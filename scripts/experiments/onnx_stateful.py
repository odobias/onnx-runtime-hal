#!/usr/bin/env python3
"""Fix the 'no KV cache': convert the neutral ONNX decoder to a STATEFUL model so
OpenVINO owns the decoder KV cache internally (ReadValue/Assign) instead of us
shuttling growing tensors through Python every step.

Flow:
  encoder(features) -> ehs
  no-past decoder(prompt, ehs) -> logits + decoder-KV (prefill) + encoder-KV (const)
  make_stateful(with_past): pair present.L.decoder.* <-> past_key_values.L.decoder.*
  seed the stateful decoder's internal state with the prefill decoder-KV
  generate: feed 1 token + constant encoder-KV; state auto-updates. No KV copies.

  .venv\\Scripts\\python.exe scripts\\onnx_stateful.py models\\whisper-tiny-en-onnx models\\eval\\eval.jsonl CPU
"""
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import openvino as ov
from openvino._offline_transformations import apply_make_stateful_transformation
import soundfile as sf


def normalize(t):
    out = []
    for ch in t:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch == "'":
            out.append("'")
        elif not ch.isalnum():
            if out and out[-1] != " ":
                out.append(" ")
    return "".join(out).strip()


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


LAYER_KV = re.compile(r"\.(\d+)\.decoder\.(key|value)")


class OVWhisperStateful:
    def __init__(self, model_dir, device, cache_dir):
        self.core = ov.Core()
        self.device = device
        cfg = {"CACHE_DIR": cache_dir}

        self.enc = self.core.compile_model(os.path.join(model_dir, "encoder_model.onnx"), device, cfg)
        self.dec0 = self.core.compile_model(os.path.join(model_dir, "decoder_model.onnx"), device, cfg)

        wp = self.core.read_model(os.path.join(model_dir, "decoder_with_past_model.onnx"))
        in_names = {i.get_any_name() for i in wp.inputs}
        out_names = {o.get_any_name() for o in wp.outputs}
        # pair decoder KV: input past_key_values.L.decoder.X -> output present.L.decoder.X
        pairs = {}
        for inp in sorted(in_names):
            if ".decoder." in inp and inp.startswith("past_key_values"):
                out = "present." + inp[len("past_key_values."):]
                if out in out_names:
                    pairs[inp] = out
        self.n_state = len(pairs)
        apply_make_stateful_transformation(wp, pairs)
        self.decp = self.core.compile_model(wp, device, cfg)
        self.decp_inputs = [i.get_any_name() for i in self.decp.inputs]

        gc = json.load(open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8"))
        self.sot = gc["decoder_start_token_id"]
        self.eos = gc["eos_token_id"]
        self.suppress = np.array(sorted(gc.get("suppress_tokens", [])), dtype=np.int64)
        self.begin_suppress = np.array(sorted(gc.get("begin_suppress_tokens", [])), dtype=np.int64)
        forced = sorted(gc.get("forced_decoder_ids", []), key=lambda x: x[0])
        self.prompt = [self.sot] + [tok for _, tok in forced]

    def transcribe(self, feats):
        er = self.enc.create_infer_request()
        er.infer({"input_features": feats})
        ehs = er.get_tensor(self.enc.outputs[0]).data.copy()

        # prefill: no-past decoder over the prompt
        r0 = self.dec0.create_infer_request()
        r0.infer({"input_ids": np.array([self.prompt], dtype=np.int64), "encoder_hidden_states": ehs})
        prefill = {}
        for port in self.dec0.outputs:
            prefill[port.get_any_name()] = r0.get_tensor(port).data.copy()
        logits = prefill["logits"][0, -1, :].astype(np.float32)

        # constant encoder cross-attn KV (inputs to the stateful decoder)
        enc_kv = {}
        for name in self.decp_inputs:
            if ".encoder." in name:
                src = "present." + name[len("past_key_values."):]
                enc_kv[name] = prefill[src]

        # seed the stateful decoder's internal decoder-KV state from the prefill
        rp = self.decp.create_infer_request()
        for st in rp.query_state():
            mm = LAYER_KV.search(st.name)  # robust to mangled/doubled variable ids
            src = f"present.{mm.group(1)}.decoder.{mm.group(2)}"
            st.state = ov.Tensor(np.ascontiguousarray(prefill[src]))

        tokens = list(self.prompt)
        logprob_sum, n_gen = 0.0, 0
        first = True
        for _ in range(128):
            logits[self.suppress] = -np.inf
            if first and self.begin_suppress.size:
                logits[self.begin_suppress] = -np.inf
            first = False
            m = logits.max()
            lse = m + np.log(np.exp(logits - m).sum())
            tok = int(logits.argmax())
            logprob_sum += float(logits[tok] - lse)
            n_gen += 1
            if tok == self.eos:
                break
            tokens.append(tok)
            feed = {"input_ids": np.array([[tok]], dtype=np.int64)}
            feed.update(enc_kv)
            rp.infer(feed)
            logits = rp.get_tensor(self.decp.outputs[0]).data[0, -1, :].astype(np.float32)

        return tokens[len(self.prompt):], logprob_sum / max(n_gen, 1), n_gen


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-onnx"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "models/eval/eval.jsonl"
    device = sys.argv[3] if len(sys.argv) > 3 else "CPU"
    cache = os.path.join("cache", f"onnx-stateful-{device}")

    from transformers import WhisperProcessor

    print(f"model    : {model_dir}")
    print(f"device   : {device}   OpenVINO {ov.get_version()}")

    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    t = time.time()
    w = OVWhisperStateful(model_dir, device, cache)
    cold_load = time.time() - t
    proc = WhisperProcessor.from_pretrained(model_dir)
    print(f"stateful KV pairs          : {w.n_state}")
    print(f"cold load (compile)        : {cold_load:.2f} s")

    t = time.time()
    _ = OVWhisperStateful(model_dir, device, cache)
    warm_load = time.time() - t
    print(f"warm load (cache hit)      : {warm_load:.2f} s\n")

    clips = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]
    feats = []
    for c in clips:
        wav, _ = sf.read(c["audio"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        feats.append(proc(wav, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32))

    t = time.time(); _ = w.transcribe(feats[0]); cold_infer = (time.time() - t) * 1000
    print(f"cold inference (clip 0)    : {cold_infer:.1f} ms")

    w_edits = w_ref = c_edits = c_ref = 0
    lats, confs = [], []
    print(f"\n{'clip':8s} {'ms':>7s} {'conf':>7s} {'wer%':>6s}  text")
    print("-" * 78)
    for c, f in zip(clips, feats):
        t = time.time()
        gen, avg_lp, ntok = w.transcribe(f)
        dt = (time.time() - t) * 1000
        lats.append(dt)
        confs.append(avg_lp)
        text = proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
        r = normalize(c["ref"]).split()
        h = normalize(text).split()
        we = edit_distance(r, h)
        w_edits += we
        w_ref += len(r)
        rc = normalize(c["ref"]).replace(" ", "")
        hc = normalize(text).replace(" ", "")
        c_edits += edit_distance(list(rc), list(hc))
        c_ref += len(rc)
        print(f"{c['id']:8s} {dt:7.1f} {avg_lp:7.3f} {100.0*we/max(len(r),1):6.2f}  {text[:44]}")

    print("-" * 78)
    print(f"device             : {device}  (stateful KV cache)")
    print(f"cold load          : {cold_load:.2f} s   warm load: {warm_load:.2f} s")
    print(f"cold infer (clip0) : {cold_infer:.1f} ms")
    print(f"hot infer mean     : {np.mean(lats):.1f} ms  (median {np.median(lats):.1f}, min {np.min(lats):.1f})")
    print(f"WER (micro)        : {100.0*w_edits/max(w_ref,1):.2f} %")
    print(f"CER (micro)        : {100.0*c_edits/max(c_ref,1):.2f} %")
    print(f"confidence mean    : {np.mean(confs):.4f}")


if __name__ == "__main__":
    main()
