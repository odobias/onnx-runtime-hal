#!/usr/bin/env python3
"""Neutral ONNX Whisper driven through OpenVINO (not ORT), with a hand-rolled
greedy KV-cache decode loop. This is the FAST path with real cold/warm caching:

  cold load = read ONNX + compile for the device (from scratch, cache cleared)
  warm load = compile with ov::cache_dir hit (blob reload, no recompile)

Same decode logic as onnx_decode.py, so quality/confidence are identical; only
the runtime under it changes (ORT -> OpenVINO). Targets CPU / GPU / NPU.

  .venv\\Scripts\\python.exe scripts\\onnx_ov_decode.py models\\whisper\\en-onnx models\\eval\\eval.jsonl CPU
"""
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import openvino as ov
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


PRESENT_RE = re.compile(r"present\.(\d+)\.(decoder|encoder)\.(key|value)")


def out_name(port):
    for n in port.get_names():
        return n
    return None


class OVWhisper:
    def __init__(self, model_dir, device, cache_dir):
        self.core = ov.Core()
        self.device = device
        cfg = {"CACHE_DIR": cache_dir}
        self.enc = self.core.compile_model(os.path.join(model_dir, "encoder_model.onnx"), device, cfg)
        self.dec0 = self.core.compile_model(os.path.join(model_dir, "decoder_model.onnx"), device, cfg)
        self.decp = self.core.compile_model(os.path.join(model_dir, "decoder_with_past_model.onnx"), device, cfg)
        self.decp_input_names = [out_name(i) for i in self.decp.inputs]

        gc = json.load(open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8"))
        self.sot = gc["decoder_start_token_id"]
        self.eos = gc["eos_token_id"]
        self.suppress = np.array(sorted(gc.get("suppress_tokens", [])), dtype=np.int64)
        self.begin_suppress = np.array(sorted(gc.get("begin_suppress_tokens", [])), dtype=np.int64)
        forced = sorted(gc.get("forced_decoder_ids", []), key=lambda x: x[0])
        self.prompt = [self.sot] + [tok for _, tok in forced]
        self.max_new = 128

    @staticmethod
    def _present_to_past(name):
        return "past_key_values." + name[len("present."):]

    def _named(self, req, compiled):
        d = {}
        for port in compiled.outputs:
            d[out_name(port)] = req.get_tensor(port).data
        return d

    def transcribe(self, feats):
        r = self.enc.create_infer_request()
        r.infer({"input_features": feats})
        ehs = r.get_tensor(self.enc.outputs[0]).data.copy()

        r0 = self.dec0.create_infer_request()
        r0.infer({"input_ids": np.array([self.prompt], dtype=np.int64), "encoder_hidden_states": ehs})
        od = {}
        for port in self.dec0.outputs:
            od[out_name(port)] = r0.get_tensor(port).data.copy()
        logits = od["logits"][0, -1, :].astype(np.float32)

        past = {}
        for n, v in od.items():
            if PRESENT_RE.match(n):
                past[self._present_to_past(n)] = v

        rp = self.decp.create_infer_request()
        tokens = list(self.prompt)
        logprob_sum, n_gen = 0.0, 0
        for step in range(self.max_new):
            logits[self.suppress] = -np.inf
            if step == 0 and self.begin_suppress.size:
                logits[self.begin_suppress] = -np.inf
            m = logits.max()
            lse = m + np.log(np.exp(logits - m).sum())
            tok = int(logits.argmax())
            logprob_sum += float(logits[tok] - lse)
            n_gen += 1
            if tok == self.eos:
                break
            tokens.append(tok)

            feed = {"input_ids": np.array([[tok]], dtype=np.int64)}
            for name in self.decp_input_names:
                if name != "input_ids":
                    feed[name] = past[name]
            rp.infer(feed)
            od = {}
            for port in self.decp.outputs:
                od[out_name(port)] = rp.get_tensor(port).data.copy()
            logits = od["logits"][0, -1, :].astype(np.float32)
            for n, v in od.items():
                if PRESENT_RE.match(n) and ".decoder." in n:
                    past[self._present_to_past(n)] = v

        return tokens[len(self.prompt):], logprob_sum / max(n_gen, 1), n_gen


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/whisper/en-onnx"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "models/eval/eval.jsonl"
    device = sys.argv[3] if len(sys.argv) > 3 else "CPU"
    cache = os.path.join("cache", f"onnx-ov-{device}")

    from transformers import WhisperProcessor

    print(f"model    : {model_dir}")
    print(f"device   : {device}   OpenVINO {ov.get_version()}")

    # COLD: clear cache, compile from ONNX
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    t = time.time()
    w = OVWhisper(model_dir, device, cache)
    cold_load = time.time() - t
    proc = WhisperProcessor.from_pretrained(model_dir)
    print(f"cold load (compile ONNX)   : {cold_load:.2f} s")

    # WARM: fresh objects, cache hit
    t = time.time()
    _ = OVWhisper(model_dir, device, cache)
    warm_load = time.time() - t
    print(f"warm load (cache hit)      : {warm_load:.2f} s   speedup {cold_load/max(warm_load,1e-3):.1f}x")
    print(f"prompt tokens              : {w.prompt}\n")

    clips = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]
    feats = []
    for c in clips:
        wav, _ = sf.read(c["audio"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        feats.append(proc(wav, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32))

    # warm up inference kernels
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
    print(f"device             : {device}")
    print(f"cold load (compile): {cold_load:.2f} s")
    print(f"warm load (cache)  : {warm_load:.2f} s")
    print(f"cold infer (clip0) : {cold_infer:.1f} ms")
    print(f"hot infer mean     : {np.mean(lats):.1f} ms  (median {np.median(lats):.1f}, min {np.min(lats):.1f})")
    print(f"WER (micro)        : {100.0*w_edits/max(w_ref,1):.2f} %")
    print(f"CER (micro)        : {100.0*c_edits/max(c_ref,1):.2f} %")
    print(f"confidence mean    : {np.mean(confs):.4f}  (mean per-token log-prob)")


if __name__ == "__main__":
    main()
