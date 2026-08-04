#!/usr/bin/env python3
"""Neutral ONNX Whisper on the NPU via a STATIC-shape decode.

The NPU compiler rejects dynamic shapes (growing KV cache). Instead of a
KV-cache decoder we reshape the *no-past* decoder to a fixed context length
MAXLEN and recompute the whole (padded) sequence each step. Whisper's decoder
is causal, so logits at position t depend only on tokens 0..t -> padding after t
is harmless. Shapes are constant every step => NPU-compilable.

  encoder: [1,80,3000] -> [1,1500,384]   (static)
  decoder: input_ids[1,MAXLEN] + enc_hidden[1,1500,384] -> logits[1,MAXLEN,V]

Cost: O(n * MAXLEN) instead of O(n) (no cache reuse), but fully static + cached.

  artifacts/venv\\Scripts\\python.exe scripts\\onnx_npu_static.py models\\whisper\\en-onnx models\\eval\\eval.jsonl NPU 64
"""
import json
import os
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


class OVWhisperStatic:
    def __init__(self, model_dir, device, cache_dir, maxlen):
        self.core = ov.Core()
        self.maxlen = maxlen
        cfg = {"CACHE_DIR": cache_dir}

        enc_m = self.core.read_model(os.path.join(model_dir, "encoder_model.onnx"))
        enc_m.reshape({"input_features": [1, 80, 3000]})
        self.enc = self.core.compile_model(enc_m, device, cfg)
        self.enc_hidden = self.enc.outputs[0]

        dec_m = self.core.read_model(os.path.join(model_dir, "decoder_model.onnx"))
        dec_m.reshape({"input_ids": [1, maxlen], "encoder_hidden_states": [1, 1500, 384]})
        self.dec = self.core.compile_model(dec_m, device, cfg)

        gc = json.load(open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8"))
        self.sot = gc["decoder_start_token_id"]
        self.eos = gc["eos_token_id"]
        self.pad = gc.get("pad_token_id", self.eos)
        self.suppress = np.array(sorted(gc.get("suppress_tokens", [])), dtype=np.int64)
        self.begin_suppress = np.array(sorted(gc.get("begin_suppress_tokens", [])), dtype=np.int64)
        forced = sorted(gc.get("forced_decoder_ids", []), key=lambda x: x[0])
        self.prompt = [self.sot] + [tok for _, tok in forced]

    def transcribe(self, feats):
        er = self.enc.create_infer_request()
        er.infer({"input_features": feats})
        ehs = er.get_tensor(self.enc_hidden).data.copy()

        ids = np.full((1, self.maxlen), self.pad, dtype=np.int64)
        for i, t in enumerate(self.prompt):
            ids[0, i] = t
        cur = len(self.prompt)  # next position to fill

        dr = self.dec.create_infer_request()
        logprob_sum, n_gen = 0.0, 0
        gen = []
        first = True
        while cur < self.maxlen:
            dr.infer({"input_ids": ids, "encoder_hidden_states": ehs})
            logits_all = dr.get_tensor(self.dec.outputs[0]).data
            logits = logits_all[0, cur - 1, :].astype(np.float32)
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
            gen.append(tok)
            ids[0, cur] = tok
            cur += 1

        return gen, logprob_sum / max(n_gen, 1), n_gen


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/workloads/whisper/models/dynamic-onnx"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "src/workloads/eval/eval.jsonl"
    device = sys.argv[3] if len(sys.argv) > 3 else "NPU"
    maxlen = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    cache = os.path.join("cache", f"onnx-static-{device}-{maxlen}")

    from transformers import WhisperProcessor

    print(f"model    : {model_dir}")
    print(f"device   : {device}   maxlen {maxlen}   OpenVINO {ov.get_version()}")

    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    t = time.time()
    w = OVWhisperStatic(model_dir, device, cache, maxlen)
    cold_load = time.time() - t
    proc = WhisperProcessor.from_pretrained(model_dir)
    print(f"cold load (compile static) : {cold_load:.2f} s")

    t = time.time()
    _ = OVWhisperStatic(model_dir, device, cache, maxlen)
    warm_load = time.time() - t
    print(f"warm load (cache hit)      : {warm_load:.2f} s   speedup {cold_load/max(warm_load,1e-3):.1f}x\n")

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
    print(f"device             : {device}   maxlen {maxlen}")
    print(f"cold load (compile): {cold_load:.2f} s")
    print(f"warm load (cache)  : {warm_load:.2f} s")
    print(f"cold infer (clip0) : {cold_infer:.1f} ms")
    print(f"hot infer mean     : {np.mean(lats):.1f} ms  (median {np.median(lats):.1f}, min {np.min(lats):.1f})")
    print(f"WER (micro)        : {100.0*w_edits/max(w_ref,1):.2f} %")
    print(f"CER (micro)        : {100.0*c_edits/max(c_ref,1):.2f} %")
    print(f"confidence mean    : {np.mean(confs):.4f}")


if __name__ == "__main__":
    main()
