#!/usr/bin/env python3
"""Undumbed ONNX Whisper: a minimal greedy KV-cache decode loop on raw ORT
sessions. No optimum generate() overhead. Correct forced/suppress tokens read
from generation_config.json. Reports cold vs hot latency + WER/CER + confidence.

  .venv\\Scripts\\python.exe scripts\\onnx_decode.py models\\whisper\\en-onnx models\\eval\\eval.jsonl [provider]
"""
import json
import os
import re
import sys
import time

import numpy as np
import onnxruntime as ort
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


def make_session(path, provider, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=[provider])


PRESENT_RE = re.compile(r"present\.(\d+)\.(decoder|encoder)\.(key|value)")


class OnnxWhisper:
    def __init__(self, model_dir, provider="CPUExecutionProvider", threads=None):
        threads = threads or os.cpu_count()
        self.enc = make_session(os.path.join(model_dir, "encoder_model.onnx"), provider, threads)
        self.dec0 = make_session(os.path.join(model_dir, "decoder_model.onnx"), provider, threads)
        self.decp = make_session(os.path.join(model_dir, "decoder_with_past_model.onnx"), provider, threads)
        self.decp_inputs = [i.name for i in self.decp.get_inputs()]

        cfg = json.load(open(os.path.join(model_dir, "generation_config.json"), encoding="utf-8"))
        self.sot = cfg["decoder_start_token_id"]
        self.eos = cfg["eos_token_id"]
        self.suppress = set(cfg.get("suppress_tokens", []))
        self.begin_suppress = set(cfg.get("begin_suppress_tokens", []))
        # forced_decoder_ids: [[pos, token], ...] -> prompt after sot
        forced = sorted(cfg.get("forced_decoder_ids", []), key=lambda x: x[0])
        self.prompt = [self.sot] + [tok for _, tok in forced]
        self.max_new = 128

    @staticmethod
    def _present_to_past(name):
        return "past_key_values." + name[len("present."):]

    def _apply_suppress(self, logits, first):
        if self.suppress:
            logits[list(self.suppress)] = -np.inf
        if first and self.begin_suppress:
            logits[list(self.begin_suppress)] = -np.inf
        return logits

    def transcribe(self, feats):
        ehs = self.enc.run(None, {"input_features": feats})[0]

        # prefill on the forced prompt with the no-past decoder
        ids = np.array([self.prompt], dtype=np.int64)
        outs = self.dec0.run(None, {"input_ids": ids, "encoder_hidden_states": ehs})
        onames = [o.name for o in self.dec0.get_outputs()]
        od = dict(zip(onames, outs))
        logits = od["logits"][0, -1, :].astype(np.float32)

        past = {}
        for n, v in od.items():
            if PRESENT_RE.match(n):
                past[self._present_to_past(n)] = v

        tokens = list(self.prompt)
        logprob_sum = 0.0
        n_gen = 0
        for step in range(self.max_new):
            logits = self._apply_suppress(logits, first=(step == 0))
            m = logits.max()
            lse = m + np.log(np.exp(logits - m).sum())
            tok = int(logits.argmax())
            logprob_sum += float(logits[tok] - lse)
            n_gen += 1
            if tok == self.eos:
                break
            tokens.append(tok)

            feed = {"input_ids": np.array([[tok]], dtype=np.int64)}
            for name in self.decp_inputs:
                if name != "input_ids":
                    feed[name] = past[name]
            outs = self.decp.run(None, feed)
            onames = [o.name for o in self.decp.get_outputs()]
            od = dict(zip(onames, outs))
            logits = od["logits"][0, -1, :].astype(np.float32)
            for n, v in od.items():
                if PRESENT_RE.match(n) and ".decoder." in n:
                    past[self._present_to_past(n)] = v

        gen = [t for t in tokens[len(self.prompt):]]
        avg_lp = logprob_sum / max(n_gen, 1)
        return gen, avg_lp, n_gen


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "src/workloads/whisper/models/dynamic-onnx"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "src/workloads/eval/eval.jsonl"
    provider = sys.argv[3] if len(sys.argv) > 3 else "CPUExecutionProvider"

    from transformers import WhisperProcessor

    print(f"model    : {model_dir}")
    print(f"provider : {provider}   ORT {ort.__version__}   threads {os.cpu_count()}")

    t = time.time()
    proc = WhisperProcessor.from_pretrained(model_dir)
    w = OnnxWhisper(model_dir, provider=provider)
    load_s = time.time() - t
    print(f"cold load (sessions init)  : {load_s:.2f} s")
    print(f"prompt tokens              : {w.prompt}\n")

    clips = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]
    feats = []
    for c in clips:
        wav, _ = sf.read(c["audio"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        feats.append(proc(wav, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32))

    # COLD: very first transcription (ORT graph warmup / allocator)
    t = time.time()
    _ = w.transcribe(feats[0])
    cold_ms = (time.time() - t) * 1000
    print(f"cold inference (clip 0)    : {cold_ms:.1f} ms")

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
    print(f"clips              : {len(clips)}")
    print(f"cold load (init)   : {load_s:.2f} s")
    print(f"cold infer (clip0) : {cold_ms:.1f} ms")
    print(f"hot infer mean     : {np.mean(lats):.1f} ms  (median {np.median(lats):.1f}, min {np.min(lats):.1f})")
    print(f"speedup cold->hot  : {cold_ms/np.mean(lats):.1f}x")
    print(f"WER (micro)        : {100.0*w_edits/max(w_ref,1):.2f} %")
    print(f"CER (micro)        : {100.0*c_edits/max(c_ref,1):.2f} %")
    print(f"confidence mean    : {np.mean(confs):.4f}  (mean per-token log-prob)")


if __name__ == "__main__":
    main()
