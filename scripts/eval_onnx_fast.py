#!/usr/bin/env python3
"""FAST end-to-end ONNX Whisper eval with explicit cold vs hot timing.

Undumbs the previous run:
  * merged KV-cache decoder (use_cache=True)  -> O(n) decode, not O(n^2)
  * IO binding on                              -> no host<->device copies per step
  * all CPU cores + ORT_ENABLE_ALL graph opt   -> real throughput
  * cold  = first generate() after session init (graph warm-up / alloc)
  * hot   = steady-state generate() (caches primed)

Usage:
  .venv\\Scripts\\python.exe scripts\\eval_onnx_fast.py models\\whisper-tiny-en-onnx models\\eval\\eval.jsonl [provider]
"""
import json
import os
import sys
import time

import numpy as np
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


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-onnx"
    eval_path = sys.argv[2] if len(sys.argv) > 2 else "models/eval/eval.jsonl"
    provider = sys.argv[3] if len(sys.argv) > 3 else "CPUExecutionProvider"

    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import WhisperProcessor

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = os.cpu_count()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    print(f"model    : {model_dir}")
    print(f"provider : {provider}   threads: {os.cpu_count()}   ORT: {ort.__version__}")

    t = time.time()
    proc = WhisperProcessor.from_pretrained(model_dir)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_dir, provider=provider, session_options=so,
        use_io_binding=True, use_cache=True,
    )
    load_s = time.time() - t
    dec = getattr(model, "decoder_with_past", None)
    print(f"session init (cold load)   : {load_s:.2f} s")
    print(f"KV-cache decoder present   : {dec is not None}")

    clips = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]

    # pre-extract features so timing is pure inference
    feats = []
    for c in clips:
        wav, _ = sf.read(c["audio"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        feats.append(proc(wav, sampling_rate=16000, return_tensors="pt").input_features)

    gen_kw = dict(max_new_tokens=128, num_beams=1)

    # --- COLD: first ever generate (graph warmup, allocator, kernel selection) ---
    t = time.time()
    _ = model.generate(feats[0], **gen_kw)
    cold_infer = (time.time() - t) * 1000
    print(f"cold inference (clip 0)    : {cold_infer:.1f} ms")

    # --- HOT: steady state over all clips (+ quality/confidence) ---
    w_edits = w_ref = c_edits = c_ref = 0
    lats, confs = [], []
    print(f"\n{'clip':8s} {'ms':>7s} {'conf':>7s} {'wer%':>6s}  text")
    print("-" * 78)
    for c, f in zip(clips, feats):
        t = time.time()
        out = model.generate(f, return_dict_in_generate=True, output_scores=True, **gen_kw)
        dt = (time.time() - t) * 1000
        lats.append(dt)
        seq = out.sequences
        text = proc.batch_decode(seq, skip_special_tokens=True)[0].strip()
        try:
            ts = model.compute_transition_scores(seq, out.scores, normalize_logits=True)
            conf = float(ts.mean())
        except Exception:
            conf = float("nan")
        confs.append(conf)
        r = normalize(c["ref"]).split()
        h = normalize(text).split()
        we = edit_distance(r, h)
        w_edits += we
        w_ref += len(r)
        rc = normalize(c["ref"]).replace(" ", "")
        hc = normalize(text).replace(" ", "")
        c_edits += edit_distance(list(rc), list(hc))
        c_ref += len(rc)
        print(f"{c['id']:8s} {dt:7.1f} {conf:7.3f} {100.0*we/max(len(r),1):6.2f}  {text[:44]}")

    print("-" * 78)
    print(f"clips              : {len(clips)}")
    print(f"cold load (init)   : {load_s:.2f} s")
    print(f"cold infer (clip0) : {cold_infer:.1f} ms")
    print(f"hot infer mean     : {np.mean(lats):.1f} ms  (median {np.median(lats):.1f}, min {np.min(lats):.1f})")
    print(f"speedup cold->hot  : {cold_infer/np.mean(lats):.1f}x")
    print(f"WER (micro)        : {100.0*w_edits/max(w_ref,1):.2f} %")
    print(f"CER (micro)        : {100.0*c_edits/max(c_ref,1):.2f} %")
    print(f"confidence mean    : {np.mean(confs):.4f}")


if __name__ == "__main__":
    main()
