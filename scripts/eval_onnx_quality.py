#!/usr/bin/env python3
"""End-to-end transcribe quality + confidence for the NEUTRAL ONNX Whisper.

Drives the exported ONNX (encoder + KV-cache decoder) through a real generate()
loop via optimum-onnx's ORTModelForSpeechSeq2Seq, so we get correct
transcriptions, WER/CER, self-confidence (mean per-token log-prob), and
end-to-end latency -- directly comparable to the OpenVINO-IR GenAI baseline.

  .venv\Scripts\python.exe scripts\eval_onnx_quality.py models\whisper-tiny-en-onnx models\eval\eval.jsonl
"""
import json
import sys
import time

import numpy as np
import soundfile as sf


# ---- tiny WER/CER matching the C++ metrics.hpp normalization ----
def normalize(t):
    out = []
    for ch in t:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch == "'":
            out.append("'")
        elif ch.isspace() or not ch.isalnum():
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

    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import WhisperProcessor

    print(f"model    : {model_dir}")
    print(f"provider : {provider}")
    proc = WhisperProcessor.from_pretrained(model_dir)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_dir, provider=provider)

    clips = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]
    root = "."
    w_edits = w_ref = c_edits = c_ref = 0
    lats, confs = [], []
    print(f"\n{'clip':8s} {'ms':>7s} {'conf':>7s} {'wer%':>6s}  text")
    print("-" * 78)
    for c in clips:
        audio_path = c["audio"]
        wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features

        t0 = time.time()
        out = model.generate(
            feats, max_new_tokens=128, num_beams=1,
            return_dict_in_generate=True, output_scores=True,
        )
        dt = (time.time() - t0) * 1000
        lats.append(dt)

        seq = out.sequences
        text = proc.batch_decode(seq, skip_special_tokens=True)[0].strip()

        # mean per-token log-prob (normalized logits) = confidence proxy
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
        wer = 100.0 * we / max(len(r), 1)
        print(f"{c['id']:8s} {dt:7.1f} {conf:7.3f} {wer:6.2f}  {text[:44]}")

    print("-" * 78)
    print(f"clips           : {len(clips)}")
    print(f"WER (micro)     : {100.0*w_edits/max(w_ref,1):.2f} %")
    print(f"CER (micro)     : {100.0*c_edits/max(c_ref,1):.2f} %")
    print(f"confidence mean : {np.mean(confs):.4f}  (mean per-token log-prob, normalized)")
    print(f"latency mean    : {np.mean(lats):.1f} ms  (median {np.median(lats):.1f})")


if __name__ == "__main__":
    main()
