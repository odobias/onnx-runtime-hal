#!/usr/bin/env python3
"""Append the stateful-KV neutral-ONNX rows to both results files."""
import csv
import datetime
import os

TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
ONNX_DIR = r"models\whisper-tiny-en-onnx"
EVAL = r"models\eval\eval.jsonl"

# --- onnx-portability.csv (its own schema) ---
PORT = os.path.join("results", "onnx-portability.csv")
port_rows = [
    ["ov-stateful", "CPU", "", "0.82", "0.79", "347.5", "236.9", "8.58", "4.53", "-0.164",
     "neutral ONNX made stateful (OV owns decoder KV); no per-step copies"],
    ["ov-stateful", "GPU", "", "7.11", "0.68", "652.9", "304.4", "8.58", "4.53", "-0.164",
     "stateful; GPU already handled dynamic KV well, no win here"],
    ["ov-stateful", "NPU", "", "", "", "", "", "", "", "",
     "COMPILE FAIL: plain stateful state is unbounded; NPU needs bounded max-context KV"],
]
with open(PORT, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(port_rows)

# --- benchmark-results.csv (extended schema) ---
BENCH = os.path.join("results", "benchmark-results.csv")
with open(BENCH, newline="", encoding="utf-8") as f:
    cols = next(csv.reader(f))


def row(device, cold, warm, hot_ms, wer, cer, conf, status, mb):
    d = {c: "" for c in cols}
    d.update({
        "timestamp_utc": TS, "requested_backend": "OnnxNeutral",
        "resolved_backend": "Neutral ONNX via OpenVINO (stateful KV)",
        "device": device, "device_name": device,
        "model_dir": ONNX_DIR, "audio_path": EVAL, "runs": "12", "warmup": "1",
        "cold_load_seconds": cold, "warm_load_seconds": warm,
        "mean_infer_seconds": "" if hot_ms == "" else f"{hot_ms/1000.0:.6g}",
        "label": "onnx-neutral", "model_size_mb": mb,
        "avg_logprob": conf,
        "wer": "" if wer == "" else f"{wer/100.0:.6g}",
        "cer": "" if cer == "" else f"{cer/100.0:.6g}",
        "runtime": "OpenVINO", "model_format": "onnx",
        "decode_strategy": "stateful-kv", "eval_clips": "12", "status": status,
    })
    return d


new = [
    row("CPU", "0.82", "0.79", 347.5, 8.58, 4.53, "-0.164", "ok", "404.5"),
    row("GPU", "7.11", "0.68", 652.9, 8.58, 4.53, "-0.164", "ok", "404.5"),
    row("NPU", "", "", "", "", "", "", "fail (unbounded state; NPU needs bounded KV)", "404.5"),
]
with open(BENCH, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    for r in new:
        w.writerow(r)

print(f"appended {len(port_rows)} rows to onnx-portability.csv and {len(new)} to benchmark-results.csv")
