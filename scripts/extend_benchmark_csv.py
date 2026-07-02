#!/usr/bin/env python3
"""Extend benchmark-results.csv with columns that capture the differences
between the C++ single-clip runs and the neutral-ONNX portability sweep, then
append the ONNX rows. New columns are appended at the END so positional readers
of the old schema keep working; existing rows are back-filled.

New columns:
  runtime          execution stack (OpenVINO / ONNX Runtime / optimum ...)
  model_format     onnx | ov-ir
  decode_strategy  stateful-kv | dynamic-kv | static-no-kv | raw-loop | optimum-generate
  max_context      static decoder context length (blank if dynamic/NA)
  eval_clips       number of clips aggregated (C++ single runs = 1)
  status           ok | fail
"""
import csv
import datetime
import os

CSV = os.path.join("results", "benchmark-results.csv")
NEW_COLS = ["runtime", "model_format", "decode_strategy", "max_context", "eval_clips", "status"]

TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
ONNX_DIR = r"models\whisper-tiny-en-onnx"
EVAL = r"models\eval\eval.jsonl"


def frac(pct):
    return f"{pct/100.0:.6g}"


def onnx_row(requested, resolved, device, cold, warm, hot_ms, wer, cer, conf,
             runtime, decode, model_mb, status="ok", maxctx="", cache=""):
    return {
        "timestamp_utc": TS,
        "requested_backend": requested,
        "resolved_backend": resolved,
        "device": device,
        "device_name": device,
        "device_full_name": "",
        "model_dir": ONNX_DIR,
        "audio_path": EVAL,
        "audio_seconds": "",
        "runs": "12",
        "warmup": "1",
        "cache_dir": cache,
        "cold_load_seconds": cold,
        "warm_load_seconds": warm,
        "mean_infer_seconds": "" if hot_ms == "" else f"{hot_ms/1000.0:.6g}",
        "rtf": "",
        "realtime_factor": "",
        "label": "onnx-neutral",
        "model_size_mb": model_mb,
        "avg_logprob": conf,
        "ttft_ms": "",
        "tpot_ms": "",
        "throughput_tps": "",
        "wer": "" if wer == "" else frac(wer),
        "cer": "" if cer == "" else frac(cer),
        "transcription": "",
        "runtime": runtime,
        "model_format": "onnx",
        "decode_strategy": decode,
        "max_context": maxctx,
        "eval_clips": "12",
        "status": status,
    }


def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    old_cols = list(rows[0].keys())
    out_cols = old_cols + NEW_COLS

    # back-fill existing single-clip rows
    for r in rows:
        rb = r.get("resolved_backend", "")
        if "OpenVINO GenAI" in rb:
            r["runtime"] = "OpenVINO GenAI"
            r["model_format"] = "ov-ir"
            r["decode_strategy"] = "stateful-kv"
        elif "VitisAI" in rb or "Ryzen" in rb:
            r["runtime"] = "ONNX Runtime + VitisAI EP"
            r["model_format"] = "onnx"
            r["decode_strategy"] = ""
        else:
            r["runtime"] = rb
            r["model_format"] = ""
            r["decode_strategy"] = ""
        r["max_context"] = ""
        r["eval_clips"] = "1"
        r["status"] = "ok"

    # new neutral-ONNX portability rows (12-clip aggregates)
    new = [
        onnx_row("OnnxNeutral", "Neutral ONNX via optimum generate() (ORT)", "CPU",
                 "1.16", "", 13910.2, 8.58, 4.53, "-0.164",
                 "ONNX Runtime", "optimum-generate", "220.3"),
        onnx_row("OnnxNeutral", "Neutral ONNX via raw ORT sessions", "CPU",
                 "1.45", "", 2029.2, 8.58, 4.53, "-0.164",
                 "ONNX Runtime", "raw-loop", "404.5"),
        onnx_row("OnnxNeutral", "Neutral ONNX via OpenVINO", "CPU",
                 "1.85", "0.32", 526.7, 8.58, 4.53, "-0.164",
                 "OpenVINO", "dynamic-kv", "404.5",
                 cache=r"cache\onnx-ov-CPU"),
        onnx_row("OnnxNeutral", "Neutral ONNX via OpenVINO", "GPU",
                 "16.39", "0.18", 316.5, 8.58, 4.53, "-0.164",
                 "OpenVINO", "dynamic-kv", "404.5",
                 cache=r"cache\onnx-ov-GPU"),
        onnx_row("OnnxNeutral", "Neutral ONNX via OpenVINO", "NPU",
                 "", "", "", "", "", "",
                 "OpenVINO", "dynamic-kv", "404.5",
                 status="fail (NPU rejects dynamic KV shapes)"),
        onnx_row("OnnxNeutral", "Neutral ONNX via OpenVINO", "NPU",
                 "9.18", "0.70", 969.7, 9.57, 5.10, "-0.1675",
                 "OpenVINO", "static-no-kv", "220.2",
                 maxctx="96", cache=r"cache\onnx-static-NPU-96"),
        onnx_row("OnnxNeutral", "OV-IR GenAI baseline (not ONNX)", "NPU",
                 "10.02", "0.70", 137.3, 9.57, 4.46, "",
                 "OpenVINO GenAI", "stateful-kv", "158.0"),
    ]
    # the baseline row is ov-ir, fix its format
    new[-1]["model_format"] = "ov-ir"
    new[-1]["label"] = "ovir-genai-baseline"

    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        for r in new:
            w.writerow(r)

    print(f"wrote {len(rows)} back-filled rows + {len(new)} new ONNX rows")
    print(f"columns: {out_cols}")


if __name__ == "__main__":
    main()
