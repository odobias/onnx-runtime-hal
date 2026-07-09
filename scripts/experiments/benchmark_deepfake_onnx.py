#!/usr/bin/env python3
"""Load/latency + correctness benchmark for the deepfake pipeline's classifier
models (FakeAudio / Generated Audio Detector, Text Scam Classifier) across
ONNX Runtime execution providers -- built to be re-run and compared across
platforms (x64 CPU/DirectML/VitisAI, ARM64 CPU/QNN).

These are plain ONNX Runtime classifiers (not OpenVINO GenAI, not the
whisper_engine IWhisperEngine contract), so they get their own small
benchmark harness rather than going through the C++ HAL/app -- same
precedent as the other one-off ONNX experiments in this directory.

Inputs are the REAL labeled samples wired up by the sibling validators
(validate_tsc_onnx.py, validate_fakeaudio_onnx.py) -- so every benchmarked
provider is timed on real data AND scored for correctness against ground
truth in the same session. See those two files' docstrings for full
provenance of the labeled data and the (documented, caveated) tokenizer /
audio-window assumptions. If a model's labeled data is missing (e.g. the
gitignored audio clips haven't been re-fetched), that model falls back to
synthetic inputs for latency-only and reports accuracy as blank.

Cross-platform notes (see scripts/experiments/README.md for the full matrix):
  - Which EPs are even importable depends on WHICH onnxruntime build is
    installed in the active env -- there is no single wheel with all of them.
    Run this under the environment that carries the vendor runtime you want:
    onnxruntime-directml (x64 GPU), the Ryzen AI conda env (VitisAI, custom
    core), or an ARM64 onnxruntime-qnn env (QNN).
  - Set WHISPER_HAL_ORT_DLL_DIR to prepend a native DLL directory to the
    loader search path before onnxruntime is imported (e.g. point it at the
    Ryzen AI deployment folder so vaip/dyn_dispatch/xclbin deps resolve).

    python scripts\\experiments\\benchmark_deepfake_onnx.py
    python scripts\\experiments\\benchmark_deepfake_onnx.py --models fakeaudio --providers cpu
    python scripts\\experiments\\benchmark_deepfake_onnx.py --providers cpu qnn --runs 30
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

# Let callers inject a vendor native-DLL directory (Ryzen AI deployment dir,
# QNN libs, ...) BEFORE onnxruntime loads, so its dependent DLLs resolve.
# This does NOT swap the ORT core the wheel links against -- for that you pick
# the right env/venv (see the module + README). It only fixes dependency
# resolution for the core that's already installed.
_ort_dll_dir = os.environ.get("WHISPER_HAL_ORT_DLL_DIR")
if _ort_dll_dir and os.path.isdir(_ort_dll_dir) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_ort_dll_dir)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

# The validators live next to this script and expose build_feeds()/predict().
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_fakeaudio_onnx as fakeaudio  # noqa: E402
import validate_tsc_onnx as tsc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_CSV = os.path.join(ROOT, "results", "deepfake-benchmark.csv")
RESULTS_MD = os.path.join(ROOT, "results", "deepfake-benchmark.md")

CSV_COLUMNS = [
    "timestamp_utc", "model", "model_path", "execution_provider", "input_kind",
    "runs", "warmup", "load_seconds", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms",
    "throughput_ips", "eval_samples", "correct", "accuracy", "threshold", "eval_detail",
    "input_shapes", "output_shapes", "status", "note",
    "host_cpu", "host_gpu", "host_arch", "host_os", "ort_version",
]

# Each model binds to its validator module (build_feeds + predict + THRESHOLD)
# so the labeled data and decision logic stay defined in exactly one place.
MODEL_SPECS = {
    "fakeaudio": {
        "module": fakeaudio,
        "desc": "Generated Audio Detector (MS-CLAP embedder + classifier)",
    },
    "tsc": {
        "module": tsc,
        "desc": "Text Scam Classifier (RoBERTa-BPE / DistilBERT-based)",
    },
}

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "gpu": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "npu-qualcomm": "QNNExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "npu-amd": "VitisAIExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}

# Short, table-friendly EP labels for the markdown.
EP_SHORT = {
    "CPUExecutionProvider": "CPU",
    "DmlExecutionProvider": "DML",
    "QNNExecutionProvider": "QNN",
    "VitisAIExecutionProvider": "VitisAI",
    "OpenVINOExecutionProvider": "OpenVINO",
}

ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def default_providers(available):
    """CPU (always) + whatever accelerator EPs this ORT build actually exposes,
    so the same command self-selects sensibly on each platform."""
    order = ["DmlExecutionProvider", "QNNExecutionProvider",
             "VitisAIExecutionProvider", "OpenVINOExecutionProvider"]
    return ["CPUExecutionProvider"] + [ep for ep in order if ep in available]


def _synthetic_feed(sess, seed=0):
    """Fallback feed (single row) when labeled data is unavailable -- latency only."""
    rng = np.random.default_rng(seed)
    feed = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        np_dtype = ORT_TO_NUMPY.get(inp.type, np.float32)
        if np_dtype == np.int64:
            feed[inp.name] = rng.integers(0, 1000, size=shape).astype(np_dtype)
        elif np_dtype == np.bool_:
            feed[inp.name] = rng.integers(0, 2, size=shape).astype(np_dtype)
        else:
            feed[inp.name] = rng.standard_normal(size=shape).astype(np_dtype)
    return feed


def host_hardware():
    cpu = platform.processor() or platform.machine()
    gpu = ""
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)"],
                stderr=subprocess.DEVNULL, timeout=10,
            )
            gpu = out.decode("utf-8", "ignore").strip()
        except Exception:
            pass
    return cpu, gpu


def percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def benchmark_one(model_key, provider_name, runs, warmup):
    spec = MODEL_SPECS[model_key]
    module = spec["module"]
    model_path = module.MODEL_PATH
    if not os.path.exists(model_path):
        return {"status": "missing-model", "note": model_path}

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(model_path, so, providers=[provider_name])
    except Exception as e:
        return {"status": "load-failed", "note": str(e)[:200]}
    load_seconds = time.perf_counter() - t0

    if provider_name not in sess.get_providers():
        return {"status": "provider-unavailable",
                "note": f"requested {provider_name}, got {sess.get_providers()}"}

    # Real labeled feeds if available; otherwise synthetic (latency-only).
    try:
        items = module.build_feeds(sess)
        build_note = ""
    except Exception as e:
        items = []
        build_note = f"labeled-data-error: {str(e)[:120]}"

    if items:
        input_kind = "labeled"
        feeds = [it["feed"] for it in items]
    else:
        input_kind = "synthetic"
        feeds = [_synthetic_feed(sess)]

    input_shapes = {k: list(v.shape) for k, v in feeds[0].items()}
    output_names = [o.name for o in sess.get_outputs()]

    # --- correctness (only meaningful for real labeled inputs) ---
    correct = eval_samples = 0
    detail = []
    if input_kind == "labeled":
        for it in items:
            predicted, p = module.predict(sess.run(output_names, it["feed"]))
            ok = predicted == it["label"]
            correct += int(ok)
            eval_samples += 1
            detail.append({"name": it["name"], "label": it["label"],
                           "pred": predicted, "p": round(p, 4)})

    # --- latency (cycles through the real inputs) ---
    for _ in range(warmup):
        sess.run(output_names, feeds[0])
    latencies_ms = []
    outputs = None
    for i in range(runs):
        feed = feeds[i % len(feeds)]
        t = time.perf_counter()
        outputs = sess.run(output_names, feed)
        latencies_ms.append((time.perf_counter() - t) * 1000.0)
    output_shapes = {n: list(o.shape) for n, o in zip(output_names, outputs)}

    mean_ms = statistics.mean(latencies_ms)
    result = {
        "status": "ok",
        "note": build_note,
        "input_kind": input_kind,
        "load_seconds": round(load_seconds, 4),
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(statistics.median(latencies_ms), 3),
        "p95_ms": round(percentile(latencies_ms, 95), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "throughput_ips": round(1000.0 / mean_ms, 3) if mean_ms > 0 else 0.0,
        "threshold": getattr(module, "THRESHOLD", ""),
        "input_shapes": json.dumps(input_shapes),
        "output_shapes": json.dumps(output_shapes),
    }
    if input_kind == "labeled":
        result["eval_samples"] = eval_samples
        result["correct"] = correct
        result["accuracy"] = round(correct / eval_samples, 4) if eval_samples else ""
        result["eval_detail"] = json.dumps(detail)
    return result


def append_csv(rows):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    # If the on-disk header differs (schema evolved), archive it rather than
    # writing ragged rows into a mixed-shape ledger.
    if os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0:
        with open(RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != CSV_COLUMNS:
            bak = RESULTS_CSV + ".bak"
            os.replace(RESULTS_CSV, bak)
            print(f"Schema changed; archived old rows to {os.path.relpath(bak, ROOT)}")
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in CSV_COLUMNS})


def _summary_table(rows):
    lines = ["## Latency + accuracy summary", "",
             "| Model | EP | Input | Status | Load s | Mean ms | P95 ms | ips | Accuracy |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("input_kind") == "labeled" and r.get("accuracy") != "":
            acc = f"{r.get('correct', '')}/{r.get('eval_samples', '')} ({r.get('accuracy', '')})"
        else:
            acc = "-"
        lines.append("| {model} | {ep} | {kind} | {status} | {load} | {mean} | {p95} | {ips} | {acc} |".format(
            model=r.get("model", ""), ep=EP_SHORT.get(r.get("execution_provider", ""), r.get("execution_provider", "")),
            kind=r.get("input_kind", ""), status=r.get("status", ""),
            load=r.get("load_seconds", ""), mean=r.get("mean_ms", ""),
            p95=r.get("p95_ms", ""), ips=r.get("throughput_ips", ""), acc=acc,
        ))
    return lines


def _per_sample_tables(rows):
    """One table per model: each labeled sample's positive-class probability
    under every EP that ran it, so per-sample scores AND cross-EP agreement are
    both visible. Misclassifications are flagged inline."""
    lines = []
    for model_key, spec in MODEL_SPECS.items():
        model_rows = [r for r in rows if r.get("model") == model_key
                      and r.get("input_kind") == "labeled" and r.get("eval_detail")]
        if not model_rows:
            continue
        pos = getattr(spec["module"], "POSITIVE_LABEL", "positive")
        eps, parsed, sample_order = [], {}, []
        for r in model_rows:
            ep = r["execution_provider"]
            eps.append(ep)
            parsed[ep] = {}
            for it in json.loads(r["eval_detail"]):
                parsed[ep][it["name"]] = it
                if it["name"] not in sample_order:
                    sample_order.append(it["name"])

        ep_cols = " | ".join(f"{EP_SHORT.get(ep, ep)} p({pos})" for ep in eps)
        lines += ["", f"### {model_key}: per-sample p({pos})  (MISS = predicted != label)", "",
                  f"| Sample | Label | {ep_cols} |",
                  "|" + "---|" * (2 + len(eps))]
        for name in sample_order:
            label = parsed[eps[0]][name]["label"]
            cells = [name, label]
            for ep in eps:
                it = parsed[ep].get(name, {})
                p = it.get("p", "")
                miss = it.get("pred") != label
                cells.append(f"{p}{'  MISS' if miss else ''}")
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return lines


def write_markdown(rows):
    lines = ["# Deepfake model benchmark (ONNX Runtime)", "",
             "Load/latency on **real labeled inputs** plus a correctness check against "
             "ground truth in the same session. `input_kind=labeled` rows are scored for "
             "accuracy; see `validate_tsc_onnx.py` / `validate_fakeaudio_onnx.py` docstrings "
             "for data provenance and caveats (small samples, best-effort preprocessing -- "
             "plausibility signals, not certified accuracy). Per-sample probabilities below "
             "also serve as a cross-EP numerical-agreement check (identical p across EPs = "
             "the accelerator isn't silently changing outputs)."]
    if rows:
        r0 = rows[0]
        lines += ["", f"Host: `{r0.get('host_cpu', '')}` / GPU `{r0.get('host_gpu', '') or 'n/a'}` / "
                  f"arch `{r0.get('host_arch', '')}` / {r0.get('host_os', '')} / "
                  f"ONNX Runtime `{r0.get('ort_version', '')}`."]
    lines += [""]
    lines += _summary_table(rows)
    lines += _per_sample_tables(rows)
    os.makedirs(os.path.dirname(RESULTS_MD), exist_ok=True)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()),
                    choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--providers", nargs="+", default=None,
                    help="cpu | gpu/dml | qnn | vitisai | openvino | any ORT EP name. "
                         "Default: CPU + whatever accelerators this ORT build exposes.")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    available = set(ort.get_available_providers())
    print(f"ONNX Runtime {ort.__version__} on {platform.machine()} "
          f"({platform.system()})\n  available providers: {sorted(available)}\n")
    cpu_name, gpu_name = host_hardware()
    host_arch = platform.machine()
    host_os = f"{platform.system()} {platform.release()}"

    provider_args = args.providers or default_providers(available)
    rows = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        print(f"== {model_key}: {spec['desc']} ==")
        for provider_arg in provider_args:
            provider = PROVIDER_ALIASES.get(str(provider_arg).lower(), provider_arg)
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if provider not in available:
                result = {"status": "provider-not-built", "note": provider}
            else:
                result = benchmark_one(model_key, provider, args.runs, args.warmup)
            row = {
                "timestamp_utc": ts, "model": model_key, "model_path": spec["module"].MODEL_PATH,
                "execution_provider": provider, "runs": args.runs, "warmup": args.warmup,
                "host_cpu": cpu_name, "host_gpu": gpu_name, "host_arch": host_arch,
                "host_os": host_os, "ort_version": ort.__version__,
                **result,
            }
            rows.append(row)
            if result.get("status") == "ok":
                acc = ""
                if result.get("input_kind") == "labeled":
                    acc = f"  acc={result.get('correct')}/{result.get('eval_samples')} " \
                          f"({result.get('accuracy')})"
                print(f"  {provider:26s} [{result.get('input_kind')}] "
                      f"load={result['load_seconds']:.3f}s  mean={result['mean_ms']:.2f}ms  "
                      f"p95={result['p95_ms']:.2f}ms  {result['throughput_ips']:.1f} inf/s{acc}")
            else:
                print(f"  {provider:26s} {result.get('status')}: {result.get('note', '')}")
        print()

    append_csv(rows)
    write_markdown(rows)
    print(f"Appended {len(rows)} row(s) to {os.path.relpath(RESULTS_CSV, ROOT)}")
    print(f"Summary written to {os.path.relpath(RESULTS_MD, ROOT)}")

    failures = [r for r in rows if r.get("status") != "ok"]
    if failures and len(failures) == len(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
