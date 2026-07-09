#!/usr/bin/env python3
"""Dump the deepfake classifiers' *validated* ONNX input tensors (and the CPU
reference probabilities) to a flat on-disk fixture format the C++ classifier
benchmark can replay.

Why fixtures instead of doing preprocessing in C++:
  The two classifiers need non-trivial, already-validated preprocessing --
  RoBERTa-style byte-level BPE for TSC, a 7s raw-PCM window for FakeAudio (its
  MS-CLAP front-end is traced into the graph). Re-implementing that in C++ would
  duplicate logic and risk drift from the Python "source of truth". Instead the
  Python validators (validate_tsc_onnx / validate_fakeaudio_onnx) build the exact
  feeds, we snapshot those tensors + the CPU-run reference probability, and the
  C++ harness just replays them across execution providers (CPU/DirectML/NPU).
  That keeps correctness anchored to the validated preprocessing while still
  measuring real per-EP latency AND per-EP numerical agreement (does the
  accelerator reproduce the CPU probability?).

Output (gitignored, under models/deepfake/fixtures/<model>/):
  model.tsv    onnx_path <tab> positive_index <tab> threshold <tab> pos_label <tab> neg_label
  samples.tsv  sample_id <tab> label <tab> expected_pred <tab> expected_p
  inputs.tsv   sample_id <tab> input_name <tab> dtype(f32|i64) <tab> shape(csv) <tab> file
  data/*.bin   raw little-endian tensors

    .venv\\Scripts\\python.exe scripts\\experiments\\dump_fixtures.py
    .venv\\Scripts\\python.exe scripts\\experiments\\dump_fixtures.py --models tsc
"""
import argparse
import importlib
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FIX_ROOT = os.path.join(ROOT, "models", "deepfake", "fixtures")

# numpy dtype -> the two tokens the C++ loader understands.
DTYPE_TOKEN = {np.dtype(np.int64): "i64", np.dtype(np.float32): "f32"}

WINDOW_SAMPLES = 308700  # FakeAudio static input (7.0s @ 44.1kHz) for the synthetic fallback


def softmax_pos(outputs, positive_index):
    x = np.asarray(outputs[0], dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    return float(p[0, positive_index])


def dump_model(name, module, positive_index=1):
    model_path = module.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[{name}] model missing: {model_path} -- run scripts\\get-deepfake-models.ps1")
        return False

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    items = module.build_feeds(sess)

    synthetic = False
    if not items and name == "fakeaudio":
        # No labeled clips on this box (gitignored). Emit one synthetic window so the
        # model is still load/latency-benchmarkable here; correctness stays blank but
        # the CPU probability is recorded so cross-EP numerical agreement still works.
        rng = np.random.default_rng(0)
        items = [{
            "name": "synthetic_0",
            "label": "unknown",
            "feed": {"input": rng.uniform(-1, 1, size=(1, WINDOW_SAMPLES)).astype(np.float32)},
        }]
        synthetic = True

    if not items:
        print(f"[{name}] no feeds to dump (no labeled data present); skipped")
        return False

    out_dir = os.path.join(FIX_ROOT, name)
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(out_dir, "model.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join([
            model_path, str(positive_index), str(module.THRESHOLD),
            module.POSITIVE_LABEL, module.NEGATIVE_LABEL,
        ]) + "\n")

    samples_rows, inputs_rows = [], []
    for item in items:
        sid = item["name"]
        outputs = sess.run(None, item["feed"])
        p = softmax_pos(outputs, positive_index)
        pred = module.POSITIVE_LABEL if p > module.THRESHOLD else module.NEGATIVE_LABEL
        samples_rows.append("\t".join([sid, item["label"], pred, f"{p:.9f}"]))

        for in_name, arr in item["feed"].items():
            arr = np.ascontiguousarray(arr)
            token = DTYPE_TOKEN.get(arr.dtype)
            if token is None:
                raise SystemExit(f"[{name}] unsupported dtype {arr.dtype} for input {in_name}")
            fname = f"{sid}__{in_name}.bin"
            arr.tofile(os.path.join(data_dir, fname))
            shape = ",".join(str(d) for d in arr.shape)
            inputs_rows.append("\t".join([sid, in_name, token, shape, fname]))

    with open(os.path.join(out_dir, "samples.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("sample_id\tlabel\texpected_pred\texpected_p\n")
        f.write("\n".join(samples_rows) + "\n")
    with open(os.path.join(out_dir, "inputs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("sample_id\tinput_name\tdtype\tshape\tfile\n")
        f.write("\n".join(inputs_rows) + "\n")

    tag = " (synthetic, latency-only)" if synthetic else ""
    print(f"[{name}] {len(items)} sample(s){tag} -> {out_dir}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tsc", "fakeaudio"],
                    choices=["tsc", "fakeaudio"])
    args = ap.parse_args()

    mods = {
        "tsc": "validate_tsc_onnx",
        "fakeaudio": "validate_fakeaudio_onnx",
    }
    any_ok = False
    for name in args.models:
        module = importlib.import_module(mods[name])
        any_ok |= dump_model(name, module)
    if any_ok:
        print(f"\nFixtures under: {FIX_ROOT}")
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
