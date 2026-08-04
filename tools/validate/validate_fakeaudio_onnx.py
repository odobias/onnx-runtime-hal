#!/usr/bin/env python3
"""Correctness (not just latency) check for the FakeAudio / Generated Audio
Detector ONNX model.

Primary labeled set (preferred):
  artifacts/workloads/classifiers/audio-samples/test_audio/{real,synthetic}/*.wav
  — the export-smoke kit from wanna-deepfk-ai (op_export.py).
  Fetch with: tools/fetch/get-fakeaudio-test-audio.ps1

Legacy fallback (flat dir, YouTube efficacy-review clips):
  artifacts/workloads/classifiers/audio-samples/*.wav with stem labels below.
  Biased (production-flagged cases); keep only if test_audio is absent.

This is still a small smoke / fixture-bake set, not a certified accuracy
number. GCS msclap_2023 chunks (Stage B) are the next step once bucket ACL
lands.

Gates (default --strict):
  * every clip's argmax/threshold decision matches the folder/stem label
  * optional --compare-model: max |p - p_ref| and zero decision flips

    .venv\\Scripts\\python.exe tools\\validate\\validate_fakeaudio_onnx.py
    .venv\\Scripts\\python.exe tools\\validate\\validate_fakeaudio_onnx.py --compare-model path\\to\\variant.onnx
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "research"))

CLASSIFIERS = os.path.join(ROOT, "artifacts", "workloads", "classifiers")
MODEL_PATH = os.path.join(CLASSIFIERS, "fakeaudio", "model.onnx")
SAMPLES_DIR = os.path.join(CLASSIFIERS, "audio-samples")
TEST_AUDIO_DIR = os.path.join(SAMPLES_DIR, "test_audio")
WINDOW_SAMPLES = 308700  # 7.0s @ 44.1kHz, static ONNX input

# Legacy flat-file stems (YouTube efficacy review). Unused when test_audio exists.
LEGACY_LABELS = {
    "real_1": "real", "real_2": "real", "real_3": "real",
    "deepfake_1": "deepfake", "deepfake_2": "deepfake",
}

POSITIVE_LABEL = "deepfake"
NEGATIVE_LABEL = "real"
THRESHOLD = 0.5
DEFAULT_MAX_ABS_P_DIFF = 0.05


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_window(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 44100:
        raise ValueError(f"{path}: expected 44100 Hz, got {sr}")
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    return audio[:WINDOW_SAMPLES].astype(np.float32)


def _discover_test_audio(root: str) -> list[dict]:
    items = []
    mapping = (("real", NEGATIVE_LABEL), ("synthetic", POSITIVE_LABEL))
    for folder, label in mapping:
        for wav in sorted(glob.glob(os.path.join(root, folder, "*.wav"))):
            stem = os.path.splitext(os.path.basename(wav))[0]
            # Stable sample ids for fixture baking: real_1 .. deepfake_5
            prefix = "real" if label == NEGATIVE_LABEL else "deepfake"
            items.append({
                "name": f"{prefix}_{stem}",
                "label": label,
                "path": wav,
                "source": "test_audio",
            })
    return items


def _discover_legacy_flat(root: str) -> list[dict]:
    items = []
    for wav in sorted(glob.glob(os.path.join(root, "*.wav"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        if stem not in LEGACY_LABELS:
            continue
        items.append({
            "name": stem,
            "label": LEGACY_LABELS[stem],
            "path": wav,
            "source": "legacy-youtube",
        })
    return items


def discover_samples() -> tuple[list[dict], str]:
    if os.path.isdir(TEST_AUDIO_DIR):
        items = _discover_test_audio(TEST_AUDIO_DIR)
        if items:
            return items, "test_audio"
    items = _discover_legacy_flat(SAMPLES_DIR)
    if items:
        return items, "legacy-youtube"
    return [], "none"


def build_feeds(sess=None):
    """Real labeled ONNX feeds: list of {name, label, feed}. `sess` ignored
    (uniform signature with the other classifiers' build_feeds)."""
    del sess
    items, _ = discover_samples()
    out = []
    for item in items:
        out.append({
            "name": item["name"],
            "label": item["label"],
            "feed": {"input": load_window(item["path"])[np.newaxis, :]},
        })
    return out


def predict(outputs) -> tuple[str, float]:
    p_fake = float(softmax(outputs[0])[0, 1])
    return (POSITIVE_LABEL if p_fake > THRESHOLD else NEGATIVE_LABEL), p_fake


def _session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_PATH, help="Primary FakeAudio ONNX (FP32 whole graph)")
    ap.add_argument("--compare-model", default="",
                    help="Optional second ONNX; fail if |Δp| or decision flips exceed gates")
    ap.add_argument("--max-abs-p-diff", type=float, default=DEFAULT_MAX_ABS_P_DIFF,
                    help="Max allowed |p_primary - p_compare| when --compare-model is set")
    ap.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                    help="Exit non-zero on label misses / compare-gate failures (default: true)")
    args = ap.parse_args()

    from _models import ensure_deepfake_models
    ensure_deepfake_models(["fakeaudio"])

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model} (run tools\\fetch\\get-classifier-models.ps1 first)")
        return 1

    samples, source = discover_samples()
    if not samples:
        print(f"No labeled sample audio under {SAMPLES_DIR}")
        print("  Fetch the balanced set:  .\\tools\\fetch\\get-fakeaudio-test-audio.ps1")
        return 1

    sess = _session(args.model)
    compare = _session(args.compare_model) if args.compare_model else None

    print(f"source={source}  n={len(samples)}  model={args.model}")
    if compare:
        print(f"compare={args.compare_model}  max|Δp|<={args.max_abs_p_diff}")
    print(f"{'label':9s} {'p(fake)':>8s}  {'Δp':>8s}  result  id")
    print("-" * 64)

    label_ok = 0
    label_miss = 0
    flip = 0
    max_dp = 0.0
    for sample in samples:
        feed = {"input": load_window(sample["path"])[np.newaxis, :]}
        pred, p_fake = predict(sess.run(None, feed))
        ok = pred == sample["label"]
        if ok:
            label_ok += 1
        else:
            label_miss += 1

        dp_s = ""
        if compare is not None:
            pred2, p2 = predict(compare.run(None, feed))
            dp = abs(p_fake - p2)
            max_dp = max(max_dp, dp)
            dp_s = f"{dp:8.4f}"
            if pred2 != pred:
                flip += 1
        else:
            dp_s = f"{'':>8s}"

        tag = "OK" if ok else "MISS"
        print(f"{sample['label']:9s} {p_fake:8.4f}  {dp_s}  [{tag}]  {sample['name']}")

    print(f"\n{label_ok}/{len(samples)} label-correct (source={source})")
    if compare is not None:
        print(f"compare: max|Δp|={max_dp:.4f}  decision_flips={flip}")

    failed = False
    if label_miss:
        print(f"FAIL: {label_miss} label miss(es)")
        failed = True
    if compare is not None:
        if flip:
            print(f"FAIL: {flip} decision flip(s) vs compare model")
            failed = True
        if max_dp > args.max_abs_p_diff:
            print(f"FAIL: max|Δp|={max_dp:.4f} > {args.max_abs_p_diff}")
            failed = True

    if failed and args.strict:
        return 1
    if failed:
        print("WARN: failures ignored because --no-strict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
