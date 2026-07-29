#!/usr/bin/env python3
"""Smoke: 7s of jfk.wav through static-onnx-tiny-multi-7s (30s mel canvas).

Supports CPU / DirectML GPU / NPU EPs. Exit 0 on a recognizable JFK transcript.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperTokenizer
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s"
AUDIO = ROOT / "artifacts/workloads/audio/jfk.wav"

N_FRAMES, ENC_SEQ, D_MODEL, MAXLEN = 3000, 1500, 384, 128
PRODUCT_SAMPLES, SR = 112000, 16000

DEVICE_PROVIDERS: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "gpu": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "npu": [
        "VitisAIExecutionProvider",
        "QNNExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ],
}


def _pick_providers(device: str, explicit: str | None) -> list[str]:
    available = set(ort.get_available_providers())
    if explicit:
        if explicit not in available:
            raise SystemExit(
                f"Provider {explicit!r} not available. Have: {sorted(available)}"
            )
        return [explicit]
    wanted = DEVICE_PROVIDERS[device]
    chosen = [p for p in wanted if p in available]
    if not chosen:
        raise SystemExit(
            f"No providers for device={device}. Wanted {wanted}; have {sorted(available)}"
        )
    # For gpu/npu, refuse silent CPU fallback when the preferred EP is missing.
    if device == "gpu" and "DmlExecutionProvider" not in chosen:
        raise SystemExit(
            "GPU requested but DmlExecutionProvider missing. "
            f"Install onnxruntime-directml. Available: {sorted(available)}"
        )
    if device == "npu" and chosen == ["CPUExecutionProvider"]:
        raise SystemExit(
            "NPU requested but no NPU EP registered "
            f"(VitisAI/QNN/OpenVINO). Available: {sorted(available)}"
        )
    return chosen


def _load_wav() -> np.ndarray:
    wav, sr = sf.read(str(AUDIO), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        t = np.linspace(0, len(wav) / sr, int(len(wav) * SR / sr), endpoint=False)
        wav = np.interp(t, np.arange(len(wav)) / sr, wav).astype(np.float32)
    return wav[:PRODUCT_SAMPLES]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--device",
        choices=sorted(DEVICE_PROVIDERS),
        default="cpu",
        help="Logical device (selects preferred ORT providers)",
    )
    ap.add_argument(
        "--provider",
        default=None,
        help="Exact ORT provider name (overrides --device preference list)",
    )
    args = ap.parse_args()

    if not MODEL.is_dir():
        print(f"FAIL: model package missing: {MODEL}", file=sys.stderr)
        return 2
    if not AUDIO.is_file():
        print(f"FAIL: audio missing: {AUDIO}", file=sys.stderr)
        return 2

    pkg = json.loads((MODEL / "npu_hal_package.json").read_text(encoding="utf-8"))
    assert pkg["n_frames"] == N_FRAMES
    assert int(pkg.get("product_window_s", 7)) == 7

    providers = _pick_providers(args.device, args.provider)
    print("ort", ort.__version__)
    print("available", ort.get_available_providers())
    print("requested_device", args.device)
    print("session_providers", providers)

    wav = _load_wav()
    fe = WhisperFeatureExtractor.from_pretrained(str(MODEL))
    feats = fe(wav, sampling_rate=SR, return_tensors="np").input_features.astype(
        np.float32
    )

    t0 = time.perf_counter()
    enc = ort.InferenceSession(str(MODEL / "encoder_model.onnx"), providers=providers)
    dec = ort.InferenceSession(str(MODEL / "decoder_model.onnx"), providers=providers)
    print("encoder_active", enc.get_providers())
    print("decoder_active", dec.get_providers())

    ehs = enc.run(None, {"input_features": feats})[0]
    assert tuple(ehs.shape) == (1, ENC_SEQ, D_MODEL), ehs.shape

    gc = json.loads((MODEL / "generation_config.json").read_text(encoding="utf-8"))
    sot = int(gc.get("decoder_start_token_id", 50258))
    eos = int(gc.get("eos_token_id", 50257))
    pad = int(gc.get("pad_token_id", eos))
    nots = int(gc.get("no_timestamps_token_id", 50363))

    ids = np.full((1, MAXLEN), pad, dtype=np.int64)
    for i, t in enumerate([sot, 50259, 50359, nots]):
        ids[0, i] = t
    cur = 4
    gen: list[int] = []
    while cur < MAXLEN:
        logits = dec.run(None, {"input_ids": ids, "encoder_hidden_states": ehs})[0]
        tok = int(logits[0, cur - 1].argmax())
        if tok == eos:
            break
        gen.append(tok)
        ids[0, cur] = tok
        cur += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    text = (
        WhisperTokenizer.from_pretrained(str(MODEL))
        .decode(gen, skip_special_tokens=True)
        .strip()
    )
    print(f"latency_ms: {elapsed_ms:.1f}")
    print("tokens:", len(gen))
    print("text:", text)

    # GPU/NPU must actually bind the preferred EP (not silent CPU).
    if args.device == "gpu" and enc.get_providers()[0] != "DmlExecutionProvider":
        print("FAIL: encoder did not bind DmlExecutionProvider", file=sys.stderr)
        return 1
    if args.device == "npu" and enc.get_providers()[0] == "CPUExecutionProvider":
        print("FAIL: encoder fell back to CPU", file=sys.stderr)
        return 1

    if "fellow" not in text.lower() and "american" not in text.lower():
        print("FAIL: unexpected transcript", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
