#!/usr/bin/env python3
"""Build the product Whisper-tiny multilingual static ONNX package.

Product Media Scan always feeds 7s of audio. Whisper's public weights are trained
on a 30s mel canvas; naively truncating the encoder to 700 mel frames / 350
encoder steps causes severe decode repetition (verified with both Torch and ONNX
encoders + stock decoder). True shape reduction needs fine-tuning — out of scope.

This export therefore:
  * ships onnx-community/whisper-tiny encoder+decoder (multilingual, static-friendly)
  * pins decoder input_ids to [1,128] for the HAL static-no-KV loop
  * records product_window_s=7 so runtimes pad 7s PCM to the 30s mel canvas

Output: artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download
from onnx import shape_inference

WINDOW_S = 7
PAD_S = 30
SAMPLE_RATE = 16000
N_SAMPLES = PAD_S * SAMPLE_RATE  # 480000 graph canvas
PRODUCT_SAMPLES = WINDOW_S * SAMPLE_RATE  # 112000
N_MELS = 80
N_FRAMES = N_SAMPLES // 160  # 3000
ENC_SEQ = N_FRAMES // 2  # 1500
D_MODEL = 384
MAXLEN = 128
ONNX_HUB = "onnx-community/whisper-tiny"
HUB = "openai/whisper-tiny"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rewrite_input_dims(model: onnx.ModelProto, name: str, dims: list[int]) -> None:
    for inp in model.graph.input:
        if inp.name != name:
            continue
        shape = inp.type.tensor_type.shape
        while len(shape.dim) < len(dims):
            shape.dim.add()
        for i, v in enumerate(dims):
            d = shape.dim[i]
            d.Clear()
            d.dim_value = int(v)
        return
    raise KeyError(name)


def build(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Snapshot {ONNX_HUB} ...", flush=True)
    snap = Path(snapshot_download(ONNX_HUB))

    # Tokenizer / configs at package root
    for name in [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "normalizer.json",
        "README.md",
    ]:
        src = snap / name
        if src.exists():
            shutil.copy2(src, dst / name)

    # Prefer openai generation_config (richer special-token fields) when present
    # on the hub cache from prior downloads; otherwise keep community copy.
    try:
        gc = hf_hub_download(HUB, "generation_config.json")
        shutil.copy2(gc, dst / "generation_config.json")
    except Exception as e:
        print(f"generation_config from {HUB} skipped: {e}", flush=True)

    enc_src = snap / "onnx" / "encoder_model.onnx"
    shutil.copy2(enc_src, dst / "encoder_model.onnx")

    print("Pinning decoder input_ids to [1,128] ...", flush=True)
    dec = onnx.load(str(snap / "onnx" / "decoder_model.onnx"))
    _rewrite_input_dims(dec, "input_ids", [1, MAXLEN])
    # encoder_hidden_states stays dynamic or 1500 — force static 1500
    _rewrite_input_dims(dec, "encoder_hidden_states", [1, ENC_SEQ, D_MODEL])
    if dec.graph.output and dec.graph.output[0].name != "logits":
        old = dec.graph.output[0].name
        dec.graph.output[0].name = "logits"
        for node in dec.graph.node:
            for i, n in enumerate(node.output):
                if n == old:
                    node.output[i] = "logits"
    try:
        dec = shape_inference.infer_shapes(dec)
    except Exception as e:
        print(f"shape_inference warning: {e}", flush=True)
    onnx.save(dec, str(dst / "decoder_model.onnx"))

    # Force encoder mel input static too
    enc = onnx.load(str(dst / "encoder_model.onnx"))
    try:
        _rewrite_input_dims(enc, "input_features", [1, N_MELS, N_FRAMES])
        enc = shape_inference.infer_shapes(enc)
        onnx.save(enc, str(dst / "encoder_model.onnx"))
    except Exception as e:
        print(f"encoder pin warning: {e}", flush=True)

    pkg = {
        "format": "onnx-static-no-kv",
        "hub_model": HUB,
        "onnx_source": ONNX_HUB,
        "multilingual": True,
        "product_window_s": WINDOW_S,
        "product_n_samples": PRODUCT_SAMPLES,
        "pad_to_s": PAD_S,
        "window_s": PAD_S,
        "sample_rate": SAMPLE_RATE,
        "n_samples": N_SAMPLES,
        "n_mels": N_MELS,
        "n_frames": N_FRAMES,
        "enc_seq": ENC_SEQ,
        "d_model": D_MODEL,
        "max_tokens": MAXLEN,
        "note": (
            "Product feeds 7s PCM; runtime pads to 30s mel canvas. "
            "Truncating the encoder to 700 frames without fine-tuning causes decode loops."
        ),
        "encoder_file": "encoder_model.onnx",
        "decoder_file": "decoder_model.onnx",
        "encoder_input": "input_features",
        "encoder_output": "last_hidden_state",
        "decoder_inputs": ["input_ids", "encoder_hidden_states"],
        "decoder_output": "logits",
    }
    (dst / "npu_hal_package.json").write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(pkg, indent=2), flush=True)


def smoke_cpu(dst: Path) -> None:
    import soundfile as sf
    from transformers import WhisperTokenizer
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

    audio = _repo_root() / "artifacts/workloads/audio/jfk.wav"
    wav, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    clip = wav[:PRODUCT_SAMPLES] if sr == SAMPLE_RATE else wav
    fe = WhisperFeatureExtractor.from_pretrained(str(dst))
    feats = fe(clip, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features.astype(np.float32)
    enc = ort.InferenceSession(str(dst / "encoder_model.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(dst / "decoder_model.onnx"), providers=["CPUExecutionProvider"])
    ehs = enc.run(None, {"input_features": feats})[0]
    assert tuple(ehs.shape) == (1, ENC_SEQ, D_MODEL), ehs.shape
    ids = np.full((1, MAXLEN), 50257, dtype=np.int64)
    for i, t in enumerate([50258, 50259, 50359, 50363]):
        ids[0, i] = t
    cur = 4
    gen = []
    while cur < MAXLEN:
        logits = dec.run(None, {"input_ids": ids, "encoder_hidden_states": ehs})[0]
        tok = int(logits[0, cur - 1].argmax())
        if tok == 50257:
            break
        gen.append(tok)
        ids[0, cur] = tok
        cur += 1
    text = WhisperTokenizer.from_pretrained(str(dst)).decode(gen, skip_special_tokens=True).strip()
    print("CPU smoke:", text)
    if "fellow" not in text.lower() and "american" not in text.lower():
        raise SystemExit(f"unexpected transcript: {text!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()
    out = args.out or (_repo_root() / "artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s")
    build(out)
    if not args.skip_smoke:
        smoke_cpu(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
