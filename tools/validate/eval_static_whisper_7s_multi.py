#!/usr/bin/env python3
"""Multi-sample WER for static-onnx-tiny-multi-7s (CPU / DML GPU / NPU).

Reads src/workloads/eval/eval.jsonl (speech WAVs under artifacts/workloads/speech)
and scores greedy static-no-KV decode against references.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperTokenizer
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s"
EVAL = ROOT / "src/workloads/eval/eval.jsonl"
SPEECH_DIR = ROOT / "artifacts/workloads/speech"
JFK = SPEECH_DIR / "jfk.wav"

N_FRAMES, ENC_SEQ, D_MODEL, MAXLEN = 3000, 1500, 384, 128
PRODUCT_SAMPLES, FULL_SAMPLES, SR = 112000, 480000, 16000

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

JFK_REF = (
    "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU "
    "ASK WHAT YOU CAN DO FOR YOUR COUNTRY"
)


def normalize(text: str) -> str:
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def wer(ref: str, hyp: str) -> float:
    r = normalize(ref).split()
    h = normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # classic Levenshtein on tokens
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev = dp[0]
        dp[0] = i
        for j, hw in enumerate(h, 1):
            cur = dp[j]
            if rw == hw:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(r)


def pick_providers(device: str, explicit: str | None) -> list[str]:
    available = set(ort.get_available_providers())
    if explicit:
        if explicit not in available:
            raise SystemExit(f"Provider {explicit!r} missing; have {sorted(available)}")
        return [explicit]
    chosen = [p for p in DEVICE_PROVIDERS[device] if p in available]
    if device == "gpu" and "DmlExecutionProvider" not in chosen:
        raise SystemExit(f"DmlExecutionProvider missing; have {sorted(available)}")
    if device == "npu" and chosen == ["CPUExecutionProvider"]:
        raise SystemExit(f"No NPU EP; have {sorted(available)}")
    if not chosen:
        raise SystemExit(f"No providers for {device}; have {sorted(available)}")
    return chosen


def load_wav(path: Path, max_samples: int) -> tuple[np.ndarray, float]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        t = np.linspace(0, len(wav) / sr, int(len(wav) * SR / sr), endpoint=False)
        wav = np.interp(t, np.arange(len(wav)) / sr, wav).astype(np.float32)
    dur = len(wav) / SR
    return wav[:max_samples], dur


@dataclass
class Clip:
    id: str
    path: Path
    ref: str
    duration_s: float


def load_clips(include_jfk: bool) -> list[Clip]:
    clips: list[Clip] = []
    seen: set[str] = set()
    for line in EVAL.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = str(row["id"])
        if not include_jfk and cid == "jfk":
            continue
        rel = str(row["audio"]).replace("\\", "/")
        path = ROOT / rel
        if not path.is_file():
            path = SPEECH_DIR / Path(rel).name
        if not path.is_file():
            raise SystemExit(f"Missing audio {path}")
        clips.append(
            Clip(
                id=cid,
                path=path,
                ref=row["ref"],
                duration_s=float(row.get("duration_s", 0.0)),
            )
        )
        seen.add(cid)
    if include_jfk and "jfk" not in seen:
        if not JFK.is_file():
            raise SystemExit(f"Missing {JFK}")
        clips.insert(
            0,
            Clip(id="jfk", path=JFK, ref=JFK_REF, duration_s=11.0),
        )
    return clips


def transcribe(
    enc: ort.InferenceSession,
    dec: ort.InferenceSession,
    fe: WhisperFeatureExtractor,
    tok: WhisperTokenizer,
    wav: np.ndarray,
    sot_ids: list[int],
    eos: int,
    pad: int,
) -> tuple[str, int, float]:
    feats = fe(wav, sampling_rate=SR, return_tensors="np").input_features.astype(
        np.float32
    )
    t0 = time.perf_counter()
    ehs = enc.run(None, {"input_features": feats})[0]
    assert tuple(ehs.shape) == (1, ENC_SEQ, D_MODEL), ehs.shape
    ids = np.full((1, MAXLEN), pad, dtype=np.int64)
    for i, t in enumerate(sot_ids):
        ids[0, i] = t
    cur = len(sot_ids)
    gen: list[int] = []
    while cur < MAXLEN:
        logits = dec.run(None, {"input_ids": ids, "encoder_hidden_states": ehs})[0]
        nxt = int(logits[0, cur - 1].argmax())
        if nxt == eos:
            break
        gen.append(nxt)
        ids[0, cur] = nxt
        cur += 1
    ms = (time.perf_counter() - t0) * 1000.0
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text, len(gen), ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=sorted(DEVICE_PROVIDERS), default="cpu")
    ap.add_argument("--provider", default=None)
    ap.add_argument(
        "--window",
        choices=["product", "full"],
        default="product",
        help="product=7s truncate (112k); full=up to 30s canvas (480k)",
    )
    ap.add_argument("--max-wer", type=float, default=0.35)
    ap.add_argument("--no-jfk", action="store_true")
    ap.add_argument(
        "--only-fit",
        action="store_true",
        help="With --window product, skip clips longer than 7s (fair WER)",
    )
    args = ap.parse_args()

    max_samples = PRODUCT_SAMPLES if args.window == "product" else FULL_SAMPLES
    providers = pick_providers(args.device, args.provider)
    print("ort", ort.__version__)
    print("available", ort.get_available_providers())
    print("device", args.device, "providers", providers, "window", args.window)

    enc = ort.InferenceSession(str(MODEL / "encoder_model.onnx"), providers=providers)
    dec = ort.InferenceSession(str(MODEL / "decoder_model.onnx"), providers=providers)
    print("encoder_active", enc.get_providers())
    print("decoder_active", dec.get_providers())
    if args.device == "gpu" and enc.get_providers()[0] != "DmlExecutionProvider":
        print("FAIL: encoder not on DML", file=sys.stderr)
        return 1

    fe = WhisperFeatureExtractor.from_pretrained(str(MODEL))
    tok = WhisperTokenizer.from_pretrained(str(MODEL))
    gc = json.loads((MODEL / "generation_config.json").read_text(encoding="utf-8"))
    sot = int(gc.get("decoder_start_token_id", 50258))
    eos = int(gc.get("eos_token_id", 50257))
    pad = int(gc.get("pad_token_id", eos))
    nots = int(gc.get("no_timestamps_token_id", 50363))
    # <|en|><|transcribe|><|notimestamps|>
    sot_ids = [sot, 50259, 50359, nots]

    clips = load_clips(include_jfk=not args.no_jfk)
    rows = []
    scored_wer: list[float] = []
    fail = False

    print()
    print(
        f"{'id':8s} {'dur':>5s} {'tok':>4s} {'ms':>7s} {'wer%':>6s}  hyp"
    )
    for clip in clips:
        if args.only_fit and args.window == "product" and clip.duration_s > 7.01:
            print(f"{clip.id:8s} {clip.duration_s:5.2f}  SKIP (>7s, --only-fit)")
            continue
        wav, _ = load_wav(clip.path, max_samples)
        hyp, ntok, ms = transcribe(enc, dec, fe, tok, wav, sot_ids, eos, pad)
        # Truncated product window vs full ref is expected to look ugly on long clips.
        truncated = args.window == "product" and clip.duration_s > 7.01
        w = wer(clip.ref, hyp)
        gate = (not truncated) and (w > args.max_wer or ntok >= MAXLEN - len(sot_ids))
        if gate:
            fail = True
        tag = "TRUNC" if truncated else ("FAIL" if gate else "ok")
        if not truncated:
            scored_wer.append(w)
        print(
            f"{clip.id:8s} {clip.duration_s:5.2f} {ntok:4d} {ms:7.1f} "
            f"{100 * w:6.1f}  [{tag}] {hyp}"
        )
        rows.append(
            {
                "id": clip.id,
                "duration_s": clip.duration_s,
                "tokens": ntok,
                "latency_ms": round(ms, 1),
                "wer": round(w, 4),
                "truncated": truncated,
                "hyp": hyp,
                "ref": clip.ref,
                "pass": not gate,
            }
        )

    mean_wer = sum(scored_wer) / len(scored_wer) if scored_wer else 1.0
    print()
    print(
        f"scored_clips={len(scored_wer)} mean_wer={100 * mean_wer:.1f}% "
        f"max_wer_gate={100 * args.max_wer:.0f}%"
    )
    out = (
        ROOT
        / "results"
        / "reports"
        / f"whisper-static-multi-7s-eval-{args.device}-{args.window}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "device": args.device,
                "window": args.window,
                "providers": enc.get_providers(),
                "mean_wer": mean_wer,
                "max_wer_gate": args.max_wer,
                "pass": not fail and bool(scored_wer),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("wrote", out)
    if fail or not scored_wer:
        print("FAIL", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
