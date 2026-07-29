#!/usr/bin/env python3
"""Compare WER: static ONNX tiny-multi vs official whisper-tiny vs larger Whisper.

(1) Export/decode fidelity: openai/whisper-tiny Transformers generate()
(3) Quality ceiling: openai/whisper-small (default) on the same clips

Uses full audio up to 30s (Whisper canvas), same eval.jsonl refs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import WhisperTokenizer
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parents[2]
ONNX = ROOT / "artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s"
EVAL = ROOT / "src/workloads/eval/eval.jsonl"
SPEECH = ROOT / "artifacts/workloads/speech"

ENC_SEQ, D_MODEL, MAXLEN = 1500, 384, 128
FULL_SAMPLES, SR = 480000, 16000


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
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev = dp[0]
        dp[0] = i
        for j, hw in enumerate(h, 1):
            cur = dp[j]
            dp[j] = prev if rw == hw else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / len(r)


def load_clips() -> list[dict]:
    clips = []
    for line in EVAL.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = ROOT / str(row["audio"]).replace("\\", "/")
        if not path.is_file():
            path = SPEECH / Path(row["audio"]).name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        clips.append(
            {
                "id": row["id"],
                "path": path,
                "ref": row["ref"],
                "duration_s": float(row.get("duration_s", 0.0)),
            }
        )
    return clips


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        t = np.linspace(0, len(wav) / sr, int(len(wav) * SR / sr), endpoint=False)
        wav = np.interp(t, np.arange(len(wav)) / sr, wav).astype(np.float32)
    return wav[:FULL_SAMPLES]


def transcribe_onnx(wav: np.ndarray, enc, dec, fe, tok, sot_ids, eos, pad) -> tuple[str, float]:
    feats = fe(wav, sampling_rate=SR, return_tensors="np").input_features.astype(np.float32)
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
    return tok.decode(gen, skip_special_tokens=True).strip(), ms


@torch.inference_mode()
def transcribe_hf(
    wav: np.ndarray,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: torch.device,
) -> tuple[str, float]:
    inputs = processor(wav, sampling_rate=SR, return_tensors="pt")
    feats = inputs.input_features.to(device)
    t0 = time.perf_counter()
    # English corpus: both sides pin <|en|> so the comparison isolates the graph,
    # not language detection. The product path detects the language per clip.
    forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    out = model.generate(
        feats,
        forced_decoder_ids=forced,
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,  # greedy — fair vs static ONNX greedy
    )
    ms = (time.perf_counter() - t0) * 1000.0
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text, ms


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--larger",
        default="openai/whisper-small",
        help="HF id for quality-ceiling model (default: openai/whisper-small)",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"torch_device={device} larger={args.larger}")

    clips = load_clips()
    print(f"clips={len(clips)}")

    # --- ONNX static tiny multi ---
    fe = WhisperFeatureExtractor.from_pretrained(str(ONNX))
    tok = WhisperTokenizer.from_pretrained(str(ONNX))
    gc = json.loads((ONNX / "generation_config.json").read_text(encoding="utf-8"))
    sot = int(gc.get("decoder_start_token_id", 50258))
    eos = int(gc.get("eos_token_id", 50257))
    pad = int(gc.get("pad_token_id", eos))
    nots = int(gc.get("no_timestamps_token_id", 50363))
    sot_ids = [sot, 50259, 50359, nots]
    enc = ort.InferenceSession(
        str(ONNX / "encoder_model.onnx"), providers=["CPUExecutionProvider"]
    )
    dec = ort.InferenceSession(
        str(ONNX / "decoder_model.onnx"), providers=["CPUExecutionProvider"]
    )

    # --- HF tiny + larger ---
    print("loading openai/whisper-tiny ...")
    tiny_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    tiny_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(
        device
    ).eval()
    print(f"loading {args.larger} ...")
    big_proc = WhisperProcessor.from_pretrained(args.larger)
    big_model = WhisperForConditionalGeneration.from_pretrained(args.larger).to(device).eval()

    rows = []
    w_onnx: list[float] = []
    w_tiny: list[float] = []
    w_big: list[float] = []

    print()
    hdr = f"{'id':8s} {'dur':>5s} {'onnx%':>7s} {'tiny%':>7s} {'big%':>7s}  onnx | tiny | big"
    print(hdr)
    print("-" * len(hdr))

    for clip in clips:
        wav = load_wav(clip["path"])
        hyp_o, ms_o = transcribe_onnx(wav, enc, dec, fe, tok, sot_ids, eos, pad)
        hyp_t, ms_t = transcribe_hf(wav, tiny_model, tiny_proc, device)
        hyp_b, ms_b = transcribe_hf(wav, big_model, big_proc, device)
        wo, wt, wb = wer(clip["ref"], hyp_o), wer(clip["ref"], hyp_t), wer(clip["ref"], hyp_b)
        w_onnx.append(wo)
        w_tiny.append(wt)
        w_big.append(wb)
        print(
            f"{clip['id']:8s} {clip['duration_s']:5.2f} "
            f"{100*wo:6.1f}% {100*wt:6.1f}% {100*wb:6.1f}%  "
            f"{hyp_o[:40]!r} | {hyp_t[:40]!r} | {hyp_b[:40]!r}"
        )
        rows.append(
            {
                "id": clip["id"],
                "duration_s": clip["duration_s"],
                "ref": clip["ref"],
                "onnx": {"hyp": hyp_o, "wer": round(wo, 4), "ms": round(ms_o, 1)},
                "hf_tiny": {"hyp": hyp_t, "wer": round(wt, 4), "ms": round(ms_t, 1)},
                "hf_larger": {
                    "model": args.larger,
                    "hyp": hyp_b,
                    "wer": round(wb, 4),
                    "ms": round(ms_b, 1),
                },
            }
        )

    mo, mt, mb = mean(w_onnx), mean(w_tiny), mean(w_big)
    print()
    print(f"mean_wer  onnx_static_tiny_multi={100*mo:.1f}%")
    print(f"mean_wer  hf_whisper_tiny_greedy={100*mt:.1f}%")
    print(f"mean_wer  hf_{args.larger.split('/')[-1]}_greedy={100*mb:.1f}%")
    print(f"delta     onnx - hf_tiny = {100*(mo-mt):+.1f} pp  (fidelity; ~0 = good export)")
    print(f"delta     onnx - larger  = {100*(mo-mb):+.1f} pp  (tiny vs ceiling)")

    out = ROOT / "results" / "reports" / "whisper-static-multi-7s-vs-hf-refs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "torch_device": str(device),
        "larger_model": args.larger,
        "window": "full_30s",
        "decode": "greedy",
        "mean_wer": {
            "onnx_static_tiny_multi": mo,
            "hf_whisper_tiny": mt,
            "hf_larger": mb,
        },
        "delta_pp": {
            "onnx_minus_hf_tiny": (mo - mt) * 100,
            "onnx_minus_larger": (mo - mb) * 100,
        },
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)

    md = ROOT / "results" / "reports" / "whisper-static-multi-7s-vs-hf-refs.md"
    lines = [
        "# WER: static ONNX tiny-multi vs HF Whisper refs",
        "",
        f"- Device: `{device}`",
        f"- Window: full audio ≤30s",
        f"- Decode: greedy (`num_beams=1`) for HF; static-no-KV greedy for ONNX",
        f"- Larger model: `{args.larger}`",
        "",
        "| System | Mean WER |",
        "|--------|----------|",
        f"| ONNX `static-onnx-tiny-multi-7s` | {100*mo:.1f}% |",
        f"| HF `openai/whisper-tiny` | {100*mt:.1f}% |",
        f"| HF `{args.larger}` | {100*mb:.1f}% |",
        "",
        f"Fidelity gap (ONNX − tiny): **{100*(mo-mt):+.1f} pp**",
        f"Ceiling gap (ONNX − larger): **{100*(mo-mb):+.1f} pp**",
        "",
        "| id | dur | onnx | tiny | larger |",
        "|----|-----|------|------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['duration_s']:.2f} | "
            f"{100*r['onnx']['wer']:.1f}% | {100*r['hf_tiny']['wer']:.1f}% | "
            f"{100*r['hf_larger']['wer']:.1f}% |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
