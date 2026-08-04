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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402  (local helpers, need the path insert above)
from providers import DEVICE_PROVIDERS, pick_providers  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s"
EVAL = ROOT / "src/workloads/eval/eval.jsonl"
SPEECH_DIR = ROOT / "artifacts/workloads/speech"
JFK = SPEECH_DIR / "jfk.wav"

N_FRAMES, ENC_SEQ, D_MODEL, MAXLEN = 3000, 1500, 384, 128
PRODUCT_SAMPLES, FULL_SAMPLES, SR = 112000, 480000, 16000

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
    prefix: "Prefix",
    eos: int,
    pad: int,
) -> tuple[str, int, float, str]:
    feats = fe(wav, sampling_rate=SR, return_tensors="np").input_features.astype(
        np.float32
    )
    t0 = time.perf_counter()
    ehs = enc.run(None, {"input_features": feats})[0]
    assert tuple(ehs.shape) == (1, ENC_SEQ, D_MODEL), ehs.shape
    lang, lang_id = prefix.resolve(dec, ehs, pad)
    sot_ids = [prefix.sot, lang_id, prefix.task, prefix.nots]
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
    return text, len(gen), ms, lang


@dataclass
class Prefix:
    """Decoder prompt: <|sot|> <|lang|> <|transcribe|> <|notimestamps|>.

    lang_id is None in auto mode, where the language is detected per clip the same
    way the C++ static engine does it -- one decoder step after <|sot|>, argmax over
    the <|xx|> tokens. Pinning <|en|> on non-English speech makes Whisper translate.
    """

    sot: int
    task: int
    nots: int
    lang_to_id: dict[str, int]
    lang: str | None = None

    def resolve(
        self, dec: ort.InferenceSession, ehs: np.ndarray, pad: int
    ) -> tuple[str, int]:
        if self.lang is not None:
            return self.lang, self.lang_to_id[self.lang]
        ids = np.full((1, MAXLEN), pad, dtype=np.int64)
        ids[0, 0] = self.sot
        row = dec.run(None, {"input_ids": ids, "encoder_hidden_states": ehs})[0][0, 0]
        codes = list(self.lang_to_id)
        ids_arr = np.fromiter(self.lang_to_id.values(), dtype=np.int64)
        best = int(row[ids_arr].argmax())
        return codes[best], int(ids_arr[best])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=sorted(DEVICE_PROVIDERS), default="cpu")
    ap.add_argument("--provider", default=None)
    ap.add_argument(
        "--window",
        choices=["product", "full"],
        default="product",
        # Note this is NOT what the shipping runner calls "product": this script keeps
        # only the first 7s of a clip (hence --only-fit), whereas the runner cuts long
        # audio into overlapping 7s windows and stitches them. Use
        # tools/validate/run-whisper-tasks.ps1 to measure the shipping behaviour.
        help="product=first 7s only (112k); full=up to 30s canvas (480k)",
    )
    ap.add_argument("--max-wer", type=float, default=0.35)
    ap.add_argument(
        "--lang-mode",
        default="auto",
        help="auto = detect per clip (product path); or an ISO code to pin, e.g. en",
    )
    ap.add_argument("--no-jfk", action="store_true")
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record this run as the reference other runners compare against",
    )
    ap.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
    ap.add_argument(
        "--baseline-tol",
        type=float,
        default=0.02,
        help="Allowed mean-WER regression vs baseline, absolute (default 0.02 = 2 pts)",
    )
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
    task_to_id = gc.get("task_to_id") or {}
    lang_to_id = {
        code.strip("<|>"): int(tid) for code, tid in (gc.get("lang_to_id") or {}).items()
    }
    prefix = Prefix(
        sot=sot,
        task=int(task_to_id.get("transcribe", 50359)),
        nots=nots,
        lang_to_id=lang_to_id,
        lang=None if args.lang_mode == "auto" else args.lang_mode,
    )
    if prefix.lang is not None and prefix.lang not in lang_to_id:
        print(f"FAIL: --lang-mode {prefix.lang} is not a language of this model", file=sys.stderr)
        return 2

    clips = load_clips(include_jfk=not args.no_jfk)
    rows = []
    scored_wer: list[float] = []
    fail = False

    print()
    print(
        f"{'id':8s} {'dur':>5s} {'lang':4s} {'tok':>4s} {'ms':>7s} {'wer%':>6s}  hyp"
    )
    n_en = 0
    for clip in clips:
        if args.only_fit and args.window == "product" and clip.duration_s > 7.01:
            print(f"{clip.id:8s} {clip.duration_s:5.2f}  SKIP (>7s, --only-fit)")
            continue
        wav, _ = load_wav(clip.path, max_samples)
        hyp, ntok, ms, lang = transcribe(enc, dec, fe, tok, wav, prefix, eos, pad)
        if lang == "en":
            n_en += 1
        # Truncated product window vs full ref is expected to look ugly on long clips.
        truncated = args.window == "product" and clip.duration_s > 7.01
        w = wer(clip.ref, hyp)
        gate = (not truncated) and (w > args.max_wer or ntok >= MAXLEN - 4)
        if gate:
            fail = True
        tag = "TRUNC" if truncated else ("FAIL" if gate else "ok")
        if not truncated:
            scored_wer.append(w)
        print(
            f"{clip.id:8s} {clip.duration_s:5.2f} {lang:4s} {ntok:4d} {ms:7.1f} "
            f"{100 * w:6.1f}  [{tag}] {hyp}"
        )
        rows.append(
            {
                "id": clip.id,
                "duration_s": clip.duration_s,
                "lang": lang,
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
        f"max_wer_gate={100 * args.max_wer:.0f}% lang_mode={args.lang_mode} "
        f"detected_en={n_en}/{len(rows)}"
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

    suite = f"english-{args.window}"
    scored_rows = [r for r in rows if not r["truncated"]]
    provider = enc.get_providers()[0]
    if args.update_baseline:
        path = baseline.update(suite, scored_rows, mean_wer, provider)
        print("updated baseline", path)
    elif not args.no_baseline:
        print()
        ok, lines = baseline.compare(suite, scored_rows, mean_wer, args.baseline_tol)
        for line in lines:
            print(line)
        if not ok:
            fail = True

    if fail or not scored_wer:
        print("FAIL", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
