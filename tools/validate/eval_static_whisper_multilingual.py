#!/usr/bin/env python3
"""Multilingual smoke for static-onnx-tiny-multi-7s.

Reads artifacts/workloads/speech/multilingual/manifest.jsonl and runs greedy
static-no-KV decode. The language token is auto-detected the same way OpenAI's
whisper.detect_language() does (one decoder step after <|startoftranscript|>,
argmax restricted to the 99 <|xx|> tokens), so transcripts stay verbatim in the
spoken language instead of being paraphrased into English.

Task is always <|transcribe|>; <|translate|> is never used.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline  # noqa: E402  (local helpers, need the path insert above)
from providers import DEVICE_PROVIDERS, pick_providers  # noqa: E402

# wer must match the C++ harness that bakes the baselines; the soft diagnostics
# below must not, because normalize_text deletes the non-ASCII letters they
# exist to look for. See scoring.py.
from scoring import normalize_text as normalize  # noqa: E402
from scoring import normalize_unicode, wer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "artifacts/workloads/whisper/models/static-onnx-tiny-multi-7s"
MANIFEST = ROOT / "artifacts/workloads/speech/multilingual/manifest.jsonl"

ENC_SEQ, D_MODEL, MAXLEN = 1500, 384, 128
FULL_SAMPLES, PRODUCT_SAMPLES, SR = 480000, 112000, 16000


def content_overlap(ref: str, hyp: str) -> float:
    """Fraction of ref tokens (len>=3) that appear in hyp — soft multilingual check."""
    r = [t for t in normalize_unicode(ref).split() if len(t) >= 3]
    if not r:
        r = normalize_unicode(ref).split()
    if not r:
        return 1.0
    h = set(normalize_unicode(hyp).split())
    return sum(1 for t in r if t in h) / len(r)


def looks_non_english(hyp: str) -> bool:
    """True if hyp has letters outside basic English orthography (é, ř, ñ, …)."""
    return any(ord(ch) > 127 and ch.isalpha() for ch in hyp)


def char_recall(ref: str, hyp: str) -> float:
    """Crude char-set recall of alnum chars from ref present in hyp (order-free)."""
    r = [c for c in normalize_unicode(ref) if c.isalnum()]
    if not r:
        return 1.0
    h = set(normalize_unicode(hyp))
    return sum(1 for c in r if c in h) / len(r)


def detect_language(
    dec: ort.InferenceSession,
    ehs: np.ndarray,
    lang_to_id: dict[str, int],
    sot: int,
    pad: int,
) -> tuple[str, int, float]:
    """OpenAI-style language detection: argmax over <|xx|> tokens after SOT."""
    ids = np.full((1, MAXLEN), pad, dtype=np.int64)
    ids[0, 0] = sot
    logits = dec.run(None, {"input_ids": ids, "encoder_hidden_states": ehs})[0]
    row = logits[0, 0]
    ids_arr = np.fromiter(lang_to_id.values(), dtype=np.int64)
    codes = list(lang_to_id.keys())
    scores = row[ids_arr]
    best = int(scores.argmax())
    # softmax confidence over the language subset only
    shifted = scores - scores.max()
    probs = np.exp(shifted)
    conf = float(probs[best] / probs.sum())
    return codes[best], int(ids_arr[best]), conf


def load_wav(path: Path, max_samples: int = FULL_SAMPLES) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        t = np.linspace(0, len(wav) / sr, int(len(wav) * SR / sr), endpoint=False)
        wav = np.interp(t, np.arange(len(wav)) / sr, wav).astype(np.float32)
    return wav[:max_samples]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=sorted(DEVICE_PROVIDERS), default="cpu")
    ap.add_argument(
        "--provider",
        default=None,
        help="Exact ORT provider name (overrides --device preference list)",
    )
    ap.add_argument("--max-wer", type=float, default=0.55)
    ap.add_argument("--min-overlap", type=float, default=0.30)
    ap.add_argument(
        "--window",
        choices=["full", "product"],
        default="full",
        help="product=7s truncate (112k samples, what AvastClient sends); full=30s canvas",
    )
    ap.add_argument(
        "--lang-mode",
        choices=["auto", "manifest"],
        default="auto",
        help="auto = detect language from audio (product path); manifest = trust the label",
    )
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
        "--en-control",
        action="store_true",
        help="Also decode Spanish with forced <|en|> (should degrade vs detected <|es|>)",
    )
    args = ap.parse_args()

    if not MANIFEST.is_file():
        print(f"FAIL: missing {MANIFEST}", file=sys.stderr)
        print("Run tools/tmp/fetch_multilingual_speech.py (or get-models after HF push).", file=sys.stderr)
        return 2

    tok = WhisperTokenizer.from_pretrained(str(MODEL))
    fe = WhisperFeatureExtractor.from_pretrained(str(MODEL))
    gc = json.loads((MODEL / "generation_config.json").read_text(encoding="utf-8"))
    sot = int(gc.get("decoder_start_token_id", 50258))
    eos = int(gc.get("eos_token_id", 50257))
    pad = int(gc.get("pad_token_id", eos))
    nots = int(gc.get("no_timestamps_token_id", 50363))
    task_to_id = gc.get("task_to_id") or {}
    transcribe = int(task_to_id.get("transcribe", tok.convert_tokens_to_ids("<|transcribe|>")))
    translate = int(task_to_id.get("translate", tok.convert_tokens_to_ids("<|translate|>")))
    if transcribe == translate:
        print("FAIL: transcribe/translate token ids collide", file=sys.stderr)
        return 2
    lang_to_id = {
        code.strip("<|>"): int(tid) for code, tid in (gc.get("lang_to_id") or {}).items()
    }
    if len(lang_to_id) < 2:
        print("FAIL: generation_config has no lang_to_id table", file=sys.stderr)
        return 2

    providers = pick_providers(args.device, args.provider)
    print(f"device {args.device} providers {providers} window {args.window}")
    enc = ort.InferenceSession(str(MODEL / "encoder_model.onnx"), providers=providers)
    dec = ort.InferenceSession(str(MODEL / "decoder_model.onnx"), providers=providers)
    print("encoder_active", enc.get_providers())
    print("decoder_active", dec.get_providers())

    rows_out = []
    fail = False
    n_lang_ok = 0
    print(
        f"{'id':12s} {'want':4s} {'got':4s} {'conf':>5s} {'wer%':>6s} "
        f"{'ovlp':>5s} {'crec':>5s}  hyp"
    )
    for line in MANIFEST.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = ROOT / row["audio"]
        if not path.is_file():
            print(f"FAIL missing {path}", file=sys.stderr)
            fail = True
            continue
        expected = row["lang"]
        if expected not in lang_to_id:
            print(f"FAIL unknown lang token {expected}", file=sys.stderr)
            fail = True
            continue
        max_samples = PRODUCT_SAMPLES if args.window == "product" else FULL_SAMPLES
        wav = load_wav(path, FULL_SAMPLES)
        # Product sends 7s; a clip cut mid-sentence cannot match the full reference,
        # so those clips are scored on language detection only.
        truncated = len(wav) > max_samples
        wav = wav[:max_samples]
        feats = fe(wav, sampling_rate=SR, return_tensors="np").input_features.astype(np.float32)
        t0 = time.perf_counter()
        ehs = enc.run(None, {"input_features": feats})[0]

        if args.lang_mode == "auto":
            lang, lang_id, conf = detect_language(dec, ehs, lang_to_id, sot, pad)
        else:
            lang, lang_id, conf = expected, lang_to_id[expected], float("nan")
        if lang == expected:
            n_lang_ok += 1
        sot_ids = [sot, lang_id, transcribe, nots]

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
        hyp = tok.decode(gen, skip_special_tokens=True).strip()
        w = wer(row["ref"], hyp)
        ov = content_overlap(row["ref"], hyp)
        cr = char_recall(row["ref"], hyp)
        # Pass if WER OK, token overlap OK, or non-English orthography + char recall
        # (tiny is noisy; empty / English-only garbage should still fail).
        ok = (
            w <= args.max_wer
            or (ov >= args.min_overlap and len(normalize(hyp).split()) >= 3)
            or (looks_non_english(hyp) and cr >= 0.55 and len(normalize(hyp).split()) >= 4)
        )
        # Verbatim requirement: never silently fall back to English on non-English
        # audio, since forcing <|en|> makes Whisper paraphrase instead of transcribe.
        lang_ok = lang == expected
        ok = (ok or truncated) and lang_ok
        if not ok:
            fail = True
        tag = "LANG" if not lang_ok else ("TRUNC" if truncated else "ok" if ok else "FAIL")
        conf_s = "  n/a" if conf != conf else f"{conf:5.2f}"
        print(
            f"{row['id']:12s} {expected:4s} {lang:4s} {conf_s} {100*w:5.1f}% "
            f"{ov:5.2f} {cr:5.2f}  [{tag}] {hyp}"
        )
        rows_out.append(
            {
                "id": row["id"],
                "lang_expected": expected,
                "lang_detected": lang,
                "lang_match": lang_ok,
                "lang_confidence": None if conf != conf else round(conf, 4),
                "ref": row["ref"],
                "hyp": hyp,
                "wer": round(w, 4),
                "overlap": round(ov, 4),
                "char_recall": round(cr, 4),
                "latency_ms": round(ms, 1),
                "pass": ok,
                "truncated": truncated,
                "lang_token_id": lang_id,
            }
        )

    if args.en_control:
        es = next((r for r in rows_out if r["lang_expected"] == "es"), None)
        if es is None:
            print("EN control skipped: no Spanish row", file=sys.stderr)
        else:
            # re-decode Spanish wav with <|en|>
            es_man = next(
                json.loads(l)
                for l in MANIFEST.read_text(encoding="utf-8-sig").splitlines()
                if l.strip() and json.loads(l).get("lang") == "es"
            )
            wav = load_wav(
                ROOT / es_man["audio"],
                PRODUCT_SAMPLES if args.window == "product" else FULL_SAMPLES,
            )
            feats = fe(wav, sampling_rate=SR, return_tensors="np").input_features.astype(
                np.float32
            )
            en_id = int(tok.convert_tokens_to_ids("<|en|>"))
            sot_ids = [sot, en_id, transcribe, nots]
            ehs = enc.run(None, {"input_features": feats})[0]
            ids = np.full((1, MAXLEN), pad, dtype=np.int64)
            for i, t in enumerate(sot_ids):
                ids[0, i] = t
            cur = len(sot_ids)
            gen = []
            while cur < MAXLEN:
                logits = dec.run(
                    None, {"input_ids": ids, "encoder_hidden_states": ehs}
                )[0]
                nxt = int(logits[0, cur - 1].argmax())
                if nxt == eos:
                    break
                gen.append(nxt)
                ids[0, cur] = nxt
                cur += 1
            hyp_en = tok.decode(gen, skip_special_tokens=True).strip()
            w_en = wer(es_man["ref"], hyp_en)
            print()
            print(
                f"CONTROL es+forced_en  wer={100*w_en:.1f}%  "
                f"(auto-detected {es['lang_detected']} was {100*es['wer']:.1f}%)"
            )
            print(f"  hyp_en: {hyp_en}")
            if w_en + 0.05 < es["wer"]:
                print(
                    "FAIL: forced English unexpectedly beat correct lang token",
                    file=sys.stderr,
                )
                fail = True
            elif es["wer"] == 0.0 and w_en > 0.15:
                print("CONTROL ok: wrong lang token hurts Spanish decode")
            else:
                print("CONTROL noted (see WER delta)")

    mean_w = sum(r["wer"] for r in rows_out) / len(rows_out) if rows_out else 1.0
    n_pass = sum(1 for r in rows_out if r["pass"])
    print()
    print(
        f"clips={len(rows_out)} pass={n_pass}/{len(rows_out)} "
        f"lang_detect={n_lang_ok}/{len(rows_out)} mean_wer={100*mean_w:.1f}% "
        f"lang_mode={args.lang_mode} task=transcribe"
    )

    stem = f"whisper-static-multi-7s-multilingual-{args.device}-{args.window}"
    out = ROOT / "results" / "reports" / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": str(MODEL),
                "device": args.device,
                "providers": enc.get_providers(),
                "window": args.window,
                "lang_mode": args.lang_mode,
                "task": "transcribe",
                "lang_detect_correct": n_lang_ok,
                "mean_wer": mean_w,
                "pass": not fail and n_pass == len(rows_out),
                "rows": rows_out,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    md = ROOT / "results" / "reports" / f"{stem}.md"
    lines = [
        f"# Multilingual smoke: static-onnx-tiny-multi-7s ({args.window} window)",
        "",
        "Source: FLEURS short clips under `artifacts/workloads/speech/multilingual/`.",
        "Decode: static-no-KV greedy, task `<|transcribe|>` (never `<|translate|>`).",
        f"Language token: `{args.lang_mode}`"
        + (
            " — detected from audio (argmax over the 99 `<|xx|>` tokens after "
            "`<|startoftranscript|>`), no label given to the model."
            if args.lang_mode == "auto"
            else " — taken from the manifest label."
        ),
        "",
        f"Pass: **{n_pass}/{len(rows_out)}** · language detected: "
        f"**{n_lang_ok}/{len(rows_out)}** · mean WER: **{100*mean_w:.1f}%**",
        "",
        "Clips cut by the 7s product window are marked `cut`: their WER is against the",
        "full reference, so it measures the missing tail, not a wrong language.",
        "",
        "| id | expected | detected | conf | WER | cut | hyp (verbatim) |",
        "|----|----------|----------|------|-----|-----|----------------|",
    ]
    for r in rows_out:
        hyp = r["hyp"].replace("|", "\\|")
        conf = "n/a" if r["lang_confidence"] is None else f"{r['lang_confidence']:.2f}"
        lines.append(
            f"| {r['id']} | {r['lang_expected']} | {r['lang_detected']} | {conf} | "
            f"{100*r['wer']:.1f}% | {'yes' if r['truncated'] else 'no'} | {hyp} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)
    print("wrote", md)

    suite = f"multilingual-{args.window}"
    provider = enc.get_providers()[0]
    if args.update_baseline:
        print("updated baseline", baseline.update(suite, rows_out, mean_w, provider))
    elif not args.no_baseline:
        print()
        ok, blines = baseline.compare(suite, rows_out, mean_w, args.baseline_tol)
        for line in blines:
            print(line)
        if not ok:
            fail = True

    if fail or not rows_out:
        print("FAIL", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
