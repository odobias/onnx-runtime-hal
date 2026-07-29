#!/usr/bin/env python3
"""Fetch a few short FLEURS clips (with refs) into artifacts/workloads/speech/multilingual/.

FLEURS is read-aloud FLoRes, so every utterance carries the FLoRes sentence `id` and
the same sentence exists in en_us. That parallel English text is recorded as `ref_en`,
which is what a <|translate|> run gets scored against -- without it a translation test
has nothing to compare to.
"""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/workloads/speech/multilingual"
# lang_code -> fleurs config name
LANGS = {
    "de": "de_de",
    "fr": "fr_fr",
    "es": "es_419",
    "cs": "cs_cz",
    "it": "it_it",
    "pl": "pl_pl",
}


def to_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    x = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def load_audio(entry) -> tuple[np.ndarray, int]:
    import soundfile as sf

    b = entry.get("bytes")
    if b:
        data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype("float32"), int(sr)


def english_refs(wanted: set[int]) -> dict[int, str]:
    """Map FLoRes sentence id -> English sentence, for translation references.

    Streamed and stopped as soon as every wanted id is seen: the en_us split is only
    needed for its text, so pulling the whole thing would be wasted bandwidth.
    """
    from datasets import Audio, load_dataset

    if not wanted:
        return {}
    found: dict[int, str] = {}
    ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    # Only the text is wanted; decoding the audio would drag in a codec stack.
    ds = ds.cast_column("audio", Audio(decode=False))
    for row in ds:
        sid = row.get("id")
        if sid not in wanted or sid in found:
            continue
        text = (row.get("raw_transcription") or row.get("transcription") or "").strip()
        if not text:
            continue
        found[sid] = text
        if len(found) == len(wanted):
            break
    return found


def main() -> int:
    import itertools

    from datasets import Audio, load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang, cfg in LANGS.items():
        print(f"== {lang} ({cfg}) ==", flush=True)
        # Streamed: a non-streaming load_dataset materialises train+validation+test for
        # the config -- gigabytes of audio per language to keep one short clip.
        ds = load_dataset("google/fleurs", cfg, split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        # Prefer clips ~3-8s with non-empty transcription
        picked = None
        for i, row in enumerate(itertools.islice(ds, 80)):
            text = (row.get("transcription") or row.get("raw_transcription") or "").strip()
            if not text or len(text.split()) < 4:
                continue
            try:
                samples, sr = load_audio(row["audio"])
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {i}: {exc}")
                continue
            if sr != 16000:
                idx = (np.arange(int(len(samples) * 16000 / sr)) * sr / 16000).astype(int)
                idx = np.clip(idx, 0, len(samples) - 1)
                samples, sr = samples[idx], 16000
            dur = len(samples) / sr
            if dur < 2.5 or dur > 10.0:
                continue
            picked = (lang, text, samples, sr, dur, i, row.get("id"))
            break
        if not picked:
            print(f"  FAIL: no suitable clip for {lang}")
            continue
        lang, text, samples, sr, dur, idx, sentence_id = picked
        cid = f"fleurs_{lang}"
        wav_path = OUT / f"{cid}.wav"
        to_wav(wav_path, samples, sr)
        rec = {
            "id": cid,
            "lang": lang,
            "audio": f"artifacts/workloads/speech/multilingual/{cid}.wav",
            "ref": text,
            "duration_s": round(dur, 2),
            "source": f"google/fleurs:{cfg}:test[{idx}]",
        }
        if sentence_id is not None:
            rec["sentence_id"] = int(sentence_id)
        rows.append(rec)
        print(f"  {cid} {dur:.2f}s :: {text[:80]}")

    print("== en_us parallel text (translation references) ==")
    try:
        refs = english_refs({r["sentence_id"] for r in rows if "sentence_id" in r})
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED ({exc}); manifest will carry no ref_en")
        refs = {}
    for r in rows:
        en = refs.get(r.get("sentence_id"))
        if en:
            r["ref_en"] = en
            print(f"  {r['id']} :: {en[:80]}")
        else:
            print(f"  {r['id']} :: no parallel English sentence found")

    man = OUT / "manifest.jsonl"
    man.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "# Multilingual ASR smoke clips\n\n"
        "Short FLEURS test clips for proving multilingual Whisper decode.\n"
        "Manifest: `manifest.jsonl` (lang + reference transcription).\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} clips -> {OUT}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

