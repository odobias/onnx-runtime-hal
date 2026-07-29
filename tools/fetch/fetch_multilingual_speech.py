#!/usr/bin/env python3
"""Fetch a few short FLEURS clips (with refs) into artifacts/workloads/speech/multilingual/."""
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


def main() -> int:
    from datasets import Audio, load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang, cfg in LANGS.items():
        print(f"== {lang} ({cfg}) ==")
        ds = load_dataset("google/fleurs", cfg, split="test")
        ds = ds.cast_column("audio", Audio(decode=False))
        # Prefer clips ~3–8s with non-empty transcription
        picked = None
        for i in range(min(80, len(ds))):
            row = ds[i]
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
            picked = (lang, text, samples, sr, dur, i)
            break
        if not picked:
            print(f"  FAIL: no suitable clip for {lang}")
            continue
        lang, text, samples, sr, dur, idx = picked
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
        rows.append(rec)
        print(f"  {cid} {dur:.2f}s :: {text[:80]}")

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

