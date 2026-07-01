#!/usr/bin/env python3
"""Build a small labeled ASR eval set as 16 kHz mono WAVs + eval.jsonl.

Source: hf-internal-testing/librispeech_asr_dummy (tiny, public, ~73 utterances).
We disable the datasets Audio auto-decoder and decode the raw flac ourselves via
soundfile, which avoids the torchcodec/ffmpeg dependency dance that recent
`datasets` versions otherwise trigger.

Output layout (repo-relative paths written into the jsonl):
    models/eval/<id>.wav
    models/eval/eval.jsonl   ->  {"id","audio","ref","duration_s"} per line
"""
import argparse
import io
import json
import os
import sys
import wave

import numpy as np


def to_wav_int16(path: str, samples: np.ndarray, sr: int) -> None:
    x = np.clip(samples, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="models/eval directory")
    ap.add_argument("--count", type=int, default=20, help="max utterances")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))

    n = min(args.count, len(ds))
    jsonl = os.path.join(args.outdir, "eval.jsonl")
    written = 0
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(n):
            row = ds[i]
            try:
                samples, sr = load_audio(row["audio"])
            except Exception as e:  # noqa: BLE001 - one bad clip shouldn't kill the set
                print(f"skip [{i}]: {e}", file=sys.stderr)
                continue
            if sr != 16000:
                # tiny nearest-neighbor resample; dummy set is already 16k so this
                # is just a guard, not a quality path.
                idx = (np.arange(int(len(samples) * 16000 / sr)) * sr / 16000).astype(int)
                idx = np.clip(idx, 0, len(samples) - 1)
                samples, sr = samples[idx], 16000
            cid = f"ls_{i:03d}"
            wav_path = os.path.join(args.outdir, f"{cid}.wav")
            to_wav_int16(wav_path, samples, sr)
            rec = {
                "id": cid,
                "audio": f"models/eval/{cid}.wav",
                "ref": str(row["text"]).strip(),
                "duration_s": round(len(samples) / sr, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} utterances to {jsonl}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
