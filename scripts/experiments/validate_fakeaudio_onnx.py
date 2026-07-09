#!/usr/bin/env python3
"""Correctness (not just latency) check for the FakeAudio / Generated Audio
Detector ONNX model, using real labeled audio pulled from the team's own
manual efficacy review -- see provenance below.

Provenance (2026-07-09, via Glean):
  - Confluence "Media detection efficacy" (CTO/854623129) documents a manual
    "Generated audio detector efficacy" review: every video in the list was
    *already flagged* by the deployed Video Analyzer's audio_deepfake
    classifier (that's the SQL WHERE clause used to pull the list); a team
    member then manually watched/listened to each flagged video and labeled
    it Real (i.e. the flag was a false positive) or Deepfake (true positive).
    So this is a sample of the *production classifier's own borderline/flagged
    cases*, not a random/balanced sample -- useful for sanity-checking that
    our on-device ONNX export reproduces the same decision boundary as the
    deployed model, but not a representative accuracy measurement (for that,
    the aggregate ~96-98% figures quoted lower on that same Confluence page,
    computed over 300-450 videos each run, are the closest thing available).
    This is the closest thing to a small labeled eval set we could find for
    this exact model -- there's no BigQuery/GCS-hosted labeled audio
    benchmark accessible to us (the "Audio benchmark dataset" section of that
    page is still "TBD").
  - models/deepfake/audio-samples/*.wav were downloaded (via yt-dlp, see
    below) from a handful of the *unambiguously* labeled entries in that list
    (skipping every "not sure" / "probably" hedge and any now-unavailable
    video): 3 Real, 2 Deepfake, ~15s each, resampled to 44.1kHz mono.
  - Model I/O: input "input" is [1, 308700] float32 -- 308700 / 44100 = 7.0s,
    matching the Media Shield deployment notes ("processes smaller audio
    chunks (approximately 7 seconds each)"), so we feed a plain 7s raw-PCM
    window in [-1, 1], no extra feature extraction (MS-CLAP's front-end
    appears to be traced into the graph itself, consistent with how MS-CLAP
    ONNX exports are normally packaged for on-device deployment). Output
    "output" is [1, 2] logits, softmax class 1 assumed = "generated/fake"
    (mirrors the TSC export's class-1-is-positive convention).
  - Caveat: sample size is tiny (5 clips) and "which 7 seconds of a 15s clip"
    is an arbitrary choice for videos where only part of the audio may be
    synthetic (the source Confluence page itself notes clips can be
    "partial" deepfakes) -- treat this as a plausibility smoke test, not a
    certified accuracy number.

Re-fetching the clips (not committed to git -- gitignored, proprietary/
copyright caveats aside, YouTube audio shouldn't live in the repo anyway):
    python -m pip install yt-dlp imageio-ffmpeg soundfile
    (see scripts/experiments/README.md or ask -- the exact yt-dlp invocation
    used is in the chat history / commit that introduced this script)

    python scripts\\experiments\\validate_fakeaudio_onnx.py
"""
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "models", "deepfake", "fakeaudio", "model.onnx")
SAMPLES_DIR = os.path.join(ROOT, "models", "deepfake", "audio-samples")
WINDOW_SAMPLES = 308700  # 7.0s @ 44.1kHz, per the model's static input shape

# Ground truth from the manual "Generated audio detector efficacy" review
# (Confluence: Media detection efficacy) -- filename prefix -> label.
LABELS = {
    "real_1": "real", "real_2": "real", "real_3": "real",
    "deepfake_1": "deepfake", "deepfake_2": "deepfake",
}

# Positive class = "deepfake" (class 1) -- mirrors the TSC export's convention.
POSITIVE_LABEL = "deepfake"
NEGATIVE_LABEL = "real"
THRESHOLD = 0.5
CAVEAT = ("every clip was already flagged non-clean by the production classifier "
          "(known FPs included); n=5, first-7s window only -- plausibility signal, "
          "not an accuracy measurement")


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


def build_feeds(sess=None):
    """Real labeled ONNX feeds: list of {name, label, feed}. `sess` is accepted
    (and ignored) for a uniform signature with the other classifiers' build_feeds."""
    items = []
    for wav in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.wav"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        items.append({
            "name": stem,
            "label": LABELS.get(stem, "?"),
            "feed": {"input": load_window(wav)[np.newaxis, :]},
        })
    return items


def predict(outputs):
    """(predicted_label, p_positive) from raw ONNX outputs."""
    p_fake = float(softmax(outputs[0])[0, 1])
    return (POSITIVE_LABEL if p_fake > THRESHOLD else NEGATIVE_LABEL), p_fake


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH} (run scripts\\get-deepfake-models.ps1 first)")
        return 1

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    items = build_feeds(sess)
    if not items:
        print(f"No labeled sample audio in {SAMPLES_DIR} -- see this script's docstring "
              "for how to re-fetch a few clips with yt-dlp.")
        return 1

    print(f"{'label':9s} {'p(fake)':>8s}  file")
    print("-" * 55)
    correct = 0
    for item in items:
        predicted, p_fake = predict(sess.run(None, item["feed"]))
        ok = predicted == item["label"]
        correct += int(ok)
        print(f"{item['label']:9s} {p_fake:8.4f}  [{'OK' if ok else 'MISS'}] {item['name']}.wav")

    print(f"\n{correct}/{len(items)} correct")
    print(f"Note: {CAVEAT}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
