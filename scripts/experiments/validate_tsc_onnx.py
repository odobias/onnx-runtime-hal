#!/usr/bin/env python3
"""Correctness (not just latency) check for the TSC ONNX model, using real
labeled samples pulled from the model's own training repo and a real
tokenizer reconstruction -- see docstring at the bottom of this file for the
full provenance trail.

Provenance (2026-07-09, via Glean + git clone of internal repos):
  - Labeled samples: src/wanna_media_nlp/core/validation_samples/{scam,clean}.txt
    from https://git.int.avast.com/ai-research/wanna-media-nlp (the actual
    TTSC training/eval repo). These are the exact sanity-check sentences the
    TSC team itself runs after every training run (see
    transcript_tsc/tasks/validate_model.py::run_benchmark_checks) -- clean
    samples must score < 0.1, scam samples > 0.9 (checked both bare and with
    random-string prefix/suffix noise added).
  - Tokenizer: models/deepfake/tsc/{vocab.json,merges.txt} is a byte-level BPE
    vocab of size 50265 with special tokens <s>=0, <pad>=1, </s>=2, <unk>=3,
    <mask>=50264 -- an exact match for roberta-base's tokenizer, confirming
    (despite the "DistilBERT" branding in internal docs) this model's
    tokenization is RoBERTa-style byte-level BPE, not WordPiece. Reconstructed
    here directly via the `tokenizers` library (ByteLevelBPETokenizer +
    RobertaProcessing for the <s>/</s> wrapping -- transformers 5.x's
    RobertaTokenizerFast refuses to bootstrap from a bare vocab.json/merges.txt
    pair without a tokenizer.json or sentencepiece present), with truncation
    and max-length padding matching the intent of
    wanna_media_nlp/transcript_tsc/tasks/train_model.py::TOKENIZER_TRAINING_SETUP
    -- matching what export_model_onnx.py used to trace the graph (input_names=["input_ids",
    "attention_mask"], output_names=["output"]), which lines up exactly with
    this ONNX file's actual input/output names.
  - Caveat: we don't have the base model's tokenizer_config.json /
    special_tokens_map.json (those live in GCS, not Artifactory), so
    add_prefix_space and any non-default special-token wiring are assumed at
    RoBERTa defaults. Good enough to check "does scam score high / clean
    score low", not a byte-exact reproduction of the training pipeline.

    python scripts\\experiments\\validate_tsc_onnx.py
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSC_DIR = os.path.join(ROOT, "models", "deepfake", "tsc")
MODEL_PATH = os.path.join(TSC_DIR, "model.onnx")

# Pulled verbatim from git.int.avast.com/ai-research/wanna-media-nlp
# src/wanna_media_nlp/core/validation_samples/{scam,clean}.txt
SCAM_SAMPLES = [
    "send me 8 bitcoins and I will send you 16 bitcoins back",
    "Today I have decided to give away a small portion of my Bitcoins. You can enter the competition by scanning this QR code.",
    "I want to share a secret of how to get rich extremely fast. This is my way of giving back to the community.",
    "I want to share a secret of how to get rich extremely fast. This is my way of giving back to the community. The secret is to use a new trading platform, you should see the link on the screen",
    "I want to show you how to code an Ethereum sniping bot. Anybody can do it and you can start earning passive income in couple of days",
    "Follow this simple trick to get rich fast and have a passive income in no time.",
    "This investment opportunity guarantees you 100% profit in 24 hours. Just send me your wallet address and I will send you the details.",
    "This investment will create a passive income stream in a week and will last for years",
]

CLEAN_SAMPLES = [
    "hi",
    "hello world",
    "this is a test",
    "music music music music music music music music music",
    "You can buy Bitcoins on an online cryptocurrency exchange or store them directly into your physical wallet.",
    "In this video I want to show you an example of how a friend of mine got scammed on the Internet.",
    "Make sure to leave a like in this video. Subscribe to help me reach 15,000",
]


# Positive class = "scam" (class 1), per TTSCLabel convention in wanna-media-nlp.
POSITIVE_LABEL = "scam"
NEGATIVE_LABEL = "clean"
THRESHOLD = 0.5
# Best-effort tokenizer reconstruction (no tokenizer_config.json available) --
# shared caveat surfaced by both the standalone validator and the benchmark.
CAVEAT = ("best-effort tokenizer reconstruction (no tokenizer_config.json); "
          "strong plausibility signal, not a certified score")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def max_length_from_session(sess) -> int:
    dim = {i.name: i for i in sess.get_inputs()}["input_ids"].shape[1]
    return dim if isinstance(dim, int) else 512  # dynamic axis -> documented context


def build_tokenizer(max_length: int):
    tokenizer = ByteLevelBPETokenizer(
        vocab=os.path.join(TSC_DIR, "vocab.json"),
        merges=os.path.join(TSC_DIR, "merges.txt"),
    )
    # RoBERTa wraps every sequence as <s> ... </s>; ids taken from vocab.json itself.
    tokenizer.post_processor = RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0))
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_id=1, pad_token="<pad>")
    return tokenizer


def labeled_samples():
    """The model's own post-training sanity set: (text, label) pairs."""
    return ([(t, POSITIVE_LABEL) for t in SCAM_SAMPLES]
            + [(t, NEGATIVE_LABEL) for t in CLEAN_SAMPLES])


def build_feeds(sess):
    """Real labeled ONNX feeds for `sess`: list of {name, label, text, feed}.

    Batch dim is 1 (this export bakes batch=1 despite the dynamic_axes in the
    original torch.onnx.export call), so each item is a single-row feed.
    """
    tokenizer = build_tokenizer(max_length_from_session(sess))
    items = []
    for i, (text, label) in enumerate(labeled_samples()):
        enc = tokenizer.encode(text)
        items.append({
            "name": f"{label}_{i}",
            "label": label,
            "text": text,
            "feed": {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            },
        })
    return items


def predict(outputs):
    """(predicted_label, p_positive) from raw ONNX outputs."""
    p_scam = float(softmax(outputs[0])[0, 1])
    return (POSITIVE_LABEL if p_scam > THRESHOLD else NEGATIVE_LABEL), p_scam


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH} (run scripts\\get-deepfake-models.ps1 first)")
        return 1

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    items = build_feeds(sess)
    print(f"ONNX inputs: {[i.name for i in sess.get_inputs()]}  "
          f"max_length={max_length_from_session(sess)}\n")

    print(f"{'label':6s} {'p(scam)':>8s}  text")
    print("-" * 90)
    correct = scam_correct = 0
    for item in items:
        predicted, p = predict(sess.run(None, item["feed"]))
        ok = predicted == item["label"]
        correct += int(ok)
        scam_correct += int(ok and item["label"] == POSITIVE_LABEL)
        print(f"{item['label']:6s} {p:8.4f}  [{'OK' if ok else 'MISS'}] {item['text'][:70]}")

    total = len(items)
    scam_total = len(SCAM_SAMPLES)
    print(f"\n{correct}/{total} correct  (scam: {scam_correct}/{scam_total}, "
          f"clean: {correct - scam_correct}/{total - scam_total})")
    print(f"Note: {CAVEAT}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
