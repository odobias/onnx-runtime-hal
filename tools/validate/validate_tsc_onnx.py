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
  - Tokenizer: artifacts/workloads/classifiers/tsc/{vocab.json,merges.txt} is a byte-level BPE
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
sys.path.insert(0, os.path.join(ROOT, "tools", "research"))
TSC_DIR = os.path.join(ROOT, "workloads", "classifiers", "tsc")
MODEL_PATH = os.path.join(TSC_DIR, "model.onnx")

# Labeled samples live beside the model in validation_samples/{scam,clean}.txt --
# gitignored and synced via HF (get-models.ps1), NOT committed to git, mirroring
# the FakeAudio audio-samples policy. Pulled verbatim from
# git.int.avast.com/ai-research/wanna-media-nlp
# src/wanna_media_nlp/core/validation_samples/{scam,clean}.txt
SAMPLES_DIR = os.path.join(TSC_DIR, "validation_samples")


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
    tokenizer.post_processor = RobertaProcessing(
        sep=("</s>", 2), cls_token=("<s>", 0)
    )
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_id=1, pad_token="<pad>")
    return tokenizer


def _read_samples(name):
    path = os.path.join(SAMPLES_DIR, f"{name}.txt")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def labeled_samples():
    """The model's own post-training sanity set: (text, label) pairs, loaded from
    validation_samples/{scam,clean}.txt beside the model (synced via HF)."""
    return ([(t, POSITIVE_LABEL) for t in _read_samples("scam")]
            + [(t, NEGATIVE_LABEL) for t in _read_samples("clean")])


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
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _models import ensure_deepfake_models
    ensure_deepfake_models(["tsc"])  # auto-fetch from HF if missing

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH} (run scripts\\get-deepfake-models.ps1 first)")
        return 1

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    items = build_feeds(sess)
    if not items:
        print(f"No labeled samples in {SAMPLES_DIR} -- run scripts\\get-models.ps1 to fetch them.")
        return 1
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
    scam_total = sum(1 for it in items if it["label"] == POSITIVE_LABEL)
    print(f"\n{correct}/{total} correct  (scam: {scam_correct}/{scam_total}, "
          f"clean: {correct - scam_correct}/{total - scam_total})")
    print(f"Note: {CAVEAT}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
