"""Ensure classifier models exist under workloads/classifiers/, auto-downloading
missing ones from the private Hugging Face mirror (remote keys stay deepfake/*).

Python twin of tools/fetch/get-classifier-models.ps1.
"""
import os
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFIERS_DIR = os.path.join(ROOT, "workloads", "classifiers")
LEGACY_DIR = os.path.join(ROOT, "models", "deepfake")
DEFAULT_REPO = os.environ.get("NPU_INFERENCE_BENCH_MODELS_REPO", "odobias/npu-hal-over-9000")

_SPECS = {
    "tsc": {
        "sentinel": os.path.join(CLASSIFIERS_DIR, "tsc", "model.onnx"),
        "legacy": os.path.join(LEGACY_DIR, "tsc", "model.onnx"),
        "patterns": ["deepfake/tsc/*"],
        "hf_dir": "deepfake/tsc",
        "local_dir": "tsc",
    },
    "fakeaudio": {
        "sentinel": os.path.join(CLASSIFIERS_DIR, "fakeaudio", "model.onnx"),
        "legacy": os.path.join(LEGACY_DIR, "fakeaudio", "model.onnx"),
        "patterns": ["deepfake/fakeaudio/*", "deepfake/audio-samples/*"],
        "hf_dir": "deepfake/fakeaudio",
        "local_dir": "fakeaudio",
    },
}


def _copytree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _migrate_legacy(models):
    """Copy from models/deepfake when workloads/classifiers is missing."""
    for m in models:
        spec = _SPECS[m]
        if os.path.exists(spec["sentinel"]) or not os.path.exists(spec["legacy"]):
            continue
        src = os.path.dirname(spec["legacy"]) if m == "tsc" else os.path.dirname(spec["legacy"])
        # family folder
        family_src = os.path.join(LEGACY_DIR, spec["local_dir"])
        family_dst = os.path.join(CLASSIFIERS_DIR, spec["local_dir"])
        if os.path.isdir(family_src):
            _copytree(family_src, family_dst)
        if m == "fakeaudio":
            samples_src = os.path.join(LEGACY_DIR, "audio-samples")
            samples_dst = os.path.join(CLASSIFIERS_DIR, "audio-samples")
            if os.path.isdir(samples_src) and not os.path.isdir(samples_dst):
                _copytree(samples_src, samples_dst)
        fix_src = os.path.join(LEGACY_DIR, "fixtures", m)
        fix_dst = os.path.join(CLASSIFIERS_DIR, "fixtures", m)
        if os.path.isdir(fix_src) and not os.path.isdir(fix_dst):
            _copytree(fix_src, fix_dst)


def ensure_deepfake_models(models=("tsc", "fakeaudio"), repo=None):
    """Download any of `models` that aren't already on disk from the HF mirror."""
    repo = repo or DEFAULT_REPO
    unknown = [m for m in models if m not in _SPECS]
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}; known: {sorted(_SPECS)}")

    _migrate_legacy(models)
    missing = [m for m in models if not os.path.exists(_SPECS[m]["sentinel"])]
    if not missing:
        return

    patterns = []
    for m in missing:
        patterns += _SPECS[m]["patterns"]
        if m == "tsc":
            patterns.append("deepfake/fixtures/tsc/*")
        if m == "fakeaudio":
            patterns.append("deepfake/fixtures/fakeaudio/*")

    print(f"[models] {', '.join(missing)} not found locally -- downloading from {repo} ...")
    staging = tempfile.mkdtemp(prefix="npu-bench-classifiers-")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            local_dir=staging,
            allow_patterns=patterns,
        )
        for m in missing:
            spec = _SPECS[m]
            src = os.path.join(staging, *spec["hf_dir"].split("/"))
            dst = os.path.join(CLASSIFIERS_DIR, spec["local_dir"])
            if os.path.exists(src):
                _copytree(src, dst)
            if m == "fakeaudio":
                samples_src = os.path.join(staging, "deepfake", "audio-samples")
                samples_dst = os.path.join(CLASSIFIERS_DIR, "audio-samples")
                if os.path.isdir(samples_src):
                    _copytree(samples_src, samples_dst)
            fix_name = m
            fix_src = os.path.join(staging, "deepfake", "fixtures", fix_name)
            fix_dst = os.path.join(CLASSIFIERS_DIR, "fixtures", fix_name)
            if os.path.isdir(fix_src):
                _copytree(fix_src, fix_dst)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[models] auto-download of {', '.join(missing)} from {repo} failed "
            f"({type(e).__name__}: {e}).\n"
            f"        For the private mirror run `hf auth login` once, or fetch "
            f"manually with tools\\fetch\\get-classifier-models.ps1."
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    still = [m for m in missing if not os.path.exists(_SPECS[m]["sentinel"])]
    if still:
        raise SystemExit(
            f"[models] {', '.join(still)} still missing after download from {repo} "
            f"(does the repo contain the deepfake/* subtree?)."
        )
    print(f"[models] ready under workloads/classifiers: {', '.join(missing)}")
