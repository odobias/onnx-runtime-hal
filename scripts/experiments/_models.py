"""Ensure the deepfake classifier models exist locally, auto-downloading the
missing ones from the private Hugging Face mirror on demand.

This is what the validators / benchmark / fixture-dumper call at startup so they
"just work" on a fresh checkout instead of dead-ending with "run
get-deepfake-models.ps1 first". It is the Python twin of
scripts/get-deepfake-models.ps1 (same repo, same deepfake/* subtree) -- pick
whichever is convenient; both pull only the deepfake/* paths and never touch any
internal/corporate artifact repository.

Private repo => a one-time `hf auth login` is required. Override the source with
WHISPER_HAL_MODELS_REPO=<user>/<repo> if you mirror it elsewhere.
"""
import os

# scripts/experiments/_models.py -> repo root is three levels up.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
DEFAULT_REPO = os.environ.get("WHISPER_HAL_MODELS_REPO", "odobias/npu-hal-over-9000")

# Per-model: a sentinel that proves it's present + the HF globs to fetch it
# (and any labeled sample data its validator scores against).
_SPECS = {
    "tsc": {
        "sentinel": os.path.join(MODELS_DIR, "deepfake", "tsc", "model.onnx"),
        "patterns": ["deepfake/tsc/*"],
    },
    "fakeaudio": {
        "sentinel": os.path.join(MODELS_DIR, "deepfake", "fakeaudio", "model.onnx"),
        "patterns": ["deepfake/fakeaudio/*", "deepfake/audio-samples/*"],
    },
}


def ensure_deepfake_models(models=("tsc", "fakeaudio"), repo=None):
    """Download any of `models` that aren't already on disk from the HF mirror.

    No-ops (and stays offline) when everything requested is already present, so
    it's cheap to call unconditionally at the top of every consumer.
    """
    repo = repo or DEFAULT_REPO
    unknown = [m for m in models if m not in _SPECS]
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}; known: {sorted(_SPECS)}")

    missing = [m for m in models if not os.path.exists(_SPECS[m]["sentinel"])]
    if not missing:
        return

    patterns = []
    for m in missing:
        patterns += _SPECS[m]["patterns"]

    print(f"[models] {', '.join(missing)} not found locally -- downloading from {repo} ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            local_dir=MODELS_DIR,
            allow_patterns=patterns,
        )
    except Exception as e:  # noqa: BLE001 -- surface a clear, actionable message
        raise SystemExit(
            f"[models] auto-download of {', '.join(missing)} from {repo} failed "
            f"({type(e).__name__}: {e}).\n"
            f"        For the private mirror run `hf auth login` once, or fetch "
            f"manually with scripts\\get-deepfake-models.ps1."
        )

    still = [m for m in missing if not os.path.exists(_SPECS[m]["sentinel"])]
    if still:
        raise SystemExit(
            f"[models] {', '.join(still)} still missing after download from {repo} "
            f"(does the repo contain the deepfake/* subtree?)."
        )
    print(f"[models] ready: {', '.join(missing)}")
