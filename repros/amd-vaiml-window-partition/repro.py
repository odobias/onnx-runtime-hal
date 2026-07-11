#!/usr/bin/env python3
"""Minimal synthetic VAIML first-inference crash reproducer.

The included ONNX graph contains only a LayerNormalization followed by the
dynamic shape/reshape/transpose sequence used to partition a [1,4096,96]
tensor into 64 windows of [64,96]. There are no trained model weights or
captured customer inputs.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
MODEL = HERE / "model.onnx"
MODEL_SHA256 = "8061c31aefe698248a5de79e8498d5b9ec78d1912dfc3381f4d36152b1f29cd0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("cpu", "vitisai"))
    parser.add_argument(
        "--cache-dir",
        default=str(Path(tempfile.gettempdir()) / "amd-vaiml-window-partition-repro"),
    )
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    actual_hash = file_sha256(MODEL)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(f"model SHA256 mismatch: {actual_hash}")

    available = ort.get_available_providers()
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "available_providers": available,
        "model": str(MODEL),
        "model_bytes": MODEL.stat().st_size,
        "model_sha256": actual_hash,
        "requested_provider": args.provider,
        "graph_optimizations": "disabled",
    }
    print(json.dumps(metadata, indent=2), flush=True)

    options = ort.SessionOptions()
    options.log_severity_level = 2
    # Keep the exported dynamic-shape sequence intact. ORT/onnxsim can fold it
    # into constants, but that causes VAIML not to claim the graph and hides the
    # bug behind CPU fallback.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = ["CPUExecutionProvider"]
    if args.provider == "vitisai":
        if "VitisAIExecutionProvider" not in available:
            raise RuntimeError("VitisAIExecutionProvider is not available in this ONNX Runtime")
        cache_dir = Path(args.cache_dir).resolve()
        if cache_dir.exists() and not args.reuse_cache:
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers = [
            (
                "VitisAIExecutionProvider",
                {
                    "cache_dir": str(cache_dir),
                    "cache_key": f"window-partition-{actual_hash[:12]}",
                },
            ),
            "CPUExecutionProvider",
        ]

    print("Creating InferenceSession...", flush=True)
    session = ort.InferenceSession(str(MODEL), options, providers=providers)
    print(f"Session created; providers={session.get_providers()}", flush=True)

    # Fully synthetic and deterministic. On the affected runtime the VitisAI
    # session compiles and is created successfully, then the native EP crashes
    # while consuming this tensor on the first Run().
    value = np.random.default_rng(20260711).standard_normal((1, 4096, 96), dtype=np.float32)
    print("Running first inference...", flush=True)
    result = session.run(None, {session.get_inputs()[0].name: value})[0]
    summary = {
        "shape": list(result.shape),
        "min": float(result.min()),
        "max": float(result.max()),
        "mean": float(result.mean()),
        "finite": bool(np.isfinite(result).all()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
