#!/usr/bin/env python3
"""Precision sweep for the neutral static ONNX Whisper encoder on Intel devices.

Compiles the SAME ONNX at fp32 / fp16 / bf16 (via INFERENCE_PRECISION_HINT) and
INT8 (NNCF weight compression), on CPU / GPU / NPU, and times the encoder forward.
Answers: what precisions does each device actually accept, and how fast.

  .venv-ovep\Scripts\python.exe scripts\probe_precisions.py models\whisper-tiny-en-onnx-static
"""
import sys
import time

import numpy as np
import openvino as ov
import openvino.properties.hint as hints


def bench(core, model, dev, prec_label, config, feats, runs=5):
    try:
        t0 = time.time()
        compiled = core.compile_model(model, dev, config)
        comp = time.time() - t0
        ir = compiled.create_infer_request()
        ir.infer(feats)  # warmup
        t0 = time.time()
        for _ in range(runs):
            ir.infer(feats)
        ms = (time.time() - t0) / runs * 1000
        print(f"  {dev:3s} {prec_label:5s} | compile {comp:6.2f}s | infer {ms:7.1f} ms")
        return ms
    except Exception as e:  # noqa: BLE001
        print(f"  {dev:3s} {prec_label:5s} | FAILED: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
        return None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-onnx-static"
    enc = f"{d}/encoder_static.onnx"
    core = ov.Core()
    print("OpenVINO:", ov.__version__, "| devices:", core.available_devices)
    feats = {"input_features": np.zeros((1, 80, 3000), dtype=np.float32)}

    base = core.read_model(enc)

    # INT8 weight-compressed variant (built once, reused across devices).
    int8_model = None
    try:
        import nncf

        int8_model = nncf.compress_weights(core.read_model(enc))
        print("INT8 weight compression: OK")
    except Exception as e:  # noqa: BLE001
        print(f"INT8 weight compression FAILED: {type(e).__name__}: {e}")

    prec_hints = {
        "fp32": {hints.inference_precision: ov.Type.f32},
        "fp16": {hints.inference_precision: ov.Type.f16},
        "bf16": {hints.inference_precision: ov.Type.bf16},
    }

    for dev in ("CPU", "GPU", "NPU"):
        print(f"\n=== {dev} ===")
        for label, cfg in prec_hints.items():
            bench(core, base, dev, label, cfg, feats)
        if int8_model is not None:
            bench(core, int8_model, dev, "int8", {}, feats)


if __name__ == "__main__":
    main()
