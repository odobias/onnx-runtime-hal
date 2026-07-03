#!/usr/bin/env python3
"""Direct format comparison: the SAME Whisper encoder as neutral ONNX vs as
OpenVINO IR, both run through OpenVINO on CPU/GPU/NPU. Isolates whether the model
*format* (ONNX vs proprietary IR) costs anything once OpenVINO compiles it.
Encoder-only (fp32 source), batch fixed to 1, timed identically.
"""
import time

import numpy as np
import openvino as ov

ONNX_ENC = "models/whisper-tiny-en-onnx-static/encoder_static.onnx"
IR_ENC = "models/variants/wten-ov-fp32/openvino_encoder_model.xml"
RESHAPE = {"input_features": [1, 80, 3000]}
FEATS = {"input_features": np.zeros((1, 80, 3000), dtype=np.float32)}


def bench(core, path, dev, runs=10):
    m = core.read_model(path)
    m.reshape(RESHAPE)
    t0 = time.time()
    compiled = core.compile_model(m, dev)
    comp = time.time() - t0
    ir = compiled.create_infer_request()
    ir.infer(FEATS)
    ir.infer(FEATS)
    t0 = time.time()
    for _ in range(runs):
        ir.infer(FEATS)
    return comp, (time.time() - t0) / runs * 1000


def main():
    core = ov.Core()
    print("OpenVINO:", ov.__version__, "| devices:", core.available_devices)
    print(f"\n{'device':6s} {'format':5s} | {'compile':>9s} | {'infer ms':>9s}")
    print("-" * 42)
    for dev in ("CPU", "GPU", "NPU"):
        row = {}
        for label, path in (("ONNX", ONNX_ENC), ("IR", IR_ENC)):
            try:
                comp, ms = bench(core, path, dev)
                row[label] = ms
                print(f"{dev:6s} {label:5s} | {comp:8.2f}s | {ms:8.1f}")
            except Exception as e:  # noqa: BLE001
                print(f"{dev:6s} {label:5s} | FAILED: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        if "ONNX" in row and "IR" in row and row["IR"] > 0:
            print(f"{dev:6s} ratio | ONNX/IR = {row['ONNX']/row['IR']:.2f}x")


if __name__ == "__main__":
    main()
