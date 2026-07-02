#!/usr/bin/env python3
"""Feasibility: import a NEUTRAL ONNX Whisper straight into OpenVINO and compile it
to the Intel NPU (and GPU/CPU). This mirrors how our C++ Intel backend runs
(OpenVINO), bypassing the ORT OpenVINO-EP pip ABI mess.

  .venv-ovep\Scripts\python.exe scripts\probe_ov_direct.py models\whisper-tiny-en-onnx-static
"""
import sys
import time

import numpy as np
import openvino as ov


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def try_compile(core, path, dev, run_feats=None, reshape=None):
    try:
        model = core.read_model(path)
        if reshape:
            model.reshape(reshape)
        t0 = time.time()
        compiled = core.compile_model(model, dev)
        comp = time.time() - t0
        msg = f"  [{dev}] compiled in {comp:6.2f}s"
        if run_feats is not None:
            ir = compiled.create_infer_request()
            ir.infer(run_feats)  # warmup
            t0 = time.time()
            for _ in range(3):
                res = ir.infer(run_feats)
            dt = (time.time() - t0) / 3
            shp = list(res.values())[0].shape
            msg += f" | infer {dt*1000:6.1f} ms | out {shp}"
        print(msg)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [{dev}] FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return False


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-onnx-static"
    core = ov.Core()
    print("OpenVINO:", ov.__version__)
    print("devices :", core.available_devices)

    enc = f"{d}/encoder_static.onnx"
    dec = f"{d}/decoder_static.onnx"

    banner("ENCODER (neutral ONNX) -> OpenVINO compile")
    feats = {"input_features": np.zeros((1, 80, 3000), dtype=np.float32)}
    for dev in ("CPU", "GPU", "NPU"):
        try_compile(core, enc, dev, run_feats=feats)

    banner("DECODER (neutral ONNX) -> OpenVINO compile")
    # Fix the growing KV-cache dims to a static max so the NPU can compile.
    reshape = {
        "input_ids": [1, 128],
        "encoder_hidden_states": [1, 1500, 384],
    }
    for dev in ("CPU", "GPU", "NPU"):
        try_compile(core, dec, dev, reshape=reshape)


if __name__ == "__main__":
    main()
