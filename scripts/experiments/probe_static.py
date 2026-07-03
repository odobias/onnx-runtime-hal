#!/usr/bin/env python3
"""Verify the static-shape Whisper ONNX and test it on the OpenVINO EP (CPU/NPU)."""
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def is_static(path):
    m = onnx.load(path, load_external_data=False)
    dyn = []
    for vi in list(m.graph.input) + list(m.graph.output):
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_param:
                dyn.append((vi.name, d.dim_param))
    print(f"{path}")
    print(f"  fully static: {not dyn}" + ("" if not dyn else f"  (dynamic: {dyn})"))
    return not dyn


def ep_test(path, device, feed=None):
    try:
        t0 = time.time()
        sess = ort.InferenceSession(
            path, providers=[("OpenVINOExecutionProvider", {"device_type": device})]
        )
        create = time.time() - t0
        active = "OpenVINOExecutionProvider" in sess.get_providers()
        line = f"  [{device}] create {create:5.2f}s | providers={sess.get_providers()} | OVEP_active={active}"
        if feed is not None and active:
            sess.run(None, feed)  # warmup
            t0 = time.time()
            for _ in range(3):
                out = sess.run(None, feed)
            line += f" | infer {(time.time()-t0)/3*1000:6.1f} ms | out {out[0].shape}"
        print(line)
    except Exception as e:  # noqa: BLE001
        print(f"  [{device}] FAILED: {type(e).__name__}: {str(e).splitlines()[0][:180]}")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-onnx-static"
    print("onnxruntime:", ort.__version__, "| providers:", ort.get_available_providers())

    enc = f"{d}/encoder_static.onnx"
    dec = f"{d}/decoder_static.onnx"

    banner("STATIC CHECK")
    is_static(enc)
    is_static(dec)

    banner("ENCODER on OpenVINO EP")
    feats = np.zeros((1, 80, 3000), dtype=np.float32)
    for dev in ("CPU", "NPU"):
        ep_test(enc, dev, {"input_features": feats})

    banner("DECODER on OpenVINO EP (session-create)")
    for dev in ("CPU", "NPU"):
        ep_test(dec, dev)


if __name__ == "__main__":
    main()
