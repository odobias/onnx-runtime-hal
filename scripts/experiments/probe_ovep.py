#!/usr/bin/env python3
"""Feasibility probe: can a NEUTRAL Whisper ONNX run on the Intel NPU via ORT's
OpenVINO Execution Provider?

For encoder + decoder graphs it reports:
  1. op domains / any custom or EPContext ops (is the graph truly standard ONNX?)
  2. input/output shapes (static vs dynamic -- NPUs require static)
  3. session creation on OpenVINO EP for device_type CPU then NPU (+ fallback/errors)
  4. a real encoder inference on CPU and NPU with timing

Run with the onnxruntime-openvino venv:
  .venv-ovep\Scripts\python.exe scripts\experiments\probe_ovep.py models\whisper-tiny-en-onnx
"""
import sys
import time
import traceback

import numpy as np
import onnx
import onnxruntime as ort


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def inspect_graph(path):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    domains = {}
    custom = set()
    for node in g.node:
        dom = node.domain or "ai.onnx"
        domains[dom] = domains.get(dom, 0) + 1
        if dom not in ("ai.onnx", "", "ai.onnx.ml"):
            custom.add(f"{dom}::{node.op_type}")
    print(f"opset imports : {[(i.domain or 'ai.onnx', i.version) for i in m.opset_import]}")
    print(f"op domains    : {domains}")
    print(f"custom ops    : {sorted(custom) if custom else 'none (standard ONNX)'}")

    def shape_of(vi):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        return dims

    print("inputs:")
    dynamic = False
    for vi in g.input:
        s = shape_of(vi)
        if any(isinstance(x, str) for x in s):
            dynamic = True
        print(f"   {vi.name:32s} {s}")
    print("outputs:")
    for vi in g.output:
        print(f"   {vi.name:32s} {shape_of(vi)}")
    print(f"shapes        : {'DYNAMIC (has symbolic dims)' if dynamic else 'fully static'}")
    return dynamic


def try_session(path, device_type):
    try:
        opts = ort.SessionOptions()
        provider = ("OpenVINOExecutionProvider", {"device_type": device_type})
        t0 = time.time()
        sess = ort.InferenceSession(path, sess_options=opts, providers=[provider])
        dt = time.time() - t0
        got = sess.get_providers()
        ok = "OpenVINOExecutionProvider" in got
        print(f"  [{device_type}] session created in {dt:.2f}s | providers={got} | OVEP_active={ok}")
        return sess
    except Exception as e:  # noqa: BLE001
        print(f"  [{device_type}] FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return None


def run_encoder(sess, tag):
    if sess is None:
        return
    try:
        inp = sess.get_inputs()[0]
        # whisper-tiny log-mel: [batch, 80, 3000]
        feats = np.zeros((1, 80, 3000), dtype=np.float32)
        # warmup + timed
        sess.run(None, {inp.name: feats})
        t0 = time.time()
        for _ in range(3):
            out = sess.run(None, {inp.name: feats})
        dt = (time.time() - t0) / 3
        print(f"  [{tag}] encoder inference OK | {dt * 1000:.1f} ms | out shape {out[0].shape}")
    except Exception as e:  # noqa: BLE001
        print(f"  [{tag}] encoder inference FAILED: {type(e).__name__}: {str(e).splitlines()[0][:200]}")


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-onnx"
    print("onnxruntime :", ort.__version__)
    print("providers   :", ort.get_available_providers())

    enc = f"{model_dir}/encoder_model.onnx"
    dec = f"{model_dir}/decoder_model.onnx"

    banner("ENCODER graph")
    inspect_graph(enc)
    banner("ENCODER on OpenVINO EP")
    for dev in ("CPU", "NPU"):
        s = try_session(enc, dev)
        run_encoder(s, dev)

    banner("DECODER graph")
    inspect_graph(dec)
    banner("DECODER on OpenVINO EP (session-create only)")
    for dev in ("CPU", "NPU"):
        try_session(dec, dev)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
