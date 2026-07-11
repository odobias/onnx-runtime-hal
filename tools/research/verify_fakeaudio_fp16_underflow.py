#!/usr/bin/env python3
"""PROVE (or refute) the fakeaudio fp16-underflow hypothesis on real audio.

Front-end (torchlibrosa power_to_db):
    power-spectrogram --MatMul(melW)--> mel energy --Clip(min=amin)--> Log --> /ln10 --> *10 (dB)

Hypothesis: whole-graph fp16 flips a real clip because the mel energies AND the
amin clamp constant both fall below fp16's floor, so Clip(0, min=0)=0 -> Log(0)
= -inf. We measure, per real labeled sample:
  1. amin (the Clip min constant) and how it renders in fp16.
  2. mel-energy min + how many elements sit below fp16 min-normal / subnormal.
  3. the fp32 Log output vs a numpy fp16 simulation vs the actual NPU-f16 output.
and check whether the flipping sample (real_3) is the one that underflows.
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_fakeaudio_onnx as fa  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
FA = os.path.join(ROOT, "models", "deepfake", "fakeaudio")
BASE = os.path.join(FA, "model.onnx")
MEL = os.path.join(FA, "_probe.melenergy.onnx")   # input -> MatMul_output_0 (pre-clamp)
LOGO = os.path.join(FA, "_probe.logout.onnx")     # input -> Log_output_0

MATMUL_OUT = "/embedder/base/htsat/logmel_extractor/MatMul_output_0"
LOG_OUT = "/embedder/base/htsat/logmel_extractor/Log_output_0"
CLIP_MIN = "/embedder/base/htsat/logmel_extractor/Constant_output_0"

FP16_MIN_NORMAL = np.finfo(np.float16).tiny      # 6.104e-5
FP16_MIN_SUBNORM = np.float32(6e-8)              # ~smallest fp16 subnormal 5.96e-8


def const_value(model, name):
    for n in model.graph.node:
        if n.op_type == "Constant" and name in n.output:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def main():
    m = onnx.load(BASE)
    amin = const_value(m, CLIP_MIN)
    amin_f = float(np.asarray(amin).reshape(-1)[0])
    amin_fp16 = np.float16(amin_f)
    print(f"Clip min (amin)        : {amin_f:.3e}  (fp32)")
    print(f"  amin cast to fp16     : {float(amin_fp16):.3e}   -> {'UNDERFLOWS to 0' if amin_fp16 == 0 else 'survives'}")
    print(f"fp16 min normal/subnorm : {FP16_MIN_NORMAL:.3e} / ~5.96e-8\n")

    onnx.utils.extract_model(BASE, MEL, ["input"], [MATMUL_OUT])
    onnx.utils.extract_model(BASE, LOGO, ["input"], [LOG_OUT])

    cpu_full = ort.InferenceSession(BASE, providers=["CPUExecutionProvider"])
    items = fa.build_feeds(cpu_full)
    onames = [o.name for o in cpu_full.get_outputs()]
    mel_sess = ort.InferenceSession(MEL, providers=["CPUExecutionProvider"])
    log_sess = ort.InferenceSession(LOGO, providers=["CPUExecutionProvider"])

    # NPU f16 readout of the Log output (front-end on-device). Falls back gracefully.
    npu_log = None
    try:
        import openvino as ov
        core = ov.Core()
        if "NPU" in core.available_devices:
            mel0 = mel_sess.run([MATMUL_OUT], items[0]["feed"])[0]
            lm = core.read_model(LOGO)
            lm.reshape({lm.inputs[0].get_any_name(): list(items[0]["feed"]["input"].shape)})
            npu_log = core.compile_model(lm, "NPU")  # fp16 by default on NPU
            print("NPU f16 front-end compiled: reading Log output on-device.\n")
    except Exception as e:
        print(f"(NPU readout unavailable: {str(e).splitlines()[-1][:100]})\n")

    hdr = (f"{'sample':10s} {'label':9s} {'p(fake)':>8s}  {'mel_min':>10s} "
           f"{'<subnorm':>8s} {'->0@fp16':>8s}  {'log_fp32_min':>12s} "
           f"{'log_fp16sim':>12s} {'log_NPU_min':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for it in items:
        feed = it["feed"]
        _, p = fa.predict(cpu_full.run(onames, feed))
        mel = mel_sess.run([MATMUL_OUT], feed)[0].astype(np.float64)
        log32 = log_sess.run([LOG_OUT], feed)[0].astype(np.float64)

        n = mel.size
        below_sub = int(np.sum(mel < FP16_MIN_SUBNORM))
        # numpy fp16 simulation of the exact clamp->log path
        mel16 = mel.astype(np.float16)
        to_zero = int(np.sum((mel16 == 0) & (mel > 0)))
        clipped16 = np.maximum(mel16.astype(np.float32), np.float32(amin_fp16))
        with np.errstate(divide="ignore"):
            log16 = np.log(np.where(clipped16 > 0, clipped16, 0.0))
            log16 = np.where(clipped16 > 0, log16, -np.inf)
        log16_min = float(np.min(log16))

        npu_min = np.nan
        if npu_log is not None:
            r = npu_log({npu_log.inputs[0].get_any_name(): feed["input"]})[npu_log.outputs[0]]
            r = np.asarray(r, np.float32)
            npu_min = float(np.min(r[np.isfinite(r)])) if np.any(np.isfinite(r)) else -np.inf

        print(f"{it['name']:10s} {it['label']:9s} {p:8.4f}  {float(mel.min()):10.2e} "
              f"{below_sub:8d} {to_zero:8d}  {log32.min():12.3f} "
              f"{log16_min:12.3f} {npu_min:12.3f}")

    # ---- CAUSAL test: raise ONLY the amin clamp to fp16's floor, keep the whole
    # rest of the graph in fp32 on CPU. If real_3 (the sole confident-real sample)
    # flips, the noise-floor lift alone -- not any other fp16 effect -- is the cause.
    print("\n== causal test: amin 1e-10 -> 6.104e-5 (fp16 floor), everything else fp32 CPU ==")
    m2 = onnx.load(BASE)
    patched = 0
    for n in m2.graph.node:
        if n.op_type == "Constant" and CLIP_MIN in n.output:
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t).copy()
                    arr[...] = np.float32(FP16_MIN_NORMAL)
                    a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
                    patched += 1
    tmp = os.path.join(FA, "_probe.aminfloor.onnx")
    onnx.save(m2, tmp)
    floor_sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    print(f"  patched {patched} clamp constant(s)\n")
    print(f"  {'sample':10s} {'label':9s} {'p_fp32':>8s} {'p_floor':>8s}  verdict")
    for it in items:
        _, p0 = fa.predict(cpu_full.run(onames, it["feed"]))
        _, p1 = fa.predict(floor_sess.run(onames, it["feed"]))
        flip = (p0 > 0.5) != (p1 > 0.5)
        print(f"  {it['name']:10s} {it['label']:9s} {p0:8.4f} {p1:8.4f}  {'FLIPPED' if flip else 'same'}")

    for p in (MEL, LOGO, tmp):
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    main()
