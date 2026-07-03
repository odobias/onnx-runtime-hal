#!/usr/bin/env python3
"""Probe: (1) is the existing OV-IR whisper decoder stateful (ReadValue/Assign)?
(2) what stateful-transform API does this OpenVINO build expose? (3) does the
neutral ONNX decoder_with_past have present<->past pairs we can make stateful?"""
import glob
import os

import openvino as ov

print("ov", ov.__version__)
core = ov.Core()

# --- (1) inspect the OV-IR variant decoder for state ---
ir_candidates = glob.glob(os.path.join("models", "variants", "wten-ov-fp32", "**", "*.xml"), recursive=True)
print("\n== OV-IR variant .xml files ==")
for x in ir_candidates:
    try:
        m = core.read_model(x)
    except Exception as e:
        print(f"  {os.path.basename(x):32s} (read failed: {str(e)[:40]})")
        continue
    try:
        sinks = m.get_sinks()
    except Exception:
        sinks = []
    states = 0
    for op in m.get_ordered_ops():
        tn = op.get_type_name()
        if tn in ("ReadValue", "Assign"):
            states += 1
    print(f"  {os.path.basename(x):32s} sinks={len(sinks)} state_ops={states} "
          f"inputs={[i.get_any_name() for i in m.inputs][:3]}...")

# --- (2) stateful transform API discovery ---
print("\n== stateful API ==")
try:
    from openvino.passes import MakeStateful  # noqa
    print("  openvino.passes.MakeStateful: OK")
except Exception as e:
    print("  openvino.passes.MakeStateful:", e)
try:
    from openvino._offline_transformations import apply_make_stateful_transformation  # noqa
    print("  _offline_transformations.apply_make_stateful_transformation: OK")
except Exception as e:
    print("  _offline_transformations:", e)

# --- (3) neutral ONNX decoder_with_past present/past pairing ---
print("\n== neutral ONNX decoder_with_past ==")
dp = os.path.join("models", "whisper-tiny-en-onnx", "decoder_with_past_model.onnx")
m = core.read_model(dp)
ins = [i.get_any_name() for i in m.inputs]
outs = [o.get_any_name() for o in m.outputs]
past = [n for n in ins if n.startswith("past_key_values")]
present = [n for n in outs if n.startswith("present")]
print(f"  past inputs   ({len(past)}): {past[:4]} ...")
print(f"  present outs  ({len(present)}): {present[:4]} ...")
dec_past = [n for n in past if ".decoder." in n]
dec_pres = [n for n in present if ".decoder." in n]
enc_past = [n for n in past if ".encoder." in n]
enc_pres = [n for n in present if ".encoder." in n]
print(f"  decoder KV: {len(dec_past)} past / {len(dec_pres)} present  (pairable -> stateful)")
print(f"  encoder KV: {len(enc_past)} past / {len(enc_pres)} present  (constant cross-attn; set once)")
