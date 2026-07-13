#!/usr/bin/env python3
"""Land the fakeaudio (MS-CLAP / HTSAT) deepfake detector on the INTEL NPU.

This answers the "UNVERIFIED on Intel OpenVINO" question left in
export_fakeaudio_variants.py: unlike AMD's VAIML (where every variant diverged),
Intel's NPU compiler runs the model correctly -- but only after TWO fixes:

  1. Precision (AMD's toolkit): the log-mel front-end must stay fp32. The
     power_to_db clamps mel energy to amin=1e-10 (a -100 dB floor) before Log; in
     fp16 that amin constant is below min-normal (6.1e-5), so the NPU raises the
     log-mel silence floor from -100 dB to -42 dB. That 58 dB lift flips the one
     confident-"real" clip (proven in verify_fakeaudio_fp16_underflow.py -- it's a
     raised floor, NOT a Log(0)/-inf underflow). The NPU has no fp32
     (INFERENCE_PRECISION_HINT accepts only f16 / i8 -- bf16 is rejected), so the
     front-end runs on CPU and only the transformer backbone goes to the NPU.

  2. Compile (this repo): the backbone still crashes OpenVINO's vpux compiler --
     it fuses HTSAT windowed attention into SDPA and mis-flattens the bias add
     (scores [nW,H,L,L] + bias [1,H,L,L] -> IE.Add 1x64x64x64 + 16x64x64, an
     LLVM ERROR abort). Fix: expand each Softmax-feeding bias-Add's [1,H,L,L]
     constant to the scores' full [nW,H,L,L] shape (numerically identical) so the
     fusion cannot mis-broadcast it.

Result on Lunar Lake NPU (OpenVINO 2026.2.1): FE(CPU fp32) + BB(NPU f16) matches
full-fp32 CPU to max|dp| ~1e-4, 0 flips, ~30 ms (~85% of it on the NPU).

Prereq: run export_fakeaudio_variants.py first (produces model.frontend-fp32 +
model.backbone-fp32). Usage:
    .venv\\Scripts\\python.exe scripts\\experiments\\fakeaudio_npu_intel.py
    optional: --full-npu (show the whole graph runs on NPU but mispredicts)
              --bf16     (show the NPU rejects bf16)
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import openvino as ov
from onnx import helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_fakeaudio_onnx as fa  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
FA = os.path.join(ROOT, "models", "deepfake", "fakeaudio")
BASE = os.path.join(FA, "model.onnx")
FE = os.path.join(FA, "model.frontend-fp32.onnx")
BB = os.path.join(FA, "model.backbone-fp32.onnx")
BB_NPU = os.path.join(FA, "model.backbone.npu-intel.onnx")


def expand_attention_bias(src, dst):
    """Expand every Softmax-feeding bias-Add's constant operand to the scores'
    full shape, so OpenVINO's SDPA fusion can't mis-flatten it on the NPU.
    Returns the number of attention blocks patched."""
    m = onnx.load(src)
    g = m.graph
    inits = {i.name for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    prepend = defaultdict(list)
    n = 0
    for node in g.node:
        if node.op_type != "Softmax":
            continue
        s = producer.get(node.input[0])
        if s is None or s.op_type != "Add" or len(s.input) != 2:
            continue
        a, b = s.input
        c = a if a in inits else (b if b in inits else None)
        if c is None:
            continue
        scores = b if c == a else a
        base = s.name.replace("/", "_").strip("_")
        so, eo = f"{base}__scsh", f"{base}__mexp"
        prepend[id(s)].append(helper.make_node("Shape", [scores], [so], name=f"{base}__Sh"))
        prepend[id(s)].append(helper.make_node("Expand", [c, so], [eo], name=f"{base}__Ex"))
        s.input[:] = [eo if x == c else x for x in s.input]
        n += 1
    rebuilt = []
    for node in g.node:
        rebuilt.extend(prepend.get(id(node), []))
        rebuilt.append(node)
    del g.node[:]
    g.node.extend(rebuilt)
    onnx.save(m, dst)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-npu", action="store_true",
                    help="also run the whole (surgered) graph on the NPU to show the fp16 flip")
    ap.add_argument("--bf16", action="store_true", help="probe whether the NPU accepts bf16")
    ap.add_argument("--runs", type=int, default=8)
    args = ap.parse_args()

    for p in (BASE, FE, BB):
        if not os.path.exists(p):
            print(f"missing {os.path.relpath(p, ROOT)} -- run export_fakeaudio_variants.py first")
            return 1

    core = ov.Core()
    print(f"OpenVINO {ov.__version__} | devices {core.available_devices}\n")

    # reference: full fp32 on CPU
    cpu = ort.InferenceSession(BASE, providers=["CPUExecutionProvider"])
    items = fa.build_feeds(cpu)
    onames = [o.name for o in cpu.get_outputs()]
    ref = {it["name"]: (fa.predict(cpu.run(onames, it["feed"]))[1], it["label"]) for it in items}

    def score(tag, probs):
        md = max(abs(ref[k][0] - probs[k]) for k in probs)
        flips = sum((ref[k][0] > 0.5) != (probs[k] > 0.5) for k in probs)
        acc = sum((probs[k] > 0.5) == (ref[k][1] == "deepfake") for k in probs)
        print(f"  {tag:34s} max|dp|={md:.4f}  flips={flips}  acc={acc}/{len(probs)}")

    # --- the working path: FE(CPU fp32) + BB(NPU f16) with attention surgery ---
    n = expand_attention_bias(BB, BB_NPU)
    print(f"attention surgery: expanded bias in {n} blocks -> {os.path.basename(BB_NPU)}")
    fe = ort.InferenceSession(FE, providers=["CPUExecutionProvider"])
    fe_out = fe.get_outputs()[0].name
    mel0 = fe.run([fe_out], items[0]["feed"])[0]
    bm = core.read_model(BB_NPU)
    bin_ = bm.inputs[0].get_any_name()
    bm.reshape({bin_: list(mel0.shape)})
    bb = core.compile_model(bm, "NPU")
    bo = bb.outputs[0]

    print("\n== split: front-end (CPU fp32) + backbone (NPU f16) ==")
    probs, fe_ms, bb_ms = {}, [], []
    for it in items:
        r = None
        for _ in range(args.runs):
            t = time.perf_counter(); mel = fe.run([fe_out], it["feed"])[0]; fe_ms.append((time.perf_counter()-t)*1000)
            t = time.perf_counter(); r = bb({bin_: mel})[bo]; bb_ms.append((time.perf_counter()-t)*1000)
        probs[it["name"]] = fa.predict([r])[1]
    score("split FE-CPU + BB-NPU", probs)
    fe_m = sum(sorted(fe_ms)[2:]) / len(sorted(fe_ms)[2:])
    bb_m = sum(sorted(bb_ms)[2:]) / len(sorted(bb_ms)[2:])
    tot = fe_m + bb_m
    print(f"  latency: FE {fe_m:.2f} ms (CPU) + BB {bb_m:.2f} ms (NPU) = {tot:.2f} ms "
          f"[CPU {100*fe_m/tot:.0f}% / NPU {100*bb_m/tot:.0f}%]")

    # --- optional: bf16 rejection probe ---
    if args.bf16:
        print("\n== bf16 probe (would fix the underflow -- if the NPU allowed it) ==")
        try:
            core.compile_model(bm, "NPU", {"INFERENCE_PRECISION_HINT": "bf16"})
            print("  bf16 ACCEPTED (unexpected)")
        except Exception as e:
            msg = next((ln.strip() for ln in str(e).splitlines()
                        if "Supported values" in ln or "Wrong value" in ln), str(e).strip()[:160])
            print(f"  bf16 REJECTED: {msg}")

    # --- optional: whole graph on NPU (runs, but fp16 front-end mispredicts) ---
    if args.full_npu:
        print("\n== whole graph on NPU (surgered): runs fully on NPU but fp16 FE flips ==")
        full_npu_path = os.path.join(FA, "model.npu-full.onnx")
        expand_attention_bias(BASE, full_npu_path)
        fm = core.read_model(full_npu_path)
        full = core.compile_model(fm, "NPU")
        fop = full.outputs[0]
        probs = {it["name"]: fa.predict([full({full.inputs[0].get_any_name(): it["feed"]["input"]})[fop]])[1]
                 for it in items}
        score("full graph on NPU (f16)", probs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
