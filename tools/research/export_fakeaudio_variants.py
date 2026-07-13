"""Regenerate the useful fakeaudio (MS-CLAP / HTSAT) precision variants from the
one checked-in base model, so we ship a *recipe* in git instead of a pile of
derived weights on HF. Every output here is a deterministic function of
workloads/classifiers/fakeaudio/model.onnx (fetched from the HF mirror on demand).

Why these variants exist (all measured, see the fidelity table this prints):

  model.fp16-safe.onnx      fp32 log-mel front-end + fp16 transformer backbone,
                            merged back into ONE graph with the original I/O
                            contract. Naive whole-graph fp16 diverges max|dp|~0.99
                            (the mel-energy MatMul underflows: values reach ~2e-17,
                            below fp16's 6.1e-5 floor -> Log(0) -> NaN). Keeping the
                            front-end in fp32 drops that to ~0.0001, zero flips.
                            >>> This is the drop-in win for CPU / GPU fp16. <<<

  model.frontend-fp32.onnx  the log-mel front-end alone (input -> Log). ~4 MB, runs
  model.backbone-fp32.onnx  on CPU in microseconds. The backbone (Log -> logits) is
                            shape-pinned + shape-inferred so a fixed-function NPU
                            compiler (VAIML / OpenVINO / QNN) can actually resolve it.
                            This is the portable "features on CPU, net on NPU" split
                            (exactly how AMD ships their own whisper NPU model).

  model.backbone-int8.onnx  calibrated INT8 QDQ of the backbone (Resize left float),
                            calibrated on the fixture mels. Holds max|dp|~0.03 on CPU.
                            Candidate for Qualcomm/Intel int8 NPUs.

Measured NPU reality on AMD XDNA2 / VitisAI (do not lose this hard-won result):
  every variant DIVERGES on VAIML -- fp32 full 0.996, fp16-safe 0.996, fp32 backbone
  0.99, calibrated int8 backbone 0.51 (and 12x slower). VAIML mishandles the HTSAT
  windowed-attention/Resize regardless of the precision we can control from ORT.
  fakeaudio therefore belongs on CPU/GPU (fp16-safe); the NPU is for whisper. The
  split/int8 artifacts remain the right thing to try on Intel OpenVINO / Qualcomm QNN,
  whose compilers are not VAIML and may well behave -- UNVERIFIED (no such HW here).

Usage:
  .venv\\Scripts\\python.exe scripts\\experiments\\export_fakeaudio_variants.py
  optional: --out-dir <dir>  --skip-int8  --no-validate
"""
import argparse
import copy
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference
from onnxruntime.transformers import float16

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FA = os.path.join(ROOT, "models", "deepfake", "fakeaudio")
MODEL = os.path.join(FA, "model.onnx")
FIX = os.path.join(ROOT, "models", "deepfake", "fixtures", "fakeaudio")
SAMPLES = ["deepfake_1", "deepfake_2", "real_1", "real_2", "real_3"]
WINDOW = 308700
# The log-mel boundary: front-end = input..CUT, backbone = CUT..output.
CUT = "/embedder/base/htsat/logmel_extractor/Log_output_0"


# --------------------------------------------------------------------------- io
def sess(path_or_bytes):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path_or_bytes, so, providers=["CPUExecutionProvider"])


def softmax_pos(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return float((e / e.sum(axis=-1, keepdims=True))[0, 1])


def load_pcm():
    """The 5 fixture raw-PCM inputs, if present (used for calibration + validation)."""
    feeds = {}
    for s in SAMPLES:
        b = os.path.join(FIX, "data", f"{s}__input.bin")
        if os.path.exists(b):
            feeds[s] = np.fromfile(b, dtype=np.float32).reshape(1, WINDOW)
    return feeds


# ------------------------------------------------------------------- fp16 merge
def _dedupe_node_names(m):
    seen = {}
    for n in m.graph.node:
        base = n.name or n.op_type
        if base in seen:
            seen[base] += 1
            n.name = f"{base}__u{seen[base]}"
        else:
            seen[base] = 0
    return m


def _dedupe_cast_outputs(m):
    seen, keep = set(), []
    for n in m.graph.node:
        if n.op_type == "Cast" and len(n.output) == 1 and n.output[0] in seen:
            continue
        seen.update(n.output)
        keep.append(n)
    del m.graph.node[:]
    m.graph.node.extend(keep)
    return m


def to_fp16(m):
    """Whole-graph fp16 with keep_io_types, plus two work-arounds for known ORT
    converter bugs (duplicate cast node names / duplicate cast output tensors)."""
    mp = float16.convert_float_to_float16(copy.deepcopy(m), keep_io_types=True)
    return _dedupe_node_names(_dedupe_cast_outputs(mp))


def _toposort(nodes, avail):
    ordered, pending, prog = [], list(nodes), True
    while pending and prog:
        prog, still = False, []
        for n in pending:
            if all(i == "" or i in avail for i in n.input):
                ordered.append(n)
                avail.update(n.output)
                prog = True
            else:
                still.append(n)
        pending = still
    if pending:
        raise RuntimeError(f"topo failed: {pending[0].name!r} needs {list(pending[0].input)}")
    return ordered


def compose(fe, bb, ir_version):
    """Concatenate fp32 front-end + fp16 backbone into one graph (seam tensor
    becomes internal) and topologically re-sort so the backbone's boundary Cast
    lands after the front-end produces the seam."""
    names = {t.name for t in fe.graph.initializer}
    inits = list(fe.graph.initializer) + [t for t in bb.graph.initializer if t.name not in names]
    avail = {i.name for i in fe.graph.input} | {t.name for t in inits}
    nodes = _toposort(list(fe.graph.node) + list(bb.graph.node), avail)
    g = onnx.helper.make_graph(nodes, "fakeaudio_fp16_safe",
                               list(fe.graph.input), list(bb.graph.output), inits)
    m = onnx.helper.make_model(g, opset_imports=list(fe.opset_import))
    m.ir_version = ir_version
    return m


# ------------------------------------------------------------------- shape pin
def cut_shape(model_path):
    """Concrete static shape of the CUT (log-mel) tensor, from shape inference on
    the fully-static base graph."""
    m = shape_inference.infer_shapes(onnx.load(model_path), data_prop=True)
    for vi in list(m.graph.value_info) + list(m.graph.output):
        if vi.name == CUT:
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if all(dims):
                return dims
    return None


def pin_and_infer(path, dims):
    m = onnx.load(path)
    inp = m.graph.input[0]
    del inp.type.tensor_type.shape.dim[:]
    for d in dims:
        inp.type.tensor_type.shape.dim.add().dim_value = d
    del m.graph.value_info[:]
    onnx.save(shape_inference.infer_shapes(m, data_prop=True), path)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=FA)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _models import ensure_deepfake_models
    ensure_deepfake_models(["fakeaudio"])  # base model.onnx from HF mirror if missing
    if not os.path.exists(MODEL):
        print(f"base model missing: {MODEL}")
        return 1

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    base_model = onnx.load(MODEL)
    ir = base_model.ir_version
    pcm = load_pcm()
    validate = pcm and not args.no_validate

    if validate:
        s0 = sess(MODEL)
        iname = s0.get_inputs()[0].name
        ref = {k: softmax_pos(s0.run(None, {iname: v})[0]) for k, v in pcm.items()}

    def report(tag, probs):
        md = max(abs(ref[s] - probs[s]) for s in probs)
        fl = sum((ref[s] > 0.5) != (probs[s] > 0.5) for s in probs)
        print(f"  {tag:34s} max|dp|={md:.4f}  flips={fl}")
        return md

    fe_path = os.path.join(out, "model.frontend-fp32.onnx")
    bb_path = os.path.join(out, "model.backbone-fp32.onnx")
    fp16_path = os.path.join(out, "model.fp16-safe.onnx")
    int8_path = os.path.join(out, "model.backbone-int8.onnx")

    # 1) split
    onnx.utils.extract_model(MODEL, fe_path, ["input"], [CUT])
    onnx.utils.extract_model(MODEL, bb_path, [CUT], ["output"])
    fe = onnx.load(fe_path)
    bb = onnx.load(bb_path)

    # 2) fp16-safe single file (fp32 front-end + fp16 backbone), pre shape-pin
    merged = compose(fe, to_fp16(bb), ir)
    onnx.checker.check_model(merged)
    onnx.save(merged, fp16_path)

    # 3) shape-pin the standalone backbone so an NPU compiler can resolve it.
    # The STFT/Conv front-end isn't statically shape-inferable, so prefer the exact
    # mel shape observed by running the front-end; fall back to inference.
    dims = None
    if pcm:
        fs = sess(fe_path)
        dims = list(fs.run(None, {fs.get_inputs()[0].name: next(iter(pcm.values()))})[0].shape)
    if not dims:
        dims = cut_shape(MODEL)
    if dims:
        pin_and_infer(bb_path, dims)
        print(f"  backbone shape-pinned to mel {dims}")
    else:
        print("  WARN: could not determine CUT shape; backbone left with dynamic dims")

    print("\n=== fidelity vs fp32 (CPU) ===" if validate else "\n=== built (validation skipped) ===")
    if validate:
        report("fp16-safe (fp32 FE + fp16 BB)", {k: softmax_pos(sess(fp16_path).run(
            None, {sess(fp16_path).get_inputs()[0].name: v})[0]) for k, v in pcm.items()})

    # 4) calibrated int8 backbone (needs the fixture PCM to build the mel calib set)
    if not args.skip_int8 and pcm and dims:
        from onnxruntime.quantization import (
            CalibrationDataReader, QuantFormat, QuantType, quantize_static)
        from onnxruntime.quantization.shape_inference import quant_pre_process

        fes = sess(fe_path)  # front-end produces the mel calibration set
        fin = fes.get_inputs()[0].name
        mels = [fes.run(None, {fin: v})[0].astype(np.float32) for v in pcm.values()]
        bin_ = sess(bb_path).get_inputs()[0].name

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{bin_: m} for m in mels])

            def get_next(self):
                return next(self.it, None)

        exclude = sorted(n.name for n in onnx.load(bb_path).graph.node
                         if n.op_type in ("Resize", "Upsample") and n.name)
        pp = bb_path + ".pp.onnx"
        quant_pre_process(bb_path, pp, skip_symbolic_shape=True)
        quantize_static(pp, int8_path, Reader(), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8,
                        activation_type=QuantType.QInt8, nodes_to_exclude=exclude)
        try:
            os.remove(pp)
        except OSError:
            pass

        if validate:
            qs = sess(int8_path)
            qi = qs.get_inputs()[0].name
            chain = {}
            for k, v in pcm.items():
                mel = fes.run(None, {fin: v})[0].astype(np.float32)
                chain[k] = softmax_pos(qs.run(None, {qi: mel})[0])
            report("int8 backbone + fp32 FE (chain)", chain)
    elif not args.skip_int8:
        print("  (int8 skipped: fixture PCM or CUT shape unavailable)")

    print("\n=== written ===")
    for p in (fp16_path, fe_path, bb_path, int8_path):
        if os.path.exists(p):
            print(f"  {os.path.getsize(p)/1e6:7.1f} MB  {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
