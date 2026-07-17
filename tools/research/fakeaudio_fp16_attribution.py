"""Per-edge precision attribution for the fakeaudio front-end: which tensor(s),
when stored in fp16 vs bf16, actually break the model.

Method: leave the whole graph in fp32, but on ONE edge insert a Cast round-trip
(fp32 -> low precision -> fp32), run the full model, and measure max |dp(fake)|
vs the untouched fp32 baseline. Do this for fp16 and for bf16 at each front-end
boundary. This isolates the responsible op and answers "does bf16 help?" directly:
  - fp16: 5-bit exponent, underflows below ~6.1e-5
  - bf16: 8-bit exponent (fp32 range, min normal ~1.2e-38), 7-bit mantissa

Also prints each edge's dynamic range so the failure lines up with the numbers.
"""
import copy
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FA = os.path.join(ROOT, "workloads", "classifiers", "fakeaudio")
MODEL = os.path.join(FA, "model.onnx")
FIX = os.path.join(ROOT, "workloads", "classifiers", "fixtures", "fakeaudio")
SAMPLES = ["deepfake_1", "deepfake_2", "real_1", "real_2", "real_3"]
WINDOW = 308700

PROBES = [
    ("Conv real (STFT)",  "/embedder/base/htsat/spectrogram_extractor/conv_real/Conv_output_0"),
    ("Pow real^2",        "/embedder/base/htsat/spectrogram_extractor/Pow_output_0"),
    ("Add power-spectrum","/embedder/base/htsat/spectrogram_extractor/Add_output_0"),
    ("MatMul mel-energy", "/embedder/base/htsat/logmel_extractor/MatMul_output_0"),
    ("Clip (floor 1e-10)","/embedder/base/htsat/logmel_extractor/Clip_output_0"),
    ("Log (log-mel)",     "/embedder/base/htsat/logmel_extractor/Log_output_0"),
]


def load_feeds():
    return {s: np.fromfile(os.path.join(FIX, "data", f"{s}__input.bin"),
                           dtype=np.float32).reshape(1, WINDOW) for s in SAMPLES}


def softmax_pos(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return float((e / e.sum(axis=-1, keepdims=True))[0, 1])


def sess(b):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(b, so, providers=["CPUExecutionProvider"])


def run_p(model, feeds):
    s = sess(model.SerializeToString())
    n = s.get_inputs()[0].name
    return {k: softmax_pos(s.run(None, {n: v})[0]) for k, v in feeds.items()}


def toposort(g):
    avail = {i.name for i in g.input} | {t.name for t in g.initializer}
    nodes, out, prog = list(g.node), [], True
    while nodes and prog:
        prog, still = False, []
        for nd in nodes:
            if all(i == "" or i in avail for i in nd.input):
                out.append(nd); avail.update(nd.output); prog = True
            else:
                still.append(nd)
        nodes = still
    out.extend(nodes)
    del g.node[:]; g.node.extend(out)


def roundtrip(model, tensor, to_type):
    m = copy.deepcopy(model)
    g = m.graph
    low, back = tensor + "_low", tensor + "_rt"
    for nd in g.node:
        for i, inp in enumerate(nd.input):
            if inp == tensor:
                nd.input[i] = back
    g.node.append(helper.make_node("Cast", [tensor], [low], to=to_type,
                                   name=tensor + "_castlow"))
    g.node.append(helper.make_node("Cast", [low], [back], to=TensorProto.FLOAT,
                                   name=tensor + "_castup"))
    toposort(g)
    return m


def ranges(feeds):
    m = onnx.load(MODEL)
    add = [t for _, t in PROBES if t not in {o.name for o in m.graph.output}]
    for t in add:
        m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    s = sess(m.SerializeToString())
    names = [o.name for o in s.get_outputs()]
    vals = dict(zip(names, s.run(None, {s.get_inputs()[0].name: feeds["real_3"]})))
    print("edge dynamic range (sample real_3): "
          "fp16 min-normal=6.1e-5, so % below that = fp16 underflow")
    for label, t in PROBES:
        a = np.abs(np.asarray(vals[t], dtype=np.float32))
        nz = a[a > 0]
        minnz = float(nz.min()) if nz.size else 0.0
        under = 100.0 * np.mean(a < 6.1e-5)
        print(f"  {label:20s} absmax={a.max():11.4g}  min|nonzero|={minnz:11.4g}  "
              f"%<6.1e-5={under:5.1f}")


def main():
    feeds = load_feeds()
    base = run_p(onnx.load(MODEL), feeds)
    print("=== fp32 baseline ===")
    for s in SAMPLES:
        print(f"  {s:12s} p={base[s]:.6f}")

    ranges(feeds)

    def maxdp(p):
        return max(abs(base[s] - p[s]) for s in SAMPLES), \
               sum((base[s] > 0.5) != (p[s] > 0.5) for s in SAMPLES)

    print(f"\n{'edge':22s} {'fp16 max|dp|':>14s} {'flips':>6s}   {'bf16 max|dp|':>14s} {'flips':>6s}")
    for label, t in PROBES:
        row = f"  {label:20s}"
        for to_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16):
            try:
                md, fl = maxdp(run_p(roundtrip(onnx.load(MODEL), t, to_type), feeds))
                row += f" {md:14.4f} {fl:6d}  "
            except Exception as e:  # noqa: BLE001
                row += f" {'ERR:'+type(e).__name__:>21s}  "
        print(row)


if __name__ == "__main__":
    sys.exit(main())
