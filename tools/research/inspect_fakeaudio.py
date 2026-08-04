"""Static structural inspection of the fakeaudio (MS-CLAP + head) ONNX graph.

Read-only. Prints opset, op-type histogram, initializer dtype/param breakdown,
graph I/O, and flags the classic precision-killer ops (STFT/DFT, LayerNorm,
ReduceMean/Sqrt/Div, Log/Exp/Pow, Softmax, large MatMul/Gemm). Also traces the
tail subgraph that produces the [1,2] logits, since softmax happens outside the
graph and logit calibration is what blows up under low precision.
"""
import os
import sys
from collections import Counter, defaultdict

import onnx
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "workloads", "classifiers", "fakeaudio", "model.onnx")

DTYPE = {v: k for k, v in onnx.TensorProto.DataType.items()}


def main() -> int:
    m = onnx.load(MODEL)
    g = m.graph
    print(f"ir_version={m.ir_version}  producer={m.producer_name!r} {m.producer_version!r}")
    for op in m.opset_import:
        print(f"opset: domain={op.domain or 'ai.onnx'!r}  version={op.version}")

    print("\n=== graph inputs ===")
    for i in g.input:
        t = i.type.tensor_type
        dims = [d.dim_value if d.dim_value else (d.dim_param or '?') for d in t.shape.dim]
        print(f"  {i.name}: {DTYPE.get(t.elem_type)} {dims}")
    print("=== graph outputs ===")
    for o in g.output:
        t = o.type.tensor_type
        dims = [d.dim_value if d.dim_value else (d.dim_param or '?') for d in t.shape.dim]
        print(f"  {o.name}: {DTYPE.get(t.elem_type)} {dims}")

    print(f"\n=== op-type histogram ({len(g.node)} nodes) ===")
    hist = Counter(n.op_type for n in g.node)
    for op, c in hist.most_common():
        print(f"  {c:5d}  {op}")

    print("\n=== initializers: dtype + param count ===")
    by_dtype = defaultdict(lambda: [0, 0])  # dtype -> [tensor_count, total_elems]
    total_bytes = 0
    biggest = []
    for init in g.initializer:
        dt = DTYPE.get(init.data_type, str(init.data_type))
        n = 1
        for d in init.dims:
            n *= d
        by_dtype[dt][0] += 1
        by_dtype[dt][1] += n
        try:
            arr = numpy_helper.to_array(init)
            total_bytes += arr.nbytes
            biggest.append((arr.nbytes, init.name, dt, tuple(init.dims)))
        except Exception:
            pass
    for dt, (cnt, elems) in sorted(by_dtype.items()):
        print(f"  {dt:10s}  tensors={cnt:5d}  elems={elems:,}")
    print(f"  total initializer bytes ~ {total_bytes/1e6:.1f} MB")
    biggest.sort(reverse=True)
    print("  largest 12 initializers:")
    for b, name, dt, dims in biggest[:12]:
        print(f"    {b/1e6:7.2f} MB  {dt:8s} {dims}  {name}")

    print("\n=== precision-sensitive op presence ===")
    sensitive = {
        "front-end (STFT/DFT/FFT)": ["STFT", "DFT", "RFFT", "Conv"],
        "normalization": ["LayerNormalization", "InstanceNormalization",
                          "BatchNormalization", "GroupNormalization",
                          "SimplifiedLayerNormalization"],
        "norm-building-blocks": ["ReduceMean", "ReduceSum", "ReduceL2",
                                 "Sqrt", "ReciprocalSqrt", "Div", "Pow"],
        "range-expanding": ["Exp", "Log", "Softmax", "LogSoftmax", "Erf",
                            "Sigmoid", "Tanh"],
        "big-matmul": ["MatMul", "Gemm", "Einsum"],
        "attention": ["Attention", "MultiHeadAttention", "Softmax"],
    }
    for group, ops in sensitive.items():
        present = {op: hist[op] for op in ops if hist.get(op)}
        if present:
            print(f"  {group:26s}: {present}")

    # Tail: nodes producing the final output (walk back a few hops).
    print("\n=== tail subgraph (producers of graph output) ===")
    producer = {}
    for n in g.node:
        for out in n.output:
            producer[out] = n
    frontier = [o.name for o in g.output]
    seen = set()
    order = []
    for _ in range(14):
        nxt = []
        for t in frontier:
            n = producer.get(t)
            if n is None or id(n) in seen:
                continue
            seen.add(id(n))
            order.append(n)
            nxt.extend(n.input)
        frontier = nxt
        if not frontier:
            break
    for n in order:
        print(f"  {n.op_type:22s} out={list(n.output)}  in={list(n.input)[:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
