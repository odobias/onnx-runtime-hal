import collections
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper

ELEM = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
    16: "bfloat16",
}


def summarize(path: Path):
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    # initializer (weight) dtype histogram + total bytes per dtype
    dt = collections.Counter()
    for init in g.initializer:
        dt[ELEM.get(init.data_type, init.data_type)] += 1
    # op-type histogram, focus on quant markers
    ops = collections.Counter(n.op_type for n in g.node)
    qdq = ops.get("QuantizeLinear", 0) + ops.get("DequantizeLinear", 0)
    print(f"==== {path.name} ====")
    print(f"  initializer dtypes: {dict(dt)}")
    print(f"  QDQ nodes (Q+DQ)  : {qdq}")
    print(f"  MatMul/Gemm       : {ops.get('MatMul',0)}/{ops.get('Gemm',0)}")
    top = ", ".join(f"{k}:{v}" for k, v in ops.most_common(8))
    print(f"  top ops           : {top}")
    # IO dtypes
    for i in g.input:
        tt = i.type.tensor_type
        print(f"  IN  {i.name:26} {ELEM.get(tt.elem_type, tt.elem_type)}")
    for o in g.output:
        tt = o.type.tensor_type
        print(f"  OUT {o.name:26} {ELEM.get(tt.elem_type, tt.elem_type)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(Path(p))
