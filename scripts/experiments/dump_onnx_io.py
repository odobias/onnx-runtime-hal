#!/usr/bin/env python3
"""Print input/output tensor names + shapes for the whisper ONNX decoders, so the
C++ dynamic (KV-cache) backend feeds exactly the right names. Read-only.

    .venv\\Scripts\\python.exe scripts\\experiments\\dump_onnx_io.py models\\whisper-tiny-en-onnx
"""
import sys
import onnxruntime as ort

model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-onnx"

for f in ("encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"):
    path = f"{model_dir}/{f}"
    try:
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        print(f"== {f}: cannot load ({e})\n")
        continue
    print(f"== {f}")
    for i in s.get_inputs():
        print(f"  IN  {i.name:40s} {i.type} {i.shape}")
    for o in s.get_outputs():
        print(f"  OUT {o.name:40s} {o.type} {o.shape}")
    print()
