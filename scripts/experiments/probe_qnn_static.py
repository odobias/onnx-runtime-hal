"""Probe whisper-tiny-en-static-onnx on ORT QNN EP (Snapdragon NPU)."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

EP_NAME = "QNNExecutionProvider"
_REGISTERED = False


def ensure_qnn_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    ort.register_execution_provider_library(EP_NAME, qnn_ep.get_library_path())
    _REGISTERED = True


def make_session(model_path: str, device: str) -> ort.InferenceSession:
    if device == "cpu":
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

    ensure_qnn_registered()
    ep_devices = [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]
    if not ep_devices:
        raise RuntimeError("QNNExecutionProvider not registered")

    if device == "gpu":
        backend = qnn_ep.get_qnn_gpu_path()
    else:
        backend = qnn_ep.get_qnn_htp_path()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_provider_for_devices(ep_devices, {"backend_path": backend})
    return ort.InferenceSession(model_path, sess_options=so)


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/whisper-tiny-en-static-onnx"
    device = (sys.argv[2] if len(sys.argv) > 2 else "npu").lower()
    enc = os.path.join(model_dir, "encoder_model.onnx")
    dec = os.path.join(model_dir, "decoder_model.onnx")

    print("ORT version:", ort.__version__)
    print("QNN EP version:", qnn_ep.__version__)
    print("Providers:", ort.get_available_providers())

    print(f"Loading encoder from {enc} device={device}")
    t0 = time.perf_counter()
    enc_sess = make_session(enc, device)
    print(f"Encoder loaded in {time.perf_counter()-t0:.2f}s, providers={enc_sess.get_providers()}")

    t0 = time.perf_counter()
    dec_sess = make_session(dec, device)
    print(f"Decoder loaded in {time.perf_counter()-t0:.2f}s, providers={dec_sess.get_providers()}")

    feats = np.zeros((1, 80, 3000), dtype=np.float32)
    t0 = time.perf_counter()
    enc_out = enc_sess.run(None, {"input_features": feats})[0]
    print(f"Encoder infer {time.perf_counter()-t0:.3f}s shape={enc_out.shape}")

    ids = np.full((1, 128), 50256, dtype=np.int64)
    ids[0, 0] = 50257
    ids[0, 1] = 50362
    t0 = time.perf_counter()
    logits = dec_sess.run(None, {"input_ids": ids, "encoder_hidden_states": enc_out})[0]
    print(f"Decoder infer {time.perf_counter()-t0:.3f}s shape={logits.shape}")
    tok = int(np.argmax(logits[0, 1]))
    print(f"First token after prompt: {tok}")

if __name__ == "__main__":
    main()
