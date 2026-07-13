#!/usr/bin/env python3
"""Measure raw ORT session latency for the ONNX Whisper pieces, to locate the
bottleneck that optimum's generate() is hiding. Times: encoder fwd, one no-past
decoder step, one with-past decoder step. Also reports IO names/shapes."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort


def sess(path, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def io(s):
    ins = [(i.name, i.shape, i.type) for i in s.get_inputs()]
    outs = [(o.name, o.shape) for o in s.get_outputs()]
    return ins, outs


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "workloads/whisper/models/dynamic-onnx"
    th = os.cpu_count()
    print(f"ORT {ort.__version__}  threads={th}\n")

    enc = sess(os.path.join(d, "encoder_model.onnx"), th)
    ei, eo = io(enc)
    print("ENCODER inputs :", ei)
    print("ENCODER outputs:", eo)
    feat = np.random.randn(1, 80, 3000).astype(np.float32)
    for _ in range(2):
        t = time.time(); ehs = enc.run(None, {"input_features": feat})[0]; et = (time.time()-t)*1000
    print(f"encoder fwd    : {et:.1f} ms  -> {ehs.shape}\n")

    dec = sess(os.path.join(d, "decoder_model.onnx"), th)
    di, do = io(dec)
    print("DECODER(no-past) inputs :", di)
    ids = np.array([[50257, 50362]], dtype=np.int64)  # sot, notimestamps
    feed = {"input_ids": ids, "encoder_hidden_states": ehs}
    for _ in range(2):
        t = time.time(); _ = dec.run(None, feed); dt = (time.time()-t)*1000
    print(f"decoder no-past (len 2) : {dt:.1f} ms")
    # simulate what optimum does: reprocess growing sequence each step
    tot = 0.0
    for L in range(2, 32):
        ids = np.array([list(range(50257, 50257 + L))], dtype=np.int64)
        t = time.time(); _ = dec.run(None, {"input_ids": ids, "encoder_hidden_states": ehs}); tot += (time.time()-t)*1000
    print(f"decoder no-past x30 growing (O(n^2) sim): {tot:.1f} ms total\n")

    mp = os.path.join(d, "decoder_with_past_model.onnx")
    if os.path.exists(mp):
        wd = sess(mp, th)
        wi, wo = io(wd)
        print("DECODER(with-past) inputs :", [x[0] for x in wi])
        print("DECODER(with-past) shapes :", [(x[0], x[1]) for x in wi[:4]], "...")


if __name__ == "__main__":
    main()
