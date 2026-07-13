"""Build C++ classifier fixtures for the fakeaudio precision variants so they can be
replayed through the --classify harness on any NPU (this closes the Qualcomm QNN /
Hexagon HTP gap the AMD/Intel toolkits could not cover).

Run tools/research/export_fakeaudio_variants.py first (it emits the variant
ONNX models), then this script. It produces four fixture dirs under
models/deepfake/fixtures/:
  fakeaudio-fp16safe     -> model.fp16-safe.onnx        (raw PCM input, whole-graph drop-in)
  fakeaudio-bb-fp32      -> model.backbone-fp32.onnx    (mel input; Qualcomm HTP runs this as-is)
  fakeaudio-bb-int8      -> model.backbone-int8.onnx    (mel input from the CPU front-end)
  fakeaudio-bb-npu-intel -> model.backbone.npu-intel.onnx (mel input; the attention-bias
                            expanded backbone that Intel's OpenVINO vpux compiler accepts --
                            this is the fixture the C++ benchmark replays for fakeaudio on the
                            Intel NPU; see results/fakeaudio-intel-npu.md and fakeaudio_npu_intel.py)

expected_p in every fixture is the FULL fp32 model's CPU probability, so the C++
harness' max_abs_p_diff measures end-to-end agreement vs the trusted CPU answer
(directly comparable to the 0.93 the unmodified model shows on the HTP).
"""
import os
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FA = os.path.join(ROOT, "models", "deepfake", "fakeaudio")
FIXROOT = os.path.join(ROOT, "models", "deepfake", "fixtures")
BASEFIX = os.path.join(FIXROOT, "fakeaudio")
SAMPLES = ["deepfake_1", "deepfake_2", "real_1", "real_2", "real_3"]
WINDOW = 308700
POS_IDX, THRESH, POS, NEG = 1, 0.5, "deepfake", "real"


def sess(p):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])


def softmax_pos(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return float((e / e.sum(axis=-1, keepdims=True))[0, POS_IDX])


def load_labels():
    labels = {}
    with open(os.path.join(BASEFIX, "samples.tsv"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) >= 2:
                labels[c[0]] = c[1]
    return labels


def load_pcm():
    feeds = {}
    for s in SAMPLES:
        b = os.path.join(BASEFIX, "data", f"{s}__input.bin")
        feeds[s] = np.fromfile(b, dtype=np.float32).reshape(1, WINDOW)
    return feeds


def write_fixture(name, onnx_rel, in_name, dtype, tensors, ref_p, labels):
    out = os.path.join(FIXROOT, name)
    data = os.path.join(out, "data")
    os.makedirs(data, exist_ok=True)
    with open(os.path.join(out, "model.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join([onnx_rel, str(POS_IDX), str(THRESH), POS, NEG]) + "\n")
    srows, irows = [], []
    for s in SAMPLES:
        p = ref_p[s]
        pred = POS if p > THRESH else NEG
        srows.append("\t".join([s, labels[s], pred, f"{p:.9f}"]))
        arr = np.ascontiguousarray(tensors[s])
        fname = f"{s}__{in_name.replace('/', '_')}.bin"
        arr.tofile(os.path.join(data, fname))
        shape = ",".join(str(d) for d in arr.shape)
        irows.append("\t".join([s, in_name, dtype, shape, fname]))
    with open(os.path.join(out, "samples.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("sample_id\tlabel\texpected_pred\texpected_p\n" + "\n".join(srows) + "\n")
    with open(os.path.join(out, "inputs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("sample_id\tinput_name\tdtype\tshape\tfile\n" + "\n".join(irows) + "\n")
    print(f"  wrote {name} ({in_name} {dtype})")


def main():
    labels = load_labels()
    pcm = load_pcm()

    base = sess(os.path.join(FA, "model.onnx"))
    bin_name = base.get_inputs()[0].name  # "input"
    ref_p = {s: softmax_pos(base.run(None, {bin_name: pcm[s]})[0]) for s in SAMPLES}
    print("full-model CPU ref p:", {s: round(ref_p[s], 4) for s in SAMPLES})

    fe = sess(os.path.join(FA, "model.frontend-fp32.onnx"))
    fe_in = fe.get_inputs()[0].name
    mels = {s: fe.run(None, {fe_in: pcm[s]})[0].astype(np.float32) for s in SAMPLES}
    print("mel shape:", mels[SAMPLES[0]].shape)

    bb_in = sess(os.path.join(FA, "model.backbone-fp32.onnx")).get_inputs()[0].name
    bbi_in = sess(os.path.join(FA, "model.backbone-int8.onnx")).get_inputs()[0].name

    write_fixture("fakeaudio-fp16safe", "../../fakeaudio/model.fp16-safe.onnx",
                  bin_name, "f32", pcm, ref_p, labels)
    write_fixture("fakeaudio-bb-fp32", "../../fakeaudio/model.backbone-fp32.onnx",
                  bb_in, "f32", mels, ref_p, labels)
    write_fixture("fakeaudio-bb-int8", "../../fakeaudio/model.backbone-int8.onnx",
                  bbi_in, "f32", mels, ref_p, labels)

    # The Intel-NPU backbone: numerically identical to backbone-fp32 (same mel input,
    # same reference p) but with each attention bias-Add's constant pre-expanded to the
    # scores' full shape so OpenVINO's vpux SDPA fusion can't mis-broadcast it (the LLVM
    # abort). This is the fixture run-suite replays for fakeaudio on the Intel NPU.
    intel_bb = os.path.join(FA, "model.backbone.npu-intel.onnx")
    if os.path.exists(intel_bb):
        bbn_in = sess(intel_bb).get_inputs()[0].name
        write_fixture("fakeaudio-bb-npu-intel", "../../fakeaudio/model.backbone.npu-intel.onnx",
                      bbn_in, "f32", mels, ref_p, labels)
    else:
        print("  (skip fakeaudio-bb-npu-intel: model.backbone.npu-intel.onnx not found -- "
              "run fakeaudio_npu_intel.py or fetch it from HF first)")


if __name__ == "__main__":
    main()
