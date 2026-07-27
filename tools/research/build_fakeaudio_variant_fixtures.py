"""Build C++ classifier fixtures for the fakeaudio precision variants so they can be
replayed through the --classify harness on any NPU (this closes the Qualcomm QNN /
Hexagon HTP gap the AMD/Intel toolkits could not cover).

Run tools/research/export_fakeaudio_variants.py first (it emits the generic
variant ONNX models), then this script. It produces four fixture dirs under
artifacts/workloads/classifiers/fixtures/:
  fakeaudio-fp16safe     -> model.fp16-safe.onnx        (raw PCM input, whole-graph drop-in)
  fakeaudio-bb-fp32      -> model.backbone-fp32.onnx    (mel input; Qualcomm HTP runs this as-is)
  fakeaudio-bb-int8      -> model.backbone-int8.onnx    (mel input from the CPU front-end)
  fakeaudio-bb-expanded-attention-bias
                          -> model.backbone.expanded-attention-bias.onnx (generated here
                             from the generic fp32 backbone by explicitly materializing
                             attention bias broadcasts)

expected_p in every fixture is the FULL fp32 model's CPU probability, so the C++
harness' max_abs_p_diff measures end-to-end agreement vs the trusted CPU answer
(directly comparable to the 0.93 the unmodified model shows on the HTP).
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FA = os.path.join(ROOT, "artifacts", "workloads", "classifiers", "fakeaudio")
FIXROOT = os.path.join(ROOT, "artifacts", "workloads", "classifiers", "fixtures")
BASEFIX = os.path.join(FIXROOT, "fakeaudio")


def _fixture_sample_ids(fix_dir):
    data = os.path.join(fix_dir, "data")
    if not os.path.isdir(data):
        return ["deepfake_1", "deepfake_2", "real_1", "real_2", "real_3"]
    ids = sorted(
        f[:-len("__input.bin")]
        for f in os.listdir(data)
        if f.endswith("__input.bin")
    )
    return ids or ["deepfake_1", "deepfake_2", "real_1", "real_2", "real_3"]


SAMPLES = _fixture_sample_ids(BASEFIX)
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_attention_bias(source_path, output_path):
    """Materialize Add->Softmax bias broadcasts that Intel VPUX mis-fuses."""
    model = onnx.load(source_path)
    graph = model.graph
    initializers = {initializer.name for initializer in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}
    prepend = defaultdict(list)
    patched = 0

    for softmax in graph.node:
        if softmax.op_type != "Softmax":
            continue
        add = producer.get(softmax.input[0])
        if add is None or add.op_type != "Add" or len(add.input) != 2:
            continue

        left, right = add.input
        constant = left if left in initializers else (
            right if right in initializers else None
        )
        if constant is None:
            continue

        scores = right if constant == left else left
        stem = add.name.replace("/", "_").strip("_") or f"attention_bias_{patched}"
        score_shape = f"{stem}__score_shape"
        expanded_bias = f"{stem}__expanded_bias"
        prepend[id(add)].append(
            helper.make_node("Shape", [scores], [score_shape], name=f"{stem}__Shape")
        )
        prepend[id(add)].append(
            helper.make_node(
                "Expand", [constant, score_shape], [expanded_bias],
                name=f"{stem}__Expand"
            )
        )
        add.input[:] = [
            expanded_bias if value == constant else value for value in add.input
        ]
        patched += 1

    rebuilt = []
    for node in graph.node:
        rebuilt.extend(prepend.get(id(node), []))
        rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return patched


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", choices=("all", "intel-npu", "generic-npu"), default="all",
        help=(
            "Build every research fixture, only the Intel OpenVINO NPU fixture, "
            "or only the generic fp32-backbone NPU fixture."
        )
    )
    args = parser.parse_args()

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

    generic_bb = os.path.join(FA, "model.backbone-fp32.onnx")
    bb_in = sess(generic_bb).get_inputs()[0].name

    if args.only == "generic-npu":
        write_fixture(
            "fakeaudio-bb-fp32", "../../fakeaudio/model.backbone-fp32.onnx",
            bb_in, "f32", mels, ref_p, labels
        )
        return

    if args.only == "all":
        bbi_in = sess(os.path.join(FA, "model.backbone-int8.onnx")).get_inputs()[0].name
        write_fixture("fakeaudio-fp16safe", "../../fakeaudio/model.fp16-safe.onnx",
                      bin_name, "f32", pcm, ref_p, labels)
        write_fixture("fakeaudio-bb-fp32", "../../fakeaudio/model.backbone-fp32.onnx",
                      bb_in, "f32", mels, ref_p, labels)
        write_fixture("fakeaudio-bb-int8", "../../fakeaudio/model.backbone-int8.onnx",
                      bbi_in, "f32", mels, ref_p, labels)

    # Generic, semantics-equivalent backbone with each attention bias broadcast
    # explicitly materialized. The OpenVINO NPU recipe uses it to avoid the VPUX
    # SDPA fusion bug, but the graph contains no vendor-specific operators.
    fixed_bb = os.path.join(FA, "model.backbone.expanded-attention-bias.onnx")
    patched = expand_attention_bias(generic_bb, fixed_bb)
    if patched != 7:
        raise RuntimeError(
            f"expected 7 Add->Softmax attention biases, patched {patched}; "
            "refusing to publish an unverified Intel NPU fixture"
        )

    generic_session = sess(generic_bb)
    fixed_session = sess(fixed_bb)
    generic_in = generic_session.get_inputs()[0].name
    fixed_in = fixed_session.get_inputs()[0].name
    max_logit_delta = max(
        float(np.max(np.abs(
            generic_session.run(None, {generic_in: mels[s]})[0] -
            fixed_session.run(None, {fixed_in: mels[s]})[0]
        )))
        for s in SAMPLES
    )
    if max_logit_delta > 1e-5:
        raise RuntimeError(
            f"attention expansion changed CPU logits by {max_logit_delta}"
        )

    fixture_name = "fakeaudio-bb-expanded-attention-bias"
    write_fixture(
        fixture_name, "../../fakeaudio/model.backbone.expanded-attention-bias.onnx",
        fixed_in, "f32", mels, ref_p, labels
    )
    metadata = {
        "schema_version": 1,
        "vendor": "Intel",
        "executor": "OpenVINO NPU",
        "transform": "expand-add-softmax-attention-bias",
        "patched_attention_blocks": patched,
        "source_model": "../../fakeaudio/model.backbone-fp32.onnx",
        "source_model_sha256": sha256(generic_bb),
        "generated_model": "../../fakeaudio/model.backbone.expanded-attention-bias.onnx",
        "generated_model_sha256": sha256(fixed_bb),
        "max_cpu_logit_delta": max_logit_delta,
    }
    metadata_path = os.path.join(FIXROOT, fixture_name, "fixture-step.json")
    with open(metadata_path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"  generated broadcast-expanded backbone: {patched} attention blocks, "
        f"max CPU logit delta {max_logit_delta:.3g}"
    )


if __name__ == "__main__":
    main()
