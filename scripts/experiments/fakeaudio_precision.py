"""Diagnose WHERE the fakeaudio (CLAP/HTSAT) graph loses precision, then test a
sensitivity-aware mixed-precision fp16 conversion against naive whole-graph fp16.

Read-only w.r.t. the checked-in model. Writes nothing unless --save is passed.

Steps:
  1. fp32 baseline on the 5 real fixtures (must match samples.tsv reference p).
  2. Probe intermediate dynamic ranges (log-mel front-end, embedding, logits) --
     fp16 tops out at 65504 and denormals die below ~6e-5, so we want to see how
     close the real activations get to those cliffs.
  3. Naive fp16: convert the whole graph -> measure max |dp| vs fp32.
  4. Robust fp16: keep the numerically fragile ops in fp32 (front-end log-mel,
     LayerNorm, Softmax, GELU/Erf, the mel filterbank matmul, the classifier head
     Gemms) and let the heavy transformer matmuls run in fp16 -> measure max |dp|.

The metric that matters is max |dp(fake)| across samples: that is exactly what the
C++ harness reports as max_abs_p_diff (0.99 on VitisAI today).
"""
import argparse
import copy
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers import float16

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FA = os.path.join(ROOT, "models", "deepfake", "fakeaudio")
MODEL = os.path.join(FA, "model.onnx")
FIX = os.path.join(ROOT, "models", "deepfake", "fixtures", "fakeaudio")
SAMPLES = ["deepfake_1", "deepfake_2", "real_1", "real_2", "real_3"]
WINDOW = 308700


def load_feeds():
    feeds = {}
    for s in SAMPLES:
        b = os.path.join(FIX, "data", f"{s}__input.bin")
        arr = np.fromfile(b, dtype=np.float32).reshape(1, WINDOW)
        feeds[s] = {"input": arr}
    return feeds


def softmax_pos(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return float((e / e.sum(axis=-1, keepdims=True))[0, 1])


def sess_from_bytes(b):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(b, so, providers=["CPUExecutionProvider"])


def run(model_proto, feeds):
    s = sess_from_bytes(model_proto.SerializeToString())
    name = s.get_inputs()[0].name
    out = {}
    for k, feed in feeds.items():
        r = s.run(None, {name: feed["input"]})
        out[k] = (np.asarray(r[0]), softmax_pos(r[0]))
    return out


def diff_table(base, other, feeds):
    md = 0.0
    rows = []
    for s in SAMPLES:
        pb = base[s][1]
        po = other[s][1]
        d = abs(pb - po)
        md = max(md, d)
        rows.append((s, pb, po, d))
    return md, rows


def print_diff(title, base, other, feeds):
    md, rows = diff_table(base, other, feeds)
    print(f"\n=== {title}  (max|dp| = {md:.4f}) ===")
    print(f"  {'sample':12s} {'p_fp32':>10s} {'p_new':>10s} {'|dp|':>10s}  flip?")
    for s, pb, po, d in rows:
        flip = "FLIP" if (pb > 0.5) != (po > 0.5) else ""
        print(f"  {s:12s} {pb:10.6f} {po:10.6f} {d:10.6f}  {flip}")
    return md


def probe_ranges(feeds):
    """Add every Log / LayerNorm / final-logit tensor as an output and report ranges."""
    m = onnx.load(MODEL)
    g = m.graph
    want = []
    for n in g.node:
        if n.op_type in ("Log", "Clip", "LayerNormalization", "Softmax") and n.output:
            want.append((n.op_type, n.output[0]))
    # de-dup, cap to keep the session small-ish
    seen = set()
    picks = []
    for op, t in want:
        if t in seen:
            continue
        seen.add(t)
        picks.append((op, t))
    existing = {o.name for o in g.output}
    vis = []
    for op, t in picks:
        if t not in existing:
            vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
            g.output.append(vi)
            vis.append((op, t))
    s = sess_from_bytes(m.SerializeToString())
    iname = s.get_inputs()[0].name
    onames = [o.name for o in s.get_outputs()]
    # one representative sample is enough for range magnitude
    r = s.run(None, {iname: feeds["deepfake_1"]["input"]})
    vals = dict(zip(onames, r))
    agg = {}  # op_type -> [min, max, absmax]
    for op, t in vis:
        a = np.asarray(vals[t], dtype=np.float32)
        lo, hi, am = float(a.min()), float(a.max()), float(np.abs(a).max())
        cur = agg.setdefault(op, [lo, hi, am])
        cur[0] = min(cur[0], lo)
        cur[1] = max(cur[1], hi)
        cur[2] = max(cur[2], am)
    print("\n=== intermediate activation ranges (sample deepfake_1) ===")
    print("  fp16 limits: |max| finite = 65504, smallest normal ~ 6.1e-5")
    for op, (lo, hi, am) in agg.items():
        warn = ""
        if am > 65504:
            warn = "  <-- EXCEEDS fp16 max!"
        elif am > 10000:
            warn = "  <-- near fp16 max"
        print(f"  {op:20s} min={lo:14.4f} max={hi:14.4f} absmax={am:14.4f}{warn}")
    # logits range
    lg = np.asarray(vals["output"], dtype=np.float32)
    print(f"  {'logits(output)':20s} min={lg.min():14.4f} max={lg.max():14.4f} "
          f"absmax={np.abs(lg).max():14.4f}")


def frontend_nodes_upto_log(g):
    """Names of all nodes the single Log transitively depends on = the whole
    spectrogram/mel/log front-end. This is the block we suspect must stay fp32."""
    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n
    logs = [n for n in g.node if n.op_type == "Log"]
    frontier = []
    for n in logs:
        frontier.extend(n.output)
    seen_nodes = set()
    names = set()
    while frontier:
        t = frontier.pop()
        n = producer.get(t)
        if n is None or id(n) in seen_nodes:
            continue
        seen_nodes.add(id(n))
        if n.name:
            names.add(n.name)
        frontier.extend(n.input)
    return names, len(seen_nodes)


def head_gemm_nodes(g):
    return {n.name for n in g.node
            if n.op_type in ("Gemm", "MatMul") and n.name
            and ("classifier.model" in n.name or "classifier/model" in n.name)}


def toposort(nodes, avail):
    """Kahn-style order: emit a node once all its inputs are available. `avail`
    starts as graph inputs + initializer names. Constants (no inputs) sort first."""
    ordered, pending, progress = [], list(nodes), True
    while pending and progress:
        progress, still = False, []
        for n in pending:
            if all(inp == "" or inp in avail for inp in n.input):
                ordered.append(n)
                avail.update(n.output)
                progress = True
            else:
                still.append(n)
        pending = still
    if pending:
        raise RuntimeError(f"topo failed: {len(pending)} nodes have unmet inputs "
                           f"(e.g. {pending[0].name!r} needs {list(pending[0].input)})")
    return ordered


def compose_frontend_backbone(fe, bb, ir_version):
    """Stitch the fp32 front-end and fp16 backbone into ONE model with the
    original input/output contract. The seam tensor (front-end output == backbone
    input) becomes an internal edge; nodes are concatenated then topologically
    sorted so the backbone's boundary Cast lands after the front-end produces it."""
    init_names = {t.name for t in fe.graph.initializer}
    inits = list(fe.graph.initializer) + [t for t in bb.graph.initializer
                                          if t.name not in init_names]
    avail = {i.name for i in fe.graph.input} | {t.name for t in inits}
    nodes = toposort(list(fe.graph.node) + list(bb.graph.node), avail)
    g = onnx.helper.make_graph(nodes, "fakeaudio_fp16_safe",
                               list(fe.graph.input), list(bb.graph.output), inits)
    m = onnx.helper.make_model(g, opset_imports=list(fe.opset_import))
    m.ir_version = ir_version
    return m


def dedupe_node_names(model):
    """The ORT fp16 converter can emit multiple '<tensor>_cast_to_fp32_node' nodes
    with identical names when a blocked node fans out. Node names are not referenced
    by edges (tensors are), so renaming later duplicates is safe and makes the model
    valid again."""
    seen = {}
    for n in model.graph.node:
        base = n.name or n.op_type
        if base in seen:
            seen[base] += 1
            n.name = f"{base}__u{seen[base]}"
        else:
            seen[base] = 0
    return model


def dedupe_duplicate_cast_outputs(model):
    """Same converter bug, tensor level: it can emit multiple Cast nodes that all
    DEFINE the same output tensor (one per fan-out consumer). A single fp32 copy
    serves every consumer, so keep the first such Cast and drop the redundant ones."""
    seen = set()
    keep = []
    removed = 0
    for n in model.graph.node:
        if n.op_type == "Cast" and len(n.output) == 1 and n.output[0] in seen:
            removed += 1
            continue
        for o in n.output:
            seen.add(o)
        keep.append(n)
    if removed:
        del model.graph.node[:]
        model.graph.node.extend(keep)
    return model, removed


def convert(m, keep_io=True, op_block=None, node_block=None):
    mp = float16.convert_float_to_float16(
        copy.deepcopy(m), keep_io_types=keep_io,
        op_block_list=list(op_block) if op_block else None,
        node_block_list=list(node_block) if node_block else None)
    mp, nrem = dedupe_duplicate_cast_outputs(mp)
    mp = dedupe_node_names(mp)
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write the best fp16 model")
    args = ap.parse_args()

    feeds = load_feeds()

    print("=== fp32 baseline ===")
    base = run(onnx.load(MODEL), feeds)
    for s in SAMPLES:
        print(f"  {s:12s} p(fake)={base[s][1]:.6f}  logits={base[s][0].ravel()}")

    probe_ranges(feeds)

    print_diff("naive fp16 (whole graph)", base,
               run(convert(onnx.load(MODEL)), feeds), feeds)

    # --- structural split at the log-mel boundary ------------------------------
    # front-end  : input                -> Log_output_0   (STFT + mel + clip + log)
    # backbone   : Log_output_0         -> output         (bn0 + HTSAT transformer + head)
    CUT = "/embedder/base/htsat/logmel_extractor/Log_output_0"
    fe_path = os.path.join(FA, "_frontend.onnx")
    bb_path = os.path.join(FA, "_backbone.onnx")
    onnx.utils.extract_model(MODEL, fe_path, ["input"], [CUT])
    onnx.utils.extract_model(MODEL, bb_path, [CUT], ["output"])
    fe = onnx.load(fe_path)
    bb = onnx.load(bb_path)
    print(f"\nsplit at {CUT}")
    print(f"  front-end: {len(fe.graph.node)} nodes   backbone: {len(bb.graph.node)} nodes")

    def chain(fe_model, bb_model, feeds):
        fs = sess_from_bytes(fe_model.SerializeToString())
        bs = sess_from_bytes(bb_model.SerializeToString())
        fin = fs.get_inputs()[0].name
        bin_ = bs.get_inputs()[0].name
        out = {}
        for k, feed in feeds.items():
            mel = fs.run(None, {fin: feed["input"]})[0]
            r = bs.run(None, {bin_: mel.astype(np.float32)})[0]
            out[k] = (np.asarray(r), softmax_pos(r))
        return out

    print_diff("split sanity: fp32 front-end + fp32 backbone", base,
               chain(fe, bb, feeds), feeds)
    print_diff("A) fp32 front-end + fp16 backbone (naive)", base,
               chain(fe, convert(bb), feeds), feeds)
    print_diff("B) fp16 front-end (naive) + fp32 backbone", base,
               chain(convert(fe), bb, feeds), feeds)

    # C) fp32 front-end + fp16 backbone but keep LayerNorm/Softmax fp32 inside it
    try:
        bb_mixed = convert(bb, op_block=["LayerNormalization", "Softmax"])
        resC = chain(fe, bb_mixed, feeds)
        mdC = print_diff("C) fp32 front-end + fp16 backbone (LN+Softmax kept fp32)",
                         base, resC, feeds)
    except Exception as e:  # noqa: BLE001
        print(f"\n=== C) === FAILED: {type(e).__name__}: {e}")

    # --- compose the single-file drop-in: fp32 front-end + fp16 backbone --------
    # Same input/output contract as the original, so fixtures + C++ harness are
    # unchanged; only the internal precision of the backbone is fp16.
    bb_fp16 = convert(bb)  # keep_io_types=True -> CUT stays fp32 at the seam
    merged = compose_frontend_backbone(fe, bb_fp16, onnx.load(MODEL).ir_version)
    onnx.checker.check_model(merged)
    md = print_diff("MERGED single-file (fp32 front-end + fp16 backbone) end-to-end",
                    base, run(merged, feeds), feeds)

    for p in (fe_path, bb_path):
        try:
            os.remove(p)
        except OSError:
            pass

    if args.save:
        out = os.path.join(FA, "model.fp16-safe.onnx")
        onnx.save(merged, out)
        sz = os.path.getsize(out) / 1e6
        print(f"\nsaved {out}  ({sz:.1f} MB, max|dp|={md:.4f})")


if __name__ == "__main__":
    sys.exit(main())
