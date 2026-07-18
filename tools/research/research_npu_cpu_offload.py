#!/usr/bin/env python3
"""Do any tsc / fakeaudio ops silently fall back to the CPU when we target the NPU?
(Excluding the fakeaudio log-mel front-end, which we run on CPU on purpose.)

Two independent views, because "on the NPU" means different things per runtime:

  A. NATIVE OpenVINO  -- core.query_model(model, "NPU") reports, per node, which
     device the plugin claims. Ops the NPU can't do are simply absent -> those are
     what a HETERO:NPU,CPU split would push to the CPU. A plain compile_model("NPU")
     is all-or-nothing (it compiled, so 100% of the claimed graph is on the NPU).

  B. ORT OpenVINO-EP  -- how the C++ harness + benchmark actually run. The EP
     partitions the ONNX graph: supported nodes fuse into OpenVINO subgraph(s),
     everything else stays on ORT's CPU EP. We read the profiler JSON and tally the
     real per-node provider, listing every op that executed on CPUExecutionProvider.

Run with the OVEP venv:
    artifacts/venv-ovep\\Scripts\\python.exe scripts\\experiments\\research_npu_cpu_offload.py
(Native-only query still works in the plain artifacts/venv.)
"""
import collections
import glob
import json
import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# NB: validate_tsc / validate_fakeaudio pull in tokenizers + librosa, which live in
# artifacts/venv but NOT artifacts/venv-ovep. Import them lazily so the OVEP profiling path (cached
# feeds only) runs without those deps.

ROOT = os.path.dirname(os.path.dirname(HERE))
DF = os.path.join(ROOT, "workloads", "classifiers")
TSC = os.path.join(DF, "tsc", "model.onnx")
FA_BASE = os.path.join(DF, "fakeaudio", "model.onnx")
FA_FE = os.path.join(DF, "fakeaudio", "model.frontend-fp32.onnx")
FA_BB = os.path.join(DF, "fakeaudio", "model.backbone-fp32.onnx")
FA_BB_NPU = os.path.join(
    DF, "fakeaudio", "model.backbone.expanded-attention-bias.onnx"
)


def op_of(node):
    return node.get_type_name()


# ---------- A. native OpenVINO: which ops does the NPU plugin claim? ----------
def query_native(name, path, static_shapes):
    import openvino as ov
    core = ov.Core()
    if "NPU" not in core.available_devices:
        print(f"[{name}] no NPU device; skipping native query")
        return
    model = core.read_model(path)
    model.reshape(static_shapes)
    total = collections.Counter(op_of(n) for n in model.get_ops())
    supported = core.query_model(model, "NPU")  # {friendly_name: device}
    claimed_names = set(supported.keys())
    off = collections.Counter()
    for n in model.get_ops():
        if n.get_friendly_name() not in claimed_names:
            off[op_of(n)] += 1
    n_total = sum(total.values())
    n_off = sum(off.values())
    print(f"\n[A native query_model NPU] {name}")
    print(f"  ops total: {n_total} | NPU-claimed: {n_total - n_off} | NOT claimed (would offload): {n_off}")
    if n_off:
        print("  offloaded op types:", ", ".join(f"{k}x{v}" for k, v in off.most_common()))
    else:
        print("  offloaded op types: NONE -- every op is claimed by the NPU plugin")
    # Confirm it actually compiles monolithically on the NPU.
    try:
        core.compile_model(model, "NPU")
        print("  compile_model('NPU'): OK (single NPU executable, no runtime CPU split)")
    except Exception as e:
        print(f"  compile_model('NPU') FAILED: {str(e).splitlines()[-1][:120]}")


# ---------- B. ORT OpenVINO-EP: what really ran on the CPU EP? ----------
def free_dim_overrides(so, path, concrete):
    """Pin every symbolic input dim to the concrete feed shape so the NPU can compile."""
    tmp = ort.InferenceSession(path, ort.SessionOptions(), providers=["CPUExecutionProvider"])
    for inp in tmp.get_inputs():
        cshape = concrete.get(inp.name)
        for j, d in enumerate(inp.shape):
            if isinstance(d, str) and cshape is not None:
                so.add_free_dimension_override_by_name(d, int(cshape[j]))


def profile_providers(name, path, feed):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(tempfile.gettempdir(), f"ovep_{name}")
    free_dim_overrides(so, path, {k: v.shape for k, v in feed.items()})
    try:
        sess = ort.InferenceSession(
            path, so, providers=[("OpenVINOExecutionProvider", {"device_type": "NPU"})])
    except Exception as e:
        print(f"\n[B ORT OVEP NPU] {name}: session build FAILED: {str(e).splitlines()[-1][:140]}")
        return
    onames = [o.name for o in sess.get_outputs()]
    for _ in range(3):
        sess.run(onames, feed)
    prof = sess.end_profiling()

    # Dedup by node name -- the profile records every one of the N iterations, so
    # counting raw kernel_time events would multiply the node count by N.
    nodes_by_provider = collections.defaultdict(set)
    cpu_ops = collections.Counter()
    with open(prof, "r", encoding="utf-8") as f:
        events = json.load(f)
    for ev in events:
        if ev.get("cat") != "Node":
            continue
        nm = ev.get("name", "")
        if not nm.endswith("_kernel_time"):
            continue
        args = ev.get("args", {}) or {}
        prov = args.get("provider")
        if not prov:
            continue
        node = nm[: -len("_kernel_time")]
        if node in nodes_by_provider[prov]:
            continue
        nodes_by_provider[prov].add(node)
        if "CPU" in prov and "OpenVINO" not in prov:
            cpu_ops[args.get("op_name", "?")] += 1
    counts = {p: len(s) for p, s in nodes_by_provider.items()}
    ov_fused = sum(len(s) for p, s in nodes_by_provider.items() if "OpenVINO" in p)
    print(f"\n[B ORT OVEP NPU] {name}")
    print(f"  distinct nodes per provider: {counts}")
    print(f"  OpenVINO fused subgraph nodes: {ov_fused}")
    if cpu_ops:
        print(f"  ran on CPUExecutionProvider ({sum(cpu_ops.values())} nodes):",
              ", ".join(f"{k}x{v}" for k, v in cpu_ops.most_common()))
    else:
        print("  ran on CPUExecutionProvider: NONE")
    try:
        os.remove(prof)
    except OSError:
        pass


FEED_CACHE = os.path.join(tempfile.gettempdir(), "npu_offload_feeds.npz")


def build_feeds_and_cache():
    """(artifacts/venv) build real feeds with the full ML stack and cache them so the OVEP
    venv -- which lacks tokenizers/librosa -- can reload them for ORT profiling."""
    import validate_tsc_onnx as tsc
    import validate_fakeaudio_onnx as fa
    from fakeaudio_npu_intel import expand_attention_bias

    tsc_sess = ort.InferenceSession(TSC, providers=["CPUExecutionProvider"])
    tsc_feed = tsc.build_feeds(tsc_sess)[0]["feed"]

    if not os.path.exists(FA_BB_NPU):
        n = expand_attention_bias(FA_BB, FA_BB_NPU)
        print(f"(surgered backbone: expanded {n} attention biases -> {os.path.basename(FA_BB_NPU)})")
    fa_cpu = ort.InferenceSession(FA_BASE, providers=["CPUExecutionProvider"])
    audio = fa.build_feeds(fa_cpu)[0]["feed"]
    fe = ort.InferenceSession(FA_FE, providers=["CPUExecutionProvider"])
    mel = fe.run([fe.get_outputs()[0].name], audio)[0]
    bb_in = ort.InferenceSession(FA_BB_NPU, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    cache = {f"tsc__{k}": v for k, v in tsc_feed.items()}
    cache[f"bb__{bb_in}"] = mel
    np.savez(FEED_CACHE, **cache)
    return tsc_feed, {bb_in: mel}


def load_cached_feeds():
    z = np.load(FEED_CACHE)
    tsc_feed = {k[len("tsc__"):]: z[k] for k in z.files if k.startswith("tsc__")}
    bb_feed = {k[len("bb__"):]: z[k] for k in z.files if k.startswith("bb__")}
    return tsc_feed, bb_feed


def main():
    print(f"ONNX Runtime {ort.__version__} | providers: {ort.get_available_providers()}")
    has_ovep = "OpenVINOExecutionProvider" in ort.get_available_providers()

    if has_ovep:
        if not os.path.exists(FEED_CACHE):
            print(f"no feed cache at {FEED_CACHE}; run this first in artifacts/venv to build it.")
            return 1
        tsc_feed, bb_feed = load_cached_feeds()
    else:
        tsc_feed, bb_feed = build_feeds_and_cache()

    bb_in = next(iter(bb_feed))

    # A. native query (best in artifacts/venv / OpenVINO 2026.2.1; also runs under OVEP's 2025.4.1)
    query_native("tsc (full)", TSC, {k: list(v.shape) for k, v in tsc_feed.items()})
    query_native("fakeaudio backbone (surgered)", FA_BB_NPU, {bb_in: list(bb_feed[bb_in].shape)})

    # B. ORT OpenVINO-EP profiling (only where the EP exists, i.e. artifacts/venv-ovep)
    if has_ovep:
        profile_providers("tsc", TSC, tsc_feed)
        profile_providers("fakeaudio-backbone", FA_BB_NPU, bb_feed)
    else:
        print("\n[B] OpenVINO EP not in this venv -> re-run with artifacts/venv-ovep for the ORT profile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
