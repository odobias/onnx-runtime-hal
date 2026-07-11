# AMD VAIML window-partition first-inference crash

Minimal, self-contained reproducer for a native crash in AMD's Vitis AI
Execution Provider (VAIML). The 17 KB ONNX model succeeds on ORT CPU. VAIML
compiles it without errors and creates an inference session, then
`onnxruntime_vitisai_ep.dll` access-violates on the first `Run()`.

## Reproduced environment

- AMD Ryzen AI 9 HX 370 / XDNA 2 NPU
- Ryzen AI SDK 1.8.0-beta
- ONNX Runtime `1.25.1.dev20260617`
- Python 3.12.11
- Windows 11 build 26200

Both a cold compile and a load from the populated VAIML cache reproduce the
same process crash.

## Run

From the repository root:

```powershell
.\repros\amd-vaiml-window-partition\run.ps1
```

Override the SDK Python if needed:

```powershell
.\repros\amd-vaiml-window-partition\run.ps1 `
  -Python C:\path\to\ryzen-ai-env\python.exe
```

The script first runs a CPU control, then starts the VitisAI case. On the
affected stack:

1. CPU returns a finite tensor with shape `[64, 64, 96]`.
2. `aiecompiler` prints `Compilation Complete` with `ERROR:0`.
3. ORT prints `Session created; providers=['VitisAIExecutionProvider', ...]`.
4. The first inference terminates Python with access violation `0xC0000005`
   (signed exit code `-1073741819`).

To verify the cached path after one cold run:

```powershell
.\repros\amd-vaiml-window-partition\run.ps1 -ReuseCache
```

## Minimal graph and data

`model.onnx` contains a `LayerNormalization` followed by the dynamic
shape/reshape/transpose sequence that partitions a `[1,4096,96]` tensor into
64 windows of `[64,96]`. Its SHA-256 is:

```text
8061c31aefe698248a5de79e8498d5b9ec78d1912dfc3381f4d36152b1f29cd0
```

The model has no trained/customer weights or captured inputs: normalization
parameters are synthetic ones/zeros, and `repro.py` generates deterministic
random input with NumPy seed `20260711`.

ORT graph rewrites are deliberately disabled. With normal optimization the
dynamic shape sequence can be folded or assigned to CPU, which prevents VAIML
from claiming the failing structure and hides the defect rather than fixing
it.

## Observed native failure

The top frame is inside `onnxruntime_vitisai_ep.dll`:

```text
Exception Code: 0xC0000005
onnxruntime_vitisai_ep.dll + 0x3524D7B
onnxruntime_vitisai_ep.dll + 0x330F67C
onnxruntime_vitisai_ep.dll + 0x33FB8DF
...
onnxruntime_providers_vitisai.dll + 0x236C4
```

The DLL is stripped and the reported nearby `xir_deserialize_cif` symbol has
large offsets, so that symbol name should not be treated as a reliable source
location.

## Relationship to the original FakeAudio failure

This graph was reduced from the first HTSAT window-partition block while
investigating FakeAudio on VAIML. The complete isolated block exposes a second
runtime defect after successful compilation:

```text
FlexMLDispatcher: HSI input slot count 1 is inconsistent with ifms.size()=4
and deviceBatchSize=1
```

That larger graph contains model-derived structure and is not required for
this report. This directory is the clean vendor-facing reproducer: tiny,
synthetic, deterministic, and free of proprietary tensors.
