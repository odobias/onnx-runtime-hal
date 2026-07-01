# whisper-npu-hal

A C++ **hardware-abstraction layer** for running `whisper-tiny(.en)` speech-to-text on
different vendor NPUs behind one stable API. Proof-of-concept for a
hardware-independent runner with swappable per-platform backends.

- **Intel** — OpenVINO GenAI (NPU / GPU / CPU). **Working reference implementation.**
- **AMD** — Ryzen AI / XDNA via ONNX Runtime + VitisAI EP. **Prepared scaffold** (wiring
  present, decode loop TODO; no AMD hardware to verify).
- **Qualcomm** — Snapdragon Hexagon via ONNX Runtime + QNN EP. **Prepared scaffold.**

Build system: **MSBuild / Visual Studio 2026** (`WhisperNpuHal.sln`).
`PlatformToolset=$(DefaultPlatformToolset)`, so it also builds on older VS if needed.

## Design

```
include/whisper_npu/whisper_engine.hpp   Public API: IWhisperEngine, Backend, factory
include/whisper_npu/audio.hpp            Dependency-free 16 kHz mono WAV loader
src/factory.cpp                          create_engine() + backend availability
src/backends/intel/                      OpenVINO GenAI backend (real)
src/backends/amd/                        Ryzen AI / VitisAI backend (scaffold)
src/backends/qualcomm/                   QNN backend (scaffold)
app/main.cpp                             CLI runner + cold/warm load + inference bench
msbuild/*.props                          Shared + per-backend build settings
projects/*/*.vcxproj, WhisperNpuHal.sln  MSBuild projects (Core static lib + App exe)
scripts/                                 setup / build / run helpers
```

The application depends only on `whisper_npu/whisper_engine.hpp`. Backends are selected
at runtime via `create_engine(Backend, EngineOptions)`; which backends exist depends on
the compile-time toggles `EnableIntel` / `EnableAmd` / `EnableQualcomm`. Backends not
compiled with their SDK still link (as throwing stubs) so the repo always builds.

## Model caching (the fast-load story)

The first NPU load compiles the model to a device blob (slow, ~seconds). Set a cache
directory (`EngineOptions::cache_dir`, CLI `--cache <dir>`) and OpenVINO persists that
blob, so subsequent **warm** loads import it and are near-instant. The CLI loads the
engine twice (cold then warm) and prints both times plus the speedup. AMD/Qualcomm
backends have matching cache hooks stubbed (VitisAI EP context cache / QNN context
binary) for when they're implemented.

## Build & run (PowerShell 7)

```powershell
cd C:\Projects\whisper-npu-hal
.\scripts\setup-intel.ps1     # links/downloads the OpenVINO GenAI C++ SDK
.\scripts\build.ps1           # MSBuild Release|x64 (prefers VS 2026)
.\scripts\run.ps1             # NPU, hybrid model, cold/warm cache demo
# or directly:
.\build\x64\Release\WhisperNpuHal.App.exe <model_dir> <audio.wav> intel npu 5 --cache .\build\cache\npu
```

## Adding a backend

1. Implement `create()` / `available()` in `src/backends/<vendor>/...cpp` behind your
   `WHISPER_HAL_<VENDOR>` macro (stub otherwise).
2. Add include/lib/define wiring in `msbuild/backend.<vendor>.props`.
3. Toggle with `/p:Enable<Vendor>=true` (see `Directory.Build.props`).

## Status / caveats

- Only the **Intel** backend is verified. AMD/Qualcomm are structural scaffolds with
  clearly-marked TODOs; they compile as throwing stubs by default.
- 16 kHz mono WAV only (no resampler).
- See `../NPU-FINDINGS.md` and `../NPU-ECOSYSTEM-STATUS.md` for the underlying research.
