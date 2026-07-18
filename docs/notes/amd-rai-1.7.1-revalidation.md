# AMD Ryzen AI re-validation: use RAI 1.7.1 GA (not 1.8.0-beta)

Scratch note carrying AMD's feedback to an AMD machine. The goal is to re-run the
NPU validation under the AMD-recommended toolchain and see whether the FakeAudio /
VitisAI divergence we documented still reproduces, or was an artifact of the beta.

## Feedback from AMD (Slack, 2026-07-14)

> Thanks for sharing the repro details — that is very helpful. I noticed a few
> differences compared to the setup I used internally, which may explain the
> discrepancy:
>
> 1. From your README.md, it looks like you are using RAI 1.8.0-beta, whereas my
>    validation was done using RAI 1.7.1 GA. The RAI 1.8 beta release was published
>    primarily to support MLPerf and is not a fully RAI 1.8-aligned release. As such,
>    we generally advise ISVs not to use it for validation or production evaluations
>    and instead recommend using the RAI 1.7.1 GA release until RAI 1.8 GA becomes
>    available.
> 2. Just to double check, were you running the compilation inside the Ryzen AI conda
>    environment? When installing RAI 1.7.1 GA from
>    https://ryzenai.docs.amd.com/en/latest/inst.html the installer creates a conda
>    environment called `ryzen-ai-1.7.1`, and the compilation script should be executed
>    from within that environment.
>
> Also, could you let me know which AMD NPU driver version is installed on your system?

## What in this repo currently pins 1.8.0-beta (change these for 1.7.1 GA)

- `tools/setup/setup-amd.ps1` (moved from `scripts/` in the repo restructure):
  - `-Version` default `"1.8.0-beta"` (L15)      -> `1.7.1`
  - `-SdkUrl`  default `.../RyzenAI/1.8.0b0/ryzen-ai-lt-1.8.0-beta.exe` (L18)
  - `-DriverUrl` default `.../RyzenAI/1.8.0b0/NPU_RAI_376_WHQL.zip` (L19)
  - `-MinDriverVersion` `32.0.20101.3760` (L20)   (the 1.8.0b0 driver; 1.7.1 GA pairs with its own driver)
  - conda env is derived as `ryzen-ai-$Version` (L35) -> pass `-Version 1.7.1` and it becomes `ryzen-ai-1.7.1`
  - `-InstallDir` defaults to `C:\Program Files\RyzenAI\$Version`
- README no longer states the RAI version after the restructure (it was rewritten), so
  the only place still pinning 1.8.0-beta is the `setup-amd.ps1` defaults above.

Get the 1.7.1 GA installer + matching NPU driver from AMD's install docs
(https://ryzenai.docs.amd.com/en/latest/inst.html) — do NOT trust the hardcoded
1.8.0b0 download URLs above for the GA re-validation.

## What to re-validate under 1.7.1 GA

The NPU findings that were measured on 1.8.0-beta / VitisAI (ORT 1.25.1) and should be
re-checked:

- `results/reports/fakeaudio-logmel-precision.md` -> "NPU reality (AMD XDNA2 / VitisAI)":
  fp32 full model `max_abs_p_diff` 0.996; fp32 backbone-only (CPU mel fed in) still 0.99;
  calibrated int8 backbone 0.51. Claim: VAIML mishandles the HTSAT backbone independently
  of the front-end underflow.
- `results/reports/int8-vitis-quantization.md` (VitisAI int8 path).
- Whisper `onnx-static` on NPU via VitisAI (should port cleanly — sanity check it still does).

## Steps on the AMD box

1. `git checkout tmp/amd-rai-1.7.1-revalidation`
2. Install RAI 1.7.1 GA from the AMD docs; confirm conda env `ryzen-ai-1.7.1` exists.
3. Record the NPU driver version:  `pnputil /enum-drivers`  (or the setup script's
   `Get-CimInstance Win32_PnPSignedDriver | ? { $_.DeviceName -match 'NPU|AMD.*AI' }`).
4. Build/run the VitisAI path FROM INSIDE the `ryzen-ai-1.7.1` conda env (build script
   moved to `tools/build/`, and the old `scripts/run.ps1` is gone — use `benchmark/`):
   `conda run -n ryzen-ai-1.7.1  .\tools\build\build.ps1 -EnableAmd -DisableIntel -RyzenAiDir "C:\Program Files\RyzenAI\1.7.1"`
   then run the Whisper + FakeAudio NPU workloads via `benchmark\run-whisper.ps1` /
   `benchmark\run-suite.ps1` (check their params for the amd/npu device selection).
5. Compare `max_abs_p_diff` against the fp32-baseline probabilities and fill in below.

## To report back to AMD

- [ ] AMD NPU driver version: __________________
- [ ] RAI version actually used: 1.7.1 GA (env `ryzen-ai-1.7.1`), ORT version: __________
- [ ] Compilation run inside the RAI conda env? (yes/no): __________
- [ ] FakeAudio NPU `max_abs_p_diff` under 1.7.1 GA: __________  (beta was 0.996)
- [ ] Whisper onnx-static NPU still clean? __________
