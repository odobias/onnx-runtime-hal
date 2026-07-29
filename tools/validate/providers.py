"""ORT execution-provider selection shared by the Whisper validators.

Keeps one definition of what "gpu" or "npu" means, and refuses a silent CPU
fallback: a runner comparing its error rate against the baseline must know which
EP actually executed, otherwise it is comparing CPU against CPU and calling it NPU.
"""

from __future__ import annotations

import onnxruntime as ort

DEVICE_PROVIDERS: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "gpu": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "npu": [
        "VitisAIExecutionProvider",
        "QNNExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ],
}


def pick_providers(device: str, explicit: str | None) -> list[str]:
    available = set(ort.get_available_providers())
    if explicit:
        if explicit not in available:
            raise SystemExit(f"Provider {explicit!r} not available. Have: {sorted(available)}")
        return [explicit]
    wanted = DEVICE_PROVIDERS[device]
    chosen = [p for p in wanted if p in available]
    if not chosen:
        raise SystemExit(
            f"No providers for device={device}. Wanted {wanted}; have {sorted(available)}"
        )
    if device == "gpu" and "DmlExecutionProvider" not in chosen:
        raise SystemExit(
            "GPU requested but DmlExecutionProvider missing. "
            f"Install onnxruntime-directml. Available: {sorted(available)}"
        )
    if device == "npu" and chosen == ["CPUExecutionProvider"]:
        raise SystemExit(
            "NPU requested but no NPU EP registered "
            f"(VitisAI/QNN/OpenVINO). Available: {sorted(available)}"
        )
    return chosen
