"""Shared baseline load/compare/update for the Whisper eval scripts.

A baseline records per-clip WER (and detected language) from a known-good host so
another runner -- especially one on an NPU EP -- can quantify its own drift instead
of eyeballing a markdown table. Comparison is by clip id, so a runner that skips
clips still gets a verdict on the overlap.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "results" / "baselines" / "whisper-static-multi-7s.json"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def load(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def update(
    suite: str,
    rows: list[dict[str, Any]],
    mean_wer: float,
    provider: str,
    path: Path = DEFAULT_PATH,
) -> Path:
    """Write this run's numbers into the baseline under `suite`."""
    data = load(path)
    data.setdefault("model", "whisper/static-onnx-tiny-multi-7s")
    data.setdefault("suites", {})
    data["suites"][suite] = {
        "recorded": {
            "host": platform.node(),
            "os": platform.platform(),
            "provider": provider,
            "commit": _git_commit(),
        },
        "mean_wer": round(mean_wer, 4),
        "clips": {
            r["id"]: {
                "wer": round(float(r["wer"]), 4),
                **({"lang": r["lang_detected"]} if "lang_detected" in r else {}),
                **({"lang": r["lang"]} if "lang" in r else {}),
            }
            for r in rows
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def compare(
    suite: str,
    rows: list[dict[str, Any]],
    mean_wer: float,
    tolerance: float,
    path: Path = DEFAULT_PATH,
) -> tuple[bool, list[str]]:
    """Return (ok, report lines) comparing this run against the stored baseline."""
    data = load(path)
    ref = (data.get("suites") or {}).get(suite)
    if not ref:
        return True, [f"baseline: no '{suite}' entry in {path.name} -- nothing to compare"]

    rec = ref.get("recorded") or {}
    lines = [
        f"baseline '{suite}' from host={rec.get('host', '?')} "
        f"provider={rec.get('provider', '?')} commit={rec.get('commit', '?')}"
    ]
    ref_clips = ref.get("clips") or {}
    ok = True
    lang_drift: list[str] = []
    worst: list[tuple[float, str]] = []
    shared = 0
    for r in rows:
        base = ref_clips.get(r["id"])
        if base is None:
            continue
        shared += 1
        delta = float(r["wer"]) - float(base["wer"])
        worst.append((delta, r["id"]))
        got_lang = r.get("lang_detected") or r.get("lang")
        if base.get("lang") and got_lang and base["lang"] != got_lang:
            lang_drift.append(f"{r['id']}: {base['lang']} -> {got_lang}")

    if not shared:
        return True, lines + ["baseline: no overlapping clip ids -- nothing to compare"]

    mean_delta = mean_wer - float(ref["mean_wer"])
    lines.append(
        f"baseline: clips={shared} mean_wer={100 * mean_wer:.1f}% "
        f"vs {100 * float(ref['mean_wer']):.1f}% (delta {100 * mean_delta:+.1f} pts, "
        f"tolerance +{100 * tolerance:.1f})"
    )
    worst.sort(reverse=True)
    for delta, clip_id in worst[:3]:
        if abs(delta) >= 0.005:
            lines.append(f"  {clip_id}: {100 * delta:+.1f} pts")

    if lang_drift:
        ok = False
        lines.append("baseline FAIL: detected language differs from baseline")
        lines.extend(f"  {d}" for d in lang_drift)
    if mean_delta > tolerance:
        ok = False
        lines.append(
            f"baseline FAIL: mean WER regressed {100 * mean_delta:+.1f} pts "
            f"(> +{100 * tolerance:.1f} allowed)"
        )
    elif ok:
        lines.append("baseline OK")
    return ok, lines
