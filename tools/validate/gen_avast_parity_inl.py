"""Generate the AvastClient parity fixture from HAL's own C++ scorer.

AvastClient cannot reach into a HAL checkout at build time, so parity has to be
frozen into something committed. Hand-writing the expectations is exactly how the
three copies of this scorer drifted apart in the first place, each passing its own
separately invented tests, so these expectations are GENERATED from the output of
HAL's metrics_tests binary instead.

Usage:
    python gen_avast_parity_inl.py <dump.tsv> <output.inl>

where dump.tsv comes from `metrics_tests.exe --dump-corpus normalization-cases.txt`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HAL = Path(__file__).resolve().parents[2]


def cpp_literal(text: str) -> str:
    """Escape a UTF-8 string as one or more adjacent C++ literals.

    Hex escapes in C++ are greedy: "\\xC5\\x99ej" parses as a single
    out-of-range escape rather than two bytes followed by "ej". The literal is
    therefore closed and reopened after every hex escape, which is always safe
    and needs no reasoning about what follows.
    """
    parts: list[str] = []
    current = '"'
    for byte in text.encode("utf-8"):
        if byte == 0x22:  # "
            current += '\\"'
        elif byte == 0x5C:  # backslash
            current += "\\\\"
        elif 0x20 <= byte <= 0x7E:
            current += chr(byte)
        else:
            current += f"\\x{byte:02X}"
            parts.append(current + '"')
            current = '"'
    parts.append(current + '"')
    # Drop the empty literals left behind by a trailing split.
    kept = [p for p in parts if p != '""'] or ['""']
    return " ".join(kept)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    dump = Path(sys.argv[1])
    out = Path(sys.argv[2])
    if not dump.exists():
        print(f"missing {dump}", file=sys.stderr)
        return 2

    try:
        commit = subprocess.run(
            ["git", "-C", str(HAL), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is nice to have, not essential
        commit = "unknown"

    rows = []
    for raw in dump.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        # Last tab: an input may contain a tab, normalised output never can.
        source, _, expected = raw.rpartition("\t")
        rows.append((source, expected))

    lines = [
        "// GENERATED FILE -- do not edit by hand.",
        "//",
        "// Expected normalisations produced by whisper-npu-hal's own scorer,",
        f"// npu_inference_bench::normalize_text, at HAL commit {commit}.",
        "// Source corpus: whisper-npu-hal src/tests/normalization-cases.txt.",
        "//",
        "// Regenerate with HAL's src/tests/build-metrics-tests.ps1 followed by",
        "// tools/validate/gen_avast_parity_inl.py. Editing an expectation by hand",
        "// defeats the purpose: these values exist so that this port cannot drift",
        "// from HAL's while both keep passing their own tests, which is precisely",
        "// what happened before.",
        "",
        "// clang-format off",
        f"inline constexpr std::array<NormalizationCase, {len(rows)}> kNormalizationParity{{{{",
    ]
    for source, expected in rows:
        lines.append(f"    {{{cpp_literal(source)},")
        lines.append(f"     {cpp_literal(expected)}}},")
    lines.append("}};")
    lines.append("// clang-format on")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} with {len(rows)} case(s) from HAL {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
