"""WER/CER scoring, kept byte-for-byte identical to the C++ harness.

The authoritative implementation is npu_inference_bench::normalize_text and
compute_error_rate in src/runner/include/npu_inference_bench/metrics.hpp. That
is what bakes the `wer` and `cer` values into src/workloads/eval/eval.jsonl, so
it defines what a baseline in this repository means.

This module is a deliberate port of it, not an independent reimplementation.
Three copies of a subtly different scorer used to live in
eval_static_whisper_7s_multi.py, eval_static_whisper_multilingual.py and
compare_whisper_wer_refs.py; all three deleted the apostrophe, which the C++
scorer keeps. That made them report a LOWER error rate than the baseline they
were being compared against -- 0.100000 against a committed 0.200000 on ls_001
and 0.250000 against 0.264706 on ls_004 -- so a genuine regression could hide
inside the gap, disguised as an improvement.

Run `python scoring.py --self-test` to check the port against every committed
baseline and against the edge cases that corpus happens not to exercise.

KNOWN LIMITATION, inherited on purpose
--------------------------------------
The C++ scorer walks bytes and uses the C-locale std::isalnum / isspace /
ispunct. Every byte >= 0x80 therefore matches no branch and is dropped, so
non-ASCII text does not survive normalisation:

    "Ondrej rekl" with accents -> "ondej ekl"      (accents deleted)
    Japanese                   -> ""               (nothing left at all)

For Latin scripts this silently degrades WER into an accent-insensitive metric.
For scripts with no ASCII at all the reference normalises to empty, ref_words is
zero, and the result collapses to 0.0 or 1.0 with no signal in between. Six
clips in artifacts/workloads/speech/multilingual/manifest.jsonl have non-ASCII
references and are affected.

This port reproduces that behaviour rather than quietly improving on it, because
diverging here would mean the Python and C++ numbers disagree again -- which is
the exact bug being fixed. Making the metric Unicode-aware is a separate change
that invalidates every committed baseline and needs a re-bake.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "normalize_text",
    "normalize_unicode",
    "split_words",
    "edit_distance",
    "error_rate",
    "wer",
    "cer",
]


def _c_isalnum(byte: int) -> bool:
    return (48 <= byte <= 57) or (65 <= byte <= 90) or (97 <= byte <= 122)


def _c_isspace(byte: int) -> bool:
    # space, \t, \n, \v, \f, \r
    return byte == 32 or 9 <= byte <= 13


def _c_ispunct(byte: int) -> bool:
    return 33 <= byte <= 126 and not _c_isalnum(byte)


def normalize_text(text: str) -> str:
    """Lowercase, keep alphanumerics and the apostrophe, collapse everything else.

    Punctuation becomes a separator rather than vanishing, so "up-guards" is two
    words and not one. Bytes >= 0x80 are dropped; see the module docstring.
    """
    out: list[str] = []
    for byte in text.encode("utf-8"):
        if _c_isalnum(byte):
            out.append(chr(byte).lower())
        elif byte == 0x27:  # '
            out.append("'")
        elif _c_isspace(byte) or _c_ispunct(byte):
            if out and out[-1] != " ":
                out.append(" ")
    while out and out[-1] == " ":
        out.pop()
    return "".join(out)


def normalize_unicode(text: str) -> str:
    """Same shape as normalize_text, but letters of every script survive.

    Do NOT score baselines with this: it disagrees with the C++ harness on any
    non-ASCII text and would reopen the bug this module exists to close. It is
    here for the soft multilingual diagnostics -- content overlap and character
    recall -- whose entire purpose is judging non-English output, and which
    normalize_text would reduce to noise by deleting the very characters that
    make a transcript Czech rather than English.
    """
    out: list[str] = []
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch == "'":
            out.append("'")
        elif out and out[-1] != " ":
            out.append(" ")
    while out and out[-1] == " ":
        out.pop()
    return "".join(out)


def split_words(text: str) -> list[str]:
    return [w for w in text.split(" ") if w]


def edit_distance(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


@dataclass
class ErrorRate:
    wer: float = 0.0
    cer: float = 0.0
    ref_words: int = 0
    word_edits: int = 0
    ref_chars: int = 0
    char_edits: int = 0


def error_rate(reference: str, hypothesis: str) -> ErrorRate:
    r = normalize_text(reference)
    h = normalize_text(hypothesis)

    result = ErrorRate()

    rw, hw = split_words(r), split_words(h)
    result.ref_words = len(rw)
    result.word_edits = edit_distance(rw, hw)
    result.wer = result.word_edits / result.ref_words if rw else (0.0 if not hw else 1.0)

    rc, hc = list(r), list(h)
    result.ref_chars = len(rc)
    result.char_edits = edit_distance(rc, hc)
    result.cer = result.char_edits / result.ref_chars if rc else (0.0 if not hc else 1.0)
    return result


def wer(reference: str, hypothesis: str) -> float:
    return error_rate(reference, hypothesis).wer


def cer(reference: str, hypothesis: str) -> float:
    return error_rate(reference, hypothesis).cer


def _self_test() -> int:
    failures: list[str] = []

    def check(label: str, got, want) -> None:
        if isinstance(want, float):
            ok = abs(got - want) < 5e-6
        else:
            ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}: {got!r}")
        if not ok:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    print("normalisation")
    check("apostrophe survives", normalize_text("Quilter's"), "quilter's")
    check("punctuation separates", normalize_text("up-guards"), "up guards")
    check("whitespace collapses", normalize_text("  a \t\n b  "), "a b")
    check("trailing punctuation trimmed", normalize_text("hello."), "hello")
    check("leading punctuation drops", normalize_text(".hello"), "hello")
    check("accents deleted (known limitation)", normalize_text("Ond\u0159ej"), "ondej")
    check("cjk deleted (known limitation)", normalize_text("\u65e5\u672c\u8a9e"), "")

    print("unicode variant, for the soft multilingual metrics only")
    check("accents survive", normalize_unicode("Ond\u0159ej"), "ond\u0159ej")
    check("cjk survives", normalize_unicode("\u65e5\u672c\u8a9e"), "\u65e5\u672c\u8a9e")
    check("still separates punctuation", normalize_unicode("up-guards"), "up guards")
    check("still keeps the apostrophe", normalize_unicode("Quilter's"), "quilter's")
    check(
        "agrees with the C++ port on pure ASCII",
        normalize_unicode("Up-Guards and at 'em!"),
        normalize_text("Up-Guards and at 'em!"),
    )

    print("scoring")
    check("identical is zero", wer("a b c", "a b c"), 0.0)
    check("one substitution of three", wer("a b c", "a b d"), 1 / 3)
    check("empty ref, empty hyp", wer("", ""), 0.0)
    check("empty ref, some hyp", wer("", "a"), 1.0)
    check("apostrophe is a real difference", wer("quilter's", "quilters"), 1.0)

    eval_path = Path(__file__).resolve().parents[2] / "src" / "workloads" / "eval" / "eval.jsonl"
    print(f"parity with committed baselines ({eval_path.name})")
    if not eval_path.exists():
        failures.append(f"missing {eval_path}")
        print(f"  FAIL  {eval_path} not found")
    else:
        checked = 0
        with eval_path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for workload, base in (row.get("baselines") or {}).items():
                    if "wer" not in base or "hyp" not in base:
                        continue
                    checked += 1
                    got = round(wer(row["ref"], base["hyp"]), 6)
                    want = float(base["wer"])
                    if abs(got - want) >= 5e-6:
                        failures.append(
                            f"{row['id']}/{workload}: committed {want:.6f}, recomputed {got:.6f}"
                        )
        print(f"  {'ok  ' if not failures else 'FAIL'}  {checked} baseline(s) reproduced")

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
    print("run with --self-test to verify the port")
