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

Both are now Unicode-aware. They were not always: the original walked bytes with
the C-locale classifiers, which answer false for everything >= 0x80, so all
non-ASCII was silently deleted. That was not a rounding error. The accented byte
was dropped rather than folded, so "Ondrej" MISMATCHED an accented reference
while two different accented letters compared EQUAL, and a reference in a script
with no ASCII normalised to nothing, which made any hypothesis in that script
score a perfect 0.0 -- Japanese against Chinese included.

Fixing it moved no committed baseline, because every ref and hyp in eval.jsonl is
pure ASCII. That was checked before the change, not hoped for afterwards.

Case folding covers ASCII, Latin-1 Supplement, Latin Extended-A, Greek and
Cyrillic. Scripts outside that set are left alone, which is right rather than
merely cheap: CJK, Kana, Hangul, Arabic, Hebrew, Devanagari and Thai have no case
to fold. Python could of course just call str.lower(), but this module has to
agree with a dependency-free C++ header byte for byte, so it deliberately does
only what that header does.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "normalize_text",
    "simple_lowercase",
    "is_non_ascii_separator",
    "split_words",
    "edit_distance",
    "error_rate",
    "wer",
    "cer",
]


def _c_isalnum(cp: int) -> bool:
    return (48 <= cp <= 57) or (65 <= cp <= 90) or (97 <= cp <= 122)


def _c_isspace(cp: int) -> bool:
    # space, \t, \n, \v, \f, \r
    return cp == 32 or 9 <= cp <= 13


def _c_ispunct(cp: int) -> bool:
    return 33 <= cp <= 126 and not _c_isalnum(cp)


def simple_lowercase(cp: int) -> int:
    """Port of npu_inference_bench::simple_lowercase.

    Not str.lower(): this has to agree with a dependency-free C++ header, which
    covers ASCII, Latin-1 Supplement, Latin Extended-A, Greek and Cyrillic and
    leaves the caseless scripts alone.
    """
    if cp < 0x80:
        return cp + 0x20 if 0x41 <= cp <= 0x5A else cp
    if 0x00C0 <= cp <= 0x00D6 or 0x00D8 <= cp <= 0x00DE:
        return cp + 0x20
    if 0x0100 <= cp <= 0x0137:
        if cp in (0x0130, 0x0131):
            return cp
        return cp + 1 if cp % 2 == 0 else cp
    if 0x0139 <= cp <= 0x0148:
        return cp + 1 if cp % 2 == 1 else cp
    if 0x014A <= cp <= 0x0177:
        return cp + 1 if cp % 2 == 0 else cp
    if cp == 0x0178:
        return 0x00FF
    if 0x0179 <= cp <= 0x017E:
        return cp + 1 if cp % 2 == 1 else cp
    if cp == 0x017F:
        return 0x73  # long s -> s
    if cp == 0x0386:
        return 0x03AC
    if 0x0388 <= cp <= 0x038A:
        return cp + 0x25
    if cp == 0x038C:
        return 0x03CC
    if 0x038E <= cp <= 0x038F:
        return cp + 0x3F
    if 0x0391 <= cp <= 0x03A1 or 0x03A3 <= cp <= 0x03AB:
        return cp + 0x20
    if 0x0400 <= cp <= 0x040F:
        return cp + 0x50
    if 0x0410 <= cp <= 0x042F:
        return cp + 0x20
    return cp


def is_non_ascii_separator(cp: int) -> bool:
    """Port of npu_inference_bench::is_non_ascii_separator."""
    return (
        0x00A0 <= cp <= 0x00BF
        or cp in (0x00D7, 0x00F7)
        or 0x2000 <= cp <= 0x206F
        or 0x2E00 <= cp <= 0x2E7F
        or 0x3000 <= cp <= 0x303F
        or 0xFE10 <= cp <= 0xFE1F
        or 0xFE30 <= cp <= 0xFE4F
        or 0xFF01 <= cp <= 0xFF20
        or 0xFF3B <= cp <= 0xFF40
        or 0xFF5B <= cp <= 0xFF65
    )


def normalize_text(text: str) -> str:
    """Lowercase, keep alphanumerics and the apostrophe, collapse everything else.

    Punctuation becomes a separator rather than vanishing, so "up-guards" is two
    words and not one. Letters of every script survive.
    """
    out: list[str] = []

    def separate() -> None:
        if out and out[-1] != " ":
            out.append(" ")

    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            if _c_isalnum(cp):
                out.append(ch.lower())
            elif cp == 0x27:  # '
                out.append("'")
            elif _c_isspace(cp) or _c_ispunct(cp):
                separate()
        elif is_non_ascii_separator(cp):
            separate()
        else:
            out.append(chr(simple_lowercase(cp)))

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

    # Code points, not bytes: over UTF-8 a single wrong accented letter would
    # count as three errors and penalise a language for its encoding.
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
    print("non-ASCII survives and folds")
    check("czech survives", normalize_text("Ond\u0159ej"), "ond\u0159ej")
    check("czech folds", normalize_text("OND\u0158EJ"), "ond\u0159ej")
    check("german folds", normalize_text("H\u00d6HLE"), "h\u00f6hle")
    check("s-caron folds", normalize_text("\u0160koda"), "\u0161koda")
    check("greek folds", normalize_text("\u0391\u0392"), "\u03b1\u03b2")
    check("cyrillic folds", normalize_text("\u0414\u0410"), "\u0434\u0430")
    check("cjk survives", normalize_text("\u65e5\u672c\u8a9e"), "\u65e5\u672c\u8a9e")

    print("unicode punctuation still separates")
    check("curly quote separates", normalize_text("don\u2019t"), "don t")
    check("ellipsis separates", normalize_text("wait\u2026 now"), "wait now")
    check("nbsp separates", normalize_text("a\u00a0b"), "a b")
    check("guillemets separate", normalize_text("\u00aboui\u00bb"), "oui")
    check(
        "ideographic space separates",
        normalize_text("\u65e5\u3000\u672c"),
        "\u65e5 \u672c",
    )

    print("errors that used to be invisible")
    check("accent is a real difference", wer("Ond\u0159ej", "Ondrej"), 1.0)
    check("two accents are not equal", wer("Ond\u0159ej", "Ond\u0161ej"), 1.0)
    check("case alone is not", wer("Ond\u0159ej", "OND\u0158EJ"), 0.0)
    check("japanese vs chinese", wer("\u65e5\u672c", "\u4e2d\u56fd"), 1.0)
    check(
        "cer counts code points, not bytes",
        error_rate("h\u00f6hle", "h\u00e4hle").char_edits,
        1,
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


def _cross_check(dump_path: str) -> int:
    """Diff this port against the C++ implementation on the shared corpus.

    The C++ side writes "<input>\\t<normalized>" for every case in
    src/tests/normalization-cases.txt; this reads that back and normalises the
    same inputs here. Checking the two implementations against EACH OTHER is the
    point: for a long time they disagreed while both passed their own tests,
    because each had its own separately invented expectations.
    """
    path = Path(dump_path)
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 2

    mismatches = []
    checked = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        # Split on the LAST tab, not the first: an input may contain a tab (one
        # of the cases does, to check that it collapses), while the normalised
        # output never can, because a tab is whitespace and becomes a space.
        source, _, expected = raw.rpartition("\t")
        checked += 1
        got = normalize_text(source)
        if got != expected:
            mismatches.append((source, expected, got))

    print(f"{checked} shared case(s) compared against the C++ implementation")
    if mismatches:
        print(f"{len(mismatches)} DISAGREEMENT(S):")
        for source, expected, got in mismatches:
            print(f"  input {source!r}")
            print(f"    c++    {expected!r}")
            print(f"    python {got!r}")
        return 1
    print("the two implementations agree on every case")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    if "--cross-check" in sys.argv:
        index = sys.argv.index("--cross-check")
        if index + 1 >= len(sys.argv):
            print("--cross-check needs the path to the C++ dump", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(_cross_check(sys.argv[index + 1]))
    print(__doc__)
    print("run with --self-test, or --cross-check <c++-dump> to compare ports")
