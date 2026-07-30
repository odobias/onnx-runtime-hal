// Tests for the scorer that defines what a committed WER baseline means.
//
// A scoring function is a bad thing to leave untested: it never crashes, it just
// returns a plausible number, and every accuracy claim in this repository and in
// AvastClient's whisper gate is expressed in terms of what it returns. The
// non-ASCII handling in particular was wrong for a long time without a single
// failure to show for it.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npu_inference_bench/metrics.hpp"

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) return;
    std::cerr << "  FAIL  " << what << "\n";
    ++failures;
}

void expect_norm(const std::string& in, const std::string& want) {
    const std::string got = npu_inference_bench::normalize_text(in);
    if (got == want) return;
    std::cerr << "  FAIL  normalize(\"" << in << "\") = \"" << got
              << "\", want \"" << want << "\"\n";
    ++failures;
}

void expect_wer(const std::string& ref, const std::string& hyp, double want) {
    const double got = npu_inference_bench::compute_error_rate(ref, hyp).wer;
    if (got > want - 1e-9 && got < want + 1e-9) return;
    std::cerr << "  FAIL  wer(\"" << ref << "\", \"" << hyp << "\") = " << got
              << ", want " << want << "\n";
    ++failures;
}

void test_ascii_rules() {
    expect_norm("Hello   WORLD", "hello world");
    expect_norm("  leading and trailing  ", "leading and trailing");
    expect_norm("tabs\tand\nnewlines", "tabs and newlines");
    expect_norm("", "");
    expect_norm("   ", "");

    // Punctuation separates rather than vanishing, so a hyphenated token is two
    // words. Deleting it instead would make "up-guards" match "upguards" and
    // not "up guards", which is the wrong way round.
    expect_norm("up-guards", "up guards");
    expect_norm("a(b)c[d]", "a b c d");
    expect_norm("Mr. Quilter, is he?", "mr quilter is he");
    expect_norm("...hello!!!", "hello");

    // The apostrophe is the one punctuation mark kept, so a possessive does not
    // compare equal to a plural.
    expect_norm("QUILTER'S", "quilter's");
    expect_wer("quilter's", "quilters", 1.0);
}

void test_non_ascii_survives_and_folds() {
    // Czech, German, French, Spanish: letters survive and fold to lowercase.
    expect_norm("Ond\xC5\x99" "ej", "ond\xC5\x99" "ej");
    expect_norm("OND\xC5\x98" "EJ", "ond\xC5\x99" "ej");   // U+0158 -> U+0159
    expect_norm("H\xC3\xB6hle", "h\xC3\xB6hle");
    expect_norm("H\xC3\x96HLE", "h\xC3\xB6hle");           // U+00D6 -> U+00F6
    expect_norm("\xC5\xA0koda", "\xC5\xA1koda");           // U+0160 -> U+0161
    expect_norm("\xC5\xBD" "ena", "\xC5\xBE" "ena");       // U+017D -> U+017E
    expect_norm("K\xC5\xAERA", "k\xC5\xAFra");             // U+016E -> U+016F

    // Greek and Cyrillic fold too.
    expect_norm("\xCE\x91\xCE\x92", "\xCE\xB1\xCE\xB2");   // ALPHA BETA
    expect_norm("\xD0\x94\xD0\x90", "\xD0\xB4\xD0\xB0");   // DA

    // Caseless scripts pass through untouched instead of disappearing.
    expect_norm("\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E",
                "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E");

    // The regressions that used to pass silently. Every one of these was 0.0 or
    // 1.0 in the byte-wise version, and every one was wrong.
    expect_wer("Ond\xC5\x99" "ej", "Ondrej", 1.0);         // still a real error
    expect_wer("Ond\xC5\x99" "ej", "Ond\xC5\xA1" "ej", 1.0);  // r-caron vs s-caron
    expect_wer("Ond\xC5\x99" "ej", "OND\xC5\x98" "EJ", 0.0);  // case only
    expect_wer("\xE6\x97\xA5\xE6\x9C\xAC", "\xE4\xB8\xAD\xE5\x9B\xBD", 1.0);
    expect_wer("\xE6\x97\xA5\xE6\x9C\xAC", "\xE6\x97\xA5\xE6\x9C\xAC", 0.0);
}

void test_unicode_punctuation_separates() {
    // U+2019 right single quote is punctuation, not the ASCII apostrophe, so it
    // separates: French "qu'on" written with a curly quote becomes two words.
    expect_norm("don\xE2\x80\x99t", "don t");
    expect_norm("wait\xE2\x80\xA6 now", "wait now");
    // U+00A0 no-break space and U+00AB/BB guillemets.
    expect_norm("a\xC2\xA0" "b", "a b");
    expect_norm("\xC2\xAB" "oui\xC2\xBB", "oui");
    // Ideographic space and CJK full stop.
    expect_norm("\xE6\x97\xA5\xE3\x80\x80\xE6\x9C\xAC", "\xE6\x97\xA5 \xE6\x9C\xAC");
}

void test_cer_counts_codepoints() {
    // One wrong accented letter is ONE character error, not three. Over bytes
    // this reference would report 3/5 and punish the language for its encoding.
    const auto er = npu_inference_bench::compute_error_rate("h\xC3\xB6hle", "h\xC3\xA4hle");
    check(er.ref_chars == 5, "cer counts code points, not bytes (ref_chars)");
    check(er.char_edits == 1, "one accented substitution is one char edit");
}

void test_malformed_utf8_is_deterministic() {
    // Truncated and stray-continuation bytes must not throw or hang; they are
    // scored as themselves so a corrupt transcript still produces a number.
    const std::string truncated = "ab\xC5";
    const std::string stray = "ab\x80" "c";
    check(!npu_inference_bench::normalize_text(truncated).empty(),
          "truncated utf-8 still normalizes");
    check(!npu_inference_bench::normalize_text(stray).empty(),
          "stray continuation byte still normalizes");
}

void test_empty_reference() {
    expect_wer("", "", 0.0);
    expect_wer("", "unexpected", 1.0);

    // Not capped: a decoder that rambles must look worse than one that merely
    // gets every word wrong.
    const double runaway =
        npu_inference_bench::compute_error_rate("a b", "x y z w v u").wer;
    check(runaway > 1.0, "insertions push wer above 1.0");
}

void test_committed_baseline_parity() {
    // Reference/hypothesis pairs exactly as committed in
    // src/workloads/eval/eval.jsonl for static-onnx-tiny-multi-7s, with the WER
    // recorded beside them. These are the numbers AvastClient's accuracy gate
    // asserts equality with, so any change to normalization that moves them has
    // broken that gate, whatever else it improved.
    struct Baseline {
        const char* id;
        const char* ref;
        const char* hyp;
        double wer;
    };

    const std::vector<Baseline> committed = {
        {"jfk",
         "AND SO MY FELLOW AMERICANS ASK NOT WHAT YOUR COUNTRY CAN DO FOR YOU "
         "ASK WHAT YOU CAN DO FOR YOUR COUNTRY",
         "And so my fellow Americans ask not what your country can do for you "
         "as what you can do for your country.",
         0.045455},
        {"ls_000",
         "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD "
         "TO WELCOME HIS GOSPEL",
         "Mr. Quilter is the apostle of the middle classes and we are glad to "
         "welcome his gospel.",
         0.058824},
        {"ls_001",
         "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER",
         "Nor is Mr. Quilters' manner less interesting than his matter.",
         0.200000},
        {"ls_006",
         "ON THE GENERAL PRINCIPLES OF ART MISTER QUILTER WRITES WITH EQUAL "
         "LUCIDITY",
         "On the general principles of art and Mr. Quilter writes with equal "
         "lucidity.",
         0.166667},
        {"ls_008",
         "AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN",
         "As for etchings, there are two kinds, British and foreign.",
         0.181818},
        {"ls_010",
         "NEAR THE FIRE AND THE ORNAMENTS FRED BROUGHT HOME FROM INDIA ON THE "
         "MANTEL BOARD",
         "Near the fire, any ornaments Fred brought home from India on the "
         "mental board.",
         0.200000},
    };

    for (const auto& b : committed) {
        const double got = npu_inference_bench::compute_error_rate(b.ref, b.hyp).wer;
        if (got > b.wer - 5e-6 && got < b.wer + 5e-6) continue;
        std::fprintf(stderr,
                     "  FAIL  %s: committed %.6f, recomputed %.6f -- the "
                     "committed baselines no longer describe this scorer\n",
                     b.id, b.wer, got);
        ++failures;
    }
}

}  // namespace

// Runs every check and reports them all before failing, rather than stopping at
// the first: when a normalization rule changes, the useful information is the
// whole set of consequences, not the alphabetically first one.
void test_metrics() {
    failures = 0;
    test_ascii_rules();
    test_non_ascii_survives_and_folds();
    test_unicode_punctuation_separates();
    test_cer_counts_codepoints();
    test_malformed_utf8_is_deterministic();
    test_empty_reference();
    test_committed_baseline_parity();

    if (failures != 0) {
        throw std::runtime_error(std::to_string(failures) +
                                 " metrics test failure(s), listed above");
    }
}

// metrics.hpp is header-only with no dependencies, so these tests can be built
// and run without the solution, ONNX Runtime or any vendor SDK. See
// tools/tmp/build-metrics-tests.ps1 for the few-second edit loop that uses this.
#ifdef NPU_INFERENCE_BENCH_METRICS_TESTS_STANDALONE
namespace {

// Emit "<input>\t<normalized>" for every case in the shared corpus, so the
// Python port can be diffed against this implementation instead of against
// expectations somebody typed out twice. See src/tests/normalization-cases.txt.
int dump_corpus(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::cerr << "cannot open " << path << "\n";
        return 2;
    }
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::cout << line << '\t' << npu_inference_bench::normalize_text(line) << '\n';
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--dump-corpus") {
        return dump_corpus(argv[2]);
    }
    try {
        test_metrics();
        std::cout << "metrics tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metrics tests failed: " << error.what() << "\n";
        return 1;
    }
}
#endif
