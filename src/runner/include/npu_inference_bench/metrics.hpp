// Backend-neutral accuracy metrics for ASR: word/char error rate with a standard
// text normalization. Header-only, no dependencies. Used by the benchmark harness
// to score any backend's transcription against a reference identically.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace npu_inference_bench {

// ---------------------------------------------------------------------------
// UTF-8 and case handling.
//
// An earlier version of this file walked bytes and classified them with the
// C-locale isalnum/isspace/ispunct. Everything >= 0x80 answers false to all
// three, so every non-ASCII byte was silently deleted. That was not a small
// inaccuracy: an accented letter was dropped rather than folded, so the
// unaccented spelling MISMATCHED while two different accented letters compared
// EQUAL, and a reference in a script with no ASCII normalized to the empty
// string, which made any hypothesis in that script score a perfect 0.0.
// Japanese scored against Chinese came out flawless.
//
// The English corpus in src/workloads/eval/eval.jsonl is pure ASCII, so none of
// its committed baselines move as a result of this fix -- verified, not assumed.
// ---------------------------------------------------------------------------

// Decode one UTF-8 sequence starting at `i`, leaving `i` on the last byte
// consumed. Malformed input is not rejected: the offending byte is returned as
// its own code point so that scoring stays deterministic on rubbish rather than
// throwing in the middle of a benchmark.
inline std::uint32_t decode_utf8(const std::string& in, std::size_t& i) {
    const auto lead = static_cast<unsigned char>(in[i]);
    if (lead < 0x80) return lead;

    int extra = 0;
    std::uint32_t cp = 0;
    if ((lead & 0xE0) == 0xC0) { extra = 1; cp = lead & 0x1Fu; }
    else if ((lead & 0xF0) == 0xE0) { extra = 2; cp = lead & 0x0Fu; }
    else if ((lead & 0xF8) == 0xF0) { extra = 3; cp = lead & 0x07u; }
    else return lead;  // stray continuation or 5+ byte lead

    if (i + static_cast<std::size_t>(extra) >= in.size()) return lead;  // truncated
    for (int k = 1; k <= extra; ++k) {
        const auto cont = static_cast<unsigned char>(in[i + static_cast<std::size_t>(k)]);
        if ((cont & 0xC0) != 0x80) return lead;  // not a continuation byte
        cp = (cp << 6) | (cont & 0x3Fu);
    }
    i += static_cast<std::size_t>(extra);
    return cp;
}

inline void encode_utf8(std::uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Simple (1:1) lowercase mapping for the cased scripts a speech benchmark
// realistically meets: ASCII, Latin-1 Supplement, Latin Extended-A, Greek and
// Cyrillic. Anything else is returned unchanged, which is correct rather than
// merely convenient for the scripts most likely to turn up next -- CJK, Kana,
// Hangul, Arabic, Hebrew, Devanagari and Thai have no case to fold.
//
// No full Unicode table and no locale: both would drag a dependency into a
// header that deliberately has none, and neither would change the outcome for
// the languages in artifacts/workloads/speech/multilingual.
inline std::uint32_t simple_lowercase(std::uint32_t cp) {
    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') ? cp + 0x20 : cp;
    }
    // Latin-1 Supplement: C0..D6 and D8..DE map to E0..F6, F8..FE.
    // D7 is the multiplication sign, not a letter.
    if ((cp >= 0x00C0 && cp <= 0x00D6) || (cp >= 0x00D8 && cp <= 0x00DE)) {
        return cp + 0x20;
    }
    // Latin Extended-A. Mostly even/odd pairs, but the parity flips twice and
    // there are three one-off cases, so the ranges are spelled out.
    if (cp >= 0x0100 && cp <= 0x0137) {
        // 0130 (I with dot above) lowercases to a two-code-point sequence and
        // 0131 (dotless i) is already lower; leave both alone.
        if (cp == 0x0130 || cp == 0x0131) return cp;
        return (cp % 2 == 0) ? cp + 1 : cp;
    }
    if (cp >= 0x0139 && cp <= 0x0148) {
        return (cp % 2 == 1) ? cp + 1 : cp;
    }
    if (cp >= 0x014A && cp <= 0x0177) {
        return (cp % 2 == 0) ? cp + 1 : cp;
    }
    if (cp == 0x0178) return 0x00FF;  // Y with diaeresis
    if (cp >= 0x0179 && cp <= 0x017E) {
        return (cp % 2 == 1) ? cp + 1 : cp;
    }
    if (cp == 0x017F) return 's';  // long s
    // Greek.
    if (cp == 0x0386) return 0x03AC;
    if (cp >= 0x0388 && cp <= 0x038A) return cp + 0x25;
    if (cp == 0x038C) return 0x03CC;
    if (cp >= 0x038E && cp <= 0x038F) return cp + 0x3F;
    if ((cp >= 0x0391 && cp <= 0x03A1) || (cp >= 0x03A3 && cp <= 0x03AB)) {
        return cp + 0x20;
    }
    // Cyrillic.
    if (cp >= 0x0400 && cp <= 0x040F) return cp + 0x50;
    if (cp >= 0x0410 && cp <= 0x042F) return cp + 0x20;
    return cp;
}

// True for a non-ASCII code point that separates words rather than forming
// them: spaces, quotes, dashes, ellipses and the like. Everything else >= 0x80
// is treated as a word character, which is the right default -- letters vastly
// outnumber punctuation, and a script this list has never heard of should not
// vanish the way it used to.
inline bool is_non_ascii_separator(std::uint32_t cp) {
    if (cp >= 0x00A0 && cp <= 0x00BF) return true;   // Latin-1 punctuation and symbols
    if (cp == 0x00D7 || cp == 0x00F7) return true;   // multiplication, division
    if (cp >= 0x2000 && cp <= 0x206F) return true;   // general punctuation
    if (cp >= 0x2E00 && cp <= 0x2E7F) return true;   // supplemental punctuation
    if (cp >= 0x3000 && cp <= 0x303F) return true;   // CJK symbols and punctuation
    if (cp >= 0xFE10 && cp <= 0xFE1F) return true;   // vertical forms
    if (cp >= 0xFE30 && cp <= 0xFE4F) return true;   // CJK compatibility forms
    if (cp >= 0xFF01 && cp <= 0xFF20) return true;   // fullwidth punctuation
    if (cp >= 0xFF3B && cp <= 0xFF40) return true;
    if (cp >= 0xFF5B && cp <= 0xFF65) return true;
    return false;
}

// Lowercase, keep alphanumerics and the apostrophe, turn every other
// punctuation or space into a single separator, and trim.
//
// This function defines what a `wer` in src/workloads/eval/eval.jsonl means,
// because the harness bakes its output into those baselines. tools/validate/
// scoring.py and the AvastClient whisper_unit_test Wer.cpp are deliberate
// ports; change one and you must change all three, or the numbers silently
// disagree while every test still passes.
//
// Note that punctuation SEPARATES rather than vanishing, so "up-guards" scores
// as two words.
inline std::string normalize_text(const std::string& in) {
    std::string out;
    out.reserve(in.size());

    const auto separate = [&out]() {
        if (!out.empty() && out.back() != ' ') out.push_back(' ');
    };

    for (std::size_t i = 0; i < in.size(); ++i) {
        const std::uint32_t cp = decode_utf8(in, i);

        if (cp < 0x80) {
            const auto c = static_cast<unsigned char>(cp);
            if (std::isalnum(c)) {
                out.push_back(static_cast<char>(std::tolower(c)));
            } else if (c == '\'') {
                out.push_back('\'');
            } else if (std::isspace(c) || std::ispunct(c)) {
                separate();
            }
            continue;
        }

        if (is_non_ascii_separator(cp)) {
            separate();
        } else {
            encode_utf8(simple_lowercase(cp), out);
        }
    }

    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

inline std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> w;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        size_t j = i;
        while (j < s.size() && s[j] != ' ') ++j;
        if (j > i) w.push_back(s.substr(i, j - i));
        i = j;
    }
    return w;
}

// Character error rate counts CODE POINTS, not bytes. Over UTF-8 bytes a single
// wrong accented letter would score as three errors, so a language that spells
// with them would look worse purely for being encoded in more bytes.
inline std::vector<std::uint32_t> to_codepoints(const std::string& s) {
    std::vector<std::uint32_t> cps;
    cps.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) cps.push_back(decode_utf8(s, i));
    return cps;
}

// Levenshtein edit distance over a generic token sequence.
template <typename T>
int edit_distance(const std::vector<T>& a, const std::vector<T>& b) {
    const size_t n = a.size(), m = b.size();
    if (n == 0) return static_cast<int>(m);
    if (m == 0) return static_cast<int>(n);
    std::vector<int> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) prev[j] = static_cast<int>(j);
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= m; ++j) {
            const int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, cur);
    }
    return prev[m];
}

struct ErrorRate {
    double wer = 0.0;   // word error rate (edits / reference words)
    double cer = 0.0;   // character error rate (edits / reference chars)
    int ref_words = 0;
    int word_edits = 0;
    int ref_chars = 0;
    int char_edits = 0;
};

// Compute WER/CER of `hypothesis` against `reference`, after normalization.
inline ErrorRate compute_error_rate(const std::string& reference, const std::string& hypothesis) {
    const std::string r = normalize_text(reference);
    const std::string h = normalize_text(hypothesis);

    ErrorRate er;

    const auto rw = split_words(r);
    const auto hw = split_words(h);
    er.ref_words = static_cast<int>(rw.size());
    er.word_edits = edit_distance(rw, hw);
    er.wer = er.ref_words ? static_cast<double>(er.word_edits) / er.ref_words
                          : (hw.empty() ? 0.0 : 1.0);

    const auto rc = to_codepoints(r);
    const auto hc = to_codepoints(h);
    er.ref_chars = static_cast<int>(rc.size());
    er.char_edits = edit_distance(rc, hc);
    er.cer = er.ref_chars ? static_cast<double>(er.char_edits) / er.ref_chars
                          : (hc.empty() ? 0.0 : 1.0);
    return er;
}

}  // namespace npu_inference_bench
