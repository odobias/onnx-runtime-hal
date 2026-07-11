// Backend-neutral accuracy metrics for ASR: word/char error rate with a standard
// text normalization. Header-only, no dependencies. Used by the benchmark harness
// to score any backend's transcription against a reference identically.
#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace npu_inference_bench {

// Lowercase, drop punctuation (keep alphanumerics + spaces + apostrophes),
// collapse whitespace. Deliberately simple and language-agnostic-ish; good enough
// for comparing variants on the same references (relative, not absolute, truth).
inline std::string normalize_text(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (unsigned char c : in) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == '\'' ) {
            out.push_back('\'');
        } else if (std::isspace(c) || std::ispunct(c)) {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
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

    const std::vector<char> rc(r.begin(), r.end());
    const std::vector<char> hc(h.begin(), h.end());
    er.ref_chars = static_cast<int>(rc.size());
    er.char_edits = edit_distance(rc, hc);
    er.cer = er.ref_chars ? static_cast<double>(er.char_edits) / er.ref_chars
                          : (hc.empty() ? 0.0 : 1.0);
    return er;
}

}  // namespace npu_inference_bench
