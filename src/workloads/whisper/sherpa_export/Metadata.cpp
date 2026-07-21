#include "stdafx.h"
#include "Metadata.h"

#include <algorithm>
#include <charconv>
#include <stdexcept>
#include <string_view>

namespace asw::whisper
{
    //=========================================================================
    // Public Metadata helpers
    //=========================================================================

    std::int32_t Metadata::LanguageTokenId(std::string_view iso_code) const noexcept
    {
        for (size_t i = 0; i < all_language_codes.size(); ++i)
        {
            if (all_language_codes[i] == iso_code)
                return all_language_tokens[i];
        }
        return -1;
    }

    std::string_view Metadata::LanguageCodeFor(std::int32_t token_id) const noexcept
    {
        for (size_t i = 0; i < all_language_tokens.size(); ++i)
        {
            if (all_language_tokens[i] == token_id)
                return all_language_codes[i];
        }
        return {};
    }

    //=========================================================================
    // Parser helpers
    //=========================================================================

    namespace
    {
        [[nodiscard]] std::int32_t ParseInt(std::string_view s, const char* key)
        {
            std::int32_t v = 0;
            const char* const begin = s.data();
            const char* const end   = s.data() + s.size();
            auto [ptr, ec] = std::from_chars(begin, end, v);
            if (ec != std::errc{})
                throw std::runtime_error(std::string("Metadata: not an int32: ") + key);
            // Reject partial consumption: from_chars stops at the first
            // non-numeric char but reports success. "448x" must NOT be
            // accepted as 448 silently.
            if (ptr != end)
                throw std::runtime_error(
                    std::string("Metadata: trailing garbage in int32 value for ") + key);
            return v;
        }

        /// Range-checked shape dimension. Shape-driving fields must be
        /// strictly positive and below a caller-supplied realistic cap so
        /// that a hostile or corrupted encoder.onnx cannot trigger massive
        /// (or even overflowing) allocations during GreedyDecoder
        /// construction or feature extraction.
        ///
        /// Per-dimension caps are set to ~10x the largest value in any known
        /// Whisper variant (whisper-large-v3 peaks at n_text_layer=32,
        /// n_text_ctx=448, n_text_state=1280, n_vocab~=52000). A COMPOSITE
        /// bound on the KV-cache product is enforced separately in
        /// ValidateCompositeShapes() below.
        [[nodiscard]] std::int32_t ParseShape(std::string_view s,
                                               const char* key,
                                               std::int32_t max_allowed)
        {
            const std::int32_t v = ParseInt(s, key);
            if (v <= 0 || v > max_allowed)
                throw std::runtime_error(
                    std::string("Metadata: shape field '") + key + "' out of range: " +
                    std::to_string(v) + " (must be in [1, " +
                    std::to_string(max_allowed) + "])");
            return v;
        }

        /// Overflow-safe product of three size_t values. Returns nullopt
        /// if any step would wrap around.
        [[nodiscard]] std::optional<size_t> SafeMul3(size_t a, size_t b, size_t c) noexcept
        {
            if (a == 0 || b == 0 || c == 0) return size_t{0};
            const size_t ab = a * b;
            if (ab / a != b) return std::nullopt;
            const size_t abc = ab * c;
            if (abc / ab != c) return std::nullopt;
            return abc;
        }

        /// Composite bound: the greedy decoder allocates TWO buffers of
        /// shape [n_text_layer, 1, n_text_ctx, n_text_state] floats (self-K
        /// and self-V). Even with realistic per-dim caps the product can
        /// legitimately be a few hundred MB, but anything above our
        /// composite budget almost certainly indicates a corrupt or
        /// hostile model and should be rejected before GreedyDecoder's
        /// ctor runs into OOM territory.
        void ValidateCompositeShapes(const Metadata& md)
        {
            // 512 MiB cap per K/V buffer; total decoder-cache allocation
            // is at most 1 GiB. whisper-large-v3 sits at ~73 MB per buffer
            // (32 * 448 * 1280 * 4 = 73 400 320), so this leaves >7x
            // headroom for future variants.
            constexpr size_t kMaxBufferBytes = 512ULL * 1024 * 1024;

            const auto elems = SafeMul3(
                static_cast<size_t>(md.n_text_layer),
                static_cast<size_t>(md.n_text_ctx),
                static_cast<size_t>(md.n_text_state));
            if (!elems)
                throw std::runtime_error(
                    "Metadata: KV-cache shape product overflows size_t "
                    "(n_text_layer * n_text_ctx * n_text_state)");

            // bytes = elems * sizeof(float); also overflow-check.
            const size_t bytes = *elems * sizeof(float);
            if (sizeof(float) != 0 && bytes / sizeof(float) != *elems)
                throw std::runtime_error(
                    "Metadata: KV-cache byte size overflows size_t");

            if (bytes > kMaxBufferBytes)
                throw std::runtime_error(
                    "Metadata: KV-cache per-buffer size " + std::to_string(bytes) +
                    " bytes exceeds budget " + std::to_string(kMaxBufferBytes) +
                    " (n_text_layer=" + std::to_string(md.n_text_layer) +
                    ", n_text_ctx=" + std::to_string(md.n_text_ctx) +
                    ", n_text_state=" + std::to_string(md.n_text_state) + ")");
        }

        /// Validate that a token id lies within the model vocabulary.
        void ValidateTokenId(std::int32_t id, const char* name, std::int32_t n_vocab)
        {
            if (id < 0 || id >= n_vocab)
                throw std::runtime_error(
                    std::string("Metadata: token id '") + name +
                    "' = " + std::to_string(id) +
                    " is outside [0, n_vocab=" + std::to_string(n_vocab) + ")");
        }

        /// Parse a comma/space-separated int32 list, bounded by `max_elements`
        /// to prevent a malformed metadata value (e.g. a list of a million
        /// ids) from forcing pathological allocation before downstream
        /// validation runs. All real Whisper variants keep each list under
        /// ~200 entries (99 languages for multilingual, 32 non-speech
        /// tokens, 3 sot_sequence entries); the cap leaves >50x headroom.
        [[nodiscard]] std::vector<std::int32_t> ParseInt32List(
            std::string_view s,
            size_t           max_elements = 10'000)
        {
            std::vector<std::int32_t> out;
            size_t i = 0;
            while (i < s.size())
            {
                // Skip whitespace and commas.
                while (i < s.size() && (s[i] == ',' || s[i] == ' '))
                    ++i;
                if (i >= s.size())
                    break;

                size_t j = i;
                while (j < s.size() && s[j] != ',' && s[j] != ' ')
                    ++j;

                if (out.size() >= max_elements)
                    throw std::runtime_error(
                        "Metadata: int32 list exceeds cap of " +
                        std::to_string(max_elements) + " elements");

                std::int32_t v = 0;
                const char* token_begin = s.data() + i;
                const char* token_end   = s.data() + j;
                auto [ptr, ec] = std::from_chars(token_begin, token_end, v);
                if (ec != std::errc{} || ptr != token_end)
                    throw std::runtime_error("Metadata: malformed int32 list");
                out.push_back(v);
                i = j;
            }
            return out;
        }

        [[nodiscard]] std::vector<std::string> ParseStringList(
            std::string_view s,
            size_t           max_elements = 10'000)
        {
            std::vector<std::string> out;
            size_t i = 0;
            while (i < s.size())
            {
                while (i < s.size() && s[i] == ',')
                    ++i;
                if (i >= s.size())
                    break;
                size_t j = i;
                while (j < s.size() && s[j] != ',')
                    ++j;
                if (out.size() >= max_elements)
                    throw std::runtime_error(
                        "Metadata: string list exceeds cap of " +
                        std::to_string(max_elements) + " elements");
                out.emplace_back(s.substr(i, j - i));
                i = j;
            }
            return out;
        }

        /// Fetch a metadata value by key or throw. Using raw-key map so we
        /// don't need to go back to the ORT allocator.
        [[nodiscard]] std::string_view Require(
            const std::unordered_map<std::string, std::string>& raw,
            const char* key)
        {
            auto it = raw.find(key);
            if (it == raw.end())
                throw std::runtime_error(std::string("Metadata: missing key: ") + key);
            return it->second;
        }

        [[nodiscard]] std::optional<std::string_view> TryGet(
            const std::unordered_map<std::string, std::string>& raw,
            const char* key)
        {
            auto it = raw.find(key);
            if (it == raw.end())
                return std::nullopt;
            return std::string_view(it->second);
        }
    } // namespace

    //=========================================================================
    // ParseMetadata
    //=========================================================================

    Metadata ParseMetadata(const Ort::Session& encoder_session)
    {
        Ort::AllocatorWithDefaultOptions allocator;
        const Ort::ModelMetadata md = encoder_session.GetModelMetadata();

        // Pull all custom metadata keys into our std::string map.
        Metadata out;
        const auto keys_owned = md.GetCustomMetadataMapKeysAllocated(allocator);
        for (size_t i = 0; i < keys_owned.size(); ++i)
        {
            const char* key = keys_owned[i].get();
            auto value_owned = md.LookupCustomMetadataMapAllocated(key, allocator);
            if (value_owned == nullptr)
                continue;
            out.raw.emplace(std::string(key), std::string(value_owned.get()));
        }

        // Architecture fields. All shape-driving dimensions are range-
        // checked so malformed encoder.onnx metadata cannot force
        // GreedyDecoder's KV-cache allocation (n_text_layer * n_text_ctx
        // * n_text_state floats) into OOM or overflow territory before
        // other Session validations have a chance to run.
        //
        // Per-dimension caps are ~10x the value used by whisper-large-v3,
        // the largest public Whisper variant as of 2024:
        //   n_mels=80, heads=20, layers=32, state=1280, ctx=1500/448,
        //   vocab=51866.
        // A composite bound on (layer * text_ctx * text_state) is enforced
        // separately below (ValidateCompositeShapes).
        out.model_type     = std::string(Require(out.raw, "model_type"));
        out.n_mels         = ParseShape(Require(out.raw, "n_mels"),         "n_mels",         256);
        out.n_audio_ctx    = ParseShape(Require(out.raw, "n_audio_ctx"),    "n_audio_ctx",    8192);
        out.n_audio_state  = ParseShape(Require(out.raw, "n_audio_state"),  "n_audio_state",  16384);
        out.n_audio_head   = ParseShape(Require(out.raw, "n_audio_head"),   "n_audio_head",   256);
        out.n_audio_layer  = ParseShape(Require(out.raw, "n_audio_layer"),  "n_audio_layer",  256);
        out.n_vocab        = ParseShape(Require(out.raw, "n_vocab"),        "n_vocab",        1'000'000);
        out.n_text_ctx     = ParseShape(Require(out.raw, "n_text_ctx"),     "n_text_ctx",     8192);
        out.n_text_state   = ParseShape(Require(out.raw, "n_text_state"),   "n_text_state",   16384);
        out.n_text_head    = ParseShape(Require(out.raw, "n_text_head"),    "n_text_head",    256);
        out.n_text_layer   = ParseShape(Require(out.raw, "n_text_layer"),   "n_text_layer",   256);

        // Special-token ids.
        out.sot           = ParseInt(Require(out.raw, "sot"),            "sot");
        out.eot           = ParseInt(Require(out.raw, "eot"),            "eot");
        out.blank_id      = ParseInt(Require(out.raw, "blank_id"),       "blank_id");
        out.no_speech     = ParseInt(Require(out.raw, "no_speech"),      "no_speech");
        out.transcribe    = ParseInt(Require(out.raw, "transcribe"),     "transcribe");
        out.translate     = ParseInt(Require(out.raw, "translate"),      "translate");
        out.sot_prev      = ParseInt(Require(out.raw, "sot_prev"),       "sot_prev");
        out.sot_lm        = ParseInt(Require(out.raw, "sot_lm"),         "sot_lm");
        out.no_timestamps = ParseInt(Require(out.raw, "no_timestamps"),  "no_timestamps");

        out.sot_sequence  = ParseInt32List(Require(out.raw, "sot_sequence"));

        out.is_multilingual = (ParseInt(Require(out.raw, "is_multilingual"),
                                        "is_multilingual") != 0);

        // Language tables (multilingual only, but the Sherpa export stamps
        // them on EN-only models too as single entries).
        if (auto v = TryGet(out.raw, "all_language_tokens"))
            out.all_language_tokens = ParseInt32List(*v);
        if (auto v = TryGet(out.raw, "all_language_codes"))
            out.all_language_codes = ParseStringList(*v);

        if (!out.all_language_tokens.empty() &&
            out.all_language_tokens.size() != out.all_language_codes.size())
        {
            throw std::runtime_error(
                "Metadata: all_language_tokens and all_language_codes length mismatch");
        }

        // Non-speech token suppression list.
        if (auto v = TryGet(out.raw, "non_speech_tokens"))
            out.non_speech_tokens = ParseInt32List(*v);

        // ---------------------------------------------------------------
        // Post-parse validation.
        // ---------------------------------------------------------------

        // Composite KV-cache size bound. Overflow-safe; fails fast with a
        // descriptive error before GreedyDecoder's ctor hits std::bad_alloc
        // or length_error from the underlying vector resize.
        ValidateCompositeShapes(out);

        // Token-id range validation. Every special and language token id
        // must lie in [0, n_vocab). Without this check, malformed metadata
        // would either trip subtle degradations later (e.g. out-of-range
        // `no_speech` silently forces `no_speech_prob=0` via SoftmaxProb's
        // range guard) or fail late inside ORT with an opaque error.
        ValidateTokenId(out.sot,           "sot",           out.n_vocab);
        ValidateTokenId(out.eot,           "eot",           out.n_vocab);
        ValidateTokenId(out.blank_id,      "blank_id",      out.n_vocab);
        ValidateTokenId(out.no_speech,     "no_speech",     out.n_vocab);
        ValidateTokenId(out.transcribe,    "transcribe",    out.n_vocab);
        ValidateTokenId(out.translate,     "translate",     out.n_vocab);
        ValidateTokenId(out.sot_prev,      "sot_prev",      out.n_vocab);
        ValidateTokenId(out.sot_lm,        "sot_lm",        out.n_vocab);
        ValidateTokenId(out.no_timestamps, "no_timestamps", out.n_vocab);

        for (std::int32_t id : out.sot_sequence)
            ValidateTokenId(id, "sot_sequence element", out.n_vocab);

        for (std::int32_t id : out.all_language_tokens)
            ValidateTokenId(id, "language token", out.n_vocab);

        for (std::int32_t id : out.non_speech_tokens)
            ValidateTokenId(id, "non_speech_tokens element", out.n_vocab);

        // Multilingual consistency: the table must exist and be non-empty
        // when the model is flagged multilingual (LanguageDetector throws
        // otherwise, but reporting this at init time is clearer).
        if (out.is_multilingual && out.all_language_tokens.empty())
            throw std::runtime_error(
                "Metadata: is_multilingual=1 but all_language_tokens is empty");

        return out;
    }

} // namespace asw::whisper
