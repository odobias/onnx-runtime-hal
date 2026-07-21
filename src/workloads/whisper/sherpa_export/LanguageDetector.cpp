#include "stdafx.h"
#include "LanguageDetector.h"

#include <limits>
#include <stdexcept>

namespace asw::whisper
{
    LanguageDetector::LanguageDetector(GreedyDecoder& decoder, const Metadata& meta)
        : _decoder(decoder)
        , _meta(meta)
    {
    }

    LanguageDetector::Detection LanguageDetector::Detect(
        Ort::Value& cross_k,
        Ort::Value& cross_v)
    {
        if (!_meta.is_multilingual)
            throw std::runtime_error("LanguageDetector: model is not multilingual");
        if (_meta.all_language_tokens.empty())
            throw std::runtime_error("LanguageDetector: metadata has no language table");

        _decoder.ResetCache();

        const std::array<std::int64_t, 1> sot_prefix = { _meta.sot };
        auto logits = _decoder.Forward(sot_prefix, 0, cross_k, cross_v);

        // The logits for the ONLY prefix position describe the distribution
        // of the next token. Pick the language-token id with the highest
        // logit.
        float        best_logit = -std::numeric_limits<float>::infinity();
        std::int32_t best_tok   = _meta.all_language_tokens.front();
        for (std::int32_t tok : _meta.all_language_tokens)
        {
            if (tok < 0 || tok >= _meta.n_vocab)
                continue;
            const float v = logits[static_cast<size_t>(tok)];
            if (v > best_logit)
            {
                best_logit = v;
                best_tok   = tok;
            }
        }

        Detection d;
        d.token_id = best_tok;
        d.code     = std::string(_meta.LanguageCodeFor(best_tok));
        return d;
    }

} // namespace asw::whisper
