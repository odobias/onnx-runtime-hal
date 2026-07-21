#include "stdafx.h"

#include <asw/framework/whisper/session.h>

#include "BpeTokenizer.h"
#include "FeatureExtractor.h"
#include "GreedyDecoder.h"
#include "LanguageDetector.h"
#include "Metadata.h"
#include "Resampler.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace asw::whisper
{
    //=========================================================================
    // Session::Impl
    //=========================================================================

    struct Session::Impl
    {
        // Member declaration order is significant -- C++ initializes members
        // in declaration order regardless of the mem-initializer-list order.
        // metadata (depends on encoder_session) must follow encoder_session;
        // decoder (depends on decoder_session + metadata) must follow both.
        Options                            options;
        Ort::Env*                          env;
        Ort::MemoryInfo                    cpu_mem;
        Ort::Session                       encoder_session;
        Ort::Session                       decoder_session;
        Metadata                           metadata;
        FeatureExtractor                   feature_extractor;
        BpeTokenizer                       tokenizer;
        GreedyDecoder                      decoder;
        std::optional<LanguageDetector>    lang_detector;

        // Precomputed suppression bitmaps indexed by vocab id. Built once
        // in the ctor (see BuildSuppressionMasks), consulted on every
        // argmax step. Kept as uint8_t for vector<bool>-free cache-friendly
        // access; 1 means "skip this id". For tiny.en (n_vocab=51864) the
        // two masks together use ~101 KB, well inside L2.
        std::vector<std::uint8_t>          suppress_mask;        ///< used on steps >= 1
        std::vector<std::uint8_t>          suppress_mask_step0;  ///< used on the first generated step (adds blank_id)

        Impl(Ort::Env& env_ref,
             const Ort::SessionOptions& so,
             const Options& opts)
            : options(opts)
            , env(&env_ref)
            , cpu_mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
            , encoder_session(env_ref, opts.encoder_path.c_str(), so)
            , decoder_session(env_ref, opts.decoder_path.c_str(), so)
            , metadata(ParseMetadata(encoder_session))
            , feature_extractor(metadata.n_mels)
            , tokenizer(opts.tokens_path)
            , decoder(decoder_session, metadata)
        {
            // Sanity: we assume the standard Whisper shape contract.
            // Whisper uses 80 mel bands for every variant up to large-v2 and
            // 128 for large-v3 / large-v3-turbo. The FeatureExtractor computes
            // its filterbank from metadata.n_mels, so both are supported; any
            // other value indicates a non-Whisper or corrupted encoder.
            if (metadata.n_mels != FeatureExtractor::kNMels &&
                metadata.n_mels != FeatureExtractor::kNMelsV3)
            {
                throw std::runtime_error(
                    "Session: unsupported model n_mels " +
                    std::to_string(metadata.n_mels) +
                    " (only 80 and 128 are supported)");
            }
            if (metadata.n_audio_ctx != FeatureExtractor::kNAudioCtx)
                throw std::runtime_error("Session: model n_audio_ctx != 1500 is not supported");

            // Validate max_new_tokens at construction time. Negative values
            // would wrap to a huge size_t in reserve() and zero/negative
            // values must NOT produce any generated tokens (see decode loop).
            // Upper-bound by n_text_ctx since the self-KV cache physically
            // cannot hold more than that many positions.
            if (options.max_new_tokens < 0 ||
                options.max_new_tokens > metadata.n_text_ctx)
            {
                throw std::runtime_error(
                    "Session: invalid max_new_tokens " +
                    std::to_string(options.max_new_tokens) +
                    " (must be in [0, " + std::to_string(metadata.n_text_ctx) + "])");
            }

            // Cross-check tokenizer vs model vocab. A truncated or mismatched
            // tokens.txt would silently produce empty BPE pieces at decode
            // time (BpeTokenizer::Piece() returns {} for out-of-range ids),
            // yielding corrupted or empty transcriptions rather than a hard
            // init failure. Fail fast at Session construction instead.
            //
            // The Sherpa Whisper export stores exactly the *decodable* BPE
            // pieces in tokens.txt: entries for ids [0 .. eot - 1]. Ids at
            // or above `eot` are special control tokens (EOT itself, SOT,
            // language tokens, timestamps) which have no printable text
            // form and are not included. Hence the exact invariant is
            // `tokenizer.Size() == metadata.eot`, NOT `== n_vocab`.
            if (tokenizer.Size() != metadata.eot)
            {
                throw std::runtime_error(
                    "Session: tokenizer size " + std::to_string(tokenizer.Size()) +
                    " does not match model eot " + std::to_string(metadata.eot) +
                    " (mismatched or truncated tokens.txt?)");
            }

            // Sanity: blank id (space token) must have a non-empty piece.
            // Catches tokens.txt that has the right line count but garbled
            // base64 payloads.
            if (metadata.blank_id < 0 || metadata.blank_id >= tokenizer.Size() ||
                tokenizer.Piece(metadata.blank_id).empty())
            {
                throw std::runtime_error(
                    "Session: tokenizer is missing blank_id " +
                    std::to_string(metadata.blank_id) +
                    " (truncated tokens.txt?)");
            }

            if (metadata.is_multilingual)
                lang_detector.emplace(decoder, metadata);

            BuildSuppressionMasks();
        }

        /// Precompute the two suppression bitmaps. Called once from the
        /// ctor. The masks encode, for every vocab id, whether the id is
        /// forbidden during greedy decoding:
        ///
        ///   suppress_mask        -- default mask, used on steps >= 1
        ///   suppress_mask_step0  -- default mask plus blank_id, used on
        ///                           the very first generated step
        ///
        /// By DEFAULT every flag below is off, so both masks are all-zero
        /// and ArgmaxMasked() degenerates to a plain full-vocabulary argmax
        /// -- exactly what sherpa-onnx's offline-Whisper greedy decoder does
        /// (`MaxElementIndex(p_logits, vocab_size)` with no logit masking).
        ///
        /// The masks can OPT IN to OpenAI-reference-style filtering:
        ///
        ///   - options.suppress_non_speech: masks `non_speech_tokens` from
        ///     the encoder metadata. NOTE this set includes `(`, `)`, `[`,
        ///     `]` and music-note tokens -- the tokens Whisper uses to
        ///     caption non-speech audio. Enabling it makes "(music)" /
        ///     "(bird chirping)" impossible and pushes greedy decoding into
        ///     hallucinated speech on music/SFX windows.
        ///
        ///   - options.suppress_timestamps: masks the full timestamp-token
        ///     range [no_timestamps + 1, n_vocab). sherpa does NOT do this;
        ///     it relies on the <|notimestamps|> prefix anchor alone.
        void BuildSuppressionMasks()
        {
            const std::int32_t nv = metadata.n_vocab;
            suppress_mask.assign(static_cast<size_t>(nv), 0);

            if (options.suppress_non_speech)
            {
                for (std::int32_t id : metadata.non_speech_tokens)
                {
                    if (id >= 0 && id < nv)
                        suppress_mask[static_cast<size_t>(id)] = 1;
                }
            }

            if (options.suppress_timestamps)
            {
                // Timestamp tokens start immediately after <|notimestamps|>.
                // Range: [no_timestamps + 1, n_vocab).
                const std::int32_t ts_begin = metadata.no_timestamps + 1;
                if (ts_begin >= 0 && ts_begin < nv)
                {
                    for (std::int32_t id = ts_begin; id < nv; ++id)
                        suppress_mask[static_cast<size_t>(id)] = 1;
                }
            }

            // Step-0 mask: default + blank.
            suppress_mask_step0 = suppress_mask;
            if (options.suppress_blank &&
                metadata.blank_id >= 0 && metadata.blank_id < nv)
            {
                suppress_mask_step0[static_cast<size_t>(metadata.blank_id)] = 1;
            }
        }

        /// Run the encoder forward. Returns a pair of Ort::Value that own the
        /// cross_k / cross_v tensors. These are passed by reference to the
        /// decoder for every subsequent step.
        [[nodiscard]] std::array<Ort::Value, 2> RunEncoder(std::span<float> mel_data)
        {
            const std::int64_t mel_shape[3] = {
                1,
                static_cast<std::int64_t>(metadata.n_mels),
                FeatureExtractor::kNFrames,
            };

            auto mel_tensor = Ort::Value::CreateTensor<float>(
                cpu_mem, mel_data.data(), mel_data.size(),
                mel_shape, 3);

            static constexpr std::array<const char*, 1> input_names  = { "mel" };
            static constexpr std::array<const char*, 2> output_names = {
                "n_layer_cross_k",
                "n_layer_cross_v",
            };

            std::array<Ort::Value, 2> outputs = {
                Ort::Value{nullptr}, Ort::Value{nullptr},
            };

            Ort::RunOptions run_opts{nullptr};
            encoder_session.Run(
                run_opts,
                input_names.data(),
                &mel_tensor, 1,
                output_names.data(),
                outputs.data(), 2);

            return outputs;
        }

        /// Resample input PCM to 16 kHz and pad/trim to exactly 30 s.
        ///
        /// The input is clipped to just the samples needed to produce a
        /// 30 s window at 16 kHz BEFORE any copying or resampling. This
        /// prevents a malformed IPC request with an oversized `samples`
        /// vector from forcing a large allocation or O(N) FIR work over
        /// data that would be thrown away a few lines later.
        [[nodiscard]] std::vector<float> PrepareAudio(
            std::span<const float> pcm, std::int32_t sample_rate)
        {
            // Sample rate must have been validated by the caller; assert
            // the contract defensively for debug builds.
            assert(sample_rate > 0);

            // Enough input samples to produce kChunkSamples (480 000) at
            // 16 kHz, plus a small FIR-margin so the resampler has edge
            // context. Everything beyond is discarded.
            constexpr std::int32_t kFirMarginSamples = 64; // > Resampler half-taps
            const size_t needed_input_samples =
                static_cast<size_t>(sample_rate) *
                    static_cast<size_t>(FeatureExtractor::kChunkSecs) +
                kFirMarginSamples;

            const size_t clip_count = std::min(pcm.size(), needed_input_samples);
            std::span<const float> pcm_clipped(pcm.data(), clip_count);

            std::vector<float> samples_16k;
            if (sample_rate == FeatureExtractor::kSampleRate)
            {
                samples_16k.assign(pcm_clipped.begin(), pcm_clipped.end());
            }
            else
            {
                // Build a per-call resampler (ratio may vary per call).
                Resampler r(sample_rate, FeatureExtractor::kSampleRate);
                samples_16k = r.Process(pcm_clipped);
            }

            samples_16k.resize(
                static_cast<size_t>(FeatureExtractor::kChunkSamples), 0.0f);
            return samples_16k;
        }

        /// Compute softmax probability of a single class `target` from a
        /// row of logits (no numerical-stability tricks needed for a single
        /// target; we use log-sum-exp).
        [[nodiscard]] static float SoftmaxProb(
            std::span<const float> logits, std::int32_t target) noexcept
        {
            if (target < 0 || static_cast<size_t>(target) >= logits.size())
                return 0.0f;

            float mx = *std::max_element(logits.begin(), logits.end());
            double denom = 0.0;
            for (float v : logits)
                denom += std::exp(static_cast<double>(v - mx));
            const double num = std::exp(static_cast<double>(logits[target] - mx));
            return static_cast<float>(num / denom);
        }

        /// Argmax over logits[0..n_vocab-1] masked by a precomputed bitmap
        /// (1 byte per vocab id; 1 means suppressed). Linear scan over
        /// vocab with a single cache-friendly array lookup per id.
        [[nodiscard]] static std::int32_t ArgmaxMasked(
            std::span<const float> logits,
            const std::vector<std::uint8_t>& suppress_mask) noexcept
        {
            std::int32_t best_id = 0;
            float        best_v  = -std::numeric_limits<float>::infinity();
            const std::int32_t n = static_cast<std::int32_t>(logits.size());
            for (std::int32_t i = 0; i < n; ++i)
            {
                if (suppress_mask[static_cast<size_t>(i)]) continue;
                if (logits[i] > best_v)
                {
                    best_v  = logits[i];
                    best_id = i;
                }
            }
            return best_id;
        }

        /// Core Whisper inference. Returns a fully-populated Result.
        [[nodiscard]] Result RunInternal(
            std::span<const float> pcm, std::int32_t sample_rate)
        {
            Result result;

            // 1. Audio preparation: resample + pad/trim to 30 s @ 16 kHz.
            auto samples_16k = PrepareAudio(pcm, sample_rate);

            // 2. Log-mel features: [80, 3000].
            auto mel = feature_extractor.Extract(samples_16k);

            // 3. Encoder forward.
            auto cross = RunEncoder(mel);
            Ort::Value& cross_k = cross[0];
            Ort::Value& cross_v = cross[1];

            // 4. Determine language + task tokens.
            std::string detected_lang;
            std::int32_t lang_tok = 0;
            if (metadata.is_multilingual)
            {
                if (options.language.empty())
                {
                    auto det = lang_detector->Detect(cross_k, cross_v);
                    detected_lang = det.code;
                    lang_tok = det.token_id;
                }
                else
                {
                    lang_tok = metadata.LanguageTokenId(options.language);
                    if (lang_tok < 0)
                        throw std::runtime_error(
                            "Session: unknown language code '" + options.language + "'");
                    detected_lang = options.language;
                }
            }
            else
            {
                detected_lang = "en";
            }

            // 5. Build the SOT prefix tokens FROM metadata.sot_sequence
            //    rather than hard-coding the order. Sherpa-ONNX's export
            //    records the authoritative template for each model:
            //      EN-only:      sot_sequence = [sot]
            //      Multilingual: sot_sequence = [sot, default_lang, default_task]
            //    We start from that template, override language/task with
            //    user choices (or auto-detected language), and append
            //    <|notimestamps|> to run the model in text-only mode.
            //
            //    The <|notimestamps|> token is critical: without it the
            //    model expects to emit timestamped output, and greedy
            //    decoding picks timestamp ids as content -- decoding to
            //    gibberish via the base64 byte-level BPE, especially when
            //    non-English audio is fed to tiny.en (the model's trained
            //    "[Foreign language]" / "[Music]" / "[Applause]" fallback
            //    text tokens get replaced by timestamp noise).
            std::vector<std::int64_t> prefix;
            prefix.reserve(metadata.sot_sequence.size() + 1);
            for (std::int32_t id : metadata.sot_sequence)
                prefix.push_back(static_cast<std::int64_t>(id));

            if (prefix.empty() || prefix[0] != metadata.sot)
                throw std::runtime_error(
                    "Session: metadata.sot_sequence is empty or does not start with sot");

            if (metadata.is_multilingual)
            {
                // Expect [sot, default_lang, default_task]. Replace the
                // language and task slots with the user's (or auto-
                // detected) choice so the same template serves every
                // (lang, task) combination the model supports.
                if (prefix.size() != 3)
                    throw std::runtime_error(
                        "Session: multilingual model metadata sot_sequence "
                        "must be [sot, lang, task], got " +
                        std::to_string(prefix.size()) + " elements");
                prefix[1] = static_cast<std::int64_t>(lang_tok);
                prefix[2] = static_cast<std::int64_t>(
                    (options.task == Task::Translate) ? metadata.translate
                                                       : metadata.transcribe);
            }
            else
            {
                // English-only: sot_sequence is just [sot]. Nothing to
                // substitute; we only need to append <|notimestamps|> below.
                if (prefix.size() != 1)
                    throw std::runtime_error(
                        "Session: English-only model metadata sot_sequence "
                        "must be [sot], got " +
                        std::to_string(prefix.size()) + " elements");
            }

            prefix.push_back(static_cast<std::int64_t>(metadata.no_timestamps));

            // 6. Prefix forward (offset=0). This also writes
            //    GreedyDecoder::LastPrefixLogits() with the last row.
            decoder.ResetCache();
            auto prefix_logits = decoder.Forward(prefix, 0, cross_k, cross_v);

            const std::int32_t nv = metadata.n_vocab;

            // 7. no_speech_prob from row 0 (distribution after SOT).
            std::span<const float> row0(prefix_logits.data(), nv);
            result.no_speech_prob = SoftmaxProb(row0, metadata.no_speech);

            // 8. First generated-token decision from the LAST prefix row.
            const size_t last_row_off =
                (prefix.size() - 1) * static_cast<size_t>(nv);
            std::span<const float> first_row(
                prefix_logits.data() + last_row_off, nv);

            // 9. Greedy loop. Suppression masks were precomputed in the
            //    ctor; the step-0 mask adds blank_id suppression on top of
            //    the default (non_speech + timestamp) mask so the very
            //    first generated token cannot be a bare space.
            //
            //    Honour the hard cap strictly: if the caller requested
            //    zero new tokens, we must emit NONE -- including the
            //    would-be first generated token. max_new_tokens >= 0 was
            //    validated at construction time, so the reserve() cast
            //    is safe.
            std::vector<std::int32_t> generated;
            generated.reserve(static_cast<size_t>(options.max_new_tokens));

            if (options.max_new_tokens > 0)
            {
                std::int32_t next = ArgmaxMasked(first_row, suppress_mask_step0);
                if (next != metadata.eot)
                {
                    generated.push_back(next);

                    std::int64_t offset = static_cast<std::int64_t>(prefix.size());
                    const std::int32_t ctx_limit = metadata.n_text_ctx;

                    // We can write into slots [0 .. ctx_limit-1]. The loop runs
                    // up to and including offset == ctx_limit-1; after the
                    // forward, offset becomes ctx_limit and the loop exits.
                    for (std::int32_t step = 1;
                         step < options.max_new_tokens && offset < ctx_limit;
                         ++step)
                    {
                        const std::int64_t tok =
                            static_cast<std::int64_t>(generated.back());
                        std::array<std::int64_t, 1> one{ tok };
                        auto step_logits =
                            decoder.Forward(one, offset, cross_k, cross_v);
                        ++offset;

                        std::span<const float> row(step_logits.data(), nv);
                        next = ArgmaxMasked(row, suppress_mask);
                        if (next == metadata.eot)
                            break;
                        generated.push_back(next);
                    }
                }
            }

            // 11. Detokenize.
            result.token_ids = std::move(generated);
            result.tokens    = tokenizer.IdsToPieces(result.token_ids);
            result.text      = tokenizer.IdsToText(result.token_ids);
            result.language  = std::move(detected_lang);

            // 12. No-speech gate (opt-in). When the model's own no_speech
            //     head is confident the window contains no speech, Whisper
            //     tends to hallucinate training artefacts (" you", " Thanks
            //     for watching.", " [Music]"). OpenAI's reference decoder
            //     clears the output under this condition; sherpa-onnx does
            //     NOT. We default to sherpa behaviour: no_speech_threshold is
            //     1.0, so this branch never fires unless the caller lowers it
            //     (0.6 is OpenAI's default). `no_speech_prob` is always
            //     populated so the caller can gate externally if it prefers.
            if (result.no_speech_prob > options.no_speech_threshold)
            {
                result.text.clear();
                result.tokens.clear();
                result.token_ids.clear();
            }

            return result;
        }
    };

    //=========================================================================
    // Session public interface
    //=========================================================================

    Session::Session(Ort::Env& env,
                     const Ort::SessionOptions& session_options,
                     const Options& opts)
        : _impl(std::make_unique<Impl>(env, session_options, opts))
    {
    }

    Session::~Session() = default;
    Session::Session(Session&&) noexcept = default;
    Session& Session::operator=(Session&&) noexcept = default;

    Result Session::Run(std::span<const float> pcm, std::int32_t sample_rate)
    {
        if (!_impl)
            throw std::runtime_error("Session: moved-from");

        // Input validation: sample_rate arrives from IPC (model_host client),
        // so we cannot trust it. Reject non-positive values (they would
        // divide-by-zero or produce bogus ratios inside Resampler, whose
        // only debug asserts are stripped in release builds) and reject
        // clearly implausible rates that would force huge allocations.
        static constexpr std::int32_t kMinSampleRate = 4000;    // 4 kHz -- below any speech-grade input
        static constexpr std::int32_t kMaxSampleRate = 384000;  // 384 kHz -- above any consumer device
        if (sample_rate < kMinSampleRate || sample_rate > kMaxSampleRate)
        {
            throw std::runtime_error(
                "Session: invalid sample_rate " + std::to_string(sample_rate) +
                " (must be between " + std::to_string(kMinSampleRate) +
                " and " + std::to_string(kMaxSampleRate) + ")");
        }

        return _impl->RunInternal(pcm, sample_rate);
    }

    const Metadata& Session::GetMetadata() const noexcept
    {
        return _impl->metadata;
    }

} // namespace asw::whisper
