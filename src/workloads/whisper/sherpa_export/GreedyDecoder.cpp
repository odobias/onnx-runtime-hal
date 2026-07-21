#include "stdafx.h"
#include "GreedyDecoder.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace asw::whisper
{
    //=========================================================================
    // GreedyDecoder
    //=========================================================================

    GreedyDecoder::GreedyDecoder(Ort::Session& decoder_session, const Metadata& meta)
        : _session(decoder_session)
        , _meta(meta)
        , _cpu_mem(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        const size_t self_cache_size =
            static_cast<size_t>(meta.n_text_layer) *
            static_cast<size_t>(meta.n_text_ctx)   *
            static_cast<size_t>(meta.n_text_state);
        _self_k_buf.resize(self_cache_size, 0.0f);
        _self_v_buf.resize(self_cache_size, 0.0f);
    }

    void GreedyDecoder::ResetCache()
    {
        std::fill(_self_k_buf.begin(), _self_k_buf.end(), 0.0f);
        std::fill(_self_v_buf.begin(), _self_v_buf.end(), 0.0f);
    }

    //=========================================================================
    // Forward one step
    //=========================================================================

    std::vector<float> GreedyDecoder::Forward(
        std::span<const std::int64_t> tokens,
        std::int64_t                  offset,
        Ort::Value&                   cross_k,
        Ort::Value&                   cross_v)
    {
        const std::int64_t n_tokens = static_cast<std::int64_t>(tokens.size());

        if (n_tokens > 1 && offset != 0)
            throw std::runtime_error("GreedyDecoder::Forward: n_tokens>1 requires offset==0");
        if (n_tokens < 1)
            throw std::runtime_error("GreedyDecoder::Forward: tokens must be non-empty");

        const std::int64_t tokens_shape[2]     = { 1, n_tokens };
        const std::int64_t self_cache_shape[4] = {
            _meta.n_text_layer, 1, _meta.n_text_ctx, _meta.n_text_state,
        };
        const std::int64_t offset_shape[1]     = { 1 };
        std::int64_t       offset_value        = offset;

        auto tokens_tensor = Ort::Value::CreateTensor<std::int64_t>(
            _cpu_mem,
            const_cast<std::int64_t*>(tokens.data()),
            static_cast<size_t>(n_tokens),
            tokens_shape, 2);

        auto self_k_tensor = Ort::Value::CreateTensor<float>(
            _cpu_mem, _self_k_buf.data(), _self_k_buf.size(),
            self_cache_shape, 4);

        auto self_v_tensor = Ort::Value::CreateTensor<float>(
            _cpu_mem, _self_v_buf.data(), _self_v_buf.size(),
            self_cache_shape, 4);

        auto offset_tensor = Ort::Value::CreateTensor<std::int64_t>(
            _cpu_mem, &offset_value, 1, offset_shape, 1);

        // Temporarily steal ownership of cross_k/cross_v into the input
        // array. They are moved back on scope exit (including exceptions)
        // so the caller's references remain valid for the next Forward().
        std::array<Ort::Value, 6> inputs = {
            std::move(tokens_tensor),
            std::move(self_k_tensor),
            std::move(self_v_tensor),
            std::move(cross_k),
            std::move(cross_v),
            std::move(offset_tensor),
        };

        struct CrossReturn
        {
            Ort::Value& caller_k;
            Ort::Value& caller_v;
            Ort::Value& slot_k;
            Ort::Value& slot_v;
            ~CrossReturn()
            {
                caller_k = std::move(slot_k);
                caller_v = std::move(slot_v);
            }
        } cross_return{ cross_k, cross_v, inputs[3], inputs[4] };

        Ort::RunOptions run_opts{nullptr};
        auto outputs = _session.Run(
            run_opts,
            kInputNames.data(),
            inputs.data(), inputs.size(),
            kOutputNames.data(),
            kOutputNames.size());

        // logits: [1, n_tokens, n_vocab]
        const auto logits_info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto logits_shape = logits_info.GetShape();
        if (logits_shape.size() != 3 ||
            logits_shape[0] != 1 ||
            logits_shape[1] != n_tokens ||
            logits_shape[2] != _meta.n_vocab)
        {
            throw std::runtime_error("GreedyDecoder: unexpected logits shape");
        }

        const float* logits_data = outputs[0].GetTensorData<float>();
        const size_t logits_total =
            static_cast<size_t>(n_tokens) * static_cast<size_t>(_meta.n_vocab);
        std::vector<float> logits_out(logits_data, logits_data + logits_total);

        // Validate that the decoder's KV outputs match the shape and element
        // count of our persistent input buffers before blindly memcpy'ing.
        // If the decoder graph does not match the encoder metadata (e.g. a
        // corrupted or mismatched decoder.onnx), this check prevents reading
        // past ORT-owned memory and corrupting the host process.
        const auto check_kv = [this](const Ort::Value& v, const char* which)
        {
            const auto info  = v.GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            const size_t expected_elems =
                _self_k_buf.size(); // == _self_v_buf.size()
            if (shape.size() != 4 ||
                shape[0] != _meta.n_text_layer ||
                shape[1] != 1 ||
                shape[2] != _meta.n_text_ctx ||
                shape[3] != _meta.n_text_state ||
                info.GetElementCount() != expected_elems)
            {
                throw std::runtime_error(
                    std::string("GreedyDecoder: unexpected ") + which +
                    " shape; encoder/decoder metadata mismatch");
            }
        };
        check_kv(outputs[1], "out_n_layer_self_k_cache");
        check_kv(outputs[2], "out_n_layer_self_v_cache");

        // Copy updated self-KV cache back into persistent buffers.
        const float* out_k = outputs[1].GetTensorData<float>();
        const float* out_v = outputs[2].GetTensorData<float>();
        std::memcpy(_self_k_buf.data(), out_k,
                    _self_k_buf.size() * sizeof(float));
        std::memcpy(_self_v_buf.data(), out_v,
                    _self_v_buf.size() * sizeof(float));

        return logits_out;
    }

} // namespace asw::whisper
