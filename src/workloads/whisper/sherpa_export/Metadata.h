#pragma once

#include <asw/framework/whisper/metadata.h>

#include <onnxruntime_cxx_api.h>

namespace asw::whisper
{
    //=========================================================================
    /// Parse the encoder ONNX's `metadata_props` into a typed Metadata
    /// struct. The Sherpa Whisper export populates every field below on
    /// every model; missing or malformed fields cause a std::runtime_error.
    //=========================================================================
    [[nodiscard]] Metadata ParseMetadata(const Ort::Session& encoder_session);

} // namespace asw::whisper
