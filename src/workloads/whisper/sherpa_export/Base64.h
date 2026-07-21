#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace asw::whisper
{
    /// Decode a standard (RFC 4648) base64 string. Padding '=' is expected.
    /// Whitespace is silently skipped. Returns std::nullopt on any invalid
    /// character or bad padding.
    ///
    /// Produces `std::string` rather than `std::vector<uint8_t>` so the
    /// result can be concatenated directly into UTF-8 output buffers.
    [[nodiscard]] std::optional<std::string> Base64Decode(std::string_view in);

} // namespace asw::whisper
