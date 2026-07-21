#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace asw::whisper
{
    //=========================================================================
    /// Loader + detokenizer for the Sherpa-exported Whisper tokens.txt
    /// format.
    ///
    /// Each line:    `<base64(raw_bytes)> <vocab_id>`
    /// After base64-decoding, each entry is the raw byte sequence of one
    /// GPT-2 byte-level BPE piece (possibly a partial UTF-8 sequence).
    ///
    /// Detokenization simply concatenates the raw bytes of generated tokens
    /// in order, then interprets the result as UTF-8 (invalid sequences are
    /// replaced with U+FFFD).
    //=========================================================================
    class BpeTokenizer
    {
    public:
        explicit BpeTokenizer(const std::wstring& tokens_path);

        /// Number of entries in tokens.txt (== Whisper's n_vocab).
        [[nodiscard]] std::int32_t Size() const noexcept
        {
            return static_cast<std::int32_t>(_id_to_bytes.size());
        }

        /// Raw byte sequence for a given token id. Returns empty view for
        /// ids that never appeared in the file.
        [[nodiscard]] std::string_view Piece(std::int32_t id) const noexcept;

        /// Returns UTF-8 piece representations for each id, in order. Pieces
        /// are the raw bytes decoded to valid UTF-8 (with U+FFFD for invalid
        /// sequences). This is what media_scan consumes for its per-token
        /// debug logging.
        [[nodiscard]] std::vector<std::string> IdsToPieces(
            std::span<const std::int32_t> ids) const;

        /// Concatenate raw bytes and convert the whole buffer to valid
        /// UTF-8 in one shot (preferred over IdsToPieces+join because BPE
        /// pieces often split UTF-8 code points across boundaries).
        [[nodiscard]] std::string IdsToText(std::span<const std::int32_t> ids) const;

    private:
        std::vector<std::string> _id_to_bytes; ///< raw decoded bytes per id
    };

    /// Replace invalid UTF-8 sequences in `s` with U+FFFD ("?" replacement).
    /// Valid UTF-8 bytes pass through untouched.
    [[nodiscard]] std::string ToValidUtf8(std::string_view s);

} // namespace asw::whisper
