#include "stdafx.h"
#include "BpeTokenizer.h"
#include "Base64.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <fstream>
#include <stdexcept>

namespace asw::whisper
{
    //=========================================================================
    // tokens.txt parser
    //=========================================================================

    BpeTokenizer::BpeTokenizer(const std::wstring& tokens_path)
    {
        std::ifstream file(tokens_path);
        if (!file)
            throw std::runtime_error("BpeTokenizer: cannot open tokens file");

        // Parallel seen-bitmap: detects duplicate ids AND, after the loop,
        // gaps in the id space (missing ids between 0 and max). We cannot
        // rely on `_id_to_bytes[id].empty()` for this because empty pieces
        // are legitimate (see the Sherpa "= <id>" placeholder entries).
        std::vector<bool> seen;

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            // Strip trailing '\r' (Windows line endings).
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                line.pop_back();

            // Split at the last space: "base64_piece <id>".
            const auto sp = line.find_last_of(' ');
            if (sp == std::string::npos)
                throw std::runtime_error("BpeTokenizer: malformed line (no space)");

            std::string_view b64 = std::string_view(line).substr(0, sp);
            std::string_view num = std::string_view(line).substr(sp + 1);

            // Full-token id parse: reject entries like "foo 123junk" where
            // from_chars would happily consume "123" and silently ignore
            // trailing garbage.
            std::int32_t id = -1;
            const char* const num_begin = num.data();
            const char* const num_end   = num.data() + num.size();
            auto [ptr, ec] = std::from_chars(num_begin, num_end, id);
            if (ec != std::errc{} || ptr != num_end || id < 0)
                throw std::runtime_error("BpeTokenizer: invalid id");

            // Upper-bound the id to prevent a malformed line like
            // "piece 999999999" from forcing `seen.resize(id+1)` and
            // `_id_to_bytes.resize(id+1)` into OOM territory before any
            // Session-level consistency check can run. The cap is ~20x
            // the largest known Whisper vocab (whisper-large-v3 has
            // n_vocab = 51866); legitimate token files fit easily below it.
            constexpr std::int32_t kMaxAllowedId = 1'000'000;
            if (id > kMaxAllowedId)
                throw std::runtime_error(
                    "BpeTokenizer: id " + std::to_string(id) +
                    " exceeds sanity cap of " + std::to_string(kMaxAllowedId));

            // Sherpa-ism: the multilingual tokens.txt export encodes some
            // placeholder/empty pieces as a bare "=" or "==" (only padding
            // chars, no data). Strict RFC 4648 rejects these, so we
            // special-case padding-only payloads to an empty piece here.
            //
            // The exception is deliberately limited to the two legal
            // padding lengths (1 and 2 '=' characters) that can appear
            // after a valid base64 quartet. Anything longer (e.g. "===",
            // "====") is not produced by any known export and is treated
            // as a hard error rather than silently yielding an empty piece.
            auto is_padding_only = [](std::string_view s) -> bool
            {
                if (s.size() != 1 && s.size() != 2) return false;
                for (char c : s) { if (c != '=') return false; }
                return true;
            };

            std::optional<std::string> decoded;
            if (b64.empty() || is_padding_only(b64))
                decoded = std::string{}; // empty piece
            else
                decoded = Base64Decode(b64);

            if (!decoded)
                throw std::runtime_error("BpeTokenizer: invalid base64 piece");

            // Duplicate id check. Silently overwriting a slot would let a
            // malformed tokens.txt produce corrupted transcriptions.
            if (static_cast<size_t>(id) < seen.size() && seen[static_cast<size_t>(id)])
                throw std::runtime_error(
                    "BpeTokenizer: duplicate id " + std::to_string(id));

            if (static_cast<size_t>(id) >= seen.size())
                seen.resize(static_cast<size_t>(id) + 1, false);
            seen[static_cast<size_t>(id)] = true;

            if (static_cast<size_t>(id) >= _id_to_bytes.size())
                _id_to_bytes.resize(static_cast<size_t>(id) + 1);

            _id_to_bytes[static_cast<size_t>(id)] = std::move(*decoded);
        }

        if (_id_to_bytes.empty())
            throw std::runtime_error("BpeTokenizer: empty tokens file");

        // Contiguous-coverage check. A well-formed Sherpa tokens.txt has an
        // entry for every id in [0 .. max_id]; any gap means the file is
        // truncated or a line was dropped, which would let the decoder emit
        // empty pieces at runtime instead of failing loudly now.
        for (size_t i = 0; i < seen.size(); ++i)
        {
            if (!seen[i])
                throw std::runtime_error(
                    "BpeTokenizer: missing id " + std::to_string(i) +
                    " (truncated tokens.txt?)");
        }
    }

    //=========================================================================
    // Lookups
    //=========================================================================

    std::string_view BpeTokenizer::Piece(std::int32_t id) const noexcept
    {
        if (id < 0 || static_cast<size_t>(id) >= _id_to_bytes.size())
            return {};
        return _id_to_bytes[static_cast<size_t>(id)];
    }

    std::vector<std::string> BpeTokenizer::IdsToPieces(
        std::span<const std::int32_t> ids) const
    {
        std::vector<std::string> result;
        result.reserve(ids.size());
        for (std::int32_t id : ids)
        {
            auto piece = Piece(id);
            result.emplace_back(ToValidUtf8(piece));
        }
        return result;
    }

    std::string BpeTokenizer::IdsToText(std::span<const std::int32_t> ids) const
    {
        // Concatenate raw bytes first so that multi-byte UTF-8 code points
        // split across BPE boundaries reassemble correctly.
        std::string buf;
        size_t total = 0;
        for (std::int32_t id : ids)
            total += Piece(id).size();
        buf.reserve(total);

        for (std::int32_t id : ids)
            buf.append(Piece(id));

        return ToValidUtf8(buf);
    }

    //=========================================================================
    // UTF-8 validation / repair
    //=========================================================================

    std::string ToValidUtf8(std::string_view s)
    {
        std::string out;
        out.reserve(s.size());

        const char kReplacement[] = "\xEF\xBF\xBD"; // U+FFFD

        size_t i = 0;
        while (i < s.size())
        {
            const auto b0 = static_cast<std::uint8_t>(s[i]);

            if (b0 < 0x80)
            {
                out.push_back(static_cast<char>(b0));
                ++i;
                continue;
            }

            std::int32_t need = 0;
            std::uint32_t cp  = 0;
            if ((b0 & 0xE0) == 0xC0) { need = 1; cp = b0 & 0x1F; }
            else if ((b0 & 0xF0) == 0xE0) { need = 2; cp = b0 & 0x0F; }
            else if ((b0 & 0xF8) == 0xF0) { need = 3; cp = b0 & 0x07; }
            else
            {
                out.append(kReplacement, 3);
                ++i;
                continue;
            }

            if (i + 1 + need > s.size())
            {
                out.append(kReplacement, 3);
                break;
            }

            bool ok = true;
            for (std::int32_t k = 0; k < need; ++k)
            {
                const auto bk = static_cast<std::uint8_t>(s[i + 1 + k]);
                if ((bk & 0xC0) != 0x80) { ok = false; break; }
                cp = (cp << 6) | (bk & 0x3F);
            }

            if (!ok)
            {
                out.append(kReplacement, 3);
                ++i;
                continue;
            }

            // Reject overlong encodings and surrogate range.
            const bool overlong =
                (need == 1 && cp < 0x80) ||
                (need == 2 && cp < 0x800) ||
                (need == 3 && cp < 0x10000);
            const bool surrogate = (cp >= 0xD800 && cp <= 0xDFFF);
            const bool too_big   = (cp > 0x10FFFF);

            if (overlong || surrogate || too_big)
            {
                // Structurally-valid bytes decoded to a forbidden code
                // point. Emit one U+FFFD for the whole maximal subpart
                // (per Unicode Best Practice for using U+FFFD), consuming
                // all (1 + need) bytes.
                out.append(kReplacement, 3);
                i += 1 + need;
                continue;
            }

            out.append(s.data() + i, 1 + need);
            i += 1 + need;
        }

        return out;
    }

} // namespace asw::whisper
