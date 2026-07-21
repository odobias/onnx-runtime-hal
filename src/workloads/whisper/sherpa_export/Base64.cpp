#include "stdafx.h"
#include "Base64.h"

namespace asw::whisper
{
    namespace
    {
        // Inverse alphabet. 0xFF marks invalid; 0xFE marks padding '='.
        [[nodiscard]] constexpr std::array<std::uint8_t, 256> MakeLookup()
        {
            std::array<std::uint8_t, 256> t{};
            for (auto& b : t) b = 0xFF;
            for (std::uint8_t i = 0; i < 26; ++i) t['A' + i] = i;
            for (std::uint8_t i = 0; i < 26; ++i) t['a' + i] = 26 + i;
            for (std::uint8_t i = 0; i < 10; ++i) t['0' + i] = 52 + i;
            t['+'] = 62;
            t['/'] = 63;
            t['='] = 0xFE;
            return t;
        }

        inline constexpr auto kLookup = MakeLookup();
    } // namespace

    std::optional<std::string> Base64Decode(std::string_view in)
    {
        std::string out;
        out.reserve((in.size() * 3) / 4);

        std::uint32_t buf     = 0;
        std::int32_t  bits    = 0;
        std::int32_t  pad     = 0;
        std::int32_t  data_chars = 0;   // non-padding, non-whitespace symbols seen
        bool          saw_pad = false;

        for (char c : in)
        {
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
                continue;

            const std::uint8_t v = kLookup[static_cast<std::uint8_t>(c)];
            if (v == 0xFF)
                return std::nullopt;

            if (v == 0xFE) // padding
            {
                saw_pad = true;
                ++pad;
                if (pad > 2)
                    return std::nullopt;
                continue;
            }

            if (saw_pad)
                return std::nullopt; // data after padding

            buf = (buf << 6) | v;
            bits += 6;
            ++data_chars;
            if (bits >= 8)
            {
                bits -= 8;
                out.push_back(static_cast<char>((buf >> bits) & 0xFF));
            }
        }

        // Strict RFC 4648 tail validation. Each valid base64 stream ends
        // in exactly one of these states:
        //
        //   data_chars % 4 == 0, pad == 0, bits == 0  -- full quartets
        //   data_chars % 4 == 2, pad == 2, bits == 4  -- 1 trailing byte
        //   data_chars % 4 == 3, pad == 1, bits == 2  -- 2 trailing bytes
        //
        // Unpadded variants (no '=' characters) with the same data-char
        // residue and the same bits residue are also accepted. Anything
        // else -- including the "bits == 6" case (a single dangling
        // sextet representing 0.75 of a byte) -- is malformed.
        //
        // When `bits > 0`, the low `bits` of `buf` are the residual bits
        // that must be zero for a well-formed stream; a non-zero residue
        // indicates the encoder dropped bits that would not have fit in a
        // whole byte, which is not valid base64.
        switch (data_chars % 4)
        {
        case 0:
            if (pad != 0) return std::nullopt;
            // bits is always 0 here.
            break;
        case 1:
            // Impossible for valid base64: 6 bits is less than one byte.
            return std::nullopt;
        case 2:
            if (pad != 0 && pad != 2) return std::nullopt;
            // Residual 4 bits must be zero.
            if ((buf & 0x0F) != 0) return std::nullopt;
            break;
        case 3:
            if (pad != 0 && pad != 1) return std::nullopt;
            // Residual 2 bits must be zero.
            if ((buf & 0x03) != 0) return std::nullopt;
            break;
        }

        return out;
    }

} // namespace asw::whisper
