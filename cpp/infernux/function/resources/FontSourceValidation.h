#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace infernux
{
// ImGui's stb backend assumes complete SFNT tables and asserts on malformed
// input. Check the source container before giving it ownership of project data.
inline void ValidateFontSource(std::string_view bytes)
{
    const auto fail = [](const char *reason) { throw std::runtime_error(std::string("Invalid font source: ") + reason); };
    const auto u16 = [&](size_t offset) -> uint16_t {
        if (offset > bytes.size() || bytes.size() - offset < 2)
            fail("truncated table");
        return (static_cast<uint8_t>(bytes[offset]) << 8) | static_cast<uint8_t>(bytes[offset + 1]);
    };
    const auto u32 = [&](size_t offset) -> uint32_t {
        return (static_cast<uint32_t>(u16(offset)) << 16) | u16(offset + 2);
    };
    if (bytes.size() <= 100 || bytes.size() > static_cast<size_t>((std::numeric_limits<int>::max)()) ||
        (u32(0) != 0x00010000 && bytes.substr(0, 4) != "OTTO" &&
                               bytes.substr(0, 4) != "true"))
        fail("expected a complete TrueType/OpenType SFNT file");
    const size_t count = u16(4);
    if (!count || count > (bytes.size() - 12) / 16)
        fail("truncated SFNT directory");
    struct Table { size_t offset, size; };
    std::unordered_map<std::string_view, Table> tables;
    for (size_t index = 0; index < count; ++index) {
        const size_t record = 12 + index * 16;
        const size_t offset = u32(record + 8), size = u32(record + 12);
        if (offset > bytes.size() || size > bytes.size() - offset ||
            !tables.emplace(bytes.substr(record, 4), Table{offset, size}).second)
            fail("invalid SFNT table range or duplicate table");
    }
    const auto require = [&](std::string_view name, size_t minimum) -> Table {
        const auto table = tables.find(name);
        if (table == tables.end() || table->second.size < minimum)
            fail("missing or truncated required SFNT table");
        return table->second;
    };
    const auto head = require("head", 54), hhea = require("hhea", 36), maxp = require("maxp", 6);
    if (u32(head.offset + 12) != 0x5f0f3cf5 || !u16(head.offset + 18))
        fail("invalid font header");
    const size_t glyphs = u16(maxp.offset + 4), metrics = u16(hhea.offset + 34);
    if (!glyphs || !metrics || metrics > glyphs)
        fail("invalid glyph metrics");
    require("hmtx", metrics * 4 + (glyphs - metrics) * 2);
    const auto cmap = require("cmap", 4);
    const size_t encodings = u16(cmap.offset + 2);
    if (!encodings || encodings > (cmap.size - 4) / 8)
        fail("invalid character map directory");
    bool unicodeMap = false;
    for (size_t index = 0; index < encodings; ++index) {
        const size_t record = cmap.offset + 4 + index * 8;
        const size_t relative = u32(record + 4);
        if (relative > cmap.size || cmap.size - relative < 6)
            fail("truncated character map");
        const size_t offset = cmap.offset + relative;
        const auto format = u16(offset);
        const size_t length = format == 14 ? u32(offset + 2) : format >= 8 ? u32(offset + 4) : u16(offset + 2);
        if (length > cmap.size - relative || length < (format == 14 ? 10u : format >= 8 ? 12u : 6u))
            fail("invalid character map range");
        if ((format == 0 && length < 262) ||
            (format == 4 && (length < 16 || !u16(offset + 6) || u16(offset + 6) % 2 ||
                             u16(offset + 6) / 2 > (length - 16) / 8)) ||
            (format == 6 && (length < 10 || u16(offset + 8) > (length - 10) / 2)) ||
            ((format == 12 || format == 13) && (length < 16 || u32(offset + 12) > (length - 16) / 12)))
            fail("truncated character map entries");
        const bool unicode = u16(record) == 0 || (u16(record) == 3 && (u16(record + 2) == 1 || u16(record + 2) == 10));
        unicodeMap |= unicode && (format == 0 || format == 4 || format == 6 || format == 12 || format == 13);
    }
    if (!unicodeMap)
        fail("no supported Unicode character map");
    if (tables.count("glyf")) {
        const auto glyf = tables.at("glyf");
        const bool longOffsets = u16(head.offset + 50) == 1;
        if (u16(head.offset + 50) > 1)
            fail("invalid glyph offset format");
        const auto loca = require("loca", (glyphs + 1) * (longOffsets ? 4 : 2));
        size_t previous = 0;
        for (size_t index = 0; index <= glyphs; ++index) {
            const size_t offset = longOffsets ? u32(loca.offset + index * 4) : u16(loca.offset + index * 2) * 2;
            if (offset < previous || offset > glyf.size || (offset != previous && offset - previous < 10))
                fail("invalid glyph table range");
            previous = offset;
        }
    } else {
        require("CFF ", 4);
    }
}
} // namespace infernux
