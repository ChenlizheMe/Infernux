#pragma once

#include <core/log/InxLog.h>
#include <core/text/TextMarkup.h>
#include <core/text/TextSegments.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <platform/filesystem/InxPath.h>

#ifdef DrawText
#undef DrawText
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux::textlayout
{

struct TextLayoutParams
{
    std::string text;
    std::string fontPath;
    // Positive sizes are authored em units; zero inherits the current ImGui
    // raster size and line height without applying em conversion a second time.
    float fontSize = 0.0f;
    float wrapWidth = 0.0f;
    float lineHeight = 1.0f;
    float letterSpacing = 0.0f;
    std::vector<std::string> fallbackFontPaths;
    bool richText = false;
};

struct TextLine
{
    size_t startOffset = 0;
    size_t endOffset = 0;
    float width = 0.0f;
    size_t glyphStart = 0;
    size_t glyphEnd = 0;
    float y = 0.0f;
    float height = 0.0f;
};

struct TextGlyph
{
    size_t startOffset = 0;
    size_t endOffset = 0;
    ImFont *font = nullptr;
    float fontSize = 0.0f;
    float advance = 0.0f;
    float inkMinY = 0.0f;
    float inkMaxY = 0.0f;
    bool visible = false;
    uint32_t codepoint = 0;
    size_t character = 0;
    size_t line = text::NoWord;
    float originX = 0.0f;
    float originY = 0.0f;
    std::array<float, 4> bounds{};
    float logicalFontSize = 0.0f;
    float baseline = 0.0f;
    std::array<float, 4> color{1, 1, 1, 1};
};

struct TextCharacter
{
    size_t startOffset;
    size_t endOffset;
    size_t glyphStart = 0;
    size_t glyphEnd = 0;
    size_t line = 0;
    size_t word = text::NoWord;
    bool whitespace = false;
    std::array<float, 4> bounds{};
};

struct TextLayoutResult
{
    std::string text;
    std::string displayText;
    std::vector<text::StyleRun> styles;
    ImFont *font = nullptr;
    std::vector<ImFont *> fallbackFonts;
    std::vector<float> fallbackFontSizes;
    float logicalFontSize = 0.0f;
    float fontSize = 0.0f;
    float lineAdvance = 0.0f;
    float baseLineHeight = 0.0f;
    float totalWidth = 0.0f;
    float totalHeight = 0.0f;
    float inkMinY = 0.0f;
    float inkMaxY = 0.0f;
    std::vector<TextLine> lines;
    std::vector<TextGlyph> glyphs;
    std::vector<TextCharacter> characters;
    text::Segments segments;
};

inline uint16_t ReadSfntU16(const unsigned char *data)
{
    return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8u) | data[1]);
}

inline int16_t ReadSfntS16(const unsigned char *data)
{
    return static_cast<int16_t>(ReadSfntU16(data));
}

inline uint32_t ReadSfntU32(const unsigned char *data)
{
    return (static_cast<uint32_t>(data[0]) << 24u) | (static_cast<uint32_t>(data[1]) << 16u) |
           (static_cast<uint32_t>(data[2]) << 8u) | static_cast<uint32_t>(data[3]);
}

inline float ReadSfntEmRasterScale(const void *fontData, int fontDataSize)
{
    constexpr uint32_t kTtcTag = 0x74746366u;
    constexpr uint32_t kHeadTag = 0x68656164u;
    constexpr uint32_t kHheaTag = 0x68686561u;
    const auto *bytes = static_cast<const unsigned char *>(fontData);
    const size_t size = fontDataSize > 0 ? static_cast<size_t>(fontDataSize) : 0u;
    if (bytes == nullptr || size < 12u)
        return 1.0f;

    size_t directoryOffset = 0u;
    if (ReadSfntU32(bytes) == kTtcTag) {
        if (size < 16u)
            return 1.0f;
        directoryOffset = static_cast<size_t>(ReadSfntU32(bytes + 12u));
    }
    if (directoryOffset > size || size - directoryOffset < 12u)
        return 1.0f;

    const uint16_t tableCount = ReadSfntU16(bytes + directoryOffset + 4u);
    const size_t recordsOffset = directoryOffset + 12u;
    if (tableCount > (size - recordsOffset) / 16u)
        return 1.0f;

    size_t headOffset = 0u;
    size_t hheaOffset = 0u;
    for (uint16_t index = 0; index < tableCount; ++index) {
        const unsigned char *record = bytes + recordsOffset + static_cast<size_t>(index) * 16u;
        const uint32_t tag = ReadSfntU32(record);
        const size_t offset = static_cast<size_t>(ReadSfntU32(record + 8u));
        const size_t length = static_cast<size_t>(ReadSfntU32(record + 12u));
        if (offset > size || length > size - offset)
            continue;
        if (tag == kHeadTag && length >= 20u)
            headOffset = offset;
        else if (tag == kHheaTag && length >= 8u)
            hheaOffset = offset;
    }
    if (headOffset == 0u || hheaOffset == 0u)
        return 1.0f;

    const uint16_t unitsPerEm = ReadSfntU16(bytes + headOffset + 18u);
    const int ascent = static_cast<int>(ReadSfntS16(bytes + hheaOffset + 4u));
    const int descent = static_cast<int>(ReadSfntS16(bytes + hheaOffset + 6u));
    const int metricHeight = ascent - descent;
    if (unitsPerEm == 0u || metricHeight <= 0)
        return 1.0f;
    const float scale = static_cast<float>(metricHeight) / static_cast<float>(unitsPerEm);
    return scale >= 0.5f && scale <= 4.0f ? scale : 1.0f;
}

inline float ResolveEmRasterScale(ImFont *font)
{
    if (font == nullptr || font->Sources.empty() || font->Sources[0] == nullptr)
        return 1.0f;
    const ImFontConfig *source = font->Sources[0];
    return ReadSfntEmRasterScale(source->FontData, source->FontDataSize);
}

// Font cache and missing-font set.  Accessed only from the main/render thread
// (ImGui is single-threaded by design).  Wrapped in accessors to avoid static
// initialization order issues across translation units.
//
// Debug-mode assertion records the first caller's thread ID and asserts on
// subsequent calls from a different thread, catching accidental cross-thread use.

namespace detail
{
inline void AssertMainThread()
{
#ifndef NDEBUG
    static const std::thread::id s_ownerThread = std::this_thread::get_id();
    assert(std::this_thread::get_id() == s_ownerThread && "InxTextLayout font cache accessed from non-owner thread");
#endif
}
} // namespace detail

inline std::unordered_map<std::string, ImFont *> &GetFontCache()
{
    detail::AssertMainThread();
    static std::unordered_map<std::string, ImFont *> cache;
    return cache;
}

inline std::unordered_set<std::string> &GetMissingFonts()
{
    detail::AssertMainThread();
    static std::unordered_set<std::string> missing;
    return missing;
}

inline uint64_t &FontCacheGeneration()
{
    static uint64_t generation = 1;
    return generation;
}

inline std::unordered_set<ImFont *> &GetRetiredFonts()
{
    detail::AssertMainThread();
    static std::unordered_set<ImFont *> fonts;
    return fonts;
}

// Remove path aliases immediately, but keep atlas-owned font storage alive
// until the GUI's next frame boundary has drained its submitted GPU work.
inline bool InvalidateFontPath(const std::string &path)
{
    const std::string requestKey = LexicalFilesystemPathKey(path);
    const std::string resolvedKey = FoldFilesystemPathCase(ResolveFilesystemPath(path));
    auto &cache = GetFontCache();
    auto &missing = GetMissingFonts();
    auto &retired = GetRetiredFonts();
    bool changed = false;
    for (const auto &key : {requestKey, resolvedKey}) {
        if (key.empty())
            continue;
        changed = missing.erase(key) != 0 || changed;
        if (const auto found = cache.find(key); found != cache.end()) {
            retired.insert(found->second);
            changed = true;
        }
    }
    if (!changed)
        return false;
    for (auto entry = cache.begin(); entry != cache.end();) {
        if (retired.count(entry->second))
            entry = cache.erase(entry);
        else
            ++entry;
    }
    ++FontCacheGeneration();
    return true;
}

inline void CollectRetiredFonts()
{
    // Caller owns the main-thread, between-frame and GPU-completion boundary.
    for (ImFont *font : GetRetiredFonts())
        font->OwnerAtlas->RemoveFont(font);
    GetRetiredFonts().clear();
}

// The editor/player GUI installs one authoritative engine font at startup.
// UIText without an authored Font asset must use that same bundled face,
// rather than inheriting whatever ambient ImGui font happens to be active.
// The pointer is valid only while the current ImGui atlas is alive; the GUI
// reload path clears it before rebuilding the atlas.
inline ImFont *&DefaultFont()
{
    static ImFont *font = nullptr;
    return font;
}

inline void SetDefaultFont(ImFont *font)
{
    DefaultFont() = font;
}

inline void ClearFontCache()
{
    GetFontCache().clear();
    GetMissingFonts().clear();
    // Atlas Clear/Destroy owns these objects when the entire GUI is reset.
    GetRetiredFonts().clear();
    ++FontCacheGeneration();
}

inline float ResolveFontSize(float fontSize)
{
    return fontSize > 0.0f ? fontSize : ImGui::GetFontSize();
}

inline std::string NormalizeFontPath(const std::string &fontPath)
{
    return ResolveFilesystemPath(fontPath);
}

inline ImFont *ResolveFont(const std::string &fontPath)
{
    if (fontPath.empty())
        return DefaultFont() != nullptr ? DefaultFont() : ImGui::GetFont();

    // Font loading owns filesystem resolution. Replaying text must not walk
    // the filesystem again; lexical absolute keys still distinguish relative
    // paths when the caller changes its working directory.
    const std::string requestKey = LexicalFilesystemPathKey(fontPath);
    auto &fontCache = GetFontCache();
    auto &missingFonts = GetMissingFonts();
    if (auto it = fontCache.find(requestKey); it != fontCache.end())
        return it->second;
    if (missingFonts.find(requestKey) != missingFonts.end())
        return nullptr;

    const std::string normalizedPath = NormalizeFontPath(fontPath);
    if (normalizedPath.empty()) {
        if (missingFonts.insert(requestKey).second)
            INXLOG_ERROR("UIText explicit font path cannot be resolved: '", fontPath, "'");
        return nullptr;
    }
    const std::string pathKey = FoldFilesystemPathCase(normalizedPath);

    if (auto it = fontCache.find(pathKey); it != fontCache.end()) {
        ImFont *font = it->second;
        fontCache[requestKey] = font;
        return font;
    }
    if (missingFonts.find(pathKey) != missingFonts.end()) {
        missingFonts.insert(requestKey);
        return nullptr;
    }

    std::error_code ec;
    if (!std::filesystem::exists(ToFsPath(normalizedPath), ec) || ec) {
        missingFonts.insert(pathKey);
        missingFonts.insert(requestKey);
        INXLOG_ERROR("UIText explicit font asset does not exist: '", normalizedPath, "'");
        return nullptr;
    }

    ImFontConfig config{};
    // World-space UIText is frequently magnified by the scene/game camera.
    // Keep a stable two-tap horizontal/vertical oversample so the dynamic
    // ImGui atlas retains edge coverage instead of magnifying a one-pixel
    // glyph raster.
    config.OversampleH = 2;
    config.OversampleV = 2;
    ImFont *font = ImGui::GetIO().Fonts->AddFontFromFileTTF(normalizedPath.c_str(), 18.0f, &config);
    if (font == nullptr) {
        missingFonts.insert(pathKey);
        missingFonts.insert(requestKey);
        INXLOG_ERROR("UIText explicit font asset could not be loaded: '", normalizedPath, "'");
        return nullptr;
    }

    fontCache[pathKey] = font;
    fontCache[requestKey] = font;
    return font;
}

inline ImFont *ResolveGlyphFont(const TextLayoutResult &layout, ImWchar codepoint)
{
    if (layout.font == nullptr)
        return nullptr;
    if (layout.font->IsGlyphInFont(codepoint))
        return layout.font;
    for (ImFont *font : layout.fallbackFonts) {
        if (font != nullptr && font->IsGlyphInFont(codepoint))
            return font;
    }
    // The chain itself is explicit. If no face contains the codepoint, use the
    // primary face's replacement glyph instead of consulting an ambient font.
    return layout.font;
}

inline float ResolveGlyphRasterSize(const TextLayoutResult &layout, ImFont *font)
{
    if (font == layout.font)
        return layout.fontSize;
    const auto it = std::find(layout.fallbackFonts.begin(), layout.fallbackFonts.end(), font);
    assert(it != layout.fallbackFonts.end());
    return layout.fallbackFontSizes[static_cast<size_t>(it - layout.fallbackFonts.begin())];
}

inline bool IsSpaceLike(uint32_t c)
{
    return c == ' ' || c == '\t';
}

inline bool IsBreakAfterChar(uint32_t c)
{
    switch (c) {
    case '-':
    case '/':
    case '\\':
    case ',':
    case '.':
    case ';':
    case ':':
    case '!':
    case '?':
    case 0x3001:
    case 0x3002:
    case 0xFF0C:
    case 0xFF01:
    case 0xFF1F:
    case 0xFF1A:
    case 0xFF1B:
        return true;
    default:
        return false;
    }
}

inline bool PopBackUtf8Codepoint(std::string &text)
{
    if (text.empty())
        return false;

    size_t codepointStart = text.size() - 1;
    while (codepointStart > 0 && (static_cast<unsigned char>(text[codepointStart]) & 0xC0u) == 0x80u)
        --codepointStart;
    text.erase(codepointStart);
    return true;
}

inline bool EraseFirstUtf8Codepoint(std::string &text)
{
    if (text.empty())
        return false;

    unsigned int codepoint = 0;
    const int consumed = ImTextCharFromUtf8(&codepoint, text.data(), text.data() + text.size());
    text.erase(0, consumed > 0 ? static_cast<size_t>(consumed) : size_t{1});
    return true;
}

inline float MeasureSegmentWidth(ImFont *font, float fontSize, const char *start, const char *end, float letterSpacing)
{
    if (font == nullptr || start == nullptr || end == nullptr || start >= end)
        return 0.0f;

    float width = 0.0f;
    int glyphCount = 0;
    const char *cursor = start;
    while (cursor < end) {
        const char *charStart = cursor;
        unsigned int codepoint = 0;
        int consumed = ImTextCharFromUtf8(&codepoint, cursor, end);
        if (consumed <= 0)
            break;
        cursor += consumed;

        if (codepoint == '\r' || codepoint == '\n')
            continue;

        const float glyphWidth = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, charStart, cursor).x;
        if (glyphCount > 0)
            width += letterSpacing;
        width += glyphWidth;
        ++glyphCount;
    }

    return width;
}

inline size_t GlyphIndexAt(const TextLayoutResult &layout, size_t offset)
{
    return static_cast<size_t>(
        std::lower_bound(layout.glyphs.begin(), layout.glyphs.end(), offset,
                         [](const TextGlyph &glyph, size_t value) { return glyph.startOffset < value; }) -
        layout.glyphs.begin());
}

inline void PushLine(TextLayoutResult &result, const char *textBegin, const char *start, const char *end, float width)
{
    const size_t first = static_cast<size_t>(start - textBegin);
    const size_t last = static_cast<size_t>(end - textBegin);
    result.lines.push_back({first, last, width, GlyphIndexAt(result, first), GlyphIndexAt(result, last)});
    result.totalWidth = std::max(result.totalWidth, width);
}

TextLayoutResult LayoutText(const TextLayoutParams &params);
std::shared_ptr<TextLayoutResult> CreateTextLayout(const TextLayoutParams &params);

void RenderLine(ImDrawList *drawList, const TextLayoutResult &layout, const TextLine &line, float x, float y,
                ImU32 color, float letterSpacing, const ImVec4 *clipRect = nullptr);

void RenderTextBox(ImDrawList *drawList, float minX, float minY, float maxX, float maxY, const TextLayoutResult &layout,
                   ImU32 color, float alignX, float alignY, float letterSpacing, const ImVec4 *clipRect = nullptr);

} // namespace infernux::textlayout
