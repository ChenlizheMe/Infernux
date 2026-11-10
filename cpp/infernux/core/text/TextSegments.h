#pragma once

#include "core/FoundationApi.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace infernux::text
{

inline constexpr size_t NoWord = static_cast<size_t>(-1);

struct Codepoint
{
    uint32_t value;
    size_t start;
    size_t end;
    size_t character;
};

struct Character
{
    size_t start;
    size_t end;
    size_t codepointStart;
    size_t codepointEnd;
    size_t word = NoWord;
    bool whitespace = false;
    bool newline = false;
};

struct Word
{
    size_t characterStart;
    size_t characterEnd;
};

struct Segments
{
    std::vector<Codepoint> codepoints;
    std::vector<Character> characters;
    std::vector<Word> words;
};

// Extended graphemes follow Unicode UAX #29. Words are non-whitespace runs;
// dictionary-based language segmentation is a separate concern.
INFERNUX_FOUNDATION_API Segments SegmentText(std::string_view text);

// Canonical composition for rendering; callers retain original source ranges.
INFERNUX_FOUNDATION_API std::string NormalizeTextNFC(std::string_view text);

} // namespace infernux::text
