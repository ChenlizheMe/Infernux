#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace infernux::text
{
struct SourceRange
{
    size_t start, end;
};

struct StyleRun
{
    size_t start, end;
    float fontSize;
    std::array<float, 4> color{1, 1, 1, 1};
};

struct MarkupText
{
    std::string text;
    std::vector<SourceRange> sources;
    std::vector<StyleRun> styles;
};

// Color (#RRGGBB[AA]), size (pixels or percent), br and noparse literal regions.
// Unsupported tags remain literal; source ranges address the original UTF-8 string.
MarkupText ParseTextMarkup(std::string_view source, float fontSize);
} // namespace infernux::text
