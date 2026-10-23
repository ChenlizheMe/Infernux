#include "TextMarkup.h"

#include <cmath>

namespace infernux::text
{
namespace
{
std::string_view Unquote(std::string_view value)
{
    if (value.size() >= 2 && (value.front() == '\'' || value.front() == '"') && value.back() == value.front())
        return value.substr(1, value.size() - 2);
    return value;
}

bool ParseColor(std::string_view value, std::array<float, 4> &color)
{
    if ((value.size() != 7 && value.size() != 9) || value.front() != '#')
        return false;
    std::array<float, 4> parsed{1, 1, 1, 1};
    for (size_t index = 1; index < value.size(); index += 2) {
        int byte = 0;
        for (size_t digit = index; digit < index + 2; ++digit) {
            const char c = value[digit];
            const int hex = c >= '0' && c <= '9'   ? c - '0'
                            : c >= 'a' && c <= 'f' ? c - 'a' + 10
                            : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                   : -1;
            if (hex < 0)
                return false;
            byte = byte * 16 + hex;
        }
        parsed[(index - 1) / 2] = static_cast<float>(byte) / 255.0f;
    }
    color = parsed;
    return true;
}

bool ParseSize(std::string_view value, float baseSize, float &size)
{
    const bool percent = !value.empty() && value.back() == '%';
    if (percent)
        value.remove_suffix(1);
    float number = 0.0f, fraction = 0.0f;
    bool digits = false, decimal = false;
    for (const char c : value) {
        if (c == '.' && !decimal) {
            decimal = true;
            fraction = 1.0f;
        } else if (c >= '0' && c <= '9') {
            digits = true;
            if (decimal) {
                fraction *= 0.1f;
                number += static_cast<float>(c - '0') * fraction;
            } else {
                number = number * 10.0f + static_cast<float>(c - '0');
            }
        } else {
            return false;
        }
    }
    const float pixels = percent ? number * baseSize / 100.0f : number;
    if (!digits || !std::isfinite(pixels) || pixels <= 0.0f || pixels > 4096.0f)
        return false;
    size = pixels;
    return true;
}
} // namespace

MarkupText ParseTextMarkup(std::string_view source, float fontSize)
{
    MarkupText result;
    result.text.reserve(source.size());
    result.sources.reserve(source.size());
    StyleRun style{0, 0, fontSize};
    struct OpenTag
    {
        std::string_view name;
        StyleRun previous;
    };
    std::vector<OpenTag> tags;
    const auto emit = [&](size_t start, size_t end, bool newline = false) {
        if (start == end)
            return;
        const size_t offset = result.text.size();
        if (newline) {
            result.text += '\n';
            result.sources.push_back({start, end});
        } else {
            result.text.append(source.substr(start, end - start));
            for (size_t byte = start; byte < end; ++byte)
                result.sources.push_back({byte, byte + 1});
        }
        if (!result.styles.empty() && result.styles.back().color == style.color &&
            result.styles.back().fontSize == style.fontSize)
            result.styles.back().end = result.text.size();
        else
            result.styles.push_back({offset, result.text.size(), style.fontSize, style.color});
    };
    size_t cursor = 0;
    while (cursor < source.size()) {
        const size_t open = source.find('<', cursor);
        if (open == std::string_view::npos) {
            emit(cursor, source.size());
            break;
        }
        emit(cursor, open);
        const size_t close = source.find('>', open + 1);
        if (close == std::string_view::npos) {
            emit(open, source.size());
            break;
        }
        const auto token = source.substr(open + 1, close - open - 1);
        if (token == "noparse") {
            constexpr std::string_view endTag = "</noparse>";
            const size_t end = source.find(endTag, close + 1);
            if (end == std::string_view::npos) {
                emit(close + 1, source.size());
                break;
            }
            emit(close + 1, end);
            cursor = end + endTag.size();
            continue;
        }
        bool consumed = false;
        if (token == "br" || token == "br/") {
            emit(open, close + 1, true);
            consumed = true;
        } else if (!token.empty() && token.front() == '/' && !tags.empty() && token.substr(1) == tags.back().name) {
            style = tags.back().previous;
            tags.pop_back();
            consumed = true;
        } else {
            const size_t equals = token.find('=');
            if (equals != std::string_view::npos) {
                const auto name = token.substr(0, equals);
                const auto value = Unquote(token.substr(equals + 1));
                auto next = style;
                if ((name == "color" && ParseColor(value, next.color)) ||
                    (name == "size" && ParseSize(value, fontSize, next.fontSize))) {
                    tags.push_back({name, style});
                    style = next;
                    consumed = true;
                }
            }
        }
        if (!consumed)
            emit(open, close + 1);
        cursor = close + 1;
    }
    return result;
}
} // namespace infernux::text
