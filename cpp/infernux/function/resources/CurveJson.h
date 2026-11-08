#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace infernux::curvejson
{
inline nlohmann::json Write(float value)
{
    if (std::isfinite(value))
        return value;
    if (std::isnan(value)) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        char text[13];
        std::snprintf(text, sizeof(text), "nan:%08x", bits);
        return text;
    }
    return value > 0 ? "infinity" : "-infinity";
}

inline float Read(const nlohmann::json &value)
{
    if (value.is_number())
        return value.get<float>();
    const auto &text = value.get_ref<const std::string &>();
    if (text == "infinity")
        return std::numeric_limits<float>::infinity();
    if (text == "-infinity")
        return -std::numeric_limits<float>::infinity();
    if (text == "nan")
        return std::numeric_limits<float>::quiet_NaN();
    if (text.size() == 12 && text.compare(0, 4, "nan:") == 0) {
        uint32_t bits;
        const auto parsed = std::from_chars(text.data() + 4, text.data() + text.size(), bits, 16);
        if (parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
            (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
            float number;
            std::memcpy(&number, &bits, sizeof(number));
            return number;
        }
    }
    throw std::invalid_argument("Curve number requires a numeric value or a nonfinite token");
}
} // namespace infernux::curvejson
