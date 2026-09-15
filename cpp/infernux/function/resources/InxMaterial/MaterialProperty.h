#pragma once

#include <array>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <variant>

namespace infernux
{

enum class MaterialPropertyType
{
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Mat4,
    Texture2D,
    Color
};

using MaterialPropertyValue = std::variant<float, glm::vec2, glm::vec3, glm::vec4, int, glm::mat4, std::string>;

struct MaterialProperty
{
    std::string name;
    MaterialPropertyType type;
    MaterialPropertyValue value;
    bool hdr = false;
    std::optional<std::array<double, 2>> range;
};

} // namespace infernux
