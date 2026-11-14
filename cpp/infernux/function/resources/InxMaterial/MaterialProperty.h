#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace infernux
{

/// Sampling state authored by one material texture binding.  Zero-valued
/// members inherit the referenced Texture asset's import settings; non-zero
/// members are exact binding overrides (for example a glTF sampler).  Keeping
/// this state on the binding avoids mutating a shared Texture asset merely
/// because one model uses it with a different sampler.
enum class MaterialSamplerFilter : uint8_t
{
    Inherit = 0,
    Nearest = 1,
    Linear = 2,
};

enum class MaterialSamplerAddress : uint8_t
{
    Inherit = 0,
    Repeat = 1,
    Clamp = 2,
    Mirror = 3,
};

struct MaterialTextureSampler
{
    MaterialSamplerFilter minFilter = MaterialSamplerFilter::Inherit;
    MaterialSamplerFilter magFilter = MaterialSamplerFilter::Inherit;
    MaterialSamplerFilter mipFilter = MaterialSamplerFilter::Inherit;
    MaterialSamplerAddress addressU = MaterialSamplerAddress::Inherit;
    MaterialSamplerAddress addressV = MaterialSamplerAddress::Inherit;
    MaterialSamplerAddress addressW = MaterialSamplerAddress::Inherit;

    [[nodiscard]] bool HasOverrides() const noexcept
    {
        return minFilter != MaterialSamplerFilter::Inherit || magFilter != MaterialSamplerFilter::Inherit ||
               mipFilter != MaterialSamplerFilter::Inherit || addressU != MaterialSamplerAddress::Inherit ||
               addressV != MaterialSamplerAddress::Inherit || addressW != MaterialSamplerAddress::Inherit;
    }

    [[nodiscard]] bool operator==(const MaterialTextureSampler &other) const noexcept
    {
        return minFilter == other.minFilter && magFilter == other.magFilter && mipFilter == other.mipFilter &&
               addressU == other.addressU && addressV == other.addressV && addressW == other.addressW;
    }

    [[nodiscard]] bool operator!=(const MaterialTextureSampler &other) const noexcept
    {
        return !(*this == other);
    }
};

enum class MaterialPropertyType
{
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Mat4,
    Texture2D,
    Color,
    FloatArray,
    Float4Array,
    Bool,
    UInt,
    Mat3,
};

inline constexpr std::array<std::string_view, 13> MaterialPropertyTypeNames = {
    "Float", "Float2",     "Float3",      "Float4", "Int",  "Mat4", "Texture2D",
    "Color", "FloatArray", "Float4Array", "Bool",   "UInt", "Mat3",
};

inline std::optional<MaterialPropertyType> MaterialPropertyTypeFromName(std::string_view name) noexcept
{
    for (size_t index = 0; index < MaterialPropertyTypeNames.size(); ++index)
        if (MaterialPropertyTypeNames[index] == name)
            return static_cast<MaterialPropertyType>(index);
    return std::nullopt;
}

inline bool MaterialPropertyTypesCompatible(MaterialPropertyType declared, MaterialPropertyType supplied)
{
    return declared == supplied ||
           ((declared == MaterialPropertyType::Color || declared == MaterialPropertyType::Float4) &&
            (supplied == MaterialPropertyType::Color || supplied == MaterialPropertyType::Float4));
}

using MaterialPropertyValue = std::variant<float, glm::vec2, glm::vec3, glm::vec4, int, glm::mat4, std::string,
                                           std::vector<float>, std::vector<glm::vec4>, bool, uint32_t, glm::mat3>;

// Color shares Float4 storage. All other logical kinds own one variant alternative.
inline constexpr std::array<size_t, 13> MaterialPropertyValueIndices = {0, 1, 2, 3, 4, 5, 6, 3, 7, 8, 9, 10, 11};

inline bool MaterialPropertyValueMatchesType(MaterialPropertyType type, const MaterialPropertyValue &value) noexcept
{
    return value.index() == MaterialPropertyValueIndices[static_cast<size_t>(type)];
}

namespace material_detail
{
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0> bool IsFinite(T value) noexcept
{
    if constexpr (std::is_floating_point_v<T>)
        return std::isfinite(value);
    else
        return true;
}

inline bool IsFinite(const std::string &) noexcept
{
    return true;
}

template <glm::length_t Size, glm::qualifier Precision>
bool IsFinite(const glm::vec<Size, float, Precision> &value) noexcept
{
    for (glm::length_t component = 0; component < Size; ++component)
        if (!IsFinite(value[component]))
            return false;
    return true;
}

template <glm::length_t Columns, glm::length_t Rows, glm::qualifier Precision>
bool IsFinite(const glm::mat<Columns, Rows, float, Precision> &value) noexcept
{
    for (glm::length_t column = 0; column < Columns; ++column)
        if (!IsFinite(value[column]))
            return false;
    return true;
}

template <typename T> bool IsFinite(const std::vector<T> &values) noexcept
{
    return std::all_of(values.begin(), values.end(), [](const auto &value) { return IsFinite(value); });
}
} // namespace material_detail

inline bool MaterialPropertyValueIsFinite(const MaterialPropertyValue &value)
{
    return std::visit([](const auto &item) { return material_detail::IsFinite(item); }, value);
}

struct MaterialProperty
{
    std::string name;
    MaterialPropertyType type;
    MaterialPropertyValue value;
    bool hdr = false;
    std::optional<std::array<double, 2>> range;
    // Derived ShaderInfo metadata; not an authored texture asset reference.
    std::string textureDefault;
    bool inheritsShaderDefault = false;
};

inline bool MaterialPropertyOverrideCompatible(const MaterialProperty &declared, const MaterialProperty &supplied)
{
    if (!MaterialPropertyTypesCompatible(declared.type, supplied.type))
        return false;
    if (declared.type == MaterialPropertyType::FloatArray)
        return std::get<std::vector<float>>(declared.value).size() ==
               std::get<std::vector<float>>(supplied.value).size();
    if (declared.type == MaterialPropertyType::Float4Array)
        return std::get<std::vector<glm::vec4>>(declared.value).size() ==
               std::get<std::vector<glm::vec4>>(supplied.value).size();
    return true;
}

} // namespace infernux
