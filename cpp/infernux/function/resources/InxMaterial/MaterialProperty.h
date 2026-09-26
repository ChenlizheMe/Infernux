#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <optional>
#include <string>
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
    Float4Array
};

using MaterialPropertyValue = std::variant<float, glm::vec2, glm::vec3, glm::vec4, int, glm::mat4, std::string,
                                           std::vector<float>, std::vector<glm::vec4>>;

struct MaterialProperty
{
    std::string name;
    MaterialPropertyType type;
    MaterialPropertyValue value;
    bool hdr = false;
    std::optional<std::array<double, 2>> range;
};

} // namespace infernux
