#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernux
{
class UIShaderMaterialError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
    UIShaderMaterialError(const std::string &message, std::string texture, uint64_t generation)
        : std::runtime_error(message), textureGuid(std::move(texture)), textureGeneration(generation)
    {
    }
    std::string textureGuid;
    uint64_t textureGeneration = 0;
};

/**
 * @brief Screen-space UI command list identifier
 */
enum class ScreenUIList
{
    Camera,  ///< Rendered before post-processing
    Overlay, ///< Rendered after post-processing
    World    ///< Local UI geometry transformed into the camera's 3D world
};

/**
 * Stable material contract attached to one retained UI draw command.
 *
 * The GUID and generation select a
 * published UI shader artifact. The text
 * pipeline key is diagnostic only; it never
 * chooses Vulkan code or
 * resources.
 * Screen/world depth rules remain owned by the UI renderer.
 */
struct UIShaderMaterialBinding
{
    std::string materialGuid;
    uint64_t generation = 0;
    std::string pipelineKey;
    // Fixed push-constant values shared by built-in and authored UI programs.
    std::array<float, 4> baseColor{1.0f, 1.0f, 1.0f, 1.0f};
    float alphaClipThreshold = 0.0f;
    bool alphaClipEnabled = false;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return !materialGuid.empty() && generation != 0 && !pipelineKey.empty();
    }

    bool operator==(const UIShaderMaterialBinding &other) const noexcept
    {
        return materialGuid == other.materialGuid && generation == other.generation &&
               pipelineKey == other.pipelineKey && baseColor == other.baseColor &&
               alphaClipThreshold == other.alphaClipThreshold && alphaClipEnabled == other.alphaClipEnabled;
    }
    bool operator!=(const UIShaderMaterialBinding &other) const noexcept
    {
        return !(*this == other);
    }
};

} // namespace infernux
