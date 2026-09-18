#pragma once

#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

class InxMaterial;
class AssetDatabase;

/// Downsampled texture data for CPU preview sampling.
struct PreviewTexture
{
    std::vector<unsigned char> pixels; ///< RGBA8
    int width = 0;
    int height = 0;

    bool IsValid() const
    {
        return !pixels.empty() && width > 0 && height > 0;
    }

    /// Bilinear-filtered sample at normalised UV (wraps).
    glm::vec4 Sample(float u, float v) const;
};

/// Callback that resolves a texture GUID to a small PreviewTexture.
/// Return an invalid PreviewTexture if the GUID cannot be resolved.
using PreviewTextureResolver = std::function<PreviewTexture(const std::string &)>;

// ============================================================================
// ShaderPreviewMapping — parsed from the surface() function
// ============================================================================

/// Mapping of a single PBR surface slot to material property names.
struct SurfaceSlotMapping
{
    std::string scalarProp;  ///< Scalar/color property name (e.g. "metallic")
    std::string textureProp; ///< Texture2D property name (e.g. "metallicMap")
    std::string scaleProp;   ///< Scale float for normal maps (e.g. "normalScale")
};

/// Parsed mapping from a shader's surface() function to PBR slots.
struct ShaderPreviewMapping
{
    /// Slot name → mapping. Known slots: "albedo", "metallic", "smoothness",
    /// "occlusion", "normal", "emission", "specularHighlights", "alpha".
    std::unordered_map<std::string, SurfaceSlotMapping> slots;

    bool IsEmpty() const
    {
        return slots.empty();
    }
};

/// Parse a shader source string and extract surface() PBR slot mappings.
ShaderPreviewMapping ParseShaderPreviewMapping(const std::string &shaderSource);

/// Get (or parse + cache) the preview mapping for a given shader ID.
/// Returns an empty mapping if the shader cannot be found.
ShaderPreviewMapping GetShaderPreviewMapping(const std::string &fragShaderId, AssetDatabase *adb);

// ============================================================================

/// All PBR parameters needed for CPU sphere rendering.
struct PreviewMaterialParams
{
    glm::vec3 baseColor{0.8f, 0.8f, 0.8f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    float ambientOcclusion = 1.0f;
    glm::vec3 emissionColor{0.0f, 0.0f, 0.0f};
    float normalScale = 1.0f;
    float specularHighlights = 1.0f;
    glm::vec4 metallicChannels{1, 0, 0, 0};
    glm::vec4 smoothnessChannels{1, 0, 0, 0};
    float smoothnessFromRoughness = 0.0f;
    float occlusionStrength = 1.0f;

    const PreviewTexture *albedoTex = nullptr;
    const PreviewTexture *metallicTex = nullptr;
    const PreviewTexture *smoothnessTex = nullptr;
    const PreviewTexture *aoTex = nullptr;
    const PreviewTexture *normalTex = nullptr;
    const PreviewTexture *emissionTex = nullptr;
};

/**
 * @brief CPU-side material preview renderer.
 *
 * Generates an RGBA pixel buffer depicting a PBR-lit sphere using the
 * material's properties (colours, metallic, roughness, textures, normal map,
 * AO, emission, etc.).  The result is intended for upload to ImGui as a
 * thumbnail texture.
 *
 * All computation happens on the CPU so no Vulkan state is required.
 */
class MaterialPreviewRenderer
{
  public:
    /// @brief Render a PBR sphere preview for the given material.
    /// @param material  The material to preview.
    /// @param size      Output image width and height (square).
    /// @param outPixels Receives RGBA8 pixel data (size*size*4 bytes).
    /// @param resolver  Optional callback to resolve texture GUIDs to pixel data.
    /// @param mapping   Optional parsed shader mapping; when null, uses hardcoded PBR defaults.
    static void RenderPreview(const InxMaterial &material, int size, std::vector<unsigned char> &outPixels,
                              const PreviewTextureResolver &resolver = nullptr,
                              const ShaderPreviewMapping *mapping = nullptr);

    /// @brief Render from a full parameter struct.
    static void RenderPreview(const PreviewMaterialParams &params, int size, std::vector<unsigned char> &outPixels);
};

} // namespace infernux
