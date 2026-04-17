/**
 * @file RenderGraphDescription.h
 * @brief Data structures for render-graph topology defined from Python
 *
 * These POD structures capture the render graph topology defined in Python,
 * allowing C++ to receive, compile, and execute the graph with automatic
 * Vulkan barrier insertion and transient resource management.
 *
 * Architecture:
 *   Python has "definition authority" — defines pass topology, resource
 *   connections, and per-pass render actions.
 *   C++ has "compilation authority" — performs DAG compilation, dead-pass
 *   culling, barrier generation, and transient resource allocation.
 */

#pragma once

#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux
{

// ============================================================================
// Pass Action Types
// ============================================================================

/**
 * @brief The rendering action a graph pass should perform
 */
enum class GraphPassActionType
{
    None,              ///< No rendering (resource-only pass)
    DrawRenderers,     ///< Draw scene renderers filtered by queue range
    DrawSkybox,        ///< Draw the procedural skybox
    Compute,           ///< Dispatch a compute shader (no render pass)
    Custom,            ///< Reserved for future Python callback support
    DrawShadowCasters, ///< Draw shadow casters into a depth-only shadow map
    DrawScreenUI,      ///< Draw screen-space UI (Camera or Overlay list)
    FullscreenQuad     ///< Draw a fullscreen triangle with a named shader (post-process)
};

// ============================================================================
// Texture Description
// ============================================================================

/**
 * @brief Description of a texture resource in the Python-defined graph
 */
struct GraphTextureDesc
{
    std::string name;                           ///< Unique resource name
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM; ///< Vulkan format
    bool isBackbuffer = false;                  ///< If true, refers to the scene's main MSAA color target
    bool isDepth = false;                       ///< If true, this is a depth/stencil texture
    uint32_t width = 0;                         ///< Custom width (0 = use scene target size)
    uint32_t height = 0;                        ///< Custom height (0 = use scene target size)
    uint32_t sizeDivisor = 0;                   ///< >0: actual = scene_size / divisor
};

// ============================================================================
// Pass Description
// ============================================================================

/**
 * @brief Description of a single render pass in the Python-defined graph
 */
struct GraphPassDesc
{
    std::string name; ///< Pass name (must be unique within the graph)

    // === Resource connections ===
    std::vector<std::string> readTextures; ///< Names of textures this pass reads
    /// MRT color outputs: list of (slot, texture_name) pairs.
    /// Slot 0 is the primary color output; higher slots enable deferred / GBuffer.
    std::vector<std::pair<int, std::string>> writeColors;
    std::string writeDepth; ///< Name of depth output texture

    // === Clear settings ===
    bool clearColor = false;
    bool clearDepth = false;
    float clearColorR = 0.0f;
    float clearColorG = 0.0f;
    float clearColorB = 0.0f;
    float clearColorA = 1.0f;
    float clearDepthValue = 1.0f;

    // === Render action ===
    GraphPassActionType action = GraphPassActionType::None;

    // DrawRenderers parameters
    int queueMin = 0;             ///< Minimum render queue (inclusive)
    int queueMax = 5000;          ///< Maximum render queue (inclusive)
    std::string sortMode;         ///< "front_to_back", "back_to_front", "none"
    std::string passTag;          ///< Filter draw calls by shader pass tag (empty = no filter)
    std::string overrideMaterial; ///< Force all objects to use this material name (empty = per-object)

    // Compute parameters
    std::string computeShaderName; ///< Compute shader name (for Compute action)
    uint32_t dispatchX = 1;        ///< Compute dispatch group count X
    uint32_t dispatchY = 1;        ///< Compute dispatch group count Y
    uint32_t dispatchZ = 1;        ///< Compute dispatch group count Z

    // DrawShadowCasters parameters
    int32_t lightIndex = 0;          ///< Index of the shadow-casting light (0 = first directional)
    std::string shadowType = "hard"; ///< Shadow quality: "hard", "soft"

    // DrawScreenUI parameters
    int screenUIList = 0; ///< 0 = Camera list (before post-process), 1 = Overlay list (after post-process)

    // FullscreenQuad parameters
    std::string shaderName; ///< Shader id for FullscreenQuad action (e.g. "bloom_prefilter")
    /// Named push-constant values (name → float) passed to the fragment shader
    std::vector<std::pair<std::string, float>> pushConstants;

    // === Input bindings (sampled texture inputs) ===
    /// Maps sampler name → texture name for textures sampled by this pass
    /// (e.g. shadow map bound as a sampled texture in a lighting pass).
    std::vector<std::pair<std::string, std::string>> inputBindings;
};

// ============================================================================
// RenderGraph Description (complete topology from Python)
// ============================================================================

/**
 * @brief Complete render graph topology defined by Python
 *
 * This structure is built by the Python RenderGraph API and sent to C++
 * via SceneRenderGraph::ApplyPythonGraph(). C++ uses it to configure
 * the SceneRenderGraph passes and underlying vk::RenderGraph.
 */
struct RenderGraphDescription
{
    std::string name; ///< Graph name for debugging

    std::vector<GraphTextureDesc> textures; ///< All texture resources
    std::vector<GraphPassDesc> passes;      ///< All passes in declaration order
    std::string outputTexture;              ///< Name of the final output texture

    /// MSAA sample count requested by the pipeline (0 = don't change, 1/2/4/8).
    int msaaSamples = 0;
};

} // namespace infernux
