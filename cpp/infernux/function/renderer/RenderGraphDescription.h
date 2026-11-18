/**
 * @file RenderGraphDescription.h
 * @brief Data structures for render-graph topology defined from Python
 *
 * These POD structures capture the render graph topology defined in Python,
 * allowing C++ to receive, compile, and execute the graph with automatic
 * backend resource transitions and transient resource management.
 *
 * Architecture:
 *   Python has "definition authority" — defines pass topology, resource
 *   connections, and per-pass render actions.
 *   C++ has "compilation authority" — performs DAG compilation, dead-pass
 *   culling, barrier generation, and transient resource allocation.
 */

#pragma once

#include "GraphEvents.h"
#include "rhi/RhiDescriptors.h"
#include "shader/ShaderParameterValue.h"
#include <core/math/Intervals.h>
#include <core/types/ShaderTypes.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infernux
{
namespace rhi
{
class RenderTexture;
class TextureResource;
class ComputeBuffer;
class ComputeWorkRecipe;
struct ComputeBufferDesc;
struct RenderTextureGeneration;
struct RenderViewContext;
} // namespace rhi
class RendererSelection;
class InxMaterial;
struct FullscreenProgram;
struct ShaderTargetCode;
struct MaterialPassPipelineDescriptor;
enum class CameraClearFlags;

/**
 * @brief Backend-neutral command recorded by a graph pass.
 *
 * Commands describe engine rendering operations
 * rather than Vulkan calls so
 * the same graph artifact can be compiled by another RHI backend later.
 * Compute commands reference a common, already-prepared work recipe; graph
 * descriptions never contain a second model or compiler representation.
 */
enum class GraphCommandType
{
    DrawRenderers,
    DrawSkybox,
    DrawShadowCasters,
    DrawWorldUI,
    DrawScreenUI,
    FullscreenQuad,
    CopyTexture,
    CopyBuffer,
    CopyTextureToBuffer,
    CopyBufferToTexture,
    ComputeWork,
    RayTailPrepare,
    RayTailMerge,
    Present
};

enum class GraphPassType
{
    Raster,
    Compute,
    Copy,
    Present
};

enum class GraphMaterialFilter : uint8_t
{
    All,
    DeferredCompatible,
    DeferredUnsupported,
    ThinSheet,
    OrdinaryTransparent,
};

/// A single camera-local witness separates ordinary transparent fragments
/// around the selected optical layer. Both partitions share this resource.
inline constexpr std::string_view kTransparentLayerInput = "transparent_layer";
inline constexpr uint32_t kTransparentLayerViewBinding = 8;

using GraphBufferUsage = rhi::BufferUsageFlags;

enum class GraphBufferAccessType
{
    StorageRead,
    UniformRead,
    StorageWrite,
    IndirectRead,
    TransferRead,
    TransferWrite
};

enum class GraphTextureRole : uint8_t
{
    Transient,
    TemporalRead,
    TemporalWrite,
    Persistent,
    Asset,
    Camera,
    WorldRayColorPages,
    WorldRayDataPages,
};

enum class GraphTextureAttachment : uint8_t
{
    Color,
    Depth,
    Resolve
};

struct GraphCommandDesc
{
    GraphCommandType type = GraphCommandType::DrawRenderers;
    ShaderCompileTarget shaderTarget = ShaderCompileTarget::Forward;
    GraphMaterialFilter materialFilter = GraphMaterialFilter::All;

    int queueMin = 0;
    int queueMax = 5000;
    std::string sortMode;
    std::string passTag;
    std::string overrideMaterial;
    std::shared_ptr<InxMaterial> overrideMaterialValue;
    /// Stable owner; entry/parameter edits do not change graph topology.
    std::shared_ptr<RendererSelection> rendererSelection;
    std::shared_ptr<RendererSelection> selectionAuthor; ///< Logical identity; never read during execution.

    int32_t lightIndex = 0;
    int screenUIList = 0;
    /// World UI pass filter, intersected with the rendering Camera's mask.
    uint32_t worldUILayerMask = 0xffffffffu;

    std::string shaderName;
    std::shared_ptr<const FullscreenProgram> fullscreenProgram;
    /// Fullscreen raster state. Renderer draws retain their material state.
    bool depthTest = false;
    bool depthWrite = false;
    rhi::CompareFunction depthCompare = rhi::CompareFunction::Always;
    bool alphaBlend = false;
    /// Stable pass-local runtime parameter block. When non-empty,
    /// parameters define the block layout and initial values rather than
    /// immutable topology. This payload does not override Material/Renderer,
    /// View or World values by name.
    std::string parameterBlock;
    std::vector<std::pair<std::string, ShaderParameterValue>> parameters;
    std::vector<std::pair<std::string, std::string>> inputBindings;

    std::string sourceResource;
    std::string destinationResource;
    uint64_t copyBytes = 0;
    uint32_t copyWidth = 0;
    uint32_t copyHeight = 0;
    uint64_t copyBufferOffset = 0; ///< Relative to the logical buffer range.
    std::string workBinding;

    [[nodiscard]] bool BindsResource(const std::string &name) const;
    [[nodiscard]] std::string_view TransparentLayer() const noexcept;
    [[nodiscard]] bool CopiesToTexture() const noexcept
    {
        return type == GraphCommandType::CopyTexture || type == GraphCommandType::CopyBufferToTexture;
    }
};

/**
 * @brief Revisioned values for one graph-owned runtime parameter block.
 *
 * Parameter updates are intentionally
 * separate from RenderGraphDescription so
 * ordinary effect edits do not rebuild or recompile graph topology.
 */
struct GraphParameterBlockUpdate
{
    std::string id;
    uint64_t revision = 0;
    std::vector<std::pair<std::string, ShaderParameterValue>> values;
};

// ============================================================================
// Texture Description
// ============================================================================

/**
 * @brief Description of a texture resource in the Python-defined graph
 */
struct GraphTextureDesc
{
    using RenderTextureLayout =
        std::tuple<rhi::PixelFormat, rhi::PixelFormat, rhi::SampleCount, rhi::FilterMode, bool, bool>;

    std::string name;                                       ///< Unique resource name
    rhi::PixelFormat format = rhi::PixelFormat::RGBA8UNorm; ///< Backend-neutral pixel format
    uint32_t width = 0;                                     ///< Custom width (0 = use scene target size)
    uint32_t height = 0;                                    ///< Custom height (0 = use scene target size)
    uint32_t sizeDivisor = 0;                               ///< >0: actual = scene_size / divisor
    uint32_t samples = 1;                                   ///< 0 = inherit frame MSAA, otherwise 1/2/4/8
    GraphTextureRole role = GraphTextureRole::Transient;    ///< Frame-local or one side of a temporal history pair
    std::string temporalKey;                                ///< Stable per-view history identity for temporal resources
    std::shared_ptr<rhi::RenderTexture> renderTexture;      ///< Persistent owner; never a disk GUID
    std::shared_ptr<const rhi::RenderTextureGeneration> renderTextureGeneration;
    [[nodiscard]] std::shared_ptr<const rhi::RenderTextureGeneration> AcquireRenderTextureGeneration() const;
    GraphTextureAttachment attachment = GraphTextureAttachment::Color;
    bool requireCurrentWrite = false; ///< Input-only import requires a producer in the same render opportunity.
    std::string assetGuid;            ///< GUID of a read-only imported Texture asset
    uint32_t depth = 1;               ///< Imported Texture3D depth; 1 for ordinary textures
    bool isVolume = false;            ///< True when the imported asset is a Texture3D
    std::optional<RenderTextureLayout> renderTextureLayout;

    [[nodiscard]] bool operator==(const GraphTextureDesc &other) const;

    void BindRenderTexture(std::shared_ptr<rhi::RenderTexture> owner, GraphTextureAttachment attachment,
                           std::optional<bool> requireWrite = {});
    [[nodiscard]] bool MatchesRenderTextureLayout(const rhi::RenderTextureGeneration &generation) const noexcept;

    [[nodiscard]] bool IsDepth() const noexcept
    {
        return rhi::IsDepthFormat(format);
    }
    [[nodiscard]] bool IsViewAttachment() const noexcept
    {
        return role == GraphTextureRole::Camera;
    }
    [[nodiscard]] bool IsWorldRayPage() const noexcept
    {
        return role == GraphTextureRole::WorldRayColorPages || role == GraphTextureRole::WorldRayDataPages;
    }
    [[nodiscard]] bool IsSampledImport() const noexcept
    {
        return role == GraphTextureRole::Asset || IsWorldRayPage();
    }
    [[nodiscard]] bool IsViewColor() const noexcept
    {
        return IsViewAttachment() && !IsDepth();
    }
    [[nodiscard]] bool IsViewDepth() const noexcept
    {
        return IsViewAttachment() && IsDepth();
    }
    [[nodiscard]] rhi::PixelFormat EffectiveFormat(const rhi::RenderViewContext &view) const noexcept;
    [[nodiscard]] uint32_t EffectiveSamples(uint32_t frameSamples) const noexcept;
    [[nodiscard]] std::pair<uint32_t, uint32_t> ResolveExtent(uint32_t viewWidth, uint32_t viewHeight) const noexcept;
};

struct GraphWorkBindingDesc
{
    std::string name;
    std::shared_ptr<rhi::ComputeWorkRecipe> recipe;
    /// Slot-indexed World input leases. Private Work storage stays renderer-owned.
    std::vector<std::shared_ptr<rhi::ComputeBuffer>> externalInputs;
    /// Input port name -> earlier graph buffer with a matching true owner.
    std::vector<std::pair<std::string, std::string>> inputBindings;
};

struct GraphBufferCapture;

/// A fixed graph input whose accepted, immutable GPU value can change between frames.
class GraphBufferInput final
{
  public:
    GraphBufferInput(const std::shared_ptr<rhi::ComputeBuffer> &value, uint64_t offset, uint64_t byteSize);
    void Publish(const std::shared_ptr<rhi::ComputeBuffer> &value);
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetValue() const noexcept
    {
        return m_value;
    }
    void SetRequirement(bool pending, bool holdsStep, std::vector<std::shared_ptr<GraphBufferCapture>> sources);
    [[nodiscard]] bool IsWaitingForResult() const noexcept
    {
        return m_waiting;
    }
    [[nodiscard]] bool HoldsStep() const noexcept
    {
        return m_holdsStep;
    }
    [[nodiscard]] const auto &GetResultSources() const noexcept
    {
        return m_sources;
    }
    [[nodiscard]] bool HasResultCycle() const noexcept
    {
        return m_cycle;
    }
    void MarkResultCycle() noexcept;

  private:
    std::shared_ptr<rhi::ComputeBuffer> m_value;
    std::vector<std::shared_ptr<GraphBufferCapture>> m_sources;
    uint64_t m_offset;
    uint64_t m_byteSize;
    bool m_waiting = false;
    bool m_holdsStep = false;
    bool m_cycle = false;
};

enum class GraphFrameBuffer : uint8_t
{
    None,
    ViewLights,
    WorldSurfaces,
    WorldRayGeometry,
    WorldRayMeshes,
    WorldRayLitMaterials,
    WorldRayTextureDescriptors,
    WorldRayTailRead,
    WorldRayTailWrite,
    WorldRayTailRequests,
    WorldRayTailControl,
    Count,
};

struct GraphBufferDesc
{
    std::string name;
    uint64_t byteSize = 0;
    uint32_t usage = static_cast<uint32_t>(GraphBufferUsage::None);
    std::shared_ptr<rhi::ComputeBuffer> computeBuffer; ///< Imported owner captured for this execution.
    std::shared_ptr<GraphBufferInput> input;           ///< Optional slot supplying the imported value.
    std::shared_ptr<rhi::ComputeBuffer> bufferAuthor;
    std::shared_ptr<GraphBufferInput> inputAuthor;
    /// Read-only data owned by the renderer and bound for this View's execution.
    GraphFrameBuffer frameBuffer = GraphFrameBuffer::None;
    [[nodiscard]] bool HasFrameBuffer() const noexcept
    {
        return frameBuffer != GraphFrameBuffer::None;
    }
    uint64_t byteOffset = 0; ///< Logical imported view; physical hazards cover the complete allocation.
    std::string workBinding;
    uint32_t workSlot = 0;
    /// Typed physical layout and logical tensor shape of a public final value.
    std::optional<rhi::ComputeBufferDesc> exportLayout;
    std::vector<uint64_t> exportShape;
};

struct GraphBufferAccessDesc
{
    std::string resource;
    GraphBufferAccessType type = GraphBufferAccessType::StorageRead;
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
    GraphPassType type = GraphPassType::Raster;

    // === Resource connections ===
    std::vector<std::string> readTextures; ///< Names of textures this pass reads
    /// MRT color outputs: list of (slot, texture_name) pairs.
    /// Slot 0 is the primary color output; higher slots enable deferred / GBuffer.
    std::vector<std::pair<int, std::string>> writeColors;
    std::string writeDepth;   ///< Name of depth output texture
    std::string resolveColor; ///< Optional 1x resolve target for color slot 0
    std::vector<GraphBufferAccessDesc> bufferAccesses;
    bool sideEffect = false;
    /// Private directional-cache batch: 0 ordinary, 1 any slice, 2 estimate slice only.
    uint32_t rayTailBatch = 0;
    std::vector<GraphEventDesc> events;

    // === Clear settings ===
    bool clearColor = false;
    bool clearDepth = false;
    float clearColorR = 0.0f;
    float clearColorG = 0.0f;
    float clearColorB = 0.0f;
    float clearColorA = 1.0f;
    float clearDepthValue = 0.0f;

    // === Typed command IR ===
    // Empty is valid for a resource-only pass. The current executor accepts
    // one command while the IR is intentionally a list for the upcoming
    // raster/compute/copy command-list executor.
    std::vector<GraphCommandDesc> commands;

    [[nodiscard]] const GraphCommandDesc *PrimaryCommand() const noexcept;
    [[nodiscard]] bool WritesTexture(const std::string &name) const;
    /// Resource reads include transfer/export sources; this does not imply sampling.
    [[nodiscard]] bool ReadsTexture(const std::string &name) const;
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

    /// Monotonic source artifact revision assigned when Python records the graph.
    /// Steady-state frames send only this value to reuse an already-applied graph.
    uint64_t sourceRevision = 0;

    std::vector<GraphTextureDesc> textures; ///< All texture resources
    std::vector<GraphBufferDesc> buffers;   ///< All buffer resources
    std::vector<GraphWorkBindingDesc> workBindings;
    std::vector<GraphPassDesc> passes; ///< All passes in declaration order
    std::string outputTexture;         ///< Name of the final output texture
    bool outputDisplayEncoded = false; ///< Sampled output already contains display-encoded RGB.
    /// Terminal linear image before display encoding/overlay. Offscreen
    /// cameras consume this prefix, not a display-encoded screen image.
    std::string linearOutputTexture;
    uint32_t linearOutputPassCount = 0;

    /// MSAA sample count requested by the pipeline (0 = don't change, 1/2/4/8).
    int msaaSamples = 0;
    /// Explicit camera sampling policy, independent of double-buffered resources.
    bool temporalJitter = false;

    /// Non-owning reuse of a capture whose definition has no external author values.
    /// World/UI/culling/parameters remain on each independent RenderViewPlan.
    mutable std::weak_ptr<const RenderGraphDescription> captureCache;

    [[nodiscard]] bool RenderTextureLayoutsCurrent() const;
};

/// Capture author values once at the completed-World boundary; no GPU submission.
RenderGraphDescription CaptureRenderGraphDescription(
    const RenderGraphDescription &source,
    const std::function<void(const std::shared_ptr<InxMaterial> &)> &prepareMaterial,
    const std::function<std::shared_ptr<const FullscreenProgram>(const std::string &)> &acquireProgram);

std::shared_ptr<const RenderGraphDescription> AcquireRenderGraphPublication(
    const RenderGraphDescription &source,
    const std::function<void(const std::shared_ptr<InxMaterial> &)> &prepareMaterial,
    const std::function<std::shared_ptr<const FullscreenProgram>(const std::string &)> &acquireProgram);

/// Topology excludes source revisions and independently updated parameter values.
bool RenderGraphTopologyEquals(const RenderGraphDescription &a, const RenderGraphDescription &b);
RenderGraphDescription NormalizeRenderGraphForOutput(RenderGraphDescription desc, bool hasOutputTexture,
                                                     rhi::SampleCount actualSamples);
/// Final description and current imported-resource checks before publication.
using GraphBufferContents = std::unordered_map<std::string, intervals::Set>;
bool RequiresGraphBufferOwners(const RenderGraphDescription &desc);
bool RequiresGraphBufferProof(const RenderGraphDescription &desc);
bool ValidateRenderGraphDescription(const RenderGraphDescription &desc, const rhi::RenderViewContext &view,
                                    GraphBufferContents *contents = nullptr);
std::optional<GraphPassDesc> ResolveGraphCameraClear(const RenderGraphDescription &graph, CameraClearFlags flags,
                                                     const glm::vec4 &linearBackground, bool preview = false);
/// The view lighting set consumes this graph image outside material set 0.
std::string ResolveGraphShadowTexture(const RenderGraphDescription &graph);
bool GraphUsesForwardPlus(const RenderGraphDescription &graph);
/// The reflected physical table layout is shared by every ray Lit consumer of a World.
std::shared_ptr<const ShaderTargetCode> ResolveGraphRayLitProgram(const RenderGraphDescription &graph);
std::pair<uint32_t, uint32_t> ResolveGraphPassExtent(const RenderGraphDescription &graph, const GraphPassDesc &pass,
                                                     uint32_t viewWidth, uint32_t viewHeight);
const GraphTextureDesc *ResolveGraphReadOnlyDepth(const RenderGraphDescription &graph, const GraphPassDesc &pass);
const rhi::TextureResource *ResolveGraphTextureAttachment(const rhi::RenderTextureGeneration &generation,
                                                          GraphTextureAttachment attachment) noexcept;
rhi::TextureHandle ResolveGraphTextureImport(const GraphTextureDesc &texture);
const GraphTextureDesc *FindGraphOutputTextureAlias(const RenderGraphDescription &graph,
                                                    const rhi::RenderTextureGeneration &output);
struct MaterialTextureRead
{
    std::string passName;
    std::shared_ptr<rhi::RenderTexture> texture;
    std::shared_ptr<const rhi::RenderTextureGeneration> generation;

    bool operator==(const MaterialTextureRead &other) const
    {
        return passName == other.passName && texture == other.texture && generation == other.generation;
    }
};
/// A material can sample a persistent texture written by an earlier graph pass.
/// Feedback or a local writer declared after its reader is an invalid graph.
bool HasLocalGraphTextureProducer(const RenderGraphDescription &graph, const MaterialTextureRead &read);
const GraphWorkBindingDesc *FindGraphWorkBinding(const RenderGraphDescription &graph, const std::string &name);
/// Actual command writes plus explicit access declarations, including every Work alias.
std::unordered_set<std::string> FindGraphBufferWrites(const RenderGraphDescription &graph);
const GraphBufferDesc &ResolveGraphBufferStorage(const RenderGraphDescription &graph, const GraphBufferDesc &buffer);
const rhi::ComputeBufferDesc *ResolveGraphComputeBufferDesc(const RenderGraphDescription &graph,
                                                            const GraphBufferDesc &buffer);
MaterialPassPipelineDescriptor ResolveMaterialPass(const RenderGraphDescription &graph, const GraphPassDesc &pass,
                                                   const rhi::RenderViewContext &view, bool invertCulling = false);

} // namespace infernux
