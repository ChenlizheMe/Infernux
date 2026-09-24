/**
 * @file InxScreenUIRenderer.h
 * @brief GPU-based 2D screen-space UI renderer for RenderGraph integration
 *
 * Renders screen-space UI elements (filled rects, text) directly into the
 * scene render target as a RenderGraph pass. Uses ImGui's ImDrawList for
 * command accumulation and font atlas for text rendering, but maintains its
 * own Vulkan pipeline and buffers to render inside scene render passes
 * (MSAA backbuffer) rather than the ImGui overlay pass.
 *
 * Two independent command lists are maintained:
 *   - Camera list: rendered before post-processing (Screen Space - Camera)
 *   - Overlay list: rendered after post-processing  (Screen Space - Overlay)
 *
 * This allows the Python-side RenderGraph to place ScreenUI passes at
 * different points in the pipeline depending on the canvas render mode.
 */

#pragma once

#include "../rhi/GpuRetirementQueue.h"
#include "../rhi/RhiRenderTexture.h"
#include "../MaterialDescriptor.h"
#include "WorldUIOcclusionPlan.h"
#include <core/types/ShaderProgramArtifact.h>
#include <function/scene/TransformECSStore.h>

#include <array>
#include <cstdint>
#include <functional>
#include <glm/mat4x4.hpp>
#include <imgui.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace infernux
{

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
 * The GUID and generation select a published UI shader artifact. The text
 * pipeline key is diagnostic only; it never
 * chooses Vulkan code or resources.
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

/**
 * @brief GPU-based 2D screen-space UI renderer
 *
 * Accumulates draw commands (filled rects, textured rects, text) per frame
 * and renders them into the scene MSAA render target during a RenderGraph pass.
 */
class InxScreenUIRenderer
{
    friend struct ScreenUIVulkanTestAccess;

  public:
    InxScreenUIRenderer();
    ~InxScreenUIRenderer();

    // Non-copyable
    InxScreenUIRenderer(const InxScreenUIRenderer &) = delete;
    InxScreenUIRenderer &operator=(const InxScreenUIRenderer &) = delete;

    /**
     * @brief Initialize the renderer
     * @param device Vulkan device
     * @param allocator VMA allocator for buffer management
     * @param colorFormat Scene color attachment format (e.g. B8G8R8A8_SRGB)
     * @param msaaSamples Scene MSAA sample count (e.g. 4x)
     * @return true if successful
     */
    bool Initialize(VkDevice device, VmaAllocator allocator, VkFormat colorFormat, VkFormat depthFormat,
                    VkSampleCountFlagBits msaaSamples, uint32_t frameCount);

    void SetRetirementQueue(GpuRetirementQueue *queue)
    {
        m_deletionQueue = queue;
    }

    void SetTextureUsageValidator(std::function<bool(uint64_t)> validator)
    {
        m_textureUsageValidator = std::move(validator);
    }
    void SetRenderTextureResolver(std::function<std::shared_ptr<rhi::RenderTexture>(uint64_t)> resolver)
    {
        m_renderTextureResolver = std::move(resolver);
    }
    void SetTextureColorSpaceQuery(std::function<bool(uint64_t)> query)
    {
        m_textureColorSpaceQuery = std::move(query);
    }
    void SetMaterialProgramResolver(
        std::function<std::shared_ptr<const ShaderProgramArtifact>(const std::string &, uint64_t, ShaderProgramDomain)>
            resolver)
    {
        m_materialProgramResolver = std::move(resolver);
    }
    void SetMaterialIdentityValidator(std::function<bool(const std::string &, uint64_t)> validator)
    {
        m_materialIdentityValidator = std::move(validator);
    }
    void SetMaterialProgramRelease(std::function<void(const ShaderProgramKey &)> release)
    {
        m_materialProgramRelease = std::move(release);
    }
    void SetMaterialProgramAcquire(std::function<void(const ShaderProgramKey &)> acquire)
    {
        m_materialProgramAcquire = std::move(acquire);
    }
    void SetMaterialProgramReleaseSweep(std::function<void()> sweep)
    {
        m_materialProgramReleaseSweep = std::move(sweep);
    }
    void SetMaterialAssetResolver(std::function<std::shared_ptr<const InxMaterial>(const std::string &, uint64_t)> resolver)
    {
        m_materialAssetResolver = std::move(resolver);
    }
    void SetMaterialTextureResolver(TextureResolver resolver)
    {
        m_materialTextureResolver = std::move(resolver);
    }
    void SetMaterialTextureGenerationResolver(std::function<uint64_t(const std::string &)> resolver)
    {
        m_materialTextureGenerationResolver = std::move(resolver);
    }
    void InvalidateMaterialProgram(const ShaderStagePair &stages);
    [[nodiscard]] size_t GetMaterialPipelineVariantCount() const noexcept
    {
        return m_materialPipelineVariants.size();
    }
    std::vector<std::shared_ptr<rhi::RenderTexture>> GetRenderTextureReads(ScreenUIList list,
                                                                           uint32_t cullingMask = 0xffffffffu) const;

    /**
     * @brief Cleanup all resources
     */
    void Destroy();

    // ========================================================================
    // Per-Frame Command Accumulation
    // ========================================================================

    /**
     * @brief Begin a new frame — clears all accumulated commands
     *
     * Must be called once per frame before any Add* calls.
     * @param width  Scene render target width  (pixels)
     * @param height Scene render target height (pixels)
     */
    void BeginFrame(uint32_t width, uint32_t height);

    /**
     * @brief Reuse the previous draw lists when UI content is unchanged.
     * @return true when cached
     * commands were retained; false when callers
     *         must submit the frame's Add* commands again.
     */
    bool BeginFrameCached(uint32_t width, uint32_t height, uint64_t contentRevision);

    // Immutable CPU geometry for one retained UI element. It owns no ImGui
    // context or GPU objects and may be released after renderer shutdown.
    class CommandPacket
    {
      public:
        CommandPacket();
        ~CommandPacket();

      private:
        friend class InxScreenUIRenderer;
        struct Data;
        std::unique_ptr<Data> m_data;
    };
    void BeginCommandPacket();
    std::shared_ptr<CommandPacket> EndCommandPacket();
    void AbortCommandPacket();
    void AppendCommandPackets(const std::vector<std::shared_ptr<CommandPacket>> &packets);
    std::array<uint64_t, 3> GetCommandPacketEpoch() const;

    /// Attach a GUID-backed material contract to subsequent commands in a
    /// packet. The binding is copied into the packet and aligned with its
    /// ImDrawCmd entries; no path or native pointer is retained. An empty
    /// GUID/generation/key triple restores the engine default material.
    void SetMaterialBinding(ScreenUIList list, const std::string &materialGuid, uint64_t generation,
                            const std::string &pipelineKey);
    void SetMaterialBinding(ScreenUIList list, const std::string &materialGuid, uint64_t generation,
                            const std::string &pipelineKey, const std::array<float, 4> &baseColor,
                            bool alphaClipEnabled = false, float alphaClipThreshold = 0.0f);
    /// Inspect the command-aligned contracts published for the current frame.
    const std::vector<UIShaderMaterialBinding> &GetCommandBindings(ScreenUIList list) const;

    /// Intersect subsequent commands with a screen-space clip rectangle.
    void PushClipRect(ScreenUIList list, float minX, float minY, float maxX, float maxY);
    void PopClipRect(ScreenUIList list);

    /// Begin one independent world UI element. Its geometry uses an ordinary
    /// scene Transform; the default scene depth test is bypassed only when
    /// alwaysOnTop is explicit. World UI has no Canvas or root plane.
    void BeginWorldElement(const std::array<float, 16> &localToWorld, float pivotX, float pivotY,
                           uint32_t layerMask = 0xffffffffu, bool alwaysOnTop = false, bool billboard = false,
                           bool constantScreenSize = false);
    /// Retain local geometry while sampling this scene object's current pose
    /// at packet publication. UI ignores scale, but inherits parent motion.
    void BeginWorldObject(GameObject *object, float pivotX, float pivotY, bool alwaysOnTop = false,
                          bool billboard = false, bool constantScreenSize = false, uint64_t ignoredOccluderId = 0);
    void BeginScreenObject(GameObject *object, ScreenUIList list, float pivotX, float pivotY, float scaleX = 1.0f,
                           float scaleY = 1.0f);
    void EndScreenObject();
    void EndWorldElement();

    /**
     * @brief Add a filled rectangle
     */
    void AddFilledRect(ScreenUIList list, float minX, float minY, float maxX, float maxY, float r, float g, float b,
                       float a, float rounding = 0.0f, float rotation = 0.0f, bool mirrorH = false,
                       bool mirrorV = false);

    /**
     * @brief Add a textured image quad
     */
    void AddImage(ScreenUIList list, uint64_t textureId, float minX, float minY, float maxX, float maxY,
                  float uv0X = 0.0f, float uv0Y = 0.0f, float uv1X = 1.0f, float uv1Y = 1.0f, float r = 1.0f,
                  float g = 1.0f, float b = 1.0f, float a = 1.0f, float rotation = 0.0f, bool mirrorH = false,
                  bool mirrorV = false, float rounding = 0.0f);

    /**
     * @brief Add aligned text (uses ImGui font atlas)
     */
    void AddText(ScreenUIList list, float minX, float minY, float maxX, float maxY, const std::string &text, float r,
                 float g, float b, float a, float alignX, float alignY, float fontSize, float wrapWidth = 0.0f,
                 float rotation = 0.0f, bool mirrorH = false, bool mirrorV = false, const std::string &fontPath = "",
                 float lineHeight = 1.0f, float letterSpacing = 0.0f, bool clip = false,
                 const std::vector<std::string> &fallbackFontPaths = {});
    std::pair<float, float> MeasureText(const std::string &text, float fontSize, float wrapWidth = 0.0f,
                                        const std::string &fontPath = "", float lineHeight = 1.0f,
                                        float letterSpacing = 0.0f,
                                        const std::vector<std::string> &fallbackFontPaths = {}) const;

    /**
     * @brief Check if a command list has any draw commands
     */
    bool HasCommands(ScreenUIList list) const;

    /**
     * @brief Number of draw calls submitted by the most recent Render() for a list
     */
    uint32_t GetLastSubmittedDrawCount(ScreenUIList list) const
    {
        return m_lastSubmittedDrawCounts[ListIndex(list)];
    }

    /// Cumulative work since initialization, for native profiling/regressions.
    struct GeometryStats
    {
        uint64_t preparations = 0;
        uint64_t uploads = 0;
        uint64_t uploadedBytes = 0;
        uint64_t packetCaptures = 0;
        uint64_t packetAppends = 0;
    };
    const GeometryStats &GetGeometryStats(ScreenUIList list) const
    {
        return m_geometryStats[ListIndex(list)];
    }

    /// Number of indices submitted by the most recent Render() for a list.
    uint64_t GetLastSubmittedIndexCount(ScreenUIList list) const
    {
        return m_lastSubmittedIndexCounts[ListIndex(list)];
    }

    /**
     * @brief Enable or disable rendering (commands still accumulate)
     *
     * When disabled, Render() becomes a no-op. Useful for suppressing
     * screen-UI in the game texture while the UI editor draws its own
     * elements on top of the same texture.
     */
    void SetEnabled(bool enabled)
    {
        m_enabled = enabled;
    }
    bool IsEnabled() const
    {
        return m_enabled;
    }

    // ========================================================================
    // Rendering (called from RenderGraph pass callback)
    // ========================================================================

    /**
     * @brief Render the specified command list into the active render pass
     *
     * Must be called inside a Vulkan render pass that is compatible with
     * the color format and MSAA settings passed to Initialize().
     *
     * @param cmdBuf Vulkan command buffer (inside active render pass)
     * @param list   Which command list to render
     * @param width  Render target width
     * @param height Render target height
     */
    // frameSlot is owned by VkCore: its previous GPU submission must be complete.
    // Publish UI commands before rendering; all cameras consume that snapshot.
    void Render(VkCommandBuffer cmdBuf, ScreenUIList list, uint32_t width, uint32_t height, uint32_t frameSlot);

    /// Draw world-space UI against the current camera depth attachment.
    void RenderWorld(VkCommandBuffer cmdBuf, uint32_t width, uint32_t height, const glm::mat4 &viewProjection,
                     const rhi::GraphicsRenderingSignature &target, uint32_t frameSlot,
                     uint32_t cullingMask = 0xffffffffu, const glm::mat4 &view = glm::mat4(1.0f),
                     const glm::mat4 &projection = glm::mat4(1.0f), uint32_t firstOrdinal = 0,
                     uint32_t endOrdinal = 0xffffffffu);

    using WorldDepthRun = WorldUIOcclusionRun;
    [[nodiscard]] bool HasSelectiveWorldOcclusion(uint32_t cullingMask = 0xffffffffu) const;
    [[nodiscard]] std::vector<WorldDepthRun> GetWorldDepthRuns(const glm::mat4 &viewProjection,
                                                               uint32_t cullingMask = 0xffffffffu) const;

  private:
    static constexpr int ListIndex(ScreenUIList list) noexcept
    {
        return list == ScreenUIList::Camera ? 0 : (list == ScreenUIList::Overlay ? 1 : 2);
    }

    /**
     * @brief Create Vulkan pipeline objects (shader modules, layouts, pipeline)
     */
    bool CreatePipeline();
    bool CreateScreenPipeline(VkShaderModule vertex, VkShaderModule fragment, VkPipeline &pipeline);
    bool CreateWorldPipeline();
    bool CreateWorldPipeline(const rhi::GraphicsRenderingSignature &target, VkPipeline &pipeline,
                             bool alwaysOnTop = false, VkShaderModule vertex = VK_NULL_HANDLE,
                             VkShaderModule fragment = VK_NULL_HANDLE);
    VkPipeline GetWorldPipeline(const rhi::GraphicsRenderingSignature &target, bool alwaysOnTop = false);
    VkPipeline GetMaterialPipeline(const UIShaderMaterialBinding &binding, ShaderProgramDomain domain,
                                   const rhi::GraphicsRenderingSignature *target = nullptr, bool alwaysOnTop = false);
    VkDescriptorSet GetMaterialDescriptor(const UIShaderMaterialBinding &binding,
                                          const ShaderProgramArtifact &artifact);
    void RetireMaterialDescriptor(const std::string &guid);
    void RetireMaterialPipelineVariants(const ShaderProgramKey &key);
    void PruneUnusedMaterialPrograms();

    // Independent per-list buffers in each engine frame slot. Cameras share
    // immutable geometry within a frame; the next frame cannot overwrite it.
    struct ListBuffers
    {
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VmaAllocation vertexAlloc = VK_NULL_HANDLE;
        VkDeviceSize vertexBufferSize = 0;

        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VmaAllocation indexAlloc = VK_NULL_HANDLE;
        VkDeviceSize indexBufferSize = 0;
        uint64_t uploadedRevision = 0;
    };

    /**
     * @brief Ensure vertex/index buffers are large enough
     */
    bool EnsureBuffers(ListBuffers &buf, VkDeviceSize vertexSize, VkDeviceSize indexSize);
    bool UploadGeometry(ListBuffers &buf, ScreenUIList list, const void *vertices, size_t vertexBytes);

    /**
     * @brief Get the ImDrawList for a given list
     */
    ImDrawList *GetDrawList(ScreenUIList list);
    const ImDrawList *GetDrawList(ScreenUIList list) const;

    struct GPUVertex
    {
        ImVec2 pos;
        ImVec2 uv;
        float color[4];
    };

    struct WorldGPUVertex
    {
        float pos[3];
        ImVec2 uv;
        float color[4];
        ImVec2 localPos;
        float anchor[3];
        ImVec2 localOffset;
        float policy = 0.0f;
    };

    struct WorldElementSpan
    {
        int vertexStart = 0;
        int vertexEnd = 0;
        int commandStart = 0;
        int commandEnd = 0;
        glm::mat4 localToWorld{1.0f};
        uint32_t layerMask = 0xffffffffu;
        float pivotX = 0.0f;
        float pivotY = 0.0f;
        bool alwaysOnTop = false;
        bool billboard = false;
        bool constantScreenSize = false;
        uint64_t ignoredOccluderId = 0;
        TransformECSStore::Handle transform;
    };

    static void ResolveWorldPose(WorldElementSpan &span);

    struct ScreenElementSpan
    {
        int vertexStart = 0;
        int vertexEnd = 0;
        ScreenUIList list = ScreenUIList::Camera;
        TransformECSStore::Handle transform;
        glm::vec3 position{0.0f};
        float rotation = 0.0f;
        glm::vec3 delta{0.0f};
        float deltaRotation = 0.0f;
        float pivotX = 0.0f;
        float pivotY = 0.0f;
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        float transformScaleX = 1.0f;
        float transformScaleY = 1.0f;
        float currentScaleX = 1.0f;
        float currentScaleY = 1.0f;
    };
    static bool ResolveScreenPose(ScreenElementSpan &span);
    static void ApplyScreenPose(const ScreenElementSpan &span, const ImVec2 &source, ImVec2 &target);

    struct HDRColorRange
    {
        int vertexStart = 0;
        int vertexEnd = 0;
        float rgbScale = 1.0f;
    };

    struct CommandBindingEvent
    {
        int commandIndex = -1;
        UIShaderMaterialBinding binding;
    };

    void TrackHDRColorRange(ScreenUIList list, int vertexStart, int vertexEnd, float rgbScale);
    std::vector<HDRColorRange> &GetHDRRanges(ScreenUIList list);
    const std::vector<HDRColorRange> &GetHDRRanges(ScreenUIList list) const;

    // Device
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    // Formats
    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    // Pipeline objects
    VkShaderModule m_vertShader = VK_NULL_HANDLE;
    VkShaderModule m_fragShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkShaderModule m_worldVertShader = VK_NULL_HANDLE;
    VkShaderModule m_worldFragShader = VK_NULL_HANDLE;
    VkPipelineLayout m_worldPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_worldPipeline = VK_NULL_HANDLE;
    VkPipeline m_worldTopPipeline = VK_NULL_HANDLE;
    struct WorldPipelineVariant
    {
        rhi::GraphicsRenderingSignature target;
        bool alwaysOnTop = false;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    std::vector<WorldPipelineVariant> m_worldPipelineVariants;
    struct MaterialPipelineKey
    {
        ShaderProgramKey key;
        ShaderProgramDomain domain = ShaderProgramDomain::ScreenUI;
        rhi::GraphicsRenderingSignature target{};
        bool alwaysOnTop = false;
        bool operator==(const MaterialPipelineKey &other) const noexcept
        {
            return key == other.key && domain == other.domain && target == other.target &&
                   alwaysOnTop == other.alwaysOnTop;
        }
    };
    struct MaterialPipelineKeyHash
    {
        size_t operator()(const MaterialPipelineKey &value) const noexcept;
    };
    std::unordered_map<MaterialPipelineKey, VkPipeline, MaterialPipelineKeyHash> m_materialPipelineVariants;
    struct ResolvedMaterialProgram
    {
        uint64_t generation = 0;
        uint64_t validatedRender = 0;
        std::shared_ptr<const ShaderProgramArtifact> artifact;
    };
    std::unordered_map<std::string, ResolvedMaterialProgram> m_resolvedMaterialPrograms;
    struct UIMaterialDescriptor;
    std::unordered_map<std::string, std::shared_ptr<UIMaterialDescriptor>> m_materialDescriptors;
    std::unordered_set<ShaderProgramKey, ShaderProgramKeyHash> m_ownedMaterialPrograms;
    std::function<std::shared_ptr<const ShaderProgramArtifact>(const std::string &, uint64_t, ShaderProgramDomain)>
        m_materialProgramResolver;
    std::function<bool(const std::string &, uint64_t)> m_materialIdentityValidator;
    std::function<void(const ShaderProgramKey &)> m_materialProgramRelease;
    std::function<void(const ShaderProgramKey &)> m_materialProgramAcquire;
    std::function<void()> m_materialProgramReleaseSweep;
    std::function<std::shared_ptr<const InxMaterial>(const std::string &, uint64_t)> m_materialAssetResolver;
    TextureResolver m_materialTextureResolver;
    std::function<uint64_t(const std::string &)> m_materialTextureGenerationResolver;
    uint64_t m_materialRenderSerial = 0;
    uint64_t m_materialBindingsRevision = 0;
    uint64_t m_prunedMaterialBindingsRevision = 0;

    // Font atlas descriptor (points to ImGui's font atlas)
    VkDescriptorSet m_fontDescriptorSet = VK_NULL_HANDLE;

    std::vector<std::array<ListBuffers, 3>> m_frameBuffers;
    std::array<uint64_t, 3> m_geometryRevision{1, 1, 1};
    std::array<uint64_t, 3> m_preparedRevision{};
    std::array<GeometryStats, 3> m_geometryStats{};
    std::array<std::vector<GPUVertex>, 2> m_screenVertices;
    // Immutable local positions used by the native screen-pose segment. The
    // GPU vertex array is rewritten every frame with the current pose.
    std::array<std::vector<ImVec2>, 2> m_screenLocalPositions;
    std::vector<WorldGPUVertex> m_worldVertices;

    // ImDrawList instances (standalone, not attached to any ImGui window)
    ImDrawList *m_cameraDrawList = nullptr;
    ImDrawList *m_overlayDrawList = nullptr;
    ImDrawList *m_worldDrawList = nullptr;
    std::array<ImDrawList *, 3> m_packetDrawLists{};
    std::array<std::vector<UIShaderMaterialBinding>, 3> m_commandBindings;
    std::shared_ptr<CommandPacket> m_recordingPacket;
    std::vector<HDRColorRange> m_cameraHDRRanges;
    std::vector<HDRColorRange> m_overlayHDRRanges;
    std::vector<HDRColorRange> m_worldHDRRanges;
    std::vector<WorldElementSpan> m_worldElementSpans;
    bool m_hasSelectiveWorldOcclusion = false;
    std::vector<ScreenElementSpan> m_screenElementSpans;
    int m_worldElementStart = -1;
    WorldElementSpan m_pendingWorldElement;

    bool m_initialized = false;
    bool m_enabled = true;
    bool m_commandCacheValid = false;
    bool m_reportedFirstDraw = false;
    uint32_t m_lastSubmittedDrawCounts[3]{};
    uint64_t m_lastSubmittedIndexCounts[3]{};
    uint32_t m_cachedWidth = 0;
    uint32_t m_cachedHeight = 0;
    uint64_t m_cachedContentRevision = 0;
    std::array<uint64_t, 3> m_cachedPacketEpoch{};
    GpuRetirementQueue *m_deletionQueue = nullptr;
    std::function<bool(uint64_t)> m_textureUsageValidator;
    std::function<std::shared_ptr<rhi::RenderTexture>(uint64_t)> m_renderTextureResolver;
    std::function<bool(uint64_t)> m_textureColorSpaceQuery;
};

} // namespace infernux
