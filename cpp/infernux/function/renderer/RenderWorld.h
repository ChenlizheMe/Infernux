#pragma once

#include <function/renderer/Frustum.h>
#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/RenderIdentity.h>
#include <function/renderer/RenderLightingSource.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

class SceneRenderExtractor;
class SceneRenderBridge;
enum class CameraClearFlags;
namespace rhi
{
class RenderTexture;
struct RenderTextureGeneration;
} // namespace rhi

/// Immutable structural values copied from a scene renderer at extraction.
/// No Scene, GameObject, Component, or Transform pointer may cross this boundary.
struct RenderProxyStructuralData
{
    uint64_t objectId = 0;
    uint32_t layerMask = 1u;
    bool isStatic = false;
    RenderProxyHandle identity;
    std::shared_ptr<InxMaterial> renderMaterial;
    int32_t renderQueue = 2000;
    uintptr_t materialSortKey = 0;
};

struct RenderProxyFrameData
{
    // CPU values serve CPU-authored poses and transform-dependent geometry.
    // For a device-only pose they are neutral placeholders, not observations.
    glm::mat4 worldMatrix{1.0f};
    glm::mat4 boundsWorldMatrix{1.0f};
    AABB worldBounds;
    RenderPoseBinding devicePose;
    bool visible = true;
};

struct RenderProxyCacheData
{
    size_t drawCallStart = 0;
    size_t drawCallCount = 0;
};

struct RenderProxy
{
    RenderProxyStructuralData structural;
    RenderProxyFrameData frame;
    RenderProxyCacheData cache;
};

struct RenderViewData
{
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::vec3 position{0.0f, 0.0f, 5.0f};
    glm::vec3 forward{0.0f, 0.0f, 1.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    uint32_t cullingMask = 0xFFFFFFFFu;
    uint64_t cameraId = 0;
    uint64_t worldId = 0;
    uint64_t completedWorldFrame = 0;
    uint64_t simulationEpoch = 0;
    uint64_t cameraObjectId = 0;
    uint64_t sceneWorldId = 0;
    std::string name;
    std::string sceneName;
    uint64_t temporalHistoryRevision = 0;
    float nearClip = 0.1f;
    float farClip = 1000.0f;
    float depth = 0.0f;
    glm::vec4 rect{0.0f, 0.0f, 1.0f, 1.0f};
    uint32_t pixelWidth = 0;
    uint32_t pixelHeight = 0;
    CameraClearFlags clearFlags = static_cast<CameraClearFlags>(0);
    glm::vec4 backgroundColor{0.0f, 0.0f, 0.0f, 1.0f};
    bool dithering = false;
    bool stopNaNs = false;
    bool invertCulling = false;
    bool editorView = false;
    bool isMainCamera = false;
    bool renderToTexture = false;
    bool renderOnDemand = false;
    bool wantsMemoryOutput = false;
    bool hasAuthoredTargetTexture = false;
    glm::uvec2 memoryOutputSize{0u, 0u};
    std::string outputTextureGuid;
    std::shared_ptr<rhi::RenderTexture> outputTexture;
    // Allocation/attachment lease only. Persistent output pixels intentionally
    // remain render-owned and may advance on every actual view execution.
    std::shared_ptr<const rhi::RenderTextureGeneration> outputGeneration;
    bool valid = false;
};

/// Immutable geometry, view, material and lighting values. Per-view graph
/// plans are published with this frame by SceneRenderBridge. UI and particle
/// device storage retain their separate publication boundaries.
/// The object is writable only
/// while owned by RenderWorldSnapshot::BeginFrame(); Publish() transfers it as
/// shared_ptr<const RenderWorldFrame>, so render consumers cannot mutate it.
class RenderWorldFrame final
{
  public:
    [[nodiscard]] const std::vector<RenderProxy> &Proxies() const noexcept
    {
        return m_proxies;
    }

    [[nodiscard]] const DrawCallResult &DrawCalls() const noexcept
    {
        return m_drawCalls;
    }

    [[nodiscard]] const RenderViewData &PrimaryView() const noexcept
    {
        return m_primaryView;
    }

    [[nodiscard]] const std::vector<RenderViewData> &Views() const noexcept
    {
        return m_views;
    }

    [[nodiscard]] const RenderViewData *FindView(uint64_t cameraId) const noexcept
    {
        const auto found = m_viewIndices.find(cameraId);
        return found == m_viewIndices.end() ? nullptr : &m_views[found->second];
    }

    [[nodiscard]] uint64_t CompletedWorldFrame() const noexcept
    {
        return m_completedWorldFrame;
    }

    [[nodiscard]] uint64_t SimulationEpoch() const noexcept
    {
        return m_simulationEpoch;
    }
    [[nodiscard]] const std::shared_ptr<const RenderLightingSource> &Lighting() const noexcept
    {
        return m_lighting;
    }

    [[nodiscard]] uint64_t WorldId() const noexcept
    {
        return m_worldId;
    }

    [[nodiscard]] uint64_t StructuralRevision() const noexcept
    {
        return m_structuralRevision;
    }

    [[nodiscard]] uint64_t TransformRevision() const noexcept
    {
        return m_transformRevision;
    }

    [[nodiscard]] uint64_t ContentRevision() const noexcept
    {
        return m_contentRevision;
    }

    [[nodiscard]] uint64_t PoseRevision() const noexcept
    {
        return m_poseRevision;
    }

    [[nodiscard]] uint64_t FrameRevision() const noexcept
    {
        return m_frameRevision;
    }

    [[nodiscard]] bool MatchesSource(uint64_t worldId, uint64_t structuralRevision) const noexcept
    {
        return m_worldId == worldId && m_structuralRevision == structuralRevision;
    }

  private:
    friend class RenderWorldSnapshot;
    friend class SceneRenderExtractor;

    std::vector<RenderProxy> m_proxies;
    DrawCallResult m_drawCalls;
    RenderViewData m_primaryView;
    std::vector<RenderViewData> m_views;
    std::unordered_map<uint64_t, size_t> m_viewIndices;
    uint64_t m_completedWorldFrame = 0;
    uint64_t m_simulationEpoch = 0;
    std::shared_ptr<const RenderLightingSource> m_lighting;
    uint64_t m_worldId = 0;
    uint64_t m_structuralRevision = 0;
    uint64_t m_transformRevision = 0;
    uint64_t m_contentRevision = 0;
    uint64_t m_sourceContentRevision = 0;
    uint64_t m_poseRevision = 0;
    uint64_t m_frameRevision = 0;
};

/// Atomic immutable publication boundary between scene extraction and render
/// consumption. Older frames remain valid while a consumer owns an acquired
/// shared pointer. Reusable frames are recycled only after all consumers drop
/// them, avoiding both frame-sized copies and mutation of published state.
class RenderWorldSnapshot final
{
  public:
    RenderWorldSnapshot() = default;
    RenderWorldSnapshot(const RenderWorldSnapshot &) = delete;
    RenderWorldSnapshot &operator=(const RenderWorldSnapshot &) = delete;

    [[nodiscard]] std::shared_ptr<const RenderWorldFrame> Acquire() const noexcept
    {
        return std::atomic_load_explicit(&m_published, std::memory_order_acquire);
    }

    void Clear()
    {
        RenderWorldFrame &frame = BeginFrame(0, 0);
        frame.m_proxies.clear();
        frame.m_drawCalls = {};
        frame.m_primaryView = {};
        frame.m_views.clear();
        frame.m_viewIndices.clear();
        frame.m_completedWorldFrame = 0;
        frame.m_simulationEpoch = 0;
        frame.m_lighting.reset();
        frame.m_worldId = 0;
        frame.m_structuralRevision = 0;
        frame.m_transformRevision = 0;
        frame.m_contentRevision = 0;
        frame.m_poseRevision = 0;
        Publish();
    }

    [[nodiscard]] uint64_t WorldId() const noexcept
    {
        const auto frame = Acquire();
        return frame ? frame->WorldId() : 0;
    }

    [[nodiscard]] uint64_t StructuralRevision() const noexcept
    {
        const auto frame = Acquire();
        return frame ? frame->StructuralRevision() : 0;
    }

    [[nodiscard]] uint64_t ContentRevision() const noexcept
    {
        const auto frame = Acquire();
        return frame ? frame->ContentRevision() : 0;
    }

    [[nodiscard]] uint64_t FrameRevision() const noexcept
    {
        const auto frame = Acquire();
        return frame ? frame->FrameRevision() : m_frameRevision;
    }

    [[nodiscard]] bool IsPublished() const noexcept
    {
        return static_cast<bool>(Acquire());
    }

    [[nodiscard]] bool MatchesSource(uint64_t worldId, uint64_t structuralRevision) const noexcept
    {
        const auto frame = Acquire();
        return frame && frame->MatchesSource(worldId, structuralRevision);
    }

  private:
    friend class SceneRenderExtractor;
    friend class SceneRenderBridge;

    // Renderer selection borrows a producer's immutable publication. Borrowed
    // frames must never enter this selector's writable recycle pool.
    void SelectPublication(std::shared_ptr<const RenderWorldFrame> frame)
    {
        std::atomic_store_explicit(&m_published, std::move(frame), std::memory_order_release);
    }

    void Reset()
    {
        SelectPublication(nullptr);
        m_writable.reset();
        m_recycled.clear();
    }

    [[nodiscard]] RenderWorldFrame &BeginFrame(uint64_t, uint64_t)
    {
        // A frame in the recycle list is writable only when this publisher is
        // its sole owner. Consumers may retain any older publication freely.
        for (auto it = m_recycled.begin(); it != m_recycled.end(); ++it) {
            if (it->use_count() != 1)
                continue;
            m_writable = std::move(*it);
            m_recycled.erase(it);
            break;
        }
        if (!m_writable)
            m_writable = std::make_shared<RenderWorldFrame>();

        m_writable->m_frameRevision = ++m_frameRevision;
        return *m_writable;
    }

    [[nodiscard]] RenderWorldFrame &WritableFrame() noexcept
    {
        return *m_writable;
    }

    void Publish()
    {
        if (!m_writable)
            return;
        std::shared_ptr<const RenderWorldFrame> next = std::move(m_writable);
        auto previous = std::atomic_exchange_explicit(&m_published, std::move(next), std::memory_order_acq_rel);
        if (previous)
            m_recycled.push_back(std::const_pointer_cast<RenderWorldFrame>(std::move(previous)));

        // Do not retain an unbounded number of frames when a slow consumer
        // intentionally keeps historical publications alive.
        if (m_recycled.size() > 8) {
            m_recycled.erase(m_recycled.begin(), m_recycled.begin() + static_cast<ptrdiff_t>(m_recycled.size() - 8));
        }
    }

    mutable std::shared_ptr<const RenderWorldFrame> m_published;
    std::shared_ptr<RenderWorldFrame> m_writable;
    std::vector<std::shared_ptr<RenderWorldFrame>> m_recycled;
    uint64_t m_frameRevision = 0;
};

} // namespace infernux
