#pragma once

#include "SceneRenderer.h"

#include <functional>
#include <memory>
#include <vector>

namespace infernux
{

class Camera;
class MeshRenderer;
class SkinnedMeshRenderer;
class Transform;
class RuntimeWorld;

/// Scene-side producer for immutable RenderWorld publications. Scene pointers
/// are confined to this type and never cross into SceneRenderer consumers.
class SceneRenderExtractor final
{
  public:
    [[nodiscard]] size_t ExtractEditorFrame(RenderWorldSnapshot &world, bool useActiveCameraCulling);
    [[nodiscard]] size_t ExtractCameraFrame(RenderWorldSnapshot &world, Camera *camera);
    // Called only after the logical frame's Late/Transform/Skeleton boundary.
    [[nodiscard]] size_t ExtractRuntimeFrame(RenderWorldSnapshot &publication, RuntimeWorld &world);
    [[nodiscard]] static RenderViewData CaptureCameraView(const Camera &camera);

    using PrepareMaterial = std::function<void(const std::shared_ptr<InxMaterial> &)>;
    void SetMaterialPreparation(PrepareMaterial prepare)
    {
        m_prepareMaterial = std::move(prepare);
    }
    void SetAspectRatio(float aspect);

#if INFERNUX_FRAME_PROFILE
    [[nodiscard]] const SceneRendererProfileSnapshot &GetProfileSnapshot() const
    {
        return m_profileSnapshot;
    }
    void ResetProfileSnapshot()
    {
        m_profileSnapshot = {};
    }
#endif

  private:
    struct SceneRenderSource
    {
        MeshRenderer *meshRenderer = nullptr;
        Transform *transform = nullptr;
        SkinnedMeshRenderer *skinnedRenderer = nullptr;
        std::shared_ptr<const void> inlineMeshOwner;
        const std::vector<Vertex> *inlineVertices = nullptr;
        const std::vector<uint32_t> *inlineIndices = nullptr;
        uint64_t inlineMeshVersion = 0;
        std::shared_ptr<rhi::ComputeBuffer> vertexPublication;
    };

    void CollectRenderables(RenderWorldFrame &frame);
    RenderPoseBinding CaptureDevicePose(const MeshRenderer &renderer);
    void CapturePrimaryView(RenderWorldFrame &frame, Camera *camera);
    void CaptureViews(RenderWorldFrame &frame, Camera *primary, bool includeEditorView);
    bool CaptureMeshPublications(RenderWorldFrame &frame);
    void CaptureMaterialPublications(RenderWorldFrame &frame);
    void CaptureLighting(RenderWorldFrame &frame);
    size_t ExtractSourceFrame(RenderWorldSnapshot &publication, Camera *primary, uint64_t worldId,
                              uint64_t completedWorldFrame, uint64_t simulationEpoch, bool includeEditorView);
    void SortRenderables(RenderWorldFrame &frame);
    void UpdateCachedRenderableTransforms(RenderWorldFrame &frame);
    void EmitDrawCallsForRenderable(DrawCallResult &result, const RenderProxy &renderable, SceneRenderSource &source,
                                    bool visible, bool bufferDirty) const;
    void RebuildDrawCalls(RenderWorldFrame &frame);

    Camera *m_activeCamera = nullptr;
    std::vector<SceneRenderSource> m_sceneSources;
    size_t m_visibleCount = 0;
    bool m_allRenderersStatic = false;
    uint64_t m_poseRevision = 0;
    uint64_t m_sourceWorldId = 0;
    uint64_t m_sourceStructureRevision = 0;
    uint64_t m_sourceRoutingRevision = 0;
    uint64_t m_structurePublicationRevision = 0;
    uint64_t m_sourceContentRevision = 0;
    uint64_t m_contentPublicationRevision = 0;
    uint64_t m_lightingGeneration = 0;
    bool m_materialPublicationsChanged = false;
    std::unordered_map<const InxMaterial *, std::weak_ptr<InxMaterial>> m_lastMaterialPublications;
    struct ParameterPublication
    {
        std::weak_ptr<const RendererParameterBlock> source;
        std::shared_ptr<const RendererParameterBlock> frozen;
    };
    std::unordered_map<const RendererParameterBlock *, ParameterPublication> m_parameterPublications;
    std::shared_ptr<const RenderLightingSource> m_lighting;
    PrepareMaterial m_prepareMaterial;
    std::shared_ptr<InxMaterial> m_baseSkybox;
    std::unordered_map<const RenderPoseBuffer *, std::shared_ptr<const RenderPoseValue>> m_framePoseValues;
    std::unordered_map<const RenderPoseBuffer *, std::shared_ptr<const RenderPoseValue>> m_previousPoseValues;

#if INFERNUX_FRAME_PROFILE
    SceneRendererProfileSnapshot m_profileSnapshot;
#endif
};

} // namespace infernux
