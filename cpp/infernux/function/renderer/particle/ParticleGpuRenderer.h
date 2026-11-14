#pragma once

#include "ParticleGpuOutputRenderer.h"
#include "ParticleGpuRibbonTopology.h"
#include "ParticleGpuSurfaceBinding.h"

#include <array>
#include <memory>
#include <vector>

namespace infernux::particle
{

class ParticleGpuGraphicsProgram;

struct GpuParticleRendererDesc
{
    GpuParticleOutputType type = GpuParticleOutputType::Sprite;
    std::shared_ptr<const ShaderCompilation> geometry;
    std::shared_ptr<const ShaderProgramArtifact> surface;
    std::shared_ptr<InxMaterial> material;
    ParticleOutputSemantics semantics;
    GpuBillboardMaterialState materialState;
    rhi::BufferHandle instances;
    rhi::BufferHandle renderIndices;
    GpuParticleStaticBuffer meshVertices;
    GpuParticleStaticBuffer meshIndices;
    uint32_t indexCount = 0;
    std::shared_ptr<void> meshOwner;
    std::shared_ptr<ParticleGpuRibbonTopology> ribbon;
    ParticleRibbonUvMode ribbonUvMode = ParticleRibbonUvMode::Stretch;
    float ribbonUvScale = 1.0f;
    uint32_t flipbookColumns = 1, flipbookRows = 1;
    GpuBillboardTextureResolver textureResolver;
    GpuRetirementQueue *retirement = nullptr;
};

/// One device draw path, with output-specific geometry buffers and constants.
/// Simulation, culling and sorting continue to own their existing resources.
class ParticleGpuRenderer final : public ParticleGpuOutputRenderer
{
  public:
    ParticleGpuRenderer();
    ~ParticleGpuRenderer() override;
    [[nodiscard]] bool Create(rhi::Device &device, const GpuParticleRendererDesc &desc);
    void Destroy() noexcept;

    [[nodiscard]] bool IsValid() const noexcept override;
    [[nodiscard]] int32_t RenderQueue() const noexcept override;
    [[nodiscard]] bool RequiresSceneDepth() const noexcept override;
    [[nodiscard]] bool CanCastShadows() const noexcept override;
    [[nodiscard]] uint32_t VertexCount() const noexcept override;
    [[nodiscard]] rhi::BufferHandle InstanceBuffer() const noexcept override;
    [[nodiscard]] rhi::BufferHandle RenderIndexBuffer() const noexcept override;
    void ReleaseViewBinding(rhi::BufferHandle renderIndices) noexcept override;
    void ReleaseView(rhi::BufferHandle globals) noexcept override;
    [[nodiscard]] size_t ViewBindingCount() const noexcept override;
    [[nodiscard]] const std::vector<GpuParticleStaticBuffer> &StaticVertexStorageBuffers() const noexcept override;

    [[nodiscard]] bool RecordDraw(const rhi::GraphicsCommandEncoder &encoder,
                                  const MaterialPassPipelineDescriptor &pass, rhi::BufferHandle indirectArguments,
                                  const GpuParticleViewConstants &view, const ShaderViewBindings &perView,
                                  rhi::BufferHandle renderIndices = {}, rhi::TextureViewHandle sceneDepth = {},
                                  bool sceneDepthIsDepth = true) override;
    [[nodiscard]] bool RecordPickingDraw(const rhi::GraphicsCommandEncoder &encoder,
                                         const MaterialPassPipelineDescriptor &pass,
                                         rhi::BufferHandle indirectArguments, const GpuParticleViewConstants &view,
                                         uint64_t ownerObjectId, const ShaderViewBindings &perView,
                                         rhi::BufferHandle renderIndices = {}) override;

  private:
    struct BufferSlot
    {
        uint32_t binding;
        uint8_t source;
    };
    struct ViewGroup
    {
        rhi::BufferHandle indices;
        rhi::BufferHandle globals;
        rhi::TextureViewHandle sceneDepth;
        uint8_t depthKind;
        rhi::BindGroupHandle group;
    };
    [[nodiscard]] rhi::BindGroupHandle GeometryGroup(rhi::BufferHandle indices, rhi::BufferHandle globals,
                                                     rhi::TextureViewHandle sceneDepth, uint8_t depthKind);
    [[nodiscard]] GpuParticleViewConstants DrawConstants(const GpuParticleViewConstants &view, ShaderCompileTarget pass,
                                                         rhi::BufferHandle indices) const;

    rhi::Device *m_device = nullptr;
    std::unique_ptr<ParticleGpuGraphicsProgram> m_program;
    ParticleGpuSurfaceBinding m_surface;
    std::shared_ptr<const ShaderProgramArtifact> m_shaderProgram;
    std::shared_ptr<void> m_meshOwner;
    std::shared_ptr<ParticleGpuRibbonTopology> m_ribbon;
    GpuParticleOutputType m_type = GpuParticleOutputType::Sprite;
    ParticleOutputSemantics m_semantics;
    std::array<rhi::BufferHandle, 4> m_buffers{};
    std::vector<BufferSlot> m_bufferSlots;
    std::vector<GpuParticleStaticBuffer> m_staticBuffers;
    uint8_t m_viewIndexSlot = 1;
    uint32_t m_vertexCount = 6;
    uint32_t m_columns = 1, m_rows = 1;
    ParticleRibbonUvMode m_uvMode = ParticleRibbonUvMode::Stretch;
    float m_uvScale = 1.0f;
    std::array<rhi::BindingLayoutHandle, 3> m_geometryLayouts{};
    std::array<uint32_t, 2> m_depthBindings{};
    std::array<bool, 2> m_depthReads{};
    std::vector<ViewGroup> m_viewGroups;
};

} // namespace infernux::particle
