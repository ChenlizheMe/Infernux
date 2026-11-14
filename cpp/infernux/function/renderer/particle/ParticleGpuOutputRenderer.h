#pragma once

#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/shader/ShaderViewBindings.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace infernux
{
struct ShaderCompilation;
}

namespace infernux::particle
{

using GpuParticleGraphicsPrograms = std::array<std::shared_ptr<const ShaderCompilation>, 3>;

enum class GpuParticleOutputType : uint8_t
{
    Sprite,
    Mesh,
    Ribbon,
};

enum class ParticleRibbonUvMode : uint8_t
{
    Stretch,
    Repeat,
};

/// Only per-draw values. Matrices and shadow parameters belong to the shared
/// camera/light binding, including picking, motion and depth passes.
struct alignas(16) GpuParticleViewConstants
{
    std::array<float, 4> cameraRight{};
    std::array<float, 4> cameraUp{};
    std::array<uint32_t, 4> objectData{};
    std::array<float, 4> depthReconstruct{};
    std::array<float, 4> lightingControl{};
    std::array<float, 4> renderingControl{};
    // xyz is the camera position or an authored world axis; w selects the
    // Sprite Output alignment mode.
    std::array<float, 4> alignmentReference{};
};

struct GpuParticleStaticBuffer
{
    rhi::BufferHandle buffer;
    uint64_t byteSize = 0;
};

class ParticleGpuOutputRenderer
{
  public:
    virtual ~ParticleGpuOutputRenderer() = default;

    [[nodiscard]] virtual bool IsValid() const noexcept = 0;
    [[nodiscard]] virtual int32_t RenderQueue() const noexcept = 0;
    [[nodiscard]] virtual bool RequiresSceneDepth() const noexcept = 0;
    [[nodiscard]] virtual bool CanCastShadows() const noexcept
    {
        return false;
    }
    [[nodiscard]] virtual uint32_t VertexCount() const noexcept = 0;
    [[nodiscard]] virtual rhi::BufferHandle InstanceBuffer() const noexcept = 0;
    [[nodiscard]] virtual rhi::BufferHandle RenderIndexBuffer() const noexcept = 0;
    // A view owns its cull/sort indices. Remove its cached geometry binding when
    // those indices retire; Device::Release keeps submitted descriptor leases alive.
    virtual void ReleaseViewBinding(rhi::BufferHandle renderIndices) noexcept = 0;
    virtual void ReleaseView(rhi::BufferHandle globals) noexcept = 0;
    [[nodiscard]] virtual size_t ViewBindingCount() const noexcept = 0;
    [[nodiscard]] virtual const std::vector<GpuParticleStaticBuffer> &StaticVertexStorageBuffers() const noexcept
    {
        static const std::vector<GpuParticleStaticBuffer> empty;
        return empty;
    }

    [[nodiscard]] virtual bool RecordDraw(const rhi::GraphicsCommandEncoder &encoder,
                                          const MaterialPassPipelineDescriptor &pass,
                                          rhi::BufferHandle indirectArguments, const GpuParticleViewConstants &view,
                                          const ShaderViewBindings &perView, rhi::BufferHandle renderIndices = {},
                                          rhi::TextureViewHandle sceneDepth = {}, bool sceneDepthIsDepth = true) = 0;
    [[nodiscard]] virtual bool
    RecordPickingDraw(const rhi::GraphicsCommandEncoder &encoder, const MaterialPassPipelineDescriptor &pass,
                      rhi::BufferHandle indirectArguments, const GpuParticleViewConstants &view, uint64_t ownerObjectId,
                      const ShaderViewBindings &perView, rhi::BufferHandle renderIndices = {}) = 0;
};

static_assert(sizeof(GpuParticleViewConstants) == 112);

} // namespace infernux::particle
