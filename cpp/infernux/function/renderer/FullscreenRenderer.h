#pragma once

#include <function/renderer/rhi/RenderCoreApi.h>
#include <function/renderer/rhi/RhiCommand.h>

#include <cstdint>
#include <memory>
#include <string>

namespace infernux::rhi
{
class Device;
}

namespace infernux
{

class InxVkCoreModular;
class FullscreenPipelineCache;

class FullscreenRendererHost
{
  public:
    virtual ~FullscreenRendererHost() = default;

    [[nodiscard]] virtual rhi::Device &GetRhiDevice() noexcept = 0;
    [[nodiscard]] virtual uint32_t GetFrameCount() const noexcept = 0;
    [[nodiscard]] virtual uint32_t GetCurrentFrame() const noexcept = 0;

    /// Return a module registration owned by the caller. The renderer releases
    /// it after pipeline creation; the concrete backend decides whether the
    /// underlying shader object is shared or newly created.
    [[nodiscard]] virtual rhi::ShaderModuleHandle AcquireShaderModule(const std::string &name, rhi::ShaderStage stage,
                                                                      uint32_t inputCount,
                                                                      uint32_t inputBufferMask) = 0;
    [[nodiscard]] virtual rhi::BindingLayoutHandle GetPerViewLayout() const noexcept = 0;
    [[nodiscard]] virtual rhi::BindingLayoutHandle GetGlobalsLayout() const noexcept = 0;
    virtual void ReportError(const std::string &message)
    {
        (void)message;
    }
    virtual void ReportInfo(const std::string &message)
    {
        (void)message;
    }
};

struct FullscreenPushConstants
{
    float values[32] = {};
};

struct FullscreenPipelineKey
{
    std::string shaderName;
    rhi::RenderTargetLayoutHandle renderTargetLayout;
    rhi::SampleCount samples = rhi::SampleCount::One;
    rhi::PixelFormat colorFormat = rhi::PixelFormat::RGBA8UNorm;
    rhi::PixelFormat depthFormat = rhi::PixelFormat::Undefined;
    rhi::DepthState depth;
    bool alphaBlend = false;
    uint32_t inputResourceCount = 0;
    uint32_t inputBufferMask = 0;
    uint32_t depthInputMask = 0;
    bool useDynamicRendering = false;

    INFERNUX_RENDER_CORE_API bool operator==(const FullscreenPipelineKey &other) const noexcept;
};

struct FullscreenPipelineKeyHash
{
    INFERNUX_RENDER_CORE_API size_t operator()(const FullscreenPipelineKey &key) const noexcept;
};

struct FullscreenPipelineEntry
{
    rhi::GraphicsPipelineHandle pipeline;
    rhi::BindingLayoutHandle inputLayout;
    bool hasPerView = false;
    bool hasGlobals = false;
};

struct FullscreenResourceInput
{
    rhi::TextureViewHandle view;
    rhi::PixelFormat format = rhi::PixelFormat::Undefined;
    bool depthRead = false;
    rhi::BufferHandle buffer;
    uint64_t byteOffset = 0;
    uint64_t byteSize = 0;
    rhi::SamplerHandle sampler;
};

/// Immutable pipelines shared by views of one device and one host binding ABI.
/// Scene teardown releases its bind groups, while this bounded cache survives
/// to reuse the same compiled programs in the next scene. Eviction uses the
/// device's normal submission-based retirement.
class FullscreenPipelineCache
{
  public:
    INFERNUX_RENDER_CORE_API explicit FullscreenPipelineCache(std::shared_ptr<FullscreenRendererHost> host,
                                                              size_t capacity = 64);
    INFERNUX_RENDER_CORE_API ~FullscreenPipelineCache();
    FullscreenPipelineCache(const FullscreenPipelineCache &) = delete;
    FullscreenPipelineCache &operator=(const FullscreenPipelineCache &) = delete;

    INFERNUX_RENDER_CORE_API const FullscreenPipelineEntry &EnsurePipeline(const FullscreenPipelineKey &key);
    INFERNUX_RENDER_CORE_API void InvalidateShader(const std::string &shaderName);
    [[nodiscard]] INFERNUX_RENDER_CORE_API rhi::Device &GetDevice() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

class FullscreenRenderer
{
  public:
    INFERNUX_RENDER_CORE_API FullscreenRenderer();
    INFERNUX_RENDER_CORE_API ~FullscreenRenderer();

    FullscreenRenderer(const FullscreenRenderer &) = delete;
    FullscreenRenderer &operator=(const FullscreenRenderer &) = delete;

    INFERNUX_RENDER_CORE_API void Initialize(std::shared_ptr<FullscreenRendererHost> host,
                                             std::shared_ptr<FullscreenPipelineCache> pipelines = {});
    void Initialize(InxVkCoreModular *vkCore);
    INFERNUX_RENDER_CORE_API void Destroy();

    INFERNUX_RENDER_CORE_API const FullscreenPipelineEntry &EnsurePipeline(const FullscreenPipelineKey &key);

    /// Retire every cached pipeline compiled from @p shaderName. The next
    /// record resolves the newly published module and builds a fresh pipeline.
    INFERNUX_RENDER_CORE_API void InvalidateShader(const std::string &shaderName);

    INFERNUX_RENDER_CORE_API rhi::BindGroupHandle AllocateBindGroup(rhi::BindingLayoutHandle layout,
                                                                    const FullscreenResourceInput *inputs,
                                                                    uint32_t inputCount,
                                                                    rhi::SamplerHandle colorSampler);

    INFERNUX_RENDER_CORE_API void Draw(rhi::GraphicsCommandEncoder &encoder, const FullscreenPipelineEntry &entry,
                                       rhi::BindGroupHandle inputGroup, rhi::BindGroupHandle perViewGroup,
                                       rhi::BindGroupHandle globalsGroup, const FullscreenPushConstants &pushConstants,
                                       uint32_t pushConstantSize);

    INFERNUX_RENDER_CORE_API void ResetPool();

    [[nodiscard]] INFERNUX_RENDER_CORE_API rhi::SamplerHandle GetLinearSampler() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
