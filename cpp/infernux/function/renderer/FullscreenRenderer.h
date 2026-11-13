#pragma once

#include <function/renderer/rhi/RenderCoreApi.h>
#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/rhi/RhiDescriptors.h>
#include <function/renderer/shader/ShaderParameterValue.h>
#include <function/renderer/shader/ShaderProgramCode.h>
#include <function/renderer/shader/ShaderValueCopy.h>

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
class DeviceShaderProgram;

struct FullscreenProgram
{
    std::string assetGuid;
    std::string shaderId;
    std::shared_ptr<const ShaderTargetCode> code;
    std::string schema;

    [[nodiscard]] bool UsesViewBinding(uint32_t binding) const noexcept
    {
        for (const auto &resource : code->resources)
            if (resource.set == 1 && resource.binding.binding == binding)
                return true;
        return false;
    }
};

/// Logical overrides and their target copy plans, prepared with the graph.
/// Each frame only copies bytes; it does not look up fields or convert types.
struct FullscreenParameters
{
    struct History
    {
        uint32_t input;
        uint32_t offset;
    };

    std::vector<char> defaults;
    std::vector<ShaderValueCopy> copies;
    std::vector<ShaderParameterValue> values;
    std::vector<History> histories;
};

// Borrowed view resources. Their owner stages and retires the allocations;
// fullscreen programs share the geometry view's camera, lights and shadows.
struct FullscreenViewResources
{
    rhi::BufferBinding globals;
    rhi::BindGroupHandle perView;
};

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
    [[nodiscard]] virtual std::shared_ptr<const FullscreenProgram> AcquireProgram(const std::string &)
    {
        return {};
    }
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
    uint8_t colorWriteMask = 0x0f;
    uint32_t inputResourceCount = 0;
    uint32_t inputBufferMask = 0;
    uint32_t depthInputMask = 0;
    bool useDynamicRendering = false;
    std::shared_ptr<const FullscreenProgram> program;

    INFERNUX_RENDER_CORE_API bool operator==(const FullscreenPipelineKey &other) const noexcept;
};

struct FullscreenPipelineKeyHash
{
    INFERNUX_RENDER_CORE_API size_t operator()(const FullscreenPipelineKey &key) const noexcept;
};

struct FullscreenPipelineEntry
{
    static constexpr int32_t Parameters = -1;
    static constexpr int32_t Globals = -2;

    rhi::GraphicsPipelineHandle pipeline;
    rhi::BindingLayoutHandle inputLayout;
    bool hasPerView = false;
    bool hasGlobals = false;
    std::shared_ptr<DeviceShaderProgram> program;
    std::vector<std::string> inputNames;
    std::vector<uint32_t> resourceIndices;
    std::vector<int32_t> resourceInputs;
};

struct FullscreenResourceInput
{
    rhi::TextureViewHandle view;
    rhi::PixelFormat format = rhi::PixelFormat::Undefined;
    bool depthRead = false;
    bool historyValid = false;
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

    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<const FullscreenProgram>
    AcquireProgram(const std::string &name) const;

    [[nodiscard]] INFERNUX_RENDER_CORE_API bool HasShader(const std::string &name, rhi::ShaderStage stage) const;

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

    [[nodiscard]] INFERNUX_RENDER_CORE_API std::shared_ptr<const FullscreenProgram>
    AcquireProgram(const std::string &name) const;

    [[nodiscard]] INFERNUX_RENDER_CORE_API static FullscreenParameters
    PrepareParameters(const FullscreenProgram &program,
                      const std::vector<std::pair<std::string, ShaderParameterValue>> &parameters);

    INFERNUX_RENDER_CORE_API void
    DrawProgram(rhi::GraphicsCommandEncoder &encoder, const FullscreenPipelineEntry &entry,
                const FullscreenResourceInput *inputs, const FullscreenParameters &parameters,
                const ShaderParameterValue *values = nullptr, const FullscreenViewResources &view = {});

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
