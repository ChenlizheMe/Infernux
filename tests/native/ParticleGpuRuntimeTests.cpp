#include "ParticleGraphicsFixtures.h"
#include "SlangShaderFixture.h"
#include <core/config/EngineConfig.h>
#include <core/types/ColorSpace.h>
#include <function/renderer/SceneDepthResolver.h>
#include <function/renderer/VkTextureCache.h>
#include <function/renderer/particle/ParticleGpuBounds.h>
#include <function/renderer/particle/ParticleGpuCollisionScene.h>
#include <function/renderer/particle/ParticleGpuCuller.h>
#include <function/renderer/particle/ParticleGpuDrawRegistry.h>
#include <function/renderer/particle/ParticleGpuMigrator.h>
#include <function/renderer/particle/ParticleGpuRenderer.h>
#include <function/renderer/particle/ParticleGpuRibbonTopology.h>
#include <function/renderer/particle/ParticleGpuRuntime.h>
#include <function/renderer/particle/ParticleGpuSorter.h>
#include <function/renderer/particle/ParticleRenderGraph.h>
#include <function/renderer/rhi/GpuRetirementQueue.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/renderer/vk/RenderGraph.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/resources/ShaderAsset/SlangNativeModule.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using namespace infernux;

particle::CompiledParticleShader CompileParticleFixture(std::string_view source,
                                                        ShaderCodeTarget target = ShaderCodeTarget::SpirV)
{
    return std::make_shared<const ShaderTargetCode>(
        CompileSlangNativeModule({"particle_fixture", "particle_fixture.slang", std::string(source)}, target));
}

bool CanRecordCompute(void *, rhi::ComputePipelineHandle pipeline, rhi::BindGroupHandle group)
{
    return pipeline.IsValid() && group.IsValid();
}

constexpr size_t RequiredGpuKernelStageCount = static_cast<size_t>(particle::GpuKernelStage::Count) - 1u;
constexpr size_t FusedGpuKernelStageCount = static_cast<size_t>(particle::GpuKernelStage::Count);

std::array<float, 4> ExpectedLinearTint(const glm::vec4 &authoredSrgb)
{
    const glm::vec4 linear = inx::color::SrgbToLinear(authoredSrgb);
    return {linear.x, linear.y, linear.z, linear.w};
}

float FloatFromBits(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool NearlyEqual(float left, float right)
{
    return std::abs(left - right) <= 1.0e-6f;
}

struct FakeDevice final : rhi::Device
{
    rhi::ShaderSourceLanguage GetShaderSourceLanguage() const noexcept override
    {
        return rhi::ShaderSourceLanguage::SpirV;
    }

    rhi::DeviceCaps capabilities;
    std::vector<rhi::BufferDesc> buffers;
    std::vector<rhi::TextureDesc> textures;
    std::vector<rhi::TextureViewDesc> textureViews;
    std::vector<rhi::SamplerDesc> samplers;
    uint32_t shaderCreates = 0;
    std::vector<rhi::BindingLayoutDesc> layouts;
    std::vector<rhi::BindGroupDesc> bindGroups;
    std::vector<uint32_t> layoutEntryCounts;
    std::vector<uint32_t> groupBufferCounts;
    std::vector<uint32_t> groupTextureCounts;
    std::vector<rhi::GraphicsPipelineDesc> graphicsPipelineDescs;
    std::vector<rhi::ComputePipelineDesc> computePipelineDescs;
    std::vector<std::vector<uint8_t>> initialBufferBytes;
    uint32_t layoutCreates = 0;
    uint32_t groupCreates = 0;
    uint32_t graphicsPipelineCreates = 0;
    uint32_t pipelineCreates = 0;
    uint32_t maxBindingLayouts = 8;
    uint32_t bufferReleases = 0;
    uint32_t textureReleases = 0;
    uint32_t samplerReleases = 0;
    uint32_t shaderReleases = 0;
    uint32_t layoutReleases = 0;
    uint32_t groupReleases = 0;
    uint32_t graphicsPipelineReleases = 0;
    uint32_t pipelineReleases = 0;
    uint32_t writes = 0;
    bool rejectNextWrite = false;
    bool rejectNextBuffer = false;
    bool rejectNextGroup = false;
    uint32_t readbacks = 0;
    bool bindlessEnabled = false;
    bool rejectNextBindlessPublication = false;
    uint32_t bindlessPublishes = 0;
    uint32_t bindlessMarks = 0;
    std::vector<rhi::ResourceIndex> markedBindlessResources;
    std::vector<std::vector<uint8_t>> writtenBytes;
    std::vector<uint64_t> writtenOffsets;
    std::vector<rhi::BufferHandle> writtenBuffers;
    uint32_t nextIndex = 1;

    const rhi::DeviceCaps &GetCapabilities() const noexcept override
    {
        return capabilities;
    }

    [[nodiscard]] rhi::BindlessTextureTableBinding GetBindlessTextureTableBinding() const noexcept override
    {
        return bindlessEnabled ? rhi::BindlessTextureTableBinding{{9000, 1}, {9001, 1}}
                               : rhi::BindlessTextureTableBinding{};
    }

    [[nodiscard]]
    rhi::ResourceIndex
    PublishBindlessTexture(const std::shared_ptr<const rhi::TextureGpuView> &texture) noexcept override
    {
        if (std::exchange(rejectNextBindlessPublication, false) || !bindlessEnabled || !texture || !texture->IsValid())
            return {};
        return {++bindlessPublishes, 1};
    }

    void MarkBindlessTexturesUsed(const rhi::ResourceIndex *resources, size_t count) noexcept override
    {
        if (!bindlessEnabled || (!resources && count != 0))
            return;
        ++bindlessMarks;
        markedBindlessResources.clear();
        if (count > 0)
            markedBindlessResources.assign(resources, resources + count);
    }

    rhi::BufferHandle CreateBuffer(const rhi::BufferDesc &desc) override
    {
        if (std::exchange(rejectNextBuffer, false))
            return {};
        buffers.push_back(desc);
        const auto *begin = static_cast<const uint8_t *>(desc.initialData);
        initialBufferBytes.emplace_back();
        if (begin && desc.initialDataBytes)
            initialBufferBytes.back().assign(begin, begin + desc.initialDataBytes);
        return {nextIndex++, 1};
    }

    rhi::TextureHandle CreateTexture(const rhi::TextureDesc &desc) override
    {
        textures.push_back(desc);
        return {nextIndex++, 1};
    }

    rhi::TextureViewHandle CreateTextureView(const rhi::TextureViewDesc &desc) override
    {
        textureViews.push_back(desc);
        return {nextIndex++, 1};
    }

    rhi::SamplerHandle CreateSampler(const rhi::SamplerDesc &desc) override
    {
        samplers.push_back(desc);
        return {nextIndex++, 1};
    }

    rhi::ShaderModuleHandle CreateShaderModule(const rhi::ShaderModuleDesc &desc) override
    {
        assert(desc.language == rhi::ShaderSourceLanguage::SpirV);
        assert(desc.code && desc.byteSize > 0);
        ++shaderCreates;
        return {nextIndex++, 1};
    }

    rhi::BindingLayoutHandle CreateBindingLayout(const rhi::BindingLayoutDesc &desc) override
    {
        layouts.push_back(desc);
        layoutEntryCounts.push_back(desc.entryCount);
        ++layoutCreates;
        return {nextIndex++, 1};
    }

    rhi::BindGroupHandle CreateBindGroup(const rhi::BindGroupDesc &desc) override
    {
        if (std::exchange(rejectNextGroup, false))
            return {};
        assert(desc.layout.IsValid());
        bindGroups.push_back(desc);
        groupBufferCounts.push_back(desc.bufferCount);
        groupTextureCounts.push_back(desc.textureCount);
        ++groupCreates;
        return {nextIndex++, 1};
    }

    rhi::ComputePipelineHandle CreateComputePipeline(const rhi::ComputePipelineDesc &desc) override
    {
        if (desc.bindingLayoutCount > maxBindingLayouts)
            return {};
        assert(desc.computeShader.IsValid() && desc.bindingLayoutCount > 0 &&
               desc.bindingLayoutCount <= rhi::ComputePipelineDesc::MaxBindingLayouts);
        computePipelineDescs.push_back(desc);
        ++pipelineCreates;
        return {nextIndex++, 1};
    }

    rhi::GraphicsPipelineHandle CreateGraphicsPipeline(const rhi::GraphicsPipelineDesc &desc) override
    {
        graphicsPipelineDescs.push_back(desc);
        ++graphicsPipelineCreates;
        return {nextIndex++, 1};
    }

    bool WriteBuffer(rhi::BufferHandle handle, uint64_t offset, const void *data, uint64_t byteSize) override
    {
        assert(handle.IsValid() && data && byteSize > 0);
        if (rejectNextWrite) {
            rejectNextWrite = false;
            return false;
        }
        const auto *begin = static_cast<const uint8_t *>(data);
        writtenBytes.emplace_back(begin, begin + byteSize);
        writtenOffsets.push_back(offset);
        writtenBuffers.push_back(handle);
        ++writes;
        return true;
    }

    bool ReadBuffer(rhi::BufferHandle, uint64_t, void *, uint64_t) override
    {
        ++readbacks;
        return false;
    }

    void Release(rhi::BufferHandle handle) noexcept override
    {
        bufferReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::TextureHandle handle) noexcept override
    {
        textureReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::TextureViewHandle handle) noexcept override
    {
        textureReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::SamplerHandle handle) noexcept override
    {
        samplerReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::ShaderModuleHandle handle) noexcept override
    {
        shaderReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::BindingLayoutHandle handle) noexcept override
    {
        layoutReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::BindGroupHandle handle) noexcept override
    {
        groupReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::GraphicsPipelineHandle handle) noexcept override
    {
        graphicsPipelineReleases += handle.IsValid() ? 1u : 0u;
    }
    void Release(rhi::ComputePipelineHandle handle) noexcept override
    {
        pipelineReleases += handle.IsValid() ? 1u : 0u;
    }
};

using TestTextureSlots = std::unordered_map<std::string, std::shared_ptr<rhi::TextureGpuViewSlot>>;

particle::GpuBillboardTextureLease AcquireTestTexture(FakeDevice &device, TestTextureSlots &slots,
                                                      const std::string &sourceId, uint64_t revision,
                                                      rhi::TextureViewHandle view, rhi::SamplerHandle sampler)
{
    const std::string canonicalSourceId = sourceId.empty() ? "white" : sourceId;
    auto &slot = slots[canonicalSourceId];
    if (!slot)
        slot = std::make_shared<rhi::TextureGpuViewSlot>(canonicalSourceId);
    auto publication = slot->Acquire();
    if (!publication || publication->GetRevision() != revision) {
        auto owner = std::shared_ptr<const void>(new uint32_t(static_cast<uint32_t>(revision)),
                                                 [&device, view, sampler](const void *value) {
                                                     device.Release(view);
                                                     device.Release(sampler);
                                                     delete static_cast<const uint32_t *>(value);
                                                 });
        auto next = std::make_shared<const rhi::TextureGpuView>(canonicalSourceId, revision, rhi::TextureHandle{}, view,
                                                                sampler, 1, std::move(owner));
        if (slot->TryPublish(next))
            publication = std::move(next);
        else
            publication = slot->Acquire();
    }
    return {particle::GpuBillboardTextureStatus::Ready, publication->GetView(), publication->GetSampler(), slot,
            std::move(publication)};
}

struct CommandTrace
{
    std::vector<rhi::ComputePipelineHandle> pipelines;
    std::vector<rhi::BindGroupHandle> groups;
    std::vector<uint32_t> groupSets;
    std::vector<particle::GpuParticlePushConstants> constants;
    std::vector<uint32_t> dispatches;
    std::vector<rhi::BufferHandle> indirectBuffers;
    std::vector<uint64_t> indirectOffsets;

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<CommandTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *context, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex < rhi::ComputePipelineDesc::MaxBindingLayouts && group.IsValid());
        auto &trace = *static_cast<CommandTrace *>(context);
        trace.groups.push_back(group);
        trace.groupSets.push_back(setIndex);
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == sizeof(particle::GpuParticlePushConstants));
        particle::GpuParticlePushConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<CommandTrace *>(context)->constants.push_back(value);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        assert(y == 1 && z == 1);
        static_cast<CommandTrace *>(context)->dispatches.push_back(x);
    }
    static void DispatchIndirect(void *context, rhi::BufferHandle buffer, uint64_t offset)
    {
        auto &trace = *static_cast<CommandTrace *>(context);
        trace.indirectBuffers.push_back(buffer);
        trace.indirectOffsets.push_back(offset);
    }
};

struct ContinuationTrace
{
    std::vector<rhi::ComputePipelineHandle> pipelines;
    std::vector<rhi::BindGroupHandle> groups;
    std::vector<uint32_t> groupSets;
    std::vector<particle::GpuParticleContinuationConstants> constants;
    std::vector<std::array<uint32_t, 3>> dispatches;
    std::vector<rhi::BufferHandle> indirectBuffers;
    std::vector<uint64_t> indirectOffsets;

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<ContinuationTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *context, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex < rhi::ComputePipelineDesc::MaxBindingLayouts && group.IsValid());
        auto &trace = *static_cast<ContinuationTrace *>(context);
        trace.groups.push_back(group);
        trace.groupSets.push_back(setIndex);
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == sizeof(particle::GpuParticleContinuationConstants));
        particle::GpuParticleContinuationConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<ContinuationTrace *>(context)->constants.push_back(value);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        static_cast<ContinuationTrace *>(context)->dispatches.push_back({x, y, z});
    }
    static void DispatchIndirect(void *context, rhi::BufferHandle buffer, uint64_t offset)
    {
        auto &trace = *static_cast<ContinuationTrace *>(context);
        trace.indirectBuffers.push_back(buffer);
        trace.indirectOffsets.push_back(offset);
    }
};

struct DepthResolveTrace
{
    rhi::ComputePipelineHandle pipeline;
    rhi::BindGroupHandle group;
    std::array<uint32_t, 4> constants{};
    std::array<uint32_t, 3> dispatch{};

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<DepthResolveTrace *>(context)->pipeline = pipeline;
    }
    static void BindGroup(void *context, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex == 0);
        static_cast<DepthResolveTrace *>(context)->group = group;
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == 16);
        std::memcpy(static_cast<DepthResolveTrace *>(context)->constants.data(), data, byteSize);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        static_cast<DepthResolveTrace *>(context)->dispatch = {x, y, z};
    }
    static void DispatchIndirect(void *, rhi::BufferHandle, uint64_t)
    {
    }
};

struct EventTransferTrace
{
    struct Copy
    {
        rhi::BufferHandle source;
        rhi::BufferHandle destination;
        rhi::BufferCopyRegion region;
    };
    std::vector<Copy> copies;
    struct Update
    {
        rhi::BufferHandle destination;
        uint64_t offset;
        std::vector<uint8_t> bytes;
    };
    std::vector<Update> updates;
    size_t rejectUpdateAt = std::numeric_limits<size_t>::max();

    static bool UpdateBuffer(void *context, rhi::BufferHandle destination, uint64_t offset, const void *data,
                             uint64_t byteSize)
    {
        auto &trace = *static_cast<EventTransferTrace *>(context);
        if (trace.updates.size() == trace.rejectUpdateAt)
            return false;
        const auto *bytes = static_cast<const uint8_t *>(data);
        trace.updates.push_back({destination, offset, {bytes, bytes + byteSize}});
        return true;
    }

    static void CopyBuffer(void *context, rhi::BufferHandle source, rhi::BufferHandle destination,
                           const rhi::BufferCopyRegion &region)
    {
        static_cast<EventTransferTrace *>(context)->copies.push_back({source, destination, region});
    }
    static void CopyTexture(void *, rhi::TextureHandle, rhi::TextureHandle, const rhi::TextureCopyRegion &)
    {
    }
    static void ResolveTexture(void *, rhi::TextureHandle, rhi::TextureHandle, const rhi::TextureResolveRegion &)
    {
    }
};

struct GraphicsTrace
{
    static void BindVertexBuffer(void *, uint32_t, rhi::BufferHandle, uint64_t, uint64_t)
    {
    }
    static void BindIndexBuffer(void *, rhi::BufferHandle, rhi::IndexFormat, uint64_t, uint64_t)
    {
    }
    static void DrawIndexed(void *, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
    {
    }
    static void DrawIndexedIndirect(void *, rhi::BufferHandle, uint64_t, uint32_t, uint32_t)
    {
    }

    static rhi::GraphicsCommandEncoder::Dispatch Commands()
    {
        return {BindPipeline,    BindGroup,   PushConstants,      Draw, DrawIndirect, BindVertexBuffer,
                BindIndexBuffer, DrawIndexed, DrawIndexedIndirect};
    }

    std::vector<rhi::GraphicsPipelineHandle> pipelines;
    std::vector<rhi::BindGroupHandle> groups;
    std::vector<uint32_t> groupSets;
    std::vector<particle::GpuParticleViewConstants> constants;
    std::vector<rhi::BufferHandle> indirectBuffers;

    static void BindPipeline(void *context, rhi::GraphicsPipelineHandle pipeline)
    {
        static_cast<GraphicsTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *context, rhi::GraphicsPipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex <= 3);
        auto &trace = *static_cast<GraphicsTrace *>(context);
        trace.groups.push_back(group);
        trace.groupSets.push_back(setIndex);
    }
    static void PushConstants(void *context, rhi::GraphicsPipelineHandle, rhi::ShaderStage stages, uint32_t byteSize,
                              const void *data)
    {
        assert(stages == (rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment) &&
               byteSize == sizeof(particle::GpuParticleViewConstants));
        particle::GpuParticleViewConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<GraphicsTrace *>(context)->constants.push_back(value);
    }
    static void Draw(void *, uint32_t, uint32_t, uint32_t, uint32_t)
    {
    }
    static void DrawIndirect(void *context, rhi::BufferHandle buffer, uint64_t offset, uint32_t drawCount,
                             uint32_t stride)
    {
        assert(offset == 0 && drawCount == 1 && stride == 16);
        static_cast<GraphicsTrace *>(context)->indirectBuffers.push_back(buffer);
    }
};

struct MigrationTrace
{
    std::vector<rhi::ComputePipelineHandle> pipelines;
    std::vector<particle::GpuParticleMigrationConstants> constants;
    std::vector<uint32_t> dispatches;

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<MigrationTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex == 0 && group.IsValid());
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == sizeof(particle::GpuParticleMigrationConstants));
        particle::GpuParticleMigrationConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<MigrationTrace *>(context)->constants.push_back(value);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        assert(y == 1 && z == 1);
        static_cast<MigrationTrace *>(context)->dispatches.push_back(x);
    }
    static void DispatchIndirect(void *, rhi::BufferHandle, uint64_t)
    {
    }
};

struct SortTrace
{
    std::vector<rhi::ComputePipelineHandle> pipelines;
    std::vector<rhi::BindGroupHandle> groups;
    std::vector<particle::GpuParticleSortConstants> constants;
    std::vector<uint32_t> dispatches;
    std::vector<rhi::BufferHandle> indirectDispatches;

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<SortTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *context, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex == 0);
        static_cast<SortTrace *>(context)->groups.push_back(group);
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == sizeof(particle::GpuParticleSortConstants));
        particle::GpuParticleSortConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<SortTrace *>(context)->constants.push_back(value);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        assert(y == 1 && z == 1);
        static_cast<SortTrace *>(context)->dispatches.push_back(x);
    }
    static void DispatchIndirect(void *context, rhi::BufferHandle buffer, uint64_t offset)
    {
        assert(offset == 0);
        static_cast<SortTrace *>(context)->indirectDispatches.push_back(buffer);
    }
};

struct RibbonTrace
{
    std::vector<rhi::ComputePipelineHandle> pipelines;
    std::vector<rhi::BindGroupHandle> groups;
    std::vector<particle::GpuParticleRibbonConstants> constants;
    std::vector<uint32_t> dispatches;
    std::vector<rhi::BufferHandle> indirectDispatches;

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<RibbonTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *context, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex == 0 && group.IsValid());
        static_cast<RibbonTrace *>(context)->groups.push_back(group);
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == sizeof(particle::GpuParticleRibbonConstants));
        particle::GpuParticleRibbonConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<RibbonTrace *>(context)->constants.push_back(value);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        assert(y == 1 && z == 1);
        static_cast<RibbonTrace *>(context)->dispatches.push_back(x);
    }
    static void DispatchIndirect(void *context, rhi::BufferHandle buffer, uint64_t offset)
    {
        assert(offset == 0 && buffer.IsValid());
        static_cast<RibbonTrace *>(context)->indirectDispatches.push_back(buffer);
    }
};

struct CullTrace
{
    std::vector<rhi::ComputePipelineHandle> pipelines;
    std::vector<rhi::BindGroupHandle> groups;
    std::vector<particle::GpuParticleCullConstants> constants;
    std::vector<uint32_t> dispatches;
    std::vector<rhi::BufferHandle> indirectDispatches;

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<CullTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *context, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex == 0);
        static_cast<CullTrace *>(context)->groups.push_back(group);
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == sizeof(particle::GpuParticleCullConstants));
        particle::GpuParticleCullConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<CullTrace *>(context)->constants.push_back(value);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        assert(y == 1 && z == 1);
        static_cast<CullTrace *>(context)->dispatches.push_back(x);
    }
    static void DispatchIndirect(void *context, rhi::BufferHandle buffer, uint64_t offset)
    {
        assert(offset == 0);
        static_cast<CullTrace *>(context)->indirectDispatches.push_back(buffer);
    }
};

struct BoundsTrace
{
    std::vector<rhi::ComputePipelineHandle> pipelines;
    std::vector<rhi::BindGroupHandle> groups;
    std::vector<particle::GpuParticleBoundsConstants> constants;
    std::vector<uint32_t> dispatches;
    std::vector<rhi::BufferHandle> indirectDispatches;

    static void BindPipeline(void *context, rhi::ComputePipelineHandle pipeline)
    {
        static_cast<BoundsTrace *>(context)->pipelines.push_back(pipeline);
    }
    static void BindGroup(void *context, rhi::ComputePipelineHandle, uint32_t setIndex, rhi::BindGroupHandle group)
    {
        assert(setIndex == 0);
        static_cast<BoundsTrace *>(context)->groups.push_back(group);
    }
    static void PushConstants(void *context, rhi::ComputePipelineHandle, uint32_t byteSize, const void *data)
    {
        assert(byteSize == sizeof(particle::GpuParticleBoundsConstants));
        particle::GpuParticleBoundsConstants value;
        std::memcpy(&value, data, sizeof(value));
        static_cast<BoundsTrace *>(context)->constants.push_back(value);
    }
    static void Dispatch(void *context, uint32_t x, uint32_t y, uint32_t z)
    {
        assert(y == 1 && z == 1);
        static_cast<BoundsTrace *>(context)->dispatches.push_back(x);
    }
    static void DispatchIndirect(void *context, rhi::BufferHandle buffer, uint64_t offset)
    {
        assert(offset == 0);
        static_cast<BoundsTrace *>(context)->indirectDispatches.push_back(buffer);
    }
};

static void VerifyParticleGraphics()
{
    {
        FakeDevice device;
        particle::GpuParticleRendererDesc desc;
        desc.geometry = particle_test::GeometryPrograms()[0];
        auto incomplete = std::make_shared<ShaderProgramArtifact>(*particle_test::Surface());
        incomplete->compilation.reset();
        assert(incomplete->IsValid());
        desc.surface = incomplete;
        desc.semantics.softParticles = true;
        particle::ParticleGpuRenderer renderer;
        assert(!renderer.Create(device, desc));
        assert(device.buffers.empty() && device.groupCreates == 0 && device.layoutCreates == 0);

        desc.surface = particle_test::CreateSurface(false, true);
        assert(renderer.Create(device, desc));
        assert(!renderer.RequiresSceneDepth());
        particle_test::ViewResources view(device);
        GraphicsTrace trace;
        const auto commands = GraphicsTrace::Commands();
        const rhi::GraphicsCommandEncoder encoder(&trace, &commands);
        MaterialPassPipelineDescriptor pass;
        pass.colorFormats = {rhi::PixelFormat::RGBA8UNorm};
        assert(renderer.RecordDraw(encoder, pass, {71, 1}, {}, view.bindings));
        assert(device.bindGroups.back().textureCount == 0);
    }
    for (size_t kind = 0; kind < 3; ++kind) {
        FakeDevice device;
        TestTextureSlots textureSlots;
        std::vector<rhi::BufferHandle> buffers;
        const auto storage = [&](uint64_t bytes, bool indirect = false) {
            const auto buffer =
                device.CreateBuffer({bytes, indirect ? rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect
                                                     : rhi::BufferUsageFlags::Storage});
            buffers.push_back(buffer);
            return buffer;
        };
        {
            particle_test::ViewResources view(device);
            particle::GpuParticleRendererDesc desc;
            desc.type = static_cast<particle::GpuParticleOutputType>(kind);
            desc.geometry = particle_test::GeometryPrograms()[kind];
            desc.surface = particle_test::Surface(true);
            desc.instances = storage(64 * sizeof(particle::GpuParticleRenderInstance));
            desc.renderIndices = storage(64 * sizeof(uint32_t));
            const auto indirect = storage(16, true);
            desc.material = std::make_shared<InxMaterial>("Particle graphics");
            desc.material->SetVector4("color", glm::vec4(0));
            desc.semantics.softParticles = true;
            const auto texture = device.CreateTextureView({});
            const auto sampler = device.CreateSampler({});
            desc.textureResolver = [&](const auto &guid, const auto &, auto) {
                return AcquireTestTexture(device, textureSlots, guid, 1, texture, sampler);
            };
            if (desc.type == particle::GpuParticleOutputType::Mesh) {
                desc.meshVertices = {storage(3 * 80), 3 * 80};
                desc.meshIndices = {storage(3 * sizeof(uint32_t)), 3 * sizeof(uint32_t)};
                desc.indexCount = 3;
            } else if (desc.type == particle::GpuParticleOutputType::Ribbon) {
                particle::GpuParticleRibbonDesc topology;
                topology.capacity = 64;
                topology.instances = desc.instances;
                topology.sourceIndices = desc.renderIndices;
                topology.sourceIndirectArguments = indirect;
                topology.simulationControl = storage(16);
                topology.program = {CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Reset()),
                                    CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Initialize()),
                                    CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Histogram()),
                                    CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Scan()),
                                    CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Scatter())};
                desc.ribbon = std::make_shared<particle::ParticleGpuRibbonTopology>();
                assert(desc.ribbon->Create(device, topology));
            }
            particle::ParticleGpuRenderer renderer;
            assert(renderer.Create(device, desc));
            GraphicsTrace trace;
            const auto commands = GraphicsTrace::Commands();
            const rhi::GraphicsCommandEncoder encoder(&trace, &commands);
            particle::GpuParticleViewConstants values;
            values.cameraRight = {1, 0, 0, 0};
            values.cameraUp = {0, 1, 0, 0};
            values.objectData[2] = 0x10;
            auto bindings = view.bindings;
            const auto draw = [&](ShaderCompileTarget target, rhi::BufferHandle indices = {}, bool depthPlane = true) {
                MaterialPassPipelineDescriptor pass;
                pass.target = target;
                pass.depthFormat = rhi::PixelFormat::D32SFloat;
                if (target != ShaderCompileTarget::Shadow && target != ShaderCompileTarget::Depth)
                    pass.colorFormats = {target == ShaderCompileTarget::Picking  ? rhi::PixelFormat::RG32UInt
                                         : target == ShaderCompileTarget::Motion ? rhi::PixelFormat::RG16SFloat
                                                                                 : rhi::PixelFormat::RGBA8UNorm};
                if (target == ShaderCompileTarget::Picking)
                    return renderer.RecordPickingDraw(encoder, pass, indirect, values, 0x123456789abcdef0ull, bindings,
                                                      indices);
                return renderer.RecordDraw(encoder, pass, indirect, values, bindings, indices,
                                           target == ShaderCompileTarget::Forward ||
                                                   target == ShaderCompileTarget::ForwardPlus
                                               ? texture
                                               : rhi::TextureViewHandle{},
                                           depthPlane);
            };
            for (const auto pass :
                 {ShaderCompileTarget::Forward, ShaderCompileTarget::ForwardPlus, ShaderCompileTarget::Picking,
                  ShaderCompileTarget::Motion, ShaderCompileTarget::Depth, ShaderCompileTarget::Shadow})
                assert(draw(pass));
            assert(draw(ShaderCompileTarget::Forward, {}, false));
            assert(trace.constants[2].objectData[0] == 0x9abcdef0u && trace.constants[2].objectData[1] == 0x12345678u &&
                   trace.constants[2].objectData[2] == 0x10u);
            assert(trace.groupSets ==
                   std::vector<uint32_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}));
            const auto pipelines = device.graphicsPipelineCreates;
            const auto oldMaterial = trace.groups[trace.groups.size() - 3];
            const auto oldGeometry = trace.groups.back();
            desc.material->SetVector4("color", glm::vec4(0.25f));
            assert(draw(ShaderCompileTarget::Forward, {}, false));
            assert(device.graphicsPipelineCreates == pipelines && trace.groups.back() == oldGeometry &&
                   trace.groups[trace.groups.size() - 3] != oldMaterial);
            const auto persistent = storage(64 * sizeof(uint32_t));
            assert(draw(ShaderCompileTarget::Forward, persistent));
            const auto baseline = renderer.ViewBindingCount();
            for (unsigned index = 0; index < 16; ++index) {
                const auto transient = storage(64 * sizeof(uint32_t));
                assert(draw(ShaderCompileTarget::Forward, transient));
                assert(renderer.ViewBindingCount() == baseline + 1);
                renderer.ReleaseViewBinding(transient);
                assert(renderer.ViewBindingCount() == baseline);
            }
            renderer.ReleaseViewBinding(persistent);
            const auto baselinePipelines = device.graphicsPipelineCreates - device.graphicsPipelineReleases;
            const auto baselineBindings = renderer.ViewBindingCount();
            for (unsigned index = 0; index < 16; ++index) {
                particle_test::ViewResources transientView(device);
                bindings = transientView.bindings;
                assert(draw(ShaderCompileTarget::Shadow));
                assert(draw(ShaderCompileTarget::Forward));
                assert(renderer.ViewBindingCount() == baselineBindings + 2);
                renderer.ReleaseView(bindings.globals);
                assert(renderer.ViewBindingCount() == baselineBindings);
                assert(device.graphicsPipelineCreates - device.graphicsPipelineReleases == baselinePipelines);
            }
            bindings = view.bindings;
            renderer.ReleaseView(bindings.globals);
            assert(renderer.ViewBindingCount() == 0);
            assert(device.graphicsPipelineCreates == device.graphicsPipelineReleases);
            renderer.Destroy();
        }
        textureSlots.clear();
        for (const auto buffer : buffers)
            device.Release(buffer);
        assert(device.buffers.size() == device.bufferReleases);
        assert(device.shaderCreates == device.shaderReleases);
        assert(device.layoutCreates == device.layoutReleases);
        assert(device.groupCreates == device.groupReleases);
        assert(device.graphicsPipelineCreates == device.graphicsPipelineReleases);
        assert(device.pipelineCreates == device.pipelineReleases);
        assert(device.textureViews.size() == device.textureReleases &&
               device.samplers.size() == device.samplerReleases);
    }
    std::cout
        << "Particle graphics: real compiled stages, typed depth, shared passes and bounded view ownership passed\n";
}

} // namespace

int main()
{
    VerifyParticleGraphics();
    {
        FakeDevice device;
        device.capabilities.limits.minStorageBufferOffsetAlignment = 256;
        device.capabilities.portable.maxStorageBufferBinding = 1u << 30;
        const std::array<uint32_t, 5> code = {0x07230203u, 0u, 0u, 0u, 0u};
        const particle::GpuParticleSpawnProgram program{
            CompileParticleFixture(particle::GpuParticleSpawnShaderSources::Advance()),
            CompileParticleFixture(particle::GpuParticleSpawnShaderSources::Prepare())};
        particle::ParticleGpuGraphSpawnDomain domain;
        assert(domain.Create(device, 871, 3, program, std::vector<uint32_t>(16, 7)));
        EventTransferTrace trace;
        const rhi::TransferCommandEncoder::DispatchTable dispatch{
            &EventTransferTrace::CopyBuffer,     &EventTransferTrace::CopyTexture,
            &EventTransferTrace::ResolveTexture, nullptr,
            &EventTransferTrace::UpdateBuffer,   nullptr};
        const rhi::TransferCommandEncoder transfer(&trace, &dispatch);
        assert(domain.HasPendingUploads() && domain.RecordPendingUploads(transfer));
        assert(trace.updates.size() == 4 && device.writes == 0);
        for (const auto &buffer : device.buffers)
            assert(buffer.memory == rhi::BufferMemory::DeviceLocal && buffer.initialData == nullptr);
        const auto parameters = trace.updates.back().destination;
        assert(domain.RecordPendingUploads(transfer) && trace.updates.size() == 4);
        domain.NotifySubmission(false);
        assert(domain.HasPendingUploads());
        trace.updates.clear();
        assert(domain.RecordPendingUploads(transfer) && trace.updates.size() == 4);
        domain.NotifySubmission(true);
        assert(!domain.HasPendingUploads());

        // Validate the complete patch list before publishing any CPU range.
        assert(!domain.UpdateParameters({{0, {2, 2, 2, 2}}, {16, {3, 3, 3, 3}}}));
        assert(!domain.UpdateParameters({{1, {2, 2, 2, 2}}}));
        assert(!domain.UpdateParameters({{0, {2, 2}}}));
        assert(!domain.UpdateParameters({{4, {2, 2, 2, 2}}, {0, {3, 3, 3, 3}}}));
        assert(!domain.UpdateParameters({{0, {2, 2, 2, 2}}, {0, {3, 3, 3, 3}}}));
        assert(!domain.HasPendingUploads());
        assert(domain.UpdateParameters({{0, {2, 2, 2, 2}}, {12, {3, 3, 3, 3}}}));
        trace.updates.clear();
        trace.rejectUpdateAt = 1;
        assert(!domain.RecordPendingUploads(transfer));
        domain.NotifySubmission(false);
        trace.updates.clear();
        trace.rejectUpdateAt = std::numeric_limits<size_t>::max();
        assert(domain.RecordPendingUploads(transfer) && trace.updates.size() == 2);
        assert(trace.updates[0].destination == parameters && trace.updates[0].offset == 0 &&
               trace.updates[0].bytes.size() == 16 && trace.updates[1].offset == 48);
        // A newer CPU edit cannot be acknowledged by an earlier recording.
        assert(domain.UpdateParameters({{0, {4, 4, 4, 4}}}));
        domain.NotifySubmission(true);
        assert(domain.HasPendingUploads());
        trace.updates.clear();
        assert(domain.RecordPendingUploads(transfer) && trace.updates.size() == 1);
        uint32_t value = 0;
        std::memcpy(&value, trace.updates[0].bytes.data(), sizeof(value));
        assert(value == 4);
        domain.NotifySubmission(true);
        assert(!domain.HasPendingUploads());
        // Same-value author commands must still override intervening GPU edits.
        assert(domain.UpdateParameters({{0, {4, 4, 4, 4}}}) && domain.HasPendingUploads());
        assert(domain.SetEmitterPlaying(1, false));
        assert(domain.SetEmitterAcceptingBurstRequests(0, true));
        assert(domain.SetEmitterAcceptingBurstRequests(1, true));
        trace.updates.clear();
        assert(domain.RecordPendingUploads(transfer) && trace.updates.size() == 3);
        assert(trace.updates[0].destination == domain.BurstRequestAcceptanceBuffer() &&
               trace.updates[0].bytes.size() == 8);
        assert(trace.updates[1].destination == domain.EmitterPlayingRequestBuffer() && trace.updates[1].offset == 4);
        std::memcpy(&value, trace.updates[1].bytes.data(), sizeof(value));
        assert(value == 1);
        domain.NotifySubmission(true);
        assert(domain.SetEmitterAcceptingBurstRequests(1, true) && !domain.HasPendingUploads());
        assert(!domain.SetEmitterPlaying(3, true) && !domain.SetEmitterAcceptingBurstRequests(3, true));
        assert(domain.SetEmitterPlaying(1, false) && domain.HasPendingUploads());
        assert(device.writes == 0);
    }
    {
        FakeDevice continuationDevice;
        const auto ownerLayout = continuationDevice.CreateBindingLayout({});
        rhi::BindGroupDesc ownerGroupDesc;
        ownerGroupDesc.layout = ownerLayout;
        const auto ownerGroup = continuationDevice.CreateBindGroup(ownerGroupDesc);
        const auto emptyLayout = continuationDevice.CreateBindingLayout({});
        rhi::BindGroupDesc emptyGroupDesc;
        emptyGroupDesc.layout = emptyLayout;
        const auto emptyGroup = continuationDevice.CreateBindGroup(emptyGroupDesc);
        std::array<uint32_t, 5> continuationShader = {0x07230203u, 0u, 0u, 0u, 0u};
        particle::GpuParticleContinuationDesc continuationDesc;
        continuationDesc.capacity = 513;
        continuationDesc.particleCapacity = 257;
        continuationDesc.recordStride = 96;
        continuationDesc.laneCount = 5;
        continuationDesc.joinCount = 2;
        continuationDesc.initialProgramGeneration = 41;
        continuationDesc.ownerLayout = ownerLayout;
        continuationDesc.ownerGroup = ownerGroup;
        continuationDesc.dataInterfaceLayout = emptyLayout;
        continuationDesc.dataInterfaceGroup = emptyGroup;
        continuationDesc.vectorFieldLayout = emptyLayout;
        continuationDesc.vectorFieldGroup = emptyGroup;
        continuationDesc.graphSpawnLayout = {0xffe0u, 1};
        continuationDesc.emptyLayout = emptyLayout;
        continuationDesc.emptyGroup = emptyGroup;
        continuationDesc.collisionSceneLayout = emptyLayout;
        continuationDesc.collisionSceneGroup = emptyGroup;
        continuationDesc.contactLayout = emptyLayout;
        continuationDesc.contactGroup = emptyGroup;
        continuationDesc.program = {
            {continuationShader.data(), continuationShader.size()},
            {continuationShader.data(), continuationShader.size()},
            {continuationShader.data(), continuationShader.size()},
        };

        auto current = std::make_shared<particle::ParticleGpuContinuationRuntime>();
        assert(current->Create(continuationDevice, continuationDesc));
        assert(current->IsValid() && current->Capacity() == 513 && current->ParticleCapacity() == 257 &&
               current->ProgramGeneration() == 41 && current->ResetSerial() == 1 && current->RecordStride() == 96 &&
               current->LaneCount() == 5 && current->JoinCount() == 2);
        assert(continuationDevice.buffers.size() == 10);
        assert(
            continuationDevice.buffers[0].byteSize ==
                uint64_t(continuationDesc.capacity) * continuationDesc.recordStride &&
            continuationDevice.buffers[1].byteSize == uint64_t(continuationDesc.capacity) * sizeof(uint32_t) &&
            continuationDevice.buffers[2].byteSize == continuationDevice.buffers[1].byteSize &&
            continuationDevice.buffers[3].byteSize == continuationDevice.buffers[1].byteSize &&
            continuationDevice.buffers[4].byteSize == continuationDevice.buffers[1].byteSize &&
            continuationDevice.buffers[5].byteSize == sizeof(particle::GpuParticleContinuationCounters) &&
            continuationDevice.buffers[6].byteSize == particle::ParticleGpuContinuationRuntime::IndirectBufferBytes &&
            continuationDevice.buffers[7].byteSize == particle::ParticleGpuContinuationRuntime::IndirectBufferBytes &&
            continuationDevice.buffers[8].byteSize ==
                uint64_t(continuationDesc.particleCapacity) * continuationDesc.laneCount * sizeof(uint32_t) &&
            continuationDevice.buffers[9].byteSize == uint64_t(continuationDesc.particleCapacity) *
                                                          continuationDesc.joinCount *
                                                          sizeof(particle::GpuParticleContinuationJoinState));
        for (size_t index = 0; index < continuationDevice.buffers.size(); ++index) {
            assert(rhi::HasBufferUsage(continuationDevice.buffers[index].usage, rhi::BufferUsageFlags::Storage));
            assert(continuationDevice.buffers[index].memory == rhi::BufferMemory::DeviceLocal &&
                   continuationDevice.buffers[index].initialData == nullptr &&
                   continuationDevice.buffers[index].initialDataBytes == 0 &&
                   rhi::HasQueueAccess(continuationDevice.buffers[index].queueAccess,
                                       rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute));
        }
        assert(rhi::HasQueueAccess(continuationDevice.buffers[5].queueAccess, rhi::QueueAccessFlags::Transfer));
        assert(rhi::HasBufferUsage(continuationDevice.buffers[5].usage, rhi::BufferUsageFlags::TransferSource));
        assert(rhi::HasBufferUsage(continuationDevice.buffers[6].usage, rhi::BufferUsageFlags::Indirect) &&
               rhi::HasBufferUsage(continuationDevice.buffers[7].usage, rhi::BufferUsageFlags::Indirect));
        assert(continuationDevice.layouts.back().entryCount == 10);
        assert(continuationDevice.bindGroups.back().bufferCount == 10);
        assert(continuationDevice.pipelineCreates == 3);
        assert(continuationDevice.computePipelineDescs.size() == 3);
        assert(continuationDevice.computePipelineDescs[0].bindingLayoutCount == 1);
        assert(continuationDevice.computePipelineDescs[1].bindingLayoutCount == 1);
        assert(continuationDevice.computePipelineDescs[2].bindingLayoutCount == 8);

        ContinuationTrace continuationTrace;
        const rhi::ComputeCommandEncoder::DispatchTable continuationDispatch = {
            &ContinuationTrace::BindPipeline, &ContinuationTrace::BindGroup,        &ContinuationTrace::PushConstants,
            &ContinuationTrace::Dispatch,     &ContinuationTrace::DispatchIndirect, CanRecordCompute};
        const rhi::ComputeCommandEncoder continuationEncoder(&continuationTrace, &continuationDispatch);
        current->RequestReset();
        assert(current->ProgramGeneration() == 42 && current->ResetSerial() == 2 && current->Telemetry().resetPending);
        constexpr uint64_t Clock = 0x1122334455667788ull;
        assert(!current->RecordClassify(continuationEncoder, 17, Clock));
        const rhi::BindGroupHandle continuationSpawnGroup{0xffe1u, 1};
        assert(!current->RecordDispatch(continuationEncoder, 17, Clock, 99, 0.125f, continuationSpawnGroup));
        assert(current->RecordPrepare(continuationEncoder, 17, Clock));
        assert(!current->RecordDispatch(continuationEncoder, 17, Clock, 99, 0.125f, continuationSpawnGroup));
        assert(current->RecordClassify(continuationEncoder, 17, Clock));
        assert(!current->RecordClassify(continuationEncoder, 17, Clock));
        assert(current->RecordDispatch(continuationEncoder, 17, Clock, 99, 0.125f, continuationSpawnGroup));
        assert(!current->RecordDispatch(continuationEncoder, 17, Clock, 99, 0.125f, continuationSpawnGroup));
        const std::vector<std::array<uint32_t, 3>> expectedContinuationDispatches = {{{6, 1, 1}}};
        assert(continuationTrace.pipelines.size() == 3 &&
               continuationTrace.groupSets == std::vector<uint32_t>({0, 0, 0, 1, 2, 3, 4, 5, 6, 7}) &&
               continuationTrace.dispatches == expectedContinuationDispatches);
        assert(continuationTrace.indirectBuffers ==
               std::vector<rhi::BufferHandle>(
                   {current->Resources().classifyIndirectArguments, current->Resources().dispatchIndirectArguments}));
        assert(continuationTrace.indirectOffsets == std::vector<uint64_t>({0, 0}));
        assert(continuationTrace.constants.size() == 3 && continuationTrace.constants[0].capacity == 513 &&
               continuationTrace.constants[0].particleCapacity == 257 &&
               continuationTrace.constants[0].programGeneration == 42 &&
               continuationTrace.constants[0].simulationStep == 17 && continuationTrace.constants[0].resetSerial == 2 &&
               continuationTrace.constants[0].resetRequested == 1 &&
               continuationTrace.constants[0].elapsedTimeLow == 0x55667788u &&
               continuationTrace.constants[0].elapsedTimeHigh == 0x11223344u &&
               continuationTrace.constants[1].resetRequested == 0 &&
               continuationTrace.constants[2].resetRequested == 0 && continuationTrace.constants[2].systemSeed == 99 &&
               continuationTrace.constants[2].deltaTime == 0.125f);
        const auto telemetry = current->Telemetry();
        assert(telemetry.capacity == 513 && telemetry.particleCapacity == 257 && telemetry.recordStride == 96 &&
               telemetry.laneCount == 5 && telemetry.joinCount == 2 && telemetry.programGeneration == 42 &&
               telemetry.resetSerial == 2 && telemetry.recordBytes == continuationDevice.buffers[0].byteSize &&
               telemetry.queueBytes == continuationDevice.buffers[1].byteSize &&
               telemetry.laneSlotBytes == continuationDevice.buffers[8].byteSize &&
               telemetry.joinStateBytes == continuationDevice.buffers[9].byteSize &&
               telemetry.prepareRecordCalls == 1 && telemetry.classifyRecordCalls == 1 &&
               telemetry.dispatchRecordCalls == 1 && !telemetry.resetPending && telemetry.gpuCountersOnly &&
               telemetry.gpuCounters == current->Resources().counters &&
               telemetry.gpuClassifyIndirectArguments == current->Resources().classifyIndirectArguments &&
               telemetry.gpuDispatchIndirectArguments == current->Resources().dispatchIndirectArguments);
        assert(continuationDevice.readbacks == 0);

        auto replacement = std::make_shared<particle::ParticleGpuContinuationRuntime>();
        assert(replacement->CreateCompatible(continuationDevice, continuationDesc, *current));
        assert(replacement->IsValid() && replacement->SharesStorageWith(*current) &&
               replacement->ProgramGeneration() == 43 && replacement->ResetSerial() == 1 &&
               continuationDevice.buffers.size() == 10 && continuationDevice.pipelineCreates == 6);

        GpuRetirementQueue retirement;
        retirement.BindSerialSource([] { return rhi::SubmissionSerial{9}; });
        const uint32_t pipelineReleasesBeforeRetire = continuationDevice.pipelineReleases;
        retirement.Retire([retired = std::move(current)]() mutable { retired.reset(); });
        assert(retirement.PendingCount() == 1 && continuationDevice.bufferReleases == 0 &&
               continuationDevice.pipelineReleases == pipelineReleasesBeforeRetire);
        assert(retirement.Collect(8) == 0 && continuationDevice.bufferReleases == 0);
        assert(retirement.Collect(9) == 1 && continuationDevice.bufferReleases == 0 &&
               continuationDevice.pipelineReleases == pipelineReleasesBeforeRetire + 3);
        replacement->Destroy();
        assert(continuationDevice.bufferReleases == 10 && continuationDevice.pipelineReleases == 6 &&
               continuationDevice.groupReleases == 2 && continuationDevice.layoutReleases == 2);
        continuationDevice.Release(ownerGroup);
        continuationDevice.Release(ownerLayout);
        continuationDevice.Release(emptyGroup);
        continuationDevice.Release(emptyLayout);

        FakeDevice invalidDevice;
        const auto invalidOwnerLayout = invalidDevice.CreateBindingLayout({});
        rhi::BindGroupDesc invalidOwnerGroupDesc;
        invalidOwnerGroupDesc.layout = invalidOwnerLayout;
        continuationDesc.ownerLayout = invalidOwnerLayout;
        continuationDesc.ownerGroup = invalidDevice.CreateBindGroup(invalidOwnerGroupDesc);
        continuationDesc.capacity = particle::ParticleGpuContinuationRuntime::MaximumCapacity + 1;
        particle::ParticleGpuContinuationRuntime invalid;
        assert(!invalid.Create(invalidDevice, continuationDesc) && invalidDevice.buffers.empty());
    }

    {
        FakeDevice runtimeDevice;
        std::array<uint32_t, 5> shader = {0x07230203u, 0u, 0u, 0u, 0u};
        particle::GpuEmitterDesc runtimeDesc;
        runtimeDesc.capacity = 64;
        runtimeDesc.stateStride = 16;
        runtimeDesc.eventTypeCount = 5;
        for (auto &kernel : runtimeDesc.kernels)
            kernel = {shader.data(), shader.size()};
        runtimeDesc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};
        runtimeDesc.collisionSceneHeader = {901, 1};
        runtimeDesc.collisionSceneColliders = {902, 1};
        runtimeDesc.collisionSceneGridOffsets = {903, 1};
        runtimeDesc.collisionSceneGridColliderIndices = {904, 1};
        runtimeDesc.collisionSceneMeshVertices = {905, 1};
        runtimeDesc.collisionSceneMeshIndices = {906, 1};
        runtimeDesc.collisionSceneMeshBvhNodes = {907, 1};
        runtimeDesc.continuation.capacity = 32;
        runtimeDesc.continuation.laneCount = 3;
        runtimeDesc.continuation.joinCount = 1;
        runtimeDesc.continuation.initialProgramGeneration = 7;
        runtimeDesc.continuation.program = {
            {shader.data(), shader.size()},
            {shader.data(), shader.size()},
            {shader.data(), shader.size()},
        };

        particle::ParticleGpuRuntime current;
        particle::ParticleGpuRuntime replacement;
        assert(current.Create(runtimeDevice, runtimeDesc));
        assert(current.IsValid() && current.HasContinuations() && current.ContinuationTelemetry().capacity == 32 &&
               current.ContinuationTelemetry().programGeneration == 7 && current.EventTypeCount() == 5 &&
               current.CounterBufferByteSize() == 112);
        assert(replacement.CreateCompatible(runtimeDevice, runtimeDesc, current));
        assert(replacement.IsValid() && replacement.HasContinuations() && replacement.SharesStateWith(current) &&
               replacement.SharesContinuationStateWith(current) &&
               replacement.ContinuationTelemetry().programGeneration == 8);
        auto incompatibleEventDesc = runtimeDesc;
        incompatibleEventDesc.eventTypeCount = 6;
        particle::ParticleGpuRuntime incompatibleEventRuntime;
        assert(!incompatibleEventRuntime.CreateCompatible(runtimeDevice, incompatibleEventDesc, current));
        auto incompatibleCollisionDesc = runtimeDesc;
        incompatibleCollisionDesc.collisionEnabled = true;
        particle::ParticleGpuRuntime incompatibleCollisionRuntime;
        assert(!incompatibleCollisionRuntime.CreateCompatible(runtimeDevice, incompatibleCollisionDesc, current));
        const auto continuationRecords = current.ContinuationResources().records;
        assert(current.AdoptCompatibleRevision(replacement));
        assert(current.ContinuationTelemetry().programGeneration == 8 &&
               replacement.ContinuationTelemetry().programGeneration == 7 &&
               current.ContinuationResources().records == continuationRecords &&
               replacement.ContinuationResources().records == continuationRecords);
        const uint32_t buffersBeforeRetiredRevision = runtimeDevice.bufferReleases;
        replacement.Destroy();
        assert(runtimeDevice.bufferReleases == buffersBeforeRetiredRevision);
        current.RequestContinuationReset();
        assert(current.ContinuationTelemetry().programGeneration == 9 && current.ContinuationTelemetry().resetPending);
        current.Destroy();
        assert(runtimeDevice.bufferReleases == buffersBeforeRetiredRevision + 23);
    }

    {
        FakeDevice cacheDevice;
        VkTextureCache cache;
        const auto makeTexture = [&](uint32_t base, uint64_t bytes) {
            return std::make_shared<rhi::TextureResource>(cacheDevice, rhi::TextureHandle{base, 1},
                                                          rhi::TextureViewHandle{base + 1, 1},
                                                          rhi::SamplerHandle{base + 2, 1}, bytes);
        };

        auto slot = cache.Insert("asset::linear", makeTexture(100, 100), 1, false, "texture-guid", 1, 7);
        assert(slot && slot->Acquire() && slot->Acquire()->GetRevision() == 1 && cache.GetEntryCount() == 1 &&
               cache.GetResidentBytes() == 100);
        auto retainedRevisionOne = slot->Acquire();
        assert(cache.RequestAssetRevision("texture-guid", 2) == 1 && slot->GetRequestedRevision() == 2 &&
               slot->NeedsRefresh());
        auto replacementSlot = cache.Insert("asset::linear", makeTexture(200, 200), 2, false, "texture-guid", 2, 8);
        assert(replacementSlot == slot && replacementSlot->Acquire()->GetRevision() == 2 &&
               !replacementSlot->NeedsRefresh() && cache.GetResidentBytes() == 300 &&
               cache.GetRetiredLeaseCount() == 1 && cache.GetRetiredLeaseBytes() == 100);
        assert(!cache.FindAsset("asset::linear", "texture-guid", 1, 8, 3));
        assert(!cache.FindAsset("asset::linear", "texture-guid", 2, 9, 3));
        assert(cache.FindAsset("asset::linear", "texture-guid", 2, 8, 3) == slot);

        retainedRevisionOne.reset();
        assert(cache.GetResidentBytes() == 200 && cache.GetRetiredLeaseCount() == 0 &&
               cacheDevice.textureReleases == 2 && cacheDevice.samplerReleases == 1);

        auto externalSlot = slot;
        assert(cache.EvictByPrefix("asset") == 1 && cache.GetEntryCount() == 0 && cache.GetResidentBytes() == 200 &&
               cache.GetRetiredLeaseCount() == 1);
        slot.reset();
        replacementSlot.reset();
        externalSlot.reset();
        assert(cache.GetResidentBytes() == 0 && cache.GetRetiredLeaseCount() == 0 && cacheDevice.textureReleases == 4 &&
               cacheDevice.samplerReleases == 2);
    }

    {
        FakeDevice resolveDevice;
        const auto compiled = SlangCompiler::Get().Compile(test::ReadSceneShaderFixture("scene_depth_resolve.slang"),
                                                           {}, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
        const auto &code = compiled.targets[0];
        SceneDepthResolver resolver;
        resolver.Initialize(SceneDepthResolver::CreateProgram(resolveDevice, code));
        assert(resolveDevice.layouts.size() == 1 && resolveDevice.layouts[0].entryCount == 3);
        assert(resolveDevice.layouts[0].entries[0].type == rhi::BindingType::UniformBuffer);
        assert(resolveDevice.layouts[0].entries[1].type == rhi::BindingType::SampledTexture);
        assert(resolveDevice.layouts[0].entries[1].depthRead && resolveDevice.layouts[0].entries[1].multisampled);
        assert(resolveDevice.layouts[0].entries[2].type == rhi::BindingType::StorageTexture);

        DepthResolveTrace trace;
        const rhi::ComputeCommandEncoder::DispatchTable dispatch = {
            &DepthResolveTrace::BindPipeline, &DepthResolveTrace::BindGroup,        &DepthResolveTrace::PushConstants,
            &DepthResolveTrace::Dispatch,     &DepthResolveTrace::DispatchIndirect, CanRecordCompute};
        const rhi::ComputeCommandEncoder encoder(&trace, &dispatch);
        const rhi::TextureViewHandle source{101, 1};
        const rhi::TextureViewHandle destination{102, 1};
        resolver.Record(encoder, source, destination, 1919, 1079, 4);
        assert(resolveDevice.groupCreates == 1 && resolveDevice.bindGroups[0].textureCount == 2);
        assert(resolveDevice.bindGroups[0].textures[0].depthRead);
        assert(resolveDevice.bindGroups[0].textures[1].type == rhi::BindingType::StorageTexture);
        std::array<uint32_t, 4> constants;
        std::memcpy(constants.data(), resolveDevice.initialBufferBytes.back().data(), sizeof(constants));
        assert((constants == std::array<uint32_t, 4>{1919, 1079, 4, 0}));
        assert((trace.dispatch == std::array<uint32_t, 3>{240, 135, 1}));
        resolver.Record(encoder, source, destination, 1919, 1079, 4);
        assert(resolveDevice.groupCreates == 1);
        GpuRetirementQueue retirement;
        retirement.BindSerialSource([] { return 7; });
        const auto buffers = resolveDevice.bufferReleases;
        const auto groups = resolveDevice.groupReleases;
        resolver.RetireBindings(retirement);
        assert(retirement.Collect(6) == 0 && resolveDevice.bufferReleases == buffers &&
               resolveDevice.groupReleases == groups);
        assert(retirement.Collect(7) == 1 && resolveDevice.bufferReleases == buffers + 1 &&
               resolveDevice.groupReleases == groups + 1);
        resolver.Destroy();
    }

    {
        FakeDevice sharedDevice;
        const auto compiled = SlangCompiler::Get().Compile(test::ReadSceneShaderFixture("scene_depth_resolve.slang"),
                                                           {}, {ShaderCodeTarget::SpirV});
        auto program = SceneDepthResolver::CreateProgram(sharedDevice, compiled.targets[0]);
        assert(program);
        DepthResolveTrace trace;
        const rhi::ComputeCommandEncoder::DispatchTable dispatch = {
            &DepthResolveTrace::BindPipeline, &DepthResolveTrace::BindGroup,        &DepthResolveTrace::PushConstants,
            &DepthResolveTrace::Dispatch,     &DepthResolveTrace::DispatchIndirect, CanRecordCompute};
        const rhi::ComputeCommandEncoder encoder(&trace, &dispatch);
        SceneDepthResolver retained;
        retained.Initialize(program);
        for (uint32_t cycle = 0; cycle < 60; ++cycle) {
            SceneDepthResolver view;
            view.Initialize(program);
            view.Record(encoder, {101 + cycle, 1}, {201 + cycle, 1}, 64, 64, 4);
            assert(sharedDevice.pipelineCreates == 1 && sharedDevice.layoutCreates == 1);
        }
        assert(sharedDevice.groupCreates == 60 && sharedDevice.groupReleases == 60);
        assert(sharedDevice.pipelineReleases == 0 && sharedDevice.samplerReleases == 0);
        program.reset();
        retained.Record(encoder, {301, 1}, {302, 1}, 128, 128, 4);
        retained.Destroy();
        assert(!retained.IsValid());
        assert(sharedDevice.pipelineReleases == 1 && sharedDevice.layoutReleases == 1 &&
               sharedDevice.samplerReleases == 0 && sharedDevice.groupReleases == 61 &&
               sharedDevice.bufferReleases == 61);
    }

    {
        FakeDevice migrationDevice;
        particle::GpuParticleMigrationDesc migrationDesc;
        migrationDesc.sourceCapacity = 512;
        migrationDesc.destinationCapacity = 256;
        migrationDesc.sourceStride = 64;
        migrationDesc.destinationStride = 80;
        migrationDesc.sourceStates = {900, 1};
        migrationDesc.sourceCounters = {901, 1};
        migrationDesc.sourceCounterByteSize = 80;
        migrationDesc.destinationStates = {902, 1};
        migrationDesc.destinationFreeList = {903, 1};
        migrationDesc.destinationCounters = {904, 1};
        migrationDesc.destinationAliveIndices = {905, 1};
        migrationDesc.destinationAliveControl = {906, 1};
        migrationDesc.copyRanges = {{4, 4, 3, 0}, {8, 12, 4, 0}};
        migrationDesc.defaultStateWords.resize(20, 0u);
        migrationDesc.defaultStateWords[16] = 0x3f800000u;
        migrationDesc.program = {CompileParticleFixture(particle::GpuParticleMigrationShaderSources::Reset()),
                                 CompileParticleFixture(particle::GpuParticleMigrationShaderSources::Migrate())};

        particle::ParticleGpuMigrator migrator;
        assert(migrator.Create(migrationDevice, migrationDesc));
        assert(migrator.IsValid() && !migrator.WasRecorded());
        assert(migrationDevice.buffers.size() == 2 && migrationDevice.writes == 2);
        assert(migrationDevice.buffers[0].queueAccess ==
                   (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute) &&
               migrationDevice.buffers[1].queueAccess ==
                   (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute));
        assert(migrator.Constants().sourceStrideWords == 16 && migrator.Constants().destinationStrideWords == 20 &&
               migrator.Constants().copyRangeCount == 2 && migrator.Constants().invocationCount == 512);

        MigrationTrace trace;
        const rhi::ComputeCommandEncoder::DispatchTable dispatch = {
            &MigrationTrace::BindPipeline, &MigrationTrace::BindGroup,        &MigrationTrace::PushConstants,
            &MigrationTrace::Dispatch,     &MigrationTrace::DispatchIndirect, CanRecordCompute};
        const rhi::ComputeCommandEncoder encoder(&trace, &dispatch);
        migrator.RecordMigrate(encoder);
        assert(!migrator.WasRecorded() && trace.pipelines.empty());
        migrator.RecordReset(encoder);
        migrator.RecordMigrate(encoder);
        migrator.RecordMigrate(encoder);
        assert(migrator.WasRecorded());
        // Reset has no per-dispatch input; only migration reads these values.
        assert(trace.pipelines.size() == 2 && trace.constants.size() == 1);
        assert(std::memcmp(&trace.constants.front(), &migrator.Constants(), sizeof(migrator.Constants())) == 0);
        assert(trace.dispatches == std::vector<uint32_t>({1, 2}));
        migrator.Destroy();
        assert(migrationDevice.bufferReleases == 2 && migrationDevice.pipelineReleases == 2 &&
               migrationDevice.groupReleases == migrationDevice.groupCreates &&
               migrationDevice.layoutReleases == migrationDevice.layoutCreates);

        migrationDesc.copyRanges.clear();
        const size_t buffersBeforeEmptyMigration = migrationDevice.buffers.size();
        assert(migrator.Create(migrationDevice, migrationDesc));
        assert(migrator.Constants().copyRangeCount == 0);
        assert(migrationDevice.buffers.size() == buffersBeforeEmptyMigration + 2 &&
               migrationDevice.buffers[buffersBeforeEmptyMigration].byteSize ==
                   sizeof(particle::GpuParticleMigrationRange) &&
               migrationDevice.buffers[buffersBeforeEmptyMigration + 1].byteSize ==
                   migrationDesc.defaultStateWords.size() * sizeof(uint32_t));
        migrator.Destroy();

        migrationDesc.copyRanges = {{15, 4, 2, 0}};
        assert(!migrator.Create(migrationDevice, migrationDesc));
    }

    {
        FakeDevice collisionDevice;
        particle::ParticleGpuCollisionScene collisionScene;
        assert(collisionScene.Create(collisionDevice, 2));
        assert(collisionScene.IsValid() && collisionScene.Capacity() == 2 && collisionScene.PublishedRevision() == 0 &&
               collisionScene.HasPendingUpload());
        assert(collisionDevice.buffers.size() == 7 && collisionDevice.writes == 0);
        const auto collisionQueueAccess =
            rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute | rhi::QueueAccessFlags::Transfer;
        assert(std::all_of(collisionDevice.buffers.begin(), collisionDevice.buffers.end(),
                           [collisionQueueAccess](const rhi::BufferDesc &buffer) {
                               return buffer.queueAccess == collisionQueueAccess &&
                                      buffer.memory == rhi::BufferMemory::DeviceLocal;
                           }));

        particle::GpuParticleCollisionSceneSnapshot collisionSnapshot;
        collisionSnapshot.revision = 2;
        collisionSnapshot.topologyRevision = 1;
        collisionSnapshot.staticColliders.resize(1);
        collisionSnapshot.staticColliders[0].identity = {1u, 0u, 1u, 0u};
        collisionSnapshot.staticColliders[0].metadata[0] =
            static_cast<uint32_t>(particle::GpuParticleColliderType::Sphere);
        std::string collisionError;
        assert(collisionScene.Publish(collisionSnapshot, &collisionError) && collisionError.empty() &&
               collisionScene.HasPendingUpload() && collisionDevice.writes == 0);
        EventTransferTrace collisionTransferTrace;
        const rhi::TransferCommandEncoder::DispatchTable collisionTransferDispatch = {
            &EventTransferTrace::CopyBuffer,     &EventTransferTrace::CopyTexture,
            &EventTransferTrace::ResolveTexture, nullptr,
            &EventTransferTrace::UpdateBuffer,   nullptr,
        };
        const rhi::TransferCommandEncoder collisionTransferEncoder(&collisionTransferTrace, &collisionTransferDispatch);
        assert(collisionScene.RecordPendingUpload(collisionTransferEncoder));
        assert(collisionScene.HasPendingUpload() && collisionScene.PublishedRevision() == 0);
        collisionScene.NotifySubmission(true);
        assert(!collisionScene.HasPendingUpload() && collisionScene.PublishedColliderCount() == 1 &&
               collisionTransferTrace.updates.size() == 4 &&
               collisionTransferTrace.updates[0].bytes.size() == sizeof(particle::GpuParticleCollisionSceneHeader) &&
               collisionTransferTrace.updates[1].bytes.size() == 2 * sizeof(uint32_t) &&
               collisionTransferTrace.updates[2].bytes.size() == sizeof(uint32_t) &&
               collisionTransferTrace.updates[3].bytes.size() == sizeof(particle::GpuParticleColliderRecord));
        assert(collisionScene.PublishedStaticColliderCount() == 1 &&
               collisionScene.PublishedDynamicColliderCount() == 0 &&
               collisionScene.PublishedGridReferenceCount() == 1);
        assert(collisionScene.PublishedRevision() == 2);
        const uint32_t writesBeforeDuplicate = collisionDevice.writes;
        assert(collisionScene.Publish(collisionSnapshot, &collisionError) &&
               collisionDevice.writes == writesBeforeDuplicate);

        collisionSnapshot.revision = 3;
        collisionSnapshot.dynamicColliders.resize(1);
        collisionSnapshot.dynamicColliders[0].identity = {2u, 0u, 2u, 0u};
        collisionSnapshot.dynamicColliders[0].metadata[0] =
            static_cast<uint32_t>(particle::GpuParticleColliderType::Box);
        collisionSnapshot.dynamicColliders[0].worldAabbMin = {8.0f, 0.0f, 0.0f, 0.0f};
        collisionSnapshot.dynamicColliders[0].worldAabbMax = {9.0f, 1.0f, 1.0f, 0.0f};
        collisionSnapshot.dynamicColliders[0].previousWorldAabbMin = {-9.0f, 0.0f, 0.0f, 0.0f};
        collisionSnapshot.dynamicColliders[0].previousWorldAabbMax = {-8.0f, 1.0f, 1.0f, 0.0f};
        const uint32_t writesBeforeDynamic = collisionDevice.writes;
        assert(collisionScene.Publish(collisionSnapshot, &collisionError) &&
               collisionDevice.writes == writesBeforeDynamic);
        EventTransferTrace dynamicTransferTrace;
        const rhi::TransferCommandEncoder dynamicTransferEncoder(&dynamicTransferTrace, &collisionTransferDispatch);
        assert(collisionScene.RecordPendingUpload(dynamicTransferEncoder));
        collisionScene.NotifySubmission(true);
        assert(dynamicTransferTrace.updates.size() == 4 &&
               dynamicTransferTrace.updates[3].offset == sizeof(particle::GpuParticleColliderRecord) &&
               dynamicTransferTrace.updates[3].bytes.size() == sizeof(particle::GpuParticleColliderRecord));
        assert(collisionScene.PublishedColliderCount() == 2 && collisionScene.PublishedStaticColliderCount() == 1 &&
               collisionScene.PublishedDynamicColliderCount() == 1);
        assert(collisionScene.PublishedGridReferenceCount() > 2);

        collisionSnapshot.revision = 4;
        collisionSnapshot.topologyRevision = 2;
        collisionSnapshot.replaceMeshTopology = true;
        collisionSnapshot.staticColliders[0].metadata[0] =
            static_cast<uint32_t>(particle::GpuParticleColliderType::Mesh);
        collisionSnapshot.staticColliders[0].identity = {0x1234u, 0u, 0u, 0u};
        particle::GpuParticleCollisionMeshGeometry triangleGeometry;
        triangleGeometry.identity = 0x1234u;
        triangleGeometry.positions = {
            {{-1.0f, 0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}};
        triangleGeometry.indices = {0u, 1u, 2u};
        collisionSnapshot.meshGeometries = {triangleGeometry};
        const uint32_t writesBeforeTopology = collisionDevice.writes;
        assert(collisionScene.Publish(collisionSnapshot, &collisionError) && collisionError.empty() &&
               collisionDevice.writes == writesBeforeTopology);
        EventTransferTrace topologyTransferTrace;
        const rhi::TransferCommandEncoder topologyTransferEncoder(&topologyTransferTrace, &collisionTransferDispatch);
        assert(collisionScene.RecordPendingUpload(topologyTransferEncoder));
        collisionScene.NotifySubmission(true);
        assert(topologyTransferTrace.updates.size() == 8);
        assert(topologyTransferTrace.updates[5].bytes.size() == 3u * sizeof(std::array<float, 4>));
        assert(topologyTransferTrace.updates[6].bytes.size() == 3u * sizeof(uint32_t));
        assert(topologyTransferTrace.updates[7].bytes.size() == sizeof(particle::GpuParticleCollisionBvhNode));
        assert(collisionScene.PublishedTopologyRevision() == 2 && collisionScene.PublishedMeshVertexCount() == 3 &&
               collisionScene.PublishedMeshIndexCount() == 3 && collisionScene.PublishedMeshBvhNodeCount() == 1);
        collisionSnapshot.replaceMeshTopology = false;
        collisionSnapshot.meshGeometries.clear();

        auto rejectedTopology = collisionSnapshot;
        rejectedTopology.revision = 5;
        rejectedTopology.topologyRevision = 3;
        rejectedTopology.replaceMeshTopology = true;
        auto invalidGeometry = triangleGeometry;
        invalidGeometry.indices[2] = 9u;
        rejectedTopology.meshGeometries = {invalidGeometry};
        const uint32_t writesBeforeRejectedTopology = collisionDevice.writes;
        assert(!collisionScene.Publish(rejectedTopology, &collisionError) && !collisionError.empty() &&
               collisionDevice.writes == writesBeforeRejectedTopology && !collisionScene.HasPendingUpload());
        assert(collisionScene.PublishedRevision() == 4 && collisionScene.PublishedTopologyRevision() == 2 &&
               collisionScene.PublishedMeshVertexCount() == 3 && collisionScene.PublishedMeshIndexCount() == 3 &&
               collisionScene.PublishedMeshBvhNodeCount() == 1);

        rejectedTopology.meshGeometries = {triangleGeometry, triangleGeometry};
        assert(!collisionScene.Publish(rejectedTopology, &collisionError) && !collisionError.empty() &&
               collisionDevice.writes == writesBeforeRejectedTopology && !collisionScene.HasPendingUpload());

        rejectedTopology.topologyRevision = 2;
        rejectedTopology.meshGeometries = {triangleGeometry};
        assert(!collisionScene.Publish(rejectedTopology, &collisionError) && !collisionError.empty() &&
               collisionDevice.writes == writesBeforeRejectedTopology && !collisionScene.HasPendingUpload());

        collisionSnapshot.revision = 5;
        collisionSnapshot.dynamicColliders[0].worldAabbMin[0] = 7.0f;
        const uint32_t writesBeforeTransformOnly = collisionDevice.writes;
        assert(collisionScene.Publish(collisionSnapshot, &collisionError) && collisionError.empty() &&
               collisionDevice.writes == writesBeforeTransformOnly);
        EventTransferTrace transformOnlyTransferTrace;
        const rhi::TransferCommandEncoder transformOnlyTransferEncoder(&transformOnlyTransferTrace,
                                                                       &collisionTransferDispatch);
        assert(collisionScene.RecordPendingUpload(transformOnlyTransferEncoder));
        collisionScene.NotifySubmission(true);
        assert(transformOnlyTransferTrace.updates.size() == 4 &&
               transformOnlyTransferTrace.updates[3].bytes.size() == sizeof(particle::GpuParticleColliderRecord));
        assert(collisionScene.PublishedRevision() == 5 && collisionScene.PublishedTopologyRevision() == 2 &&
               collisionScene.PublishedMeshVertexCount() == 3 && collisionScene.PublishedMeshIndexCount() == 3 &&
               collisionScene.PublishedMeshBvhNodeCount() == 1);

        collisionSnapshot.revision = 3;
        assert(!collisionScene.Publish(collisionSnapshot, &collisionError) && !collisionError.empty());

        collisionSnapshot.revision = 6;
        collisionSnapshot.staticColliders.resize(2);
        collisionSnapshot.dynamicColliders.resize(1);
        assert(!collisionScene.Publish(collisionSnapshot, &collisionError) && !collisionError.empty());
        collisionScene.Destroy();
        assert(!collisionScene.IsValid() && collisionDevice.bufferReleases == 7);

        FakeDevice sortedCollisionDevice;
        particle::ParticleGpuCollisionScene sortedCollisionScene;
        assert(sortedCollisionScene.Create(sortedCollisionDevice, 3));
        particle::GpuParticleCollisionSceneSnapshot sortedSnapshot;
        sortedSnapshot.revision = 2;
        sortedSnapshot.topologyRevision = 1;
        sortedSnapshot.staticColliders.resize(2);
        sortedSnapshot.staticColliders[0].identity = {20u, 0u, 2u, 0u};
        sortedSnapshot.staticColliders[1].identity = {10u, 0u, 1u, 0u};
        sortedSnapshot.dynamicColliders.resize(1);
        sortedSnapshot.dynamicColliders[0].identity = {15u, 0u, 3u, 0u};
        assert(sortedCollisionScene.Publish(sortedSnapshot, &collisionError) && collisionError.empty());
        EventTransferTrace sortedTrace;
        const rhi::TransferCommandEncoder sortedEncoder(&sortedTrace, &collisionTransferDispatch);
        assert(sortedCollisionScene.RecordPendingUpload(sortedEncoder));
        sortedCollisionScene.NotifySubmission(true);
        const auto staticWrite =
            std::find_if(sortedTrace.updates.begin(), sortedTrace.updates.end(), [&](const auto &update) {
                return update.destination == sortedCollisionScene.ColliderBuffer() && update.offset == 0;
            });
        assert(staticWrite != sortedTrace.updates.end());
        std::array<particle::GpuParticleColliderRecord, 2> sortedStatic{};
        std::memcpy(sortedStatic.data(), staticWrite->bytes.data(), staticWrite->bytes.size());
        assert(sortedStatic[0].identity[0] == 10u && sortedStatic[1].identity[0] == 20u);

        sortedSnapshot.revision = 3;
        sortedSnapshot.replaceMeshTopology = false;
        sortedSnapshot.staticColliders[1].identity = sortedSnapshot.staticColliders[0].identity;
        assert(!sortedCollisionScene.Publish(sortedSnapshot, &collisionError));
        assert(collisionError.find("duplicate collider identity") != std::string::npos);
        sortedCollisionScene.Destroy();

        // Coalesce CPU edits without losing static/topology data from an earlier
        // edit, and acknowledge publication only after a successful submission.
        FakeDevice coalescedDevice;
        particle::ParticleGpuCollisionScene coalesced;
        assert(coalesced.Create(coalescedDevice, 4));
        auto first = collisionSnapshot;
        first.revision = 2;
        first.topologyRevision = 2;
        first.replaceMeshTopology = true;
        first.staticColliders.resize(1);
        first.dynamicColliders.clear();
        first.meshGeometries = {triangleGeometry};
        assert(coalesced.Publish(first));
        auto latest = first;
        latest.revision = 3;
        latest.replaceMeshTopology = false;
        latest.meshGeometries.clear();
        latest.dynamicColliders = {sortedSnapshot.dynamicColliders.front()};
        latest.dynamicColliders.front().material[0] = 0.75f;
        assert(coalesced.Publish(latest));
        EventTransferTrace failedTrace;
        failedTrace.rejectUpdateAt = 2;
        assert(!coalesced.RecordPendingUpload({&failedTrace, &collisionTransferDispatch}));
        coalesced.NotifySubmission(false);
        assert(coalesced.PublishedRevision() == 0 && coalesced.HasPendingUpload());

        EventTransferTrace rejectedTrace;
        assert(coalesced.RecordPendingUpload({&rejectedTrace, &collisionTransferDispatch}));
        assert(rejectedTrace.updates.size() == 8);
        coalesced.NotifySubmission(false);
        assert(coalesced.PublishedRevision() == 0 && coalesced.HasPendingUpload());
        EventTransferTrace acceptedTrace;
        assert(coalesced.RecordPendingUpload({&acceptedTrace, &collisionTransferDispatch}));
        assert(acceptedTrace.updates.size() == rejectedTrace.updates.size());
        for (size_t index = 0; index < acceptedTrace.updates.size(); ++index) {
            assert(acceptedTrace.updates[index].destination == rejectedTrace.updates[index].destination);
            assert(acceptedTrace.updates[index].offset == rejectedTrace.updates[index].offset);
            assert(acceptedTrace.updates[index].bytes == rejectedTrace.updates[index].bytes);
        }

        // An edit between recording and acknowledgement must remain pending;
        // the already recorded bytes must retain the prior immutable snapshot.
        latest.revision = 4;
        latest.dynamicColliders.front().material[0] = 0.25f;
        assert(coalesced.Publish(latest));
        coalesced.NotifySubmission(true);
        assert(coalesced.PublishedRevision() == 3 && coalesced.HasPendingUpload());
        EventTransferTrace newerTrace;
        assert(coalesced.RecordPendingUpload({&newerTrace, &collisionTransferDispatch}));
        assert(newerTrace.updates[4].bytes != acceptedTrace.updates[4].bytes);
        particle::GpuParticleColliderRecord captured{};
        std::memcpy(&captured, acceptedTrace.updates[4].bytes.data(), sizeof(captured));
        assert(captured.material[0] == 0.75f);
        coalesced.NotifySubmission(true);
        assert(coalesced.PublishedRevision() == 4 && !coalesced.HasPendingUpload());
        assert(coalesced.PublishedTopologyRevision() == 2 && coalesced.PublishedMeshVertexCount() == 3);
        assert(coalesced.Publish(latest));
        EventTransferTrace noChangeTrace;
        assert(coalesced.RecordPendingUpload({&noChangeTrace, &collisionTransferDispatch}));
        assert(noChangeTrace.updates.empty() && coalescedDevice.writes == 0);
    }

    {
        FakeDevice sharingDevice;
        std::array<std::array<uint32_t, 5>, static_cast<size_t>(particle::GpuKernelStage::Count)> sharingWords{};
        particle::GpuEmitterDesc sharingDesc;
        sharingDesc.capacity = 128;
        sharingDesc.stateStride = 32;
        sharingDesc.eventTypeCount = 3;
        sharingDesc.collisionSceneHeader = {0xfff0u, 1};
        sharingDesc.collisionSceneColliders = {0xfff1u, 1};
        sharingDesc.collisionSceneGridOffsets = {0xfff2u, 1};
        sharingDesc.collisionSceneGridColliderIndices = {0xfff3u, 1};
        sharingDesc.collisionSceneMeshVertices = {0xfff4u, 1};
        sharingDesc.collisionSceneMeshIndices = {0xfff5u, 1};
        sharingDesc.collisionSceneMeshBvhNodes = {0xfff6u, 1};
        for (size_t index = 0; index < sharingWords.size(); ++index) {
            sharingWords[index][0] = 0x07230203;
            sharingDesc.kernels[index] = {sharingWords[index].data(), sharingWords[index].size()};
        }
        sharingDesc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};

        particle::ParticleGpuRuntime previous;
        particle::ParticleGpuRuntime compatible;
        assert(previous.Create(sharingDevice, sharingDesc));
        assert(previous.EventTypeCount() == 3 && previous.CounterBufferByteSize() == 96);
        assert(previous.NeedsBootstrap());
        CommandTrace sharingTrace;
        const rhi::ComputeCommandEncoder::DispatchTable sharingDispatch = {
            &CommandTrace::BindPipeline, &CommandTrace::BindGroup,        &CommandTrace::PushConstants,
            &CommandTrace::Dispatch,     &CommandTrace::DispatchIndirect, CanRecordCompute};
        const rhi::ComputeCommandEncoder sharingEncoder(&sharingTrace, &sharingDispatch);
        const rhi::BindGroupHandle sharingSpawnGroup{0xffe2u, 1};
        assert(previous.RecordBootstrap(sharingEncoder, 13, sharingSpawnGroup));
        assert(!previous.NeedsBootstrap());
        const auto state = previous.StateBuffer();
        const auto counters = previous.CounterBuffer();
        const auto visibility = previous.VisibilityBuffer();
        assert(compatible.CreateCompatible(sharingDevice, sharingDesc, previous));
        assert(compatible.SharesStateWith(previous));
        assert(!compatible.NeedsBootstrap());
        assert(compatible.StateBuffer() == state && compatible.CounterBuffer() == counters &&
               compatible.VisibilityBuffer() == visibility);
        assert(sharingDevice.buffers.size() == 13);
        assert(previous.AdoptCompatibleRevision(compatible));
        assert(previous.IsValid() && compatible.IsValid() && previous.SharesStateWith(compatible));
        compatible.RequestBootstrap();
        assert(previous.NeedsBootstrap() && compatible.NeedsBootstrap());
        assert(compatible.RecordBootstrap(sharingEncoder, 13, sharingSpawnGroup));
        assert(!previous.NeedsBootstrap() && !compatible.NeedsBootstrap());
        previous.Destroy();
        assert(compatible.IsValid() && sharingDevice.bufferReleases == 0);
        compatible.Destroy();
        assert(sharingDevice.bufferReleases == 13 &&
               sharingDevice.pipelineReleases == 2u * RequiredGpuKernelStageCount && sharingDevice.groupReleases == 4 &&
               sharingDevice.layoutReleases == 6);
    }

    {
        FakeDevice collisionRuntimeDevice;
        std::array<std::array<uint32_t, 5>, static_cast<size_t>(particle::GpuKernelStage::Count)> collisionWords{};
        particle::GpuEmitterDesc collisionDesc;
        collisionDesc.capacity = 32;
        collisionDesc.stateStride = 32;
        collisionDesc.collisionEnabled = true;
        collisionDesc.collisionSceneHeader = {0xffd0u, 1};
        collisionDesc.collisionSceneColliders = {0xffd1u, 1};
        collisionDesc.collisionSceneGridOffsets = {0xffd2u, 1};
        collisionDesc.collisionSceneGridColliderIndices = {0xffd3u, 1};
        collisionDesc.collisionSceneMeshVertices = {0xffd4u, 1};
        collisionDesc.collisionSceneMeshIndices = {0xffd5u, 1};
        collisionDesc.collisionSceneMeshBvhNodes = {0xffd6u, 1};
        for (size_t index = 0; index < collisionWords.size(); ++index) {
            collisionWords[index][0] = 0x07230203u;
            collisionDesc.kernels[index] = {collisionWords[index].data(), collisionWords[index].size()};
        }
        collisionDesc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};

        particle::ParticleGpuRuntime collisionRuntime;
        assert(collisionRuntime.Create(collisionRuntimeDevice, collisionDesc));
        assert(collisionRuntime.CollisionEnabled() && collisionRuntime.HasContactRuntime() &&
               !collisionRuntime.SupportsFusedUpdateRendering());
        const auto contactTelemetry = collisionRuntime.ContactTelemetry();
        assert(contactTelemetry.particleCapacity == 32 && contactTelemetry.contactsPerParticle == 8 &&
               contactTelemetry.contactRecordCapacity == 256 && contactTelemetry.contactHashCapacity == 512 &&
               contactTelemetry.workItemCapacity == 512 && contactTelemetry.continuationSnapshotCapacity == 1 &&
               contactTelemetry.continuationJoinCapacity == 2 &&
               contactTelemetry.contactBytes == 2u * 256u * sizeof(particle::GpuParticleContactRecord) &&
               contactTelemetry.hashBytes == 2u * 512u * sizeof(particle::GpuParticleContactHashSlot) &&
               contactTelemetry.workItemBytes == 512u * sizeof(particle::GpuParticleContactWorkItem) &&
               contactTelemetry.continuationJoinBytes == 2u * sizeof(particle::GpuParticleContactJoinState));
        assert(collisionRuntimeDevice.layoutEntryCounts == std::vector<uint32_t>({13, 0, 6, 7, 9}));
        assert(collisionRuntimeDevice.groupBufferCounts == std::vector<uint32_t>({13, 0, 7, 9}));
        assert(collisionRuntimeDevice.computePipelineDescs.size() == RequiredGpuKernelStageCount);
        for (const auto &pipeline : collisionRuntimeDevice.computePipelineDescs)
            assert(pipeline.bindingLayoutCount == 8);

        CommandTrace contactTrace;
        const rhi::ComputeCommandEncoder::DispatchTable contactDispatch = {
            &CommandTrace::BindPipeline, &CommandTrace::BindGroup,        &CommandTrace::PushConstants,
            &CommandTrace::Dispatch,     &CommandTrace::DispatchIndirect, CanRecordCompute};
        const rhi::ComputeCommandEncoder contactEncoder(&contactTrace, &contactDispatch);
        const rhi::BindGroupHandle contactSpawnGroup{0xffcfu, 1};
        collisionRuntime.RecordContactPrepare(contactEncoder, 17u, contactSpawnGroup);
        assert(contactTrace.pipelines.size() == 1 && contactTrace.groups.size() == 8 &&
               contactTrace.groupSets == std::vector<uint32_t>({0, 1, 2, 3, 4, 5, 6, 7}) &&
               contactTrace.constants.size() == 1 && contactTrace.constants[0].simulationStep == 17u &&
               contactTrace.constants[0].invocationCount == 512u &&
               contactTrace.dispatches == std::vector<uint32_t>({2u}));
        assert(collisionRuntime.ContactPrepareRecordCalls() == 1u);
        collisionRuntime.RecordContactSolve(contactEncoder, 17u, contactSpawnGroup);
        assert(contactTrace.pipelines.size() == 2 && contactTrace.groups.size() == 16 &&
               contactTrace.constants.size() == 2 && contactTrace.constants[1].simulationStep == 17u &&
               contactTrace.constants[1].invocationCount == 32u &&
               contactTrace.dispatches == std::vector<uint32_t>({2u, 1u}));
        assert(collisionRuntime.ContactSolveRecordCalls() == 1u);
        collisionRuntime.RecordContactDispatch(contactEncoder, 19u, 17u, 1.0f / 60.0f, contactSpawnGroup, true);
        assert(contactTrace.pipelines.size() == 3 && contactTrace.groups.size() == 24 &&
               contactTrace.constants.size() == 3 && contactTrace.constants[2].systemSeed == 19u &&
               contactTrace.constants[2].simulationStep == 17u && contactTrace.constants[2].diagnosticFlags == 1u &&
               contactTrace.indirectBuffers ==
                   std::vector<rhi::BufferHandle>({collisionRuntime.ContactResources().dispatchIndirect}) &&
               contactTrace.indirectOffsets == std::vector<uint64_t>({0u}));
        assert(collisionRuntime.ContactDispatchRecordCalls() == 1u);
        collisionRuntime.RecordContactPrepare(contactEncoder, 18u, contactSpawnGroup, true);
        assert(contactTrace.constants.size() == 4 && contactTrace.constants[3].simulationStep == 18u &&
               contactTrace.constants[3].diagnosticFlags == 4u &&
               contactTrace.dispatches == std::vector<uint32_t>({2u, 1u, 2u}));
        assert(collisionRuntime.ContactPrepareRecordCalls() == 2u);

        particle::ParticleGpuRuntime compatibleCollisionRuntime;
        assert(compatibleCollisionRuntime.CreateCompatible(collisionRuntimeDevice, collisionDesc, collisionRuntime));
        assert(compatibleCollisionRuntime.SharesStateWith(collisionRuntime) &&
               compatibleCollisionRuntime.SharesContactStateWith(collisionRuntime));
        const auto contactRecords = collisionRuntime.ContactResources().contactRecords;
        assert(collisionRuntime.AdoptCompatibleRevision(compatibleCollisionRuntime));
        assert(collisionRuntime.ContactResources().contactRecords == contactRecords &&
               compatibleCollisionRuntime.ContactResources().contactRecords == contactRecords);
        compatibleCollisionRuntime.Destroy();
        collisionRuntime.Destroy();

        auto missingCollisionBuffer = collisionDesc;
        missingCollisionBuffer.collisionSceneMeshBvhNodes = {};
        assert(!collisionRuntime.Create(collisionRuntimeDevice, missingCollisionBuffer));
    }

    {
        FakeDevice meshInterfaceDevice;
        std::array<uint32_t, 5> shader = {0x07230203u, 0u, 0u, 0u, 0u};
        particle::GpuEmitterDesc meshInterfaceDesc;
        meshInterfaceDesc.capacity = 64;
        meshInterfaceDesc.stateStride = 16;
        for (auto &kernel : meshInterfaceDesc.kernels)
            kernel = {shader.data(), shader.size()};
        meshInterfaceDesc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};

        const auto meshLease = std::make_shared<uint32_t>(1u);
        particle::GpuMeshInterfaceDesc firstMesh;
        firstMesh.stableId = "surface-a";
        firstMesh.interfaceIndex = 0;
        firstMesh.metadataOffsetWords = 0;
        firstMesh.vertexBinding = 1;
        firstMesh.triangleBinding = 2;
        firstMesh.vertexCount = 24;
        firstMesh.triangleCount = 12;
        firstMesh.edgeCount = 30;
        firstMesh.vertices = {0xfa01u, 1};
        firstMesh.triangles = {0xfa02u, 1};
        firstMesh.vertexBufferBytes = 1920;
        firstMesh.triangleBufferBytes = 672;
        firstMesh.keepAlive = meshLease;
        firstMesh.meshToSpace = {
            -2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        };

        auto secondMesh = firstMesh;
        secondMesh.stableId = "surface-b";
        secondMesh.interfaceIndex = 1;
        secondMesh.metadataOffsetWords = 32;
        secondMesh.vertexBinding = 5;
        secondMesh.triangleBinding = 6;
        secondMesh.influenceBinding = 7;
        secondMesh.paletteBinding = 8;
        secondMesh.worldSpace = true;
        secondMesh.vertexCount = 36;
        secondMesh.triangleCount = 18;
        secondMesh.edgeCount = 48;
        secondMesh.vertices = {0xfa03u, 1};
        secondMesh.triangles = {0xfa04u, 1};
        secondMesh.boneCount = 1;
        secondMesh.poseRevision = 4;
        secondMesh.influences = {0xfa05u, 1};
        secondMesh.vertexBufferBytes = 2880;
        secondMesh.triangleBufferBytes = 1056;
        secondMesh.influenceBufferBytes = 1152;
        secondMesh.initialPalette = {glm::mat4(1.0f)};
        meshInterfaceDesc.meshInterfaces = {firstMesh, secondMesh};

        particle::ParticleGpuRuntime meshInterfaceRuntime;
        assert(meshInterfaceRuntime.Create(meshInterfaceDevice, meshInterfaceDesc));
        assert(meshInterfaceRuntime.IsValid());
        assert(meshInterfaceRuntime.MeshBufferBindings().size() == 9);
        for (uint32_t invalidRange = 0; invalidRange < 4; ++invalidRange) {
            auto invalidDesc = meshInterfaceDesc;
            if (invalidRange == 0)
                invalidDesc.meshInterfaces[0].vertexBufferBytes = 0;
            else if (invalidRange == 1)
                invalidDesc.meshInterfaces[0].triangleBufferBytes = 0;
            else if (invalidRange == 2)
                invalidDesc.meshInterfaces[1].influenceBufferBytes = 0;
            else
                invalidDesc.meshInterfaces[0].influenceBufferBytes = 32;
            particle::ParticleGpuRuntime rejected;
            assert(!rejected.Create(meshInterfaceDevice, invalidDesc));
            assert(rejected.MeshBufferBindings().empty() && !rejected.HasPendingUploads());
        }
        assert(std::find(meshInterfaceDevice.layoutEntryCounts.begin(), meshInterfaceDevice.layoutEntryCounts.end(),
                         9u) != meshInterfaceDevice.layoutEntryCounts.end());
        assert(std::find(meshInterfaceDevice.groupBufferCounts.begin(), meshInterfaceDevice.groupBufferCounts.end(),
                         9u) != meshInterfaceDevice.groupBufferCounts.end());

        EventTransferTrace uploadTrace;
        const rhi::TransferCommandEncoder::DispatchTable uploadDispatch{
            &EventTransferTrace::CopyBuffer,     &EventTransferTrace::CopyTexture,
            &EventTransferTrace::ResolveTexture, nullptr,
            &EventTransferTrace::UpdateBuffer,   nullptr};
        const rhi::TransferCommandEncoder uploadEncoder(&uploadTrace, &uploadDispatch);
        const auto flush = [&] {
            assert(meshInterfaceRuntime.RecordPendingUploads(uploadEncoder));
            meshInterfaceRuntime.NotifySubmission(true);
            assert(!meshInterfaceRuntime.HasPendingUploads() && meshInterfaceDevice.writes == 0);
        };
        particle::GpuParticleTransforms transforms;
        const std::array<float, 16> identity = {
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        };
        transforms.emitterToWorld = identity;
        transforms.worldToEmitter = identity;
        transforms.simulationToWorld = identity;
        transforms.worldToSimulation = identity;
        auto updatedPalette = std::make_shared<const std::vector<glm::mat4>>(
            std::vector<glm::mat4>{glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f))});
        auto movedSource = identity;
        movedSource[3] = 5.0f;
        assert(meshInterfaceRuntime.UpdateSkinnedMeshSources({{1u, 5u, movedSource, updatedPalette}}));
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        assert(uploadTrace.updates.size() == 4);
        auto metadataBuffer = uploadTrace.updates[2].destination;
        const auto metadataBytes = [&]() -> const std::vector<uint8_t> & {
            const auto found = std::find_if(uploadTrace.updates.rbegin(), uploadTrace.updates.rend(),
                                            [&](const auto &update) { return update.destination == metadataBuffer; });
            assert(found != uploadTrace.updates.rend());
            return found->bytes;
        };
        assert(metadataBytes().size() == 64u * sizeof(uint32_t));
        std::array<uint32_t, 64> metadata{};
        std::memcpy(metadata.data(), metadataBytes().data(), sizeof(metadata));
        assert(metadata[0] == 24u && metadata[1] == 12u && metadata[2] == 30u);
        assert(metadata[32] == 36u && metadata[33] == 18u && metadata[34] == 48u && metadata[35] == 1u);
        assert(NearlyEqual(FloatFromBits(metadata[32 + 4 + 12]), 5.0f));
        assert(NearlyEqual(FloatFromBits(metadata[4]), -2.0f));
        assert(NearlyEqual(FloatFromBits(metadata[9]), 3.0f));
        assert(NearlyEqual(FloatFromBits(metadata[14]), 4.0f));
        assert(NearlyEqual(FloatFromBits(metadata[20]), -0.5f));
        assert(NearlyEqual(FloatFromBits(metadata[25]), 1.0f / 3.0f));
        assert(NearlyEqual(FloatFromBits(metadata[30]), 0.25f));

        auto writesBeforeChange = uploadTrace.updates.size();
        assert(meshInterfaceRuntime.UpdateSkinnedMeshSources({{1u, 5u, movedSource, updatedPalette}}));
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        assert(uploadTrace.updates.size() == writesBeforeChange);

        // The emitter and pose revision remain fixed while the source moves,
        // rotates and scales. Only its metadata needs another upload.
        movedSource = {0, -3, 0, 8, 2, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1};
        assert(meshInterfaceRuntime.UpdateSkinnedMeshSources({{1u, 5u, movedSource, updatedPalette}}));
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        assert(uploadTrace.updates.size() == writesBeforeChange + 1);
        assert(uploadTrace.updates.back().destination == metadataBuffer);
        std::memcpy(metadata.data(), metadataBytes().data(), sizeof(metadata));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 4 + 12]), 8.0f));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 4 + 1]), 2.0f));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 4 + 4]), -3.0f));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 20 + 1]), 0.5f));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 20 + 4]), -1.0f / 3.0f));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 20 + 10]), 0.25f));

        // Emitter-local and scene-owned sources use distinct coordinate bases.
        transforms.emitterToWorld[12] = 10.0f;
        transforms.worldToEmitter[12] = -10.0f;
        transforms.simulationToWorld[12] = 4.0f;
        transforms.worldToSimulation[12] = -4.0f;
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        std::memcpy(metadata.data(), metadataBytes().data(), sizeof(metadata));
        assert(NearlyEqual(FloatFromBits(metadata[4 + 12]), 6.0f));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 4 + 12]), 4.0f));
        writesBeforeChange = uploadTrace.updates.size();
        assert(meshInterfaceRuntime.UpdateSkinnedMeshSources({{1u, 6u, movedSource, updatedPalette}}));
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        assert(uploadTrace.updates.size() == writesBeforeChange + 1);
        assert(uploadTrace.updates.back().destination != metadataBuffer);

        movedSource[3] = 9.0f;
        assert(meshInterfaceRuntime.UpdateSkinnedMeshSources({{1u, 6u, movedSource, updatedPalette}}));
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        uploadTrace.rejectUpdateAt = uploadTrace.updates.size();
        assert(!meshInterfaceRuntime.RecordPendingUploads(uploadEncoder));
        meshInterfaceRuntime.NotifySubmission(false);
        assert(meshInterfaceRuntime.HasPendingUploads());
        uploadTrace.rejectUpdateAt = std::numeric_limits<size_t>::max();
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        assert(uploadTrace.updates.back().destination == metadataBuffer);
        std::memcpy(metadata.data(), metadataBytes().data(), sizeof(metadata));
        assert(NearlyEqual(FloatFromBits(metadata[32 + 4 + 12]), 5.0f));
        writesBeforeChange = uploadTrace.updates.size();
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        assert(uploadTrace.updates.size() == writesBeforeChange);

        // A precomputed replacement owns a different metadata coordinate basis.
        // Adopting it cannot reuse the receiving runtime's old cache decision.
        auto replacementDesc = meshInterfaceDesc;
        replacementDesc.meshInterfaces[0].meshToSpace[3] = 11.0f;
        particle::ParticleGpuRuntime replacement;
        assert(replacement.CreateCompatible(meshInterfaceDevice, replacementDesc, meshInterfaceRuntime));
        auto replacementTransforms = transforms;
        replacementTransforms.worldToSimulation = identity;
        replacementTransforms.simulationToWorld = identity;
        assert(replacement.UpdateTransforms(replacementTransforms));
        assert(meshInterfaceRuntime.AdoptCompatibleRevision(replacement));
        // Two palettes precede this revision's metadata in its prepared batch.
        const auto replacementStart = uploadTrace.updates.size();
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        metadataBuffer = uploadTrace.updates[replacementStart + 2].destination;
        std::memcpy(metadata.data(), metadataBytes().data(), sizeof(metadata));
        assert(NearlyEqual(FloatFromBits(metadata[4 + 12]), 17.0f));
        writesBeforeChange = uploadTrace.updates.size();
        assert(meshInterfaceRuntime.UpdateTransforms(transforms));
        flush();
        assert(uploadTrace.updates.size() == writesBeforeChange);
        meshInterfaceRuntime.Destroy();

        auto invalidMeshDesc = meshInterfaceDesc;
        invalidMeshDesc.meshInterfaces[1].metadataOffsetWords = 0;
        assert(!meshInterfaceRuntime.Create(meshInterfaceDevice, invalidMeshDesc));
    }

    FakeDevice device;
    GpuRetirementQueue deletionQueue;
    deletionQueue.BindSerialSource([] { return rhi::SubmissionSerial{1}; });
    std::array<std::array<uint32_t, 5>, static_cast<size_t>(particle::GpuKernelStage::Count)> words{};
    particle::GpuEmitterDesc desc;
    desc.capacity = 1000;
    desc.stateStride = 64;
    desc.eventTypeCount = 5;
    desc.collisionSceneHeader = {0xfff2u, 1};
    desc.collisionSceneColliders = {0xfff3u, 1};
    desc.collisionSceneGridOffsets = {0xfff4u, 1};
    desc.collisionSceneGridColliderIndices = {0xfff5u, 1};
    desc.collisionSceneMeshVertices = {0xfff6u, 1};
    desc.collisionSceneMeshIndices = {0xfff7u, 1};
    desc.collisionSceneMeshBvhNodes = {0xfff8u, 1};
    for (size_t index = 0; index < words.size(); ++index) {
        words[index][0] = 0x07230203;
        desc.kernels[index] = {words[index].data(), words[index].size()};
    }
    desc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};

    particle::ParticleGpuRuntime runtime;
    device.maxBindingLayouts = 4;
    assert(runtime.Create(device, desc));
    assert(std::all_of(device.computePipelineDescs.begin(), device.computePipelineDescs.end(),
                       [](const auto &pipeline) { return pipeline.bindingLayoutCount == 4; }));
    assert(runtime.IsValid() && runtime.Capacity() == 1000 && runtime.StateStride() == 64 &&
           runtime.EventTypeCount() == 5 && !runtime.CollisionEnabled() && !runtime.SupportsFusedUpdateRendering() &&
           runtime.CounterBufferByteSize() == 112);
    assert(device.buffers.size() == 13);
    assert(device.buffers[0].byteSize == 64000);
    assert(rhi::HasBufferUsage(device.buffers[0].usage, rhi::BufferUsageFlags::Storage) &&
           rhi::HasBufferUsage(device.buffers[0].usage, rhi::BufferUsageFlags::TransferSource));
    assert(device.buffers[0].queueAccess == (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute) &&
           device.buffers[1].queueAccess == (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute) &&
           device.buffers[2].queueAccess == (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute));
    assert(device.buffers[2].byteSize == 112);
    assert(rhi::HasBufferUsage(device.buffers[2].usage, rhi::BufferUsageFlags::TransferSource));
    assert(device.buffers[3].byteSize ==
           static_cast<uint64_t>(desc.capacity) * particle::ParticleGpuRuntime::RenderInstanceStride);
    assert(device.buffers[4].byteSize ==
           static_cast<uint64_t>(desc.capacity) * particle::ParticleGpuRuntime::VisibilityInstanceStride);
    assert(rhi::HasBufferUsage(device.buffers[5].usage, rhi::BufferUsageFlags::Indirect));
    assert(device.buffers[6].byteSize == 4000 && device.buffers[6].usage == rhi::BufferUsageFlags::Storage);
    assert(device.buffers[7].byteSize == 4000 && device.buffers[7].usage == rhi::BufferUsageFlags::Storage);
    assert(device.buffers[8].byteSize == 4000 && device.buffers[8].usage == rhi::BufferUsageFlags::Storage);
    assert(device.buffers[9].byteSize == 32 &&
           device.buffers[9].usage == (rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::Indirect));
    assert(device.buffers[10].byteSize == 16 && device.buffers[10].usage == rhi::BufferUsageFlags::Storage);
    assert(device.buffers[11].byteSize == sizeof(particle::GpuParticleTransforms) &&
           device.buffers[11].memory == rhi::BufferMemory::DeviceLocal &&
           device.buffers[11].queueAccess == (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute));
    assert(device.buffers[12].byteSize == sizeof(particle::GpuParticleSimulationControl) &&
           device.buffers[12].usage == (rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferSource) &&
           device.buffers[12].memory == rhi::BufferMemory::DeviceLocal &&
           device.buffers[12].queueAccess == (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute) &&
           device.initialBufferBytes[12].empty());
    assert(runtime.AliveIndexBuffer(0).IsValid() && runtime.AliveIndexBuffer(1).IsValid() &&
           runtime.AliveIndexBuffer(0) != runtime.AliveIndexBuffer(1) && runtime.AliveDispatchBuffer().IsValid() &&
           runtime.AliveControlBuffer().IsValid() && !runtime.IsAliveListReady());
    assert(device.shaderCreates == RequiredGpuKernelStageCount && device.shaderReleases == RequiredGpuKernelStageCount);
    assert(device.layoutCreates == 3 && device.groupCreates == 2 &&
           device.pipelineCreates == RequiredGpuKernelStageCount && device.layouts.size() == 3 &&
           device.layouts[0].entryCount == 13 && device.bindGroups.size() == 2 &&
           device.bindGroups[0].bufferCount == 13);

    particle::GpuParticleTransforms transforms;
    assert(runtime.UpdateTransforms(transforms));
    assert(device.writes == 0 && runtime.HasPendingUploads());
    {
        const rhi::TransferCommandEncoder::DispatchTable uploads{
            &EventTransferTrace::CopyBuffer,     &EventTransferTrace::CopyTexture,
            &EventTransferTrace::ResolveTexture, nullptr,
            &EventTransferTrace::UpdateBuffer,   nullptr};
        EventTransferTrace first;
        assert(runtime.RecordPendingUploads({&first, &uploads}) && first.updates.size() == 1);
        assert(runtime.HasPendingUploads());
        transforms.emitterToWorld[12] = 7.0f;
        assert(runtime.UpdateTransforms(transforms));
        runtime.NotifySubmission(true);
        assert(runtime.HasPendingUploads());
        EventTransferTrace second;
        assert(runtime.RecordPendingUploads({&second, &uploads}) && second.updates.size() == 1);
        assert(first.updates[0].bytes != second.updates[0].bytes);
        runtime.NotifySubmission(false);
        assert(runtime.HasPendingUploads());
        EventTransferTrace retry;
        assert(runtime.RecordPendingUploads({&retry, &uploads}));
        assert(retry.updates[0].bytes == second.updates[0].bytes);
        runtime.NotifySubmission(true);
        assert(!runtime.HasPendingUploads());
        assert(runtime.UpdateTransforms(transforms));
        assert(runtime.RecordPendingUploads({&retry, &uploads}) && retry.updates.size() == 1);

        FakeDevice fieldDevice;
        auto fieldDesc = desc;
        particle::GpuVectorFieldDesc field;
        field.textureBinding = 1;
        field.worldSpace = false;
        field.texture = {0xab01, 1};
        field.sampler = {0xab02, 1};
        field.keepAlive = std::make_shared<int>(1);
        field.vectorScale = 1.5f;
        auto sdf = field;
        sdf.kind = particle::GpuVectorFieldDesc::Kind::SignedDistanceField;
        sdf.interfaceIndex = 1;
        sdf.textureBinding = 2;
        fieldDesc.vectorFields.vectorFields = {field, sdf};
        particle::ParticleGpuRuntime fieldRuntime;
        assert(fieldRuntime.Create(fieldDevice, fieldDesc));
        particle::GpuParticleTransforms fieldTransforms{field.fieldToSpace, field.fieldToSpace, field.fieldToSpace,
                                                        field.fieldToSpace};
        fieldTransforms.emitterToWorld[0] = 2;
        fieldTransforms.emitterToWorld[5] = 3;
        fieldTransforms.emitterToWorld[10] = 4;
        fieldTransforms.emitterToWorld[12] = 10;
        fieldTransforms.worldToSimulation[12] = -4;
        assert(fieldRuntime.UpdateTransforms(fieldTransforms));
        EventTransferTrace fieldTrace;
        assert(fieldRuntime.RecordPendingUploads({&fieldTrace, &uploads}));
        fieldRuntime.NotifySubmission(true);
        assert(fieldTrace.updates.size() == 2 && fieldDevice.writes == 0);
        std::array<float, 64> fieldMetadata{};
        assert(fieldTrace.updates[0].bytes.size() == sizeof(fieldMetadata));
        std::memcpy(fieldMetadata.data(), fieldTrace.updates[0].bytes.data(), sizeof(fieldMetadata));
        assert(NearlyEqual(fieldMetadata[0], 0.5f) && NearlyEqual(fieldMetadata[12], -3.0f));
        assert(NearlyEqual(fieldMetadata[16], 2.0f) && NearlyEqual(fieldMetadata[21], 3.0f));
        assert(NearlyEqual(fieldMetadata[28], 1.5f));
        assert(NearlyEqual(fieldMetadata[32 + 16], 0.5f) && NearlyEqual(fieldMetadata[32 + 21], 1.0f / 3.0f));
        assert(NearlyEqual(fieldMetadata[32 + 28], 3.0f));
        assert(fieldRuntime.UpdateTransforms(fieldTransforms));
        assert(fieldRuntime.RecordPendingUploads({&fieldTrace, &uploads}) && fieldTrace.updates.size() == 2);
        auto singular = fieldTransforms;
        singular.emitterToWorld[0] = 0;
        assert(!fieldRuntime.UpdateTransforms(singular));
        assert(!fieldRuntime.HasPendingUploads());
        assert(fieldRuntime.RecordPendingUploads({&fieldTrace, &uploads}) && fieldTrace.updates.size() == 2);
    }

    CommandTrace trace;
    const rhi::ComputeCommandEncoder::DispatchTable dispatch = {
        &CommandTrace::BindPipeline, &CommandTrace::BindGroup,        &CommandTrace::PushConstants,
        &CommandTrace::Dispatch,     &CommandTrace::DispatchIndirect, CanRecordCompute};
    const rhi::ComputeCommandEncoder encoder(&trace, &dispatch);
    const rhi::BindGroupHandle graphSpawnGroup{0xffe3u, 1};
    const rhi::BufferHandle spawnMetadata{0xffe4u, 1};
    assert(!runtime.RecordBootstrap(encoder, 7, {}) && runtime.NeedsBootstrap() && !runtime.IsAliveListReady());
    assert(runtime.RecordBootstrap(encoder, 7, graphSpawnGroup));
    assert(!runtime.IsAliveListReady() && runtime.AliveReadSlot() == 0 && runtime.AliveWriteSlot() == 1);
    runtime.RecordInitIndirect(encoder, 300, 100, 2, 7, 9, 1.0f / 60.0f, graphSpawnGroup, spawnMetadata,
                               sizeof(uint32_t) * 4u);
    assert(runtime.RecordUpdate(encoder, 7, 9, 1.0f / 60.0f, graphSpawnGroup, true));
    runtime.PublishAliveWrite();
    assert(runtime.IsAliveListReady() && runtime.AliveReadSlot() == 1 && runtime.AliveWriteSlot() == 0);
    assert(!runtime.RecordRenderReset(encoder, {}, true));
    assert(runtime.RecordRenderReset(encoder, graphSpawnGroup, true));
    assert(runtime.RecordRendering(encoder, 7, 9, graphSpawnGroup));
    assert(trace.pipelines.size() == 5 && trace.groups.size() == 20 && trace.constants.size() == 5);
    assert(trace.groupSets == std::vector<uint32_t>({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}));
    assert(trace.dispatches == std::vector<uint32_t>({4, 4, 1}));
    assert(trace.indirectBuffers == std::vector<rhi::BufferHandle>({spawnMetadata, runtime.AliveDispatchBuffer()}));
    assert(trace.indirectOffsets == std::vector<uint64_t>({16u, 16u}));
    assert(trace.constants[1].spawnBaseId == 100 && trace.constants[1].spawnGeneration == 2);
    assert(trace.constants[1].reserved == 0u);
    assert(trace.constants[2].simulationStep == 9);
    assert(trace.constants[1].diagnosticFlags == 0u);
    assert(trace.constants[2].diagnosticFlags == 1u);
    assert(trace.constants[3].diagnosticFlags == 2u);

    trace = {};
    runtime.RecordInitIndirect(encoder, 12, 700, 9, 7, 10, 1.0f / 60.0f, graphSpawnGroup, {}, 0);
    assert(trace.constants.size() == 1 && trace.constants[0].invocationCount == 12 &&
           trace.constants[0].spawnBaseId == 700 && trace.constants[0].spawnGeneration == 9 &&
           trace.constants[0].reserved == 1u && trace.dispatches == std::vector<uint32_t>({1u}) &&
           trace.indirectBuffers.empty());

    auto fusedDesc = desc;
    fusedDesc.supportsFusedUpdateRendering = true;
    fusedDesc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {
        words[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)].data(),
        words[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)].size()};
    particle::ParticleGpuRuntime missingFusedRuntime;
    auto missingFusedDesc = fusedDesc;
    missingFusedDesc.kernels[static_cast<size_t>(particle::GpuKernelStage::UpdateRenderingFused)] = {};
    assert(!missingFusedRuntime.CreateCompatible(device, missingFusedDesc, runtime));
    particle::ParticleGpuRuntime fusedRuntime;
    assert(fusedRuntime.CreateCompatible(device, fusedDesc, runtime));
    assert(fusedRuntime.SupportsFusedUpdateRendering());
    trace = {};
    assert(fusedRuntime.RecordUpdateRenderingFused(encoder, 7, 9, 1.0f / 60.0f, graphSpawnGroup));
    assert(trace.pipelines.size() == 1 && trace.dispatches.empty() &&
           trace.indirectBuffers == std::vector<rhi::BufferHandle>({runtime.AliveDispatchBuffer()}) &&
           trace.indirectOffsets == std::vector<uint64_t>({16u}) && trace.constants.size() == 1 &&
           trace.constants[0].simulationStep == 9);
    assert(!fusedRuntime.RecordUpdateRenderingFused(encoder, 7, 9, 1.0f / 60.0f, {}) &&
           fusedRuntime.AliveReadSlot() == 1u);
    fusedRuntime.PublishAliveWrite();
    assert(runtime.IsAliveListReady() && runtime.AliveReadSlot() == 0 && runtime.AliveWriteSlot() == 1);

    // Recording multiple views/export consumers is read-only with respect to
    // the simulation ping-pong cursor. Both recordings must consume the same
    // published list and indirect argument slot.
    trace = {};
    assert(runtime.RecordRendering(encoder, 7, 10, graphSpawnGroup));
    assert(fusedRuntime.RecordRendering(encoder, 7, 10, graphSpawnGroup));
    assert(runtime.AliveReadSlot() == 0 && fusedRuntime.AliveReadSlot() == 0 &&
           trace.indirectOffsets == std::vector<uint64_t>({0u, 0u}));

    trace = {};
    assert(runtime.RecordUpdate(encoder, 7, 10, 1.0f / 60.0f, graphSpawnGroup));
    assert(trace.constants.size() == 1 && trace.constants[0].aliveReadSlot == 0u &&
           trace.constants[0].aliveWriteSlot == 1u && trace.indirectOffsets == std::vector<uint64_t>({0u}));
    // The ordinary update path is sized by the GPU-owned live-count indirect
    // arguments. It neither dispatches the complete capacity from the CPU nor
    // reads the live list back merely to learn how much work remains.
    assert(trace.dispatches.empty() &&
           trace.indirectBuffers == std::vector<rhi::BufferHandle>({runtime.AliveDispatchBuffer()}) &&
           device.readbacks == 0u);
    runtime.PublishAliveWrite();
    assert(runtime.AliveReadSlot() == 1 && runtime.AliveWriteSlot() == 0);
    assert(runtime.AdoptCompatibleRevision(fusedRuntime));
    assert(runtime.SupportsFusedUpdateRendering() && !fusedRuntime.SupportsFusedUpdateRendering());
    particle::GpuParticleFrameRequest routeRequest;
    assert(particle::ParticleRenderGraph::ShouldUseFusedUpdateRendering(routeRequest, runtime));
    routeRequest.render = false;
    assert(!particle::ParticleRenderGraph::ShouldUseFusedUpdateRendering(routeRequest, runtime));
    routeRequest.render = true;
    routeRequest.simulate = false;
    assert(!particle::ParticleRenderGraph::ShouldUseFusedUpdateRendering(routeRequest, runtime));

    FakeDevice boundsDevice;
    particle::GpuParticleBoundsDesc boundsDesc;
    boundsDesc.capacity = runtime.Capacity();
    boundsDesc.visibility = runtime.VisibilityBuffer();
    boundsDesc.sourceIndices = runtime.RenderIndexBuffer();
    boundsDesc.sourceIndirectArguments = runtime.IndirectBuffer();
    boundsDesc.simulationControl = runtime.SimulationControlBuffer();
    boundsDesc.program = {CompileParticleFixture(particle::GpuParticleBoundsShaderSources::Prepare()),
                          CompileParticleFixture(particle::GpuParticleBoundsShaderSources::Reset()),
                          CompileParticleFixture(particle::GpuParticleBoundsShaderSources::Reduce())};
    particle::GpuParticleBoundsProgramStorage boundsProgramStorage;
    assert(boundsProgramStorage.Assign(boundsDesc.program) && boundsProgramStorage.IsValid());
    assert(boundsProgramStorage.View().prepare == boundsDesc.program.prepare &&
           boundsProgramStorage.View().reset == boundsDesc.program.reset);
    particle::ParticleGpuBounds bounds;
    assert(bounds.Create(boundsDevice, boundsDesc));
    assert(bounds.IsValid() && bounds.Capacity() == 1000 && bounds.VisibilityBuffer() == runtime.VisibilityBuffer() &&
           bounds.SourceIndexBuffer() == runtime.RenderIndexBuffer() &&
           bounds.SourceIndirectBuffer() == runtime.IndirectBuffer());
    assert(boundsDevice.buffers.size() == 2 &&
           boundsDevice.buffers[0].byteSize == particle::ParticleGpuBounds::BoundsBufferBytes &&
           rhi::HasBufferUsage(boundsDevice.buffers[0].usage, rhi::BufferUsageFlags::Storage) &&
           rhi::HasBufferUsage(boundsDevice.buffers[0].usage, rhi::BufferUsageFlags::TransferSource) &&
           boundsDevice.buffers[0].queueAccess == (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute) &&
           boundsDevice.buffers[1].byteSize == particle::ParticleGpuBounds::DispatchBufferBytes &&
           rhi::HasBufferUsage(boundsDevice.buffers[1].usage, rhi::BufferUsageFlags::Storage) &&
           rhi::HasBufferUsage(boundsDevice.buffers[1].usage, rhi::BufferUsageFlags::Indirect) &&
           boundsDevice.buffers[1].queueAccess == (rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute));
    assert(std::all_of(boundsDevice.layouts.begin(), boundsDevice.layouts.end(),
                       [](const auto &layout) { return layout.entryCount <= 8; }) &&
           std::all_of(boundsDevice.bindGroups.begin(), boundsDevice.bindGroups.end(),
                       [](const auto &group) { return group.bufferCount <= 8; }));

    BoundsTrace boundsTrace;
    const rhi::ComputeCommandEncoder::DispatchTable boundsDispatch = {
        &BoundsTrace::BindPipeline, &BoundsTrace::BindGroup,        &BoundsTrace::PushConstants,
        &BoundsTrace::Dispatch,     &BoundsTrace::DispatchIndirect, CanRecordCompute};
    const rhi::ComputeCommandEncoder boundsEncoder(&boundsTrace, &boundsDispatch);
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::PauseWhenOffscreen, false);
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::AlwaysSimulate, false);
    const size_t alwaysSimulatePrepareCount = boundsTrace.dispatches.size();
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::AlwaysSimulate, false);
    assert(boundsTrace.dispatches.size() == alwaysSimulatePrepareCount &&
           "AlwaysSimulate prepare must be cached until a control input changes");
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::AlwaysSimulate, true);
    const size_t forcedPrepareCount = boundsTrace.dispatches.size();
    assert(forcedPrepareCount == alwaysSimulatePrepareCount + 1 &&
           "forceSimulation changes must invalidate the AlwaysSimulate prepare cache");
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::AlwaysSimulate, true);
    assert(boundsTrace.dispatches.size() == forcedPrepareCount &&
           "unchanged forceSimulation must reuse the prepared control state");
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::AlwaysSimulate, false);
    const size_t unforcedPrepareCount = boundsTrace.dispatches.size();
    assert(unforcedPrepareCount == forcedPrepareCount + 1 &&
           "clearing forceSimulation must invalidate the AlwaysSimulate prepare cache");
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::AlwaysSimulate, false);
    assert(boundsTrace.dispatches.size() == unforcedPrepareCount &&
           "an unchanged unforced AlwaysSimulate state must remain cached");
    bounds.RecordPrepare(boundsEncoder, particle::GpuParticleOffscreenPolicy::PauseWhenOffscreen, false);
    bounds.RecordReset(boundsEncoder, particle::GpuParticleBoundsMode::Automatic, {}, {});
    const std::array<float, 3> manualLower = {-4.0f, -2.0f, 1.0f};
    const std::array<float, 3> manualUpper = {8.0f, 6.0f, 9.0f};
    bounds.RecordReset(boundsEncoder, particle::GpuParticleBoundsMode::Manual, manualLower, manualUpper);
    bounds.RecordReduce(boundsEncoder);
    assert(boundsTrace.pipelines.size() == 8 && boundsTrace.groups.size() == 8 && boundsTrace.constants.size() == 8);
    assert(boundsTrace.dispatches == std::vector<uint32_t>({1, 1, 1, 1, 1, 1, 1}));
    assert(boundsTrace.indirectDispatches == std::vector<rhi::BufferHandle>({bounds.DispatchBuffer()}));
    assert(boundsTrace.constants[0].capacity == 1000 &&
           boundsTrace.constants[0].offscreenPolicy == particle::GpuParticleOffscreenPolicy::PauseWhenOffscreen &&
           boundsTrace.constants[1].capacity == 1000 &&
           boundsTrace.constants[1].offscreenPolicy == particle::GpuParticleOffscreenPolicy::AlwaysSimulate &&
           boundsTrace.constants[2].capacity == 1000 &&
           boundsTrace.constants[2].offscreenPolicy == particle::GpuParticleOffscreenPolicy::AlwaysSimulate &&
           boundsTrace.constants[2].forceSimulation == 1 && boundsTrace.constants[3].capacity == 1000 &&
           boundsTrace.constants[3].offscreenPolicy == particle::GpuParticleOffscreenPolicy::AlwaysSimulate &&
           boundsTrace.constants[3].forceSimulation == 0 && boundsTrace.constants[4].capacity == 1000 &&
           boundsTrace.constants[4].offscreenPolicy == particle::GpuParticleOffscreenPolicy::PauseWhenOffscreen &&
           boundsTrace.constants[5].capacity == 1000 &&
           boundsTrace.constants[5].boundsMode == particle::GpuParticleBoundsMode::Automatic &&
           boundsTrace.constants[6].capacity == 1000 &&
           boundsTrace.constants[6].boundsMode == particle::GpuParticleBoundsMode::Manual &&
           std::equal(manualLower.begin(), manualLower.end(), boundsTrace.constants[6].manualLower.begin()) &&
           std::equal(manualUpper.begin(), manualUpper.end(), boundsTrace.constants[6].manualUpper.begin()) &&
           boundsTrace.constants[7].capacity == 1000);
    bounds.Destroy();
    assert(!bounds.IsValid() && boundsDevice.bufferReleases == 2 &&
           boundsDevice.groupReleases == boundsDevice.groupCreates &&
           boundsDevice.layoutReleases == boundsDevice.layoutCreates && boundsDevice.pipelineReleases == 3);

    FakeDevice cullDevice;
    particle::GpuParticleCullerDesc cullerDesc;
    cullerDesc.capacity = runtime.Capacity();
    cullerDesc.vertexCount = 6;
    cullerDesc.visibility = runtime.VisibilityBuffer();
    cullerDesc.ribbonInstances = runtime.InstanceBuffer();
    cullerDesc.sourceIndirectArguments = runtime.IndirectBuffer();
    cullerDesc.sourceIndices = runtime.RenderIndexBuffer();
    cullerDesc.bounds = {800, 1};
    cullerDesc.simulationControl = runtime.SimulationControlBuffer();
    cullerDesc.program = {CompileParticleFixture(particle::GpuParticleCullShaderSources::Reset()),
                          CompileParticleFixture(particle::GpuParticleCullShaderSources::Cull()),
                          CompileParticleFixture(particle::GpuParticleCullShaderSources::Finalize())};
    particle::GpuParticleCullProgramStorage cullProgramStorage;
    assert(cullProgramStorage.Assign(cullerDesc.program) && cullProgramStorage.IsValid());
    assert(cullProgramStorage.View().reset == cullerDesc.program.reset);
    auto portableCull = CompileParticleFixture(particle::GpuParticleCullShaderSources::Reset(), ShaderCodeTarget::Wgsl);
    const std::weak_ptr<const ShaderTargetCode> portableWeak = portableCull;
    particle::GpuParticleCullProgramStorage portableStorage;
    assert(portableStorage.Assign({portableCull, portableCull, portableCull}));
    portableCull.reset();
    assert(!portableWeak.expired() && portableStorage.View().reset->target == ShaderCodeTarget::Wgsl &&
           portableStorage.View().reset->entries.front().rootConstants);
    particle::ParticleGpuCuller sceneCuller;
    particle::ParticleGpuCuller gameCuller;
    assert(sceneCuller.Create(cullDevice, cullerDesc));
    assert(gameCuller.Create(cullDevice, cullerDesc));
    assert(sceneCuller.IsValid() && gameCuller.IsValid() && sceneCuller.Capacity() == 1000 &&
           sceneCuller.VisibilityBuffer() == runtime.VisibilityBuffer() &&
           sceneCuller.SourceIndirectBuffer() == runtime.IndirectBuffer() &&
           sceneCuller.SourceIndexBuffer() == runtime.RenderIndexBuffer() &&
           sceneCuller.Mode() == particle::GpuParticleCullMode::Instances &&
           sceneCuller.BoundsBuffer() == cullerDesc.bounds);
    assert(sceneCuller.VisibleIndexBuffer() != gameCuller.VisibleIndexBuffer() &&
           sceneCuller.DrawIndirectBuffer() != gameCuller.DrawIndirectBuffer() &&
           sceneCuller.SortDispatchBuffer() != gameCuller.SortDispatchBuffer());
    assert(cullDevice.buffers.size() == 6 && cullDevice.buffers[0].byteSize == 4000 &&
           cullDevice.buffers[0].usage == rhi::BufferUsageFlags::Storage && cullDevice.buffers[1].byteSize == 16 &&
           rhi::HasBufferUsage(cullDevice.buffers[1].usage, rhi::BufferUsageFlags::Indirect) &&
           cullDevice.buffers[2].byteSize == sizeof(particle::GpuParticleCullDispatchState) &&
           rhi::HasBufferUsage(cullDevice.buffers[2].usage, rhi::BufferUsageFlags::Indirect) &&
           rhi::HasBufferUsage(cullDevice.buffers[2].usage, rhi::BufferUsageFlags::TransferSource));
    assert(std::all_of(cullDevice.layouts.begin(), cullDevice.layouts.end(),
                       [](const auto &layout) { return layout.entryCount <= 8; }) &&
           std::all_of(cullDevice.bindGroups.begin(), cullDevice.bindGroups.end(),
                       [](const auto &group) { return group.bufferCount <= 8; }));

    CullTrace cullTrace;
    const rhi::ComputeCommandEncoder::DispatchTable cullDispatch = {&CullTrace::BindPipeline,     &CullTrace::BindGroup,
                                                                    &CullTrace::PushConstants,    &CullTrace::Dispatch,
                                                                    &CullTrace::DispatchIndirect, CanRecordCompute};
    const rhi::ComputeCommandEncoder cullEncoder(&cullTrace, &cullDispatch);
    std::array<float, particle::ParticleGpuCuller::PlaneCount * 4> frustumPlanes{};
    for (size_t index = 0; index < frustumPlanes.size(); ++index)
        frustumPlanes[index] = static_cast<float>(index + 1);
    sceneCuller.RecordReset(cullEncoder, frustumPlanes);
    sceneCuller.RecordCull(cullEncoder, frustumPlanes);
    sceneCuller.RecordFinalize(cullEncoder);
    assert(cullTrace.pipelines.size() == 3 && cullTrace.groups.size() == 3 && cullTrace.constants.size() == 3);
    assert(cullTrace.dispatches == std::vector<uint32_t>({1, 1}));
    assert(cullTrace.indirectDispatches == std::vector<rhi::BufferHandle>({sceneCuller.SortDispatchBuffer()}));
    assert(cullTrace.constants[0].capacity == 1000 && cullTrace.constants[0].frustumPlanes == frustumPlanes &&
           cullTrace.constants[1].capacity == 1000 && cullTrace.constants[1].frustumPlanes == frustumPlanes &&
           cullTrace.constants[2].capacity == 1000 &&
           cullTrace.constants[0].mode == particle::GpuParticleCullMode::Instances);
    assert(cullTrace.constants[0].viewEnabled == 1u);
    sceneCuller.RecordReset(cullEncoder, frustumPlanes, false);
    assert(cullTrace.constants.back().viewEnabled == 0u);
    sceneCuller.RecordReset(cullEncoder, frustumPlanes, true);
    assert(cullTrace.constants.back().viewEnabled == 1u);
    sceneCuller.Destroy();
    gameCuller.Destroy();
    assert(!sceneCuller.IsValid() && !gameCuller.IsValid() && cullDevice.bufferReleases == 6 &&
           cullDevice.groupReleases == cullDevice.groupCreates &&
           cullDevice.layoutReleases == cullDevice.layoutCreates && cullDevice.pipelineReleases == 6);

    FakeDevice sortDevice;
    particle::GpuParticleSorterDesc sorterDesc;
    sorterDesc.capacity = runtime.Capacity();
    sorterDesc.visibility = runtime.VisibilityBuffer();
    sorterDesc.indirectArguments = runtime.IndirectBuffer();
    sorterDesc.sourceIndices = runtime.RenderIndexBuffer();
    sorterDesc.dispatchArguments = {700, 1};
    sorterDesc.program = {CompileParticleFixture(particle::GpuParticleSortShaderSources::Small()),
                          CompileParticleFixture(particle::GpuParticleSortShaderSources::Generate()),
                          CompileParticleFixture(particle::GpuParticleSortShaderSources::Histogram()),
                          CompileParticleFixture(particle::GpuParticleSortShaderSources::Scan()),
                          CompileParticleFixture(particle::GpuParticleSortShaderSources::Scatter())};
    particle::GpuParticleSortProgramStorage sortProgramStorage;
    assert(sortProgramStorage.Assign(sorterDesc.program) && sortProgramStorage.IsValid());
    assert(sortProgramStorage.View().generate == sorterDesc.program.generate);
    particle::ParticleGpuSorter sorter;
    assert(sorter.Create(sortDevice, sorterDesc));
    particle::ParticleGpuSorter gameViewSorter;
    assert(gameViewSorter.Create(sortDevice, sorterDesc));
    const rhi::BufferHandle expectedSortDispatch{700, 1};
    assert(sorter.IsValid() && sorter.Capacity() == 1000 && sorter.BlockCount() == 4 &&
           sorter.SourceIndexBuffer() == runtime.RenderIndexBuffer() &&
           sorter.DispatchBuffer() == expectedSortDispatch && sorter.SortedIndices() == sorter.IndexBuffer(0) &&
           gameViewSorter.IsValid() && gameViewSorter.SortedIndices() != sorter.SortedIndices());
    static_assert(particle::ParticleGpuSorter::PassCount == 8);
    static_assert(particle::ParticleGpuSorter::PackedKeyStride == sizeof(uint32_t));
    assert(sortDevice.buffers.size() == 14 && sortDevice.buffers[0].byteSize == 4000 &&
           sortDevice.buffers[4].byteSize == 256 && sortDevice.buffers[5].byteSize == 256 &&
           sortDevice.buffers[6].byteSize == 64);
    assert(std::all_of(sortDevice.layouts.begin(), sortDevice.layouts.end(),
                       [](const auto &layout) { return layout.entryCount <= 8; }) &&
           std::all_of(sortDevice.bindGroups.begin(), sortDevice.bindGroups.end(),
                       [](const auto &group) { return group.bufferCount <= 8; }));
    assert(sortDevice.shaderCreates == 10 && sortDevice.shaderReleases == 0 && sortDevice.pipelineCreates == 10);

    SortTrace sortTrace;
    const rhi::ComputeCommandEncoder::DispatchTable sortDispatch = {&SortTrace::BindPipeline,     &SortTrace::BindGroup,
                                                                    &SortTrace::PushConstants,    &SortTrace::Dispatch,
                                                                    &SortTrace::DispatchIndirect, CanRecordCompute};
    const rhi::ComputeCommandEncoder sortEncoder(&sortTrace, &sortDispatch);
    std::array<float, 16> sortView{};
    sortView[0] = sortView[5] = sortView[10] = sortView[15] = 1.0f;
    const auto preparedSortGroups = sortDevice.groupCreates;
    // Infernux uses GLM's left-handed view convention: visible depth grows
    // along +Z. Back-to-front therefore reverses the ordered float key so the
    // radix/odd-even sort emits farther particles before nearer particles.
    sorter.RecordGenerate(sortEncoder, sortView, particle::ParticleSortMode::BackToFront);
    sorter.RecordHistogram(sortEncoder, 0);
    sorter.RecordScan(sortEncoder, 0);
    sorter.RecordScatter(sortEncoder, 0);
    sorter.RecordHistogram(sortEncoder, 1);
    assert(sortTrace.dispatches == std::vector<uint32_t>({1}));
    assert(sortTrace.indirectDispatches ==
           std::vector<rhi::BufferHandle>({sorterDesc.dispatchArguments, sorterDesc.dispatchArguments,
                                           sorterDesc.dispatchArguments, sorterDesc.dispatchArguments}));
    assert(sortTrace.constants.size() == 5 && sortTrace.constants[0].view == sortView &&
           sortTrace.constants[0].descending == 1 && sortTrace.constants[0].capacity == 1000 &&
           sortTrace.constants[0].blockCount == 4 && sortTrace.constants[1].digitShift == 0 &&
           sortTrace.constants[4].digitShift == 4);
    assert(sortTrace.groups[1] != sortTrace.groups[4] && sortDevice.groupCreates == preparedSortGroups);

    SortTrace gameSortTrace;
    const rhi::ComputeCommandEncoder gameSortEncoder(&gameSortTrace, &sortDispatch);
    auto gameSortView = sortView;
    gameSortView[12] = 12.0f;
    gameViewSorter.RecordGenerate(gameSortEncoder, gameSortView, particle::ParticleSortMode::FrontToBack);
    assert(gameSortTrace.constants.size() == 1 && gameSortTrace.constants[0].view == gameSortView &&
           gameSortTrace.constants[0].descending == 0 && gameSortTrace.groups[0] != sortTrace.groups[0] &&
           gameSortTrace.indirectDispatches == std::vector<rhi::BufferHandle>({sorterDesc.dispatchArguments}));
    sorter.Destroy();
    gameViewSorter.Destroy();
    assert(!sorter.IsValid() && !gameViewSorter.IsValid() && sortDevice.bufferReleases == 14 &&
           sortDevice.groupReleases == sortDevice.groupCreates &&
           sortDevice.layoutReleases == sortDevice.layoutCreates && sortDevice.pipelineReleases == 10);

    static_assert(particle::ParticleGpuSorter::SmallSortCapacity == 1024);
    FakeDevice smallSortDevice;
    auto smallSorterDesc = sorterDesc;
    smallSorterDesc.capacity = particle::ParticleGpuSorter::SmallSortCapacity;
    particle::ParticleGpuSorter smallSorter;
    assert(smallSorter.Create(smallSortDevice, smallSorterDesc));
    assert(smallSorter.IsValid() && smallSorter.UsesSmallSort() && smallSorter.BlockCount() == 4);
    SortTrace smallSortTrace;
    const rhi::ComputeCommandEncoder smallSortEncoder(&smallSortTrace, &sortDispatch);
    smallSorter.RecordSmall(smallSortEncoder, sortView, particle::ParticleSortMode::BackToFront);
    assert(smallSortTrace.dispatches == std::vector<uint32_t>({1}) && smallSortTrace.indirectDispatches.empty());
    assert(smallSortTrace.constants.size() == 1 && smallSortTrace.constants[0].view == sortView &&
           smallSortTrace.constants[0].descending == 1 &&
           smallSortTrace.constants[0].capacity == particle::ParticleGpuSorter::SmallSortCapacity &&
           smallSortTrace.constants[0].blockCount == 4);
    smallSorter.Destroy();
    assert(!smallSorter.IsValid() && smallSortDevice.bufferReleases == 7 &&
           smallSortDevice.groupReleases == smallSortDevice.groupCreates &&
           smallSortDevice.layoutReleases == smallSortDevice.layoutCreates && smallSortDevice.pipelineReleases == 5);

    FakeDevice radixThresholdDevice;
    auto radixThresholdDesc = sorterDesc;
    radixThresholdDesc.capacity = particle::ParticleGpuSorter::SmallSortCapacity + 1u;
    particle::ParticleGpuSorter radixThresholdSorter;
    assert(radixThresholdSorter.Create(radixThresholdDevice, radixThresholdDesc));
    assert(radixThresholdSorter.IsValid() && !radixThresholdSorter.UsesSmallSort() &&
           radixThresholdSorter.BlockCount() == 5);
    radixThresholdSorter.Destroy();

    FakeDevice ribbonDevice;
    particle::GpuParticleRibbonDesc ribbonDesc;
    ribbonDesc.capacity = runtime.Capacity();
    ribbonDesc.instances = runtime.InstanceBuffer();
    ribbonDesc.sourceIndices = runtime.RenderIndexBuffer();
    ribbonDesc.sourceIndirectArguments = runtime.IndirectBuffer();
    ribbonDesc.simulationControl = runtime.SimulationControlBuffer();
    ribbonDesc.program = {CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Reset()),
                          CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Initialize()),
                          CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Histogram()),
                          CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Scan()),
                          CompileParticleFixture(particle::GpuParticleRibbonShaderSources::Scatter())};
    particle::GpuParticleRibbonProgramStorage ribbonTopologyStorage;
    assert(ribbonTopologyStorage.Assign(ribbonDesc.program) && ribbonTopologyStorage.IsValid());
    auto ribbonTopology = std::make_shared<particle::ParticleGpuRibbonTopology>();
    assert(ribbonTopology->Create(ribbonDevice, ribbonDesc));
    assert(ribbonTopology->IsValid() && ribbonTopology->Capacity() == 1000 && ribbonTopology->BlockCount() == 4 &&
           ribbonTopology->InstanceBuffer() == runtime.InstanceBuffer() &&
           ribbonTopology->SourceIndexBuffer() == runtime.RenderIndexBuffer() &&
           ribbonTopology->SourceIndirectBuffer() == runtime.IndirectBuffer() &&
           ribbonTopology->SortedIndexBuffer() == ribbonTopology->IndexBuffer(0));
    assert(ribbonDevice.buffers.size() == 7 && ribbonDevice.buffers[0].byteSize == 4000 &&
           ribbonDevice.buffers[1].byteSize == 4000 && ribbonDevice.buffers[2].byteSize == 16 &&
           rhi::HasBufferUsage(ribbonDevice.buffers[2].usage, rhi::BufferUsageFlags::Indirect) &&
           ribbonDevice.buffers[3].byteSize == 12 &&
           rhi::HasBufferUsage(ribbonDevice.buffers[3].usage, rhi::BufferUsageFlags::Indirect) &&
           ribbonDevice.buffers[4].byteSize == 256 && ribbonDevice.buffers[5].byteSize == 256 &&
           ribbonDevice.buffers[6].byteSize == 64);
    assert(std::all_of(ribbonDevice.layouts.begin(), ribbonDevice.layouts.end(),
                       [](const auto &layout) { return layout.entryCount <= 8; }) &&
           std::all_of(ribbonDevice.bindGroups.begin(), ribbonDevice.bindGroups.end(),
                       [](const auto &group) { return group.bufferCount <= 8; }));

    particle::GpuParticleCullerDesc ribbonCullerDesc;
    ribbonCullerDesc.capacity = runtime.Capacity();
    ribbonCullerDesc.vertexCount = 6;
    ribbonCullerDesc.visibility = runtime.VisibilityBuffer();
    ribbonCullerDesc.ribbonInstances = runtime.InstanceBuffer();
    ribbonCullerDesc.sourceIndirectArguments = ribbonTopology->DrawIndirectBuffer();
    ribbonCullerDesc.sourceIndices = ribbonTopology->SortedIndexBuffer();
    ribbonCullerDesc.bounds = {801, 1};
    ribbonCullerDesc.simulationControl = runtime.SimulationControlBuffer();
    ribbonCullerDesc.mode = particle::GpuParticleCullMode::RibbonSegments;
    ribbonCullerDesc.program = cullProgramStorage.View();
    particle::ParticleGpuCuller ribbonCuller;
    assert(ribbonCuller.Create(ribbonDevice, ribbonCullerDesc));
    assert(ribbonCuller.IsValid() && ribbonCuller.Mode() == particle::GpuParticleCullMode::RibbonSegments &&
           ribbonCuller.SourceIndexBuffer() == ribbonTopology->SortedIndexBuffer() &&
           ribbonCuller.SourceIndirectBuffer() == ribbonTopology->DrawIndirectBuffer());

    RibbonTrace ribbonTrace;
    const rhi::ComputeCommandEncoder::DispatchTable ribbonDispatch = {
        &RibbonTrace::BindPipeline, &RibbonTrace::BindGroup,        &RibbonTrace::PushConstants,
        &RibbonTrace::Dispatch,     &RibbonTrace::DispatchIndirect, CanRecordCompute};
    const rhi::ComputeCommandEncoder ribbonEncoder(&ribbonTrace, &ribbonDispatch);
    ribbonTopology->RecordReset(ribbonEncoder);
    ribbonTopology->RecordInitialize(ribbonEncoder);
    ribbonTopology->RecordHistogram(ribbonEncoder, 0);
    ribbonTopology->RecordScan(ribbonEncoder, 0);
    ribbonTopology->RecordScatter(ribbonEncoder, 0);
    ribbonTopology->RecordHistogram(ribbonEncoder, 8);
    ribbonTopology->RecordHistogram(ribbonEncoder, 16);
    ribbonTopology->RecordHistogram(ribbonEncoder, 23);
    assert(ribbonTrace.dispatches == std::vector<uint32_t>({1, 1}));
    assert(ribbonTrace.indirectDispatches.size() == 6 &&
           std::all_of(ribbonTrace.indirectDispatches.begin(), ribbonTrace.indirectDispatches.end(),
                       [&](auto buffer) { return buffer == ribbonTopology->DispatchBuffer(); }));
    assert(ribbonTrace.constants.size() == 8 && ribbonTrace.constants[0].capacity == 1000 &&
           ribbonTrace.constants[0].blockCount == 4 && ribbonTrace.constants[2].keyField == 2 &&
           ribbonTrace.constants[2].digitShift == 0 && ribbonTrace.constants[5].keyField == 1 &&
           ribbonTrace.constants[6].keyField == 0 && ribbonTrace.constants[7].keyField == 0 &&
           ribbonTrace.constants[7].digitShift == 28);

    ribbonCuller.Destroy();
    ribbonTopology->Destroy();
    assert(!ribbonTopology->IsValid());

    const uint32_t bufferReleasesBeforeRuntimeDestroy = device.bufferReleases;
    const uint32_t pipelineReleasesBeforeRuntimeDestroy = device.pipelineReleases;
    const uint32_t shaderReleasesBeforeRuntimeDestroy = device.shaderReleases;
    const uint32_t groupReleasesBeforeRuntimeDestroy = device.groupReleases;
    const uint32_t layoutReleasesBeforeRuntimeDestroy = device.layoutReleases;
    const uint32_t textureReleasesBeforeRuntimeDestroy = device.textureReleases;
    const uint32_t samplerReleasesBeforeRuntimeDestroy = device.samplerReleases;
    runtime.Destroy();
    assert(!runtime.IsValid() && runtime.StateStride() == 0);
    assert(device.pipelineReleases == pipelineReleasesBeforeRuntimeDestroy + FusedGpuKernelStageCount);
    assert(device.groupReleases == groupReleasesBeforeRuntimeDestroy + 2u);
    assert(device.layoutReleases == layoutReleasesBeforeRuntimeDestroy + 3u);
    assert(device.bufferReleases == bufferReleasesBeforeRuntimeDestroy);
    assert(device.shaderReleases == shaderReleasesBeforeRuntimeDestroy);
    assert(device.textureReleases == textureReleasesBeforeRuntimeDestroy);
    assert(device.samplerReleases == samplerReleasesBeforeRuntimeDestroy);

    {
        using infernux::rhi::PipelineStage;
        using infernux::vk::PassBuilder;
        using infernux::vk::PassHandle;
        using infernux::vk::RenderGraph;
        using infernux::vk::ResourceHandle;

        struct EmitterPasses
        {
            uint64_t graphInstanceId = 0;
            PassHandle update{};
            PassHandle renderReset{};
            ResourceHandle counters{};
        };

        const auto buildTwoMultiEmitterGraphs = [](RenderGraph &graph, bool linkCrossGraphExportToSim) {
            std::array<EmitterPasses, 4> emitters{};
            ResourceHandle spawn[2]{};
            for (uint32_t graphIndex = 0; graphIndex < 2; ++graphIndex) {
                const uint64_t graphInstanceId = 100 + graphIndex;
                const std::string graphPrefix = "G" + std::to_string(graphIndex);
                graph.AddComputePass(graphPrefix + "/Advance", [&spawn, graphIndex, graphPrefix](PassBuilder &builder) {
                    spawn[graphIndex] =
                        builder.ImportBuffer(graphPrefix + "/Spawn", static_cast<VkBuffer>(VK_NULL_HANDLE), 16);
                    spawn[graphIndex] = builder.ReadWrite(spawn[graphIndex], PipelineStage::ComputeShader);
                    return [](infernux::vk::RenderContext &) {};
                });
                for (uint32_t emitterIndex = 0; emitterIndex < 2; ++emitterIndex) {
                    const uint32_t slot = graphIndex * 2 + emitterIndex;
                    const std::string prefix = graphPrefix + "/E" + std::to_string(emitterIndex);
                    emitters[slot].graphInstanceId = graphInstanceId;
                    emitters[slot].update =
                        graph.AddComputePass(prefix + "/Update", [&spawn, graphIndex](PassBuilder &builder) {
                            spawn[graphIndex] = builder.ReadWrite(spawn[graphIndex], PipelineStage::ComputeShader);
                            return [](infernux::vk::RenderContext &) {};
                        });
                    emitters[slot].renderReset =
                        graph.AddComputePass(prefix + "/RenderReset", [&emitters, slot, prefix](PassBuilder &builder) {
                            emitters[slot].counters =
                                builder.ImportBuffer(prefix + "/Counters", static_cast<VkBuffer>(VK_NULL_HANDLE), 16);
                            emitters[slot].counters =
                                builder.ReadWrite(emitters[slot].counters, PipelineStage::ComputeShader);
                            return [](infernux::vk::RenderContext &) {};
                        });
                    graph.AddComputePass(
                        prefix + "/Rendering", [&spawn, &emitters, graphIndex, slot](PassBuilder &builder) {
                            spawn[graphIndex] = builder.ReadWrite(spawn[graphIndex], PipelineStage::ComputeShader);
                            emitters[slot].counters =
                                builder.ReadWrite(emitters[slot].counters, PipelineStage::ComputeShader);
                            return [](infernux::vk::RenderContext &) {};
                        });
                }
            }
            if (!linkCrossGraphExportToSim)
                return;
            for (const auto &later : emitters) {
                for (const auto &earlier : emitters) {
                    if (later.graphInstanceId == earlier.graphInstanceId)
                        continue;
                    graph.AddPassDependency(later.renderReset, earlier.update);
                }
            }
        };

        RenderGraph acyclic;
        buildTwoMultiEmitterGraphs(acyclic, false);
        assert(acyclic.CompileExecutionOrder());

        RenderGraph cyclic;
        buildTwoMultiEmitterGraphs(cyclic, true);
        assert(!cyclic.CompileExecutionOrder());
    }

    fusedRuntime.Destroy();
    assert(!fusedRuntime.IsValid() && fusedRuntime.StateStride() == 0);
    assert(device.pipelineReleases ==
           pipelineReleasesBeforeRuntimeDestroy + FusedGpuKernelStageCount + RequiredGpuKernelStageCount);
    assert(device.groupReleases == groupReleasesBeforeRuntimeDestroy + 4u);
    assert(device.layoutReleases == layoutReleasesBeforeRuntimeDestroy + 6u);
    assert(device.bufferReleases == bufferReleasesBeforeRuntimeDestroy + 13u);
    assert(device.shaderReleases == shaderReleasesBeforeRuntimeDestroy);
    assert(device.textureReleases == textureReleasesBeforeRuntimeDestroy);
    assert(device.samplerReleases == samplerReleasesBeforeRuntimeDestroy);
    return 0;
}
