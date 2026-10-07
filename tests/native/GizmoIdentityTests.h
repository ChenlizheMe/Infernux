#pragma once

#include <function/renderer/GizmosDrawCallBuffer.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace gizmo_identity_test
{
using namespace infernux;
using namespace infernux::rhi;

// Metadata-only device: production ComputeBuffer owns real lifetime/size
// metadata, but no graphics commands or pixels are claimed by this test.
class MetadataDevice final : public Device
{
    DeviceCaps caps;
    std::shared_ptr<DeviceLifetime> lifetime = std::make_shared<DeviceLifetime>();
    uint32_t next = 1;
  public:
    const DeviceCaps &GetCapabilities() const noexcept override { return caps; }
    std::shared_ptr<DeviceLifetime> GetLifetime() const noexcept override { return lifetime; }
    BufferHandle CreateBuffer(const BufferDesc &) override { return {next++, ComposeHandleGeneration(1, 1)}; }
    TextureHandle CreateTexture(const TextureDesc &) override { throw std::logic_error("unexpected texture"); }
    TextureViewHandle CreateTextureView(const TextureViewDesc &) override { throw std::logic_error("unexpected view"); }
    SamplerHandle CreateSampler(const SamplerDesc &) override { throw std::logic_error("unexpected sampler"); }
    ShaderModuleHandle CreateShaderModule(const ShaderModuleDesc &) override { throw std::logic_error("unexpected shader"); }
    BindingLayoutHandle CreateBindingLayout(const BindingLayoutDesc &) override { throw std::logic_error("unexpected layout"); }
    BindGroupHandle CreateBindGroup(const BindGroupDesc &) override { throw std::logic_error("unexpected group"); }
    GraphicsPipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &) override { throw std::logic_error("unexpected pipeline"); }
    ComputePipelineHandle CreateComputePipeline(const ComputePipelineDesc &) override { throw std::logic_error("unexpected compute"); }
    bool WriteBuffer(BufferHandle, uint64_t, const void *, uint64_t) override { throw std::logic_error("unexpected upload"); }
    void Release(BufferHandle) noexcept override {}
    void Release(TextureHandle) noexcept override {}
    void Release(TextureViewHandle) noexcept override {}
    void Release(SamplerHandle) noexcept override {}
    void Release(ShaderModuleHandle) noexcept override {}
    void Release(BindingLayoutHandle) noexcept override {}
    void Release(BindGroupHandle) noexcept override {}
    void Release(GraphicsPipelineHandle) noexcept override {}
    void Release(ComputePipelineHandle) noexcept override {}
};

class MetadataQueue final : public ComputeQueue
{
  public:
    SubmissionTicket Submit(const Recorder &, std::shared_ptr<void>) override { throw std::logic_error("unexpected submit"); }
    void Wait(SubmissionTicket) override { throw std::logic_error("unexpected wait"); }
    bool IsComplete(SubmissionTicket) override { throw std::logic_error("unexpected ticket"); }
    void Collect() override {}
    bool SetProfilingEnabled(bool) override { return false; }
    GpuTimestampFrame GetProfile() const override { return {}; }
    uint64_t GetPendingSubmissionCount() const noexcept override { return 0; }
};

inline bool Run()
{
    int passed = 0, failed = 0;
    auto check = [&](bool ok, const char *name) {
        std::cout << "GIZMO_IDENTITY " << name << (ok ? " PASS\n" : " FAIL\n");
        ok ? ++passed : ++failed;
    };
    auto unique = [](const DrawCallResult &result) {
        std::set<uint64_t> ids;
        std::unordered_set<RenderDrawIdentity, RenderDrawIdentityHash> draws;
        for (const auto &draw : result.drawCalls) {
            if (!draw.identity.IsValid() || !ids.insert(draw.objectId).second || !draws.insert(draw.identity).second)
                return false;
        }
        return true;
    };
    MetadataDevice device;
    MetadataQueue queue;
    ComputeHost host(device, queue);
    auto storage = std::make_shared<ComputeBuffer>(host, ComputeBufferDesc{2 * sizeof(Vertex) / 4});
    auto cpu = [](float x) {
        GizmosDrawCallBuffer::DrawDescriptor descriptor{};
        descriptor.indexCount = 2;
        const glm::mat4 world = glm::translate(glm::mat4(1), glm::vec3(x, 0, 0));
        std::memcpy(descriptor.worldMatrix, &world, sizeof(world));
        return descriptor;
    };
    auto resident = [&](uint64_t id, float x = 0) {
        GizmosDrawCallBuffer::ResidentDrawDescriptor descriptor;
        descriptor.identity = id;
        descriptor.vertexCount = 2;
        descriptor.vertexBuffer = storage;
        descriptor.indices = {0, 1};
        const auto world = cpu(x);
        std::memcpy(descriptor.worldMatrix, world.worldMatrix, sizeof(world.worldMatrix));
        return descriptor;
    };
    GizmosDrawCallBuffer buffer;
    buffer.SetData(std::vector<Vertex>(2), {0, 1}, {cpu(0), cpu(1)});
    buffer.SetResidentData({resident(1)});
    auto mixed = buffer.GetDrawCalls(nullptr);
    check(mixed.drawCalls.size() == 3 && unique(mixed), "cpu_resident_separate");
    buffer.SetResidentData({resident(1), resident(0x100000001ULL), resident(UINT64_MAX)});
    auto wide = buffer.GetDrawCalls(nullptr);
    check(wide.drawCalls.size() == 5 && unique(wide), "full_64_bit_source_identity");
    buffer.SetResidentData({resident(1, 5), resident(0x100000001ULL), resident(UINT64_MAX)});
    auto stable = buffer.GetDrawCalls(nullptr);
    check(stable.drawCalls[2].objectId == wide.drawCalls[2].objectId &&
          stable.drawCalls[2].identity == wide.drawCalls[2].identity &&
          stable.drawCalls[2].worldMatrix[3].x == 5, "stable_resident_move");
    buffer.SetData(std::vector<Vertex>(2), {0, 1}, {cpu(5), cpu(6)});
    check(buffer.GetDrawCalls(nullptr).drawCalls[1].identity == mixed.drawCalls[1].identity, "stable_cpu_move");
    buffer.SetResidentData({});
    buffer.SetResidentData({resident(1)});
    check(buffer.GetDrawCalls(nullptr).drawCalls[2].identity != mixed.drawCalls[2].identity, "retired_resident_not_reused");
    GizmosDrawCallBuffer second;
    second.SetData(std::vector<Vertex>(2), {0, 1}, {cpu(0)});
    second.SetResidentData({resident(1)});
    auto combined = buffer.GetDrawCalls(nullptr);
    for (const auto &draw : second.GetDrawCalls(nullptr).drawCalls)
        combined.drawCalls.push_back(draw);
    check(unique(combined), "buffer_owners_separate");

    GizmosDrawCallBuffer::IconMaterials materials;
    // Truth-only material route, never dereferenced by GetIconDrawCalls.
    materials.fallback = std::shared_ptr<InxMaterial>(reinterpret_cast<InxMaterial *>(1), [](InxMaterial *) {});
    const auto projection = glm::perspective(glm::radians(60.0f), 1.0f, .1f, 100.0f);
    auto icons = [&]() { return buffer.GetIconDrawCalls(materials, {0, 0, -10}, {1, 0, 0}, {0, 1, 0}, projection, 800, 1); };
    buffer.SetIconData({{{0, 0, 0}, 42}, {{1, 0, 0}, 0x10000002AULL}});
    auto iconDraws = icons();
    check(iconDraws.drawCalls.size() == 2 && unique(iconDraws), "icon_full_owner_identity");
    check(iconDraws.drawCalls[0].pickingObjectId == 42 && iconDraws.drawCalls[1].pickingObjectId == 0x10000002AULL,
          "icon_picking_retains_owner");
    auto iconStable = icons();
    check(iconStable.drawCalls[1].identity == iconDraws.drawCalls[1].identity, "icon_stable_identity");
    buffer.ClearIcons();
    buffer.SetIconData({{{0, 0, 0}, 42}});
    check(icons().drawCalls[0].identity != iconDraws.drawCalls[0].identity, "retired_icon_not_reused");
    auto cpuBeforeClear = buffer.GetDrawCalls(nullptr).drawCalls[0].identity;
    buffer.ClearCpuData();
    buffer.SetData(std::vector<Vertex>(2), {0, 1}, {cpu(0)});
    check(buffer.GetDrawCalls(nullptr).drawCalls[0].identity != cpuBeforeClear, "retired_cpu_not_reused");
    std::cout << "GIZMO_IDENTITY_SUMMARY passed=" << passed << " failed=" << failed << '\n';
    return failed == 0;
}
} // namespace gizmo_identity_test
