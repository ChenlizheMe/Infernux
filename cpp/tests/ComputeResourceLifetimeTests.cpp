#include <function/renderer/rhi/RhiComputeKernel.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <optional>
#include <set>
#include <stdexcept>

using namespace infernux::rhi;

namespace
{
class TestDevice final : public Device
{
  public:
    DeviceCaps caps;
    std::shared_ptr<DeviceLifetime> lifetime = std::make_shared<DeviceLifetime>();
    std::set<uint32_t> live;
    uint32_t next = 1;
    const DeviceCaps &GetCapabilities() const noexcept override
    {
        return caps;
    }
    std::shared_ptr<DeviceLifetime> GetLifetime() const noexcept override
    {
        return lifetime;
    }
    template <class H> H Allocate()
    {
        live.insert(next);
        return H{next++, ComposeHandleGeneration(1, 1)};
    }
    template <class H> void Free(H h)
    {
        if (h.IsValid())
            assert(live.erase(h.index) == 1);
    }
    TextureHandle CreateTexture(const TextureDesc &) override
    {
        assert(false);
        return {};
    }
    TextureViewHandle CreateTextureView(const TextureViewDesc &) override
    {
        assert(false);
        return {};
    }
    SamplerHandle CreateSampler(const SamplerDesc &) override
    {
        assert(false);
        return {};
    }
    BufferHandle CreateBuffer(const BufferDesc &) override
    {
        return Allocate<BufferHandle>();
    }
    ShaderModuleHandle CreateShaderModule(const ShaderModuleDesc &) override
    {
        return Allocate<ShaderModuleHandle>();
    }
    BindingLayoutHandle CreateBindingLayout(const BindingLayoutDesc &) override
    {
        return Allocate<BindingLayoutHandle>();
    }
    BindGroupHandle CreateBindGroup(const BindGroupDesc &) override
    {
        return Allocate<BindGroupHandle>();
    }
    GraphicsPipelineHandle CreateGraphicsPipeline(const GraphicsPipelineDesc &) override
    {
        assert(false);
        return {};
    }
    ComputePipelineHandle CreateComputePipeline(const ComputePipelineDesc &) override
    {
        return Allocate<ComputePipelineHandle>();
    }
    bool WriteBuffer(BufferHandle, uint64_t, const void *, uint64_t) override
    {
        return true;
    }
    bool ReadBuffer(BufferHandle, uint64_t, void *data, uint64_t size) override
    {
        std::memset(data, 0, static_cast<size_t>(size));
        return true;
    }
    void Release(TextureHandle h) noexcept override
    {
        Free(h);
    }
    void Release(TextureViewHandle h) noexcept override
    {
        Free(h);
    }
    void Release(SamplerHandle h) noexcept override
    {
        Free(h);
    }
    void Release(BufferHandle h) noexcept override
    {
        Free(h);
    }
    void Release(ShaderModuleHandle h) noexcept override
    {
        Free(h);
    }
    void Release(BindingLayoutHandle h) noexcept override
    {
        Free(h);
    }
    void Release(BindGroupHandle h) noexcept override
    {
        Free(h);
    }
    void Release(GraphicsPipelineHandle h) noexcept override
    {
        Free(h);
    }
    void Release(ComputePipelineHandle h) noexcept override
    {
        Free(h);
    }
};

// Ownership test only: no GPU commands are executed. Vulkan tests cover data
// correctness; this queue records which service survives the host wrapper.
class TestQueue final : public ComputeQueue
{
  public:
    uint64_t submissions = 0;
    uint64_t waits = 0;
    SubmissionTicket Submit(const Recorder &) override
    {
        return {1, QueueRole::Compute, ++submissions};
    }
    void Wait(SubmissionTicket ticket) override
    {
        assert(ticket.IsValid());
        ++waits;
    }
    bool IsComplete(SubmissionTicket) override
    {
        return true;
    }
    void Collect() override
    {
    }
    bool SetProfilingEnabled(bool) override
    {
        return false;
    }
    GpuTimestampFrame GetProfile() const override
    {
        return {};
    }
    uint64_t GetPendingSubmissionCount() const noexcept override
    {
        return 0;
    }
};
} // namespace

int main()
{
    TestDevice device;
    TestQueue originalQueue, replacementQueue;
    std::optional<ComputeHost> wrapper;
    wrapper.emplace(device, originalQueue);
    auto buffer = std::make_shared<ComputeBuffer>(*wrapper, ComputeBufferDesc{4});
    const uint32_t spirv[] = {0x07230203u, 0, 0, 0, 0};
    auto kernel = std::make_shared<ComputeKernel>(*wrapper, spirv, 5, 1, 0);
    auto readback = buffer->GetDataAsync(0, 4);
    buffer->AttachUpload({1, QueueRole::Compute, 1});

    // Replace the complete host object at the same address. This deterministically
    // catches borrowing the wrapper instead of its services, without a UAF in the test.
    wrapper.emplace(device, replacementQueue);
    assert(&buffer->GetHost().queue == &originalQueue);
    assert(&kernel->GetHost().queue == &originalQueue);
    readback->Wait();
    assert(originalQueue.waits == 1 && replacementQueue.waits == 0);

    // Different wrappers for the same device/queue are the same compute host.
    ComputeHost otherWrapper(device, originalQueue);
    auto otherReadback = std::make_shared<ComputeReadback>(otherWrapper, buffer, 0, 4);
    auto bytes = ReadComputeBatch(otherWrapper, {{buffer, 0, 4}});
    assert(bytes.size() == 1 && bytes.front().size() == 4);
    ComputeDispatchDesc dispatch;
    dispatch.kernel = kernel;
    dispatch.buffers = {buffer};
    dispatch.bufferAccesses = {ComputeBufferAccess::ReadWrite};
    assert(SubmitComputeBatch(otherWrapper, {{buffer, 0, {0, 0, 0, 0}}}, {dispatch}).IsValid());

    // The real queue boundary still rejects a buffer from a different host.
    bool rejected = false;
    try {
        ReadComputeBatch(*wrapper, {{buffer, 0, 4}});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    wrapper.reset();
    readback.reset();
    otherReadback.reset();
    kernel->Wait();
    dispatch = {};
    kernel.reset();
    buffer.reset();
    assert(device.live.empty());
    assert(originalQueue.waits > 1 && replacementQueue.waits == 0);
}
