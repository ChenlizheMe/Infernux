#include <function/renderer/rhi/RhiComputeKernel.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
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
    TestDevice()
    {
        caps.limits.maxComputeWorkgroupCount[0] = 4;
        caps.limits.maxComputeWorkgroupCount[1] = 3;
        caps.limits.maxComputeWorkgroupCount[2] = 2;
    }
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
    bool complete = true;
    std::map<uint64_t, std::shared_ptr<void>> retained;
    SubmissionTicket Submit(const Recorder &, std::shared_ptr<void> resources = {}) override
    {
        retained.emplace(submissions + 1, std::move(resources));
        return {1, QueueRole::Compute, ++submissions};
    }
    void Wait(SubmissionTicket ticket) override
    {
        assert(ticket.IsValid());
        ++waits;
        retained.erase(retained.begin(), retained.upper_bound(ticket.serial));
    }
    bool IsComplete(SubmissionTicket) override
    {
        return complete;
    }
    void Collect() override
    {
        if (complete)
            retained.clear();
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

bool VerifyDispatchLimits()
{
    struct Scenario
    {
        std::array<uint32_t, 3> groups;
        bool valid;
    };
    const std::array<Scenario, 11> scenarios{{
        {{1, 1, 1}, true}, {{4, 3, 2}, true},
        {{0, 1, 1}, false}, {{1, 0, 1}, false}, {{1, 1, 0}, false},
        {{5, 1, 1}, false}, {{1, 4, 1}, false}, {{1, 1, 3}, false},
        {{UINT32_MAX, 1, 1}, false}, {{1, UINT32_MAX, 1}, false}, {{1, 1, UINT32_MAX}, false},
    }};
    unsigned passed = 0, failed = 0;
    for (int path = 0; path < 3; ++path) {
        for (const auto &scenario : scenarios) {
            TestDevice device;
            TestQueue queue;
            ComputeHost host(device, queue);
            bool rejected = false;
            {
                const uint32_t spirv[] = {0x07230203u, 0, 0, 0, 0};
                auto buffer = std::make_shared<ComputeBuffer>(host, ComputeBufferDesc{4});
                auto kernel = std::make_shared<ComputeKernel>(host, spirv, 5, 1, 0);
                try {
                    if (path == 0) {
                        kernel->Dispatch({buffer}, nullptr, 0, scenario.groups[0], scenario.groups[1], scenario.groups[2]);
                    } else {
                        ComputeDispatchDesc invalid;
                        invalid.kernel = kernel;
                        invalid.buffers = {buffer};
                        invalid.bufferAccesses = {ComputeBufferAccess::ReadWrite};
                        invalid.groupCountX = scenario.groups[0];
                        invalid.groupCountY = scenario.groups[1];
                        invalid.groupCountZ = scenario.groups[2];
                        if (path == 1) {
                            SubmitComputeBatch(host, {invalid});
                        } else {
                            auto valid = invalid;
                            valid.groupCountX = valid.groupCountY = valid.groupCountZ = 1;
                            SubmitComputeBatch(host, {{buffer, 0, {0, 0, 0, 0}}}, {valid, invalid});
                        }
                    }
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
            }
            const bool correct = rejected == !scenario.valid && queue.submissions == (scenario.valid ? 1u : 0u) &&
                                 device.live.empty();
            if (correct)
                ++passed;
            else {
                ++failed;
                std::cerr << "dispatch limits failed path=" << path << " groups=" << scenario.groups[0] << ','
                          << scenario.groups[1] << ',' << scenario.groups[2] << " submissions=" << queue.submissions
                          << '\n';
            }
        }
    }
    std::cout << "Compute dispatch limits: passed=" << passed << " failed=" << failed << '\n';
    return failed == 0;
}
} // namespace

int main()
{
    if (!VerifyDispatchLimits())
        return 1;
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

    // Abandoning a pending readback must neither wait nor release either GPU
    // allocation. The existing queue owns both until its completion boundary.
    TestQueue pendingQueue;
    pendingQueue.complete = false;
    ComputeHost pendingHost(device, pendingQueue);
    auto pendingSource = std::make_shared<ComputeBuffer>(pendingHost, ComputeBufferDesc{4});
    auto pendingRead = pendingSource->GetDataAsync(0, 4);
    std::weak_ptr<ComputeBuffer> sourceLifetime = pendingSource;
    pendingSource.reset();
    pendingRead.reset();
    assert(pendingQueue.waits == 0);
    assert(!sourceLifetime.expired() && device.live.size() == 2);
    pendingQueue.Collect();
    assert(!sourceLifetime.expired() && device.live.size() == 2);
    pendingQueue.complete = true;
    pendingQueue.Collect();
    assert(sourceLifetime.expired() && device.live.empty());
    assert(pendingQueue.waits == 0);

    TestQueue reuseQueue;
    ComputeHost reuseHost(device, reuseQueue);
    auto reuseSource = std::make_shared<ComputeBuffer>(reuseHost, ComputeBufferDesc{4});
    auto first = reuseSource->GetDataAsync(0, 16);
    first->Wait();
    first.reset();
    assert(reuseQueue.GetStatistics().stagingAllocationCount == 1);
    auto smaller = reuseSource->GetDataAsync(4, 8);
    smaller->Wait();
    assert(reuseQueue.GetStatistics().stagingAllocationCount == 1);
    const auto waits = reuseQueue.waits;
    // Completion alone is insufficient: the caller still owns this snapshot.
    auto concurrent = reuseSource->GetDataAsync(0, 16);
    assert(reuseQueue.GetStatistics().stagingAllocationCount == 2);
    assert(reuseQueue.waits == waits);
    concurrent->Wait();
    smaller.reset();
    concurrent.reset();
    assert(device.live.size() == 2); // Source plus one cached staging allocation.
    reuseSource.reset();
    assert(device.live.empty());
}
