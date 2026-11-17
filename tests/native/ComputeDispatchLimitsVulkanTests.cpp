#include "SlangShaderFixture.h"
#include <SDL3/SDL.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanComputeQueue.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace infernux;
using namespace infernux::rhi;

// Exercise preparation with real hardware capabilities, pipeline and buffers,
// without executing a possibly enormous maximum-sized workload on the GPU.
class PreparationQueue final : public ComputeQueue
{
  public:
    DeviceId device;
    uint64_t submissions = 0;
    explicit PreparationQueue(DeviceId id) : device(id)
    {
    }
    SubmissionTicket Submit(const Recorder &, ComputeSubmissionResources resources = {}) override
    {
        if (resources.receipt)
            resources.receipt->complete = true;
        return {device, QueueRole::Compute, ++submissions};
    }
    std::optional<SubmissionTicket> TrySubmit(const Recorder &record, const ComputeSubmissionResources &resources = {},
                                              const Admission &admit = {}) override
    {
        if (admit && !admit())
            return std::nullopt;
        return Submit(record, resources);
    }
    void Wait(SubmissionTicket) override
    {
    }
    bool WaitFor(SubmissionTicket, uint64_t) override
    {
        return true;
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
    uint64_t GetRetainedUploadBytes() const noexcept override
    {
        return 0;
    }
    SubmissionTicket GetCapacityPredecessor() const noexcept override
    {
        return {}; // This test queue has no bounded submission pool.
    }
};

static void Dispatch(ComputeHost &host, const std::shared_ptr<ComputeKernel> &kernel,
                     const std::shared_ptr<ComputeBuffer> &buffer, std::array<uint32_t, 3> groups, bool batch)
{
    if (!batch) {
        kernel->Dispatch({buffer}, nullptr, 0, groups[0], groups[1], groups[2]);
        return;
    }
    ComputeDispatchDesc dispatch;
    dispatch.kernel = kernel;
    dispatch.buffers = {buffer};
    dispatch.bufferAccesses = {ComputeBufferAccess::Write};
    dispatch.groupCountX = groups[0];
    dispatch.groupCountY = groups[1];
    dispatch.groupCountZ = groups[2];
    assert(SubmitComputeBatch(host, {dispatch}).IsValid());
}

static const ShaderEntryPointCode &Entry(const ShaderTargetCode &program, const std::string &name)
{
    const auto found = std::find_if(program.entries.begin(), program.entries.end(),
                                    [&](const auto &entry) { return entry.name == name; });
    assert(found != program.entries.end());
    return *found;
}

static std::vector<uint32_t> Words(const ShaderEntryPointCode &entry)
{
    assert(!entry.code.empty() && entry.code.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> words(entry.code.size() / sizeof(uint32_t));
    std::memcpy(words.data(), entry.code.data(), entry.code.size());
    return words;
}

static std::shared_ptr<ComputeKernel> ReflectedKernel(ComputeHost &host, const ShaderTargetCode &program,
                                                      const std::string &name)
{
    const auto &entry = Entry(program, name);
    std::vector<ComputeBufferBinding> layout;
    for (const auto &resource : program.resources) {
        const std::array location{resource.set, resource.binding.binding};
        if (std::find(entry.bindings.begin(), entry.bindings.end(), location) != entry.bindings.end()) {
            assert(resource.set == 0);
            layout.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
        }
    }
    std::sort(layout.begin(), layout.end(), [](const auto &left, const auto &right) { return left.slot < right.slot; });
    const auto code = Words(entry);
    return std::make_shared<ComputeKernel>(host, ShaderModuleDesc::FromSpirV(code.data(), code.size()), entry.name,
                                           std::move(layout), 0);
}

static void TestDeviceCountedPayload(ComputeHost &host, const ShaderTargetCode &program)
{
    constexpr uint32_t capacity = 67, padding = 4, bytes = capacity * sizeof(uint32_t);
    constexpr uint32_t sentinel = 999999;
    constexpr auto scalar = ComputeScalarType::UInt32;
    constexpr auto read = ComputeBufferAccess::Read, write = ComputeBufferAccess::Write;
    const ComputeWorkRange request{0, 0, sizeof(uint32_t)}, count{1, sizeof(uint32_t), sizeof(uint32_t)};
    const ComputeWorkRange payload{2, padding * sizeof(uint32_t), bytes}, result{3, 0, bytes};
    const std::vector<ComputeWorkSlot> slots{
        {{1, scalar}, {}}, {{3, scalar}, {}}, {{capacity + padding, scalar}, {}}, {{capacity, scalar}, {}}};
    const std::vector<ComputeWorkPort> ports{{"request", request, true}, {"result", result, false}};
    ComputeWorkDispatch makeCount, reset, makePayload, readPayload;
    makeCount.kernel = ReflectedKernel(host, program, "make_count");
    makeCount.bindings = {{request, read}, {count, write}};
    makeCount.completeWrites = {{1, count.offset, count.byteSize}};
    reset.kernel = ReflectedKernel(host, program, "reset_result");
    reset.bindings = {{result, write}};
    reset.completeWrites = {{0, result.offset, result.byteSize}};
    const auto groupsFor = [&](const std::string &entry) {
        const auto width = Entry(program, entry).workgroupSize[0];
        assert(width != 0);
        return (capacity + width - 1) / width;
    };
    reset.groupCountX = groupsFor("reset_result");
    makePayload.kernel = ReflectedKernel(host, program, "make_payload");
    makePayload.bindings = {{count, read}, {payload, write}};
    makePayload.groupCountX = groupsFor("make_payload");
    makePayload.prefix = ComputePrefixAccess{{0, count.offset, count.byteSize},
                                             {1, payload.offset, payload.byteSize},
                                             write,
                                             Entry(program, "make_payload").workgroupSize};
    readPayload.kernel = ReflectedKernel(host, program, "read_payload");
    readPayload.bindings = {{count, read}, {payload, read}, {result, write}};
    readPayload.groupCountX = groupsFor("read_payload");
    readPayload.prefix = ComputePrefixAccess{{0, count.offset, count.byteSize},
                                             {1, payload.offset, payload.byteSize},
                                             read,
                                             Entry(program, "read_payload").workgroupSize};
    const std::vector<ComputeWorkDispatch> dispatches{makeCount, reset, makePayload, readPayload};
    const auto recipe = std::make_shared<ComputeWorkRecipe>(host, slots, ports, dispatches);
    const auto instance = recipe->Instantiate();
    assert(instance);
    const auto &buffers = instance->GetBuffers();
    const auto verify = [&](const auto &output, uint32_t rows) {
        const auto data = output->GetData(0, bytes);
        assert(data.size() == bytes);
        std::array<uint32_t, capacity> words;
        std::memcpy(words.data(), data.data(), data.size());
        for (uint32_t row = 0; row < capacity; ++row)
            assert(words[row] == (row < rows ? (row * 3 + 7) * 2 : sentinel));
    };
    std::shared_ptr<ComputeBuffer> fullResult;
    for (const auto &[selector, rows] :
         std::array<std::pair<uint32_t, uint32_t>, 4>{{{34, capacity}, {1, 1}, {0, 0}, {35, capacity}}}) {
        std::vector<uint8_t> value(sizeof(selector));
        std::memcpy(value.data(), &selector, sizeof(selector));
        const auto ticket =
            instance->PrepareBatch({{buffers[request.slot], request.offset, std::move(value)}})->Submit();
        assert(ticket.IsValid() && buffers[result.slot]->GetLastWriteSubmission() == ticket);
        assert(buffers[result.slot]->HasInitializedRange(result.offset, result.byteSize));
        assert(!buffers[payload.slot]->HasInitializedRange(payload.offset, payload.byteSize));
        verify(buffers[result.slot], rows);
        if (!fullResult)
            fullResult = buffers[result.slot]->Freeze();
    }
    assert(fullResult->GetBuffer() != buffers[result.slot]->GetBuffer());
    verify(fullResult, capacity);

    const auto rejected = [&](std::vector<ComputeWorkPort> badPorts, std::vector<ComputeWorkDispatch> badDispatches,
                              const std::string &diagnostic) {
        try {
            ComputeWorkRecipe invalid(host, slots, std::move(badPorts), std::move(badDispatches));
        } catch (const std::invalid_argument &error) {
            return std::string(error.what()).find(diagnostic) != std::string::npos;
        }
        return false;
    };
    auto exported = ports;
    exported.push_back({"payload", payload, false});
    assert(rejected(std::move(exported), dispatches, "lacks a complete writer"));
    assert(rejected(ports, {makeCount, reset, makePayload, makeCount, readPayload}, "matching current count"));
    auto incomplete = dispatches;
    --incomplete[2].groupCountX;
    assert(rejected(ports, std::move(incomplete), "does not cover the payload capacity"));
    std::cout << "GPU counted payload correct: full, one row, empty, clamped and retained result\n";
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    const auto fixtures = infernux::test::CompileSlangFixture(argv[1]);
    const auto spirv = Words(Entry(fixtures, "main"));
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("Compute dispatch limits", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.enableValidationLayers = true;
    assert(context.Initialize(window, config));
    auto &device = context.GetRhiDevice();
    const auto &limits = device.GetCapabilities().limits;
    assert(limits.maxPushConstantBytes > 0 && limits.maxBindingLayouts > 0);
    const auto &portable = device.GetCapabilities().portable;
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context.GetPhysicalDevice(), &physical);
    // A byte range must not accidentally publish the number of binding slots.
    assert(portable.maxStorageBufferBinding == physical.limits.maxStorageBufferRange);
    assert(portable.maxStorageBufferBinding >= 128ull * 1024ull * 1024ull);
    assert(portable.maxWorkgroupInvocations == physical.limits.maxComputeWorkGroupInvocations);
    for (size_t axis = 0; axis < 3; ++axis)
        assert(portable.maxWorkgroupSize[axis] == physical.limits.maxComputeWorkGroupSize[axis]);
    const auto &enabled = device.GetVulkanFeatures();
    assert(portable.shaderInt16 == enabled.shaderInt16.IsEnabled());
    assert(portable.shaderInt64 == enabled.shaderInt64.IsEnabled());
    assert(portable.shaderFloat64 == enabled.shaderFloat64.IsEnabled());
    assert(device.GetCapabilities().BackendName() == "vulkan");
    {
        const auto shader = device.CreateShaderModule(ShaderModuleDesc::FromSpirV(spirv.data(), spirv.size()));
        assert(shader.IsValid());
        BindingLayoutDesc layoutDesc;
        layoutDesc.entries[0] = {0, BindingType::StorageBuffer, ShaderStage::Compute, 1};
        layoutDesc.entryCount = 1;
        const auto layout = device.CreateBindingLayout(layoutDesc);
        assert(layout.IsValid());
        ComputePipelineDesc compute;
        compute.computeShader = shader;
        compute.bindingLayouts.fill(layout);
        compute.bindingLayoutCount = 1;
        const auto baseline = device.CreateComputePipeline(compute);
        assert(baseline.IsValid());
        device.Release(baseline);
        compute.pushConstantBytes = limits.maxPushConstantBytes + 4;
        assert(!device.CreateComputePipeline(compute).IsValid());
        compute.pushConstantBytes = 0;
        compute.bindingLayoutCount = limits.maxBindingLayouts + 1;
        assert(!device.CreateComputePipeline(compute).IsValid());

        // Capability rejection must happen before shader-stage or native
        // layout creation. These shared handles never reach a graphics API.
        GraphicsPipelineDesc graphics;
        graphics.vertexShader = graphics.fragmentShader = shader;
        graphics.useDynamicRendering = true;
        graphics.renderingSignature.colorFormatCount = graphics.colorTargetCount = 1;
        graphics.renderingSignature.colorFormats[0] = graphics.colorTargets[0].format = PixelFormat::RGBA8UNorm;
        graphics.depth.testEnabled = graphics.depth.writeEnabled = false;
        graphics.bindingLayouts.fill(layout);
        graphics.bindingLayoutCount = 1;
        graphics.pushConstantStages = ShaderStage::Vertex;
        graphics.pushConstantBytes = limits.maxPushConstantBytes + 4;
        assert(graphics.HasValidRenderingContract());
        assert(!device.CreateGraphicsPipeline(graphics).IsValid());
        graphics.pushConstantBytes = 0;
        graphics.bindingLayoutCount = limits.maxBindingLayouts + 1;
        assert(!device.CreateGraphicsPipeline(graphics).IsValid());
        device.Release(layout);
        device.Release(shader);
        std::cout << "Pipeline layout limits rejected before Vulkan: sets=" << limits.maxBindingLayouts
                  << " push-bytes=" << limits.maxPushConstantBytes << '\n';
    }
    const std::array<uint32_t, 3> maxima{limits.maxComputeWorkgroupCount[0], limits.maxComputeWorkgroupCount[1],
                                         limits.maxComputeWorkgroupCount[2]};
    std::cout << "Actual workgroup-count limits: " << maxima[0] << ',' << maxima[1] << ',' << maxima[2] << '\n';
    for (const auto limit : maxima)
        assert(limit > 0);
    for (const bool batch : {false, true}) {
        PreparationQueue queue(device.GetDeviceId());
        ComputeHost host(device, queue);
        auto buffer = std::make_shared<ComputeBuffer>(host, ComputeBufferDesc{24, ComputeScalarType::UInt32});
        auto kernel = std::make_shared<ComputeKernel>(host, spirv.data(), spirv.size(), 1, 0);
        for (size_t axis = 0; axis < 3; ++axis) {
            if (maxima[axis] == std::numeric_limits<uint32_t>::max())
                continue; // There is no representable positive value above this axis limit.
            std::array<uint32_t, 3> groups{1, 1, 1};
            groups[axis] = maxima[axis] + 1;
            bool rejected = false;
            try {
                Dispatch(host, kernel, buffer, groups, batch);
            } catch (const std::invalid_argument &error) {
                const std::string diagnostic = error.what();
                rejected = diagnostic.find(std::string(1, "XYZ"[axis]) + "=") != std::string::npos &&
                           diagnostic.find(std::to_string(maxima[axis])) != std::string::npos;
            }
            assert(rejected && queue.submissions == 0);
            assert(kernel->GetCachedBindingGroupCount() == 0);
        }
        Dispatch(host, kernel, buffer, maxima, batch);
        assert(queue.submissions == 1);
    }

    vk::VulkanQueueManager queues;
    assert(queues.Initialize(context, 2));
    vk::VulkanComputeQueue queue;
    assert(queue.Initialize(context, queues));
    {
        ComputeHost host(device, queue);
        for (const bool batch : {false, true}) {
            auto buffer = std::make_shared<ComputeBuffer>(host, ComputeBufferDesc{24, ComputeScalarType::UInt32});
            auto kernel = std::make_shared<ComputeKernel>(host, spirv.data(), spirv.size(), 1, 0);
            Dispatch(host, kernel, buffer, {4, 3, 2}, batch);
            const auto bytes = buffer->GetData(0, 24 * sizeof(uint32_t));
            assert(bytes.size() == 24 * sizeof(uint32_t));
            for (uint32_t index = 0; index < 24; ++index) {
                uint32_t value;
                std::memcpy(&value, bytes.data() + index * sizeof(uint32_t), sizeof(value));
                assert(value == index * 3 + 7);
            }
            std::cout << "GPU readback correct: path=" << (batch ? "batch" : "direct") << " groups=4,3,2\n";
        }
        TestDeviceCountedPayload(host, fixtures);
    }
    queue.Destroy();
    queues.Destroy();
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
