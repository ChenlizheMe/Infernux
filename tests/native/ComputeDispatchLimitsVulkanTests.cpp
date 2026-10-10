#include <SDL3/SDL.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanComputeQueue.h>
#include <function/renderer/vk/VulkanQueueManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
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
    SubmissionTicket Submit(const Recorder &, std::shared_ptr<void> = {}) override
    {
        return {device, QueueRole::Compute, ++submissions};
    }
    void Wait(SubmissionTicket) override
    {
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

int main(int argc, char **argv)
{
    assert(argc == 2);
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    assert(file);
    const auto size = static_cast<size_t>(file.tellg());
    assert(size && size % sizeof(uint32_t) == 0);
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(spirv.data()), static_cast<std::streamsize>(size));
    assert(file);
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
    }
    queue.Destroy();
    queues.Destroy();
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
