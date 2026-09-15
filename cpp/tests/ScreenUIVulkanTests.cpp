#include <SDL3/SDL.h>
#include <function/renderer/gui/InxScreenUIRenderer.h>
#include <function/renderer/gui/InxTextLayout.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/vk/RenderGraph.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/scene/SceneManager.h>
#include <glm/gtc/type_ptr.hpp>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>

using namespace infernux;

int main(int argc, char **argv)
{
    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("UI Vulkan regression", 128, 128, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.appName = "UI Vulkan regression";
    // Correctness runs retain validation; performance runs opt out explicitly
    // so layer bookkeeping cannot be mistaken for engine/driver work.
    assert(argc == 1 || (argc == 2 && std::string(argv[1]) == "--performance"));
    config.enableValidationLayers = argc == 1;
    std::cout << "UI_BENCH_VALIDATION requested=" << config.enableValidationLayers << std::endl;
    assert(window && context.Initialize(window, config));
    auto &device = context.GetRhiDevice();
    rhi::SubmissionSerial epoch = 1;
    device.UseSubmissionSerials([&] { return epoch; });
    GpuRetirementQueue retirement;
    retirement.BindSerialSource([&] { return epoch; });

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    assert(vkCreateCommandPool(context.GetDevice(), &poolInfo, nullptr, &pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command;
    assert(vkAllocateCommandBuffers(context.GetDevice(), &allocate, &command) == VK_SUCCESS);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    assert(vkCreateFence(context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS);
    auto begin = [&] {
        assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        assert(vkBeginCommandBuffer(command, &info) == VK_SUCCESS);
    };
    auto finish = [&] {
        assert(vkEndCommandBuffer(command) == VK_SUCCESS);
        assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
        assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        retirement.Collect(epoch++);
    };

    // One white texel makes solid-quad/color readback independent of font art.
    // The real ImGui atlas still generates the text benchmark's glyph geometry.
    rhi::TextureDesc whiteDesc;
    whiteDesc.format = rhi::PixelFormat::RGBA8UNorm;
    whiteDesc.usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::TransferDestination;
    const auto white = device.CreateTexture(whiteDesc);
    rhi::TextureViewDesc viewDesc;
    viewDesc.texture = white;
    viewDesc.format = whiteDesc.format;
    const auto view = device.CreateTextureView(viewDesc);
    const auto sampler = device.CreateSampler({});
    rhi::BindingLayoutDesc layoutDesc;
    layoutDesc.entryCount = 1;
    layoutDesc.entries[0] = {0, rhi::BindingType::CombinedTextureSampler, rhi::ShaderStage::Fragment, 1};
    const auto layout = device.CreateBindingLayout(layoutDesc);
    rhi::BindGroupDesc groupDesc;
    groupDesc.layout = layout;
    groupDesc.textureCount = 1;
    groupDesc.textures[0] = {0, rhi::BindingType::CombinedTextureSampler, view, sampler};
    const auto group = device.CreateBindGroup(groupDesc);
    assert(white.IsValid() && view.IsValid() && sampler.IsValid() && group.IsValid());
    begin();
    VkImageMemoryBarrier image{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image.srcQueueFamilyIndex = image.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image.image = device.Resolve(white);
    image.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    image.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &image);
    const VkClearColorValue whiteValue{{1, 1, 1, 1}};
    vkCmdClearColorImage(command, image.image, image.newLayout, &whiteValue, 1, &image.subresourceRange);
    image.oldLayout = image.newLayout;
    image.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &image);
    finish();

    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(128, 128);
    io.DeltaTime = 1.f / 60;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
    io.Fonts->AddFontDefault();
    assert(io.Fonts->Build());
    io.Fonts->SetTexID(static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(device.Resolve(group))));
    ImGui::NewFrame();
    InxScreenUIRenderer renderer;
    assert(renderer.Initialize(context.GetDevice(), context.GetVmaAllocator(), VK_FORMAT_R8G8B8A8_UNORM,
                               VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, 4));
    renderer.SetRetirementQueue(&retirement);
    std::shared_ptr<InxScreenUIRenderer::CommandPacket> shutdownPacket;
    {
        rhi::RenderTextureDesc desc;
        desc.width = desc.height = 128;
        desc.colorFormat = rhi::PixelFormat::RGBA8UNorm;
        desc.depthFormat = rhi::PixelFormat::D32SFloat;
        rhi::RenderTexture target(device, "UI readback", desc);
        rhi::GraphicsRenderingSignature signature;
        signature.colorFormatCount = 1;
        signature.colorFormats[0] = desc.colorFormat;
        signature.depthFormat = desc.depthFormat;
        rhi::BufferDesc readback;
        readback.byteSize = 128 * 128 * 4;
        readback.usage = rhi::BufferUsageFlags::TransferDestination;
        readback.memory = rhi::BufferMemory::Readback;
        const auto output = device.CreateBuffer(readback);
        ScreenUIList list = ScreenUIList::Overlay;
        glm::mat4 camera(1.f);
        uint32_t frameSlot = 0;
        uint32_t cullingMask = 0xffffffffu;
        bool isolateSlots = false;
        double renderMs = 0;
        vk::RenderGraph graph;
        vk::ResourceHandle color;
        auto buildGraph = [&] {
            graph.Destroy();
            graph.Initialize(&context);
            const auto attachments = graph.ImportRenderTexture("UI readback", target.Acquire());
            graph.AddPass("draw-ui", [&](vk::PassBuilder &builder) {
                color = builder.WriteColor(attachments.color);
                if (list == ScreenUIList::World)
                    builder.WriteDepth(attachments.depth);
                builder.SetClearColor(0, 0, 0, 0);
                builder.SetClearDepth(1, 0);
                builder.SetDepthTest(true);
                builder.SetRenderArea(128, 128);
                return [&](vk::RenderContext &ctx) {
                    const auto start = std::chrono::steady_clock::now();
                    if (isolateSlots) {
                        // Both draws are recorded before either executes on the GPU.
                        // A shared writable buffer makes the red quad disappear.
                        renderer.BeginFrame(128, 128);
                        renderer.AddFilledRect(list, 8, 8, 56, 120, 1, 0, 0, 1);
                        renderer.Render(ctx.GetCommandBuffer(), list, 128, 128, 0);
                        renderer.BeginFrame(128, 128);
                        renderer.AddFilledRect(list, 72, 8, 120, 120, 0, 1, 0, 1);
                        renderer.Render(ctx.GetCommandBuffer(), list, 128, 128, 3);
                        return;
                    }
                    if (list == ScreenUIList::World)
                        renderer.RenderWorld(ctx.GetCommandBuffer(), 128, 128, camera, signature, frameSlot,
                                             cullingMask);
                    else
                        renderer.Render(ctx.GetCommandBuffer(), list, 128, 128, frameSlot);
                    renderMs =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                };
            });
            graph.AddTransferPass("read-ui", [&](vk::PassBuilder &builder) {
                builder.TransferRead(color);
                builder.TransferWrite(builder.ImportBuffer("pixels", output, readback.byteSize));
                builder.SetQueueRole(rhi::QueueRole::Graphics);
                builder.SetSideEffect();
                return [&](vk::RenderContext &ctx) {
                    VkBufferImageCopy copy{};
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.imageExtent = {128, 128, 1};
                    vkCmdCopyImageToBuffer(ctx.GetCommandBuffer(), device.Resolve(ctx.GetTextureHandle(color)),
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(output), 1, &copy);
                };
            });
            graph.AddPass("export-ui", [&](vk::PassBuilder &builder) {
                builder.Read(color);
                builder.SetSideEffect();
                return [](vk::RenderContext &) {};
            });
            assert(graph.Compile());
        };
        auto frame = [&] {
            begin();
            graph.Execute(command);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = device.Resolve(output);
            barrier.size = readback.byteSize;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                                 &barrier, 0, nullptr);
            finish();
            frameSlot = (frameSlot + 1) % 4;
        };
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            const uint64_t contentRevision = 100 + static_cast<uint64_t>(kind);
            assert(!renderer.BeginFrameCached(128, 128, contentRevision));
            for (int i = 0; i < 1000; ++i) {
                if (list == ScreenUIList::World) {
                    std::array<float, 16> pose{};
                    glm::mat4 matrix(1.f);
                    matrix[3].z = .5f;
                    std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                    renderer.BeginWorldElement(pose, 50, 20);
                }
                renderer.AddText(list, 0, 0, 100, 40, "UI 041", 1, 1, 1, 1, 0, 0, 18);
                if (list == ScreenUIList::World)
                    renderer.EndWorldElement();
            }
            std::vector<double> samples;
            for (int i = 0; i < 24; ++i) {
                assert(renderer.BeginFrameCached(128, 128, contentRevision));
                frame();
                if (i >= 4)
                    samples.push_back(renderMs);
            }
            std::sort(samples.begin(), samples.end());
            std::cout << "UI_NATIVE_BENCH list=" << int(list)
                      << " count=1000 render_record_p50_ms=" << samples[samples.size() / 2]
                      << " draws=" << renderer.GetLastSubmittedDrawCount(list) << std::endl;
            assert(renderer.GetLastSubmittedIndexCount(list) > 0);
            assert(renderer.GetLastSubmittedDrawCount(list) == 1);
            const auto stats = renderer.GetGeometryStats(list);
            assert(stats.preparations == 1 && stats.uploads == 4 && stats.uploadedBytes > 0);
        }
        // Changed content forces a real rebuild, unlike the static benchmark.
        // Distinct descriptor sets deliberately prevent texture batching.
        std::vector<rhi::BindGroupHandle> imageGroups;
        for (int i = 0; i < 64; ++i) {
            imageGroups.push_back(device.CreateBindGroup(groupDesc));
            assert(imageGroups.back().IsValid());
        }
        for (auto kind : {ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            for (bool textured : {false, true}) {
                std::vector<std::array<uint8_t, 128 * 128 * 4>> expectedFrames;
                for (bool retained : {false, true}) {
                    std::vector<std::shared_ptr<InxScreenUIRenderer::CommandPacket>> packets(1000);
                    const auto capturesBefore = renderer.GetGeometryStats(list).packetCaptures;
                    std::vector<double> buildSamples, renderSamples;
                    for (int iteration = 0; iteration < 20; ++iteration) {
                        const auto started = std::chrono::steady_clock::now();
                        renderer.BeginFrame(128, 128);
                        for (int i = 0; i < 1000; ++i) {
                            if (retained && packets[i] && i != iteration && i != iteration - 1)
                                continue;
                            if (retained)
                                renderer.BeginCommandPacket();
                            if (list == ScreenUIList::World) {
                                glm::mat4 matrix(1.f);
                                matrix[3].z = .5f;
                                std::array<float, 16> pose{};
                                std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
                                renderer.BeginWorldElement(pose, 50, 20);
                            }
                            if (textured) {
                                const auto textureId = static_cast<uint64_t>(
                                    reinterpret_cast<uintptr_t>(device.Resolve(imageGroups[i % imageGroups.size()])));
                                renderer.AddImage(list, textureId, 0, 0, 100, 40);
                            } else {
                                renderer.AddText(list, 0, 0, 100, 40, i == iteration ? "updated UI" : "UI 041", 1, 1, 1,
                                                 1, 0, 0, 18);
                            }
                            if (list == ScreenUIList::World)
                                renderer.EndWorldElement();
                            if (retained)
                                packets[i] = renderer.EndCommandPacket();
                        }
                        if (retained)
                            renderer.AppendCommandPackets(packets);
                        const double buildMs =
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                                .count();
                        frame();
                        std::array<uint8_t, 128 * 128 * 4> observed{};
                        assert(device.ReadBuffer(output, 0, observed.data(), observed.size()));
                        if (retained)
                            assert(observed == expectedFrames[iteration]);
                        else
                            expectedFrames.push_back(observed);
                        if (iteration >= 4) {
                            buildSamples.push_back(buildMs);
                            renderSamples.push_back(renderMs);
                        }
                    }
                    std::sort(buildSamples.begin(), buildSamples.end());
                    std::sort(renderSamples.begin(), renderSamples.end());
                    std::cout << "UI_DYNAMIC_NATIVE_BENCH list=" << int(list) << " textured=" << textured
                              << " retained=" << retained
                              << " count=1000 build_p50_ms=" << buildSamples[buildSamples.size() / 2]
                              << " record_p50_ms=" << renderSamples[renderSamples.size() / 2]
                              << " draws=" << renderer.GetLastSubmittedDrawCount(list) << std::endl;
                    assert(renderer.GetLastSubmittedDrawCount(list) == (textured ? 1000u : 1u));
                    assert(renderer.GetGeometryStats(list).packetCaptures - capturesBefore == (retained ? 1038u : 0u));
                }
            }
        }
        for (auto imageGroup : imageGroups)
            device.Release(imageGroup);
        list = ScreenUIList::Overlay;
        buildGraph();
        renderer.BeginFrame(128, 128);
        renderer.AddFilledRect(list, 16, 16, 112, 112, 1, 0, 0, 1);
        frame();
        std::array<uint8_t, 128 * 128 * 4> pixels{};
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        const size_t center = (64 * 128 + 64) * 4;
        assert(pixels[center] == 255 && pixels[center + 1] == 0 && pixels[center + 3] == 255);
        renderer.BeginFrame(128, 128);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] == 0 && pixels[center + 3] == 0);
        isolateSlots = true;
        frame();
        isolateSlots = false;
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        const size_t left = (64 * 128 + 32) * 4;
        const size_t right = (64 * 128 + 96) * 4;
        assert(pixels[left] == 255 && pixels[left + 1] == 0);
        assert(pixels[right] == 0 && pixels[right + 1] == 255);

        list = ScreenUIList::World;
        buildGraph();
        renderer.BeginFrame(128, 128);
        auto worldQuad = [&](float z, float red, float blue, float alpha, uint32_t layer) {
            glm::mat4 matrix(1.f);
            matrix[3].z = z;
            std::array<float, 16> pose{};
            std::copy_n(glm::value_ptr(matrix), 16, pose.begin());
            renderer.BeginWorldElement(pose, 50, 50, layer);
            renderer.AddFilledRect(list, 0, 0, 100, 100, red, 0, blue, alpha);
            renderer.EndWorldElement();
        };
        worldQuad(.2f, 1, 0, .5f, 1); // submitted near before far; sort must reverse it
        worldQuad(.8f, 0, 1, 1, 2);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 127 && pixels[center] <= 129 && pixels[center + 2] >= 126);
        const auto prepared = renderer.GetGeometryStats(list).preparations;
        // Reverse camera depth without changing the world geometry.
        camera[2][2] = -1;
        camera[3][2] = 1;
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] == 0 && pixels[center + 2] == 255);
        cullingMask = 1;
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 127 && pixels[center + 2] == 0);
        assert(renderer.GetGeometryStats(list).preparations == prepared);
        cullingMask = 0;
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center + 3] == 0 && renderer.GetLastSubmittedDrawCount(list) == 0);
        camera = glm::mat4(1.f);
        cullingMask = 0xffffffffu;
        renderer.BeginFrame(128, 128);
        renderer.PushClipRect(list, 0, 0, 50, 100);
        worldQuad(.5f, 1, 0, 1, 1);
        renderer.PopClipRect(list);
        renderer.PushClipRect(list, 50, 0, 100, 100);
        worldQuad(.5f, 0, 1, .5f, 1);
        renderer.PopClipRect(list);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        const size_t innerLeft = (64 * 128 + 48) * 4;
        const size_t innerRight = (64 * 128 + 80) * 4;
        // World UI is not constrained by screen/canvas clip rectangles. Both
        // elements therefore remain visible even though their authored ImGui
        // clip regions are disjoint; only scene scissor/depth apply.
        for (const size_t sample : {innerLeft, innerRight}) {
            assert(pixels[sample] >= 126 && pixels[sample] <= 129);
            assert(pixels[sample + 2] >= 126 && pixels[sample + 2] <= 129);
            assert(pixels[sample + 3] == 255);
        }
        // Ignored canvas clips cannot split otherwise compatible world draws.
        assert(renderer.GetLastSubmittedDrawCount(list) == 1);
        // Stable equal-depth order and alpha blending must survive batching.
        renderer.BeginFrame(128, 128);
        worldQuad(.5f, 1, 0, 1, 1);
        worldQuad(.5f, 0, 1, .5f, 1);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 126 && pixels[center + 2] >= 127 && pixels[center + 3] == 255);
        assert(renderer.GetLastSubmittedDrawCount(list) == 1);
        renderer.SetEnabled(false);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center + 3] == 0);
        renderer.SetEnabled(true);
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center + 3] == 255);

        // A different texture descriptor is a hard batch boundary.
        const auto secondGroup = device.CreateBindGroup(groupDesc);
        renderer.BeginFrame(128, 128);
        worldQuad(.5f, 1, 0, 1, 1);
        glm::mat4 poseMatrix(1.f);
        poseMatrix[3].z = .5f;
        std::array<float, 16> pose{};
        std::copy_n(glm::value_ptr(poseMatrix), 16, pose.begin());
        renderer.BeginWorldElement(pose, 50, 50);
        renderer.AddImage(list, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device.Resolve(secondGroup))), 0, 0,
                          100, 100, 0, 0, 1, 1, 0, 0, 1, .5f);
        renderer.EndWorldElement();
        frame();
        assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
        assert(pixels[center] >= 126 && pixels[center + 2] >= 127);
        assert(renderer.GetLastSubmittedDrawCount(list) == 2);
        renderer.BeginFrame(128, 128);
        device.Release(secondGroup);

        // Crossing ImGui's 16-bit index range requires distinct base vertices.
        for (int i = 0; i < 4000; ++i) {
            renderer.BeginWorldElement(pose, 50, 50);
            renderer.AddText(list, 0, 0, 100, 100, "UI 041", 1, 1, 1, 1, 0, 0, 18);
            renderer.EndWorldElement();
        }
        frame();
        assert(renderer.GetLastSubmittedDrawCount(list) == (sizeof(ImDrawIdx) == 2 ? 2 : 1));

        // Compare retained and fresh geometry byte-for-byte through real GPU
        // readback: clip, rotation/mirror, HDR tint, textures, world layers,
        // alpha order, mixed direct/retained submission and large index ranges.
        using Packet = std::shared_ptr<InxScreenUIRenderer::CommandPacket>;
        for (auto kind : {ScreenUIList::Camera, ScreenUIList::Overlay, ScreenUIList::World}) {
            list = kind;
            buildGraph();
            const auto decorate = [&](int item, bool large = false) {
                glm::mat4 matrix(1.f);
                matrix[3] = glm::vec4(float(item) * .05f, .1f, item == 1 ? .7f : .4f, 1.f);
                std::array<float, 16> transform{};
                std::copy_n(glm::value_ptr(matrix), 16, transform.begin());
                if (list == ScreenUIList::World)
                    renderer.BeginWorldElement(transform, 64, 64, 1u << item);
                renderer.PushClipRect(list, 12, 17, 116, 107);
                renderer.AddFilledRect(list, 8, 8, 112, 114, 1.7f, .2f, .4f, .4f, 8, 23, false, true);
                renderer.AddImage(list, static_cast<uint64_t>(io.Fonts->TexRef.GetTexID()), 10, 21, 82, 101, 0, 0, 1, 1,
                                  .2f, .6f, 1, .6f, -18, true, false, 5);
                const int repeats = large ? 3500 : 1;
                for (int n = 0; n < repeats; ++n)
                    renderer.AddText(list, 13, 20, 109, 106, "Alpha\nBeta UI", .2f, 1.4f, .6f, .2f, .5f, .5f, 18, 72,
                                     13, true, false, "", 1.2f, 2, true);
                renderer.PopClipRect(list);
                if (list == ScreenUIList::World)
                    renderer.EndWorldElement();
            };
            for (bool large : {false, true}) {
                renderer.BeginFrame(128, 128);
                decorate(0);
                decorate(1, large);
                decorate(2);
                frame();
                std::array<uint8_t, 128 * 128 * 4> expected{};
                assert(device.ReadBuffer(output, 0, expected.data(), expected.size()));
                const auto indices = renderer.GetLastSubmittedIndexCount(list);
                renderer.BeginFrame(128, 128);
                renderer.BeginCommandPacket();
                decorate(1, large);
                Packet retained = renderer.EndCommandPacket();
                shutdownPacket = retained;
                decorate(0);
                renderer.AppendCommandPackets({retained});
                decorate(2);
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                assert(pixels == expected);
                assert(renderer.GetLastSubmittedIndexCount(list) == indices);
                if (list == ScreenUIList::World) {
                    cullingMask = 2;
                    frame();
                    std::array<uint8_t, 128 * 128 * 4> masked{};
                    assert(device.ReadBuffer(output, 0, masked.data(), masked.size()));
                    renderer.BeginFrame(128, 128);
                    decorate(0);
                    decorate(1, large);
                    decorate(2);
                    frame();
                    assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                    assert(pixels == masked);
                    cullingMask = 0xffffffffu;
                }
                renderer.BeginFrame(128, 128);
                decorate(0);
                renderer.AppendCommandPackets({retained});
                decorate(2);
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                assert(pixels == expected);
            }
        }
        // Scene-bound packets keep local glyph/shape geometry immutable.
        // Pose/layer is sampled at publication, including uncommitted frame
        // cache writes. Compare with explicit fresh geometry through Vulkan.
        list = ScreenUIList::World;
        buildGraph();
        auto *uiScene = SceneManager::Instance().CreateScene("Retained world UI");
        auto *parent = uiScene->CreateGameObject("Parent");
        auto *object = uiScene->CreateGameObject("Animated UI");
        object->SetParent(parent, false);
        object->GetTransform()->SetLocalPosition(.1f, .1f, .4f);
        const auto drawLocal = [&] {
            renderer.AddFilledRect(list, 10, 20, 85, 70, .3f, .8f, 1.2f, .7f, 7);
            renderer.AddText(list, 10, 20, 85, 70, "UI", 1, 1, 1, .8f, .5f, .5f, 18);
        };
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.BeginWorldObject(object, 50, 50);
        drawLocal();
        renderer.EndWorldElement();
        auto animated = renderer.EndCommandPacket();
        const auto captureCount = renderer.GetGeometryStats(list).packetCaptures;
        auto &transforms = TransformECSStore::Instance();
        std::array<uint8_t, 128 * 128 * 4> previousPose{};
        for (int step = 0; step < 4; ++step) {
            parent->GetTransform()->SetLocalEulerAngles(8, float(step * 13), 7);
            parent->GetTransform()->SetLocalScale(1.5f, .7f, 1.f);
            // Its own scale must not stretch UI, but parent scale affects its position.
            object->GetTransform()->SetLocalScale(3.f, .2f, 2.f);
            transforms.BeginFrameCache();
            object->GetTransform()->SetPosition(float(step) * .12f, .05f, .4f);
            object->SetLayer(step);
            for (bool visible : {true, false}) {
                cullingMask = visible ? (1u << step) : (1u << (step + 1));
                renderer.BeginFrame(128, 128);
                renderer.AppendCommandPackets({animated});
                frame();
                std::array<uint8_t, 128 * 128 * 4> retained{};
                assert(device.ReadBuffer(output, 0, retained.data(), retained.size()));
                glm::mat4 current = glm::mat4_cast(object->GetTransform()->GetRotation());
                current[3] = glm::vec4(object->GetTransform()->GetPosition(), 1);
                std::array<float, 16> explicitPose{};
                std::copy_n(glm::value_ptr(current), 16, explicitPose.begin());
                renderer.BeginFrame(128, 128);
                renderer.BeginWorldElement(explicitPose, 50, 50, 1u << step);
                drawLocal();
                renderer.EndWorldElement();
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                assert(pixels == retained);
                assert(renderer.GetGeometryStats(list).packetCaptures == captureCount);
                if (visible) {
                    assert(pixels != previousPose);
                    previousPose = pixels;
                } else {
                    assert(std::all_of(pixels.begin(), pixels.end(), [](uint8_t value) { return value == 0; }));
                }
            }
            (void)transforms.EndFrameCache();
        }
        cullingMask = 0xffffffffu;
        // Screen-bound packets resolve Transform pose at publication too.
        // A pose-only change must reach the GPU even when geometry revision
        // is unchanged.
        list = ScreenUIList::Overlay;
        buildGraph();
        auto *screenObject = uiScene->CreateGameObject("Animated screen UI");
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.BeginScreenObject(screenObject, list, 40, 40);
        renderer.AddFilledRect(list, 8, 8, 72, 56, 1, 0, 0, 1);
        renderer.EndScreenObject();
        auto screenPacket = renderer.EndCommandPacket();
        frame();
        std::array<uint8_t, 128 * 128 * 4> screenBefore{};
        assert(device.ReadBuffer(output, 0, screenBefore.data(), screenBefore.size()));
        screenObject->GetTransform()->SetPosition(.5f, 0.0f, 0.0f);
        renderer.BeginFrame(128, 128);
        renderer.AppendCommandPackets({screenPacket});
        frame();
        std::array<uint8_t, 128 * 128 * 4> screenAfter{};
        assert(device.ReadBuffer(output, 0, screenAfter.data(), screenAfter.size()));
        assert(screenAfter != screenBefore);
        screenObject->GetTransform()->SetLocalEulerAngles(0.0f, 0.0f, 25.0f);
        screenObject->GetTransform()->SetLocalScale(1.5f, 0.75f, 1.0f);
        renderer.BeginFrame(128, 128);
        renderer.AppendCommandPackets({screenPacket});
        frame();
        std::array<uint8_t, 128 * 128 * 4> screenAfterRotateScale{};
        assert(device.ReadBuffer(output, 0, screenAfterRotateScale.data(), screenAfterRotateScale.size()));
        assert(screenAfterRotateScale != screenAfter);
        screenPacket.reset();

        // No owning scene pointer is retained; recycling a Transform slot
        // must reject an expired packet rather than attach it to a new object.
        uiScene->DestroyGameObject(object);
        uiScene->ProcessPendingDestroys();
        auto *replacement = uiScene->CreateGameObject("Replacement");
        (void)replacement;
        renderer.BeginFrame(128, 128);
        bool expired = false;
        try {
            renderer.AppendCommandPackets({animated});
        } catch (const std::runtime_error &) {
            expired = true;
        }
        assert(expired);
        animated.reset();

        // Moving 1000 scene objects must not regenerate 1000 text packets.
        // Both policies use the same native backend and render identical frames.
        auto *crowdRoot = uiScene->CreateGameObject("Animated UI crowd");
        std::vector<GameObject *> crowd;
        for (int index = 0; index < 1000; ++index) {
            auto *label = uiScene->CreateGameObject("Label");
            label->SetParent(crowdRoot, false);
            label->GetTransform()->SetLocalPosition(0, 0, .5f);
            crowd.push_back(label);
        }
        std::vector<std::array<uint8_t, 128 * 128 * 4>> poseFrames;
        for (bool retain : {false, true}) {
            std::vector<Packet> packets(1000);
            std::vector<double> buildTimes, recordTimes;
            const auto captures = renderer.GetGeometryStats(list).packetCaptures;
            for (int iteration = 0; iteration < 12; ++iteration) {
                crowdRoot->GetTransform()->SetLocalEulerAngles(0, float(iteration * 2), 0);
                const auto started = std::chrono::steady_clock::now();
                renderer.BeginFrame(128, 128);
                if (!retain || iteration == 0) {
                    for (int index = 0; index < 1000; ++index) {
                        renderer.BeginCommandPacket();
                        renderer.BeginWorldObject(crowd[index], 50, 20);
                        renderer.AddText(list, 0, 0, 100, 40, "UI 041", 1, 1, 1, 1, 0, 0, 18);
                        renderer.EndWorldElement();
                        packets[index] = renderer.EndCommandPacket();
                    }
                }
                renderer.AppendCommandPackets(packets);
                const double buildTime =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
                frame();
                assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
                if (retain)
                    assert(pixels == poseFrames[iteration]);
                else
                    poseFrames.push_back(pixels);
                if (iteration >= 4) {
                    buildTimes.push_back(buildTime);
                    recordTimes.push_back(renderMs);
                }
            }
            std::sort(buildTimes.begin(), buildTimes.end());
            std::sort(recordTimes.begin(), recordTimes.end());
            std::cout << "UI_WORLD_POSE_NATIVE_BENCH count=1000 retained=" << retain
                      << " build_p50_ms=" << buildTimes[buildTimes.size() / 2]
                      << " record_p50_ms=" << recordTimes[recordTimes.size() / 2] << std::endl;
            assert(renderer.GetGeometryStats(list).packetCaptures - captures == (retain ? 1000u : 12000u));
        }
        renderer.BeginFrame(128, 128);
        renderer.BeginCommandPacket();
        renderer.BeginWorldElement(pose, 50, 50);
        renderer.AbortCommandPacket();
        renderer.BeginCommandPacket();
        const auto empty = renderer.EndCommandPacket();
        renderer.AppendCommandPackets({empty});
        assert(!renderer.HasCommands(ScreenUIList::World));
        assert(!renderer.BeginFrameCached(128, 128, 7788));
        assert(renderer.BeginFrameCached(128, 128, 7788));
        const auto fontEpoch = renderer.GetCommandPacketEpoch();
        textlayout::ClearFontCache();
        assert(renderer.GetCommandPacketEpoch() != fontEpoch);
        assert(!renderer.BeginFrameCached(128, 128, 7788));
        graph.Destroy();
        device.Release(output);
    }
    renderer.Destroy();
    ImGui::EndFrame();
    ImGui::DestroyContext();
    shutdownPacket.reset(); // CPU packets have no dangling ImGui/GPU owners.
    device.Release(group);
    device.Release(layout);
    device.Release(sampler);
    device.Release(view);
    device.Release(white);
    device.CollectDescriptorRetirements(epoch);
    device.CollectResourceRetirements(epoch);
    retirement.FlushAll();
    vkDestroyFence(context.GetDevice(), fence, nullptr);
    vkDestroyCommandPool(context.GetDevice(), pool, nullptr);
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::cout << "PASS UI readback, retained geometry, frame-slot isolation and camera depth/layer order\n";
}
