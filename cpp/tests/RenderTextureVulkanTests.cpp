#include <SDL3/SDL.h>
#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/SceneRenderGraph.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/shader/ShaderReflection.h>
#include <function/renderer/vk/RenderGraph.h>
#include <function/renderer/vk/RhiVulkanTypes.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace infernux;

static void CheckDepthSamplingDeclaration(vk::VkDeviceContext &context)
{
    for (const auto samples : {rhi::SampleCount::One, rhi::SampleCount::Four}) {
        for (const bool sampled : {false, true}) {
            rhi::RenderTextureDesc desc;
            desc.width = desc.height = 16;
            desc.depthFormat = rhi::PixelFormat::D32SFloat;
            desc.samples = samples;
            desc.sampledDepth = sampled;
            rhi::RenderTexture target(context.GetRhiDevice(), "depth-contract", desc);
            vk::RenderGraph graph;
            graph.Initialize(&context);
            const auto attachments = graph.ImportRenderTexture("depth-contract", target.Acquire());
            graph.AddComputePass("depth-consumer", [&](vk::PassBuilder &builder) {
                builder.SetQueueRole(rhi::QueueRole::Graphics);
                builder.ReadSampledDepth(attachments.depth, rhi::PipelineStage::ComputeShader);
                builder.SetSideEffect();
                return [](vk::RenderContext &) {};
            });
            // Invalid authored usage must fail compilation, not throw from a native frame.
            assert(graph.Compile() == sampled);
        }
    }
}

static void CheckPersistentGraphContract(const std::shared_ptr<rhi::RenderTexture> &target)
{
    auto material = InxMaterial::CreateDefaultUnlit();
    material->SetTextureGuid("texSampler", "white");
    const auto authored = material->Serialize();
    material->SetRenderTexture("texSampler", target);
    assert(material->GetRenderTexture("texSampler") == target);
    assert(material->Serialize() == authored); // Runtime owner never replaces the authored asset reference.
    auto clone = material->Clone();
    assert(clone->GetRenderTexture("texSampler") == target);
    clone->ClearTexture("texSampler");
    assert(!clone->GetRenderTexture("texSampler") && material->GetRenderTexture("texSampler") == target);
    material->SetTextureGuid("texSampler", "white");
    assert(!material->GetRenderTexture("texSampler"));
    material->SetRenderTexture("texSampler", target);
    material->SetFloat("texSampler", 1.0f);
    assert(!material->GetRenderTexture("texSampler"));
    const auto generation = target->Acquire();
    RenderGraphDescription description;
    GraphTextureDesc texture;
    texture.name = "target";
    texture.format = generation->description.colorFormat;
    texture.width = generation->width;
    texture.height = generation->height;
    texture.role = GraphTextureRole::Persistent;
    texture.renderTexture = target;
    texture.samples = static_cast<uint32_t>(generation->description.samples);
    description.textures.push_back(texture);
    GraphPassDesc clear;
    clear.name = "clear";
    clear.writeColors.emplace_back(0, texture.name);
    clear.clearColor = true;
    if (generation->multisampleColor) {
        auto resolve = texture;
        resolve.name = "resolved";
        resolve.samples = 1;
        resolve.attachment = GraphTextureAttachment::Resolve;
        description.textures.push_back(resolve);
        clear.resolveColor = resolve.name;
    }
    auto depth = texture;
    depth.name = "depth";
    depth.isDepth = true;
    depth.format = generation->description.depthFormat;
    depth.attachment = GraphTextureAttachment::Depth;
    description.textures.push_back(depth);
    clear.writeDepth = depth.name;
    clear.clearDepth = true;
    description.passes.push_back(clear);
    description.outputTexture = generation->multisampleColor ? "resolved" : texture.name;
    assert(SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.passes[0].clearDepth = false;
    assert(!SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.passes[0].clearDepth = true;
    description.passes[0].clearColor = false;
    assert(!SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.passes[0].clearColor = true;
    description.passes[0].readTextures.push_back(texture.name);
    assert(!SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.passes[0].readTextures.clear();
    GraphPassDesc reader;
    reader.name = "reader";
    reader.readTextures.push_back(texture.name);
    description.passes.insert(description.passes.begin(), reader);
    assert(!SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.passes.erase(description.passes.begin());
    // An input-only import is resolved by the frame scheduler, not by a
    // fabricated local clear which would erase the producer camera's image.
    auto incoming = description;
    incoming.passes.clear();
    reader.readTextures = {incoming.outputTexture};
    incoming.passes.push_back(reader);
    assert(SceneRenderGraph::ValidateGraphDescription(incoming, 1));
    // Implicit material/UI reads obey the same local producer contract. An
    // MSAA consumer reads resolve, never the multisampled color attachment.
    const SceneRenderGraph::MaterialTextureRead implicitRead{"reader", target, generation};
    assert(!SceneRenderGraph::HasLocalTextureProducer(incoming, implicitRead));
    auto local = description;
    GraphPassDesc implicitPass;
    implicitPass.name = "reader";
    local.passes.push_back(implicitPass);
    assert(SceneRenderGraph::HasLocalTextureProducer(local, implicitRead));
    const auto rejectImplicit = [&](const RenderGraphDescription &candidate) {
        bool rejected = false;
        try {
            (void)SceneRenderGraph::HasLocalTextureProducer(candidate, implicitRead);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    };
    std::swap(local.passes[0], local.passes[1]);
    rejectImplicit(local); // Cannot silently use previous-frame pixels.
    std::swap(local.passes[0], local.passes[1]);
    local.passes.back().writeColors.emplace_back(0, local.outputTexture);
    rejectImplicit(local); // Same sampled attachment read/write feedback.
    local.passes.back().writeColors.clear();
    auto rewritten = local.passes.front();
    rewritten.name = "rewrite";
    local.passes.push_back(rewritten);
    assert(SceneRenderGraph::HasLocalTextureProducer(local, implicitRead));
    // Names are aliases, not independent resources.
    auto alias = local.textures[generation->multisampleColor ? 1 : 0];
    alias.name = "sample_alias";
    local.textures.push_back(alias);
    local.passes[1].writeColors.emplace_back(0, alias.name);
    rejectImplicit(local);
    if (!generation->multisampleColor) {
        // Single-sample resolve is another name for color, not a second image.
        local.textures.back().attachment = GraphTextureAttachment::Resolve;
        rejectImplicit(local);
        local.passes[1].writeColors.clear();
        local.passes[0].writeColors = {{0, alias.name}};
        local.passes.pop_back();
        assert(SceneRenderGraph::HasLocalTextureProducer(local, implicitRead));
    }
    // A depth-only writer cannot supply sampled color from the same owner.
    local = incoming;
    auto depthOnly = clear;
    depthOnly.writeColors.clear();
    depthOnly.resolveColor.clear();
    local.passes.insert(local.passes.begin(), depthOnly);
    assert(!SceneRenderGraph::HasLocalTextureProducer(local, implicitRead));
    // A transfer producer is a writer too: no clear is needed for a complete
    // copy, but reading the destination before the copy remains invalid.
    RenderGraphDescription copied;
    auto destination = description.textures[generation->multisampleColor ? 1 : 0];
    copied.textures.push_back(destination);
    auto source = destination;
    source.name = "copy_source";
    source.role = GraphTextureRole::Transient;
    source.renderTexture.reset();
    copied.textures.push_back(source);
    GraphPassDesc initialize;
    initialize.name = "initialize_source";
    initialize.writeColors.emplace_back(0, source.name);
    initialize.clearColor = true;
    copied.passes.push_back(initialize);
    GraphPassDesc copy;
    copy.name = "copy";
    copy.type = GraphPassType::Copy;
    copy.commands.push_back({GraphCommandType::CopyTexture});
    copy.commands[0].sourceResource = source.name;
    copy.commands[0].destinationResource = destination.name;
    copied.passes.push_back(copy);
    copied.outputTexture = destination.name;
    assert(SceneRenderGraph::ValidateGraphDescription(copied, 1));
    reader.readTextures = {destination.name};
    copied.passes.insert(copied.passes.begin(), reader);
    assert(!SceneRenderGraph::ValidateGraphDescription(copied, 1));
    description.textures[0].samples = texture.samples == 1 ? 4 : 1;
    assert(!SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.textures[0].samples = texture.samples;
    // Shader reads may consume resolved color; raw MSAA color/depth must not
    // be bound as a regular sampler2D. A depth attachment read is separate.
    reader.commands.push_back({GraphCommandType::DrawRenderers});
    reader.readTextures = {description.outputTexture};
    reader.commands[0].inputBindings = {{"color", description.outputTexture}};
    description.passes.push_back(reader);
    assert(SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.passes.back().readTextures = {"depth"};
    description.passes.back().commands[0].inputBindings = {{"depth", "depth"}};
    assert(SceneRenderGraph::ValidateGraphDescription(description, 1) ==
           (texture.samples == 1 && generation->description.sampledDepth));
    description.passes.back().commands[0].inputBindings.clear();
    assert(SceneRenderGraph::ValidateGraphDescription(description, 1));
    description.passes.pop_back();
    description.textures[0].renderTexture.reset();
    assert(!SceneRenderGraph::ValidateGraphDescription(description, 1));
}

class FullscreenTestHost final : public FullscreenRendererHost
{
  public:
    FullscreenTestHost(rhi::Device &device, const char *vertex, const char *fragment)
        : device(device), vertex(vertex), fragment(fragment)
    {
    }
    rhi::Device &GetRhiDevice() noexcept override
    {
        return device;
    }
    uint32_t GetFrameCount() const noexcept override
    {
        return 1;
    }
    uint32_t GetCurrentFrame() const noexcept override
    {
        return 0;
    }
    rhi::BindingLayoutHandle GetPerViewLayout() const noexcept override
    {
        return {};
    }
    rhi::BindingLayoutHandle GetGlobalsLayout() const noexcept override
    {
        return {};
    }
    rhi::BindGroupHandle GetCurrentGlobalsGroup() override
    {
        return {};
    }
    rhi::ShaderModuleHandle AcquireShaderModule(const std::string &, rhi::ShaderStage stage, uint32_t,
                                                uint32_t) override
    {
        std::ifstream file(stage == rhi::ShaderStage::Vertex ? vertex : fragment, std::ios::binary | std::ios::ate);
        assert(file);
        const size_t bytes = static_cast<size_t>(file.tellg());
        assert(bytes && bytes % 4 == 0);
        std::vector<uint32_t> code(bytes / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char *>(code.data()), bytes);
        assert(file);
        return device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(code.data(), code.size()));
    }
    void ReportError(const std::string &message) override
    {
        std::cerr << message << '\n';
        assert(false);
    }
    rhi::Device &device;
    const char *vertex;
    const char *fragment;
};

static void CheckFullscreenRasterState(vk::VkDeviceContext &context, VkCommandBuffer command, VkFence fence,
                                       const char *vertex, const char *fragment, rhi::SubmissionSerial &epoch)
{
    auto &device = context.GetRhiDevice();
    FullscreenRenderer renderer;
    renderer.Initialize(std::make_shared<FullscreenTestHost>(device, vertex, fragment));
    // Actual fragment-depth writes, disabled writes, rejected fragments and
    // straight-alpha blending on both depth formats and single/MSAA targets.
    for (auto depthFormat : {rhi::PixelFormat::D32SFloat, rhi::PixelFormat::D24UNormS8UInt}) {
        for (auto samples : {rhi::SampleCount::One, rhi::SampleCount::Four}) {
            rhi::RenderTextureDesc desc;
            desc.width = 9;
            desc.height = 7;
            desc.colorFormat = rhi::PixelFormat::RGBA8UNorm;
            desc.depthFormat = depthFormat;
            desc.samples = samples;
            auto target = std::make_shared<rhi::RenderTexture>(device, "fullscreen-state", desc);
            vk::RenderGraph graph;
            graph.Initialize(&context);
            const auto attachments = graph.ImportRenderTexture("fullscreen", target->Acquire());
            auto color = attachments.color;
            auto depth = attachments.depth;
            auto resolved = attachments.resolve;
            graph.AddPass("initialize", [&](vk::PassBuilder &builder) {
                color = builder.WriteColor(color);
                depth = builder.WriteDepth(depth);
                if (resolved.IsValid())
                    resolved = builder.WriteResolve(resolved);
                builder.SetClearColor(0, 0, 1, 1);
                builder.SetClearDepth(1, 0);
                builder.SetRenderArea(desc.width, desc.height);
                return [](vk::RenderContext &) {};
            });
            const auto snapshot =
                graph.RegisterTransientTexture("independent-depth-copy", desc.width, desc.height,
                                               rhi::ToVkFormat(depthFormat), rhi::ToVkSampleCount(samples), true);
            assert(snapshot.id != depth.id);
            graph.AddTransferPass("snapshot-before-overlays", [&](vk::PassBuilder &builder) {
                const auto sourceDepth = depth;
                builder.SetQueueRole(rhi::QueueRole::Graphics);
                builder.TransferRead(sourceDepth);
                const auto written = builder.TransferWrite(snapshot);
                builder.SetSideEffect();
                return [=](vk::RenderContext &render) {
                    render.GetTransferCommandEncoder().CopyTexture(
                        render.GetTextureHandle(sourceDepth), render.GetTextureHandle(written),
                        {rhi::TextureAspect::Depth, 0, 0, 0, 0, desc.width, desc.height, 1});
                };
            });
            const std::array<std::array<float, 5>, 4> values{{
                {1, 0, 0, .5f, .4f}, // Red over blue, depth becomes .4.
                {0, 1, 0, 1, .7f},   // Behind .4: reject, including alpha.
                {0, 1, 1, .5f, .2f}, // Cyan blends but must NOT write depth.
                {1, 1, 0, .5f, .3f}, // Passes .4; would fail if .2 were written.
            }};
            for (size_t i = 0; i < values.size(); ++i) {
                FullscreenPipelineKey key;
                key.shaderName = "Fullscreen state";
                key.useDynamicRendering = true;
                key.depthFormat = depthFormat;
                key.samples = samples;
                key.alphaBlend = true;
                key.depth = {true, i != 2, i == 0 ? rhi::CompareFunction::Always : rhi::CompareFunction::Less};
                const auto pipeline = renderer.EnsurePipeline(key);
                assert(pipeline.pipeline.IsValid());
                assert(renderer.EnsurePipeline(key).pipeline == pipeline.pipeline);
                FullscreenPushConstants parameters;
                std::copy(values[i].begin(), values[i].end(), parameters.values);
                graph.AddPass("overlay-" + std::to_string(i), [&](vk::PassBuilder &builder) {
                    color = builder.WriteColor(color);
                    depth = builder.WriteDepth(depth);
                    if (resolved.IsValid())
                        resolved = builder.WriteResolve(resolved);
                    builder.SetDepthTest(true);
                    builder.SetRenderArea(desc.width, desc.height);
                    return [&, pipeline, parameters](vk::RenderContext &render) {
                        renderer.Draw(render.GetGraphicsCommandEncoder(), pipeline, {}, {}, parameters, 20);
                    };
                });
            }
            const auto source = resolved.IsValid() ? resolved : color;
            rhi::BufferDesc buffer;
            buffer.byteSize = desc.width * desc.height * 4;
            buffer.usage = rhi::BufferUsageFlags::TransferDestination;
            buffer.memory = rhi::BufferMemory::Readback;
            const auto output = device.CreateBuffer(buffer);
            assert(output.IsValid());
            graph.AddTransferPass("verify", [&](vk::PassBuilder &builder) {
                builder.TransferRead(source);
                builder.TransferWrite(builder.ImportBuffer("result", output, buffer.byteSize));
                builder.SetQueueRole(rhi::QueueRole::Graphics);
                builder.SetSideEffect();
                return [&](vk::RenderContext &render) {
                    VkBufferImageCopy copy{};
                    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copy.imageExtent = {desc.width, desc.height, 1};
                    vkCmdCopyImageToBuffer(render.GetCommandBuffer(), device.Resolve(render.GetTextureHandle(source)),
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(output), 1, &copy);
                };
            });
            assert(graph.Compile());
            assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
            graph.Execute(command);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = device.Resolve(output);
            barrier.size = buffer.byteSize;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                                 &barrier, 0, nullptr);
            assert(vkEndCommandBuffer(command) == VK_SUCCESS);
            assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
            assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
            std::vector<uint8_t> result(buffer.byteSize);
            assert(device.ReadBuffer(output, 0, result.data(), result.size()));
            const std::array<float, 4> expected{.625f, .75f, .375f, 1};
            for (size_t i = 0; i < result.size(); ++i)
                assert(std::abs(result[i] / 255.f - expected[i % 4]) < .008f);
            graph.Destroy();
            target.reset();
            device.Release(output);
            device.CollectResourceRetirements(epoch++);
            std::cout << "PASS fullscreen fragment-depth/write-disabled/blend/load depth=" << int(depthFormat)
                      << " samples=" << int(samples) << '\n';
        }
    }
    renderer.Destroy();
    device.CollectDescriptorRetirements(epoch);
    device.CollectResourceRetirements(epoch++);
}

static void CheckFullscreenSamples(vk::VkDeviceContext &context, VkCommandBuffer command, VkFence fence,
                                   const char *vertex, const char *write, const char *read,
                                   rhi::SubmissionSerial &epoch)
{
    if (!context.GetDeviceFeatures().sampleRateShading) {
        std::cout << "SKIP sample-frequency shader: device does not support sampleRateShading\n";
        return;
    }
    auto &device = context.GetRhiDevice();
    FullscreenRenderer writer, reader;
    writer.Initialize(std::make_shared<FullscreenTestHost>(device, vertex, write));
    reader.Initialize(std::make_shared<FullscreenTestHost>(device, vertex, read));
    std::ifstream input(read, std::ios::binary);
    const std::vector<char> code{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ShaderReflection reflection;
    assert(reflection.Reflect(code, VK_SHADER_STAGE_FRAGMENT_BIT));
    assert(reflection.RequiresSampleRateShading());
    assert(reflection.GetSampledImages().size() == 2);
    for (const auto &image : reflection.GetSampledImages())
        assert(image.multisampled);
    reflection.Clear();
    assert(!reflection.RequiresSampleRateShading());

    rhi::RenderTextureDesc desc;
    desc.width = 9;
    desc.height = 7;
    desc.samples = rhi::SampleCount::Four;
    desc.colorFormat = rhi::PixelFormat::RGBA32SFloat;
    desc.depthFormat = rhi::PixelFormat::D32SFloat;
    desc.sampledDepth = true;
    auto owners = std::make_shared<rhi::RenderTexture>(device, "sample-identities", desc);
    desc.colorFormat = rhi::PixelFormat::RGBA8UNorm;
    auto result = std::make_shared<rhi::RenderTexture>(device, "sample-colors", desc);
    vk::RenderGraph graph;
    graph.Initialize(&context);
    const auto source = graph.ImportRenderTexture("owners", owners->Acquire());
    const auto destination = graph.ImportRenderTexture("result", result->Acquire());
    auto color = source.color, depth = source.depth, resolved = destination.resolve;
    FullscreenPipelineKey key;
    key.shaderName = "sample-write";
    key.useDynamicRendering = true;
    key.samples = rhi::SampleCount::Four;
    key.colorFormat = rhi::PixelFormat::RGBA32SFloat;
    key.depthFormat = desc.depthFormat;
    key.depth = {true, true, rhi::CompareFunction::Always};
    const auto writePipeline = writer.EnsurePipeline(key);
    assert(writePipeline.pipeline.IsValid());
    graph.AddPass("write-samples", [&](vk::PassBuilder &builder) {
        color = builder.WriteColor(color);
        depth = builder.WriteDepth(depth);
        builder.SetClearColor(0, 0, 0, 0);
        builder.SetClearDepth(1, 0);
        builder.SetDepthTest(true);
        builder.SetRenderArea(desc.width, desc.height);
        return [&, writePipeline](vk::RenderContext &ctx) {
            writer.Draw(ctx.GetGraphicsCommandEncoder(), writePipeline, {}, {}, {}, 0);
        };
    });
    key.shaderName = "sample-read";
    key.colorFormat = desc.colorFormat;
    key.depthFormat = rhi::PixelFormat::Undefined;
    key.depth = {};
    key.inputResourceCount = 2;
    key.depthInputMask = 2;
    const auto readPipeline = reader.EnsurePipeline(key);
    assert(readPipeline.pipeline.IsValid());
    graph.AddPass("shade-samples", [&](vk::PassBuilder &builder) {
        builder.Read(color);
        builder.ReadSampledDepth(depth);
        builder.WriteColor(destination.color);
        resolved = builder.WriteResolve(resolved);
        builder.SetClearColor(0, 0, 0, 0);
        builder.SetRenderArea(desc.width, desc.height);
        return [&, readPipeline](vk::RenderContext &ctx) {
            const FullscreenResourceInput inputs[] = {
                {ctx.GetTextureView(color), rhi::PixelFormat::RGBA32SFloat, false},
                {ctx.GetTextureView(depth), rhi::PixelFormat::D32SFloat, true}};
            const auto group = reader.AllocateBindGroup(readPipeline.inputLayout, inputs, 2, reader.GetLinearSampler());
            assert(group.IsValid());
            reader.Draw(ctx.GetGraphicsCommandEncoder(), readPipeline, group, {}, {}, 0);
        };
    });
    rhi::BufferDesc buffer;
    buffer.byteSize = desc.width * desc.height * 4;
    buffer.usage = rhi::BufferUsageFlags::TransferDestination;
    buffer.memory = rhi::BufferMemory::Readback;
    const auto output = device.CreateBuffer(buffer);
    graph.AddTransferPass("readback", [&](vk::PassBuilder &builder) {
        builder.TransferRead(resolved);
        builder.TransferWrite(builder.ImportBuffer("pixels", output, buffer.byteSize));
        builder.SetQueueRole(rhi::QueueRole::Graphics);
        builder.SetSideEffect();
        return [&](vk::RenderContext &ctx) {
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {desc.width, desc.height, 1};
            vkCmdCopyImageToBuffer(ctx.GetCommandBuffer(), device.Resolve(ctx.GetTextureHandle(resolved)),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(output), 1, &copy);
        };
    });
    assert(graph.Compile());
    assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
    graph.Execute(command);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = device.Resolve(output);
    barrier.size = buffer.byteSize;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &barrier, 0, nullptr);
    assert(vkEndCommandBuffer(command) == VK_SUCCESS);
    assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
    assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
    std::vector<uint8_t> pixels(buffer.byteSize);
    assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
    const int expected[] = {128, 0, 128, 255};
    for (size_t i = 0; i < pixels.size(); ++i) {
        if (std::abs(int(pixels[i]) - expected[i % 4]) > 1)
            std::cerr << "Sample resolve mismatch byte=" << i << " actual=" << int(pixels[i])
                      << " expected=" << expected[i % 4] << " first RGBA=" << int(pixels[0]) << ',' << int(pixels[1])
                      << ',' << int(pixels[2]) << ',' << int(pixels[3]) << '\n';
        assert(std::abs(int(pixels[i]) - expected[i % 4]) <= 1);
    }
    graph.Destroy();
    owners.reset();
    result.reset();
    device.Release(output);
    reader.Destroy();
    writer.Destroy();
    device.CollectDescriptorRetirements(epoch);
    device.CollectResourceRetirements(epoch++);
    std::cout << "PASS per-sample owner/depth identity -> per-sample color -> 4x coverage resolve\n";
}

static void CheckFullscreenStorageRead(vk::VkDeviceContext &context, VkCommandBuffer command, VkFence fence,
                                       const char *vertex, const char *fragment, rhi::SubmissionSerial &epoch)
{
    std::ifstream shaderInput(fragment, std::ios::binary);
    const std::vector<char> shaderCode{std::istreambuf_iterator<char>(shaderInput), std::istreambuf_iterator<char>()};
    ShaderReflection reflection;
    assert(reflection.Reflect(shaderCode, VK_SHADER_STAGE_FRAGMENT_BIT));
    assert(reflection.GetStorageBuffers().size() == 1);
    assert(reflection.GetStorageBuffers()[0].set == 0 && reflection.GetStorageBuffers()[0].binding == 0 &&
           reflection.GetStorageBuffers()[0].readOnly);
    auto &device = context.GetRhiDevice();
    rhi::BufferDesc sourceDesc;
    sourceDesc.byteSize = sizeof(uint32_t);
    sourceDesc.usage = rhi::BufferUsageFlags::Storage;
    sourceDesc.memory = rhi::BufferMemory::Upload;
    assert(sourceDesc.byteSize <= context.GetDeviceProperties().limits.maxStorageBufferRange);
    const auto source = device.CreateBuffer(sourceDesc);
    const uint32_t value = 128;
    assert(source.IsValid() && device.WriteBuffer(source, 0, &value, sizeof(value)));

    rhi::RenderTextureDesc imageDesc;
    imageDesc.width = imageDesc.height = 2;
    imageDesc.colorFormat = rhi::PixelFormat::RGBA8UNorm;
    auto target = std::make_shared<rhi::RenderTexture>(device, "fullscreen-storage", imageDesc);
    rhi::BufferDesc readbackDesc;
    readbackDesc.byteSize = 16;
    readbackDesc.usage = rhi::BufferUsageFlags::TransferDestination;
    readbackDesc.memory = rhi::BufferMemory::Readback;
    const auto readback = device.CreateBuffer(readbackDesc);
    assert(readback.IsValid());

    FullscreenRenderer renderer;
    renderer.Initialize(std::make_shared<FullscreenTestHost>(device, vertex, fragment));
    FullscreenPipelineKey key;
    key.shaderName = "storage-read";
    key.useDynamicRendering = true;
    key.colorFormat = imageDesc.colorFormat;
    key.inputResourceCount = 1;
    key.inputBufferMask = 1;
    auto textureInputKey = key;
    textureInputKey.inputBufferMask = 0;
    assert(!(key == textureInputKey));
    const auto pipeline = renderer.EnsurePipeline(key);
    assert(pipeline.pipeline.IsValid());

    vk::RenderGraph graph;
    graph.Initialize(&context);
    auto color = graph.ImportRenderTexture("storage-result", target->Acquire()).color;
    graph.AddPass("storage-fragment", [&](vk::PassBuilder &builder) {
        const auto imported = builder.ImportBuffer("storage-source", source, sourceDesc.byteSize);
        builder.ReadStorageBuffer(imported, rhi::PipelineStage::FragmentShader);
        color = builder.WriteColor(color);
        builder.SetClearColor(0, 0, 0, 1);
        builder.SetRenderArea(2, 2);
        return [&, imported, pipeline](vk::RenderContext &render) {
            FullscreenResourceInput input{};
            input.buffer = render.GetBufferHandle(imported);
            input.byteSize = sourceDesc.byteSize;
            const auto group = renderer.AllocateBindGroup(pipeline.inputLayout, &input, 1, renderer.GetLinearSampler());
            assert(group.IsValid());
            renderer.Draw(render.GetGraphicsCommandEncoder(), pipeline, group, {}, {}, 0);
        };
    });
    graph.AddTransferPass("storage-readback", [&](vk::PassBuilder &builder) {
        builder.SetQueueRole(rhi::QueueRole::Graphics);
        builder.TransferRead(color);
        builder.TransferWrite(builder.ImportBuffer("storage-pixels", readback, readbackDesc.byteSize));
        builder.SetSideEffect();
        return [&](vk::RenderContext &render) {
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {2, 2, 1};
            vkCmdCopyImageToBuffer(render.GetCommandBuffer(), device.Resolve(render.GetTextureHandle(color)),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(readback), 1, &copy);
        };
    });
    assert(graph.Compile());
    assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
    graph.Execute(command);
    assert(vkEndCommandBuffer(command) == VK_SUCCESS);
    assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
    assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
    std::array<uint8_t, 16> pixels{};
    assert(device.ReadBuffer(readback, 0, pixels.data(), pixels.size()));
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (std::abs(int(pixels[i]) - 128) > 1 || pixels[i + 1] || pixels[i + 2] || pixels[i + 3] != 255)
            std::cerr << "Storage pixel=" << int(pixels[i]) << ',' << int(pixels[i + 1]) << ',' << int(pixels[i + 2])
                      << ',' << int(pixels[i + 3]) << '\n';
        assert(std::abs(int(pixels[i]) - 128) <= 1 && pixels[i + 1] == 0 && pixels[i + 2] == 0 && pixels[i + 3] == 255);
    }
    graph.Destroy();
    renderer.Destroy();
    target.reset();
    device.Release(readback);
    device.Release(source);
    device.CollectDescriptorRetirements(epoch);
    device.CollectResourceRetirements(epoch++);
    std::cout << "PASS fullscreen storage-buffer GPU read -> color output\n";
}

static void CheckFullscreenVolumeRead(vk::VkDeviceContext &context, VkCommandBuffer command, VkFence fence,
                                      const char *vertex, const char *fragment, rhi::SubmissionSerial &epoch)
{
    auto &device = context.GetRhiDevice();
    constexpr uint32_t width = 2;
    constexpr uint32_t height = 2;
    constexpr uint32_t depth = 2;
    std::array<uint8_t, width * height * depth * 4> texels{};
    for (uint32_t z = 0; z < depth; ++z) {
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                const size_t offset = ((z * height + y) * width + x) * 4;
                texels[offset + 0] = z == 0 ? 255 : 0;
                texels[offset + 2] = z == 1 ? 255 : 0;
                texels[offset + 3] = 255;
            }
        }
    }
    rhi::TextureDesc volumeDesc;
    volumeDesc.dimension = rhi::TextureDimension::Texture3D;
    volumeDesc.width = width;
    volumeDesc.height = height;
    volumeDesc.depthOrLayers = depth;
    volumeDesc.format = rhi::PixelFormat::RGBA8UNorm;
    volumeDesc.usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::TransferDestination;
    const auto volume = device.CreateTexture(volumeDesc);
    rhi::TextureViewDesc viewDesc;
    viewDesc.texture = volume;
    viewDesc.dimension = rhi::TextureViewDimension::Texture3D;
    const auto volumeView = device.CreateTextureView(viewDesc);
    rhi::BufferDesc uploadDesc;
    uploadDesc.byteSize = texels.size();
    uploadDesc.usage = rhi::BufferUsageFlags::TransferSource;
    uploadDesc.memory = rhi::BufferMemory::Upload;
    const auto upload = device.CreateBuffer(uploadDesc);
    assert(volume.IsValid() && volumeView.IsValid() && upload.IsValid());
    assert(device.WriteBuffer(upload, 0, texels.data(), texels.size()));

    assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = device.Resolve(volume);
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, depth};
    vkCmdCopyBufferToImage(command, device.Resolve(upload), device.Resolve(volume),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
    assert(vkEndCommandBuffer(command) == VK_SUCCESS);
    assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
    assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);

    rhi::RenderTextureDesc targetDesc;
    targetDesc.width = width;
    targetDesc.height = height;
    targetDesc.colorFormat = rhi::PixelFormat::RGBA8UNorm;
    auto target = std::make_shared<rhi::RenderTexture>(device, "fullscreen-volume", targetDesc);
    rhi::BufferDesc readbackDesc;
    readbackDesc.byteSize = width * height * 4;
    readbackDesc.usage = rhi::BufferUsageFlags::TransferDestination;
    readbackDesc.memory = rhi::BufferMemory::Readback;
    const auto readback = device.CreateBuffer(readbackDesc);
    assert(readback.IsValid());
    rhi::SamplerDesc samplerDesc;
    samplerDesc.minFilter = rhi::FilterMode::Nearest;
    samplerDesc.magFilter = rhi::FilterMode::Nearest;
    samplerDesc.mipFilter = rhi::FilterMode::Nearest;
    samplerDesc.addressU = samplerDesc.addressV = samplerDesc.addressW = rhi::AddressMode::ClampToEdge;
    const auto sampler = device.CreateSampler(samplerDesc);
    assert(sampler.IsValid());
    FullscreenRenderer renderer;
    renderer.Initialize(std::make_shared<FullscreenTestHost>(device, vertex, fragment));
    FullscreenPipelineKey key;
    key.shaderName = "volume-read";
    key.useDynamicRendering = true;
    key.colorFormat = targetDesc.colorFormat;
    key.inputResourceCount = 1;
    const auto pipeline = renderer.EnsurePipeline(key);
    assert(pipeline.pipeline.IsValid());

    vk::RenderGraph graph;
    graph.Initialize(&context);
    const auto input = graph.ImportTexture("volume", volume, volumeView, VK_FORMAT_R8G8B8A8_UNORM, width, height,
                                           VK_SAMPLE_COUNT_1_BIT, depth, true);
    assert(input.IsValid());
    graph.SetResourceInitialState(input, rhi::TextureLayout::ShaderReadOnly, rhi::Access::ShaderRead,
                                  rhi::PipelineStage::FragmentShader);
    auto color = graph.ImportRenderTexture("volume-result", target->Acquire()).color;
    graph.AddPass("sample-volume", [&](vk::PassBuilder &builder) {
        builder.Read(input, rhi::PipelineStage::FragmentShader);
        color = builder.WriteColor(color);
        builder.SetClearColor(0, 0, 0, 1);
        builder.SetRenderArea(width, height);
        return [&, pipeline](vk::RenderContext &render) {
            FullscreenResourceInput resource{};
            resource.view = render.GetTextureView(input);
            resource.format = rhi::PixelFormat::RGBA8UNorm;
            resource.sampler = sampler;
            const auto group =
                renderer.AllocateBindGroup(pipeline.inputLayout, &resource, 1, renderer.GetLinearSampler());
            assert(group.IsValid());
            renderer.Draw(render.GetGraphicsCommandEncoder(), pipeline, group, {}, {}, 0);
        };
    });
    graph.AddTransferPass("volume-readback", [&](vk::PassBuilder &builder) {
        builder.TransferRead(color);
        builder.TransferWrite(builder.ImportBuffer("volume-pixels", readback, readbackDesc.byteSize));
        builder.SetQueueRole(rhi::QueueRole::Graphics);
        builder.SetSideEffect();
        return [&](vk::RenderContext &render) {
            VkBufferImageCopy resultCopy{};
            resultCopy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            resultCopy.imageExtent = {width, height, 1};
            vkCmdCopyImageToBuffer(render.GetCommandBuffer(), device.Resolve(render.GetTextureHandle(color)),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(readback), 1, &resultCopy);
        };
    });
    assert(graph.Compile());
    assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
    assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
    graph.Execute(command);
    VkBufferMemoryBarrier readbackBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    readbackBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    readbackBarrier.srcQueueFamilyIndex = readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.buffer = device.Resolve(readback);
    readbackBarrier.size = readbackDesc.byteSize;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                         &readbackBarrier, 0, nullptr);
    assert(vkEndCommandBuffer(command) == VK_SUCCESS);
    assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
    assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
    assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
    std::array<uint8_t, width * height * 4> pixels{};
    assert(device.ReadBuffer(readback, 0, pixels.data(), pixels.size()));
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t offset = (y * width + x) * 4;
            const std::array<uint8_t, 4> expected =
                x == 0 ? std::array<uint8_t, 4>{255, 0, 0, 255} : std::array<uint8_t, 4>{0, 0, 255, 255};
            if (!std::equal(expected.begin(), expected.end(), pixels.begin() + offset))
                std::cerr << "Volume pixel (" << x << ',' << y << ")=" << int(pixels[offset]) << ','
                          << int(pixels[offset + 1]) << ',' << int(pixels[offset + 2]) << ',' << int(pixels[offset + 3])
                          << '\n';
            assert(std::equal(expected.begin(), expected.end(), pixels.begin() + offset));
        }
    }
    graph.Destroy();
    renderer.Destroy();
    target.reset();
    device.Release(sampler);
    device.Release(readback);
    device.Release(upload);
    device.Release(volumeView);
    device.Release(volume);
    device.CollectDescriptorRetirements(epoch);
    device.CollectResourceRetirements(epoch++);
    std::cout << "PASS imported Texture3D -> fullscreen sampler3D -> two depth slices -> GPU readback\n";
}

int main(int argc, char **argv)
{
    assert(argc == 8);
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    assert(input);
    const auto bytes = static_cast<size_t>(input.tellg());
    assert(bytes > 0 && bytes % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(bytes / sizeof(uint32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(code.data()), bytes);
    assert(input);

    assert(SDL_Init(SDL_INIT_VIDEO));
    auto *window = SDL_CreateWindow("RenderTexture GPU regression", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    assert(window);
    vk::VkDeviceContext context;
    vk::DeviceConfig config;
    config.appName = "RenderTexture GPU regression";
    config.enableValidationLayers = true;
    assert(context.Initialize(window, config));
    auto &device = context.GetRhiDevice();
    rhi::SubmissionSerial epoch = 1;
    device.UseSubmissionSerials([&] { return epoch; });

    const auto shader = device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(code.data(), code.size()));
    rhi::BindingLayoutDesc layoutDesc;
    layoutDesc.entries[0] = {0, rhi::BindingType::CombinedTextureSampler, rhi::ShaderStage::Compute, 1};
    layoutDesc.entries[1] = {1, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute, 1};
    layoutDesc.entryCount = 2;
    const auto layout = device.CreateBindingLayout(layoutDesc);
    rhi::ComputePipelineDesc pipelineDesc;
    pipelineDesc.computeShader = shader;
    pipelineDesc.bindingLayouts[0] = layout;
    pipelineDesc.bindingLayoutCount = 1;
    const auto pipeline = device.CreateComputePipeline(pipelineDesc);
    assert(shader.IsValid() && layout.IsValid() && pipeline.IsValid());

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    assert(vkCreateCommandPool(context.GetDevice(), &poolInfo, nullptr, &pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    assert(vkAllocateCommandBuffers(context.GetDevice(), &allocate, &command) == VK_SUCCESS);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    assert(vkCreateFence(context.GetDevice(), &fenceInfo, nullptr, &fence) == VK_SUCCESS);

    CheckDepthSamplingDeclaration(context);
    CheckFullscreenRasterState(context, command, fence, argv[2], argv[3], epoch);
    CheckFullscreenSamples(context, command, fence, argv[2], argv[4], argv[5], epoch);
    CheckFullscreenStorageRead(context, command, fence, argv[2], argv[6], epoch);
    CheckFullscreenVolumeRead(context, command, fence, argv[2], argv[7], epoch);

    // Alternate independent targets and formats. Each recorded generation must
    // remain usable after resize and after both the target and graph are gone.
    for (int caseIndex = 0; caseIndex != 6; ++caseIndex) {
        const bool msaa = (caseIndex % 2) != 0;
        rhi::RenderTextureDesc desc;
        desc.width = 17 + caseIndex * 3;
        desc.height = 11 + caseIndex * 2;
        desc.colorFormat = msaa ? rhi::PixelFormat::RGBA16SFloat : rhi::PixelFormat::RGBA8UNorm;
        desc.depthFormat = rhi::PixelFormat::D32SFloat;
        desc.sampledDepth = true;
        desc.samples = msaa ? rhi::SampleCount::Four : rhi::SampleCount::One;
        auto target = std::make_shared<rhi::RenderTexture>(device, "monitor-" + std::to_string(caseIndex), desc);
        CheckPersistentGraphContract(target);
        auto generation = target->Acquire();
        const auto width = generation->width;
        const auto height = generation->height;
        const auto oldImage = generation->color->GetTexture();
        std::weak_ptr<const rhi::RenderTextureGeneration> weakGeneration = generation;
        const float expected[4] = {msaa ? 2.0f : 1.0f, 0.0f, (caseIndex % 3) == 0 ? 1.0f : 0.0f, 1.0f};
        const float beforeRewrite[4] = {0.125f, 0.5f, 0.75f, 1.0f};
        const uint64_t outputBytes = uint64_t(width) * height * 4 * sizeof(float);
        rhi::BufferDesc outputDesc;
        outputDesc.byteSize = outputBytes;
        outputDesc.usage = rhi::BufferUsageFlags::Storage;
        outputDesc.memory = rhi::BufferMemory::Readback;
        const auto output = device.CreateBuffer(outputDesc);
        const auto beforeOutput = device.CreateBuffer(outputDesc);
        assert(output.IsValid() && beforeOutput.IsValid());
        rhi::BindGroupDesc groupDesc;
        groupDesc.layout = layout;
        groupDesc.textures[0] = {0, rhi::BindingType::CombinedTextureSampler, generation->color->GetView(),
                                 generation->color->GetSampler()};
        groupDesc.textureCount = 1;
        groupDesc.buffers[0] = {1, rhi::BindingType::StorageBuffer, output, 0, outputBytes};
        groupDesc.bufferCount = 1;
        const auto group = device.CreateBindGroup(groupDesc);
        groupDesc.buffers[0].buffer = beforeOutput;
        const auto beforeGroup = device.CreateBindGroup(groupDesc);
        assert(group.IsValid() && beforeGroup.IsValid());

        vk::RenderGraph graph;
        graph.Initialize(&context);
        auto attachments = graph.ImportRenderTexture("monitor", generation);
        graph.AddPass("clear-monitor", [&](vk::PassBuilder &builder) {
            attachments.color = builder.WriteColor(attachments.color);
            attachments.depth = builder.WriteDepth(attachments.depth);
            if (msaa)
                attachments.resolve = builder.WriteResolve(attachments.resolve);
            builder.SetRenderArea(width, height);
            builder.SetClearColor(beforeRewrite[0], beforeRewrite[1], beforeRewrite[2], beforeRewrite[3]);
            builder.SetClearDepth(0.25f);
            return [](vk::RenderContext &) {};
        });
        const auto resourceCount = graph.GetResourceCount();
        const auto consumer = graph.ImportRenderTexture("another-consumer", generation);
        assert(graph.GetResourceCount() == resourceCount);
        assert(consumer.color == attachments.color && consumer.depth == attachments.depth);
        assert(consumer.resolve == attachments.resolve);
        graph.AddPass("sample-before-rewrite", [&](vk::PassBuilder &builder) {
            builder.Read(msaa ? consumer.resolve : consumer.color, rhi::PipelineStage::ComputeShader);
            builder.WriteStorageBuffer(builder.ImportBuffer("before-readback", beforeOutput, outputBytes));
            builder.SetQueueRole(rhi::QueueRole::Graphics);
            builder.SetSideEffect();
            return [=](vk::RenderContext &render) {
                auto &encoder = render.GetComputeCommandEncoder();
                encoder.BindPipeline(pipeline);
                encoder.BindGroup(pipeline, 0, beforeGroup);
                encoder.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
            };
        });
        graph.AddPass("rewrite-monitor", [&](vk::PassBuilder &builder) {
            attachments.color = builder.WriteColor(attachments.color);
            if (msaa)
                attachments.resolve = builder.WriteResolve(attachments.resolve);
            builder.SetRenderArea(width, height);
            builder.SetDepthTest(false);
            builder.SetClearColor(expected[0], expected[1], expected[2], expected[3]);
            return [](vk::RenderContext &) {};
        });
        // One physical image, two logical versions: earlier readers must finish
        // before the overwrite, and later imports must acquire its new version.
        const auto latest = graph.ImportRenderTexture("after-rewrite", generation);
        assert(graph.GetResourceCount() == resourceCount + 1); // Only the readback buffer was added.
        assert(latest.color == attachments.color && latest.color != consumer.color);
        assert(latest.resolve == attachments.resolve);
        const auto sample = msaa ? latest.resolve : latest.color;
        graph.AddPass("export-monitor", [&](vk::PassBuilder &builder) {
            builder.Read(sample, rhi::PipelineStage::ComputeShader);
            builder.SetSideEffect();
            return [](vk::RenderContext &) {};
        });
        vk::RenderGraph overlayGraph;
        overlayGraph.Initialize(&context);
        auto overlay = overlayGraph.ImportRenderTexture("shared-output", generation);
        overlayGraph.AddPass("preserve-output", [&](vk::PassBuilder &builder) {
            overlay.color = builder.WriteColor(overlay.color);
            if (msaa)
                overlay.resolve = builder.WriteResolve(overlay.resolve);
            builder.SetRenderArea(width, height);
            builder.SetDepthTest(false);
            // No clear: rebinding a second writer must preserve the first.
            return [](vk::RenderContext &) {};
        });
        overlayGraph.AddPass("export-preserved", [&](vk::PassBuilder &builder) {
            builder.Read(msaa ? overlay.resolve : overlay.color, rhi::PipelineStage::ComputeShader);
            builder.SetSideEffect();
            return [](vk::RenderContext &) {};
        });
        // Consume this execution's image in a different graph, without a
        // readback, intermediate color copy, or previous-frame substitution.
        vk::RenderGraph readerGraph;
        readerGraph.Initialize(&context);
        const auto imported = readerGraph.ImportRenderTexture("camera-output", generation);
        const auto incoming = msaa ? imported.resolve : imported.color;
        readerGraph.SetResourceInitialState(incoming, rhi::TextureLayout::ShaderReadOnly, rhi::Access::ShaderRead,
                                            rhi::PipelineStage::ComputeShader);
        readerGraph.AddPass("sample-monitor", [&](vk::PassBuilder &builder) {
            builder.Read(incoming, rhi::PipelineStage::ComputeShader);
            builder.WriteStorageBuffer(builder.ImportBuffer("sample-readback", output, outputBytes));
            builder.SetQueueRole(rhi::QueueRole::Graphics);
            builder.SetSideEffect();
            return [=](vk::RenderContext &render) {
                auto &encoder = render.GetComputeCommandEncoder();
                encoder.BindPipeline(pipeline);
                encoder.BindGroup(pipeline, 0, group);
                encoder.Dispatch((width + 7) / 8, (height + 7) / 8, 1);
            };
        });
        assert(graph.Compile());
        assert(overlayGraph.Compile());
        assert(readerGraph.Compile());
        const auto signature = graph.GetPassRenderingSignature("clear-monitor");
        assert(signature.samples == desc.samples && signature.depthFormat == desc.depthFormat);
        assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
        assert(graph.NeedsPersistentImageInitialization());
        assert(readerGraph.NeedsPersistentImageInitialization());
        graph.Execute(command);
        assert(generation->graphLayoutsInitialized);
        assert(!overlayGraph.NeedsPersistentImageInitialization());
        assert(!readerGraph.NeedsPersistentImageInitialization());
        overlayGraph.Execute(command);
        readerGraph.Execute(command);
        VkBufferMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.buffer = device.Resolve(output);
        hostBarrier.size = outputBytes;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                             1, &hostBarrier, 0, nullptr);
        hostBarrier.buffer = device.Resolve(beforeOutput);
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr,
                             1, &hostBarrier, 0, nullptr);
        assert(vkEndCommandBuffer(command) == VK_SUCCESS);

        desc.width += 5;
        assert(target->Reconfigure(desc));
        assert(!target->Acquire()->graphLayoutsInitialized);
        assert(target->Acquire()->color->GetTexture() != oldImage);
        generation.reset();
        target.reset();
        assert(!weakGeneration.expired()); // Graph still owns the recorded allocation.
        graph.Destroy();
        overlayGraph.Destroy();
        assert(!weakGeneration.expired());
        readerGraph.Destroy();
        assert(weakGeneration.expired());
        assert(device.Resolve(oldImage) == VK_NULL_HANDLE); // Handle retired, allocation not freed yet.
        device.Release(group);
        device.Release(beforeGroup);
        assert(device.CollectResourceRetirements(epoch - 1) == 0);
        assert(device.CollectDescriptorRetirements(epoch - 1) == 0);

        assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
        assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
        std::vector<float> pixels(width * height * 4);
        assert(device.ReadBuffer(output, 0, pixels.data(), outputBytes));
        for (size_t i = 0; i < pixels.size(); ++i)
            assert(std::abs(pixels[i] - expected[i % 4]) < 0.002f);
        assert(device.ReadBuffer(beforeOutput, 0, pixels.data(), outputBytes));
        for (size_t i = 0; i < pixels.size(); ++i)
            assert(std::abs(pixels[i] - beforeRewrite[i % 4]) < 0.002f);
        device.Release(output);
        device.Release(beforeOutput);
        device.CollectDescriptorRetirements(epoch);
        assert(device.CollectResourceRetirements(epoch) > 0);
        ++epoch;
        std::cout << "PASS RT " << width << 'x' << height << (msaa ? " HDR MSAA4" : " UNORM")
                  << " initialize/clear/resolve/read-rewrite-read/shared-load/readback/retirement\n";
    }
    {
        rhi::RenderTextureDesc desc;
        desc.width = 3;
        desc.height = 2;
        std::array<std::shared_ptr<rhi::RenderTexture>, 2> histories;
        for (auto &history : histories)
            history = std::make_shared<rhi::RenderTexture>(device, "ping-pong", desc);
        rhi::BufferDesc readbackDesc;
        readbackDesc.byteSize = 48; // Two RGBA8 3x2 images.
        readbackDesc.usage = rhi::BufferUsageFlags::TransferDestination;
        readbackDesc.memory = rhi::BufferMemory::Readback;
        const auto output = device.CreateBuffer(readbackDesc);
        assert(output.IsValid());
        vk::RenderGraph graph;
        graph.Initialize(&context);
        const auto readSlot = graph.ImportRenderTexture("read", histories[0]->Acquire()).color;
        const auto writeSlot = graph.ImportRenderTexture("write", histories[1]->Acquire()).color;
        vk::ResourceHandle initialized, written;
        bool reset = true;
        graph.AddTransferPass("initialize-history", [&](vk::PassBuilder &builder) {
            initialized = builder.TransferWrite(readSlot);
            builder.SetQueueRole(rhi::QueueRole::Graphics);
            return [&](vk::RenderContext &render) {
                if (!reset)
                    return;
                const VkClearColorValue zero{};
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(render.GetCommandBuffer(), device.Resolve(render.GetTextureHandle(readSlot)),
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            };
        });
        graph.AddPass("write-history", [&](vk::PassBuilder &builder) {
            written = builder.WriteColor(writeSlot);
            builder.SetClearColor(1, 0, 0, 1);
            builder.SetRenderArea(3, 2);
            return [](vk::RenderContext &) {};
        });
        graph.AddTransferPass("read-history-pair", [&](vk::PassBuilder &builder) {
            builder.TransferRead(initialized);
            builder.TransferRead(written);
            builder.TransferWrite(builder.ImportBuffer("readback", output, 48));
            builder.SetQueueRole(rhi::QueueRole::Graphics);
            builder.SetSideEffect();
            return [&](vk::RenderContext &render) {
                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy.imageExtent = {3, 2, 1};
                for (const auto source : {initialized, written}) {
                    vkCmdCopyImageToBuffer(render.GetCommandBuffer(), device.Resolve(render.GetTextureHandle(source)),
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, device.Resolve(output), 1, &copy);
                    copy.bufferOffset += 24;
                }
            };
        });
        graph.AddPass("export-history-pair", [&](vk::PassBuilder &builder) {
            builder.Read(initialized);
            builder.Read(written);
            builder.SetSideEffect();
            return [](vk::RenderContext &) {};
        });
        assert(graph.Compile());
        std::array<uint8_t, 4> previous{};
        for (uint32_t frame = 0; frame != 8; ++frame) {
            reset = frame == 0 || frame == 4;
            if (reset)
                previous = {};
            graph.UpdateImportedRenderTextureColor(readSlot, histories[frame % 2]->Acquire());
            graph.UpdateImportedRenderTextureColor(writeSlot, histories[(frame % 2) ^ 1]->Acquire());
            const std::array<uint8_t, 4> current{uint8_t(frame % 2 ? 255 : 0), uint8_t(frame % 3 ? 255 : 0), 255, 255};
            graph.UpdatePassClearColor("write-history", current[0] / 255.f, current[1] / 255.f, 1, 1);
            assert(vkResetCommandBuffer(command, 0) == VK_SUCCESS);
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            assert(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
            graph.Execute(command);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = device.Resolve(output);
            barrier.size = 48;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                                 &barrier, 0, nullptr);
            assert(vkEndCommandBuffer(command) == VK_SUCCESS);
            assert(vkResetFences(context.GetDevice(), 1, &fence) == VK_SUCCESS);
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            assert(vkQueueSubmit(context.GetGraphicsQueue(), 1, &submit, fence) == VK_SUCCESS);
            assert(vkWaitForFences(context.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS);
            std::array<uint8_t, 48> pixels{};
            assert(device.ReadBuffer(output, 0, pixels.data(), pixels.size()));
            std::cerr << "history frame " << frame << " read=" << int(pixels[0]) << ',' << int(pixels[1]) << ','
                      << int(pixels[2]) << ',' << int(pixels[3]) << " write=" << int(pixels[24]) << ','
                      << int(pixels[25]) << ',' << int(pixels[26]) << ',' << int(pixels[27]) << '\n';
            for (size_t i = 0; i < 24; ++i) {
                assert(pixels[i] == previous[i % 4]);
                assert(pixels[i + 24] == current[i % 4]);
            }
            previous = current;
            device.CollectResourceRetirements(epoch++);
        }
        graph.Destroy();
        device.Release(output);
        std::cout << "PASS history raster-write/copy-read/rebind/zero-reset across 8 executions\n";
    }
    device.Release(pipeline);
    device.Release(layout);
    device.Release(shader);
    device.CollectDescriptorRetirements(epoch);
    device.CollectResourceRetirements(epoch);
    vkDestroyFence(context.GetDevice(), fence, nullptr);
    vkDestroyCommandPool(context.GetDevice(), pool, nullptr);
    context.Destroy();
    SDL_DestroyWindow(window);
    SDL_Quit();
}
