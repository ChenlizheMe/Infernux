#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/SceneRenderGraph.h>
#include <function/scene/Camera.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <string_view>
#include <utility>

using namespace infernux;

namespace
{

void CheckCameraHistoryResetIsolation()
{
    Camera source, other;
    SceneRenderGraph firstView, secondViewOfSource, unrelatedView;
    const glm::mat4 identity(1.0f);
    glm::mat4 moved(1.0f);
    moved[3][0] = -0.25f; // Below automatic camera-cut detection.
    firstView.SetCachedCameraVP(&source, identity, identity);
    secondViewOfSource.SetCachedCameraVP(&source, identity, identity);
    unrelatedView.SetCachedCameraVP(&other, identity, identity);
    firstView.CommitCameraHistory();
    secondViewOfSource.CommitCameraHistory();
    unrelatedView.CommitCameraHistory();

    source.ResetHistory();
    source.ResetHistory(); // Several requests before a frame still reset once per view.
    assert(source.GetTemporalHistoryRevision() == 2);
    assert(other.GetTemporalHistoryRevision() == 0);
    firstView.SetCachedCameraVP(&source, moved, identity);
    assert(firstView.GetPreviousViewProj() == moved);
    // A second view cannot lose the request because the first rendered earlier.
    secondViewOfSource.SetCachedCameraVP(&source, moved, identity);
    assert(secondViewOfSource.GetPreviousViewProj() == moved);
    unrelatedView.SetCachedCameraVP(&other, moved, identity);
    assert(unrelatedView.GetPreviousViewProj() == identity);

    firstView.CommitCameraHistory();
    firstView.SetCachedCameraVP(&source, identity, identity);
    assert(firstView.GetPreviousViewProj() == moved); // Accumulation resumes.
    firstView.ClearCachedViewSubmission();
    source.ResetHistory();
    firstView.SetCachedCameraVP(&source, moved, identity);
    assert(firstView.GetPreviousViewProj() == moved);
}

void CheckFullscreenState()
{
    RenderGraphDescription graph;
    graph.textures.push_back({"color", rhi::PixelFormat::RGBA8UNorm});
    graph.textures.push_back({"depth", rhi::PixelFormat::D32SFloat, false, true});
    GraphPassDesc initialize;
    initialize.name = "initialize";
    initialize.writeColors = {{0, "color"}};
    initialize.writeDepth = "depth";
    initialize.clearColor = initialize.clearDepth = true;
    graph.passes.push_back(initialize);
    GraphPassDesc overlay;
    overlay.name = "overlay";
    overlay.writeColors = {{0, "color"}};
    overlay.writeDepth = "depth";
    GraphCommandDesc command;
    command.type = GraphCommandType::FullscreenQuad;
    command.shaderName = "effect";
    command.depthTest = command.depthWrite = command.alphaBlend = true;
    command.depthCompare = rhi::CompareFunction::Always;
    overlay.commands.push_back(command);
    graph.passes.push_back(overlay);
    graph.outputTexture = "color";
    const auto valid = [](const auto &desc) { return SceneRenderGraph::ValidateGraphDescription(desc, 1); };
    assert(valid(graph));
    auto broken = graph;
    broken.passes.erase(broken.passes.begin());
    assert(!valid(broken)); // No implicit first-use attachment load.
    broken.passes[0].clearColor = broken.passes[0].clearDepth = true;
    assert(valid(broken));
    broken = graph;
    broken.passes.back().commands[0].depthTest = false;
    assert(!valid(broken)); // Vulkan depth writes require enabled depth test.
    broken = graph;
    broken.passes.back().writeDepth.clear();
    assert(!valid(broken));
    broken = graph;
    broken.passes.back().readTextures = {"depth"};
    assert(!valid(broken));
    broken = graph;
    broken.passes.back().commands[0].inputBindings = {{"_Depth", "depth"}};
    assert(!valid(broken));
    broken = graph;
    broken.passes.back().commands[0].type = GraphCommandType::DrawRenderers;
    assert(!valid(broken));

    // Cache identity includes every explicit raster state, not just shader name.
    FullscreenPipelineKey key;
    key.shaderName = "effect";
    std::unordered_map<FullscreenPipelineKey, int, FullscreenPipelineKeyHash> variants;
    variants.emplace(key, 0);
    key.depthFormat = rhi::PixelFormat::D32SFloat;
    variants.emplace(key, 1);
    key.depth.testEnabled = true;
    variants.emplace(key, 2);
    key.depth.writeEnabled = true;
    variants.emplace(key, 3);
    key.depth.compare = rhi::CompareFunction::Always;
    variants.emplace(key, 4);
    key.alphaBlend = true;
    variants.emplace(key, 5);
    assert(variants.size() == 6 && variants.at(key) == 5);
}

void CheckGraphBufferValidation()
{
    RenderGraphDescription graph;
    graph.textures.push_back({"color", rhi::PixelFormat::RGBA8UNorm});
    graph.buffers.push_back(
        {"producer/data", 64,
         static_cast<uint32_t>(GraphBufferUsage::Storage) | static_cast<uint32_t>(GraphBufferUsage::TransferSource)});
    graph.buffers.push_back({"consumer/data", 64, static_cast<uint32_t>(GraphBufferUsage::TransferDestination)});
    GraphPassDesc clear;
    clear.name = "clear";
    clear.writeColors = {{0, "color"}};
    clear.clearColor = true;
    graph.passes.push_back(clear);
    GraphPassDesc copy;
    copy.name = "copy";
    copy.type = GraphPassType::Copy;
    copy.bufferAccesses.push_back({"producer/data", GraphBufferAccessType::TransferRead});
    copy.bufferAccesses.push_back({"consumer/data", GraphBufferAccessType::TransferWrite});
    GraphCommandDesc command;
    command.type = GraphCommandType::CopyBuffer;
    command.sourceResource = "producer/data";
    command.destinationResource = "consumer/data";
    command.copyBytes = 64;
    copy.commands.push_back(command);
    graph.passes.push_back(copy);
    graph.outputTexture = "color";

    const auto valid = [](const auto &desc) { return SceneRenderGraph::ValidateGraphDescription(desc, 1); };
    assert(valid(graph));
    auto broken = graph;
    broken.passes.back().bufferAccesses[0].resource = "other_view/data";
    assert(!valid(broken));
    broken = graph;
    broken.buffers[0].usage = static_cast<uint32_t>(GraphBufferUsage::Storage);
    assert(!valid(broken));
    broken = graph;
    broken.passes.back().commands[0].copyBytes = 65;
    assert(!valid(broken));
}

void CheckViewLightListValidation()
{
    RenderGraphDescription graph;
    graph.textures.push_back({"color", rhi::PixelFormat::RGBA8UNorm});
    GraphBufferDesc lightList;
    lightList.name = "light_list";
    lightList.byteSize = sizeof(uint32_t) * 4;
    lightList.usage = static_cast<uint32_t>(GraphBufferUsage::Storage);
    lightList.viewLightList = true;
    graph.buffers.push_back(lightList);

    GraphPassDesc pass;
    pass.name = "light-consumer";
    pass.writeColors = {{0, "color"}};
    pass.clearColor = true;
    pass.bufferAccesses.push_back({"light_list", GraphBufferAccessType::StorageRead});
    GraphCommandDesc command;
    command.type = GraphCommandType::FullscreenQuad;
    command.shaderName = "light-consumer";
    command.inputBindings = {{"lightList", "light_list"}};
    pass.commands.push_back(command);
    graph.passes.push_back(pass);
    graph.outputTexture = "color";

    const auto valid = [](const auto &desc) { return SceneRenderGraph::ValidateGraphDescription(desc, 1); };
    assert(valid(graph));

    auto writable = graph;
    writable.passes.back().bufferAccesses[0].type = GraphBufferAccessType::StorageWrite;
    assert(!valid(writable));

    auto wrongUsage = graph;
    wrongUsage.buffers[0].usage =
        static_cast<uint32_t>(GraphBufferUsage::Storage) | static_cast<uint32_t>(GraphBufferUsage::TransferDestination);
    assert(!valid(wrongUsage));

    auto duplicate = graph;
    auto second = lightList;
    second.name = "second_light_list";
    second.viewLightList = true;
    duplicate.buffers.push_back(second);
    assert(!valid(duplicate));
}

void CheckSampledAssetTextureValidation()
{
    RenderGraphDescription graph;
    graph.textures.push_back({"color", rhi::PixelFormat::RGBA8UNorm});
    GraphTextureDesc volume;
    volume.name = "density";
    volume.role = GraphTextureRole::Asset;
    volume.assetGuid = "0123456789abcdef0123456789abcdef";
    volume.width = 8;
    volume.height = 8;
    volume.depth = 4;
    volume.isVolume = true;
    volume.samples = 1;
    graph.textures.push_back(volume);
    GraphPassDesc sample;
    sample.name = "sample";
    sample.writeColors = {{0, "color"}};
    sample.clearColor = true;
    GraphCommandDesc command;
    command.type = GraphCommandType::FullscreenQuad;
    command.shaderName = "Tests/VolumeInput";
    command.inputBindings = {{"density", "density"}};
    sample.commands.push_back(command);
    graph.passes.push_back(sample);
    graph.outputTexture = "color";

    const auto valid = [](const auto &desc) { return SceneRenderGraph::ValidateGraphDescription(desc, 1); };
    assert(valid(graph));
    auto broken = graph;
    broken.textures[1].assetGuid.clear();
    assert(!valid(broken));
    broken = graph;
    broken.textures[1].depth = 0;
    assert(!valid(broken));
    broken = graph;
    broken.textures[1].isVolume = false;
    assert(!valid(broken));
    broken = graph;
    broken.textures[1].samples = 4;
    assert(!valid(broken));
    broken = graph;
    broken.passes[0].writeColors[0].second = "density";
    assert(!valid(broken));
    broken = graph;
    broken.passes[0].commands.clear();
    broken.passes[0].readTextures.clear();
    broken.passes[0].name = "copy";
    broken.passes[0].type = GraphPassType::Copy;
    GraphCommandDesc copy;
    copy.type = GraphCommandType::CopyTexture;
    copy.sourceResource = "density";
    copy.destinationResource = "color";
    broken.passes[0].commands.push_back(copy);
    assert(!valid(broken));
}

void CheckWorldUIPassAttachments()
{
    RenderGraphDescription graph;
    graph.textures.push_back({"color", rhi::PixelFormat::RGBA8UNorm});
    graph.textures.push_back({"depth", rhi::PixelFormat::D32SFloat, false, true});
    graph.textures.push_back({"label_depth", rhi::PixelFormat::D32SFloat, false, true});
    GraphPassDesc ordinary;
    ordinary.name = "ordinary";
    ordinary.writeColors = {{0, "color"}};
    ordinary.writeDepth = "depth";
    ordinary.clearColor = ordinary.clearDepth = true;
    GraphCommandDesc command;
    command.type = GraphCommandType::DrawWorldUI;
    command.worldUILayerMask = 0xffffffffu ^ (1u << 30);
    ordinary.commands.push_back(command);
    graph.passes.push_back(ordinary);
    auto labels = ordinary;
    labels.name = "labels";
    labels.clearColor = false;
    labels.writeDepth = "label_depth";
    labels.commands[0].worldUILayerMask = 1u << 30;
    graph.passes.push_back(labels);
    graph.outputTexture = "color";
    const auto valid = [](const auto &desc) { return SceneRenderGraph::ValidateGraphDescription(desc, 1); };
    assert(valid(graph));
    auto broken = graph;
    broken.passes.back().writeDepth.clear();
    broken.passes.back().clearDepth = false;
    assert(!valid(broken)); // A filtered UI pass still requires explicit depth.
    broken = graph;
    broken.passes.back().writeColors.clear();
    assert(!valid(broken));
    graph.passes.back().commands[0].worldUILayerMask = 0;
    assert(valid(graph)); // Empty selection is intentional, not an error.
}

void CheckViewResourceSchedule()
{
    int screen, monitor, reflection, missing;
    const auto forward =
        RenderViewSchedule::Build({{{&monitor}, {&screen}}, {{&reflection}, {&monitor}}, {{}, {&reflection}}});
    assert((forward.order == std::vector<uint32_t>{2, 1, 0}));
    assert((forward.predecessors[0] == std::vector<uint32_t>{1}));
    const auto layered =
        RenderViewSchedule::Build({{{&monitor}, {&screen}}, {{}, {&monitor}}, {{}, {&screen}}, {{}, {&monitor}}});
    assert((layered.order == std::vector<uint32_t>{1, 3, 0, 2}));
    const auto independent = RenderViewSchedule::Build({{{}, {&screen}}, {{}, {&monitor}}});
    assert((independent.order == std::vector<uint32_t>{0, 1}));
    assert(independent.predecessors[1].empty());
    for (const auto &invalid : std::vector<std::vector<RenderViewAccess>>{
             {{{&missing}, {&screen}}},
             {{{&monitor}, {&monitor}}},
             {{{&monitor}, {&screen}}, {{&screen}, {&monitor}}},
         }) {
        bool rejected = false;
        try {
            (void)RenderViewSchedule::Build(invalid);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
    }
    // The camera's resource is consumed by several views; consumers are not
    // unnecessarily serialized against each other by unrelated Camera.depth.
    const auto fanout =
        RenderViewSchedule::Build({{{&monitor}, {&screen}}, {{&monitor}, {&reflection}}, {{}, {&monitor}}});
    assert((fanout.order == std::vector<uint32_t>{2, 0, 1}));
    assert(fanout.predecessors[0] == fanout.predecessors[1]);
}

void CheckViewMaterialContracts()
{
    RenderGraphDescription graph;
    graph.msaaSamples = 8; // An authored request is not an allocated target.
    graph.textures.push_back({"color", rhi::PixelFormat::RGBA16SFloat, true, false, 0, 0, 0, 0});
    graph.textures.push_back({"depth", rhi::PixelFormat::D32SFloat, false, true, 0, 0, 0, 0});
    graph.textures.push_back({"fixed", rhi::PixelFormat::RGBA16SFloat, false, false, 37, 23, 0, 2});
    graph.textures.push_back({"fixedDepth", rhi::PixelFormat::D32SFloat, false, true, 37, 23, 0, 2});
    graph.textures.push_back({"reducedDepth", rhi::PixelFormat::D32SFloat, false, true, 0, 0, 2, 1});
    GraphPassDesc pass;
    pass.commands.push_back({GraphCommandType::DrawRenderers});
    pass.writeColors = {{0, "color"}};
    pass.writeDepth = "depth";
    rhi::RenderViewContext view;
    view.colorFormat = rhi::PixelFormat::RGBA8UNorm;
    view.depthFormat = rhi::PixelFormat::D24UNormS8UInt;
    view.samples = rhi::SampleCount::One;
    const auto first = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(first.colorFormats == std::vector{rhi::PixelFormat::RGBA8UNorm});
    assert(first.depthFormat == rhi::PixelFormat::D24UNormS8UInt);
    assert(first.samples == rhi::SampleCount::One);
    assert(!first.depthReadOnly && first.IsValid());
    view.samples = rhi::SampleCount::Four;
    const auto second = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(second.samples == rhi::SampleCount::Four);
    assert(first.RenderingSignature() != second.RenderingSignature());
    view.samples = rhi::SampleCount::One;
    view.colorFormat = rhi::PixelFormat::RGBA16SFloat;
    const auto third = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(first.RenderingSignature() != third.RenderingSignature());
    assert(first.samples == rhi::SampleCount::One); // Other views were not mutated.

    // A viewport-sized mask is not the Camera depth just because its dimensions match.
    auto independentDepth = graph.textures[1];
    independentDepth.name = "selection_depth";
    independentDepth.samples = 1;
    assert(graph.textures[1].IsViewDepth() && !independentDepth.IsViewDepth());
    graph.textures.push_back(independentDepth);
    auto independentPass = pass;
    independentPass.writeColors = {{0, "fixed"}};
    independentPass.writeDepth = independentDepth.name;
    const auto independent = SceneRenderGraph::ResolveMaterialPass(graph, independentPass, view);
    assert(independent.depthFormat == rhi::PixelFormat::D32SFloat);
    assert(independent.samples == rhi::SampleCount::One);

    // World UI uses the same actual attachments as geometry, including an
    // offscreen camera's HDR/MSAA and depth-stencil contract.
    pass.commands[0].type = GraphCommandType::DrawWorldUI;
    for (const auto samples : {rhi::SampleCount::One, rhi::SampleCount::Four}) {
        view.samples = samples;
        const auto world = SceneRenderGraph::ResolveMaterialPass(graph, pass, view).RenderingSignature();
        assert(world.colorFormats[0] == view.colorFormat);
        assert(world.depthFormat == view.depthFormat);
        assert(world.stencilFormat == rhi::PixelFormat::D24UNormS8UInt);
        assert(world.samples == samples);
    }
    pass.commands[0].type = GraphCommandType::DrawRenderers;
    view.samples = rhi::SampleCount::One;

    // A custom raster attachment retains its own format/sample/depth contract.
    pass.writeColors = {{0, "fixed"}};
    pass.writeDepth = "fixedDepth";
    const auto fixed = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(fixed.colorFormats == std::vector{rhi::PixelFormat::RGBA16SFloat});
    assert(fixed.depthFormat == rhi::PixelFormat::D32SFloat);
    assert(fixed.samples == rhi::SampleCount::Two);

    // Skybox and transparency attach the view's depth read-only. Sampling it
    // instead must not accidentally attach it or turn on fixed-function depth.
    pass.commands[0].type = GraphCommandType::DrawSkybox;
    pass.writeColors = {{0, "color"}};
    pass.writeDepth.clear();
    pass.readTextures = {"depth"};
    auto read = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(read.depthFormat == view.depthFormat && read.depthReadOnly);
    pass.commands[0].inputBindings = {{"_Depth", "depth"}};
    read = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(read.depthFormat == rhi::PixelFormat::Undefined && !read.depthReadOnly);
    pass.commands[0].inputBindings.clear();
    pass.readTextures = {"reducedDepth"};
    read = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(read.depthFormat == rhi::PixelFormat::D32SFloat && read.depthReadOnly);

    pass.commands[0].type = GraphCommandType::DrawRenderers;
    pass.commands[0].shaderTarget = ShaderCompileTarget::Depth;
    pass.writeColors.clear();
    pass.readTextures.clear();
    pass.writeDepth = "fixedDepth";
    const auto depthOnly = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert(depthOnly.colorFormats.empty() && depthOnly.IsValid());
    assert(depthOnly.samples == rhi::SampleCount::Two);
    const auto reflectedDepth = SceneRenderGraph::ResolveMaterialPass(graph, pass, view, true);
    assert(reflectedDepth.invertCulling && reflectedDepth != depthOnly);
    assert(reflectedDepth.RenderingSignature() == depthOnly.RenderingSignature());
    pass.commands[0].type = GraphCommandType::DrawShadowCasters;
    pass.commands[0].shaderTarget = ShaderCompileTarget::Shadow;
    const auto lightSpace = SceneRenderGraph::ResolveMaterialPass(graph, pass, view, true);
    assert(lightSpace.IsValid() && !lightSpace.invertCulling);
    pass.commands[0].type = GraphCommandType::DrawRenderers;

    // Attachment slots, not author declaration order, determine the signature.
    pass.commands[0].shaderTarget = ShaderCompileTarget::Forward;
    pass.writeDepth.clear();
    graph.textures[0].format = rhi::PixelFormat::RGBA8UNorm;
    graph.textures[0].isBackbuffer = false;
    graph.textures[0].samples = 2;
    pass.writeColors = {{1, "fixed"}, {0, "color"}};
    const auto mrt = SceneRenderGraph::ResolveMaterialPass(graph, pass, view);
    assert((mrt.colorFormats == std::vector{rhi::PixelFormat::RGBA8UNorm, rhi::PixelFormat::RGBA16SFloat}));
    assert(mrt.samples == rhi::SampleCount::Two);
}

RenderGraphDescription MakeShadowGraph(ShaderCompileTarget target)
{
    RenderGraphDescription graph;
    graph.textures.push_back({"ShadowDepth", rhi::PixelFormat::D32SFloat, false, true, 16, 16, 0, 1});

    GraphPassDesc pass;
    pass.name = "ShadowCasters";
    pass.commands.push_back({GraphCommandType::DrawShadowCasters});
    pass.commands.front().shaderTarget = target;
    pass.writeDepth = "ShadowDepth";
    graph.passes.push_back(std::move(pass));
    return graph;
}

RenderGraphDescription MakeMotionGraph(rhi::PixelFormat colorFormat, bool readableDepth)
{
    RenderGraphDescription graph;
    graph.textures.push_back({"Motion", colorFormat, false, false, 0, 0, 0, 1});
    graph.textures.push_back({"Depth", rhi::PixelFormat::D32SFloat, false, true, 0, 0, 0, 1});

    GraphPassDesc pass;
    pass.name = "Motion";
    pass.commands.push_back({GraphCommandType::DrawRenderers});
    pass.commands.front().shaderTarget = ShaderCompileTarget::Motion;
    pass.writeColors.push_back({0, "Motion"});
    if (readableDepth)
        pass.readTextures.push_back("Depth");
    graph.passes.push_back(std::move(pass));
    graph.outputTexture = "Motion";
    return graph;
}

RenderGraphDescription MakeTemporalGraph(bool completePair, bool singleSample = true)
{
    RenderGraphDescription graph;
    GraphTextureDesc read{"HistoryRead", rhi::PixelFormat::RGBA16SFloat, false, false, 0, 0, 0, singleSample ? 1u : 4u};
    read.role = GraphTextureRole::TemporalRead;
    read.temporalKey = "taa";
    graph.textures.push_back(read);
    if (completePair) {
        GraphTextureDesc write{"HistoryWrite", rhi::PixelFormat::RGBA16SFloat, false, false, 0, 0, 0, 1};
        write.role = GraphTextureRole::TemporalWrite;
        write.temporalKey = "taa";
        graph.textures.push_back(write);
        GraphPassDesc commit;
        commit.name = "HistoryRasterWrite";
        commit.writeColors.emplace_back(0, write.name);
        commit.clearColor = true;
        graph.passes.push_back(commit);
    }
    return graph;
}

RenderGraphDescription MakeNormalGraph(rhi::PixelFormat colorFormat, bool readableDepth)
{
    RenderGraphDescription graph;
    graph.textures.push_back({"Normal", colorFormat, false, false, 0, 0, 0, 1});
    graph.textures.push_back({"Depth", rhi::PixelFormat::D32SFloat, false, true, 0, 0, 0, 1});

    GraphPassDesc pass;
    pass.name = "Normal";
    pass.commands.push_back({GraphCommandType::DrawRenderers});
    pass.commands.front().shaderTarget = ShaderCompileTarget::Normal;
    pass.writeColors.push_back({0, "Normal"});
    if (readableDepth)
        pass.readTextures.push_back("Depth");
    graph.passes.push_back(std::move(pass));
    graph.outputTexture = "Normal";
    return graph;
}

RenderGraphDescription MakeBaseColorGraph(rhi::PixelFormat colorFormat, bool readableDepth)
{
    RenderGraphDescription graph;
    graph.textures.push_back({"BaseColor", colorFormat, false, false, 0, 0, 0, 1});
    graph.textures.push_back({"Depth", rhi::PixelFormat::D32SFloat, false, true, 0, 0, 0, 1});

    GraphPassDesc pass;
    pass.name = "BaseColor";
    pass.commands.push_back({GraphCommandType::DrawRenderers});
    pass.commands.front().shaderTarget = ShaderCompileTarget::BaseColor;
    pass.writeColors.push_back({0, "BaseColor"});
    if (readableDepth)
        pass.readTextures.push_back("Depth");
    graph.passes.push_back(std::move(pass));
    graph.outputTexture = "BaseColor";
    return graph;
}

RenderGraphDescription MakeAttachmentGraph(int secondColorSlot, uint32_t readOnlyDepthCount, bool writesDepth)
{
    RenderGraphDescription graph;
    graph.textures.push_back({"Color0", rhi::PixelFormat::RGBA16SFloat, false, false, 0, 0, 0, 1});
    graph.textures.push_back({"Color1", rhi::PixelFormat::RGBA16SFloat, false, false, 0, 0, 0, 1});
    graph.textures.push_back({"Depth0", rhi::PixelFormat::D32SFloat, false, true, 0, 0, 0, 1});
    graph.textures.push_back({"Depth1", rhi::PixelFormat::D32SFloat, false, true, 0, 0, 0, 1});

    GraphPassDesc pass;
    pass.name = "AttachmentContract";
    pass.commands.push_back({GraphCommandType::DrawRenderers});
    pass.commands.front().shaderTarget = ShaderCompileTarget::Forward;
    pass.writeColors = {{0, "Color0"}, {secondColorSlot, "Color1"}};
    if (readOnlyDepthCount > 0)
        pass.readTextures.push_back("Depth0");
    if (readOnlyDepthCount > 1)
        pass.readTextures.push_back("Depth1");
    if (writesDepth)
        pass.writeDepth = "Depth1";
    graph.passes.push_back(std::move(pass));
    graph.outputTexture = "Color0";
    return graph;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--graph-buffer-contract") {
        CheckGraphBufferValidation();
        return 0;
    }
    CheckCameraHistoryResetIsolation();
    CheckFullscreenState();
    CheckGraphBufferValidation();
    CheckViewLightListValidation();
    CheckSampledAssetTextureValidation();
    CheckWorldUIPassAttachments();
    CheckViewResourceSchedule();
    CheckViewMaterialContracts();
    const auto invalidTarget = MakeShadowGraph(ShaderCompileTarget::Forward);
    assert(!SceneRenderGraph::ValidateGraphDescription(invalidTarget, 1));

    const auto validTarget = MakeShadowGraph(ShaderCompileTarget::Shadow);
    assert(SceneRenderGraph::ValidateGraphDescription(validTarget, 1));

    assert(SceneRenderGraph::ValidateGraphDescription(MakeMotionGraph(rhi::PixelFormat::RG16SFloat, true), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeMotionGraph(rhi::PixelFormat::RGBA8UNorm, true), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeMotionGraph(rhi::PixelFormat::RG16SFloat, false), 1));
    assert(SceneRenderGraph::ValidateGraphDescription(MakeNormalGraph(rhi::PixelFormat::RGBA16SFloat, true), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeNormalGraph(rhi::PixelFormat::RGBA8UNorm, true), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeNormalGraph(rhi::PixelFormat::RGBA16SFloat, false), 1));
    assert(SceneRenderGraph::ValidateGraphDescription(MakeBaseColorGraph(rhi::PixelFormat::RGBA16SFloat, true), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeBaseColorGraph(rhi::PixelFormat::RGBA8UNorm, true), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeBaseColorGraph(rhi::PixelFormat::RGBA16SFloat, false), 1));
    assert(SceneRenderGraph::ValidateGraphDescription(MakeTemporalGraph(true), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeTemporalGraph(false), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeTemporalGraph(true, false), 1));
    auto historyContract = MakeTemporalGraph(true);
    for (auto &texture : historyContract.textures) {
        texture.width = 37;
        texture.height = 23;
    }
    assert(SceneRenderGraph::ValidateGraphDescription(historyContract, 1));
    historyContract.textures[0].width = 38;
    assert(!SceneRenderGraph::ValidateGraphDescription(historyContract, 1));
    historyContract = MakeTemporalGraph(true);
    historyContract.passes.clear();
    assert(!SceneRenderGraph::ValidateGraphDescription(historyContract, 1));
    historyContract = MakeTemporalGraph(true);
    historyContract.passes[0].writeColors[0].second = "HistoryRead";
    assert(!SceneRenderGraph::ValidateGraphDescription(historyContract, 1));
    historyContract = MakeTemporalGraph(true);
    GraphPassDesc readCurrent;
    readCurrent.name = "ReadCurrent";
    readCurrent.readTextures.push_back("HistoryWrite");
    historyContract.passes.push_back(readCurrent);
    assert(SceneRenderGraph::ValidateGraphDescription(historyContract, 1));
    std::swap(historyContract.passes[0], historyContract.passes[1]);
    assert(!SceneRenderGraph::ValidateGraphDescription(historyContract, 1));
    historyContract = MakeTemporalGraph(true);
    historyContract.passes[0].readTextures.push_back("HistoryWrite");
    assert(!SceneRenderGraph::ValidateGraphDescription(historyContract, 1));
    assert(SceneRenderGraph::ValidateGraphDescription(MakeAttachmentGraph(1, 1, false), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeAttachmentGraph(2, 1, false), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeAttachmentGraph(1, 2, false), 1));
    assert(!SceneRenderGraph::ValidateGraphDescription(MakeAttachmentGraph(1, 1, true), 1));

    const glm::vec2 firstJitter = SceneRenderGraph::ComputeTemporalJitterNdc(0, 200, 100);
    assert(std::abs(firstJitter.x) < 1e-7f);
    assert(std::abs(firstJitter.y + (1.0f / 300.0f)) < 1e-6f);
    assert(SceneRenderGraph::ComputeTemporalJitterNdc(0, 200, 100) ==
           SceneRenderGraph::ComputeTemporalJitterNdc(8, 200, 100));
    assert(SceneRenderGraph::ComputeTemporalJitterNdc(0, 0, 100) == glm::vec2(0.0f));

    glm::mat4 perspective(1.0f);
    perspective[2][3] = -1.0f;
    const glm::vec2 offset{0.01f, -0.02f};
    const glm::mat4 jitteredPerspective = SceneRenderGraph::ApplyTemporalJitter(perspective, offset);
    assert(SceneRenderGraph::ProjectionForPass(perspective, offset, false) == jitteredPerspective);
    assert(SceneRenderGraph::ProjectionForPass(perspective, offset, true) == perspective);
    assert(std::abs(jitteredPerspective[2][0] + 0.01f) < 1e-7f);
    assert(std::abs(jitteredPerspective[2][1] - 0.02f) < 1e-7f);

    glm::mat4 orthographic(1.0f);
    const glm::mat4 jitteredOrthographic = SceneRenderGraph::ApplyTemporalJitter(orthographic, offset);
    assert(SceneRenderGraph::ProjectionForPass(orthographic, offset, false) == jitteredOrthographic);
    assert(SceneRenderGraph::ProjectionForPass(orthographic, offset, true) == orthographic);
    assert(std::abs(jitteredOrthographic[3][0] - 0.01f) < 1e-7f);
    assert(std::abs(jitteredOrthographic[3][1] + 0.02f) < 1e-7f);

    DrawCall dynamicCaster;
    assert(SceneRenderGraph::ShadowCasterRequiresContinuousUpdate(dynamicCaster));
    dynamicCaster.isStatic = true;
    assert(!SceneRenderGraph::ShadowCasterRequiresContinuousUpdate(dynamicCaster));
    std::vector<glm::mat4> skinPose{glm::mat4(1.0f)};
    dynamicCaster.skinBoneMatrices = &skinPose;
    assert(SceneRenderGraph::ShadowCasterRequiresContinuousUpdate(dynamicCaster));
    dynamicCaster.skinBoneMatrices = nullptr;
    dynamicCaster.previousSkinBoneMatrices = &skinPose;
    assert(SceneRenderGraph::ShadowCasterRequiresContinuousUpdate(dynamicCaster));

    SceneRenderGraph graph;
    graph.SetCachedRendererList(RendererList{});
    graph.SetCachedSubmissionSignature(42, {}, 7);
    assert(graph.CanReuseCachedSubmission(42, 7));
    assert(!graph.CanReuseCachedSubmission(42, 8));
    assert(!graph.CanReuseCachedSubmission(42, 0));
    assert(!graph.CanReuseCachedSubmission(0, 7));
    graph.ClearCachedViewSubmission();
    assert(!graph.HasCachedDrawCalls());
    assert(!graph.CanReuseCachedSubmission(42, 7));

    graph.SetCachedRendererList(RendererList{});
    graph.SetCachedSubmissionSignature(42, {}, 7);
    graph.InvalidateParticleViews();
    assert(graph.NeedsRebuild());
    assert(!graph.IsGraphBuilt());
    graph.ClearCachedFrameState();
    assert(!graph.HasCachedDrawCalls());
    assert(graph.NeedsRebuild());
    assert(!graph.IsGraphBuilt());
    return 0;
}
