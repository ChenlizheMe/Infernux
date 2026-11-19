#include "RenderGraphDescription.h"
#include "FullscreenRenderer.h"
#include "MaterialPassPipeline.h"
#include "MsaaPolicy.h"
#include "RendererSelection.h"
#include "rhi/RenderViewContext.h"
#include "rhi/RhiComputeBuffer.h"
#include "rhi/RhiComputeKernel.h"
#include "rhi/RhiRenderTexture.h"
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/scene/Camera.h>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace infernux
{
bool GraphUsesForwardPlus(const RenderGraphDescription &graph)
{
    for (const auto &pass : graph.passes)
        for (const auto &command : pass.commands) {
            if (command.type == GraphCommandType::DrawRenderers && IsForwardPlusMaterialPass(command.shaderTarget))
                return true;
            const auto &program = command.fullscreenProgram;
            if (program && (program->UsesViewBinding(2) || program->UsesViewBinding(3)))
                return true;
        }
    return false;
}

GraphRayTailUse ResolveGraphRayTailUse(const RenderGraphDescription &graph)
{
    bool prepare = false, merge = false;
    for (const auto &pass : graph.passes) {
        if (const auto *command = pass.PrimaryCommand()) {
            prepare |= command->type == GraphCommandType::RayTailPrepare;
            merge |= command->type == GraphCommandType::RayTailMerge;
        }
    }
    if (prepare != merge)
        throw std::invalid_argument("A ray cache update requires both its prepare and merge passes");
    if (merge)
        return GraphRayTailUse::Update;
    const bool reads = std::any_of(graph.buffers.begin(), graph.buffers.end(), [](const GraphBufferDesc &buffer) {
        return buffer.frameBuffer == GraphFrameBuffer::WorldRayTailRead;
    });
    return reads ? GraphRayTailUse::Read : GraphRayTailUse::None;
}

std::shared_ptr<const ShaderTargetCode> ResolveGraphRayLitProgram(const RenderGraphDescription &graph)
{
    const auto isLitTable = [](const GraphBufferDesc &buffer) {
        return buffer.frameBuffer == GraphFrameBuffer::WorldRayLitMaterials;
    };
    const bool hasTables = std::any_of(graph.buffers.begin(), graph.buffers.end(), isLitTable);
    const bool hasPages = std::any_of(graph.textures.begin(), graph.textures.end(),
                                      [](const auto &texture) { return texture.IsWorldRayPage(); });
    if (!hasTables && !hasPages)
        return {};
    constexpr std::array bufferPorts{
        std::pair{"world_ray_geometry", GraphFrameBuffer::WorldRayGeometry},
        std::pair{"world_ray_meshes", GraphFrameBuffer::WorldRayMeshes},
        std::pair{"ray_lit_materials", GraphFrameBuffer::WorldRayLitMaterials},
    };
    constexpr std::array pagePorts{
        std::pair{"ray_color_pages", GraphTextureRole::WorldRayColorPages},
        std::pair{"ray_data_pages", GraphTextureRole::WorldRayDataPages},
    };

    std::shared_ptr<const ShaderTargetCode> program;
    const ShaderResourceLayout *layout = nullptr;
    for (const auto &pass : graph.passes) {
        const auto *command = pass.PrimaryCommand();
        if (!command ||
            (command->type != GraphCommandType::FullscreenQuad && command->type != GraphCommandType::ComputeWork))
            continue;
        const bool readsTables =
            std::any_of(command->inputBindings.begin(), command->inputBindings.end(), [&](const auto &binding) {
                return std::any_of(graph.buffers.begin(), graph.buffers.end(), [&](const auto &buffer) {
                    return isLitTable(buffer) && buffer.name == binding.second;
                });
            });
        if (!readsTables)
            continue;
        const auto bound = [&](const char *port) {
            const auto it = std::find_if(command->inputBindings.begin(), command->inputBindings.end(),
                                         [&](const auto &binding) { return binding.first == port; });
            return it == command->inputBindings.end() ? std::string_view{} : std::string_view(it->second);
        };
        for (const auto &[name, source] : bufferPorts) {
            const auto resource = bound(name);
            if (std::none_of(graph.buffers.begin(), graph.buffers.end(),
                             [&](const auto &item) { return item.frameBuffer == source && item.name == resource; }))
                throw std::invalid_argument("World ray Lit pass must bind its World buffer: " + std::string(name));
        }
        for (const auto &[name, source] : pagePorts) {
            const auto resource = bound(name);
            if (std::none_of(graph.textures.begin(), graph.textures.end(),
                             [&](const auto &item) { return item.role == source && item.name == resource; }))
                throw std::invalid_argument("World ray Lit pass must bind its World page: " + std::string(name));
        }
        std::vector<std::shared_ptr<const ShaderTargetCode>> candidates;
        if (command->type == GraphCommandType::FullscreenQuad) {
            if (!command->fullscreenProgram)
                throw std::invalid_argument("World ray Lit pass has no captured shader: " + pass.name);
            candidates.push_back(command->fullscreenProgram->code);
        } else {
            const auto *work = FindGraphWorkBinding(graph, command->workBinding);
            for (const auto &dispatch : work->recipe->GetDispatches())
                if (dispatch.kernel->GetGraphProgram())
                    candidates.push_back(dispatch.kernel->GetGraphProgram());
        }
        for (const auto &candidate : candidates) {
            const auto reflected =
                std::find_if(candidate->resources.begin(), candidate->resources.end(), [](const auto &r) {
                    return r.name == "ray_lit_materials" && r.set == 0 &&
                           r.binding.type == rhi::BindingType::StorageBuffer && r.binding.readOnlyStorage;
                });
            if (reflected == candidate->resources.end())
                throw std::invalid_argument("World ray Lit pass must read its material table: " + pass.name);
            if (!program) {
                program = candidate;
                layout = &*reflected;
                continue;
            }
            if (reflected->elementStride != layout->elementStride || !(reflected->value == layout->value))
                throw std::invalid_argument("World ray Lit pass uses a different table layout: " + pass.name);
        }
    }
    if (!program)
        throw std::invalid_argument("World ray Lit resources require a captured graphics or compute consumer");
    return program;
}

const GraphTextureDesc *FindViewShadowAtlas(const RenderGraphDescription &graph) noexcept
{
    for (const auto &pass : graph.passes) {
        const auto *command = pass.PrimaryCommand();
        if (command && command->type == GraphCommandType::DrawShadowCasters) {
            const auto texture = std::find_if(graph.textures.begin(), graph.textures.end(),
                                              [&](const auto &entry) { return entry.name == pass.writeDepth; });
            return texture == graph.textures.end() ? nullptr : &*texture;
        }
    }
    return nullptr;
}

bool ReadsViewShadow(const GraphCommandDesc &command) noexcept
{
    if (command.type == GraphCommandType::FullscreenQuad)
        return command.fullscreenProgram && command.fullscreenProgram->UsesViewBinding(0);
    return command.type == GraphCommandType::DrawRenderers &&
           (command.shaderTarget == ShaderCompileTarget::Forward || IsForwardPlusMaterialPass(command.shaderTarget));
}

std::optional<GraphPassDesc> ResolveGraphCameraClear(const RenderGraphDescription &graph, CameraClearFlags flags,
                                                     const glm::vec4 &linearBackground, bool preview)
{
    const auto persistent = [&](const std::string &name) {
        return std::any_of(graph.textures.begin(), graph.textures.end(), [&](const auto &texture) {
            return texture.name == name && texture.role == GraphTextureRole::Persistent;
        });
    };
    if (preview && (flags == CameraClearFlags::DepthOnly || flags == CameraClearFlags::DontClear))
        flags = CameraClearFlags::SolidColor;
    for (const auto &pass : graph.passes) {
        if (!pass.clearColor || persistent(pass.writeDepth) || persistent(pass.resolveColor) ||
            std::any_of(pass.writeColors.begin(), pass.writeColors.end(),
                        [&](const auto &output) { return persistent(output.second); }))
            continue;
        auto result = pass;
        result.clearDepth = !pass.writeDepth.empty() && flags != CameraClearFlags::DontClear;
        result.clearColor = flags == CameraClearFlags::SolidColor || flags == CameraClearFlags::Skybox;
        if (flags == CameraClearFlags::SolidColor) {
            result.clearColorR = linearBackground.r;
            result.clearColorG = linearBackground.g;
            result.clearColorB = linearBackground.b;
            result.clearColorA = linearBackground.a;
        }
        return result;
    }
    return std::nullopt;
}

namespace
{
GraphTextureDesc::RenderTextureLayout AttachmentLayout(const rhi::RenderTextureGeneration &generation) noexcept
{
    const auto &desc = generation.description;
    return {desc.colorFormat, desc.depthFormat, desc.samples, desc.filter, desc.storage, desc.sampledDepth};
}

uint32_t AttachmentSamples(const rhi::RenderTextureGeneration &generation, GraphTextureAttachment attachment) noexcept
{
    return attachment == GraphTextureAttachment::Resolve ? 1u : static_cast<uint32_t>(generation.description.samples);
}

bool BufferDescEquals(const GraphBufferDesc &a, const GraphBufferDesc &b)
{
    return a.name == b.name && a.byteSize == b.byteSize && a.byteOffset == b.byteOffset && a.usage == b.usage &&
           (a.inputAuthor ? a.inputAuthor : a.input) == (b.inputAuthor ? b.inputAuthor : b.input) &&
           (a.input || (a.bufferAuthor ? a.bufferAuthor : a.computeBuffer) ==
                           (b.bufferAuthor ? b.bufferAuthor : b.computeBuffer)) &&
           a.frameBuffer == b.frameBuffer && a.workBinding == b.workBinding && a.workSlot == b.workSlot &&
           a.exportLayout.has_value() == b.exportLayout.has_value() && a.exportShape == b.exportShape &&
           (!a.exportLayout || (a.exportLayout->elementCount == b.exportLayout->elementCount &&
                                a.exportLayout->scalarType == b.exportLayout->scalarType &&
                                a.exportLayout->lanes == b.exportLayout->lanes));
}

bool BufferAccessEquals(const GraphBufferAccessDesc &a, const GraphBufferAccessDesc &b)
{
    return a.resource == b.resource && a.type == b.type;
}

bool CommandDescEquals(const GraphCommandDesc &a, const GraphCommandDesc &b)
{
    const auto parameterLayoutEquals = [](const GraphCommandDesc &lhs, const GraphCommandDesc &rhs) {
        if (lhs.parameterBlock.empty())
            return lhs.parameters == rhs.parameters;
        if (lhs.parameters.size() != rhs.parameters.size())
            return false;
        for (size_t i = 0; i < lhs.parameters.size(); ++i) {
            if (lhs.parameters[i].first != rhs.parameters[i].first ||
                lhs.parameters[i].second.Kind() != rhs.parameters[i].second.Kind())
                return false;
        }
        return true;
    };
    return a.type == b.type && a.shaderTarget == b.shaderTarget && a.materialFilter == b.materialFilter &&
           a.queueMin == b.queueMin && a.queueMax == b.queueMax && a.sortMode == b.sortMode && a.passTag == b.passTag &&
           a.overrideMaterial == b.overrideMaterial &&
           (a.selectionAuthor ? a.selectionAuthor : a.rendererSelection) ==
               (b.selectionAuthor ? b.selectionAuthor : b.rendererSelection) &&
           a.depthTest == b.depthTest && a.depthWrite == b.depthWrite && a.depthCompare == b.depthCompare &&
           a.alphaBlend == b.alphaBlend && a.lightIndex == b.lightIndex && a.screenUIList == b.screenUIList &&
           a.worldUILayerMask == b.worldUILayerMask && a.shaderName == b.shaderName &&
           a.parameterBlock == b.parameterBlock && parameterLayoutEquals(a, b) && a.inputBindings == b.inputBindings &&
           a.sourceResource == b.sourceResource && a.destinationResource == b.destinationResource &&
           a.copyBytes == b.copyBytes && a.copyWidth == b.copyWidth && a.copyHeight == b.copyHeight &&
           a.copyBufferOffset == b.copyBufferOffset && a.workBinding == b.workBinding;
}

bool CommandListEquals(const std::vector<GraphCommandDesc> &a, const std::vector<GraphCommandDesc> &b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const auto &lhs, const auto &rhs) {
               return CommandDescEquals(lhs, rhs);
           });
}

bool PassDescEquals(const GraphPassDesc &a, const GraphPassDesc &b)
{
    return a.name == b.name && a.type == b.type && a.readTextures == b.readTextures && a.writeColors == b.writeColors &&
           a.writeDepth == b.writeDepth && a.resolveColor == b.resolveColor && a.clearColor == b.clearColor &&
           a.clearDepth == b.clearDepth && a.clearColorR == b.clearColorR && a.clearColorG == b.clearColorG &&
           a.clearColorB == b.clearColorB && a.clearColorA == b.clearColorA && a.clearDepthValue == b.clearDepthValue &&
           a.sideEffect == b.sideEffect && a.events == b.events && a.bufferAccesses.size() == b.bufferAccesses.size() &&
           std::equal(a.bufferAccesses.begin(), a.bufferAccesses.end(), b.bufferAccesses.begin(), BufferAccessEquals) &&
           CommandListEquals(a.commands, b.commands);
}
} // namespace

std::shared_ptr<const rhi::RenderTextureGeneration> GraphTextureDesc::AcquireRenderTextureGeneration() const
{
    return renderTextureGeneration ? renderTextureGeneration : renderTexture ? renderTexture->Acquire() : nullptr;
}

RenderGraphDescription CaptureRenderGraphDescription(
    const RenderGraphDescription &source,
    const std::function<void(const std::shared_ptr<InxMaterial> &)> &prepareMaterial,
    const std::function<std::shared_ptr<const FullscreenProgram>(const std::string &)> &acquireProgram)
{
    auto result = source;
    const auto capture = [](const std::shared_ptr<rhi::ComputeBuffer> &buffer) {
        return !buffer || buffer->GetAllocation()->IsGraphWritable() ? buffer : buffer->Freeze();
    };
    for (auto &texture : result.textures)
        if (texture.renderTexture)
            texture.renderTextureGeneration = texture.renderTexture->Acquire();
    for (auto &buffer : result.buffers) {
        if (buffer.input) {
            buffer.inputAuthor = buffer.input;
            buffer.input = std::make_shared<GraphBufferInput>(*buffer.input);
            buffer.computeBuffer = buffer.input->GetValue();
        } else if (buffer.computeBuffer) {
            buffer.bufferAuthor = buffer.computeBuffer;
            buffer.computeBuffer = capture(buffer.computeBuffer);
        }
    }
    for (auto &binding : result.workBindings) {
        binding.externalInputs.clear();
        for (const auto &slot : binding.recipe->GetSlots())
            binding.externalInputs.push_back(capture(slot.external));
    }
    std::unordered_map<const RendererSelection *, std::shared_ptr<RendererSelection>> selections;
    for (auto &pass : result.passes)
        for (auto &command : pass.commands) {
            if (command.type == GraphCommandType::FullscreenQuad)
                command.fullscreenProgram = acquireProgram(command.shaderName);
            if (command.rendererSelection) {
                command.selectionAuthor = command.rendererSelection;
                auto &selected = selections[command.rendererSelection.get()];
                if (!selected) {
                    prepareMaterial(command.rendererSelection->Material());
                    selected = command.rendererSelection->CaptureRenderPublication();
                }
                command.rendererSelection = selected;
            } else if (!command.overrideMaterial.empty()) {
                auto &registry = AssetRegistry::Instance();
                auto material = registry.GetBuiltinMaterial(command.overrideMaterial);
                if (!material)
                    material = registry.LoadAsset<InxMaterial>(command.overrideMaterial, ResourceType::Material);
                if (!material)
                    throw std::invalid_argument("Render override material is unavailable: " + command.overrideMaterial);
                prepareMaterial(material);
                command.overrideMaterialValue = material->CaptureRenderPublication();
            }
        }
    return result;
}

std::shared_ptr<const RenderGraphDescription> AcquireRenderGraphPublication(
    const RenderGraphDescription &source,
    const std::function<void(const std::shared_ptr<InxMaterial> &)> &prepareMaterial,
    const std::function<std::shared_ptr<const FullscreenProgram>(const std::string &)> &acquireProgram)
{
    const bool immutable =
        source.sourceRevision != 0 && source.workBindings.empty() &&
        std::none_of(source.textures.begin(), source.textures.end(),
                     [](const auto &texture) {
                         return texture.role == GraphTextureRole::Persistent || texture.renderTexture ||
                                texture.renderTextureGeneration;
                     }) &&
        std::none_of(source.buffers.begin(), source.buffers.end(),
                     [](const auto &buffer) {
                         return buffer.computeBuffer || buffer.input || buffer.bufferAuthor || buffer.inputAuthor;
                     }) &&
        std::all_of(source.passes.begin(), source.passes.end(), [](const auto &pass) {
            return std::all_of(pass.commands.begin(), pass.commands.end(), [](const auto &command) {
                return !command.rendererSelection && !command.selectionAuthor && command.overrideMaterial.empty() &&
                       !command.overrideMaterialValue;
            });
        });
    if (immutable) {
        if (auto cached = source.captureCache.lock(); cached && cached->sourceRevision == source.sourceRevision) {
            if (cached->passes.size() != source.passes.size())
                throw std::invalid_argument("A RenderGraph revision cannot change its pass layout: " + source.name);
            bool programsCurrent = true;
            for (size_t pass = 0; pass < source.passes.size(); ++pass) {
                if (cached->passes[pass].commands.size() != source.passes[pass].commands.size())
                    throw std::invalid_argument("A RenderGraph revision cannot change its command layout: " +
                                                source.name);
                for (size_t command = 0; command < source.passes[pass].commands.size(); ++command) {
                    const auto &value = source.passes[pass].commands[command];
                    if (value.type == GraphCommandType::FullscreenQuad)
                        programsCurrent &= cached->passes[pass].commands[command].fullscreenProgram ==
                                           acquireProgram(value.shaderName);
                }
            }
            if (programsCurrent)
                return cached;
        }
    }
    auto captured = std::make_shared<RenderGraphDescription>(
        CaptureRenderGraphDescription(source, prepareMaterial, acquireProgram));
    captured->captureCache.reset();
    if (immutable)
        source.captureCache = captured;
    return captured;
}

GraphBufferInput::GraphBufferInput(const std::shared_ptr<rhi::ComputeBuffer> &value, uint64_t offset, uint64_t byteSize)
    : m_offset(offset), m_byteSize(byteSize)
{
    Publish(value);
}

void GraphBufferInput::Publish(const std::shared_ptr<rhi::ComputeBuffer> &value)
{
    if (!value || !value->GetHost().IsAlive() || !value->GetLastWriteSubmission().IsValid() ||
        !value->GetAllocation()->HasInitializedRange(m_offset, m_byteSize))
        throw std::invalid_argument("Render buffer input requires an accepted value for its full range");

    if (m_value) {
        const auto &layout = value->GetDesc();
        const auto &current = m_value->GetDesc();
        if (!value->GetHost().SharesServicesWith(m_value->GetHost()) || layout.elementCount != current.elementCount ||
            layout.scalarType != current.scalarType || layout.lanes != current.lanes)
            throw std::invalid_argument("Render buffer input publication must preserve its device and scalar layout");
        if (value->GetAllocation() == m_value->GetAllocation())
            return;
    }
    m_value = value->Freeze();
}

void GraphBufferInput::SetRequirement(bool pending, bool holdsStep,
                                      std::vector<std::shared_ptr<GraphBufferCapture>> sources)
{
    m_waiting = pending;
    m_holdsStep = holdsStep;
    if (!pending)
        m_cycle = false;
    m_sources = std::move(sources);
}

void GraphBufferInput::MarkResultCycle() noexcept
{
    m_cycle = true;
    m_sources.clear();
}

bool GraphCommandDesc::BindsResource(const std::string &name) const
{
    return std::any_of(inputBindings.begin(), inputBindings.end(),
                       [&](const auto &binding) { return binding.second == name; });
}

std::string_view GraphCommandDesc::TransparentLayer() const noexcept
{
    if (type != GraphCommandType::DrawRenderers || !IsTransparentPartition(shaderTarget))
        return {};
    const auto binding = std::find_if(inputBindings.begin(), inputBindings.end(),
                                      [](const auto &value) { return value.first == kTransparentLayerInput; });
    return binding == inputBindings.end() ? std::string_view{} : std::string_view(binding->second);
}

const GraphCommandDesc *GraphPassDesc::PrimaryCommand() const noexcept
{
    return commands.empty() ? nullptr : &commands.front();
}

bool GraphPassDesc::WritesTexture(const std::string &name) const
{
    return writeDepth == name || resolveColor == name ||
           std::any_of(writeColors.begin(), writeColors.end(),
                       [&](const auto &output) { return output.second == name; }) ||
           std::any_of(commands.begin(), commands.end(), [&](const auto &command) {
               return command.CopiesToTexture() && command.destinationResource == name;
           });
}

bool GraphPassDesc::ReadsTexture(const std::string &name) const
{
    return std::find(readTextures.begin(), readTextures.end(), name) != readTextures.end() ||
           std::any_of(commands.begin(), commands.end(), [&](const auto &command) {
               const bool textureSource = command.type == GraphCommandType::CopyTexture ||
                                          command.type == GraphCommandType::CopyTextureToBuffer ||
                                          command.type == GraphCommandType::Present;
               return (textureSource && command.sourceResource == name) || command.BindsResource(name);
           });
}

bool GraphTextureDesc::operator==(const GraphTextureDesc &other) const
{
    return name == other.name && format == other.format && width == other.width && height == other.height &&
           sizeDivisor == other.sizeDivisor && samples == other.samples && role == other.role &&
           temporalKey == other.temporalKey && renderTexture == other.renderTexture && attachment == other.attachment &&
           requireCurrentWrite == other.requireCurrentWrite && assetGuid == other.assetGuid && depth == other.depth &&
           isVolume == other.isVolume && dependsOnRayWorld == other.dependsOnRayWorld &&
           respondsToLighting == other.respondsToLighting && renderTextureLayout == other.renderTextureLayout;
}

rhi::PixelFormat GraphTextureDesc::EffectiveFormat(const rhi::RenderViewContext &view) const noexcept
{
    if (!IsViewAttachment())
        return format;
    return IsDepth() ? view.depthFormat : view.colorFormat;
}

uint32_t GraphTextureDesc::EffectiveSamples(uint32_t frameSamples) const noexcept
{
    return samples == 0 ? frameSamples : samples;
}

void GraphTextureDesc::BindRenderTexture(std::shared_ptr<rhi::RenderTexture> owner, GraphTextureAttachment requested,
                                         std::optional<bool> requireWrite)
{
    const auto generation = owner ? owner->Acquire() : nullptr;
    if (!generation || !generation->color->IsValid())
        throw std::invalid_argument("Cannot import a RenderTexture whose device has been destroyed");
    if (requested == GraphTextureAttachment::Resolve && !generation->multisampleColor)
        requested = GraphTextureAttachment::Color;
    const auto *image = ResolveGraphTextureAttachment(*generation, requested);
    if (!image)
        throw std::invalid_argument("RenderTexture has no depth attachment");
    role = GraphTextureRole::Persistent;
    attachment = requested;
    renderTexture = std::move(owner);
    renderTextureLayout = AttachmentLayout(*generation);
    format = image->GetFormat();
    samples = AttachmentSamples(*generation, attachment);
    requireCurrentWrite = requireWrite.value_or(IsDepth() || samples > 1);
}

bool GraphTextureDesc::MatchesRenderTextureLayout(const rhi::RenderTextureGeneration &generation) const noexcept
{
    const auto *image = ResolveGraphTextureAttachment(generation, attachment);
    return image && image->IsValid() && format == image->GetFormat() &&
           samples == AttachmentSamples(generation, attachment) && renderTextureLayout == AttachmentLayout(generation);
}

bool RenderGraphDescription::RenderTextureLayoutsCurrent() const
{
    return std::all_of(textures.begin(), textures.end(), [](const auto &texture) {
        if (texture.role != GraphTextureRole::Persistent)
            return true;
        const auto generation = texture.AcquireRenderTextureGeneration();
        return generation && texture.MatchesRenderTextureLayout(*generation);
    });
}

std::pair<uint32_t, uint32_t> GraphTextureDesc::ResolveExtent(uint32_t viewWidth, uint32_t viewHeight) const noexcept
{
    if (role == GraphTextureRole::Persistent && renderTexture) {
        const auto generation = AcquireRenderTextureGeneration();
        return {generation->width, generation->height};
    }
    if (sizeDivisor > 0)
        return {viewWidth ? std::max(1u, viewWidth / sizeDivisor) : 0,
                viewHeight ? std::max(1u, viewHeight / sizeDivisor) : 0};
    return {width ? width : viewWidth, height ? height : viewHeight};
}

std::pair<uint32_t, uint32_t> ResolveGraphPassExtent(const RenderGraphDescription &graph, const GraphPassDesc &pass,
                                                     uint32_t viewWidth, uint32_t viewHeight)
{
    std::string name = pass.writeDepth;
    if (!pass.writeColors.empty()) {
        const auto primary = std::min_element(pass.writeColors.begin(), pass.writeColors.end(),
                                              [](const auto &a, const auto &b) { return a.first < b.first; });
        name = primary->second;
    }
    const auto found = std::find_if(graph.textures.begin(), graph.textures.end(),
                                    [&](const auto &texture) { return texture.name == name; });
    return found == graph.textures.end() ? std::pair{viewWidth, viewHeight}
                                         : found->ResolveExtent(viewWidth, viewHeight);
}

const GraphTextureDesc *ResolveGraphReadOnlyDepth(const RenderGraphDescription &graph, const GraphPassDesc &pass)
{
    if (!pass.writeDepth.empty())
        return nullptr;
    const auto *command = pass.PrimaryCommand();
    for (const auto &name : pass.readTextures) {
        if (command && command->BindsResource(name))
            continue;
        const auto found = std::find_if(graph.textures.begin(), graph.textures.end(),
                                        [&](const auto &texture) { return texture.name == name; });
        if (found != graph.textures.end() && found->IsDepth())
            return &*found;
    }
    return nullptr;
}

const rhi::TextureResource *ResolveGraphTextureAttachment(const rhi::RenderTextureGeneration &generation,
                                                          GraphTextureAttachment attachment) noexcept
{
    switch (attachment) {
    case GraphTextureAttachment::Color:
        return &generation.ColorAttachment();
    case GraphTextureAttachment::Depth:
        return generation.depth.get();
    case GraphTextureAttachment::Resolve:
        return generation.color.get();
    }
    return {};
}

rhi::TextureHandle ResolveGraphTextureImport(const GraphTextureDesc &texture)
{
    if (texture.role != GraphTextureRole::Persistent || !texture.renderTexture)
        return {};
    const auto generation = texture.AcquireRenderTextureGeneration();
    const auto *attachment = ResolveGraphTextureAttachment(*generation, texture.attachment);
    return attachment ? attachment->GetTexture() : rhi::TextureHandle{};
}

const GraphTextureDesc *FindGraphOutputTextureAlias(const RenderGraphDescription &graph,
                                                    const rhi::RenderTextureGeneration &output)
{
    for (const auto &texture : graph.textures) {
        const auto image = ResolveGraphTextureImport(texture);
        if (image.IsValid() && (image == output.ColorAttachment().GetTexture() ||
                                (output.multisampleColor && image == output.color->GetTexture()) ||
                                (output.depth && image == output.depth->GetTexture())))
            return &texture;
    }
    return nullptr;
}

bool RenderGraphTopologyEquals(const RenderGraphDescription &a, const RenderGraphDescription &b)
{
    if (a.name != b.name || a.outputTexture != b.outputTexture || a.msaaSamples != b.msaaSamples ||
        a.outputDisplayEncoded != b.outputDisplayEncoded || a.temporalJitter != b.temporalJitter ||
        a.linearOutputTexture != b.linearOutputTexture || a.linearOutputPassCount != b.linearOutputPassCount ||
        a.textures != b.textures || a.buffers.size() != b.buffers.size() || a.passes.size() != b.passes.size() ||
        a.workBindings.size() != b.workBindings.size()) {
        return false;
    }

    for (size_t i = 0; i < a.workBindings.size(); ++i) {
        const auto &lhs = a.workBindings[i];
        const auto &rhs = b.workBindings[i];
        if (lhs.name != rhs.name || lhs.recipe != rhs.recipe || lhs.inputBindings != rhs.inputBindings)
            return false;
    }

    for (size_t i = 0; i < a.buffers.size(); ++i) {
        if (!BufferDescEquals(a.buffers[i], b.buffers[i])) {
            return false;
        }
    }

    for (size_t i = 0; i < a.passes.size(); ++i) {
        if (!PassDescEquals(a.passes[i], b.passes[i])) {
            return false;
        }
    }

    return true;
}

bool HasLocalGraphTextureProducer(const RenderGraphDescription &description, const MaterialTextureRead &read)
{
    bool reachedReader = false, hasWriter = false, priorWriter = false;
    for (const auto &pass : description.passes) {
        const bool writes =
            std::any_of(description.textures.begin(), description.textures.end(), [&](const GraphTextureDesc &texture) {
                if (texture.role != GraphTextureRole::Persistent || texture.renderTexture != read.texture ||
                    !pass.WritesTexture(texture.name))
                    return false;
                const auto generation = texture.AcquireRenderTextureGeneration();
                return generation == read.generation &&
                       ResolveGraphTextureAttachment(*generation, texture.attachment) == generation->color.get();
            });
        if (pass.name == read.passName) {
            if (writes)
                throw std::invalid_argument("RenderTexture sampled and written by the same pass: " + read.passName);
            reachedReader = true;
        }
        if (writes) {
            hasWriter = true;
            priorWriter |= !reachedReader;
        }
    }
    if (!reachedReader)
        throw std::invalid_argument("RenderTexture reader pass is not in this graph: " + read.passName);
    if (hasWriter && !priorWriter)
        throw std::invalid_argument("RenderTexture sampled before its local producer completes: " + read.passName);
    return hasWriter;
}

const GraphWorkBindingDesc *FindGraphWorkBinding(const RenderGraphDescription &graph, const std::string &name)
{
    const auto found = std::find_if(graph.workBindings.begin(), graph.workBindings.end(),
                                    [&](const auto &binding) { return binding.name == name; });
    return found == graph.workBindings.end() ? nullptr : &*found;
}

bool RequiresGraphBufferOwners(const RenderGraphDescription &graph)
{
    return !graph.workBindings.empty() ||
           std::any_of(graph.buffers.begin(), graph.buffers.end(),
                       [](const auto &buffer) { return buffer.exportLayout.has_value(); });
}

bool RequiresGraphBufferProof(const RenderGraphDescription &graph)
{
    return RequiresGraphBufferOwners(graph) ||
           std::any_of(graph.passes.begin(), graph.passes.end(), [](const auto &pass) {
               const auto *command = pass.PrimaryCommand();
               return command && (command->type == GraphCommandType::CopyTextureToBuffer ||
                                  command->type == GraphCommandType::CopyBufferToTexture);
           });
}

std::unordered_set<std::string> FindGraphBufferWrites(const RenderGraphDescription &graph)
{
    std::unordered_set<std::string> writes;
    for (const auto &pass : graph.passes) {
        for (const auto &access : pass.bufferAccesses)
            if (access.type == GraphBufferAccessType::StorageWrite ||
                access.type == GraphBufferAccessType::TransferWrite)
                writes.insert(access.resource);
        const auto *command = pass.PrimaryCommand();
        if (!command)
            continue;
        if (command->type == GraphCommandType::CopyBuffer || command->type == GraphCommandType::CopyTextureToBuffer)
            writes.insert(command->destinationResource);
        if (command->type == GraphCommandType::ComputeWork) {
            const auto *work = FindGraphWorkBinding(graph, command->workBinding);
            for (const auto &dispatch : work->recipe->GetDispatches())
                for (const auto &binding : dispatch.bindings)
                    if (static_cast<uint8_t>(binding.access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Write))
                        for (const auto &buffer : graph.buffers)
                            if (buffer.workBinding == work->name && buffer.workSlot == binding.range.slot)
                                writes.insert(buffer.name);
        }
    }
    return writes;
}

const GraphBufferDesc &ResolveGraphBufferStorage(const RenderGraphDescription &graph, const GraphBufferDesc &buffer)
{
    const auto *storage = &buffer;
    while (!storage->workBinding.empty()) {
        const auto &work = *FindGraphWorkBinding(graph, storage->workBinding);
        const auto &ports = work.recipe->GetPorts();
        const auto input = std::find_if(work.inputBindings.begin(), work.inputBindings.end(), [&](const auto &binding) {
            return std::any_of(ports.begin(), ports.end(), [&](const auto &port) {
                return port.input && port.name == binding.first && port.range.slot == storage->workSlot;
            });
        });
        if (input == work.inputBindings.end())
            break;
        storage = &*std::find_if(graph.buffers.begin(), graph.buffers.end(),
                                 [&](const auto &item) { return item.name == input->second; });
    }
    return *storage;
}

const rhi::ComputeBufferDesc *ResolveGraphComputeBufferDesc(const RenderGraphDescription &graph,
                                                            const GraphBufferDesc &buffer)
{
    if (buffer.computeBuffer)
        return &buffer.computeBuffer->GetDesc();
    if (buffer.workBinding.empty() && buffer.exportLayout)
        return &*buffer.exportLayout;
    const auto *binding = FindGraphWorkBinding(graph, buffer.workBinding);
    if (!binding || !binding->recipe || buffer.workSlot >= binding->recipe->GetSlots().size())
        return nullptr;
    return &binding->recipe->GetSlots()[buffer.workSlot].allocation;
}

RenderGraphDescription NormalizeRenderGraphForOutput(RenderGraphDescription desc, bool hasOutputTexture,
                                                     rhi::SampleCount actualSamples)
{
    if (hasOutputTexture) {
        desc.msaaSamples = static_cast<int>(actualSamples);
        if (!desc.linearOutputTexture.empty()) {
            if (desc.linearOutputPassCount > desc.passes.size())
                throw std::invalid_argument("RenderGraph linear output boundary exceeds its pass list");
            desc.passes.resize(desc.linearOutputPassCount);
            desc.outputTexture = desc.linearOutputTexture;
            desc.outputDisplayEncoded = false;
        }
        desc.passes.erase(std::remove_if(desc.passes.begin(), desc.passes.end(),
                                         [](const GraphPassDesc &pass) {
                                             const auto *command = pass.PrimaryCommand();
                                             return command && command->type == GraphCommandType::DrawScreenUI;
                                         }),
                          desc.passes.end());
    }
    return desc;
}

MaterialPassPipelineDescriptor ResolveMaterialPass(const RenderGraphDescription &graph, const GraphPassDesc &pass,
                                                   const rhi::RenderViewContext &view, bool invertCulling)
{
    MaterialPassPipelineDescriptor result;
    const auto *command = pass.PrimaryCommand();
    result.target = command ? command->shaderTarget : ShaderCompileTarget::Forward;
    // Light-space geometry is not reflected with the observing camera.
    result.invertCulling = invertCulling && command && command->type != GraphCommandType::DrawShadowCasters;
    result.samples = view.samples;
    const auto texture = [&](const std::string &name) -> const GraphTextureDesc & {
        const auto found = std::find_if(graph.textures.begin(), graph.textures.end(),
                                        [&](const GraphTextureDesc &item) { return item.name == name; });
        if (found == graph.textures.end())
            throw std::invalid_argument("Material pass references undeclared texture '" + name + "'");
        return *found;
    };
    const auto samples = [&](const GraphTextureDesc &item) {
        return ToRhiSampleCount(static_cast<int>(item.EffectiveSamples(static_cast<uint32_t>(view.samples))));
    };
    auto colors = pass.writeColors;
    std::sort(colors.begin(), colors.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    for (const auto &[slot, name] : colors) {
        (void)slot;
        const auto &item = texture(name);
        result.colorFormats.push_back(item.EffectiveFormat(view));
        result.samples = samples(item);
    }
    // The topology builder's implicit color target applies only to a pass
    // without an explicit depth output; depth-only really means depth-only.
    if (colors.empty() && pass.writeDepth.empty() && result.target != ShaderCompileTarget::Depth &&
        result.target != ShaderCompileTarget::Shadow)
        result.colorFormats = {view.colorFormat};
    const auto setDepth = [&](const GraphTextureDesc &item) {
        result.depthFormat = item.EffectiveFormat(view);
        result.samples = samples(item);
    };
    if (!pass.writeDepth.empty()) {
        setDepth(texture(pass.writeDepth));
    } else if (const auto *depth = ResolveGraphReadOnlyDepth(graph, pass)) {
        setDepth(*depth);
        result.depthReadOnly = true;
    }
    return result;
}

} // namespace infernux
