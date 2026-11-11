#include "RenderGraphDescription.h"
#include "MaterialPassPipeline.h"
#include "MsaaPolicy.h"
#include "rhi/RenderViewContext.h"
#include "rhi/RhiComputeBuffer.h"
#include "rhi/RhiComputeKernel.h"
#include "rhi/RhiRenderTexture.h"

#include <algorithm>
#include <stdexcept>

namespace infernux
{
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
           a.input == b.input && (a.input || a.computeBuffer == b.computeBuffer) &&
           a.viewLightList == b.viewLightList && a.workBinding == b.workBinding && a.workSlot == b.workSlot &&
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
           a.overrideMaterial == b.overrideMaterial && a.rendererSelection == b.rendererSelection &&
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
           isVolume == other.isVolume && renderTextureLayout == other.renderTextureLayout;
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
        const auto generation = texture.renderTexture ? texture.renderTexture->Acquire() : nullptr;
        return generation && texture.MatchesRenderTextureLayout(*generation);
    });
}

std::pair<uint32_t, uint32_t> GraphTextureDesc::ResolveExtent(uint32_t viewWidth, uint32_t viewHeight) const noexcept
{
    if (role == GraphTextureRole::Persistent && renderTexture) {
        const auto generation = renderTexture->Acquire();
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
    const auto generation = texture.renderTexture->Acquire();
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
        a.temporalJitter != b.temporalJitter || a.linearOutputTexture != b.linearOutputTexture ||
        a.linearOutputPassCount != b.linearOutputPassCount || a.textures != b.textures ||
        a.buffers.size() != b.buffers.size() || a.passes.size() != b.passes.size() ||
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
    } else {
        for (const auto &name : pass.readTextures) {
            if (command && command->BindsResource(name))
                continue;
            const auto &item = texture(name);
            if (item.IsDepth()) {
                setDepth(item);
                result.depthReadOnly = true;
                break;
            }
        }
    }
    return result;
}

} // namespace infernux
