#include "MsaaPolicy.h"
#include "RenderGraphDescription.h"
#include "rhi/RenderViewContext.h"
#include "rhi/RhiCommand.h"
#include "rhi/RhiComputeBuffer.h"
#include "rhi/RhiComputeKernel.h"
#include "rhi/RhiRenderTexture.h"
#include <core/error/InxError.h>
#include <core/math/Intervals.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace infernux
{
namespace
{
bool TextureExtentsMatch(const GraphTextureDesc &a, const GraphTextureDesc &b, uint32_t viewWidth, uint32_t viewHeight)
{
    const auto left = a.ResolveExtent(viewWidth, viewHeight);
    const auto right = b.ResolveExtent(viewWidth, viewHeight);
    if (left.first && left.second && right.first && right.second)
        return left == right;
    // A metadata-only caller can prove equal formulas, not equality between
    // two unresolved zero extents. The actual view can prove more at Build.
    return a.width == b.width && a.height == b.height && a.sizeDivisor == b.sizeDivisor;
}

using Intervals = intervals::Set;

Intervals RectangleIntervals(uint32_t stride, uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0)
        return {};
    if (width == stride)
        return {{0, uint64_t(width) * height}};
    Intervals result;
    result.reserve(height);
    for (uint32_t row = 0; row < height; ++row)
        result.emplace_back(uint64_t(row) * stride, uint64_t(row) * stride + width);
    return result;
}

// This is a prefix proof over the declared graph, not an execution graph or a
// content-version ledger. Only concrete full writers initialize private work.
void ValidateWorkDataflow(const RenderGraphDescription &graph,
                          const std::unordered_map<std::string, const GraphBufferDesc *> &buffers,
                          const rhi::RenderViewContext &view, GraphBufferContents *contents)
{
    if (!RequiresGraphBufferProof(graph))
        return;
    const auto require = [](bool condition, const std::string &message) {
        if (!condition)
            throw std::invalid_argument(message);
    };
    struct Identity
    {
        uint32_t allocation = 0;
        bool readOnly = false;
        bool work = false;
    };
    struct TextureContents
    {
        Identity identity;
        uint32_t width = 0;
        uint32_t height = 0;
        rhi::PixelFormat format;
    };
    std::unordered_map<std::string, TextureContents> textureContents;
    std::unordered_map<rhi::TextureHandle, uint32_t, rhi::HandleHash<rhi::TextureTag>> textureImports;
    std::unordered_map<std::string, Identity> identities;
    std::unordered_map<std::string, std::vector<Identity>> slots;
    std::unordered_map<rhi::BufferHandle, uint32_t, rhi::HandleHash<rhi::BufferTag>> imports;
    std::unordered_map<uint32_t, Intervals> initialized;
    uint32_t nextIdentity = 0;
    const auto add = [&](Identity identity, uint64_t offset, uint64_t size) {
        intervals::Add(initialized[identity.allocation], offset, offset + size);
    };
    const auto contains = [&](Identity identity, uint64_t offset, uint64_t size) {
        const auto found = initialized.find(identity.allocation);
        return found != initialized.end() && intervals::Contains(found->second, offset, offset + size);
    };
    const auto importedIdentity = [&](const std::shared_ptr<rhi::ComputeBuffer> &buffer) {
        auto [found, inserted] = imports.emplace(buffer->GetBuffer(), nextIdentity);
        if (inserted)
            ++nextIdentity;
        return Identity{found->second, true, false};
    };
    for (const auto &texture : graph.textures) {
        const auto [width, height] = texture.ResolveExtent(view.width, view.height);
        require(width != 0 && height != 0,
                "Graph work validation requires resolved texture extents from the actual view");
        auto identity = nextIdentity++;
        if (texture.role == GraphTextureRole::Persistent) {
            const auto image = ResolveGraphTextureImport(texture);
            const auto [found, inserted] = textureImports.emplace(image, identity);
            identity = found->second;
        }
        textureContents.emplace(
            texture.name, TextureContents{{identity, false, false}, width, height, texture.EffectiveFormat(view)});
    }
    for (const auto &buffer : graph.buffers) {
        if (!buffer.workBinding.empty())
            continue;
        auto identity = buffer.computeBuffer ? importedIdentity(buffer.computeBuffer)
                                             : Identity{nextIdentity++, buffer.viewLightList, false};
        identity.work = buffer.exportLayout.has_value();
        identities.emplace(buffer.name, identity);
        if (buffer.computeBuffer)
            add(identity, buffer.byteOffset, buffer.byteSize);
    }
    for (const auto &binding : graph.workBindings) {
        require(!binding.name.empty() && binding.recipe && !slots.count(binding.name),
                "Work bindings require unique names and a prepared recipe");
        const auto &recipe = *binding.recipe;
        std::vector<Identity> physical;
        physical.reserve(recipe.GetSlots().size());
        for (const auto &slot : recipe.GetSlots()) {
            auto identity = slot.external ? importedIdentity(slot.external) : Identity{nextIdentity++, false, true};
            identity.work = true;
            physical.push_back(identity);
        }
        std::unordered_set<std::string> assignedPorts;
        std::unordered_map<uint32_t, uint32_t> assignedSlots;
        for (const auto &[portName, sourceName] : binding.inputBindings) {
            const auto port = std::find_if(
                recipe.GetPorts().begin(), recipe.GetPorts().end(),
                [&, &portName = portName](const auto &item) { return item.input && item.name == portName; });
            const auto source = buffers.find(sourceName);
            const auto identity = identities.find(sourceName);
            require(port != recipe.GetPorts().end() && assignedPorts.insert(portName).second &&
                        source != buffers.end() && identity != identities.end(),
                    "Work input must reference an earlier work or imported buffer and a unique input port");
            const auto *type = ResolveGraphComputeBufferDesc(graph, *source->second);
            const auto &expected = recipe.GetSlots()[port->range.slot].allocation;
            require(type && type->elementCount == expected.elementCount && type->scalarType == expected.scalarType &&
                        type->lanes == expected.lanes && source->second->byteOffset == port->range.offset &&
                        source->second->byteSize == port->range.byteSize,
                    "Work input override does not match its prepared allocation and logical range");
            const auto [previous, inserted] = assignedSlots.emplace(port->range.slot, identity->second.allocation);
            require(inserted || previous->second == identity->second.allocation,
                    "Input ports of one work allocation must reuse the same physical owner");
            physical[port->range.slot] = identity->second;
            physical[port->range.slot].work = true;
        }
        slots.emplace(binding.name, physical);
        std::vector<bool> represented(physical.size(), false);
        for (const auto &buffer : graph.buffers) {
            if (buffer.workBinding != binding.name)
                continue;
            require(buffer.workSlot < physical.size(), "Graph work buffer references an unknown slot");
            identities.emplace(buffer.name, physical[buffer.workSlot]);
            if (buffer.byteOffset == 0 &&
                buffer.byteSize == recipe.GetSlots()[buffer.workSlot].allocation.GetByteSize())
                represented[buffer.workSlot] = true;
        }
        require(std::all_of(represented.begin(), represented.end(), [](bool value) { return value; }),
                "Every work allocation requires one complete physical graph resource declaration");
        for (const auto &required : recipe.GetRequiredInputs()) {
            if (recipe.GetSlots()[required.slot].external)
                add(physical[required.slot], required.offset, required.byteSize);
        }
    }
    const auto read = [&](const GraphBufferDesc &buffer, uint64_t size) {
        const auto identity = identities.at(buffer.name);
        require(!identity.work || contains(identity, buffer.byteOffset, size),
                "Work buffer is read before a complete current-graph writer: " + buffer.name);
    };
    const auto writable = [&](const GraphBufferDesc &buffer) {
        require(!identities.at(buffer.name).readOnly,
                "Graph buffer aliases read-only external storage: " + buffer.name);
    };
    std::unordered_set<uint32_t> localTextureWriters;
    for (const auto &texture : graph.textures) {
        if (std::any_of(graph.passes.begin(), graph.passes.end(),
                        [&](const auto &pass) { return pass.WritesTexture(texture.name); }))
            localTextureWriters.insert(textureContents.at(texture.name).identity.allocation);
    }
    for (const auto &texture : graph.textures) {
        const auto &contents = textureContents.at(texture.name);
        if (texture.role == GraphTextureRole::TemporalRead ||
            (texture.role == GraphTextureRole::Persistent && !localTextureWriters.count(contents.identity.allocation)))
            add(contents.identity, 0, uint64_t(contents.width) * contents.height);
    }
    const auto copyTexture = [&](const TextureContents &source, const TextureContents &destination, uint32_t width,
                                 uint32_t height) {
        // Snapshot first: names may be aliases of the same physical image.
        const auto known = intervals::Intersect(initialized[source.identity.allocation],
                                                RectangleIntervals(source.width, width, height));
        intervals::Remove(initialized[destination.identity.allocation],
                          RectangleIntervals(destination.width, width, height));
        Intervals mapped;
        for (const auto &range : known) {
            auto cursor = range.first;
            while (cursor < range.second) {
                const auto row = cursor / source.width;
                const auto end = std::min(range.second, (row + 1) * source.width);
                const auto begin = row * destination.width + cursor % source.width;
                mapped.emplace_back(begin, begin + end - cursor);
                cursor = end;
            }
        }
        intervals::Add(initialized[destination.identity.allocation], mapped);
    };
    for (const auto &pass : graph.passes) {
        for (const auto &access : pass.bufferAccesses) {
            const auto &buffer = *buffers.at(access.resource);
            if (access.type == GraphBufferAccessType::StorageWrite ||
                access.type == GraphBufferAccessType::TransferWrite)
                writable(buffer);
            else
                read(buffer, buffer.byteSize);
        }
        const auto *command = pass.PrimaryCommand();
        if (command) {
            for (const auto &[name, resource] : command->inputBindings) {
                const auto buffer = buffers.find(resource);
                if (buffer != buffers.end())
                    read(*buffer->second, buffer->second->byteSize);
            }
        }
        if (command && command->type == GraphCommandType::ComputeWork) {
            const auto *binding = FindGraphWorkBinding(graph, command->workBinding);
            require(binding && binding->recipe, "Compute pass references an unknown work binding");
            const auto &physical = slots.at(binding->name);
            for (const auto &required : binding->recipe->GetRequiredInputs())
                require(contains(physical[required.slot], required.offset, required.byteSize),
                        "Compute work requires a complete input writer: " + binding->name);
            for (const auto &dispatch : binding->recipe->GetDispatches()) {
                for (const auto &item : dispatch.bindings) {
                    const auto identity = physical[item.range.slot];
                    if ((static_cast<uint8_t>(item.access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Read)) != 0)
                        require(contains(identity, item.range.offset, item.range.byteSize),
                                "Compute work binding is read before a complete writer: " + binding->name);
                    if ((static_cast<uint8_t>(item.access) & static_cast<uint8_t>(rhi::ComputeBufferAccess::Write)) !=
                        0)
                        require(!identity.readOnly, "Compute work input aliases read-only external storage");
                }
                for (const auto &write : dispatch.completeWrites) {
                    const auto slot = dispatch.bindings[write.binding].range.slot;
                    add(physical[slot], write.offset, write.byteSize);
                }
            }
        } else if (command && command->type == GraphCommandType::CopyBuffer) {
            const auto &source = *buffers.at(command->sourceResource);
            const auto &destination = *buffers.at(command->destinationResource);
            const auto count =
                command->copyBytes ? command->copyBytes : std::min(source.byteSize, destination.byteSize);
            writable(destination);
            require(identities.at(source.name).allocation != identities.at(destination.name).allocation ||
                        source.byteOffset + count <= destination.byteOffset ||
                        destination.byteOffset + count <= source.byteOffset,
                    "CopyBuffer source and destination ranges overlap in one allocation");
            const auto known = intervals::Intersect(initialized[identities.at(source.name).allocation],
                                                    {{source.byteOffset, source.byteOffset + count}});
            auto &target = initialized[identities.at(destination.name).allocation];
            intervals::Remove(target, {{destination.byteOffset, destination.byteOffset + count}});
            Intervals mapped;
            for (const auto &range : known)
                mapped.emplace_back(destination.byteOffset + range.first - source.byteOffset,
                                    destination.byteOffset + range.second - source.byteOffset);
            intervals::Add(target, mapped);
        } else if (command && command->type == GraphCommandType::CopyTextureToBuffer) {
            const auto &destination = *buffers.at(command->destinationResource);
            writable(destination);
            const auto &source = textureContents.at(command->sourceResource);
            require(command->copyWidth <= source.width && command->copyHeight <= source.height,
                    "Texture-to-buffer rectangle exceeds the actual source extent");
            const auto region = RectangleIntervals(source.width, command->copyWidth, command->copyHeight);
            require(intervals::Intersect(initialized[source.identity.allocation], region) == region,
                    "Texture-to-buffer source has no complete earlier producer");
            add(identities.at(destination.name), destination.byteOffset + command->copyBufferOffset,
                rhi::TextureBufferCopyRegion::ByteSize(source.format, command->copyWidth, command->copyHeight));
        } else if (command && command->type == GraphCommandType::CopyBufferToTexture) {
            const auto &source = *buffers.at(command->sourceResource);
            const auto &destination = textureContents.at(command->destinationResource);
            const auto pixels = uint64_t{destination.width} * destination.height;
            require(contains(identities.at(source.name), source.byteOffset + command->copyBufferOffset,
                             rhi::TextureBufferCopyRegion::ByteSize(destination.format, destination.width,
                                                                    destination.height)),
                    "Buffer-to-texture source has no complete earlier producer: " + source.name);
            add(destination.identity, 0, pixels);
        } else if (command && command->type == GraphCommandType::CopyTexture) {
            const auto &source = textureContents.at(command->sourceResource);
            const auto &destination = textureContents.at(command->destinationResource);
            require(source.identity.allocation != destination.identity.allocation,
                    "Texture copy source and destination alias the same image");
            copyTexture(source, destination, std::min(source.width, destination.width),
                        std::min(source.height, destination.height));
        }
        const auto extent = ResolveGraphPassExtent(graph, pass, view.width, view.height);
        const auto coverAttachment = [&](const std::string &name, bool writesEveryPixel) {
            const auto &target = textureContents.at(name);
            require(extent.first <= target.width && extent.second <= target.height,
                    "Render area exceeds an attachment's actual extent");
            if (writesEveryPixel)
                intervals::Add(initialized[target.identity.allocation],
                               RectangleIntervals(target.width, extent.first, extent.second));
        };
        const bool coversColor = pass.clearColor || (command && command->type == GraphCommandType::FullscreenQuad &&
                                                     !command->alphaBlend && !command->depthTest);
        for (const auto &[slot, name] : pass.writeColors)
            coverAttachment(name, coversColor);
        if (!pass.writeDepth.empty()) {
            const bool clearsDepth =
                pass.clearDepth && (!command || command->type != GraphCommandType::DrawShadowCasters);
            coverAttachment(pass.writeDepth, clearsDepth);
        }
        if (!pass.resolveColor.empty()) {
            const auto primary = std::min_element(pass.writeColors.begin(), pass.writeColors.end(),
                                                  [](const auto &a, const auto &b) { return a.first < b.first; });
            const auto &source = textureContents.at(primary->second);
            const auto &destination = textureContents.at(pass.resolveColor);
            copyTexture(source, destination, std::min({extent.first, source.width, destination.width}),
                        std::min({extent.second, source.height, destination.height}));
        }
    }
    for (const auto &buffer : graph.buffers) {
        if (buffer.exportLayout)
            require(contains(identities.at(buffer.name), buffer.byteOffset, buffer.byteSize),
                    "Exported buffer lacks a complete current-graph producer: " + buffer.name);
        if (contents)
            contents->emplace(buffer.name, initialized[identities.at(buffer.name).allocation]);
    }
}
} // namespace

bool ValidateRenderGraphDescription(const RenderGraphDescription &desc, const rhi::RenderViewContext &view,
                                    GraphBufferContents *contents)
{
    const uint32_t frameSamples =
        desc.msaaSamples > 0 ? static_cast<uint32_t>(desc.msaaSamples) : static_cast<uint32_t>(view.samples);
    std::unordered_map<std::string, const GraphTextureDesc *> textures;
    textures.reserve(desc.textures.size());
    struct TemporalPair
    {
        const GraphTextureDesc *read = nullptr;
        const GraphTextureDesc *write = nullptr;
    };
    std::unordered_map<std::string, TemporalPair> temporalPairs;
    std::unordered_set<rhi::TextureHandle, rhi::HandleHash<rhi::TextureTag>> checkedPersistentImports;
    bool hasViewColor = false;
    bool hasViewDepth = false;

    for (const auto &tex : desc.textures) {
        if (tex.name.empty()) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: texture name cannot be empty");
            return false;
        }
        if (!textures.emplace(tex.name, &tex).second) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: duplicate texture '", tex.name, "'");
            return false;
        }
        if (tex.sizeDivisor == 1) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: texture '", tex.name,
                         "' uses sizeDivisor=1; use 0 or >1");
            return false;
        }
        if (tex.width > 0 && tex.height > 0 && tex.sizeDivisor > 0) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: texture '", tex.name,
                         "' cannot use both explicit size and sizeDivisor");
            return false;
        }
        if ((tex.width == 0) != (tex.height == 0)) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: texture '", tex.name,
                         "' must specify both width and height together");
            return false;
        }
        if (tex.samples != 0 && !IsValidMsaaSampleCount(static_cast<int>(tex.samples))) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: texture '", tex.name, "' has invalid sample count ",
                         tex.samples);
            return false;
        }
        if (tex.IsViewAttachment()) {
            if (tex.samples != 0 || tex.width || tex.height || tex.sizeDivisor || tex.isVolume || tex.renderTexture ||
                !tex.assetGuid.empty() || !tex.temporalKey.empty()) {
                INXLOG_ERROR("Camera attachment '", tex.name, "' must use its view layout and storage");
                return false;
            }
            auto &declared = tex.IsDepth() ? hasViewDepth : hasViewColor;
            if (declared) {
                INXLOG_ERROR("Camera attachments allow one graph name per color or depth image: ", tex.name);
                return false;
            }
            declared = true;
        }
        if (tex.role != GraphTextureRole::Asset && !rhi::IsValidPixelFormat(tex.EffectiveFormat(view))) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: texture '", tex.name, "' has an undefined pixel format");
            return false;
        }
        if (tex.IsViewAttachment())
            continue;
        if (tex.role == GraphTextureRole::Persistent) {
            if (!tex.renderTexture || tex.sizeDivisor != 0 || !tex.temporalKey.empty()) {
                INXLOG_ERROR("Persistent graph textures require an explicit RenderTexture attachment");
                return false;
            }
            const auto generation = tex.renderTexture->Acquire();
            if (!tex.MatchesRenderTextureLayout(*generation)) {
                INXLOG_ERROR("Persistent graph texture description does not match its live owner");
                return false;
            }
            // Writer order belongs to the physical import, including native
            // descriptions that give one attachment several logical names.
            const auto imported = ResolveGraphTextureImport(tex);
            if (checkedPersistentImports.insert(imported).second) {
                std::vector<std::string> aliases;
                for (const auto &candidate : desc.textures)
                    if (ResolveGraphTextureImport(candidate) == imported)
                        aliases.push_back(candidate.name);
                const auto writer = std::find_if(desc.passes.begin(), desc.passes.end(), [&](const auto &pass) {
                    return std::any_of(aliases.begin(), aliases.end(),
                                       [&](const auto &name) { return pass.WritesTexture(name); });
                });
                if (writer != desc.passes.end() &&
                    (!writer->PrimaryCommand() || !writer->PrimaryCommand()->CopiesToTexture()) &&
                    (tex.IsDepth()
                         ? !writer->clearDepth
                         : (!writer->clearColor &&
                            std::find(aliases.begin(), aliases.end(), writer->resolveColor) == aliases.end() &&
                            (!writer->PrimaryCommand() ||
                             writer->PrimaryCommand()->type != GraphCommandType::FullscreenQuad)))) {
                    INXLOG_ERROR("Persistent graph target requires a clear on its first writer: ", tex.name);
                    return false;
                }
                for (auto pass = desc.passes.begin(); writer != desc.passes.end() && pass != std::next(writer);
                     ++pass) {
                    if (std::any_of(aliases.begin(), aliases.end(),
                                    [&](const auto &name) { return pass->ReadsTexture(name); })) {
                        INXLOG_ERROR("Persistent graph target cannot be read before its first writer completes: ",
                                     tex.name);
                        return false;
                    }
                }
            }
            for (const auto &pass : desc.passes) {
                const bool sampled =
                    std::any_of(pass.commands.begin(), pass.commands.end(), [&](const GraphCommandDesc &command) {
                        return command.BindsResource(tex.name) ||
                               (command.type == GraphCommandType::FullscreenQuad &&
                                std::find(pass.readTextures.begin(), pass.readTextures.end(), tex.name) !=
                                    pass.readTextures.end());
                    });
                const bool sampledByMaterial =
                    std::any_of(pass.commands.begin(), pass.commands.end(), [&](const GraphCommandDesc &command) {
                        return command.type != GraphCommandType::FullscreenQuad && command.BindsResource(tex.name);
                    });
                if (sampled && ((tex.samples != 1 && sampledByMaterial) ||
                                (tex.IsDepth() && !generation->description.sampledDepth))) {
                    INXLOG_ERROR("Material sampling requires resolved color; depth sampling requires sampledDepth: ",
                                 tex.name);
                    return false;
                }
            }
            continue;
        }
        if (tex.role == GraphTextureRole::Asset) {
            if (tex.assetGuid.empty() || tex.renderTexture || tex.IsDepth() || tex.sizeDivisor != 0 ||
                !tex.temporalKey.empty() || tex.samples != 1 || tex.width == 0 || tex.height == 0 || tex.depth == 0 ||
                (!tex.isVolume && tex.depth != 1)) {
                INXLOG_ERROR("Sampled graph texture assets require a GUID, explicit dimensions, and one sample: ",
                             tex.name);
                return false;
            }
            const bool written = std::any_of(desc.passes.begin(), desc.passes.end(),
                                             [&](const GraphPassDesc &pass) { return pass.WritesTexture(tex.name); });
            if (written) {
                INXLOG_ERROR("Sampled graph texture assets are read-only: ", tex.name);
                return false;
            }
            continue;
        }
        if (tex.renderTexture) {
            INXLOG_ERROR("Only persistent graph textures may retain a RenderTexture owner");
            return false;
        }
        if (tex.role == GraphTextureRole::Transient) {
            if (!tex.temporalKey.empty()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: transient texture '", tex.name,
                             "' cannot declare a temporal key");
                return false;
            }
            continue;
        }
        if (tex.temporalKey.empty() || tex.IsDepth() || tex.samples != 1) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: temporal texture '", tex.name,
                         "' must be a single-sample color texture with a temporal key");
            return false;
        }
        auto &pair = temporalPairs[tex.temporalKey];
        const GraphTextureDesc **slot = tex.role == GraphTextureRole::TemporalRead ? &pair.read : &pair.write;
        if (*slot != nullptr) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: duplicate temporal role for '", tex.temporalKey, "'");
            return false;
        }
        *slot = &tex;
    }
    for (const auto &temporalEntry : temporalPairs) {
        const auto &key = temporalEntry.first;
        const auto &pair = temporalEntry.second;
        if (!pair.read || !pair.write || pair.read->format != pair.write->format ||
            pair.read->width != pair.write->width || pair.read->height != pair.write->height ||
            pair.read->sizeDivisor != pair.write->sizeDivisor) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: temporal history '", key,
                         "' requires one matching read/write pair");
            return false;
        }
        bool written = false;
        for (const auto &pass : desc.passes) {
            if (pass.WritesTexture(pair.read->name)) {
                INXLOG_ERROR("Temporal history input is read-only: ", pair.read->name);
                return false;
            }
            const bool readsOutput = pass.ReadsTexture(pair.write->name);
            if (readsOutput && (!written || pass.WritesTexture(pair.write->name))) {
                INXLOG_ERROR("Temporal output must be produced before a separate pass reads it: ", pair.write->name);
                return false;
            }
            written = written || pass.WritesTexture(pair.write->name);
        }
        if (!written) {
            INXLOG_ERROR("Temporal history requires an output write on every execution: ", key);
            return false;
        }
    }

    constexpr uint32_t kKnownBufferUsages =
        static_cast<uint32_t>(GraphBufferUsage::Storage) | static_cast<uint32_t>(GraphBufferUsage::Indirect) |
        static_cast<uint32_t>(GraphBufferUsage::TransferSource) |
        static_cast<uint32_t>(GraphBufferUsage::TransferDestination) | static_cast<uint32_t>(GraphBufferUsage::Uniform);
    std::unordered_set<std::string> workNames;
    for (const auto &work : desc.workBindings) {
        if (work.name.empty() || !work.recipe || !workNames.insert(work.name).second) {
            INXLOG_ERROR("Graph work bindings require unique names and a prepared recipe");
            return false;
        }
    }
    std::unordered_map<std::string, const GraphBufferDesc *> buffers;
    buffers.reserve(desc.buffers.size());
    size_t viewLightListCount = 0;
    for (const auto &buffer : desc.buffers) {
        if (buffer.name.empty() || textures.find(buffer.name) != textures.end()) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: invalid or duplicate buffer name '", buffer.name, "'");
            return false;
        }
        if (!buffers.emplace(buffer.name, &buffer).second || buffer.byteSize == 0 || buffer.usage == 0 ||
            (buffer.usage & ~kKnownBufferUsages) != 0) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: invalid buffer description for '", buffer.name, "'");
            return false;
        }
        const bool workBuffer = !buffer.workBinding.empty();
        if (static_cast<unsigned>(buffer.computeBuffer != nullptr) + static_cast<unsigned>(buffer.viewLightList) +
                static_cast<unsigned>(workBuffer) >
            1) {
            INXLOG_ERROR("Graph buffer has multiple storage sources: ", buffer.name);
            return false;
        }
        if (workBuffer) {
            const auto *type = ResolveGraphComputeBufferDesc(desc, buffer);
            if (!type || buffer.byteOffset > type->GetByteSize() ||
                buffer.byteSize > type->GetByteSize() - buffer.byteOffset || buffer.byteOffset % 4 ||
                buffer.byteSize % 4) {
                INXLOG_ERROR("Graph work buffer has an invalid slot or logical range: ", buffer.name);
                return false;
            }
        } else if (buffer.computeBuffer) {
            const auto &owner = *buffer.computeBuffer;
            const auto capacity = owner.GetByteSize();
            const auto &type = owner.GetDesc();
            const auto producer = owner.GetLastWriteSubmission();
            if (!owner.GetBuffer().IsValid() || type.lanes != 1 ||
                (type.scalarType != rhi::ComputeScalarType::Float32 &&
                 type.scalarType != rhi::ComputeScalarType::UInt32) ||
                buffer.byteOffset % 4 != 0 || buffer.byteSize % 4 != 0 || buffer.byteOffset > capacity ||
                buffer.byteSize > capacity - buffer.byteOffset || !producer.IsValid() ||
                producer.device != owner.GetBuffer().Device() ||
                !owner.HasInitializedRange(buffer.byteOffset, buffer.byteSize)) {
                INXLOG_ERROR(
                    "SceneRenderGraph::ApplyPythonGraph: imported buffer '", buffer.name,
                    "' requires a live scalar f32/u32 view with an accepted producer for its entire initialized range");
                return false;
            }
        } else if (buffer.byteOffset != 0) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: only imported buffers can have a byte offset: ",
                         buffer.name);
            return false;
        }
        if (buffer.viewLightList) {
            ++viewLightListCount;
            if (buffer.computeBuffer || buffer.byteSize < sizeof(uint32_t) * 4 ||
                buffer.usage != static_cast<uint32_t>(GraphBufferUsage::Storage)) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: view light list '", buffer.name,
                             "' must be a read-only storage resource with a 16-byte header");
                return false;
            }
        }
    }
    if (viewLightListCount > 1) {
        INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: only one view light list is allowed");
        return false;
    }

    std::unordered_set<std::string> passNames;
    passNames.reserve(desc.passes.size());
    std::unordered_set<std::string> parameterBlockIds;
    std::unordered_set<std::string> producedTextures;
    for (const auto &pass : desc.passes) {
        if (pass.name.empty()) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass name cannot be empty");
            return false;
        }
        if (!passNames.insert(pass.name).second) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: duplicate pass '", pass.name, "'");
            return false;
        }
        for (const auto &event : pass.events) {
            if (event.label.empty()) {
                INXLOG_ERROR("Graph event requires a label: ", pass.name);
                return false;
            }
            for (const auto &[name, buffer] : event.buffers) {
                const auto output = buffers.find(buffer);
                if (name.empty() || output == buffers.end() || !output->second->exportLayout) {
                    INXLOG_ERROR("Graph event requires a name and exported buffer for each source: ", pass.name);
                    return false;
                }
            }
        }
        if (pass.commands.size() > 1) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                         "' records multiple commands; command-list execution is not enabled yet");
            return false;
        }
        const GraphCommandDesc *command = pass.PrimaryCommand();
        if (command) {
            if (command->type == GraphCommandType::FullscreenQuad) {
                if (pass.writeColors.size() != 1 || pass.writeColors.front().first != 0 ||
                    (command->depthTest && pass.writeDepth.empty()) || (command->depthWrite && !command->depthTest) ||
                    static_cast<uint8_t>(command->depthCompare) > static_cast<uint8_t>(rhi::CompareFunction::Always)) {
                    INXLOG_ERROR("Fullscreen pass '", pass.name, "' has an invalid color/depth contract");
                    return false;
                }
                const auto requiresExisting = [&](const std::string &name) {
                    const auto texture = textures.find(name);
                    return texture == textures.end() ||
                           (texture->second->role != GraphTextureRole::Persistent && !producedTextures.count(name));
                };
                if (((command->alphaBlend || command->depthTest) && !pass.clearColor &&
                     requiresExisting(pass.writeColors.front().second)) ||
                    (!pass.writeDepth.empty() && !pass.clearDepth && requiresExisting(pass.writeDepth))) {
                    INXLOG_ERROR("Fullscreen pass '", pass.name, "' must clear or load an earlier attachment output");
                    return false;
                }
                if (std::find(pass.readTextures.begin(), pass.readTextures.end(), pass.writeDepth) !=
                        pass.readTextures.end() ||
                    command->BindsResource(pass.writeDepth)) {
                    INXLOG_ERROR("Fullscreen pass '", pass.name, "' cannot sample its attached depth; copy it first");
                    return false;
                }
            } else if (command->depthTest || command->depthWrite || command->alphaBlend) {
                INXLOG_ERROR("Explicit fullscreen raster state used by a non-fullscreen pass: ", pass.name);
                return false;
            }
            if (command->rendererSelection &&
                (command->type != GraphCommandType::DrawRenderers || !command->overrideMaterial.empty())) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: renderer selection requires DrawRenderers "
                             "without another override material in pass '",
                             pass.name, "'");
                return false;
            }
            std::unordered_set<std::string> parameterNames;
            for (const auto &[name, value] : command->parameters) {
                if (name.empty() || !parameterNames.insert(name).second) {
                    INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                                 "' has an invalid parameter layout");
                    return false;
                }
            }
            if (!command->parameterBlock.empty()) {
                if (command->type != GraphCommandType::FullscreenQuad ||
                    !parameterBlockIds.insert(command->parameterBlock).second) {
                    INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                                 "' has an invalid or duplicate runtime parameter block '", command->parameterBlock,
                                 "'");
                    return false;
                }
            }
        }
        const bool rasterCommand =
            command &&
            (command->type == GraphCommandType::DrawRenderers || command->type == GraphCommandType::DrawSkybox ||
             command->type == GraphCommandType::DrawShadowCasters || command->type == GraphCommandType::DrawWorldUI ||
             command->type == GraphCommandType::DrawScreenUI || command->type == GraphCommandType::FullscreenQuad);
        const bool copyCommand =
            command && (command->CopiesToTexture() || command->type == GraphCommandType::CopyBuffer ||
                        command->type == GraphCommandType::CopyTextureToBuffer);
        if ((pass.type == GraphPassType::Raster && command && !rasterCommand) ||
            (pass.type == GraphPassType::Copy && !copyCommand) ||
            (pass.type == GraphPassType::Compute && (!command || command->type != GraphCommandType::ComputeWork)) ||
            (pass.type == GraphPassType::Present && (!command || command->type != GraphCommandType::Present))) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                         "' command does not match its execution domain");
            return false;
        }
        if (pass.type != GraphPassType::Raster && (!pass.writeColors.empty() || !pass.writeDepth.empty() ||
                                                   !pass.resolveColor.empty() || pass.clearColor || pass.clearDepth)) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: non-raster pass '", pass.name,
                         "' declares raster attachments");
            return false;
        }
        for (const auto &access : pass.bufferAccesses) {
            const auto buffer = buffers.find(access.resource);
            if (buffer == buffers.end()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' references unknown buffer '",
                             access.resource, "'");
                return false;
            }
            uint32_t requiredUsage = 0;
            switch (access.type) {
            case GraphBufferAccessType::UniformRead:
                requiredUsage = static_cast<uint32_t>(GraphBufferUsage::Uniform);
                break;
            case GraphBufferAccessType::StorageRead:
            case GraphBufferAccessType::StorageWrite:
                requiredUsage = static_cast<uint32_t>(GraphBufferUsage::Storage);
                break;
            case GraphBufferAccessType::IndirectRead:
                requiredUsage = static_cast<uint32_t>(GraphBufferUsage::Indirect);
                break;
            case GraphBufferAccessType::TransferRead:
                requiredUsage = static_cast<uint32_t>(GraphBufferUsage::TransferSource);
                break;
            case GraphBufferAccessType::TransferWrite:
                requiredUsage = static_cast<uint32_t>(GraphBufferUsage::TransferDestination);
                break;
            }
            if ((buffer->second->usage & requiredUsage) == 0) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' buffer '", access.resource,
                             "' does not declare the required usage");
                return false;
            }
            if (buffer->second->viewLightList && access.type != GraphBufferAccessType::StorageRead) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: view light list '", access.resource,
                             "' is read-only");
                return false;
            }
            if (buffer->second->computeBuffer && (access.type == GraphBufferAccessType::StorageWrite ||
                                                  access.type == GraphBufferAccessType::TransferWrite)) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: imported buffer '", access.resource,
                             "' is read-only in the render graph");
                return false;
            }
        }
        if (command && command->type == GraphCommandType::CopyTexture) {
            const auto source = textures.find(command->sourceResource);
            const auto destination = textures.find(command->destinationResource);
            if (source == textures.end() || destination == textures.end() || source == destination ||
                source->second->IsViewColor() || destination->second->IsViewColor() ||
                source->second->role == GraphTextureRole::Asset ||
                destination->second->role == GraphTextureRole::Asset ||
                source->second->EffectiveFormat(view) != destination->second->EffectiveFormat(view) ||
                source->second->EffectiveSamples(frameSamples) != destination->second->EffectiveSamples(frameSamples) ||
                (ResolveGraphTextureImport(*source->second).IsValid() &&
                 ResolveGraphTextureImport(*source->second) == ResolveGraphTextureImport(*destination->second))) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: texture copy pass '", pass.name,
                             "' has incompatible resources");
                return false;
            }
        } else if (command && command->type == GraphCommandType::CopyBuffer) {
            const auto source = buffers.find(command->sourceResource);
            const auto destination = buffers.find(command->destinationResource);
            if (source == buffers.end() || destination == buffers.end() ||
                command->sourceResource == command->destinationResource ||
                command->copyBytes > std::min(source->second->byteSize, destination->second->byteSize)) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: buffer copy pass '", pass.name,
                             "' has incompatible resources");
                return false;
            }
            const uint32_t transferSource = static_cast<uint32_t>(GraphBufferUsage::TransferSource);
            const uint32_t transferDestination = static_cast<uint32_t>(GraphBufferUsage::TransferDestination);
            if ((source->second->usage & transferSource) == 0 ||
                (destination->second->usage & transferDestination) == 0) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: buffer copy pass '", pass.name,
                             "' resources do not declare transfer usage");
                return false;
            }
            if (destination->second->computeBuffer) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: imported buffer '", command->destinationResource,
                             "' cannot be a copy destination");
                return false;
            }
            const auto count = command->copyBytes ? command->copyBytes
                                                  : std::min(source->second->byteSize, destination->second->byteSize);
            if (source->second->byteOffset % 4 || destination->second->byteOffset % 4 || count % 4) {
                INXLOG_ERROR("Buffer copy offsets and size must be four-byte aligned");
                return false;
            }
        } else if (command && (command->type == GraphCommandType::CopyTextureToBuffer ||
                               command->type == GraphCommandType::CopyBufferToTexture)) {
            const bool toBuffer = command->type == GraphCommandType::CopyTextureToBuffer;
            const auto image = textures.find(toBuffer ? command->sourceResource : command->destinationResource);
            const auto buffer = buffers.find(toBuffer ? command->destinationResource : command->sourceResource);
            if (image == textures.end() || buffer == buffers.end()) {
                INXLOG_ERROR("Texture-buffer copy references an unknown resource in pass '", pass.name, "'");
                return false;
            }
            const auto &texture = *image->second;
            const auto &range = *buffer->second;
            const auto [width, height] = texture.ResolveExtent(view.width, view.height);
            const auto format = texture.EffectiveFormat(view);
            const rhi::TextureBufferCopyRegion region{rhi::TextureBufferCopyRegion::AspectFor(format),
                                                      0,
                                                      0,
                                                      toBuffer ? command->copyWidth : width,
                                                      toBuffer ? command->copyHeight : height,
                                                      range.byteOffset + command->copyBufferOffset};
            const auto usage = toBuffer ? GraphBufferUsage::TransferDestination : GraphBufferUsage::TransferSource;
            if (texture.IsViewColor() || texture.isVolume || texture.role == GraphTextureRole::Asset ||
                texture.EffectiveSamples(frameSamples) != 1 ||
                (!toBuffer && texture.role != GraphTextureRole::Persistent &&
                 (!texture.width || !texture.height || texture.sizeDivisor)) ||
                (width && region.width > width) || (height && region.height > height) ||
                command->copyBufferOffset > range.byteSize ||
                !region.Fits(format, range.byteSize - command->copyBufferOffset) || range.viewLightList ||
                (toBuffer && range.computeBuffer) || !(range.usage & static_cast<uint32_t>(usage))) {
                INXLOG_ERROR("Texture-buffer copy requires a packed color or depth layout and aligned buffer range: ",
                             pass.name);
                return false;
            }
        } else if (command && command->type == GraphCommandType::ComputeWork &&
                   !FindGraphWorkBinding(desc, command->workBinding)) {
            INXLOG_ERROR("Compute pass references unknown work: ", command->workBinding);
            return false;
        } else if (command && command->type == GraphCommandType::Present &&
                   textures.find(command->sourceResource) == textures.end()) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: present pass '", pass.name,
                         "' references an unknown texture");
            return false;
        } else if (command && command->type == GraphCommandType::Present &&
                   textures.at(command->sourceResource)->IsDepth()) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: present pass '", pass.name,
                         "' cannot export a depth texture");
            return false;
        }
        if (command && command->type == GraphCommandType::DrawShadowCasters) {
            if (command->shaderTarget != ShaderCompileTarget::Shadow) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: shadow pass '", pass.name,
                             "' requires the Shadow shader target");
                return false;
            }
            if (!pass.writeColors.empty()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: shadow pass '", pass.name,
                             "' cannot write color targets");
                return false;
            }
            if (pass.writeDepth.empty()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: shadow pass '", pass.name,
                             "' requires a depth output");
                return false;
            }
        }
        if (command && command->type == GraphCommandType::DrawWorldUI &&
            (pass.writeColors.size() != 1 || pass.writeDepth.empty())) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: world UI pass '", pass.name,
                         "' requires one color output and one depth output");
            return false;
        }
        if (pass.clearDepth && pass.writeDepth.empty()) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                         "' clears depth but has no depth output");
            return false;
        }

        const bool depthOnlyMaterialPass = command && command->type == GraphCommandType::DrawRenderers &&
                                           (command->shaderTarget == ShaderCompileTarget::Depth ||
                                            command->shaderTarget == ShaderCompileTarget::Shadow);
        if (depthOnlyMaterialPass && (!pass.writeColors.empty() || pass.writeDepth.empty())) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: depth-only material pass '", pass.name,
                         "' requires one depth output and no color outputs");
            return false;
        }
        const bool motionMaterialPass = command && command->type == GraphCommandType::DrawRenderers &&
                                        command->shaderTarget == ShaderCompileTarget::Motion;
        const bool rgbaGeometryMaterialPass = command && command->type == GraphCommandType::DrawRenderers &&
                                              (command->shaderTarget == ShaderCompileTarget::Normal ||
                                               command->shaderTarget == ShaderCompileTarget::BaseColor);
        if (motionMaterialPass || rgbaGeometryMaterialPass) {
            const auto format = motionMaterialPass ? rhi::PixelFormat::RG16SFloat : rhi::PixelFormat::RGBA16SFloat;
            const auto color =
                pass.writeColors.size() == 1 ? textures.find(pass.writeColors.front().second) : textures.end();
            const bool hasReadableDepth =
                std::any_of(pass.readTextures.begin(), pass.readTextures.end(), [&](const std::string &textureName) {
                    const auto texture = textures.find(textureName);
                    const bool sampled = command->BindsResource(textureName);
                    return texture != textures.end() && texture->second->IsDepth() && !sampled;
                });
            if (color == textures.end() || pass.writeColors.front().first != 0 ||
                color->second->EffectiveFormat(view) != format || !hasReadableDepth || !pass.writeDepth.empty()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: ", ShaderCompileTargetName(command->shaderTarget),
                             " pass '", pass.name, "' requires one ",
                             motionMaterialPass ? "RG16SFloat" : "RGBA16SFloat",
                             " color[0], one readable depth attachment, and no depth write");
                return false;
            }
        }

        std::unordered_set<int> colorSlots;
        uint32_t attachmentSamples = 0;
        const auto acceptAttachmentSamples = [&](const GraphTextureDesc &texture) {
            const uint32_t samples = texture.EffectiveSamples(frameSamples);
            if (attachmentSamples == 0) {
                attachmentSamples = samples;
                return true;
            }
            return attachmentSamples == samples;
        };

        for (const auto &[slot, textureName] : pass.writeColors) {
            if (slot < 0 || slot >= 8 || !colorSlots.insert(slot).second) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                             "' has an invalid or duplicate color slot ", slot);
                return false;
            }
            auto texIt = textures.find(textureName);
            if (texIt == textures.end()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' writes unknown color target '",
                             textureName, "'");
                return false;
            }
            if (texIt->second->IsDepth()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' writes depth texture '",
                             textureName, "' as color slot ", slot);
                return false;
            }
            if (texIt->second->role == GraphTextureRole::Asset) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                             "' cannot write sampled Texture asset '", textureName, "'");
                return false;
            }
            if (!acceptAttachmentSamples(*texIt->second)) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                             "' uses color and depth attachments with different sample counts");
                return false;
            }
        }
        for (int slot = 0; slot < static_cast<int>(colorSlots.size()); ++slot) {
            if (colorSlots.find(slot) == colorSlots.end()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                             "' must use contiguous color slots starting at zero");
                return false;
            }
        }

        if (!pass.writeDepth.empty()) {
            auto texIt = textures.find(pass.writeDepth);
            if (texIt == textures.end()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' writes unknown depth target '",
                             pass.writeDepth, "'");
                return false;
            }
            if (!texIt->second->IsDepth()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' writes color texture '",
                             pass.writeDepth, "' as depth");
                return false;
            }
            if (texIt->second->role == GraphTextureRole::Asset) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                             "' cannot write sampled Texture asset '", pass.writeDepth, "'");
                return false;
            }
            if (!acceptAttachmentSamples(*texIt->second)) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                             "' uses color and depth attachments with different sample counts");
                return false;
            }
        }

        uint32_t readOnlyDepthAttachmentCount = 0;
        if (command && command->type != GraphCommandType::FullscreenQuad) {
            for (const auto &textureName : pass.readTextures) {
                const bool sampledInput = command->BindsResource(textureName);
                if (sampledInput)
                    continue;
                auto texIt = textures.find(textureName);
                if (texIt == textures.end() || !texIt->second->IsDepth())
                    continue;
                ++readOnlyDepthAttachmentCount;
                if (!acceptAttachmentSamples(*texIt->second)) {
                    INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                                 "' uses color and depth attachments with different sample counts");
                    return false;
                }
            }
        }
        if (readOnlyDepthAttachmentCount > 1 || (readOnlyDepthAttachmentCount != 0 && !pass.writeDepth.empty())) {
            INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                         "' must declare exactly one depth attachment as either read-only or writable");
            return false;
        }

        if (!pass.resolveColor.empty()) {
            const auto resolve = textures.find(pass.resolveColor);
            const auto source = pass.writeColors.size() == 1 && pass.writeColors.front().first == 0
                                    ? textures.find(pass.writeColors.front().second)
                                    : textures.end();
            if (resolve == textures.end() || source == textures.end() || resolve->second->IsViewColor() ||
                resolve->second->role == GraphTextureRole::Asset || resolve->second->IsDepth() ||
                pass.resolveColor == pass.writeColors.front().second ||
                source->second->EffectiveSamples(frameSamples) <= 1 ||
                resolve->second->EffectiveSamples(frameSamples) != 1 ||
                source->second->EffectiveFormat(view) != resolve->second->EffectiveFormat(view) ||
                !TextureExtentsMatch(*source->second, *resolve->second, view.width, view.height)) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name,
                             "' has an invalid MSAA color resolve contract");
                return false;
            }
        }

        for (const auto &textureName : pass.readTextures) {
            if (textures.find(textureName) == textures.end()) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' reads unknown texture '",
                             textureName, "'");
                return false;
            }
        }

        for (const auto &[slot, name] : pass.writeColors)
            producedTextures.insert(name);
        if (!pass.writeDepth.empty())
            producedTextures.insert(pass.writeDepth);
        if (!pass.resolveColor.empty())
            producedTextures.insert(pass.resolveColor);
        if (command && command->CopiesToTexture())
            producedTextures.insert(command->destinationResource);

        if (!command)
            continue;
        for (const auto &[samplerName, textureName] : command->inputBindings) {
            if (textures.find(textureName) == textures.end() &&
                (command->type != GraphCommandType::FullscreenQuad || buffers.find(textureName) == buffers.end())) {
                INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: pass '", pass.name, "' input '", samplerName,
                             "' references unknown resource '", textureName, "'");
                return false;
            }
        }
    }

    if (!desc.outputTexture.empty() && textures.find(desc.outputTexture) == textures.end()) {
        INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: output texture '", desc.outputTexture, "' is not declared");
        return false;
    }
    if (!desc.outputTexture.empty() && textures.at(desc.outputTexture)->role == GraphTextureRole::Asset) {
        INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: a sampled Texture asset cannot be the graph output");
        return false;
    }

    try {
        for (const auto &buffer : desc.buffers) {
            if (!buffer.exportLayout)
                continue;
            const auto &layout = *buffer.exportLayout;
            if (buffer.viewLightList || !layout.elementCount || layout.lanes == 0 || layout.lanes > 4 ||
                layout.scalarType > rhi::ComputeScalarType::Int64 ||
                layout.elementCount > (std::numeric_limits<uint64_t>::max)() / layout.GetElementStride())
                throw std::invalid_argument("Buffer export requires an owned, valid numerical layout");
            const auto *actual = ResolveGraphComputeBufferDesc(desc, buffer);
            if (!actual || actual->elementCount != layout.elementCount || actual->scalarType != layout.scalarType ||
                actual->lanes != layout.lanes || buffer.byteOffset > layout.GetByteSize() ||
                buffer.byteSize > layout.GetByteSize() - buffer.byteOffset)
                throw std::invalid_argument("Buffer export layout does not match the actual storage");
            if (!buffer.computeBuffer && buffer.workBinding.empty() && layout.GetByteSize() != buffer.byteSize)
                throw std::invalid_argument("Transient buffer export layout must match its allocation size");
            uint64_t elements = 1;
            for (const auto dimension : buffer.exportShape) {
                if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension)
                    throw std::invalid_argument("Buffer export requires a nonempty, bounded shape");
                elements *= dimension;
            }
            if (buffer.exportShape.empty() || elements != buffer.byteSize / layout.GetScalarBytes() ||
                buffer.byteSize % layout.GetScalarBytes())
                throw std::invalid_argument("Buffer export shape must cover its complete logical range");
        }
        if (contents)
            contents->clear();
        ValidateWorkDataflow(desc, buffers, view, contents);
    } catch (const std::exception &error) {
        INXLOG_ERROR("SceneRenderGraph::ApplyPythonGraph: ", error.what());
        return false;
    }
    return true;
}

} // namespace infernux
