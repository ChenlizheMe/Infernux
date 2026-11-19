#include "RenderGraphWork.h"
#include "RenderViewSchedule.h"
#include "raytracing/RayWorldMaterials.h"
#include "rhi/RhiBuffer.h"
#include "rhi/RhiComputeBuffer.h"
#include "rhi/RhiComputeKernel.h"

#include <algorithm>
#include <stdexcept>

namespace infernux
{
namespace
{
std::shared_ptr<rhi::ComputeBuffer> ReuseGraphBuffer(const RenderGraphWorkStorage &storage, rhi::ComputeHost &host,
                                                     const std::string &name, const rhi::ComputeBufferDesc &layout)
{
    const auto found = storage.buffers.find(name);
    if (found == storage.buffers.end())
        return {};
    const auto &owner = found->second;
    const auto &previous = owner->GetDesc();
    if (!owner->GetHost().SharesServicesWith(host) || owner->GetAllocation()->IsReadProtected() ||
        !owner->GetAllocation()->IsGraphWritable() || previous.elementCount != layout.elementCount ||
        previous.scalarType != layout.scalarType || previous.lanes != layout.lanes ||
        !rhi::HasBufferUsage(previous.usage, layout.usage))
        return {};
    return owner;
}
} // namespace

std::optional<RenderGraphWorkStorage> PrepareRenderGraphWork(rhi::ComputeHost &host,
                                                             const RenderGraphDescription &graph,
                                                             const RenderGraphWorkStorage &previous)
{
    for (const auto &binding : graph.workBindings)
        if (!binding.recipe->GetHost().SharesServicesWith(host))
            throw std::invalid_argument("Graph work belongs to an expired or different compute host");

    RenderGraphWorkStorage next;
    for (const auto &buffer : graph.buffers) {
        if (buffer.computeBuffer) {
            if (!buffer.computeBuffer->GetHost().SharesServicesWith(host))
                throw std::invalid_argument("Graph buffer belongs to an expired or different compute host: " +
                                            buffer.name);
            next.buffers.emplace(buffer.name, buffer.computeBuffer);
        } else if (buffer.workBinding.empty() && buffer.exportLayout) {
            auto layout = *buffer.exportLayout;
            layout.usage = static_cast<rhi::BufferUsageFlags>(buffer.usage);
            auto owner = ReuseGraphBuffer(previous, host, buffer.name, layout);
            if (!owner)
                owner = rhi::ComputeBuffer::TryCreateGraphBuffer(host, layout);
            if (!owner)
                return std::nullopt;
            next.buffers.emplace(buffer.name, std::move(owner));
        }
    }
    for (const auto &binding : graph.workBindings) {
        std::vector<rhi::ComputeWorkInput> inputs;
        inputs.reserve(binding.inputBindings.size());
        for (const auto &[port, resource] : binding.inputBindings) {
            const auto owner = next.buffers.find(resource);
            if (owner == next.buffers.end())
                throw std::invalid_argument("Work input has no earlier typed graph allocation owner: " + resource);
            inputs.push_back({port, owner->second});
        }
        const auto old = previous.instances.find(binding.name);
        auto instance = old != previous.instances.end() && &old->second->GetRecipe() == binding.recipe.get()
                            ? old->second->Rebind(std::move(inputs), binding.externalInputs)
                            : binding.recipe->Instantiate(std::move(inputs), binding.externalInputs);
        if (!instance)
            return std::nullopt;
        for (const auto &buffer : graph.buffers)
            if (buffer.workBinding == binding.name)
                next.buffers.emplace(buffer.name, instance->GetBuffers()[buffer.workSlot]);
        next.instances.emplace(binding.name, std::move(instance));
    }
    return next;
}

bool BindRenderGraphWorkReads(const RenderGraphDescription &graph, RenderGraphWorkStorage &storage,
                              const rhi::ComputeGraphReads &publications)
{
    decltype(storage.passInstances) next;
    for (const auto &pass : graph.passes) {
        const auto *command = pass.PrimaryCommand();
        if (!command || command->type != GraphCommandType::ComputeWork)
            continue;
        const auto &base = storage.instances.at(command->workBinding);
        std::shared_ptr<rhi::ComputeGraphReads> reads;
        for (const auto &operation : base->GetRecipe().GetDispatches())
            for (const auto &resource : operation.kernel->GetGraphReadBindings()) {
                if (!reads)
                    reads = std::make_shared<rhi::ComputeGraphReads>();
                auto port = resource.name;
                if (port == "ray_color_sampler")
                    port = "ray_color_pages";
                else if (port == "ray_data_sampler")
                    port = "ray_data_pages";
                const auto binding = std::find_if(command->inputBindings.begin(), command->inputBindings.end(),
                                                  [&](const auto &item) { return item.first == port; });
                if (binding == command->inputBindings.end())
                    throw std::invalid_argument("Graph compute input is not bound: " + pass.name + "." + port);
                if (resource.binding.type == rhi::BindingType::StorageBuffer) {
                    const auto found = publications.buffers.find(binding->second);
                    if (found == publications.buffers.end())
                        return false;
                    reads->buffers.emplace(resource.name, found->second);
                } else {
                    const auto found = publications.textures.find(binding->second);
                    if (found == publications.textures.end())
                        return false;
                    reads->textures.emplace(resource.name, found->second);
                }
            }
        auto instance = reads ? base->WithGraphReads(std::move(reads)) : base;
        (void)instance->GetDispatches();
        next.emplace(pass.name, std::move(instance));
    }
    storage.passInstances = std::move(next);
    return true;
}

rhi::ComputeGraphReads
WorldRayGraphReadPublications(const RenderGraphDescription &graph,
                              const std::array<std::shared_ptr<const rhi::BufferResource>, 2> &geometry,
                              const std::shared_ptr<const raytracing::RayWorldMaterialPublication> &materials)
{
    rhi::ComputeGraphReads result;
    for (const auto &buffer : graph.buffers) {
        std::shared_ptr<const rhi::BufferResource> owner;
        switch (buffer.frameBuffer) {
        case GraphFrameBuffer::WorldRayGeometry:
            owner = geometry[0];
            break;
        case GraphFrameBuffer::WorldRayMeshes:
            owner = geometry[1];
            break;
        case GraphFrameBuffer::WorldRayLitMaterials:
            owner = materials ? materials->materials : nullptr;
            break;
        default:
            continue;
        }
        if (owner)
            result.buffers.emplace(buffer.name,
                                   rhi::ComputeGraphReads::Buffer{owner->GetBuffer(), buffer.byteOffset,
                                                                  owner->GetByteSize() - buffer.byteOffset, owner});
    }
    if (materials)
        for (const auto &texture : graph.textures)
            if (texture.IsWorldRayPage()) {
                const auto &owner = materials->pages[texture.role == GraphTextureRole::WorldRayDataPages ? 1 : 0];
                result.textures.emplace(texture.name,
                                        rhi::ComputeGraphReads::Texture{owner->GetView(), owner->GetSampler(), owner});
            }
    return result;
}

void RenderGraphWorkStorage::Publish() const
{
    const auto publish = [](const auto &owner) {
        if (owner->GetAllocation()->IsGraphWritable())
            owner->PublishStorage();
    };
    for (const auto &[name, owner] : buffers)
        publish(owner);
    for (const auto &[name, instance] : instances)
        for (const auto &owner : instance->GetBuffers())
            publish(owner);
}

std::vector<std::string> GraphWorkSlotNames(const RenderGraphDescription &graph, const GraphWorkBindingDesc &work)
{
    const auto &slots = work.recipe->GetSlots();
    std::vector<std::string> names(slots.size());
    for (const auto &buffer : graph.buffers)
        if (buffer.workBinding == work.name && buffer.byteOffset == 0 &&
            buffer.byteSize == slots[buffer.workSlot].allocation.GetByteSize())
            names[buffer.workSlot] = buffer.name;
    return names;
}

bool AppendPendingGraphBufferResults(RenderViewAccess &access, const std::vector<const GraphBufferDesc *> &inputs)
{
    bool pending = false;
    for (const auto *buffer : inputs) {
        const auto &input = *buffer->input;
        if (!input.IsWaitingForResult())
            continue;
        pending = true;
        auto *author = buffer->inputAuthor ? buffer->inputAuthor.get() : buffer->input.get();
        for (const auto &source : input.GetResultSources())
            if (source->status == GraphBufferCapture::Status::Pending)
                access.pendingResults.push_back({author, source->view});
    }
    return pending;
}

void GraphBufferWrite::Publish(rhi::SubmissionTicket ticket) noexcept
{
    allocation->PublishGraphValue(ticket, rhi::QueueRole::Graphics, std::move(initialized), std::move(written));
}

std::vector<GraphBufferWrite> CaptureGraphBufferWrites(const RenderGraphDescription &graph,
                                                       const RenderGraphWorkStorage &storage,
                                                       const GraphBufferContents &contents,
                                                       const std::unordered_set<std::string> &writes)
{
    std::unordered_set<const rhi::ComputeBufferAllocation *> writers;
    for (const auto &name : writes) {
        const auto owner = storage.buffers.find(name);
        if (owner != storage.buffers.end())
            writers.insert(owner->second->GetAllocation().get());
    }
    std::vector<GraphBufferWrite> result;
    for (const auto &buffer : graph.buffers) {
        if (!buffer.exportLayout)
            continue;
        const auto &allocation = storage.buffers.at(buffer.name)->GetAllocation();
        if (writers.erase(allocation.get())) {
            const auto &known = contents.at(buffer.name);
            result.push_back({allocation, known, known});
        }
    }
    return result;
}

GraphFramePublication::~GraphFramePublication()
{
    for (const auto &capture : captures)
        capture.request->Cancel("graph submission did not succeed");
}

void GraphFramePublication::Accept(rhi::SubmissionTicket completion, GraphEventQueue &acceptedEvents) noexcept
{
    for (auto &write : writes)
        write.Publish(completion);
    for (auto &capture : captures)
        if (capture.request->status == GraphBufferCapture::Status::Pending) {
            capture.request->value = std::move(capture.value);
            capture.request->status = GraphBufferCapture::Status::Accepted;
        }
    captures.clear();
    for (auto &event : events)
        event.submission = completion;
    acceptedEvents.splice(acceptedEvents.end(), events);
}

bool RenderGraphResults::HasPendingCaptures() const
{
    return std::any_of(m_captures.begin(), m_captures.end(), [](const auto &weak) {
        const auto capture = weak.lock();
        return capture && capture->status == GraphBufferCapture::Status::Pending;
    });
}

void RenderGraphResults::Reset(const std::string &reason)
{
    Cancel(reason);
    ++m_generation;
}

void RenderGraphResults::Cancel(const std::string &reason)
{
    for (const auto &weak : m_captures)
        if (const auto capture = weak.lock())
            capture->Cancel(reason);
    m_captures.clear();
}

static const GraphBufferDesc &FindBufferExport(const RenderGraphDescription &graph, const std::string &name)
{
    const auto buffer = std::find_if(graph.buffers.begin(), graph.buffers.end(), [&](const auto &item) {
        return item.name == name && item.exportLayout.has_value();
    });
    if (buffer == graph.buffers.end())
        throw std::invalid_argument("No exported buffer in this view: " + name);
    return *buffer;
}

std::shared_ptr<GraphBufferCapture> RenderGraphResults::MakeCapture(const GraphBufferDesc &buffer,
                                                                    rhi::RenderViewId view) const
{
    auto capture = std::make_shared<GraphBufferCapture>();
    capture->name = buffer.name;
    capture->shape = buffer.exportShape;
    capture->scalarType = buffer.exportLayout->scalarType;
    capture->byteOffset = buffer.byteOffset;
    capture->storageByteSize = buffer.exportLayout->GetByteSize();
    capture->view = view;
    capture->generation = m_generation;
    capture->occurrence = m_occurrence + 1;
    return capture;
}

std::shared_ptr<GraphBufferCapture> RenderGraphResults::CaptureBuffer(const RenderGraphDescription &graph,
                                                                      rhi::RenderViewId view, const std::string &name)
{
    const auto &buffer = FindBufferExport(graph, name);
    auto capture = MakeCapture(buffer, view);
    const auto &storage = ResolveGraphBufferStorage(graph, buffer);
    auto imported = storage.computeBuffer;
    if (!storage.workBinding.empty()) {
        const auto *work = FindGraphWorkBinding(graph, storage.workBinding);
        imported = work->recipe->GetSlots()[storage.workSlot].external;
        capture->storageClaim = "work:" + work->name + "/" + std::to_string(storage.workSlot);
    }
    if (storage.input) {
        capture->storageClaim = "input:" + std::to_string(reinterpret_cast<uintptr_t>(storage.input.get()));
    } else if (imported && imported->IsFrozen()) {
        capture->storageBuffer = imported;
        const auto handle = imported->GetBuffer();
        capture->storageClaim = "frozen:" + std::to_string(handle.index) + ":" + std::to_string(handle.generation);
    } else if (imported)
        capture->storageClaim = "import:" + std::to_string(reinterpret_cast<uintptr_t>(imported.get()));
    else if (capture->storageClaim.empty())
        capture->storageClaim = "buffer:" + storage.name;
    m_captures.push_back(capture);
    return capture;
}

std::shared_ptr<GraphEventInterest> FindActiveGraphEventInterest(const GraphEventInterests &interests,
                                                                 const std::string &label)
{
    const auto found = interests.find(label);
    if (found == interests.end())
        return {};
    auto interest = found->second.lock();
    return interest && interest->active ? interest : nullptr;
}

std::shared_ptr<GraphFramePublication> RenderGraphResults::Prepare(
    const RenderGraphDescription &graph, const RenderGraphWorkStorage &storage, const GraphBufferContents &contents,
    const std::unordered_set<std::string> &writes, const std::vector<uint32_t> &eventPasses,
    const GraphEventInterests &interests, rhi::RenderViewId view, uint64_t historyEpoch,
    const std::function<std::shared_ptr<GraphEventViewSample>()> &sample,
    const std::shared_ptr<const GameScreenAdmission> &screenAdmission, const rhi::ComputeBufferBindings *recordedReads)
{
    if (storage.buffers.empty() && eventPasses.empty())
        return {};
    std::shared_ptr<GraphFramePublication> publication;
    const auto prepare = [&]() -> GraphFramePublication & {
        if (!publication)
            publication = std::make_shared<GraphFramePublication>();
        return *publication;
    };
    auto written = CaptureGraphBufferWrites(graph, storage, contents, writes);
    if (!written.empty())
        prepare().writes = std::move(written);
    ++m_occurrence;
    std::unordered_map<std::string, std::shared_ptr<rhi::ComputeBuffer>> frozen;
    const auto prepareCapture = [&](const std::shared_ptr<GraphBufferCapture> &capture) {
        auto [entry, inserted] = frozen.try_emplace(capture->name);
        if (inserted) {
            const auto &owner = storage.buffers.at(capture->name);
            // Keep the allocation used by this execution's descriptors and commands.
            if (recordedReads && owner->GetAllocation() != recordedReads->at(owner.get()))
                throw std::logic_error("Graph output changed after actual resource capture");
            entry->second = owner->FreezeScalars();
        }
        prepare().captures.push_back({capture, entry->second});
        capture->occurrence = m_occurrence;
        capture->screenAdmission = screenAdmission;
        capture->selected = true;
    };
    auto output = m_captures.begin();
    for (auto it = m_captures.begin(); it != m_captures.end(); ++it) {
        const auto capture = it->lock();
        if (!capture || capture->status != GraphBufferCapture::Status::Pending)
            continue;
        *output++ = *it;
        if (!capture->selected)
            prepareCapture(capture);
    }
    m_captures.erase(output, m_captures.end());
    std::shared_ptr<GraphEventViewSample> viewSample;
    for (size_t index : eventPasses) {
        const auto &pass = graph.passes[index];
        for (size_t ordinal = 0; ordinal < pass.events.size(); ++ordinal) {
            const auto &declaration = pass.events[ordinal];
            const auto interest = FindActiveGraphEventInterest(interests, declaration.label);
            if (!interest)
                continue;
            if (!viewSample)
                viewSample = sample();
            decltype(GraphEventRecord::sources) sources;
            sources.reserve(declaration.buffers.size());
            for (const auto &[name, resource] : declaration.buffers) {
                auto capture = MakeCapture(FindBufferExport(graph, resource), view);
                prepareCapture(capture);
                sources.emplace_back(name, std::move(capture));
            }
            prepare().events.push_back({declaration.label,
                                        declaration.metadata,
                                        interest,
                                        pass.name,
                                        ordinal,
                                        view,
                                        m_generation,
                                        m_occurrence,
                                        historyEpoch,
                                        {},
                                        viewSample,
                                        std::move(sources)});
        }
    }
    return publication;
}

} // namespace infernux
