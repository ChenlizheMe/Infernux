#include "JoltGpuNarrowphase.h"
#include <Jolt/GPU/ConvexGeometryTypes.h>
#include "JoltGpuGeometryStore.h"
#include "JoltGpuProgramSources.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernux
{
namespace
{
using namespace JPH::GPU;

enum Port : uint8_t
{
    Params,
    Work,
    Shapes,
    States,
    Patches,
    Contacts,
    Geometry,
    Scratch,
    PortCount
};
enum Entry : uint8_t
{
    Initialize,
    Advance,
    Progress,
    EntryCount
};
constexpr std::array<std::string_view, EntryCount> EntryNames{"narrowphase_init_jobs", "narrowphase_advance_jobs",
                                                              "narrowphase_job_progress"};
constexpr std::array<uint32_t, EntryCount> EntryPorts{0x83u, 0xfdu, 0x80u};
constexpr uint32_t HeaderWords = 4u, MaxSlots = 32u, AdvancesPerPacket = 8u;
constexpr uint64_t WordBytes = sizeof(UInt4);

struct PortLayout
{
    std::string_view name;
    uint64_t stride;
    std::array<std::string_view, 8> fields;
    uint32_t integers;
    bool readOnly;
};
constexpr std::array<PortLayout, PortCount> Ports{{
    {"step_params",
     sizeof(StepParams),
     {"counts", "solve", "gravityDt", "contact", "restitution", "selection", "selectedCounts"},
     0x63u,
     true},
    {"stage_work", sizeof(StageWork), {"range"}, 1u, true},
    {"shape_models",
     sizeof(ShapeModel),
     {"meta", "geometry", "localPositionCOM", "localRotation", "material", "surface", "source", "geometryReference"},
     0xe1u,
     true},
    {"body_states", sizeof(BodyState), {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0u, true},
    {"contact_patches", sizeof(ContactPatch), {"leaves"}, 1u, false},
    {"contacts", sizeof(Contact), {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"}, 1u, false},
    {"convex_geometry", WordBytes, {}, 0u, true},
    {"convex_scratch", WordBytes, {}, 0u, false},
}};

uint32_t Groups(uint32_t count)
{
    return count / 64u + uint32_t(count % 64u != 0u);
}

void ValidateLayout(Port port, const ShaderResourceLayout &resource)
{
    const auto &expected = Ports[port];
    const bool uniform = port == Params || port == Work;
    const auto &value = resource.value;
    if (resource.set != 0u || resource.binding.binding != port ||
        resource.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? value.byteSize : resource.elementStride) != expected.stride ||
        (!uniform && resource.binding.readOnlyStorage != expected.readOnly))
        throw std::invalid_argument("GPU narrowphase binding differs from shared POD");
    if (port == Geometry || port == Scratch) {
        if (value.kind != ShaderValueKind::Vector || value.scalar != ShaderScalarKind::UInt32 || value.rows != 1u ||
            value.columns != 4u || value.byteSize != WordBytes || !value.fields.empty())
            throw std::invalid_argument("GPU narrowphase geometry/scratch requires raw uint4 storage");
        return;
    }
    if (value.kind != ShaderValueKind::Struct || value.fields.size() != expected.stride / WordBytes)
        throw std::invalid_argument("GPU narrowphase struct differs from shared POD");
    for (size_t i = 0; i < value.fields.size(); ++i) {
        const auto &field = value.fields[i];
        const auto scalar = expected.integers & (1u << i) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32;
        if (field.name != expected.fields[i] || field.offset != i * WordBytes || field.byteSize != WordBytes ||
            field.kind != ShaderValueKind::Vector || field.scalar != scalar || field.rows != 1u || field.columns != 4u)
            throw std::invalid_argument("GPU narrowphase field differs from shared POD");
    }
}
} // namespace

class JoltGpuNarrowphase::Impl
{
  public:
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Port> ports;
    };
    struct Execution;

    Impl(rhi::ComputeHost &source, const ShaderTargetCode &target) : host(source)
    {
        if (target.target != ShaderCodeTarget::SpirV && target.target != ShaderCodeTarget::Wgsl)
            throw std::invalid_argument("GPU narrowphase requires the engine SPIR-V/WGSL target");
        for (uint8_t i = 0; i < EntryCount; ++i) {
            const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                           [&](const auto &item) { return item.name == EntryNames[i]; });
            if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
                code->workgroupSize != std::array<uint32_t, 3>{64u, 1u, 1u})
                throw std::invalid_argument("GPU narrowphase entry/workgroup differs from module");
            auto &program = entries[i];
            std::vector<rhi::ComputeBufferBinding> bindings;
            uint32_t mask = 0;
            for (const auto &resource : target.resources) {
                const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
                if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                    continue;
                const auto port = std::find_if(Ports.begin(), Ports.end(),
                                               [&](const auto &item) { return item.name == resource.name; });
                if (port == Ports.end())
                    throw std::invalid_argument("GPU narrowphase entry contains an unknown resource");
                const auto index = static_cast<Port>(port - Ports.begin());
                ValidateLayout(index, resource);
                if (mask & (1u << index))
                    throw std::invalid_argument("GPU narrowphase entry repeats a resource");
                mask |= 1u << index;
                program.ports.push_back(index);
                bindings.push_back({resource.binding.binding, resource.binding.type, resource.binding.readOnlyStorage});
            }
            if (mask != EntryPorts[i] || code->bindings.size() != program.ports.size())
                throw std::invalid_argument("GPU narrowphase entry differs from its resource mask");
            program.kernel = PrepareJoltGpuKernel(host, target.target, *code, std::move(bindings));
        }
    }
    rhi::ComputeHost host;
    std::array<Program, EntryCount> entries;
    double advanceNanoseconds = 0;
};

struct JoltGpuNarrowphase::Impl::Execution : std::enable_shared_from_this<Execution>
{
    enum class Phase
    {
        InitializeJobs,
        AdvanceJobs,
        Done
    };

    Execution(std::shared_ptr<Impl> implementation, const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
              FrameWorkOwner &budget, std::function<void()> completed)
        : implementation(std::move(implementation)), contactLease(inputs.GetContacts()), geometry(inputs.GetGeometry()),
          budget(budget), completed(std::move(completed))
    {
        if (!this->implementation->host.SharesServicesWith(storage.GetHost()) || !contactLease || !this->completed)
            throw std::invalid_argument("GPU narrowphase requires the accepted Tick host/contact workset");
        frozen[Params] = contactLease->parameters;
        frozen[Shapes] = inputs.GetCaptured(Resource::ShapeModels);
        frozen[States] = storage.Get(Resource::BodyStates);
        frozen[Patches] = contactLease->patches;
        frozen[Contacts] = contactLease->contacts;
        if (!geometry || !geometry->buffer)
            throw std::logic_error("GPU convex candidates require their captured geometry lease");
        frozen[Geometry] = geometry->buffer;
        const auto &caps = this->implementation->host.device.GetCapabilities();
        const uint64_t words = std::min(caps.portable.maxStorageBufferBinding / WordBytes,
                                        uint64_t(caps.limits.maxComputeWorkgroupCount[0]) * 64u);
        if (words < HeaderWords + ConvexJobSlotWords)
            throw std::length_error("Device cannot bind and initialize one convex job slot");
        slots = static_cast<uint32_t>(std::min(
            {uint64_t(MaxSlots), uint64_t(contactLease->patchCount), (words - HeaderWords) / ConvexJobSlotWords}));
        scratchWords = HeaderWords + slots * ConvexJobSlotWords;
    }

    JoltGpuControlWork Retain(JoltGpuControlWork work) const
    {
        work.geometry = geometry;
        return work;
    }

    bool Allocate(std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes)
    {
        if (!buffer)
            buffer = rhi::ComputeBuffer::TryCreatePrivate(
                implementation->host, {bytes / 4u, rhi::ComputeScalarType::UInt32, 1u, rhi::BufferUsageFlags::None});
        return buffer != nullptr;
    }

    bool AllocateScratch()
    {
        if (scratch)
            return true;
        // Choose concurrency before any job has started. Each grant runs the
        // same strided jobs; reducing slots never restarts numerical work.
        uint32_t count = slots;
        while (true) {
            const uint32_t words = HeaderWords + count * ConvexJobSlotWords;
            if (Allocate(scratch, uint64_t(words) * WordBytes)) {
                slots = count;
                scratchWords = words;
                return true;
            }
            if (count == 1u)
                return false;
            count = (count + 1u) / 2u;
        }
    }

    rhi::ComputeDispatchDesc Dispatch(Entry entry) const
    {
        const auto &program = implementation->entries[entry];
        rhi::ComputeDispatchDesc dispatch;
        dispatch.kernel = program.kernel;
        dispatch.groupCountX = entry == Initialize ? Groups(scratchWords) : 1u;
        rhi::ComputeDispatchProof proof;
        for (const auto port : program.ports) {
            const auto &buffer = port == Scratch ? scratch : port == Work ? workUniform : frozen[port];
            const uint32_t binding = static_cast<uint32_t>(dispatch.buffers.size());
            dispatch.buffers.push_back(buffer);
            const bool initialize = entry == Initialize && port == Scratch;
            const bool mutablePort = port == Scratch || port == Patches || port == Contacts;
            dispatch.bufferAccesses.push_back(initialize    ? rhi::ComputeBufferAccess::Write
                                              : mutablePort ? rhi::ComputeBufferAccess::ReadWrite
                                                            : rhi::ComputeBufferAccess::Read);
            if (initialize)
                proof.writes.push_back({binding, 0u, buffer->GetByteSize()});
            else if (port == Geometry) {
                for (const auto &range : geometry->wordRanges)
                    proof.reads.push_back({binding, range.first * WordBytes, (range.second - range.first) * WordBytes});
            } else
                proof.reads.push_back({binding, 0u, buffer->GetByteSize()});
        }
        dispatch.proof = std::move(proof);
        return dispatch;
    }

    void Finish()
    {
        phase = Phase::Done;
        auto callback = std::move(completed);
        callback();
    }

    JoltGpuControlWork Next(uint32_t maximumAdvances = AdvancesPerPacket)
    {
        if (phase == Phase::Done)
            return {};
        if (contactLease->patchCount == 0u) {
            Finish();
            return {};
        }
        if (!Allocate(workUniform, sizeof(StageWork)) || !AllocateScratch())
            return Retain({nullptr, [self = shared_from_this()] { return self->Next(); }});
        if (phase == Phase::InitializeJobs) {
            StageWork work{{0u, contactLease->patchCount, 0u, 0u}};
            std::vector<uint8_t> bytes(sizeof(work));
            std::memcpy(bytes.data(), &work, sizeof(work));
            auto packet = rhi::PrepareComputeBatch(implementation->host, {{workUniform, 0u, std::move(bytes)}},
                                                   {Dispatch(Initialize)});
            return Retain({packet,
                           [self = shared_from_this()] {
                               self->phase = Phase::AdvanceJobs;
                               return self->Next();
                           },
                           JoltGpuControlWork::ResumeAfter::Acceptance});
        }
        // Each invocation advances a fixed number of real support/EPA/clip
        // operations. The persisted cursor, not Host retry count, determines
        // remaining work. Only this small completion record crosses to CPU.
        const auto available = budget.ResourceRemaining(FrameWorkResource::Gpu);
        const auto window = available ? available : budget.ResourceState(FrameWorkResource::Gpu).limit;
        const double unitCost = implementation->advanceNanoseconds;
        const uint32_t count =
            unitCost > 0 ? static_cast<uint32_t>(std::clamp(double(window) / unitCost, 1.0, double(maximumAdvances)))
                         : 1u;
        const auto started = std::chrono::steady_clock::now();
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        for (uint32_t i = 0; i < count; ++i)
            dispatches.push_back(Dispatch(Advance));
        dispatches.push_back(Dispatch(Progress));
        auto packet =
            rhi::PrepareComputeBatch(implementation->host, {}, std::move(dispatches), {{scratch, 0u, WordBytes}});
        auto work = Retain(
            {packet, [self = shared_from_this(), packet, started, count]() -> JoltGpuControlWork {
                 // Planning only: Tick remains the sole resource accountant.
                 // Without timestamps use the conservative observed interval,
                 // including queue/Host lag, never assume missing time is zero.
                 const double observed = packet->GetCompletion().GetDeviceElapsedNanoseconds().value_or(
                     std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - started).count());
                 const double sample = observed / count;
                 self->implementation->advanceNanoseconds =
                     std::max(sample, self->implementation->advanceNanoseconds * .75);
                 UInt4 result{};
                 auto read = packet->Readback(0u);
                 read->ReadInto(&result, sizeof(result));
                 if (result.x == QuerySuccess) {
                     if (result.y != self->slots)
                         throw std::runtime_error("GPU convex completion omitted an active job slot");
                     self->Finish();
                     return {};
                 }
                 if (result.x != QueryPending)
                     throw std::runtime_error("GPU convex narrowphase failed at candidate " + std::to_string(result.z) +
                                              " (reason " + std::to_string(result.w) + ")");
                 return self->Next();
             }});
        if (count > 1u)
            work.prepareMinimum = [self = shared_from_this()] { return self->Next(1u); };
        return work;
    }

    std::shared_ptr<Impl> implementation;
    std::shared_ptr<const JoltGpuContactWorkset> contactLease;
    std::shared_ptr<const JoltGpuGeometryLease> geometry;
    FrameWorkOwner &budget;
    std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> frozen;
    std::shared_ptr<rhi::ComputeBuffer> scratch, workUniform;
    std::function<void()> completed;
    Phase phase = Phase::InitializeJobs;
    uint32_t slots = 0u, scratchWords = 0u;
};

JoltGpuNarrowphase::JoltGpuNarrowphase(rhi::ComputeHost &host, const ShaderTargetCode &target)
    : m_impl(std::make_shared<Impl>(host, target))
{
}

JoltGpuControlWork JoltGpuNarrowphase::Prepare(const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs,
                                               FrameWorkOwner &budget, std::function<void()> completed)
{
    return std::make_shared<Impl::Execution>(m_impl, storage, inputs, budget, std::move(completed))->Next();
}

} // namespace infernux
