#include "JoltGpuContactReports.h"
#include "JoltGpuProgramSources.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
enum Port : uint8_t
{
    Params,
    Patches,
    Contacts,
    Models,
    States,
    Targets,
    LeafInterests,
    Indices,
    Words,
    PortCount
};
constexpr uint32_t Bit(Port p)
{
    return 1u << p;
}
constexpr uint32_t Masks[] = {(1u << PortCount) - 1u, Bit(Params) | Bit(Contacts) | Bit(Indices) | Bit(Words),
                              Bit(Params) | Bit(Indices) | Bit(Words)};
struct Layout
{
    std::string_view name;
    uint32_t stride;
    std::array<std::string_view, 5> fields;
    uint32_t integers;
    bool read;
};
constexpr Layout Layouts[] = {
    {"contact_report_params", 48, {"range", "selection", "requests"}, 7, true},
    {"contact_patches", 16, {"leaves"}, 1, true},
    {"contacts", 80, {"meta", "normalDepth", "localAnchorA", "localAnchorB", "impulse"}, 1, true},
    {"body_models", 80, {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"}, 1, true},
    {"body_states", 64, {"positionCOM", "rotation", "linearVelocity", "angularVelocity"}, 0, true},
    {"kinematic_targets", 64, {"positionSpeed", "rotation", "rootCOMDt", "state"}, 8, true},
    {"contact_report_interest", 4, {}, 0, true},
    {"contact_report_indices", 4, {}, 0, false},
    {"contact_report_words", 16, {"value"}, 1, false}};
template <class T> std::vector<uint8_t> Bytes(const T *p, size_t n = 1)
{
    if (n > SIZE_MAX / sizeof(T))
        throw std::length_error("Contact report upload overflow");
    std::vector<uint8_t> r(n * sizeof(T));
    if (!r.empty())
        std::memcpy(r.data(), p, r.size());
    return r;
}
void Validate(Port p, const ShaderResourceLayout &r)
{
    const auto &e = Layouts[p];
    const bool uniform = p == Params;
    if (r.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? r.value.byteSize : r.elementStride) != e.stride || (!uniform && r.binding.readOnlyStorage != e.read))
        throw std::invalid_argument("Contact report binding layout mismatch");
    if (p == LeafInterests || p == Indices) {
        if (r.value.kind != ShaderValueKind::Scalar || r.value.scalar != ShaderScalarKind::UInt32 ||
            r.value.byteSize != 4 || r.value.rows != 1 || r.value.columns != 1 || !r.value.fields.empty())
            throw std::invalid_argument("Contact report scalar layout mismatch");
        return;
    }
    if (r.value.kind != ShaderValueKind::Struct || r.value.fields.size() != e.stride / 16)
        throw std::invalid_argument("Contact report record layout mismatch");
    for (size_t i = 0; i < r.value.fields.size(); ++i) {
        const auto &f = r.value.fields[i];
        if (f.name != e.fields[i] || f.offset != 16 * i || f.byteSize != 16 || f.kind != ShaderValueKind::Vector ||
            f.scalar != ((e.integers & (1u << i)) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32) ||
            f.rows != 1 || f.columns != 4)
            throw std::invalid_argument("Contact report field layout mismatch");
    }
}
} // namespace
struct JoltGpuContactReports::Impl
{
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Port> ports;
    };
    rhi::ComputeHost host;
    std::array<Program, 3> programs;
    Impl(rhi::ComputeHost &h, const ShaderTargetCode &target) : host(h)
    {
        if (target.target != ShaderCodeTarget::SpirV && target.target != ShaderCodeTarget::Wgsl)
            throw std::invalid_argument("Contact reports require SPIR-V/WGSL");
        constexpr std::string_view names[]{"contact_report_seed", "contact_report_finish", "contact_report_initialize"};
        for (uint32_t i = 0; i < 3; ++i) {
            const auto code = std::find_if(target.entries.begin(), target.entries.end(),
                                           [&](const auto &v) { return v.name == names[i]; });
            if (code == target.entries.end() || code->stage != rhi::ShaderStage::Compute ||
                code->workgroupSize != std::array<uint32_t, 3>{64, 1, 1})
                throw std::invalid_argument("Contact report entry mismatch");
            std::vector<rhi::ComputeBufferBinding> bindings;
            uint32_t mask = 0, storage = 0;
            auto &program = programs[i];
            for (const auto &r : target.resources) {
                const std::array<uint32_t, 2> key{r.set, r.binding.binding};
                if (std::find(code->bindings.begin(), code->bindings.end(), key) == code->bindings.end())
                    continue;
                const auto f = std::find_if(std::begin(Layouts), std::end(Layouts),
                                            [&](const auto &v) { return v.name == r.name; });
                if (f == std::end(Layouts) || r.set)
                    throw std::invalid_argument("Unknown contact report binding");
                const auto p = static_cast<Port>(f - std::begin(Layouts));
                Validate(p, r);
                if (mask & Bit(p))
                    throw std::invalid_argument("Repeated contact report binding");
                mask |= Bit(p);
                storage += r.binding.type == rhi::BindingType::StorageBuffer;
                program.ports.push_back(p);
                bindings.push_back({r.binding.binding, r.binding.type, r.binding.readOnlyStorage});
            }
            if (mask != Masks[i] || storage > 8 || bindings.size() != code->bindings.size())
                throw std::invalid_argument("Contact report resource mask mismatch");
            program.kernel = PrepareJoltGpuKernel(host, target.target, *code, std::move(bindings));
        }
    }
    bool Allocate(std::shared_ptr<rhi::ComputeBuffer> &p, uint64_t bytes)
    {
        if (p)
            return true;
        if (!bytes || bytes % 4 || bytes > SIZE_MAX ||
            !host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
            throw std::length_error("Contact report buffer exceeds device range");
        p = rhi::ComputeBuffer::TryCreatePrivate(
            host, {bytes / 4, rhi::ComputeScalarType::UInt32, 1, rhi::BufferUsageFlags::None});
        return p != nullptr;
    }
    struct Selection
    {
        Interest factory;
        std::vector<uint32_t> values;
        std::shared_ptr<rhi::ComputeBuffer> buffer;
        size_t uploadFirst = 0;
        bool captured = false, uploaded = false;
    };
    struct Job : std::enable_shared_from_this<Job>
    {
        std::shared_ptr<Impl> impl;
        std::shared_ptr<JoltGpuContactReportResult> result;
        std::shared_ptr<Selection> selection;
        JoltGpuContactReportResult::Frame *frame = nullptr;
        uint32_t stage = 0, first = 0, initFirst = 0;
        bool swept = false;
        std::function<JoltGpuControlWork()> done;
        ContactReportParams params{};
        std::array<std::shared_ptr<rhi::ComputeBuffer>, PortCount> ports;
        std::shared_ptr<rhi::ComputeBuffer> uniform;
        bool initialized = false;
        JoltGpuControlWork Next()
        {
            auto &f = *frame;
            if (first == f.capacity) {
                if (stage == 0)
                    f.seeded = true;
                if (stage != 0 || swept) {
                    f.finished = true;
                    f.parameters.reset();
                }
                return done();
            }
            if (stage != 0 && !f.parameters)
                throw std::logic_error("Contact velocity report lost its admitted pre-velocity parameters");
            if (!impl->Allocate(f.indices, 4ull * (2ull + f.capacity)) ||
                !impl->Allocate(f.words, uint64_t(f.capacity) * sizeof(ContactReportPatch)) ||
                !impl->Allocate(selection->buffer, 4ull * std::max<size_t>(1, selection->values.size())) ||
                !impl->Allocate(f.parameters, sizeof(params)))
                return {nullptr, [s = shared_from_this()] { return s->Next(); }};
            uniform = f.parameters;
            f.interest = selection->buffer;
            ports[Params] = uniform;
            ports[LeafInterests] = f.interest;
            ports[Indices] = f.indices;
            ports[Words] = f.words;
            if (!selection->uploaded) {
                const size_t count = std::min<size_t>(2048, selection->values.size() - selection->uploadFirst);
                if (!count)
                    throw std::logic_error("Nonempty contact workset needs a selected leaf range");
                const auto next = selection->uploadFirst + count;
                auto upload =
                    rhi::PrepareComputeBatch(impl->host,
                                             {{selection->buffer, selection->uploadFirst * 4,
                                               Bytes(selection->values.data() + selection->uploadFirst, count)}},
                                             {}, {});
                return {upload,
                        [s = shared_from_this(), next] {
                            s->selection->uploadFirst = next;
                            s->selection->uploaded = next == s->selection->values.size();
                            return s->Next();
                        },
                        JoltGpuControlWork::ResumeAfter::Acceptance};
            }
            const bool initialize = stage == 0 && !initialized;
            const auto entry = initialize ? 2u : stage;
            params.range.x = initialize ? initFirst : first;
            params.range.y = std::min(2048u, f.capacity - params.range.x);
            std::vector<rhi::ComputeBufferUpdate> uploads{{uniform, 0, Bytes(&params)}};
            if (initialize && !initFirst) {
                const std::array<uint32_t, 2> header{};
                uploads.push_back({f.indices, 0, Bytes(header.data(), header.size())});
            }
            rhi::ComputeDispatchDesc d;
            d.kernel = impl->programs[entry].kernel;
            d.groupCountX = (params.range.y + 63) / 64;
            if (d.groupCountX > impl->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0])
                throw std::length_error("Contact report window exceeds device limit");
            rhi::ComputeDispatchProof proof;
            for (auto p : impl->programs[entry].ports) {
                auto &b = ports[p];
                if (!b || !impl->host.SharesServicesWith(b->GetHost()))
                    throw std::invalid_argument("Contact report missing/shared-host endpoint");
                const auto binding = static_cast<uint32_t>(d.buffers.size());
                d.buffers.push_back(b);
                if (initialize && (p == Indices || p == Words)) {
                    d.bufferAccesses.push_back(rhi::ComputeBufferAccess::Write);
                    const uint64_t stride = p == Indices ? 4 : sizeof(ContactReportPatch);
                    proof.writes.push_back({binding, (params.range.x + (p == Indices ? 2ull : 0ull)) * stride,
                                            uint64_t(params.range.y) * stride});
                } else {
                    d.bufferAccesses.push_back(p == Indices || p == Words ? rhi::ComputeBufferAccess::ReadWrite
                                                                          : rhi::ComputeBufferAccess::Read);
                    proof.reads.push_back({binding, 0, b->GetByteSize()});
                }
            }
            d.proof = std::move(proof);
            auto packet = rhi::PrepareComputeBatch(impl->host, std::move(uploads), {std::move(d)}, {});
            const auto next = params.range.x + params.range.y;
            return {packet,
                    [s = shared_from_this(), next, initialize] {
                        if (initialize) {
                            s->initFirst = next;
                            s->initialized = next == s->frame->capacity;
                        } else
                            s->first = next;
                        s->uniform.reset();
                        return s->Next();
                    },
                    JoltGpuControlWork::ResumeAfter::Acceptance};
        }
    };
    struct Reader : std::enable_shared_from_this<Reader>
    {
        std::shared_ptr<Impl> impl;
        std::shared_ptr<JoltGpuContactReportResult> result;
        std::vector<JoltGpuContactReportResult::Frame *> frames;
        std::function<void()> done;
        size_t next = 0;
        uint32_t prefixFirst = 0;
        void RetireFrame()
        {
            auto &frame = *frames[next++];
            frame.interest.reset();
            frame.indices.reset();
            frame.words.reset();
            prefixFirst = 0;
        }
        JoltGpuControlWork Prefix()
        {
            auto &frame = *frames[next];
            const auto count = std::min<size_t>(2048, frame.records.size() - prefixFirst);
            auto packet = rhi::PrepareComputeBatch(impl->host, {}, {},
                                                   {{frame.words, uint64_t(prefixFirst) * sizeof(ContactReportPatch),
                                                     count * sizeof(ContactReportPatch)}});
            return {packet, [s = shared_from_this(), packet, count] {
                        auto &v = s->frames[s->next]->records;
                        auto r = packet->Readback(0);
                        if (!r || !r->IsComplete())
                            throw std::logic_error("Contact prefix requires completed receipt");
                        r->ReadInto(v.data() + s->prefixFirst, count * sizeof(ContactReportPatch));
                        s->prefixFirst += static_cast<uint32_t>(count);
                        if (s->prefixFirst < v.size())
                            return s->Prefix();
                        s->RetireFrame();
                        return s->Next();
                    }};
        }
        JoltGpuControlWork Next()
        {
            while (next < frames.size() && !frames[next]->capacity)
                ++next;
            if (next == frames.size()) {
                done();
                return {};
            }
            auto &f = *frames[next];
            if (!f.seeded || !f.finished)
                throw std::logic_error("Partial contact substep cannot publish");
            auto packet = rhi::PrepareComputeBatch(impl->host, {}, {}, {{f.indices, 0, 8}});
            return {packet, [s = shared_from_this(), packet]() -> JoltGpuControlWork {
                        uint32_t h[2]{};
                        auto read = packet->Readback(0);
                        if (!read || !read->IsComplete())
                            throw std::logic_error("Contact count requires completed receipt");
                        read->ReadInto(h, sizeof(h));
                        auto &frame = *s->frames[s->next];
                        if (h[1] || h[0] > frame.capacity)
                            throw std::runtime_error("GPU contact report count/status rejected");
                        if (!h[0]) {
                            s->RetireFrame();
                            return s->Next();
                        }
                        frame.records.resize(h[0]);
                        return s->Prefix();
                    }};
        }
    };
};
JoltGpuContactReports::JoltGpuContactReports(rhi::ComputeHost &h, const ShaderTargetCode &t)
    : m_impl(std::make_shared<Impl>(h, t))
{
}
JoltGpuContactReports::Capture JoltGpuContactReports::Prepare(std::shared_ptr<const JoltGpuScene::Publication> p,
                                                              uint32_t substeps, uint32_t flags,
                                                              Interest interest) const
{
    if (!p || !substeps || flags & ~7u || !interest)
        throw std::invalid_argument("Invalid contact capture request");
    auto result = std::make_shared<JoltGpuContactReportResult>();
    result->publication = std::move(p);
    result->expectedSubsteps = substeps;
    auto selection = std::make_shared<Impl::Selection>();
    selection->factory = std::move(interest);
    Capture capture;
    capture.result = result;
    capture.phase = [impl = m_impl, result, flags, selection](
                        const JoltGpuStorage &storage, const JoltGpuStepInputs &inputs, const StepParams &params,
                        uint32_t substep, JoltGpuContactReportStage stage, std::function<void()> done) {
        if (!done || !impl->host.SharesServicesWith(storage.GetHost()) || substep >= result->expectedSubsteps ||
            params.solve.y != result->expectedSubsteps || !result->publication->directory)
            throw std::logic_error("Contact phase identity mismatch");
        const bool swept = stage == JoltGpuContactReportStage::Sweep;
        const auto lease = swept ? inputs.GetSweepContacts() : inputs.GetContacts();
        if (!lease)
            throw std::logic_error("Contact phase lacks accepted workset");
        const auto &publication = *result->publication;
        if (publication.worldRow != params.selection.x || publication.bodyBase != params.selection.y ||
            publication.bodyCount != params.selectedCounts.x || publication.leafBase != params.selection.z ||
            publication.leafCount != params.selectedCounts.y || uint64_t(lease->patchSlots) * 22u > UINT32_MAX)
            throw std::logic_error("Contact phase differs from accepted publication/range");
        if (stage == JoltGpuContactReportStage::BeforeVelocity) {
            if (substep != result->frames.size())
                throw std::logic_error("Repeated/out-of-order contact seed");
            result->frames.push_back({substep, lease->patchSlots});
        }
        auto &frame = swept ? result->sweepFrames[substep].emplace_back(
                                  JoltGpuContactReportResult::Frame{substep, lease->patchSlots})
                            : result->frames.at(substep);
        if (frame.capacity != lease->patchSlots ||
            (stage == JoltGpuContactReportStage::AfterVelocity && (!frame.seeded || frame.finished)))
            throw std::logic_error("Contact phase lease mismatch");
        auto job = std::make_shared<Impl::Job>();
        job->impl = impl;
        job->result = result;
        job->selection = selection;
        job->frame = &frame;
        job->swept = swept;
        job->stage = stage == JoltGpuContactReportStage::AfterVelocity ? 1 : 0;
        job->done = [impl, result, swept, frame = &frame, done = std::move(done)]() mutable -> JoltGpuControlWork {
            if (!swept) {
                done();
                return {};
            }
            // The next sweep reuses the collision workset. Keep its compact
            // witnesses and retire the GPU capture before continuing motion.
            auto reader = std::make_shared<Impl::Reader>();
            reader->impl = impl;
            reader->result = result;
            reader->frames.push_back(frame);
            reader->done = std::move(done);
            return reader->Next();
        };
        job->params = {{0, 0, frame.capacity, substep},
                       {params.selection.x, params.selection.y, params.selectedCounts.x, params.selection.z},
                       {params.selectedCounts.y, flags, uint32_t(swept), 0}};
        if (!selection->captured) {
            selection->values = selection->factory(*result->publication);
            selection->captured = true;
        }
        if (selection->values.size() != params.selectedCounts.y)
            throw std::logic_error("Contact interest differs from accepted leaf range");
        job->ports[Patches] = lease->patches;
        job->ports[Contacts] = lease->contacts;
        job->ports[Models] = inputs.GetCaptured(Resource::BodyModels);
        job->ports[States] = storage.Get(Resource::BodyStates);
        job->ports[Targets] = storage.Get(Resource::KinematicTargets);
        return job->Next();
    };
    capture.finish = [impl = m_impl, result](const std::shared_ptr<const JoltGpuScene::Publication> &publication) {
        if (publication.get() != result->publication.get() || result->complete)
            throw std::logic_error("Contact final publication mismatch");
        if (!publication->bodyCount && result->frames.empty())
            for (uint32_t i = 0; i < result->expectedSubsteps; ++i) {
                result->frames.push_back({i, 0});
                result->frames.back().seeded = result->frames.back().finished = true;
            }
        if (result->frames.size() != result->expectedSubsteps)
            throw std::logic_error("Contact Fixed lacks complete substeps");
        auto reader = std::make_shared<Impl::Reader>();
        reader->impl = impl;
        reader->result = result;
        reader->done = [result] { result->complete = true; };
        for (auto &frame : result->frames)
            reader->frames.push_back(&frame);
        return reader->Next();
    };
    return capture;
}
} // namespace infernux
