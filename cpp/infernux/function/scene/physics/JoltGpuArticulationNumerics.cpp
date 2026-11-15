#include "JoltGpuArticulationNumerics.h"
#include <Jolt/GPU/ArticulationIntegrationTypes.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace infernux
{
namespace
{
using namespace JPH::GPU;
#include "JoltGpuArticulationEntries.inl"
template <class T> std::vector<uint8_t> Bytes(const std::vector<T> &v)
{
    std::vector<uint8_t> b(v.size() * sizeof(T));
    if (!b.empty())
        std::memcpy(b.data(), v.data(), b.size());
    return b;
}
template <class T> std::vector<uint8_t> Bytes(const T &v)
{
    std::vector<uint8_t> b(sizeof(T));
    std::memcpy(b.data(), &v, sizeof(T));
    return b;
}
uint32_t Load(const JoltGpuArticulationState &s, uint32_t byte)
{
    uint32_t v;
    std::memcpy(&v, s.prefix.data() + byte, 4);
    return v;
}
std::vector<JoltGpuArticulationNumericState::Origin> Origins(const JoltGpuArticulationState &s)
{
    std::vector<JoltGpuArticulationNumericState::Origin> result;
    for (uint32_t tree = 0; tree < s.trees.size(); ++tree) {
        const auto &t = s.trees[tree];
        for (uint32_t local = 1; local < t.range.y; ++local) {
            const auto &l = s.links[t.range.x + local];
            result.push_back({(uint64_t(l.identity.y) << 32u) | l.identity.x, l.topology.w, tree});
        }
    }
    return result;
}
void MatchRows(const JoltGpuStorage &storage, const JoltGpuArticulationState &s, JoltGpuArticulationNumericState &n)
{
    std::unordered_map<uint64_t, JoltGpuArticulationNumericState::Origin> old;
    for (const auto &o : n.origin)
        old.emplace(o.id, o);
    n.transitions.clear();
    // These are accepted native rows, not public A/B, component order or CPU pose.
    const auto &model = storage.GetModel();
    for (uint32_t row = 0; row < model.joints.size(); ++row) {
        const auto &j = model.joints[row];
        if (j.identity.z != s.worldRow)
            continue;
        const uint64_t id = (uint64_t(j.identity.y) << 32u) | j.identity.x;
        const auto found = old.find(id);
        if (found == old.end() || found->second.kind != j.bodies.z)
            continue;
        uint32_t tree = ArtControlInvalid;
        const auto h = s.header;
        if (row >= h.selected.z && row - h.selected.z < h.selected.w)
            tree = Load(s, h.maps.z + 4u * (row - h.selected.z));
        n.transitions.push_back({found->second.tree, row, tree, j.identity.z});
    }
}
bool Has(std::string_view list, std::string_view name)
{
    size_t first = 0;
    while (first < list.size()) {
        size_t end = list.find('|', first);
        if (end == std::string_view::npos)
            end = list.size();
        if (list.substr(first, end - first) == name)
            return true;
        first = end + 1;
    }
    return false;
}
bool Writer(std::string_view entry, std::string_view port)
{
    if (entry == "articulation_retire_declined" || entry == "articulation_retire_captured")
        return port == "articulation_retirement" || port == "joint_stream";
    if (entry == "articulation_clear_retirement_summary" || entry == "articulation_advance_retirement")
        return port == "articulation_retirement";
    if (entry == "articulation_retirement_summary")
        return port == "articulation_words";
    if (entry == "normal_capture_initialize")
        return true;
    if (entry == "articulation_clear_summary" || entry == "articulation_numeric_summary")
        return port == "articulation_summary";
    if (entry == "articulation_capture_transition")
        return port == "articulation_transition_witness";
    if (entry == "articulation_reset_transition")
        return port == "joint_stream" || port == "articulation_summary";
    if (entry.substr(0, 13) == "articulation_")
        return port == "articulation_words";
    if (entry == "normal_capture_begin")
        return port == "sf_trees" || port == "normal_capture_control";
    if (entry == "normal_capture_advance" || entry == "normal_seed_advance")
        return port == "sf_trees" || port == "sf_roots" || port == "sf_coordinates" || port == "normal_capture_control";
    if (entry == "normal_promote_begin")
        return port == "sf_trees" || port == "normal_capture_control";
    if (entry == "normal_promote_advance")
        return port == "sf_trees" || port == "normal_capture_control" || port == "normal_resident_roots" ||
               port == "normal_resident_coordinates";
    if (port == "art_trees")
        return true;
    if (entry == "normal_dynamics_preserve" || entry == "normal_dynamics_solve" || entry == "normal_reaction_advance")
        return port == "art_scratch";
    if (entry == "normal_history_advance")
        return port == "art_joints";
    if (entry == "normal_velocity_advance")
        return port == "art_states";
    if (entry == "normal_pose_advance")
        return port == "art_states" || port == "art_joints" || port == "dyn_roots" || port == "dyn_coordinates";
    return false;
}
void Validate(const ShaderResourceLayout &r)
{
    const auto &name = r.name;
    const auto &v = r.value;
    const bool uniform = name == "sf_work" || name == "art_work" || name == "articulation_params";
    const bool readOnly = name == "sf_models" || name == "sf_accepted" || name == "sf_joints" || name == "sf_links" ||
                          name == "art_models" || name == "art_links" || name == "normal_active_roots" ||
                          name == "normal_active_coordinates" || name == "normal_dynamics_trees" ||
                          name == "articulation_trees" || name == "articulation_capture_trees" ||
                          name == "articulation_previous_control" || name == "articulation_transitions";
    const bool scalar =
        name == "normal_capture_control" || name == "articulation_words" || name == "articulation_summary";
    const bool floatVector = name == "sf_coordinates" || name == "dyn_coordinates" ||
                             name == "normal_active_coordinates" || name == "normal_resident_coordinates" ||
                             name == "articulation_transition_witness";
    const bool uintVector = name == "sf_joints" || name == "art_joints" || name == "art_scratch" ||
                            name == "articulation_transitions" || name == "articulation_retirement";
    uint32_t stride = scalar ? 4u : 16u, integers = 0;
    std::vector<std::string_view> fields;
    if (uniform) {
        stride = 32;
        fields = {"range", name == "sf_work" ? "tolerance" : name == "art_work" ? "integration" : "references"};
        integers = name == "articulation_params" ? 3u : 1u;
    } else if (name == "sf_models" || name == "art_models") {
        stride = 80;
        fields = {"meta", "mass", "invInertiaDiagonal", "inertiaRotation", "damping"};
        integers = 1;
    } else if (name == "sf_accepted" || name == "art_states" || name == "sf_roots" || name == "dyn_roots" ||
               name == "normal_active_roots" || name == "normal_resident_roots") {
        stride = 64;
        fields = {"positionCOM", "rotation", "linearVelocity", "angularVelocity"};
    } else if (name == "sf_links" || name == "art_links") {
        stride = 32;
        fields = {"topology", "identity"};
        integers = 3;
    } else if (name == "sf_trees" || name == "articulation_capture_trees") {
        stride = 48;
        fields = {"range", "owner", "progress"};
        integers = 7;
    } else if (name == "art_trees" || name == "articulation_trees" || name == "normal_dynamics_trees") {
        stride = 48;
        fields = {"range", "rootMode", "progress"};
        integers = 7;
    } else if (name == "joint_stream" || name == "articulation_previous_control") {
        fields = {"value"};
        integers = 1;
    } else if (!scalar && !floatVector && !uintVector)
        throw std::invalid_argument("Unknown reduced numeric resource");
    if (r.set != 0 || r.binding.type != (uniform ? rhi::BindingType::UniformBuffer : rhi::BindingType::StorageBuffer) ||
        (uniform ? v.byteSize : r.elementStride) != stride || (!uniform && r.binding.readOnlyStorage != readOnly))
        throw std::invalid_argument("Reduced numeric resource ABI differs from actual source");
    if (scalar) {
        if (v.kind != ShaderValueKind::Scalar || v.scalar != ShaderScalarKind::UInt32 || v.byteSize != 4)
            throw std::invalid_argument("Reduced scalar is not uint32");
        return;
    }
    if (floatVector || uintVector) {
        if (v.kind != ShaderValueKind::Vector ||
            v.scalar != (floatVector ? ShaderScalarKind::Float32 : ShaderScalarKind::UInt32) || v.rows != 1 ||
            v.columns != 4)
            throw std::invalid_argument("Reduced raw word ABI differs");
        return;
    }
    if (v.kind != ShaderValueKind::Struct || v.fields.size() != fields.size())
        throw std::invalid_argument("Reduced POD field count differs");
    for (uint32_t i = 0; i < fields.size(); ++i) {
        const auto &f = v.fields[i];
        if (f.name != fields[i] || f.offset != 16u * i || f.byteSize != 16 || f.kind != ShaderValueKind::Vector ||
            f.rows != 1 || f.columns != 4 ||
            f.scalar != ((integers & (1u << i)) ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32))
            throw std::invalid_argument("Reduced POD field ABI differs");
    }
}
} // namespace

class JoltGpuArticulationNumerics::Impl
{
  public:
    struct Binding
    {
        std::string name;
        bool uniform;
    };
    struct Program
    {
        std::shared_ptr<rhi::ComputeKernel> kernel;
        std::vector<Binding> bindings;
    };
    struct Execution;
    explicit Impl(rhi::ComputeHost &source, const ShaderTargetCode &step, const ShaderTargetCode &capture,
                  const ShaderTargetCode &dynamics)
        : host(source)
    {
        const ShaderTargetCode *targets[]{&step, &capture, &dynamics};
        for (const auto &d : NumericEntries) {
            const auto &target = *targets[d.source];
            const auto e = std::find_if(target.entries.begin(), target.entries.end(),
                                        [&](const auto &v) { return v.name == d.name; });
            if (e == target.entries.end() || e->stage != rhi::ShaderStage::Compute ||
                e->workgroupSize != std::array<uint32_t, 3>{64, 1, 1} || e->rootConstants)
                throw std::invalid_argument("Reduced numeric artifact lacks actual bounded entry");
            Program p;
            std::set<std::string> names;
            std::vector<rhi::ComputeBufferBinding> bindings;
            for (const auto &r : target.resources) {
                if (std::find(e->bindings.begin(), e->bindings.end(),
                              std::array<uint32_t, 2>{r.set, r.binding.binding}) == e->bindings.end())
                    continue;
                if (!Has(d.resources, r.name) || !names.insert(r.name).second)
                    throw std::invalid_argument("Reduced numeric entry has unexpected resource");
                Validate(r);
                p.bindings.push_back({r.name, r.binding.type == rhi::BindingType::UniformBuffer});
                bindings.push_back({r.binding.binding, r.binding.type, r.binding.readOnlyStorage});
            }
            uint32_t expected = 1;
            for (const char *c = d.resources; *c; ++c)
                if (*c == '|')
                    ++expected;
            if (names.size() != expected || names.size() != e->bindings.size())
                throw std::invalid_argument("Reduced numeric entry resource closure incomplete");
            if (target.target == ShaderCodeTarget::SpirV) {
                if (e->code.empty() || e->code.size() % 4u)
                    throw std::invalid_argument("Reduced SPIR-V extent invalid");
                std::vector<uint32_t> words(e->code.size() / 4);
                std::memcpy(words.data(), e->code.data(), e->code.size());
                p.kernel = std::make_shared<rhi::ComputeKernel>(
                    host, rhi::ShaderModuleDesc::FromSpirV(words.data(), words.size()), e->name, std::move(bindings),
                    0);
            } else if (target.target == ShaderCodeTarget::Wgsl)
                p.kernel = std::make_shared<rhi::ComputeKernel>(
                    host, rhi::ShaderModuleDesc::FromWgsl(e->code.data(), e->code.size()), e->name, std::move(bindings),
                    0);
            else
                throw std::invalid_argument("Reduced numeric artifact target unsupported");
            programs.emplace(d.name, std::move(p));
        }
    }
    bool Allocate(std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes)
    {
        if (!bytes)
            return true;
        if (bytes % 4u)
            throw std::logic_error("Reduced private allocation is not a word range");
        if (bytes > host.device.GetCapabilities().portable.maxStorageBufferBinding || bytes > UINT32_MAX)
            return false; // Hard capacity is a cold lowering refusal, like managed pressure.
        if (buffer && buffer->GetByteSize() >= bytes)
            return true;
        auto value = rhi::ComputeBuffer::TryCreatePrivate(
            host, {bytes / 4u, rhi::ComputeScalarType::UInt32, 1u, rhi::BufferUsageFlags::None});
        if (!value)
            return false;
        buffer = std::move(value);
        return true;
    }
    rhi::ComputeHost host;
    std::map<std::string, Program> programs;
};

struct JoltGpuArticulationNumerics::Impl::Execution : std::enable_shared_from_this<Execution>
{
    struct Recipe
    {
        std::string entry;
        uint32_t first, count, repeat, units, mode;
        bool parallel;
    };
    struct Upload
    {
        std::shared_ptr<rhi::ComputeBuffer> buffer;
        std::vector<uint8_t> bytes;
        uint64_t base = 0;
    };
    struct Bound
    {
        std::shared_ptr<rhi::ComputeBuffer> buffer;
        uint64_t readable = 0;
    };
    std::shared_ptr<Impl> owner;
    std::shared_ptr<JoltGpuArticulationState> state;
    std::shared_ptr<JoltGpuArticulationNumericState> numeric;
    std::map<std::string, Bound> sources;
    std::vector<Recipe> plan;
    std::vector<Upload> uploads;
    size_t upload = 0, cursor = 0;
    uint64_t uploadOffset = 0;
    uint32_t repeat = 0, offset = 0;
    bool initialize = false, refresh = false, configured = false, readSummary = false, declinedRetirement = false;
    float dt = 0;
    std::function<void()> completed;
    std::function<JoltGpuControlWork(bool)> admission;
    std::function<JoltGpuControlWork()> successor;
    std::shared_ptr<rhi::ComputeBuffer> summarySource;
    uint64_t summaryOffset = 0, retirementBytes = 0;
    uint32_t summaryItems = 0;
    void Source(std::string name, std::shared_ptr<rhi::ComputeBuffer> value, uint64_t bytes = UINT64_MAX)
    {
        sources.emplace(std::move(name), Bound{value, value ? std::min(bytes, value->GetByteSize()) : 0});
    }
    void Bind(const JoltGpuStorage *storage, const JoltGpuStepInputs *inputs)
    {
        const auto models = inputs ? inputs->GetCaptured(Resource::BodyModels) : nullptr;
        const auto states = storage ? storage->Get(Resource::BodyStates) : nullptr;
        const auto joints = storage ? storage->Get(Resource::Joints) : nullptr;
        for (const auto *name : {"sf_models", "art_models"})
            Source(name, models);
        for (const auto *name : {"sf_accepted", "art_states"})
            Source(name, states);
        for (const auto *name : {"sf_joints", "art_joints", "joint_stream"})
            Source(name, joints);
        for (const auto *name : {"sf_links", "art_links"})
            Source(name, state->linkBuffer);
        for (const auto *name : {"art_trees", "articulation_trees", "normal_dynamics_trees"})
            Source(name, state->treeBuffer);
        for (const auto *name : {"sf_trees", "articulation_capture_trees"})
            Source(name, numeric->captureTrees);
        for (const auto *name : {"sf_roots", "dyn_roots"})
            Source(name, numeric->stagingRoots);
        for (const auto *name : {"sf_coordinates", "dyn_coordinates"})
            Source(name, numeric->stagingCoordinates);
        for (const auto *name : {"normal_active_roots", "normal_resident_roots"})
            Source(name, numeric->roots);
        for (const auto *name : {"normal_active_coordinates", "normal_resident_coordinates"})
            Source(name, numeric->coordinates);
        for (const auto *name : {"normal_capture_control", "articulation_words"})
            Source(name, state->control, state->header.storage.z);
        Source("art_scratch", state->control);
        Source("articulation_previous_control", numeric->previousControl, numeric->previousPrefixBytes);
        Source("articulation_transitions", numeric->rows, uint64_t(numeric->transitions.size()) * 16u);
        Source("articulation_transition_witness", numeric->witness, uint64_t(numeric->transitions.size()) * 16u);
        Source("articulation_summary", numeric->summary, 16u);
    }
    void Add(std::string entry, uint32_t first, uint32_t count, uint32_t repetitions = 1, uint32_t units = 8,
             bool parallel = false, uint32_t mode = 0)
    {
        if (count && repetitions)
            plan.push_back({std::move(entry), first, count, repetitions, units, mode, parallel});
    }
    void Windows(std::string entry, uint32_t passes = 0)
    {
        for (const auto &w : numeric->windows)
            Add(entry, w.first, w.count, passes ? uint32_t((uint64_t(w.maxLinks) * passes + 7u) / 8u) : 1u);
    }
    bool Allocate()
    {
        const uint64_t trees = state->trees.size(), links = state->links.size(), rows = numeric->transitions.size();
        const uint64_t rowCapacity = std::max(rows, links - trees);
        if (!(owner->Allocate(numeric->roots, trees * 64u) && owner->Allocate(numeric->stagingRoots, trees * 64u) &&
              owner->Allocate(numeric->coordinates, links * 16u) &&
              owner->Allocate(numeric->stagingCoordinates, links * 16u) &&
              owner->Allocate(numeric->captureTrees, trees * 48u) &&
              owner->Allocate(numeric->rows, rowCapacity * 16u) &&
              owner->Allocate(numeric->witness, rowCapacity * 16u) && owner->Allocate(numeric->summary, 16u)))
            return false;
        for (auto &uniform : numeric->uniforms)
            if (!owner->Allocate(uniform, 32u))
                return false;
        return true;
    }
    void CompleteProof(const Recipe &r, const Bound &source, uint32_t binding, uint32_t first, uint32_t count,
                       rhi::ComputeDispatchProof &proof)
    {
        const auto write = [&](uint64_t begin, uint64_t bytes) {
            if (bytes)
                proof.writes.push_back({binding, begin, bytes});
        };
        if (r.entry == "normal_capture_initialize") {
            const bool roots = source.buffer == numeric->roots || source.buffer == numeric->stagingRoots;
            const uint64_t total = roots ? state->trees.size() : state->links.size(), stride = roots ? 64u : 16u;
            if (first < total)
                write(uint64_t(first) * stride, std::min<uint64_t>(count, total - first) * stride);
        } else if (r.entry == "articulation_initialize_numeric")
            write(uint64_t(first) * 4u, uint64_t(count) * 4u);
        else if (r.entry == "articulation_clear_summary")
            write(0, 16);
        else if (r.entry == "articulation_capture_transition")
            write(uint64_t(first) * 16u, uint64_t(count) * 16u);
        // All other conditional/native/numeric writes preserve initialized data.
        // No unconditionally complete claim is made for a rejected tree/row.
    }
    JoltGpuControlWork Next(uint32_t maximum = 8)
    {
        if (!configured) {
            if (!declinedRetirement && !Allocate()) {
                if (initialize) {
                    auto declined = std::move(admission);
                    successor = {};
                    state->numeric.reset();
                    numeric.reset();
                    sources.clear();
                    uploads.clear();
                    return declined(false);
                }
                throw std::logic_error("Accepted reduced resources changed capacity outside a cold boundary");
            }
            if (initialize) {
                if (!numeric->captureMetadata.empty())
                    uploads.push_back({numeric->captureTrees, Bytes(numeric->captureMetadata)});
                if (!numeric->transitions.empty())
                    uploads.push_back({numeric->rows, Bytes(numeric->transitions)});
            } else if (refresh && !numeric->transitions.empty())
                uploads.push_back({numeric->rows, Bytes(numeric->transitions)});
            // Init does not read any live model/state source. Other callers bound
            // these after allocation, below, preserving exact fresh identities.
            configured = true;
        }
        if (upload < uploads.size()) {
            const auto &u = uploads[upload];
            const uint64_t bytes = std::min<uint64_t>(16384u, u.bytes.size() - uploadOffset);
            std::vector<uint8_t> chunk(u.bytes.begin() + uploadOffset, u.bytes.begin() + uploadOffset + bytes);
            auto packet =
                rhi::PrepareComputeBatch(owner->host, {{u.buffer, u.base + uploadOffset, std::move(chunk)}}, {}, {});
            return {packet,
                    [self = shared_from_this(), bytes] {
                        self->uploadOffset += bytes;
                        if (self->uploadOffset == self->uploads[self->upload].bytes.size()) {
                            ++self->upload;
                            self->uploadOffset = 0;
                        }
                        return self->Next();
                    },
                    JoltGpuControlWork::ResumeAfter::Acceptance};
        }
        if (cursor == plan.size()) {
            if (successor) {
                auto next = std::move(successor);
                return next();
            }
            auto done = std::move(completed);
            done();
            return {};
        }
        // Refresh allocation aliases once all managed creations succeeded.
        sources["sf_roots"] = {numeric->stagingRoots, numeric->stagingRoots ? numeric->stagingRoots->GetByteSize() : 0};
        sources["dyn_roots"] = sources["sf_roots"];
        sources["sf_coordinates"] = {numeric->stagingCoordinates,
                                     numeric->stagingCoordinates ? numeric->stagingCoordinates->GetByteSize() : 0};
        sources["dyn_coordinates"] = sources["sf_coordinates"];
        sources["normal_active_roots"] = {numeric->roots, numeric->roots ? numeric->roots->GetByteSize() : 0};
        sources["normal_resident_roots"] = sources["normal_active_roots"];
        sources["normal_active_coordinates"] = {numeric->coordinates,
                                                numeric->coordinates ? numeric->coordinates->GetByteSize() : 0};
        sources["normal_resident_coordinates"] = sources["normal_active_coordinates"];
        sources["sf_trees"] = {numeric->captureTrees, numeric->captureTrees ? numeric->captureTrees->GetByteSize() : 0};
        sources["articulation_capture_trees"] = sources["sf_trees"];
        sources["articulation_transitions"] = {numeric->rows, uint64_t(numeric->transitions.size()) * 16u};
        sources["articulation_transition_witness"] = {numeric->witness, uint64_t(numeric->transitions.size()) * 16u};
        sources["articulation_summary"] = {numeric->summary, 16};
        size_t next = cursor;
        uint32_t nextRepeat = repeat, nextOffset = offset;
        std::vector<rhi::ComputeBufferUpdate> uniforms;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        while (next < plan.size() && dispatches.size() < maximum) {
            const auto &r = plan[next];
            const uint32_t quantum = uint32_t(std::min<uint64_t>(
                2048u, uint64_t(owner->host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0]) * 64u));
            if (!quantum)
                throw std::length_error("Reduced numeric device has no dispatch capacity");
            uint32_t count = r.parallel ? std::min(quantum, r.count - nextOffset) : r.count;
            uint32_t first = r.first + nextOffset;
            auto &p = owner->programs.at(r.entry);
            const bool needsUniform =
                std::any_of(p.bindings.begin(), p.bindings.end(), [](const auto &b) { return b.uniform; });
            const auto uniform = needsUniform ? numeric->uniforms[dispatches.size()] : nullptr;
            if (needsUniform && !uniform)
                throw std::logic_error("Reduced owner lost its cold-admitted uniform pool");
            const UInt4 range{first, count, r.units, 0};
            if (needsUniform) {
                if (r.entry.substr(0, 13) == "articulation_")
                    uniforms.push_back({uniform, 0, Bytes(ArticulationClassifyWork{{first, count, 0, r.mode}, {}})});
                else if (std::any_of(p.bindings.begin(), p.bindings.end(),
                                     [](const auto &b) { return b.name == "sf_work"; }))
                    uniforms.push_back({uniform, 0, Bytes(StateFkWork{range, {3e-5f, 3e-5f, 3e-5f, 3e-5f}})});
                else
                    uniforms.push_back({uniform, 0, Bytes(ArticulationDynamicsWork{range, {dt, 3e-5f, 3e-5f, 3e-5f}})});
            }
            rhi::ComputeDispatchDesc d;
            d.kernel = p.kernel;
            d.groupCountX = (count + 63u) / 64u;
            rhi::ComputeDispatchProof proof;
            for (const auto &b : p.bindings) {
                const Bound source = b.uniform ? Bound{uniform, 32} : sources.at(b.name);
                if (!source.buffer)
                    throw std::logic_error("Reduced numeric work lacks exact retained source");
                const uint32_t binding = uint32_t(d.buffers.size());
                d.buffers.push_back(source.buffer);
                const bool complete =
                    r.entry == "normal_capture_initialize" || r.entry == "articulation_initialize_numeric" ||
                    r.entry == "articulation_clear_summary" ||
                    (r.entry == "articulation_capture_transition" && b.name == "articulation_transition_witness");
                const bool writer = !b.uniform && Writer(r.entry, b.name);
                d.bufferAccesses.push_back(
                    writer ? (complete ? rhi::ComputeBufferAccess::Write : rhi::ComputeBufferAccess::ReadWrite)
                           : rhi::ComputeBufferAccess::Read);
                if (!writer || !complete) {
                    if (declinedRetirement && (b.name == "articulation_retirement" || b.name == "articulation_words")) {
                        proof.reads.push_back({binding, 0, numeric->previousPrefixBytes});
                        proof.reads.push_back({binding, numeric->previousPrefixBytes, retirementBytes});
                    } else
                        proof.reads.push_back({binding, 0, source.readable});
                }
                if (writer && complete)
                    CompleteProof(r, source, binding, first, count, proof);
            }
            d.proof = std::move(proof);
            dispatches.push_back(std::move(d));
            if (r.parallel && nextOffset + count < r.count)
                nextOffset += count;
            else {
                nextOffset = 0;
                if (++nextRepeat == r.repeat) {
                    ++next;
                    nextRepeat = 0;
                }
            }
        }
        if (dispatches.empty())
            return {nullptr, [self = shared_from_this()] { return self->Next(); }};
        const bool final = next == plan.size(), multi = dispatches.size() > 1;
        std::vector<rhi::ComputeBufferRead> reads;
        if (final && readSummary)
            reads.push_back({summarySource ? summarySource : numeric->summary, summaryOffset, 16});
        auto packet =
            rhi::PrepareComputeBatch(owner->host, std::move(uniforms), std::move(dispatches), std::move(reads));
        JoltGpuControlWork work{
            packet,
            [self = shared_from_this(), packet, next, nextRepeat, nextOffset, final] {
                if (final && self->readSummary) {
                    UInt4 summary{};
                    packet->Readback(0)->ReadInto(&summary, sizeof(summary));
                    const auto expected =
                        self->declinedRetirement ? self->summaryItems : uint32_t(self->state->trees.size());
                    if (summary.x || summary.w != expected || summary.z > summary.y || summary.y > summary.w ||
                        summary.y != summary.z)
                        throw std::runtime_error("Reduced articulation commit owner invariant failed");
                }
                self->cursor = next;
                self->repeat = nextRepeat;
                self->offset = nextOffset;
                return self->Next();
            },
            final && readSummary ? JoltGpuControlWork::ResumeAfter::Completion
                                 : JoltGpuControlWork::ResumeAfter::Acceptance};
        if (multi)
            work.prepareMinimum = [self = shared_from_this()] { return self->Next(1); };
        return work;
    }
};

JoltGpuArticulationNumerics::JoltGpuArticulationNumerics(rhi::ComputeHost &h, const ShaderTargetCode &s,
                                                         const ShaderTargetCode &c, const ShaderTargetCode &d)
    : m_impl(std::make_shared<Impl>(h, s, c, d))
{
}

JoltGpuControlWork JoltGpuArticulationNumerics::PrepareInitialize(const JoltGpuStorage &storage,
                                                                  std::shared_ptr<JoltGpuArticulationState> state,
                                                                  std::shared_ptr<JoltGpuArticulationState> previous,
                                                                  std::function<JoltGpuControlWork(bool)> admitted)
{
    auto n = std::make_shared<JoltGpuArticulationNumericState>();
    state->numeric = n;
    n->pendingOrigin = previous != nullptr;
    if (previous && previous->numeric && previous->numeric->pendingOrigin) {
        n->origin = previous->numeric->origin;
        n->previousControl = previous->numeric->previousControl;
        n->previousPrefixBytes = previous->numeric->previousPrefixBytes;
    } else if (previous) {
        n->origin = Origins(*previous);
        n->previousControl = previous->control;
        n->previousPrefixBytes = previous->header.storage.z;
    } else {
        n->origin = Origins(*state);
        n->previousControl = state->control;
        n->previousPrefixBytes = state->header.storage.z;
    }
    MatchRows(storage, *state, *n);
    if (state->trees.empty() && n->transitions.empty()) {
        n->initialized = true;
        n->pendingOrigin = false;
        n->previousControl = state->control;
        n->previousPrefixBytes = state->header.storage.z;
        std::vector<JoltGpuArticulationNumericState::Origin>().swap(n->origin);
        return admitted(true);
    }
    for (uint32_t first = 0; first < state->trees.size(); first += 64u) {
        uint32_t count = std::min<uint32_t>(64u, uint32_t(state->trees.size()) - first), maximum = 0;
        for (uint32_t i = first; i < first + count; ++i)
            maximum = std::max(maximum, state->trees[i].range.y);
        n->windows.push_back({first, count, maximum});
    }
    for (const auto &t : state->trees)
        n->captureMetadata.push_back({{t.range.x, t.range.y, t.range.w, 0}, {t.rootMode.z, t.rootMode.w, 0, 0}, {}});
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = state;
    e->numeric = n;
    e->initialize = true;
    e->Bind(nullptr, nullptr);
    e->Add("articulation_initialize_numeric", state->header.storage.z / 4u,
           (state->header.storage.w - state->header.storage.z) / 4u, 1, 8, true);
    e->Add("normal_capture_initialize", 0, uint32_t(state->links.size()), 1, 8, true);
    e->admission = admitted;
    e->successor = [n, admitted = std::move(admitted)] {
        n->initialized = true;
        return admitted(true);
    };
    return e->Next();
}
JoltGpuControlWork JoltGpuArticulationNumerics::PrepareDeclined(const JoltGpuStorage &storage,
                                                                std::shared_ptr<JoltGpuArticulationState> state,
                                                                std::shared_ptr<JoltGpuArticulationState> previous,
                                                                std::function<void()> done, bool capturedWitness)
{
    if (!previous) {
        done();
        return {};
    }
    auto n = std::make_shared<JoltGpuArticulationNumericState>();
    if (previous->numeric && previous->numeric->pendingOrigin) {
        n->origin = previous->numeric->origin;
        n->previousControl = previous->numeric->previousControl;
        n->previousPrefixBytes = previous->numeric->previousPrefixBytes;
    } else {
        n->origin = Origins(*previous);
        n->previousControl = previous->control;
        n->previousPrefixBytes = previous->header.storage.z;
    }
    if (n->origin.empty()) {
        done();
        return {};
    }
    MatchRows(storage, *state, *n);
    if (n->transitions.empty()) {
        done();
        return {};
    }
    if (capturedWitness) {
        if (!previous->numeric || !previous->numeric->witness ||
            previous->numeric->transitions.size() != n->transitions.size())
            throw std::logic_error("Pressure retirement lost its pre-Setup witness rows");
        // Next refreshes all numeric aliases from this execution's owner.
        // Retain the actual admitted witness there, not only in sources.
        n->witness = previous->numeric->witness;
        for (size_t i = 0; i < n->transitions.size(); ++i) {
            const auto &a = n->transitions[i], &b = previous->numeric->transitions[i];
            if (a.x != b.x || a.y != b.y || a.w != b.w)
                throw std::logic_error("Pressure retirement crosses its captured native row map");
        }
    }
    const uint32_t count = uint32_t(n->transitions.size()), first = n->previousPrefixBytes / 16u;
    const uint64_t bytes = (uint64_t(count) + 2u) * 16u;
    if (n->previousPrefixBytes % 16u || !n->previousControl ||
        n->previousPrefixBytes > n->previousControl->GetByteSize() ||
        bytes > n->previousControl->GetByteSize() - n->previousPrefixBytes)
        throw std::logic_error("Retired reduced generation has no proven dead numeric range");
    // Header + current native rows + compact result, in the OLD dead numeric
    // tail. No new device-local allocation is needed to retire a declined tree.
    std::vector<UInt4> payload;
    payload.reserve(size_t(count) + 2u);
    payload.push_back({first + 1u, count, first + 1u + count, 0u});
    for (const auto &row : n->transitions)
        payload.push_back({row.x, row.y, row.w, 0u});
    payload.push_back({});
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = state;
    e->numeric = n;
    e->configured = true;
    e->declinedRetirement = true;
    e->retirementBytes = bytes;
    e->summaryItems = count;
    e->Bind(&storage, nullptr);
    e->sources["articulation_words"] = {n->previousControl, n->previousPrefixBytes};
    if (capturedWitness)
        e->sources["articulation_transition_witness"] = {previous->numeric->witness, uint64_t(count) * 16u};
    e->Source("articulation_retirement", n->previousControl, n->previousPrefixBytes);
    e->uploads.push_back({n->previousControl, Bytes(payload), n->previousPrefixBytes});
    for (uint32_t offset = 0; offset < count;) {
        const uint32_t size = std::min(2048u, count - offset);
        e->Add(capturedWitness ? "articulation_retire_captured" : "articulation_retire_declined", offset, size);
        e->Add("articulation_advance_retirement", 0, 1);
        offset += size;
    }
    e->Add("articulation_clear_retirement_summary", 0, 1);
    for (uint32_t offset = 0; offset < count;) {
        const uint32_t size = std::min(2048u, count - offset);
        e->Add("articulation_retirement_summary", offset, size);
        e->Add("articulation_advance_retirement", 0, 1);
        offset += size;
    }
    e->summarySource = n->previousControl;
    e->summaryOffset = uint64_t(first + 1u + count) * 16u;
    e->readSummary = true;
    e->completed = std::move(done);
    return e->Next();
}
JoltGpuControlWork JoltGpuArticulationNumerics::PrepareBeforeForces(const JoltGpuStorage &storage,
                                                                    const JoltGpuStepInputs &inputs,
                                                                    std::function<void()> done)
{
    auto s = inputs.GetArticulations();
    if (!s || !s->numeric) {
        done();
        return {};
    }
    auto n = s->numeric;
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = s;
    e->numeric = n;
    if (n->rowsNeedRefresh) {
        n->origin = Origins(*s);
        n->previousControl = s->control;
        n->previousPrefixBytes = s->header.storage.z;
        MatchRows(storage, *s, *n);
        e->refresh = true;
        n->rowsNeedRefresh = false;
    }
    if (s->trees.empty() && n->transitions.empty()) {
        // The previous transition window completed before this factory. No
        // reader can still need these retired row/witness allocations.
        n->rows.reset();
        n->witness.reset();
        n->summary.reset();
        done();
        return {};
    }
    e->dt = inputs.GetParameters().gravityDt.w;
    e->Bind(&storage, &inputs);
    e->Add("articulation_capture_transition", 0, uint32_t(n->transitions.size()), 1, 8, true);
    e->Windows("normal_capture_begin");
    for (const auto &w : n->windows) {
        uint32_t repeat = (w.maxLinks + 7u) / 8u;
        e->Add("normal_capture_advance", w.first, w.count, repeat);
        e->Add("normal_seed_advance", w.first, w.count, repeat);
    }
    e->Windows("normal_dynamics_begin");
    e->Windows("normal_dynamics_preserve", 1);
    e->completed = std::move(done);
    return e->Next();
}
JoltGpuControlWork JoltGpuArticulationNumerics::PrepareCandidate(const JoltGpuStorage &storage,
                                                                 const JoltGpuStepInputs &inputs,
                                                                 std::function<void()> done)
{
    auto s = inputs.GetArticulations();
    if (!s || !s->numeric) {
        done();
        return {};
    }
    auto n = s->numeric;
    if (s->trees.empty() && n->transitions.empty()) {
        done();
        return {};
    }
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = s;
    e->numeric = n;
    e->dt = inputs.GetParameters().gravityDt.w;
    e->Bind(&storage, &inputs);
    e->Add("articulation_clear_summary", 0, 4);
    const bool solve = inputs.GetParameters().counts.w != 0u;
    if (solve) {
        e->Windows("normal_dynamics_solve_begin");
        e->Windows("normal_dynamics_solve", 4);
        e->Windows("normal_reaction_begin");
        e->Windows("normal_reaction_advance", 1);
    }
    e->Add("articulation_mark_candidate", 0, uint32_t(s->trees.size()), 1, 8, true);
    e->Add("articulation_reset_transition", 0, uint32_t(n->transitions.size()), 1, 8, true);
    if (solve) {
        e->Windows("normal_history_begin");
        e->Windows("normal_history_advance", 1);
        e->Windows("normal_velocity_begin");
        e->Windows("normal_velocity_advance", 1);
    }
    e->Add("articulation_admit", 0, uint32_t(s->trees.size()), 1, 8, true);
    e->Add("articulation_numeric_summary", 0, uint32_t(s->trees.size()), 1, 8, true, 0);
    e->readSummary = true;
    e->completed = [n, s, done = std::move(done)] {
        if (n->pendingOrigin)
            n->rowsNeedRefresh = true;
        n->pendingOrigin = false;
        n->previousControl = s->control;
        n->previousPrefixBytes = s->header.storage.z;
        std::vector<JoltGpuArticulationNumericState::Origin>().swap(n->origin);
        done();
    };
    return e->Next();
}
JoltGpuControlWork JoltGpuArticulationNumerics::PreparePose(const JoltGpuStorage &storage,
                                                            const JoltGpuStepInputs &inputs, std::function<void()> done)
{
    auto s = inputs.GetArticulations();
    if (!s || !s->numeric || s->trees.empty()) {
        done();
        return {};
    }
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = s;
    e->numeric = s->numeric;
    e->dt = inputs.GetParameters().gravityDt.w;
    e->Bind(&storage, &inputs);
    e->Add("articulation_clear_summary", 0, 4);
    e->Windows("normal_pose_begin");
    e->Windows("normal_pose_advance", 1);
    e->Windows("normal_promote_begin");
    e->Windows("normal_promote_advance", 1);
    e->Add("articulation_numeric_summary", 0, uint32_t(s->trees.size()), 1, 8, true, 1);
    e->readSummary = true;
    e->completed = std::move(done);
    return e->Next();
}
JoltGpuControlWork JoltGpuArticulationNumerics::PrepareDirty(std::shared_ptr<JoltGpuArticulationState> s,
                                                             std::function<void()> done)
{
    if (!s || s->trees.empty()) {
        done();
        return {};
    }
    auto e = std::make_shared<Impl::Execution>();
    e->owner = m_impl;
    e->state = s;
    e->numeric = s->numeric;
    e->Bind(nullptr, nullptr);
    e->Add("articulation_dirty", 0, uint32_t(s->trees.size()), 1, 8, true);
    e->completed = std::move(done);
    return e->Next();
}
} // namespace infernux
