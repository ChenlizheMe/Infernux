#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/scene/ComponentDataStore.h>
#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>
#include <function/scene/TransformECSStore.h>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "BindingRegistration.h"

namespace py = pybind11;
using namespace infernux;

namespace
{

using StepPermission = std::optional<std::pair<uint64_t, uint64_t>>;

bool PythonTruth(py::handle value)
{
    const int truth = PyObject_IsTrue(value.ptr());
    if (truth < 0)
        throw py::error_already_set();
    return truth != 0;
}

class ComponentExecutionState
{
  public:
    enum class Status
    {
        Active,
        Paused,
        Stepping,
        Closed
    };

    ComponentExecutionState(py::handle owner, SceneManager &manager, uint64_t epoch, bool playing)
        : m_owner(owner), m_manager(manager), m_epoch(epoch), m_playing(playing)
    {
    }

    Status Read(py::handle scope, const StepPermission &permission) const
    {
        if (PythonTruth(scope.attr("_closed")))
            return Status::Closed;
        const py::object owner = m_owner();
        if (owner.is_none() || PythonTruth(owner.attr("_is_destroyed")))
            return Status::Closed;
        const auto fieldIsTrue = [&owner](const char *key) {
            const py::object fields = owner.attr("__dict__");
            if (!PyDict_CheckExact(fields.ptr()))
                return PythonTruth(fields.attr("get")(key));
            const py::str name(key);
            PyObject *value = PyDict_GetItemWithError(fields.ptr(), name.ptr());
            if (!value && PyErr_Occurred())
                throw py::error_already_set();
            return value && PythonTruth(py::reinterpret_borrow<py::object>(value));
        };
        if (fieldIsTrue("_is_destroying") || fieldIsTrue("_play_domain_destroy_finalized") ||
            m_manager.GetSimulationEpoch() != m_epoch || m_manager.IsPlaying() != m_playing)
            return Status::Closed;
        if (!PythonTruth(owner.attr("_runtime_active_in_hierarchy")) || !PythonTruth(owner.attr("_enabled")))
            return Status::Paused;
        if (m_playing && m_manager.IsPaused()) {
            if (!m_manager.IsStepping() ||
                permission != std::pair(m_manager.GetSimulationEpoch(), m_manager.GetStepPermissionIndex()))
                return Status::Paused;
            return PythonTruth(scope.attr("_paused")) ? Status::Paused : Status::Stepping;
        }
        return PythonTruth(scope.attr("_paused")) ? Status::Paused : Status::Active;
    }

    const char *State(py::handle scope, const StepPermission &permission) const
    {
        switch (Read(scope, permission)) {
        case Status::Active:
            return "active";
        case Status::Paused:
            return "paused";
        case Status::Stepping:
            return "stepping";
        case Status::Closed:
            return "closed";
        }
        return "closed";
    }

  private:
    py::weakref m_owner;
    const SceneManager &m_manager;
    uint64_t m_epoch;
    bool m_playing;
};

struct CpuTransformPose
{
    std::array<float, 3> position;
    std::array<float, 4> rotation;
    std::array<float, 3> scale;

    static CpuTransformPose Read(Transform &target)
    {
        const auto position = target.GetWorldPosition();
        const auto rotation = target.GetWorldRotation();
        const auto scale = target.GetLocalScale();
        return {{position.x, position.y, position.z},
                {rotation.x, rotation.y, rotation.z, rotation.w},
                {scale.x, scale.y, scale.z}};
    }

    static CpuTransformPose FromPython(py::handle pose)
    {
        return {pose.attr("position").cast<std::array<float, 3>>(), pose.attr("rotation").cast<std::array<float, 4>>(),
                pose.attr("scale").cast<std::array<float, 3>>()};
    }

    static CpuTransformPose Sample(const py::array &array)
    {
        std::array<float, 12> values;
        std::memcpy(values.data(), array.data(), sizeof(values));
        return {{values[0], values[1], values[2]},
                {values[4], values[5], values[6], values[7]},
                {values[8], values[9], values[10]}};
    }

    py::object ToPython(const py::object &factory) const
    {
        return factory(py::make_tuple(position[0], position[1], position[2]),
                       py::make_tuple(rotation[0], rotation[1], rotation[2], rotation[3]),
                       py::make_tuple(scale[0], scale[1], scale[2]));
    }

    void Publish(Transform &target) const
    {
        target.SetWorldPosition(position[0], position[1], position[2]);
        target.SetWorldRotation(glm::quat(rotation[3], rotation[0], rotation[1], rotation[2]));
        target.SetLocalScale(scale[0], scale[1], scale[2]);
    }

    bool ChangedFrom(const CpuTransformPose &other) const
    {
        const auto close = [](double a, double b) {
            return a == b || (std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 1.0e-5);
        };
        return !std::equal(position.begin(), position.end(), other.position.begin(), close) ||
               !std::equal(rotation.begin(), rotation.end(), other.rotation.begin(), close) ||
               !std::equal(scale.begin(), scale.end(), other.scale.begin(), close);
    }
};

std::optional<py::array> CpuPoseArray(py::handle buffer)
{
    const py::object storage = buffer.attr("_array");
    if (!py::isinstance<py::array>(storage))
        return std::nullopt;
    auto array = py::reinterpret_borrow<py::array>(storage);
    if (array.size() != 12 || !array.dtype().is(py::dtype::of<float>()) || !(array.flags() & py::array::c_style))
        return std::nullopt;
    return array;
}

// Derived once from the current open builder and prepared FIFO for each host
// used in a native span. The caller ends that span at every Python action.
class PendingTransformWriters
{
  public:
    PendingTransformWriters(const py::dict &streams, const py::function &openWrites)
        : m_streams(streams), m_openWrites(openWrites)
    {
    }

    bool Contains(py::handle wrapper, const rhi::ComputeBuffer &buffer)
    {
        // Same ordered compute lane identity as the existing Python stream.
        const auto identity = reinterpret_cast<uintptr_t>(&buffer.GetHost().queue);
        auto [entry, fresh] = m_writers.try_emplace(identity);
        auto &writers = entry->second;
        if (fresh) {
            const py::object stream = m_streams.attr("get")(identity);
            if (!stream.is_none()) {
                for (const py::handle open : m_openWrites(stream))
                    if (!open.is_none())
                        writers.open.insert(open.ptr());
                for (const py::handle packet : stream["pending"]) {
                    const py::object native = packet.attr("native");
                    if (!native.is_none())
                        native.cast<const rhi::PreparedComputeBatch &>().AppendPendingWriterStates(writers.prepared);
                }
            }
        }
        // Open views deliberately share their Python native wrapper. Sealing
        // records the mutable state, which survives COW allocation changes.
        return writers.open.find(wrapper.ptr()) != writers.open.end() ||
               writers.prepared.find(buffer.GetWriterStateIdentity()) != writers.prepared.end();
    }

  private:
    struct HostWriters
    {
        std::unordered_set<PyObject *> open;
        rhi::PreparedComputeBatch::WriterStateSet prepared;
    };
    const py::dict &m_streams;
    const py::function &m_openWrites;
    std::unordered_map<uintptr_t, HostWriters> m_writers;
};

class TransformExchange
{
  public:
    TransformExchange(ObjectHandle target, bool cpu) : m_target(target), m_cpu(cpu)
    {
    }

    void AdoptPythonState(py::handle binding)
    {
        const py::object pending = binding.attr("_pending_cpu_pose");
        if (!pending.is_none() && !CpuPoseArray(binding.attr("_pose"))) {
            binding.attr("_native_transform_exchange") = py::none();
            return;
        }
        m_published = CpuTransformPose::FromPython(binding.attr("_published_pose"));
        m_pending = pending.is_none() ? std::nullopt : std::optional(CpuTransformPose::FromPython(pending));
        ClearPythonPoses(binding);
    }

    // True means Python code ran and the caller must refresh the context-local Step permit.
    bool Advance(py::handle binding, const StepPermission &permission, const py::object &factory,
                 PendingTransformWriters &writers)
    {
        using Status = ComponentExecutionState::Status;
        const py::object scope = binding.attr("_execution_scope");
        const py::object state = scope.attr("_component_state");
        const auto &componentScope = state.cast<const ComponentExecutionState &>();
        if (componentScope.Read(scope, permission) == Status::Closed) {
            binding.attr("close")();
            return true;
        }
        const py::object owner = binding.attr("_owner")();
        const py::object buffer = binding.attr("_pose");
        auto *object = owner.is_none() || PythonTruth(py::getattr(owner, "_is_destroyed", py::bool_(false))) ||
                               PythonTruth(buffer.attr("_closed"))
                           ? nullptr
                           : SceneManager::Instance().FindRuntimeObjectByID(m_target.id);
        if (!object || object->GetHandle() != m_target) {
            binding.attr("close")();
            return true;
        }
        auto &transform = *object->GetTransform();
        const auto current = CpuTransformPose::Read(transform);
        const bool paused = componentScope.Read(scope, permission) == Status::Paused;
        if (!m_cpu) {
            const py::object published = binding.attr("_published_pose");
            if (!published.is_none() && current.ChangedFrom(CpuTransformPose::FromPython(published)))
                return Poll(binding);
            if (paused)
                return false;
            const py::object observed = binding.attr("_readback_source");
            const py::object readback = binding.attr("_readback");
            if (!readback.is_none()) {
                const py::object task = readback.attr("_task");
                const py::object source = readback.attr("_source_write_submission");
                // The binding must observe the accepted source before the batch can wait on it.
                if (!task.is_none() && !observed.is_none() && source.is(observed)) {
                    auto &native = task.cast<rhi::ComputeReadback &>();
                    if (native.GetSubmission().IsValid() && !native.IsCancelled() && !native.IsComplete())
                        return false;
                }
                return Poll(binding);
            }
            const py::object wrapper = buffer.attr("_native");
            const auto &native = wrapper.cast<const rhi::ComputeBuffer &>();
            const auto ticket = native.GetLastWriteSubmission();
            using WriteIdentity = std::tuple<rhi::DeviceId, uint32_t, rhi::SubmissionSerial>;
            if (observed.is_none() ||
                observed.cast<WriteIdentity>() !=
                    WriteIdentity(ticket.device, static_cast<uint32_t>(ticket.queue), ticket.serial) ||
                writers.Contains(wrapper, native))
                return Poll(binding);
            return false;
        }
        if (m_published && current.ChangedFrom(*m_published)) {
            binding.attr("_apply_native_cpu_authoring")(m_published->ToPython(factory), current.ToPython(factory),
                                                        paused);
            return true;
        }
        if (paused)
            return false;

        const auto array = CpuPoseArray(buffer);
        if (!array) {
            binding.attr("_published_pose") = m_published ? m_published->ToPython(factory) : py::none();
            binding.attr("_pending_cpu_pose") = m_pending ? m_pending->ToPython(factory) : py::none();
            binding.attr("_native_transform_exchange") = py::none();
            return Poll(binding);
        }
        if (m_pending) {
            try {
                m_pending->Publish(transform);
            } catch (const std::runtime_error &) {
                binding.attr("close")();
                throw;
            }
            m_published = m_pending;
        }
        m_pending = CpuTransformPose::Sample(*array);
        return false;
    }

  private:
    static bool Poll(py::handle binding)
    {
        binding.attr("_poll")();
        return true;
    }

    static void ClearPythonPoses(py::handle binding)
    {
        binding.attr("_published_pose") = py::none();
        binding.attr("_pending_cpu_pose") = py::none();
    }

    ObjectHandle m_target;
    bool m_cpu;
    std::optional<CpuTransformPose> m_published;
    std::optional<CpuTransformPose> m_pending;
};

size_t AdvanceTransformBindings(const py::tuple &bindings, size_t first, const py::object &factory,
                                const StepPermission &permission, const py::dict &streams,
                                const py::function &openWrites)
{
    PendingTransformWriters writers(streams, openWrites);
    for (size_t index = first; index < bindings.size(); ++index) {
        const py::handle binding = bindings[index];
        const py::object exchange = binding.attr("_native_transform_exchange");
        if (exchange.is_none() || !binding.attr("_initial_pose").is_none())
            return index;
        if (exchange.cast<TransformExchange &>().Advance(binding, permission, factory, writers))
            return index + 1;
    }
    return bindings.size();
}

} // namespace

// ── helpers ──────────────────────────────────────────────────────────────

/// Extract a C-contiguous vector of Transform* from a Python list.
static std::vector<Transform *> ExtractTransforms(const py::list &pyList)
{
    const size_t n = pyList.size();
    std::vector<Transform *> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        auto *transform = pyList[i].cast<Transform *>();
        if (!transform)
            throw py::type_error("targets[" + std::to_string(i) + "] must be a Transform, not None");
        out.push_back(transform);
    }
    return out;
}

static void ValidateTransformDataShape(const py::array &data, size_t rows, py::ssize_t columns)
{
    if (data.ndim() != 2 || data.shape(1) != columns)
        throw py::value_error("data must have shape (N, " + std::to_string(columns) + ")");
    if (static_cast<size_t>(data.shape(0)) < rows)
        throw py::value_error("data array has fewer rows than targets");
}

// ── TransformBatchHandle ─────────────────────────────────────────────────

struct TransformBatchHandle
{
    enum class Mode : uint8_t
    {
        Strict,
        Compact,
    };

    std::vector<TransformECSStore::Handle> handles;
    std::vector<uint64_t> worldIds;
    Mode mode = Mode::Strict;
    mutable std::vector<Transform *> resolved;
    mutable std::vector<size_t> resolvedIndices;
    mutable std::vector<uint8_t> validMask;
    mutable std::vector<float> valueScratch;
    mutable uint64_t resolvedStructuralVersion = UINT64_MAX;

    explicit TransformBatchHandle(std::vector<Transform *> transforms, Mode validationMode = Mode::Strict)
        : mode(validationMode)
    {
        handles.reserve(transforms.size());
        worldIds.reserve(transforms.size());
        resolved.reserve(transforms.size());
        resolvedIndices.reserve(transforms.size());
        validMask.resize(transforms.size());
        for (const Transform *transform : transforms) {
            handles.push_back(transform->GetECSHandle());
            worldIds.push_back(transform->GetHandle().worldId);
        }
    }

    explicit TransformBatchHandle(const py::list &pyList, Mode validationMode = Mode::Strict)
        : TransformBatchHandle(ExtractTransforms(pyList), validationMode)
    {
    }

    [[nodiscard]] size_t size() const
    {
        return handles.size();
    }

    [[nodiscard]] bool IsCompact() const noexcept
    {
        return mode == Mode::Compact;
    }

    Transform *const *Resolve() const
    {
        auto &store = TransformECSStore::Instance();
        const uint64_t structuralVersion = store.GetStructuralVersion();
        if (resolvedStructuralVersion == structuralVersion)
            return resolved.data();

        resolved.clear();
        resolvedIndices.clear();
        std::fill(validMask.begin(), validMask.end(), uint8_t{0});
        for (size_t i = 0; i < handles.size(); ++i) {
            Transform *owner = store.IsValid(handles[i]) ? store.GetOwner(handles[i]) : nullptr;
            const bool valid = owner && owner->GetHandle().worldId == worldIds[i];
            if (!valid) {
                if (mode == Mode::Strict)
                    throw std::runtime_error("TransformBatchHandle contains a stale transform");
                continue;
            }
            validMask[i] = 1;
            resolved.push_back(owner);
            resolvedIndices.push_back(i);
        }
        resolvedStructuralVersion = structuralVersion;
        return resolved.data();
    }
};

// ── Transform batch read/write ───────────────────────────────────────────

using GatherTransformFn = void (TransformECSStore::*)(Transform *const *, float *, size_t) const;
using ScatterTransformFn = void (TransformECSStore::*)(Transform *const *, const float *, size_t);

struct TransformFieldOps
{
    GatherTransformFn gather;
    ScatterTransformFn scatter;
    py::ssize_t width;
};

static const std::unordered_map<std::string, TransformFieldOps> kTransformFields = {
    {"local_position", {&TransformECSStore::GatherLocalPositions, &TransformECSStore::ScatterLocalPositions, 3}},
    {"local_scale", {&TransformECSStore::GatherLocalScales, &TransformECSStore::ScatterLocalScales, 3}},
    {"local_euler_angles",
     {&TransformECSStore::GatherLocalEulerAngles, &TransformECSStore::ScatterLocalEulerAngles, 3}},
    {"position", {&TransformECSStore::GatherWorldPositions, &TransformECSStore::ScatterWorldPositions, 3}},
    {"euler_angles", {&TransformECSStore::GatherWorldEulerAngles, &TransformECSStore::ScatterWorldEulerAngles, 3}},
    {"local_rotation", {&TransformECSStore::GatherLocalRotations, &TransformECSStore::ScatterLocalRotations, 4}},
    {"rotation", {&TransformECSStore::GatherWorldRotations, &TransformECSStore::ScatterWorldRotations, 4}},
};

static const TransformFieldOps &TransformField(const std::string &name)
{
    const auto it = kTransformFields.find(name);
    if (it == kTransformFields.end())
        throw py::value_error("Unknown Transform property: '" + name + "'");
    return it->second;
}

static py::array_t<float> GatherTransformField(Transform *const *targets, size_t count, const TransformFieldOps &field)
{
    auto result = py::array_t<float>({static_cast<py::ssize_t>(count), field.width});
    float *output = result.mutable_data();
    {
        py::gil_scoped_release release;
        (TransformECSStore::Instance().*field.gather)(targets, output, count);
    }
    return result;
}

static void SnapshotTransformFields(const TransformBatchHandle &handle, py::iterable fields)
{
    if (handle.IsCompact())
        throw py::value_error("Transform source snapshots require a strict batch");
    struct Field
    {
        GatherTransformFn gather;
        py::array output;
    };
    std::vector<Field> columns;
    for (const auto &value : fields) {
        const auto field = py::cast<py::tuple>(value);
        const auto &ops = TransformField(field[1].cast<std::string>());
        auto output = field[3].cast<py::array>();
        if (!output.dtype().is(py::dtype::of<float>()) || !(output.flags() & py::array::c_style) ||
            output.ndim() != 2 || output.shape(0) != static_cast<py::ssize_t>(handle.size()) ||
            output.shape(1) != ops.width || !output.writeable())
            throw py::value_error("transform snapshot destination must be a writable contiguous float32 field matrix");
        columns.push_back({ops.gather, std::move(output)});
    }
    Transform *const *targets = handle.Resolve();
    // One owner-thread sample writes directly into the request's private inputs.
    // Do not release the GIL between fields or allocate a second value batch.
    auto &store = TransformECSStore::Instance();
    for (auto &column : columns)
        (store.*column.gather)(targets, static_cast<float *>(column.output.mutable_data()), handle.size());
}

static py::object TransformBatchRead(const py::list &targets, const std::string &prop)
{
    const auto &field = TransformField(prop);
    auto transforms = ExtractTransforms(targets);
    return GatherTransformField(transforms.data(), transforms.size(), field);
}

static void TransformBatchWrite(const py::list &targets, py::array data, const std::string &prop)
{
    const auto &field = TransformField(prop);
    auto fdata = py::array_t<float, py::array::c_style>::ensure(data);
    if (!fdata)
        throw py::type_error("data must be convertible to float32 array");
    auto transforms = ExtractTransforms(targets);
    ValidateTransformDataShape(fdata, transforms.size(), field.width);
    const float *input = fdata.data();
    py::gil_scoped_release release;
    (TransformECSStore::Instance().*field.scatter)(transforms.data(), input, transforms.size());
}

// ── Handle-based batch read/write (avoids repeated ExtractTransforms) ──

static py::array_t<bool> BuildBatchMask(const TransformBatchHandle &handle)
{
    auto mask = py::array_t<bool>(static_cast<py::ssize_t>(handle.validMask.size()));
    auto output = mask.mutable_unchecked<1>();
    for (size_t i = 0; i < handle.validMask.size(); ++i)
        output(static_cast<py::ssize_t>(i)) = handle.validMask[i] != 0;
    return mask;
}

static py::object HandleBatchRead(const TransformBatchHandle &handle, const std::string &prop)
{
    const auto &field = TransformField(prop);
    Transform *const *targets = handle.Resolve();
    auto result = GatherTransformField(targets, handle.resolved.size(), field);
    if (handle.IsCompact())
        return py::make_tuple(std::move(result), BuildBatchMask(handle));
    return std::move(result);
}

static py::object HandleBatchWrite(const TransformBatchHandle &handle, py::array data, const std::string &prop)
{
    const auto &field = TransformField(prop);
    auto fdata = py::array_t<float, py::array::c_style>::ensure(data);
    if (!fdata)
        throw py::type_error("data must be convertible to float32 array");
    Transform *const *targets = handle.Resolve();
    const size_t count = handle.resolved.size();
    ValidateTransformDataShape(fdata, handle.IsCompact() ? handle.size() : count, field.width);
    const float *input = fdata.data();
    if (handle.IsCompact()) {
        handle.valueScratch.resize(count * field.width);
        for (size_t i = 0; i < count; ++i)
            std::copy_n(input + handle.resolvedIndices[i] * field.width, field.width,
                        handle.valueScratch.data() + i * field.width);
        input = handle.valueScratch.data();
    }
    {
        py::gil_scoped_release release;
        (TransformECSStore::Instance().*field.scatter)(targets, input, count);
    }
    if (handle.IsCompact())
        return BuildBatchMask(handle);
    return py::none();
}

// ── ComponentDataStore bindings ───────────────────────────────────────────

static ComponentDataStore::DataType CDS_ParseDataType(int typeCode)
{
    if (typeCode < static_cast<int>(ComponentDataStore::DataType::Float64) ||
        typeCode > static_cast<int>(ComponentDataStore::DataType::Vec4)) {
        throw py::value_error("Invalid ComponentDataStore type code");
    }
    return static_cast<ComponentDataStore::DataType>(typeCode);
}

static uint32_t CDS_RegisterClass(const std::string &name)
{
    return ComponentDataStore::Instance().RegisterClass(name);
}

static uint32_t CDS_RegisterField(uint32_t classId, const std::string &name, int typeCode)
{
    return ComponentDataStore::Instance().RegisterField(classId, name, CDS_ParseDataType(typeCode));
}

static py::object CDS_OptionalPreparedClass(ComponentDataStore::PreparedClassId preparedClassId)
{
    if (preparedClassId == ComponentDataStore::InvalidPreparedClassId)
        return py::none();
    return py::cast(preparedClassId);
}

static py::object CDS_OptionalPublishedClass(uint32_t classId)
{
    if (classId == UINT32_MAX)
        return py::none();
    return py::cast(classId);
}

static ComponentDataStore::SlotHandle CDS_ParseHandle(py::handle value)
{
    if (!py::isinstance<py::tuple>(value) && !py::isinstance<py::list>(value))
        throw py::type_error("CDS slot handle must be a two-item (index, generation) sequence");
    const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    if (sequence.size() != 2)
        throw py::value_error("CDS slot handle must contain exactly index and generation");
    return {sequence[0].cast<uint32_t>(), sequence[1].cast<uint32_t>()};
}

static py::tuple CDS_AllocSlot(uint32_t classId)
{
    const auto handle = ComponentDataStore::Instance().AllocateSlot(classId);
    return py::make_tuple(handle.index, handle.generation);
}

static void CDS_FreeSlot(uint32_t classId, py::handle handle)
{
    ComponentDataStore::Instance().ReleaseSlot(classId, CDS_ParseHandle(handle));
}

static bool CDS_IsAlive(uint32_t classId, py::handle handle)
{
    return ComponentDataStore::Instance().IsAlive(classId, CDS_ParseHandle(handle));
}

static void CDS_ReserveClass(uint32_t classId, size_t capacity)
{
    ComponentDataStore::Instance().ReserveClass(classId, capacity);
}

static py::tuple CDS_SchemaAllocSlot(ComponentDataStore::SchemaTransactionId transactionId,
                                     ComponentDataStore::PreparedClassId preparedClassId)
{
    const auto handle = ComponentDataStore::Instance().AllocatePreparedSlot(transactionId, preparedClassId);
    return py::make_tuple(handle.index, handle.generation);
}

static py::object CDS_SchemaGet(ComponentDataStore::SchemaTransactionId transactionId,
                                ComponentDataStore::PreparedClassId preparedClassId, uint32_t fieldId, py::handle slot,
                                int typeCode)
{
    auto &store = ComponentDataStore::Instance();
    const auto type = CDS_ParseDataType(typeCode);
    const auto handle = CDS_ParseHandle(slot);
    switch (type) {
    case ComponentDataStore::DataType::Float64:
        return py::cast(store.GetPreparedFloat(transactionId, preparedClassId, fieldId, handle));
    case ComponentDataStore::DataType::Int64:
        return py::cast(store.GetPreparedInt(transactionId, preparedClassId, fieldId, handle));
    case ComponentDataStore::DataType::Bool:
        return py::cast(store.GetPreparedBool(transactionId, preparedClassId, fieldId, handle));
    case ComponentDataStore::DataType::Vec2: {
        float value[2];
        store.GetPreparedVec2(transactionId, preparedClassId, fieldId, handle, value);
        return py::make_tuple(value[0], value[1]);
    }
    case ComponentDataStore::DataType::Vec3: {
        float value[3];
        store.GetPreparedVec3(transactionId, preparedClassId, fieldId, handle, value);
        return py::make_tuple(value[0], value[1], value[2]);
    }
    case ComponentDataStore::DataType::Vec4: {
        float value[4];
        store.GetPreparedVec4(transactionId, preparedClassId, fieldId, handle, value);
        return py::make_tuple(value[0], value[1], value[2], value[3]);
    }
    }
    throw py::value_error("Invalid ComponentDataStore type code");
}

static void CDS_SchemaSet(ComponentDataStore::SchemaTransactionId transactionId,
                          ComponentDataStore::PreparedClassId preparedClassId, uint32_t fieldId, py::handle slot,
                          int typeCode, py::object value)
{
    auto &store = ComponentDataStore::Instance();
    const auto type = CDS_ParseDataType(typeCode);
    const auto handle = CDS_ParseHandle(slot);
    switch (type) {
    case ComponentDataStore::DataType::Float64:
        store.SetPreparedFloat(transactionId, preparedClassId, fieldId, handle, value.cast<double>());
        return;
    case ComponentDataStore::DataType::Int64:
        store.SetPreparedInt(transactionId, preparedClassId, fieldId, handle, value.cast<int64_t>());
        return;
    case ComponentDataStore::DataType::Bool:
        store.SetPreparedBool(transactionId, preparedClassId, fieldId, handle, value.cast<bool>());
        return;
    case ComponentDataStore::DataType::Vec2: {
        float data[2];
        if (py::hasattr(value, "x")) {
            data[0] = value.attr("x").cast<float>();
            data[1] = value.attr("y").cast<float>();
        } else {
            const auto sequence = value.cast<py::sequence>();
            if (sequence.size() != 2)
                throw py::value_error("CDS Vec2 value must contain two elements");
            data[0] = sequence[0].cast<float>();
            data[1] = sequence[1].cast<float>();
        }
        store.SetPreparedVec2(transactionId, preparedClassId, fieldId, handle, data);
        return;
    }
    case ComponentDataStore::DataType::Vec3: {
        float data[3];
        if (py::hasattr(value, "x")) {
            data[0] = value.attr("x").cast<float>();
            data[1] = value.attr("y").cast<float>();
            data[2] = value.attr("z").cast<float>();
        } else {
            const auto sequence = value.cast<py::sequence>();
            if (sequence.size() != 3)
                throw py::value_error("CDS Vec3 value must contain three elements");
            data[0] = sequence[0].cast<float>();
            data[1] = sequence[1].cast<float>();
            data[2] = sequence[2].cast<float>();
        }
        store.SetPreparedVec3(transactionId, preparedClassId, fieldId, handle, data);
        return;
    }
    case ComponentDataStore::DataType::Vec4: {
        float data[4];
        if (py::hasattr(value, "x")) {
            data[0] = value.attr("x").cast<float>();
            data[1] = value.attr("y").cast<float>();
            data[2] = value.attr("z").cast<float>();
            data[3] = value.attr("w").cast<float>();
        } else {
            const auto sequence = value.cast<py::sequence>();
            if (sequence.size() != 4)
                throw py::value_error("CDS Vec4 value must contain four elements");
            data[0] = sequence[0].cast<float>();
            data[1] = sequence[1].cast<float>();
            data[2] = sequence[2].cast<float>();
            data[3] = sequence[3].cast<float>();
        }
        store.SetPreparedVec4(transactionId, preparedClassId, fieldId, handle, data);
        return;
    }
    }
    throw py::value_error("Invalid ComponentDataStore type code");
}

static py::tuple CDS_SchemaMigrateSlot(ComponentDataStore::SchemaTransactionId transactionId,
                                       ComponentDataStore::PreparedClassId preparedClassId, uint32_t sourceClassId,
                                       py::handle sourceSlot,
                                       const std::vector<std::pair<uint32_t, uint32_t>> &fieldMap)
{
    std::vector<ComponentDataStore::FieldMigration> migrations;
    migrations.reserve(fieldMap.size());
    for (const auto &[sourceFieldId, destinationFieldId] : fieldMap)
        migrations.push_back({sourceFieldId, destinationFieldId});
    const auto handle = ComponentDataStore::Instance().MigrateSlotToPrepared(transactionId, preparedClassId,
                                                                             sourceClassId, CDS_ParseHandle(sourceSlot),
                                                                             migrations.data(), migrations.size());
    return py::make_tuple(handle.index, handle.generation);
}

static std::vector<ComponentDataStore::SlotHandle>
CDS_ParseHandles(const py::array_t<uint32_t, py::array::c_style> &handles)
{
    if (handles.ndim() != 2 || handles.shape(1) != 2)
        throw py::value_error("CDS handles must have shape (N, 2)");
    const auto input = handles.unchecked<2>();
    std::vector<ComponentDataStore::SlotHandle> parsed(static_cast<size_t>(input.shape(0)));
    for (py::ssize_t i = 0; i < input.shape(0); ++i)
        parsed[static_cast<size_t>(i)] = {input(i, 0), input(i, 1)};
    return parsed;
}

// ── per-element accessors ────────────────────────────────────────────────

static py::object CDS_Get(uint32_t classId, uint32_t fieldId, py::handle slot, int typeCode)
{
    auto &store = ComponentDataStore::Instance();
    auto type = CDS_ParseDataType(typeCode);
    const auto handle = CDS_ParseHandle(slot);
    switch (type) {
    case ComponentDataStore::DataType::Float64:
        return py::cast(store.GetFloat(classId, fieldId, handle));
    case ComponentDataStore::DataType::Int64:
        return py::cast(store.GetInt(classId, fieldId, handle));
    case ComponentDataStore::DataType::Bool:
        return py::cast(store.GetBool(classId, fieldId, handle));
    case ComponentDataStore::DataType::Vec2: {
        float v[2];
        store.GetVec2(classId, fieldId, handle, v);
        return py::make_tuple(v[0], v[1]);
    }
    case ComponentDataStore::DataType::Vec3: {
        float v[3];
        store.GetVec3(classId, fieldId, handle, v);
        return py::make_tuple(v[0], v[1], v[2]);
    }
    case ComponentDataStore::DataType::Vec4: {
        float v[4];
        store.GetVec4(classId, fieldId, handle, v);
        return py::make_tuple(v[0], v[1], v[2], v[3]);
    }
    }
    throw py::value_error("Invalid ComponentDataStore type code");
}

static void CDS_Set(uint32_t classId, uint32_t fieldId, py::handle slot, int typeCode, py::object value)
{
    auto &store = ComponentDataStore::Instance();
    auto type = CDS_ParseDataType(typeCode);
    const auto handle = CDS_ParseHandle(slot);
    switch (type) {
    case ComponentDataStore::DataType::Float64:
        store.SetFloat(classId, fieldId, handle, value.cast<double>());
        break;
    case ComponentDataStore::DataType::Int64:
        store.SetInt(classId, fieldId, handle, value.cast<int64_t>());
        break;
    case ComponentDataStore::DataType::Bool:
        store.SetBool(classId, fieldId, handle, value.cast<bool>());
        break;
    case ComponentDataStore::DataType::Vec2: {
        // Accept anything with .x, .y (Vector2, tuple)
        float v[2];
        if (py::hasattr(value, "x")) {
            v[0] = value.attr("x").cast<float>();
            v[1] = value.attr("y").cast<float>();
        } else {
            auto t = value.cast<py::tuple>();
            v[0] = t[0].cast<float>();
            v[1] = t[1].cast<float>();
        }
        store.SetVec2(classId, fieldId, handle, v);
        break;
    }
    case ComponentDataStore::DataType::Vec3: {
        float v[3];
        if (py::hasattr(value, "x")) {
            v[0] = value.attr("x").cast<float>();
            v[1] = value.attr("y").cast<float>();
            v[2] = value.attr("z").cast<float>();
        } else {
            auto t = value.cast<py::tuple>();
            v[0] = t[0].cast<float>();
            v[1] = t[1].cast<float>();
            v[2] = t[2].cast<float>();
        }
        store.SetVec3(classId, fieldId, handle, v);
        break;
    }
    case ComponentDataStore::DataType::Vec4: {
        float v[4];
        if (py::hasattr(value, "x")) {
            v[0] = value.attr("x").cast<float>();
            v[1] = value.attr("y").cast<float>();
            v[2] = value.attr("z").cast<float>();
            v[3] = value.attr("w").cast<float>();
        } else {
            auto t = value.cast<py::tuple>();
            v[0] = t[0].cast<float>();
            v[1] = t[1].cast<float>();
            v[2] = t[2].cast<float>();
            v[3] = t[3].cast<float>();
        }
        store.SetVec4(classId, fieldId, handle, v);
        break;
    }
    }
}

// ── batch gather/scatter for ComponentDataStore ──────────────────────────

static py::array CDS_BatchGather(uint32_t classId, uint32_t fieldId, int typeCode,
                                 py::array_t<uint32_t, py::array::c_style> slots)
{
    auto &store = ComponentDataStore::Instance();
    auto type = CDS_ParseDataType(typeCode);
    const auto handles = CDS_ParseHandles(slots);
    const size_t n = handles.size();

    switch (type) {
    case ComponentDataStore::DataType::Float64: {
        auto out = py::array_t<double>(static_cast<py::ssize_t>(n));
        store.GatherFloat(classId, fieldId, handles.data(), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Int64: {
        auto out = py::array_t<int64_t>(static_cast<py::ssize_t>(n));
        store.GatherInt(classId, fieldId, handles.data(), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Bool: {
        auto out = py::array_t<uint8_t>(static_cast<py::ssize_t>(n));
        store.GatherBool(classId, fieldId, handles.data(), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Vec2: {
        auto out = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(2)});
        store.GatherVec2(classId, fieldId, handles.data(), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Vec3: {
        auto out = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(3)});
        store.GatherVec3(classId, fieldId, handles.data(), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Vec4: {
        auto out = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(4)});
        store.GatherVec4(classId, fieldId, handles.data(), n, out.mutable_data());
        return out;
    }
    }
    throw py::value_error("Invalid ComponentDataStore type code");
}

static void CDS_BatchScatter(uint32_t classId, uint32_t fieldId, int typeCode,
                             py::array_t<uint32_t, py::array::c_style> slots, py::array data)
{
    auto &store = ComponentDataStore::Instance();
    auto type = CDS_ParseDataType(typeCode);
    const auto handles = CDS_ParseHandles(slots);
    const size_t n = handles.size();

    switch (type) {
    case ComponentDataStore::DataType::Float64: {
        auto d = py::array_t<double, py::array::c_style>::ensure(data);
        if (!d || d.ndim() != 1 || static_cast<size_t>(d.shape(0)) != n)
            throw py::value_error("CDS Float64 scatter data must have shape (N,)");
        store.ScatterFloat(classId, fieldId, handles.data(), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Int64: {
        auto d = py::array_t<int64_t, py::array::c_style>::ensure(data);
        if (!d || d.ndim() != 1 || static_cast<size_t>(d.shape(0)) != n)
            throw py::value_error("CDS Int64 scatter data must have shape (N,)");
        store.ScatterInt(classId, fieldId, handles.data(), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Bool: {
        auto d = py::array_t<uint8_t, py::array::c_style>::ensure(data);
        if (!d || d.ndim() != 1 || static_cast<size_t>(d.shape(0)) != n)
            throw py::value_error("CDS Bool scatter data must have shape (N,)");
        store.ScatterBool(classId, fieldId, handles.data(), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Vec2: {
        auto d = py::array_t<float, py::array::c_style>::ensure(data);
        if (!d || d.ndim() != 2 || static_cast<size_t>(d.shape(0)) != n || d.shape(1) != 2)
            throw py::value_error("CDS Vec2 scatter data must have shape (N, 2)");
        store.ScatterVec2(classId, fieldId, handles.data(), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Vec3: {
        auto d = py::array_t<float, py::array::c_style>::ensure(data);
        if (!d || d.ndim() != 2 || static_cast<size_t>(d.shape(0)) != n || d.shape(1) != 3)
            throw py::value_error("CDS Vec3 scatter data must have shape (N, 3)");
        store.ScatterVec3(classId, fieldId, handles.data(), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Vec4: {
        auto d = py::array_t<float, py::array::c_style>::ensure(data);
        if (!d || d.ndim() != 2 || static_cast<size_t>(d.shape(0)) != n || d.shape(1) != 4)
            throw py::value_error("CDS Vec4 scatter data must have shape (N, 4)");
        store.ScatterVec4(classId, fieldId, handles.data(), n, d.data());
        break;
    }
    }
}

// ── Module registration ──────────────────────────────────────────────────

namespace infernux
{

void RegisterBatchBindings(py::module_ &m)
{
    py::class_<ComponentExecutionState>(m, "_ComponentExecutionState")
        .def(py::init<py::handle, SceneManager &, uint64_t, bool>())
        .def("state", &ComponentExecutionState::State);
    py::class_<TransformExchange>(m, "_TransformExchange")
        .def(py::init<ObjectHandle, bool>())
        .def("adopt_python_state", &TransformExchange::AdoptPythonState);
    m.def("_advance_transform_bindings", &AdvanceTransformBindings);

    m.def("_transform_batch_read", &TransformBatchRead, py::arg("targets"), py::arg("property"),
          "Read a Transform property from all targets into a numpy array.\n"
          "Supported properties: 'position', 'local_position', 'local_scale',\n"
          "'euler_angles', 'local_euler_angles', 'rotation', 'local_rotation'.");

    m.def("_transform_batch_write", &TransformBatchWrite, py::arg("targets"), py::arg("data"), py::arg("property"),
          "Write a numpy array back to a Transform property on all targets.\n"
          "data.shape[0] must be >= len(targets).");

    py::enum_<TransformBatchHandle::Mode>(m, "TransformBatchMode")
        .value("STRICT", TransformBatchHandle::Mode::Strict)
        .value("COMPACT", TransformBatchHandle::Mode::Compact);

    py::class_<TransformBatchHandle>(m, "TransformBatchHandle")
        .def(py::init<const py::list &, TransformBatchHandle::Mode>(), py::arg("targets"),
             py::arg("mode") = TransformBatchHandle::Mode::Strict)
        .def("__len__", &TransformBatchHandle::size)
        .def_property_readonly("is_compact", &TransformBatchHandle::IsCompact)
        .def("validate", [](const TransformBatchHandle &handle) { handle.Resolve(); })
        .def_static("_field_width", [](const std::string &name) { return TransformField(name).width; })
        .def("_snapshot_fields", &SnapshotTransformFields, py::arg("fields"));

    m.def(
        "_create_scene_transform_batch_handle",
        [](Scene &scene, const std::string &namePrefix, TransformBatchHandle::Mode mode) {
            std::vector<Transform *> transforms;
            const auto objects = scene.GetAllObjects();
            transforms.reserve(objects.size());
            for (GameObject *object : objects) {
                if (!object || (!namePrefix.empty() && object->GetName().rfind(namePrefix, 0) != 0))
                    continue;
                transforms.push_back(object->GetTransform());
            }
            return TransformBatchHandle(std::move(transforms), mode);
        },
        py::arg("scene"), py::arg("name_prefix") = "", py::arg("mode") = TransformBatchHandle::Mode::Strict,
        "Build a Transform batch in native code without materializing GameObjects in Python.");

    m.def("_transform_batch_read", &HandleBatchRead, py::arg("handle"), py::arg("property"),
          "Read through generational handles; compact mode returns (values, valid_mask).");

    m.def("_transform_batch_write", &HandleBatchWrite, py::arg("handle"), py::arg("data"), py::arg("property"),
          "Write through generational handles; compact mode returns valid_mask.");

    // ── ComponentDataStore ──
    m.def("_cds_register_class", &CDS_RegisterClass, py::arg("name"));
    m.def("_cds_register_field", &CDS_RegisterField, py::arg("class_id"), py::arg("name"), py::arg("type_code"));

    m.def("_cds_schema_begin", [] { return ComponentDataStore::Instance().BeginSchemaTransaction(); });
    m.def(
        "_cds_schema_prepare_class",
        [](ComponentDataStore::SchemaTransactionId transactionId, const std::string &name) {
            return ComponentDataStore::Instance().PrepareClass(transactionId, name);
        },
        py::arg("transaction_id"), py::arg("name"));
    m.def(
        "_cds_schema_prepare_field",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId,
           const std::string &name, int typeCode) {
            return ComponentDataStore::Instance().PrepareField(transactionId, preparedClassId, name,
                                                               CDS_ParseDataType(typeCode));
        },
        py::arg("transaction_id"), py::arg("candidate_id"), py::arg("name"), py::arg("type_code"));
    m.def(
        "_cds_schema_has_class",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId) {
            return ComponentDataStore::Instance().HasPreparedClass(transactionId, preparedClassId);
        },
        py::arg("transaction_id"), py::arg("candidate_id"));
    m.def(
        "_cds_schema_find_class",
        [](ComponentDataStore::SchemaTransactionId transactionId, const std::string &name) {
            return CDS_OptionalPreparedClass(ComponentDataStore::Instance().FindPreparedClass(transactionId, name));
        },
        py::arg("transaction_id"), py::arg("name"));
    m.def(
        "_cds_schema_get_field_id",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId,
           const std::string &name) {
            const uint32_t fieldId =
                ComponentDataStore::Instance().GetPreparedFieldId(transactionId, preparedClassId, name);
            return CDS_OptionalPublishedClass(fieldId);
        },
        py::arg("transaction_id"), py::arg("candidate_id"), py::arg("name"));
    m.def(
        "_cds_schema_discard_class",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId) {
            return ComponentDataStore::Instance().DiscardPreparedClass(transactionId, preparedClassId);
        },
        py::arg("transaction_id"), py::arg("candidate_id"));
    m.def(
        "_cds_schema_reserve",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId,
           size_t capacity) {
            ComponentDataStore::Instance().ReservePreparedClass(transactionId, preparedClassId, capacity);
        },
        py::arg("transaction_id"), py::arg("candidate_id"), py::arg("capacity"));
    m.def("_cds_schema_alloc", &CDS_SchemaAllocSlot, py::arg("transaction_id"), py::arg("candidate_id"));
    m.def(
        "_cds_schema_free",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId,
           py::handle slot) {
            ComponentDataStore::Instance().ReleasePreparedSlot(transactionId, preparedClassId, CDS_ParseHandle(slot));
        },
        py::arg("transaction_id"), py::arg("candidate_id"), py::arg("slot"));
    m.def(
        "_cds_schema_is_alive",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId,
           py::handle slot) {
            return ComponentDataStore::Instance().IsPreparedSlotAlive(transactionId, preparedClassId,
                                                                      CDS_ParseHandle(slot));
        },
        py::arg("transaction_id"), py::arg("candidate_id"), py::arg("slot"));
    m.def("_cds_schema_get", &CDS_SchemaGet, py::arg("transaction_id"), py::arg("candidate_id"), py::arg("field_id"),
          py::arg("slot"), py::arg("type_code"));
    m.def("_cds_schema_set", &CDS_SchemaSet, py::arg("transaction_id"), py::arg("candidate_id"), py::arg("field_id"),
          py::arg("slot"), py::arg("type_code"), py::arg("value"));
    m.def("_cds_schema_migrate_slot", &CDS_SchemaMigrateSlot, py::arg("transaction_id"), py::arg("candidate_id"),
          py::arg("source_class_id"), py::arg("source_slot"), py::arg("field_map"));
    m.def(
        "_cds_schema_seal",
        [](ComponentDataStore::SchemaTransactionId transactionId) {
            return ComponentDataStore::Instance().SealSchemaTransaction(transactionId);
        },
        py::arg("transaction_id"));
    m.def(
        "_cds_schema_final_class_id",
        [](ComponentDataStore::SchemaTransactionId transactionId, ComponentDataStore::PreparedClassId preparedClassId) {
            return CDS_OptionalPublishedClass(
                ComponentDataStore::Instance().GetPreparedFinalClassId(transactionId, preparedClassId));
        },
        py::arg("transaction_id"), py::arg("candidate_id"));
    m.def(
        "_cds_schema_commit",
        [](ComponentDataStore::SchemaTransactionId transactionId) {
            return ComponentDataStore::Instance().CommitSchemaTransaction(transactionId);
        },
        py::arg("transaction_id"));
    m.def(
        "_cds_schema_finalize",
        [](ComponentDataStore::SchemaTransactionId transactionId) {
            return ComponentDataStore::Instance().FinalizeSchemaTransaction(transactionId);
        },
        py::arg("transaction_id"));
    m.def(
        "_cds_schema_rollback",
        [](ComponentDataStore::SchemaTransactionId transactionId) {
            return ComponentDataStore::Instance().RollbackSchemaTransaction(transactionId);
        },
        py::arg("transaction_id"));
    m.def(
        "_cds_schema_active",
        [](ComponentDataStore::SchemaTransactionId transactionId) {
            return ComponentDataStore::Instance().IsSchemaTransactionActive(transactionId);
        },
        py::arg("transaction_id"));

    m.def("_cds_alloc", &CDS_AllocSlot, py::arg("class_id"));
    m.def("_cds_free", &CDS_FreeSlot, py::arg("class_id"), py::arg("slot"));
    m.def("_cds_is_alive", &CDS_IsAlive, py::arg("class_id"), py::arg("slot"));
    m.def("_cds_reserve", &CDS_ReserveClass, py::arg("class_id"), py::arg("capacity"));
    m.def(
        "_cds_capacity", [](uint32_t classId) { return ComponentDataStore::Instance().GetClassCapacity(classId); },
        py::arg("class_id"));
    m.def(
        "_cds_alive_count", [](uint32_t classId) { return ComponentDataStore::Instance().GetClassAliveCount(classId); },
        py::arg("class_id"));
    m.def("_cds_get", &CDS_Get, py::arg("class_id"), py::arg("field_id"), py::arg("slot"), py::arg("type_code"));
    m.def("_cds_set", &CDS_Set, py::arg("class_id"), py::arg("field_id"), py::arg("slot"), py::arg("type_code"),
          py::arg("value"));
    m.def("_cds_batch_gather", &CDS_BatchGather, py::arg("class_id"), py::arg("field_id"), py::arg("type_code"),
          py::arg("slots"));
    m.def("_cds_batch_scatter", &CDS_BatchScatter, py::arg("class_id"), py::arg("field_id"), py::arg("type_code"),
          py::arg("slots"), py::arg("data"));
}

} // namespace infernux
