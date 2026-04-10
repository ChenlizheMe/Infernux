#include <function/scene/ComponentDataStore.h>
#include <function/scene/Transform.h>
#include <function/scene/TransformECSStore.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace infernux;

// ── helpers ──────────────────────────────────────────────────────────────

/// Extract a C-contiguous vector of Transform* from a Python list.
static std::vector<Transform *> ExtractTransforms(const py::list &pyList)
{
    const size_t n = pyList.size();
    std::vector<Transform *> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(pyList[i].cast<Transform *>());
    }
    return out;
}

// ── TransformBatchHandle (caches Transform* array) ───────────────────────

/// Caches the extracted Transform* pointers so that repeated batch_read /
/// batch_write calls with the same handle skip the O(N) pybind11 cast loop.
struct TransformBatchHandle
{
    std::vector<Transform *> transforms;

    explicit TransformBatchHandle(const py::list &pyList) : transforms(ExtractTransforms(pyList))
    {
    }

    [[nodiscard]] size_t size() const
    {
        return transforms.size();
    }
    Transform *const *data() const
    {
        return transforms.data();
    }
};

// ── Transform batch read/write ───────────────────────────────────────────

using GatherVec3Fn = void (TransformECSStore::*)(Transform *const *, float *, size_t) const;
using ScatterVec3Fn = void (TransformECSStore::*)(Transform *const *, const float *, size_t);
using GatherQuatFn = void (TransformECSStore::*)(Transform *const *, float *, size_t) const;
using ScatterQuatFn = void (TransformECSStore::*)(Transform *const *, const float *, size_t);

/// batch_read for vec3 properties → numpy (N, 3) float32
static py::array_t<float> BatchReadVec3(const py::list &targets, GatherVec3Fn gatherFn)
{
    auto transforms = ExtractTransforms(targets);
    const size_t n = transforms.size();
    auto result = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(3)});
    auto buf = result.mutable_unchecked<2>();
    float *outPtr = buf.mutable_data(0, 0);
    Transform *const *tPtr = transforms.data();
    {
        py::gil_scoped_release release;
        (TransformECSStore::Instance().*gatherFn)(tPtr, outPtr, n);
    }
    return result;
}

/// batch_write for vec3 properties from numpy (N, 3) float32
static void BatchWriteVec3(const py::list &targets, py::array data, ScatterVec3Fn scatterFn)
{
    // Accept any numeric dtype and convert to contiguous float32 on demand.
    auto fdata = py::array_t<float, py::array::c_style>::ensure(data);
    if (!fdata) {
        throw py::type_error("data must be convertible to float32 array");
    }
    auto transforms = ExtractTransforms(targets);
    const size_t n = transforms.size();
    auto buf = fdata.unchecked<2>();
    if (static_cast<size_t>(buf.shape(0)) < n) {
        throw py::value_error("data array has fewer rows than targets");
    }
    const float *inPtr = buf.data(0, 0);
    Transform *const *tPtr = transforms.data();
    {
        py::gil_scoped_release release;
        (TransformECSStore::Instance().*scatterFn)(tPtr, inPtr, n);
    }
}

/// batch_read for quaternion properties → numpy (N, 4) float32
static py::array_t<float> BatchReadQuat(const py::list &targets, GatherQuatFn gatherFn)
{
    auto transforms = ExtractTransforms(targets);
    const size_t n = transforms.size();
    auto result = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(4)});
    auto buf = result.mutable_unchecked<2>();
    float *outPtr = buf.mutable_data(0, 0);
    Transform *const *tPtr = transforms.data();
    {
        py::gil_scoped_release release;
        (TransformECSStore::Instance().*gatherFn)(tPtr, outPtr, n);
    }
    return result;
}

/// batch_write for quaternion properties from numpy (N, 4) float32
static void BatchWriteQuat(const py::list &targets, py::array data, ScatterQuatFn scatterFn)
{
    auto fdata = py::array_t<float, py::array::c_style>::ensure(data);
    if (!fdata) {
        throw py::type_error("data must be convertible to float32 array");
    }
    auto transforms = ExtractTransforms(targets);
    const size_t n = transforms.size();
    auto buf = fdata.unchecked<2>();
    if (static_cast<size_t>(buf.shape(0)) < n) {
        throw py::value_error("data array has fewer rows than targets");
    }
    const float *inPtr = buf.data(0, 0);
    Transform *const *tPtr = transforms.data();
    {
        py::gil_scoped_release release;
        (TransformECSStore::Instance().*scatterFn)(tPtr, inPtr, n);
    }
}

// ── Dispatch table: property name → gather/scatter function ──────────────

struct Vec3BatchOps
{
    GatherVec3Fn gather;
    ScatterVec3Fn scatter;
};

struct QuatBatchOps
{
    GatherQuatFn gather;
    ScatterQuatFn scatter;
};

static const std::unordered_map<std::string, Vec3BatchOps> kTransformVec3Ops = {
    {"local_position", {&TransformECSStore::GatherLocalPositions, &TransformECSStore::ScatterLocalPositions}},
    {"local_scale", {&TransformECSStore::GatherLocalScales, &TransformECSStore::ScatterLocalScales}},
    {"local_euler_angles", {&TransformECSStore::GatherLocalEulerAngles, &TransformECSStore::ScatterLocalEulerAngles}},
    {"position", {&TransformECSStore::GatherWorldPositions, &TransformECSStore::ScatterWorldPositions}},
    {"euler_angles", {&TransformECSStore::GatherWorldEulerAngles, &TransformECSStore::ScatterWorldEulerAngles}},
};

static const std::unordered_map<std::string, QuatBatchOps> kTransformQuatOps = {
    {"local_rotation", {&TransformECSStore::GatherLocalRotations, &TransformECSStore::ScatterLocalRotations}},
    {"rotation", {&TransformECSStore::GatherWorldRotations, &TransformECSStore::ScatterWorldRotations}},
};

// ── Python-facing free functions ─────────────────────────────────────────

static py::object TransformBatchRead(const py::list &targets, const std::string &prop)
{
    {
        auto it = kTransformVec3Ops.find(prop);
        if (it != kTransformVec3Ops.end()) {
            return BatchReadVec3(targets, it->second.gather);
        }
    }
    {
        auto it = kTransformQuatOps.find(prop);
        if (it != kTransformQuatOps.end()) {
            return BatchReadQuat(targets, it->second.gather);
        }
    }
    throw py::value_error("Unknown Transform property: '" + prop + "'");
}

static void TransformBatchWrite(const py::list &targets, py::array data, const std::string &prop)
{
    {
        auto it = kTransformVec3Ops.find(prop);
        if (it != kTransformVec3Ops.end()) {
            BatchWriteVec3(targets, data, it->second.scatter);
            return;
        }
    }
    {
        auto it = kTransformQuatOps.find(prop);
        if (it != kTransformQuatOps.end()) {
            BatchWriteQuat(targets, data, it->second.scatter);
            return;
        }
    }
    throw py::value_error("Unknown Transform property: '" + prop + "'");
}

// ── Handle-based batch read/write (avoids repeated ExtractTransforms) ──

static py::object HandleBatchRead(const TransformBatchHandle &handle, const std::string &prop)
{
    const size_t n = handle.size();
    Transform *const *tPtr = handle.data();
    {
        auto it = kTransformVec3Ops.find(prop);
        if (it != kTransformVec3Ops.end()) {
            auto result = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(3)});
            float *outPtr = result.mutable_unchecked<2>().mutable_data(0, 0);
            {
                py::gil_scoped_release release;
                (TransformECSStore::Instance().*(it->second.gather))(tPtr, outPtr, n);
            }
            return result;
        }
    }
    {
        auto it = kTransformQuatOps.find(prop);
        if (it != kTransformQuatOps.end()) {
            auto result = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(4)});
            float *outPtr = result.mutable_unchecked<2>().mutable_data(0, 0);
            {
                py::gil_scoped_release release;
                (TransformECSStore::Instance().*(it->second.gather))(tPtr, outPtr, n);
            }
            return result;
        }
    }
    throw py::value_error("Unknown Transform property: '" + prop + "'");
}

static void HandleBatchWrite(const TransformBatchHandle &handle, py::array data, const std::string &prop)
{
    auto fdata = py::array_t<float, py::array::c_style>::ensure(data);
    if (!fdata) {
        throw py::type_error("data must be convertible to float32 array");
    }
    const size_t n = handle.size();
    Transform *const *tPtr = handle.data();
    {
        auto it = kTransformVec3Ops.find(prop);
        if (it != kTransformVec3Ops.end()) {
            auto buf = fdata.unchecked<2>();
            if (static_cast<size_t>(buf.shape(0)) < n) {
                throw py::value_error("data array has fewer rows than targets");
            }
            const float *inPtr = buf.data(0, 0);
            {
                py::gil_scoped_release release;
                (TransformECSStore::Instance().*(it->second.scatter))(tPtr, inPtr, n);
            }
            return;
        }
    }
    {
        auto it = kTransformQuatOps.find(prop);
        if (it != kTransformQuatOps.end()) {
            auto buf = fdata.unchecked<2>();
            if (static_cast<size_t>(buf.shape(0)) < n) {
                throw py::value_error("data array has fewer rows than targets");
            }
            const float *inPtr = buf.data(0, 0);
            {
                py::gil_scoped_release release;
                (TransformECSStore::Instance().*(it->second.scatter))(tPtr, inPtr, n);
            }
            return;
        }
    }
    throw py::value_error("Unknown Transform property: '" + prop + "'");
}

// ── ComponentDataStore bindings ───────────────────────────────────────────

static uint32_t CDS_RegisterClass(const std::string &name)
{
    return ComponentDataStore::Instance().RegisterClass(name);
}

static uint32_t CDS_RegisterField(uint32_t classId, const std::string &name, int typeCode)
{
    return ComponentDataStore::Instance().RegisterField(classId, name,
                                                        static_cast<ComponentDataStore::DataType>(typeCode));
}

static uint32_t CDS_AllocSlot(uint32_t classId)
{
    return ComponentDataStore::Instance().AllocateSlot(classId);
}

static void CDS_FreeSlot(uint32_t classId, uint32_t slot)
{
    ComponentDataStore::Instance().ReleaseSlot(classId, slot);
}

// ── per-element accessors ────────────────────────────────────────────────

static py::object CDS_Get(uint32_t classId, uint32_t fieldId, uint32_t slot, int typeCode)
{
    auto &store = ComponentDataStore::Instance();
    auto type = static_cast<ComponentDataStore::DataType>(typeCode);
    switch (type) {
    case ComponentDataStore::DataType::Float64:
        return py::cast(store.GetFloat(classId, fieldId, slot));
    case ComponentDataStore::DataType::Int64:
        return py::cast(store.GetInt(classId, fieldId, slot));
    case ComponentDataStore::DataType::Bool:
        return py::cast(store.GetBool(classId, fieldId, slot));
    case ComponentDataStore::DataType::Vec2: {
        float v[2];
        store.GetVec2(classId, fieldId, slot, v);
        return py::make_tuple(v[0], v[1]);
    }
    case ComponentDataStore::DataType::Vec3: {
        float v[3];
        store.GetVec3(classId, fieldId, slot, v);
        return py::make_tuple(v[0], v[1], v[2]);
    }
    case ComponentDataStore::DataType::Vec4: {
        float v[4];
        store.GetVec4(classId, fieldId, slot, v);
        return py::make_tuple(v[0], v[1], v[2], v[3]);
    }
    }
    return py::none();
}

static void CDS_Set(uint32_t classId, uint32_t fieldId, uint32_t slot, int typeCode, py::object value)
{
    auto &store = ComponentDataStore::Instance();
    auto type = static_cast<ComponentDataStore::DataType>(typeCode);
    switch (type) {
    case ComponentDataStore::DataType::Float64:
        store.SetFloat(classId, fieldId, slot, value.cast<double>());
        break;
    case ComponentDataStore::DataType::Int64:
        store.SetInt(classId, fieldId, slot, value.cast<int64_t>());
        break;
    case ComponentDataStore::DataType::Bool:
        store.SetBool(classId, fieldId, slot, value.cast<bool>());
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
        store.SetVec2(classId, fieldId, slot, v);
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
        store.SetVec3(classId, fieldId, slot, v);
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
        store.SetVec4(classId, fieldId, slot, v);
        break;
    }
    }
}

// ── batch gather/scatter for ComponentDataStore ──────────────────────────

static py::array CDS_BatchGather(uint32_t classId, uint32_t fieldId, int typeCode,
                                 py::array_t<uint32_t, py::array::c_style> slots)
{
    auto &store = ComponentDataStore::Instance();
    auto type = static_cast<ComponentDataStore::DataType>(typeCode);
    auto slotBuf = slots.unchecked<1>();
    const size_t n = static_cast<size_t>(slotBuf.shape(0));

    switch (type) {
    case ComponentDataStore::DataType::Float64: {
        auto out = py::array_t<double>({static_cast<py::ssize_t>(n)});
        store.GatherFloat(classId, fieldId, slotBuf.data(0), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Int64: {
        auto out = py::array_t<int64_t>({static_cast<py::ssize_t>(n)});
        store.GatherInt(classId, fieldId, slotBuf.data(0), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Bool: {
        auto out = py::array_t<uint8_t>({static_cast<py::ssize_t>(n)});
        store.GatherBool(classId, fieldId, slotBuf.data(0), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Vec2: {
        auto out = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(2)});
        store.GatherVec2(classId, fieldId, slotBuf.data(0), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Vec3: {
        auto out = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(3)});
        store.GatherVec3(classId, fieldId, slotBuf.data(0), n, out.mutable_data());
        return out;
    }
    case ComponentDataStore::DataType::Vec4: {
        auto out = py::array_t<float>({static_cast<py::ssize_t>(n), py::ssize_t(4)});
        store.GatherVec4(classId, fieldId, slotBuf.data(0), n, out.mutable_data());
        return out;
    }
    }
    return py::array();
}

static void CDS_BatchScatter(uint32_t classId, uint32_t fieldId, int typeCode,
                             py::array_t<uint32_t, py::array::c_style> slots, py::array data)
{
    auto &store = ComponentDataStore::Instance();
    auto type = static_cast<ComponentDataStore::DataType>(typeCode);
    auto slotBuf = slots.unchecked<1>();
    const size_t n = static_cast<size_t>(slotBuf.shape(0));

    switch (type) {
    case ComponentDataStore::DataType::Float64: {
        auto d = py::array_t<double, py::array::c_style>::ensure(data);
        store.ScatterFloat(classId, fieldId, slotBuf.data(0), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Int64: {
        auto d = py::array_t<int64_t, py::array::c_style>::ensure(data);
        store.ScatterInt(classId, fieldId, slotBuf.data(0), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Bool: {
        auto d = py::array_t<uint8_t, py::array::c_style>::ensure(data);
        store.ScatterBool(classId, fieldId, slotBuf.data(0), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Vec2: {
        auto d = py::array_t<float, py::array::c_style>::ensure(data);
        store.ScatterVec2(classId, fieldId, slotBuf.data(0), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Vec3: {
        auto d = py::array_t<float, py::array::c_style>::ensure(data);
        store.ScatterVec3(classId, fieldId, slotBuf.data(0), n, d.data());
        break;
    }
    case ComponentDataStore::DataType::Vec4: {
        auto d = py::array_t<float, py::array::c_style>::ensure(data);
        store.ScatterVec4(classId, fieldId, slotBuf.data(0), n, d.data());
        break;
    }
    }
}

// ── Module registration ──────────────────────────────────────────────────

namespace infernux
{

void RegisterBatchBindings(py::module_ &m)
{
    m.def("_transform_batch_read", &TransformBatchRead, py::arg("targets"), py::arg("property"),
          "Read a Transform property from all targets into a numpy array.\n"
          "Supported properties: 'position', 'local_position', 'local_scale',\n"
          "'euler_angles', 'local_euler_angles', 'rotation', 'local_rotation'.");

    m.def("_transform_batch_write", &TransformBatchWrite, py::arg("targets"), py::arg("data"), py::arg("property"),
          "Write a numpy array back to a Transform property on all targets.\n"
          "data.shape[0] must be >= len(targets).");

    // ── TransformBatchHandle (cached Transform* pointers) ──
    py::class_<TransformBatchHandle>(m, "TransformBatchHandle")
        .def(py::init<const py::list &>(), py::arg("targets"))
        .def("__len__", &TransformBatchHandle::size);

    m.def("_transform_batch_read", &HandleBatchRead, py::arg("handle"), py::arg("property"),
          "Like _transform_batch_read but uses a cached TransformBatchHandle.");

    m.def("_transform_batch_write", &HandleBatchWrite, py::arg("handle"), py::arg("data"), py::arg("property"),
          "Like _transform_batch_write but uses a cached TransformBatchHandle.");

    // ── ComponentDataStore ──
    m.def("_cds_register_class", &CDS_RegisterClass, py::arg("name"));
    m.def("_cds_register_field", &CDS_RegisterField, py::arg("class_id"), py::arg("name"), py::arg("type_code"));
    m.def("_cds_alloc", &CDS_AllocSlot, py::arg("class_id"));
    m.def("_cds_free", &CDS_FreeSlot, py::arg("class_id"), py::arg("slot"));
    m.def("_cds_get", &CDS_Get, py::arg("class_id"), py::arg("field_id"), py::arg("slot"), py::arg("type_code"));
    m.def("_cds_set", &CDS_Set, py::arg("class_id"), py::arg("field_id"), py::arg("slot"), py::arg("type_code"),
          py::arg("value"));
    m.def("_cds_batch_gather", &CDS_BatchGather, py::arg("class_id"), py::arg("field_id"), py::arg("type_code"),
          py::arg("slots"));
    m.def("_cds_batch_scatter", &CDS_BatchScatter, py::arg("class_id"), py::arg("field_id"), py::arg("type_code"),
          py::arg("slots"), py::arg("data"));
    m.def("_cds_clear", []() { ComponentDataStore::Instance().Clear(); });
}

} // namespace infernux
