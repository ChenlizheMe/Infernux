#include "BindingRegistration.h"
#include "BufferPyView.h"
#include "SpirvPyBridge.h"
#include <function/renderer/RenderPoseBuffer.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/rhi/RhiUpload.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <unordered_set>

using namespace infernux;
namespace py = pybind11;

namespace
{
using ComputeStorage = std::map<std::tuple<uintptr_t, uint32_t, uint32_t>, uint64_t>;

void AppendComputeStorage(ComputeStorage &blocks, const std::shared_ptr<rhi::ComputeBuffer> &buffer)
{
    // Resolve the current allocation, including copy-on-write replacement.
    const auto allocation = buffer->GetAllocation();
    const auto handle = allocation->GetBuffer();
    blocks[{reinterpret_cast<uintptr_t>(&buffer->GetHost().queue), handle.index, handle.generation}] =
        allocation->GetByteSize();
}

py::dict ComputeStorageDictionary(const ComputeStorage &blocks)
{
    py::dict result;
    for (const auto &[key, bytes] : blocks)
        result[py::make_tuple("gpu", std::get<0>(key), std::get<1>(key), std::get<2>(key))] = bytes;
    return result;
}

class ComputeStorageGroup
{
  public:
    explicit ComputeStorageGroup(const std::vector<std::shared_ptr<rhi::ComputeBuffer>> &buffers)
    {
        std::unordered_set<rhi::ComputeBuffer *> seen;
        for (const auto &buffer : buffers)
            if (seen.insert(buffer.get()).second)
                m_buffers.push_back(buffer);
    }

    py::dict Blocks() const
    {
        ComputeStorage blocks;
        for (const auto &entry : m_buffers)
            if (const auto buffer = entry.lock())
                AppendComputeStorage(blocks, buffer);
        return ComputeStorageDictionary(blocks);
    }

  private:
    // Accounting must neither pin a dead buffer nor freeze a writer's storage.
    std::vector<std::weak_ptr<rhi::ComputeBuffer>> m_buffers;
};

auto SubmissionIdentity(rhi::SubmissionTicket ticket)
{
    return py::make_tuple(ticket.device, static_cast<uint32_t>(ticket.queue), ticket.serial);
}
template <typename Owner> auto RetainedComputeStorage(const Owner &owner)
{
    std::vector<rhi::ComputeStorageBlock> blocks;
    owner.AppendRetainedStorage(blocks);
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> result;
    result.reserve(blocks.size());
    for (const auto &block : blocks)
        result.emplace_back(block.buffer.index, block.buffer.generation, block.byteSize);
    return result;
}
rhi::ComputeBufferAccess DecodeComputeAccess(const std::string &access)
{
    if (access == "read")
        return rhi::ComputeBufferAccess::Read;
    if (access == "write")
        return rhi::ComputeBufferAccess::Write;
    if (access == "read_write")
        return rhi::ComputeBufferAccess::ReadWrite;
    throw py::value_error("Compute buffer access must be read, write, or read_write");
}

const char *ComputeScalarName(rhi::ComputeScalarType type)
{
    switch (type) {
    case rhi::ComputeScalarType::Float32:
        return "float32";
    case rhi::ComputeScalarType::Int32:
        return "int32";
    case rhi::ComputeScalarType::UInt32:
        return "uint32";
    case rhi::ComputeScalarType::Int64:
        return "int64";
    }
    throw std::runtime_error("Unknown compute buffer scalar type");
}

std::vector<uint8_t> DecodeComputeBytes(py::handle value)
{
    const auto payload = bindings::ContiguousBuffer(value);
    if (!payload.size)
        return {};
    const auto *begin = static_cast<const uint8_t *>(payload.ptr);
    return {begin, begin + payload.size * payload.itemsize};
}

std::vector<rhi::ComputeBindingRange> DecodeComputeRanges(py::iterable values)
{
    std::vector<rhi::ComputeBindingRange> result;
    for (const auto value : values) {
        const auto range = py::cast<py::tuple>(value);
        if (range.size() != 3)
            throw py::value_error("Compute range must contain binding index, byte offset and byte size");
        result.push_back({range[0].cast<uint32_t>(), range[1].cast<uint64_t>(), range[2].cast<uint64_t>()});
    }
    return result;
}

std::vector<rhi::ComputeDispatchDesc> DecodeComputeDispatches(py::iterable values)
{
    std::vector<rhi::ComputeDispatchDesc> dispatches;
    for (const auto value : values) {
        const auto dispatch = py::cast<py::tuple>(value);
        if (dispatch.size() != 8)
            throw py::value_error(
                "Compute dispatch requires kernel, buffers, accesses, constants, three group counts and a proof");
        std::vector<rhi::ComputeBufferAccess> accesses;
        for (const auto access : dispatch[2].cast<py::iterable>())
            accesses.push_back(DecodeComputeAccess(py::cast<std::string>(access)));
        std::optional<rhi::ComputeDispatchProof> proof;
        if (!dispatch[7].is_none()) {
            const auto ranges = dispatch[7].cast<py::tuple>();
            if (ranges.size() != 2)
                throw py::value_error("Compute proof requires read ranges and complete writes");
            proof = rhi::ComputeDispatchProof{DecodeComputeRanges(ranges[0].cast<py::iterable>()),
                                              DecodeComputeRanges(ranges[1].cast<py::iterable>())};
        }
        dispatches.push_back({dispatch[0].cast<std::shared_ptr<rhi::ComputeKernel>>(),
                              dispatch[1].cast<std::vector<std::shared_ptr<rhi::ComputeBuffer>>>(), std::move(accesses),
                              DecodeComputeBytes(dispatch[3]), dispatch[4].cast<uint32_t>(),
                              dispatch[5].cast<uint32_t>(), dispatch[6].cast<uint32_t>(), std::move(proof)});
    }
    return dispatches;
}

std::vector<rhi::ComputeWorkSlot> DecodeComputeWorkSlots(rhi::ComputeHost &host, py::iterable values)
{
    std::vector<rhi::ComputeWorkSlot> result;
    for (const auto value : values) {
        const auto slot = py::cast<py::tuple>(value);
        if (slot.size() != 2)
            throw py::value_error("Compute work slot must contain a prototype buffer and private-storage flag");
        auto prototype = slot[0].cast<std::shared_ptr<rhi::ComputeBuffer>>();
        if (!prototype || !prototype->GetHost().SharesServicesWith(host))
            throw py::value_error("Compute work prototypes must belong to its host");
        result.push_back({prototype->GetDesc(), slot[1].cast<bool>() ? nullptr : std::move(prototype)});
    }
    return result;
}

std::vector<rhi::ComputeWorkPort> DecodeComputeWorkPorts(py::iterable values)
{
    std::vector<rhi::ComputeWorkPort> result;
    for (const auto value : values) {
        const auto port = py::cast<py::tuple>(value);
        if (port.size() != 5)
            throw py::value_error("Compute work port must contain name, slot, offset, bytes and input flag");
        result.push_back({port[0].cast<std::string>(),
                          {port[1].cast<uint32_t>(), port[2].cast<uint64_t>(), port[3].cast<uint64_t>()},
                          port[4].cast<bool>()});
    }
    return result;
}

std::vector<rhi::ComputeWorkDispatch> DecodeComputeWorkDispatches(py::iterable values)
{
    std::vector<rhi::ComputeWorkDispatch> result;
    for (const auto value : values) {
        const auto dispatch = py::cast<py::tuple>(value);
        if (dispatch.size() != 7)
            throw py::value_error("Compute work dispatch must contain kernel, bindings, complete-write indices, "
                                  "push constants and three group counts");
        rhi::ComputeWorkDispatch item;
        item.kernel = dispatch[0].cast<std::shared_ptr<rhi::ComputeKernel>>();
        for (const auto bindingValue : dispatch[1].cast<py::iterable>()) {
            const auto binding = py::cast<py::tuple>(bindingValue);
            if (binding.size() != 4)
                throw py::value_error("Compute work binding must contain slot, offset, bytes and access");
            item.bindings.push_back(
                {{binding[0].cast<uint32_t>(), binding[1].cast<uint64_t>(), binding[2].cast<uint64_t>()},
                 DecodeComputeAccess(binding[3].cast<std::string>())});
        }
        item.completeWrites = DecodeComputeRanges(dispatch[2].cast<py::iterable>());
        item.pushConstants = DecodeComputeBytes(dispatch[3]);
        item.groupCountX = dispatch[4].cast<uint32_t>();
        item.groupCountY = dispatch[5].cast<uint32_t>();
        item.groupCountZ = dispatch[6].cast<uint32_t>();
        result.push_back(std::move(item));
    }
    return result;
}

std::vector<rhi::ComputeBufferUpdate> DecodeComputeUpdates(py::iterable values)
{
    std::vector<rhi::ComputeBufferUpdate> updates;
    for (const auto value : values) {
        if (!py::isinstance<py::tuple>(value) && !py::isinstance<py::list>(value))
            throw py::type_error("Compute buffer update must be a tuple or list");
        const auto update = py::reinterpret_borrow<py::sequence>(value);
        if (update.size() != 3)
            throw py::value_error("Compute buffer update must contain buffer, bytes, and byte offset");
        updates.push_back({py::cast<std::shared_ptr<rhi::ComputeBuffer>>(update[0]), py::cast<uint64_t>(update[2]),
                           DecodeComputeBytes(update[1])});
    }
    return updates;
}

std::vector<rhi::ComputeBufferRead> DecodeComputeReads(py::iterable values)
{
    std::vector<rhi::ComputeBufferRead> reads;
    for (const auto value : values) {
        if (!py::isinstance<py::tuple>(value) && !py::isinstance<py::list>(value))
            throw py::type_error("Compute buffer read must be a tuple or list");
        const auto read = py::reinterpret_borrow<py::sequence>(value);
        if (read.size() != 3)
            throw py::value_error("Compute buffer read must contain buffer, byte offset, and byte size");
        reads.push_back({
            py::cast<std::shared_ptr<rhi::ComputeBuffer>>(read[0]),
            py::cast<uint64_t>(read[1]),
            py::cast<uint64_t>(read[2]),
        });
    }
    return reads;
}

std::vector<rhi::ComputeBufferCopy> DecodeComputeCopies(py::iterable values)
{
    std::vector<rhi::ComputeBufferCopy> copies;
    for (const auto &value : values) {
        const auto entry = py::cast<py::tuple>(value);
        if (entry.size() != 5)
            throw py::value_error(
                "Compute copies require source, destination, source offset, destination offset and byte size");
        copies.push_back({entry[0].cast<std::shared_ptr<rhi::ComputeBuffer>>(),
                          entry[1].cast<std::shared_ptr<rhi::ComputeBuffer>>(),
                          {entry[2].cast<uint64_t>(), entry[3].cast<uint64_t>(), entry[4].cast<uint64_t>()}});
    }
    return copies;
}
} // namespace

void infernux::RegisterComputeBindings(py::module_ &m)
{
    py::class_<ComputeStorageGroup>(m, "_ComputeStorageGroup")
        .def(py::init<const std::vector<std::shared_ptr<rhi::ComputeBuffer>> &>())
        .def_property_readonly("storage_blocks", &ComputeStorageGroup::Blocks);
    m.def("_compute_buffer_storage", [](const std::vector<std::shared_ptr<rhi::ComputeBuffer>> &buffers) {
        ComputeStorage blocks;
        for (const auto &buffer : buffers)
            AppendComputeStorage(blocks, buffer);
        return ComputeStorageDictionary(blocks);
    });
    m.def("_compute_upload_staging_bytes",
          [](uint64_t bytes) { return bytes > rhi::InlineBufferUpdateBytes ? rhi::UploadStagingClass(bytes) : 0; });
    py::enum_<rhi::ComputeScalarType>(m, "ComputeScalarType")
        .value("FLOAT32", rhi::ComputeScalarType::Float32)
        .value("INT32", rhi::ComputeScalarType::Int32)
        .value("UINT32", rhi::ComputeScalarType::UInt32)
        .value("INT64", rhi::ComputeScalarType::Int64);
    py::class_<rhi::ComputeBufferDesc>(m, "ComputeBufferDesc")
        .def(py::init<>())
        .def_readwrite("element_count", &rhi::ComputeBufferDesc::elementCount)
        .def_readwrite("scalar_type", &rhi::ComputeBufferDesc::scalarType)
        .def_readwrite("lanes", &rhi::ComputeBufferDesc::lanes);
    m.def(
        "_snapshot_material_fields",
        [](const std::vector<std::shared_ptr<InxMaterial>> &materials, py::iterable fields) {
            struct Field
            {
                std::string kind, name;
                float fallback;
                py::array_t<float, py::array::c_style> output;
            };
            std::vector<Field> columns;
            for (const auto &value : fields) {
                auto row = py::cast<py::tuple>(value);
                columns.push_back({row[0].cast<std::string>(), row[1].cast<std::string>(), row[2].cast<float>(),
                                   row[3].cast<py::array_t<float, py::array::c_style>>()});
                const auto &column = columns.back();
                const auto width = column.kind == "float"     ? 1
                                   : column.kind == "vector2" ? 2
                                   : column.kind == "vector3" ? 3
                                                              : 4;
                if (column.output.ndim() != 2 || column.output.shape(0) != static_cast<py::ssize_t>(materials.size()) ||
                    column.output.shape(1) != width || !column.output.writeable())
                    throw py::value_error("material snapshot destination must be a writable float32 field matrix");
            }
            // The owning thread reads all requested fields together. These private
            // CPU arrays become the actual NN inputs; no second packing copy follows.
            for (size_t row = 0; row < materials.size(); ++row) {
                if (!materials[row])
                    throw py::value_error("material snapshot source has retired");
                for (auto &column : columns) {
                    auto *output = column.output.mutable_data() + row * column.output.shape(1);
                    const auto *property = materials[row]->GetProperty(column.name);
                    glm::vec4 value(0.0f);
                    if (column.kind == "float") {
                        output[0] = column.fallback;
                        if (property) {
                            if (property->type == MaterialPropertyType::Float)
                                output[0] = std::get<float>(property->value);
                            else if (property->type == MaterialPropertyType::Int)
                                output[0] = static_cast<float>(std::get<int>(property->value));
                            else
                                throw py::type_error("get_float requires a scalar material property");
                        }
                        continue;
                    }
                    if (column.kind == "color")
                        value.w = 1.0f;
                    if (property) {
                        if (column.kind == "vector2" && property->type == MaterialPropertyType::Float2) {
                            const auto v = std::get<glm::vec2>(property->value);
                            value.x = v.x;
                            value.y = v.y;
                        } else if (column.kind == "vector3" && property->type == MaterialPropertyType::Float3) {
                            const auto v = std::get<glm::vec3>(property->value);
                            value.x = v.x;
                            value.y = v.y;
                            value.z = v.z;
                        } else if ((column.kind == "vector4" || column.kind == "color") &&
                                   (property->type == MaterialPropertyType::Float4 ||
                                    property->type == MaterialPropertyType::Color))
                            value = std::get<glm::vec4>(property->value);
                    }
                    for (py::ssize_t lane = 0; lane < column.output.shape(1); ++lane)
                        output[lane] = value[static_cast<int>(lane)];
                }
            }
        },
        py::arg("materials"), py::arg("fields"));

    py::class_<rhi::ComputeReadback, std::shared_ptr<rhi::ComputeReadback>>(m, "_ComputeReadback")
        .def_property_readonly("storage_blocks", &RetainedComputeStorage<rhi::ComputeReadback>)
        .def_property_readonly("done",
                               [](rhi::ComputeReadback &readback) {
                                   if (readback.GetHost().IsAlive())
                                       readback.GetHost().Queue().RecordNativeBoundary();
                                   return readback.IsComplete();
                               })
        .def_property_readonly("cancelled", &rhi::ComputeReadback::IsCancelled)
        .def_property_readonly("submission_serial",
                               [](const rhi::ComputeReadback &readback) { return readback.GetSubmission().serial; })
        .def_property_readonly(
            "source_write_serial",
            [](const rhi::ComputeReadback &readback) { return readback.GetSourceWriteSubmission().serial; })
        .def_property_readonly("source_write_submission",
                               [](const rhi::ComputeReadback &readback) {
                                   return SubmissionIdentity(readback.GetSourceWriteSubmission());
                               })
        .def_property_readonly("byte_size", &rhi::ComputeReadback::GetByteSize)
        .def("cancel",
             [](rhi::ComputeReadback &readback) {
                 if (readback.GetHost().IsAlive())
                     readback.GetHost().Queue().RecordNativeBoundary();
                 return readback.Cancel();
             })
        .def("get_bytes",
             [](rhi::ComputeReadback &readback) {
                 readback.GetHost().Queue().RecordNativeBoundary();
                 const auto size = readback.GetByteSize();
                 if (size > static_cast<uint64_t>(PY_SSIZE_T_MAX))
                     throw py::value_error("Compute readback exceeds the Python bytes address range");
                 auto *object = PyBytes_FromStringAndSize(nullptr, static_cast<Py_ssize_t>(size));
                 if (!object)
                     throw py::error_already_set();
                 auto result = py::reinterpret_steal<py::bytes>(object);
                 void *destination = PyBytes_AS_STRING(result.ptr());
                 {
                     py::gil_scoped_release release;
                     readback.ReadInto(destination, size);
                 }
                 return result;
             })
        .def(
            "copy_into",
            [](rhi::ComputeReadback &readback, py::array destination) {
                const auto dtype = destination.dtype();
                if (!(dtype.equal(py::dtype::of<float>()) || dtype.equal(py::dtype::of<int32_t>()) ||
                      dtype.equal(py::dtype::of<uint32_t>()) || dtype.equal(py::dtype::of<int64_t>())))
                    throw py::type_error(
                        "Compute readback destination requires native float32, int32, uint32 or int64");
                if (!(destination.flags() & py::array::c_style) || !destination.writeable())
                    throw py::value_error("Compute readback destination must be writable and C-contiguous");
                const auto size = static_cast<uint64_t>(destination.nbytes());
                if (size != readback.GetByteSize())
                    throw py::value_error("Compute readback destination byte size does not match the snapshot");
                void *data = destination.mutable_data();
                readback.GetHost().Queue().RecordNativeBoundary();
                py::gil_scoped_release release;
                readback.ReadInto(data, size);
            },
            py::arg("destination").noconvert());

    py::class_<rhi::ComputeBuffer, std::shared_ptr<rhi::ComputeBuffer>>(m, "_ComputeBuffer")
        .def("freeze", &rhi::ComputeBuffer::Freeze)
        .def("freeze_scalars", &rhi::ComputeBuffer::FreezeScalars)
        .def("publish_storage", &rhi::ComputeBuffer::PublishStorage)
        .def("belongs_to", [](const rhi::ComputeBuffer &buffer,
                              const rhi::ComputeHost &host) { return buffer.GetHost().SharesServicesWith(host); })
        .def_property_readonly("storage_blocks", &RetainedComputeStorage<rhi::ComputeBuffer>)
        .def_property_readonly("byte_size", &rhi::ComputeBuffer::GetByteSize)
        .def_property_readonly("last_write_serial",
                               [](const rhi::ComputeBuffer &buffer) { return buffer.GetLastWriteSubmission().serial; })
        .def_property_readonly(
            "last_write_submission",
            [](const rhi::ComputeBuffer &buffer) { return SubmissionIdentity(buffer.GetLastWriteSubmission()); })
        .def("has_initialized_range", &rhi::ComputeBuffer::HasInitializedRange, py::arg("offset"), py::arg("byte_size"))
        .def_property_readonly("resource_index",
                               [](const rhi::ComputeBuffer &buffer) { return buffer.GetBuffer().index; })
        .def_property_readonly("resource_generation",
                               [](const rhi::ComputeBuffer &buffer) { return buffer.GetBuffer().generation; })
        .def_property_readonly("element_count",
                               [](const rhi::ComputeBuffer &buffer) { return buffer.GetDesc().elementCount; })
        .def_property_readonly("element_stride",
                               [](const rhi::ComputeBuffer &buffer) { return buffer.GetDesc().GetElementStride(); })
        .def_property_readonly("lanes", [](const rhi::ComputeBuffer &buffer) { return buffer.GetDesc().lanes; })
        .def_property_readonly(
            "scalar_type",
            [](const rhi::ComputeBuffer &buffer) { return ComputeScalarName(buffer.GetDesc().scalarType); })
        .def(
            "set_bytes",
            [](rhi::ComputeBuffer &buffer, py::bytes value, uint64_t offset) {
                buffer.GetHost().Queue().RecordNativeBoundary();
                std::string bytes = value;
                buffer.SetData(offset, bytes.data(), static_cast<uint64_t>(bytes.size()));
            },
            py::arg("value"), py::arg("offset") = 0)
        .def(
            "get_bytes",
            [](rhi::ComputeBuffer &buffer, uint64_t byteSize, uint64_t offset) {
                buffer.GetHost().Queue().RecordNativeBoundary();
                auto bytes = buffer.GetData(offset, byteSize);
                return py::bytes(reinterpret_cast<const char *>(bytes.data()), static_cast<py::ssize_t>(bytes.size()));
            },
            py::arg("byte_size"), py::arg("offset") = 0);

    py::class_<RenderPoseBuffer, std::shared_ptr<RenderPoseBuffer>>(m, "_RenderPoseBuffer")
        .def(py::init<const std::shared_ptr<rhi::ComputeBuffer> &, const std::shared_ptr<rhi::ComputeBuffer> &,
                      uint32_t, uint32_t, uint32_t>(),
             py::arg("models"), py::arg("bounds"), py::arg("count"), py::arg("first_model") = 0,
             py::arg("first_bounds") = 0)
        .def("publish", &RenderPoseBuffer::Publish, py::arg("models"), py::arg("bounds"), py::arg("first_model") = 0,
             py::arg("first_bounds") = 0)
        .def_property_readonly("count", &RenderPoseBuffer::GetCount)
        .def_property_readonly("generation",
                               [](const RenderPoseBuffer &source) { return source.Acquire()->generation; });

    py::class_<rhi::ComputeKernel, std::shared_ptr<rhi::ComputeKernel>>(m, "_ComputeKernel")
        .def_property_readonly("buffer_binding_count", &rhi::ComputeKernel::GetBufferBindingCount)
        .def_property_readonly("buffer_bindings", &rhi::ComputeKernel::GetBufferBindings)
        .def_property_readonly("push_constant_bytes", &rhi::ComputeKernel::GetPushConstantBytes)
        .def_property_readonly("pending_dispatch_count", &rhi::ComputeKernel::GetPendingDispatchCount)
        .def_property_readonly("cached_binding_group_count", &rhi::ComputeKernel::GetCachedBindingGroupCount)
        .def(
            "dispatch",
            [](rhi::ComputeKernel &kernel, std::vector<std::shared_ptr<rhi::ComputeBuffer>> buffers,
               py::bytes pushConstants, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
                kernel.GetHost().Queue().RecordNativeBoundary();
                const std::string constants = pushConstants;
                kernel.Dispatch(std::move(buffers), constants.empty() ? nullptr : constants.data(),
                                static_cast<uint32_t>(constants.size()), groupCountX, groupCountY, groupCountZ);
            },
            py::arg("buffers"), py::arg("push_constants") = py::bytes(), py::arg("group_count_x") = 1,
            py::arg("group_count_y") = 1, py::arg("group_count_z") = 1)
        .def("collect", &rhi::ComputeKernel::Collect)
        .def("wait", &rhi::ComputeKernel::Wait);

    py::class_<rhi::ComputeWorkRecipe, std::shared_ptr<rhi::ComputeWorkRecipe>>(m, "_ComputeWorkRecipe")
        .def_property_readonly("slots",
                               [](const rhi::ComputeWorkRecipe &work) {
                                   py::list result;
                                   for (const auto &slot : work.GetSlots())
                                       result.append(py::make_tuple(slot.allocation.elementCount,
                                                                    ComputeScalarName(slot.allocation.scalarType),
                                                                    slot.allocation.lanes, slot.external));
                                   return result;
                               })
        .def_property_readonly("ports",
                               [](const rhi::ComputeWorkRecipe &work) {
                                   py::list result;
                                   for (const auto &port : work.GetPorts())
                                       result.append(py::make_tuple(port.name, port.range.slot, port.range.offset,
                                                                    port.range.byteSize, port.input));
                                   return result;
                               })
        .def_property_readonly("required_inputs",
                               [](const rhi::ComputeWorkRecipe &work) {
                                   py::list result;
                                   for (const auto &range : work.GetRequiredInputs())
                                       result.append(py::make_tuple(range.slot, range.offset, range.byteSize));
                                   return result;
                               })
        .def_property_readonly("dispatch_count",
                               [](const rhi::ComputeWorkRecipe &work) { return work.GetDispatches().size(); })
        .def("instantiate", [](const rhi::ComputeWorkRecipe &work,
                               const std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> &values) {
            work.GetHost().Queue().RecordNativeBoundary();
            std::vector<rhi::ComputeWorkInput> inputs;
            for (const auto &[name, buffer] : values)
                inputs.push_back({name, buffer});
            return work.Instantiate(std::move(inputs));
        });

    py::class_<rhi::ComputeWorkInstance, std::shared_ptr<rhi::ComputeWorkInstance>>(m, "_ComputeWorkInstance")
        .def("rebind",
             [](const rhi::ComputeWorkInstance &work,
                const std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> &values) {
                 work.GetRecipe().GetHost().Queue().RecordNativeBoundary();
                 std::vector<rhi::ComputeWorkInput> inputs;
                 for (const auto &[name, buffer] : values)
                     inputs.push_back({name, buffer});
                 return work.Rebind(std::move(inputs));
             })
        .def("buffer", [](const rhi::ComputeWorkInstance &work, size_t slot) { return work.GetBuffers().at(slot); })
        .def(
            "prepare_batch",
            [](const rhi::ComputeWorkInstance &work, py::iterable updates) {
                work.GetRecipe().GetHost().Queue().RecordNativeBoundary();
                return work.PrepareBatch(DecodeComputeUpdates(updates));
            },
            py::arg("updates") = py::tuple());

    py::class_<rhi::ComputeCompletion>(m, "_ComputeCompletion")
        .def_property_readonly(
            "done",
            [](rhi::ComputeCompletion &completion) {
                const auto &host = completion.GetHost();
                if (host.IsAlive())
                    host.Queue().RecordNativeBoundary();
                return completion.IsComplete();
            },
            "Poll this accepted submission on the engine owner thread without waiting")
        .def_property_readonly("retired", &rhi::ComputeCompletion::HasRetired)
        .def(
            "wait",
            [](rhi::ComputeCompletion &completion) {
                const auto &host = completion.GetHost();
                if (host.IsAlive())
                    host.Queue().RecordNativeBoundary();
                completion.Wait();
            },
            "Wait for exactly this accepted submission at an explicit owner-thread synchronization boundary")
        .def_property_readonly(
            "submission_serial",
            [](const rhi::ComputeCompletion &completion) { return completion.GetSubmission().serial; })
        .def_property_readonly("device_elapsed_ns", &rhi::ComputeCompletion::GetDeviceElapsedNanoseconds,
                               "Exact submission's collected GPU command-region time; None when pending or "
                               "unavailable. Does not poll or wait.");

    py::class_<rhi::PreparedComputeBatch, std::shared_ptr<rhi::PreparedComputeBatch>>(m, "_PreparedComputeBatch")
        .def_property_readonly("storage_blocks", &RetainedComputeStorage<rhi::PreparedComputeBatch>)
        .def_property_readonly("payload_bytes", &rhi::PreparedComputeBatch::GetRetainedPayloadBytes)
        .def_property_readonly("pending_staging_bytes", &rhi::PreparedComputeBatch::GetPendingStagingBytes)
        .def_property_readonly("transfer_bytes", &rhi::PreparedComputeBatch::GetTransferBytes)
        .def_property_readonly("execution_signature", &rhi::PreparedComputeBatch::GetExecutionSignature)
        .def_property_readonly("requires_preparation", &rhi::PreparedComputeBatch::RequiresPreparation)
        .def_property_readonly("preparation_bytes", &rhi::PreparedComputeBatch::GetPreparationBytes)
        .def_property_readonly("has_preservation_work", &rhi::PreparedComputeBatch::HasPreservationWork)
        .def(
            "prepare_preservation",
            [](rhi::PreparedComputeBatch &batch, uint64_t maxBytes, bool managed) {
                auto preparation = batch.PreparePreservation(maxBytes, managed);
                return py::make_tuple(preparation.capacityGranted, std::move(preparation.copy));
            },
            py::arg("max_bytes"), py::arg("managed"))
        .def_property_readonly("has_work", &rhi::PreparedComputeBatch::HasWork)
        .def("has_pending_write", &rhi::PreparedComputeBatch::HasPendingWrite, py::arg("buffer"))
        .def("completion", &rhi::PreparedComputeBatch::GetCompletion,
             "Return an accepted submission ticket without retaining this batch or a host lease")
        .def("readback", &rhi::PreparedComputeBatch::Readback, py::arg("index"), py::keep_alive<0, 1>())
        .def("cancel",
             [](rhi::PreparedComputeBatch &batch) {
                 if (batch.GetHost().IsAlive())
                     batch.GetHost().Queue().RecordNativeBoundary();
                 return batch.Cancel();
             })
        .def("submit",
             [](rhi::PreparedComputeBatch &batch) {
                 batch.GetHost().Queue().RecordNativeBoundary();
                 return batch.Submit().serial;
             })
        .def("try_submit",
             [](rhi::PreparedComputeBatch &batch) -> std::optional<uint64_t> {
                 batch.GetHost().Queue().RecordNativeBoundary();
                 const auto ticket = batch.TrySubmit();
                 return ticket ? std::optional{ticket->serial} : std::nullopt;
             })
        .def_property_readonly("submission_serial",
                               [](const rhi::PreparedComputeBatch &batch) { return batch.GetSubmission().serial; });

    py::class_<rhi::ComputeHost>(m, "_ComputeHost")
        .def_property_readonly("shader_target",
                               [](const rhi::ComputeHost &host) {
                                   host.RequireAlive();
                                   switch (host.device.GetShaderSourceLanguage()) {
                                   case rhi::ShaderSourceLanguage::SpirV:
                                       return "spirv";
                                   case rhi::ShaderSourceLanguage::Wgsl:
                                       return "wgsl";
                                   }
                                   throw std::logic_error("Compute device declared an unknown shader source language");
                               })
        .def_property_readonly("shader_float32_preserve",
                               [](rhi::ComputeHost &host) {
                                   host.RequireAlive();
                                   return host.device.GetCapabilities().portable.shaderFloat32Preserve;
                               })
        .def_property_readonly("shader_int64",
                               [](rhi::ComputeHost &host) {
                                   host.RequireAlive();
                                   return host.device.GetCapabilities().portable.shaderInt64;
                               })
        .def_property_readonly("max_dispatch_groups_x",
                               [](const rhi::ComputeHost &host) {
                                   host.RequireAlive();
                                   return host.device.GetCapabilities().limits.maxComputeWorkgroupCount[0];
                               })
        .def("_release_lease", &rhi::ComputeHost::ReleaseLease,
             "Release the engine teardown lease after all Python compute resources are closed")
        .def("collect",
             [](rhi::ComputeHost &host) {
                 host.Queue().RecordNativeBoundary();
                 host.Collect();
                 return host.Queue().GetPendingSubmissionCount();
             })
        .def_property_readonly(
            "queue_pressure",
            [](rhi::ComputeHost &host) {
                host.Queue().RecordNativeBoundary();
                const auto pressure = host.GetQueuePressure();
                py::dict result;
                result["frame_slots"] = pressure.frameSlots;
                result["pending_frames"] = pressure.pendingFrames;
                result["older_pending_frames"] = pressure.olderPendingFrames;
                result["recording_frames"] = pressure.recordingFrames;
                result["compute_submissions"] = pressure.computeSubmissions;
                return result;
            },
            "Non-waiting collection and original GPU pool/fence facts;no submission or graphics slot retirement")
        .def_property_readonly("identity",
                               [](rhi::ComputeHost &host) {
                                   // Leases acquired from one engine are distinct wrapper objects,
                                   // but they borrow the same ordered compute lane.  That queue is
                                   // the compatibility identity for buffers and kernels.
                                   return reinterpret_cast<uintptr_t>(&host.queue);
                               })
        .def(
            "create_buffer",
            [](rhi::ComputeHost &host, uint64_t elementCount, const std::string &scalarType, uint8_t lanes,
               bool managed) {
                host.Queue().RecordNativeBoundary();
                rhi::ComputeScalarType scalar;
                if (scalarType == "float32")
                    scalar = rhi::ComputeScalarType::Float32;
                else if (scalarType == "int32")
                    scalar = rhi::ComputeScalarType::Int32;
                else if (scalarType == "uint32")
                    scalar = rhi::ComputeScalarType::UInt32;
                else if (scalarType == "int64")
                    scalar = rhi::ComputeScalarType::Int64;
                else
                    throw std::invalid_argument("Compute buffer scalar_type must be float32, int32, uint32, or int64");
                const rhi::ComputeBufferDesc desc{elementCount, scalar, lanes};
                return managed ? rhi::ComputeBuffer::TryCreatePrivate(host, desc)
                               : std::make_shared<rhi::ComputeBuffer>(host, desc);
            },
            py::arg("element_count"), py::arg("scalar_type"), py::arg("lanes") = 1, py::kw_only(),
            py::arg("managed") = false, py::keep_alive<0, 1>())
        .def(
            "create_kernel",
            [](rhi::ComputeHost &host, py::bytes spirv, uint32_t bufferBindingCount, uint32_t pushConstantBytes) {
                host.Queue().RecordNativeBoundary();
                const auto words = bindings::DecodeSpirvBytes(spirv, "compute kernel");
                return std::make_shared<rhi::ComputeKernel>(host, words.data(), words.size(), bufferBindingCount,
                                                            pushConstantBytes);
            },
            py::arg("spirv"), py::arg("buffer_binding_count"), py::arg("push_constant_bytes") = 0,
            py::keep_alive<0, 1>())
        .def(
            "create_kernel_with_bindings",
            [](rhi::ComputeHost &host, py::bytes spirv, std::vector<uint32_t> bufferBindings,
               uint32_t pushConstantBytes) {
                host.Queue().RecordNativeBoundary();
                const auto words = bindings::DecodeSpirvBytes(spirv, "compute kernel");
                return std::make_shared<rhi::ComputeKernel>(host, words.data(), words.size(), std::move(bufferBindings),
                                                            pushConstantBytes);
            },
            py::arg("spirv"), py::arg("buffer_bindings"), py::arg("push_constant_bytes") = 0, py::keep_alive<0, 1>())
        .def(
            "_create_kernel_from_compiler",
            [](rhi::ComputeHost &host, py::bytes code, const std::vector<std::pair<uint32_t, std::string>> &bindings,
               uint32_t pushConstantBytes, const std::string &entryPoint, const std::string &target) {
                host.Queue().RecordNativeBoundary();
                std::vector<rhi::ComputeBufferBinding> bufferBindings;
                bufferBindings.reserve(bindings.size());
                for (const auto &[slot, type] : bindings) {
                    if (type == "uniform")
                        bufferBindings.push_back({slot, rhi::BindingType::UniformBuffer});
                    else if (type == "storage")
                        bufferBindings.push_back({slot, rhi::BindingType::StorageBuffer});
                    else if (type == "read_only_storage")
                        bufferBindings.push_back({slot, rhi::BindingType::StorageBuffer, true});
                    else
                        throw py::value_error(
                            "Compiler buffer binding type must be uniform, storage or read_only_storage");
                }
                if (target == "spirv") {
                    const auto words = bindings::DecodeSpirvBytes(code, "compiled compute kernel");
                    return std::make_shared<rhi::ComputeKernel>(
                        host, rhi::ShaderModuleDesc::FromSpirV(words.data(), words.size()), entryPoint,
                        std::move(bufferBindings), pushConstantBytes);
                }
                if (target != "wgsl")
                    throw py::value_error("Compiler target must be spirv or wgsl");
                const std::string text = code;
                return std::make_shared<rhi::ComputeKernel>(host,
                                                            rhi::ShaderModuleDesc::FromWgsl(text.data(), text.size()),
                                                            entryPoint, std::move(bufferBindings), pushConstantBytes);
            },
            py::arg("code"), py::arg("bindings"), py::arg("push_constant_bytes") = 0, py::arg("entry_point") = "main",
            py::arg("target") = "spirv", py::keep_alive<0, 1>())
        .def(
            "prepare_batch",
            [](rhi::ComputeHost &host, py::iterable values, py::iterable updateValues, py::iterable readValues,
               py::iterable copyValues) {
                host.Queue().RecordNativeBoundary();
                return rhi::PrepareComputeBatch(host, DecodeComputeUpdates(updateValues),
                                                DecodeComputeDispatches(values), DecodeComputeReads(readValues),
                                                DecodeComputeCopies(copyValues));
            },
            py::arg("dispatches"), py::arg("updates") = py::tuple(), py::arg("reads") = py::tuple(),
            py::arg("copies") = py::tuple(), py::keep_alive<0, 1>())
        .def(
            "prepare_work",
            [](rhi::ComputeHost &host, py::iterable slots, py::iterable ports, py::iterable dispatches) {
                host.Queue().RecordNativeBoundary();
                return std::make_shared<rhi::ComputeWorkRecipe>(host, DecodeComputeWorkSlots(host, slots),
                                                                DecodeComputeWorkPorts(ports),
                                                                DecodeComputeWorkDispatches(dispatches));
            },
            py::arg("slots"), py::arg("ports"), py::arg("dispatches"), py::keep_alive<0, 1>())
        .def(
            "dispatch_batch",
            [](rhi::ComputeHost &host, py::iterable values, py::iterable updateValues) {
                host.Queue().RecordNativeBoundary();
                rhi::SubmitComputeBatch(host, DecodeComputeUpdates(updateValues), DecodeComputeDispatches(values));
            },
            py::arg("dispatches"), py::arg("updates") = py::tuple())
        .def(
            "dispatch_batch_and_read",
            [](rhi::ComputeHost &host, py::iterable values, py::iterable updateValues, py::iterable readValues) {
                host.Queue().RecordNativeBoundary();
                auto dispatches = DecodeComputeDispatches(values);
                auto updates = DecodeComputeUpdates(updateValues);
                auto reads = DecodeComputeReads(readValues);
                std::vector<std::vector<uint8_t>> payloads;
                {
                    py::gil_scoped_release release;
                    payloads = rhi::SubmitComputeBatchAndRead(host, std::move(updates), std::move(dispatches),
                                                              std::move(reads));
                }
                py::list result;
                for (const auto &payload : payloads)
                    result.append(py::bytes(reinterpret_cast<const char *>(payload.data()), payload.size()));
                return result;
            },
            py::arg("dispatches"), py::arg("updates"), py::arg("reads"))
        .def(
            "set_profiling_enabled",
            [](rhi::ComputeHost &host, bool enabled) { return host.Queue().SetProfilingEnabled(enabled); },
            py::arg("enabled"))
        .def("get_profile",
             [](rhi::ComputeHost &host) {
                 host.Collect();
                 const auto frame = host.Queue().GetProfile();
                 py::dict result;
                 result["serial"] = frame.serial;
                 result["available"] = frame.available;
                 py::dict samples;
                 for (uint32_t i = 0; i < frame.sampleCount; ++i)
                     samples[py::str(frame.samples[i].Name())] = frame.samples[i].milliseconds;
                 result["samples_ms"] = std::move(samples);
                 return result;
             })
        .def_property_readonly("retained_upload_bytes",
                               [](const rhi::ComputeHost &host) {
                                   if (!host.IsAlive())
                                       throw std::runtime_error("Compute host has retired");
                                   return host.Queue().GetRetainedUploadBytes();
                               })
        .def("get_statistics",
             [](rhi::ComputeHost &host) {
                 host.Collect();
                 const auto statistics = host.Queue().GetStatistics();
                 const auto profile = host.Queue().GetProfile();
                 py::dict result;
                 result["submission_count"] = statistics.submissionCount;
                 result["dispatch_count"] = statistics.dispatchCount;
                 result["upload_request_count"] = statistics.uploadRequestCount;
                 result["upload_bytes"] = statistics.uploadBytes;
                 result["copy_request_count"] = statistics.copyRequestCount;
                 result["copy_bytes"] = statistics.copyBytes;
                 result["readback_request_count"] = statistics.readbackRequestCount;
                 result["readback_bytes"] = statistics.readbackBytes;
                 result["staging_allocation_count"] = statistics.stagingAllocationCount;
                 result["host_map_count"] = statistics.hostMapCount;
                 result["native_boundary_count"] = statistics.nativeBoundaryCount;
                 result["wait_count"] = statistics.waitCount;
                 result["cpu_submit_ms"] = statistics.cpuSubmitMilliseconds;
                 result["wait_ms"] = statistics.waitMilliseconds;
                 result["pending_submission_count"] = host.Queue().GetPendingSubmissionCount();
                 result["gpu_profile_available"] = profile.available;
                 result["gpu_profile_serial"] = profile.serial;
                 double gpuMilliseconds = 0.0;
                 for (uint32_t i = 0; i < profile.sampleCount; ++i)
                     gpuMilliseconds += profile.samples[i].milliseconds;
                 result["gpu_time_ms"] = profile.available ? py::cast(gpuMilliseconds) : py::none();
                 return result;
             })
        .def("reset_statistics", [](rhi::ComputeHost &host) { host.Queue().ResetStatistics(); });
}
