#include "InxPackPyBridge.h"
#include "JsonPyBridge.h"
#include "MaterialPropertyName.h"
#include "MatrixPyBridge.h"
#include "ResourceMetaPyView.h"
#include <array>
#include <core/threading/FrameWorkOwner.h>
#include <fstream>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/AssetRegistry/AssetSource.h>
#include <function/resources/InxFileLoader/InxTextureLoader.hpp>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/InxTexture/TextureLoader.h>
#include <function/resources/InxTexture/TexturePixels.h>
#include <function/resources/PhysicMaterial/PhysicMaterial.h>
#include <iomanip>
#include <platform/filesystem/AssetDocument.h>
#include <platform/filesystem/AtomicFile.h>
#include <platform/filesystem/DocumentStore.h>
#include <platform/filesystem/InxPack.h>
#include <platform/filesystem/InxPackTask.h>
#include <platform/filesystem/InxPath.h>
#include <platform/filesystem/NativeModule.h>
#include <platform/filesystem/VerifiedFile.h>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "BindingRegistration.h"

namespace py = pybind11;

namespace infernux
{

namespace
{

/// Shared conversion from one reflected material property to its Python value.
///
/// The single-property and bulk getters both use this so their two switches
/// cannot drift apart. The returned shapes are a public binding contract:
/// Mat4 is a flat 16-float list in GLM column-major order, Float4Array is a
/// list of 4-tuples, and Color shares the Float4 branch.
py::object PyValueFromMaterialProperty(const MaterialProperty &prop)
{
    switch (prop.type) {
    case MaterialPropertyType::Float:
        return py::cast(std::get<float>(prop.value));
    case MaterialPropertyType::Float2:
        return py::cast(std::get<glm::vec2>(prop.value));
    case MaterialPropertyType::Float3:
        return py::cast(std::get<glm::vec3>(prop.value));
    case MaterialPropertyType::Float4:
    case MaterialPropertyType::Color:
        return py::cast(std::get<glm::vec4>(prop.value));
    case MaterialPropertyType::Int:
        return py::cast(std::get<int>(prop.value));
    case MaterialPropertyType::Mat4: {
        // mat4 is not registered as a pybind11 type: expose the GLM storage
        // order (column-major) as 16 floats.
        auto &matrix = std::get<glm::mat4>(prop.value);
        py::list result;
        const float *data = &matrix[0][0];
        for (int index = 0; index < 16; ++index)
            result.append(data[index]);
        return result;
    }
    case MaterialPropertyType::Texture2D:
        return py::cast(std::get<std::string>(prop.value));
    case MaterialPropertyType::FloatArray:
        return py::cast(std::get<std::vector<float>>(prop.value));
    case MaterialPropertyType::Float4Array: {
        py::list result;
        for (const auto &value : std::get<std::vector<glm::vec4>>(prop.value))
            result.append(py::make_tuple(value.x, value.y, value.z, value.w));
        return result;
    }
    }
    return py::none();
}

template <typename Value, typename... Options, typename Access>
void BindMetadataReadMethods(py::class_<Value, Options...> &binding, Access access)
{
    binding.def("get_resource_name", [access](const Value &self) { return access(self).GetResourceName(); })
        .def("get_guid", [access](const Value &self) { return access(self).GetGuid(); })
        .def("get_resource_type", [access](const Value &self) { return access(self).GetResourceType(); })
        .def(
            "has_key", [access](const Value &self, const std::string &key) { return access(self).HasKey(key); },
            py::arg("key"))
        .def(
            "get_string",
            [access](const Value &self, const std::string &key) {
                const InxResourceMeta &meta = access(self);
                return meta.HasKey(key) ? meta.GetDataAs<std::string>(key) : std::string{};
            },
            py::arg("key"))
        .def(
            "get_int",
            [access](const Value &self, const std::string &key) {
                const InxResourceMeta &meta = access(self);
                return meta.HasKey(key) ? meta.GetDataAs<int>(key) : 0;
            },
            py::arg("key"))
        .def(
            "get_float",
            [access](const Value &self, const std::string &key) {
                const InxResourceMeta &meta = access(self);
                return meta.HasKey(key) ? meta.GetDataAs<float>(key) : 0.0f;
            },
            py::arg("key"))
        .def(
            "get_bool",
            [access](const Value &self, const std::string &key) {
                const InxResourceMeta &meta = access(self);
                return meta.HasKey(key) ? meta.GetDataAs<bool>(key) : false;
            },
            py::arg("key"))
        .def("serialize_document",
             [access](const Value &self) { return JsonToPython(access(self).SerializeDocument()); })
        .def(
            "serialize_document_portable",
            [access](const Value &self, const std::string &projectRoot) {
                return JsonToPython(access(self).SerializeDocumentPortable(projectRoot));
            },
            py::arg("project_root"));
}

py::object ShaderReferenceToPython(const ShaderAssetReference &reference)
{
    return JsonToPython({
        {"guid", reference.guid},
        {"shader_id", reference.shaderId},
        {"path_hint", reference.pathHint},
    });
}

ShaderAssetReference ShaderReferenceFromPython(py::handle value)
{
    const nlohmann::json document = PythonToJson(value);
    if (!document.is_object())
        throw std::invalid_argument("shader reference must be a dictionary");
    for (const char *field : {"guid", "shader_id", "path_hint"}) {
        if (!document.contains(field) || !document[field].is_string())
            throw std::invalid_argument(std::string("shader reference field '") + field + "' must be a string");
    }
    ShaderAssetReference reference{
        document["guid"].get<std::string>(),
        document["shader_id"].get<std::string>(),
        document["path_hint"].get<std::string>(),
    };
    if (!reference.IsAssigned())
        throw std::invalid_argument("shader reference requires guid or shader_id");
    return reference;
}

} // namespace

void RegisterResourceBindings(py::module_ &m)
{
    py::tuple shaderSourceKinds(asset_formats::kShaderSourceKinds.size());
    size_t sourceKindIndex = 0;
    for (const auto &kind : asset_formats::kShaderSourceKinds)
        shaderSourceKinds[sourceKindIndex++] = py::make_tuple(kind.suffix, kind.stage, kind.icon, kind.dragPayload);
    m.attr("_shader_source_kinds") = std::move(shaderSourceKinds);

    py::register_exception<PathWriteBusy>(m, "PathWriteBusy", PyExc_RuntimeError);
    py::class_<PathWriteScope>(m, "PathWriteScope")
        .def(py::init<>())
        .def_static("begin_write", &PathWriteScope::BeginWrite, py::arg("paths"), py::arg("parent") = PathWriteScope{})
        .def_static("begin_mutation", &PathWriteScope::BeginMutation, py::arg("paths"),
                    py::arg("directories") = std::vector<std::string>{}, py::arg("parent") = PathWriteScope{})
        .def_property_readonly("is_open", &PathWriteScope::IsOpen)
        .def("covers", &PathWriteScope::Covers)
        .def("close", &PathWriteScope::Close)
        .def(
            "__enter__", [](PathWriteScope &scope) -> PathWriteScope & { return scope; },
            py::return_value_policy::reference_internal)
        .def("__exit__", [](PathWriteScope &scope, py::object, py::object, py::object) { scope.Close(); });

    m.def(
        "_asset_source_stat",
        [](const std::string &path) {
            const auto file = ToFsPath(path);
            if (!std::filesystem::is_regular_file(file))
                throw std::runtime_error("Asset source is not a file: " + path);
            return std::make_pair(
                std::filesystem::file_size(file),
                static_cast<int64_t>(std::filesystem::last_write_time(file).time_since_epoch().count()));
        },
        py::call_guard<py::gil_scoped_release>());
    m.def(
        "_asset_source_content_hash",
        [](const std::string &path) {
            const auto file = ToFsPath(path);
            const auto size = std::filesystem::file_size(file);
            const auto modified = std::filesystem::last_write_time(file);
            AssetSource source(path);
            const auto hash = source.ContentHash();
            if (size != std::filesystem::file_size(file) || modified != std::filesystem::last_write_time(file))
                throw std::runtime_error("Asset source changed or could not be read: " + path);
            return hash;
        },
        py::call_guard<py::gil_scoped_release>());
    m.def("_encode_asset_document",
          [](py::handle value) { return py::bytes(EncodeAssetDocument(PythonToJson(value))); });
    m.def("_decode_asset_document", [](const py::bytes &value) {
        const std::string bytes = value;
        return JsonToPython(DecodeAssetDocument(bytes));
    });
    m.def("_read_asset_document", [](const std::string &path) { return JsonToPython(ReadAssetDocument(path)); });
    binding::BindPackFunctions(m);
    py::class_<IoWork>(m, "_IoWork")
        .def_readonly("operation", &IoWork::operation)
        .def_readonly("bytes", &IoWork::bytes)
        .def_readonly("remaining_bytes", &IoWork::remainingBytes)
        .def_readonly("parameter", &IoWork::parameter);
    py::class_<IoSpan>(m, "_IoSpan")
        .def_readonly("kind", &IoSpan::kind)
        .def_readonly("operation", &IoSpan::operation)
        .def_readonly("entry_index", &IoSpan::entryIndex)
        .def_readonly("parameter", &IoSpan::parameter)
        .def_readonly("input_bytes", &IoSpan::inputBytes)
        .def_readonly("output_bytes", &IoSpan::outputBytes)
        .def_readonly("finish_operation", &IoSpan::finishOperation)
        .def_readonly("entry_bytes", &IoSpan::entryBytes);
    py::class_<VerifiedFile, std::shared_ptr<VerifiedFile>>(m, "_VerifiedFile")
        .def_property_readonly("path", [](const VerifiedFile &file) { return FromFsPath(file.Path()); })
        .def_property_readonly("sha256", &VerifiedFile::Sha256)
        .def_property_readonly("byte_size", &VerifiedFile::Bytes)
        .def_property_readonly("retained_bytes", &VerifiedFile::RetainedBytes);
#if defined(INX_PLATFORM_WINDOWS) && (defined(_M_X64) || defined(__x86_64__))
    m.attr("_native_cpu_program_target") = "windows-x86_64";
#elif defined(INX_PLATFORM_LINUX) && defined(__x86_64__)
    m.attr("_native_cpu_program_target") = "linux-x86_64";
#else
    m.attr("_native_cpu_program_target") = py::none();
#endif
#if defined(INX_PLATFORM_WINDOWS) || defined(INX_PLATFORM_LINUX)
    py::class_<NativeModule, std::shared_ptr<NativeModule>>(m, "_NativeModule")
        .def_static("load", &NativeModule::Load, py::arg("file"), py::arg("symbols"),
                    py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("identity", &NativeModule::Identity)
        .def_property_readonly("image_bytes", &NativeModule::ImageBytes)
        .def_property_readonly("published", &NativeModule::IsPublished)
        .def("address", &NativeModule::Address, py::arg("symbol"))
        .def("require_published", &NativeModule::RequirePublished)
        .def("publish", &NativeModule::Publish)
        .def("discard_unpublished", &NativeModule::Discard, py::call_guard<py::gil_scoped_release>());
    py::class_<NativeModuleTask>(m, "_NativeModuleTask")
        .def(py::init<std::shared_ptr<VerifiedFile>, std::vector<std::string>>(), py::arg("file"), py::arg("symbols"))
        .def("advance", &NativeModuleTask::Advance, py::arg("max_bytes"), py::call_guard<py::gil_scoped_release>())
        .def("describe_work", &NativeModuleTask::DescribeWork, py::arg("max_bytes"))
        .def("describe_remaining_work", &NativeModuleTask::DescribeRemainingWork)
        .def_property_readonly("submitted_work", &NativeModuleTask::GetSubmittedWork)
        .def_property_readonly("module", &NativeModuleTask::GetModule)
        .def("commit", &NativeModuleTask::Commit)
        .def("cancel", &NativeModuleTask::Cancel, py::call_guard<py::gil_scoped_release>())
        .def("wait", &NativeModuleTask::Wait, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("pending", &NativeModuleTask::IsPending)
        .def_property_readonly("done", &NativeModuleTask::IsComplete)
        .def_property_readonly("state", &NativeModuleTask::GetStateName)
        .def_property_readonly("error", &NativeModuleTask::GetError)
        .def_property_readonly("retained_bytes", &NativeModuleTask::GetRetainedBytes)
        .def_property_readonly("reservation_bytes", &NativeModuleTask::GetReservationBytes)
        .def_property_readonly("image_bytes", &NativeModuleTask::GetImageBytes)
        .def_property_readonly("module_identity", &NativeModuleTask::GetModuleIdentity)
        .def_property_readonly("read_bytes", &NativeModuleTask::GetReadBytes)
        .def_property_readonly("load_ns", &NativeModuleTask::GetLoadNs)
        .def_property_readonly("execution_ns", &NativeModuleTask::GetExecutionNs)
        .def_property_readonly("submissions", &NativeModuleTask::GetSubmissionCount)
        .def_property_readonly("ran_on_worker", &NativeModuleTask::RanOnWorker);
#endif
    py::class_<inxpack::WriteTask>(m, "_InxPackWriteTask")
        .def(py::init([](const std::string &path, py::sequence entries, size_t blockBytes, std::string textHeader,
                         const PathWriteScope &writeScope, const py::tuple &residentInputs) {
                 if (residentInputs.size() && residentInputs.size() != entries.size())
                     throw std::invalid_argument("resident input slots must match the InxPack write entries");
                 auto pins = std::make_shared<std::vector<py::buffer_info>>();
                 std::vector<inxpack::WriteEntry> sources;
                 sources.reserve(entries.size());
                 for (const auto item : entries) {
                     const auto row = py::cast<py::tuple>(item);
                     if (row.size() != 3)
                         throw std::invalid_argument("InxPack write entry requires (path, byte_count, store)");
                     sources.push_back(
                         {py::cast<std::string>(row[0]), py::cast<uint64_t>(row[1]), py::cast<bool>(row[2])});
                     if (residentInputs.size()) {
                         const auto buffer = residentInputs[sources.size() - 1];
                         if (buffer.is_none())
                             continue;
                         auto info = bindings::ContiguousBuffer(buffer);
                         if (!info.readonly || static_cast<uint64_t>(info.size * info.itemsize) != sources.back().bytes)
                             throw std::invalid_argument("resident input must be read-only and match its entry size");
                         sources.back().residentData = static_cast<const uint8_t *>(info.ptr);
                         pins->push_back(std::move(info));
                     }
                 }
                 inxpack::WriteOptions options;
                 options.writeScope = writeScope;
                 if (!textHeader.empty())
                     options.textHeader = std::make_shared<const std::string>(std::move(textHeader));
                 return std::make_unique<inxpack::WriteTask>(ToFsPath(path), std::move(sources), options, blockBytes,
                                                             std::move(pins));
             }),
             py::arg("path"), py::arg("entries"), py::arg("block_bytes") = 64u * 1024u, py::arg("text_header") = "",
             py::kw_only(), py::arg("write_scope") = PathWriteScope{}, py::arg("resident_inputs") = py::tuple())
        .def("append",
             [](inxpack::WriteTask &task, py::handle input) {
                 auto info = bindings::ContiguousBuffer(input);
                 py::gil_scoped_release release;
                 task.Append(static_cast<const uint8_t *>(info.ptr), static_cast<size_t>(info.size * info.itemsize));
             })
        .def("advance", &inxpack::WriteTask::Advance, py::arg("max_bytes"), py::arg("target_ns") = 0,
             py::arg("continuous") = false, py::call_guard<py::gil_scoped_release>())
        .def("describe_work", &inxpack::WriteTask::DescribeWork, py::arg("max_bytes"), py::arg("publish") = false)
        .def("describe_remaining_work", &inxpack::WriteTask::DescribeRemainingWork)
        .def_property_readonly("submitted_work", &inxpack::WriteTask::GetSubmittedWork)
        .def("publish", &inxpack::WriteTask::Publish, py::call_guard<py::gil_scoped_release>())
        .def("cancel", &inxpack::WriteTask::Cancel, py::call_guard<py::gil_scoped_release>())
        .def("wait", &inxpack::WriteTask::Wait, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("pending", &inxpack::WriteTask::IsPending)
        .def_property_readonly("done", &inxpack::WriteTask::IsComplete)
        .def_property_readonly("state", &inxpack::WriteTask::GetStateName)
        .def_property_readonly("error", &inxpack::WriteTask::GetError)
        .def_property_readonly("input_path", &inxpack::WriteTask::GetInputPath)
        .def_property_readonly("input_bytes", &inxpack::WriteTask::GetInputBytes)
        .def_property_readonly("retained_bytes", &inxpack::WriteTask::GetRetainedBytes)
        .def_property_readonly("reservation_bytes", &inxpack::WriteTask::GetReservationBytes)
        .def_property_readonly("execution_ns", &inxpack::WriteTask::GetExecutionNs)
        .def_property_readonly("submissions", &inxpack::WriteTask::GetSubmissionCount)
        .def_property_readonly("ran_on_worker", &inxpack::WriteTask::RanOnWorker)
        .def_property_readonly("progress",
                               [](const inxpack::WriteTask &task) {
                                   const auto progress = task.GetProgress();
                                   py::dict result;
                                   result["queued_bytes"] = progress.queuedBytes;
                                   result["consumed_bytes"] = progress.consumedBytes;
                                   result["written_bytes"] = progress.writtenBytes;
                                   result["hashed_bytes"] = progress.hashedBytes;
                                   return result;
                               })
        .def_property_readonly("manifest", [](const inxpack::WriteTask &task) {
            return binding::InxPackManifestToPython(task.GetManifest());
        });
    using PackReadBuffer = inxpack::ReadTask::ReadBuffer;
    py::class_<PackReadBuffer, std::shared_ptr<PackReadBuffer>>(m, "_InxPackReadBuffer", py::buffer_protocol())
        .def_buffer([](PackReadBuffer &buffer) {
            return py::buffer_info(buffer.data.get(), static_cast<py::ssize_t>(buffer.bytes));
        });
    py::class_<inxpack::ReadTask>(m, "_InxPackReadTask")
        .def(py::init([](const std::string &path, size_t metadataLimit, size_t blockBytes) {
                 return std::make_unique<inxpack::ReadTask>(ToFsPath(path), metadataLimit, blockBytes);
             }),
             py::arg("path"), py::arg("metadata_bytes"), py::arg("block_bytes") = 64u * 1024u)
        .def("advance", &inxpack::ReadTask::Advance, py::arg("max_bytes"), py::arg("target_ns") = 0,
             py::arg("directory_extra_bytes") = 0, py::call_guard<py::gil_scoped_release>())
        .def("describe_work", &inxpack::ReadTask::DescribeWork, py::arg("max_bytes"))
        .def("describe_remaining_work", &inxpack::ReadTask::DescribeRemainingWork)
        .def("describe_read_work", &inxpack::ReadTask::DescribeReadWork, py::arg("entry"), py::arg("windowed") = false)
        .def_property_readonly("submitted_work", &inxpack::ReadTask::GetSubmittedWork)
        .def("begin_checksum", &inxpack::ReadTask::BeginChecksum)
        .def("begin_read_buffer", &inxpack::ReadTask::BeginReadBuffer, py::arg("entry"), py::arg("owner"),
             py::arg("reserved") = nullptr, py::call_guard<py::gil_scoped_release>())
        .def("begin_read_stream", &inxpack::ReadTask::BeginReadStream, py::arg("entry"))
        .def("reserve_read_window", &inxpack::ReadTask::ReserveReadWindow, py::arg("owner"), py::arg("maximum"),
             py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("read_buffer_remaining_bytes", &inxpack::ReadTask::ReadBufferRemainingBytes)
        .def("take_read_buffer", &inxpack::ReadTask::TakeReadBuffer)
        .def(
            "begin_materialize",
            [](inxpack::ReadTask &task, const std::string &entry, const std::string &destination,
               const std::string &sha256) { task.BeginMaterialize(entry, ToFsPath(destination), sha256); },
            py::arg("entry"), py::arg("destination"), py::arg("sha256"))
        .def("take_materialized_file", &inxpack::ReadTask::TakeMaterializedFile)
        .def("close", &inxpack::ReadTask::Close, py::call_guard<py::gil_scoped_release>())
        .def("cancel", &inxpack::ReadTask::Cancel)
        .def("wait", &inxpack::ReadTask::Wait, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("pending", &inxpack::ReadTask::IsPending)
        .def_property_readonly("done", &inxpack::ReadTask::IsComplete)
        .def_property_readonly("state", &inxpack::ReadTask::GetStateName)
        .def_property_readonly("error", &inxpack::ReadTask::GetError)
        .def_property_readonly("checksum_verified", &inxpack::ReadTask::IsChecksumVerified)
        .def_property_readonly("retained_bytes", &inxpack::ReadTask::GetRetainedBytes)
        .def_property_readonly("reservation_bytes", &inxpack::ReadTask::GetReservationBytes)
        .def_property_readonly("execution_ns", &inxpack::ReadTask::GetExecutionNs)
        .def_property_readonly("submissions", &inxpack::ReadTask::GetSubmissionCount)
        .def_property_readonly("ran_on_worker", &inxpack::ReadTask::RanOnWorker)
        .def_property_readonly("progress",
                               [](const inxpack::ReadTask &task) {
                                   const auto progress = task.GetProgress();
                                   py::dict result;
                                   result["read_bytes"] = progress.readBytes;
                                   result["hashed_bytes"] = progress.hashedBytes;
                                   result["codec_input_bytes"] = progress.codecInputBytes;
                                   result["decoded_bytes"] = progress.decodedBytes;
                                   result["consumed_bytes"] = progress.consumedBytes;
                                   result["written_bytes"] = progress.writtenBytes;
                                   result["buffered_bytes"] = progress.bufferedBytes;
                                   result["buffer_work_bytes"] = progress.bufferWorkBytes;
                                   result["overlapped_reads"] = progress.overlappedReads;
                                   result["pending_overlapped_reads"] = progress.pendingOverlappedReads;
                                   result["range_completion_ns"] = progress.fileCompletionNs;
                                   return result;
                               })
        .def_property_readonly("manifest", [](const inxpack::ReadTask &task) {
            return binding::InxPackManifestToPython(task.GetManifest());
        });
    py::class_<inxpack::Reader>(m, "_InxPackReader")
        .def(py::init([](const std::string &path) { return std::make_unique<inxpack::Reader>(ToFsPath(path)); }))
        .def_property_readonly(
            "manifest",
            [](const inxpack::Reader &reader) { return binding::InxPackManifestToPython(reader.GetManifest()); })
        .def("verify_checksum", &inxpack::Reader::VerifyChecksum, py::call_guard<py::gil_scoped_release>())
        .def("read",
             [](inxpack::Reader &reader, const std::string &name) {
                 std::vector<uint8_t> bytes;
                 {
                     py::gil_scoped_release release;
                     bytes = reader.Read(name);
                 }
                 return py::bytes(reinterpret_cast<const char *>(bytes.data()), bytes.size());
             })
        .def("read_into",
             [](inxpack::Reader &reader, const std::string &name, py::handle output) {
                 auto info = bindings::ContiguousBuffer(output, true);
                 py::gil_scoped_release release;
                 reader.ReadInto(name, static_cast<uint8_t *>(info.ptr),
                                 static_cast<size_t>(info.size * info.itemsize));
             })
        .def("extract_entry",
             [](inxpack::Reader &reader, const std::string &name, const std::string &destination,
                const std::string &expectedSha256) {
                 py::gil_scoped_release release;
                 return reader.ExtractEntry(name, ToFsPath(destination), expectedSha256);
             })
        .def("close", &inxpack::Reader::Close);
    m.def("_copy_file_verified",
          [](const std::string &source, const std::string &destination, const std::string &expectedSha256) {
              py::gil_scoped_release release;
              return inxpack::CopyVerifiedFile(ToFsPath(source), ToFsPath(destination), expectedSha256);
          });
    py::class_<FileInput>(m, "_FileInput")
        .def(py::init([](const std::string &path) { return std::make_unique<FileInput>(ToFsPath(path)); }))
        .def_property_readonly("size", &FileInput::Size)
        .def("seek", &FileInput::Seek)
        .def("read",
             [](FileInput &file, size_t bytes) {
                 std::string data(bytes, '\0');
                 {
                     py::gil_scoped_release release;
                     file.Read(data.data(), bytes);
                 }
                 return py::bytes(data);
             })
        .def(
            "sha256", [](FileInput &file) { return inxpack::HashToHex(inxpack::HashFile(file)); },
            py::call_guard<py::gil_scoped_release>())
        .def(
            "__enter__", [](FileInput &file) -> FileInput & { return file; },
            py::return_value_policy::reference_internal)
        .def("__exit__", [](FileInput &file, py::object, py::object, py::object) { file.Close(); });
    py::register_exception<DocumentWriteSuperseded>(m, "DocumentWriteSuperseded");
    py::register_exception<DocumentWriteCancelled>(m, "DocumentWriteCancelled");

    py::class_<DocumentFileState>(m, "DocumentFileState")
        .def(py::init<>())
        .def_readwrite("exists", &DocumentFileState::exists)
        .def_readwrite("size", &DocumentFileState::size)
        .def_readwrite("modified_ns", &DocumentFileState::modifiedNs)
        .def_readwrite("content_hash", &DocumentFileState::contentHash);

    py::enum_<FileWriteMode>(m, "FileWriteMode")
        .value("Replace", FileWriteMode::Replace)
        .value("Create", FileWriteMode::Create);
    py::enum_<FilePublication>(m, "FilePublication")
        .value("Written", FilePublication::Written)
        .value("AlreadyExists", FilePublication::AlreadyExists);

    py::class_<DocumentWriteOptions>(m, "DocumentWriteOptions")
        .def(py::init<>())
        .def_readwrite("mode", &DocumentWriteOptions::mode)
        .def_readwrite("expected_file_state", &DocumentWriteOptions::expectedFileState)
        .def_readwrite("commit_chain_token", &DocumentWriteOptions::commitChainToken)
        .def_readwrite("write_scope", &DocumentWriteOptions::writeScope)
        // Keep the short name available to Python tooling while the native
        // field remains explicit about its role in the write chain.
        .def_readwrite("chain_id", &DocumentWriteOptions::commitChainToken);

    py::class_<DocumentPathMetrics>(m, "DocumentPathMetrics")
        .def_readonly("latest_submitted_generation", &DocumentPathMetrics::latestSubmittedGeneration)
        .def_readonly("latest_succeeded_generation", &DocumentPathMetrics::latestSucceededGeneration)
        .def_readonly("latest_failed_generation", &DocumentPathMetrics::latestFailedGeneration)
        .def_readonly("pending_generation", &DocumentPathMetrics::pendingGeneration)
        .def_readonly("active_generation", &DocumentPathMetrics::activeGeneration);

    py::class_<DocumentWriteTicket, std::shared_ptr<DocumentWriteTicket>>(m, "DocumentWriteTicket")
        .def_property_readonly("path", &DocumentWriteTicket::GetPath)
        .def_property_readonly("generation", &DocumentWriteTicket::GetGeneration)
        .def_property_readonly("is_complete", &DocumentWriteTicket::IsComplete)
        .def_property_readonly("status", &DocumentWriteTicket::GetStatusName)
        .def_property_readonly("error", &DocumentWriteTicket::GetError)
        .def_property_readonly("committed_file_state", &DocumentWriteTicket::GetCommittedFileState)
        .def_property_readonly("publication", &DocumentWriteTicket::GetPublication)
        .def("wait", &DocumentWriteTicket::Wait, py::call_guard<py::gil_scoped_release>());

    py::class_<DocumentStore>(m, "NativeDocumentStore")
        .def_static("is_temporary_path", [](const std::string &path) { return IsAtomicTemporaryPath(ToFsPath(path)); })
        .def_static("instance", &DocumentStore::Instance, py::return_value_policy::reference)
        .def("submit", &DocumentStore::Submit, py::arg("path"), py::arg("content"),
             py::arg("options") = DocumentWriteOptions{}, py::call_guard<py::gil_scoped_release>())
        .def(
            "write_and_wait",
            [](DocumentStore &store, const std::string &path, py::object content, DocumentWriteOptions options) {
                std::string snapshot;
                std::string_view view;
                if (PyUnicode_Check(content.ptr())) {
                    Py_ssize_t size = 0;
                    const auto *bytes = PyUnicode_AsUTF8AndSize(content.ptr(), &size);
                    if (!bytes)
                        throw py::error_already_set();
                    view = {bytes, static_cast<size_t>(size)};
                } else if (PyBytes_Check(content.ptr())) {
                    view = {PyBytes_AS_STRING(content.ptr()), static_cast<size_t>(PyBytes_GET_SIZE(content.ptr()))};
                } else {
                    snapshot = content.cast<std::string>();
                    view = snapshot;
                }
                // The binding keeps the Python owner alive through the synchronous write.
                py::gil_scoped_release release;
                return store.WriteAndWait(path, view, std::move(options));
            },
            py::arg("path"), py::arg("content"), py::arg("options") = DocumentWriteOptions{})
        .def("cancel", &DocumentStore::Cancel, py::arg("ticket"))
        .def("get_metrics", &DocumentStore::GetMetrics, py::arg("path"))
        .def("capture_file_state", &DocumentStore::CaptureFileState, py::arg("path"))
        .def_static(
            "read_text_snapshot",
            [](const std::string &path) -> py::tuple {
                if (!std::filesystem::exists(ToFsPath(path)))
                    return py::make_tuple(py::none(), DocumentFileState{});
                const auto snapshot = ReadTextFileSnapshot(path);
                return py::make_tuple(snapshot.content, snapshot.state);
            },
            py::arg("path"), "Read UTF-8 content and its consumed file state; return None for an absent file")
        .def_property_readonly("is_idle", &DocumentStore::IsIdle, "Whether all queued document writes have completed")
        .def("flush_all", py::overload_cast<>(&DocumentStore::Flush), py::call_guard<py::gil_scoped_release>())
        .def("flush_path", py::overload_cast<const std::string &>(&DocumentStore::Flush), py::arg("path"),
             py::call_guard<py::gil_scoped_release>())
        .def("shutdown", &DocumentStore::Shutdown, py::call_guard<py::gil_scoped_release>());

    // ResourceType enum
    py::enum_<ResourceType>(m, "ResourceType")
        .value("Meta", ResourceType::Meta)
        .value("Shader", ResourceType::Shader)
        .value("Texture", ResourceType::Texture)
        .value("Mesh", ResourceType::Mesh)
        .value("Material", ResourceType::Material)
        .value("Script", ResourceType::Script)
        .value("Audio", ResourceType::Audio)
        .value("DefaultText", ResourceType::DefaultText)
        .value("DefaultBinary", ResourceType::DefaultBinary)
        .value("PhysicMaterial", ResourceType::PhysicMaterial)
        .value("RenderEffect", ResourceType::RenderEffect)
        .value("ParticleGraph", ResourceType::ParticleGraph)
        .value("DataAsset", ResourceType::DataAsset)
        .value("RenderTexture", ResourceType::RenderTexture);

    // Independent authoring values are editable; database queries expose only
    // a read view of the retained published generation.
    auto metadata = py::class_<InxResourceMeta, std::shared_ptr<InxResourceMeta>>(m, "ResourceMeta");
    BindMetadataReadMethods(metadata, [](const InxResourceMeta &self) -> const InxResourceMeta & { return self; });
    metadata.def(py::init<>())
        .def(
            "deserialize_document",
            [](InxResourceMeta &self, const py::object &document) { self.DeserializeDocument(PythonToJson(document)); },
            py::arg("document"));
    auto metadataView = py::class_<ResourceMetaView>(m, "ResourceMetaView");
    BindMetadataReadMethods(metadataView,
                            [](const ResourceMetaView &self) -> const InxResourceMeta & { return self.Get(); });

    py::class_<PhysicMaterial, std::shared_ptr<PhysicMaterial>>(m, "InxPhysicMaterial")
        .def(py::init<>())
        .def_property_readonly("name", &PhysicMaterial::GetName)
        .def_property_readonly("guid", &PhysicMaterial::GetGuid)
        .def_property_readonly("file_path", &PhysicMaterial::GetFilePath)
        .def_property("friction", &PhysicMaterial::GetFriction, &PhysicMaterial::SetFriction)
        .def_property("bounciness", &PhysicMaterial::GetBounciness, &PhysicMaterial::SetBounciness)
        .def_property(
            "friction_combine",
            [](const PhysicMaterial &material) { return static_cast<int>(material.GetFrictionCombine()); },
            [](PhysicMaterial &material, int value) {
                if (value < 0 || value > static_cast<int>(PhysicsMaterialCombine::Maximum))
                    throw std::invalid_argument("PhysicMaterial friction_combine is invalid");
                material.SetFrictionCombine(static_cast<PhysicsMaterialCombine>(value));
            })
        .def_property(
            "bounce_combine",
            [](const PhysicMaterial &material) { return static_cast<int>(material.GetBounceCombine()); },
            [](PhysicMaterial &material, int value) {
                if (value < 0 || value > static_cast<int>(PhysicsMaterialCombine::Maximum))
                    throw std::invalid_argument("PhysicMaterial bounce_combine is invalid");
                material.SetBounceCombine(static_cast<PhysicsMaterialCombine>(value));
            })
        .def("serialize_document",
             [](const PhysicMaterial &material) { return JsonToPython(material.SerializeDocument()); })
        .def("deserialize_document", [](PhysicMaterial &material,
                                        py::handle document) { material.DeserializeDocument(PythonToJson(document)); })
        .def("save", py::overload_cast<>(&PhysicMaterial::SaveToFile, py::const_))
        .def("save_to", py::overload_cast<const std::string &>(&PhysicMaterial::SaveToFile), py::arg("path"));

    // InxTextureData - raw texture data accessible from Python
    py::class_<InxTextureData>(m, "TextureData")
        .def(py::init<>())
        .def_readonly("width", &InxTextureData::width, "Texture width in pixels")
        .def_readonly("height", &InxTextureData::height, "Texture height in pixels")
        .def_readonly("channels", &InxTextureData::channels, "Number of color channels (always 4 for RGBA)")
        .def_readonly("name", &InxTextureData::name, "Texture name/identifier")
        .def_readonly("source_path", &InxTextureData::sourcePath, "Original file path")
        .def("is_valid", &InxTextureData::IsValid, "Check if texture data is valid")
        .def("get_size_bytes", &InxTextureData::GetSizeBytes, "Get total size in bytes")
        .def(
            "get_pixels",
            [](const InxTextureData &self) {
                // Return pixels as bytes for Python access
                return py::bytes(reinterpret_cast<const char *>(self.pixels.data()), self.pixels.size());
            },
            "Get raw pixel data as bytes (RGBA format)")
        .def(
            "get_pixels_list",
            [](const InxTextureData &self) {
                // Return pixels as RGBA8 bytes for submit_imgui_texture.
                return self.pixels;
            },
            "Get raw RGBA8 pixel data for submit_imgui_texture");

    // InxTextureLoader - static methods for loading textures
    py::class_<InxTextureLoader>(m, "TextureLoader")
        .def_static(
            "load_asset",
            [](const std::string &guid) -> std::optional<InxTextureData> {
                auto *database = AssetRegistry::Instance().GetAssetDatabase();
                if (!database)
                    throw std::runtime_error("Texture asset loading requires an AssetDatabase");
                const auto path = database->GetPathFromGuid(guid);
                if (path.empty())
                    return std::nullopt;
                TextureLoader loader;
                const auto payload = loader.ReadPixels(path, guid, database);
                const auto &mip = payload->mipLevels.front();
                InxTextureData result;
                result.width = static_cast<int>(mip.width);
                result.height = static_cast<int>(mip.height);
                result.channels = 4;
                result.sourcePath = path;
                result.name = FromFsPath(ToFsPath(path).stem());
                result.pixels = texture_pixels::BaseMipRgba8(*payload);
                return result;
            },
            py::arg("guid"), "Read imported Texture2D pixels by GUID as RGBA8")
        .def_static("load_from_file", &InxTextureLoader::LoadFromFile, py::arg("file_path"), py::arg("name") = "",
                    py::arg("svg_max_size") = 0,
                    "Load texture from file; SVG uses the requested longest edge (0: 8192)")
        .def_static(
            "load_from_memory",
            [](py::bytes data, const std::string &name, int svgMaxSize) {
                std::string str = data;
                return InxTextureLoader::LoadFromMemory(reinterpret_cast<const unsigned char *>(str.data()), str.size(),
                                                        name, svgMaxSize);
            },
            py::arg("data"), py::arg("name") = "", py::arg("svg_max_size") = 0, "Load texture from memory buffer")
        .def_static("create_solid_color", &InxTextureLoader::CreateSolidColor, py::arg("width"), py::arg("height"),
                    py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a"), py::arg("name") = "solid_color",
                    "Create a solid color texture")
        .def_static("create_checkerboard", &InxTextureLoader::CreateCheckerboard, py::arg("width"), py::arg("height"),
                    py::arg("checker_size") = 8, py::arg("name") = "checkerboard",
                    "Create a checkerboard texture (for error indication)");

    // InxMaterial - material definition (named InxMaterial to avoid conflict with ResourceType.Material)
    py::class_<InxMaterial, std::shared_ptr<InxMaterial>>(m, "InxMaterial")
        .def_static("property_to_id", &ShaderPropertyIds::Get, py::arg("name"))
        .def_static(
            "property_name", [](const bindings::MaterialPropertyName &name) { return std::string(name); },
            py::arg("name_or_id"))
        .def(py::init<>())
        .def(py::init<const std::string &>(), py::arg("name"))
        .def(py::init<const std::string &, const std::string &>(), py::arg("name"), py::arg("shader_name"))
        .def_property("name", &InxMaterial::GetName, &InxMaterial::SetName, "Material name")
        .def_property_readonly("guid", &InxMaterial::GetGuid, "Material GUID (read-only, set by AssetDatabase)")
        .def_property("file_path", &InxMaterial::GetFilePath, &InxMaterial::SetFilePath, "File path for saving")
        .def_property_readonly("material_key", &InxMaterial::GetMaterialKey,
                               "Unique key used by the pipeline manager (GUID > filePath > name)")
        .def_property("is_builtin", &InxMaterial::IsBuiltin, &InxMaterial::SetBuiltin,
                      "Whether this is a built-in material (shader cannot be changed)")
        .def_property(
            "shader_name", &InxMaterial::GetShaderName,
            [](InxMaterial &m, const std::string &name) { m.SetShader(name); },
            "Shader name (e.g. 'lit', 'unlit') — sets both vert and frag")
        .def_property(
            "vert_shader_name", &InxMaterial::GetVertShaderName,
            [](InxMaterial &m, const std::string &name) { m.SetVertShader(name); }, "Vertex shader name")
        .def_property(
            "frag_shader_name", &InxMaterial::GetFragShaderName,
            [](InxMaterial &m, const std::string &name) { m.SetFragShader(name); }, "Fragment shader name")
        .def_property(
            "vert_shader_reference",
            [](const InxMaterial &material) { return ShaderReferenceToPython(material.GetVertShaderReference()); },
            [](InxMaterial &material, py::handle reference) {
                material.SetVertShaderReference(ShaderReferenceFromPython(reference));
            },
            "Stable vertex shader asset reference")
        .def_property(
            "frag_shader_reference",
            [](const InxMaterial &material) { return ShaderReferenceToPython(material.GetFragShaderReference()); },
            [](InxMaterial &material, py::handle reference) {
                material.SetFragShaderReference(ShaderReferenceFromPython(reference));
            },
            "Stable fragment shader asset reference")
        .def("set_shader", &InxMaterial::SetShader, py::arg("shader_name"),
             "Set the material's shader by name (sets both vert and frag)")
        .def("get_render_queue", &InxMaterial::GetRenderQueue, "Get the render queue value")
        .def("set_render_queue", &InxMaterial::SetRenderQueue, py::arg("queue"), "Set the render queue value")
        .def("save", py::overload_cast<>(&InxMaterial::SaveToFile, py::const_), "Save material to its file path")
        .def("save_to", py::overload_cast<const std::string &>(&InxMaterial::SaveToFile), py::arg("path"),
             "Save material to specified path")
        .def("is_deleted", &InxMaterial::IsDeleted, "True if the backing .mat file has been deleted")
        .def("mark_as_deleted", &InxMaterial::MarkAsDeleted, "Mark this material as deleted (prevents save)")
        // Property setters (accept both tuple and individual args)
        .def("set_float", bindings::MaterialNameMethod(&InxMaterial::SetFloat), py::arg("name"), py::arg("value"),
             "Set a float property")
        .def(
            "set_vector2",
            [](InxMaterial &mat, const bindings::MaterialPropertyName &name, py::args args) {
                glm::vec2 v;
                if (args.size() == 1) {
                    py::object obj = args[0];
                    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
                        py::sequence seq = obj.cast<py::sequence>();
                        v = glm::vec2(seq[0].cast<float>(), seq[1].cast<float>());
                    } else {
                        v = obj.cast<glm::vec2>();
                    }
                } else if (args.size() >= 2) {
                    v = glm::vec2(args[0].cast<float>(), args[1].cast<float>());
                } else {
                    throw std::runtime_error("set_vector2: expected (name, x, y) or (name, vec2)");
                }
                mat.SetVector2(name, v);
            },
            py::arg("name"), "Set a vec2 property: set_vector2(name, x, y) or set_vector2(name, (x,y))")
        .def(
            "set_vector3",
            [](InxMaterial &mat, const bindings::MaterialPropertyName &name, py::args args) {
                glm::vec3 v;
                if (args.size() == 1) {
                    py::object obj = args[0];
                    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
                        py::sequence seq = obj.cast<py::sequence>();
                        v = glm::vec3(seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>());
                    } else {
                        v = obj.cast<glm::vec3>();
                    }
                } else if (args.size() >= 3) {
                    v = glm::vec3(args[0].cast<float>(), args[1].cast<float>(), args[2].cast<float>());
                } else {
                    throw std::runtime_error("set_vector3: expected (name, x, y, z) or (name, vec3)");
                }
                mat.SetVector3(name, v);
            },
            py::arg("name"), "Set a vec3 property: set_vector3(name, x, y, z) or set_vector3(name, (x,y,z))")
        .def(
            "set_vector4",
            [](InxMaterial &mat, const bindings::MaterialPropertyName &name, py::args args) {
                glm::vec4 v;
                if (args.size() == 1) {
                    py::object obj = args[0];
                    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
                        py::sequence seq = obj.cast<py::sequence>();
                        v = glm::vec4(seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>(),
                                      seq[3].cast<float>());
                    } else {
                        v = obj.cast<glm::vec4>();
                    }
                } else if (args.size() >= 4) {
                    v = glm::vec4(args[0].cast<float>(), args[1].cast<float>(), args[2].cast<float>(),
                                  args[3].cast<float>());
                } else {
                    throw std::runtime_error("set_vector4: expected (name, x, y, z, w) or (name, vec4)");
                }
                mat.SetVector4(name, v);
            },
            py::arg("name"), "Set a vec4 property: set_vector4(name, x, y, z, w) or set_vector4(name, (x,y,z,w))")
        .def(
            "set_color",
            [](InxMaterial &mat, const bindings::MaterialPropertyName &name, py::args args) {
                glm::vec4 color;
                if (args.size() == 1) {
                    py::object obj = args[0];
                    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
                        py::sequence seq = obj.cast<py::sequence>();
                        color = glm::vec4(seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>(),
                                          py::len(seq) >= 4 ? seq[3].cast<float>() : 1.0f);
                    } else {
                        color = obj.cast<glm::vec4>();
                    }
                } else if (args.size() >= 3) {
                    float r = args[0].cast<float>();
                    float g = args[1].cast<float>();
                    float b = args[2].cast<float>();
                    float a = args.size() >= 4 ? args[3].cast<float>() : 1.0f;
                    color = glm::vec4(r, g, b, a);
                } else {
                    throw std::runtime_error("set_color: expected (name, r, g, b[, a]) or (name, color_tuple)");
                }
                mat.SetColor(name, color);
            },
            py::arg("name"), "Set a color property: set_color(name, r, g, b[, a]) or set_color(name, (r,g,b,a))")
        .def("set_int", bindings::MaterialNameMethod(&InxMaterial::SetInt), py::arg("name"), py::arg("value"),
             "Set an int property")
        .def(
            "set_matrix",
            [](InxMaterial &mat, const bindings::MaterialPropertyName &name, py::handle value) {
                mat.SetMatrix(name, binding::Matrix4FromPython(value, "Material matrix"));
            },
            py::arg("name"), py::arg("value"), "Set a mat4 from a [row, column] array")
        .def("set_float_array", bindings::MaterialNameMethod(&InxMaterial::SetFloatArray), py::arg("name"),
             py::arg("values"), "Set a reflected float array property")
        .def(
            "set_vector4_array",
            [](InxMaterial &material, const bindings::MaterialPropertyName &name, py::sequence values) {
                std::vector<glm::vec4> native;
                native.reserve(py::len(values));
                for (py::handle item : values) {
                    py::sequence vector = py::reinterpret_borrow<py::sequence>(item);
                    if (py::len(vector) != 4)
                        throw py::value_error("set_vector4_array requires four-component vectors");
                    native.emplace_back(vector[0].cast<float>(), vector[1].cast<float>(), vector[2].cast<float>(),
                                        vector[3].cast<float>());
                }
                material.SetVector4Array(name, native);
            },
            py::arg("name"), py::arg("values"), "Set a reflected vec4 array property")
        .def("set_texture_guid", bindings::MaterialNameMethod(&InxMaterial::SetTextureGuid), py::arg("name"),
             py::arg("texture_guid"), "Set a texture property by GUID")
        .def("set_buffer", bindings::MaterialNameMethod(&InxMaterial::SetBuffer), py::arg("name"), py::arg("buffer"),
             "Bind a reflected runtime storage buffer")
        .def("get_buffer", bindings::MaterialNameMethod(&InxMaterial::GetBuffer), py::arg("name"),
             "Get a reflected runtime storage buffer binding")
        .def(
            "set_param",
            [](InxMaterial &mat, const bindings::MaterialPropertyName &name, py::object value) {
                const MaterialProperty *prop = mat.GetProperty(name);

                if ((prop && prop->type == MaterialPropertyType::Mat4) || py::isinstance<py::array>(value)) {
                    mat.SetMatrix(name, binding::Matrix4FromPython(value, "Material matrix"));
                    return;
                }

                if (py::isinstance<py::bool_>(value)) {
                    mat.SetInt(name, value.cast<bool>() ? 1 : 0);
                    return;
                }
                if (py::isinstance<py::int_>(value) && !py::isinstance<py::bool_>(value)) {
                    if (prop && prop->type == MaterialPropertyType::Float) {
                        mat.SetFloat(name, value.cast<float>());
                    } else {
                        mat.SetInt(name, value.cast<int>());
                    }
                    return;
                }
                if (py::isinstance<py::float_>(value)) {
                    if (prop && prop->type == MaterialPropertyType::Int) {
                        mat.SetInt(name, static_cast<int>(value.cast<float>()));
                    } else {
                        mat.SetFloat(name, value.cast<float>());
                    }
                    return;
                }
                if (py::isinstance<py::tuple>(value) || py::isinstance<py::list>(value)) {
                    py::sequence seq = value.cast<py::sequence>();
                    const auto len = py::len(seq);
                    if (prop && prop->type == MaterialPropertyType::FloatArray) {
                        mat.SetFloatArray(name, value.cast<std::vector<float>>());
                        return;
                    }
                    if (prop && prop->type == MaterialPropertyType::Float4Array) {
                        std::vector<glm::vec4> values;
                        values.reserve(len);
                        for (py::handle item : seq) {
                            py::sequence vector = py::reinterpret_borrow<py::sequence>(item);
                            if (py::len(vector) != 4)
                                throw py::value_error("Float4Array values require four-component vectors");
                            values.emplace_back(vector[0].cast<float>(), vector[1].cast<float>(),
                                                vector[2].cast<float>(), vector[3].cast<float>());
                        }
                        mat.SetVector4Array(name, values);
                        return;
                    }
                    if (len == 2) {
                        mat.SetVector2(name, glm::vec2(seq[0].cast<float>(), seq[1].cast<float>()));
                        return;
                    }
                    if (len == 3) {
                        mat.SetVector3(name,
                                       glm::vec3(seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>()));
                        return;
                    }
                    if (len == 4) {
                        glm::vec4 v(seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>(),
                                    seq[3].cast<float>());
                        if (prop && prop->type == MaterialPropertyType::Color) {
                            mat.SetColor(name, v);
                        } else {
                            mat.SetVector4(name, v);
                        }
                        return;
                    }
                }

                throw std::runtime_error(
                    "set_param: unsupported value type. Expected int/float/bool, vec2/3/4 tuple, or a (4, 4) matrix.");
            },
            py::arg("name"), py::arg("value"), "Set a non-texture material property using value-shape/type dispatch")
        .def("clear_texture", bindings::MaterialNameMethod(&InxMaterial::ClearTexture), py::arg("name"),
             "Clear a texture property (remove texture reference)")
        .def("_set_render_texture", bindings::MaterialNameMethod(&InxMaterial::SetRenderTexture), py::arg("name"),
             py::arg("texture"))
        .def("_get_render_texture", bindings::MaterialNameMethod(&InxMaterial::GetRenderTexture), py::arg("name"))
        .def_property_readonly("_texture_assets_pending", &InxMaterial::NeedsTextureAssetResolution)
        .def("remove_property", bindings::MaterialNameMethod(&InxMaterial::RemoveProperty), py::arg("name"),
             "Remove a material property and its asset dependency")
        .def(
            "set_texture",
            [](InxMaterial &mat, const bindings::MaterialPropertyName &name, py::object value) {
                if (value.is_none()) {
                    mat.ClearTexture(name);
                    return;
                }

                auto applyString = [&](const std::string &text) {
                    if (text.empty()) {
                        mat.ClearTexture(name);
                        return;
                    }
                    mat.SetTextureGuid(name, text);
                };

                if (py::isinstance<py::str>(value)) {
                    applyString(value.cast<std::string>());
                    return;
                }

                if (py::hasattr(value, "guid")) {
                    py::object guidObj = value.attr("guid");
                    if (!guidObj.is_none()) {
                        std::string guid = py::cast<std::string>(guidObj);
                        if (!guid.empty()) {
                            mat.SetTextureGuid(name, guid);
                            return;
                        }
                    }
                }

                throw std::runtime_error(
                    "set_texture: expected None, a Texture GUID/builtin token, or an object with guid.");
            },
            py::arg("name"), py::arg("value"), "Set a texture property from a Texture GUID or texture object")
        // Individual property getters (convenience wrappers over GetProperty)
        .def(
            "get_float",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name, float defaultVal) -> float {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (prop && prop->type == MaterialPropertyType::Float)
                    return std::get<float>(prop->value);
                return defaultVal;
            },
            py::arg("name"), py::arg("default_value") = 0.0f, "Get a float property")
        .def(
            "get_int",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name, int defaultVal) -> int {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (prop && prop->type == MaterialPropertyType::Int)
                    return std::get<int>(prop->value);
                return defaultVal;
            },
            py::arg("name"), py::arg("default_value") = 0, "Get an int property")
        .def(
            "get_color",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name) -> glm::vec4 {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (prop && (prop->type == MaterialPropertyType::Float4 || prop->type == MaterialPropertyType::Color)) {
                    return std::get<glm::vec4>(prop->value);
                }
                return glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            },
            py::arg("name"), "Get a color property as vec4f")
        .def(
            "get_vector2",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name) -> glm::vec2 {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (prop && prop->type == MaterialPropertyType::Float2) {
                    return std::get<glm::vec2>(prop->value);
                }
                return glm::vec2(0.0f);
            },
            py::arg("name"), "Get a vec2 property as Vector2")
        .def(
            "get_vector3",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name) -> glm::vec3 {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (prop && prop->type == MaterialPropertyType::Float3) {
                    return std::get<glm::vec3>(prop->value);
                }
                return glm::vec3(0.0f);
            },
            py::arg("name"), "Get a vec3 property as Vector3")
        .def(
            "get_vector4",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name) -> glm::vec4 {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (prop && (prop->type == MaterialPropertyType::Float4 || prop->type == MaterialPropertyType::Color)) {
                    return std::get<glm::vec4>(prop->value);
                }
                return glm::vec4(0.0f);
            },
            py::arg("name"), "Get a vec4 property as vec4f")
        .def(
            "get_texture",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name) -> py::object {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (prop && prop->type == MaterialPropertyType::Texture2D)
                    return py::cast(std::get<std::string>(prop->value));
                return py::none();
            },
            py::arg("name"), "Get a texture property GUID (or None)")
        // Generic property access
        .def("has_property", bindings::MaterialNameMethod(&InxMaterial::HasProperty), py::arg("name"),
             "Check if material has a property")
        .def(
            "get_property",
            [](const InxMaterial &mat, const bindings::MaterialPropertyName &name) -> py::object {
                const MaterialProperty *prop = mat.GetProperty(name);
                if (!prop) {
                    return py::none();
                }
                return PyValueFromMaterialProperty(*prop);
            },
            py::arg("name"), "Get a property value by name")
        .def(
            "get_all_properties",
            [](const InxMaterial &mat) -> py::dict {
                py::dict result;
                for (const auto &[name, prop] : mat.GetAllProperties()) {
                    result[py::str(name)] = PyValueFromMaterialProperty(prop);
                }
                return result;
            },
            "Get all properties as a dictionary")
        // Pipeline state
        .def("is_pipeline_dirty", &InxMaterial::IsPipelineDirty, "Check if pipeline needs recreation")
        .def("clear_pipeline_dirty", &InxMaterial::ClearPipelineDirty, "Clear the pipeline dirty flag")
        .def("get_pipeline_hash", &InxMaterial::GetPipelineHash, "Get a hash of the pipeline configuration")
        .def("get_version", &InxMaterial::GetVersion, "Monotonic version counter incremented on every property change")
        // Serialization
        .def("serialize", &InxMaterial::Serialize, "Serialize material to JSON string")
        .def("deserialize", &InxMaterial::Deserialize, py::arg("json_str"), "Deserialize material from JSON string")
        .def(
            "serialize_document",
            [](const InxMaterial &material) { return JsonToPython(material.SerializeDocument()); },
            "Serialize material to a Python document")
        .def(
            "deserialize_document",
            [](InxMaterial &material, py::handle document) {
                return material.DeserializeDocument(PythonToJson(document));
            },
            py::arg("document"), "Deserialize material from a Python document")
        .def_static("create_default_lit", &InxMaterial::CreateDefaultLit,
                    "Create the default lit opaque material (built-in)")
        .def_static("create_default_unlit", &InxMaterial::CreateDefaultUnlit, "Create a default unlit opaque material")
        // Render state access
        .def(
            "get_render_state", [](const InxMaterial &mat) -> RenderState { return mat.GetRenderState(); },
            "Get a copy of the material's render state")
        .def(
            "set_render_state", [](InxMaterial &mat, const RenderState &state) { mat.SetRenderState(state); },
            py::arg("state"),
            "Explicitly author the full render state; every field becomes an override so shader "
            "annotation defaults can never replace it (switching shaders resets authorship)")
        // RenderState override mechanism
        .def_property("render_state_overrides", &InxMaterial::GetRenderStateOverrides,
                      &InxMaterial::SetRenderStateOverrides,
                      "Bitmask of user-overridden RenderState fields (see RenderStateOverride enum)")
        .def("mark_override", &InxMaterial::MarkOverride, py::arg("flag"),
             "Mark a RenderState field as user-overridden")
        .def("clear_override", &InxMaterial::ClearOverride, py::arg("flag"),
             "Clear a RenderState field override (revert to shader default)")
        .def("has_override", &InxMaterial::HasOverride, py::arg("flag"),
             "Check if a RenderState field is user-overridden")
        .def("apply_shader_render_meta", &InxMaterial::ApplyShaderRenderMeta, py::arg("cull_mode"),
             py::arg("depth_write"), py::arg("depth_test"), py::arg("blend"), py::arg("queue"),
             py::arg("pass_tag") = "", py::arg("stencil") = "", py::arg("alpha_clip") = "",
             "Apply shader annotation defaults to render-state fields the material has not authored")
        .def("sync_alpha_clip_property", &InxMaterial::SyncAlphaClipProperty,
             "Sync internal _AlphaClipThreshold material property from RenderState")
        // Clone / Instantiate (Unity-style Object.Instantiate for materials)
        .def("clone", &InxMaterial::Clone,
             "Create a deep copy of this material (Unity: new Material(original)). "
             "Copies all properties, shader names, and render state. "
             "GPU state is lazily recreated. The clone has no GUID (runtime-only).")
        .def_static(
            "instantiate",
            [](const std::shared_ptr<InxMaterial> &original) -> std::shared_ptr<InxMaterial> {
                if (!original)
                    return nullptr;
                return original->Clone();
            },
            py::arg("original"),
            "Clone a material (Unity: Object.Instantiate). Deep-copies all properties; "
            "shader and texture references are shared. Returns a new runtime material instance.");

    // RenderStateOverride — bitmask flags for per-material render state overrides
    py::enum_<RenderStateOverride>(m, "RenderStateOverride", py::arithmetic())
        .value("NONE", RenderStateOverride::None)
        .value("CULL_MODE", RenderStateOverride::CullMode)
        .value("DEPTH_WRITE", RenderStateOverride::DepthWrite)
        .value("DEPTH_TEST", RenderStateOverride::DepthTest)
        .value("DEPTH_COMPARE_OP", RenderStateOverride::DepthCompareOp)
        .value("BLEND_ENABLE", RenderStateOverride::BlendEnable)
        .value("BLEND_MODE", RenderStateOverride::BlendMode)
        .value("RENDER_QUEUE", RenderStateOverride::RenderQueue)
        .value("SURFACE_TYPE", RenderStateOverride::SurfaceType)
        .value("ALPHA_CLIP", RenderStateOverride::AlphaClip)
        .export_values();

    // RenderState — GPU pipeline state for materials
    py::class_<RenderState>(m, "RenderState")
        .def(py::init<>())
        // Rasterization
        .def_property(
            "cull_mode", [](const RenderState &rs) { return static_cast<int>(rs.cullMode); },
            [](RenderState &rs, int v) { rs.cullMode = static_cast<MaterialCullMode>(v); },
            "Material cull mode: 0=None, 1=Front, 2=Back, 3=FrontAndBack")
        .def_property(
            "front_face", [](const RenderState &rs) { return static_cast<int>(rs.frontFace); },
            [](RenderState &rs, int v) { rs.frontFace = static_cast<MaterialFrontFace>(v); },
            "Material front face: 0=CounterClockwise, 1=Clockwise")
        .def_property(
            "polygon_mode", [](const RenderState &rs) { return static_cast<int>(rs.polygonMode); },
            [](RenderState &rs, int v) { rs.polygonMode = static_cast<MaterialPolygonMode>(v); },
            "Material polygon mode: 0=Fill, 1=Line, 2=Point")
        .def_readwrite("line_width", &RenderState::lineWidth)
        // Depth
        .def_readwrite("depth_test_enable", &RenderState::depthTestEnable)
        .def_readwrite("depth_write_enable", &RenderState::depthWriteEnable)
        .def_property(
            "depth_compare_op", [](const RenderState &rs) { return static_cast<int>(rs.depthCompareOp); },
            [](RenderState &rs, int v) { rs.depthCompareOp = static_cast<MaterialCompareOp>(v); },
            "Material depth compare: "
            "0=Never,1=Less,2=Equal,3=LessOrEqual,4=Greater,5=NotEqual,6=GreaterOrEqual,7=Always")
        // Blending
        .def_readwrite("blend_enable", &RenderState::blendEnable)
        .def_property(
            "src_color_blend_factor", [](const RenderState &rs) { return static_cast<int>(rs.srcColorBlendFactor); },
            [](RenderState &rs, int v) { rs.srcColorBlendFactor = static_cast<MaterialBlendFactor>(v); })
        .def_property(
            "dst_color_blend_factor", [](const RenderState &rs) { return static_cast<int>(rs.dstColorBlendFactor); },
            [](RenderState &rs, int v) { rs.dstColorBlendFactor = static_cast<MaterialBlendFactor>(v); })
        .def_property(
            "color_blend_op", [](const RenderState &rs) { return static_cast<int>(rs.colorBlendOp); },
            [](RenderState &rs, int v) { rs.colorBlendOp = static_cast<MaterialBlendOp>(v); })
        .def_property(
            "src_alpha_blend_factor", [](const RenderState &rs) { return static_cast<int>(rs.srcAlphaBlendFactor); },
            [](RenderState &rs, int v) { rs.srcAlphaBlendFactor = static_cast<MaterialBlendFactor>(v); })
        .def_property(
            "dst_alpha_blend_factor", [](const RenderState &rs) { return static_cast<int>(rs.dstAlphaBlendFactor); },
            [](RenderState &rs, int v) { rs.dstAlphaBlendFactor = static_cast<MaterialBlendFactor>(v); })
        .def_property(
            "alpha_blend_op", [](const RenderState &rs) { return static_cast<int>(rs.alphaBlendOp); },
            [](RenderState &rs, int v) { rs.alphaBlendOp = static_cast<MaterialBlendOp>(v); })
        // Render queue
        .def_readwrite("render_queue", &RenderState::renderQueue, "Sorting order: 2000=Opaque, 3000=Transparent")
        // Alpha clip
        .def_readwrite("alpha_clip_enabled", &RenderState::alphaClipEnabled, "Whether alpha clipping is enabled")
        .def_readwrite("alpha_clip_threshold", &RenderState::alphaClipThreshold, "Alpha clip threshold (0.0-1.0)");
}

} // namespace infernux
