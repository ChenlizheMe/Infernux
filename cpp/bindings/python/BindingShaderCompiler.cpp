#include "BindingRegistration.h"
#include "ShaderDiagnosticPyBridge.h"
#include <core/threading/JobSystem.h>
#include <function/renderer/shader/ShaderTargetArchive.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderAuthorInputs.h>
#include <function/resources/ShaderAsset/ShaderAuthorProgramCache.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/resources/ShaderAsset/SlangParticleModule.h>
#include <function/resources/ShaderAsset/SlangUIModule.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <pybind11/stl.h>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace py = pybind11;

namespace infernux
{
py::list ShaderDiagnosticsToPython(const std::vector<ShaderDiagnostic> &diagnostics)
{
    py::list result;
    for (const auto &value : diagnostics) {
        py::dict item;
        item["code"] = value.code;
        item["severity"] = value.severity;
        item["path"] = value.path;
        item["line"] = value.line;
        item["column"] = value.column;
        item["end_line"] = value.endLine;
        item["end_column"] = value.endColumn;
        item["message"] = value.message;
        item["phase"] = value.phase;
        item["target"] = value.target;
        result.append(std::move(item));
    }
    return result;
}

namespace
{
PyObject *g_shaderCompilationError = nullptr;
using SourceRangeTuple = std::tuple<std::string, uint32_t, uint32_t, uint32_t, uint32_t>;

std::vector<ShaderAuthorRange> SourceRanges(const std::vector<SourceRangeTuple> &locations)
{
    std::vector<ShaderAuthorRange> result;
    result.reserve(locations.size());
    for (const auto &[path, line, column, endLine, endColumn] : locations)
        result.push_back({path, line, column, endLine, endColumn});
    return result;
}

std::vector<ShaderDiagnostic> AuthorDiagnostics(py::error_already_set &error, const std::string &path)
{
    const auto value = error.value();
    std::vector<ShaderDiagnostic> result;
    if (py::hasattr(value, "diagnostics")) {
        for (const auto record : value.attr("diagnostics")) {
            const auto item = py::reinterpret_borrow<py::dict>(record);
            result.push_back({py::cast<std::string>(item["code"]),
                              py::cast<std::string>(item["severity"]),
                              py::cast<std::string>(item["path"]),
                              py::cast<uint32_t>(item["line"]),
                              py::cast<uint32_t>(item["column"]),
                              py::cast<uint32_t>(item["end_line"]),
                              py::cast<uint32_t>(item["end_column"]),
                              py::cast<std::string>(item["message"]),
                              "author",
                              {}});
        }
    } else
        result.push_back({"author.internal", "error", path, 0, 0, 0, 0, py::str(value), "author", {}});
    error.restore();
    PyErr_Clear();
    return result;
}

py::dict ValueLayout(const ShaderValueLayout &value)
{
    py::dict result;
    result["name"] = value.name;
    result["kind"] = static_cast<uint8_t>(value.kind);
    result["scalar"] = static_cast<uint8_t>(value.scalar);
    result["offset"] = value.offset;
    result["byte_size"] = value.byteSize;
    result["stride"] = value.stride;
    result["rows"] = value.rows;
    result["columns"] = value.columns;
    result["element_count"] = value.elementCount;
    result["row_major"] = value.rowMajor;
    py::list fields;
    for (const auto &field : value.fields)
        fields.append(ValueLayout(field));
    result["fields"] = std::move(fields);
    return result;
}

using LibraryTuple = std::tuple<std::string, std::string, std::string>;
using ShaderInputKinds = std::vector<std::pair<std::string, std::string>>;

struct ShaderCompileRequest
{
    ShaderSourceModule module;
    std::vector<ShaderSourceModule> libraries;
    std::vector<ShaderCodeTarget> targets;
    std::vector<ShaderAuthorRange> ranges;
    ShaderInputKinds inputKinds;
    std::optional<std::string> entryPoint;
    bool precise = false;
    bool rootConstants = false;
    std::string authorSchema;
};

ShaderCompileRequest MakeCompileRequest(const std::string &source, const std::string &sourcePath,
                                        const std::string &moduleName, const std::vector<LibraryTuple> &libraries,
                                        const std::vector<std::string> &targetNames, bool precise,
                                        const std::vector<SourceRangeTuple> &sourceRanges,
                                        const ShaderInputKinds &inputKinds, bool rootConstants,
                                        const std::optional<std::string> &entryPoint, const std::string &authorSchema)
{
    ShaderCompileRequest request;
    request.authorSchema = authorSchema;
    request.module = {moduleName, sourcePath, source};
    request.libraries.reserve(libraries.size());
    for (const auto &[name, path, text] : libraries)
        request.libraries.push_back({name, path, text});
    request.targets.reserve(targetNames.size());
    for (const auto &name : targetNames) {
        if (name == "spirv")
            request.targets.push_back(ShaderCodeTarget::SpirV);
        else if (name == "wgsl")
            request.targets.push_back(ShaderCodeTarget::Wgsl);
        else
            throw std::invalid_argument("Shader target must be spirv or wgsl: " + name);
    }
    if (request.targets.empty())
        throw std::invalid_argument("A shader program needs at least one target");
    request.ranges = SourceRanges(sourceRanges);
    request.inputKinds = inputKinds;
    request.entryPoint = entryPoint;
    request.precise = precise;
    request.rootConstants = rootConstants;
    return request;
}

struct ShaderCompileResult
{
    ShaderCompilation compilation;
    std::optional<bool> frontendReused;
};

ShaderCompileResult CompileProgram(const ShaderCompileRequest &request)
{
    auto &compiler = SlangCompiler::Get();
    ShaderCompileResult result;
    if (request.entryPoint) {
        bool frontendReused = false;
        result.compilation =
            compiler.CompileComputeEntry(request.module, request.libraries, request.targets, *request.entryPoint,
                                         request.precise, request.ranges, frontendReused);
        result.frontendReused = frontendReused;
    } else if (request.rootConstants) {
        result.compilation = compiler.CompileRootProgram(request.module, request.libraries, request.targets,
                                                         request.precise, request.ranges);
    } else {
        result.compilation =
            compiler.Compile(request.module, request.libraries, request.targets, request.precise, request.ranges);
    }
    ApplyShaderAuthorResources(result.compilation, request.inputKinds);
    if (!request.authorSchema.empty())
        SlangCompiler::ApplyAuthorMetadata(result.compilation, request.authorSchema);
    return result;
}

py::dict CompileResultToPython(const ShaderCompileResult &resultData)
{
    const auto &compiled = resultData.compilation;
    py::dict result;
    result["diagnostics"] = ShaderDiagnosticsToPython(compiled.diagnostics);
    if (resultData.frontendReused.has_value())
        result["frontend_reused"] = *resultData.frontendReused;
    py::list targetPrograms;
    for (const auto &target : compiled.targets) {
        py::dict program;
        program["target"] = target.target == ShaderCodeTarget::SpirV ? "spirv" : "wgsl";
        program["reflection"] = target.reflection;
        py::list resources;
        for (const auto &resource : target.resources) {
            py::dict item;
            item["name"] = resource.name;
            item["set"] = resource.set;
            item["binding"] = resource.binding.binding;
            item["type"] = static_cast<uint8_t>(resource.binding.type);
            item["count"] = resource.binding.count;
            item["visibility"] = static_cast<uint8_t>(resource.binding.visibility);
            item["read_only"] = resource.binding.readOnlyStorage;
            item["depth"] = resource.binding.depthRead;
            item["dimension"] = static_cast<uint8_t>(resource.binding.viewDimension);
            item["multisampled"] = resource.binding.multisampled;
            item["sample_type"] = static_cast<uint8_t>(resource.binding.sampleType);
            item["storage_format"] = static_cast<uint8_t>(resource.binding.storageFormat);
            item["storage_access"] = static_cast<uint8_t>(resource.binding.storageAccess);
            item["sampler_type"] = static_cast<uint8_t>(resource.binding.samplerType);
            item["min_buffer_binding_size"] = resource.binding.minBufferBindingSize;
            item["element_stride"] = resource.elementStride;
            item["value"] = ValueLayout(resource.value);
            resources.append(std::move(item));
        }
        program["resources"] = std::move(resources);
        py::list entries;
        for (const auto &entry : target.entries) {
            py::dict item;
            item["name"] = entry.name;
            switch (entry.stage) {
            case rhi::ShaderStage::Vertex:
                item["stage"] = "vertex";
                break;
            case rhi::ShaderStage::Fragment:
                item["stage"] = "fragment";
                break;
            default:
                item["stage"] = "compute";
                break;
            }
            item["workgroup_size"] = entry.workgroupSize;
            item["bindings"] = entry.bindings;
            item["root_constants"] =
                entry.rootConstants ? py::object(ValueLayout(*entry.rootConstants)) : py::object(py::none());
            const auto stageValues = [](const auto &values) {
                py::list result;
                for (const auto &value : values) {
                    py::dict variable;
                    variable["name"] = value.name;
                    variable["location"] = value.location;
                    variable["scalar"] = static_cast<uint8_t>(value.scalar);
                    variable["components"] = value.components;
                    result.append(std::move(variable));
                }
                return result;
            };
            item["inputs"] = stageValues(entry.inputs);
            item["outputs"] = stageValues(entry.outputs);
            item["sample_rate_shading"] = entry.sampleRateShading;
            item["code"] = py::bytes(entry.code.data(), entry.code.size());
            entries.append(std::move(item));
        }
        program["entries"] = std::move(entries);
        const bool graphCompute =
            std::any_of(target.entries.begin(), target.entries.end(),
                        [](const auto &entry) { return entry.stage == rhi::ShaderStage::Compute; }) &&
            std::any_of(target.resources.begin(), target.resources.end(), [](const auto &resource) {
                return resource.name == "world_ray_geometry" || resource.name == "world_ray_meshes" ||
                       resource.name == "ray_lit_materials";
            });
        if (graphCompute) {
            auto metadata = target;
            for (auto &entry : metadata.entries)
                entry.code.clear();
            program["graph_metadata"] = py::bytes(EncodeShaderTarget(metadata));
        }
        targetPrograms.append(std::move(program));
    }
    result["targets"] = std::move(targetPrograms);
    return result;
}

// Only C++ values cross the worker boundary. The owner maps the native result to
// Python, then prepares pipelines through its compute host.
class ShaderCompileTicket
{
  public:
    explicit ShaderCompileTicket(ShaderCompileRequest request) : m_state(std::make_shared<State>())
    {
        auto compile = [state = m_state, request = std::move(request)] {
            if (state->cancelRequested.load(std::memory_order_acquire))
                return;
            state->producer = std::this_thread::get_id();
            try {
                state->result = CompileProgram(request);
            } catch (...) {
                state->failure = std::current_exception();
            }
        };
        auto &jobs = JobSystem::Get();
        if (jobs.IsInline())
            compile();
        else
            m_job = jobs.Schedule(std::move(compile), JobDomain::Runtime, JobPriority::Normal, JobAffinity::WorkerOnly);
    }

    ~ShaderCompileTicket()
    {
        Cancel();
    }

    ShaderCompileTicket(const ShaderCompileTicket &) = delete;
    ShaderCompileTicket &operator=(const ShaderCompileTicket &) = delete;

    [[nodiscard]] bool Done() const noexcept
    {
        return m_job.IsComplete();
    }

    [[nodiscard]] bool ProducedOnWorker() const noexcept
    {
        return Done() && m_state->producer != std::thread::id{} && m_state->producer != m_owner;
    }

    bool Cancel() noexcept
    {
        if (m_state->cancelRequested.exchange(true, std::memory_order_acq_rel))
            return false;
        m_job.Cancel();
        return true;
    }

    [[nodiscard]] py::object Result() const
    {
        if (std::this_thread::get_id() != m_owner)
            throw std::logic_error("Shader compile results must be read on the requesting thread");
        if (!Done() || m_state->cancelRequested.load(std::memory_order_acquire) || m_job.IsCancellationRequested())
            return py::none();
        if (m_state->failure)
            std::rethrow_exception(m_state->failure);
        return CompileResultToPython(m_state->result.value());
    }

  private:
    struct State
    {
        std::optional<ShaderCompileResult> result;
        std::exception_ptr failure;
        std::thread::id producer;
        std::atomic<bool> cancelRequested{false};
    };

    const std::thread::id m_owner = std::this_thread::get_id();
    std::shared_ptr<State> m_state;
    JobHandle m_job;
};
} // namespace

void RegisterShaderCompilerBindings(py::module_ &m)
{
    g_shaderCompilationError =
        PyErr_NewException("infernux.lib._Infernux.ShaderCompilationError", PyExc_RuntimeError, nullptr);
    m.add_object("ShaderCompilationError", py::reinterpret_steal<py::object>(g_shaderCompilationError));
    py::register_local_exception_translator([](std::exception_ptr exception) {
        try {
            if (exception)
                std::rethrow_exception(exception);
        } catch (const ShaderCompilationError &error) {
            auto value = py::reinterpret_borrow<py::object>(g_shaderCompilationError)(error.what());
            value.attr("diagnostics") = ShaderDiagnosticsToPython(error.diagnostics);
            PyErr_SetObject(g_shaderCompilationError, value.ptr());
        }
    });
    m.def(
        "_shader_source_path",
        [](const std::string &identifier, const std::string &kind) {
            auto *database = AssetRegistry::Instance().GetAssetDatabase();
            if (!database)
                throw std::logic_error("Resolving a shader ID requires an AssetDatabase");
            return database->FindShaderPathById(identifier, kind);
        },
        py::arg("identifier"), py::arg("kind"));
    {
        py::gil_scoped_release release;
        SlangCompiler::Get();
    }
    SlangCompiler::Get().SetAuthorFrontend([](const ShaderSourceModule &source, const ShaderTextReader &reader,
                                              ShaderAuthorPhase phase, const ShaderSourceModule &vertex,
                                              const ShaderAuthorSurfaceReader &surfaces) {
        py::gil_scoped_acquire acquire;
        try {
            const auto prepare = py::module_::import("infernux._compiler.shader").attr("_prepare_source");
            const auto prepared = prepare(
                source.source, source.path, source.name, py::cpp_function(reader),
                py::arg("ray_surfaces") = surfaces ? py::object(py::cpp_function(surfaces)) : py::object(py::tuple()),
                py::arg("metadata_only") = phase == ShaderAuthorPhase::Metadata,
                py::arg("vertex_source") = vertex.path.empty()
                                               ? py::object(py::none())
                                               : py::object(py::make_tuple(vertex.path, vertex.source)));
            const auto fields = py::cast<py::tuple>(prepared);
            ShaderAuthorSource result;
            result.source = {source.name, source.path, py::cast<std::string>(fields[0])};
            for (const auto &[name, path, text] :
                 py::cast<std::vector<std::tuple<std::string, std::string, std::string>>>(fields[1]))
                result.libraries.push_back({name, path, text});
            result.schema = py::cast<std::string>(fields[2]);
            result.sourceRanges = SourceRanges(py::cast<std::vector<SourceRangeTuple>>(fields[3]));
            return result;
        } catch (py::error_already_set &error) {
            // Worker tickets retain native diagnostics, never Python exception
            // objects whose destruction could run after interpreter shutdown.
            throw ShaderCompilationError(AuthorDiagnostics(error, source.path));
        }
    });
    m.def("_shader_author_contract", [] { return std::string(SlangCompiler::GetAuthorContract()); });
    m.def(
        "_cook_shader_author_programs",
        [](const std::string &contentRoot, const std::string &resourceRoot, const std::string &buildCache,
           const std::string &catalog, bool web, const std::vector<std::pair<std::string, std::string>> &selections) {
            AssetDatabase database;
            database.InitializeRuntime(contentRoot);
            database.AddReadOnlyScanRoot(resourceRoot);
            database.InstallRuntimeAssetCatalog(catalog, true);
            const auto library = FromFsPath(ToFsPath(contentRoot) / "Library");
            const auto target = web ? ShaderCodeTarget::Wgsl : ShaderCodeTarget::SpirV;
            for (const auto &[author, vertex] : selections)
                ShaderLoader::CookAuthorProgram(database, author, vertex, target, buildCache, library);
        },
        py::call_guard<py::gil_scoped_release>());
    m.def(
        "_check_shader_source",
        [](const std::string &path, const std::optional<std::string> &text) {
            ShaderAuthorSource prepared;
            std::vector<ShaderDiagnostic> diagnostics;
            bool valid = false;
            {
                py::gil_scoped_release release;
                auto *database = AssetRegistry::Instance().GetAssetDatabase();
                if (!database)
                    throw std::logic_error("GPU source checking requires an AssetDatabase");
                const auto reader = [database](const std::string &path) -> std::optional<std::string> {
                    std::vector<char> contents;
                    if (!database->ReadFile(path, contents))
                        return std::nullopt;
                    return std::string(contents.begin(), contents.end());
                };
                const auto source = text ? text : reader(path);
                if (!source)
                    throw std::invalid_argument("GPU author source could not be read: " + path);
                auto &compiler = SlangCompiler::Get();
                try {
                    prepared = compiler.PrepareAuthorSource({"check_program", path, *source}, reader,
                                                            ShaderAuthorPhase::Program, {},
                                                            ShaderProjectSurfaceReader(*database));
                    auto compiled =
                        compiler.CompileAuthorProgram(prepared, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
                    if (nlohmann::json::parse(prepared.schema).at("domain") == "fullscreen")
                        ShaderLoader::ValidateFullscreenProgram(compiled, path);
                    diagnostics = std::move(compiled.diagnostics);
                    valid = true;
                } catch (const ShaderCompilationError &error) {
                    diagnostics = error.diagnostics;
                }
            }
            py::dict result;
            result["valid"] = valid;
            result["schema"] = prepared.schema;
            result["diagnostics"] = ShaderDiagnosticsToPython(diagnostics);
            return result;
        },
        py::arg("path"), py::arg("source") = py::none());
    m.def(
        "_shader_source_schema",
        [](const std::string &guid) {
            // Logical fields are authoring data. Loading ShaderAsset here also
            // compiles every GPU target, making a cold Inspector query block UI.
            auto *database = AssetRegistry::Instance().GetAssetDatabase();
            if (!database)
                throw std::logic_error("Reading a shader schema requires an AssetDatabase");
            const auto path = database->GetPathFromGuid(guid);
            if (!asset_formats::FindShaderSourceKind(path))
                throw std::invalid_argument("Asset has no static GPU author schema: " + guid);
            // An accepted resident program owns its fields until the next
            // publication. An unfinished disk edit cannot change that schema.
            if (const auto resident = AssetRegistry::Instance().GetAsset<ShaderAsset>(guid))
                return resident->authoringSchema;
            const auto reader = [database](const std::string &sourcePath) -> std::optional<std::string> {
                std::vector<char> contents;
                if (!database->ReadFile(sourcePath, contents))
                    return std::nullopt;
                return std::string(contents.begin(), contents.end());
            };
            const auto source = reader(path);
            if (!source)
                throw std::invalid_argument("GPU author source could not be read: " + path);
            return SlangCompiler::Get()
                .PrepareAuthorSource({"schema_" + guid, path, *source}, reader, ShaderAuthorPhase::Program, {},
                                     ShaderProjectSurfaceReader(*database))
                .schema;
        },
        py::arg("guid"), py::call_guard<py::gil_scoped_release>());
    m.def("_slang_version", [] { return SlangCompiler::Get().GetVersion(); }, py::call_guard<py::gil_scoped_release>());
    m.def(
        "_shader_project_surface_sources",
        [](const std::string &contentRoot, const std::string &resourceRoot, const std::string &catalog) {
            if (!contentRoot.empty()) {
                AssetDatabase database;
                database.InitializeRuntime(contentRoot);
                database.AddReadOnlyScanRoot(resourceRoot);
                database.InstallRuntimeAssetCatalog(catalog, true);
                return ShaderAuthorSurfaceSources(ShaderProjectSurfaceReader(database)());
            }
            const auto *database = AssetRegistry::Instance().GetAssetDatabase();
            return database ? ShaderAuthorSurfaceSources(ShaderProjectSurfaceReader(*database)())
                            : ShaderAuthorSurfaceSources{};
        },
        py::arg("content_root") = "", py::arg("resource_root") = "", py::arg("catalog") = "",
        py::call_guard<py::gil_scoped_release>());

    m.def("_shader_ui_author_source", &GetSlangUIAuthorSource, py::arg("world"), py::arg("depth"),
          py::call_guard<py::gil_scoped_release>());

    py::class_<ShaderCompileTicket>(m, "_ShaderCompileTicket")
        .def_property_readonly("done", &ShaderCompileTicket::Done)
        .def_property_readonly("produced_on_worker", &ShaderCompileTicket::ProducedOnWorker)
        .def("cancel", &ShaderCompileTicket::Cancel)
        .def("result", &ShaderCompileTicket::Result);

    const auto bindCompile = [&m](const char *name, auto compile) {
        m.def(
            name,
            [compile](const std::string &source, const std::string &sourcePath, const std::string &moduleName,
                      const std::vector<LibraryTuple> &libraries, const std::vector<std::string> &targetNames,
                      bool precise, const std::vector<SourceRangeTuple> &sourceRanges,
                      const ShaderInputKinds &inputKinds, bool rootConstants,
                      const std::optional<std::string> &entryPoint, const std::string &authorSchema) {
                return compile(MakeCompileRequest(source, sourcePath, moduleName, libraries, targetNames, precise,
                                                  sourceRanges, inputKinds, rootConstants, entryPoint, authorSchema));
            },
            py::arg("source"), py::arg("source_path"), py::arg("module_name"),
            py::arg("libraries") = std::vector<LibraryTuple>{},
            py::arg("targets") = std::vector<std::string>{"spirv", "wgsl"}, py::arg("precise") = false,
            py::arg("source_ranges") = std::vector<SourceRangeTuple>{}, py::arg("input_kinds") = ShaderInputKinds{},
            py::arg("root_constants") = false, py::arg("entry_point") = py::none(),
            py::arg("author_schema") = std::string{});
    };
    bindCompile("_compile_shader_program", [](const ShaderCompileRequest &request) {
        ShaderCompileResult result;
        {
            py::gil_scoped_release release;
            result = CompileProgram(request);
        }
        return CompileResultToPython(result);
    });
    bindCompile("_begin_compile_shader_program", [](ShaderCompileRequest request) {
        py::gil_scoped_release release;
        return std::make_unique<ShaderCompileTicket>(std::move(request));
    });
}

} // namespace infernux
