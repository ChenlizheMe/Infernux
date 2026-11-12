#include "BindingRegistration.h"
#include "ShaderDiagnosticPyBridge.h"
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <pybind11/stl.h>
#include <stdexcept>
#include <tuple>

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
    m.attr("_slang_standard_module_name") = SlangCompiler::GetStandardModuleName();
    {
        py::gil_scoped_release release;
        SlangCompiler::Get();
    }
    SlangCompiler::Get().SetAuthorFrontend([](const ShaderSourceModule &source, const ShaderTextReader &reader) {
        py::gil_scoped_acquire acquire;
        try {
            const auto prepare = py::module_::import("infernux._compiler.shader").attr("_prepare_source");
            const auto prepared = prepare(source.source, source.path, source.name, py::cpp_function(reader));
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
                    prepared = compiler.PrepareAuthorSource({"check_program", path, *source}, reader);
                    auto compiled = compiler.Compile(prepared.source, prepared.libraries,
                                                     {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl}, false,
                                                     prepared.sourceRanges);
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
            const auto asset = AssetRegistry::Instance().LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
            if (!asset || asset->authoringSchema.empty())
                throw std::invalid_argument("Asset has no static GPU author schema: " + guid);
            return asset->authoringSchema;
        },
        py::arg("guid"), py::call_guard<py::gil_scoped_release>());
    m.def("_slang_version", [] { return SlangCompiler::Get().GetVersion(); }, py::call_guard<py::gil_scoped_release>());
    m.def(
        "_set_shader_module_directory",
        [](const std::string &directory) { SlangCompiler::Get().SetStandardModuleDirectory(directory); },
        py::call_guard<py::gil_scoped_release>());
    m.def(
        "_compile_shader_program",
        [](const std::string &source, const std::string &sourcePath, const std::string &moduleName,
           const std::vector<std::tuple<std::string, std::string, std::string>> &libraries,
           const std::vector<std::string> &targetNames, bool precise,
           const std::vector<SourceRangeTuple> &sourceRanges) {
            std::vector<ShaderSourceModule> modules;
            for (const auto &[name, path, text] : libraries)
                modules.push_back({name, path, text});
            std::vector<ShaderCodeTarget> targets;
            for (const auto &name : targetNames) {
                if (name == "spirv")
                    targets.push_back(ShaderCodeTarget::SpirV);
                else if (name == "wgsl")
                    targets.push_back(ShaderCodeTarget::Wgsl);
                else
                    throw std::invalid_argument("Shader target must be spirv or wgsl: " + name);
            }
            if (targets.empty())
                throw std::invalid_argument("A shader program needs at least one target");

            ShaderCompilation compiled;
            {
                py::gil_scoped_release release;
                compiled = SlangCompiler::Get().Compile({moduleName, sourcePath, source}, modules, targets, precise,
                                                        SourceRanges(sourceRanges));
            }
            py::dict result;
            result["diagnostics"] = ShaderDiagnosticsToPython(compiled.diagnostics);
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
                targetPrograms.append(std::move(program));
            }
            result["targets"] = std::move(targetPrograms);
            return result;
        },
        py::arg("source"), py::arg("source_path"), py::arg("module_name"),
        py::arg("libraries") = std::vector<std::tuple<std::string, std::string, std::string>>{},
        py::arg("targets") = std::vector<std::string>{"spirv", "wgsl"}, py::arg("precise") = false,
        py::arg("source_ranges") = std::vector<SourceRangeTuple>{});
}

} // namespace infernux
