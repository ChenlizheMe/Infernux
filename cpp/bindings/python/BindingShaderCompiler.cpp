#include "BindingRegistration.h"
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <pybind11/stl.h>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;

namespace infernux
{
namespace
{
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
            return result;
        } catch (py::error_already_set &error) {
            // Worker tickets retain native diagnostics, never Python exception
            // objects whose destruction could run after interpreter shutdown.
            const std::string message = error.what();
            error.restore();
            PyErr_Clear();
            throw std::runtime_error(message);
        }
    });
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
           const std::vector<std::string> &targetNames, bool precise) {
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
                compiled = SlangCompiler::Get().Compile({moduleName, sourcePath, source}, modules, targets, precise);
            }
            py::dict result;
            result["diagnostics"] = compiled.diagnostics;
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
        py::arg("targets") = std::vector<std::string>{"spirv", "wgsl"}, py::arg("precise") = false);
}

} // namespace infernux
