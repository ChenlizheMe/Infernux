#include "BindingRegistration.h"
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <pybind11/stl.h>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;

namespace infernux
{

void RegisterShaderCompilerBindings(py::module_ &m)
{
    m.attr("_slang_standard_module_name") = SlangCompiler::GetStandardModuleName();
    {
        py::gil_scoped_release release;
        SlangCompiler::Get();
    }
    m.def("_slang_version", [] { return SlangCompiler::Get().GetVersion(); }, py::call_guard<py::gil_scoped_release>());
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
