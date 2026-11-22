#include <function/renderer/gui/BuiltinUIProgram.h>
#include <function/renderer/shader/BuiltinSceneProgram.h>
#include <function/renderer/shader/ShaderTargetArchive.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/resources/ShaderAsset/SlangUIModule.h>
#include <function/scene/physics/inxphygpu/InxPhyGpuProgramSources.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace
{
using namespace infernux;

std::string ReadSource(const std::filesystem::path &path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open shader source: " + path.u8string());
    return {std::istreambuf_iterator<char>(file), {}};
}

ShaderSourceModule LoadModule(const std::filesystem::path &root, const char *name, const char *path)
{
    return {name, path, ReadSource(root / path)};
}

// C arrays store exactly the payload bytes. Bound each literal for the
// target C compiler and keep the linked record contiguous for the decoder.
void WriteArchive(std::ostream &out, unsigned index, std::string_view bytes)
{
    constexpr size_t blockSize = 2048;
    constexpr char digits[] = "0123456789abcdef";
    out << "static const struct {\n";
    for (size_t offset = 0; offset < bytes.size(); offset += blockSize)
        out << "    char block" << offset / blockSize << '[' << bytes.substr(offset, blockSize).size() << "];\n";
    out << "} archive" << index << " = {\n";

    std::array<char, blockSize * 4 + 4> literal;
    for (size_t offset = 0; offset < bytes.size(); offset += blockSize) {
        size_t count = 0;
        literal[count++] = '"';
        for (const auto byte : bytes.substr(offset, blockSize)) {
            const auto value = static_cast<unsigned char>(byte);
            literal[count++] = '\\';
            literal[count++] = 'x';
            literal[count++] = digits[value >> 4];
            literal[count++] = digits[value & 15];
        }
        literal[count++] = '"';
        literal[count++] = ',';
        literal[count++] = '\n';
        out.write(literal.data(), static_cast<std::streamsize>(count));
    }
    out << "};\n_Static_assert(sizeof(archive" << index << ") == " << bytes.size()
        << ", \"Shader archive must be contiguous\");\n";
}

template <typename Compile>
void WritePrograms(const std::filesystem::path &output, const char *accessor, unsigned programCount, Compile compile)
{
    std::ofstream file(output, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    // MSVC warns when a C string initializer deliberately fills its array without a terminator.
    file << "#include <stddef.h>\n#ifdef _MSC_VER\n#pragma warning(disable : 4295)\n#endif\n";
    for (unsigned index = 0; index < programCount; ++index) {
        auto compiled = compile(index);
        auto &artifact = compiled.targets.front();
        artifact.reflection.clear(); // Keep typed layout and entry bindings.
        WriteArchive(file, index, EncodeShaderTarget(artifact));
    }

    file << "const void *" << accessor << "(unsigned program, size_t *size)\n{\n"
         << "    static const struct { const void *bytes; size_t size; } archives[] = {\n";
    for (unsigned index = 0; index < programCount; ++index)
        file << "        { &archive" << index << ", sizeof(archive" << index << ") },\n";
    file << "    };\n    *size = archives[program].size;\n    return archives[program].bytes;\n}\n";
    file.close();
}

void CookInxPhyGpuPhysics(const std::filesystem::path &root, ShaderCodeTarget target,
                          const std::filesystem::path &output)
{
    const auto directory = root / "cpp/infernux/function/scene/physics/inxphygpu";
    std::vector<ShaderSourceModule> sources;
    for (const auto &file : std::filesystem::directory_iterator(directory)) {
        if (file.path().extension() == ".slang")
            sources.push_back({"inxphygpu_" + file.path().stem().u8string(), file.path().filename().u8string(),
                               ReadSource(file.path())});
    }
    std::sort(sources.begin(), sources.end(), [](const auto &a, const auto &b) { return a.path < b.path; });
    using physics::inxphygpu::InxPhyGpuProgramSources;
    WritePrograms(output, "InxPhyGpuArchive", std::size(InxPhyGpuProgramSources), [&](unsigned index) {
        const auto &program = InxPhyGpuProgramSources[index];
        const std::string path = std::string(program.module) + ".slang";
        const auto found =
            std::find_if(sources.begin(), sources.end(), [&](const auto &file) { return file.path == path; });
        if (found == sources.end())
            throw std::runtime_error("Missing inxphygpu physics source: " + path);
        auto source = *found;
        if (program.awakeWorlds)
            source.source = "#define INX_AWAKE_SOLVER_WORLDS 1\n#line 1\n" + source.source;
        std::vector<ShaderSourceModule> libraries;
        libraries.reserve(sources.size() - 1);
        libraries.insert(libraries.end(), sources.begin(), found);
        libraries.insert(libraries.end(), found + 1, sources.end());
        return SlangCompiler::Get().Compile(source, libraries, {target}, true);
    });
}

void CookUI(ShaderCodeTarget target, const std::filesystem::path &output)
{
    const ShaderSourceModule *const programs[] = {&GetSlangEditorUIModule(), &GetSlangScreenUIModule(),
                                                  &GetSlangWorldUIModule()};
    static_assert(std::size(programs) == static_cast<unsigned>(BuiltinUIProgram::Count));
    WritePrograms(output, "InfernuxUIShaderArchive", std::size(programs), [&](unsigned index) {
        return SlangCompiler::Get().CompileRootProgram(*programs[index], {}, {target});
    });
}

void CookScene(const std::filesystem::path &root, ShaderCodeTarget target, const std::filesystem::path &output)
{
    const auto shaders = root / "cpp/infernux/function/renderer/shaders";
    const ShaderSourceModule programs[] = {LoadModule(shaders, "inx_scene_depth_resolve", "scene_depth_resolve.slang"),
                                           LoadModule(shaders, "inx_forward_plus_grid", "forward_plus_grid.slang")};
    static_assert(std::size(programs) == static_cast<unsigned>(BuiltinSceneProgram::Count));
    WritePrograms(output, "InfernuxSceneShaderArchive", std::size(programs),
                  [&](unsigned index) { return SlangCompiler::Get().Compile(programs[index], {}, {target}); });
}
} // namespace

int main(int argc, char **argv)
try {
    if (argc != 5)
        throw std::invalid_argument(
            "Usage: InfernuxBuiltinShaderCook <inxphygpu|ui|scene> <engine-source> <spirv|wgsl> <output.c>");
    const std::string_view kind = argv[1];
    const auto root = std::filesystem::u8path(argv[2]);
    const std::string_view format = argv[3];
    if (format != "spirv" && format != "wgsl")
        throw std::invalid_argument("Unknown shader target");
    const auto target = format == "spirv" ? ShaderCodeTarget::SpirV : ShaderCodeTarget::Wgsl;
    const auto output = std::filesystem::u8path(argv[4]);
    if (kind == "inxphygpu")
        CookInxPhyGpuPhysics(root, target, output);
    else if (kind == "ui")
        CookUI(target, output);
    else if (kind == "scene")
        CookScene(root, target, output);
    else
        throw std::invalid_argument("Unknown built-in program group");
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
