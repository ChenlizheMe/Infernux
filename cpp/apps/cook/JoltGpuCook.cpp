#include <function/renderer/gui/BuiltinUIProgram.h>
#include <function/renderer/shader/BuiltinSceneProgram.h>
#include <function/renderer/shader/ShaderTargetArchive.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/resources/ShaderAsset/SlangUIModule.h>
#include <function/scene/physics/JoltGpuProgramSources.h>
#include <function/scene/physics/gpu/PhysicsStages.h>
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

ShaderSourceModule LoadProgramSource(const std::filesystem::path &physics, JoltGpuProgram program)
{
    switch (program) {
    case JoltGpuProgram::ArticulationCapture:
        return LoadModule(physics, "jph_gpu_articulation_capture", "gpu/Shaders/articulation_normal_capture.slang");
    case JoltGpuProgram::ArticulationDynamics:
        return LoadModule(physics, "jph_gpu_articulation_dynamics_owner",
                          "gpu/Shaders/articulation_normal_dynamics.slang");
    case JoltGpuProgram::Step:
        return LoadModule(physics, JPH::GPU::StepModuleName, JPH::GPU::StepSourcePath);
    case JoltGpuProgram::Broadphase:
        return LoadModule(physics, "jph_gpu_broadphase", "gpu/Shaders/rigid_broadphase.slang");
    case JoltGpuProgram::Narrowphase:
        return LoadModule(physics, "jph_gpu_narrowphase", "gpu/Shaders/rigid_convex_narrowphase.slang");
    case JoltGpuProgram::ContactReports:
        return LoadModule(physics, "jph_gpu_contact_reports", "gpu/Shaders/rigid_contact_reports.slang");
    case JoltGpuProgram::Constraints:
        return LoadModule(physics, "jph_gpu_constraints", "gpu/Shaders/rigid_constraints.slang");
    case JoltGpuProgram::BodyMutations:
        return LoadModule(physics, "jph_gpu_body_mutations", "gpu/Shaders/rigid_body_mutations.slang");
    case JoltGpuProgram::BodyCommands:
        return LoadModule(physics, "jph_gpu_body_commands", "gpu/Shaders/body_commands.slang");
    case JoltGpuProgram::Joints:
        return LoadModule(physics, "jph_gpu_joints", "gpu/Shaders/joint_commands.slang");
    case JoltGpuProgram::Count:
        break;
    }
    throw std::logic_error("Missing GPU physics program source");
}

std::vector<ShaderSourceModule> LoadProgramLibraries(const std::filesystem::path &physics)
{
    const auto load = [&](const char *name, const char *path) { return LoadModule(physics, name, path); };
    return {load("jph_gpu_rigid_math", "gpu/Shaders/rigid_math.slang"),
            load("jph_gpu_component_union", "gpu/Shaders/component_union.slang"),
            load("jph_gpu_key_sort", "gpu/Shaders/key_sort.slang"),
            load("jph_gpu_joint_fields", "gpu/JointPayloadFields.inl"),
            load("jph_gpu_physics_types", "gpu/PhysicsTypes.h"),
            load("jph_gpu_continuous_types", "gpu/ContinuousTypes.h"),
            load("jph_gpu_articulation_integration_types", "gpu/ArticulationIntegrationTypes.h"),
            load("jph_gpu_state_fk_types", "gpu/StateFkTypes.h"),
            load("jph_gpu_articulation_dynamics_types", "gpu/ArticulationDynamicsTypes.h"),
            load("jph_gpu_articulation_aba", "gpu/Shaders/articulation_aba.slang"),
            load("jph_gpu_articulation_joint_load", "gpu/Shaders/articulation_joint_load.slang"),
            load("jph_gpu_articulation_kinematics", "gpu/Shaders/articulation_kinematics.slang"),
            load("jph_gpu_articulation_state_fk", "gpu/Shaders/articulation_state_fk.slang"),
            load("jph_gpu_articulation_dynamics", "gpu/Shaders/articulation_dynamics.slang"),
            load("jph_gpu_articulation_types", "gpu/ArticulationTypes.h"),
            load("jph_gpu_articulation_control", "gpu/ArticulationControlTypes.h"),
            load("jph_gpu_articulations", "gpu/Shaders/rigid_articulations.slang"),
            load("jph_gpu_sleep_types", "gpu/SleepTypes.h"),
            load("jph_gpu_sleep_math", "gpu/Shaders/rigid_sleep_math.slang"),
            load("jph_gpu_sleep", "gpu/Shaders/rigid_sleep.slang"),
            load("jph_gpu_convex_types", "gpu/ConvexGeometryTypes.h"),
            load("jph_gpu_convex", "gpu/Shaders/rigid_convex.slang"),
            load("jph_gpu_convex_manifold", "gpu/Shaders/rigid_convex_manifold.slang"),
            load("jph_gpu_contact_report_types", "gpu/ContactReportTypes.h"),
            load("jph_gpu_broadphase_types", "gpu/BroadphaseTypes.h"),
            load("jph_gpu_constraint_types", "gpu/ConstraintTypes.h"),
            load("jph_gpu_island_owner", "gpu/Shaders/island_owner_schedule.slang"),
            load("jph_gpu_rigid_psd", "gpu/Shaders/rigid_psd.slang"),
            load("jph_gpu_rigid_joints", "gpu/Shaders/rigid_joints.slang"),
            load("jph_gpu_swing_twist", "gpu/Shaders/rigid_swing_twist.slang"),
            load("jph_gpu_six_dof", "gpu/Shaders/rigid_six_dof.slang")};
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

void CookPhysics(const std::filesystem::path &root, ShaderCodeTarget target, const std::filesystem::path &output)
{
    const auto physics = root / "cpp/infernux/function/scene/physics";
    const auto libraries = LoadProgramLibraries(physics);
    WritePrograms(output, "InfernuxJoltGpuArchive", static_cast<unsigned>(JoltGpuProgram::Count), [&](unsigned index) {
        const auto source = LoadProgramSource(physics, static_cast<JoltGpuProgram>(index));
        return SlangCompiler::Get().Compile(source, libraries, {target}, true);
    });
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
            "Usage: InfernuxJoltGpuCook <physics|inxphygpu|ui|scene> <engine-source> <spirv|wgsl> <output.c>");
    const std::string_view kind = argv[1];
    const auto root = std::filesystem::u8path(argv[2]);
    const std::string_view format = argv[3];
    if (format != "spirv" && format != "wgsl")
        throw std::invalid_argument("Unknown shader target");
    const auto target = format == "spirv" ? ShaderCodeTarget::SpirV : ShaderCodeTarget::Wgsl;
    const auto output = std::filesystem::u8path(argv[4]);
    if (kind == "physics")
        CookPhysics(root, target, output);
    else if (kind == "inxphygpu")
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
