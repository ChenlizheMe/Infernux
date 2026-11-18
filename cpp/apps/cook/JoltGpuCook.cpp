#include <Jolt/GPU/PhysicsStages.h>
#include <function/renderer/shader/ShaderTargetArchive.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/scene/physics/JoltGpuProgramSources.h>

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
        throw std::runtime_error("Cannot open GPU physics source: " + path.u8string());
    return {std::istreambuf_iterator<char>(file), {}};
}

ShaderSourceModule LoadModule(const std::filesystem::path &jolt, const char *name, const char *path)
{
    return {name, path, ReadSource(jolt / path)};
}

ShaderSourceModule LoadProgramSource(const std::filesystem::path &root, const std::filesystem::path &jolt,
                                     JoltGpuProgram program)
{
    switch (program) {
    case JoltGpuProgram::ArticulationCapture:
        return LoadModule(jolt, "jph_gpu_articulation_capture", "Jolt/GPU/Shaders/articulation_normal_capture.slang");
    case JoltGpuProgram::ArticulationDynamics:
        return LoadModule(jolt, "jph_gpu_articulation_dynamics_owner",
                          "Jolt/GPU/Shaders/articulation_normal_dynamics.slang");
    case JoltGpuProgram::Step:
        return LoadModule(jolt, JPH::GPU::StepModuleName, JPH::GPU::StepSourcePath);
    case JoltGpuProgram::Broadphase:
        return LoadModule(jolt, "jph_gpu_broadphase", "Jolt/GPU/Shaders/rigid_broadphase.slang");
    case JoltGpuProgram::Narrowphase:
        return LoadModule(jolt, "jph_gpu_narrowphase", "Jolt/GPU/Shaders/rigid_convex_narrowphase.slang");
    case JoltGpuProgram::ContactReports:
        return LoadModule(jolt, "jph_gpu_contact_reports", "Jolt/GPU/Shaders/rigid_contact_reports.slang");
    case JoltGpuProgram::Constraints:
        return LoadModule(jolt, "jph_gpu_constraints", "Jolt/GPU/Shaders/rigid_constraints.slang");
    case JoltGpuProgram::BodyMutations:
        return LoadModule(jolt, "jph_gpu_body_mutations", "Jolt/GPU/Shaders/rigid_body_mutations.slang");
    case JoltGpuProgram::BodyCommands:
        return LoadModule(jolt, "jph_gpu_body_commands", "Jolt/GPU/Shaders/body_commands.slang");
    case JoltGpuProgram::Queries: {
        auto source = LoadModule(jolt, "jph_gpu_queries", "Jolt/GPU/Shaders/rigid_queries.slang");
        source.source += "\n#include \"../../../infernux/renderer/particle/collision_export.slang\"\n";
        return source;
    }
    case JoltGpuProgram::RenderPoses:
        return LoadModule(jolt, "jph_gpu_render_poses", "Jolt/GPU/Shaders/render_poses.slang");
    case JoltGpuProgram::Joints:
        return LoadModule(jolt, "jph_gpu_joints", "Jolt/GPU/Shaders/joint_commands.slang");
    case JoltGpuProgram::CrossDomainCandidates:
        return {"inx_cross_domain_candidates", "Jolt/GPU/Shaders/cross_domain_candidates.slang",
                ReadSource(root / "cpp/infernux/function/scene/physics/shaders/cross_domain_candidates.slang")};
    case JoltGpuProgram::CrossDomainContactSolver:
        return {"inx_cross_domain_contact_solver", "Jolt/GPU/Shaders/cross_domain_contact_solver.slang",
                ReadSource(root / "cpp/infernux/function/scene/physics/shaders/cross_domain_contact_solver.slang")};
    case JoltGpuProgram::Count:
        break;
    }
    throw std::logic_error("Missing GPU physics program source");
}

std::vector<ShaderSourceModule> LoadProgramLibraries(const std::filesystem::path &root,
                                                     const std::filesystem::path &jolt)
{
    const auto load = [&](const char *name, const char *path) { return LoadModule(jolt, name, path); };
    return {load("jph_gpu_rigid_math", "Jolt/GPU/Shaders/rigid_math.slang"),
            load("jph_gpu_component_union", "Jolt/GPU/Shaders/component_union.slang"),
            load("jph_gpu_key_sort", "Jolt/GPU/Shaders/key_sort.slang"),
            load("jph_gpu_joint_fields", "Jolt/GPU/JointPayloadFields.inl"),
            load("jph_gpu_physics_types", "Jolt/GPU/PhysicsTypes.h"),
            load("jph_gpu_continuous_types", "Jolt/GPU/ContinuousTypes.h"),
            load("jph_gpu_articulation_integration_types", "Jolt/GPU/ArticulationIntegrationTypes.h"),
            load("jph_gpu_state_fk_types", "Jolt/GPU/StateFkTypes.h"),
            load("jph_gpu_articulation_dynamics_types", "Jolt/GPU/ArticulationDynamicsTypes.h"),
            load("jph_gpu_articulation_aba", "Jolt/GPU/Shaders/articulation_aba.slang"),
            load("jph_gpu_articulation_joint_load", "Jolt/GPU/Shaders/articulation_joint_load.slang"),
            load("jph_gpu_articulation_kinematics", "Jolt/GPU/Shaders/articulation_kinematics.slang"),
            load("jph_gpu_articulation_state_fk", "Jolt/GPU/Shaders/articulation_state_fk.slang"),
            load("jph_gpu_articulation_dynamics", "Jolt/GPU/Shaders/articulation_dynamics.slang"),
            load("jph_gpu_articulation_types", "Jolt/GPU/ArticulationTypes.h"),
            load("jph_gpu_articulation_control", "Jolt/GPU/ArticulationControlTypes.h"),
            load("jph_gpu_articulations", "Jolt/GPU/Shaders/rigid_articulations.slang"),
            load("jph_gpu_sleep_types", "Jolt/GPU/SleepTypes.h"),
            load("jph_gpu_sleep_math", "Jolt/GPU/Shaders/rigid_sleep_math.slang"),
            load("jph_gpu_sleep", "Jolt/GPU/Shaders/rigid_sleep.slang"),
            load("jph_gpu_convex_types", "Jolt/GPU/ConvexGeometryTypes.h"),
            load("jph_gpu_convex", "Jolt/GPU/Shaders/rigid_convex.slang"),
            load("jph_gpu_convex_manifold", "Jolt/GPU/Shaders/rigid_convex_manifold.slang"),
            load("jph_gpu_convex_queries", "Jolt/GPU/Shaders/rigid_convex_queries.slang"),
            load("jph_gpu_contact_report_types", "Jolt/GPU/ContactReportTypes.h"),
            load("jph_gpu_broadphase_types", "Jolt/GPU/BroadphaseTypes.h"),
            load("jph_gpu_constraint_types", "Jolt/GPU/ConstraintTypes.h"),
            load("jph_gpu_island_owner", "Jolt/GPU/Shaders/island_owner_schedule.slang"),
            load("jph_gpu_rigid_psd", "Jolt/GPU/Shaders/rigid_psd.slang"),
            load("jph_gpu_rigid_joints", "Jolt/GPU/Shaders/rigid_joints.slang"),
            load("jph_gpu_swing_twist", "Jolt/GPU/Shaders/rigid_swing_twist.slang"),
            load("jph_gpu_six_dof", "Jolt/GPU/Shaders/rigid_six_dof.slang"),
            {"inx_particle_collision_export", "infernux/renderer/particle/collision_export.slang",
             ReadSource(root / "cpp/infernux/function/renderer/particle/shaders/collision_export.slang")}};
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
        << ", \"GPU physics archive must be contiguous\");\n";
}

void Cook(const std::filesystem::path &root, ShaderCodeTarget target, const std::filesystem::path &output)
{
    const auto jolt = root / "external/joltphysics_for_infernux";
    const auto libraries = LoadProgramLibraries(root, jolt);
    auto &compiler = SlangCompiler::Get();
    std::ofstream file(output, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    // MSVC warns when a C string initializer deliberately fills its array without a terminator.
    file << "#include <stddef.h>\n#ifdef _MSC_VER\n#pragma warning(disable : 4295)\n#endif\n";
    constexpr unsigned programCount = static_cast<unsigned>(JoltGpuProgram::Count);
    for (unsigned index = 0; index < programCount; ++index) {
        const auto source = LoadProgramSource(root, jolt, static_cast<JoltGpuProgram>(index));
        auto compiled = compiler.Compile(source, libraries, {target}, true);
        auto &artifact = compiled.targets.front();
        artifact.reflection.clear(); // Keep typed layout and entry bindings.
        WriteArchive(file, index, EncodeShaderTarget(artifact));
    }

    file << "const void *InfernuxJoltGpuArchive(unsigned program, size_t *size)\n{\n"
         << "    static const struct { const void *bytes; size_t size; } archives[] = {\n";
    for (unsigned index = 0; index < programCount; ++index)
        file << "        { &archive" << index << ", sizeof(archive" << index << ") },\n";
    file << "    };\n    *size = archives[program].size;\n    return archives[program].bytes;\n}\n";
    file.close();
}
} // namespace

int main(int argc, char **argv)
try {
    if (argc != 4)
        throw std::invalid_argument("Usage: InfernuxJoltGpuCook <engine-source> <spirv|wgsl> <output.c>");
    const std::string_view format = argv[2];
    if (format != "spirv" && format != "wgsl")
        throw std::invalid_argument("Unknown GPU physics shader target");
    Cook(std::filesystem::u8path(argv[1]), format == "spirv" ? ShaderCodeTarget::SpirV : ShaderCodeTarget::Wgsl,
         std::filesystem::u8path(argv[3]));
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
