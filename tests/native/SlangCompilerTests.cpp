#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace infernux;

namespace
{
void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

bool HasSpirvEntry(const std::vector<char> &code, const std::string &name)
{
    std::vector<uint32_t> words(code.size() / sizeof(uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    Require(words.at(0) == 0x07230203u, "Slang did not produce SPIR-V");
    for (size_t cursor = 5; cursor < words.size(); cursor += words[cursor] >> 16) {
        if ((words[cursor] & 0xffffu) == 15u &&
            std::string(reinterpret_cast<const char *>(words.data() + cursor + 3)) == name)
            return true;
    }
    return false;
}
} // namespace

int main()
{
    try {
        for (const auto &kind : asset_formats::kShaderSourceKinds) {
            const auto path = "Assets/nested/program" + std::string(kind.suffix);
            Require(asset_formats::SourceExtension(path) == kind.suffix, "Compound shader suffix lost");
        }
        Require(asset_formats::SourceExtension("Assets/nested/program.FRaG.PY") == ".frag.py",
                "Shader suffix classification must be case insensitive");
        Require(asset_formats::SourceExtension("Assets/program.frag.py.txt") == ".txt",
                "Only the final complete shader suffix classifies a source");
        auto &compiler = SlangCompiler::Get();
        const ShaderSourceModule library{"palette", "palette.shaderlib.py", R"(
public float3 tint(float3 color) { return color * float3(1.0, 0.5, 0.25); }
)"};
        ShaderSourceModule source{"fixture", "fixture.compute.py", R"(
import palette;
struct Parameters { float scale; uint count; float2 bias; };
ConstantBuffer<Parameters> params;
StructuredBuffer<float> inputValues;
RWStructuredBuffer<float> outputValues;
[Differentiable]
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
[shader("vertex")]
float4 triangle(uint id : SV_VertexID) : SV_Position
{
    return float4(float(id & 1) * 2.0 - 1.0, float(id >> 1) * 2.0 - 1.0, 0.0, 1.0);
}
[shader("fragment")]
float4 color() : SV_Target { return float4(tint(float3(1.0)), 1.0); }
[shader("compute")]
[numthreads(32, 1, 1)]
void affine(uint3 id : SV_DispatchThreadID)
{
    if (id.x < params.count)
        outputValues[id.x] = inputValues[id.x] * params.scale + params.bias.x;
}
[shader("compute")]
[numthreads(32, 1, 1)]
void derivative(uint3 id : SV_DispatchThreadID)
{
    if (id.x < params.count) {
        DifferentialPair<float> x = diffPair(inputValues[id.x], 0.0);
        bwd_diff(sigmoid)(x, 1.0);
        outputValues[id.x] = x.d;
    }
}
)"};
        const auto start = std::chrono::steady_clock::now();
        const auto result = compiler.Compile(source, {library}, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
        Require(result.targets.size() == 2, "Both targets must be generated");
        for (const auto &target : result.targets) {
            Require(target.entries.size() == 4, "The module's four entries must be retained");
            Require(target.reflection.find("params") != std::string::npos, "Constant-buffer reflection is missing");
            Require(target.reflection.find("outputValues") != std::string::npos, "Storage reflection is missing");
            for (const auto &entry : target.entries) {
                Require(!entry.code.empty(), "Shader code is empty");
                if (entry.stage == rhi::ShaderStage::Compute)
                    Require(entry.workgroupSize == std::array<uint32_t, 3>{32, 1, 1}, "Workgroup size changed");
                if (target.target == ShaderCodeTarget::SpirV)
                    Require(HasSpirvEntry(entry.code, entry.name), "Reflection and SPIR-V entry names differ");
                else {
                    const std::string text(entry.code.begin(), entry.code.end());
                    Require(text.find("fn " + entry.name + "(") != std::string::npos,
                            "Reflection and WGSL entry names differ");
                }
            }
        }

        // A changed revision with the same virtual module name must not reuse
        // the module cached inside an earlier Slang session.
        const auto previous = result.targets[0].entries[1].code;
        const std::string oldColor = "float3(1.0)), 1.0";
        source.source.replace(source.source.find(oldColor), oldColor.size(), "float3(0.5)), 1.0");
        const auto edited = compiler.Compile(source, {library}, {ShaderCodeTarget::SpirV});
        Require(edited.targets[0].entries[1].code != previous, "Shader edit reused the previous module");

        bool diagnosed = false;
        try {
            compiler.Compile({"broken", "broken.frag.py", "float4 broken( {"}, {}, {ShaderCodeTarget::SpirV});
        } catch (const std::runtime_error &error) {
            diagnosed = std::string(error.what()).find("broken.frag.py") != std::string::npos;
        }
        Require(diagnosed, "Compiler diagnostics lost the source location");
        std::cout
            << "Slang " << compiler.GetVersion()
            << ": vertex/fragment/compute/AD, direct SPIR-V + WGSL, layout, imports, edit and diagnostics passed in "
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << " ms\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
