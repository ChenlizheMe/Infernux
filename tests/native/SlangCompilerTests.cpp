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
            Require(target.resources.size() == 3, "Typed resource reflection is missing");
            const auto &parameters = target.resources[0];
            Require(parameters.name == "params" && parameters.binding.type == rhi::BindingType::UniformBuffer &&
                        parameters.value.byteSize == 16 && parameters.value.fields.size() == 3 &&
                        parameters.value.fields[2].name == "bias" && parameters.value.fields[2].offset == 8,
                    "Target parameter block layout changed");
            Require(target.resources[1].binding.readOnlyStorage && !target.resources[2].binding.readOnlyStorage,
                    "Storage access qualifiers changed");
            Require(target.reflection.find("params") != std::string::npos, "Constant-buffer reflection is missing");
            Require(target.reflection.find("outputValues") != std::string::npos, "Storage reflection is missing");
            for (const auto &entry : target.entries) {
                Require(!entry.code.empty(), "Shader code is empty");
                if (entry.stage == rhi::ShaderStage::Compute)
                    Require(entry.workgroupSize == std::array<uint32_t, 3>{32, 1, 1}, "Workgroup size changed");
                Require(entry.bindings.size() == (entry.stage == rhi::ShaderStage::Compute ? 3 : 0),
                        "Entry resource use must come from the compiled target metadata");
                if (target.target == ShaderCodeTarget::SpirV)
                    Require(HasSpirvEntry(entry.code, entry.name), "Reflection and SPIR-V entry names differ");
                else {
                    const std::string text(entry.code.begin(), entry.code.end());
                    Require(text.find("fn " + entry.name + "(") != std::string::npos,
                            "Reflection and WGSL entry names differ");
                }
            }
        }

        const auto depthProgram = compiler.Compile({"depth", "depth.frag.py", R"(
DepthTexture2D sceneDepth;
SamplerComparisonState depthSampler;
[shader("fragment")]
float4 shadow(float2 uv : TEXCOORD0) : SV_Target0
{
    return sceneDepth.SampleCmpLevelZero(depthSampler, uv, 0.5).xxxx;
}
)"},
                                                   {}, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
        for (const auto &target : depthProgram.targets) {
            Require(target.resources.size() == 2 && target.resources[0].binding.depthRead &&
                        target.resources[1].binding.samplerType == rhi::SamplerBindingType::Comparison,
                    "Depth texture and comparison sampler lost their target binding types");
        }

        const auto blocks = compiler.Compile({"blocks", "blocks.compute.py", R"(
struct Material { float3 tint; float weight; Texture2D<float4> image; SamplerState sampler; };
struct Scene { float4x4 world; float2 offsets[3]; ParameterBlock<Material> material; };
ParameterBlock<Scene> scene;
float gain;
RWStructuredBuffer<float4> output;
[shader("compute")][numthreads(1,1,1)]
void main()
{
    output[0] = mul(scene.world, float4(scene.material.tint, gain)) * scene.material.weight
        + scene.material.image.SampleLevel(scene.material.sampler, scene.offsets[1], 0);
}
)"},
                                             {}, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
        for (const auto &target : blocks.targets) {
            Require(target.resources.size() == 6 && target.entries[0].bindings.size() == 6,
                    "Nested blocks or implicit globals lost a descriptor or entry use");
            const auto find = [&](const std::string &name) -> const ShaderResourceLayout & {
                for (const auto &resource : target.resources)
                    if (resource.name == name)
                        return resource;
                throw std::runtime_error("Missing logical resource: " + name);
            };
            const auto &scene = find("scene");
            const auto &material = find("scene.material");
            Require(find("$global").set == 0 && find("output").binding.binding == 1 && scene.set == 1 &&
                        scene.binding.binding == 0 && material.set == 2 && material.binding.binding == 0 &&
                        find("scene.material.image").binding.binding == 1 &&
                        find("scene.material.sampler").binding.binding == 2,
                    "Container offsets were applied twice or nested spaces were flattened");
            Require(scene.value.byteSize == 112 && scene.value.fields.size() == 2 &&
                        scene.value.fields[0].kind == ShaderValueKind::Matrix && scene.value.fields[0].stride == 16 &&
                        scene.value.fields[1].kind == ShaderValueKind::Array && scene.value.fields[1].offset == 64 &&
                        scene.value.fields[1].stride == 16 && material.value.fields[1].offset == 12,
                    "Matrix, array or nested uniform contents lost the target's physical layout");
        }

        const auto matrices = compiler.Compile({"matrices", "matrices.compute.py", R"(
struct Values { float3x3 rotation; column_major float2x3 rectangular; };
ConstantBuffer<Values> uniforms;
StructuredBuffer<Values> records;
StructuredBuffer<float3> directions;
RWStructuredBuffer<float> output;
[shader("compute")][numthreads(1,1,1)]
void main() { output[0] = uniforms.rotation[1][2] + uniforms.rectangular[1][2] + records[1].rectangular[1][2] + directions[1].x; }
)"},
                                               {}, {ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl});
        for (const auto &target : matrices.targets) {
            const auto &uniforms = target.resources[0].value.fields;
            const auto &records = target.resources[1];
            Require(uniforms[0].stride == 16 && uniforms[0].rowMajor && uniforms[1].stride == 16 &&
                        !uniforms[1].rowMajor && records.value.fields[1].stride == 8 &&
                        !records.value.fields[1].rowMajor && records.elementStride == 80 &&
                        records.value.byteSize == 80 && target.resources[2].value.byteSize == 12 &&
                        target.resources[2].elementStride == 16,
                    "Rectangular matrices or structured-buffer tail padding lost their target layout");
        }

        // A changed revision with the same virtual module name must not reuse
        // the module cached inside an earlier Slang session.
        const auto previous = result.targets[0].entries[1].code;
        const std::string oldColor = "float3(1.0)), 1.0";
        source.source.replace(source.source.find(oldColor), oldColor.size(), "float3(0.5)), 1.0");
        const auto edited = compiler.Compile(source, {library}, {ShaderCodeTarget::SpirV});
        Require(edited.targets[0].entries[1].code != previous, "Shader edit reused the previous module");

        const auto precise = compiler.Compile(source, {library}, {ShaderCodeTarget::SpirV}, true);
        const auto &numericalCode = precise.targets[0].entries.back().code;
        std::vector<uint32_t> words(numericalCode.size() / sizeof(uint32_t));
        std::memcpy(words.data(), numericalCode.data(), numericalCode.size());
        bool preserve = false;
        bool separate = false;
        for (size_t cursor = 5; cursor < words.size(); cursor += words[cursor] >> 16) {
            const auto opcode = words[cursor] & 0xffffu;
            preserve |= opcode == 16 && words[cursor + 2] == 4461 && words[cursor + 3] == 32;
            separate |= opcode == 71 && words[cursor + 2] == 42;
        }
        Require(preserve && separate, "Numerical programs lost float32 preservation or NoContraction");

        // Standard modules are loaded beside the compiler. Their experimental
        // gate must diagnose the module itself, not a missing packaged file.
        bool standardModuleFound = false;
        try {
            compiler.Compile({"standard", "standard.compute.py",
                              "import slang.numerics; [shader(\"compute\")] [numthreads(1,1,1)] void main() {}"},
                             {}, {ShaderCodeTarget::SpirV});
        } catch (const std::runtime_error &error) {
            standardModuleFound = std::string(error.what()).find("experimental module") != std::string::npos;
        }
        Require(standardModuleFound, "Packaged Slang standard modules were not found");

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
