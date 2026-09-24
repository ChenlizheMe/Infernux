#include <core/types/ShaderProgramArtifact.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <string>

namespace
{

using infernux::InxMaterial;
using infernux::MaterialBlendFactor;
using infernux::MaterialCompareOp;
using infernux::MaterialCullMode;
using infernux::MaterialSamplerAddress;
using infernux::MaterialSamplerFilter;
using infernux::MaterialTextureSampler;
using infernux::RenderStateOverride;
using infernux::ShaderAssetReference;
using infernux::ShaderProgramArtifact;
using infernux::ShaderProgramPropertyBinding;
using infernux::ShaderProgramStageMask;

void VerifyRetiredFieldsAreIgnored()
{
    InxMaterial material("Current", "Lit");
    auto document = material.SerializeDocument();
    document["material_version"] = 4;
    document["shaders"]["vertex"]["path_hint"] = "Assets/Shaders/Stale.vert";
    document["shaders"]["vertex"]["unexpected"] = true;

    assert(material.DeserializeDocument(document));
    const auto current = material.SerializeDocument();
    assert(!current.contains("material_version"));
    assert(!current["shaders"]["vertex"].contains("path_hint"));
    assert(!current["shaders"]["vertex"].contains("unexpected"));
}

void VerifyGizmoIconPreservesAuthoredAlpha()
{
    // Every built-in icon uses this same material factory; only its texture
    // differs. Transparent interiors must blend, not become binary holes.
    const auto material = InxMaterial::CreateComponentGizmoIconMaterial();
    const auto &state = material->GetRenderState();
    assert(state.blendEnable);
    assert(state.srcColorBlendFactor == MaterialBlendFactor::SourceAlpha);
    assert(state.dstColorBlendFactor == MaterialBlendFactor::OneMinusSourceAlpha);
    assert(!state.alphaClipEnabled);
    assert(state.alphaClipThreshold == 0.0f);
    assert(state.depthTestEnable && !state.depthWriteEnable);
    // Shader defaults cannot turn the explicitly authored blend into a mask.
    material->ApplyShaderRenderMeta("", "", "", "", 2000, "", "", "0.5");
    assert(!material->GetRenderState().alphaClipEnabled);
}

void VerifyStableReferencesAndClone()
{
    InxMaterial material("Stable", "Unlit");
    const ShaderAssetReference vertex{"vertex-guid", "Standard", "Assets/Shaders/Standard.vert"};
    const ShaderAssetReference fragment{"fragment-guid", "Unlit", "Assets/Shaders/Unlit.frag"};
    material.SetVertShaderReference(vertex);
    material.SetFragShaderReference(fragment);

    const auto document = material.SerializeDocument();
    assert(!document["shaders"]["vertex"].contains("path_hint"));
    assert(!document["shaders"]["fragment"].contains("path_hint"));
    InxMaterial restored;
    assert(restored.DeserializeDocument(document));
    const ShaderAssetReference persistedVertex{"vertex-guid", "Standard", ""};
    const ShaderAssetReference persistedFragment{"fragment-guid", "Unlit", ""};
    assert(restored.GetVertShaderReference() == persistedVertex);
    assert(restored.GetFragShaderReference() == persistedFragment);
    assert(restored.GetVertShaderReference().pathHint.empty());
    assert(restored.GetFragShaderReference().pathHint.empty());
    assert(restored.GetShaderId() == "vertex-guid|fragment-guid");

    const std::shared_ptr<InxMaterial> clone = restored.Clone();
    assert(clone);
    assert(clone->GetVertShaderReference() == persistedVertex);
    assert(clone->GetFragShaderReference() == persistedFragment);
}

void VerifyTransactionalFailure()
{
    InxMaterial material("Stable", "Unlit");
    material.SetVertShaderReference({"vertex-guid", "Standard", "Assets/Shaders/Standard.vert"});
    material.SetFragShaderReference({"fragment-guid", "Unlit", "Assets/Shaders/Unlit.frag"});
    const auto before = material.SerializeDocument();

    auto invalid = before;
    invalid["name"] = "PartialMutation";
    invalid["shaders"]["fragment"] = {
        {"guid", ""},
        {"shader_id", ""},
        {"path_hint", "Assets/Shaders/Missing.frag"},
    };
    assert(!material.DeserializeDocument(invalid));
    assert(material.SerializeDocument() == before);
}

void VerifyMaterialIdentityIsNotSourceProvenance()
{
    InxMaterial first("Embedded", "Lit");
    InxMaterial second("Embedded", "Lit");
    const auto firstKey = first.GetMaterialKey();
    const auto secondKey = second.GetMaterialKey();
    assert(firstKey != secondKey);
    first.SetFilePath("Assets/Models/shared.obj::submat:0");
    second.SetFilePath(first.GetFilePath());
    assert(first.GetMaterialKey() == firstKey);
    assert(second.GetMaterialKey() == secondKey);
    first.SetName("Renamed");
    first.SetFilePath("Assets/Models/moved.obj::submat:0");
    assert(first.GetMaterialKey() == firstKey);
    const InxMaterial copied(first);
    assert(copied.GetMaterialKey() != firstKey);

    first.SetGuid("shared-material-guid");
    second.SetGuid("shared-material-guid");
    assert(first.GetMaterialKey() == "shared-material-guid");
    assert(first.GetMaterialKey() == second.GetMaterialKey());
    const auto clone = first.Clone();
    clone->SetFilePath(first.GetFilePath());
    clone->SetName(first.GetName());
    assert(clone->GetMaterialKey() != first.GetMaterialKey());
    assert(clone->GetMaterialKey() != copied.GetMaterialKey());
}

void VerifyRenderStateVersioning()
{
    InxMaterial material("LiveState", "Unlit");
    const uint64_t initialVersion = material.GetVersion();
    material.SetRenderQueue(3042);
    assert(material.GetVersion() == initialVersion + 1);
    material.SetRenderQueue(3042);
    assert(material.GetVersion() == initialVersion + 1);

    auto state = material.GetRenderState();
    state.blendEnable = !state.blendEnable;
    material.SetRenderState(state);
    assert(material.GetVersion() == initialVersion + 2);
    material.SetRenderState(state);
    assert(material.GetVersion() == initialVersion + 2);
}

void VerifyShaderReferenceVersioning()
{
    InxMaterial material("LiveShader", "Unlit");
    const uint64_t initialVersion = material.GetVersion();
    material.SetVertShader("Standard");
    assert(material.GetVersion() == initialVersion + 1);
    material.SetVertShader("Standard");
    assert(material.GetVersion() == initialVersion + 1);

    const ShaderAssetReference reference{"vertex-guid", "Standard", "Assets/Shaders/Standard.vert"};
    material.SetVertShaderReference(reference);
    assert(material.GetVersion() == initialVersion + 2);
    material.SetVertShaderReference(reference);
    assert(material.GetVersion() == initialVersion + 2);

    material.SetShader("Unlit");
    assert(material.GetVersion() == initialVersion + 3);
    assert((material.GetVertShaderReference() == ShaderAssetReference{"", "Unlit", ""}));
    assert((material.GetFragShaderReference() == ShaderAssetReference{"", "Unlit", ""}));
}

void VerifyPropertyRemoval()
{
    InxMaterial material("PropertyRemoval", "Unlit");
    material.SetFloat("blend_enable", 1.0f);
    const uint64_t populatedVersion = material.GetVersion();
    assert(material.HasProperty("blend_enable"));

    assert(material.RemoveProperty("blend_enable"));
    assert(!material.HasProperty("blend_enable"));
    assert(material.GetVersion() == populatedVersion + 1);
    assert(!material.RemoveProperty("blend_enable"));
    assert(material.GetVersion() == populatedVersion + 1);
}

void VerifyShaderDefaultsReplacePreviousShaderState()
{
    // Shader annotations only supply defaults for fields the material has not
    // authored. SetRenderState is explicit authoring of every field, so
    // annotation defaults must not touch the state while the shader stays.
    InxMaterial material("ShaderDefaults", "Particle Unlit");
    auto particleState = material.GetRenderState();
    particleState.cullMode = MaterialCullMode::None;
    particleState.depthWriteEnable = false;
    particleState.blendEnable = true;
    particleState.renderQueue = 3000;
    material.SetRenderState(particleState);
    assert(material.HasOverride(RenderStateOverride::CullMode));
    assert(material.HasOverride(RenderStateOverride::RenderQueue));

    material.ApplyShaderRenderMeta("back", "true", "", "off", 2000, "", "", "");
    const auto &authored = material.GetRenderState();
    assert(authored.cullMode == MaterialCullMode::None);
    assert(!authored.depthWriteEnable);
    assert(authored.blendEnable);
    assert(authored.renderQueue == 3000);

    // Switching to a different shader hands authorship back to the new
    // shader's annotation defaults.
    material.SetShader("Lit");
    assert(material.GetRenderStateOverrides() == 0);
    material.ApplyShaderRenderMeta("", "", "", "", 2000, "", "", "");
    const auto &litState = material.GetRenderState();
    assert(litState.cullMode == MaterialCullMode::Back);
    assert(litState.depthWriteEnable);
    assert(!litState.blendEnable);
    assert(litState.renderQueue == 2000);
    assert(!litState.stencilTestEnable);
    assert(material.GetPassTag().empty());

    material.ApplyShaderRenderMeta("none", "false", "", "alpha", 3000, "particle", "", "");
    const auto &nextParticleState = material.GetRenderState();
    assert(nextParticleState.cullMode == MaterialCullMode::None);
    assert(!nextParticleState.depthWriteEnable);
    assert(nextParticleState.blendEnable);
    assert(nextParticleState.renderQueue == 3000);
    assert(material.GetPassTag() == "particle");

    material.ApplyShaderRenderMeta("none", "false", "", "premultiplied", 3000, "particle", "", "");
    const auto &premultipliedState = material.GetRenderState();
    assert(premultipliedState.blendEnable);
    assert(premultipliedState.srcColorBlendFactor == MaterialBlendFactor::One);
    assert(premultipliedState.dstColorBlendFactor == MaterialBlendFactor::OneMinusSourceAlpha);

    // Resolving the same shader to a concrete asset reference is not a
    // switch: authored fields keep their override bits.
    material.SetRenderState(material.GetRenderState());
    assert(material.GetRenderStateOverrides() != 0);
    material.SetFragShaderReference(ShaderAssetReference{"frag-guid", "Lit", "Assets/Shaders/Lit.frag"});
    assert(material.GetRenderStateOverrides() != 0);

    // Moving the file (new pathHint) or migrating a renamed shader id under
    // the same GUID references the same asset: authorship survives.
    material.SetFragShaderReference(ShaderAssetReference{"frag-guid", "Lit", "Assets/Moved/Lit.frag"});
    assert(material.GetRenderStateOverrides() != 0);
    material.SetFragShaderReference(ShaderAssetReference{"frag-guid", "Lit Renamed", "Assets/Moved/LitRenamed.frag"});
    assert(material.GetRenderStateOverrides() != 0);

    // A different GUID is a real switch even when the display id matches.
    material.SetFragShaderReference(ShaderAssetReference{"other-guid", "Lit Renamed", "Assets/Other.frag"});
    assert(material.GetRenderStateOverrides() == 0);
}

void VerifyMaterialOverridesSurviveShaderDefaults()
{
    InxMaterial material("ShaderOverrides", "Lit");
    auto state = material.GetRenderState();
    state.cullMode = MaterialCullMode::Front;
    state.depthWriteEnable = false;
    state.blendEnable = true;
    state.renderQueue = 3456;
    material.SetRenderState(state);
    material.MarkOverride(RenderStateOverride::CullMode);
    material.MarkOverride(RenderStateOverride::DepthWrite);
    material.MarkOverride(RenderStateOverride::BlendEnable);
    material.MarkOverride(RenderStateOverride::RenderQueue);

    material.ApplyShaderRenderMeta("back", "true", "", "off", 2000, "", "", "");
    const auto &preserved = material.GetRenderState();
    assert(preserved.cullMode == MaterialCullMode::Front);
    assert(!preserved.depthWriteEnable);
    assert(preserved.blendEnable);
    assert(preserved.renderQueue == 3456);
}

void VerifyBuiltinSixWaySmokeMaterial()
{
    const auto material = InxMaterial::CreateParticleSixWaySmokeMaterial();
    assert(material);
    assert(material->IsBuiltin());
    assert(material->GetVertShaderName() == "Particle Sprite");
    assert(material->GetFragShaderName() == "Particle Six-Way Smoke");

    const auto document = material->SerializeDocument();
    assert(document.at("builtin").get<bool>());
    assert(document.at("properties").at("positiveAxesMap").at("guid") == "white");
    assert(document.at("properties").at("negativeAxesMap").at("guid") == "black");
    assert(document.at("properties").at("ambientIntensity").at("value") == 0.0f);

    const auto &state = material->GetRenderState();
    assert(state.renderQueue == 3000);
    assert(state.cullMode == MaterialCullMode::None);
    assert(state.depthTestEnable);
    assert(!state.depthWriteEnable);
    assert(state.blendEnable);
    assert(state.srcColorBlendFactor == MaterialBlendFactor::One);
    assert(state.dstColorBlendFactor == MaterialBlendFactor::OneMinusSourceAlpha);
}

void VerifyBackendNeutralRenderStateSchema()
{
    static_assert(static_cast<uint32_t>(MaterialCullMode::None) == 0);
    static_assert(static_cast<uint32_t>(MaterialCullMode::Front) == 1);
    static_assert(static_cast<uint32_t>(MaterialCullMode::Back) == 2);
    static_assert(static_cast<uint32_t>(MaterialCompareOp::Always) == 7);
    static_assert(static_cast<uint32_t>(MaterialBlendFactor::One) == 1);
    static_assert(static_cast<uint32_t>(MaterialBlendFactor::OneMinusSourceAlpha) == 7);

    InxMaterial material("PortableSchema", "Lit");
    auto state = material.GetRenderState();
    state.cullMode = MaterialCullMode::Front;
    state.depthCompareOp = MaterialCompareOp::GreaterOrEqual;
    state.srcColorBlendFactor = MaterialBlendFactor::One;
    state.dstColorBlendFactor = MaterialBlendFactor::OneMinusSourceAlpha;
    material.SetRenderState(state);

    const auto document = material.SerializeDocument();
    const auto &renderState = document.at("renderState");
    assert(renderState.at("cullMode") == 1);
    assert(renderState.at("depthCompareOp") == 6);
    assert(renderState.at("srcColorBlendFactor") == 1);
    assert(renderState.at("dstColorBlendFactor") == 7);

    InxMaterial restored;
    assert(restored.DeserializeDocument(document));
    assert(restored.GetRenderState() == state);
}

void VerifySparseMaterialUsesLinkedShaderDefaults()
{
    InxMaterial material("Sparse", "Lit");
    material.SetColor("baseColor", glm::vec4(0.25f, 0.5f, 0.75f, 1.0f));

    ShaderProgramArtifact artifact;
    ShaderProgramPropertyBinding baseColor;
    baseColor.name = "baseColor";
    baseColor.type = "Color";
    baseColor.defaultValue = "[1.0,1.0,1.0,1.0]";
    baseColor.stages = ShaderProgramStageMask::Fragment;
    baseColor.bufferOffset = 0;
    baseColor.byteSize = 16;
    baseColor.byteAlignment = 16;
    artifact.properties.push_back(baseColor);

    ShaderProgramPropertyBinding smoothness;
    smoothness.name = "smoothness";
    smoothness.type = "Float";
    smoothness.defaultValue = "0.5";
    smoothness.stages = ShaderProgramStageMask::Fragment;
    smoothness.bufferOffset = 16;
    smoothness.byteSize = 4;
    smoothness.byteAlignment = 4;
    artifact.properties.push_back(smoothness);

    const uint64_t authoredVersion = material.GetAuthoredVersion();
    assert(material.SynchronizeShaderPropertyDefaults(artifact));
    assert(material.GetAuthoredVersion() == authoredVersion);
    assert(std::get<glm::vec4>(material.GetProperty("baseColor")->value) == glm::vec4(0.25f, 0.5f, 0.75f, 1.0f));
    assert(std::get<float>(material.GetProperty("smoothness")->value) == 0.5f);
    const uint64_t synchronizedVersion = material.GetVersion();
    assert(!material.SynchronizeShaderPropertyDefaults(artifact));
    assert(material.GetVersion() == synchronizedVersion);
    material.ApplyShaderRenderMeta("", "", "", "", 2000, "", "", "0.3");
    assert(material.GetAuthoredVersion() == authoredVersion);
    material.InvalidateTextureAssets("unused", false);
    assert(material.GetAuthoredVersion() == authoredVersion);
    material.SetColor("baseColor", glm::vec4(1.0f));
    assert(material.GetAuthoredVersion() > authoredVersion);
}

void VerifyColorVectorShaderTransitionsPreserveAuthoredValues()
{
    InxMaterial material("AuthoredTint", "Unlit");
    const glm::vec4 tint(1.0f, 0.55f, 0.12f, 1.0f);
    material.SetVector4("baseColor", tint);
    ShaderProgramArtifact artifact;
    ShaderProgramPropertyBinding binding;
    binding.name = "baseColor";
    binding.type = "Color";
    binding.defaultValue = "[1,1,1,1]";
    artifact.properties.push_back(binding);

    // Color and Float4 share their numeric representation. Shader semantic
    // changes must not discard an authored tint on first pipeline creation.
    for (const auto *type : {"Color", "Float4", "Color"}) {
        artifact.properties[0].type = type;
        const auto version = material.GetVersion();
        assert(material.SynchronizeShaderPropertyDefaults(artifact));
        assert(material.GetVersion() > version);
        const auto expected = std::string(type) == "Color" ? infernux::MaterialPropertyType::Color
                                                           : infernux::MaterialPropertyType::Float4;
        assert(material.GetProperty("baseColor")->type == expected);
        assert(std::get<glm::vec4>(material.GetProperty("baseColor")->value) == tint);
        assert(material.SerializeDocument()["properties"]["baseColor"]["type"] == static_cast<int>(expected));
        const auto synchronized = material.GetVersion();
        assert(!material.SynchronizeShaderPropertyDefaults(artifact));
        assert(material.GetVersion() == synchronized);
    }
    // A genuinely incompatible shape still takes the shader's typed default.
    material.SetFloat("baseColor", 0.25f);
    assert(material.SynchronizeShaderPropertyDefaults(artifact));
    assert(std::get<glm::vec4>(material.GetProperty("baseColor")->value) == glm::vec4(1.0f));
}

void VerifyTextureSamplerBindingRoundTrip()
{
    InxMaterial material("SamplerBinding", "Lit");
    material.SetTextureGuid("texSampler", "white");
    MaterialTextureSampler sampler;
    sampler.minFilter = MaterialSamplerFilter::Nearest;
    sampler.magFilter = MaterialSamplerFilter::Linear;
    sampler.mipFilter = MaterialSamplerFilter::Nearest;
    sampler.addressU = MaterialSamplerAddress::Clamp;
    sampler.addressV = MaterialSamplerAddress::Mirror;
    sampler.addressW = MaterialSamplerAddress::Repeat;
    material.SetTextureSampler("texSampler", sampler);
    assert(material.GetTextureSampler("texSampler") && *material.GetTextureSampler("texSampler") == sampler);

    const auto document = material.SerializeDocument();
    assert(document.at("textureSamplers").at("texSampler").at("addressV") ==
           static_cast<uint32_t>(MaterialSamplerAddress::Mirror));
    InxMaterial restored;
    assert(restored.DeserializeDocument(document));
    assert(restored.GetTextureSampler("texSampler") && *restored.GetTextureSampler("texSampler") == sampler);
    assert(restored.Clone()->GetTextureSampler("texSampler") &&
           *restored.Clone()->GetTextureSampler("texSampler") == sampler);

    const auto before = restored.SerializeDocument();
    auto invalid = before;
    invalid["textureSamplers"]["texSampler"]["minFilter"] = 99;
    assert(!restored.DeserializeDocument(invalid));
    assert(restored.SerializeDocument() == before);
    invalid = before;
    invalid["textureSamplers"]["missing"] = invalid["textureSamplers"]["texSampler"];
    assert(!restored.DeserializeDocument(invalid));
    assert(restored.SerializeDocument() == before);

    assert(restored.RemoveProperty("texSampler"));
    assert(restored.GetTextureSampler("texSampler") == nullptr);
    assert(!restored.SerializeDocument().contains("textureSamplers"));
}

void VerifyReflectedArrayRoundTripAndLengthAuthority()
{
    InxMaterial material("ArrayContract", "Unlit");
    material.SetFloatArray("curve", {0.0f, 0.5f, 1.0f});
    material.SetVector4Array("palette", {{1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});

    const auto document = material.SerializeDocument();
    InxMaterial restored;
    assert(restored.DeserializeDocument(document));
    assert(std::get<std::vector<float>>(restored.GetProperty("curve")->value) ==
           std::vector<float>({0.0f, 0.5f, 1.0f}));
    const auto &palette = std::get<std::vector<glm::vec4>>(restored.GetProperty("palette")->value);
    assert(palette.size() == 2 && palette[1] == glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));

    bool rejected = false;
    try {
        restored.SetFloatArray("curve", {1.0f, 2.0f});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main()
{
    VerifyGizmoIconPreservesAuthoredAlpha();
    VerifyRetiredFieldsAreIgnored();
    VerifyStableReferencesAndClone();
    VerifyMaterialIdentityIsNotSourceProvenance();
    VerifyTransactionalFailure();
    VerifyRenderStateVersioning();
    VerifyShaderReferenceVersioning();
    VerifyPropertyRemoval();
    VerifyTextureSamplerBindingRoundTrip();
    VerifyShaderDefaultsReplacePreviousShaderState();
    VerifyMaterialOverridesSurviveShaderDefaults();
    VerifyBuiltinSixWaySmokeMaterial();
    VerifyBackendNeutralRenderStateSchema();
    VerifySparseMaterialUsesLinkedShaderDefaults();
    VerifyColorVectorShaderTransitionsPreserveAuthoredValues();
    VerifyReflectedArrayRoundTripAndLengthAuthority();
    std::cout << "Material document tests passed\n";
    return 0;
}
