#include "ShaderAuthorMaterial.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{

ShaderProgramArtifact CreateShaderAuthorMaterial(std::string guid,
                                                 const std::shared_ptr<const ShaderCompilation> &compilation,
                                                 const std::string &schema, std::string vertexKey)
{
    const auto source = nlohmann::json::parse(schema);
    const auto target = std::find_if(compilation->targets.begin(), compilation->targets.end(),
                                     [](const auto &value) { return value.target == ShaderCodeTarget::SpirV; });
    if (target == compilation->targets.end())
        throw std::logic_error("Mesh author has no native shader target");
    const auto entry = [&](const char *name) -> const ShaderEntryPointCode & {
        const auto found = std::find_if(target->entries.begin(), target->entries.end(),
                                        [name](const auto &value) { return value.name == name; });
        if (found == target->entries.end())
            throw std::logic_error("Mesh author has an incomplete graphics program");
        return *found;
    };
    ShaderProgramArtifact result;
    result.key.stages = {std::move(vertexKey), std::move(guid)};
    result.domain = ShaderProgramDomain::Mesh;
    const bool surface = source.value("surface", false);
    result.shadingModel = surface ? source.at("shading_model").get<std::string>() : "";
    result.compatibilitySignature =
        ComputeShaderProgramRevision(schema, "InxMeshInput", ShaderCompileTarget::Forward, 0);
    result.materialLayoutSignature = result.compatibilitySignature;
    const auto uniform = std::find_if(target->resources.begin(), target->resources.end(), [](const auto &resource) {
        return resource.set == 0 && resource.binding.binding == 0 &&
               resource.binding.type == rhi::BindingType::UniformBuffer;
    });
    if (uniform != target->resources.end())
        result.materialBufferSize = static_cast<uint32_t>((uniform->value.byteSize + 15u) & ~uint64_t{15u});
    static const std::unordered_map<std::string, std::string> propertyTypes = {
        {"float", "Float"}, {"int", "Int"},     {"vec2", "Float2"}, {"vec3", "Float3"},
        {"vec4", "Float4"}, {"color", "Color"}, {"mat4", "Mat4"},
    };
    uint32_t textureSlot = 0;
    for (const auto &parameter : source.at("parameters")) {
        ShaderProgramPropertyBinding property;
        property.name = parameter.at("name");
        for (const auto &stage : parameter.at("stages"))
            property.stages = property.stages |
                              (stage == "vertex" ? ShaderProgramStageMask::Vertex : ShaderProgramStageMask::Fragment);
        const auto kind = parameter.at("kind").get<std::string>();
        if (kind == "texture_2d") {
            property.type = "Texture2D";
            property.defaultValue = "\"\"";
            property.textureDefault = parameter.at("default");
            property.textureSlot = textureSlot++;
            property.resourcePath = "inx_" + property.name;
        } else {
            if (uniform == target->resources.end())
                throw std::logic_error("Mesh author parameters have no compiler layout");
            const auto field = std::find_if(uniform->value.fields.begin(), uniform->value.fields.end(),
                                            [&](const auto &value) { return value.name == property.name; });
            if (field == uniform->value.fields.end())
                throw std::logic_error("Mesh author field is absent from the compiler layout: " + property.name);
            property.type = propertyTypes.at(kind);
            auto value = parameter.at("default");
            if (property.type == "Mat4") {
                nlohmann::json columns = nlohmann::json::array();
                for (size_t column = 0; column < 4; ++column)
                    for (size_t row = 0; row < 4; ++row)
                        columns.push_back(value.at(row).at(column));
                value = std::move(columns);
            }
            property.defaultValue = value.dump();
            property.hdr = parameter.at("hdr");
            if (!parameter.at("range").is_null())
                property.range = parameter.at("range").get<std::array<double, 2>>();
            property.bufferOffset = static_cast<uint32_t>(field->offset);
            property.byteSize = static_cast<uint32_t>(field->byteSize);
            // Offsets already satisfy the compiler's target alignment.
            property.byteAlignment = 1;
        }
        if (property.name == "_alpha_clip_threshold")
            result.alphaClipThresholdOffset = property.bufferOffset;
        result.properties.push_back(std::move(property));
    }
    const auto publication = std::shared_ptr<const ShaderTargetCode>(compilation, &*target);
    const auto addVariant = [&](ShaderCompileTarget pass, const char *vertexName, const char *fragmentName) {
        const auto &vertex = entry(vertexName);
        const auto &fragment = entry(fragmentName);
        ShaderProgramArtifact::PassVariant variant;
        variant.target = pass;
        variant.compatibilitySignature = result.compatibilitySignature;
        variant.vertexSpirv = vertex.code;
        variant.fragmentSpirv = fragment.code;
        variant.vertexEntryPoint = vertex.name;
        variant.fragmentEntryPoint = fragment.name;
        variant.vertexTarget = publication;
        variant.fragmentTarget = publication;
        result.variants.push_back(std::move(variant));
    };
    addVariant(ShaderCompileTarget::Forward, "vertex_main", "fragment_main");
    addVariant(ShaderCompileTarget::ForwardPlus, "vertex_main", surface ? "fragment_forward_plus" : "fragment_main");
    if (surface) {
        if (source.at("info").at("cast_shadows").get<bool>())
            addVariant(ShaderCompileTarget::Shadow, "vertex_shadow", "fragment_depth");
        if (!source.at("deferred_model").is_null())
            addVariant(ShaderCompileTarget::GBuffer, "vertex_main", "fragment_gbuffer");
        addVariant(ShaderCompileTarget::Depth, "vertex_main", "fragment_depth");
        addVariant(ShaderCompileTarget::Picking, "vertex_main", "fragment_picking");
        addVariant(ShaderCompileTarget::Motion, "vertex_motion", "fragment_motion");
        addVariant(ShaderCompileTarget::Normal, "vertex_main", "fragment_normal");
        addVariant(ShaderCompileTarget::BaseColor, "vertex_main", "fragment_base_color");
    }
    result.key.revision = ComputeShaderProgramArtifactRevision(result);
    return result;
}

} // namespace infernux
