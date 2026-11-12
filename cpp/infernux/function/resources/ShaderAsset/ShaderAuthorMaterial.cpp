#include "ShaderAuthorMaterial.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace infernux
{

ShaderProgramArtifact CreateShaderAuthorMaterial(std::string guid,
                                                 const std::shared_ptr<const ShaderCompilation> &compilation,
                                                 const std::string &schema)
{
    const auto source = nlohmann::json::parse(schema);
    const auto target = std::find_if(compilation->targets.begin(), compilation->targets.end(),
                                     [](const auto &value) { return value.target == ShaderCodeTarget::SpirV; });
    if (target == compilation->targets.end())
        throw std::logic_error("Mesh author has no native shader target");
    const auto entry = [&](rhi::ShaderStage stage) -> const ShaderEntryPointCode & {
        const auto found = std::find_if(target->entries.begin(), target->entries.end(),
                                        [stage](const auto &value) { return value.stage == stage; });
        if (found == target->entries.end())
            throw std::logic_error("Mesh author has an incomplete graphics program");
        return *found;
    };
    ShaderProgramArtifact result;
    result.key.stages = {"Standard", std::move(guid)};
    result.domain = ShaderProgramDomain::Mesh;
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
        property.stages = ShaderProgramStageMask::Fragment;
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
        result.properties.push_back(std::move(property));
    }
    const auto &vertex = entry(rhi::ShaderStage::Vertex);
    const auto &fragment = entry(rhi::ShaderStage::Fragment);
    const auto publication = std::shared_ptr<const ShaderTargetCode>(compilation, &*target);
    for (const auto pass : {ShaderCompileTarget::Forward, ShaderCompileTarget::ForwardPlus}) {
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
    }
    result.key.revision = ComputeShaderProgramArtifactRevision(result);
    return result;
}

} // namespace infernux
