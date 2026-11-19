#include "ShaderAuthorMaterial.h"
#include "SlangCompiler.h"

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
    if (compilation->targets.empty())
        throw std::logic_error("Material author has no compiled target");
    const auto &layout = compilation->targets.front();
    const auto entry = [](const ShaderTargetCode &module, rhi::ShaderStage stage,
                          const char *name) -> const ShaderEntryPointCode & {
        const auto found = std::find_if(module.entries.begin(), module.entries.end(),
                                        [&](const auto &value) { return value.stage == stage && value.name == name; });
        if (found == module.entries.end())
            throw std::logic_error(std::string("Material author is missing a graphics entry: ") + name);
        return *found;
    };
    ShaderProgramArtifact result;
    result.compilation = compilation;
    result.key.stages = {std::move(vertexKey), std::move(guid)};
    const bool particle = source.at("domain") == "particle";
    const bool screenUI = source.at("domain") == "screen_ui";
    const bool worldUI = source.at("domain") == "world_ui";
    const bool ui = screenUI || worldUI;
    result.domain = screenUI   ? ShaderProgramDomain::ScreenUI
                    : worldUI  ? ShaderProgramDomain::WorldUI
                    : particle ? ShaderProgramDomain::ParticleSprite
                               : ShaderProgramDomain::Mesh;
    const bool surface = source.value("surface", false);
    result.shadingModel = surface ? source.at("shading_model").get<std::string>() : "";
    if (surface) {
        const auto &deferredModel = source.at("deferred_model");
        if (!deferredModel.is_null())
            result.deferredModel = ShadingModelId(deferredModel.get<std::string>());
        result.receivesShadows = source.at("info").at("receive_shadows").get<bool>();
    }
    result.compatibilitySignature =
        ComputeShaderProgramRevision(schema, source.at("domain").get<std::string>(), ShaderCompileTarget::Forward, 0);
    result.materialLayoutSignature = result.compatibilitySignature;
    const auto uniform = std::find_if(layout.resources.begin(), layout.resources.end(), [](const auto &resource) {
        return resource.set == 0 && resource.binding.binding == 0 &&
               resource.binding.type == rhi::BindingType::UniformBuffer;
    });
    if (uniform != layout.resources.end())
        result.materialBufferSize = static_cast<uint32_t>((uniform->value.byteSize + 15u) & ~uint64_t{15u});
    static const std::unordered_map<std::string, std::string> propertyTypes = {
        {"float", "Float"}, {"int", "Int"},   {"vec2", "Float2"}, {"vec3", "Float3"}, {"vec4", "Float4"},
        {"color", "Color"}, {"mat4", "Mat4"}, {"bool", "Bool"},   {"uint", "UInt"},   {"mat3", "Mat3"},
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
        } else {
            if (uniform == layout.resources.end())
                throw std::logic_error("Mesh author parameters have no compiler layout");
            const auto field = std::find_if(uniform->value.fields.begin(), uniform->value.fields.end(),
                                            [&](const auto &value) { return value.name == property.name; });
            if (field == uniform->value.fields.end())
                throw std::logic_error("Mesh author field is absent from the compiler layout: " + property.name);
            property.type = propertyTypes.at(kind);
            auto value = parameter.at("default");
            if (property.type == "Mat3" || property.type == "Mat4") {
                nlohmann::json columns = nlohmann::json::array();
                const size_t dimension = property.type == "Mat3" ? 3 : 4;
                for (size_t column = 0; column < dimension; ++column)
                    for (size_t row = 0; row < dimension; ++row)
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
    const auto addVariant = [&](ShaderCompileTarget pass, const char *vertexName, const char *fragmentName) {
        ShaderProgramArtifact::PassVariant variant;
        variant.target = pass;
        variant.compatibilitySignature = result.compatibilitySignature;
        variant.outputs.reserve(compilation->targets.size());
        for (const auto &module : compilation->targets) {
            const auto *vertex = vertexName ? &entry(module, rhi::ShaderStage::Vertex, vertexName) : nullptr;
            variant.outputs.push_back({std::shared_ptr<const ShaderTargetCode>(compilation, &module), vertex,
                                       &entry(module, rhi::ShaderStage::Fragment, fragmentName)});
        }
        result.variants.push_back(std::move(variant));
    };
    const auto addPartitions = [&](const char *vertexName) {
        const bool compiled = std::any_of(layout.entries.begin(), layout.entries.end(),
                                          [](const auto &code) { return code.name == "fragment_transparent_behind"; });
        if (compiled) {
            addVariant(ShaderCompileTarget::TransparentBehind, vertexName, "fragment_transparent_behind");
            addVariant(ShaderCompileTarget::TransparentFront, vertexName, "fragment_transparent_front");
        }
    };
    if (ui) {
        addVariant(ShaderCompileTarget::Forward, "vertex_main", "fragment_main");
        result.key.revision = ComputeShaderProgramArtifactRevision(result);
        return result;
    }
    if (particle) {
        addVariant(ShaderCompileTarget::Forward, nullptr, "fragment_main");
        addVariant(ShaderCompileTarget::ForwardPlus, nullptr, "fragment_forward_plus");
        addPartitions(nullptr);
        addVariant(ShaderCompileTarget::Depth, nullptr, "fragment_depth");
        addVariant(ShaderCompileTarget::Picking, nullptr, "fragment_picking");
        addVariant(ShaderCompileTarget::Motion, nullptr, "fragment_motion");
        if (source.at("info").at("cast_shadows").get<bool>()) {
            // A caster cannot sample the atlas that this pass is writing.
            // Check the compiled entry once, before publishing any device work.
            for (const auto &code : compilation->targets) {
                const auto coverage = std::find_if(code.entries.begin(), code.entries.end(),
                                                   [](const auto &value) { return value.name == "fragment_depth"; });
                for (const auto &binding : coverage->bindings)
                    if (binding[0] == 1 && binding[1] != 5)
                        throw ShaderCompilationError(
                            {{"author.shadow_resources", "error", result.key.stages.fragmentKey, 0, 0, 0, 0,
                              "Particle shadow coverage depends on scene lighting. Keep alpha, clip and depth "
                              "independent of lighting, or set cast_shadows = False.",
                              "shadow", code.target == ShaderCodeTarget::SpirV ? "spirv" : "wgsl"}});
            }
            addVariant(ShaderCompileTarget::Shadow, nullptr, "fragment_depth");
        }
        result.key.revision = ComputeShaderProgramArtifactRevision(result);
        return result;
    }
    addVariant(ShaderCompileTarget::Forward, "vertex_main", "fragment_main");
    addVariant(ShaderCompileTarget::ForwardPlus, "vertex_main", surface ? "fragment_forward_plus" : "fragment_main");
    addPartitions("vertex_main");
    if (surface) {
        if (source.at("thin_sheet").get<bool>())
            addVariant(ShaderCompileTarget::ThinSurface, "vertex_motion", "fragment_thin_surface");
        if (source.at("info").at("cast_shadows").get<bool>())
            addVariant(ShaderCompileTarget::Shadow, "vertex_shadow", "fragment_shadow");
        if (!source.at("deferred_model").is_null()) {
            addVariant(ShaderCompileTarget::GBuffer, "vertex_main", "fragment_gbuffer");
            addVariant(ShaderCompileTarget::RaySurface, "vertex_motion", "fragment_ray_surface");
        }
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
