#include "RayLitMaterials.h"
#include "RayMaterial.h"

#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/shader/MaterialUniformLayout.h>
#include <function/renderer/shader/ShaderProgramCode.h>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace infernux::raytracing
{
namespace
{

const ShaderResourceLayout &Buffer(const std::vector<ShaderResourceLayout> &resources, std::string_view name)
{
    const auto it =
        std::find_if(resources.begin(), resources.end(), [name](const auto &item) { return item.name == name; });
    if (it == resources.end() || it->binding.type != rhi::BindingType::StorageBuffer || !it->binding.readOnlyStorage ||
        it->value.kind != ShaderValueKind::Struct || it->value.byteSize > it->elementStride)
        throw std::invalid_argument("Expected a reflected ray material storage record: " + std::string(name));
    return *it;
}

const ShaderValueLayout &Field(const ShaderValueLayout &record, std::string_view name)
{
    const auto it = std::find_if(record.fields.begin(), record.fields.end(),
                                 [name](const auto &field) { return field.name == name; });
    if (it == record.fields.end())
        throw std::invalid_argument("Ray material storage record has no field: " + std::string(name));
    return *it;
}

size_t UIntOffset(const ShaderValueLayout &field)
{
    if (field.kind != ShaderValueKind::Scalar || field.scalar != ShaderScalarKind::UInt32)
        throw std::invalid_argument("Ray material table index needs a reflected uint32 field: " + field.name);
    return field.offset;
}

std::vector<std::byte> Rows(uint64_t stride, uint64_t count)
{
    if (!stride || count > (std::numeric_limits<size_t>::max)() / stride)
        throw std::length_error("Ray material table exceeds the native address range");
    return std::vector<std::byte>(static_cast<size_t>(stride * count));
}

} // namespace

RayMaterialParticipation ClassifyRayMaterial(const InxMaterial &material) noexcept
{
    if (material.GetRenderState().blendEnable)
        return RayMaterialParticipation::Excluded;
    const auto &shader = material.GetFragShaderName();
    if (shader == "infernux/lit")
        return RayMaterialParticipation::Lit;
    if (shader == "infernux/unlit")
        return RayMaterialParticipation::Unlit;
    const auto &program = material.GetSelectedShaderArtifact();
    return program && program->hasRaySurface ? RayMaterialParticipation::Lit : RayMaterialParticipation::Unknown;
}

struct RayLitMaterialLayout::State
{
    struct Payload
    {
        Payload(const ShaderResourceLayout &record, bool authored, uint32_t index,
                const std::vector<std::string> &textureNames = {})
            : stride(record.elementStride), entry(index)
        {
            std::vector<ShaderValueLayout> fields;
            if (authored) {
                for (const auto &slot : record.value.fields) {
                    auto value = Field(slot, "value");
                    value.name = slot.name;
                    value.offset += slot.offset;
                    if (std::find(textureNames.begin(), textureNames.end(), slot.name) != textureNames.end())
                        textures.emplace_back(slot.name, UIntOffset(value));
                    else
                        fields.push_back(std::move(value));
                }
            } else {
                const auto &values = Field(record.value, "values");
                for (const auto &slot : values.fields) {
                    auto field = Field(slot, "value");
                    field.name = slot.name;
                    field.offset += values.offset + slot.offset;
                    fields.push_back(std::move(field));
                }
                for (const auto &field : record.value.fields)
                    if (field.name != "values" && field.name != "participation")
                        textures.emplace_back(field.name, UIntOffset(field));
            }
            uniforms = MaterialUniformLayout(fields);
        }
        uint64_t stride;
        uint32_t entry;
        MaterialUniformLayout uniforms;
        std::vector<std::pair<std::string, size_t>> textures;
    };

    explicit State(const ShaderTargetCode &program)
        : builtin(Buffer(program.resources, "inx_ray_schema_builtin"), false, 0)
    {
        const auto manifest =
            nlohmann::json::parse(program.reflection).value("infernux_ray_surfaces", nlohmann::json::array());
        for (const auto &surface : manifest) {
            std::vector<std::string> textureNames;
            for (const auto &parameter : surface.at("parameters"))
                if (parameter.at("kind") == "texture_2d")
                    textureNames.push_back(parameter.at("name").get<std::string>());
            authored.emplace(surface.at("guid").get<std::string>(),
                             Payload(Buffer(program.resources, surface.at("schema_resource").get<std::string>()), true,
                                     surface.at("entry_index").get<uint32_t>(), textureNames));
        }
    }

    Payload builtin;
    std::unordered_map<std::string, Payload> authored;
};

RayLitMaterialLayout::RayLitMaterialLayout(const ShaderTargetCode &program) : m_state(std::make_unique<State>(program))
{
}

RayLitMaterialLayout::~RayLitMaterialLayout() = default;

RayLitTextureReference ResolveRayLitTextureReference(const InxMaterial &material,
                                                     const RendererParameterBlock *parameters,
                                                     std::string_view property)
{
    const std::string name(property);
    if (material.GetRenderTextureGeneration(name))
        throw std::invalid_argument("Ray Lit cannot capture a live RenderTexture: " + name);
    const auto *value = material.GetProperty(name);
    if (parameters) {
        const auto override = parameters->properties.find(name);
        if (override != parameters->properties.end())
            value = &override->second;
    }
    const auto *guid = value ? std::get_if<std::string>(&value->value) : nullptr;
    const auto key = guid && !guid->empty() ? std::string_view(*guid) : material.GetTextureDefault(name);
    if (key == "white" || key == "White")
        return RayLitSolidTexture::White;
    if (key == "black" || key == "Black")
        return RayLitSolidTexture::Black;
    if (key == "normal" || key == "Normal")
        return RayLitSolidTexture::FlatNormal;
    if (guid && !guid->empty())
        return *guid;
    throw std::invalid_argument("Ray Lit texture parameter has no default: " + name);
}

std::optional<RayLitMaterialInput> RayLitMaterialLayout::Capture(const DrawCallResult &world,
                                                                 const ResolveRayLitTexture &resolve) const
{
    const auto &layout = *m_state;
    RayLitMaterialInput input;
    uint32_t maxSurface = 0;
    for (const auto &draw : world.drawCalls)
        if (draw.indexCount)
            maxSurface = std::max(maxSurface, draw.surfaceId);
    // One compact header per SurfaceID, followed by target-reflected payloads.
    // Header: family, byte address, actual compiled author entry, alpha threshold bits.
    constexpr uint64_t headerStride = 16;
    input.materials = Rows(headerStride, uint64_t{maxSurface} + 1);
    for (const auto &draw : world.drawCalls) {
        if (!draw.indexCount || !draw.surfaceId || !draw.material)
            continue;
        const auto &material = *draw.material;
        auto participation = ClassifyRayMaterial(material);
        const auto custom = layout.authored.find(material.GetShaderStages().fragmentKey);
        const bool authored = custom != layout.authored.end();
        const auto &program = material.GetSelectedShaderArtifact();
        if (program && program->hasRaySurface && !authored)
            participation = RayMaterialParticipation::Unknown;
        const auto *payload = authored ? &custom->second : &layout.builtin;
        const float alpha =
            material.GetRenderState().alphaClipEnabled ? material.GetRenderState().alphaClipThreshold : 0.0f;
        std::array<uint32_t, 4> header{
            static_cast<uint32_t>(participation), 0,
            authored ? payload->entry : uint32_t(participation == RayMaterialParticipation::Unlit), 0};
        std::memcpy(&header[3], &alpha, sizeof(alpha));
        if (participation == RayMaterialParticipation::Lit || participation == RayMaterialParticipation::Unlit) {
            const auto offset = (input.materials.size() + 15u) & ~size_t{15u};
            header[1] = static_cast<uint32_t>(offset);
            input.materials.resize(offset + payload->stride);
            auto *values = input.materials.data() + offset;
            payload->uniforms.Write(values, material);
            if (draw.parameterBlock)
                payload->uniforms.Apply(values, material, *draw.parameterBlock);
            payload->uniforms.WriteValue(values, "_alpha_clip_threshold", &alpha);
            for (const auto &[name, field] : payload->textures) {
                if (participation == RayMaterialParticipation::Unlit && !material.HasProperty(name))
                    continue;
                auto texture = resolve(material, draw.parameterBlock.get(), name);
                if (!texture)
                    return std::nullopt;
                if (auto *source = std::get_if<RayTexturePageSource>(&*texture)) {
                    input.textureOffsets.push_back(offset + field);
                    input.sources.push_back(std::move(*source));
                } else {
                    const auto value = static_cast<uint32_t>(std::get<RayLitSolidTexture>(*texture));
                    std::memcpy(values + field, &value, sizeof(value));
                }
            }
        }
        std::memcpy(input.materials.data() + draw.surfaceId * headerStride, header.data(), headerStride);
    }
    return input;
}

void RayLitMaterialInput::ApplyPageSlots(const std::vector<RayTexturePageSlot> &slots)
{
    // The low three bits select a page or a uniform texel. The rest address its array layer.
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto &slot = slots[i];
        const uint32_t value = (slot.layer << 3u) | static_cast<uint32_t>(slot.pageClass);
        std::memcpy(materials.data() + textureOffsets[i], &value, sizeof(value));
    }
}

} // namespace infernux::raytracing
