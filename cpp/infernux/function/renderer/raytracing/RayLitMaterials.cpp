#include "RayLitMaterials.h"

#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/shader/MaterialUniformLayout.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace infernux::raytracing
{
namespace
{

// Matches the material participation values in the Slang ray evaluator.
enum class RayMaterialParticipation : uint32_t
{
    Unknown = 0,
    Lit = 1,
    Excluded = 2,
    Unlit = 3,
};

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

struct RayLitMaterialLayout::State
{
    explicit State(const ShaderResourceLayout &materials)
        : materialStride(materials.elementStride), values(Field(materials.value, "values")), uniforms(values.fields),
          participation(UIntOffset(Field(materials.value, "participation")))
    {
        if (values.kind != ShaderValueKind::Struct ||
            Field(values, "_alpha_clip_threshold").scalar != ShaderScalarKind::Float32)
            throw std::invalid_argument("Ray Lit needs the reflected material values block and alpha clip threshold");
        for (const auto &field : materials.value.fields)
            if (field.name != "values" && field.name != "participation")
                textureFields.emplace_back(field.name, UIntOffset(field));
    }

    const uint64_t materialStride;
    const ShaderValueLayout values;
    const MaterialUniformLayout uniforms;
    const size_t participation;
    std::vector<std::pair<std::string, size_t>> textureFields;
};

RayLitMaterialLayout::RayLitMaterialLayout(const std::vector<ShaderResourceLayout> &resources)
    : m_state(std::make_unique<State>(Buffer(resources, "ray_lit_materials")))
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
    input.materials = Rows(layout.materialStride, uint64_t{maxSurface} + 1);
    for (const auto &draw : world.drawCalls) {
        if (!draw.indexCount || !draw.surfaceId || !draw.material)
            continue;
        auto *row = input.materials.data() + draw.surfaceId * layout.materialStride;
        const auto &material = *draw.material;
        const auto &renderState = material.GetRenderState();
        auto participation = RayMaterialParticipation::Unknown;
        if (renderState.blendEnable)
            participation = RayMaterialParticipation::Excluded;
        else if (material.GetFragShaderName() == "infernux/lit")
            participation = RayMaterialParticipation::Lit;
        else if (material.GetFragShaderName() == "infernux/unlit")
            participation = RayMaterialParticipation::Unlit;
        const auto kind = static_cast<uint32_t>(participation);
        std::memcpy(row + layout.participation, &kind, sizeof(kind));
        if (participation != RayMaterialParticipation::Lit && participation != RayMaterialParticipation::Unlit)
            continue;
        auto *values = row + layout.values.offset;
        layout.uniforms.Write(values, material);
        if (draw.parameterBlock)
            layout.uniforms.Apply(values, material, *draw.parameterBlock);
        const float alpha = renderState.alphaClipEnabled ? renderState.alphaClipThreshold : 0.0f;
        layout.uniforms.WriteValue(values, "_alpha_clip_threshold", &alpha);

        for (const auto &[name, offset] : layout.textureFields) {
            if (participation == RayMaterialParticipation::Unlit && !material.HasProperty(name))
                continue;
            auto texture = resolve(material, draw.parameterBlock.get(), name);
            if (!texture)
                return std::nullopt;
            if (auto *source = std::get_if<RayTexturePageSource>(&*texture)) {
                input.textureOffsets.push_back(static_cast<size_t>(row - input.materials.data()) + offset);
                input.sources.push_back(std::move(*source));
            } else {
                const auto value = static_cast<uint32_t>(std::get<RayLitSolidTexture>(*texture));
                std::memcpy(row + offset, &value, sizeof(value));
            }
        }
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
