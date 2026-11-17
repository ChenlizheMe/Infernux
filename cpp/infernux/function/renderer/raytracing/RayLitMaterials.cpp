#include "RayLitMaterials.h"

#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/shader/MaterialUniformLayout.h>

#include <algorithm>
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

ShaderValueCopy UIntField(const ShaderValueLayout &field)
{
    if (field.kind != ShaderValueKind::Scalar || field.scalar != ShaderScalarKind::UInt32)
        throw std::invalid_argument("Ray material table index needs a reflected uint32 field: " + field.name);
    return ShaderValueCopy(field);
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
    explicit State(const ShaderResourceLayout &materials, const ShaderResourceLayout &descriptors)
        : materialStride(materials.elementStride), descriptorStride(descriptors.elementStride),
          values(Field(materials.value, "values")), uniforms(values.fields),
          participation(UIntField(Field(materials.value, "participation"))),
          pageClass(UIntField(Field(descriptors.value, "page_class"))),
          layer(UIntField(Field(descriptors.value, "layer")))
    {
        if (values.kind != ShaderValueKind::Struct ||
            Field(values, "_alpha_clip_threshold").scalar != ShaderScalarKind::Float32)
            throw std::invalid_argument("Ray Lit needs the reflected material values block and alpha clip threshold");
        for (const auto &field : materials.value.fields)
            if (field.name != "values" && field.name != "participation")
                textureIndices.emplace_back(field.name, UIntField(field));
    }

    const uint64_t materialStride;
    const uint64_t descriptorStride;
    const ShaderValueLayout values;
    const MaterialUniformLayout uniforms;
    const ShaderValueCopy participation;
    const ShaderValueCopy pageClass;
    const ShaderValueCopy layer;
    std::vector<std::pair<std::string, ShaderValueCopy>> textureIndices;
};

RayLitMaterialLayout::RayLitMaterialLayout(const std::vector<ShaderResourceLayout> &resources)
    : m_state(
          std::make_unique<State>(Buffer(resources, "ray_lit_materials"), Buffer(resources, "ray_texture_descriptors")))
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
        const auto kind = static_cast<uint32_t>(participation);
        layout.participation.Write(row, &kind);
        if (participation != RayMaterialParticipation::Lit)
            continue;
        auto *values = row + layout.values.offset;
        layout.uniforms.Write(values, material);
        if (draw.parameterBlock)
            layout.uniforms.Apply(values, material, *draw.parameterBlock);
        const float alpha = renderState.alphaClipEnabled ? renderState.alphaClipThreshold : 0.0f;
        layout.uniforms.WriteValue(values, "_alpha_clip_threshold", &alpha);

        for (const auto &[name, copy] : layout.textureIndices) {
            auto texture = resolve(material, draw.parameterBlock.get(), name);
            if (!texture)
                return std::nullopt;
            if (input.descriptors.size() >= UINT32_MAX)
                throw std::length_error("Ray Lit texture references exceed the shader index range");
            const auto descriptor = static_cast<uint32_t>(input.descriptors.size());
            copy.Write(row, &descriptor);
            if (auto *source = std::get_if<RayTexturePageSource>(&*texture)) {
                input.descriptors.emplace_back(static_cast<uint32_t>(input.sources.size()));
                input.sources.push_back(std::move(*source));
            } else {
                input.descriptors.emplace_back(std::get<RayLitSolidTexture>(*texture));
            }
        }
    }
    return input;
}

std::vector<std::byte> RayLitMaterialLayout::PackDescriptors(const RayLitMaterialInput &input,
                                                             const std::vector<RayTexturePageSlot> &slots) const
{
    if (slots.size() != input.sources.size())
        throw std::invalid_argument("Ray Lit texture page slots do not match the captured sources");
    const auto &layout = *m_state;
    auto bytes = Rows(layout.descriptorStride, input.descriptors.size());
    auto *record = bytes.data();
    for (const auto &descriptor : input.descriptors) {
        uint32_t pageClass;
        uint32_t layer = 0;
        if (const auto *source = std::get_if<uint32_t>(&descriptor)) {
            const auto &slot = slots[*source];
            pageClass = static_cast<uint32_t>(slot.pageClass);
            layer = slot.layer;
        } else {
            pageClass = static_cast<uint32_t>(std::get<RayLitSolidTexture>(descriptor));
        }
        layout.pageClass.Write(record, &pageClass);
        layout.layer.Write(record, &layer);
        record += layout.descriptorStride;
    }
    return bytes;
}

} // namespace infernux::raytracing
