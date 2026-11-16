#pragma once

#include <cstddef>
#include <cstdint>

namespace infernux
{

// Material documents are shared by Vulkan, WebGPU, and future renderer
// backends. Keep their persisted values independent from backend headers while
// retaining the numeric schema already used by .mat files and Python.
enum class MaterialCullMode : uint32_t
{
    None = 0,
    Front = 1,
    Back = 2,
    FrontAndBack = 3,
};

enum class MaterialFrontFace : uint32_t
{
    CounterClockwise = 0,
    Clockwise = 1,
};

enum class MaterialPolygonMode : uint32_t
{
    Fill = 0,
    Line = 1,
    Point = 2,
};

enum class MaterialPrimitiveTopology : uint32_t
{
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
};

enum class MaterialCompareOp : uint32_t
{
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    Always = 7,
};

enum class MaterialStencilOp : uint32_t
{
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementAndClamp = 3,
    DecrementAndClamp = 4,
    Invert = 5,
    IncrementAndWrap = 6,
    DecrementAndWrap = 7,
};

struct MaterialStencilOpState
{
    MaterialStencilOp failOp = MaterialStencilOp::Keep;
    MaterialStencilOp passOp = MaterialStencilOp::Keep;
    MaterialStencilOp depthFailOp = MaterialStencilOp::Keep;
    MaterialCompareOp compareOp = MaterialCompareOp::Never;
    uint32_t compareMask = 0;
    uint32_t writeMask = 0;
    uint32_t reference = 0;

    [[nodiscard]] bool operator==(const MaterialStencilOpState &other) const noexcept
    {
        return failOp == other.failOp && passOp == other.passOp && depthFailOp == other.depthFailOp &&
               compareOp == other.compareOp && compareMask == other.compareMask && writeMask == other.writeMask &&
               reference == other.reference;
    }

    [[nodiscard]] bool operator!=(const MaterialStencilOpState &other) const noexcept
    {
        return !(*this == other);
    }
};

enum class MaterialBlendFactor : uint32_t
{
    Zero = 0,
    One = 1,
    SourceColor = 2,
    OneMinusSourceColor = 3,
    DestinationColor = 4,
    OneMinusDestinationColor = 5,
    SourceAlpha = 6,
    OneMinusSourceAlpha = 7,
    DestinationAlpha = 8,
    OneMinusDestinationAlpha = 9,
    ConstantColor = 10,
    OneMinusConstantColor = 11,
    ConstantAlpha = 12,
    OneMinusConstantAlpha = 13,
    SourceAlphaSaturate = 14,
};

enum class MaterialBlendOp : uint32_t
{
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Minimum = 3,
    Maximum = 4,
};

/**
 * @brief Render state configuration for materials
 *
 * This defines how the GPU should render geometry with this material.
 */
struct RenderState
{
    // Rasterization
    MaterialCullMode cullMode = MaterialCullMode::Back;
    MaterialFrontFace frontFace = MaterialFrontFace::Clockwise;
    MaterialPolygonMode polygonMode = MaterialPolygonMode::Fill;
    float lineWidth = 1.0f;

    // Depth bias (polygon offset) — pushes fragments in depth to avoid z-fighting
    bool depthBiasEnable = false;
    float depthBiasConstantFactor = 0.0f;
    float depthBiasSlopeFactor = 0.0f;
    float depthBiasClamp = 0.0f;

    // Primitive topology (default: triangle list)
    MaterialPrimitiveTopology topology = MaterialPrimitiveTopology::TriangleList;

    // Depth/Stencil
    bool depthTestEnable = true;
    bool depthWriteEnable = true;
    MaterialCompareOp depthCompareOp = MaterialCompareOp::Less;
    bool stencilTestEnable = false;
    MaterialStencilOpState stencilFront{}; // front-face stencil operations
    MaterialStencilOpState stencilBack{};  // back-face stencil operations

    // Blending
    bool blendEnable = false;
    MaterialBlendFactor srcColorBlendFactor = MaterialBlendFactor::SourceAlpha;
    MaterialBlendFactor dstColorBlendFactor = MaterialBlendFactor::OneMinusSourceAlpha;
    MaterialBlendOp colorBlendOp = MaterialBlendOp::Add;
    MaterialBlendFactor srcAlphaBlendFactor = MaterialBlendFactor::Zero;
    MaterialBlendFactor dstAlphaBlendFactor = MaterialBlendFactor::One;
    MaterialBlendOp alphaBlendOp = MaterialBlendOp::Add;

    // Alpha clip (runtime toggle — controls _alpha_clip_threshold material property)
    bool alphaClipEnabled = false;
    float alphaClipThreshold = 0.5f;

    // Render queue (for sorting)
    int32_t renderQueue = 2000; // 2000 = Opaque, 3000 = Transparent

    bool operator==(const RenderState &other) const;
    size_t Hash() const;
};

} // namespace infernux
