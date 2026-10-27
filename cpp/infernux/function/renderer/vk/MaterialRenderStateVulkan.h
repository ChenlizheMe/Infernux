#pragma once

#include <function/resources/InxMaterial/InxMaterial.h>
#include <vulkan/vulkan.h>

namespace infernux::vk
{

[[nodiscard]] constexpr VkCullModeFlags ToVkCullMode(MaterialCullMode value) noexcept
{
    return static_cast<VkCullModeFlags>(value);
}

[[nodiscard]] constexpr VkFrontFace ToVkFrontFace(MaterialFrontFace value) noexcept
{
    return static_cast<VkFrontFace>(value);
}

[[nodiscard]] constexpr VkPolygonMode ToVkPolygonMode(MaterialPolygonMode value) noexcept
{
    return static_cast<VkPolygonMode>(value);
}

[[nodiscard]] constexpr VkPrimitiveTopology ToVkPrimitiveTopology(MaterialPrimitiveTopology value) noexcept
{
    return static_cast<VkPrimitiveTopology>(value);
}

[[nodiscard]] constexpr VkCompareOp ToVkCompareOp(MaterialCompareOp value) noexcept
{
    return static_cast<VkCompareOp>(value);
}

// Material comparisons retain Unity-style near/far semantics. Stencil
// comparisons above remain literal numeric comparisons.
[[nodiscard]] constexpr VkCompareOp ToVkDepthCompareOp(MaterialCompareOp value) noexcept
{
    switch (value) {
    case MaterialCompareOp::Less:
        return VK_COMPARE_OP_GREATER;
    case MaterialCompareOp::LessOrEqual:
        return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case MaterialCompareOp::Greater:
        return VK_COMPARE_OP_LESS;
    case MaterialCompareOp::GreaterOrEqual:
        return VK_COMPARE_OP_LESS_OR_EQUAL;
    default:
        return ToVkCompareOp(value);
    }
}

[[nodiscard]] constexpr VkStencilOp ToVkStencilOp(MaterialStencilOp value) noexcept
{
    return static_cast<VkStencilOp>(value);
}

[[nodiscard]] constexpr VkStencilOpState ToVkStencilOpState(const MaterialStencilOpState &value) noexcept
{
    return {ToVkStencilOp(value.failOp),
            ToVkStencilOp(value.passOp),
            ToVkStencilOp(value.depthFailOp),
            ToVkCompareOp(value.compareOp),
            value.compareMask,
            value.writeMask,
            value.reference};
}

[[nodiscard]] constexpr VkBlendFactor ToVkBlendFactor(MaterialBlendFactor value) noexcept
{
    return static_cast<VkBlendFactor>(value);
}

[[nodiscard]] constexpr VkBlendOp ToVkBlendOp(MaterialBlendOp value) noexcept
{
    return static_cast<VkBlendOp>(value);
}

} // namespace infernux::vk
