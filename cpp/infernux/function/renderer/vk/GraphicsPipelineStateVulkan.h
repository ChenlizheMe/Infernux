#pragma once

#include <function/renderer/rhi/RhiDescriptors.h>
#include <vulkan/vulkan.h>

namespace infernux::vk
{

inline VkPrimitiveTopology GraphicsTopology(rhi::PrimitiveTopology topology)
{
    constexpr std::array values{VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                                VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                                VK_PRIMITIVE_TOPOLOGY_LINE_STRIP};
    return values[static_cast<size_t>(topology)];
}

inline VkPipelineRasterizationStateCreateInfo GraphicsRasterState(const rhi::RasterState &state)
{
    VkPipelineRasterizationStateCreateInfo result{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    result.polygonMode = static_cast<VkPolygonMode>(state.polygonMode);
    result.cullMode = static_cast<VkCullModeFlags>(state.cullMode);
    result.frontFace =
        state.frontFace == rhi::FrontFace::Clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    result.lineWidth = state.lineWidth;
    result.depthBiasEnable = state.depthBias != 0.0f || state.depthBiasSlope != 0.0f;
    result.depthBiasConstantFactor = state.depthBias;
    result.depthBiasSlopeFactor = state.depthBiasSlope;
    result.depthBiasClamp = state.depthBiasClamp;
    return result;
}

inline VkPipelineDepthStencilStateCreateInfo GraphicsDepthStencilState(const rhi::DepthState &depth,
                                                                       const rhi::StencilState &stencil)
{
    VkPipelineDepthStencilStateCreateInfo result{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    result.depthTestEnable = depth.testEnabled;
    result.depthWriteEnable = depth.writeEnabled;
    result.depthCompareOp = static_cast<VkCompareOp>(depth.compare);
    const auto face = [](const rhi::StencilFaceState &value) -> VkStencilOpState {
        return {static_cast<VkStencilOp>(value.fail),
                static_cast<VkStencilOp>(value.pass),
                static_cast<VkStencilOp>(value.depthFail),
                static_cast<VkCompareOp>(value.compare),
                value.readMask,
                value.writeMask,
                value.reference};
    };
    result.stencilTestEnable = stencil.enabled;
    result.front = face(stencil.front);
    result.back = face(stencil.back);
    return result;
}

inline VkPipelineColorBlendAttachmentState GraphicsColorState(const rhi::ColorTargetState &state)
{
    VkPipelineColorBlendAttachmentState result{};
    result.blendEnable = state.blendEnabled;
    result.srcColorBlendFactor = static_cast<VkBlendFactor>(state.colorBlend.source);
    result.dstColorBlendFactor = static_cast<VkBlendFactor>(state.colorBlend.destination);
    result.colorBlendOp = static_cast<VkBlendOp>(state.colorBlend.operation);
    result.srcAlphaBlendFactor = static_cast<VkBlendFactor>(state.alphaBlend.source);
    result.dstAlphaBlendFactor = static_cast<VkBlendFactor>(state.alphaBlend.destination);
    result.alphaBlendOp = static_cast<VkBlendOp>(state.alphaBlend.operation);
    result.colorWriteMask = state.writeMask;
    return result;
}

} // namespace infernux::vk
