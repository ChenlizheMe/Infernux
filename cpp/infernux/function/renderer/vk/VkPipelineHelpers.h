#pragma once

#include <array>
#include <vulkan/vulkan.h>

namespace infernux
{
namespace vkrender
{

/// Create a VkPipelineShaderStageCreateInfo for a single stage.
/// Reusable across all pipeline creators that use "main" as entry point.
inline VkPipelineShaderStageCreateInfo MakeShaderStageInfo(VkShaderStageFlagBits stage, VkShaderModule module,
                                                           const char *entryPoint = "main")
{
    VkPipelineShaderStageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage = stage;
    info.module = module;
    info.pName = entryPoint;
    return info;
}

/// Build a vert+frag shader stage pair.
inline std::array<VkPipelineShaderStageCreateInfo, 2> MakeVertFragStages(VkShaderModule vert, VkShaderModule frag)
{
    return {MakeShaderStageInfo(VK_SHADER_STAGE_VERTEX_BIT, vert),
            MakeShaderStageInfo(VK_SHADER_STAGE_FRAGMENT_BIT, frag)};
}

/// Dynamic viewport + scissor state (used by virtually every pipeline).
struct DynamicViewportScissorState
{
    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    VkPipelineViewportStateCreateInfo viewportState{};

    DynamicViewportScissorState()
    {
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
    }
};

/// Multisampling state with a configurable sample count.
inline VkPipelineMultisampleStateCreateInfo MakeMultisampleState(
    VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT)
{
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = sampleCount;
    ms.sampleShadingEnable = VK_FALSE;
    return ms;
}

/// Input assembly for triangle lists (the most common topology).
inline VkPipelineInputAssemblyStateCreateInfo MakeTriangleListInputAssembly()
{
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    ia.primitiveRestartEnable = VK_FALSE;
    return ia;
}

} // namespace vkrender
} // namespace infernux
