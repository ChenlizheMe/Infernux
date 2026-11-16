#pragma once

#include <function/renderer/MeshVertexLayout.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>

namespace infernux::vk
{

[[nodiscard]] inline VkVertexInputBindingDescription GetVertexBindingDescription() noexcept
{
    VkVertexInputBindingDescription description{};
    description.binding = 0;
    description.stride = sizeof(Vertex);
    description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return description;
}

[[nodiscard]] inline std::array<VkVertexInputAttributeDescription, 8> GetVertexAttributeDescriptions() noexcept
{
    std::array<VkVertexInputAttributeDescription, 8> descriptions{};
    for (size_t index = 0; index < MeshVertexAttributes.size(); ++index) {
        const auto &attribute = MeshVertexAttributes[index];
        VkFormat format;
        switch (attribute.format) {
        case rhi::VertexFormat::Float32x2:
            format = VK_FORMAT_R32G32_SFLOAT;
            break;
        case rhi::VertexFormat::Float32x3:
            format = VK_FORMAT_R32G32B32_SFLOAT;
            break;
        case rhi::VertexFormat::Float32x4:
            format = VK_FORMAT_R32G32B32A32_SFLOAT;
            break;
        case rhi::VertexFormat::Uint32x4:
            format = VK_FORMAT_R32G32B32A32_UINT;
            break;
        default:
            format = VK_FORMAT_UNDEFINED;
            break;
        }
        descriptions[index] = {attribute.location, attribute.bufferSlot, format, attribute.offset};
    }
    return descriptions;
}

} // namespace infernux::vk
