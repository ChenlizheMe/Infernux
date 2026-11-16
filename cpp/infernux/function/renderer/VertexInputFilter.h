#pragma once

#include "InxRenderStruct.h"
#include "shader/ShaderReflection.h"
#include "vk/VertexInputVulkan.h"

#include <vector>

#include <vulkan/vulkan.h>

namespace infernux
{

/// Keep only vertex buffer attributes whose locations are actually read by the vertex shader.
inline std::vector<VkVertexInputAttributeDescription>
FilterVertexAttributesForReflection(const ShaderReflection &vertexReflection)
{
    const auto allAttributes = vk::GetVertexAttributeDescriptions();
    const auto layout = MeshVertexLayout(vertexReflection.GetInputs());
    std::vector<VkVertexInputAttributeDescription> filtered;
    filtered.reserve(layout.attributeCount);
    for (uint32_t index = 0; index < layout.attributeCount; ++index)
        filtered.push_back(allAttributes[layout.attributes[index].location]);
    return filtered;
}

} // namespace infernux
