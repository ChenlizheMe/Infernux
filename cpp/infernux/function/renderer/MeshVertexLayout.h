#pragma once

#include "InxRenderStruct.h"
#include "rhi/RhiDescriptors.h"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace infernux
{

/// The engine mesh stream has one layout on every device. Shader compilation
/// selects the locations it reads; backend adapters only translate formats.
inline constexpr std::array<rhi::VertexAttribute, 8> MeshVertexAttributes{{
    {0, 0, rhi::VertexFormat::Float32x3, offsetof(Vertex, pos)},
    {1, 0, rhi::VertexFormat::Float32x3, offsetof(Vertex, normal)},
    {2, 0, rhi::VertexFormat::Float32x4, offsetof(Vertex, tangent)},
    {3, 0, rhi::VertexFormat::Float32x3, offsetof(Vertex, color)},
    {4, 0, rhi::VertexFormat::Float32x2, offsetof(Vertex, texCoord)},
    {5, 0, rhi::VertexFormat::Uint32x4, offsetof(Vertex, boneIndices)},
    {6, 0, rhi::VertexFormat::Float32x4, offsetof(Vertex, boneWeights)},
    {7, 0, rhi::VertexFormat::Float32x2, offsetof(Vertex, texCoord1)},
}};

template <class Inputs> rhi::VertexInputLayout MeshVertexLayout(const Inputs &inputs)
{
    rhi::VertexInputLayout layout;
    for (const auto &input : inputs) {
        if (input.location >= MeshVertexAttributes.size())
            throw std::invalid_argument("Mesh vertex input is outside the engine vertex stream: " + input.name);
        layout.attributes[layout.attributeCount++] = MeshVertexAttributes[input.location];
    }
    if (layout.attributeCount) {
        layout.bufferCount = 1;
        layout.buffers[0] = {sizeof(Vertex), rhi::VertexStepMode::Vertex};
    }
    return layout;
}

} // namespace infernux
