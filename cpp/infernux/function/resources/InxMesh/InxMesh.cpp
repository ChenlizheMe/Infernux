#include "InxMesh.h"

#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>

#include <core/log/InxLog.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernux
{

size_t InxMesh::GetRuntimeMemoryBytes() const noexcept
{
    size_t bytes = sizeof(*this) + m_name.capacity() + m_guid.capacity() + m_filePath.capacity();
    bytes += sizeof(MeshGeometry);
    bytes += m_geometry->vertices.capacity() * sizeof(Vertex);
    bytes += m_geometry->indices.capacity() * sizeof(uint32_t);
    bytes += m_geometry->subMeshes.capacity() * sizeof(SubMesh);
    for (const auto &subMesh : m_geometry->subMeshes)
        bytes += subMesh.name.capacity();
    bytes += m_materialSlotNames.capacity() * sizeof(std::string);
    for (const auto &name : m_materialSlotNames)
        bytes += name.capacity();
    bytes += m_materialSlotData.capacity() * sizeof(MaterialSlotData);
    bytes += m_nodeNames.capacity() * sizeof(std::string);
    for (const auto &name : m_nodeNames)
        bytes += name.capacity();
    if (m_skinnedData)
        bytes += m_skinnedData->GetRuntimeMemoryBytes();
    return bytes;
}

void InxMesh::SetSkinnedData(std::shared_ptr<const InxSkinnedMesh> skinnedData)
{
    if (skinnedData && !skinnedData->IsAssetPayloadValid())
        throw std::invalid_argument("InxMesh cannot attach invalid skinned data");
    m_skinnedData = std::move(skinnedData);
    ++m_generation;
}

void InxMesh::SetData(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<SubMesh> subMeshes)
{
    auto geometry = std::make_shared<MeshGeometry>();
    geometry->vertices = std::move(vertices);
    geometry->indices = std::move(indices);
    geometry->subMeshes = std::move(subMeshes);
    if (!geometry->vertices.empty()) {
        constexpr float INF = std::numeric_limits<float>::max();
        geometry->boundsMin = glm::vec3(INF);
        geometry->boundsMax = glm::vec3(-INF);
        for (const auto &vertex : geometry->vertices) {
            geometry->boundsMin = glm::min(geometry->boundsMin, vertex.pos);
            geometry->boundsMax = glm::max(geometry->boundsMax, vertex.pos);
        }
    }
    m_geometry = std::move(geometry);
    ++m_generation;
}

void InxMesh::UpdateVertexRange(size_t first, const std::vector<Vertex> &replacement)
{
    if (first > m_geometry->vertices.size() || replacement.size() > m_geometry->vertices.size() - first)
        throw std::invalid_argument("Mesh vertex update exceeds the existing vertex range");
    if (replacement.empty())
        return;

    auto vertices = m_geometry->vertices;
    auto subMeshes = m_geometry->subMeshes;
    std::copy(replacement.begin(), replacement.end(), vertices.begin() + first);
    for (auto &subMesh : subMeshes) {
        subMesh.boundsMin = glm::vec3(0.0f);
        subMesh.boundsMax = glm::vec3(0.0f);
        if (subMesh.vertexCount == 0)
            continue;
        subMesh.boundsMin = glm::vec3(std::numeric_limits<float>::max());
        subMesh.boundsMax = glm::vec3(-std::numeric_limits<float>::max());
        for (size_t offset = 0; offset < subMesh.vertexCount; ++offset) {
            const auto &position = vertices.at(static_cast<size_t>(subMesh.vertexStart) + offset).pos;
            subMesh.boundsMin = glm::min(subMesh.boundsMin, position);
            subMesh.boundsMax = glm::max(subMesh.boundsMax, position);
        }
    }
    SetData(std::move(vertices), m_geometry->indices, std::move(subMeshes));
}

} // namespace infernux
