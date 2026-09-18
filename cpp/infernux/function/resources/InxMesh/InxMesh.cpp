#include "InxMesh.h"

#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>

#include <core/log/InxLog.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace infernux
{

std::vector<std::string> InxMesh::GetModelNodePath(size_t index) const
{
    std::vector<std::string> path;
    for (int32_t node = static_cast<int32_t>(index); node >= 0; node = m_modelNodes.at(node).parentIndex)
        path.push_back(m_modelNodes.at(node).name);
    std::reverse(path.begin(), path.end());
    return path;
}

void InxMesh::UpgradeLegacyModelNodePath(std::vector<std::string> &path) const
{
    // Early 041 OBJ references persisted Assimp's synthetic memory-IO root.
    // Migrate that exact root once; authored child names remain authoritative.
    if (path.empty() || path.front() != "$$$___magic___$$$.obj" || m_modelNodes.empty())
        return;
    const auto &root = m_modelNodes.front();
    if (root.parentIndex >= 0 || root.name.size() < 4)
        return;
    auto extension = root.name.substr(root.name.size() - 4);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".obj")
        path.front() = root.name;
}

int32_t InxMesh::RequireModelNode(const std::vector<std::string> &path) const
{
    auto canonical = path;
    UpgradeLegacyModelNodePath(canonical);
    int32_t found = -1;
    for (size_t i = 0; i < m_modelNodes.size(); ++i) {
        if (m_modelNodes[i].nodeGroup < 0 || GetModelNodePath(i) != canonical)
            continue;
        if (found >= 0)
            throw std::invalid_argument("Ambiguous model mesh node path");
        found = static_cast<int32_t>(i);
    }
    if (found < 0 || !m_modelSourceGeometry)
        throw std::invalid_argument("Model mesh node no longer exists; choose an existing mesh");
    return found;
}

std::shared_ptr<InxMesh> InxMesh::CreateModelNodeCopy(const std::vector<std::string> &path) const
{
    const auto &node = m_modelNodes[RequireModelNode(path)];
    const auto &source = *m_modelSourceGeometry;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> subMeshes;
    for (const auto &sub : source.subMeshes) {
        if (sub.nodeGroup != static_cast<uint32_t>(node.nodeGroup))
            continue;
        auto copy = sub;
        copy.vertexStart = static_cast<uint32_t>(vertices.size());
        copy.indexStart = static_cast<uint32_t>(indices.size());
        copy.nodeGroup = 0;
        vertices.insert(vertices.end(), source.vertices.begin() + sub.vertexStart,
                        source.vertices.begin() + sub.vertexStart + sub.vertexCount);
        for (uint32_t i = 0; i < sub.indexCount; ++i)
            indices.push_back(source.indices[sub.indexStart + i] - sub.vertexStart + copy.vertexStart);
        subMeshes.push_back(std::move(copy));
    }
    auto result = std::make_shared<InxMesh>();
    result->SetName(node.name);
    result->SetData(std::move(vertices), std::move(indices), std::move(subMeshes));
    result->SetMaterialSlotNames(m_materialSlotNames);
    result->SetMaterialSlotData(m_materialSlotData);
    return result;
}

size_t InxMesh::GetRuntimeMemoryBytes() const noexcept
{
    size_t bytes = sizeof(*this) + m_name.capacity() + m_guid.capacity() + m_filePath.capacity();
    bytes += sizeof(MeshGeometry);
    bytes += m_geometry->vertices.capacity() * sizeof(Vertex);
    bytes += m_geometry->indices.capacity() * sizeof(uint32_t);
    bytes += m_geometry->subMeshes.capacity() * sizeof(SubMesh);
    for (const auto &subMesh : m_geometry->subMeshes)
        bytes += subMesh.name.capacity();
    if (m_modelSourceGeometry) {
        bytes += sizeof(MeshGeometry) + m_modelSourceGeometry->vertices.capacity() * sizeof(Vertex) +
                 m_modelSourceGeometry->indices.capacity() * sizeof(uint32_t) +
                 m_modelSourceGeometry->subMeshes.capacity() * sizeof(SubMesh);
        for (const auto &subMesh : m_modelSourceGeometry->subMeshes)
            bytes += subMesh.name.capacity();
    }
    bytes += m_materialSlotNames.capacity() * sizeof(std::string);
    for (const auto &name : m_materialSlotNames)
        bytes += name.capacity();
    bytes += m_materialSlotData.capacity() * sizeof(MaterialSlotData);
    for (const auto &material : m_materialSlotData)
        bytes += material.sourceId.capacity() + material.materialGuid.capacity();
    bytes += m_nodeNames.capacity() * sizeof(std::string);
    for (const auto &name : m_nodeNames)
        bytes += name.capacity();
    bytes += m_modelNodes.capacity() * sizeof(ImportedModelNode);
    for (const auto &node : m_modelNodes)
        bytes += node.name.capacity();
    if (m_skinnedData)
        bytes += m_skinnedData->GetRuntimeMemoryBytes();
    return bytes;
}

void InxMesh::SetModelNodes(std::vector<ImportedModelNode> nodes)
{
    if (m_modelSourceGeometry) {
        SetModelData(m_modelSourceGeometry->vertices, m_modelSourceGeometry->indices, m_modelSourceGeometry->subMeshes,
                     std::move(nodes));
        return;
    }
    std::vector<bool> assignedGroups(m_nodeNames.size(), false);
    for (size_t index = 0; index < nodes.size(); ++index) {
        const auto &node = nodes[index];
        if (node.parentIndex < -1 || (node.parentIndex >= 0 && static_cast<size_t>(node.parentIndex) >= index))
            throw std::invalid_argument("Model nodes require parent-before-child order");
        if (node.nodeGroup < -1 || (node.nodeGroup >= 0 && static_cast<size_t>(node.nodeGroup) >= m_nodeNames.size()))
            throw std::invalid_argument("Model node refers to an invalid geometry group");
        if (node.nodeGroup >= 0) {
            if (assignedGroups[node.nodeGroup])
                throw std::invalid_argument("Model geometry group belongs to more than one node");
            assignedGroups[node.nodeGroup] = true;
        }
        for (glm::length_t column = 0; column < 4; ++column)
            for (glm::length_t row = 0; row < 4; ++row)
                if (!std::isfinite(node.localTransform[column][row]))
                    throw std::invalid_argument("Model node transform must be finite");
        if (node.localTransform[0][3] != 0.0f || node.localTransform[1][3] != 0.0f ||
            node.localTransform[2][3] != 0.0f || node.localTransform[3][3] != 1.0f)
            throw std::invalid_argument("Model node transform must be affine");
    }
    m_modelNodes = std::move(nodes);
}

void InxMesh::SetSkinnedData(std::shared_ptr<const InxSkinnedMesh> skinnedData)
{
    if (skinnedData && !skinnedData->IsAssetPayloadValid())
        throw std::invalid_argument("InxMesh cannot attach invalid skinned data");
    m_skinnedData = std::move(skinnedData);
    ++m_generation;
}

void InxMesh::SetModelData(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<SubMesh> subMeshes,
                           std::vector<ImportedModelNode> nodes)
{
    // Validate and derive away from the published generation. Neither source
    // geometry nor its model-space view may be partially replaced on failure.
    InxMesh candidate;
    candidate.SetNodeNames(m_nodeNames);
    candidate.SetModelNodes(std::move(nodes));
    candidate.SetData(std::move(vertices), std::move(indices), std::move(subMeshes));
    const auto source = candidate.GetGeometrySnapshot();
    auto bakedVertices = source->vertices;
    auto bakedSubMeshes = source->subMeshes;
    std::vector<glm::mat4> world(candidate.m_modelNodes.size());
    std::vector<int32_t> groupNodes(m_nodeNames.size(), -1);
    for (size_t index = 0; index < candidate.m_modelNodes.size(); ++index) {
        const auto &node = candidate.m_modelNodes[index];
        world[index] = node.parentIndex < 0 ? node.localTransform : world[node.parentIndex] * node.localTransform;
        if (node.nodeGroup >= 0)
            groupNodes[node.nodeGroup] = static_cast<int32_t>(index);
    }
    const auto normalized = [](const glm::vec3 &value) {
        const float squared = glm::dot(value, value);
        return squared > 0.0f ? value / std::sqrt(squared) : glm::vec3(0.0f);
    };
    std::vector<int32_t> vertexGroups(source->vertices.size(), -1);
    for (auto &sub : bakedSubMeshes) {
        if (sub.nodeGroup >= groupNodes.size() || groupNodes[sub.nodeGroup] < 0 ||
            sub.vertexStart > bakedVertices.size() || sub.vertexCount > bakedVertices.size() - sub.vertexStart ||
            sub.indexStart > source->indices.size() || sub.indexCount > source->indices.size() - sub.indexStart)
            throw std::invalid_argument("Model source geometry has an invalid node or submesh range");
        const auto &matrix = world[groupNodes[sub.nodeGroup]];
        const glm::mat3 linear(matrix);
        const float orientation = glm::determinant(linear) < 0.0f ? -1.0f : 1.0f;
        // Cofactors transform normals even at singular authored scales. A
        // fully collapsed surface has a zero normal, not NaNs from inverse().
        const glm::mat3 normals(glm::cross(linear[1], linear[2]) * orientation,
                                glm::cross(linear[2], linear[0]) * orientation,
                                glm::cross(linear[0], linear[1]) * orientation);
        sub.boundsMin = sub.vertexCount ? glm::vec3(std::numeric_limits<float>::max()) : glm::vec3(0.0f);
        sub.boundsMax = sub.vertexCount ? glm::vec3(std::numeric_limits<float>::lowest()) : glm::vec3(0.0f);
        for (size_t offset = 0; offset < sub.vertexCount; ++offset) {
            const size_t index = sub.vertexStart + offset;
            if (vertexGroups[index] >= 0 && vertexGroups[index] != static_cast<int32_t>(sub.nodeGroup))
                throw std::invalid_argument("Model source vertices cannot belong to different node spaces");
            vertexGroups[index] = static_cast<int32_t>(sub.nodeGroup);
            const auto &original = source->vertices[index];
            auto &vertex = bakedVertices[index];
            vertex.pos = glm::vec3(matrix * glm::vec4(original.pos, 1.0f));
            vertex.normal = normalized(normals * original.normal);
            vertex.tangent =
                glm::vec4(normalized(linear * glm::vec3(original.tangent)), original.tangent.w * orientation);
            sub.boundsMin = glm::min(sub.boundsMin, vertex.pos);
            sub.boundsMax = glm::max(sub.boundsMax, vertex.pos);
        }
        for (size_t offset = 0; offset < sub.indexCount; ++offset) {
            const uint32_t vertex = source->indices[sub.indexStart + offset];
            if (vertex < sub.vertexStart || vertex - sub.vertexStart >= sub.vertexCount)
                throw std::invalid_argument("Model source index crosses its local vertex range");
        }
    }
    if (std::find(vertexGroups.begin(), vertexGroups.end(), -1) != vertexGroups.end())
        throw std::invalid_argument("Model source vertices must belong to a node space");
    candidate.SetData(std::move(bakedVertices), source->indices, std::move(bakedSubMeshes));
    m_geometry = candidate.m_geometry;
    m_modelSourceGeometry = source;
    m_modelNodes = std::move(candidate.m_modelNodes);
    ++m_generation;
}

void InxMesh::ReplaceImportedContent(const InxMesh &source)
{
    InxMesh replacement(source);
    replacement.m_guid = m_guid;
    replacement.m_generation = m_generation + 1;
    *this = std::move(replacement);
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
    m_modelSourceGeometry.reset();
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
