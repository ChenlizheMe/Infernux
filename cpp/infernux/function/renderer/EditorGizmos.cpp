#include "EditorGizmos.h"

namespace infernux
{

EditorGizmos::EditorGizmos()
{
    CreateGridMesh();
}

const std::vector<Vertex> &EditorGizmos::GetGridVertices()
{
    if (m_gridDirty) {
        CreateGridMesh();
        m_gridDirty = false;
    }
    return m_gridVertices;
}

const std::vector<uint32_t> &EditorGizmos::GetGridIndices()
{
    if (m_gridDirty) {
        CreateGridMesh();
        m_gridDirty = false;
    }
    return m_gridIndices;
}

DrawCallResult EditorGizmos::GetDrawCalls(std::shared_ptr<InxMaterial> gizmoMaterial,
                                          std::shared_ptr<InxMaterial> gridMaterial)
{
    DrawCallResult result;

    // Grid draw call — use persistent cached grid data
    {
        if (m_showGrid) {
            const auto &gridVerts = GetGridVertices();
            const auto &gridInds = GetGridIndices();

            if (!gridVerts.empty()) {
                DrawCall dc;
                dc.indexStart = 0;
                dc.indexCount = static_cast<uint32_t>(gridInds.size());
                dc.worldMatrix = glm::mat4(1.0f); // Grid shader consumes clip-space quad positions directly
                dc.material = gridMaterial ? gridMaterial : gizmoMaterial;
                dc.objectId = 0; // Gizmo objectId = 0
                dc.identity = RenderProxyHandle::Synthetic(RenderDomain::EditorGizmo, 1).MakeDrawIdentity();
                dc.meshVertices = &gridVerts;
                dc.meshIndices = &gridInds;
                result.drawCalls.push_back(dc);
            }
        }
    }

    return result;
}

void EditorGizmos::CreateGridMesh()
{
    m_gridVertices.clear();
    m_gridIndices.clear();

    // Infinite grid: a fullscreen clip-space quad. The shader unprojects each
    // fragment ray and intersects it with the XZ plane at Y=0.
    glm::vec3 color(0.5f, 0.5f, 0.5f); // Not used by procedural shader, kept for vertex format
    glm::vec3 normal(0.0f, 0.0f, 1.0f);
    glm::vec4 tangent(1.0f, 0.0f, 0.0f, 1.0f);

    m_gridVertices.push_back(Vertex::CreateFull({-1.0f, -1.0f, 0.0f}, normal, tangent, color, {0.0f, 0.0f}));
    m_gridVertices.push_back(Vertex::CreateFull({1.0f, -1.0f, 0.0f}, normal, tangent, color, {1.0f, 0.0f}));
    m_gridVertices.push_back(Vertex::CreateFull({1.0f, 1.0f, 0.0f}, normal, tangent, color, {1.0f, 1.0f}));
    m_gridVertices.push_back(Vertex::CreateFull({-1.0f, 1.0f, 0.0f}, normal, tangent, color, {0.0f, 1.0f}));

    // Two triangles covering the whole viewport.
    m_gridIndices.push_back(0);
    m_gridIndices.push_back(1);
    m_gridIndices.push_back(2);
    m_gridIndices.push_back(2);
    m_gridIndices.push_back(3);
    m_gridIndices.push_back(0);
}

} // namespace infernux
