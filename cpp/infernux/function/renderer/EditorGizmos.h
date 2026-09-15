#pragma once

#include "InxRenderStruct.h"
#include <cstdint>
#include <vector>

namespace infernux
{

/**
 * @brief Editor gizmos for scene visualization.
 *
 * Provides visual aids in the editor:
 * - Grid on XZ plane
 * (Unity-style)
 *
 * These are rendered as overlays in a dedicated Gizmos pass.
 */
class EditorGizmos
{
  public:
    EditorGizmos();
    ~EditorGizmos() = default;

    // ========================================================================
    // Visibility toggles
    // ========================================================================

    void SetShowGrid(bool show)
    {
        m_showGrid = show;
    }
    [[nodiscard]] bool IsShowGrid() const
    {
        return m_showGrid;
    }

    // ========================================================================
    // Grid mesh data access
    // ========================================================================

    /// @brief Get infinite-grid fullscreen quad vertices
    [[nodiscard]] const std::vector<Vertex> &GetGridVertices();

    /// @brief Get infinite-grid fullscreen quad indices
    [[nodiscard]] const std::vector<uint32_t> &GetGridIndices();

    /// @brief Build draw calls for all editor gizmos (grid).
    /// Returns a DrawCallResult that can be appended to the scene draw calls.
    /// @param gizmoMaterial Material for non-grid gizmo rendering
    /// @param gridMaterial Material for grid rendering (distance-fading)
    [[nodiscard]] DrawCallResult GetDrawCalls(std::shared_ptr<InxMaterial> gizmoMaterial,
                                              std::shared_ptr<InxMaterial> gridMaterial);

  private:
    void CreateGridMesh();

    bool m_showGrid = true;

    bool m_gridDirty = true;

    std::vector<Vertex> m_gridVertices;
    std::vector<uint32_t> m_gridIndices;
};

} // namespace infernux
