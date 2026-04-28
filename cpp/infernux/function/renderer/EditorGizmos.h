#pragma once

#include "InxRenderStruct.h"
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

namespace infernux
{

class Scene;

/**
 * @brief Editor gizmos for scene visualization.
 *
 * Provides visual aids in the editor:
 * - Grid on XZ plane (Unity-style)
 * - Selection outline using normal expansion (Unity-style)
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

    void SetOutlineWidth(float width)
    {
        m_outlineWidth = width;
        m_selectionDirty = true;
    }
    [[nodiscard]] float GetOutlineWidth() const
    {
        return m_outlineWidth;
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
    /// @param selectedObjectId Currently selected object ID (0 = none)
    /// @param activeScene Active scene for selected object transform lookup
    /// @param cameraPos Camera position for distance-based outline width
    [[nodiscard]] DrawCallResult GetDrawCalls(std::shared_ptr<InxMaterial> gizmoMaterial,
                                              std::shared_ptr<InxMaterial> gridMaterial, uint64_t selectedObjectId,
                                              Scene *activeScene, const glm::vec3 &cameraPos);

    // ========================================================================
    // Selection Outline (normal expansion method)
    // ========================================================================

    /// @brief Set the outline mesh for selection visualization
    /// @param positions Mesh vertex positions
    /// @param normals Mesh vertex normals (for expansion)
    /// @param indices Mesh triangle indices
    /// @param worldMatrix Object's world transform matrix
    void SetSelectionOutline(const std::vector<glm::vec3> &positions, const std::vector<glm::vec3> &normals,
                             const std::vector<uint32_t> &indices, const glm::mat4 &worldMatrix);
    void ClearSelectionOutline();
    [[nodiscard]] bool HasSelectionOutline() const
    {
        return m_hasSelectionOutline;
    }

    /// @brief Update world matrix for selection outline (for Transform changes)
    void UpdateSelectionWorldMatrix(const glm::mat4 &worldMatrix)
    {
        m_selectionWorldMatrix = worldMatrix;
        // No need to rebuild mesh - just update matrix
    }

    /// @brief Get outline mesh data for rendering with outline shader
    /// @note This returns the original mesh that will be expanded by the shader
    void GetOutlineMeshData(std::vector<Vertex> &outVertices, std::vector<uint32_t> &outIndices);

    /// @brief Get the world matrix for the selection outline
    [[nodiscard]] const glm::mat4 &GetOutlineWorldMatrix() const
    {
        return m_selectionWorldMatrix;
    }

    /// @brief Get scaled world matrix for outline rendering (slightly larger than original)
    [[nodiscard]] glm::mat4 GetOutlineScaledWorldMatrix() const;

    /// @brief Set outline scale factor (default 1.03 = 3% larger)
    void SetOutlineScale(float scale)
    {
        m_outlineScale = scale;
        m_selectionDirty = true;
    }
    [[nodiscard]] float GetOutlineScale() const
    {
        return m_outlineScale;
    }

    /// @brief Set camera position for distance-based outline width calculation
    void SetCameraPosition(const glm::vec3 &cameraPos)
    {
        m_cameraPosition = cameraPos;
    }

    /// @brief Set screen-space outline width in pixels (default 2.0)
    void SetOutlinePixelWidth(float pixels)
    {
        m_outlinePixelWidth = pixels;
    }
    [[nodiscard]] float GetOutlinePixelWidth() const
    {
        return m_outlinePixelWidth;
    }

  private:
    void CreateGridMesh();
    void CreateOutlineMesh();

    bool m_showGrid = true;

    bool m_gridDirty = true;

    std::vector<Vertex> m_gridVertices;
    std::vector<uint32_t> m_gridIndices;

    // Selection outline (normal expansion method)
    bool m_hasSelectionOutline = false;
    float m_outlineWidth = 0.05f;
    float m_outlineScale = 1.03f;                 // Base scale factor for outline
    float m_outlinePixelWidth = 3.0f;             // Desired outline width in screen pixels
    glm::vec3 m_cameraPosition{0.0f, 0.0f, 5.0f}; // Camera position for distance calculation
    std::vector<glm::vec3> m_selectionPositions;
    std::vector<glm::vec3> m_selectionNormals;
    std::vector<uint32_t> m_selectionIndices;
    glm::mat4 m_selectionWorldMatrix{1.0f};

    bool m_selectionDirty = true;
    std::vector<Vertex> m_outlineVertices;
    std::vector<uint32_t> m_outlineIndices;
};

} // namespace infernux
