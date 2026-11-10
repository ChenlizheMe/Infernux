#pragma once

#include "TransformECSStore.h"
#include "core/math/RectLayout.h"
#include <array>
#include <utility>

namespace infernux
{
class Transform;

// A checked handle to the optional layout owned by the native Transform store.
// Retaining this value never keeps a destroyed scene object alive.
class RectTransformLayout
{
  public:
    explicit RectTransformLayout(Transform *owner);
    glm::vec2 GetAnchorMin() const;
    glm::vec2 GetAnchorMax() const;
    glm::vec2 GetPivot() const;
    glm::vec2 GetSizeDelta() const;
    glm::vec2 GetAnchoredPosition() const;
    glm::vec3 GetAnchoredPosition3D() const;
    glm::vec4 GetRect() const;
    glm::vec2 GetOffsetMin() const;
    glm::vec2 GetOffsetMax() const;
    std::array<glm::vec3, 4> GetLocalCorners() const;
    std::array<glm::vec3, 4> GetWorldCorners() const;
    uint64_t GetRevision() const;
    float GetPositionScale() const;
    void SetPositionScale(float scale);
    int GetUnitMode() const;
    void SetUnitMode(int mode);
    std::pair<glm::vec4, glm::vec2> ResolveParent(glm::vec2 size, glm::vec2 pivot) const;
    glm::vec4 GetAuthoringRect() const;
    Transform *GetDriver() const;
    bool DriveSize(Transform *driver, glm::vec2 size, uint8_t sizeMask = 3);
    bool DriveRect(Transform *driver, glm::vec2 size, glm::vec2 localPosition);
    bool ClearDriver(Transform *driver);
    void SetAuthoringSizeWithCurrentAnchors(int axis, float size);
    void SetAuthoringLocalPosition(glm::vec3 position);
    void Validate() const;
    void SetAnchorMin(glm::vec2 value);
    void SetAnchorMax(glm::vec2 value);
    void SetPivot(glm::vec2 value);
    void SetSizeDelta(glm::vec2 value);
    void SetAnchoredPosition(glm::vec2 value);
    void SetAnchoredPosition3D(glm::vec3 value);
    void SetOffsetMin(glm::vec2 value);
    void SetOffsetMax(glm::vec2 value);
    void SetSizeWithCurrentAnchors(int axis, float size);
    void SetInsetAndSizeFromParentEdge(int edge, float inset, float size);

    static void RefreshSubtree(Transform *root, bool authored = true);

  private:
    Transform *Owner() const;
    ui::RectTransformState &State() const;
    ui::RectLayoutGeometry Geometry() const;
    void SetDimension(ui::RectPoint ui::RectLayoutSpec::*field, glm::vec2 value);
    bool ApplyDrive(Transform *driver, glm::vec2 size, const glm::vec2 *position, uint8_t sizeMask = 3);
    TransformECSStore::Handle m_owner;
};
} // namespace infernux
