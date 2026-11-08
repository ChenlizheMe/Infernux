#include "RectTransformLayout.h"
#include "Transform.h"
#include <stdexcept>

namespace infernux
{
namespace
{
glm::vec2 Vector(ui::RectPoint value)
{
    return {value[0], value[1]};
}

uint64_t DriverToken(Transform *driver)
{
    const auto handle = driver->GetECSHandle();
    return uint64_t(handle.generation) << 32 | handle.index;
}

Transform *DriverOwner(uint64_t token)
{
    const TransformECSStore::Handle handle{uint32_t(token), uint32_t(token >> 32)};
    auto &store = TransformECSStore::Instance();
    return store.IsValid(handle) ? store.GetOwner(handle) : nullptr;
}

void ParentFrame(Transform *owner, ui::RectTransformState &state)
{
    auto *parent = owner->GetParent();
    auto *layout = parent ? TransformECSStore::Instance().GetRectLayout(parent->GetECSHandle()) : nullptr;
    state.screenSpace = state.unitMode == ui::RectUnitMode::ScreenCanvas || (layout && layout->screenSpace);
    if (state.unitMode != ui::RectUnitMode::Explicit) {
        const float scale = state.screenSpace ? 1.f : .01f;
        if (state.positionScale != scale) {
            auto &store = TransformECSStore::Instance();
            const auto handle = owner->GetECSHandle();
            store.SetRectLayoutZ(handle, store.GetLocalPosition(handle).z * (scale / state.positionScale));
            state.positionScale = scale;
        }
    }
    if (layout) {
        const auto rect = ui::CompileRectLayout(layout->spec).rect;
        state.spec.parentSize = {rect[2], rect[3]};
        state.spec.parentPivot = layout->spec.pivot;
        state.drivenParent = layout->drive.has_value() || layout->drivenParent;
        if (state.drivenParent) {
            const auto authored = layout->AuthoringSpec();
            const auto rect = ui::CompileRectLayout(authored).rect;
            state.authoringParentSize = {rect[2], rect[3]};
            state.authoringParentPivot = authored.pivot;
        }
    } else {
        state.spec.parentSize = {};
        state.spec.parentPivot = {.5f, .5f};
        state.drivenParent = false;
    }
    state.RefreshReference();
}

void ResolvePendingPositions(Transform *root)
{
    auto &store = TransformECSStore::Instance();
    if (!store.GetRectLayout(root->GetECSHandle()))
        return;
    (void)root->GetLocalPosition();
    for (size_t index = 0; index < root->GetChildCount(); ++index)
        ResolvePendingPositions(root->GetChild(index));
}
} // namespace

RectTransformLayout::RectTransformLayout(Transform *owner) : m_owner(owner->GetECSHandle())
{
    auto &store = TransformECSStore::Instance();
    if (store.GetRectLayout(m_owner))
        return;
    const auto position = owner->GetLocalPosition();
    auto &state = store.EnableRectLayout(m_owner);
    ParentFrame(owner, state);
    state.SetLocalPosition({position.x, position.y});
    RefreshSubtree(owner);
}

Transform *RectTransformLayout::Owner() const
{
    auto &store = TransformECSStore::Instance();
    if (!store.IsValid(m_owner))
        throw std::runtime_error("Rectangle layout outlived its Transform");
    return store.GetOwner(m_owner);
}

ui::RectTransformState &RectTransformLayout::State() const
{
    (void)Owner()->GetLocalPosition();
    auto *state = TransformECSStore::Instance().GetRectLayout(m_owner);
    if (!state)
        throw std::runtime_error("Rectangle layout was removed from its Transform");
    return *state;
}

void RectTransformLayout::Validate() const
{
    (void)Owner();
    if (!TransformECSStore::Instance().GetRectLayout(m_owner))
        throw std::runtime_error("Rectangle layout was removed from its Transform");
}

ui::RectLayoutGeometry RectTransformLayout::Geometry() const
{
    return ui::CompileRectLayout(State().spec);
}
glm::vec2 RectTransformLayout::GetAnchorMin() const
{
    return Vector(State().spec.anchorMin);
}
glm::vec2 RectTransformLayout::GetAnchorMax() const
{
    return Vector(State().spec.anchorMax);
}
glm::vec2 RectTransformLayout::GetPivot() const
{
    return Vector(State().spec.pivot);
}
glm::vec2 RectTransformLayout::GetSizeDelta() const
{
    return Vector(State().spec.sizeDelta);
}
glm::vec2 RectTransformLayout::GetAnchoredPosition() const
{
    return Vector(State().spec.anchoredPosition);
}
glm::vec3 RectTransformLayout::GetAnchoredPosition3D() const
{
    const auto position = GetAnchoredPosition();
    return {position, Owner()->GetLocalPosition().z / State().positionScale};
}
glm::vec4 RectTransformLayout::GetRect() const
{
    const auto rect = Geometry().rect;
    return {rect[0], rect[1], rect[2], rect[3]};
}

glm::vec4 RectTransformLayout::GetAuthoringRect() const
{
    const auto rect = ui::CompileRectLayout(State().AuthoringSpec()).rect;
    return {rect[0], rect[1], rect[2], rect[3]};
}

Transform *RectTransformLayout::GetDriver() const
{
    const auto &state = State();
    return state.drive ? DriverOwner(state.drive->owner) : nullptr;
}
glm::vec2 RectTransformLayout::GetOffsetMin() const
{
    return Vector(Geometry().offsetMin);
}
glm::vec2 RectTransformLayout::GetOffsetMax() const
{
    return Vector(Geometry().offsetMax);
}
uint64_t RectTransformLayout::GetRevision() const
{
    return State().revision;
}
float RectTransformLayout::GetPositionScale() const
{
    return State().positionScale;
}

void RectTransformLayout::SetPositionScale(float scale)
{
    if (!std::isfinite(scale) || scale <= 0)
        throw std::invalid_argument("Rectangle position scale must be positive and finite");
    auto &state = State();
    if (state.positionScale == scale && state.unitMode == ui::RectUnitMode::Explicit)
        return;
    const auto position = Owner()->GetLocalPosition();
    state.unitMode = ui::RectUnitMode::Explicit;
    state.positionScale = scale;
    TransformECSStore::Instance().MarkRectLayoutInputChanged();
    state.SetLocalPosition({position.x, position.y});
    RefreshSubtree(Owner());
}

int RectTransformLayout::GetUnitMode() const
{
    return static_cast<int>(State().unitMode);
}

void RectTransformLayout::SetUnitMode(int mode)
{
    if (mode < 0 || mode > 2)
        throw std::invalid_argument("Unknown rectangle unit mode");
    auto &state = State();
    const auto value = static_cast<ui::RectUnitMode>(mode);
    if (state.unitMode == value)
        return;
    auto *owner = Owner();
    const auto position = owner->GetLocalPosition();
    state.unitMode = value;
    ParentFrame(owner, state);
    state.SetLocalPosition({position.x, position.y});
    auto &store = TransformECSStore::Instance();
    store.SetRectLayoutZ(m_owner, position.z);
    store.MarkRectLayoutInputChanged();
    RefreshSubtree(owner);
}

std::pair<glm::vec4, glm::vec2> RectTransformLayout::ResolveParent(glm::vec2 size, glm::vec2 pivot) const
{
    auto state = State();
    state.spec.parentSize = {size.x, size.y};
    state.spec.parentPivot = {pivot.x, pivot.y};
    state.positionScale = 1;
    state.RefreshReference();
    const auto rect = ui::CompileRectLayout(state.spec).rect;
    return {{rect[0], rect[1], rect[2], rect[3]}, Vector(state.LocalPosition())};
}

std::array<glm::vec3, 4> RectTransformLayout::GetLocalCorners() const
{
    const auto geometry = Geometry();
    std::array<glm::vec3, 4> result;
    for (size_t index = 0; index < 4; ++index) {
        const auto &point = geometry.localCorners[index];
        result[index] = {point[0], point[1], point[2]};
    }
    return result;
}

std::array<glm::vec3, 4> RectTransformLayout::GetWorldCorners() const
{
    auto result = GetLocalCorners();
    const auto *owner = Owner();
    const auto scale = State().positionScale;
    for (auto &point : result)
        point = owner->TransformPoint(point * scale);
    return result;
}

void RectTransformLayout::RefreshSubtree(Transform *root, bool authored)
{
    auto &store = TransformECSStore::Instance();
    auto *state = store.GetRectLayout(root->GetECSHandle());
    if (!state)
        return;
    if (state->drive) {
        auto *driver = DriverOwner(state->drive->owner);
        if (!driver || (driver != root && driver != root->GetParent()))
            state->EndDrive();
    }
    ParentFrame(root, *state);
    state->revision = store.NextRectLayoutRevision();
    store.ApplyRectLayout(root->GetECSHandle(), authored);
    for (size_t index = 0; index < root->GetChildCount(); ++index)
        RefreshSubtree(root->GetChild(index), authored);
}

bool RectTransformLayout::ApplyDrive(Transform *driver, glm::vec2 size, const glm::vec2 *position, uint8_t sizeMask)
{
    auto *owner = Owner();
    if (driver != owner && driver != owner->GetParent())
        throw std::invalid_argument("Rectangle driver must own this frame or its direct parent");
    auto &state = State();
    const auto token = DriverToken(driver);
    if (state.drive && state.drive->owner != token)
        throw std::runtime_error("Rectangle already belongs to another layout driver");
    const auto previous = state.spec;
    const bool changedOwner =
        !state.drive || state.drive->position != (position != nullptr) || state.drive->sizeMask != sizeMask;
    ResolvePendingPositions(owner);
    state.BeginDrive(token, position != nullptr, sizeMask);
    for (size_t axis = 0; axis < 2; ++axis) {
        if (sizeMask & (1 << axis))
            state.spec.sizeDelta[axis] =
                size[axis] - state.spec.parentSize[axis] * (state.spec.anchorMax[axis] - state.spec.anchorMin[axis]);
        if (position)
            state.spec.anchoredPosition[axis] = (*position)[axis] - state.anchorReference[axis];
    }
    const bool changed = changedOwner || previous.sizeDelta != state.spec.sizeDelta ||
                         previous.anchoredPosition != state.spec.anchoredPosition;
    if (changed)
        RefreshSubtree(owner, false);
    return changed;
}

bool RectTransformLayout::DriveSize(Transform *driver, glm::vec2 size, uint8_t sizeMask)
{
    return ApplyDrive(driver, size, nullptr, sizeMask);
}
bool RectTransformLayout::DriveRect(Transform *driver, glm::vec2 size, glm::vec2 position)
{
    return ApplyDrive(driver, size, &position);
}

bool RectTransformLayout::ClearDriver(Transform *driver)
{
    auto &state = State();
    if (!state.drive || state.drive->owner != DriverToken(driver))
        return false;
    ResolvePendingPositions(Owner());
    state.EndDrive();
    RefreshSubtree(Owner(), false);
    return true;
}

void RectTransformLayout::SetAuthoringSizeWithCurrentAnchors(int axis, float size)
{
    if (axis < 0 || axis > 1)
        throw std::out_of_range("Rectangle axis must be 0 or 1");
    auto &state = State();
    if (!state.drive || !(state.drive->sizeMask & (1 << axis))) {
        SetSizeWithCurrentAnchors(axis, size);
        return;
    }
    const auto authored = state.AuthoringSpec();
    const auto delta = size - authored.parentSize[axis] * (authored.anchorMax[axis] - authored.anchorMin[axis]);
    if (state.drive->sizeDelta[axis] == delta)
        return;
    state.drive->sizeDelta[axis] = delta;
    TransformECSStore::Instance().MarkRectLayoutInputChanged();
    RefreshSubtree(Owner(), false);
}

void RectTransformLayout::SetAuthoringLocalPosition(glm::vec3 position)
{
    auto &state = State();
    auto authored = state;
    authored.spec = state.AuthoringSpec();
    authored.RefreshReference();
    authored.SetLocalPosition({position.x, position.y});
    if (state.drive && state.drive->position)
        state.drive->anchoredPosition = authored.spec.anchoredPosition;
    else
        state.spec.anchoredPosition = authored.spec.anchoredPosition;
    auto &store = TransformECSStore::Instance();
    store.SetRectLayoutZ(m_owner, position.z);
    store.MarkRectLayoutInputChanged();
    store.ApplyRectLayout(m_owner, false);
}

void RectTransformLayout::SetDimension(ui::RectPoint ui::RectLayoutSpec::*field, glm::vec2 value)
{
    auto &spec = State().spec;
    const ui::RectPoint point{value.x, value.y};
    if (spec.*field == point)
        return;
    ResolvePendingPositions(Owner());
    spec.*field = point;
    TransformECSStore::Instance().MarkRectLayoutInputChanged();
    RefreshSubtree(Owner());
}

void RectTransformLayout::SetAnchorMin(glm::vec2 value)
{
    SetDimension(&ui::RectLayoutSpec::anchorMin, value);
}
void RectTransformLayout::SetAnchorMax(glm::vec2 value)
{
    SetDimension(&ui::RectLayoutSpec::anchorMax, value);
}
void RectTransformLayout::SetPivot(glm::vec2 value)
{
    SetDimension(&ui::RectLayoutSpec::pivot, value);
}
void RectTransformLayout::SetSizeDelta(glm::vec2 value)
{
    SetDimension(&ui::RectLayoutSpec::sizeDelta, value);
}

void RectTransformLayout::SetAnchoredPosition(glm::vec2 value)
{
    if (State().spec.anchoredPosition == ui::RectPoint{value.x, value.y})
        return;
    State().spec.anchoredPosition = {value.x, value.y};
    TransformECSStore::Instance().MarkRectLayoutInputChanged();
    TransformECSStore::Instance().ApplyRectLayout(m_owner);
}

void RectTransformLayout::SetAnchoredPosition3D(glm::vec3 value)
{
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
        return;
    State().spec.anchoredPosition = {value.x, value.y};
    TransformECSStore::Instance().MarkRectLayoutInputChanged();
    auto &store = TransformECSStore::Instance();
    // Preserve the authored XY parameterization while updating ordinary local Z.
    store.SetRectLayoutZ(m_owner, value.z * State().positionScale);
    store.ApplyRectLayout(m_owner);
}

void RectTransformLayout::SetOffsetMin(glm::vec2 value)
{
    const auto delta = value - GetOffsetMin();
    const auto pivot = GetPivot();
    SetSizeDelta(GetSizeDelta() - delta);
    SetAnchoredPosition(GetAnchoredPosition() + delta * (glm::vec2(1) - pivot));
}

void RectTransformLayout::SetOffsetMax(glm::vec2 value)
{
    const auto delta = value - GetOffsetMax();
    const auto pivot = GetPivot();
    SetSizeDelta(GetSizeDelta() + delta);
    SetAnchoredPosition(GetAnchoredPosition() + delta * pivot);
}

void RectTransformLayout::SetSizeWithCurrentAnchors(int axis, float size)
{
    if (axis < 0 || axis > 1)
        throw std::out_of_range("Rectangle axis must be 0 or 1");
    const auto &spec = State().spec;
    auto delta = GetSizeDelta();
    delta[axis] = size - spec.parentSize[axis] * (spec.anchorMax[axis] - spec.anchorMin[axis]);
    SetSizeDelta(delta);
}

void RectTransformLayout::SetInsetAndSizeFromParentEdge(int edge, float inset, float size)
{
    if (edge < 0 || edge > 3)
        throw std::out_of_range("Rectangle edge must be 0 through 3");
    const int axis = edge >= 2 ? 1 : 0;
    const bool end = edge == 1 || edge == 2;
    auto minimum = GetAnchorMin(), maximum = GetAnchorMax(), delta = GetSizeDelta();
    minimum[axis] = maximum[axis] = end ? 1.f : 0.f;
    SetAnchorMin(minimum);
    SetAnchorMax(maximum);
    delta[axis] = size;
    SetSizeDelta(delta);
    auto position = GetAnchoredPosition();
    const auto pivot = GetPivot();
    position[axis] = end ? -inset - size * (1 - pivot[axis]) : inset + size * pivot[axis];
    SetAnchoredPosition(position);
}
} // namespace infernux
