#include "RectLayoutTypes.h"
#include <cstddef>

namespace infernux::ui
{
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void RectTransformState::RefreshReference()
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (size_t axis = 0; axis < 2; ++axis) {
        const float origin = -spec.parentSize[axis] * spec.parentPivot[axis];
        const float minimum = origin + spec.parentSize[axis] * spec.anchorMin[axis];
        const float maximum = origin + spec.parentSize[axis] * spec.anchorMax[axis];
        anchorReference[axis] = minimum + (maximum - minimum) * spec.pivot[axis];
    }
}

RectPoint RectTransformState::LocalPosition() const
{
    return {(spec.anchoredPosition[0] + anchorReference[0]) * positionScale,
            (spec.anchoredPosition[1] + anchorReference[1]) * positionScale};
}

RectPoint RectTransformState::SetLocalPosition(RectPoint value)
{
    for (size_t axis = 0; axis < 2; ++axis)
        spec.anchoredPosition[axis] = value[axis] / positionScale - anchorReference[axis];
    return LocalPosition();
}

RectLayoutSpec RectTransformState::AuthoringSpec() const
{
    auto authored = spec;
    if (drivenParent) {
        authored.parentSize = authoringParentSize;
        authored.parentPivot = authoringParentPivot;
    }
    if (drive) {
        for (size_t axis = 0; axis < 2; ++axis)
            if (drive->sizeMask & (1 << axis))
                authored.sizeDelta[axis] = drive->sizeDelta[axis];
        if (drive->position)
            authored.anchoredPosition = drive->anchoredPosition;
    }
    return authored;
}

RectPoint RectTransformState::AuthoringLocalPosition() const
{
    auto authored = *this;
    authored.spec = AuthoringSpec();
    authored.RefreshReference();
    return authored.LocalPosition();
}

void RectTransformState::BeginDrive(uint64_t owner, bool position, uint8_t sizeMask)
{
    if (!drive)
        drive = RectLayoutDrive{owner, spec.anchoredPosition, spec.sizeDelta, position, sizeMask};
    else {
        if (drive->position != position) {
            if (position)
                drive->anchoredPosition = spec.anchoredPosition;
            else
                spec.anchoredPosition = drive->anchoredPosition;
            drive->position = position;
        }
        for (size_t axis = 0; axis < 2; ++axis) {
            const uint8_t bit = 1 << axis;
            if ((drive->sizeMask & bit) && !(sizeMask & bit))
                spec.sizeDelta[axis] = drive->sizeDelta[axis];
            if (!(drive->sizeMask & bit) && (sizeMask & bit))
                drive->sizeDelta[axis] = spec.sizeDelta[axis];
        }
        drive->sizeMask = sizeMask;
    }
}

void RectTransformState::EndDrive()
{
    if (!drive)
        return;
    for (size_t axis = 0; axis < 2; ++axis)
        if (drive->sizeMask & (1 << axis))
            spec.sizeDelta[axis] = drive->sizeDelta[axis];
    if (drive->position)
        spec.anchoredPosition = drive->anchoredPosition;
    drive.reset();
}
} // namespace infernux::ui
