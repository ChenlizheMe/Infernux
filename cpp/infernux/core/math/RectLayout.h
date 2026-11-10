#pragma once

#include "RectLayoutTypes.h"
#include <cstddef>

namespace infernux::ui
{

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline RectLayoutGeometry
CompileRectLayout(const RectLayoutSpec &spec)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    RectLayoutGeometry result;
    for (std::size_t axis = 0; axis < 2; ++axis) {
        // Parent origin participates before endpoint subtraction. A span-first
        // rewrite loses the reference rounding for off-center parent pivots.
        const float origin = -spec.parentSize[axis] * spec.parentPivot[axis];
        const float minimum = origin + spec.parentSize[axis] * spec.anchorMin[axis];
        const float maximum = origin + spec.parentSize[axis] * spec.anchorMax[axis];
        const float size = spec.sizeDelta[axis] + (maximum - minimum);
        result.rect[axis] = -spec.pivot[axis] * size;
        result.rect[axis + 2] = size;
        result.offsetMin[axis] = spec.anchoredPosition[axis] - spec.sizeDelta[axis] * spec.pivot[axis];
        result.offsetMax[axis] = spec.anchoredPosition[axis] + spec.sizeDelta[axis] * (1.0f - spec.pivot[axis]);
    }
    const float left = result.rect[0], bottom = result.rect[1];
    const float right = left + result.rect[2], top = bottom + result.rect[3];
    result.localCorners = {{{left, bottom, 0}, {left, top, 0}, {right, top, 0}, {right, bottom, 0}}};
    return result;
}

} // namespace infernux::ui
