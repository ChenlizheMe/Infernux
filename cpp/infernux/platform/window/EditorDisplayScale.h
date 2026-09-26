#pragma once

#include <cmath>
#include <stdexcept>

namespace infernux
{

/// Editor geometry and pointer positions use SDL window coordinates. SDL's
/// display scale describes physical pixels per authored UI unit; pixel density
/// describes physical pixels per window unit. They are different on Retina and
/// Wayland, but pixel density is normally 1 on Windows, including at 200% DPI.
inline float ResolveEditorDisplayScale(float displayScale, float pixelDensity)
{
    if (!std::isfinite(displayScale) || displayScale <= 0.0f || !std::isfinite(pixelDensity) || pixelDensity <= 0.0f)
        throw std::runtime_error("SDL reported an invalid editor display scale or pixel density");
    return displayScale / pixelDensity;
}

} // namespace infernux
