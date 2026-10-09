#pragma once

#include <function/resources/InxFileLoader/InxTextureLoader.hpp>

namespace infernux
{
// Rasterize at the requested longest edge, including upscaling. Unlimited (0)
// and oversized requests resolve to 8192, retaining the source aspect ratio.
inline constexpr int MaximumSvgDimension = 8192;
bool IsSvgSource(const unsigned char *data, size_t size);
bool SvgUsesFilters(const unsigned char *data, size_t size);
// Documentation preserves its authored viewport instead of filling the budget.
InxTextureData RasterizeSvg(const unsigned char *data, size_t size, int maxSize, bool preserveViewport = false);
} // namespace infernux
