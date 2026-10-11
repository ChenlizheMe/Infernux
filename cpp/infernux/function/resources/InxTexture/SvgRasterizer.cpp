#include "SvgRasterizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <lunasvg.h>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace infernux
{
bool SvgUsesFilters(const unsigned char *data, size_t size)
{
    if (!data)
        return false;
    const std::string_view text(reinterpret_cast<const char *>(data), size);
    size_t pos = 0;
    while ((pos = text.find('<', pos)) != std::string_view::npos) {
        if (text.substr(pos, 4) == "<!--" || text.substr(pos, 9) == "<![CDATA[") {
            const auto end = text.find(text.substr(pos, 4) == "<!--" ? "-->" : "]]>", pos);
            if (end == std::string_view::npos)
                return false;
            pos = end + 3;
            continue;
        }
        ++pos;
        const auto end = text.find_first_of(" \t\r\n/>", pos);
        auto tag = text.substr(pos, end == std::string_view::npos ? text.size() - pos : end - pos);
        if (const auto colon = tag.find(':'); colon != std::string_view::npos)
            tag.remove_prefix(colon + 1);
        if (tag == "filter")
            return true;
    }
    return false;
}

bool IsSvgSource(const unsigned char *data, size_t size)
{
    if (!data || size == 0)
        return false;
    std::string_view text(reinterpret_cast<const char *>(data), size);
    if (text.substr(0, 3) == "\xef\xbb\xbf")
        text.remove_prefix(3);
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos)
        return false;
    text.remove_prefix(start);
    return text.substr(0, 4) == "<svg" || text.substr(0, 5) == "<?xml" || text.substr(0, 4) == "<!--" ||
           text.substr(0, 9) == "<!DOCTYPE";
}

InxTextureData RasterizeSvg(const unsigned char *data, size_t size, int maxSize, bool preserveViewport)
{
    if (!data || size == 0 || maxSize < 0)
        throw std::invalid_argument("failed to decode SVG: empty source or invalid raster size");
    // Import and preview workers share LunaSVG's font-face cache.
    static std::mutex rasterMutex;
    std::lock_guard<std::mutex> lock(rasterMutex);
    // XML permits a UTF-8 BOM; LunaSVG expects the XML text after that marker.
    if (size >= 3 && data[0] == 0xef && data[1] == 0xbb && data[2] == 0xbf) {
        data += 3;
        size -= 3;
    }
    const auto document = lunasvg::Document::loadFromData(reinterpret_cast<const char *>(data), size);
    if (!document)
        throw std::runtime_error("failed to decode SVG: invalid document");
    const double width = document->width();
    const double height = document->height();
    if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0)
        throw std::runtime_error("failed to decode SVG: width, height or viewBox must define a positive viewport");

    const int edge = maxSize == 0 ? MaximumSvgDimension : (std::min)(maxSize, MaximumSvgDimension);
    const double scale =
        preserveViewport ? (std::min)(1.0, edge / (std::max)(width, height)) : edge / (std::max)(width, height);
    InxTextureData result;
    result.width = std::clamp(static_cast<int>(std::lround(width * scale)), 1, edge);
    result.height = std::clamp(static_cast<int>(std::lround(height * scale)), 1, edge);
    auto bitmap = document->renderToBitmap(result.width, result.height);
    if (bitmap.isNull())
        throw std::runtime_error("failed to decode SVG: rasterization failed");
    // The pipeline requires straight RGBA, not LunaSVG's premultiplied ARGB.
    bitmap.convertToRGBA();
    const size_t rowBytes = static_cast<size_t>(result.width) * 4;
    result.pixels.resize(rowBytes * result.height);
    for (int y = 0; y < result.height; ++y)
        std::memcpy(result.pixels.data() + y * rowBytes, bitmap.data() + y * bitmap.stride(), rowBytes);
    return result;
}
} // namespace infernux
