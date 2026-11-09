#pragma once

#include "InxTexture.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernux
{

class InxResourceMeta;
class AssetSource;

class TextureDecoder final
{
  public:
    struct PngPlan
    {
        uint32_t width = 0;
        uint32_t height = 0;
        uint64_t bytes = 0;
        size_t resizeScratch = 0;
        size_t mipBytes = 0;
    };

    [[nodiscard]] static PngPlan PlanPng(AssetSource &source, const InxResourceMeta &metadata);
    [[nodiscard]] static std::pair<uint32_t, uint32_t> ReadPngDimensions(AssetSource &source);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData>
    Decode(const std::string &sourcePath, const InxResourceMeta &metadata, bool preserveSvgViewport = false);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData>
    Decode(AssetSource &source, const InxResourceMeta &metadata, bool preserveSvgViewport = false);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> DecodeMemory(const std::vector<unsigned char> &source,
                                                                            const InxResourceMeta &metadata,
                                                                            const std::string &sourceName,
                                                                            bool preserveSvgViewport = false);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData>
    CreateRgba8(const uint8_t *pixels, size_t byteCount, uint32_t width, uint32_t height, bool generateMipmaps);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> DecodeRgba8(const std::vector<unsigned char> &pixels,
                                                                           uint32_t width, uint32_t height,
                                                                           const InxResourceMeta &metadata);

  private:
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> DecodeBytes(std::string_view source,
                                                                           const InxResourceMeta &metadata,
                                                                           const std::string &sourceName,
                                                                           bool preserveSvgViewport);
};

} // namespace infernux
