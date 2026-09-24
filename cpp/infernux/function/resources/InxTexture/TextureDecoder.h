#pragma once

#include "InxTexture.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace infernux
{

class InxResourceMeta;

class TextureDecoder final
{
  public:
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> Decode(const std::string &sourcePath,
                                                                      const InxResourceMeta &metadata);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> DecodeMemory(const std::vector<unsigned char> &source,
                                                                            const InxResourceMeta &metadata,
                                                                            const std::string &sourceName);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData>
    CreateRgba8(const uint8_t *pixels, size_t byteCount, uint32_t width, uint32_t height, bool generateMipmaps);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> DecodeRgba8(const std::vector<unsigned char> &pixels,
                                                                           uint32_t width, uint32_t height,
                                                                           const InxResourceMeta &metadata);
};

} // namespace infernux
