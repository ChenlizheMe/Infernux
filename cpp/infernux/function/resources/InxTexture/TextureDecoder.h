#pragma once

#include "InxTexture.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{

class InxResourceMeta;
class AssetSource;
struct InxTextureData;

class TextureDecoder final
{
  public:
    struct SourceInfo
    {
        uint32_t width = 0;
        uint32_t height = 0;
        uint8_t channels = 0;
    };

    struct SourcePreviewPlan
    {
        enum class Codec
        {
            Png,
            Jpeg,
            Raster
        };

        SourceInfo source;
        uint32_t previewWidth = 0;
        uint32_t previewHeight = 0;
        uint64_t bytes = 0;
        Codec codec = Codec::Png;
        TextureFormat format = TextureFormat::Rgba8UNorm;
    };

    [[nodiscard]] static bool IsRasterSource(const std::string &path);
    [[nodiscard]] static uint64_t PlanSourceStorage(AssetSource &source, const InxResourceMeta &metadata);
    [[nodiscard]] static SourceInfo ReadSequentialSourceInfo(AssetSource &source);
    [[nodiscard]] static SourcePreviewPlan PlanSourcePreview(AssetSource &source, uint32_t maxSize, bool document);
    [[nodiscard]] static InxTextureData DecodeSourcePreview(AssetSource &source, const SourcePreviewPlan &plan,
                                                            const std::function<bool()> &retired);
    static void ResizeRgba8Preview(InxTextureData &image, uint32_t maxSize);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData>
    TakePreview(InxTextureData image, bool srgb, TextureFormat format = TextureFormat::Rgba8UNorm);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData>
    Decode(const std::string &sourcePath, const InxResourceMeta &metadata, bool preserveSvgViewport = false);
    [[nodiscard]] static std::shared_ptr<const TextureCpuData> Decode(AssetSource &source,
                                                                      const InxResourceMeta &metadata,
                                                                      bool preserveSvgViewport = false,
                                                                      int *sequentialChannels = nullptr);
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
    [[nodiscard]] static std::shared_ptr<const TextureCpuData>
    DecodeBytes(std::string_view source, const InxResourceMeta &metadata, const std::string &sourceName,
                bool preserveSvgViewport, size_t plannedMipBytes = 0);
};

namespace detail
{
struct JpegInfo
{
    TextureDecoder::SourceInfo source;
    uint64_t decodeBytes = 0;
};

[[nodiscard]] JpegInfo ReadJpegInfo(AssetSource &source);
[[nodiscard]] JpegInfo ReadJpegInfo(std::string_view bytes);

struct RasterInfo
{
    TextureDecoder::SourceInfo source;
    uint64_t decodeBytes = 0;
    uint64_t previewDecodeBytes = 0;
    bool sixteenBit = false;
    bool hdr = false;
};

[[nodiscard]] RasterInfo ReadRasterInfo(AssetSource &source);
[[nodiscard]] RasterInfo ReadRasterInfo(std::string_view bytes);
[[nodiscard]] TextureDecoder::SourceInfo ReadPnmInfo(AssetSource &source);
[[nodiscard]] TextureDecoder::SourceInfo ReadPnmInfo(std::string_view bytes);
} // namespace detail

} // namespace infernux
