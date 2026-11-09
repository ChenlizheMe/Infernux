#include "TextureDecoder.h"
#include "SignedDistanceFieldSource.h"
#include "SvgRasterizer.h"
#include "TextureProcessor.h"
#include "VectorFieldSource.h"
#include <core/log/InxLog.h>

#include <function/resources/AssetRegistry/AssetSource.h>
#include <function/resources/InxFileLoader/InxTextureLoader.hpp>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <stb_image.h>
#include <stb_image_resize2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernux
{
namespace
{
constexpr uint32_t MaximumDimension = 65'536;
constexpr uint64_t MaximumDecodedBytes = 1ULL << 30;

uint32_t ReadMaxSize(const InxResourceMeta &metadata)
{
    const int maxSize = metadata.HasKey("max_size") ? metadata.GetDataAs<int>("max_size") : 2048;
    if (maxSize < 0 || maxSize > static_cast<int>(MaximumDimension))
        throw std::invalid_argument("texture max_size is outside the supported range");
    return maxSize == 0 ? MaximumDimension : static_cast<uint32_t>(maxSize);
}

bool ReadGenerateMipmaps(const InxResourceMeta &metadata)
{
    return metadata.HasKey("generate_mipmaps") ? metadata.GetDataAs<bool>("generate_mipmaps") : true;
}

TextureCompression ReadCompression(const InxResourceMeta &metadata)
{
    const std::string value =
        metadata.HasKey("texture_compression") ? metadata.GetDataAs<std::string>("texture_compression") : "auto";
    if (value == "none")
        return TextureCompression::None;
    if (value == "auto")
        return TextureCompression::Automatic;
    if (value == "bc1")
        return TextureCompression::BC1;
    if (value == "bc3")
        return TextureCompression::BC3;
    if (value == "bc4")
        return TextureCompression::BC4;
    if (value == "bc5")
        return TextureCompression::BC5;
    throw std::invalid_argument("texture_compression has an unsupported value: " + value);
}

TextureCompressionQuality ReadCompressionQuality(const InxResourceMeta &metadata)
{
    const std::string value = metadata.HasKey("texture_compression_quality")
                                  ? metadata.GetDataAs<std::string>("texture_compression_quality")
                                  : "normal";
    if (value == "fast")
        return TextureCompressionQuality::Fast;
    if (value == "normal")
        return TextureCompressionQuality::Normal;
    if (value == "high")
        return TextureCompressionQuality::High;
    throw std::invalid_argument("texture_compression_quality has an unsupported value: " + value);
}

TextureTargetFormat ReadTargetFormat(const InxResourceMeta &metadata)
{
    const std::string value =
        metadata.HasKey("texture_format") ? metadata.GetDataAs<std::string>("texture_format") : "auto";
    if (value == "auto")
        return TextureTargetFormat::Automatic;
    if (value == "rgba8")
        return TextureTargetFormat::Rgba8;
    if (value == "rgba4444")
        return TextureTargetFormat::Rgba4UNorm;
    if (value == "rgba16_unorm")
        return TextureTargetFormat::Rgba16UNorm;
    if (value == "rgba16_float")
        return TextureTargetFormat::Rgba16Float;
    if (value == "rgba32_float")
        return TextureTargetFormat::Rgba32Float;
    throw std::invalid_argument("texture_format has an unsupported value: " + value);
}

bool ReadSrgb(const InxResourceMeta &metadata)
{
    return metadata.HasKey("srgb") ? metadata.GetDataAs<bool>("srgb") : true;
}

TextureSemantic ReadSemantic(const InxResourceMeta &metadata)
{
    const std::string type =
        metadata.HasKey("texture_type") ? metadata.GetDataAs<std::string>("texture_type") : "default";
    if (type == "default" || type.empty())
        return TextureSemantic::Color;
    if (type == "normal_map")
        return TextureSemantic::Normal;
    if (type == "data")
        return TextureSemantic::Data;
    if (type == "ui")
        return TextureSemantic::UserInterface;
    if (type == "sprite")
        return TextureSemantic::Sprite;
    if (type == "vector_field")
        return TextureSemantic::VectorField;
    if (type == "sdf")
        return TextureSemantic::SignedDistanceField;
    throw std::invalid_argument("texture_type has an unsupported value: " + type);
}

uint64_t LevelByteSize(uint32_t width, uint32_t height, TextureFormat format)
{
    const uint64_t bytesPerPixel = TextureFormatBytesPerTexel(format);
    if (bytesPerPixel == 0)
        throw std::invalid_argument("decoded texture requires an uncompressed working format");
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (pixels == 0 || pixels > MaximumDecodedBytes / bytesPerPixel)
        throw std::overflow_error("decoded texture exceeds the CPU artifact size limit");
    return pixels * bytesPerPixel;
}

const std::array<float, 65536> &Srgb16ToLinear()
{
    static const auto values = [] {
        std::array<float, 65536> table;
        for (size_t index = 0; index < table.size(); ++index) {
            const float value = static_cast<float>(index) / 65535.0f;
            table[index] = value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
        }
        return table;
    }();
    return values;
}

TextureMipLevel DescribeBaseLevel(uint32_t width, uint32_t height, TextureFormat format)
{
    const auto size = LevelByteSize(width, height, format);
    const auto rowPitch = static_cast<uint64_t>(width) * TextureFormatBytesPerTexel(format);
    return {width, height, 1, 0, size, rowPitch, size};
}

void CopyBaseLevel(TextureCpuData &texture, const TextureMipLevel &level, const void *pixels)
{
    const auto *begin = static_cast<const uint8_t *>(pixels);
    texture.bytes.assign(begin, begin + static_cast<size_t>(level.byteSize));
    texture.mipLevels.push_back(level);
}

std::pair<uint32_t, uint32_t> FitDimensions(uint32_t width, uint32_t height, uint32_t maxSize)
{
    const float scale = static_cast<float>(maxSize) / static_cast<float>((std::max)(width, height));
    if (scale >= 1.0f)
        return {width, height};
    return {(std::max)(1U, static_cast<uint32_t>(width * scale)),
            (std::max)(1U, static_cast<uint32_t>(height * scale))};
}

void SetRgba8BaseLevel(TextureCpuData &texture, const uint8_t *pixels, uint32_t width, uint32_t height,
                       uint32_t maxSize)
{
    const auto [targetWidth, targetHeight] = FitDimensions(width, height, maxSize);
    const auto level = DescribeBaseLevel(targetWidth, targetHeight, texture.format);
    if (targetWidth == width && targetHeight == height) {
        CopyBaseLevel(texture, level, pixels);
        return;
    }
    texture.bytes.resize(static_cast<size_t>(level.byteSize));
    const bool colorSrgb = TextureFormatIsSrgb(texture.format) && (texture.semantic == TextureSemantic::Color ||
                                                                   texture.semantic == TextureSemantic::UserInterface ||
                                                                   texture.semantic == TextureSemantic::Sprite);
    const auto resize = colorSrgb ? stbir_resize_uint8_srgb : stbir_resize_uint8_linear;
    if (!resize(pixels, width, height, 0, texture.bytes.data(), targetWidth, targetHeight, 0, STBIR_RGBA))
        throw std::runtime_error("failed to resize RGBA texture");
    texture.mipLevels.push_back(level);
}

void ExpandRgba16(const stbi_us *pixels, size_t byteCount, bool srgb, void *destination)
{
    // The working RGBA32F layout stores linear RGB and a linear alpha channel.
    // Write through bytes so the same conversion can fill either the final
    // payload or the typed scratch input required by the float resizer.
    const auto *linear = srgb ? Srgb16ToLinear().data() : nullptr;
    auto *output = static_cast<uint8_t *>(destination);
    using Pixel = std::array<float, 4>;
    constexpr size_t pixelBytes = 4 * sizeof(float);
    for (size_t offset = 0; offset < byteCount; offset += pixelBytes, pixels += 4) {
        const Pixel value{linear ? linear[pixels[0]] : static_cast<float>(pixels[0]) / 65535.0f,
                          linear ? linear[pixels[1]] : static_cast<float>(pixels[1]) / 65535.0f,
                          linear ? linear[pixels[2]] : static_cast<float>(pixels[2]) / 65535.0f,
                          static_cast<float>(pixels[3]) / 65535.0f};
        std::memcpy(output + offset, value.data(), pixelBytes);
    }
}

} // namespace

std::shared_ptr<const TextureCpuData> TextureDecoder::Decode(const std::string &sourcePath,
                                                             const InxResourceMeta &metadata, bool preserveSvgViewport)
{
    AssetSource source(sourcePath);
    return Decode(source, metadata, preserveSvgViewport);
}

std::shared_ptr<const TextureCpuData> TextureDecoder::Decode(AssetSource &source, const InxResourceMeta &metadata,
                                                             bool preserveSvgViewport)
{
    if (!source.Size() || source.Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("texture source is empty or exceeds decoder limits: " + source.Path());
    return DecodeBytes(source.Bytes(), metadata, source.Path(), preserveSvgViewport);
}

std::shared_ptr<const TextureCpuData> TextureDecoder::DecodeMemory(const std::vector<unsigned char> &source,
                                                                   const InxResourceMeta &metadata,
                                                                   const std::string &sourcePath,
                                                                   bool preserveSvgViewport)
{
    return DecodeBytes({reinterpret_cast<const char *>(source.data()), source.size()}, metadata, sourcePath,
                       preserveSvgViewport);
}

std::shared_ptr<const TextureCpuData> TextureDecoder::DecodeBytes(std::string_view source,
                                                                  const InxResourceMeta &metadata,
                                                                  const std::string &sourcePath,
                                                                  bool preserveSvgViewport)
{
    if (source.empty() || source.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("texture source is empty or exceeds decoder limits");
    const uint32_t maxSize = ReadMaxSize(metadata);
    const auto *bytes = reinterpret_cast<const unsigned char *>(source.data());

    std::string extension = FromFsPath(ToFsPath(sourcePath).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == ".inxvfield") {
        TextureCpuData volume = VectorFieldSource::Decode(source);
        const auto &base = volume.mipLevels.front();
        if (base.width > maxSize || base.height > maxSize || base.depth > maxSize)
            throw std::invalid_argument("vector field dimensions exceed texture max_size");
        return TextureProcessor::Process(
            std::move(volume), TextureProcessOptions{ReadGenerateMipmaps(metadata), ReadCompression(metadata),
                                                     ReadCompressionQuality(metadata), ReadTargetFormat(metadata)});
    }
    if (extension == ".inxsdf") {
        TextureCpuData volume = SignedDistanceFieldSource::Decode(source);
        const auto &base = volume.mipLevels.front();
        if (base.width > maxSize || base.height > maxSize || base.depth > maxSize)
            throw std::invalid_argument("signed distance field dimensions exceed texture max_size");
        return TextureProcessor::Process(std::move(volume), TextureProcessOptions{false, ReadCompression(metadata),
                                                                                  ReadCompressionQuality(metadata),
                                                                                  ReadTargetFormat(metadata)});
    }

    int sourceWidth = 0;
    int sourceHeight = 0;
    int sourceChannels = 0;

    TextureCpuData texture;
    texture.semantic = ReadSemantic(metadata);
    if (texture.semantic == TextureSemantic::VectorField)
        throw std::invalid_argument("VectorField textures must use the .inxvfield source format");
    if (texture.semantic == TextureSemantic::SignedDistanceField)
        throw std::invalid_argument("SignedDistanceField textures must use the .inxsdf source format");
    if (extension == ".svg" || IsSvgSource(bytes, source.size())) {
        if (SvgUsesFilters(bytes, source.size()))
            INXLOG_WARN("SVG filters are not supported and will be omitted: ", sourcePath,
                        ". Export filter effects as an embedded raster image for faithful rendering.");
        auto svg = RasterizeSvg(bytes, source.size(), static_cast<int>(maxSize), preserveSvgViewport);
        return DecodeRgba8(svg.pixels, svg.width, svg.height, metadata);
    }
    // PNM has an authoritative decoder for its sample maximum and byte order.
    const bool pnmSource = InxTextureLoader::IsPnmSource(bytes, source.size());
    const bool source16Bit = !pnmSource && stbi_is_16_bit_from_memory(bytes, static_cast<int>(source.size())) != 0;
    if (source16Bit || stbi_is_hdr_from_memory(bytes, static_cast<int>(source.size())) != 0) {
        texture.format = TextureFormat::Rgba32Float;
        std::unique_ptr<stbi_us, decltype(&stbi_image_free)> pixels16(nullptr, &stbi_image_free);
        std::unique_ptr<float, decltype(&stbi_image_free)> decodedHdr(nullptr, &stbi_image_free);
        if (source16Bit) {
            pixels16.reset(stbi_load_16_from_memory(bytes, static_cast<int>(source.size()), &sourceWidth, &sourceHeight,
                                                    &sourceChannels, STBI_rgb_alpha));
            if (!pixels16)
                throw std::runtime_error("failed to decode 16-bit texture: " + sourcePath);
        } else {
            decodedHdr.reset(stbi_loadf_from_memory(bytes, static_cast<int>(source.size()), &sourceWidth, &sourceHeight,
                                                    &sourceChannels, STBI_rgb_alpha));
            if (!decodedHdr)
                throw std::runtime_error("failed to decode HDR texture: " + sourcePath);
        }
        const auto [width, height] = FitDimensions(sourceWidth, sourceHeight, maxSize);
        const auto level = DescribeBaseLevel(width, height, texture.format);
        const bool srgb = source16Bit && ReadSrgb(metadata);
        if (width != static_cast<uint32_t>(sourceWidth) || height != static_cast<uint32_t>(sourceHeight)) {
            std::vector<float> precisePixels;
            const float *decoded = decodedHdr.get();
            if (pixels16) {
                const auto sourceSize = static_cast<size_t>(LevelByteSize(sourceWidth, sourceHeight, texture.format));
                precisePixels.resize(sourceSize / sizeof(float));
                ExpandRgba16(pixels16.get(), sourceSize, srgb, precisePixels.data());
                pixels16.reset();
                decoded = precisePixels.data();
            }
            std::vector<float> resized(static_cast<size_t>(level.byteSize / sizeof(float)));
            if (!stbir_resize_float_linear(decoded, sourceWidth, sourceHeight, 0, resized.data(),
                                           static_cast<int>(width), static_cast<int>(height), 0, STBIR_RGBA))
                throw std::runtime_error("failed to resize HDR texture: " + sourcePath);
            CopyBaseLevel(texture, level, resized.data());
        } else if (pixels16) {
            texture.bytes.resize(static_cast<size_t>(level.byteSize));
            ExpandRgba16(pixels16.get(), texture.bytes.size(), srgb, texture.bytes.data());
            texture.mipLevels.push_back(level);
        } else {
            CopyBaseLevel(texture, level, decodedHdr.get());
        }
    } else {
        texture.format = ReadSrgb(metadata) ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
        stbi_uc *decoded = nullptr;
        std::vector<unsigned char> pnmPixels;
        if (pnmSource) {
            InxTextureData pnm = InxTextureLoader::LoadFromMemory(bytes, source.size(), sourcePath);
            if (!pnm.IsValid())
                throw std::runtime_error("failed to decode texture: " + sourcePath);
            sourceWidth = pnm.width;
            sourceHeight = pnm.height;
            pnmPixels = std::move(pnm.pixels);
        } else {
            decoded = stbi_load_from_memory(bytes, static_cast<int>(source.size()), &sourceWidth, &sourceHeight,
                                            &sourceChannels, STBI_rgb_alpha);
            if (!decoded)
                throw std::runtime_error("failed to decode texture: " + sourcePath);
        }
        auto release = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>(decoded, &stbi_image_free);
        SetRgba8BaseLevel(texture, decoded ? decoded : pnmPixels.data(), sourceWidth, sourceHeight, maxSize);
    }

    return TextureProcessor::Process(
        std::move(texture), TextureProcessOptions{ReadGenerateMipmaps(metadata), ReadCompression(metadata),
                                                  ReadCompressionQuality(metadata), ReadTargetFormat(metadata)});
}

std::shared_ptr<const TextureCpuData> TextureDecoder::DecodeRgba8(const std::vector<unsigned char> &pixels,
                                                                  uint32_t width, uint32_t height,
                                                                  const InxResourceMeta &metadata)
{
    if (!width || !height || pixels.size() != LevelByteSize(width, height, TextureFormat::Rgba8UNorm))
        throw std::invalid_argument("embedded RGBA texture dimensions do not match its payload");
    TextureCpuData texture;
    texture.dimension = TextureDimension::Texture2D;
    texture.semantic = ReadSemantic(metadata);
    texture.format = ReadSrgb(metadata) ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
    SetRgba8BaseLevel(texture, pixels.data(), width, height, ReadMaxSize(metadata));
    return TextureProcessor::Process(
        std::move(texture), TextureProcessOptions{ReadGenerateMipmaps(metadata), ReadCompression(metadata),
                                                  ReadCompressionQuality(metadata), ReadTargetFormat(metadata)});
}

std::shared_ptr<const TextureCpuData> TextureDecoder::CreateRgba8(const uint8_t *pixels, size_t byteCount,
                                                                  uint32_t width, uint32_t height, bool generateMipmaps)
{
    if (!pixels)
        throw std::invalid_argument("RGBA8 texture payload has no pixels");
    const uint64_t expectedSize = LevelByteSize(width, height, TextureFormat::Rgba8UNorm);
    if (expectedSize != byteCount)
        throw std::invalid_argument("RGBA8 texture payload byte count does not match its dimensions");

    TextureCpuData texture;
    texture.semantic = TextureSemantic::Data;
    texture.format = TextureFormat::Rgba8UNorm;
    CopyBaseLevel(texture, DescribeBaseLevel(width, height, texture.format), pixels);
    return TextureProcessor::Process(
        std::move(texture),
        TextureProcessOptions{generateMipmaps, TextureCompression::None, TextureCompressionQuality::Normal});
}

} // namespace infernux
