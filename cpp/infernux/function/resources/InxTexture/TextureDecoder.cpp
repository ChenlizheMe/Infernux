#include "TextureDecoder.h"
#include "SignedDistanceFieldSource.h"
#include "SvgRasterizer.h"
#include "TextureProcessor.h"
#include "VectorFieldSource.h"
#include <core/log/InxLog.h>

#include <function/resources/InxFileLoader/InxTextureLoader.hpp>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <stb_image.h>
#include <stb_image_resize2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
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

std::vector<unsigned char> ReadSourceBytes(const std::string &path)
{
    std::ifstream file(ToFsPath(path), std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("failed to open texture source: " + path);
    const auto size = file.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("texture source is empty or exceeds decoder limits: " + path);
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!file)
        throw std::runtime_error("failed to read texture source: " + path);
    return bytes;
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

void AppendLevel(TextureCpuData &texture, uint32_t width, uint32_t height, const void *pixels, uint64_t byteSize)
{
    if (!pixels || byteSize != LevelByteSize(width, height, texture.format) ||
        byteSize > MaximumDecodedBytes - texture.bytes.size())
        throw std::invalid_argument("decoded texture level has an invalid payload");
    TextureMipLevel level;
    level.width = width;
    level.height = height;
    level.depth = 1;
    level.byteOffset = texture.bytes.size();
    level.byteSize = byteSize;
    level.rowPitch = static_cast<uint64_t>(width) * TextureFormatBytesPerTexel(texture.format);
    level.slicePitch = level.rowPitch * height;
    const auto *begin = static_cast<const uint8_t *>(pixels);
    texture.bytes.insert(texture.bytes.end(), begin, begin + static_cast<size_t>(byteSize));
    texture.mipLevels.push_back(level);
}

} // namespace

std::shared_ptr<const TextureCpuData> TextureDecoder::Decode(const std::string &sourcePath,
                                                             const InxResourceMeta &metadata, bool preserveSvgViewport)
{
    const auto source = ReadSourceBytes(sourcePath);
    return DecodeMemory(source, metadata, sourcePath, preserveSvgViewport);
}

std::shared_ptr<const TextureCpuData> TextureDecoder::DecodeMemory(const std::vector<unsigned char> &source,
                                                                   const InxResourceMeta &metadata,
                                                                   const std::string &sourcePath,
                                                                   bool preserveSvgViewport)
{
    if (source.empty() || source.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("texture source is empty or exceeds decoder limits");
    const uint32_t maxSize = ReadMaxSize(metadata);

    std::string extension = FromFsPath(ToFsPath(sourcePath).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (extension == ".inxvfield") {
        TextureCpuData volume =
            VectorFieldSource::Decode(std::string_view(reinterpret_cast<const char *>(source.data()), source.size()));
        const auto &base = volume.mipLevels.front();
        if (base.width > maxSize || base.height > maxSize || base.depth > maxSize)
            throw std::invalid_argument("vector field dimensions exceed texture max_size");
        return TextureProcessor::Process(
            std::move(volume), TextureProcessOptions{ReadGenerateMipmaps(metadata), ReadCompression(metadata),
                                                     ReadCompressionQuality(metadata), ReadTargetFormat(metadata)});
    }
    if (extension == ".inxsdf") {
        TextureCpuData volume = SignedDistanceFieldSource::Decode(
            std::string_view(reinterpret_cast<const char *>(source.data()), source.size()));
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

    auto texture = std::make_shared<TextureCpuData>();
    texture->dimension = TextureDimension::Texture2D;
    texture->semantic = ReadSemantic(metadata);
    if (texture->semantic == TextureSemantic::VectorField)
        throw std::invalid_argument("VectorField textures must use the .inxvfield source format");
    if (texture->semantic == TextureSemantic::SignedDistanceField)
        throw std::invalid_argument("SignedDistanceField textures must use the .inxsdf source format");
    if (extension == ".svg" || IsSvgSource(source.data(), source.size())) {
        if (SvgUsesFilters(source.data(), source.size()))
            INXLOG_WARN("SVG filters are not supported and will be omitted: ", sourcePath,
                        ". Export filter effects as an embedded raster image for faithful rendering.");
        auto svg = RasterizeSvg(source.data(), source.size(), static_cast<int>(maxSize), preserveSvgViewport);
        return DecodeRgba8(svg.pixels, svg.width, svg.height, metadata);
    }
    const bool preserve16BitUi = texture->semantic == TextureSemantic::UserInterface &&
                                 ReadCompression(metadata) == TextureCompression::None &&
                                 ReadTargetFormat(metadata) == TextureTargetFormat::Automatic &&
                                 stbi_is_16_bit_from_memory(source.data(), static_cast<int>(source.size())) != 0;
    if (preserve16BitUi || stbi_is_hdr_from_memory(source.data(), static_cast<int>(source.size())) != 0) {
        texture->format = TextureFormat::Rgba32Float;
        std::vector<float> precisePixels;
        std::unique_ptr<float, decltype(&stbi_image_free)> decodedHdr(nullptr, &stbi_image_free);
        if (preserve16BitUi) {
            std::unique_ptr<stbi_us, decltype(&stbi_image_free)> pixels(
                stbi_load_16_from_memory(source.data(), static_cast<int>(source.size()), &sourceWidth, &sourceHeight,
                                         &sourceChannels, STBI_rgb_alpha),
                &stbi_image_free);
            if (!pixels)
                throw std::runtime_error("failed to decode 16-bit UI texture: " + sourcePath);
            precisePixels.resize(
                static_cast<size_t>(LevelByteSize(sourceWidth, sourceHeight, texture->format) / sizeof(float)));
            // Vulkan has no RGBA16 sRGB format. Preserve all source precision
            // in linear floats instead of truncating to 8-bit or treating sRGB
            // values as linear (which would brighten documentation images).
            const bool srgb = ReadSrgb(metadata);
            for (size_t index = 0; index < precisePixels.size(); ++index) {
                float value = static_cast<float>(pixels.get()[index]) / 65535.0f;
                if (srgb && index % 4 != 3)
                    value = value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
                precisePixels[index] = value;
            }
        } else {
            decodedHdr.reset(stbi_loadf_from_memory(source.data(), static_cast<int>(source.size()), &sourceWidth,
                                                    &sourceHeight, &sourceChannels, STBI_rgb_alpha));
        }
        const float *decoded = preserve16BitUi ? precisePixels.data() : decodedHdr.get();
        if (!decoded)
            throw std::runtime_error("failed to decode HDR texture: " + sourcePath);
        const float scale = static_cast<float>(maxSize) / static_cast<float>((std::max)(sourceWidth, sourceHeight));
        const uint32_t width = scale < 1.0f ? (std::max)(1U, static_cast<uint32_t>(sourceWidth * scale))
                                            : static_cast<uint32_t>(sourceWidth);
        const uint32_t height = scale < 1.0f ? (std::max)(1U, static_cast<uint32_t>(sourceHeight * scale))
                                             : static_cast<uint32_t>(sourceHeight);
        if (width != static_cast<uint32_t>(sourceWidth) || height != static_cast<uint32_t>(sourceHeight)) {
            std::vector<float> resized(static_cast<size_t>(width) * height * 4);
            if (!stbir_resize_float_linear(decoded, sourceWidth, sourceHeight, 0, resized.data(),
                                           static_cast<int>(width), static_cast<int>(height), 0, STBIR_RGBA))
                throw std::runtime_error("failed to resize HDR texture: " + sourcePath);
            AppendLevel(*texture, width, height, resized.data(), resized.size() * sizeof(float));
        } else {
            AppendLevel(*texture, width, height, decoded, static_cast<uint64_t>(width) * height * 4 * sizeof(float));
        }
    } else {
        texture->format = ReadSrgb(metadata) ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
        stbi_uc *decoded = nullptr;
        std::vector<unsigned char> pnmPixels;
        if (InxTextureLoader::IsPnmSource(source.data(), source.size())) {
            InxTextureData pnm = InxTextureLoader::LoadFromMemory(source.data(), source.size(), sourcePath);
            if (!pnm.IsValid())
                throw std::runtime_error("failed to decode texture: " + sourcePath);
            sourceWidth = pnm.width;
            sourceHeight = pnm.height;
            pnmPixels = std::move(pnm.pixels);
        } else {
            decoded = stbi_load_from_memory(source.data(), static_cast<int>(source.size()), &sourceWidth, &sourceHeight,
                                            &sourceChannels, STBI_rgb_alpha);
            if (!decoded)
                throw std::runtime_error("failed to decode texture: " + sourcePath);
        }
        auto release = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>(decoded, &stbi_image_free);
        const unsigned char *base = decoded ? decoded : pnmPixels.data();
        const float scale = static_cast<float>(maxSize) / static_cast<float>((std::max)(sourceWidth, sourceHeight));
        const uint32_t width = scale < 1.0f ? (std::max)(1U, static_cast<uint32_t>(sourceWidth * scale))
                                            : static_cast<uint32_t>(sourceWidth);
        const uint32_t height = scale < 1.0f ? (std::max)(1U, static_cast<uint32_t>(sourceHeight * scale))
                                             : static_cast<uint32_t>(sourceHeight);
        if (width != static_cast<uint32_t>(sourceWidth) || height != static_cast<uint32_t>(sourceHeight)) {
            std::vector<uint8_t> resized(static_cast<size_t>(width) * height * 4);
            const bool colorSrgb =
                TextureFormatIsSrgb(texture->format) &&
                (texture->semantic == TextureSemantic::Color || texture->semantic == TextureSemantic::UserInterface ||
                 texture->semantic == TextureSemantic::Sprite);
            const bool resizedOk = colorSrgb
                                       ? stbir_resize_uint8_srgb(base, sourceWidth, sourceHeight, 0, resized.data(),
                                                                 static_cast<int>(width), static_cast<int>(height), 0,
                                                                 STBIR_RGBA) != nullptr
                                       : stbir_resize_uint8_linear(base, sourceWidth, sourceHeight, 0, resized.data(),
                                                                   static_cast<int>(width), static_cast<int>(height), 0,
                                                                   STBIR_RGBA) != nullptr;
            if (!resizedOk)
                throw std::runtime_error("failed to resize texture: " + sourcePath);
            AppendLevel(*texture, width, height, resized.data(), resized.size());
        } else {
            AppendLevel(*texture, width, height, base, static_cast<uint64_t>(width) * height * 4);
        }
    }

    return TextureProcessor::Process(
        std::move(*texture), TextureProcessOptions{ReadGenerateMipmaps(metadata), ReadCompression(metadata),
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
    const float scale = static_cast<float>(ReadMaxSize(metadata)) / static_cast<float>((std::max)(width, height));
    if (scale < 1.0f) {
        const auto targetWidth = (std::max)(1U, static_cast<uint32_t>(width * scale));
        const auto targetHeight = (std::max)(1U, static_cast<uint32_t>(height * scale));
        std::vector<unsigned char> resized(static_cast<size_t>(targetWidth) * targetHeight * 4);
        const bool colorSrgb =
            TextureFormatIsSrgb(texture.format) &&
            (texture.semantic == TextureSemantic::Color || texture.semantic == TextureSemantic::UserInterface ||
             texture.semantic == TextureSemantic::Sprite);
        const auto result = colorSrgb ? stbir_resize_uint8_srgb(pixels.data(), width, height, 0, resized.data(),
                                                                targetWidth, targetHeight, 0, STBIR_RGBA)
                                      : stbir_resize_uint8_linear(pixels.data(), width, height, 0, resized.data(),
                                                                  targetWidth, targetHeight, 0, STBIR_RGBA);
        if (!result)
            throw std::runtime_error("failed to resize embedded RGBA texture");
        AppendLevel(texture, targetWidth, targetHeight, resized.data(), resized.size());
    } else {
        AppendLevel(texture, width, height, pixels.data(), pixels.size());
    }
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

    auto texture = std::make_shared<TextureCpuData>();
    texture->dimension = TextureDimension::Texture2D;
    texture->semantic = TextureSemantic::Data;
    texture->format = TextureFormat::Rgba8UNorm;
    AppendLevel(*texture, width, height, pixels, expectedSize);
    return TextureProcessor::Process(
        std::move(*texture),
        TextureProcessOptions{generateMipmaps, TextureCompression::None, TextureCompressionQuality::Normal});
}

} // namespace infernux
