#pragma once

#include "InxTexture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <glm/gtc/packing.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infernux::texture_pixels
{
namespace detail
{
inline std::array<uint8_t, 4> DecodeRgb565(uint16_t packed)
{
    const uint8_t red = static_cast<uint8_t>((packed >> 11u) & 0x1fu);
    const uint8_t green = static_cast<uint8_t>((packed >> 5u) & 0x3fu);
    const uint8_t blue = static_cast<uint8_t>(packed & 0x1fu);
    return {static_cast<uint8_t>((red << 3u) | (red >> 2u)), static_cast<uint8_t>((green << 2u) | (green >> 4u)),
            static_cast<uint8_t>((blue << 3u) | (blue >> 2u)), 255u};
}

inline std::array<std::array<uint8_t, 4>, 4> DecodeBcColorTable(const uint8_t *block, bool allowTransparent)
{
    const uint16_t endpoint0 = static_cast<uint16_t>(block[0]) | static_cast<uint16_t>(block[1]) << 8u;
    const uint16_t endpoint1 = static_cast<uint16_t>(block[2]) | static_cast<uint16_t>(block[3]) << 8u;
    std::array<std::array<uint8_t, 4>, 4> table{};
    table[0] = DecodeRgb565(endpoint0);
    table[1] = DecodeRgb565(endpoint1);
    if (!allowTransparent || endpoint0 > endpoint1) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            table[2][channel] = static_cast<uint8_t>((2u * table[0][channel] + table[1][channel]) / 3u);
            table[3][channel] = static_cast<uint8_t>((table[0][channel] + 2u * table[1][channel]) / 3u);
        }
        table[2][3] = 255u;
        table[3][3] = 255u;
    } else {
        for (uint32_t channel = 0; channel < 3; ++channel)
            table[2][channel] = static_cast<uint8_t>((table[0][channel] + table[1][channel]) / 2u);
        table[2][3] = 255u;
        table[3] = {0u, 0u, 0u, 0u};
    }
    return table;
}

inline std::array<uint8_t, 8> DecodeBc3AlphaTable(uint8_t endpoint0, uint8_t endpoint1)
{
    std::array<uint8_t, 8> table{endpoint0, endpoint1};
    if (endpoint0 > endpoint1) {
        for (uint32_t index = 1; index <= 6; ++index)
            table[index + 1] = static_cast<uint8_t>(((7u - index) * endpoint0 + index * endpoint1) / 7u);
    } else {
        for (uint32_t index = 1; index <= 4; ++index)
            table[index + 1] = static_cast<uint8_t>(((5u - index) * endpoint0 + index * endpoint1) / 5u);
        table[6] = 0u;
        table[7] = 255u;
    }
    return table;
}

inline uint64_t DecodeBcScalarIndices(const uint8_t *block)
{
    uint64_t indices = 0;
    for (uint32_t byte = 0; byte < 6; ++byte)
        indices |= static_cast<uint64_t>(block[2 + byte]) << (8u * byte);
    return indices;
}

inline void DecodeBcMip(const TextureCpuData &source, const TextureMipLevel &mip, std::vector<uint8_t> &pixels)
{
    const bool bc1 = source.format == TextureFormat::BC1RgbaUNorm || source.format == TextureFormat::BC1RgbaSrgb;
    const bool bc3 = source.format == TextureFormat::BC3UNorm || source.format == TextureFormat::BC3Srgb;
    const bool bc4 = source.format == TextureFormat::BC4UNorm;
    const bool bc5 = source.format == TextureFormat::BC5UNorm;
    if (!bc1 && !bc3 && !bc4 && !bc5)
        throw std::invalid_argument("Unsupported block-compressed texture pixel format");
    const uint32_t blockBytes = (bc1 || bc4) ? 8u : 16u;
    const uint32_t blockColumns = std::max(1u, (mip.width + 3u) / 4u);
    const uint32_t blockRows = std::max(1u, (mip.height + 3u) / 4u);
    const auto blockRowBytes = static_cast<uint64_t>(blockColumns) * blockBytes;
    if (mip.width == 0 || mip.height == 0 || mip.depth != 1 || mip.rowPitch < blockRowBytes ||
        mip.byteOffset > source.bytes.size() || mip.byteSize > source.bytes.size() - mip.byteOffset ||
        mip.byteSize < static_cast<uint64_t>(blockRows - 1u) * mip.rowPitch + blockRowBytes)
        throw std::invalid_argument("Invalid block-compressed texture pixel layout");
    const auto rowPitch = static_cast<size_t>(mip.width) * 4u;
    for (uint32_t blockY = 0; blockY < blockRows; ++blockY) {
        for (uint32_t blockX = 0; blockX < blockColumns; ++blockX) {
            const uint8_t *block = source.bytes.data() + mip.byteOffset + static_cast<uint64_t>(blockY) * mip.rowPitch +
                                   static_cast<uint64_t>(blockX) * blockBytes;
            std::array<std::array<uint8_t, 4>, 4> colors{};
            uint32_t colorIndices = 0;
            std::array<uint8_t, 8> alphaOrRed{};
            std::array<uint8_t, 8> green{};
            uint64_t alphaOrRedIndices = 0;
            uint64_t greenIndices = 0;
            if (bc1 || bc3) {
                const uint8_t *colorBlock = block + (bc3 ? 8u : 0u);
                colors = DecodeBcColorTable(colorBlock, bc1);
                colorIndices = static_cast<uint32_t>(colorBlock[4]) | static_cast<uint32_t>(colorBlock[5]) << 8u |
                               static_cast<uint32_t>(colorBlock[6]) << 16u |
                               static_cast<uint32_t>(colorBlock[7]) << 24u;
                if (bc3) {
                    alphaOrRed = DecodeBc3AlphaTable(block[0], block[1]);
                    alphaOrRedIndices = DecodeBcScalarIndices(block);
                }
            } else {
                alphaOrRed = DecodeBc3AlphaTable(block[0], block[1]);
                alphaOrRedIndices = DecodeBcScalarIndices(block);
                if (bc5) {
                    green = DecodeBc3AlphaTable(block[8], block[9]);
                    greenIndices = DecodeBcScalarIndices(block + 8u);
                }
            }

            for (uint32_t pixel = 0; pixel < 16; ++pixel) {
                const uint32_t x = blockX * 4u + pixel % 4u;
                const uint32_t y = blockY * 4u + pixel / 4u;
                if (x >= mip.width || y >= mip.height)
                    continue;
                std::array<uint8_t, 4> rgba{};
                if (bc1 || bc3) {
                    rgba = colors[(colorIndices >> (pixel * 2u)) & 0x3u];
                    if (bc3)
                        rgba[3] = alphaOrRed[(alphaOrRedIndices >> (pixel * 3u)) & 0x7u];
                } else {
                    const uint8_t red = alphaOrRed[(alphaOrRedIndices >> (pixel * 3u)) & 0x7u];
                    const uint8_t greenValue = bc5 ? green[(greenIndices >> (pixel * 3u)) & 0x7u] : red;
                    rgba = {red, greenValue, static_cast<uint8_t>(bc5 ? 255u : red), 255u};
                }
                std::memcpy(pixels.data() + static_cast<size_t>(y) * rowPitch + x * 4u, rgba.data(), rgba.size());
            }
        }
    }
}

inline uint8_t QuantizeRgba8(float value)
{
    const float scaled = (std::isnan(value) ? 0.0f : std::clamp(value, 0.0f, 1.0f)) * 255.0f;
    const auto integral = static_cast<uint8_t>(scaled);
    return static_cast<uint8_t>(integral + (scaled - integral >= 0.5f));
}
} // namespace detail

inline std::vector<uint8_t> BaseMipRgba8(const TextureCpuData &source)
{
    if (source.dimension != TextureDimension::Texture2D || !source.IsValid())
        throw std::invalid_argument("Texture pixel access requires a Texture2D with pixels");
    const auto &mip = source.mipLevels.front();
    const auto rowBytes = static_cast<uint64_t>(mip.width) * 4u;
    if (rowBytes * mip.height > (std::numeric_limits<size_t>::max)())
        throw std::length_error("RGBA8 pixel output is too large");
    std::vector<uint8_t> pixels(static_cast<size_t>(rowBytes * mip.height));
    if (TextureFormatIsBlockCompressed(source.format)) {
        detail::DecodeBcMip(source, mip, pixels);
        return pixels;
    }
    const auto texelBytes = TextureFormatBytesPerTexel(source.format);
    for (uint32_t y = 0; y < mip.height; ++y) {
        const auto *row = source.bytes.data() + mip.byteOffset + y * mip.rowPitch;
        auto *output = pixels.data() + y * rowBytes;
        if (source.format == TextureFormat::Rgba8UNorm || source.format == TextureFormat::Rgba8Srgb) {
            std::memcpy(output, row, static_cast<size_t>(rowBytes));
            continue;
        }
        for (uint32_t x = 0; x < mip.width; ++x) {
            const auto *pixel = row + x * texelBytes;
            for (uint32_t channel = 0; channel < 4; ++channel) {
                float value;
                if (source.format == TextureFormat::Rgba4UNormPack16) {
                    uint16_t packed;
                    std::memcpy(&packed, pixel, sizeof(packed));
                    value = ((packed >> ((3 - channel) * 4)) & 15) / 15.0f;
                } else if (source.format == TextureFormat::Rgba32Float) {
                    std::memcpy(&value, pixel + channel * 4, sizeof(value));
                } else {
                    uint16_t packed;
                    std::memcpy(&packed, pixel + channel * 2, sizeof(packed));
                    value =
                        source.format == TextureFormat::Rgba16Float ? glm::unpackHalf1x16(packed) : packed / 65535.0f;
                }
                output[x * 4 + channel] = detail::QuantizeRgba8(value);
            }
        }
    }
    return pixels;
}
} // namespace infernux::texture_pixels
