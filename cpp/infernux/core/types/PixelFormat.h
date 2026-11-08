#pragma once

#include <cstddef>
#include <cstdint>

namespace infernux::rhi
{

enum class PixelFormat : uint8_t
{
    Undefined = 0,
    R8UNorm,
    RG8UNorm,
    RGBA8UNorm,
    RGBA8Srgb,
    BGRA8UNorm,
    BGRA8Srgb,
    R16SFloat,
    RG16SFloat,
    RGBA16SFloat,
    RGBA16UNorm,
    R32SFloat,
    RG32UInt,
    RGBA32SFloat,
    RGB10A2UNorm,
    RGBA4UNormPack16,
    BC1RgbaUNorm,
    BC1RgbaSrgb,
    BC3UNorm,
    BC3Srgb,
    BC4UNorm,
    BC5UNorm,
    BC6HUFloat,
    BC7UNorm,
    BC7Srgb,
    D32SFloat,
    D24UNormS8UInt,
    RGBA32UInt,
    D32SFloatS8UInt,
    Count,
};

constexpr std::size_t kPixelFormatCount = static_cast<std::size_t>(PixelFormat::Count);

struct PixelFormatLayout
{
    uint32_t blockWidth = 1;
    uint32_t blockHeight = 1;
    uint32_t bytesPerBlock = 0;

    [[nodiscard]] constexpr uint64_t RowBytes(uint32_t width) const noexcept
    {
        return ((uint64_t{width} + blockWidth - 1) / blockWidth) * bytesPerBlock;
    }

    [[nodiscard]] constexpr uint64_t RowCount(uint32_t height) const noexcept
    {
        return (uint64_t{height} + blockHeight - 1) / blockHeight;
    }
};

[[nodiscard]] constexpr PixelFormatLayout GetPixelFormatLayout(PixelFormat format) noexcept
{
    switch (format) {
    case PixelFormat::R8UNorm:
        return {1, 1, 1};
    case PixelFormat::RG8UNorm:
    case PixelFormat::R16SFloat:
    case PixelFormat::RGBA4UNormPack16:
        return {1, 1, 2};
    case PixelFormat::RGBA8UNorm:
    case PixelFormat::RGBA8Srgb:
    case PixelFormat::BGRA8UNorm:
    case PixelFormat::BGRA8Srgb:
    case PixelFormat::RG16SFloat:
    case PixelFormat::R32SFloat:
    case PixelFormat::RGB10A2UNorm:
    case PixelFormat::D32SFloat:
    case PixelFormat::D24UNormS8UInt:
        return {1, 1, 4};
    case PixelFormat::RGBA16SFloat:
    case PixelFormat::RGBA16UNorm:
    case PixelFormat::RG32UInt:
    case PixelFormat::D32SFloatS8UInt:
        return {1, 1, 8};
    case PixelFormat::RGBA32SFloat:
    case PixelFormat::RGBA32UInt:
        return {1, 1, 16};
    case PixelFormat::BC1RgbaUNorm:
    case PixelFormat::BC1RgbaSrgb:
    case PixelFormat::BC4UNorm:
        return {4, 4, 8};
    case PixelFormat::BC3UNorm:
    case PixelFormat::BC3Srgb:
    case PixelFormat::BC5UNorm:
    case PixelFormat::BC6HUFloat:
    case PixelFormat::BC7UNorm:
    case PixelFormat::BC7Srgb:
        return {4, 4, 16};
    default:
        return {};
    }
}

[[nodiscard]] constexpr bool IsDepthFormat(PixelFormat format) noexcept
{
    return format == PixelFormat::D32SFloat || format == PixelFormat::D24UNormS8UInt ||
           format == PixelFormat::D32SFloatS8UInt;
}

[[nodiscard]] constexpr bool IsStencilFormat(PixelFormat format) noexcept
{
    return format == PixelFormat::D24UNormS8UInt || format == PixelFormat::D32SFloatS8UInt;
}

[[nodiscard]] constexpr bool IsIntegerFormat(PixelFormat format) noexcept
{
    return format == PixelFormat::RG32UInt || format == PixelFormat::RGBA32UInt;
}

[[nodiscard]] constexpr bool IsValidPixelFormat(PixelFormat format) noexcept
{
    return format > PixelFormat::Undefined && format < PixelFormat::Count;
}

[[nodiscard]] constexpr bool IsBlockCompressedFormat(PixelFormat format) noexcept
{
    return GetPixelFormatLayout(format).blockWidth > 1;
}

[[nodiscard]] constexpr PixelFormat LinearColorFormat(PixelFormat format) noexcept
{
    switch (format) {
    case PixelFormat::RGBA8Srgb:
        return PixelFormat::RGBA8UNorm;
    case PixelFormat::BGRA8Srgb:
        return PixelFormat::BGRA8UNorm;
    case PixelFormat::BC1RgbaSrgb:
        return PixelFormat::BC1RgbaUNorm;
    case PixelFormat::BC3Srgb:
        return PixelFormat::BC3UNorm;
    case PixelFormat::BC7Srgb:
        return PixelFormat::BC7UNorm;
    default:
        return format;
    }
}

[[nodiscard]] constexpr bool IsSrgbFormat(PixelFormat format) noexcept
{
    return LinearColorFormat(format) != format;
}

[[nodiscard]] constexpr bool AreColorSpaceViewFormatsCompatible(PixelFormat imageFormat,
                                                                PixelFormat viewFormat) noexcept
{
    return LinearColorFormat(imageFormat) == LinearColorFormat(viewFormat);
}

} // namespace infernux::rhi
