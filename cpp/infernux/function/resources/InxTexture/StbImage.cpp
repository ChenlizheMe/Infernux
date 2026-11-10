#include "TextureDecoder.h"
#include <function/resources/AssetRegistry/AssetSource.h>
#include <platform/filesystem/InxPath.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>

namespace
{
struct ResizeAllocation
{
    size_t maximum;
    size_t requested = 0;
};

thread_local ResizeAllocation *s_resizeAllocation = nullptr;

void *AllocateResize(size_t size)
{
    if (s_resizeAllocation) {
        s_resizeAllocation->requested = size;
        if (size > s_resizeAllocation->maximum)
            return nullptr;
    }
    return std::malloc(size);
}
} // namespace

#define STBIR_MALLOC(size, user) ((void)(user), AllocateResize(size))
#define STBIR_FREE(pointer, user) ((void)(user), std::free(pointer))
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

namespace infernux::detail
{
namespace
{
bool HasJpegMarker(const unsigned char *bytes)
{
    return bytes[0] == 0xff && bytes[1] == 0xd8;
}

JpegInfo ReadJpegFrame(stbi__context &context)
{
    stbi__jpeg jpeg{};
    jpeg.s = &context;
    if (!stbi__decode_jpeg_header(&jpeg, STBI__SCAN_header))
        throw std::invalid_argument("JPEG header could not be decoded");

    const uint32_t width = context.img_x;
    const uint32_t height = context.img_y;
    int maxH = 1, maxV = 1;
    for (int i = 0; i < context.img_n; ++i) {
        maxH = (std::max)(maxH, jpeg.img_comp[i].h);
        maxV = (std::max)(maxV, jpeg.img_comp[i].v);
    }
    const uint64_t mcuWidth = uint64_t(maxH) * 8;
    const uint64_t mcuHeight = uint64_t(maxV) * 8;
    const auto columns = (width + mcuWidth - 1) / mcuWidth;
    const auto rows = (height + mcuHeight - 1) / mcuHeight;
    // stb retains the padded component and coefficient planes until after
    // converting the full image to RGBA, along with its resample rows.
    uint64_t bytes = sizeof(stbi__jpeg) + uint64_t(width) * height * 4 + 1;
    for (int i = 0; i < context.img_n; ++i) {
        const auto &component = jpeg.img_comp[i];
        if (maxH % component.h || maxV % component.v)
            throw std::invalid_argument("JPEG component sampling is unsupported");
        const auto plane = columns * rows * component.h * component.v * 64;
        bytes += plane + 15 + width + 3;
        if (jpeg.progressive)
            bytes += plane * sizeof(short) + 15;
    }
    return {{width, height, context.img_n >= 3 ? uint8_t{3} : uint8_t{1}}, bytes};
}

struct StbFileInput
{
    FileInput &file;
    uint64_t size;
    uint64_t offset = 0;

    void Start(stbi__context &context)
    {
        stbi_io_callbacks callbacks{[](void *opaque, char *destination, int bytes) {
                                        auto &input = *static_cast<StbFileInput *>(opaque);
                                        const auto count = (std::min)(uint64_t(bytes), input.size - input.offset);
                                        if (count) {
                                            input.file.ReadAt(input.offset, destination, static_cast<size_t>(count));
                                            input.offset += count;
                                        }
                                        return static_cast<int>(count);
                                    },
                                    [](void *opaque, int bytes) {
                                        auto &input = *static_cast<StbFileInput *>(opaque);
                                        input.offset += (std::min)(uint64_t(bytes), input.size - input.offset);
                                    },
                                    [](void *opaque) {
                                        const auto &input = *static_cast<StbFileInput *>(opaque);
                                        return int(input.offset == input.size);
                                    }};
        stbi__start_callbacks(&context, &callbacks, this);
    }
};

RasterInfo ReadRasterFrame(stbi__context &context)
{
    int width = 0, height = 0, channels = 0;
    uint64_t paletteBytes = 0;
    bool gif = false, sixteenBit = false, hdr = false, tga = false;
    bool valid;
    if (stbi__bmp_test(&context)) {
        valid = stbi__bmp_info(&context, &width, &height, &channels);
        // A negative BMP height is the top-down row order.
        if (height < 0 && height >= -STBI_MAX_DIMENSIONS)
            height = -height;
    } else if (stbi__gif_test(&context)) {
        stbi__gif header{};
        valid = stbi__gif_header(&context, &header, &channels, 1);
        width = header.w;
        height = header.h;
        gif = true;
    } else if (stbi__psd_test(&context)) {
        valid = stbi__psd_info(&context, &width, &height, &channels);
        stbi__rewind(&context);
        sixteenBit = stbi__psd_is16(&context);
    } else if (stbi__pic_test(&context)) {
        valid = stbi__pic_info(&context, &width, &height, &channels);
    } else if (stbi__hdr_test(&context)) {
        valid = stbi__hdr_info(&context, &width, &height, &channels);
        hdr = true;
    } else {
        valid = stbi__tga_info(&context, &width, &height, &channels);
        tga = true;
        if (valid) {
            stbi__rewind(&context);
            stbi__skip(&context, 1);
            if (stbi__get8(&context)) {
                stbi__skip(&context, 3);
                paletteBytes = uint64_t(stbi__get16le(&context)) * channels;
            }
        }
    }
    if (!valid || width <= 0 || height <= 0 || width > STBI_MAX_DIMENSIONS || height > STBI_MAX_DIMENSIONS)
        return {};

    const uint64_t pixels = uint64_t(width) * height;
    uint64_t bytes = 4 * pixels;
    if (gif)
        bytes = 9 * pixels; // Output, background and per-pixel history for the first frame.
    if (tga && channels != 4)
        bytes += pixels * channels; // stb converts its source buffer to RGBA.
    if (tga)
        bytes = std::max(bytes, pixels * channels + paletteBytes);
    if (sixteenBit)
        bytes = 12 * pixels; // A compressed 16-bit PSD can widen its decoded 8-bit buffer.
    if (hdr)
        bytes = 16 * pixels + (width >= 8 && width < 32768 ? 4 * uint64_t(width) : 0);
    const auto previewBytes = hdr ? std::max(bytes, 20 * pixels) : sixteenBit ? 4 * pixels : bytes;
    return {{uint32_t(width), uint32_t(height), uint8_t(channels)}, bytes, previewBytes, sixteenBit, hdr};
}
} // namespace

JpegInfo ReadJpegInfo(AssetSource &source)
{
    auto *file = source.OpenedFile();
    std::array<unsigned char, 2> marker{};
    if (!file || source.Size() < marker.size())
        return {};
    file->ReadAt(0, marker.data(), marker.size());
    if (!HasJpegMarker(marker.data()))
        return {};

    StbFileInput input{*file, source.Size()};
    stbi__context context{};
    input.Start(context);
    return ReadJpegFrame(context);
}

RasterInfo ReadRasterInfo(AssetSource &source)
{
    auto *file = source.OpenedFile();
    if (!file || !source.Size())
        return {};
    StbFileInput input{*file, source.Size()};
    stbi__context context{};
    input.Start(context);
    if (const auto raster = ReadRasterFrame(context); raster.source.width)
        return raster;
    const auto pnm = ReadPnmInfo(source);
    const uint64_t bytes = uint64_t(pnm.width) * pnm.height * 4;
    return {pnm, bytes, bytes};
}

RasterInfo ReadRasterInfo(std::string_view bytes)
{
    stbi__context context{};
    stbi__start_mem(&context, reinterpret_cast<const unsigned char *>(bytes.data()), static_cast<int>(bytes.size()));
    if (const auto raster = ReadRasterFrame(context); raster.source.width)
        return raster;
    const auto pnm = ReadPnmInfo(bytes);
    const uint64_t rgbaBytes = uint64_t(pnm.width) * pnm.height * 4;
    return {pnm, rgbaBytes, rgbaBytes};
}

JpegInfo ReadJpegInfo(std::string_view bytes)
{
    const auto *data = reinterpret_cast<const unsigned char *>(bytes.data());
    if (bytes.size() < 2 || !HasJpegMarker(data))
        return {};
    stbi__context context{};
    stbi__start_mem(&context, data, static_cast<int>(bytes.size()));
    return ReadJpegFrame(context);
}

size_t MeasureTextureResize(STBIR_RESIZE &resize)
{
    ResizeAllocation allocation{0};
    s_resizeAllocation = &allocation;
    (void)stbir_build_samplers(&resize);
    s_resizeAllocation = nullptr;
    return allocation.requested;
}

bool RunTextureResize(STBIR_RESIZE &resize, size_t maximum)
{
    ResizeAllocation allocation{maximum};
    s_resizeAllocation = &allocation;
    const bool success = stbir_resize_extended(&resize) != 0;
    s_resizeAllocation = nullptr;
    return success;
}
} // namespace infernux::detail
