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
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernux
{
namespace detail
{
size_t MeasureTextureResize(STBIR_RESIZE &resize);
bool RunTextureResize(STBIR_RESIZE &resize, size_t maximum);
} // namespace detail

namespace
{
constexpr uint32_t MaximumDimension = 65'536;
constexpr uint64_t MaximumDecodedBytes = 1ULL << 30;
constexpr size_t PngWindowBytes = 64 * 1024;
// zlib uses a 32 KiB history window plus its inflate state and tables.
constexpr size_t PngInflateBytes = 64 * 1024;

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

uint32_t NearestSample(uint32_t index, uint32_t sourceSize, uint32_t targetSize)
{
    return std::min(static_cast<uint32_t>((index + 0.5f) * sourceSize / targetSize), sourceSize - 1);
}

template <typename ReadRow>
std::vector<uint8_t> SamplePreview(uint32_t width, uint32_t height, uint32_t outWidth, uint32_t outHeight, ReadRow row)
{
    const auto bytes = LevelByteSize(outWidth, outHeight, TextureFormat::Rgba8UNorm);
    std::vector<uint8_t> output(static_cast<size_t>(bytes));
    for (uint32_t y = 0; y < outHeight; ++y) {
        const auto *source = row(NearestSample(y, height, outHeight));
        auto *dest = output.data() + size_t(y) * outWidth * 4;
        if (width == outWidth) {
            std::memcpy(dest, source, size_t(width) * 4);
            continue;
        }
        for (uint32_t x = 0; x < outWidth; ++x) {
            const auto sourceX = NearestSample(x, width, outWidth);
            std::memcpy(dest + size_t(x) * 4, source + size_t(sourceX) * 4, 4);
        }
    }
    return output;
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

uint32_t PngU32(const unsigned char *bytes)
{
    return uint32_t(bytes[0]) << 24 | uint32_t(bytes[1]) << 16 | uint32_t(bytes[2]) << 8 | bytes[3];
}

struct PngHeader
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t depth = 0;
    uint8_t color = 0;
    bool interlaced = false;

    uint8_t Channels() const
    {
        return color == 3 ? 1 : uint8_t((color & 2 ? 3 : 1) + (color & 4 ? 1 : 0));
    }

    uint16_t MaxSample() const
    {
        return uint16_t((1u << depth) - 1);
    }

    size_t RowBytes(uint32_t pixels) const
    {
        return (size_t(pixels) * Channels() * depth + 7) / 8 + 1;
    }

    size_t ExpansionBytes(TextureFormat format) const
    {
        return color == 6 && depth == 8 ? 0 : size_t(width) * TextureFormatBytesPerTexel(format);
    }
};

PngHeader ParsePngHeader(const std::array<unsigned char, 33> &bytes)
{
    constexpr std::array<unsigned char, 8> signature{137, 80, 78, 71, 13, 10, 26, 10};
    if (!std::equal(signature.begin(), signature.end(), bytes.begin()))
        return {};
    const PngHeader header{PngU32(bytes.data() + 16), PngU32(bytes.data() + 20), bytes[24], bytes[25], bytes[28] != 0};
    if (PngU32(bytes.data() + 8) != 13 || std::memcmp(bytes.data() + 12, "IHDR", 4) != 0)
        throw std::invalid_argument("invalid PNG header");
    if (!header.width || !header.height || header.width > MaximumDimension || header.height > MaximumDimension)
        throw std::invalid_argument("PNG source dimensions exceed texture decoder limits");
    const bool lowDepth = header.depth == 1 || header.depth == 2 || header.depth == 4;
    const bool validLayout = (header.color == 0 && lowDepth) ||
                             (header.color == 3 && (lowDepth || header.depth == 8)) ||
                             ((header.color == 0 || header.color == 2 || header.color == 4 || header.color == 6) &&
                              (header.depth == 8 || header.depth == 16));
    if (!validLayout || bytes[26] != 0 || bytes[27] != 0 || bytes[28] > 1)
        throw std::invalid_argument("unsupported PNG layout");
    return header;
}

PngHeader ReadPngHeader(AssetSource &source)
{
    std::array<unsigned char, 33> bytes{};
    if (!source.OpenedFile() || source.Size() < bytes.size() ||
        source.Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};
    source.OpenedFile()->ReadAt(0, bytes.data(), bytes.size());
    return ParsePngHeader(bytes);
}

class PngRows
{
  public:
    PngRows(AssetSource &source, PngHeader expected, TextureFormat format, bool srgb,
            const std::function<bool()> *retired = nullptr)
        : m_source(source), m_retired(retired), m_header(expected), m_channels(expected.Channels()),
          m_pixelBytes(TextureFormatBytesPerTexel(format)),
          m_linear(srgb && format == TextureFormat::Rgba32Float ? Srgb16ToLinear().data() : nullptr)
    {
        std::array<unsigned char, 33> header{};
        Read(header.data(), header.size());
        const auto actual = ParsePngHeader(header);
        if (crc32(0, header.data() + 12, 17) != PngU32(header.data() + 29))
            throw std::invalid_argument("invalid PNG header checksum");
        if (actual.width != expected.width || actual.height != expected.height || actual.color != expected.color ||
            actual.depth != expected.depth || actual.interlaced != expected.interlaced)
            throw std::runtime_error("PNG layout changed after admission");
        m_rows.resize(m_header.RowBytes(m_header.width) * 2);
        m_expanded.resize(m_header.ExpansionBytes(format));
        BeginPass(m_header.width, m_header.height);
        m_inflate.zalloc = &Allocate;
        m_inflate.zfree = &Free;
        m_inflate.opaque = this;
        if (inflateInit(&m_inflate) != Z_OK)
            throw std::runtime_error("could not initialize PNG decompression");
    }

    ~PngRows()
    {
        inflateEnd(&m_inflate);
    }

    PngRows(const PngRows &) = delete;
    PngRows &operator=(const PngRows &) = delete;

    const unsigned char *Row(int y)
    {
        while (m_lastRow < y)
            Advance();
        const auto *input = m_rows.data() + size_t(y % 2) * m_rowBytes + 1;
        if (m_expanded.empty())
            return input;
        auto *output = m_expanded.data();
        if (m_header.color == 2 && m_header.depth == 8) {
            for (uint32_t x = 0; x < m_passWidth; ++x, input += 3, output += 4) {
                std::copy_n(input, 3, output);
                output[3] = m_transparent && std::equal(m_transparent->begin(), m_transparent->end(), input) ? 0 : 255;
            }
        } else {
            for (uint32_t x = 0; x < m_passWidth; ++x, output += m_pixelBytes)
                ExpandPixel(input, x, output);
        }
        return m_expanded.data();
    }

    void CopyImage(unsigned char *output, uint32_t width, uint32_t height)
    {
        if (m_header.interlaced) {
            constexpr std::array<Pass, 7> adam7{
                {{0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8}, {2, 0, 4, 4}, {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2}}};
            for (const auto pass : adam7)
                CopyPass(pass, output, width, height);
        } else {
            CopyPass({0, 0, 1, 1}, output, width, height);
        }
    }

    int SourceChannels() const
    {
        return m_header.color == 3 ? 3 + m_paletteAlpha : m_channels + m_transparent.has_value();
    }

    static const void *ReadRow(void *, const void *, int pixels, int x, int y, void *context) noexcept
    {
        auto &self = *static_cast<PngRows *>(context);
        if (!self.m_error) {
            try {
                if (x < 0 || y < 0 || uint32_t(x) > self.m_passWidth || uint32_t(y) >= self.m_passHeight ||
                    pixels < 0 || uint32_t(pixels) > self.m_passWidth - uint32_t(x) || y < self.m_lastRow - 1)
                    throw std::runtime_error("PNG resize requested a scanline outside its window");
                return self.Row(y) + size_t(x) * self.m_pixelBytes;
            } catch (...) {
                self.m_error = std::current_exception();
            }
        }
        // The C resize callback has no failure return. Retire its scratch
        // normally, then discard this unpublished output and report the error.
        return self.m_expanded.empty() ? self.m_rows.data() + 1 : self.m_expanded.data();
    }

    void Finish()
    {
        if (m_error)
            std::rethrow_exception(m_error);
        FinishPass();
        unsigned char excess = 0;
        m_inflate.next_out = &excess;
        m_inflate.avail_out = 1;
        Inflate();
        if (!m_streamEnded || !m_inflate.avail_out)
            throw std::invalid_argument("PNG data does not end at its last scanline");
        // The last IDAT may contain unused bytes after the zlib stream.
        SkipChunk();
        bool afterImageData = false;
        for (;;) {
            OpenChunk();
            if (ChunkIs("IEND")) {
                if (m_chunkBytes)
                    throw std::invalid_argument("invalid PNG end chunk");
                FinishChunk();
                break;
            }
            if (ChunkIs("IDAT")) {
                if (afterImageData)
                    throw std::invalid_argument("PNG image data chunks are not consecutive");
            } else {
                if (!(m_chunkType[0] & 32) || ChunkIs("tRNS"))
                    throw std::invalid_argument("unexpected PNG chunk after image data");
                afterImageData = true;
            }
            SkipChunk();
        }
        // Imported assets also consume the entire encoded source for its identity.
        while (m_source.IsSequential() && m_cursor != m_source.Size()) {
            size_t bytes = m_window.size();
            (void)ReadSpan(bytes);
        }
    }

  private:
    struct Pass
    {
        uint32_t x, y, stepX, stepY;
    };

    void BeginPass(uint32_t width, uint32_t height)
    {
        m_passWidth = width;
        m_passHeight = height;
        m_rowBytes = m_header.RowBytes(width);
        m_lastRow = -1;
        std::fill_n(m_rows.data() + m_rowBytes, m_rowBytes, 0);
    }

    void FinishPass()
    {
        while (uint32_t(m_lastRow + 1) < m_passHeight)
            Advance();
    }

    void CopyPass(Pass pass, unsigned char *output, uint32_t width, uint32_t height)
    {
        if (pass.x >= m_header.width || pass.y >= m_header.height)
            return;
        BeginPass((m_header.width - 1 - pass.x) / pass.stepX + 1, (m_header.height - 1 - pass.y) / pass.stepY + 1);
        const size_t pitch = size_t(width) * m_pixelBytes;
        if (width == m_header.width && height == m_header.height) {
            for (uint32_t y = 0; y < m_passHeight; ++y) {
                const auto *row = Row(static_cast<int>(y));
                auto *destination = output + size_t(pass.y + y * pass.stepY) * pitch;
                if (pass.stepX == 1) {
                    std::memcpy(destination, row, pitch);
                } else {
                    for (uint32_t x = 0; x < m_passWidth; ++x)
                        std::memcpy(destination + size_t(pass.x + x * pass.stepX) * m_pixelBytes,
                                    row + size_t(x) * m_pixelBytes, m_pixelBytes);
                }
            }
        } else {
            // Adam7 stores disjoint sets of pixels. Write just the samples the
            // preview will use and still drain every pass through the same inflater.
            for (uint32_t y = 0; y < height; ++y) {
                const auto sourceY = NearestSample(y, m_header.height, height);
                if (sourceY < pass.y || (sourceY - pass.y) % pass.stepY)
                    continue;
                const auto *row = Row(static_cast<int>((sourceY - pass.y) / pass.stepY));
                auto *destination = output + size_t(y) * pitch;
                if (pass.stepX == 1 && width == m_header.width) {
                    std::memcpy(destination, row, pitch);
                    continue;
                }
                for (uint32_t x = 0; x < width; ++x) {
                    const auto sourceX = NearestSample(x, m_header.width, width);
                    if (sourceX >= pass.x && (sourceX - pass.x) % pass.stepX == 0)
                        std::memcpy(destination + size_t(x) * m_pixelBytes,
                                    row + size_t((sourceX - pass.x) / pass.stepX) * m_pixelBytes, m_pixelBytes);
                }
            }
        }
        FinishPass();
    }

    uint16_t ReadSample(const unsigned char *row, size_t index) const
    {
        const auto depth = m_header.depth;
        if (depth == 16)
            return uint16_t(row[2 * index] << 8 | row[2 * index + 1]);
        if (depth == 8)
            return row[index];
        const auto bit = index * depth;
        return uint16_t((row[bit / 8] >> (8 - depth - bit % 8)) & m_header.MaxSample());
    }

    void ExpandPixel(const unsigned char *row, uint32_t x, unsigned char *output) const
    {
        const auto index = size_t(x) * m_channels;
        const auto red = ReadSample(row, index);
        if (m_header.color == 3) {
            if (red >= m_paletteSize)
                throw std::invalid_argument("PNG pixel references a missing palette entry");
            std::memcpy(output, m_palette[red].data(), 4);
            return;
        }
        const bool gray = m_channels < 3;
        const auto maximum = m_header.MaxSample();
        const std::array<uint16_t, 3> rgb{red, gray ? red : ReadSample(row, index + 1),
                                          gray ? red : ReadSample(row, index + 2)};
        const auto alpha = m_transparent && *m_transparent == rgb ? uint16_t{0}
                           : (m_header.color & 4)                 ? ReadSample(row, index + m_channels - 1)
                                                                  : maximum;
        if (m_pixelBytes == 4) {
            const auto toByte = [&](uint16_t value) {
                return static_cast<unsigned char>(m_header.depth == 16 ? value >> 8 : value * 255u / maximum);
            };
            for (size_t channel = 0; channel < rgb.size(); ++channel)
                output[channel] = toByte(rgb[channel]);
            output[3] = toByte(alpha);
        } else {
            const auto linear = [&](uint16_t value) {
                return m_linear ? m_linear[value] : static_cast<float>(value) / 65535.0f;
            };
            const std::array<float, 4> rgba{linear(rgb[0]), linear(rgb[1]), linear(rgb[2]),
                                            static_cast<float>(alpha) / 65535.0f};
            std::memcpy(output, rgba.data(), sizeof(rgba));
        }
    }

    struct alignas(std::max_align_t) Allocation
    {
        size_t size;
    };

    static voidpf Allocate(voidpf opaque, uInt items, uInt size)
    {
        auto &self = *static_cast<PngRows *>(opaque);
        const auto remaining = PngInflateBytes - self.m_allocated;
        if (remaining < sizeof(Allocation) || (size && items > (remaining - sizeof(Allocation)) / size))
            return nullptr;
        const auto bytes = size_t(items) * size + sizeof(Allocation);
        auto *allocation = static_cast<Allocation *>(std::malloc(bytes));
        if (!allocation)
            return nullptr;
        allocation->size = bytes;
        self.m_allocated += bytes;
        return allocation + 1;
    }

    static void Free(voidpf opaque, voidpf address)
    {
        if (!address)
            return;
        auto &self = *static_cast<PngRows *>(opaque);
        auto *allocation = static_cast<Allocation *>(address) - 1;
        self.m_allocated -= allocation->size;
        std::free(allocation);
    }

    void RequireActive() const
    {
        if (m_retired && (*m_retired)())
            throw std::runtime_error("PNG preview was retired");
    }

    const unsigned char *ReadSpan(size_t &bytes)
    {
        if (m_cursor == m_source.Size())
            throw std::invalid_argument("truncated PNG");
        if (m_cursor == m_windowEnd) {
            RequireActive();
            m_windowBegin = m_cursor;
            const auto count = size_t(std::min<uint64_t>(m_window.size(), m_source.Size() - m_cursor));
            if (m_source.IsSequential())
                m_source.ReadSequential(m_window.data(), count);
            else
                m_source.OpenedFile()->ReadAt(m_cursor, m_window.data(), count);
            m_windowEnd += count;
        }
        const auto *span = m_window.data() + (m_cursor - m_windowBegin);
        bytes = size_t(std::min<uint64_t>(bytes, m_windowEnd - m_cursor));
        m_cursor += bytes;
        return span;
    }

    void Read(unsigned char *output, size_t bytes)
    {
        while (bytes) {
            auto count = bytes;
            const auto *input = ReadSpan(count);
            std::memcpy(output, input, count);
            output += count;
            bytes -= count;
        }
    }

    bool ChunkIs(const char *type) const
    {
        return std::memcmp(m_chunkType.data(), type, 4) == 0;
    }

    void OpenChunk()
    {
        std::array<unsigned char, 8> header{};
        Read(header.data(), header.size());
        m_chunkBytes = PngU32(header.data());
        if (m_chunkBytes > 0x7fffffff || uint64_t(m_chunkBytes) + 4 > m_source.Size() - m_cursor)
            throw std::invalid_argument("truncated PNG chunk");
        std::copy_n(header.data() + 4, m_chunkType.size(), m_chunkType.begin());
        m_chunkCrc = crc32(0, m_chunkType.data(), static_cast<uInt>(m_chunkType.size()));
        m_chunkOpen = true;
    }

    const unsigned char *ReadChunk(size_t &bytes)
    {
        bytes = std::min<size_t>(bytes, m_chunkBytes);
        const auto *span = ReadSpan(bytes);
        m_chunkCrc = crc32(m_chunkCrc, span, static_cast<uInt>(bytes));
        m_chunkBytes -= static_cast<uint32_t>(bytes);
        return span;
    }

    void FinishChunk()
    {
        std::array<unsigned char, 4> crc{};
        Read(crc.data(), crc.size());
        if (PngU32(crc.data()) != m_chunkCrc)
            throw std::invalid_argument("invalid PNG chunk checksum");
        m_chunkOpen = false;
    }

    void SkipChunk()
    {
        while (m_chunkBytes) {
            size_t bytes = m_chunkBytes;
            (void)ReadChunk(bytes);
        }
        FinishChunk();
    }

    void ReadTransparency()
    {
        if (m_header.color == 3) {
            if (!m_paletteSize || m_chunkBytes > m_paletteSize || m_paletteAlpha)
                throw std::invalid_argument("invalid PNG palette transparency");
            for (size_t index = 0; m_chunkBytes; ++index) {
                size_t bytes = 1;
                m_palette[index][3] = *ReadChunk(bytes);
            }
            m_paletteAlpha = true;
        } else {
            if ((m_header.color & 4) || m_transparent || m_chunkBytes != uint32_t(m_channels) * 2)
                throw std::invalid_argument("invalid PNG transparent color");
            std::array<uint16_t, 3> key{};
            for (uint8_t channel = 0; channel < m_channels; ++channel) {
                size_t bytes = 1;
                const auto high = *ReadChunk(bytes);
                const auto low = *ReadChunk(bytes);
                key[channel] = uint16_t((high << 8 | low) & m_header.MaxSample());
            }
            if (m_channels == 1)
                key[1] = key[2] = key[0];
            m_transparent = key;
        }
        FinishChunk();
    }

    void ReadPalette()
    {
        if (m_paletteSize || !m_chunkBytes || m_chunkBytes % 3 || m_chunkBytes / 3 > m_palette.size())
            throw std::invalid_argument("invalid PNG palette");
        m_paletteSize = m_chunkBytes / 3;
        for (uint32_t index = 0; index < m_paletteSize; ++index) {
            size_t bytes = 1;
            auto &color = m_palette[index];
            for (size_t channel = 0; channel < 3; ++channel)
                color[channel] = *ReadChunk(bytes);
            color[3] = 255;
        }
        FinishChunk();
    }

    void SupplyInput()
    {
        while (!m_chunkBytes) {
            if (m_chunkOpen)
                FinishChunk();
            OpenChunk();
            if (ChunkIs("IDAT")) {
                if (m_header.color == 3 && !m_paletteSize)
                    throw std::invalid_argument("indexed PNG has no palette");
                m_startedImage = true;
            } else if (!m_startedImage && ChunkIs("tRNS"))
                ReadTransparency();
            else if (!m_startedImage && m_header.color == 3 && ChunkIs("PLTE"))
                ReadPalette();
            else if (!m_startedImage && ((m_chunkType[0] & 32) || ChunkIs("PLTE")))
                SkipChunk();
            else
                throw std::invalid_argument("missing or incomplete PNG image data");
        }
        size_t bytes = m_chunkBytes;
        m_inflate.next_in = const_cast<Bytef *>(ReadChunk(bytes));
        m_inflate.avail_in = static_cast<uInt>(bytes);
    }

    void Inflate()
    {
        while (m_inflate.avail_out && !m_streamEnded) {
            // inflate can still emit buffered output with no encoded input.
            auto result = inflate(&m_inflate, Z_NO_FLUSH);
            if (result == Z_BUF_ERROR && !m_inflate.avail_in) {
                SupplyInput();
                result = inflate(&m_inflate, Z_NO_FLUSH);
            }
            if (result == Z_STREAM_END)
                m_streamEnded = true;
            else if (result != Z_OK)
                throw std::invalid_argument("invalid PNG deflate stream");
        }
    }

    void Advance()
    {
        RequireActive();
        const auto index = size_t(++m_lastRow % 2);
        const auto rowBytes = m_rowBytes;
        auto *row = m_rows.data() + index * rowBytes;
        const auto *previous = m_rows.data() + (1 - index) * rowBytes;
        m_inflate.next_out = row;
        m_inflate.avail_out = static_cast<uInt>(rowBytes);
        Inflate();
        if (m_inflate.avail_out || row[0] > 4)
            throw std::invalid_argument("invalid PNG scanline");
        if (!row[0])
            return;
        const auto stride = (m_channels * m_header.depth + 7u) / 8u;
        for (size_t i = 1; i < rowBytes; ++i) {
            const int left = i > stride ? row[i - stride] : 0;
            const int above = previous[i];
            const int diagonal = i > stride ? previous[i - stride] : 0;
            int prediction = 0;
            switch (row[0]) {
            case 1:
                prediction = left;
                break;
            case 2:
                prediction = above;
                break;
            case 3:
                prediction = (left + above) / 2;
                break;
            case 4: {
                const int point = left + above - diagonal;
                const int l = std::abs(point - left), a = std::abs(point - above), d = std::abs(point - diagonal);
                prediction = l <= a && l <= d ? left : a <= d ? above : diagonal;
                break;
            }
            }
            row[i] = static_cast<unsigned char>(row[i] + prediction);
        }
    }

    AssetSource &m_source;
    const std::function<bool()> *m_retired;
    PngHeader m_header;
    uint8_t m_channels;
    size_t m_pixelBytes;
    const float *m_linear;
    uint32_t m_passWidth = 0, m_passHeight = 0;
    size_t m_rowBytes = 0;
    std::vector<unsigned char> m_rows;
    std::vector<unsigned char> m_expanded;
    std::optional<std::array<uint16_t, 3>> m_transparent;
    std::array<std::array<unsigned char, 4>, 256> m_palette;
    uint32_t m_paletteSize = 0;
    std::array<unsigned char, PngWindowBytes> m_window;
    std::array<unsigned char, 4> m_chunkType{};
    uint64_t m_cursor = 0;
    uint64_t m_windowBegin = 0;
    uint64_t m_windowEnd = 0;
    uint32_t m_chunkBytes = 0;
    uLong m_chunkCrc = 0;
    size_t m_allocated = 0;
    z_stream m_inflate{};
    std::exception_ptr m_error;
    int m_lastRow = -1;
    bool m_chunkOpen = false;
    bool m_startedImage = false;
    bool m_streamEnded = false;
    bool m_paletteAlpha = false;
};

uint64_t PngRowStorage(PngHeader header, TextureFormat format)
{
    return sizeof(PngRows) + 2 * header.RowBytes(header.width) + header.ExpansionBytes(format) + PngInflateBytes;
}

bool UsesSrgbResize(const InxResourceMeta &metadata)
{
    const auto semantic = ReadSemantic(metadata);
    return ReadSrgb(metadata) && (semantic == TextureSemantic::Color || semantic == TextureSemantic::UserInterface ||
                                  semantic == TextureSemantic::Sprite);
}

void InitRgbaResize(STBIR_RESIZE &resize, uint32_t width, uint32_t height, uint32_t targetWidth, uint32_t targetHeight,
                    const InxResourceMeta &metadata, TextureFormat format = TextureFormat::Rgba8UNorm,
                    const void *input = nullptr, void *output = nullptr)
{
    const auto type = format == TextureFormat::Rgba32Float ? STBIR_TYPE_FLOAT
                      : UsesSrgbResize(metadata)           ? STBIR_TYPE_UINT8_SRGB
                                                           : STBIR_TYPE_UINT8;
    stbir_resize_init(&resize, input, static_cast<int>(width), static_cast<int>(height), 0, output,
                      static_cast<int>(targetWidth), static_cast<int>(targetHeight), 0, STBIR_RGBA, type);
}

struct RgbaCookStorage
{
    uint64_t mipBytes = 0;
    uint64_t processingBytes = 0;
};

RgbaCookStorage PlanRgbaCook(uint32_t width, uint32_t height, const InxResourceMeta &metadata,
                             TextureFormat format = TextureFormat::Rgba8UNorm)
{
    const auto target = ReadTargetFormat(metadata);
    const bool mipmaps = ReadGenerateMipmaps(metadata);
    uint64_t mipBytes = 0;
    uint64_t blocks = 0;
    for (auto w = width, h = height;; w = std::max(1u, w / 2u), h = std::max(1u, h / 2u)) {
        mipBytes += LevelByteSize(w, h, format);
        blocks += uint64_t((w + 3u) / 4u) * ((h + 3u) / 4u);
        if (!mipmaps || (w == 1 && h == 1))
            break;
    }
    if (mipBytes > MaximumDecodedBytes)
        throw std::overflow_error("texture mip chain exceeds the payload limit");
    const auto texels = mipBytes / TextureFormatBytesPerTexel(format);
    uint64_t outputBytes = mipBytes;
    switch (target) {
    case TextureTargetFormat::Automatic: {
        auto compression = ReadCompression(metadata);
        if (compression == TextureCompression::Automatic)
            compression = format == TextureFormat::Rgba32Float
                              ? TextureCompression::None
                              : TextureProcessor::AutomaticRgba8Compression(ReadSemantic(metadata), true);
        if (compression != TextureCompression::None) {
            const bool shortBlock = compression == TextureCompression::BC1 || compression == TextureCompression::BC4;
            outputBytes = blocks * (shortBlock ? 8u : 16u);
        }
        break;
    }
    case TextureTargetFormat::Rgba4UNorm:
        outputBytes = texels * 2;
        break;
    case TextureTargetFormat::Rgba8:
        outputBytes = texels * 4;
        break;
    case TextureTargetFormat::Rgba16UNorm:
    case TextureTargetFormat::Rgba16Float:
        outputBytes = texels * 8;
        break;
    case TextureTargetFormat::Rgba32Float:
        outputBytes = texels * 16;
        break;
    }
    if (outputBytes > MaximumDecodedBytes)
        throw std::overflow_error("texture mip chain exceeds the payload limit");
    const uint64_t mipScratch =
        mipmaps ? uint64_t(std::max(1u, width / 2u)) * sizeof(std::pair<uint32_t, uint32_t>) : 0;
    // Compression/conversion retains the source. Serialization retains the
    // final texture alongside the artifact string.
    return {mipBytes, std::max({mipBytes + mipScratch, mipBytes + outputBytes, 2 * outputBytes})};
}

constexpr uint64_t ArtifactControlBytes = 64 * 1024;

struct JpegPlan
{
    uint64_t bytes;
    size_t mipBytes;
};

JpegPlan PlanJpeg(const detail::JpegInfo &jpeg, uint64_t sourceBytes, const InxResourceMeta &metadata)
{
    const auto [width, height] = FitDimensions(jpeg.source.width, jpeg.source.height, ReadMaxSize(metadata));
    const auto cook = PlanRgbaCook(width, height, metadata);
    size_t resizeScratch = 0;
    if (width != jpeg.source.width || height != jpeg.source.height) {
        STBIR_RESIZE resize;
        InitRgbaResize(resize, jpeg.source.width, jpeg.source.height, width, height, metadata);
        resizeScratch = detail::MeasureTextureResize(resize);
        if (!resizeScratch)
            throw std::invalid_argument("cannot plan the JPEG resize workspace");
    }
    const auto rgbaBytes = uint64_t(jpeg.source.width) * jpeg.source.height * 4;
    const auto workingBytes =
        std::max({jpeg.decodeBytes, rgbaBytes + cook.mipBytes + resizeScratch, cook.processingBytes});
    return {sourceBytes + workingBytes + ArtifactControlBytes, static_cast<size_t>(cook.mipBytes)};
}

struct PngPlan
{
    PngHeader source;
    uint32_t width = 0, height = 0;
    TextureFormat format = TextureFormat::Rgba8UNorm;
    uint64_t bytes = 0;
    size_t resizeScratch = 0;
    size_t mipBytes = 0;
    size_t interlacedBytes = 0;
};

PngPlan PlanPng(AssetSource &source, const InxResourceMeta &metadata)
{
    const auto header = ReadPngHeader(source);
    if (!header.width)
        return {};
    const auto semantic = ReadSemantic(metadata);
    if (semantic == TextureSemantic::VectorField || semantic == TextureSemantic::SignedDistanceField)
        throw std::invalid_argument("PNG source cannot supply a volume texture");
    const auto [width, height] = FitDimensions(header.width, header.height, ReadMaxSize(metadata));
    const auto format = header.depth == 16   ? TextureFormat::Rgba32Float
                        : ReadSrgb(metadata) ? TextureFormat::Rgba8Srgb
                                             : TextureFormat::Rgba8UNorm;
    size_t resizeScratch = 0;
    size_t interlacedBytes = 0;
    if (width != header.width || height != header.height) {
        STBIR_RESIZE resize;
        InitRgbaResize(resize, header.width, header.height, width, height, metadata, format);
        if (header.interlaced)
            interlacedBytes = static_cast<size_t>(LevelByteSize(header.width, header.height, format));
        else
            stbir_set_pixel_callbacks(&resize, &PngRows::ReadRow, nullptr);
        resizeScratch = detail::MeasureTextureResize(resize);
        if (!resizeScratch)
            throw std::invalid_argument("cannot plan the PNG resize workspace");
    }
    const auto cook = PlanRgbaCook(width, height, metadata, format);
    const auto rows = PngRowStorage(header, format);
    const auto workspace =
        interlacedBytes ? interlacedBytes + std::max(rows, uint64_t(resizeScratch)) : rows + resizeScratch;
    const auto bytes = std::max(cook.mipBytes + workspace, cook.processingBytes) + ArtifactControlBytes;
    return {header, width, height, format, bytes, resizeScratch, static_cast<size_t>(cook.mipBytes), interlacedBytes};
}

} // namespace

TextureDecoder::SourceInfo TextureDecoder::ReadSequentialSourceInfo(AssetSource &source)
{
    if (const auto jpeg = detail::ReadJpegInfo(source); jpeg.source.width)
        return jpeg.source;
    const auto png = ReadPngHeader(source);
    // PNG without an explicit alpha channel can gain one from tRNS.
    return {png.width, png.height, png.color & 4 ? png.Channels() : uint8_t{0}};
}

uint64_t TextureDecoder::PlanSourceStorage(AssetSource &source, const InxResourceMeta &metadata)
{
    try {
        if (const auto jpeg = detail::ReadJpegInfo(source); jpeg.source.width)
            return PlanJpeg(jpeg, source.Size(), metadata).bytes;
        return PlanPng(source, metadata).bytes;
    } catch (const std::overflow_error &error) {
        throw std::invalid_argument(error.what());
    }
}

TextureDecoder::SourcePreviewPlan TextureDecoder::PlanSourcePreview(AssetSource &source, uint32_t maxSize,
                                                                    bool document)
{
    const auto header = ReadPngHeader(source);
    maxSize = maxSize ? maxSize : MaximumDimension;
    if (header.width) {
        const auto [width, height] = FitDimensions(header.width, header.height, maxSize);
        const auto format = document && header.depth == 16 ? TextureFormat::Rgba32Float : TextureFormat::Rgba8UNorm;
        const auto pixels = LevelByteSize(width, height, format);
        return {{header.width, header.height, header.Channels()},
                width,
                height,
                pixels + PngRowStorage(header, format),
                SourcePreviewPlan::Codec::Png,
                format};
    }
    const auto jpeg = detail::ReadJpegInfo(source);
    if (!jpeg.source.width)
        return {};
    const auto [width, height] = FitDimensions(jpeg.source.width, jpeg.source.height, maxSize);
    const auto output = LevelByteSize(width, height, TextureFormat::Rgba8UNorm);
    const auto input = uint64_t(jpeg.source.width) * jpeg.source.height * 4;
    const auto bytes = source.Size() + std::max(jpeg.decodeBytes, input + 1 + output);
    return {jpeg.source, width, height, bytes, SourcePreviewPlan::Codec::Jpeg};
}

InxTextureData TextureDecoder::DecodeSourcePreview(AssetSource &source, const SourcePreviewPlan &plan,
                                                   const std::function<bool()> &retired)
{
    InxTextureData image;
    image.width = static_cast<int>(plan.previewWidth);
    image.height = static_cast<int>(plan.previewHeight);
    if (plan.codec == SourcePreviewPlan::Codec::Png) {
        const auto header = ReadPngHeader(source);
        const auto pixels = LevelByteSize(plan.previewWidth, plan.previewHeight, plan.format);
        if (header.width != plan.source.width || header.height != plan.source.height ||
            (plan.format == TextureFormat::Rgba32Float && header.depth != 16) ||
            pixels + PngRowStorage(header, plan.format) > plan.bytes)
            throw std::runtime_error("PNG preview layout changed after admission");
        PngRows rows(source, header, plan.format, true, &retired);
        image.pixels.resize(static_cast<size_t>(pixels));
        rows.CopyImage(image.pixels.data(), plan.previewWidth, plan.previewHeight);
        rows.Finish();
    } else {
        std::vector<unsigned char> encoded(static_cast<size_t>(source.Size()));
        source.OpenedFile()->ReadAt(0, encoded.data(), encoded.size());
        const auto jpeg = detail::ReadJpegInfo({reinterpret_cast<const char *>(encoded.data()), encoded.size()});
        if (jpeg.source.width != plan.source.width || jpeg.source.height != plan.source.height ||
            encoded.size() + jpeg.decodeBytes > plan.bytes)
            throw std::runtime_error("JPEG preview source changed after admission");
        int width, height, channels;
        const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
            stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, &channels,
                                  STBI_rgb_alpha),
            &stbi_image_free);
        if (!pixels)
            throw std::runtime_error("failed to decode JPEG preview");
        image.pixels = SamplePreview(plan.source.width, plan.source.height, plan.previewWidth, plan.previewHeight,
                                     [&](uint32_t row) { return pixels.get() + size_t(row) * plan.source.width * 4; });
    }
    return image;
}

void TextureDecoder::ResizeRgba8Preview(InxTextureData &image, uint32_t maxSize)
{
    const auto width = static_cast<uint32_t>(image.width);
    const auto height = static_cast<uint32_t>(image.height);
    const auto [outWidth, outHeight] = FitDimensions(width, height, maxSize);
    if (outWidth == width && outHeight == height)
        return;
    image.pixels = SamplePreview(width, height, outWidth, outHeight,
                                 [&](uint32_t row) { return image.pixels.data() + size_t(row) * width * 4; });
    image.width = static_cast<int>(outWidth);
    image.height = static_cast<int>(outHeight);
}

std::shared_ptr<const TextureCpuData> TextureDecoder::TakePreview(InxTextureData image, bool srgb, TextureFormat format)
{
    TextureCpuData texture;
    texture.semantic = srgb ? TextureSemantic::UserInterface : TextureSemantic::Data;
    texture.format = srgb && format == TextureFormat::Rgba8UNorm ? TextureFormat::Rgba8Srgb : format;
    const auto level = DescribeBaseLevel(image.width, image.height, texture.format);
    if (level.byteSize != image.pixels.size())
        throw std::invalid_argument("preview pixels do not match image dimensions");
    texture.bytes = std::move(image.pixels);
    texture.mipLevels.push_back(level);
    return TextureProcessor::Process(
        std::move(texture), TextureProcessOptions{false, TextureCompression::None, TextureCompressionQuality::Normal});
}

std::shared_ptr<const TextureCpuData> TextureDecoder::Decode(const std::string &sourcePath,
                                                             const InxResourceMeta &metadata, bool preserveSvgViewport)
{
    AssetSource source(sourcePath);
    return Decode(source, metadata, preserveSvgViewport);
}

std::shared_ptr<const TextureCpuData> TextureDecoder::Decode(AssetSource &source, const InxResourceMeta &metadata,
                                                             bool preserveSvgViewport, int *sequentialChannels)
{
    if (!source.Size() || source.Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("texture source is empty or exceeds decoder limits: " + source.Path());
    if (source.IsSequential()) {
        auto *file = source.OpenedFile();
        if (!file)
            throw std::logic_error("texture source must be opened for its admitted decode");
        const auto revision = file->InspectRevision();
        if (detail::ReadJpegInfo(source).source.width) {
            if (source.Size() > source.SequentialStorageLimit())
                throw std::logic_error("JPEG source exceeded its admitted storage");
            std::string encoded(static_cast<size_t>(source.Size()), '\0');
            source.ReadSequential(encoded.data(), encoded.size());
            if (!(file->InspectRevision() == revision))
                throw std::runtime_error("JPEG source changed during capture: " + source.Path());
            source.CloseFile();
            // The decoder sees these exact bytes, even if the file changes
            // between admission, capture and the caller's publication check.
            const auto jpeg = detail::ReadJpegInfo(encoded);
            if (!jpeg.source.width)
                throw std::runtime_error("JPEG source changed before decode: " + source.Path());
            const auto plan = PlanJpeg(jpeg, encoded.size(), metadata);
            if (plan.bytes > source.SequentialStorageLimit())
                throw std::logic_error("JPEG options changed after bounded admission");
            if (sequentialChannels)
                *sequentialChannels = jpeg.source.channels;
            return DecodeBytes(encoded, metadata, source.Path(), preserveSvgViewport, plan.mipBytes);
        }
        const auto plan = PlanPng(source, metadata);
        if (!plan.bytes || plan.bytes > source.SequentialStorageLimit())
            throw std::logic_error("PNG options changed after bounded admission");
        TextureCpuData texture;
        texture.semantic = ReadSemantic(metadata);
        texture.format = plan.format;
        const auto level = DescribeBaseLevel(plan.width, plan.height, texture.format);
        texture.bytes.reserve(plan.mipBytes);
        texture.bytes.resize(static_cast<size_t>(level.byteSize));
        texture.mipLevels.push_back(level);
        {
            std::vector<unsigned char> interlaced(plan.interlacedBytes);
            STBIR_RESIZE resize;
            if (plan.resizeScratch) {
                InitRgbaResize(resize, plan.source.width, plan.source.height, plan.width, plan.height, metadata,
                               plan.format, interlaced.empty() ? nullptr : interlaced.data(), texture.bytes.data());
            }
            {
                PngRows rows(source, plan.source, plan.format, ReadSrgb(metadata));
                if (!interlaced.empty()) {
                    rows.CopyImage(interlaced.data(), plan.source.width, plan.source.height);
                } else if (plan.resizeScratch) {
                    stbir_set_pixel_callbacks(&resize, &PngRows::ReadRow, nullptr);
                    stbir_set_user_data(&resize, &rows);
                    if (!detail::RunTextureResize(resize, plan.resizeScratch))
                        throw std::runtime_error("failed to resize PNG texture");
                } else {
                    rows.CopyImage(texture.bytes.data(), plan.width, plan.height);
                }
                rows.Finish();
                if (sequentialChannels)
                    *sequentialChannels = rows.SourceChannels();
            }
            if (!interlaced.empty() && !detail::RunTextureResize(resize, plan.resizeScratch))
                throw std::runtime_error("failed to resize PNG texture");
        }
        if (!(file->InspectRevision() == revision))
            throw std::runtime_error("PNG source changed during decode: " + source.Path());
        source.CloseFile();
        return TextureProcessor::Process(
            std::move(texture), TextureProcessOptions{ReadGenerateMipmaps(metadata), ReadCompression(metadata),
                                                      ReadCompressionQuality(metadata), ReadTargetFormat(metadata)});
    }
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
                                                                  bool preserveSvgViewport, size_t plannedMipBytes)
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
        if (plannedMipBytes)
            texture.bytes.reserve(plannedMipBytes);
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
