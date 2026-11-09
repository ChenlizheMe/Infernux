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

template <typename ReadRow>
std::vector<uint8_t> SamplePreview(uint32_t width, uint32_t height, uint32_t outWidth, uint32_t outHeight, ReadRow row)
{
    const auto bytes = LevelByteSize(outWidth, outHeight, TextureFormat::Rgba8UNorm);
    std::vector<uint8_t> output(static_cast<size_t>(bytes));
    for (uint32_t y = 0; y < outHeight; ++y) {
        const auto sourceY = std::min(static_cast<uint32_t>((y + 0.5f) * height / outHeight), height - 1);
        const auto *source = row(sourceY);
        auto *dest = output.data() + size_t(y) * outWidth * 4;
        if (width == outWidth) {
            std::memcpy(dest, source, size_t(width) * 4);
            continue;
        }
        for (uint32_t x = 0; x < outWidth; ++x) {
            const auto sourceX = std::min(static_cast<uint32_t>((x + 0.5f) * width / outWidth), width - 1);
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

using PngHeader = TextureDecoder::PngSourceInfo;

size_t PngScanlineBytes(PngHeader header)
{
    return size_t(header.width) * header.channels + 1;
}

size_t PngExpansionBytes(PngHeader header)
{
    return header.channels == 3 ? size_t(header.width) * 4 : 0;
}

PngHeader ParsePngHeader(const std::array<unsigned char, 33> &header)
{
    constexpr std::array<unsigned char, 16> prefix{137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 'I', 'H', 'D', 'R'};
    const auto colorType = header[25];
    if (!std::equal(prefix.begin(), prefix.end(), header.begin()) || header[24] != 8 ||
        (colorType != 2 && colorType != 6) || header[26] != 0 || header[27] != 0 || header[28] != 0)
        return {};
    const auto width = PngU32(header.data() + 16);
    const auto height = PngU32(header.data() + 20);
    if (!width || !height || width > MaximumDimension || height > MaximumDimension)
        throw std::invalid_argument("PNG source dimensions exceed texture decoder limits");
    return {width, height, colorType == 2 ? uint8_t{3} : uint8_t{4}};
}

class PngRows
{
  public:
    PngRows(AssetSource &source, PngHeader expected, const std::function<bool()> *retired = nullptr)
        : m_source(source), m_retired(retired), m_header(expected)
    {
        std::array<unsigned char, 33> header{};
        Read(header.data(), header.size());
        const auto actual = ParsePngHeader(header);
        if (actual.width != expected.width || actual.height != expected.height ||
            actual.channels != expected.channels || crc32(0, header.data() + 12, 17) != PngU32(header.data() + 29))
            throw std::invalid_argument("invalid PNG header");
        m_rows.resize(PngScanlineBytes(m_header) * 2);
        m_expanded.resize(PngExpansionBytes(m_header));
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
        const auto *input = m_rows.data() + size_t(y % 2) * PngScanlineBytes(m_header) + 1;
        if (m_expanded.empty())
            return input;
        // The PNG filter needs the original RGB row. Each consumer reads
        // the expanded pixels before requesting another row.
        auto *output = m_expanded.data();
        for (uint32_t x = 0; x < m_header.width; ++x, input += 3, output += 4) {
            std::copy_n(input, 3, output);
            output[3] = m_transparent && std::equal(m_transparent->begin(), m_transparent->end(), input) ? 0 : 255;
        }
        return m_expanded.data();
    }

    int SourceChannels() const
    {
        return m_header.channels + m_transparent.has_value();
    }

    static const void *ReadRow(void *, const void *, int pixels, int x, int y, void *context) noexcept
    {
        auto &self = *static_cast<PngRows *>(context);
        if (!self.m_error) {
            try {
                if (x < 0 || y < 0 || uint32_t(x) > self.m_header.width || uint32_t(y) >= self.m_header.height ||
                    pixels < 0 || uint32_t(pixels) > self.m_header.width - uint32_t(x) || y < self.m_lastRow - 1)
                    throw std::runtime_error("PNG resize requested a scanline outside its window");
                return self.Row(y) + size_t(x) * 4;
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
        while (uint32_t(m_lastRow + 1) < m_header.height)
            Advance();
        unsigned char excess = 0;
        m_inflate.next_out = &excess;
        m_inflate.avail_out = 1;
        Inflate();
        if (!m_streamEnded || !m_inflate.avail_out || m_inflate.avail_in || m_chunkBytes)
            throw std::invalid_argument("PNG data does not end at its last scanline");
        FinishChunk();
        for (;;) {
            OpenChunk();
            if (ChunkIs("IEND")) {
                if (m_chunkBytes)
                    throw std::invalid_argument("invalid PNG end chunk");
                FinishChunk();
                break;
            }
            if (ChunkIs("IDAT") ? m_chunkBytes != 0 : (!(m_chunkType[0] & 32) || ChunkIs("tRNS")))
                throw std::invalid_argument("unexpected PNG chunk after image data");
            SkipChunk();
        }
        // Imported assets also consume the entire encoded source for its identity.
        while (m_source.IsSequential() && m_cursor != m_source.Size()) {
            size_t bytes = m_window.size();
            (void)ReadSpan(bytes);
        }
    }

  private:
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

    const unsigned char *ReadSpan(size_t &bytes)
    {
        if (m_cursor == m_source.Size())
            throw std::invalid_argument("truncated PNG");
        if (m_cursor == m_windowEnd) {
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
        std::array<unsigned char, 6> key{};
        if (m_header.channels != 3 || m_chunkBytes != key.size())
            throw std::invalid_argument("invalid PNG transparent color");
        for (auto &value : key) {
            size_t bytes = 1;
            value = *ReadChunk(bytes);
        }
        FinishChunk();
        // The color key stores the eight-bit samples in 16-bit fields.
        m_transparent = std::array{key[1], key[3], key[5]};
    }

    void SupplyInput()
    {
        while (!m_chunkBytes) {
            if (m_chunkOpen)
                FinishChunk();
            OpenChunk();
            if (ChunkIs("IDAT"))
                m_startedImage = true;
            else if (!m_startedImage && ChunkIs("tRNS"))
                ReadTransparency();
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
        if (m_retired && (*m_retired)())
            throw std::runtime_error("PNG preview was retired");
        const auto index = size_t(++m_lastRow % 2);
        const auto rowBytes = PngScanlineBytes(m_header);
        auto *row = m_rows.data() + index * rowBytes;
        const auto *previous = m_rows.data() + (1 - index) * rowBytes;
        m_inflate.next_out = row;
        m_inflate.avail_out = static_cast<uInt>(rowBytes);
        Inflate();
        if (m_inflate.avail_out || row[0] > 4)
            throw std::invalid_argument("invalid PNG scanline");
        if (!row[0])
            return;
        const auto stride = m_header.channels;
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
    std::vector<unsigned char> m_rows;
    std::vector<unsigned char> m_expanded;
    std::optional<std::array<unsigned char, 3>> m_transparent;
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
};

uint64_t PngRowStorage(PngHeader header)
{
    return sizeof(PngRows) + 2 * PngScanlineBytes(header) + PngExpansionBytes(header) + PngInflateBytes;
}

bool UsesSrgbResize(const InxResourceMeta &metadata)
{
    const auto semantic = ReadSemantic(metadata);
    return ReadSrgb(metadata) && (semantic == TextureSemantic::Color || semantic == TextureSemantic::UserInterface ||
                                  semantic == TextureSemantic::Sprite);
}

void InitPngResize(STBIR_RESIZE &resize, uint32_t width, uint32_t height, uint32_t targetWidth, uint32_t targetHeight,
                   const InxResourceMeta &metadata, void *output = nullptr)
{
    stbir_resize_init(&resize, nullptr, static_cast<int>(width), static_cast<int>(height), 0, output,
                      static_cast<int>(targetWidth), static_cast<int>(targetHeight), 0, STBIR_RGBA,
                      UsesSrgbResize(metadata) ? STBIR_TYPE_UINT8_SRGB : STBIR_TYPE_UINT8);
    stbir_set_pixel_callbacks(&resize, &PngRows::ReadRow, nullptr);
}

} // namespace

TextureDecoder::PngSourceInfo TextureDecoder::ReadPngSourceInfo(AssetSource &source)
{
    std::array<unsigned char, 33> header{};
    if (!source.OpenedFile() || source.Size() < header.size() ||
        source.Size() > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        return {};
    source.OpenedFile()->ReadAt(0, header.data(), header.size());
    return ParsePngHeader(header);
}

TextureDecoder::PngPreviewPlan TextureDecoder::PlanPngPreview(AssetSource &source, uint32_t maxSize)
{
    const auto header = ReadPngSourceInfo(source);
    if (!header.width)
        return {};
    const auto [outWidth, outHeight] = FitDimensions(header.width, header.height, maxSize ? maxSize : MaximumDimension);
    const auto pixels = LevelByteSize(outWidth, outHeight, TextureFormat::Rgba8UNorm);
    return {header, outWidth, outHeight, pixels + PngRowStorage(header)};
}

InxTextureData TextureDecoder::DecodePngPreview(AssetSource &source, const PngPreviewPlan &plan,
                                                const std::function<bool()> &retired)
{
    PngRows rows(source, plan.source, &retired);
    InxTextureData image;
    image.width = static_cast<int>(plan.previewWidth);
    image.height = static_cast<int>(plan.previewHeight);
    image.pixels = SamplePreview(plan.source.width, plan.source.height, plan.previewWidth, plan.previewHeight,
                                 [&](uint32_t row) { return rows.Row(static_cast<int>(row)); });
    rows.Finish();
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

std::shared_ptr<const TextureCpuData> TextureDecoder::TakeRgba8Preview(InxTextureData image, bool srgb)
{
    TextureCpuData texture;
    texture.semantic = srgb ? TextureSemantic::UserInterface : TextureSemantic::Data;
    texture.format = srgb ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
    const auto level = DescribeBaseLevel(image.width, image.height, texture.format);
    if (level.byteSize != image.pixels.size())
        throw std::invalid_argument("preview pixels do not match image dimensions");
    texture.bytes = std::move(image.pixels);
    texture.mipLevels.push_back(level);
    return TextureProcessor::Process(
        std::move(texture), TextureProcessOptions{false, TextureCompression::None, TextureCompressionQuality::Normal});
}

TextureDecoder::PngPlan TextureDecoder::PlanPng(AssetSource &source, const InxResourceMeta &metadata)
{
    const auto header = ReadPngSourceInfo(source);
    const auto width = header.width;
    const auto height = header.height;
    if (!width || ReadTargetFormat(metadata) != TextureTargetFormat::Automatic ||
        ReadSemantic(metadata) == TextureSemantic::VectorField ||
        ReadSemantic(metadata) == TextureSemantic::SignedDistanceField)
        return {};
    const auto [targetWidth, targetHeight] = FitDimensions(width, height, ReadMaxSize(metadata));
    if (width == targetWidth && height == targetHeight)
        return {};

    STBIR_RESIZE resize;
    InitPngResize(resize, width, height, targetWidth, targetHeight, metadata);
    const auto resizeScratch = detail::MeasureTextureResize(resize);
    if (!resizeScratch)
        throw std::invalid_argument("cannot plan the PNG resize workspace");

    uint64_t mipBytes = 0;
    uint64_t blockBytes = 0;
    auto w = targetWidth;
    auto h = targetHeight;
    for (;;) {
        mipBytes += LevelByteSize(w, h, TextureFormat::Rgba8UNorm);
        blockBytes += uint64_t((w + 3u) / 4u) * ((h + 3u) / 4u) * 16u;
        if (!ReadGenerateMipmaps(metadata) || (w == 1 && h == 1))
            break;
        w = std::max(1u, w / 2u);
        h = std::max(1u, h / 2u);
    }
    const uint64_t outputBytes = std::max(mipBytes, blockBytes);
    const uint64_t decodeBytes = PngRowStorage(header) + resizeScratch;
    // The mip vector is reserved once. BC encoding uses a second planned vector;
    // serialization holds the final output and its string at the same time.
    constexpr uint64_t artifactHeaderAndControlBytes = 64 * 1024;
    const uint64_t bytes =
        std::max({mipBytes + decodeBytes, mipBytes + blockBytes, 2 * outputBytes}) + artifactHeaderAndControlBytes;
    return {header, bytes, resizeScratch, static_cast<size_t>(mipBytes)};
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
            throw std::logic_error("PNG source must be opened for its admitted decode");
        const auto revision = file->InspectRevision();
        const auto plan = PlanPng(source, metadata);
        if (!plan.bytes || plan.bytes > source.SequentialStorageLimit())
            throw std::logic_error("PNG options changed after bounded admission");
        const auto [width, height] = FitDimensions(plan.source.width, plan.source.height, ReadMaxSize(metadata));
        TextureCpuData texture;
        texture.semantic = ReadSemantic(metadata);
        texture.format = ReadSrgb(metadata) ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
        const auto level = DescribeBaseLevel(width, height, texture.format);
        texture.bytes.reserve(plan.mipBytes);
        texture.bytes.resize(static_cast<size_t>(level.byteSize));
        texture.mipLevels.push_back(level);
        {
            PngRows rows(source, plan.source);
            STBIR_RESIZE resize;
            InitPngResize(resize, plan.source.width, plan.source.height, width, height, metadata, texture.bytes.data());
            stbir_set_user_data(&resize, &rows);
            if (!detail::RunTextureResize(resize, plan.resizeScratch))
                throw std::runtime_error("failed to resize PNG texture");
            rows.Finish();
            if (sequentialChannels)
                *sequentialChannels = rows.SourceChannels();
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
