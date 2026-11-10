#include "InxTextureLoader.hpp"
#include <function/resources/AssetRegistry/AssetSource.h>
#include <function/resources/InxTexture/SvgRasterizer.h>
#include <function/resources/InxTexture/TextureDecoder.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <core/log/InxLog.h>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <platform/filesystem/InxPath.h>
#include <stb_image.h>
#include <vector>

namespace infernux
{
namespace
{

class PnmReader
{
  public:
    explicit PnmReader(std::string_view bytes) : m_bytes(bytes), m_size(bytes.size())
    {
    }
    explicit PnmReader(AssetSource &source) : m_file(source.OpenedFile()), m_size(source.Size())
    {
    }

    int Peek()
    {
        if (position == m_size)
            return EOF;
        if (!m_file)
            return static_cast<unsigned char>(m_bytes[static_cast<size_t>(position)]);
        if (position == m_windowEnd) {
            m_windowBegin = position;
            const auto count = (std::min)(uint64_t(m_window.size()), m_size - position);
            m_file->ReadAt(position, m_window.data(), static_cast<size_t>(count));
            m_windowEnd = position + count;
        }
        return m_window[static_cast<size_t>(position - m_windowBegin)];
    }

    uint64_t position = 0;

  private:
    FileInput *m_file = nullptr;
    std::string_view m_bytes;
    uint64_t m_size;
    uint64_t m_windowBegin = 0, m_windowEnd = 0;
    std::array<unsigned char, 4096> m_window;
};

bool ReadPnmInt(PnmReader &input, int &out)
{
    int next = input.Peek();
    for (;;) {
        while (std::isspace(next)) {
            ++input.position;
            next = input.Peek();
        }
        if (next != '#')
            break;
        while (next != EOF && next != '\n' && next != '\r') {
            ++input.position;
            next = input.Peek();
        }
    }
    const bool negative = next == '-';
    if (negative || next == '+') {
        ++input.position;
        next = input.Peek();
    }
    if (next < '0' || next > '9')
        return false;
    out = 0;
    while (next >= '0' && next <= '9') {
        out = out * 10 + next - '0';
        if (out > 65535)
            return false;
        ++input.position;
        next = input.Peek();
    }
    return (!negative || out == 0) && (next == EOF || next == '#' || std::isspace(next));
}

struct PnmHeader
{
    int width = 0, height = 0, maxValue = 0, channels = 0;
    bool binary = false;
};

PnmHeader ReadPnmHeader(PnmReader &input)
{
    if (input.Peek() != 'P')
        return {};
    ++input.position;
    const int magic = input.Peek();
    if (magic != '2' && magic != '3' && magic != '5' && magic != '6')
        return {};
    ++input.position;
    PnmHeader header;
    if (!ReadPnmInt(input, header.width) || !ReadPnmInt(input, header.height) || !ReadPnmInt(input, header.maxValue) ||
        !header.width || !header.height || !header.maxValue)
        return {};
    header.binary = magic == '5' || magic == '6';
    header.channels = magic == '3' || magic == '6' ? 3 : 1;
    return header;
}

TextureDecoder::SourceInfo PnmSourceInfo(PnmReader &input)
{
    const auto header = ReadPnmHeader(input);
    return {static_cast<uint32_t>(header.width), static_cast<uint32_t>(header.height),
            static_cast<uint8_t>(header.binary ? header.channels : 4)};
}

unsigned char ScalePnmSample(int value, int maxValue)
{
    return static_cast<unsigned char>((value * 255 + maxValue / 2) / maxValue);
}

bool DecodePnmToRgba(const unsigned char *data, size_t dataSize, InxTextureData &result)
{
    PnmReader input({reinterpret_cast<const char *>(data), dataSize});
    const auto [width, height, maxValue, srcChannels, binary] = ReadPnmHeader(input);
    if (!width)
        return false;

    size_t pos = static_cast<size_t>(input.position);
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const int bytesPerSample = maxValue > 255 ? 2 : 1;
    // Even ASCII needs at least one byte per sample. Reject impossible headers
    // before allocating pixels, and never expose a partially decoded image.
    if (pixelCount > (dataSize - pos) / static_cast<size_t>(srcChannels) ||
        pixelCount > (std::numeric_limits<size_t>::max)() / 4)
        return false;
    if (binary) {
        // The single separator ends the header. Subsequent whitespace and '#'
        // bytes are pixel values, not padding or comments.
        if (pos >= dataSize || !std::isspace(static_cast<unsigned char>(data[pos])))
            return false;
        ++pos;
        if (pixelCount > (dataSize - pos) / (static_cast<size_t>(srcChannels) * bytesPerSample))
            return false;
    }
    std::vector<unsigned char> pixels(pixelCount * 4);

    if (!binary) {
        for (size_t i = 0; i < pixelCount; ++i) {
            int r = 0;
            int g = 0;
            int b = 0;
            if (srcChannels == 3) {
                if (!ReadPnmInt(input, r) || !ReadPnmInt(input, g) || !ReadPnmInt(input, b)) {
                    return false;
                }
            } else {
                if (!ReadPnmInt(input, r)) {
                    return false;
                }
                g = r;
                b = r;
            }
            if (r > maxValue || g > maxValue || b > maxValue)
                return false;
            const size_t out = i * 4;
            pixels[out + 0] = ScalePnmSample(r, maxValue);
            pixels[out + 1] = ScalePnmSample(g, maxValue);
            pixels[out + 2] = ScalePnmSample(b, maxValue);
            pixels[out + 3] = 255;
        }
    } else {
        for (size_t i = 0; i < pixelCount; ++i) {
            int samples[3] = {0, 0, 0};
            for (int c = 0; c < srcChannels; ++c) {
                if (bytesPerSample == 1) {
                    samples[c] = data[pos++];
                } else {
                    samples[c] = (static_cast<int>(data[pos]) << 8) | static_cast<int>(data[pos + 1]);
                    pos += 2;
                }
                if (samples[c] > maxValue)
                    return false;
            }
            if (srcChannels == 1) {
                samples[1] = samples[0];
                samples[2] = samples[0];
            }
            const size_t out = i * 4;
            pixels[out + 0] = ScalePnmSample(samples[0], maxValue);
            pixels[out + 1] = ScalePnmSample(samples[1], maxValue);
            pixels[out + 2] = ScalePnmSample(samples[2], maxValue);
            pixels[out + 3] = 255;
        }
    }
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.pixels = std::move(pixels);
    return true;
}

} // namespace

TextureDecoder::SourceInfo detail::ReadPnmInfo(AssetSource &source)
{
    PnmReader input(source);
    return PnmSourceInfo(input);
}

TextureDecoder::SourceInfo detail::ReadPnmInfo(std::string_view bytes)
{
    PnmReader input(bytes);
    return PnmSourceInfo(input);
}

bool InxTextureLoader::IsPnmSource(const unsigned char *data, size_t dataSize)
{
    return data && dataSize >= 2 && data[0] == 'P' &&
           (data[1] == '2' || data[1] == '3' || data[1] == '5' || data[1] == '6');
}

InxTextureData InxTextureLoader::LoadFromFile(const std::string &filePath, const std::string &name, int svgMaxSize)
{
    InxTextureData result;
    result.sourcePath = filePath;
    result.name = name.empty() ? FromFsPath(ToFsPath(filePath).stem()) : name;

    // Read file bytes first to support Unicode paths on Windows
    std::vector<unsigned char> fileBytes;
    if (!ReadFileBytes(filePath, fileBytes) || fileBytes.empty()) {
        INXLOG_ERROR("Failed to read texture file: ", filePath);
        return result;
    }
    result = LoadFromMemory(fileBytes.data(), fileBytes.size(), result.name, svgMaxSize);
    result.sourcePath = filePath;
    return result;
}

InxTextureData InxTextureLoader::LoadFromMemory(const unsigned char *data, size_t dataSize, const std::string &name,
                                                int svgMaxSize)
{
    InxTextureData result;
    result.name = name;

    if (!data || dataSize == 0 || dataSize > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        INXLOG_ERROR("Texture source is empty or exceeds decoder limits: ", name);
        return result;
    }
    if (IsSvgSource(data, dataSize)) {
        if (SvgUsesFilters(data, dataSize))
            INXLOG_WARN("SVG filters are not supported and will be omitted: ", name,
                        ". Export filter effects as an embedded raster image for faithful rendering.");
        try {
            result = RasterizeSvg(data, dataSize, svgMaxSize);
            result.name = name;
        } catch (const std::exception &error) {
            INXLOG_ERROR("SVG texture decode failed: ", name, ": ", error.what());
        }
        return result;
    }
    if (IsPnmSource(data, dataSize)) {
        if (!DecodePnmToRgba(data, dataSize, result))
            INXLOG_ERROR("Failed to decode PNM texture: ", name);
        return result;
    }
    int width, height, channels;
    stbi_uc *pixels =
        stbi_load_from_memory(data, static_cast<int>(dataSize), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        INXLOG_ERROR("stbi_load_from_memory failed: ", stbi_failure_reason());
        return result;
    }

    result.width = width;
    result.height = height;
    result.channels = 4;
    size_t pixelDataSize = static_cast<size_t>(width) * height * 4;
    result.pixels.assign(pixels, pixels + pixelDataSize);

    stbi_image_free(pixels);

    return result;
}

InxTextureData InxTextureLoader::CreateSolidColor(int width, int height, unsigned char r, unsigned char g,
                                                  unsigned char b, unsigned char a, const std::string &name)
{
    InxTextureData result;
    result.name = name;
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.pixels.resize(static_cast<size_t>(width) * height * 4);

    for (int i = 0; i < width * height; ++i) {
        result.pixels[i * 4 + 0] = r;
        result.pixels[i * 4 + 1] = g;
        result.pixels[i * 4 + 2] = b;
        result.pixels[i * 4 + 3] = a;
    }

    return result;
}

InxTextureData InxTextureLoader::CreateCheckerboard(int width, int height, int checkerSize, const std::string &name)
{
    InxTextureData result;
    result.name = name;
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.pixels.resize(static_cast<size_t>(width) * height * 4);

    // Magenta and black checkerboard (classic "missing texture" pattern)
    const unsigned char color1[4] = {255, 0, 255, 255}; // Magenta
    const unsigned char color2[4] = {0, 0, 0, 255};     // Black

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int checkerX = x / checkerSize;
            int checkerY = y / checkerSize;
            const unsigned char *color = ((checkerX + checkerY) % 2 == 0) ? color1 : color2;

            int idx = (y * width + x) * 4;
            result.pixels[idx + 0] = color[0];
            result.pixels[idx + 1] = color[1];
            result.pixels[idx + 2] = color[2];
            result.pixels[idx + 3] = color[3];
        }
    }

    return result;
}

} // namespace infernux
