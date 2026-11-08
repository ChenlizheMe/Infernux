#include "TextureArtifact.h"

#include <core/serialization/BinaryCodec.h>
#include <platform/filesystem/FileFingerprint.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace infernux
{
namespace
{
constexpr std::string_view Magic = "INXTEXTUREX";
constexpr uint32_t EndianMarker = 0x01020304U;
constexpr uint32_t MaximumDimension = 65'536;
constexpr uint32_t MaximumMipLevels = 32;
constexpr uint64_t MaximumPayloadBytes = 1ULL << 30;
constexpr uint32_t MaximumHashBytes = 1024;

using binary_codec::AppendU32;

using binary_codec::AppendU64;

using binary_codec::AppendFloat;

void AppendString(std::string &out, std::string_view value)
{
    if (value.empty() || value.size() > MaximumHashBytes)
        throw std::invalid_argument("texture artifact requires a bounded source content hash");
    AppendU32(out, static_cast<uint32_t>(value.size()));
    out.append(value);
}

using Reader = binary_codec::Reader<MaximumHashBytes, false>;

bool IsValidDimension(TextureDimension dimension)
{
    return dimension == TextureDimension::Texture2D || dimension == TextureDimension::Texture3D;
}

bool IsValidSemantic(TextureSemantic semantic)
{
    return semantic >= TextureSemantic::Color && semantic <= TextureSemantic::SignedDistanceField;
}

bool IsValidFormat(TextureFormat format)
{
    return GetTextureFormatInfo(format).pixelFormat != rhi::PixelFormat::Undefined;
}

struct MipLayout
{
    uint64_t rowPitch = 0;
    uint64_t slicePitch = 0;
    uint64_t byteSize = 0;
};

MipLayout ComputeMipLayout(uint32_t width, uint32_t height, uint32_t depth, TextureFormat format)
{
    const auto layout = GetTextureFormatLayout(format);
    if (!layout.bytesPerBlock)
        throw std::invalid_argument("texture artifact has an unsupported concrete format");
    const uint64_t rowPitch = layout.RowBytes(width);
    const uint64_t rows = layout.RowCount(height);
    if (rows != 0 && rowPitch > MaximumPayloadBytes / rows)
        throw std::invalid_argument("texture artifact mip row layout exceeds the payload limit");
    const uint64_t slicePitch = rowPitch * rows;
    if (depth != 0 && slicePitch > MaximumPayloadBytes / depth)
        throw std::invalid_argument("texture artifact mip volume exceeds the payload limit");
    return {rowPitch, slicePitch, slicePitch * depth};
}

void ValidateTexture(const TextureCpuData &texture)
{
    if (!texture.IsValid() || !IsValidDimension(texture.dimension) || !IsValidSemantic(texture.semantic) ||
        !IsValidFormat(texture.format) || texture.mipLevels.size() > MaximumMipLevels ||
        texture.bytes.size() > MaximumPayloadBytes)
        throw std::invalid_argument("texture artifact has invalid payload dimensions");
    if (!std::all_of(texture.bakeBasis.begin(), texture.bakeBasis.end(),
                     [](float value) { return std::isfinite(value); }) ||
        !std::all_of(texture.valueMin.begin(), texture.valueMin.end(),
                     [](float value) { return std::isfinite(value); }) ||
        !std::all_of(texture.valueMax.begin(), texture.valueMax.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("texture artifact metadata must contain finite values");
    for (size_t channel = 0; channel < texture.valueMin.size(); ++channel) {
        if (texture.valueMin[channel] > texture.valueMax[channel])
            throw std::invalid_argument("texture artifact value range is inverted");
    }
    uint64_t expectedOffset = 0;
    uint32_t previousWidth = 0;
    uint32_t previousHeight = 0;
    uint32_t previousDepth = 0;
    for (size_t index = 0; index < texture.mipLevels.size(); ++index) {
        const auto &mip = texture.mipLevels[index];
        if (mip.width == 0 || mip.height == 0 || mip.depth == 0 || mip.width > MaximumDimension ||
            mip.height > MaximumDimension || mip.depth > MaximumDimension ||
            (texture.dimension == TextureDimension::Texture2D && mip.depth != 1))
            throw std::invalid_argument("texture artifact has an invalid mip dimension");
        if (index > 0 &&
            (mip.width != (std::max)(1U, previousWidth / 2U) || mip.height != (std::max)(1U, previousHeight / 2U) ||
             mip.depth != (texture.dimension == TextureDimension::Texture3D ? (std::max)(1U, previousDepth / 2U) : 1U)))
            throw std::invalid_argument("texture artifact has a non-contiguous mip chain");
        const MipLayout expected = ComputeMipLayout(mip.width, mip.height, mip.depth, texture.format);
        if (mip.byteOffset != expectedOffset || mip.byteSize != expected.byteSize ||
            mip.rowPitch != expected.rowPitch || mip.slicePitch != expected.slicePitch ||
            expected.byteSize > MaximumPayloadBytes - expectedOffset)
            throw std::invalid_argument("texture artifact has an invalid mip byte range");
        expectedOffset += expected.byteSize;
        previousWidth = mip.width;
        previousHeight = mip.height;
        previousDepth = mip.depth;
    }
    if (expectedOffset != texture.bytes.size())
        throw std::invalid_argument("texture artifact payload size does not match its mip chain");
}
// A complete maximum-size header is small. Read it once, parse with the same
// codec as memory artifacts, and retain the open FileInput for later blocks.
constexpr size_t MaximumHeaderBytes = Magic.size() + 4 + 4 + MaximumHashBytes + 12 + 96 + 4 + MaximumMipLevels * 36 + 8;

struct ArtifactLayout
{
    TextureArtifactDescription description;
    size_t payloadOffset;
};

ArtifactLayout ParseLayout(std::string_view prefix, uint64_t fileSize, std::string_view expectedSourceContentHash)
{
    Reader reader(prefix);
    if (!reader.ReadPrefix(Magic))
        throw std::invalid_argument("texture artifact has an invalid header");
    if (reader.ReadU32() != EndianMarker)
        throw std::invalid_argument("texture artifact has an invalid endian marker");
    if (reader.ReadString() != expectedSourceContentHash)
        throw std::invalid_argument("texture artifact does not match the imported source content");
    TextureArtifactDescription description;
    description.dimension = static_cast<TextureDimension>(reader.ReadU32());
    description.semantic = static_cast<TextureSemantic>(reader.ReadU32());
    description.format = static_cast<TextureFormat>(reader.ReadU32());
    if (!IsValidDimension(description.dimension) || !IsValidSemantic(description.semantic) ||
        !IsValidFormat(description.format))
        throw std::invalid_argument("texture artifact description has invalid type metadata");
    for (float &value : description.bakeBasis)
        value = reader.ReadFloat();
    for (float &value : description.valueMin)
        value = reader.ReadFloat();
    for (float &value : description.valueMax)
        value = reader.ReadFloat();
    for (size_t channel = 0; channel < description.valueMin.size(); ++channel) {
        if (description.valueMin[channel] > description.valueMax[channel])
            throw std::invalid_argument("texture artifact value range is inverted");
    }

    const uint32_t mipCount = reader.ReadU32();
    if (mipCount == 0 || mipCount > MaximumMipLevels)
        throw std::invalid_argument("texture artifact has an invalid mip count");
    description.mipLevels.reserve(mipCount);
    uint64_t offset = 0;
    uint32_t previousWidth = 0;
    uint32_t previousHeight = 0;
    uint32_t previousDepth = 0;
    for (uint32_t index = 0; index < mipCount; ++index) {
        TextureMipLevel mip;
        mip.width = reader.ReadU32();
        mip.height = reader.ReadU32();
        mip.depth = reader.ReadU32();
        mip.byteOffset = offset;
        mip.byteSize = reader.ReadU64();
        mip.rowPitch = reader.ReadU64();
        mip.slicePitch = reader.ReadU64();
        if (mip.width == 0 || mip.height == 0 || mip.depth == 0 || mip.width > MaximumDimension ||
            mip.height > MaximumDimension || mip.depth > MaximumDimension ||
            (description.dimension == TextureDimension::Texture2D && mip.depth != 1))
            throw std::invalid_argument("texture artifact has an invalid mip dimension");
        if (index > 0 &&
            (mip.width != (std::max)(1U, previousWidth / 2U) || mip.height != (std::max)(1U, previousHeight / 2U) ||
             mip.depth !=
                 (description.dimension == TextureDimension::Texture3D ? (std::max)(1U, previousDepth / 2U) : 1U)))
            throw std::invalid_argument("texture artifact has a non-contiguous mip chain");
        const MipLayout expected = ComputeMipLayout(mip.width, mip.height, mip.depth, description.format);
        if (mip.byteSize != expected.byteSize || mip.rowPitch != expected.rowPitch ||
            mip.slicePitch != expected.slicePitch || expected.byteSize > MaximumPayloadBytes - offset)
            throw std::invalid_argument("texture artifact has an invalid mip byte range");
        offset += expected.byteSize;
        previousWidth = mip.width;
        previousHeight = mip.height;
        previousDepth = mip.depth;
        description.mipLevels.push_back(mip);
    }
    description.payloadBytes = reader.ReadU64();
    if (description.payloadBytes != offset || description.payloadBytes > MaximumPayloadBytes)
        throw std::invalid_argument("texture artifact has an invalid payload size");
    const size_t payloadOffset = reader.Cursor();
    if (payloadOffset + description.payloadBytes + sizeof(uint64_t) != fileSize)
        throw std::invalid_argument("texture artifact file size does not match its description");
    return {std::move(description), payloadOffset};
}

ArtifactLayout ReadLayout(FileInput &file, std::string_view sourceHash, detail::FileFingerprint *fingerprint = nullptr)
{
    const auto fileSize = file.Size();
    std::string prefix(static_cast<size_t>((std::min)(fileSize, uint64_t(MaximumHeaderBytes))), '\0');
    file.ReadAt(0, prefix.data(), prefix.size());
    auto layout = ParseLayout(prefix, fileSize, sourceHash);
    if (fingerprint)
        fingerprint->Append(std::string_view(prefix).substr(0, layout.payloadOffset));
    return layout;
}

void ReadPayload(FileInput &file, uint64_t offset, uint8_t *output, size_t bytes, detail::FileFingerprint &fingerprint)
{
    file.ReadAt(offset, output, bytes);
    fingerprint.Append({reinterpret_cast<const char *>(output), bytes});
}

void VerifyFooter(std::string_view footer, uint64_t checksum)
{
    if (Reader(footer).ReadU64() != checksum)
        throw std::invalid_argument("texture artifact checksum mismatch");
}

void VerifyFooter(FileInput &file, uint64_t offset, const detail::FileFingerprint &fingerprint)
{
    std::array<char, sizeof(uint64_t)> footer{};
    file.ReadAt(offset, footer.data(), footer.size());
    VerifyFooter({footer.data(), footer.size()}, fingerprint.Value());
}

std::shared_ptr<TextureCpuData> MakeCpuData(TextureArtifactDescription &&description)
{
    auto texture = std::make_shared<TextureCpuData>();
    texture->dimension = description.dimension;
    texture->semantic = description.semantic;
    texture->format = description.format;
    texture->mipLevels = std::move(description.mipLevels);
    texture->bakeBasis = description.bakeBasis;
    texture->valueMin = description.valueMin;
    texture->valueMax = description.valueMax;
    return texture;
}

} // namespace

TextureArtifactDescription TextureArtifact::ReadDescription(const std::string &artifactPath,
                                                            std::string_view expectedSourceContentHash)
{
    FileInput file(ToFsPath(artifactPath));
    return ReadLayout(file, expectedSourceContentHash).description;
}

struct TextureReadCursor::Impl
{
    std::string path, sourceHash;
    std::unique_ptr<FileInput> file;
    TextureArtifactDescription description;
    std::vector<uint8_t> output;
    uint64_t payloadOffset = 0, readBytes = 0;
    detail::FileFingerprint hash;
    TextureReadState state = TextureReadState::Opening;
};

TextureReadCursor::TextureReadCursor(std::string artifactPath, std::string sourceContentHash)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->path = std::move(artifactPath);
    m_impl->sourceHash = std::move(sourceContentHash);
}
TextureReadCursor::~TextureReadCursor() = default;
TextureReadState TextureReadCursor::GetState() const noexcept
{
    return m_impl->state;
}
size_t TextureReadCursor::GetBufferedBytes() const noexcept
{
    return m_impl->output.size();
}

IoWork TextureReadCursor::DescribeWork(size_t maximum) const
{
    const auto &s = *m_impl;
    switch (s.state) {
    case TextureReadState::Opening:
        return {"texture.open"};
    case TextureReadState::Reading:
        return IoByteWork("texture.read", s.description.payloadBytes - s.readBytes, maximum);
    case TextureReadState::Closing:
        return {"texture.close"};
    default:
        return {};
    }
}

void TextureReadCursor::Advance(size_t maximum, size_t batchBytes)
{
    auto &s = *m_impl;
    switch (s.state) {
    case TextureReadState::Opening: {
        auto file = std::make_unique<FileInput>(ToFsPath(s.path));
        auto layout = ReadLayout(*file, s.sourceHash, &s.hash);
        s.payloadOffset = layout.payloadOffset;
        s.description = std::move(layout.description);
        s.file = std::move(file);
        s.state = TextureReadState::Reading;
        return;
    }
    case TextureReadState::Reading: {
        const auto offset = s.output.size();
        const auto work = DescribeWork((std::min)(maximum, batchBytes - offset));
        if (!work.bytes)
            throw std::invalid_argument("texture read step requires a positive byte allowance");
        if (batchBytes > s.output.capacity()) {
            std::vector<uint8_t>().swap(s.output);
            s.output.reserve(batchBytes);
        }
        s.output.resize(offset + work.bytes);
        ReadPayload(*s.file, s.payloadOffset + s.readBytes, s.output.data() + offset, work.bytes, s.hash);
        s.readBytes += work.bytes;
        if (s.output.size() == batchBytes || s.readBytes == s.description.payloadBytes)
            s.state = TextureReadState::Output;
        return;
    }
    case TextureReadState::Closing: {
        VerifyFooter(*s.file, s.payloadOffset + s.description.payloadBytes, s.hash);
        s.file.reset();
        std::vector<uint8_t>().swap(s.output);
        s.state = TextureReadState::Complete;
        return;
    }
    default:
        throw std::logic_error("texture cursor has no IO step ready");
    }
}

const TextureArtifactDescription &TextureReadCursor::GetDescription() const
{
    if (!m_impl->description.IsValid())
        throw std::logic_error("texture description is not ready");
    return m_impl->description;
}
const std::vector<uint8_t> &TextureReadCursor::GetOutput() const
{
    if (m_impl->state != TextureReadState::Output)
        throw std::logic_error("texture output is not ready");
    return m_impl->output;
}
void TextureReadCursor::ConsumeOutput()
{
    auto &s = *m_impl;
    if (s.state != TextureReadState::Output)
        throw std::logic_error("texture output is not ready");
    s.output.clear();
    s.state = s.readBytes == s.description.payloadBytes ? TextureReadState::Closing : TextureReadState::Reading;
}
void TextureReadCursor::Close()
{
    auto &s = *m_impl;
    s.file.reset();
    std::vector<uint8_t>().swap(s.output);
    s.state = TextureReadState::Closed;
}
size_t TextureReadCursor::GetRetainedBytes() const noexcept
{
    const auto &s = *m_impl;
    return sizeof(*this) + sizeof(Impl) + s.path.capacity() + s.sourceHash.capacity() +
           s.description.mipLevels.capacity() * sizeof(TextureMipLevel) + s.output.capacity() +
           (s.file ? sizeof(FileInput) : 0);
}
size_t TextureReadCursor::GetReservationBytes(size_t batchBytes) const
{
    const auto &s = *m_impl;
    if (s.state == TextureReadState::Opening)
        return GetRetainedBytes() + sizeof(FileInput) + MaximumHeaderBytes + MaximumHashBytes +
               MaximumMipLevels * sizeof(TextureMipLevel);
    if (s.state == TextureReadState::Reading)
        return GetRetainedBytes() + (std::max)(s.output.capacity(), batchBytes) - s.output.capacity();
    return GetRetainedBytes();
}

std::shared_ptr<const TextureCpuData> TextureArtifact::Read(const std::string &artifactPath,
                                                            std::string_view expectedSourceContentHash)
{
    FileInput file(ToFsPath(artifactPath));
    detail::FileFingerprint fingerprint;
    auto layout = ReadLayout(file, expectedSourceContentHash, &fingerprint);
    const auto bytes = static_cast<size_t>(layout.description.payloadBytes);
    auto texture = MakeCpuData(std::move(layout.description));
    texture->bytes.resize(bytes);
    ReadPayload(file, layout.payloadOffset, texture->bytes.data(), bytes, fingerprint);
    VerifyFooter(file, layout.payloadOffset + bytes, fingerprint);
    return texture;
}

std::string TextureArtifact::Serialize(const TextureCpuData &texture, std::string_view sourceContentHash)
{
    ValidateTexture(texture);
    std::string bytes(Magic);
    AppendU32(bytes, EndianMarker);
    AppendString(bytes, sourceContentHash);
    AppendU32(bytes, static_cast<uint32_t>(texture.dimension));
    AppendU32(bytes, static_cast<uint32_t>(texture.semantic));
    AppendU32(bytes, static_cast<uint32_t>(texture.format));
    for (float value : texture.bakeBasis)
        AppendFloat(bytes, value);
    for (float value : texture.valueMin)
        AppendFloat(bytes, value);
    for (float value : texture.valueMax)
        AppendFloat(bytes, value);
    AppendU32(bytes, static_cast<uint32_t>(texture.mipLevels.size()));
    for (const auto &mip : texture.mipLevels) {
        AppendU32(bytes, mip.width);
        AppendU32(bytes, mip.height);
        AppendU32(bytes, mip.depth);
        AppendU64(bytes, mip.byteSize);
        AppendU64(bytes, mip.rowPitch);
        AppendU64(bytes, mip.slicePitch);
    }
    AppendU64(bytes, texture.bytes.size());
    bytes.reserve(bytes.size() + texture.bytes.size() + sizeof(uint64_t));
    bytes.append(reinterpret_cast<const char *>(texture.bytes.data()), texture.bytes.size());
    AppendU64(bytes, detail::FileFingerprint::Of(bytes));
    return bytes;
}

std::shared_ptr<const TextureCpuData> TextureArtifact::Deserialize(std::string_view bytes,
                                                                   std::string_view expectedSourceContentHash)
{
    if (bytes.size() < Magic.size() + sizeof(uint32_t) * 3 + sizeof(uint64_t) * 2 ||
        bytes.substr(0, Magic.size()) != Magic)
        throw std::invalid_argument("texture artifact has an invalid header");
    const size_t checksumOffset = bytes.size() - sizeof(uint64_t);
    VerifyFooter(bytes.substr(checksumOffset), detail::FileFingerprint::Of(bytes.substr(0, checksumOffset)));

    auto layout = ParseLayout(bytes, bytes.size(), expectedSourceContentHash);
    const auto payload = bytes.substr(layout.payloadOffset, static_cast<size_t>(layout.description.payloadBytes));
    auto texture = MakeCpuData(std::move(layout.description));
    texture->bytes.assign(payload.begin(), payload.end());
    return texture;
}

bool TextureArtifact::HasCurrentHeader(std::string_view bytes) noexcept
{
    return bytes.size() >= Magic.size() + sizeof(uint32_t) && bytes.substr(0, Magic.size()) == Magic &&
           static_cast<unsigned char>(bytes[Magic.size()]) == 0x04 &&
           static_cast<unsigned char>(bytes[Magic.size() + 1]) == 0x03 &&
           static_cast<unsigned char>(bytes[Magic.size() + 2]) == 0x02 &&
           static_cast<unsigned char>(bytes[Magic.size() + 3]) == 0x01;
}

} // namespace infernux
