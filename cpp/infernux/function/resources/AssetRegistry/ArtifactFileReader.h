#pragma once

#include <core/serialization/BinaryCodec.h>
#include <platform/filesystem/InxPath.h>

#include <array>
#include <vector>

namespace infernux
{

namespace artifact_layout
{
constexpr uint32_t LittleEndian = 0x01020304U;

inline void AppendSignature(std::string &bytes, std::string_view magic, uint32_t schema)
{
    bytes.append(magic);
    binary_codec::AppendU32(bytes, LittleEndian);
    binary_codec::AppendU32(bytes, schema);
}

inline void AppendHeader(std::string &bytes, std::string_view magic, uint32_t schema, uint64_t workspace)
{
    AppendSignature(bytes, magic, schema);
    binary_codec::AppendU64(bytes, workspace);
}

template <typename Reader> bool ReadSignature(Reader &reader, std::string_view magic, uint32_t schema)
{
    return reader.ReadPrefix(magic) && reader.ReadU32() == LittleEndian && reader.ReadU32() == schema;
}

inline bool HasHeader(std::string_view bytes, std::string_view magic, uint32_t schema) noexcept
{
    if (bytes.size() < magic.size() + 2 * sizeof(uint32_t) + sizeof(uint64_t))
        return false;
    binary_codec::Reader<0> reader(bytes);
    return ReadSignature(reader, magic, schema);
}

template <typename Reader>
uint64_t ReadHeader(Reader &reader, std::string_view magic, uint32_t schema, uint64_t workspaceLimit = UINT64_MAX)
{
    if (!ReadSignature(reader, magic, schema))
        throw std::invalid_argument("invalid cooked asset layout");
    const auto workspace = reader.ReadU64();
    if (workspace > workspaceLimit)
        throw std::invalid_argument("cooked asset workspace changed after admission");
    return workspace;
}

inline uint64_t ReadWorkspace(FileInput &file, std::string_view magic, uint32_t schema)
{
    std::string header(magic.size() + 2 * sizeof(uint32_t) + sizeof(uint64_t), '\0');
    file.ReadAt(0, header.data(), header.size());
    binary_codec::Reader<0> reader(header);
    return ReadHeader(reader, magic, schema);
}

} // namespace artifact_layout

/// A sequential file window for the same binary parser used by in-memory artifacts.
/// Hash each input block as it is read and check the footer before publication.
class ArtifactFileReader
{
  public:
    static constexpr size_t BlockBytes = 64u * 1024u;

    ArtifactFileReader(FileInput &source, uint64_t maximumBytes, std::vector<char> &block)
        : m_source(source), m_payloadBytes(PayloadSize(source, maximumBytes)), m_block(block)
    {
        if (block.empty())
            throw std::invalid_argument("artifact file read requires a nonempty block");
    }

    template <typename Reader> Reader Begin()
    {
        return Reader(m_payloadBytes, [this] {
            const size_t count = (std::min)(m_block.size(), m_payloadBytes - m_offset);
            m_source.ReadAt(m_offset, m_block.data(), count);
            m_offset += count;
            const std::string_view bytes(m_block.data(), count);
            m_hash = binary_codec::Fnv1a64(bytes, m_hash);
            return bytes;
        });
    }

    void Verify()
    {
        std::array<char, sizeof(uint64_t)> footer;
        m_source.ReadAt(m_payloadBytes, footer.data(), footer.size());
        if (binary_codec::Reader<0>(std::string_view(footer.data(), footer.size())).ReadU64() != m_hash)
            throw std::invalid_argument("artifact checksum mismatch");
    }

  private:
    static size_t PayloadSize(FileInput &source, uint64_t maximum)
    {
        const uint64_t bytes = source.Size();
        if (bytes < sizeof(uint64_t) || bytes > maximum)
            throw std::invalid_argument("artifact size exceeds its format limits");
        return static_cast<size_t>(bytes - sizeof(uint64_t));
    }

    FileInput &m_source;
    size_t m_payloadBytes;
    std::vector<char> &m_block;
    size_t m_offset = 0;
    uint64_t m_hash = binary_codec::Fnv1a64Basis;
};

} // namespace infernux
