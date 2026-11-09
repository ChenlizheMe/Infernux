#include "AssetSource.h"
#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <platform/filesystem/InxPath.h>
#include <sstream>

namespace infernux
{
namespace
{
constexpr uint64_t InitialHash = 14695981039346656037ULL;

uint64_t HashBytes(std::string_view bytes, uint64_t hash)
{
    for (const unsigned char value : bytes) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}
} // namespace

AssetSource::AssetSource(std::string path, ReadMode mode)
    : m_path(std::move(path)),
      m_input(std::make_unique<FileInput>(ToFsPath(m_path), mode == ReadMode::Retained ? FileInput::Sharing::Snapshot
                                                                                       : FileInput::Sharing::ReadWrite))
{
    const auto state = m_input->Inspect();
    m_size = state.size;
    if (mode == ReadMode::Retained)
        m_modified = state.modified;
}

AssetSource::AssetSource(std::string path, std::string_view bytes)
    : m_path(std::move(path)), m_size(bytes.size()), m_view(bytes)
{
}

AssetSource::~AssetSource() = default;

std::string_view AssetSource::Bytes()
{
    if (IsSequential())
        throw std::logic_error("A sequential asset source cannot be materialized");
    ReadNext((std::numeric_limits<size_t>::max)());
    return m_view;
}

void AssetSource::BeginSequentialRead(uint64_t storageLimit)
{
    if (!m_input || m_bytes || m_hash || IsSequential() || !storageLimit)
        throw std::logic_error("A sequential asset source requires an unconsumed file");
    m_sequentialHash = InitialHash;
    m_sequentialStorageLimit = storageLimit;
}

void AssetSource::OpenSequentialRead()
{
    if (!IsSequential() || m_input || m_sequentialOffset)
        throw std::logic_error("Sequential asset source cannot be reopened after reading");
    m_input = std::make_unique<FileInput>(ToFsPath(m_path));
}

void AssetSource::CloseFile()
{
    if (!IsSequential() && m_view.size() != m_size)
        throw std::logic_error("Cannot close a partially materialized asset source");
    m_input.reset();
}

void AssetSource::ReadSequential(void *destination, size_t bytes)
{
    if (!IsSequential() || !m_input || bytes > m_size - m_sequentialOffset)
        throw std::logic_error("Sequential asset read exceeds the source");
    m_input->ReadAt(m_sequentialOffset, destination, bytes);
    *m_sequentialHash = HashBytes({static_cast<const char *>(destination), bytes}, *m_sequentialHash);
    m_sequentialOffset += bytes;
}

bool AssetSource::ReadNext(size_t maximum)
{
    if (!m_input)
        return true;
    if (!m_bytes) {
        if (m_size > std::string{}.max_size())
            throw std::length_error("asset source is too large to materialize");
        std::string bytes;
        bytes.reserve(static_cast<size_t>(m_size));
        m_bytes.emplace(std::move(bytes));
    }
    const auto offset = m_bytes->size();
    const auto count = static_cast<size_t>((std::min<uint64_t>)(m_size - offset, maximum));
    if (count) {
        m_bytes->resize(offset + count);
        try {
            m_input->ReadAt(offset, m_bytes->data() + offset, count);
            CheckRevision();
        } catch (...) {
            m_bytes->resize(offset);
            throw;
        }
    }
    if (m_bytes->size() != m_size)
        return false;
    m_view = *m_bytes;
    return true;
}

void AssetSource::CheckRevision()
{
    if (!m_modified)
        return;
    const auto state = m_input->Inspect();
    if (state.size != m_size || state.modified != *m_modified)
        throw std::runtime_error("Asset source changed while being read: " + m_path);
}

std::string AssetSource::TakeBytes() &&
{
    (void)Bytes();
    std::string bytes = m_bytes ? std::move(*m_bytes) : std::string(m_view);
    m_input.reset();
    m_bytes.reset();
    m_view = {};
    m_size = 0;
    m_hash.reset();
    return bytes;
}

const std::string &AssetSource::ContentHash()
{
    if (!m_hash) {
        uint64_t hash = InitialHash;
        if (IsSequential()) {
            if (m_sequentialOffset != m_size)
                throw std::logic_error("Sequential asset source has not reached the end of the file");
            hash = *m_sequentialHash;
        } else if (!m_input || m_view.size() == m_size) {
            hash = HashBytes(m_view, hash);
        } else {
            std::array<char, 64 * 1024> block;
            for (uint64_t offset = 0; offset < m_size;) {
                const auto count = static_cast<size_t>(std::min<uint64_t>(block.size(), m_size - offset));
                m_input->ReadAt(offset, block.data(), count);
                hash = HashBytes({block.data(), count}, hash);
                offset += count;
            }
            CheckRevision();
        }
        std::ostringstream text;
        text << std::hex << std::setfill('0') << std::setw(16) << hash;
        m_hash = text.str();
    }
    return *m_hash;
}
} // namespace infernux
