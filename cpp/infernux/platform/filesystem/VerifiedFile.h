#pragma once

#include "core/FoundationApi.h"

#include "InxPath.h"
#include <memory>
#include <stdexcept>

namespace infernux
{
class VerifiedFile;
}
namespace infernux::inxpack
{
class ReadCursor;
INFERNUX_FOUNDATION_API std::shared_ptr<VerifiedFile>
CopyVerifiedFile(const std::filesystem::path &, const std::filesystem::path &, const std::string &);
} // namespace infernux::inxpack

namespace infernux
{
/// The completed file sink's content fact and actual open-file lifetime.
/// Only native verification can construct it; Python cannot substitute metadata.
class VerifiedFile
{
  public:
    const std::filesystem::path &Path() const
    {
        return m_path;
    }
    const std::string &Sha256() const
    {
        return m_sha256;
    }
    uint64_t Bytes() const
    {
        return m_bytes;
    }
    size_t RetainedBytes() const
    {
        return sizeof(*this) + sizeof(FileInput) +
               m_path.native().capacity() * sizeof(std::filesystem::path::value_type) + m_sha256.capacity() + 1;
    }
    void ReadAt(uint64_t offset, void *output, size_t bytes) const
    {
        if (offset > m_bytes || bytes > m_bytes - offset)
            throw std::out_of_range("verified file read exceeds its content");
        m_source->ReadAt(offset, output, bytes);
    }
#if defined(INX_PLATFORM_WINDOWS) || defined(INX_PLATFORM_LINUX)
    std::pair<uint64_t, uint64_t> Identity() const
    {
        return m_identity;
    }
#endif
#ifdef INX_PLATFORM_LINUX
    int Descriptor() const
    {
        return m_source->Descriptor();
    }
#endif

  private:
    friend class inxpack::ReadCursor;
    friend INFERNUX_FOUNDATION_API std::shared_ptr<VerifiedFile>
    inxpack::CopyVerifiedFile(const std::filesystem::path &, const std::filesystem::path &, const std::string &);
    VerifiedFile(std::filesystem::path path, uint64_t bytes, std::string sha256, std::shared_ptr<FileInput> source,
                 std::shared_ptr<void> anchors)
        : m_path(std::move(path)), m_sha256(std::move(sha256)), m_bytes(bytes), m_source(std::move(source)),
          m_anchors(std::move(anchors))
    {
#ifdef INX_PLATFORM_WINDOWS
        m_identity = m_source->Identity();
#elif defined(INX_PLATFORM_LINUX)
        m_identity = m_source->InspectRevision().identity;
#endif
    }
    std::filesystem::path m_path;
    std::string m_sha256;
    uint64_t m_bytes;
    std::shared_ptr<FileInput> m_source;
    std::shared_ptr<void> m_anchors;
#if defined(INX_PLATFORM_WINDOWS) || defined(INX_PLATFORM_LINUX)
    std::pair<uint64_t, uint64_t> m_identity;
#endif
};
} // namespace infernux
