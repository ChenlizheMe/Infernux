#pragma once

#include <core/config/InxPlatform.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <system_error>

namespace infernux
{
class FrameWorkOwner;
class OverlappedFile;

namespace detail
{
/// Native, exclusive sibling output. Naming and ownership start before IO.
class AtomicFileOutput final
{
  public:
    enum class Access
    {
        Write,
        ReadWrite
    };

    explicit AtomicFileOutput(const std::filesystem::path &target, uint64_t completeBytes = 0, bool namedStage = false);
    ~AtomicFileOutput();
    AtomicFileOutput(const AtomicFileOutput &) = delete;
    AtomicFileOutput &operator=(const AtomicFileOutput &) = delete;

    const std::filesystem::path &Path() const noexcept
    {
        return m_path;
    }
    size_t PathStorageBytes() const noexcept;
    void Open(Access access = Access::Write);
    void Write(std::string_view content, const FrameWorkOwner *storageOwner = nullptr);
    void Seek(uint64_t offset);
    void Read(void *destination, size_t bytes);
    void Finish();
    void Remove();
    /// Transfer a finished named stage to its transaction.
    void Release() noexcept;
    /// The atomic rename has consumed the private name. Persist the directory entry.
    std::error_code Published() noexcept;
    bool UsesDirectOutput() const noexcept;

  private:
    std::error_code Close() noexcept;

    std::filesystem::path m_path;
    bool m_private = false;
#ifdef INX_PLATFORM_WINDOWS
    HANDLE m_handle = INVALID_HANDLE_VALUE;
    uint64_t m_directBytes = 0;
    std::shared_ptr<OverlappedFile> m_ranges;
#else
    const std::filesystem::path m_leaf;
    int m_directory = -1;
    int m_descriptor = -1;
#endif
};
} // namespace detail
} // namespace infernux
