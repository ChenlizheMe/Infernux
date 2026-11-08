#pragma once

#define XXH_INLINE_ALL
#include <zstd_for_infernux/lib/common/xxhash.h>

#include <cstdint>
#include <string_view>

namespace infernux::detail
{
class FileFingerprint
{
  public:
    FileFingerprint()
    {
        XXH64_reset(&m_state, 0);
    }
    void Append(std::string_view bytes)
    {
        XXH64_update(&m_state, bytes.data(), bytes.size());
    }
    uint64_t Value() const
    {
        return XXH64_digest(&m_state);
    }
    static uint64_t Of(std::string_view bytes)
    {
        return XXH64(bytes.data(), bytes.size(), 0);
    }

  private:
    XXH64_state_t m_state{};
};
} // namespace infernux::detail
