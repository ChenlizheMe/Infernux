#pragma once

#include <string>
#include <string_view>
#include <utility>

namespace infernux
{
// One immutable source revision. Jobs retain this object while the registry
// may invalidate its cache and publish a different revision of the same GUID.
class AssetBytes final
{
  public:
    explicit AssetBytes(std::string bytes) : m_bytes(std::move(bytes))
    {
    }
    [[nodiscard]] std::string_view View() const noexcept
    {
        return m_bytes;
    }
    [[nodiscard]] size_t OwnedBytes() const noexcept
    {
        return sizeof(*this) + m_bytes.capacity() + 1;
    }

  private:
    const std::string m_bytes;
};
} // namespace infernux
