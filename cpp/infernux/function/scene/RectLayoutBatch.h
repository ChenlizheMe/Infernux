#pragma once

#include "RectTransformLayout.h"
#include <string>
#include <vector>

namespace infernux
{
// Resolve field dispatch once. Scene reads/writes remain on the owner thread;
// pure numeric snapshots have separate, independently schedulable kernels.
class RectLayoutBatch
{
  public:
    RectLayoutBatch(std::vector<RectTransformLayout> layouts, const std::string &field);
    size_t Size() const noexcept
    {
        return m_layouts.size();
    }
    size_t Channels() const noexcept
    {
        return m_channels;
    }
    void Validate() const;
    std::array<float, 3> Read(size_t row) const
    {
        return m_read(m_layouts[row]);
    }
    void Write(size_t row, const std::array<float, 3> &value)
    {
        m_write(m_layouts[row], value);
    }

  private:
    using Value = std::array<float, 3>;
    using Reader = Value (*)(const RectTransformLayout &);
    using Writer = void (*)(RectTransformLayout &, const Value &);
    std::vector<RectTransformLayout> m_layouts;
    Reader m_read;
    Writer m_write;
    size_t m_channels;
};
} // namespace infernux
