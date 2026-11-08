#pragma once

#include "RhiRenderTexture.h"
#include <glm/glm.hpp>

namespace infernux::rhi
{
/// A camera-local work allocation and its clipped destination in an output.
/// The allocation is shared across frames; moving a viewport without resizing
/// it never allocates. All attachment lifetimes use normal RenderTexture
/// generations and device retirement.
class CameraViewportTarget
{
  public:
    bool Prepare(Device &device, const std::string &identity, std::shared_ptr<const RenderTextureGeneration> output,
                 const glm::vec4 &rect);
    void Reset();
    [[nodiscard]] bool IsPartial() const
    {
        return m_work != nullptr;
    }
    [[nodiscard]] bool IsEmpty() const
    {
        return m_width == 0 || m_height == 0;
    }
    [[nodiscard]] uint32_t X() const
    {
        return m_x;
    }
    [[nodiscard]] uint32_t Y() const
    {
        return m_y;
    }
    [[nodiscard]] uint32_t Width() const
    {
        return m_width;
    }
    [[nodiscard]] uint32_t Height() const
    {
        return m_height;
    }
    [[nodiscard]] const auto &Output() const
    {
        return m_output;
    }
    [[nodiscard]] const auto &Work() const
    {
        return m_work;
    }

  private:
    std::shared_ptr<RenderTexture> m_work;
    std::shared_ptr<const RenderTextureGeneration> m_output;
    uint32_t m_x = 0, m_y = 0, m_width = 0, m_height = 0;
};
} // namespace infernux::rhi
