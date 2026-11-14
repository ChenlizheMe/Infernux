#include "CameraViewportTarget.h"
#include "infernux/core/math/ViewportRect.h"
#include <algorithm>
#include <cmath>

namespace infernux::rhi
{
bool CameraViewportTarget::Prepare(Device &device, const std::string &identity,
                                   std::shared_ptr<const RenderTextureGeneration> output, const glm::vec4 &rect)
{
    const auto previous = m_work ? m_work->Acquire() : nullptr;
    const auto previousOutput = m_output;
    m_output = std::move(output);
    if (!m_output) {
        Reset();
        return previous != nullptr || previousOutput != nullptr;
    }
    const auto pixels = RasterViewportRect(rect, m_output->width, m_output->height);
    m_x = pixels.x;
    m_y = pixels.y;
    m_width = pixels.z;
    m_height = pixels.w;
    if (IsEmpty() || (m_x == 0 && m_y == 0 && m_width == m_output->width && m_height == m_output->height)) {
        m_work.reset();
        return previous != nullptr || previousOutput != m_output;
    }
    auto description = m_output->description;
    description.sizeMode = RenderTextureSizeMode::Absolute;
    description.width = m_width;
    description.height = m_height;
    if (!m_work)
        m_work = std::make_shared<RenderTexture>(device, identity, description);
    else
        m_work->Reconfigure(description);
    return previous != m_work->Acquire() || previousOutput != m_output;
}

void CameraViewportTarget::Reset()
{
    m_work.reset();
    m_output.reset();
    m_x = m_y = m_width = m_height = 0;
}
} // namespace infernux::rhi
