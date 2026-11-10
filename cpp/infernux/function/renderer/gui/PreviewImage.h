#pragma once

#include <memory>

namespace infernux
{
namespace rhi
{
class TextureGpuView;
}
namespace vk
{
class GraphicsSubmissionTicket;
}

// A private result of the original graphics submission. Allocation alone does
// not publish pixels; the request also checks its source generation at commit.
struct PreviewImage
{
    std::shared_ptr<const rhi::TextureGpuView> texture;
    std::shared_ptr<vk::GraphicsSubmissionTicket> submission;

    [[nodiscard]] bool IsComplete() const noexcept;
};
} // namespace infernux
