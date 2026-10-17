#pragma once

#include <array>
#include <function/renderer/InxVkCoreModular.h>

namespace infernux
{

inline void RetirePreviewAttachments(InxVkCoreModular &core, vk::VkImageHandle &color, vk::VkImageHandle &resolve,
                                     vk::VkImageHandle &depth)
{
    if (!color.IsValid() && !resolve.IsValid() && !depth.IsValid())
        return;
    auto retired = std::make_shared<std::array<vk::VkImageHandle, 3>>();
    (*retired)[0] = std::move(color);
    (*retired)[1] = std::move(resolve);
    (*retired)[2] = std::move(depth);
    core.RetireGpuResource([retired = std::move(retired)] {});
}

} // namespace infernux
