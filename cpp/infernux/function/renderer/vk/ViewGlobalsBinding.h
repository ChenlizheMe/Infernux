#pragma once

#include "VkDescriptorManager.h"
#include <function/renderer/rhi/RhiHandles.h>

namespace infernux::vk
{

// One RenderView and frame slot own the camera globals. Instance streams are
// shared, so their descriptor revision is copied without replacing this UBO.
struct ViewGlobalsBinding
{
    rhi::BufferHandle buffer;
    DescriptorLease descriptor;
    rhi::BindGroupHandle group;
    uint64_t instanceSourceLeaseId = 0;
};

} // namespace infernux::vk
