#pragma once

#include "RhiIdentityTests.h"
#include <function/renderer/vk/VkDeviceContext.h>

namespace rhi_identity_test
{
inline void RealBufferRecycling(vk::VkDeviceContext &context)
{
    auto &device = context.GetRhiDevice();
    rhi::BufferDesc desc;
    desc.byteSize = 64;
    desc.memory = rhi::BufferMemory::Readback;
    desc.usage = rhi::BufferUsageFlags::TransferDestination;
    const auto first = device.CreateBuffer(desc);
    Require(first.IsValid(), "Initial real buffer allocation failed");
    auto current = first;
    for (unsigned generation = 1; generation <= std::numeric_limits<uint16_t>::max(); ++generation) {
        Require(current.index == first.index && current.Version() == generation, "Real buffer reuse order changed");
        device.Release(current);
        current = device.CreateBuffer(desc);
        Require(current.IsValid(), "Real buffer recycling allocation failed");
        Require(device.Resolve(first) == VK_NULL_HANDLE, "Stale real buffer resolved after recycling");
        device.Release(first);
        Require(device.Resolve(current) != VK_NULL_HANDLE, "Stale release destroyed a new real buffer");
    }
    Require(current.index != first.index, "Real buffer allocation reused an exhausted slot");

    const auto native = context.GetDevice();
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.GetQueueIndices().graphicsFamily.value();
    VkCommandPool pool = VK_NULL_HANDLE;
    Require(vkCreateCommandPool(native, &poolInfo, nullptr, &pool) == VK_SUCCESS, "GPU proof pool creation failed");
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    Require(vkAllocateCommandBuffers(native, &allocate, &command) == VK_SUCCESS, "GPU proof command allocation failed");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    Require(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "GPU proof recording failed");
    const uint32_t expected = 0x241abcde;
    vkCmdFillBuffer(command, device.Resolve(current), 0, desc.byteSize, expected);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &barrier, 0, nullptr, 0, nullptr);
    Require(vkEndCommandBuffer(command) == VK_SUCCESS, "GPU proof recording completion failed");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    const auto queue = context.GetGraphicsQueue();
    Require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS, "GPU proof submission failed");
    Require(vkQueueWaitIdle(queue) == VK_SUCCESS, "GPU proof completion failed");
    std::array<uint32_t, 16> actual{};
    Require(device.ReadBuffer(current, 0, actual.data(), sizeof(actual)), "GPU proof readback failed");
    for (auto word : actual)
        Require(word == expected, "Recycled real buffer returned incorrect GPU data");
    vkDestroyCommandPool(native, pool, nullptr);
    device.Release(current);
    std::cout << "rhi_identity real_allocations=65536 stale_release_rejected=true gpu_readback_words=16\n";
}
} // namespace rhi_identity_test
