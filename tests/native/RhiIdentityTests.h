#pragma once

#include <function/renderer/vk/VulkanRhiDevice.h>

#include <array>
#include <atomic>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace rhi_identity_test
{
using namespace infernux;

inline void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Native> Native Opaque(uintptr_t value)
{
    if constexpr (std::is_pointer_v<Native>)
        return reinterpret_cast<Native>(value);
    else
        return static_cast<Native>(value);
}

inline void SlotExhaustion(bool reset)
{
    vk::VulkanRhiDevice device;
    const auto native = Opaque<VkImageView>(0x241); // Registration only; never submitted to Vulkan.
    const auto first = device.RegisterTextureView(native);
    auto current = first;
    const unsigned maximum = std::numeric_limits<uint16_t>::max();
    for (unsigned generation = 1; generation <= maximum; ++generation) {
        Require(current.index == first.index && current.Version() == generation, "Unexpected slot reuse order");
        Require(device.Resolve(current) == native, "Live registration did not resolve");
        if (generation > 1) {
            Require(device.Resolve(first) == VK_NULL_HANDLE, "Old handle resolved during slot reuse");
            device.Release(first);
            Require(device.Resolve(current) == native, "Old release destroyed a live registration");
        }
        if (reset)
            device.Reset();
        else
            device.Release(current);
        if (generation < maximum)
            current = device.RegisterTextureView(native);
    }
    // Reset must never resurrect a slot retired by either Release or Reset.
    device.Reset();
    const auto replacement = device.RegisterTextureView(native);
    Require(replacement.index != first.index, "Exhausted generation reused a retired slot");
    Require(device.Resolve(first) == VK_NULL_HANDLE && device.Resolve(current) == VK_NULL_HANDLE,
            "Exhausted handle resolved a replacement");
    device.Release(first);
    device.Release(current);
    Require(device.Resolve(replacement) == native, "Retired handle released a replacement");
    device.Release(replacement);
    for (unsigned repeat = 0; repeat < 4; ++repeat)
        device.Reset();
    const auto reused = device.RegisterTextureView(native);
    Require(reused.index == replacement.index && reused.Version() == replacement.Version() + 1,
            "Reset consumed unpublished free generations or revived a retired slot");
    device.Release(reused);
    std::cout << "rhi_identity slot=" << (reset ? "reset" : "release") << " generations=" << maximum
              << " stale_resolve_release_rejected=true\n";
}

inline void DeviceExhaustion()
{
    vk::VulkanRhiDevice retained;
    constexpr unsigned count = 4;
    std::array<std::vector<rhi::DeviceId>, count> ids;
    std::array<std::thread, count> threads;
    std::atomic<unsigned> exhausted{0};
    for (unsigned worker = 0; worker < count; ++worker) {
        threads[worker] = std::thread([&, worker] {
            for (unsigned attempt = 0; attempt < 65536; ++attempt) {
                try {
                    ids[worker].push_back(rhi::AllocateDeviceId());
                } catch (const std::overflow_error &) {
                    ++exhausted;
                    break;
                }
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
    std::unordered_set<rhi::DeviceId> unique{retained.GetDeviceId()};
    for (const auto &allocated : ids)
        for (auto id : allocated)
            Require(id != rhi::InvalidDeviceId && unique.insert(id).second, "Device identity wrapped or duplicated");
    Require(exhausted == count, "Device identity exhaustion was not reported to every caller");
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        bool rejected = false;
        try {
            vk::VulkanRhiDevice unavailable;
        } catch (const std::overflow_error &) {
            rejected = true;
        }
        Require(rejected, "Device construction reused an exhausted identity");
    }
    // Construction taking a native device must also propagate exhaustion,
    // rather than terminate through a noexcept constructor.
    bool rejected = false;
    try {
        vk::VulkanRhiDevice unavailable(VK_NULL_HANDLE);
    } catch (const std::overflow_error &) {
        rejected = true;
    }
    Require(rejected, "Native-device construction reused an exhausted identity");
    const auto native = Opaque<VkImageView>(0x242);
    const auto handle = retained.RegisterTextureView(native);
    Require(retained.Resolve(handle) == native, "Exhaustion damaged an existing device");
    retained.Release(handle);
    std::cout << "rhi_identity device_unique=" << unique.size() << " exhausted_callers=" << exhausted << '\n';
}
} // namespace rhi_identity_test
