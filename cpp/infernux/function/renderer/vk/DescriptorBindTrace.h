#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>

#include <core/error/InxError.h>
#include <vulkan/vulkan.h>

namespace infernux::vkdebug
{

struct DescriptorBindTraceSnapshot
{
    uint64_t sequence = 0;
    const char *site = nullptr;
    uint64_t commandBufferRaw = 0;
    uint64_t pipelineLayoutRaw = 0;
    uint32_t firstSet = 0;
    uint32_t descriptorSetCount = 0;
    std::array<uint64_t, 4> descriptorSetRaws{};
};

inline bool IsSuspiciousDescriptorRaw(uint64_t raw)
{
    if (raw == 0ull)
        return false;

    const uint32_t lo = static_cast<uint32_t>(raw & 0xffffffffull);
    const uint32_t hi = static_cast<uint32_t>((raw >> 32) & 0xffffffffull);
    return hi == lo && lo <= 0x000fffffu;
}

inline std::mutex &DescriptorBindTraceMutex()
{
    static std::mutex m;
    return m;
}

inline constexpr size_t kDescriptorBindTraceHistorySize = 256;

inline std::array<DescriptorBindTraceSnapshot, kDescriptorBindTraceHistorySize> &DescriptorBindTraceHistory()
{
    static std::array<DescriptorBindTraceSnapshot, kDescriptorBindTraceHistorySize> history{};
    return history;
}

inline size_t &DescriptorBindTraceHistoryWriteIndex()
{
    static size_t idx = 0;
    return idx;
}

inline DescriptorBindTraceSnapshot &LastDescriptorBindSnapshotStorage()
{
    static DescriptorBindTraceSnapshot s;
    return s;
}

inline std::atomic<uint64_t> &DescriptorBindTraceSequence()
{
    static std::atomic<uint64_t> seq{0};
    return seq;
}

inline std::atomic<int> &DescriptorBindSuspiciousWarnCount()
{
    static std::atomic<int> count{0};
    return count;
}

inline void RecordDescriptorBind(const char *site, VkCommandBuffer cmdBuf, VkPipelineLayout layout, uint32_t firstSet,
                                 uint32_t descriptorSetCount, const VkDescriptorSet *descriptorSets)
{
    DescriptorBindTraceSnapshot snapshot;
    snapshot.sequence = DescriptorBindTraceSequence().fetch_add(1, std::memory_order_relaxed) + 1;
    snapshot.site = site;
    snapshot.commandBufferRaw = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cmdBuf));
    snapshot.pipelineLayoutRaw = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(layout));
    snapshot.firstSet = firstSet;
    snapshot.descriptorSetCount = descriptorSetCount;

    const uint32_t copyCount =
        std::min<uint32_t>(descriptorSetCount, static_cast<uint32_t>(snapshot.descriptorSetRaws.size()));
    for (uint32_t i = 0; i < copyCount; ++i) {
        snapshot.descriptorSetRaws[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(descriptorSets[i]));
    }

    {
        std::lock_guard<std::mutex> lock(DescriptorBindTraceMutex());
        LastDescriptorBindSnapshotStorage() = snapshot;
        auto &history = DescriptorBindTraceHistory();
        size_t &writeIdx = DescriptorBindTraceHistoryWriteIndex();
        history[writeIdx] = snapshot;
        writeIdx = (writeIdx + 1) % kDescriptorBindTraceHistorySize;
    }

    for (uint32_t i = 0; i < copyCount; ++i) {
        const uint64_t raw = snapshot.descriptorSetRaws[i];
        if (!IsSuspiciousDescriptorRaw(raw))
            continue;

        const int warnIndex = DescriptorBindSuspiciousWarnCount().fetch_add(1, std::memory_order_relaxed);
        if (warnIndex < 48) {
            INXLOG_WARN("[VkBindTrace] suspicious descriptor raw=0x", raw, " site=", (site ? site : "<null>"),
                        " firstSet=", firstSet, " localIndex=", i, " count=", descriptorSetCount, " cmd=0x",
                        snapshot.commandBufferRaw, " layout=0x", snapshot.pipelineLayoutRaw);
        }
    }
}

inline DescriptorBindTraceSnapshot GetLastDescriptorBindSnapshot()
{
    std::lock_guard<std::mutex> lock(DescriptorBindTraceMutex());
    return LastDescriptorBindSnapshotStorage();
}

inline bool FindRecentDescriptorBindByRaw(uint64_t descriptorRaw, DescriptorBindTraceSnapshot &outMatch,
                                          uint32_t &outLocalIndex)
{
    if (descriptorRaw == 0ull)
        return false;

    std::lock_guard<std::mutex> lock(DescriptorBindTraceMutex());
    const auto &history = DescriptorBindTraceHistory();
    const size_t writeIdx = DescriptorBindTraceHistoryWriteIndex();

    for (size_t i = 0; i < kDescriptorBindTraceHistorySize; ++i) {
        const size_t idx = (writeIdx + kDescriptorBindTraceHistorySize - 1 - i) % kDescriptorBindTraceHistorySize;
        const auto &snap = history[idx];
        if (snap.sequence == 0)
            continue;

        const uint32_t count =
            std::min<uint32_t>(snap.descriptorSetCount, static_cast<uint32_t>(snap.descriptorSetRaws.size()));
        for (uint32_t local = 0; local < count; ++local) {
            if (snap.descriptorSetRaws[local] == descriptorRaw) {
                outMatch = snap;
                outLocalIndex = local;
                return true;
            }
        }
    }
    return false;
}

inline void CmdBindDescriptorSetsTracked(const char *site, VkCommandBuffer cmdBuf,
                                         VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout,
                                         uint32_t firstSet, uint32_t descriptorSetCount,
                                         const VkDescriptorSet *descriptorSets, uint32_t dynamicOffsetCount,
                                         const uint32_t *dynamicOffsets)
{
    RecordDescriptorBind(site, cmdBuf, layout, firstSet, descriptorSetCount, descriptorSets);
    vkCmdBindDescriptorSets(cmdBuf, pipelineBindPoint, layout, firstSet, descriptorSetCount, descriptorSets,
                            dynamicOffsetCount, dynamicOffsets);
}

} // namespace infernux::vkdebug
