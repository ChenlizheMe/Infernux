#pragma once

#include <function/renderer/RenderIdentity.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace infernux
{

static constexpr uint32_t kGPUInstanceAuxFlagValidHistory = 1u;
static constexpr uint32_t kGPUInstanceAuxFlagNoReceiveShadows = 2u;

/// Per-instance identity, transform history and geometry lighting controls.
/// Shadow draws keep using the compact model-matrix-only stream.
struct alignas(16) GPUInstanceAuxData
{
    glm::mat4 previousModel{1.0f};
    glm::uvec2 objectId{0u};
    uint32_t flags = 0;
    uint32_t layerMask = 1u;
};

static_assert(alignof(GPUInstanceAuxData) == 16, "GPU instance auxiliary data must remain std430-compatible");
static_assert(sizeof(GPUInstanceAuxData) == 80, "GPU instance auxiliary data layout changed");
static_assert(offsetof(GPUInstanceAuxData, objectId) == 64, "GPU object identity must follow the previous model");

[[nodiscard]] constexpr glm::uvec2 PackGPUObjectId(uint64_t objectId) noexcept
{
    return {static_cast<uint32_t>(objectId), static_cast<uint32_t>(objectId >> 32u)};
}

[[nodiscard]] constexpr uint64_t UnpackGPUObjectId(const glm::uvec2 &objectId) noexcept
{
    return static_cast<uint64_t>(objectId.x) | (static_cast<uint64_t>(objectId.y) << 32u);
}

/// Rigid and skinned motion refer to the same accepted execution of one view.
/// Recording can touch a primitive in several passes or be abandoned entirely.
class RenderInstanceHistory
{
  public:
    struct Pose
    {
        glm::mat4 model{1.0f};
        std::shared_ptr<const std::vector<glm::mat4>> skin;
        uint64_t skinContinuity = 0;
    };

    void BeginFrame() noexcept
    {
        ++m_recordingFrame;
    }

    [[nodiscard]] const Pose *Previous(const RenderDrawIdentity &identity, uint64_t skinContinuity) const
    {
        if (m_acceptedFrame == 0)
            return nullptr;
        const auto entry = m_entries.find(identity);
        if (entry == m_entries.end())
            return nullptr;
        const auto &previous = entry->second.frames[m_recordingSlot ^ 1u];
        return previous.serial == m_acceptedFrame && previous.pose.skinContinuity == skinContinuity ? &previous.pose
                                                                                                    : nullptr;
    }

    void Record(const RenderDrawIdentity &identity, const glm::mat4 &model,
                const std::shared_ptr<const std::vector<glm::mat4>> &skin, uint64_t skinContinuity)
    {
        if (!identity.IsValid())
            return;
        auto &entry = m_entries[identity];
        auto &current = entry.frames[m_recordingSlot];
        if (current.serial != m_recordingFrame) {
            current.pose = {model, skin, skinContinuity};
            current.serial = m_recordingFrame;
        }
    }

    void CommitFrame()
    {
        m_acceptedFrame = m_recordingFrame;
        m_recordingSlot ^= 1u;
        if (m_acceptedFrame - m_lastPruneFrame < kPruneInterval)
            return;
        for (auto entry = m_entries.begin(); entry != m_entries.end();) {
            const auto &frames = entry->second.frames;
            if (m_acceptedFrame - std::max(frames[0].serial, frames[1].serial) > kRetentionFrames)
                entry = m_entries.erase(entry);
            else
                ++entry;
        }
        m_lastPruneFrame = m_acceptedFrame;
    }

    void Clear() noexcept
    {
        m_entries.clear();
        m_acceptedFrame = 0;
    }

  private:
    struct FramePose
    {
        Pose pose;
        uint64_t serial = 0;
    };

    struct Entry
    {
        FramePose frames[2];
    };

    static constexpr uint64_t kRetentionFrames = 120;
    static constexpr uint64_t kPruneInterval = 60;

    std::unordered_map<RenderDrawIdentity, Entry, RenderDrawIdentityHash> m_entries;
    uint64_t m_recordingFrame = 0;
    uint64_t m_acceptedFrame = 0;
    uint64_t m_lastPruneFrame = 0;
    unsigned m_recordingSlot = 0;
};

} // namespace infernux
