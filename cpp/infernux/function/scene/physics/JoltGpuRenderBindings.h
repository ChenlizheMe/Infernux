#pragma once

#include "JoltGpuScene.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace JPH::GPU
{
struct PoseNode;
}

namespace infernux
{
class Rigidbody;

class JoltGpuRenderBindings final
{
  public:
    struct Snapshot;
    using SnapshotPtr = std::shared_ptr<const Snapshot>;
    struct PhysicsPose
    {
        glm::vec3 position;
        glm::quat rotation;
        uint8_t channels;

        bool operator!=(const PhysicsPose &other) const noexcept
        {
            return channels != other.channels || position != other.position || rotation != other.rotation;
        }
    };
    using MixedPose = std::function<std::optional<PhysicsPose>(uint32_t bodyId, bool cpu)>;

    struct Prepared
    {
        std::vector<JPH::GPU::PoseNode> nodes;
        std::vector<JPH::GPU::PoseBinding> bindings;

      private:
        friend class JoltGpuRenderBindings;
        struct Target
        {
            size_t renderer = 0;
            std::shared_ptr<const JoltGpuScene::PublishedActor> actor;
            uint64_t continuity = 0;
        };
        SnapshotPtr snapshot;
        std::vector<Target> targets;
    };

    JoltGpuRenderBindings();
    ~JoltGpuRenderBindings();
    JoltGpuRenderBindings(const JoltGpuRenderBindings &) = delete;
    JoltGpuRenderBindings &operator=(const JoltGpuRenderBindings &) = delete;

    /// Capture at the logical job's author boundary. Ordinary local channels
    /// come directly from CPU storage; physics frame-cache world channels are
    /// resolved through cached local getters without acquiring GPU samples.
    [[nodiscard]] SnapshotPtr Capture(RuntimeWorld &world, bool captureCpuConsumers = false) const;
    [[nodiscard]] bool NeedsExport(const Snapshot &snapshot, uint64_t numericalGeneration) const;
    [[nodiscard]] bool HasMixedHierarchy(const Snapshot &) const;
    [[nodiscard]] std::vector<Rigidbody *> MixedGpuAncestors(const Snapshot &) const;
    [[nodiscard]] SnapshotPtr ResolveMixedAnchors(SnapshotPtr snapshot, const MixedPose &pose) const;

    /// Resolves accepted actor identities and COMs against the captured author
    /// hierarchy. Never reads newer Transform values while preparing GPU work.
    [[nodiscard]] Prepared Prepare(SnapshotPtr snapshot, const JoltGpuScene::Publication &publication) const;

    /// Called only for the completed export packet, including an empty mapping.
    /// Each renderer keeps a count-one endpoint selecting the packed buffers.
    /// False retains the previous automatic generation when live author
    /// metadata no longer matches. Manual and retired targets leave this set.
    [[nodiscard]] bool Publish(const Prepared &prepared, const std::shared_ptr<rhi::ComputeBuffer> &models,
                               const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint64_t numericalGeneration,
                               const std::function<void()> &commitTransforms = {});
    /// World calls this for teleport/reset, independently of ordinary ticks.
    void MarkDiscontinuity(uint32_t bodyId);
    void RetireBody(uint32_t bodyId) noexcept;
    /// Stop/retirement clears only bindings still owned by this producer.
    void Clear() noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
