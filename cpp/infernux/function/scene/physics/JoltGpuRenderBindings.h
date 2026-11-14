#pragma once

#include "JoltGpuScene.h"

#include <memory>
#include <vector>

namespace JPH::GPU
{
struct PoseNode;
}

namespace infernux
{

class JoltGpuRenderBindings final
{
  public:
    struct Snapshot;
    using SnapshotPtr = std::shared_ptr<const Snapshot>;

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
    [[nodiscard]] SnapshotPtr Capture(RuntimeWorld &world) const;
    [[nodiscard]] bool NeedsExport(const Snapshot &snapshot, uint64_t numericalGeneration) const;

    /// Resolves accepted actor identities and COMs against the captured author
    /// hierarchy. Never reads newer Transform values while preparing GPU work.
    [[nodiscard]] Prepared Prepare(SnapshotPtr snapshot, const JoltGpuScene::Publication &publication) const;

    /// Called only for the completed export packet, including an empty mapping.
    /// Each renderer keeps a count-one endpoint selecting the packed buffers.
    /// False retains the previous automatic generation when live author
    /// metadata no longer matches. Manual and retired targets leave this set.
    [[nodiscard]] bool Publish(const Prepared &prepared, const std::shared_ptr<rhi::ComputeBuffer> &models,
                               const std::shared_ptr<rhi::ComputeBuffer> &bounds, uint64_t numericalGeneration);
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
