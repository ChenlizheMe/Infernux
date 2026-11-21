#pragma once

#include "PhysicsBodyState.h"
#include "PhysicsECSStore.h"
#include "PhysicsGpuPoseSource.h"
#include "inxphygpu/render_pose_layout.slang"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace infernux
{
class RuntimeWorld;
class PhysicsGpuRenderPoses;
namespace rhi
{
class ComputeHost;
class PreparedComputeBatch;
} // namespace rhi

/// Read the origin and rotation from the selected frozen pose rows. The read
/// packet carries only their two float4s and has no dependency on a solver.
struct PhysicsGpuPoseRead
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;

    [[nodiscard]] static PhysicsGpuPoseRead Prepare(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source,
                                                    std::vector<PhysicsGpuBodyPoseBinding> selected);
    [[nodiscard]] std::vector<PhysicsBodyPoseUpdate> Read() const;

  private:
    PhysicsGpuPoseSource::Layout m_layout{};
    std::vector<PhysicsGpuBodyPoseBinding> m_selected;
};

class PhysicsGpuRenderBindings final
{
  public:
    struct Snapshot;
    using SnapshotPtr = std::shared_ptr<const Snapshot>;
    using ResolveBody = std::function<std::optional<PhysicsGpuBodyPoseBinding>(PhysicsActorHandle, uint32_t bodyId)>;
    struct Source
    {
        PhysicsGpuPoseSource poses;
        ResolveBody resolve;
    };
    struct Prepared
    {
        std::vector<physics::gpu_render::PoseNode> nodes;
        std::vector<physics::gpu_render::PoseBinding> bindings;

      private:
        friend class PhysicsGpuRenderBindings;
        struct Target
        {
            size_t renderer = 0;
            PhysicsActorHandle actor;
            uint32_t bodyId = UINT32_MAX;
            uint64_t continuity = 0;
        };
        SnapshotPtr snapshot;
        std::vector<Target> targets;
    };

    struct Export
    {
        [[nodiscard]] const std::shared_ptr<rhi::PreparedComputeBatch> &Packet() const noexcept
        {
            return packet;
        }
        /// Seal after the export receipt, while these endpoints still name this work.
        void Seal();

      private:
        friend class PhysicsGpuRenderBindings;
        Prepared bindings;
        std::shared_ptr<rhi::PreparedComputeBatch> packet;
        std::shared_ptr<rhi::ComputeBuffer> models, bounds;
    };

    PhysicsGpuRenderBindings();
    ~PhysicsGpuRenderBindings();
    PhysicsGpuRenderBindings(const PhysicsGpuRenderBindings &) = delete;
    PhysicsGpuRenderBindings &operator=(const PhysicsGpuRenderBindings &) = delete;

    /// Capture at the logical job's author boundary. Ordinary local channels
    /// come directly from CPU storage; physics frame-cache world channels are
    /// resolved through cached local getters without acquiring GPU samples.
    [[nodiscard]] SnapshotPtr Capture(RuntimeWorld &world) const;
    [[nodiscard]] bool NeedsExport(const Snapshot &snapshot, uint64_t numericalGeneration) const;
    /// Resolves accepted actor identities and COMs against the captured author
    /// hierarchy. Never reads newer Transform values while preparing GPU work.
    [[nodiscard]] Export PrepareExport(SnapshotPtr snapshot, const Source &source,
                                       const PhysicsGpuRenderPoses &program) const;

    /// Called only for the completed export packet, including an empty mapping.
    /// Each renderer keeps a count-one endpoint selecting the packed buffers.
    /// False retains the previous automatic generation when live author
    /// metadata no longer matches. Manual and retired targets leave this set.
    [[nodiscard]] bool Publish(const Export &output, uint64_t numericalGeneration);
    /// World calls this for teleport/reset, independently of ordinary ticks.
    void MarkDiscontinuity(uint32_t bodyId);
    void RetireBody(uint32_t bodyId) noexcept;
    /// Stop/retirement clears only bindings still owned by this producer.
    void Clear() noexcept;

  private:
    [[nodiscard]] Prepared Prepare(SnapshotPtr snapshot, const ResolveBody &resolve) const;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace infernux
