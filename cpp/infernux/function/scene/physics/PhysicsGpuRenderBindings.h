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

struct PhysicsGpuMotionBinding
{
    PhysicsGpuBodyPoseBinding pose;
    uint64_t shapeRevision = 0;
    float inverseMass = 0;
    glm::vec3 inverseInertia{0.0f};
    glm::quat inertiaFrame{1.0f, 0.0f, 0.0f, 0.0f};
    uint32_t frozenTranslation = 0, frozenRotation = 0;
};

/// Reads only the selected channels of frozen body rows. Mass, lock axes and
/// shape identity belong to the same accepted author record as the motion.
struct PhysicsGpuMotionRead
{
    std::shared_ptr<rhi::PreparedComputeBatch> packet;

    [[nodiscard]] static PhysicsGpuMotionRead Prepare(rhi::ComputeHost &host, const PhysicsGpuPoseSource &source,
                                                      std::vector<PhysicsGpuMotionBinding> selected,
                                                      PhysicsStateFields fields);
    [[nodiscard]] bool Empty() const noexcept
    {
        return m_selected.empty();
    }
    [[nodiscard]] std::vector<PhysicsBodyMotionState> Read() const;

  private:
    struct Range
    {
        uint32_t first, count;
    };
    PhysicsGpuPoseSource::Layout m_layout{};
    PhysicsStateFields m_fields = PhysicsStateFields::None;
    std::vector<PhysicsGpuMotionBinding> m_selected;
    std::vector<uint32_t> m_rows;
    std::vector<Range> m_ranges;
    size_t m_uniqueRows = 0;
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
