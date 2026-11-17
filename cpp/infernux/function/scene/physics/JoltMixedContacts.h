#pragma once

#include <Jolt/Jolt.h>

#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Body/BodyID.h>
#include "JoltGpuScene.h"
#include "JoltJointTransfer.h"
#include "PhysicsContactPublication.h"
#include <core/math/Bvh.h>
#include <functional>
#include <unordered_map>
#include <utility>

namespace infernux
{
class JoltCpuScene;
class JoltCpuContactPublication;
class JoltMixedConstraintSolver;

struct JoltMixedFixedJoint
{
    uint32_t cpuBody, gpuBody;
    bool cpuOwner;
    JPH::Vec3 ownerBakedAnchor, ownerLocalAnchor;
    // This frame is captured once from the accepted poses at constraint creation.
    std::optional<JoltJointTransfer> reference;
};

/// One accepted Fixed step's CPU geometry and substep positions. The CPU
/// world still owns integration, and this capture owns no solver bodies.
class JoltMixedContactStep final
{
  public:
    using Policy = PhysicsContactPublication::Policy;
    using Frame = PhysicsContactPublication::Frame;
    struct CpuEndpoint
    {
        uint32_t body;
        bool allowResponse;
    };

    struct Candidate
    {
        JPH::GPU::UInt4 source; // GPU leaf index, layer, mask, motion type and activity.
        JPH::GPU::Float4 center, rotation, linear, angular, sweepMin, sweepMax;
        JPH::GPU::Float4 material; // Friction, restitution, combine modes as bits.
    };
    static_assert(sizeof(Candidate) == 128);

    struct ResponsePoint
    {
        JPH::GPU::Float4 positionDepth;
        glm::vec3 gpuVelocity;
        glm::vec3 localAnchorA, localAnchorB;
    };

    struct ResponsePatch
    {
        PhysicsContactPublication::Endpoint cpu, gpu;
        std::array<uint64_t, 2> shapeRevisions;
        JPH::BodyID native;
        uint32_t gpuSlot, pointCount;
        JPH::GPU::Float4 normalFriction;
        float restitution;
        bool cpuDynamic, gpuDynamic, wakeGpu;
        std::array<ResponsePoint, JPH::GPU::MaxContactPointsPerPair> points;
    };

    struct SleepLink
    {
        std::pair<uint32_t, uint32_t> bodies;
        size_t island;
    };

    using BodyPair = std::pair<uint32_t, uint32_t>;
    void Begin(const JoltCpuScene &, std::vector<CpuEndpoint>, std::vector<uint32_t> gpuResponseBodies,
               uint32_t substeps, std::vector<std::shared_ptr<JoltMixedFixedJoint>>, std::vector<BodyPair> suppressed);
    void BeginSleepObservation();
    [[nodiscard]] std::vector<uint32_t> TakeWakingIslands();
    [[nodiscard]] std::vector<uint32_t> TakeUnavailableIslands(const std::function<bool(uint32_t)> &canCouple);
    void AcceptSleepLinks();
    void CaptureCpuBoundary(uint32_t boundary);
    [[nodiscard]] const std::vector<JPH::GPU::UInt4> &EncodeTree(uint32_t substep, float predictedInterval);
    void Collect(uint32_t substep, const JoltGpuScene::Publication &, const Policy &, const Candidate &,
                 const JoltGpuModel *responseModel);
    [[nodiscard]] std::vector<ResponsePatch> &Responses() noexcept
    {
        return m_responses;
    }
    [[nodiscard]] Frame &ContactFrame(uint32_t substep) noexcept
    {
        return m_frames[substep];
    }
    [[nodiscard]] const JoltCpuScene &Scene() const noexcept
    {
        return *m_scene;
    }
    void AppendFrame(uint32_t substep, Frame &);
    void ClearStep();
    void InvalidateImpulseHistory(uint32_t bodyId);
    void DiscardImpulseHistory() noexcept;

  private:
    friend class JoltMixedConstraintSolver;
    struct MassFrame
    {
        JPH::Vec3 scale, localCOM;
    };
    [[nodiscard]] MassFrame CpuMassFrame(uint32_t id) const;
    [[nodiscard]] bool CpuJointMayWake(const JPH::Body &, std::unordered_map<uint32_t, bool> &) const;
    void ObserveSleepingPair(BodyPair);
    struct ImpulseHistory;
    struct SleepingContact
    {
        SleepLink link;
        bool observed = false;
    };
    [[nodiscard]] std::vector<uint32_t> TakeSleepingIslands(const std::function<bool(const SleepingContact &)> &wakes);
    struct CpuProxy
    {
        uint32_t id;
        uint64_t shapeRevision;
        JPH::RefConst<JPH::Shape> shape;
        PhysicsTargetReference target;
        bool sensor, allowResponse, dynamic;
    };
    struct Pose
    {
        JPH::RVec3 center;
        JPH::Quat rotation;
        JPH::Vec3 linear, angular;
        JPH::AABox bounds;
        bool active;
    };
    const JoltCpuScene *m_scene = nullptr;
    std::vector<CpuProxy> m_proxies;
    std::vector<std::vector<Pose>> m_pose;
    std::vector<std::vector<JPH::GPU::UInt4>> m_treeWords;
    std::vector<Frame> m_frames;
    std::vector<uint32_t> m_gpuResponseBodies;
    std::vector<ResponsePatch> m_responses;
    std::vector<std::shared_ptr<JoltMixedFixedJoint>> m_fixedJoints;
    std::vector<BodyPair> m_suppressedPairs;
    std::vector<SleepLink> m_pendingSleepLinks;
    std::vector<SleepingContact> m_sleepLinks;
    size_t m_nextSleepIsland = 0;
    std::shared_ptr<ImpulseHistory> m_impulseHistory;
    std::vector<uint32_t> m_ids, m_escape;
    spatial::Bvh m_tree;
};

/// Device traversal of the accepted CPU BVH against resident GPU leaf bounds.
/// This uses the owning physics cursor's packets throughout.
class JoltGpuMixedContacts final
{
  public:
    JoltGpuMixedContacts(rhi::ComputeHost &, const ShaderTargetCode &);
    [[nodiscard]] JoltGpuScene::ContactObserver Observe(const std::shared_ptr<const JoltGpuScene::Publication> &,
                                                        JoltMixedContactStep &, const JoltMixedContactStep::Policy &);
    [[nodiscard]] JoltGpuControlWork Discover(const JoltGpuStorage &, const JPH::GPU::StepParams &, uint32_t substep,
                                              const std::shared_ptr<const JoltGpuScene::Publication> &,
                                              JoltMixedContactStep &, const JoltMixedContactStep::Policy &,
                                              std::function<JoltGpuControlWork()>);

  private:
    struct Job;
    rhi::ComputeHost m_host;
    std::shared_ptr<rhi::ComputeKernel> m_kernel;
    std::vector<uint32_t> m_ports;
};
} // namespace infernux
