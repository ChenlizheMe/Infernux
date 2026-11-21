// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsBodyState.h"
#include "../PhysicsContactTypes.h"
#include "WarpStep.h"

#include <core/threading/FrameWorkOwner.h>
#include <unordered_map>

namespace infernux::physics::warp
{

struct WarpSceneModel;

struct WarpRegisteredBody
{
    uint32_t bodyId;
    uint64_t shapeRevision;
};

struct WarpRegisteredWorld
{
    uint64_t worldId;
    std::vector<WarpRegisteredBody> bodies; // Scene model author order.
};

struct WarpFixedWorld
{
    ContactStreamIdentity identity;
    float time;
    uint32_t status;
    uint32_t acceptedStep;
};

struct WarpFixedPublication
{
    rhi::SubmissionTicket submission;
    std::shared_ptr<rhi::ComputeBuffer> bodyStates;
    std::vector<WarpFixedWorld> worlds;
    std::vector<WarpBodyState> hostBodies;
};

struct WarpHostBody
{
    PhysicsBodyPoseUpdate pose;
    uint64_t shapeRevision;
    glm::vec3 centerOfMass, linearVelocity, angularVelocity;
};

/// Fixed-step source for one compiled layout and its enrolled scene worlds.
/// Admission and receipt collection use the engine's FrameWorkOwner.
class WarpSceneRuntime final
{
  public:
    WarpSceneRuntime(FrameWorkOwner &work, rhi::ComputeHost &host, ShaderCodeTarget target, const WarpSceneModel &scene,
                     std::vector<WarpRegisteredWorld> worlds, float timestep, const ModelVector &gravity,
                     WarpCollisionCapacity collisionCapacity, const WarpContactSolver &solver,
                     uint32_t commandCapacity);
    ~WarpSceneRuntime();
    WarpSceneRuntime(const WarpSceneRuntime &) = delete;
    WarpSceneRuntime &operator=(const WarpSceneRuntime &) = delete;

    [[nodiscard]] FrameWorkOwner::SourceId SourceId() const noexcept
    {
        return m_source;
    }
    [[nodiscard]] std::optional<uint32_t> BodyRow(uint32_t world, uint32_t engineBody) const;

    /// Own the fixed identity until its publication is consumed. The device
    /// profile reports status only, while the host profile includes motion.
    void BeginFixed(std::vector<ContactStreamIdentity> identities, std::vector<WarpBodyCommand> commands, bool host);
    [[nodiscard]] std::optional<WarpFixedPublication> TakeFixed();
    [[nodiscard]] std::optional<WarpHostBody> ReadHostBody(const WarpFixedPublication &publication, uint32_t world,
                                                           uint32_t engineBody) const;
    [[nodiscard]] FrameWorkWaitResult WaitForProgress(uint64_t nanoseconds);
    void Retire() noexcept;
    [[nodiscard]] bool CanRetire() const noexcept;

  private:
    enum class Packet : uint8_t
    {
        Initialize,
        Preserve,
        Fixed
    };
    struct World
    {
        uint64_t id;
        std::vector<WarpRegisteredBody> bodies; // Device row, including world row zero.
        std::unordered_map<uint32_t, uint32_t> rows;
    };
    struct Fixed
    {
        std::vector<ContactStreamIdentity> identities;
        std::vector<WarpBodyCommand> commands;
        bool host;
    };
    struct Receipt
    {
        rhi::ComputeCompletion completion;
        Packet kind;
        uint64_t opportunity, gpuGrant, transferGrant, transferBytes;
        bool ready = false;
    };

    [[nodiscard]] FrameWorkFacts Describe() const;
    bool Advance(bool gpu, uint64_t token);
    void Observe();
    bool Submit();
    void Finish();
    void PublishFixed();

    FrameWorkOwner &m_work;
    rhi::ComputeHost m_host;
    FrameWorkOwner::SourceId m_source = 0;
    std::unique_ptr<WarpStep> m_step;
    std::vector<World> m_worlds;
    uint32_t m_bodyCount;
    std::shared_ptr<rhi::PreparedComputeBatch> m_packet, m_preservation;
    std::optional<Receipt> m_receipt;
    std::optional<Fixed> m_fixed;
    std::optional<WarpFixedPublication> m_publication;
    uint64_t m_generation = 1;
    uint64_t m_fixedGpuNanoseconds = 0;
    bool m_initialized = false;
    bool m_retiring = false;
};

} // namespace infernux::physics::warp
