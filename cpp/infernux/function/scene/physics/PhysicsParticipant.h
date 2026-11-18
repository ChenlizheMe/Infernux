#pragma once

#include "JoltGpuTick.h"
#include "PhysicsTargetReference.h"
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <map>
#include <string>
#include <type_traits>

namespace infernux
{
class Component;
class Rigidbody;
class PhysicsParticipant;
struct RuntimeWorldHandle;

/// A deferred immutable compact observation, owned by ordinary BufferSource leases.
/// The participant only keeps weak entries. Queue bundles retain submitted storage.
class PhysicsParticipantOutput final
{
  public:
    std::weak_ptr<PhysicsParticipant> owner;
    std::string name, error;
    uint64_t token = 0, participantGeneration = 0, worldId = 0;
    uint64_t componentId = 0, componentGeneration = 0, componentSceneId = 0;
    uint64_t offset = 0, sourceByteOffset = 0;
    rhi::ComputeBufferDesc layout;
    uint64_t simulationEpoch = 0, fixedStepIndex = 0;
    bool isFixedStep = false, ready = false, cancelled = false;
    rhi::SubmissionTicket sourceWrite, copySubmission;
    std::shared_ptr<rhi::ComputeBuffer> storage, value;
    [[nodiscard]] bool IsDone() const;
    [[nodiscard]] bool IsReady() const
    {
        return ready && !cancelled && error.empty();
    }
    [[nodiscard]] std::string Error() const;
    void Cancel() noexcept;
    void Fail(std::string reason);
};

/// Exact scalar-array port ABI; independent from any project state layout.
struct PhysicsParticipantStep
{
    float substepDeltaTime, fixedDeltaTime;
};
struct PhysicsParticipantIndices
{
    uint32_t substep, substeps, iteration, iterations, count, stage;
};
static_assert(std::is_standard_layout_v<PhysicsParticipantStep> &&
              std::is_trivially_copyable_v<PhysicsParticipantStep>);
static_assert(std::is_standard_layout_v<PhysicsParticipantIndices> &&
              std::is_trivially_copyable_v<PhysicsParticipantIndices>);
static_assert(sizeof(PhysicsParticipantStep) == 8 && sizeof(PhysicsParticipantIndices) == 24);

/// Author values only. The allocation is per participant, never the prototype.
struct PhysicsParticipantControl
{
    std::shared_ptr<rhi::ComputeBuffer> prototype;
    std::vector<uint8_t> initial;
};

/// Native owner for public Work recipes. No project callback runs during a tick.
/// The selected tick captures shared owners; removal revokes later admission only.
class PhysicsParticipant final : public std::enable_shared_from_this<PhysicsParticipant>
{
  public:
    using Recipes = std::map<std::string, std::shared_ptr<rhi::ComputeWorkRecipe>>;
    using Controls = std::map<std::string, PhysicsParticipantControl>;
    using ControlPayloads = std::map<std::string, std::shared_ptr<const std::vector<uint8_t>>>;
    using RigidExchange = std::function<JoltGpuControlWork(JoltGpuParticipantRigidPort, const JPH::GPU::StepParams &,
                                                           const std::shared_ptr<rhi::ComputeBuffer> &)>;
    using RigidFactory = std::function<RigidExchange(std::vector<uint32_t>)>;
    struct Prefix
    {
        std::shared_ptr<PhysicsParticipant> owner;
        ControlPayloads controls;
        uint32_t count, stages;
        std::vector<uint32_t> rigidBodyIds;
        RigidExchange rigidExchange;
    };

    PhysicsParticipant(const Component &component, Recipes recipes, Controls controls, uint32_t count,
                       const std::vector<Rigidbody *> &rigidbodies);
    [[nodiscard]] bool IsReady() const noexcept
    {
        return m_ready && !m_retired;
    }
    [[nodiscard]] bool IsRetired() const noexcept
    {
        return m_retired || !m_component.GetComponent();
    }
    [[nodiscard]] bool IsActive() const;
    void Retire() noexcept;
    [[nodiscard]] std::shared_ptr<PhysicsParticipantOutput> CaptureOutput(const std::string &name, uint64_t offset,
                                                                          std::optional<uint64_t> count);
    [[nodiscard]] bool HasPendingOutputs();
    [[nodiscard]] JoltGpuControlWork PrepareOutputs(uint64_t epoch, uint64_t step, bool isFixedStep);
    void SetControl(const std::string &name, std::vector<uint8_t> bytes);
    void SetCount(uint32_t count);
    void AcceptControls(const ControlPayloads &values);
    [[nodiscard]] Prefix CapturePrefix();
    [[nodiscard]] JoltGpuControlWork PrepareInitialize();
    [[nodiscard]] JoltGpuControlWork PrepareStage(JoltGpuParticipantStage stage, const JPH::GPU::StepParams &params,
                                                  float fixedDeltaTime, uint32_t substep, uint32_t iteration,
                                                  const Prefix &prefix);

  private:
    void ValidateRigidPorts();
    [[nodiscard]] JoltGpuControlWork PrepareRecipeStage(JoltGpuParticipantStage stage,
                                                        const JPH::GPU::StepParams &params, float fixedDeltaTime,
                                                        uint32_t substep, uint32_t iteration, const Prefix &prefix);
    [[nodiscard]] JoltGpuControlWork PrepareOutput(const std::shared_ptr<PhysicsParticipantOutput> &capture,
                                                   uint64_t epoch, uint64_t step, bool isFixedStep);
    bool BindInstances();
    std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> Inputs(const rhi::ComputeWorkRecipe &recipe,
                                                                      bool initialize) const;
    PhysicsTargetReference m_component;
    std::vector<PhysicsTargetReference> m_rigidbodies;
    Recipes m_recipes;
    Controls m_controls;
    std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> m_state, m_controlStorage;
    std::map<uint32_t, std::shared_ptr<rhi::ComputeBuffer>> m_stateSlots;
    ControlPayloads m_pendingControls, m_initialControls;
    struct BoundStage
    {
        BoundStage(std::shared_ptr<rhi::ComputeWorkInstance> instance, const rhi::ComputeBuffer *step);
        std::shared_ptr<rhi::ComputeWorkInstance> work;
        bool usesStep = false, usesIndices = false, writesStep = false;
    };
    std::map<std::string, BoundStage> m_instances;
    std::optional<PhysicsParticipantStep> m_uploadedStep;
    std::shared_ptr<rhi::ComputeBuffer> m_step, m_indices, m_rigidMotion;
    uint32_t m_rigidMotionSlot = UINT32_MAX, m_rigidImpulseSlot = UINT32_MAX;
    std::vector<std::weak_ptr<PhysicsParticipantOutput>> m_outputs;
    uint64_t m_generation = 0, m_nextOutput = 1;
    uint32_t m_count, m_initialCount, m_stages = 0;
    bool m_ready = false, m_retired = false;
};

/// A frozen author/control prefix, prepared once when the actual tick is admitted.
[[nodiscard]] JoltGpuParticipantStages
CapturePhysicsParticipants(const std::vector<std::weak_ptr<PhysicsParticipant>> &participants,
                           const PhysicsParticipant::RigidFactory &bindBodies);
} // namespace infernux
