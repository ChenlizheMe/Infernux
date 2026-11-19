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
    using ExchangeStates = std::map<std::string, std::pair<std::shared_ptr<PhysicsParticipant>, std::string>>;
    using ControlPayloads = std::map<std::string, std::shared_ptr<const std::vector<uint8_t>>>;
    struct RigidExchange
    {
        using Recipe = std::function<JoltGpuControlWork(std::vector<rhi::ComputeBufferUpdate>,
                                                        std::vector<rhi::ComputeBufferRead>)>;
        std::function<JoltGpuControlWork(JoltGpuParticipantRigidPort, const JPH::GPU::StepParams &,
                                         const std::shared_ptr<rhi::ComputeBuffer> &)>
            prepare;
        // The body owner may include bounded CPU ports in the same recipe
        // packet. It owns the original segmented chain when contraction is needed.
        std::function<JoltGpuControlWork(const JPH::GPU::StepParams &, const std::shared_ptr<rhi::ComputeBuffer> &,
                                         const std::shared_ptr<rhi::ComputeBuffer> &, Recipe)>
            exchange;
        struct ClosedRounds
        {
            std::vector<rhi::ComputeBufferUpdate> updates;
            std::vector<rhi::ComputeBufferRead> reads;
            std::vector<rhi::ComputeDispatchDesc> responses;
            std::function<void(const std::shared_ptr<rhi::PreparedComputeBatch> &)> apply;
        };
        // No intermediate native consumer: optional preparation may decline
        // without submitting, changing the accepted cursor or retaining debt.
        std::function<std::optional<ClosedRounds>(const JPH::GPU::StepParams &,
                                                  const std::shared_ptr<rhi::ComputeBuffer> &,
                                                  const std::shared_ptr<rhi::ComputeBuffer> &, uint32_t)>
            closeRounds;
        explicit operator bool() const noexcept
        {
            return bool(prepare);
        }
    };
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
                       const std::vector<Rigidbody *> &rigidbodies, const ExchangeStates &exchangeStates);
    [[nodiscard]] bool IsReady() const noexcept
    {
        return m_ready && !m_retired;
    }
    [[nodiscard]] bool IsRetired() const noexcept
    {
        return m_retired || !m_component.GetComponent();
    }
    [[nodiscard]] bool IsActive() const;
    /// Cold preparation hint only; Fixed routing still resolves accepted body ownership.
    [[nodiscard]] bool HasCpuOnlyRigidBindings() const;
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

    struct PreparedStage
    {
        rhi::ComputeHost *host = nullptr;
        std::vector<rhi::ComputeBufferUpdate> updates;
        std::vector<rhi::ComputeDispatchDesc> dispatches;
        std::function<void()> accepted;
    };
    /// Lower one GPU-only stage without changing its bound state or private
    /// scratch identities. A runtime slot supplies immutable per-round ports;
    /// writable/aliased runtime ports decline before preparing any GPU work.
    [[nodiscard]] std::optional<PreparedStage> PrepareGpuStage(JoltGpuParticipantStage, const JPH::GPU::StepParams &,
                                                               float fixedDeltaTime, uint32_t substep,
                                                               uint32_t iteration, const Prefix &,
                                                               std::optional<uint32_t> runtimeSlot = std::nullopt);

    [[nodiscard]] std::optional<JoltGpuControlWork>
    PrepareClosedRounds(const JPH::GPU::StepParams &, float fixedDeltaTime, uint32_t substep, uint32_t iteration,
                        uint32_t rounds, const Prefix &, std::function<void(uint32_t)> done);

  private:
    struct ExchangeState
    {
        std::weak_ptr<PhysicsParticipant> owner;
        std::string output;
        std::shared_ptr<rhi::ComputeBuffer> buffer;
    };
    void DeclareExchangeStates(const ExchangeStates &states);
    void ReleasePeerExchange();
    void ValidateRigidPorts();
    [[nodiscard]] JoltGpuControlWork PrepareRecipeStage(JoltGpuParticipantStage stage,
                                                        const JPH::GPU::StepParams &params, float fixedDeltaTime,
                                                        uint32_t substep, uint32_t iteration, const Prefix &prefix,
                                                        std::vector<rhi::ComputeBufferUpdate> updates = {},
                                                        std::vector<rhi::ComputeBufferRead> reads = {});
    [[nodiscard]] JoltGpuControlWork PrepareOutput(const std::shared_ptr<PhysicsParticipantOutput> &capture,
                                                   uint64_t epoch, uint64_t step, bool isFixedStep);
    bool BindInstances();
    std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> Inputs(const rhi::ComputeWorkRecipe &recipe,
                                                                      bool initialize) const;
    PhysicsTargetReference m_component;
    std::vector<PhysicsTargetReference> m_rigidbodies;
    Recipes m_recipes;
    Controls m_controls;
    std::map<std::string, ExchangeState> m_exchangeStates;
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
    struct BatchRuntime
    {
        std::shared_ptr<rhi::ComputeBuffer> step, indices;
    };
    std::vector<BatchRuntime> m_batchRuntime;
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
