#include "JoltGpuTick.h"
#include "JoltGpuArticulations.h"
#include "JoltGpuBroadphase.h"
#include "JoltGpuConstraints.h"
#include "JoltGpuNarrowphase.h"
#include "JoltGpuSleep.h"
#include <core/config/EngineConfig.h>
#include <core/log/InxLog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace infernux
{
namespace
{
uint64_t NowNanoseconds()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
} // namespace

class JoltGpuTick::Impl
{
    struct Lane;
    struct SharedStage;

  public:
    Impl(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs, JoltGpuBroadphase &broadphase,
         JoltGpuNarrowphase &narrowphase, JoltGpuConstraints &constraints, std::function<void()> onFailure,
         std::shared_ptr<Lane> sharedLane = {}, bool observation = false)
        : work(work), storage(storage), programs(programs), broadphase(broadphase), narrowphase(narrowphase),
          constraints(constraints), host(storage.GetHost()), failureObserver(std::move(onFailure)),
          lane(sharedLane ? std::move(sharedLane) : std::make_shared<Lane>())
    {
        if (lane->members[0]) {
            const size_t slot = observation ? 2 : 1;
            if (lane->members[slot])
                throw std::logic_error("GPU source already owns its bounded member slot");
            lane->members[slot] = this;
            sourceId = lane->members[0]->sourceId;
            isolateParameters = !observation;
            return;
        }
        lane->members[0] = this;
        FrameWorkSource source;
        source.observe = [this] {
            try {
                lane->Poll();
                if (pressureObserve)
                    pressureObserved = pressureObserve();
            } catch (...) {
                Fail();
                throw;
            }
        };
        source.refresh = source.observe;
        source.describe = [this] { return lane->Describe(); };
        source.advance = [this](bool gpu, uint64_t token) { return lane->Advance(gpu, token); };
        sourceId = work.AddSource(std::move(source));
    }

    ~Impl()
    {
        // Shutdown only. Normal Stop cancels, observes CanRetire, then destroys.
        Cancel();
        if (!receipts.empty()) {
            receipts.back().completion.Wait();
            Poll();
        }
        while (!sharedStages.empty()) {
            sharedStages.front()->completion->Wait();
            ++lane->pollSerial;
            Poll();
        }
        const bool primary = lane->members[0] == this;
        if (lane->focus == this)
            lane->focus = nullptr;
        for (auto &member : lane->members)
            if (member == this)
                member = nullptr;
        if (primary)
            work.RemoveSource(sourceId);
    }

    void Begin(uint32_t worldRow, float deltaTime, const JoltGpuStepParameters *settings,
               JoltGpuTickControl pendingControl, std::optional<std::vector<JPH::GPU::BodyForces>> forces)
    {
        FrameWorkOwner::Scope scope(work);
        if (pressureActive || (status != Status::Idle && status != Status::Complete))
            throw std::logic_error("GPU tick Begin requires an idle or fully completed owner");
        if (settings && (!std::isfinite(deltaTime) || deltaTime <= 0))
            throw std::invalid_argument("GPU fixed delta time must be finite and positive");
        if (!pendingControl.commandProgram && std::any_of(pendingControl.groups.begin(), pendingControl.groups.end(),
                                                          [](const auto &group) { return !group.commands.empty(); }))
            throw std::invalid_argument("GPU tick commands require their command program");
        ValidateControlPacket(pendingControl.initialization);
        if (!settings && this == lane->members[0])
            lane->bodyBatchInputs = {};
        selectedRow = worldRow;
        parameters = settings ? std::make_optional(*settings) : std::nullopt;
        reuseParameters = false;
        stageCursor = {};
        earlyFeedbackOpportunity = 0;
        reducedGuardWindow = false;
        if (parameters) {
            auto gravityDt = parameters->Get().gravityDt;
            gravityDt.w = deltaTime / static_cast<float>(parameters->Get().solve.y);
            parameters->SetGravityAndDeltaTime(gravityDt);
        }

        fixedDeltaTime = deltaTime;
        controlOnly = settings == nullptr;
        cpuForces = std::move(forces);
        forceWakePending = false;
        constraintTopology.reset();
        prescribedBounds.reset();
        inputs = {};
        control = std::move(pendingControl);
        // All normal and native callers use the current public configuration,
        // including a standalone Tick that has no Scene control factories.
        control.sleepParameters.timing = {0.0f, EngineConfig::Get().physicsTimeBeforeSleep,
                                          EngineConfig::Get().physicsPointVelocitySleepThreshold, 0.0f};
        workContinuation = {};
        groupCursor = 0;
        modelPrepared = false;
        modelUpdate.reset();
        commandCursor.reset();
        controlAccepted = false;
        finalPrepared = finalAccepted = false;
        finalOutput = {};
        snapshotRegion = 0;
        snapshotOffset = 0;
        snapshotMetadataOffset = 0;
        snapshotMetadata.clear();
        snapshotComplete = false;
        cursor = 0;
        sharedFallbackCursor.reset();
        inputAccepted = controlOnly;
        submissionBusy = false;
        status = Status::Running;
        try {
            if (!control.initialization && control.groups.empty())
                CaptureControlInputs();
        } catch (...) {
            Fail();
            throw;
        }
        work.Wake(sourceId);
    }

    // The observer is owner bookkeeping only: it must not throw, submit, wait,
    // or destroy this source. Accepted receipts remain owned until retirement.
    void Fail() noexcept
    {
        const bool firstFailure = status != Status::Failed;
        status = Status::Failed;
        Cancel();
        if (firstFailure && failureObserver)
            failureObserver();
    }

    void Cancel() noexcept
    {
        lane->DropShared(this);
        if (!controlOnly && (status == Status::Running || status == Status::Failed))
            storage.ClearContactHistory(selectedRow);
        if (status == Status::Running)
            status = Status::Cancelled;
        CancelPackets();
        if (suspendedWork) {
            if (suspendedWork->batch)
                suspendedWork->batch->Cancel();
            if (suspendedWork->preservation)
                suspendedWork->preservation->Cancel();
            suspendedWork.reset();
        }
        pressureActive = false;
        reducedGuardWindow = false;
        broadphase.ResetCapacityHint();
        inputs = {};
        workGeometry.reset();
        parameterInput = {};
        acceptedParameters.reset();
        parameters.reset();
        constraintTopology.reset();
        prescribedBounds.reset();
        cpuForces.reset();
        control = {};
        workContinuation = {};
        workMinimum = {};
        finalOutput = {};
        modelUpdate.reset();
        commandCursor.reset();
        submissionBusy = false;
    }

    bool CanRetire() const noexcept
    {
        return status != Status::Running && !pressureActive && receipts.empty() && sharedStages.empty();
    }

    FrameWorkWaitResult WaitForProgress(uint64_t maxWaitNanoseconds)
    {
        if (receipts.empty()) {
            if (lane->prepared && lane->prepared->peer == this && lane->prepared->owner != this)
                return lane->prepared->owner->WaitForProgress(maxWaitNanoseconds);
            for (const auto &stage : sharedStages) {
                if (!stage->completion)
                    continue;
                const auto count = sharedStages.size();
                const bool done = host.Queue().WaitFor(stage->completion->GetSubmission(), maxWaitNanoseconds);
                lane->Poll();
                return sharedStages.size() != count ? FrameWorkWaitResult::Progress
                       : done                       ? FrameWorkWaitResult::Blocked
                                                    : FrameWorkWaitResult::Waiting;
            }
        }
        if (receipts.empty()) {
            try {
                if (batch)
                    if (const auto prepared = batch->WaitForPreparation(maxWaitNanoseconds))
                        return *prepared ? FrameWorkWaitResult::Progress : FrameWorkWaitResult::Waiting;
            } catch (...) {
                Fail();
                throw;
            }
            // A refused prepared packet may need a slot held by another
            // accepted user of this same queue, outside this Tick source.
            const auto dependency =
                submissionBusy && Submission() ? host.Queue().GetCapacityPredecessor() : rhi::SubmissionTicket{};
            if (!dependency.IsValid())
                return FrameWorkWaitResult::Blocked;
            try {
                const bool completed = host.Queue().WaitFor(dependency, maxWaitNanoseconds);
                lane->Poll();
                return completed ? FrameWorkWaitResult::Progress : FrameWorkWaitResult::Waiting;
            } catch (...) {
                Fail();
                throw;
            }
        }
        const auto count = receipts.size();
        try {
            const bool completed =
                host.Queue().WaitFor(receipts.front().completion.GetSubmission(), maxWaitNanoseconds);
            Poll();
            if (receipts.size() != count)
                return FrameWorkWaitResult::Progress;
            return completed ? FrameWorkWaitResult::Blocked : FrameWorkWaitResult::Waiting;
        } catch (...) {
            Fail();
            throw;
        }
    }

    FrameWorkOwner &work;
    JoltGpuStorage &storage;
    JoltGpuPrograms &programs;
    JoltGpuBroadphase &broadphase;
    JoltGpuNarrowphase &narrowphase;
    JoltGpuConstraints &constraints;
    rhi::ComputeHost host;
    std::function<void()> failureObserver;
    FrameWorkOwner::SourceId sourceId = 0;
    Status status = Status::Idle;
    bool admissionEnabled = true;
    float fixedDeltaTime = 0;
    size_t cursor = 0;
    JoltGpuTickOutput completedOutput;

  private:
    enum class PacketKind
    {
        Initialization,
        Model,
        Commands,
        Work,
        NumericalWork,
        FinalWork,
        Inputs,
        Stages,
        Preservation,
        Pressure
    };
    enum class Phase
    {
        InitializeSleep,
        CapturePrescribedBounds,
        PrepareKinematics,
        CaptureArticulationRotors,
        ApplyForces,
        ParticipantPredict,
        ParticipantPrepare,
        ParticipantExchange,
        ParticipantAccept,
        WakePrescribedBounds,
        BuildBounds,
        Broadphase,
        CollidePairs,
        CompleteConvexContacts,
        BuildTopology,
        WakeComponents,
        PrepareCoupledContacts,
        WakeExternalComponents,
        SetupContacts,
        SetupJoints,
        ReportContactsBeforeVelocity,
        BuildConstraints,
        ClassifyArticulations,
        WarmJoints,
        WarmContacts,
        WarmCoupledContacts,
        SolveJointVelocity,
        SolveVelocity,
        ExchangeCoupledContacts,
        ReportContactsAfterVelocity,
        IntegrateCoupledBodies,
        PrepareCoupledSweep,
        PrepareSweep,
        CaptureSweepMotion,
        BuildSweptBounds,
        SweptBroadphase,
        ResolveSweep,
        ResolveCoupledSweep,
        SettleCoupledSweep,
        SolveSweepEvent,
        ReportSweep,
        Integrate,
        IntegrateSwept,
        CommitArticulationPose,
        SolveJointPosition,
        SolvePosition,
        ProjectCoupledContacts,
        FinishJoints,
        PrepareContactStore,
        StoreContactHistory,
        UpdateSleep,
        WakeFinalPrescribedBounds,
        FinishTick,
        Done
    };
    struct StageCursor
    {
        Phase phase = Phase::Done;
        uint32_t substep = 0;
        uint32_t iteration = 0;
        uint32_t islandBatchFirst = 0;
        uint32_t islandBatchLanes = 0;
        uint32_t localFirst = 0;
        float sweepRemaining = 0.0f;
        uint64_t sameTimeImpacts = 0;
        bool sweepAdvanced = false;
        std::optional<float> localSweepImpact;
    };
    struct StageSpan
    {
        JPH::GPU::Stage stage;
        JPH::GPU::Domain domain;
    };

    struct UnitEstimate
    {
        size_t count = 0;
        uint64_t nanoseconds = 0;
    };
    static constexpr size_t ColdPacketUnits = 8;
    static constexpr size_t MaxPacketUnits = 16;
    using UnitHistory = std::deque<UnitEstimate>;
    using UnitCostKey = std::tuple<JPH::GPU::Stage, uint32_t, uint32_t>;
    using NumericalCostKey = std::vector<UnitCostKey>;
    using CostKey = std::tuple<std::vector<uint64_t>, uint64_t, NumericalCostKey>;
    struct Receipt
    {
        rhi::ComputeCompletion completion;
        CostKey key;
        PacketKind kind;
        size_t stageCount;
        std::optional<JPH::GPU::Stage> homogeneousStage;
        uint64_t submittedOpportunity;
        uint64_t gpuGrant;
        uint64_t transferGrant;
        uint64_t transferBytes;
        uint64_t submitStarted;
        std::shared_ptr<const JoltGpuGeometryLease> geometry;
    };

    struct SharedStage
    {
        Impl *owner = nullptr, *peer = nullptr;
        StageCursor peerEnd;
        size_t peerCursorEnd = 0;
        CostKey key;
        JPH::GPU::Stage stage;
        std::shared_ptr<const JoltGpuGeometryLease> ownerGeometry, peerGeometry;
        std::optional<rhi::ComputeCompletion> completion;
        uint64_t opportunity = 0, gpuGrant = 0, transferGrant = 0, bytes = 0, started = 0;
        bool observed = false;
        uint64_t lastPoll = 0;
    };
    struct Lane
    {
        std::array<Impl *, 3> members{};
        std::shared_ptr<SharedStage> prepared;
        JoltGpuBodyBatchInputs bodyBatchInputs;
        size_t nextHost = 0, nextGpu = 0;
        uint64_t pollSerial = 0;
        Impl *focus = nullptr;
        void Poll();
        FrameWorkFacts Describe() const;
        bool Advance(bool gpu, uint64_t token);
        bool PrepareShared();
        void DropShared(Impl *member) noexcept;
        bool OtherReceipts(const Impl *member) const noexcept;
    };

    static NumericalCostKey NumericalKey(const std::vector<JoltGpuStageWork> &ready)
    {
        NumericalCostKey key;
        key.reserve(ready.size());
        for (const auto &unit : ready)
            key.emplace_back(unit.stage, unit.parameters.range.y, unit.parameters.range.w);
        return key;
    }

    static CostKey Key(const rhi::PreparedComputeBatch &batch, const NumericalCostKey &numerical = {})
    {
        // Keep the physical loop counts separate from the RHI signature. In a
        // island solver dispatch, local item count does not appear in its group counts.
        return {batch.GetExecutionSignature(), batch.GetTransferBytes(), numerical};
    }

    CostKey SubmissionKey(const rhi::PreparedComputeBatch &packet) const
    {
        if (!preservation && batchKind == PacketKind::Stages)
            return Key(packet, batchNumericalKey);
        return Key(packet);
    }

    static void ObserveUnitCost(UnitHistory &history, size_t count, uint64_t nanoseconds)
    {
        if (!count || !nanoseconds)
            return;
        if (history.size() == 8)
            history.pop_front();
        history.push_back({count, nanoseconds});
    }

    static long double PlanningCost(const UnitHistory &history)
    {
        long double cost = 0;
        for (const auto &sample : history)
            cost = std::max(cost, static_cast<long double>(sample.nanoseconds) / sample.count);
        return cost;
    }

    uint64_t Prediction(const CostKey &key) const
    {
        const auto found = costs.find(key);
        return found == costs.end() ? 0 : *std::max_element(found->second.begin(), found->second.end());
    }

    void ObserveCost(const CostKey &key, uint64_t estimate)
    {
        if (!estimate)
            return;
        auto &samples = costs[key];
        if (samples.size() == 8)
            samples.pop_front();
        samples.push_back(estimate);
    }

    void Poll()
    {
        for (auto it = sharedStages.begin(); it != sharedStages.end();) {
            const auto stage = *it;
            if (!stage->completion) {
                ++it;
                continue;
            }
            if (!stage->completion->HasRetired()) {
                if (stage->lastPoll == lane->pollSerial) {
                    ++it;
                    continue;
                }
                stage->lastPoll = lane->pollSerial;
                try {
                    if (!stage->completion->IsComplete()) {
                        ++it;
                        continue;
                    }
                } catch (...) {
                    stage->owner->Fail();
                    stage->peer->Fail();
                    throw;
                }
            }
            if (!stage->observed) {
                stage->observed = true;
                const auto elapsed = stage->completion->GetDeviceElapsedNanoseconds();
                if (elapsed)
                    work.ObserveResource(FrameWorkResource::Gpu, static_cast<uint64_t>(std::ceil(*elapsed)),
                                         stage->opportunity, stage->gpuGrant);
                if (stage->bytes)
                    work.ObserveResource(FrameWorkResource::Transfer, stage->bytes, stage->opportunity,
                                         stage->transferGrant);
                const auto estimate =
                    elapsed ? static_cast<uint64_t>(std::ceil(*elapsed)) : NowNanoseconds() - stage->started;
                stage->owner->ObserveCost(stage->key, estimate);
                ObserveUnitCost(stage->owner->stageEstimates[stage->stage], 2, estimate);
                // One packet measurement; the peer does not receive a second sample.
            }
            it = sharedStages.erase(it);
            work.ReportProgress();
        }
        if (submissionBusy && (busyOpportunity != work.Opportunity() || busyProgress != work.ProgressSerial() ||
                               busySubmissions != host.Queue().GetPendingSubmissionCount()))
            submissionBusy = false;
        while (!receipts.empty() && receipts.front().completion.IsComplete()) {
            auto &receipt = receipts.front();
            const auto elapsed = receipt.completion.GetDeviceElapsedNanoseconds();
            uint64_t estimate = 0;
            if (elapsed) {
                const auto actual = static_cast<uint64_t>(std::ceil(*elapsed));
                work.ObserveResource(FrameWorkResource::Gpu, actual, receipt.submittedOpportunity, receipt.gpuGrant);
                estimate = actual;
            }
            // This is a conservative host-observed completion interval,
            // including recording, queueing and observation lag. It is only an
            // estimate: missing GPU time never refunds a grant as zero work.
            if (!estimate)
                estimate = NowNanoseconds() - receipt.submitStarted;
            ObserveCost(receipt.key, estimate);
            if (receipt.transferBytes)
                work.ObserveResource(FrameWorkResource::Transfer, receipt.transferBytes, receipt.submittedOpportunity,
                                     receipt.transferGrant);
            if (receipt.kind == PacketKind::Stages) {
                if (receipt.homogeneousStage)
                    ObserveUnitCost(stageEstimates[*receipt.homogeneousStage], receipt.stageCount, estimate);
                else
                    ObserveUnitCost(mixedEstimates, receipt.stageCount, estimate);
            } else if (receipt.kind == PacketKind::Commands) {
                lastCommandCount = receipt.stageCount;
                lastCommandEstimate = estimate;
            }
            receipts.pop_front();
            submissionBusy = false;
            work.ReportProgress();
        }
        FinishCompleted();
    }

    void CancelPackets() noexcept
    {
        for (auto *packet : {&batch, &preservation}) {
            if (*packet)
                (*packet)->Cancel();
            packet->reset();
        }
        batchNumericalKey.clear();
    }

    struct PressureSavedWork
    {
        std::shared_ptr<rhi::PreparedComputeBatch> batch, preservation;
        std::function<JoltGpuControlWork()> continuation, minimum;
        std::shared_ptr<const JoltGpuGeometryLease> geometry;
        PacketKind kind = PacketKind::Inputs, continuationKind = PacketKind::Work;
        JoltGpuControlWork::ResumeAfter resumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
        NumericalCostKey numericalKey;
        std::optional<JPH::GPU::Stage> homogeneousStage;
        size_t begin = 0, end = 0;
        StageCursor stageEnd;
        bool busy = false, minimumReducesDispatchWork = true;
        uint64_t opportunity = 0, progress = 0, submissions = 0;
    };

    void SwapPressureWork(PressureSavedWork &saved) noexcept
    {
        using std::swap;
        swap(batch, saved.batch);
        swap(preservation, saved.preservation);
        swap(workContinuation, saved.continuation);
        swap(workMinimum, saved.minimum);
        swap(minimumReducesDispatchWork, saved.minimumReducesDispatchWork);
        swap(workGeometry, saved.geometry);
        swap(batchKind, saved.kind);
        swap(continuationKind, saved.continuationKind);
        swap(continuationResumeAfter, saved.resumeAfter);
        swap(batchNumericalKey, saved.numericalKey);
        swap(batchHomogeneousStage, saved.homogeneousStage);
        swap(batchBegin, saved.begin);
        swap(batchEnd, saved.end);
        swap(batchStageEnd, saved.stageEnd);
        swap(submissionBusy, saved.busy);
        swap(busyOpportunity, saved.opportunity);
        swap(busyProgress, saved.progress);
        swap(busySubmissions, saved.submissions);
    }

    bool GuardConsumersFinished() const noexcept
    {
        return stageCursor.phase == Phase::PrepareContactStore || stageCursor.phase == Phase::StoreContactHistory ||
               stageCursor.phase == Phase::ParticipantAccept || stageCursor.phase == Phase::UpdateSleep ||
               stageCursor.phase == Phase::WakeFinalPrescribedBounds ||
               stageCursor.phase == Phase::CaptureArticulationRotors || stageCursor.phase == Phase::FinishTick ||
               stageCursor.phase == Phase::Done;
    }

    bool PressureSafe() const noexcept
    {
        if (lane->OtherReceipts(this))
            return false;
        if (lane->members[1] && lane->members[1] != this && lane->members[1]->tickParameters.selectedCounts.z &&
            ((lane->members[1]->control.articulations && lane->members[1]->control.articulations->preparing) ||
             (lane->members[1]->reducedGuardWindow && !lane->members[1]->GuardConsumersFinished())))
            return false;
        if (!pressurePrepare || (status != Status::Running && status != Status::Idle && status != Status::Complete))
            return false;
        if (status == Status::Running && control.articulations && control.articulations->preparing)
            return false; // Complete or decline this private cold proposal first.
        // Admitted is also read by native position guards after reduced pose.
        // Keep the whole original law through the final guarded joint work.
        // Normalize can skip an empty FinishJoints without accepting a native
        // stage packet (for example a zero-joint World). Use the actual cursor
        // as well as the latch so that such Worlds cannot strand peer caches.
        return !reducedGuardWindow || GuardConsumersFinished();
    }

    bool PressureRequested() const
    {
        return !pressureActive && pressurePending && pressurePending() && PressureSafe();
    }

    bool StartPressure()
    {
        lane->DropShared(this);
        if (!receipts.empty())
            return false;
        const bool capturedWitness =
            status == Status::Running && controlAccepted && !controlOnly &&
            (stageCursor.phase == Phase::SetupJoints || stageCursor.phase == Phase::ReportContactsBeforeVelocity ||
             stageCursor.phase == Phase::BuildConstraints || stageCursor.phase == Phase::ClassifyArticulations);

        if (batch && batchKind == PacketKind::Model) {
            // The transaction has not been accepted. Drop its frozen old
            // runtime and private remap target, then prepare it from converted
            // histories through the original author factory exactly once.
            CancelPackets();
            modelUpdate.reset();
            modelPrepared = false;
            workContinuation = control.groups[groupCursor].reprepareModel;
            continuationKind = PacketKind::Work;
            continuationResumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
            workMinimum = {};
            submissionBusy = false;
        } else if (status == Status::Running && !controlAccepted && !modelPrepared && !batch && !workContinuation &&
                   groupCursor < control.groups.size() && control.groups[groupCursor].reprepareModel) {
            // The layout preparation group can be complete while its commit
            // group has not prepared a packet yet. Retirement invalidates that
            // private joint history too; rebuild before calling Commit.
            workContinuation = control.groups[groupCursor].reprepareModel;
            continuationKind = PacketKind::Work;
            continuationResumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
            submissionBusy = false;
        } else if (status == Status::Running && controlAccepted && !controlOnly &&
                   stageCursor.phase == Phase::CaptureArticulationRotors) {
            // Only scratch/capture work has run; no Setup or force is replayed.
            // Its accepted predecessors have settled. Re-enter this phase with
            // the declined descriptor, which advances directly to ApplyForces.
            CancelPackets();
            workContinuation = {};
            workMinimum = {};
            workGeometry.reset();
            submissionBusy = false;
        } else if (batch && batchKind == PacketKind::Stages) {
            // Native stageCursor advances only on acceptance. Rebuilding this
            // unaccepted prefix replaces all old control bindings safely.
            CancelPackets();
            batchNumericalKey.clear();
            submissionBusy = false;
        }

        if (status == Status::Running && finalPrepared && !snapshotComplete &&
            (continuationKind == PacketKind::FinalWork || (batch && batchKind == PacketKind::FinalWork))) {
            CancelPackets();
            finalOutput.states.reset();
            finalOutput.joints.reset();
            finalOutput.jointLayout.reset();
            snapshotRegion = 0;
            snapshotOffset = snapshotMetadataOffset = 0;
            snapshotMetadata.clear();
            workContinuation = [this] { return PrepareSnapshot(); };
            continuationKind = PacketKind::FinalWork;
            continuationResumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
            workMinimum = {};
            submissionBusy = false;
        }

        suspendedWork = std::make_unique<PressureSavedWork>();
        SwapPressureWork(*suspendedWork);
        pressureActive = true;
        pressureChanged = false;
        SelectControlWork(pressurePrepare(capturedWitness ? selectedRow : UINT32_MAX,
                                          [this](uint32_t world, auto empty) {
                                              pressureChanged = true;
                                              if (status == Status::Running && world == selectedRow) {
                                                  inputs.SetArticulations(empty);
                                                  if (control.articulations)
                                                      control.articulations->value = std::move(empty);
                                              }
                                          }),
                          PacketKind::Pressure);
        return true;
    }

    void FinishPressure()
    {
        // All conversion summaries and actual receipts have completed. Existing
        // unpublished layout destinations must copy the converted joint history.
        if (pressureChanged)
            for (auto &group : control.groups)
                if (group.retireJointRuntime)
                    group.retireJointRuntime();
        SwapPressureWork(*suspendedWork);
        suspendedWork.reset();
        pressureActive = false;
        // The retirement itself changed capacity; retry the saved work now.
        if (pressureChanged) {
            submissionBusy = false;
            work.ReportProgress();
        }
    }

    static bool ProducerPhase(Phase phase)
    {
        return phase == Phase::ParticipantPredict || phase == Phase::ParticipantPrepare ||
               phase == Phase::ParticipantExchange || phase == Phase::ParticipantAccept || phase == Phase::Broadphase ||
               phase == Phase::CompleteConvexContacts || phase == Phase::InitializeSleep ||
               phase == Phase::CapturePrescribedBounds || phase == Phase::WakePrescribedBounds ||
               phase == Phase::BuildTopology || phase == Phase::WakeComponents ||
               phase == Phase::WakeExternalComponents || phase == Phase::UpdateSleep ||
               phase == Phase::WakeFinalPrescribedBounds || phase == Phase::BuildConstraints ||
               phase == Phase::CommitArticulationPose || phase == Phase::CaptureArticulationRotors ||
               phase == Phase::ClassifyArticulations || phase == Phase::ReportContactsBeforeVelocity ||
               phase == Phase::ReportContactsAfterVelocity || phase == Phase::PrepareCoupledContacts ||
               phase == Phase::IntegrateCoupledBodies || phase == Phase::PrepareCoupledSweep ||
               phase == Phase::ResolveCoupledSweep || phase == Phase::SettleCoupledSweep ||
               phase == Phase::PrepareContactStore || phase == Phase::PrepareSweep || phase == Phase::SweptBroadphase ||
               phase == Phase::ResolveSweep || phase == Phase::SolveSweepEvent || phase == Phase::ReportSweep ||
               CoupledFeedbackPhase(phase);
    }

    StageSpan Span(Phase phase) const
    {
        using JPH::GPU::Domain;
        using JPH::GPU::Stage;
        switch (phase) {
        case Phase::PrepareKinematics:
            return {Stage::PrepareKinematics, Domain::Bodies};
        case Phase::ApplyForces:
            return {Stage::ApplyForces, Domain::Bodies};
        case Phase::BuildBounds:
            return {Stage::BuildBounds, Domain::Shapes};
        case Phase::CaptureSweepMotion:
            return {Stage::CaptureSweepMotion, Domain::Bodies};
        case Phase::BuildSweptBounds:
            return {Stage::BuildSweptBounds, Domain::Shapes};
        case Phase::CollidePairs:
            return {Stage::CollidePairs, Domain::Patches};
        case Phase::SetupContacts:
            return {Stage::SetupContacts, Domain::Contacts};
        case Phase::SetupJoints:
            return {Stage::SetupJoints, Domain::Joints};
        case Phase::WarmJoints:
            return {Stage::WarmJoints, Domain::IslandJoints};
        case Phase::WarmContacts:
            return {Stage::WarmContacts, Domain::IslandPatches};
        case Phase::SolveJointVelocity:
            return {Stage::SolveJointVelocity, Domain::IslandJoints};
        case Phase::SolveVelocity:
            return {Stage::SolveVelocity, Domain::IslandPatches};
        case Phase::StoreContactHistory:
            return {Stage::StoreContactHistory, Domain::Patches};
        case Phase::Integrate:
            return {Stage::Integrate, Domain::Bodies};
        case Phase::IntegrateSwept:
            return {Stage::IntegrateSwept, Domain::Bodies};
        case Phase::SolveJointPosition:
            return {Stage::SolveJointPosition, Domain::IslandJoints};
        case Phase::SolvePosition:
            return {Stage::SolvePosition, Domain::IslandPatches};
        case Phase::FinishJoints:
            return {Stage::FinishJoints, Domain::Joints};
        case Phase::FinishTick:
            return {Stage::FinishTick, Domain::Bodies};
        case Phase::CommitArticulationPose:
        case Phase::CaptureArticulationRotors:
        case Phase::ClassifyArticulations:
        case Phase::Broadphase:
        case Phase::PrepareSweep:
        case Phase::SweptBroadphase:
        case Phase::ResolveSweep:
        case Phase::SolveSweepEvent:
        case Phase::ReportSweep:
        case Phase::InitializeSleep:
        case Phase::CapturePrescribedBounds:
        case Phase::WakePrescribedBounds:
        case Phase::CompleteConvexContacts:
        case Phase::BuildTopology:
        case Phase::WakeComponents:
        case Phase::WakeExternalComponents:
        case Phase::UpdateSleep:
        case Phase::WakeFinalPrescribedBounds:
        case Phase::BuildConstraints:
        case Phase::PrepareContactStore:
        case Phase::ReportContactsBeforeVelocity:
        case Phase::ReportContactsAfterVelocity:
        case Phase::PrepareCoupledContacts:
        case Phase::IntegrateCoupledBodies:
        case Phase::PrepareCoupledSweep:
        case Phase::ResolveCoupledSweep:
        case Phase::SettleCoupledSweep:
        case Phase::WarmCoupledContacts:
        case Phase::ExchangeCoupledContacts:
        case Phase::ProjectCoupledContacts:
        case Phase::Done:
            break;
        }
        throw std::logic_error("A completed GPU tick has no stage span");
    }

    static bool IslandDomain(JPH::GPU::Domain domain)
    {
        using JPH::GPU::Domain;
        return domain == Domain::IslandJoints || domain == Domain::IslandPatches;
    }

    uint32_t IslandCount(JPH::GPU::Domain domain) const
    {
        const auto &plan = inputs.GetConstraints();
        if (!plan)
            throw std::logic_error("Island phase requires its installed constraint plan");
        return domain == JPH::GPU::Domain::IslandJoints ? plan->jointIslandCount : plan->contactIslandCount;
    }

    uint32_t IslandMaxItems(JPH::GPU::Domain domain) const
    {
        const auto &plan = inputs.GetConstraints();
        if (!plan)
            throw std::logic_error("Island phase requires its installed constraint plan");
        return domain == JPH::GPU::Domain::IslandJoints ? plan->maxJointItems : plan->maxPatchItems;
    }

    uint32_t ItemCount(JPH::GPU::Domain domain) const
    {
        using JPH::GPU::Domain;
        switch (domain) {
        case Domain::Bodies:
            return tickParameters.selectedCounts.x;
        case Domain::Shapes:
            return tickParameters.selectedCounts.y;
        case Domain::Patches:
            return inputs.GetContacts() ? inputs.GetContacts()->patchCount : 0;
        case Domain::Contacts:
            return inputs.GetContacts() ? inputs.GetContacts()->patchCount * JPH::GPU::MaxContactPointsPerPair : 0;
        case Domain::Joints:
            return tickParameters.selectedCounts.z;
        default:
            throw std::logic_error("Island-local GPU work requires its installed schedule");
        }
    }

    bool HasRigidWork() const noexcept
    {
        return control.simulateRigidBodies && tickParameters.selectedCounts.x;
    }

    void NextSpan(StageCursor &next) const
    {
        const auto &plan = inputs.GetConstraints();
        const bool localRows = plan && (plan->jointIslandCount || plan->contactIslandCount);
        const bool exchange = control.participants.Has(JoltGpuParticipantStage::Exchange, next.substep);
        const bool participantCpu = control.participants.hasCpuTargets;
        // Discovery runs each substep. Only its actual rows or a CPU participant need a round boundary.
        const bool coupledRows =
            control.prepareCoupledContacts && (participantCpu || !inputs.GetExternalContactBodies().empty());
        const bool velocity = tickParameters.counts.w && (localRows || exchange || coupledRows);
        const bool position = tickParameters.solve.x && (localRows || coupledRows);
        const bool coupledSweep = inputs.NeedsCoupledSweep();
        const auto localDrift =
            storage.HasSweepBodies(selectedRow) || coupledSweep ? Phase::PrepareSweep : Phase::Integrate;
        const auto drift = control.prepareCoupledContacts
                               ? (coupledSweep ? Phase::PrepareCoupledSweep : Phase::IntegrateCoupledBodies)
                               : localDrift;
        next.islandBatchFirst = 0;
        next.islandBatchLanes = 0;
        next.localFirst = 0;
        switch (next.phase) {
        case Phase::InitializeSleep:
            next.phase = Phase::CapturePrescribedBounds;
            break;
        case Phase::CapturePrescribedBounds:
            next.phase = Phase::PrepareKinematics;
            break;
        case Phase::PrepareKinematics:
            next.phase = Phase::CaptureArticulationRotors;
            break;
        case Phase::CaptureArticulationRotors:
            next.phase = Phase::ApplyForces;
            break;
        case Phase::ApplyForces:
            next.phase = control.participants ? Phase::ParticipantPredict : Phase::WakePrescribedBounds;
            break;
        case Phase::ParticipantPredict:
            next.phase = HasRigidWork() ? Phase::WakePrescribedBounds : Phase::ParticipantPrepare;
            break;
        case Phase::ParticipantPrepare:
            next.iteration = 0;
            next.phase = HasRigidWork()   ? Phase::BuildTopology
                         : participantCpu ? Phase::PrepareCoupledContacts
                         : velocity       ? Phase::ParticipantExchange
                                          : Phase::ParticipantAccept;
            break;
        case Phase::ParticipantExchange:
            if (HasRigidWork())
                next.phase = Phase::SolveJointVelocity;
            else if (participantCpu)
                next.phase = Phase::ExchangeCoupledContacts;
            else
                next.phase =
                    ++next.iteration < tickParameters.counts.w ? Phase::ParticipantExchange : Phase::ParticipantAccept;
            break;
        case Phase::ParticipantAccept:
            if (HasRigidWork())
                next.phase = Phase::UpdateSleep;
            else {
                next.iteration = 0;
                next.phase = ++next.substep < tickParameters.solve.y ? Phase::ParticipantPredict : Phase::Done;
            }
            break;
        case Phase::WakePrescribedBounds:
            next.phase = Phase::BuildBounds;
            break;
        case Phase::BuildBounds:
            next.phase = Phase::Broadphase;
            break;
        case Phase::Broadphase:
            next.phase = Phase::CollidePairs;
            break;
        case Phase::CollidePairs:
            next.phase = inputs.HasGenericGeometry() ? Phase::CompleteConvexContacts
                         : control.participants      ? Phase::ParticipantPrepare
                                                     : Phase::BuildTopology;
            break;
        case Phase::CompleteConvexContacts:
            next.phase = control.participants ? Phase::ParticipantPrepare : Phase::BuildTopology;
            break;
        case Phase::BuildTopology:
            next.phase = Phase::WakeComponents;
            break;
        case Phase::WakeComponents:
            next.phase = control.prepareCoupledContacts ? Phase::PrepareCoupledContacts : Phase::SetupContacts;
            break;
        case Phase::PrepareCoupledContacts:
            if (HasRigidWork())
                next.phase = inputs.HasExternalWakePending() ? Phase::WakeExternalComponents : Phase::SetupContacts;
            else
                next.phase = velocity ? Phase::WarmCoupledContacts : Phase::IntegrateCoupledBodies;
            break;
        case Phase::WakeExternalComponents:
            next.phase = Phase::SetupContacts;
            break;
        case Phase::SetupContacts:
            next.phase = Phase::SetupJoints;
            break;
        case Phase::SetupJoints:
            next.phase = control.prepareContactReport ? Phase::ReportContactsBeforeVelocity : Phase::BuildConstraints;
            break;
        case Phase::ReportContactsBeforeVelocity:
            next.phase = Phase::BuildConstraints;
            break;
        case Phase::ReportContactsAfterVelocity:
            next.phase = drift;
            break;
        case Phase::IntegrateCoupledBodies:
            if (HasRigidWork()) {
                next.phase = localDrift;
            } else {
                next.iteration = 0;
                next.phase = position ? Phase::ProjectCoupledContacts : Phase::ParticipantAccept;
            }
            break;
        case Phase::PrepareCoupledSweep:
            next.phase = Phase::PrepareSweep;
            break;
        case Phase::BuildConstraints:
            next.phase = Phase::ClassifyArticulations;
            break;
        case Phase::ClassifyArticulations:
            next.iteration = 0;
            next.phase = velocity                       ? Phase::WarmJoints
                         : control.prepareContactReport ? Phase::ReportContactsAfterVelocity
                                                        : drift;
            break;
        case Phase::WarmJoints:
            next.phase = Phase::WarmContacts;
            break;
        case Phase::WarmContacts:
            if (coupledRows) {
                next.phase = Phase::WarmCoupledContacts;
                break;
            }
            [[fallthrough]];
        case Phase::WarmCoupledContacts:
            next.phase = exchange ? Phase::ParticipantExchange : Phase::SolveJointVelocity;
            break;
        case Phase::SolveJointVelocity:
            next.phase = Phase::SolveVelocity;
            break;
        case Phase::SolveVelocity:
            if (coupledRows) {
                next.phase = Phase::ExchangeCoupledContacts;
                break;
            }
            [[fallthrough]];
        case Phase::ExchangeCoupledContacts:
            next.phase = ++next.iteration < tickParameters.counts.w
                             ? (exchange ? Phase::ParticipantExchange : Phase::SolveJointVelocity)
                         : HasRigidWork() && control.prepareContactReport ? Phase::ReportContactsAfterVelocity
                                                                          : drift;
            break;
        case Phase::PrepareSweep:
            next.phase = Phase::CaptureSweepMotion;
            break;
        case Phase::CaptureSweepMotion:
            next.phase = Phase::BuildSweptBounds;
            break;
        case Phase::BuildSweptBounds:
            next.phase = Phase::SweptBroadphase;
            break;
        case Phase::SweptBroadphase:
            next.phase =
                inputs.GetSweepContacts()->patchCount || coupledSweep ? Phase::ResolveSweep : Phase::IntegrateSwept;
            break;
        case Phase::ResolveSweep:
            if (coupledSweep) {
                next.phase = Phase::ResolveCoupledSweep;
                break;
            }
            [[fallthrough]];
        case Phase::ResolveCoupledSweep:
            next.phase = inputs.SweepImpactTime() ? Phase::SolveSweepEvent : Phase::IntegrateSwept;
            break;
        case Phase::SolveSweepEvent:
            if (control.prepareContactReport && inputs.GetSweepContacts()->patchCount) {
                next.phase = Phase::ReportSweep;
                break;
            }
            [[fallthrough]];
        case Phase::ReportSweep:
            if (inputs.SweepSettlementPending()) {
                next.phase = Phase::SettleCoupledSweep;
                break;
            }
            next.phase = next.sweepRemaining > 0.0f ? Phase::BuildSweptBounds : Phase::IntegrateSwept;
            break;
        case Phase::SettleCoupledSweep:
            next.localSweepImpact.reset();
            next.phase = inputs.SweepImpactTime()     ? Phase::SolveSweepEvent
                         : next.sweepRemaining > 0.0f ? Phase::BuildSweptBounds
                                                      : Phase::IntegrateSwept;
            break;
        case Phase::Integrate:
        case Phase::IntegrateSwept:
            next.phase = Phase::CommitArticulationPose;
            break;
        case Phase::CommitArticulationPose:
            next.iteration = 0;
            next.phase = position ? Phase::SolveJointPosition : Phase::FinishJoints;
            break;
        case Phase::SolveJointPosition:
            next.phase = Phase::SolvePosition;
            break;
        case Phase::SolvePosition:
            if (coupledRows) {
                next.phase = Phase::ProjectCoupledContacts;
                break;
            }
            [[fallthrough]];
        case Phase::ProjectCoupledContacts:
            if (HasRigidWork())
                next.phase =
                    ++next.iteration < tickParameters.solve.x ? Phase::SolveJointPosition : Phase::FinishJoints;
            else
                next.phase = ++next.iteration < tickParameters.solve.x ? Phase::ProjectCoupledContacts
                                                                       : Phase::ParticipantAccept;
            break;
        case Phase::FinishJoints:
            if (inputs.NextContactHistory().patchCount) {
                next.phase =
                    inputs.NextContactHistory().buffer ? Phase::StoreContactHistory : Phase::PrepareContactStore;
                break;
            }
            [[fallthrough]];
        case Phase::StoreContactHistory:
            next.phase = control.participants ? Phase::ParticipantAccept : Phase::UpdateSleep;
            break;
        case Phase::PrepareContactStore:
            next.phase = Phase::StoreContactHistory;
            break;
        case Phase::UpdateSleep:
            next.phase = Phase::WakeFinalPrescribedBounds;
            break;
        case Phase::WakeFinalPrescribedBounds:
            next.phase = ++next.substep < tickParameters.solve.y ? Phase::CaptureArticulationRotors : Phase::FinishTick;
            break;
        case Phase::FinishTick:
            next.phase = Phase::Done;
            break;
        case Phase::Done:
            break;
        }
    }

    static std::optional<JoltGpuParticipantStage> ParticipantStage(Phase phase)
    {
        switch (phase) {
        case Phase::ParticipantPredict:
            return JoltGpuParticipantStage::Predict;
        case Phase::ParticipantPrepare:
            return JoltGpuParticipantStage::Prepare;
        case Phase::ParticipantExchange:
            return JoltGpuParticipantStage::Exchange;
        case Phase::ParticipantAccept:
            return JoltGpuParticipantStage::Accept;
        default:
            return {};
        }
    }

    void Normalize(StageCursor &next) const
    {
        while (next.phase != Phase::Done) {
            if (const auto stage = ParticipantStage(next.phase);
                stage && !control.participants.Has(*stage, next.substep)) {
                NextSpan(next);
                continue;
            }
            // CPU Jolt samples sleep history only at the final collision substep.
            if (next.phase == Phase::UpdateSleep && next.substep + 1u < tickParameters.solve.y) {
                NextSpan(next);
                continue;
            }
            if (next.phase == Phase::WarmContacts && !inputs.PreviousContactHistory().buffer) {
                NextSpan(next);
                continue;
            }
            if (ProducerPhase(next.phase))
                return;
            const auto span = Span(next.phase);
            if (IslandDomain(span.domain)) {
                if (next.islandBatchFirst < IslandCount(span.domain) && IslandMaxItems(span.domain))
                    return;
            } else if (next.localFirst < ItemCount(span.domain)) {
                return;
            }
            NextSpan(next);
        }
    }

    JoltGpuStageWork TakeUnit(StageCursor &next,
                              uint32_t bodyLimit = JPH::GPU::WorkItemLimit(JPH::GPU::Domain::Bodies)) const
    {
        const auto span = Span(next.phase);
        const auto quantum = span.domain == JPH::GPU::Domain::Bodies
                                 ? std::min(JPH::GPU::WorkItemLimit(span.domain), bodyLimit)
                                 : JPH::GPU::WorkItemLimit(span.domain);
        JPH::GPU::StageWork parameters{};
        if (IslandDomain(span.domain)) {
            const auto maximum = IslandMaxItems(span.domain);
            if (next.localFirst == 0) {
                const auto remaining = IslandCount(span.domain) - next.islandBatchFirst;
                const auto originalLanes = std::min({uint32_t{64}, quantum, remaining});
                const auto originalCount = std::min(quantum / originalLanes, maximum);
                // Spend unused rectangular items on independent islands without
                // reducing the old local slice. Large islands keep their slice
                // length; a small maximum can fill more than one workgroup.
                next.islandBatchLanes = std::min(remaining, quantum / originalCount);
            }
            // A batch owns these same islands until every local slice is done.
            // Recomputing its width for a short final slice would skip the
            // prefix of newly included islands after a yield/retry.
            const auto lanes = next.islandBatchLanes;
            const auto count = std::min(quantum / lanes, maximum - next.localFirst);
            parameters.range = {next.localFirst, count, next.islandBatchFirst, lanes};
            next.localFirst += count;
            if (next.localFirst == maximum) {
                next.localFirst = 0;
                next.islandBatchFirst += lanes;
                next.islandBatchLanes = 0;
            }
        } else {
            const auto count = std::min(quantum, ItemCount(span.domain) - next.localFirst);
            const uint32_t base = span.domain == JPH::GPU::Domain::Bodies   ? tickParameters.selection.y
                                  : span.domain == JPH::GPU::Domain::Shapes ? tickParameters.selection.z
                                  : span.domain == JPH::GPU::Domain::Joints ? tickParameters.selection.w
                                                                            : 0u;
            parameters.range = {base + next.localFirst, count, 0, 0};
            next.localFirst += count;
        }
        Normalize(next);
        return {span.stage, parameters};
    }

    void CaptureStageSpans()
    {
        stageCursor = {};
        stageCursor.phase = HasRigidWork()         ? Phase::InitializeSleep
                            : control.participants ? Phase::ParticipantPredict
                                                   : Phase::Done;
        Normalize(stageCursor);
    }

    void AcceptContactWorkset(std::shared_ptr<const JoltGpuContactWorkset> contacts)
    {
        const bool collisionDispatched = contacts->collisionDispatched;
        inputs.SetContacts(std::move(contacts));
        stageCursor.phase = Phase::CollidePairs;
        if (collisionDispatched)
            NextSpan(stageCursor);
        stageCursor.localFirst = 0;
        stageCursor.islandBatchFirst = 0;
        stageCursor.islandBatchLanes = 0;
        Normalize(stageCursor);
    }

    void AcceptConstraintWorkset(std::shared_ptr<const JoltGpuConstraintWorkset> plan)
    {
        inputs.SetConstraints(std::move(plan));
        // The producer calls back after GPU construction, required CC checks
        // and summary installation. Setup/forces are not rerun when this
        // source resumes a graph continuation.
        NextSpan(stageCursor);
        Normalize(stageCursor);
    }

    void CaptureContactHistory()
    {
        const auto &plan = inputs.GetConstraints();
        if (!tickParameters.counts.w || !plan || !plan->activePatches) {
            storage.ClearContactHistory(selectedRow);
            return;
        }

        auto previous = storage.GetContactHistory(selectedRow);
        JoltGpuContactHistory next{previous.buffer, inputs.GetContacts()->patchCount};
        if (next.buffer &&
            (next.buffer->GetByteSize() < next.ByteSize() || next.buffer->GetByteSize() / 4u > next.ByteSize()))
            next.buffer.reset();
        inputs.SetContactHistory(std::move(previous), std::move(next));
    }

    JoltGpuControlWork PrepareContactStore()
    {
        auto next = inputs.NextContactHistory();
        // All previous-history readers have already been submitted. The actual
        // queue receipts keep their allocation alive until GPU access ends.
        storage.ClearContactHistory(selectedRow);
        inputs.SetContactHistory({}, next);
        next.buffer = rhi::ComputeBuffer::TryCreatePrivate(
            host, {next.ByteSize() / sizeof(uint32_t), rhi::ComputeScalarType::UInt32, 1u});
        if (!next.buffer)
            return {nullptr, [this] { return PrepareContactStore(); }};
        inputs.SetContactHistory({}, std::move(next));
        NextSpan(stageCursor);
        Normalize(stageCursor);
        return {};
    }

    JoltGpuControlWork PrepareSweep()
    {
        inputs.SetSweepContacts(nullptr);
        inputs.SetSweepChoices(nullptr);
        inputs.SetSweepImpactTime({});
        if (!inputs.GetSweepBodies()) {
            auto bodies = rhi::ComputeBuffer::TryCreatePrivate(
                host, {uint64_t(tickParameters.selectedCounts.x) * sizeof(JPH::GPU::SweepBody) / sizeof(uint32_t),
                       rhi::ComputeScalarType::UInt32, 1u});
            if (!bodies)
                return {nullptr, [this] { return PrepareSweep(); }};
            inputs.SetSweepBodies(std::move(bodies));
        }
        stageCursor.sweepRemaining = tickParameters.gravityDt.w;
        stageCursor.sameTimeImpacts = 0;
        stageCursor.sweepAdvanced = false;
        NextSpan(stageCursor);
        Normalize(stageCursor);
        return {};
    }

    void PublishContactHistory()
    {
        const auto &next = inputs.NextContactHistory();
        if (next.buffer)
            storage.PublishContactHistory(selectedRow, next);
    }

    void CaptureControlInputs()
    {
        if (selectedRow >= storage.GetModel().worlds.size())
            throw std::invalid_argument("GPU tick requires its accepted packed world row");
        const auto &selected = storage.GetModel().worlds[selectedRow];
        tickParameters.selection = {selectedRow, selected.bodies.x, selected.leaves.x, selected.leaves.z};
        tickParameters.selectedCounts = {selected.bodies.y, selected.leaves.y, selected.leaves.w, 0u};
        if (!controlOnly) {
            // Model controls can repack leaves/patches. All sizes, parameter
            // reuse and stage domains come from that accepted model boundary.
            auto actual = parameters->Get();
            actual.counts.x = storage.GetBodyCount();
            actual.counts.y = storage.GetWorldCount();
            actual.counts.z = storage.GetShapeCount();
            actual.solve.z = actual.solve.w = 0;
            actual.selection = tickParameters.selection;
            actual.selectedCounts = tickParameters.selectedCounts;
            tickParameters = actual;
            parameters = JoltGpuStepParameters(actual);
            reuseParameters = acceptedParameters && std::memcmp(&*acceptedParameters, &actual, sizeof(actual)) == 0;
            inputs = reuseParameters ? parameterInput : JoltGpuStepInputs{};
            inputAccepted = (reuseParameters && !cpuForces) || !storage.GetBodyCount();
            inputs.SetSelection(actual);
            inputs.SetArticulations(control.articulations ? control.articulations->value : nullptr);
            inputs.SetColliderPairs(control.colliderPairs ? control.colliderPairs->value : nullptr);
            CaptureStageSpans();
            // This Fixed advances outside the native GPU solver. Its old
            // contact warm history must not survive across that elapsed time.
            if (!HasRigidWork())
                storage.ClearContactHistory(selectedRow);
            for (const auto resource :
                 {JPH::GPU::Resource::BodyModels, JPH::GPU::Resource::ShapeModels, JPH::GPU::Resource::Worlds})
                inputs.Capture(storage, resource);
            if (control.acceptedGeometry)
                inputs.CaptureGeometry(control.acceptedGeometry, control.acceptedGenericGeometry);
            else
                inputs.Capture(storage, JPH::GPU::Resource::ConvexGeometry);
            if (!cpuForces)
                inputs.Capture(storage, JPH::GPU::Resource::BodyForces);
        }
        controlAccepted = true;
        FinishCompleted();
    }

    void Complete()
    {
        constraintTopology.reset();
        prescribedBounds.reset();
        completedOutput = std::move(finalOutput);
        status = Status::Complete;
        // Factories have been consumed and every actual receipt retired.
        PublishContactHistory();
        control = {};
        inputs = {};
        parameters.reset();
        cpuForces.reset();
    }

    bool NumericalInputsAccepted() const noexcept
    {
        return controlAccepted && (controlOnly || stageCursor.phase == Phase::Done);
    }

    void FinishCompleted()
    {
        if (!pressureActive && status == Status::Running && NumericalInputsAccepted() && finalAccepted &&
            !workContinuation && receipts.empty() && sharedStages.empty())
            Complete();
    }

    void ValidateControlPacket(const std::shared_ptr<rhi::PreparedComputeBatch> &packet) const
    {
        if (packet && (!host.SharesServicesWith(packet->GetHost()) || packet->GetSubmission().IsValid()))
            throw std::invalid_argument("GPU tick control requires an unaccepted packet from its compute owner");
    }

    void SelectControlWork(JoltGpuControlWork selected, PacketKind kind)
    {
        ValidateControlPacket(selected.packet);
        batch = std::move(selected.packet);
        workGeometry = std::move(selected.geometry);
        workMinimum = std::move(selected.prepareMinimum);
        minimumReducesDispatchWork = selected.minimumReducesDispatchWork;
        workContinuation = std::move(selected.continueWork);
        continuationKind = kind;
        continuationResumeAfter = selected.resumeAfter;
        if (batch)
            batchKind = kind;
        else if (continuationResumeAfter == JoltGpuControlWork::ResumeAfter::HostWork)
            submissionBusy = false;
        else if (workContinuation) {
            submissionBusy = true;
            busyOpportunity = work.Opportunity();
            busyProgress = work.ProgressSerial();
            busySubmissions = host.Queue().GetPendingSubmissionCount();
        }
    }

    bool ContinuationNeedsWait() const
    {
        return continuationResumeAfter == JoltGpuControlWork::ResumeAfter::Completion &&
               (!receipts.empty() || !sharedStages.empty());
    }

    static bool CoupledFeedbackPhase(Phase phase)
    {
        return phase == Phase::WarmCoupledContacts || phase == Phase::ExchangeCoupledContacts ||
               phase == Phase::IntegrateCoupledBodies || phase == Phase::PrepareCoupledSweep ||
               phase == Phase::ResolveCoupledSweep || phase == Phase::SettleCoupledSweep ||
               phase == Phase::ProjectCoupledContacts;
    }

    void ModelAccepted()
    {
        auto &group = control.groups[groupCursor];
        if (group.modelAccepted) {
            auto callback = std::move(group.modelAccepted);
            callback();
        }
        group.prepareModel = {};
    }

    void FinishControlGroup()
    {
        if (groupCursor < control.groups.size()) {
            const auto &group = control.groups[groupCursor];
            if (!modelPrepared || modelUpdate || !group.commands.empty() || group.prepareWork || workContinuation ||
                (commandCursor && !commandCursor->IsComplete()))
                return;
            commandCursor.reset();
            modelPrepared = false;
            ++groupCursor;
        }
        if (!control.initialization && groupCursor == control.groups.size()) {
            control.groups.clear();
            control.commandProgram = nullptr;
            CaptureControlInputs();
        }
    }

    rhi::PreparedComputeBatch *Submission() const
    {
        return preservation ? preservation.get() : batch.get();
    }

    uint64_t PreservationWindow() const
    {
        if (!preservation)
            return std::max(uint64_t{4}, work.ResourceRemaining(FrameWorkResource::Transfer));
        // Wait for a fresh opportunity if only this frame is full. Resize an
        // unsubmitted copy when even an entire GPU or transfer window cannot fit it.
        auto bytes = work.ResourceState(FrameWorkResource::Transfer).limit;
        if (Prediction(SubmissionKey(*preservation)) > work.ResourceState(FrameWorkResource::Gpu).limit)
            bytes = std::min(bytes, preservation->GetTransferBytes() / 2);
        return std::max(uint64_t{4}, bytes);
    }

    bool NeedsSmallerPrefix() const
    {
        const bool adjustable =
            (batchKind == PacketKind::NumericalWork || batchKind == PacketKind::Pressure) && bool(workMinimum);
        const bool prefix =
            (lane->prepared && lane->prepared->owner == this) ||
            ((batchKind == PacketKind::Commands || batchKind == PacketKind::Stages) && batchEnd - batchBegin > 1);
        if (!batch || preservation || (!adjustable && !prefix) || batch->RequiresPreparation() ||
            batch->HasPreservationWork())
            return false;
        const auto remaining = work.ResourceRemaining(FrameWorkResource::Gpu);
        const auto window = adjustable && remaining ? remaining : work.ResourceState(FrameWorkResource::Gpu).limit;
        return ((!adjustable || minimumReducesDispatchWork) && Prediction(SubmissionKey(*batch)) > window) ||
               batch->GetTransferBytes() > work.ResourceState(FrameWorkResource::Transfer).limit;
    }

    bool CanSubmit() const
    {
        const auto *packet = Submission();
        if (!packet)
            return false;
        const bool indivisible = (lane->prepared && lane->prepared->owner == this && !preservation) ? false
                                 : preservation ? packet->GetTransferBytes() <= 4
                                 : batchKind == PacketKind::Commands || batchKind == PacketKind::Stages
                                     ? batchEnd - batchBegin == 1
                                     : true;
        return work.CanAdmitResource(FrameWorkResource::Gpu, Prediction(SubmissionKey(*packet)), indivisible) &&
               (!packet->GetTransferBytes() ||
                work.CanAdmitResource(FrameWorkResource::Transfer, packet->GetTransferBytes(), indivisible));
    }

    FrameWorkFacts Describe() const
    {
        FrameWorkFacts facts;
        facts.observe = pressureObserved;
        if (lane->prepared && lane->prepared->peer == this) {
            facts.pending = true;
            return facts;
        }
        if (PressureRequested()) {
            facts.pending = true;
            if (receipts.empty())
                facts.host.ready = true;
            else
                facts.gpuBusy = true;
            return facts;
        }
        if (!admissionEnabled && !pressureActive) {
            facts.pending = facts.gpuBusy = !receipts.empty();
            return facts;
        }
        facts.pending = status == Status::Running || pressureActive || !receipts.empty() || !sharedStages.empty();
        if (status != Status::Running && !pressureActive)
            return facts;
        if (pressureActive && !batch && !workContinuation) {
            if (receipts.empty())
                facts.host.ready = true;
            else
                facts.gpuBusy = true;
            return facts;
        }
        // Only explicitly acceptance-ready producer preparation may overlap its
        // accepted GPU predecessor. Completion consumers retain the same gate.
        if (!batch && workContinuation && ContinuationNeedsWait()) {
            facts.gpuBusy = true;
            return facts;
        }
        if (!batch && submissionBusy)
            return facts;
        if (!pressureActive && NumericalInputsAccepted() && finalAccepted) {
            facts.gpuBusy = !receipts.empty() || !sharedStages.empty();
            return facts;
        }
        // Advance a closed partition's retirement and already completed coupling
        // feedback while the host still has room before foreground scene work.
        const bool early = !work.IsLate() && !pressureActive;
        const bool cleanup = early && control.partitionCleanup;
        const bool coupled = early && !controlOnly;
        const bool nextFeedbackPacket = coupled && earlyFeedbackOpportunity == work.Opportunity() &&
                                        stageCursor.phase != Phase::Done &&
                                        (!ProducerPhase(stageCursor.phase) || CoupledFeedbackPhase(stageCursor.phase));
        const bool preparePacket = preservation ? preservation->GetTransferBytes() > PreservationWindow()
                                                : batch && (batch->RequiresPreparation() ||
                                                            batch->HasPreservationWork() || NeedsSmallerPrefix());
        if (preparePacket || (!batch && (workContinuation || !controlAccepted || !inputAccepted ||
                                         stageCursor.phase != Phase::Done || !finalPrepared))) {
            facts.host.ready = true;
            facts.host.reserved = cleanup || nextFeedbackPacket ||
                                  (coupled && workContinuation && CoupledFeedbackPhase(stageCursor.phase) &&
                                   continuationResumeAfter == JoltGpuControlWork::ResumeAfter::Completion);
        } else if (!submissionBusy && CanSubmit()) {
            facts.gpu = {0, 0, true, cleanup || (nextFeedbackPacket && !preservation)};
        }
        // Accepted predecessors do not block this source's ready successor.
        // Its actual buffer dependencies and queue-slot admission order it.
        facts.gpuBusy = !receipts.empty() && !facts.gpu.ready && !facts.host.ready;
        return facts;
    }

    bool PreparePrefix(bool commands = false, bool minimum = false)
    {
        const size_t begin = commands ? commandCursor->GetAcceptedDispatchCount() : cursor;
        const auto available = work.ResourceRemaining(FrameWorkResource::Gpu);
        const auto window = available ? available : work.ResourceState(FrameWorkResource::Gpu).limit;
        const auto transferAvailable = work.ResourceRemaining(FrameWorkResource::Transfer);
        const auto transferWindow =
            transferAvailable ? transferAvailable : work.ResourceState(FrameWorkResource::Transfer).limit;
        size_t count = 1;
        StageCursor next = stageCursor;
        std::vector<JoltGpuStageWork> ready;
        if (commands) {
            if (!minimum && lastCommandCount && lastCommandEstimate) {
                const long double width = static_cast<long double>(window) * lastCommandCount / lastCommandEstimate;
                count = width >= MaxPacketUnits ? MaxPacketUnits : std::max(size_t{1}, static_cast<size_t>(width));
            }
            constexpr uint64_t minimumBytes =
                sizeof(JPH::GPU::BodyCommand) + sizeof(JPH::GPU::BodyCommandSpan) + sizeof(JPH::GPU::BodyCommandParams);
            count = std::min(count, std::max(size_t{1}, static_cast<size_t>(transferWindow / minimumBytes)));
            count = std::min(count, commandCursor->GetDispatchCount() - begin);
        } else {
            // The current ready numerical phases may share a bounded packet,
            // including across iterations and substeps. TakeUnit preserves all
            // setup/warm and joint/contact ordering; control/final work stays outside.
            ready.reserve(MaxPacketUnits);
            long double planned = 0;
            bool cold = false;
            while (ready.size() < MaxPacketUnits && next.phase != Phase::Done && !ProducerPhase(next.phase)) {
                const auto found = stageEstimates.find(Span(next.phase).stage);
                // A mixed packet's measured average is only a planning proxy,
                // never a measurement attributed to an individual stage.
                const auto &history = found == stageEstimates.end() ? mixedEstimates : found->second;
                const auto cost = PlanningCost(history);
                const bool unknown = cost == 0;
                if (ready.size() >= (cold || unknown ? ColdPacketUnits : MaxPacketUnits))
                    break;
                // Always retain the minimum unit, even if its estimate exceeds
                // this window. The existing indivisible admission rule decides it.
                if (!ready.empty() && planned + cost > window)
                    break;
                ready.push_back(TakeUnit(next));
                planned += cost;
                cold |= unknown;
                if (minimum)
                    break;
            }
            count = ready.size();
        }
        auto candidate = commands ? commandCursor->PrepareNext(host, static_cast<uint32_t>(count))
                                  : programs.PrepareStages(storage, inputs, ready);
        if (!candidate)
            throw std::logic_error("A ready GPU work unit did not produce its dispatch packet");
        // Admission still uses the exact packet key. An unseen composition
        // remains Prediction == 0 and uses the owner's real calibration gate.
        auto numericalKey = commands ? NumericalCostKey{} : NumericalKey(ready);
        const auto prediction = Prediction(Key(*candidate, numericalKey));
        // A known oversized packet or excessive transfer gets one minimum
        // replacement, not another search over the remaining logical tick.
        if (count > 1 && ((prediction && prediction > window) || candidate->GetTransferBytes() > transferWindow)) {
            candidate->Cancel();
            count = 1;
            if (commands) {
                commandCursor->DiscardPreparedPrefix();
                candidate = commandCursor->PrepareNext(host, 1);
            } else {
                next = stageCursor;
                ready = {TakeUnit(next)};
                candidate = programs.PrepareStages(storage, inputs, ready);
                numericalKey = NumericalKey(ready);
            }
        }
        batch = std::move(candidate);
        batchBegin = begin;
        batchEnd = begin + count;
        batchStageEnd = next;
        batchNumericalKey = std::move(numericalKey);
        batchHomogeneousStage.reset();
        if (!commands && std::all_of(ready.begin(), ready.end(),
                                     [&](const auto &unit) { return unit.stage == ready.front().stage; }))
            batchHomogeneousStage = ready.front().stage;
        batchKind = commands ? PacketKind::Commands : PacketKind::Stages;
        return true;
    }

    JoltGpuControlWork PrepareSnapshot()
    {
        if (!control.needsSnapshot)
            return {}; // Pure observations and closed partitions have no publication consumer.
        using namespace JPH::GPU;
        const auto &liveLayout = *storage.GetJointLayout();
        const auto slice = liveLayout.WorldSlice(finalOutput.jointBase, finalOutput.jointCount);
        if (!finalOutput.jointLayout) {
            finalOutput.jointLayout = liveLayout.LocalLayout(slice);
            snapshotMetadata = finalOutput.jointLayout->MetadataBytes();
        }
        const uint64_t stateBytes = uint64_t(finalOutput.bodyCount) * sizeof(BodyState);
        const uint64_t authorBytes = uint64_t(slice.authorWords) * 16u;
        const uint64_t jointStateBytes = uint64_t(slice.stateWords) * 16u;
        const auto allocate = [&](std::shared_ptr<rhi::ComputeBuffer> &buffer, uint64_t bytes, bool body) {
            if (!bytes)
                return true;
            if (!host.device.GetCapabilities().IsStorageBufferRangeValid(0, bytes))
                throw std::length_error("GPU selected-world snapshot exceeds device storage range");
            if (!buffer)
                buffer = rhi::ComputeBuffer::TryCreatePrivate(
                    host,
                    {bytes / (body ? 16u : 4u), body ? rhi::ComputeScalarType::Float32 : rhi::ComputeScalarType::UInt32,
                     static_cast<uint8_t>(body ? 4u : 1u)});
            return bool(buffer);
        };
        if (!allocate(finalOutput.states, stateBytes, true) ||
            !allocate(finalOutput.joints, finalOutput.jointLayout->ByteSize(), false))
            return {nullptr, [this] { return PrepareSnapshot(); }};
        const auto available = work.ResourceRemaining(FrameWorkResource::Transfer);
        const auto window = available ? available : work.ResourceState(FrameWorkResource::Transfer).limit;
        uint64_t remaining = std::max(uint64_t{16}, window - window % 16u);
        std::vector<rhi::ComputeBufferUpdate> uploads;
        std::vector<rhi::ComputeBufferCopy> copies;
        if (snapshotMetadataOffset < snapshotMetadata.size()) {
            const auto bytes = std::min(remaining, uint64_t(snapshotMetadata.size()) - snapshotMetadataOffset);
            const auto first = snapshotMetadata.begin() + static_cast<size_t>(snapshotMetadataOffset);
            uploads.push_back({finalOutput.joints, snapshotMetadataOffset,
                               std::vector<uint8_t>(first, first + static_cast<size_t>(bytes))});
            snapshotMetadataOffset += bytes;
            remaining -= bytes;
        }
        while (snapshotRegion < 3u && remaining) {
            const uint64_t length = snapshotRegion == 0u   ? stateBytes
                                    : snapshotRegion == 1u ? authorBytes
                                                           : jointStateBytes;
            if (snapshotOffset == length) {
                snapshotOffset = 0;
                ++snapshotRegion;
                continue;
            }
            const bool bodies = snapshotRegion == 0u;
            const auto &source = storage.Get(bodies ? Resource::BodyStates : Resource::Joints);
            const auto &target = bodies ? finalOutput.states : finalOutput.joints;
            const uint64_t sourceBase =
                bodies ? uint64_t(finalOutput.bodyBase) * sizeof(BodyState)
                       : uint64_t(snapshotRegion == 1u ? slice.authorFirst : slice.stateFirst) * 16u;
            const uint64_t targetBase = bodies ? 0u
                                        : snapshotRegion == 1u
                                            ? 16u + uint64_t(finalOutput.jointCount) * sizeof(JointDirectory)
                                            : finalOutput.jointLayout->AuthorEndBytes();
            const auto bytes = std::min(remaining, length - snapshotOffset);
            // Borrow the mutable live writer generation; only the compact
            // destination gets a retained publication reader after receipts.
            copies.push_back({source, target, {sourceBase + snapshotOffset, targetBase + snapshotOffset, bytes}});
            snapshotOffset += bytes;
            remaining -= bytes;
        }
        if (uploads.empty() && copies.empty()) {
            snapshotMetadata.clear();
            if (finalOutput.states)
                finalOutput.states = finalOutput.states->Freeze();
            finalOutput.joints = finalOutput.joints->Freeze();
            snapshotComplete = true;
            auto prepare = std::move(control.prepareFinalWork);
            control.prepareFinalWork = {};
            return prepare ? prepare(finalOutput) : JoltGpuControlWork{};
        }
        auto packet = rhi::PrepareComputeBatch(host, std::move(uploads), {}, {}, std::move(copies));
        return {packet, [this] { return PrepareSnapshot(); }};
    }

    bool Prepare()
    {
        // This is the actual accepted cursor, never speculative NextSpan state.
        // Empty joint domains can reach it through producer Normalize, so clear
        // the latch here as well as after accepted native stage packets.
        if (GuardConsumersFinished())
            reducedGuardWindow = false;
        if (PressureRequested())
            return StartPressure();
        if (pressureActive && !batch && !workContinuation) {
            if (!receipts.empty())
                return false;
            FinishPressure();
            return true;
        }
        if (NeedsSmallerPrefix()) {
            if (lane->prepared && lane->prepared->owner == this) {
                // This exact frontier already declined a shared packet. Keep
                // its singleton fallback until acceptance advances the cursor.
                sharedFallbackCursor = cursor;
                lane->prepared->peer->sharedFallbackCursor = lane->prepared->peer->cursor;
                lane->DropShared(this);
                return PreparePrefix(false, true);
            }
            if (workMinimum) {
                auto minimum = std::move(workMinimum);
                const auto kind = batchKind;
                batch->Cancel();
                batch.reset();
                workContinuation = {};
                batchNumericalKey.clear();
                submissionBusy = false;
                SelectControlWork(minimum(), kind);
                return true;
            }
            const bool commands = batchKind == PacketKind::Commands;
            // Only the unaccepted prefix changes. The accepted stage/command
            // cursor and all outstanding receipts retain their original work.
            batch->Cancel();
            batch.reset();
            submissionBusy = false;
            if (commands)
                commandCursor->DiscardPreparedPrefix();
            return PreparePrefix(commands, true);
        }
        if (!batch) {
            if (workContinuation) {
                if (ContinuationNeedsWait())
                    return false;
                const auto kind = continuationKind;
                auto continuation = std::move(workContinuation);
                workContinuation = {};
                SelectControlWork(continuation(), kind);
                if (!batch && !workContinuation) {
                    if (kind == PacketKind::Pressure) {
                        if (receipts.empty())
                            FinishPressure();
                    } else if (kind == PacketKind::FinalWork)
                        finalAccepted = true;
                    else if (kind != PacketKind::NumericalWork)
                        FinishControlGroup();
                    FinishCompleted();
                }
                return true;
            }
            if (pressureActive)
                return false;
            if (NumericalInputsAccepted()) {
                if (!finalPrepared) {
                    finalPrepared = true;
                    finalOutput.worldRow = tickParameters.selection.x;
                    finalOutput.bodyBase = tickParameters.selection.y;
                    finalOutput.bodyCount = tickParameters.selectedCounts.x;
                    finalOutput.leafBase = tickParameters.selection.z;
                    finalOutput.leafCount = tickParameters.selectedCounts.y;
                    finalOutput.jointBase = tickParameters.selection.w;
                    finalOutput.jointCount = tickParameters.selectedCounts.z;
                    finalOutput.geometryStore = storage.GetGeometryStore();
                    finalOutput.geometry = controlOnly ? storage.GetGeometry() : inputs.GetGeometry();
                    finalOutput.hasGenericGeometry =
                        controlOnly ? storage.HasGenericGeometry() : inputs.HasGenericGeometry();
                    SelectControlWork(PrepareSnapshot(), PacketKind::FinalWork);
                    if (!batch && !workContinuation) {
                        finalAccepted = true;
                        FinishCompleted();
                    }
                    return true;
                }
                return false;
            }
            if (!controlAccepted) {
                if (control.initialization) {
                    batch = control.initialization;
                    batchKind = PacketKind::Initialization;
                    return true;
                }
                auto &group = control.groups[groupCursor];
                if (!modelPrepared) {
                    modelUpdate = group.prepareModel ? group.prepareModel() : nullptr;
                    if (!modelUpdate && group.modelDeferred && group.modelDeferred()) {
                        submissionBusy = true;
                        busyOpportunity = work.Opportunity();
                        busyProgress = work.ProgressSerial();
                        busySubmissions = host.Queue().GetPendingSubmissionCount();
                        return false;
                    }
                    modelPrepared = true;
                    if (modelUpdate)
                        ValidateControlPacket(modelUpdate->GetBatch());
                    else
                        ModelAccepted();
                }
                if (modelUpdate) {
                    batch = modelUpdate->GetBatch();
                    batchKind = PacketKind::Model;
                    return true;
                }
                if (!commandCursor && !group.commands.empty()) {
                    commandCursor = group.prepareCommands
                                        ? group.prepareCommands(std::move(group.commands))
                                        : control.commandProgram->Prepare(storage, std::move(group.commands));
                    group.prepareCommands = {};
                    group.commands.clear();
                }
                if (commandCursor && !commandCursor->IsComplete())
                    return PreparePrefix(true);
                if (group.prepareWork) {
                    auto prepareWork = std::move(group.prepareWork);
                    group.prepareWork = {};
                    SelectControlWork(prepareWork(), PacketKind::Work);
                    if (batch || workContinuation)
                        return true;
                }
                FinishControlGroup();
                return true;
            }
            if (!inputAccepted) {
                if (isolateParameters && !memberParameters) {
                    memberParameters = rhi::ComputeBuffer::TryCreatePrivate(
                        host, {sizeof(JPH::GPU::StepParams) / 4u, rhi::ComputeScalarType::UInt32, 1});
                    if (!memberParameters)
                        return false;
                }
                batch = reuseParameters ? storage.PrepareForces(*cpuForces, tickParameters)
                                        : storage.PrepareInputs(*parameters, cpuForces ? &*cpuForces : nullptr,
                                                                isolateParameters ? memberParameters : nullptr);
                batchKind = PacketKind::Inputs;
                return true;
            }
            if (const auto stage = ParticipantStage(stageCursor.phase)) {
                if (*stage == JoltGpuParticipantStage::Exchange && !HasRigidWork() &&
                    control.participants.prepareClosedRounds && tickParameters.counts.w - stageCursor.iteration > 1) {
                    // Same finite round cap as the coupled producer. Owners
                    // decline if a round needs an intermediate native consumer.
                    const auto rounds = std::min(tickParameters.counts.w - stageCursor.iteration, 4u);
                    auto closed =
                        control.participants.prepareClosedRounds(tickParameters, fixedDeltaTime, stageCursor.substep,
                                                                 stageCursor.iteration, rounds, [this](uint32_t count) {
                                                                     stageCursor.iteration += count - 1;
                                                                     NextSpan(stageCursor);
                                                                     Normalize(stageCursor);
                                                                 });
                    if (closed) {
                        SelectControlWork(std::move(*closed), PacketKind::NumericalWork);
                        return true;
                    }
                }
                SelectControlWork(control.participants.prepare(*stage, tickParameters, fixedDeltaTime,
                                                               stageCursor.substep, stageCursor.iteration,
                                                               [this] {
                                                                   NextSpan(stageCursor);
                                                                   Normalize(stageCursor);
                                                               }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::Broadphase) {
                PublishContactHistory();
                inputs.ClearContacts();
                SelectControlWork(
                    broadphase.Prepare(storage, inputs, programs, tickParameters,
                                       [this](auto contacts) { AcceptContactWorkset(std::move(contacts)); }),
                    PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::PrepareSweep) {
                SelectControlWork(PrepareSweep(), PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::SweptBroadphase) {
                SelectControlWork(broadphase.Prepare(
                                      storage, inputs, programs, tickParameters,
                                      [this](auto contacts) {
                                          inputs.SetSweepContacts(std::move(contacts));
                                          NextSpan(stageCursor);
                                          Normalize(stageCursor);
                                      },
                                      true),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::ResolveSweep) {
                SelectControlWork(narrowphase.PrepareSweeps(storage, inputs, work, stageCursor.sweepRemaining,
                                                            stageCursor.sweepAdvanced,
                                                            [this](std::optional<float> fraction) {
                                                                stageCursor.localSweepImpact = fraction;
                                                                inputs.SetSweepImpactTime(fraction);
                                                                NextSpan(stageCursor);
                                                                Normalize(stageCursor);
                                                            }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::SolveSweepEvent) {
                const float fraction = *inputs.SweepImpactTime();
                const float remaining = stageCursor.sweepRemaining * (1.0f - fraction);
                if (remaining < stageCursor.sweepRemaining)
                    stageCursor.sameTimeImpacts = 0;
                else if (++stageCursor.sameTimeImpacts > 64u + 4u * uint64_t(tickParameters.selectedCounts.x))
                    throw std::runtime_error("Continuous impacts did not settle at the current time");
                SelectControlWork(narrowphase.PrepareSweepEvent(storage, inputs, work, fraction, remaining,
                                                                stageCursor.localSweepImpact == fraction,
                                                                bool(control.prepareContactReport),
                                                                [this, remaining] {
                                                                    stageCursor.sweepRemaining = remaining;
                                                                    stageCursor.sweepAdvanced = true;
                                                                    NextSpan(stageCursor);
                                                                    Normalize(stageCursor);
                                                                }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::PrepareContactStore) {
                SelectControlWork(PrepareContactStore(), PacketKind::NumericalWork);
                return true;
            }
            if (forceWakePending) {
                forceWakePending = false;
                SelectControlWork(programs.GetSleep().PrepareForceWake(storage, tickParameters, [] {}),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::CapturePrescribedBounds) {
                std::vector<uint32_t> sources;
                if (control.prepareSleepSources)
                    sources = control.prepareSleepSources();
                else {
                    // Native standalone owners without a Scene intent directory
                    // conservatively nominate author static/kinematic rows.
                    // Actual GPU motion remains the sole wake predicate.
                    const auto &models = storage.GetModel().bodies;
                    for (uint32_t local = 0; local < tickParameters.selectedCounts.x; ++local) {
                        const auto body = tickParameters.selection.y + local;
                        const auto motion = models[body].meta.z & ~JPH::GPU::MotionNotResident;
                        if (motion == JPH::GPU::MotionStatic || motion == JPH::GPU::MotionKinematic)
                            sources.push_back(body);
                    }
                }
                SelectControlWork(programs.GetSleep().PreparePrescribedCapture(storage, tickParameters,
                                                                               std::move(sources),
                                                                               [this](auto bounds) {
                                                                                   prescribedBounds = std::move(bounds);
                                                                                   NextSpan(stageCursor);
                                                                                   Normalize(stageCursor);
                                                                               }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::CommitArticulationPose) {
                SelectControlWork(programs.GetArticulations().PreparePose(storage, inputs,
                                                                          [this] {
                                                                              NextSpan(stageCursor);
                                                                              Normalize(stageCursor);
                                                                          }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::CaptureArticulationRotors ||
                stageCursor.phase == Phase::ClassifyArticulations) {
                const auto done = [this] {
                    NextSpan(stageCursor);
                    Normalize(stageCursor);
                };
                if (stageCursor.phase == Phase::ClassifyArticulations) {
                    CaptureContactHistory();
                    reducedGuardWindow = true;
                }
                SelectControlWork(stageCursor.phase == Phase::CaptureArticulationRotors
                                      ? programs.GetArticulations().PrepareRotors(storage, inputs, done)
                                      : programs.GetArticulations().PrepareClassification(
                                            storage, inputs, control.participants.gpuBodySlots.get(), done),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::InitializeSleep) {
                SelectControlWork(programs.GetSleep().PrepareInitialize(storage, tickParameters,
                                                                        [this] {
                                                                            NextSpan(stageCursor);
                                                                            Normalize(stageCursor);
                                                                        }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::WakePrescribedBounds ||
                stageCursor.phase == Phase::WakeFinalPrescribedBounds) {
                // The final-pose pass runs AFTER this substep's sleep vote and
                // commit. A previously asleep destination did not accumulate
                // quiet time here; waking it now preserves one real following
                // forces/contact opportunity even when TimeBeforeSleep is 0.
                SelectControlWork(programs.GetSleep().PreparePrescribedWake(storage, tickParameters, prescribedBounds,
                                                                            [this] {
                                                                                NextSpan(stageCursor);
                                                                                Normalize(stageCursor);
                                                                            }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::BuildTopology) {
                inputs.ClearConstraints();
                constraintTopology.reset();
                SelectControlWork(constraints.PrepareTopology(storage, inputs, tickParameters,
                                                              [this](auto topology) {
                                                                  constraintTopology = std::move(topology);
                                                                  NextSpan(stageCursor);
                                                                  Normalize(stageCursor);
                                                              }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::WakeComponents || stageCursor.phase == Phase::WakeExternalComponents) {
                if (!constraintTopology)
                    throw std::logic_error("GPU sleep wake lost its accepted topology");
                SelectControlWork(programs.GetSleep().PrepareWake(storage, inputs, *constraintTopology,
                                                                  [this] {
                                                                      NextSpan(stageCursor);
                                                                      Normalize(stageCursor);
                                                                  }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::UpdateSleep) {
                auto settings = control.sleepParameters;
                settings.timing.x = tickParameters.gravityDt.w;
                SelectControlWork(programs.GetSleep().PrepareHistory(storage, inputs, settings,
                                                                     inputs.TakeSleepCoupling(),
                                                                     [this] {
                                                                         NextSpan(stageCursor);
                                                                         Normalize(stageCursor);
                                                                     }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::CompleteConvexContacts) {
                SelectControlWork(narrowphase.Prepare(storage, inputs, work,
                                                      [this] {
                                                          NextSpan(stageCursor);
                                                          Normalize(stageCursor);
                                                      }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::ReportContactsBeforeVelocity ||
                stageCursor.phase == Phase::ReportContactsAfterVelocity || stageCursor.phase == Phase::ReportSweep) {
                const auto phase = stageCursor.phase == Phase::ReportContactsBeforeVelocity
                                       ? JoltGpuContactReportStage::BeforeVelocity
                                   : stageCursor.phase == Phase::ReportSweep ? JoltGpuContactReportStage::Sweep
                                                                             : JoltGpuContactReportStage::AfterVelocity;
                SelectControlWork(control.prepareContactReport(storage, inputs, tickParameters, stageCursor.substep,
                                                               phase,
                                                               [this] {
                                                                   NextSpan(stageCursor);
                                                                   Normalize(stageCursor);
                                                               }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::PrepareCoupledContacts ||
                stageCursor.phase == Phase::IntegrateCoupledBodies || CoupledFeedbackPhase(stageCursor.phase)) {
                using CoupledStage = JoltGpuCoupledContactStage;
                const bool preparing = stageCursor.phase == Phase::PrepareCoupledContacts;
                const bool warming = stageCursor.phase == Phase::WarmCoupledContacts;
                const bool projecting = stageCursor.phase == Phase::ProjectCoupledContacts;
                const bool integrating = stageCursor.phase == Phase::IntegrateCoupledBodies;
                const bool preparingSweep = stageCursor.phase == Phase::PrepareCoupledSweep;
                const bool settlingSweep = stageCursor.phase == Phase::SettleCoupledSweep;
                const bool resolvingSweep = stageCursor.phase == Phase::ResolveCoupledSweep || settlingSweep;
                const auto phase = preparing        ? CoupledStage::Prepare
                                   : warming        ? CoupledStage::WarmVelocity
                                   : projecting     ? CoupledStage::Position
                                   : integrating    ? CoupledStage::Integrate
                                   : preparingSweep ? CoupledStage::PrepareSweep
                                   : settlingSweep  ? CoupledStage::SettleSweep
                                   : resolvingSweep ? CoupledStage::Sweep
                                                    : CoupledStage::Velocity;
                const bool iterating = !preparing && !warming && !integrating && !preparingSweep && !resolvingSweep;
                const bool exchangeBetweenRounds =
                    !projecting && control.participants.Has(JoltGpuParticipantStage::Exchange, stageCursor.substep);
                const auto &local = inputs.GetConstraints();
                const bool canBatch = iterating && !exchangeBetweenRounds &&
                                      !(local && (local->contactIslandCount || local->jointIslandCount));
                const uint32_t iterations = projecting ? tickParameters.solve.x : tickParameters.counts.w;
                const uint32_t maxRounds = canBatch ? std::min(iterations - stageCursor.iteration, 4u) : 1u;
                auto phaseParams = tickParameters;
                if (resolvingSweep)
                    phaseParams.gravityDt.w = stageCursor.sweepRemaining;
                SelectControlWork(control.prepareCoupledContacts(storage, inputs, phaseParams, stageCursor.substep,
                                                                 stageCursor.iteration, phase, maxRounds,
                                                                 [this, iterating](uint32_t rounds) {
                                                                     if (iterating)
                                                                         stageCursor.iteration += rounds - 1;
                                                                     NextSpan(stageCursor);
                                                                     Normalize(stageCursor);
                                                                 }),
                                  PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::BuildConstraints) {
                SelectControlWork(
                    constraints.PrepareNumericalPlan(storage, inputs, constraintTopology,
                                                     [this](auto plan) { AcceptConstraintWorkset(std::move(plan)); }),
                    PacketKind::NumericalWork);
                return true;
            }
            return PreparePrefix();
        }
        // Resolving a full-overwrite COW allocation can require no transfer at
        // all. A minimum preparation range does not grant its copy submission.
        const auto prepared = batch->PreparePreservation(PreservationWindow(), true);
        if (!prepared.ready)
            return false;
        preservation = prepared.copy;
        return true;
    }

    void Accepted(const std::shared_ptr<rhi::PreparedComputeBatch> &packet, CostKey key, uint64_t prediction,
                  uint64_t started)
    {
        const auto kind = preservation ? PacketKind::Preservation : batchKind;
        if (kind == PacketKind::Stages && lane->prepared && lane->prepared->owner == this) {
            const auto stage = lane->prepared;
            stage->completion.emplace(packet->GetCompletion());
            stage->opportunity = work.Opportunity();
            stage->gpuGrant = work.RecordAcceptedResource(FrameWorkResource::Gpu, prediction);
            stage->bytes = packet->GetTransferBytes();
            stage->transferGrant =
                stage->bytes ? work.RecordAcceptedResource(FrameWorkResource::Transfer, stage->bytes) : 0;
            stage->started = started;
            stage->peer->cursor = stage->peerCursorEnd;
            stage->peer->stageCursor = stage->peerEnd;
            cursor = batchEnd;
            stageCursor = batchStageEnd;
            batch.reset();
            batchNumericalKey.clear();
            lane->prepared.reset();
            try {
                if (INXLOG_GET_LEVEL() == LOG_DEBUG)
                    INXLOG_DIAGNOSTIC("P4_SHARED_ACCEPT opportunity=", stage->opportunity, " owner_row=", selectedRow,
                                      " peer_row=", stage->peer->selectedRow,
                                      " stage=", static_cast<uint32_t>(stage->stage),
                                      " ticket=", packet->GetSubmission().serial,
                                      " units=2 dispatches=", packet->GetExecutionSignature().front());
            } catch (...) {
                // A diagnostic failure cannot revoke or fail an accepted numerical packet.
            }
            return;
        }
        const auto bytes = packet->GetTransferBytes();
        const bool prefix = kind == PacketKind::Stages || kind == PacketKind::Commands;
        const auto homogeneousStage = kind == PacketKind::Stages ? batchHomogeneousStage : std::nullopt;
        receipts.emplace_back(
            Receipt{packet->GetCompletion(), std::move(key), kind, prefix ? batchEnd - batchBegin : 0, homogeneousStage,
                    work.Opportunity(), work.RecordAcceptedResource(FrameWorkResource::Gpu, prediction),
                    bytes ? work.RecordAcceptedResource(FrameWorkResource::Transfer, bytes) : 0, bytes, started,
                    kind == PacketKind::Stages ? inputs.GetGeometry() : workGeometry});
        if (kind == PacketKind::Preservation) {
            preservation.reset();
        } else {
            batch.reset();
            workMinimum = {};
            batchNumericalKey.clear();
            if (kind == PacketKind::Initialization) {
                control.initialization.reset();
            } else if (kind == PacketKind::Model) {
                modelUpdate.reset();
                ModelAccepted();
            } else if (kind == PacketKind::Inputs) {
                if (!reuseParameters) {
                    const auto &parameterBuffer =
                        isolateParameters ? memberParameters : storage.Get(JPH::GPU::Resource::StepParams);
                    inputs.CaptureParameters(storage, parameterBuffer);
                    parameterInput.CaptureParameters(storage, parameterBuffer);
                    acceptedParameters = tickParameters;
                }
                if (cpuForces)
                    inputs.Capture(storage, JPH::GPU::Resource::BodyForces);
                forceWakePending = cpuForces.has_value();
                cpuForces.reset();
                inputAccepted = true;
            } else if (kind == PacketKind::Stages) {
                cursor = batchEnd;
                stageCursor = batchStageEnd;
                if (GuardConsumersFinished())
                    reducedGuardWindow = false;
            } else if (kind == PacketKind::FinalWork && !workContinuation) {
                finalAccepted = true;
            }
            if (kind != PacketKind::Pressure && !controlAccepted)
                FinishControlGroup();
        }
    }

    bool Submit()
    {
        if (!CanSubmit())
            return false;
        const auto packet = preservation ? preservation : batch;
        auto key = SubmissionKey(*packet);
        const auto prediction = Prediction(key);
        const auto started = NowNanoseconds();
        try {
            const auto ticket = preservation                        ? packet->TrySubmit()
                                : batchKind == PacketKind::Model    ? modelUpdate->TrySubmit()
                                : batchKind == PacketKind::Commands ? commandCursor->TrySubmit()
                                                                    : packet->TrySubmit();
            if (!ticket) {
                // Slot-free admission can refuse a packet whose own temporary
                // allocations leave no room for its staging. Drop the old
                // unaccepted prefix before preparing its minimum on the host.
                const bool adjustable = batchKind == PacketKind::NumericalWork || batchKind == PacketKind::Pressure;
                if (adjustable && !preservation && workMinimum && !host.Queue().GetCapacityPredecessor().IsValid()) {
                    batch->Cancel();
                    batch.reset();
                    workContinuation = std::exchange(workMinimum, {});
                    continuationKind = batchKind;
                    continuationResumeAfter = JoltGpuControlWork::ResumeAfter::Acceptance;
                    batchNumericalKey.clear();
                    submissionBusy = false;
                    return true;
                }
                // Retained retry state, not a second execution queue. Retry
                // only after a receipt, peer progress, pressure change or a
                // new host opportunity can have released the blocked resource.
                submissionBusy = true;
                busyOpportunity = work.Opportunity();
                busyProgress = work.ProgressSerial();
                busySubmissions = host.Queue().GetPendingSubmissionCount();
                return true;
            }
        } catch (...) {
            // Acceptance is final even if RHI host bookkeeping subsequently
            // throws. Retain its receipt and account it before failing the tick.
            if (packet->GetSubmission().IsValid())
                Accepted(packet, std::move(key), prediction, started);
            throw;
        }
        Accepted(packet, std::move(key), prediction, started);
        submissionBusy = false;
        return true;
    }

    bool Advance(bool gpu)
    {
        if ((!admissionEnabled || status != Status::Running) && !pressureActive && !PressureRequested())
            return false;
        try {
            if (gpu)
                return Submit();
            return Prepare();
        } catch (...) {
            Fail();
            throw;
        }
    }

    std::shared_ptr<Lane> lane;
    std::vector<std::shared_ptr<SharedStage>> sharedStages;
    std::optional<JoltGpuStepParameters> parameters;
    JoltGpuTickControl control;
    std::function<JoltGpuControlWork()> workContinuation, workMinimum;
    bool minimumReducesDispatchWork = true;
    std::shared_ptr<const JoltGpuGeometryLease> workGeometry;
    PacketKind continuationKind = PacketKind::Work;
    JoltGpuControlWork::ResumeAfter continuationResumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
    std::unique_ptr<JoltGpuModelUpdate> modelUpdate;
    size_t groupCursor = 0;
    bool modelPrepared = false;
    std::unique_ptr<JoltGpuCommandCursor> commandCursor;
    bool controlAccepted = false;
    bool controlOnly = false;
    bool finalPrepared = true, finalAccepted = true;
    JoltGpuTickOutput finalOutput;
    uint32_t snapshotRegion = 0;
    uint64_t snapshotOffset = 0;
    uint64_t snapshotMetadataOffset = 0;
    std::vector<uint8_t> snapshotMetadata;
    bool snapshotComplete = false;
    std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces;
    JoltGpuStepInputs inputs;
    JoltGpuStepInputs parameterInput;
    std::optional<JPH::GPU::StepParams> acceptedParameters;
    JPH::GPU::StepParams tickParameters{};
    uint32_t selectedRow = 0;
    bool reuseParameters = false;
    bool isolateParameters = false;
    std::shared_ptr<rhi::ComputeBuffer> memberParameters;
    StageCursor stageCursor, batchStageEnd;
    std::map<JPH::GPU::Stage, UnitHistory> stageEstimates;
    UnitHistory mixedEstimates;
    bool inputAccepted = false;
    bool reducedGuardWindow = false;
    bool pressureActive = false, pressureObserved = false, pressureChanged = false;
    std::function<bool()> pressureObserve, pressurePending;
    std::function<JoltGpuControlWork(uint32_t, JoltGpuTick::PressureCompletion)> pressurePrepare;
    std::unique_ptr<PressureSavedWork> suspendedWork;
    bool forceWakePending = false;
    std::shared_ptr<JoltGpuConstraintTopology> constraintTopology;
    std::shared_ptr<const JoltGpuPrescribedSleepBounds> prescribedBounds;
    std::shared_ptr<rhi::PreparedComputeBatch> batch, preservation;
    PacketKind batchKind = PacketKind::Inputs;
    std::optional<JPH::GPU::Stage> batchHomogeneousStage;
    NumericalCostKey batchNumericalKey;
    size_t batchBegin = 0, batchEnd = 0;
    std::optional<size_t> sharedFallbackCursor;
    bool submissionBusy = false;
    uint64_t busyOpportunity = 0, busyProgress = 0, busySubmissions = 0;
    uint64_t earlyFeedbackOpportunity = 0;
    std::deque<Receipt> receipts;
    std::map<CostKey, std::deque<uint64_t>> costs;
    size_t lastCommandCount = 0;
    uint64_t lastCommandEstimate = 0;

    friend class JoltGpuTick;
};

void JoltGpuTick::Impl::Lane::Poll()
{
    ++pollSerial;
    for (auto *member : members) {
        if (!member)
            continue;
        try {
            member->Poll();
        } catch (...) {
            member->Fail();
            if (!members[1] && !members[2])
                throw;
        }
    }
}

bool JoltGpuTick::Impl::Lane::OtherReceipts(const Impl *member) const noexcept
{
    for (const auto *peer : members) {
        if (!peer || peer == member)
            continue;
        if (!peer->receipts.empty())
            return true;
        for (const auto &stage : peer->sharedStages)
            if (stage->completion)
                return true;
    }
    return false;
}

FrameWorkFacts JoltGpuTick::Impl::Lane::Describe() const
{
    if (members[0]->pressureActive || members[0]->PressureRequested())
        return members[0]->Describe();
    if (focus)
        return prepared && prepared->peer == focus ? prepared->owner->Describe() : focus->Describe();
    FrameWorkFacts result;
    const auto reserve = [this](FrameWorkCandidate &current, const FrameWorkCandidate &candidate, size_t slot,
                                size_t next) {
        const auto distance = [this, next](size_t index) { return (index + members.size() - next) % members.size(); };
        if (candidate.reserved && (!current.reserved || distance(slot) < distance(current.token - 1)))
            current = {slot + 1, candidate.hostNanoseconds, true, true};
    };
    for (size_t slot = 0; slot < members.size(); ++slot) {
        const auto *member = members[slot];
        if (!member)
            continue;
        const auto facts = member->Describe();
        result.pending |= facts.pending;
        result.observe |= facts.observe;
        result.host.ready |= facts.host.ready;
        result.gpu.ready |= facts.gpu.ready;
        reserve(result.host, facts.host, slot, nextHost);
        reserve(result.gpu, facts.gpu, slot, nextGpu);
        result.gpuBusy |= facts.gpuBusy;
    }
    result.gpuBusy &= !result.host.ready && !result.gpu.ready;
    return result;
}

void JoltGpuTick::Impl::Lane::DropShared(Impl *member) noexcept
{
    // Pressure, numerical cancellation and budget decline may surrender this
    // optional source owner. Accepted packets retain their actual allocations.
    if (member != members[2])
        bodyBatchInputs = {};
    if (!prepared || (prepared->owner != member && prepared->peer != member))
        return;
    const auto stage = std::move(prepared);
    if (stage->owner->batch)
        stage->owner->batch->Cancel();
    if (stage->owner->preservation)
        stage->owner->preservation->Cancel();
    stage->owner->batch.reset();
    stage->owner->preservation.reset();
    stage->owner->batchNumericalKey.clear();
    stage->owner->submissionBusy = false;
    for (auto *affected : {stage->owner, stage->peer}) {
        const auto found = std::find(affected->sharedStages.begin(), affected->sharedStages.end(), stage);
        if (found != affected->sharedStages.end())
            affected->sharedStages.erase(found);
    }
}

bool JoltGpuTick::Impl::Lane::PrepareShared()
{
    auto *a = members[0], *b = members[1];
    if (!a || !b || prepared || a->pressureActive || a->PressureRequested())
        return false;
    const auto ready = [](const Impl *member) {
        return member->admissionEnabled && member->status == Status::Running && member->controlAccepted &&
               member->inputAccepted && !member->preservation && !member->workContinuation && !member->workMinimum &&
               !member->submissionBusy && member->sharedFallbackCursor != member->cursor &&
               std::none_of(member->receipts.begin(), member->receipts.end(),
                            [](const auto &receipt) { return receipt.kind == PacketKind::Preservation; });
    };
    if (!ready(a) || !ready(b) || a->stageCursor.phase != b->stageCursor.phase ||
        a->stageCursor.substep != b->stageCursor.substep || a->stageCursor.iteration != b->stageCursor.iteration ||
        a->fixedDeltaTime != b->fixedDeltaTime || a->tickParameters.counts.w != b->tickParameters.counts.w ||
        a->tickParameters.selectedCounts.z || b->tickParameters.selectedCounts.z || a->control.participants ||
        b->control.participants || a->control.prepareCoupledContacts || b->control.prepareCoupledContacts)
        return false;
    const auto phase = a->stageCursor.phase;
    if (phase != Phase::PrepareKinematics && phase != Phase::ApplyForces && phase != Phase::Integrate)
        return false;
    if (std::memcmp(&a->tickParameters.solve, &b->tickParameters.solve, sizeof(a->tickParameters.solve)) ||
        std::memcmp(&a->tickParameters.contact, &b->tickParameters.contact, sizeof(a->tickParameters.contact)) ||
        std::memcmp(&a->tickParameters.restitution, &b->tickParameters.restitution,
                    sizeof(a->tickParameters.restitution)))
        return false;
    auto fullEndA = a->stageCursor, fullEndB = b->stageCursor;
    const auto fullA = a->TakeUnit(fullEndA), fullB = b->TakeUnit(fullEndB);
    const auto replaceable = [](const Impl *member, const JoltGpuStageWork &unit, const StageCursor &end) {
        if (!member->batch)
            return true;
        const auto &preparedEnd = member->batchStageEnd;
        return member->batchKind == PacketKind::Stages && member->batchBegin == member->cursor &&
               member->batchEnd == member->cursor + 1 && member->batchHomogeneousStage == unit.stage &&
               member->batch->HasWork() && !member->batch->GetSubmission().IsValid() &&
               !member->batch->RequiresPreparation() && !member->batch->HasPreservationWork() &&
               preparedEnd.phase == end.phase && preparedEnd.substep == end.substep &&
               preparedEnd.iteration == end.iteration && preparedEnd.localFirst == end.localFirst &&
               preparedEnd.islandBatchFirst == end.islandBatchFirst &&
               preparedEnd.islandBatchLanes == end.islandBatchLanes;
    };
    if (!replaceable(a, fullA, fullEndA) || !replaceable(b, fullB, fullEndB))
        return false;
    // Verify a prepared singleton at its original exact frontier first, then
    // bound the shared prefix through TakeUnit itself so both end cursors match.
    // Share the original quantum, including padding; give a short peer's unused
    // tiles back to the other member without admitting a rectangle of Worlds.
    constexpr uint32_t tileWidth = 64;
    const uint32_t maxTiles = JPH::GPU::WorkItemLimit(JPH::GPU::Domain::Bodies) / tileWidth;
    const auto tilesFor = [](uint32_t count) { return count / tileWidth + uint32_t(count % tileWidth != 0); };
    uint32_t tilesA = std::min(tilesFor(fullA.parameters.range.y), maxTiles / 2u);
    const uint32_t tilesB = std::min(tilesFor(fullB.parameters.range.y), maxTiles - tilesA);
    tilesA = std::min(tilesFor(fullA.parameters.range.y), maxTiles - tilesB);
    auto endA = a->stageCursor, endB = b->stageCursor;
    const auto unitA = a->TakeUnit(endA, tilesA * tileWidth);
    const auto unitB = b->TakeUnit(endB, tilesB * tileWidth);
    auto stage = std::make_shared<SharedStage>();
    stage->owner = a;
    stage->peer = b;
    stage->stage = unitA.stage;
    stage->ownerGeometry = a->inputs.GetGeometry();
    stage->peerGeometry = b->inputs.GetGeometry();
    stage->peerEnd = endB;
    stage->peerCursorEnd = b->cursor + 1;
    a->sharedStages.reserve(a->sharedStages.size() + 1);
    b->sharedStages.reserve(b->sharedStages.size() + 1);
    auto packet = a->programs.PrepareBodyBatch(a->storage, {{&a->inputs, unitA}, {&b->inputs, unitB}}, bodyBatchInputs);
    if (!packet)
        return false; // Managed defer accepts/cancels neither member's existing singleton.
    stage->key = Impl::Key(*packet, Impl::NumericalKey({unitA, unitB}));
    auto numericalKey = Impl::NumericalKey({unitA, unitB});
    // Any old packets are exact, unaccepted singletons at these same
    // accepted cursors. Finish all fallible preparation before replacing them;
    // their accepted receipts and the numerical cursors remain untouched.
    for (auto *member : {a, b}) {
        if (member->batch)
            member->batch->Cancel();
        member->batch.reset();
        member->batchNumericalKey.clear();
        member->batchHomogeneousStage.reset();
    }
    a->batchNumericalKey = std::move(numericalKey);
    a->batch = std::move(packet);
    a->batchKind = PacketKind::Stages;
    a->batchBegin = a->cursor;
    a->batchEnd = a->cursor + 1;
    a->batchStageEnd = endA;
    a->batchHomogeneousStage = unitA.stage;
    a->sharedStages.push_back(stage);
    b->sharedStages.push_back(stage);
    prepared = std::move(stage);
    return true;
}

bool JoltGpuTick::Impl::Lane::Advance(bool gpu, uint64_t token)
{
    if (members[0]->pressureActive || members[0]->PressureRequested())
        return members[0]->Advance(gpu);
    if (!focus && !token) {
        try {
            if (PrepareShared())
                return true;
        } catch (...) {
            members[0]->Fail();
            if (members[1])
                members[1]->Fail();
            return true;
        }
    }
    auto &next = gpu ? nextGpu : nextHost;
    const size_t start = token ? static_cast<size_t>(token - 1) : next;
    const size_t count = token ? 1 : members.size();
    for (size_t i = 0; i < count; ++i) {
        const size_t index = (start + i) % members.size();
        auto *member = members[index];
        if (!member)
            continue;
        auto *target = focus && prepared && prepared->peer == focus ? prepared->owner : focus;
        if (target && member != target)
            continue;
        const auto facts = member->Describe();
        const auto &candidate = gpu ? facts.gpu : facts.host;
        if (!candidate.ready || (token && !candidate.reserved))
            continue;
        if (token && gpu) {
            const auto pressure = member->host.GetQueuePressure();
            if (!pressure.frameSlots || pressure.recordingFrames || pressure.pendingFrames ||
                pressure.computeSubmissions)
                return false;
        }
        next = (index + 1) % members.size();
        const auto affected = prepared && prepared->owner == member ? prepared : nullptr;
        try {
            if (token && !gpu && !member->controlOnly)
                member->earlyFeedbackOpportunity = member->work.Opportunity();
            if (member->Advance(gpu)) {
                if (gpu)
                    member->earlyFeedbackOpportunity = 0;
                return true;
            }
        } catch (...) {
            if (affected && affected->completion)
                affected->peer->Fail();
            if (!members[1] && !members[2])
                throw;
            return true; // Each World observes its own retained Failed status.
        }
    }
    return false;
}

JoltGpuTick::JoltGpuTick(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs,
                         JoltGpuBroadphase &broadphase, JoltGpuNarrowphase &narrowphase,
                         JoltGpuConstraints &constraints, std::function<void()> failureObserver)
    : m_impl(std::make_unique<Impl>(work, storage, programs, broadphase, narrowphase, constraints,
                                    std::move(failureObserver)))
{
}
JoltGpuTick::JoltGpuTick(JoltGpuTick &owner, std::function<void()> failureObserver, bool observation)
    : m_impl(std::make_unique<Impl>(owner.m_impl->work, owner.m_impl->storage, owner.m_impl->programs,
                                    owner.m_impl->broadphase, owner.m_impl->narrowphase, owner.m_impl->constraints,
                                    std::move(failureObserver), owner.m_impl->lane, observation))
{
}

JoltGpuTick &JoltGpuTick::GetSharedMember(std::function<void()> failureObserver)
{
    if (!m_sharedMember)
        m_sharedMember.reset(new JoltGpuTick(*this, std::move(failureObserver)));
    return *m_sharedMember;
}

JoltGpuTick &JoltGpuTick::GetObservationMember()
{
    if (!m_observationMember)
        m_observationMember.reset(new JoltGpuTick(*this, {}, true));
    m_observationMember->ReleaseObservation();
    return *m_observationMember;
}

void JoltGpuTick::ReleaseBodyBatchInputs() noexcept
{
    m_impl->lane->bodyBatchInputs = {};
}

void JoltGpuTick::ReleaseObservation() noexcept
{
    if (!m_impl->CanRetire())
        return;
    m_impl->completedOutput = {};
    if (m_impl->status == Status::Failed || m_impl->status == Status::Cancelled)
        m_impl->status = Status::Idle;
}

void JoltGpuTick::TracePublicationRead(const char *boundary, uint32_t row) const noexcept
{
    try {
        if (INXLOG_GET_LEVEL() != LOG_DEBUG)
            return;
        const auto *a = m_impl->lane->members[0], *b = m_impl->lane->members[1];
        const auto future = [](const Impl *member) {
            return member && member->status == Status::Running && !member->controlOnly &&
                   member->stageCursor.phase != Impl::Phase::Done;
        };
        INXLOG_DIAGNOSTIC(
            "P4_PUBLICATION_READ boundary=", boundary, " row=", row, " slot0_row=", a ? a->selectedRow : UINT32_MAX,
            " slot0_status=", a ? uint32_t(a->status) : 0,
            " slot0_phase=", a ? uint32_t(a->stageCursor.phase) : UINT32_MAX,
            " slot0_substep=", a ? a->stageCursor.substep : 0, " slot0_iteration=", a ? a->stageCursor.iteration : 0,
            " slot0_local=", a ? a->stageCursor.localFirst : 0, " slot0_accepted=", a ? a->cursor : 0,
            " slot0_future=", future(a), " slot1_row=", b ? b->selectedRow : UINT32_MAX,
            " slot1_status=", b ? uint32_t(b->status) : 0,
            " slot1_phase=", b ? uint32_t(b->stageCursor.phase) : UINT32_MAX,
            " slot1_substep=", b ? b->stageCursor.substep : 0, " slot1_iteration=", b ? b->stageCursor.iteration : 0,
            " slot1_local=", b ? b->stageCursor.localFirst : 0, " slot1_accepted=", b ? b->cursor : 0,
            " slot1_future=", future(b));
    } catch (...) {
    }
}

void JoltGpuTick::SetSourceFocus(JoltGpuTick *member) noexcept
{
    m_impl->lane->focus = member ? member->m_impl.get() : nullptr;
    m_impl->work.Wake(m_impl->sourceId);
}

FrameWorkWaitResult JoltGpuTick::WaitForSourceProgress(uint64_t maximum)
{
    if (m_impl->lane->focus) {
        const auto focused = m_impl->lane->focus->WaitForProgress(maximum);
        if (focused != FrameWorkWaitResult::Blocked || m_impl->lane->focus != m_impl->lane->members[2])
            return focused;
        // A full queue may need an already accepted numerical ticket to retire.
        // Poll/wait that real predecessor only; never advance its future phases.
        for (size_t slot = 0; slot < 2; ++slot)
            if (auto *member = m_impl->lane->members[slot]) {
                const auto dependency = member->WaitForProgress(maximum);
                if (dependency != FrameWorkWaitResult::Blocked)
                    return dependency;
            }
        return FrameWorkWaitResult::Blocked;
    }
    const auto own = WaitForProgress(maximum);
    return own != FrameWorkWaitResult::Blocked || !m_sharedMember ? own : m_sharedMember->WaitForProgress(maximum);
}

JoltGpuTick::~JoltGpuTick() = default;
void JoltGpuTick::Begin(uint32_t worldRow, float deltaTime, const JoltGpuStepParameters &parameters,
                        std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces)
{
    m_impl->Begin(worldRow, deltaTime, &parameters, {}, std::move(cpuForces));
}
void JoltGpuTick::Begin(uint32_t worldRow, float deltaTime, const JoltGpuStepParameters &parameters,
                        JoltGpuTickControl control, std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces)
{
    m_impl->Begin(worldRow, deltaTime, &parameters, std::move(control), std::move(cpuForces));
}
void JoltGpuTick::BeginControl(uint32_t worldRow, JoltGpuTickControl control)
{
    m_impl->Begin(worldRow, 0, nullptr, std::move(control), std::nullopt);
}
void JoltGpuTick::Cancel() noexcept
{
    m_impl->Cancel();
}
void JoltGpuTick::ReopenForPartitionCleanup()
{
    if (!m_impl->CanRetire() || (m_impl->status != Status::Cancelled && m_impl->status != Status::Failed))
        throw std::logic_error("GPU partition cleanup requires settled cancelled source receipts");
    m_impl->status = Status::Idle;
}
void JoltGpuTick::SetAdmissionEnabled(bool enabled) noexcept
{
    if (m_impl->admissionEnabled == enabled)
        return;
    m_impl->admissionEnabled = enabled;
    if (!enabled)
        m_impl->broadphase.ResetCapacityHint();
    if (enabled)
        m_impl->work.Wake(m_impl->sourceId);
}
void JoltGpuTick::SetPressureOwner(std::function<bool()> observe, std::function<bool()> pending,
                                   std::function<JoltGpuControlWork(uint32_t, PressureCompletion)> prepare)
{
    m_impl->pressureObserve = std::move(observe);
    m_impl->pressurePending = std::move(pending);
    m_impl->pressurePrepare = std::move(prepare);
    m_impl->work.Wake(m_impl->sourceId);
}
bool JoltGpuTick::IsPressureActive() const noexcept
{
    return m_impl->pressureActive;
}
bool JoltGpuTick::IsAdmissionEnabled() const noexcept
{
    return m_impl->admissionEnabled;
}
FrameWorkOwner::SourceId JoltGpuTick::GetSourceId() const noexcept
{
    return m_impl->sourceId;
}
JoltGpuTick::Status JoltGpuTick::GetStatus() const noexcept
{
    return m_impl->status;
}
bool JoltGpuTick::IsPending() const noexcept
{
    return GetStatus() == Status::Running;
}
bool JoltGpuTick::IsComplete() const noexcept
{
    return GetStatus() == Status::Complete;
}
bool JoltGpuTick::AreControlInputsAccepted() const noexcept
{
    return m_impl->controlAccepted;
}
FrameWorkWaitResult JoltGpuTick::WaitForProgress(uint64_t maxWaitNanoseconds)
{
    return m_impl->WaitForProgress(maxWaitNanoseconds);
}
bool JoltGpuTick::HasReceipts() const noexcept
{
    return !m_impl->receipts.empty() || std::any_of(m_impl->sharedStages.begin(), m_impl->sharedStages.end(),
                                                    [](const auto &stage) { return stage->completion.has_value(); });
}
bool JoltGpuTick::CanRetire() const noexcept
{
    return m_impl->CanRetire();
}
float JoltGpuTick::GetFixedDeltaTime() const noexcept
{
    return m_impl->fixedDeltaTime;
}
size_t JoltGpuTick::GetAcceptedStageCount() const noexcept
{
    return m_impl->cursor;
}
const JoltGpuTickOutput &JoltGpuTick::GetCompletedOutput() const noexcept
{
    return m_impl->completedOutput;
}

} // namespace infernux
