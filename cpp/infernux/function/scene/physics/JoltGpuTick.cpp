#include "JoltGpuTick.h"
#include "JoltGpuBroadphase.h"
#include "JoltGpuConstraints.h"

#include <algorithm>
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
  public:
    Impl(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs, JoltGpuBroadphase &broadphase,
         JoltGpuConstraints &constraints)
        : work(work), storage(storage), programs(programs), broadphase(broadphase), constraints(constraints),
          host(storage.GetHost())
    {
        FrameWorkSource source;
        source.observe = [this] { Poll(); };
        source.refresh = source.observe;
        source.describe = [this] { return Describe(); };
        source.advance = [this](bool gpu, uint64_t token) { return Advance(gpu, token); };
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
        work.RemoveSource(sourceId);
    }

    void Begin(float deltaTime, const JoltGpuStepParameters *settings, JoltGpuTickControl pendingControl,
               std::optional<std::vector<JPH::GPU::BodyForces>> forces)
    {
        FrameWorkOwner::Scope scope(work);
        if (status != Status::Idle && status != Status::Complete)
            throw std::logic_error("GPU tick Begin requires an idle or fully completed owner");
        if (settings && (!std::isfinite(deltaTime) || deltaTime <= 0))
            throw std::invalid_argument("GPU fixed delta time must be finite and positive");
        if (!pendingControl.commandProgram && std::any_of(pendingControl.groups.begin(), pendingControl.groups.end(),
                                                          [](const auto &group) { return !group.commands.empty(); }))
            throw std::invalid_argument("GPU tick commands require their command program");
        ValidateControlPacket(pendingControl.initialization);
        parameters = settings ? std::make_optional(*settings) : std::nullopt;
        reuseParameters = false;
        stageCursor = {};
        if (parameters) {
            auto gravityDt = parameters->Get().gravityDt;
            gravityDt.w = deltaTime / static_cast<float>(parameters->Get().solve.y);
            parameters->SetGravityAndDeltaTime(gravityDt);
        }

        fixedDeltaTime = deltaTime;
        controlOnly = settings == nullptr;
        cpuForces = std::move(forces);
        inputs = {};
        control = std::move(pendingControl);
        workContinuation = {};
        groupCursor = 0;
        modelPrepared = false;
        modelUpdate.reset();
        commandCursor.reset();
        controlAccepted = false;
        finalPrepared = finalAccepted = !control.prepareFinalWork;
        finalStates.reset();
        cursor = 0;
        inputAccepted = controlOnly;
        submissionBusy = false;
        status = Status::Running;
        try {
            if (!control.initialization && control.groups.empty())
                CaptureControlInputs();
        } catch (...) {
            status = Status::Failed;
            Cancel();
            throw;
        }
        work.Wake(sourceId);
    }

    void Cancel() noexcept
    {
        if (status == Status::Running)
            status = Status::Cancelled;
        CancelPackets();
        inputs = {};
        parameterInput = {};
        acceptedParameters.reset();
        parameters.reset();
        cpuForces.reset();
        control = {};
        workContinuation = {};
        finalStates.reset();
        modelUpdate.reset();
        commandCursor.reset();
        submissionBusy = false;
    }

    bool CanRetire() const noexcept
    {
        return status != Status::Running && receipts.empty();
    }

    FrameWorkWaitResult WaitForProgress(uint64_t maxWaitNanoseconds)
    {
        if (receipts.empty())
            return FrameWorkWaitResult::Blocked;
        const auto count = receipts.size();
        try {
            const bool completed =
                host.Queue().WaitFor(receipts.front().completion.GetSubmission(), maxWaitNanoseconds);
            Poll();
            if (receipts.size() != count)
                return FrameWorkWaitResult::Progress;
            return completed ? FrameWorkWaitResult::Blocked : FrameWorkWaitResult::Waiting;
        } catch (...) {
            status = Status::Failed;
            Cancel();
            throw;
        }
    }

    FrameWorkOwner &work;
    JoltGpuStorage &storage;
    JoltGpuPrograms &programs;
    JoltGpuBroadphase &broadphase;
    JoltGpuConstraints &constraints;
    rhi::ComputeHost host;
    FrameWorkOwner::SourceId sourceId = 0;
    Status status = Status::Idle;
    bool admissionEnabled = true;
    float fixedDeltaTime = 0;
    size_t cursor = 0;
    std::shared_ptr<rhi::ComputeBuffer> completedStates;

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
        Preservation
    };
    enum class HostAction : uint64_t
    {
        Prepare = 1,
        Wait
    };
    enum class Phase
    {
        PrepareKinematics,
        ApplyForces,
        BuildBounds,
        Broadphase,
        CollidePairs,
        SetupContacts,
        SetupJoints,
        BuildConstraints,
        WarmJoints,
        SolveJointVelocity,
        SolveVelocity,
        Integrate,
        SolveJointPosition,
        SolvePosition,
        FinishJoints,
        FinishTick,
        Done
    };
    struct StageCursor
    {
        Phase phase = Phase::Done;
        uint32_t substep = 0;
        uint32_t iteration = 0;
        uint32_t islandBatchFirst = 0;
        uint32_t localFirst = 0;
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
        case Phase::CollidePairs:
            return {Stage::CollidePairs, Domain::Patches};
        case Phase::SetupContacts:
            return {Stage::SetupContacts, Domain::Contacts};
        case Phase::SetupJoints:
            return {Stage::SetupJoints, Domain::Joints};
        case Phase::WarmJoints:
            return {Stage::WarmJoints, Domain::IslandJoints};
        case Phase::SolveJointVelocity:
            return {Stage::SolveJointVelocity, Domain::IslandJoints};
        case Phase::SolveVelocity:
            return {Stage::SolveVelocity, Domain::IslandPatches};
        case Phase::Integrate:
            return {Stage::Integrate, Domain::Bodies};
        case Phase::SolveJointPosition:
            return {Stage::SolveJointPosition, Domain::IslandJoints};
        case Phase::SolvePosition:
            return {Stage::SolvePosition, Domain::IslandContacts};
        case Phase::FinishJoints:
            return {Stage::FinishJoints, Domain::Joints};
        case Phase::FinishTick:
            return {Stage::FinishTick, Domain::Bodies};
        case Phase::Broadphase:
        case Phase::BuildConstraints:
        case Phase::Done:
            break;
        }
        throw std::logic_error("A completed GPU tick has no stage span");
    }

    static bool IslandDomain(JPH::GPU::Domain domain)
    {
        using JPH::GPU::Domain;
        return domain == Domain::IslandJoints || domain == Domain::IslandPatches || domain == Domain::IslandContacts;
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
        if (domain == JPH::GPU::Domain::IslandJoints)
            return plan->maxJointItems;
        return domain == JPH::GPU::Domain::IslandContacts ? plan->maxPatchItems * JPH::GPU::MaxContactPointsPerPair
                                                          : plan->maxPatchItems;
    }

    uint32_t ItemCount(JPH::GPU::Domain domain) const
    {
        using JPH::GPU::Domain;
        switch (domain) {
        case Domain::Bodies:
            return storage.GetBodyCount();
        case Domain::Shapes:
            return storage.GetShapeCount();
        case Domain::Patches:
            return inputs.GetContacts() ? inputs.GetContacts()->patchCount : 0;
        case Domain::Contacts:
            return inputs.GetContacts() ? inputs.GetContacts()->patchCount * JPH::GPU::MaxContactPointsPerPair : 0;
        case Domain::Joints:
            return storage.GetJointCount();
        default:
            throw std::logic_error("Island-local GPU work requires its installed schedule");
        }
    }

    void NextSpan(StageCursor &next) const
    {
        const auto &plan = inputs.GetConstraints();
        const bool responding = plan && (plan->jointIslandCount || plan->contactIslandCount);
        const bool velocity = tickParameters.counts.w && responding;
        const bool position = tickParameters.solve.x && responding;
        next.islandBatchFirst = 0;
        next.localFirst = 0;
        switch (next.phase) {
        case Phase::PrepareKinematics:
            next.phase = Phase::ApplyForces;
            break;
        case Phase::ApplyForces:
            next.phase = Phase::BuildBounds;
            break;
        case Phase::BuildBounds:
            next.phase = Phase::Broadphase;
            break;
        case Phase::Broadphase:
            next.phase = Phase::CollidePairs;
            break;
        case Phase::CollidePairs:
            next.phase = Phase::SetupContacts;
            break;
        case Phase::SetupContacts:
            next.phase = Phase::SetupJoints;
            break;
        case Phase::SetupJoints:
            next.phase = Phase::BuildConstraints;
            break;
        case Phase::BuildConstraints:
            next.iteration = 0;
            next.phase = velocity ? Phase::WarmJoints : Phase::Integrate;
            break;
        case Phase::WarmJoints:
            next.phase = Phase::SolveJointVelocity;
            break;
        case Phase::SolveJointVelocity:
            next.phase = Phase::SolveVelocity;
            break;
        case Phase::SolveVelocity:
            next.phase = ++next.iteration < tickParameters.counts.w ? Phase::SolveJointVelocity : Phase::Integrate;
            break;
        case Phase::Integrate:
            next.iteration = 0;
            next.phase = position ? Phase::SolveJointPosition : Phase::FinishJoints;
            break;
        case Phase::SolveJointPosition:
            next.phase = Phase::SolvePosition;
            break;
        case Phase::SolvePosition:
            next.phase = ++next.iteration < tickParameters.solve.x ? Phase::SolveJointPosition : Phase::FinishJoints;
            break;
        case Phase::FinishJoints:
            next.phase = ++next.substep < tickParameters.solve.y ? Phase::ApplyForces : Phase::FinishTick;
            break;
        case Phase::FinishTick:
            next.phase = Phase::Done;
            break;
        case Phase::Done:
            break;
        }
    }

    void Normalize(StageCursor &next) const
    {
        while (next.phase != Phase::Done) {
            if (next.phase == Phase::Broadphase || next.phase == Phase::BuildConstraints)
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

    JoltGpuStageWork TakeUnit(StageCursor &next) const
    {
        const auto span = Span(next.phase);
        const auto quantum = JPH::GPU::WorkItemLimit(span.domain);
        JPH::GPU::StageWork parameters{};
        if (IslandDomain(span.domain)) {
            const auto lanes = std::min({uint32_t{64}, quantum, IslandCount(span.domain) - next.islandBatchFirst});
            const auto maximum = IslandMaxItems(span.domain);
            const auto count = std::min(quantum / lanes, maximum - next.localFirst);
            parameters.range = {next.localFirst, count, next.islandBatchFirst, lanes};
            next.localFirst += count;
            if (next.localFirst == maximum) {
                next.localFirst = 0;
                next.islandBatchFirst += lanes;
            }
        } else {
            const auto count = std::min(quantum, ItemCount(span.domain) - next.localFirst);
            parameters.range = {next.localFirst, count, 0, 0};
            next.localFirst += count;
        }
        Normalize(next);
        return {span.stage, parameters};
    }

    void CaptureStageSpans()
    {
        stageCursor.phase = storage.GetBodyCount() ? Phase::PrepareKinematics : Phase::Done;
        Normalize(stageCursor);
    }

    void AcceptContactWorkset(std::shared_ptr<const JoltGpuContactWorkset> contacts)
    {
        inputs.SetContacts(std::move(contacts));
        stageCursor.phase = Phase::CollidePairs;
        stageCursor.localFirst = 0;
        stageCursor.islandBatchFirst = 0;
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

    void CaptureControlInputs()
    {
        if (!controlOnly) {
            // Model controls can repack leaves/patches. All sizes, parameter
            // reuse and stage domains come from that accepted model boundary.
            auto actual = parameters->Get();
            actual.counts.x = storage.GetBodyCount();
            actual.counts.y = storage.GetWorldCount();
            actual.counts.z = storage.GetShapeCount();
            actual.solve.z = actual.solve.w = 0;
            tickParameters = actual;
            reuseParameters = acceptedParameters && std::memcmp(&*acceptedParameters, &actual, sizeof(actual)) == 0;
            inputs = reuseParameters ? parameterInput : JoltGpuStepInputs{};
            inputAccepted = (reuseParameters && !cpuForces) || !storage.GetBodyCount();
            CaptureStageSpans();
            for (const auto resource :
                 {JPH::GPU::Resource::BodyModels, JPH::GPU::Resource::ShapeModels, JPH::GPU::Resource::Worlds})
                inputs.Capture(storage, resource);
            if (!cpuForces)
                inputs.Capture(storage, JPH::GPU::Resource::BodyForces);
        }
        controlAccepted = true;
        FinishCompleted();
    }

    void Complete()
    {
        completedStates = finalPrepared && finalStates ? std::move(finalStates)
                          : storage.GetBodyCount()     ? storage.Get(JPH::GPU::Resource::BodyStates)->Freeze()
                                                       : nullptr;
        status = Status::Complete;
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
        if (status == Status::Running && NumericalInputsAccepted() && finalAccepted && !workContinuation &&
            receipts.empty())
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
        workContinuation = std::move(selected.continueWork);
        continuationKind = kind;
        if (batch)
            batchKind = kind;
        else if (workContinuation) {
            submissionBusy = true;
            busyOpportunity = work.Opportunity();
            busyProgress = work.ProgressSerial();
            busySubmissions = host.Queue().GetPendingSubmissionCount();
        }
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

    bool NeedsSmallerPrefix() const
    {
        if (!batch || preservation || (batchKind != PacketKind::Commands && batchKind != PacketKind::Stages) ||
            batchEnd - batchBegin <= 1 || batch->RequiresPreparation() || batch->HasPreservationWork())
            return false;
        return Prediction(SubmissionKey(*batch)) > work.ResourceState(FrameWorkResource::Gpu).limit ||
               batch->GetTransferBytes() > work.ResourceState(FrameWorkResource::Transfer).limit;
    }

    bool CanSubmit() const
    {
        const auto *packet = Submission();
        if (!packet)
            return false;
        const bool indivisible = preservation ? packet->GetTransferBytes() <= 4
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
        if (!admissionEnabled) {
            facts.pending = facts.gpuBusy = !receipts.empty();
            return facts;
        }
        facts.pending = status == Status::Running || !receipts.empty();
        if (status != Status::Running)
            return facts;
        // A completion-dependent consumer stays on this source. Waiting for
        // its preceding receipts neither reruns its factory nor admits another
        // control group, solver stage or final render consumer.
        if (!batch && workContinuation && !receipts.empty()) {
            facts.host = {static_cast<uint64_t>(HostAction::Wait), 0, true, false};
            facts.gpuBusy = true;
            return facts;
        }
        if (!batch && workContinuation && submissionBusy)
            return facts;
        if (NumericalInputsAccepted() && finalAccepted) {
            if (!receipts.empty())
                facts.host = {static_cast<uint64_t>(HostAction::Wait), 0, true, false};
            facts.gpuBusy = !receipts.empty();
            return facts;
        }
        if ((!batch && (workContinuation || !controlAccepted || !inputAccepted || stageCursor.phase != Phase::Done ||
                        !finalPrepared)) ||
            (batch && !preservation &&
             (batch->RequiresPreparation() || batch->HasPreservationWork() || NeedsSmallerPrefix()))) {
            facts.host = {static_cast<uint64_t>(HostAction::Prepare), 0, true, false};
        } else if (!submissionBusy && CanSubmit()) {
            facts.gpu = {0, 0, true, false};
        }
        // Accepted predecessors do not block this source's ready successor.
        // Its actual buffer dependencies and queue-slot admission order it.
        if (!facts.host.ready && !facts.gpu.ready && !receipts.empty())
            facts.host = {static_cast<uint64_t>(HostAction::Wait), 0, true, false};
        facts.gpuBusy = !receipts.empty() && !facts.gpu.ready &&
                        (!facts.host.ready || facts.host.token == static_cast<uint64_t>(HostAction::Wait));
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
            while (ready.size() < MaxPacketUnits && next.phase != Phase::Done && next.phase != Phase::Broadphase &&
                   next.phase != Phase::BuildConstraints) {
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

    bool Prepare()
    {
        if (NeedsSmallerPrefix()) {
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
                if (!receipts.empty())
                    return false;
                const auto kind = continuationKind;
                auto continuation = std::move(workContinuation);
                workContinuation = {};
                SelectControlWork(continuation(), kind);
                if (!batch && !workContinuation) {
                    if (kind == PacketKind::FinalWork)
                        finalAccepted = true;
                    else if (kind != PacketKind::NumericalWork)
                        FinishControlGroup();
                    FinishCompleted();
                }
                return true;
            }
            if (NumericalInputsAccepted()) {
                if (!finalPrepared) {
                    finalStates =
                        storage.GetBodyCount() ? storage.Get(JPH::GPU::Resource::BodyStates)->Freeze() : nullptr;
                    auto prepareFinalWork = std::move(control.prepareFinalWork);
                    control.prepareFinalWork = {};
                    finalPrepared = true;
                    SelectControlWork(prepareFinalWork(finalStates), PacketKind::FinalWork);
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
                    commandCursor = control.commandProgram->Prepare(storage, std::move(group.commands));
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
                batch = reuseParameters ? storage.PrepareForces(*cpuForces)
                                        : storage.PrepareInputs(*parameters, cpuForces ? &*cpuForces : nullptr);
                batchKind = PacketKind::Inputs;
                return true;
            }
            if (stageCursor.phase == Phase::Broadphase) {
                inputs.ClearContacts();
                SelectControlWork(
                    broadphase.Prepare(storage, inputs, tickParameters,
                                       [this](auto contacts) { AcceptContactWorkset(std::move(contacts)); }),
                    PacketKind::NumericalWork);
                return true;
            }
            if (stageCursor.phase == Phase::BuildConstraints) {
                inputs.ClearConstraints();
                SelectControlWork(constraints.Prepare(storage, inputs, tickParameters,
                                                      [this](auto plan) { AcceptConstraintWorkset(std::move(plan)); }),
                                  PacketKind::NumericalWork);
                return true;
            }
            return PreparePrefix();
        }
        // Resolving a full-overwrite COW allocation can require no transfer at
        // all. A minimum preparation range does not grant its copy submission.
        const auto allowance = std::max(uint64_t{4}, work.ResourceRemaining(FrameWorkResource::Transfer));
        const auto prepared = batch->PreparePreservation(allowance, true);
        if (!prepared.capacityGranted)
            return false;
        preservation = prepared.copy;
        return true;
    }

    void Accepted(const std::shared_ptr<rhi::PreparedComputeBatch> &packet, CostKey key, uint64_t prediction,
                  uint64_t started)
    {
        const auto kind = preservation ? PacketKind::Preservation : batchKind;
        const auto bytes = packet->GetTransferBytes();
        const bool prefix = kind == PacketKind::Stages || kind == PacketKind::Commands;
        const auto homogeneousStage = kind == PacketKind::Stages ? batchHomogeneousStage : std::nullopt;
        receipts.emplace_back(
            Receipt{packet->GetCompletion(), std::move(key), kind, prefix ? batchEnd - batchBegin : 0, homogeneousStage,
                    work.Opportunity(), work.RecordAcceptedResource(FrameWorkResource::Gpu, prediction),
                    bytes ? work.RecordAcceptedResource(FrameWorkResource::Transfer, bytes) : 0, bytes, started});
        if (kind == PacketKind::Preservation) {
            preservation.reset();
        } else {
            batch.reset();
            batchNumericalKey.clear();
            if (kind == PacketKind::Initialization) {
                control.initialization.reset();
            } else if (kind == PacketKind::Model) {
                modelUpdate.reset();
                ModelAccepted();
            } else if (kind == PacketKind::Inputs) {
                if (!reuseParameters) {
                    inputs.Capture(storage, JPH::GPU::Resource::StepParams);
                    parameterInput.Capture(storage, JPH::GPU::Resource::StepParams);
                    acceptedParameters = tickParameters;
                }
                if (cpuForces)
                    inputs.Capture(storage, JPH::GPU::Resource::BodyForces);
                cpuForces.reset();
                inputAccepted = true;
            } else if (kind == PacketKind::Stages) {
                cursor = batchEnd;
                stageCursor = batchStageEnd;
            } else if (kind == PacketKind::FinalWork && !workContinuation) {
                finalAccepted = true;
            }
            if (!controlAccepted)
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

    bool Advance(bool gpu, uint64_t token)
    {
        if (!admissionEnabled || status != Status::Running)
            return false;
        try {
            if (gpu)
                return Submit();
            if (token == static_cast<uint64_t>(HostAction::Wait)) {
                if (receipts.empty())
                    return false;
                const auto remaining = work.ServiceRemaining();
                if (remaining <= 0 || !host.Queue().WaitFor(receipts.front().completion.GetSubmission(),
                                                            static_cast<uint64_t>(remaining)))
                    return false;
                Poll();
                return true;
            }
            return Prepare();
        } catch (...) {
            status = Status::Failed;
            Cancel();
            throw;
        }
    }

    std::optional<JoltGpuStepParameters> parameters;
    JoltGpuTickControl control;
    std::function<JoltGpuControlWork()> workContinuation;
    PacketKind continuationKind = PacketKind::Work;
    std::unique_ptr<JoltGpuModelUpdate> modelUpdate;
    size_t groupCursor = 0;
    bool modelPrepared = false;
    std::unique_ptr<JoltGpuCommandCursor> commandCursor;
    bool controlAccepted = false;
    bool controlOnly = false;
    bool finalPrepared = true, finalAccepted = true;
    std::shared_ptr<rhi::ComputeBuffer> finalStates;
    std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces;
    JoltGpuStepInputs inputs;
    JoltGpuStepInputs parameterInput;
    std::optional<JPH::GPU::StepParams> acceptedParameters;
    JPH::GPU::StepParams tickParameters{};
    bool reuseParameters = false;
    StageCursor stageCursor, batchStageEnd;
    std::map<JPH::GPU::Stage, UnitHistory> stageEstimates;
    UnitHistory mixedEstimates;
    bool inputAccepted = false;
    std::shared_ptr<rhi::PreparedComputeBatch> batch, preservation;
    PacketKind batchKind = PacketKind::Inputs;
    std::optional<JPH::GPU::Stage> batchHomogeneousStage;
    NumericalCostKey batchNumericalKey;
    size_t batchBegin = 0, batchEnd = 0;
    bool submissionBusy = false;
    uint64_t busyOpportunity = 0, busyProgress = 0, busySubmissions = 0;
    std::deque<Receipt> receipts;
    std::map<CostKey, std::deque<uint64_t>> costs;
    size_t lastCommandCount = 0;
    uint64_t lastCommandEstimate = 0;

    friend class JoltGpuTick;
};

JoltGpuTick::JoltGpuTick(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs,
                         JoltGpuBroadphase &broadphase, JoltGpuConstraints &constraints)
    : m_impl(std::make_unique<Impl>(work, storage, programs, broadphase, constraints))
{
}
JoltGpuTick::~JoltGpuTick() = default;
void JoltGpuTick::Begin(float deltaTime, const JoltGpuStepParameters &parameters,
                        std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces)
{
    m_impl->Begin(deltaTime, &parameters, {}, std::move(cpuForces));
}
void JoltGpuTick::Begin(float deltaTime, const JoltGpuStepParameters &parameters, JoltGpuTickControl control,
                        std::optional<std::vector<JPH::GPU::BodyForces>> cpuForces)
{
    m_impl->Begin(deltaTime, &parameters, std::move(control), std::move(cpuForces));
}
void JoltGpuTick::BeginControl(JoltGpuTickControl control)
{
    m_impl->Begin(0, nullptr, std::move(control), std::nullopt);
}
void JoltGpuTick::Cancel() noexcept
{
    m_impl->Cancel();
}
void JoltGpuTick::SetAdmissionEnabled(bool enabled) noexcept
{
    if (m_impl->admissionEnabled == enabled)
        return;
    m_impl->admissionEnabled = enabled;
    if (enabled)
        m_impl->work.Wake(m_impl->sourceId);
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
    return !m_impl->receipts.empty();
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
const std::shared_ptr<rhi::ComputeBuffer> &JoltGpuTick::GetCompletedBodyStates() const noexcept
{
    return m_impl->completedStates;
}

} // namespace infernux
