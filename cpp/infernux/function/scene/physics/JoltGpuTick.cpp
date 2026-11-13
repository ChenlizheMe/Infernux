#include "JoltGpuTick.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
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
    Impl(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs)
        : work(work), storage(storage), programs(programs), host(storage.GetHost())
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
        stages.clear();
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
        planLength = 0;
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
    using CostKey = std::pair<std::vector<uint64_t>, uint64_t>;
    struct Receipt
    {
        rhi::ComputeCompletion completion;
        CostKey key;
        PacketKind kind;
        size_t stageCount;
        uint64_t submittedOpportunity;
        uint64_t gpuGrant;
        uint64_t transferGrant;
        uint64_t transferBytes;
        uint64_t submitStarted;
    };

    static CostKey Key(const rhi::PreparedComputeBatch &batch)
    {
        return {batch.GetExecutionSignature(), batch.GetTransferBytes()};
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
                lastStageCount = receipt.stageCount;
                lastStageEstimate = estimate;
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
        for (auto *packet : {&batch, &preservation, &probeBatch}) {
            if (*packet)
                (*packet)->Cancel();
            packet->reset();
        }
        planLength = 0;
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
            actual.solve.z = storage.GetContactCount();
            actual.solve.w = storage.GetPatchCount();
            tickParameters = actual;
            reuseParameters = acceptedParameters && std::memcmp(&*acceptedParameters, &actual, sizeof(actual)) == 0;
            inputs = reuseParameters ? parameterInput : JoltGpuStepInputs{};
            inputAccepted = (reuseParameters && !cpuForces) || !storage.GetBodyCount();
            if (storage.GetBodyCount()) {
                stages.push_back(JPH::GPU::Stage::PrepareKinematics);
                for (uint32_t substep = 0; substep < actual.solve.y; ++substep) {
                    stages.push_back(JPH::GPU::Stage::ApplyForces);
                    if (storage.GetShapeCount()) {
                        stages.push_back(JPH::GPU::Stage::BuildBounds);
                        stages.push_back(JPH::GPU::Stage::CollidePairs);
                        stages.push_back(JPH::GPU::Stage::SetupContacts);
                    }
                    if (storage.GetJointCount())
                        stages.push_back(JPH::GPU::Stage::SetupJoints);
                    if (storage.GetPatchCount() || storage.GetJointCount()) {
                        for (uint32_t iteration = 0; iteration < actual.counts.w; ++iteration)
                            stages.push_back(JPH::GPU::Stage::SolveVelocity);
                    }
                    stages.push_back(JPH::GPU::Stage::Integrate);
                    if (storage.GetPatchCount() || storage.GetJointCount())
                        for (uint32_t iteration = 0; iteration < actual.solve.x; ++iteration)
                            stages.push_back(JPH::GPU::Stage::SolvePosition);
                    if (storage.GetJointCount())
                        stages.push_back(JPH::GPU::Stage::FinishJoints);
                }
                stages.push_back(JPH::GPU::Stage::FinishTick);
            }
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
        return controlAccepted && (controlOnly || storage.GetBodyCount() == 0 || cursor == stages.size());
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
        if (!selected.packet && selected.continueAfterCompletion)
            throw std::invalid_argument("GPU control continuation requires a preceding packet receipt");
        batch = std::move(selected.packet);
        workContinuation = std::move(selected.continueAfterCompletion);
        continuationKind = kind;
        if (batch)
            batchKind = kind;
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

    bool CanSubmit() const
    {
        const auto *packet = Submission();
        if (!packet)
            return false;
        const bool indivisible = preservation ? packet->GetTransferBytes() <= 4
                                 : batchKind == PacketKind::Commands || batchKind == PacketKind::Stages
                                     ? batchEnd - batchBegin == 1
                                     : true;
        return work.CanAdmitResource(FrameWorkResource::Gpu, Prediction(Key(*packet)), indivisible) &&
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
        if (NumericalInputsAccepted() && finalAccepted) {
            if (!receipts.empty())
                facts.host = {static_cast<uint64_t>(HostAction::Wait), 0, true, false};
            facts.gpuBusy = !receipts.empty();
            return facts;
        }
        if ((!batch &&
             (workContinuation || !controlAccepted || !inputAccepted || cursor < stages.size() || !finalPrepared)) ||
            (batch && !preservation && (batch->RequiresPreparation() || batch->HasPreservationWork()))) {
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

    void Select(std::shared_ptr<rhi::PreparedComputeBatch> selected, size_t end)
    {
        if (probeBatch && probeBatch != selected)
            probeBatch->Cancel();
        probeBatch.reset();
        batch = std::move(selected);
        batchEnd = end;
        planLength = 0;
    }

    bool PreparePrefix(bool commands = false)
    {
        const size_t begin = commands ? commandCursor->GetAcceptedDispatchCount() : cursor;
        const size_t remaining = commands ? commandCursor->GetDispatchCount() - begin : stages.size() - begin;
        if (!planLength) {
            planLength = remaining;
            probeLimit = remaining;
            // An unknown predecessor temporarily books the whole remaining
            // allowance. Plan its successor for the configured window instead
            // of sealing a one-stage packet merely because that grant has not
            // retired yet. Actual submission still uses ResourceRemaining.
            const auto available = work.ResourceRemaining(FrameWorkResource::Gpu);
            planWindow = available ? available : work.ResourceState(FrameWorkResource::Gpu).limit;
            if (commands) {
                const auto transfer = work.ResourceRemaining(FrameWorkResource::Transfer);
                planTransferWindow = transfer ? transfer : work.ResourceState(FrameWorkResource::Transfer).limit;
                // Every command dispatch uploads at least one command/span
                // and one parameter record. Larger counts cannot fit even
                // before preparing their exact payloads.
                constexpr uint64_t minimumBytes = sizeof(JPH::GPU::BodyCommand) + sizeof(JPH::GPU::BodyCommandSpan) +
                                                  sizeof(JPH::GPU::BodyCommandParams);
                planLength =
                    std::min(planLength, std::max(size_t{1}, static_cast<size_t>(planTransferWindow / minimumBytes)));
            }
            // Only sizes an unknown calibration; it does not pretend a packet
            // receipt measured individual stages. Exact signature costs below
            // still decide whether a known prefix fits.
            const auto count = commands ? lastCommandCount : lastStageCount;
            const auto estimate = commands ? lastCommandEstimate : lastStageEstimate;
            if (estimate && count) {
                const long double width = static_cast<long double>(planWindow) * count / estimate;
                probeLimit = width >= remaining ? remaining : std::max(size_t{1}, static_cast<size_t>(width));
            }
        }
        const size_t count = planLength--;
        auto candidate = commands ? commandCursor->PrepareNext(host, static_cast<uint32_t>(count))
                                  : programs.PrepareStages(storage, inputs,
                                                           std::vector<JPH::GPU::Stage>(
                                                               stages.begin() + begin, stages.begin() + begin + count));
        const auto prediction = Prediction(Key(*candidate));
        const bool transferFits = !commands || candidate->GetTransferBytes() <= planTransferWindow;
        if (transferFits &&
            ((prediction && prediction <= planWindow) || (commands && !prediction && count <= probeLimit))) {
            Select(std::move(candidate), begin + count);
        } else if (!commands && !prediction && count <= probeLimit && !probeBatch) {
            probeBatch = std::move(candidate);
            probeEnd = begin + count;
            // The first finite small-world tick can calibrate its whole ready
            // suffix instead of artificially requiring one host per stage.
            if (count == remaining)
                Select(probeBatch, probeEnd);
        }
        if (candidate) {
            candidate->Cancel();
            if (commands)
                commandCursor->DiscardPreparedPrefix();
        }
        if (!batch && !planLength) {
            if (probeBatch)
                Select(probeBatch, probeEnd);
            else {
                // No shorter known prefix fits. Retain the minimum legal
                // stage; ordinary indivisible admission can make progress.
                batch = commands ? commandCursor->PrepareNext(host, 1)
                                 : programs.PrepareStages(storage, inputs, {stages[cursor]});
                batchEnd = begin + 1;
            }
        }
        if (batch) {
            batchBegin = begin;
            batchKind = commands ? PacketKind::Commands : PacketKind::Stages;
        }
        return true; // A selected packet or the retained descending plan advanced.
    }

    bool Prepare()
    {
        if (!batch) {
            if (workContinuation) {
                if (!receipts.empty())
                    return false;
                const auto kind = continuationKind;
                auto continuation = std::move(workContinuation);
                workContinuation = {};
                SelectControlWork(continuation(), kind);
                if (!batch) {
                    if (kind == PacketKind::FinalWork)
                        finalAccepted = true;
                    else
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
                    if (!batch) {
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
                    if (batch)
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
        receipts.emplace_back(
            Receipt{packet->GetCompletion(), std::move(key), kind, prefix ? batchEnd - batchBegin : 0,
                    work.Opportunity(), work.RecordAcceptedResource(FrameWorkResource::Gpu, prediction),
                    bytes ? work.RecordAcceptedResource(FrameWorkResource::Transfer, bytes) : 0, bytes, started});
        if (kind == PacketKind::Preservation) {
            preservation.reset();
        } else {
            batch.reset();
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
        auto key = Key(*packet);
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
    std::vector<JPH::GPU::Stage> stages;
    bool inputAccepted = false;
    std::shared_ptr<rhi::PreparedComputeBatch> batch, preservation, probeBatch;
    PacketKind batchKind = PacketKind::Inputs;
    size_t batchBegin = 0, batchEnd = 0, planLength = 0, probeLimit = 0, probeEnd = 0;
    uint64_t planWindow = 0, planTransferWindow = 0;
    bool submissionBusy = false;
    uint64_t busyOpportunity = 0, busyProgress = 0, busySubmissions = 0;
    std::deque<Receipt> receipts;
    std::map<CostKey, std::deque<uint64_t>> costs;
    size_t lastStageCount = 0;
    uint64_t lastStageEstimate = 0;
    size_t lastCommandCount = 0;
    uint64_t lastCommandEstimate = 0;

    friend class JoltGpuTick;
};

JoltGpuTick::JoltGpuTick(FrameWorkOwner &work, JoltGpuStorage &storage, JoltGpuPrograms &programs)
    : m_impl(std::make_unique<Impl>(work, storage, programs))
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
