#include "PhysicsParticipant.h"
#include "../Component.h"
#include "../GameObject.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>

namespace infernux
{
namespace
{
const char *StageName(JoltGpuParticipantStage stage)
{
    switch (stage) {
    case JoltGpuParticipantStage::Predict:
        return "predict";
    case JoltGpuParticipantStage::Prepare:
        return "prepare";
    case JoltGpuParticipantStage::Exchange:
        return "exchange";
    case JoltGpuParticipantStage::Accept:
        return "accept";
    case JoltGpuParticipantStage::Initialize:
        return "initialize";
    }
    throw std::logic_error("Unknown participant stage");
}
std::vector<rhi::ComputeWorkInput> Bindings(const std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> &values)
{
    std::vector<rhi::ComputeWorkInput> result;
    for (const auto &[name, buffer] : values)
        result.push_back({name, buffer});
    return result;
}
template <class Value> std::vector<uint8_t> Bytes(const Value &value)
{
    std::vector<uint8_t> result(sizeof(Value));
    std::memcpy(result.data(), &value, sizeof(Value));
    return result;
}
JoltGpuControlWork StageChain(std::shared_ptr<const std::vector<PhysicsParticipant::Prefix>> owners, size_t cursor,
                              JoltGpuParticipantStage stage, JPH::GPU::StepParams params, float fixedDeltaTime,
                              uint32_t substep, uint32_t iteration, std::function<void()> done)
{
    if (cursor == owners->size()) {
        done();
        return {};
    }
    auto work =
        owners->at(cursor).owner->PrepareStage(stage, params, fixedDeltaTime, substep, iteration, owners->at(cursor));
    auto next = [owners, cursor, stage, params, fixedDeltaTime, substep, iteration, done = std::move(done)] {
        if (stage == JoltGpuParticipantStage::Predict && substep == 0)
            owners->at(cursor).owner->AcceptControls(owners->at(cursor).controls);
        return StageChain(owners, cursor + 1, stage, params, fixedDeltaTime, substep, iteration, done);
    };
    if (!work.packet && !work.continueWork)
        return {nullptr, std::move(next), JoltGpuControlWork::ResumeAfter::HostWork};
    return ContinueJoltGpuControlWork(std::move(work), std::move(next));
}
} // namespace

PhysicsParticipant::PhysicsParticipant(const Component &component, Recipes recipes, Controls controls, uint32_t count)
    : m_component(&component), m_recipes(std::move(recipes)), m_controls(std::move(controls)), m_count(count),
      m_initialCount(count)
{
    static std::atomic<uint64_t> nextGeneration{1};
    m_generation = nextGeneration.fetch_add(1, std::memory_order_relaxed);
    const auto initialize = m_recipes.find("initialize");
    if (initialize == m_recipes.end() || !initialize->second)
        throw std::invalid_argument("Participant needs initialize Work");
    const auto &host = initialize->second->GetHost();
    for (const auto &[stage, recipe] : m_recipes) {
        if (stage != "initialize" && stage != "predict" && stage != "prepare" && stage != "exchange" &&
            stage != "accept")
            throw std::invalid_argument("Unknown participant Work stage");
        if (!recipe || !recipe->GetHost().SharesServicesWith(host))
            throw std::invalid_argument("Participant Work must share one compute host");
    }
    for (auto &[name, value] : m_controls) {
        if (name == "step" || name == "indices" || !value.prototype ||
            !value.prototype->GetHost().SharesServicesWith(host) ||
            value.initial.size() != value.prototype->GetDesc().GetByteSize())
            throw std::invalid_argument("Participant controls need complete matching author values");
        auto payload = std::make_shared<const std::vector<uint8_t>>(std::move(value.initial));
        m_initialControls.emplace(name, payload);
        m_pendingControls.emplace(name, std::move(payload));
    }
    bool state = false;
    for (const auto &port : initialize->second->GetPorts())
        if (!port.input) {
            if (port.name == "step" || port.name == "indices" || m_controls.count(port.name))
                throw std::invalid_argument("Participant state and runtime input names must be distinct");
            state = true;
        }
    if (!state)
        throw std::invalid_argument("Participant initialize must declare persistent outputs");
    for (const auto &[name, recipe] : m_recipes)
        for (const auto &port : recipe->GetPorts()) {
            if (!port.input || port.name == "step" || port.name == "indices" || m_controls.count(port.name))
                continue;
            const auto &outputs = initialize->second->GetPorts();
            const auto found = std::find_if(outputs.begin(), outputs.end(), [&](const auto &output) {
                return !output.input && output.name == port.name;
            });
            if (name == "initialize" || found == outputs.end() || port.range.offset != found->range.offset ||
                port.range.byteSize > found->range.byteSize)
                throw std::invalid_argument("Participant state input exceeds its named initialized output");
        }
}

bool PhysicsParticipantOutput::IsDone() const
{
    const auto participant = owner.lock();
    return ready || cancelled || !error.empty() || !participant || participant->IsRetired();
}
std::string PhysicsParticipantOutput::Error() const
{
    if (!error.empty())
        return error;
    if (cancelled)
        return "Participant output was cancelled";
    const auto participant = owner.lock();
    return !ready && (!participant || participant->IsRetired()) ? "Participant owner is retired" : "";
}
void PhysicsParticipantOutput::Cancel() noexcept
{
    cancelled = true;
    value.reset();
    storage.reset();
}
void PhysicsParticipantOutput::Fail(std::string reason)
{
    error = std::move(reason);
    value.reset();
    storage.reset();
}
void PhysicsParticipant::Retire() noexcept
{
    m_retired = true;
    for (const auto &entry : m_outputs)
        if (const auto capture = entry.lock(); capture && !capture->ready)
            capture->Cancel();
    m_outputs.clear();
}
std::shared_ptr<PhysicsParticipantOutput> PhysicsParticipant::CaptureOutput(const std::string &name, uint64_t offset,
                                                                            std::optional<uint64_t> count)
{
    if (IsRetired())
        throw std::logic_error("Participant owner is retired");
    const auto &recipe = *m_recipes.at("initialize");
    const auto port = std::find_if(recipe.GetPorts().begin(), recipe.GetPorts().end(),
                                   [&](const auto &value) { return !value.input && value.name == name; });
    if (port == recipe.GetPorts().end())
        throw std::invalid_argument("Participant output must name an initialized state port");
    const auto &allocation = recipe.GetSlots().at(port->range.slot).allocation;
    const auto stride = allocation.GetElementStride();
    const uint64_t size = port->range.byteSize / stride;
    if (allocation.lanes != 1 || port->range.offset % stride || port->range.byteSize % stride || offset > size ||
        (count && (!*count || *count > size - offset)) || (!count && offset == size))
        throw std::invalid_argument("Participant output range exceeds its scalar port");
    auto result = std::make_shared<PhysicsParticipantOutput>();
    result->owner = shared_from_this();
    result->name = name;
    result->token = m_nextOutput++;
    result->participantGeneration = m_generation;
    result->offset = offset;
    result->sourceByteOffset = port->range.offset + offset * stride;
    result->layout = {count.value_or(size - offset), allocation.scalarType, 1};
    const auto handle = m_component.GetComponent()->GetHandle();
    result->componentId = handle.id;
    result->componentGeneration = handle.generation;
    result->componentSceneId = handle.worldId;
    const auto *world = m_component.GetComponent()->GetGameObject()->GetScene()->GetRuntimeWorld();
    if (!world || world->handle->destroyRequested ||
        (world->handle->state != "preparing" && world->handle->state != "ready"))
        throw std::logic_error("Participant output requires a live RuntimeWorld");
    result->worldId = world->id;
    m_outputs.push_back(result);
    return result;
}
bool PhysicsParticipant::HasPendingOutputs()
{
    m_outputs.erase(std::remove_if(m_outputs.begin(), m_outputs.end(),
                                   [](const auto &entry) {
                                       const auto capture = entry.lock();
                                       return !capture || capture->IsDone();
                                   }),
                    m_outputs.end());
    return !m_outputs.empty();
}
JoltGpuControlWork PhysicsParticipant::PrepareOutput(const std::shared_ptr<PhysicsParticipantOutput> &capture,
                                                     uint64_t epoch, uint64_t step, bool isFixedStep)
{
    if (IsRetired() || capture->IsDone())
        return {};
    auto self = shared_from_this();
    auto &host = m_recipes.at("initialize")->GetHost();
    if (!capture->storage)
        capture->storage = rhi::ComputeBuffer::TryCreatePrivate(host, capture->layout);
    const std::weak_ptr<PhysicsParticipantOutput> observed = capture;
    if (!capture->storage)
        return {nullptr, [self, observed, epoch, step, isFixedStep] {
                    const auto capture = observed.lock();
                    return capture ? self->PrepareOutput(capture, epoch, step, isFixedStep) : JoltGpuControlWork{};
                }};
    const auto &source = m_state.at(capture->name);
    capture->sourceWrite = source->GetLastWriteSubmission();
    capture->simulationEpoch = epoch;
    capture->fixedStepIndex = step;
    capture->isFixedStep = isFixedStep;
    auto packet = rhi::PrepareComputeBatch(
        host, {}, {}, {}, {{source, capture->storage, {capture->sourceByteOffset, 0, capture->layout.GetByteSize()}}});
    return {packet, [self, observed, packet] {
                // This continuation runs after the original Tick's actual copy receipt.
                // Freezing only the independent target never protects the live state.
                const auto capture = observed.lock();
                if (capture && !self->IsRetired() && !capture->IsDone()) {
                    capture->copySubmission = packet->GetSubmission();
                    capture->value = capture->storage->Freeze();
                    capture->ready = true;
                    capture->storage.reset();
                }
                return JoltGpuControlWork{};
            }};
}
JoltGpuControlWork PhysicsParticipant::PrepareOutputs(uint64_t epoch, uint64_t step, bool isFixedStep)
{
    if (!IsReady())
        return {};
    for (const auto &entry : m_outputs)
        if (const auto capture = entry.lock(); capture && !capture->IsDone()) {
            auto self = shared_from_this();
            auto work = PrepareOutput(capture, epoch, step, isFixedStep);
            return ContinueJoltGpuControlWork(std::move(work), [self, epoch, step, isFixedStep] {
                return self->PrepareOutputs(epoch, step, isFixedStep);
            });
        }
    return {};
}

bool PhysicsParticipant::IsActive() const
{
    const auto *component = m_component.GetComponent();
    return IsReady() && component && component->HasAwake() && component->IsPreparationReady() &&
           component->IsEnabled() && component->GetGameObject() && component->GetGameObject()->IsActiveInHierarchy();
}
void PhysicsParticipant::SetControl(const std::string &name, std::vector<uint8_t> bytes)
{
    if (IsRetired())
        throw std::logic_error("Participant owner is retired");
    const auto found = m_controls.find(name);
    if (found == m_controls.end() || bytes.size() != found->second.prototype->GetDesc().GetByteSize())
        throw std::invalid_argument("Control author value must match its declared storage");
    // New owner even for equal bytes: a caller may be overwriting GPU changes.
    m_pendingControls[name] = std::make_shared<const std::vector<uint8_t>>(std::move(bytes));
}
void PhysicsParticipant::AcceptControls(const ControlPayloads &values)
{
    for (const auto &[name, payload] : values) {
        const auto pending = m_pendingControls.find(name);
        if (pending != m_pendingControls.end() && pending->second == payload)
            m_pendingControls.erase(pending);
    }
}
PhysicsParticipant::Prefix PhysicsParticipant::CapturePrefix()
{
    return {shared_from_this(), m_pendingControls, m_count};
}
void PhysicsParticipant::SetCount(uint32_t count)
{
    if (IsRetired())
        throw std::logic_error("Participant owner is retired");
    m_count = count;
}

std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>>
PhysicsParticipant::Inputs(const rhi::ComputeWorkRecipe &recipe, bool initialize) const
{
    std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> result;
    for (const auto &port : recipe.GetPorts()) {
        if (!port.input)
            continue;
        if (port.name == "step")
            result.emplace(port.name, m_step);
        else if (port.name == "indices")
            result.emplace(port.name, m_indices);
        else if (const auto control = m_controlStorage.find(port.name); control != m_controlStorage.end())
            result.emplace(port.name, control->second);
        else if (!initialize && m_state.count(port.name))
            result.emplace(port.name, m_state.at(port.name));
        else
            throw std::invalid_argument("Participant Work input must name initialized state, step or control");
    }
    if (initialize)
        for (const auto &[name, storage] : m_state)
            result.emplace(name, storage);
    return result;
}
PhysicsParticipant::BoundStage::BoundStage(std::shared_ptr<rhi::ComputeWorkInstance> instance,
                                           const rhi::ComputeBuffer *step)
    : work(std::move(instance))
{
    const auto &recipe = work->GetRecipe();
    for (const auto &port : recipe.GetPorts()) {
        usesStep |= port.input && port.name == "step";
        usesIndices |= port.input && port.name == "indices";
    }
    if (step)
        for (const auto &dispatch : recipe.GetDispatches())
            for (const auto &binding : dispatch.bindings)
                if (binding.access != rhi::ComputeBufferAccess::Read &&
                    work->GetBuffers()[binding.range.slot].get() == step) {
                    writesStep = true;
                    return;
                }
}
bool PhysicsParticipant::BindInstances()
{
    auto &host = m_recipes.at("initialize")->GetHost();
    for (const auto &[name, control] : m_controls)
        if (!m_controlStorage.count(name)) {
            auto buffer = rhi::ComputeBuffer::TryCreatePrivate(host, control.prototype->GetDesc());
            if (!buffer)
                return false;
            m_controlStorage.emplace(name, std::move(buffer));
        }
    for (const auto &[name, recipe] : m_recipes)
        for (const auto &port : recipe->GetPorts())
            if (port.input && (port.name == "step" || port.name == "indices")) {
                const bool indices = port.name == "indices";
                const auto &layout = recipe->GetSlots().at(port.range.slot).allocation;
                const uint64_t bytes = indices ? sizeof(PhysicsParticipantIndices) : sizeof(PhysicsParticipantStep);
                if (layout.GetByteSize() != bytes ||
                    layout.scalarType != (indices ? rhi::ComputeScalarType::UInt32 : rhi::ComputeScalarType::Float32) ||
                    layout.lanes != 1 || port.range.offset || port.range.byteSize != bytes)
                    throw std::invalid_argument("Participant runtime ports require step f32[2] / indices u32[6]");
                auto &storage = indices ? m_indices : m_step;
                if (!storage) {
                    storage = rhi::ComputeBuffer::TryCreatePrivate(host, layout);
                    if (!storage)
                        return false;
                }
            }
    const auto &initialize = *m_recipes.at("initialize");
    for (const auto &port : initialize.GetPorts()) {
        if (port.input)
            continue;
        auto &storage = m_stateSlots[port.range.slot];
        if (!storage) {
            // These are Compute FIFO writers. Explicit frozen published values
            // still protect their accepted allocation through the common API.
            storage = rhi::ComputeBuffer::TryCreatePrivate(host, initialize.GetSlots().at(port.range.slot).allocation);
            if (!storage)
                return false;
        }
        m_state.emplace(port.name, storage);
    }
    if (!m_instances.count("initialize")) {
        auto instance = m_recipes.at("initialize")->Instantiate(Bindings(Inputs(*m_recipes.at("initialize"), true)));
        if (!instance)
            return false;
        for (const auto &port : instance->GetRecipe().GetPorts())
            if (!port.input)
                m_state.emplace(port.name, instance->GetBuffers().at(port.range.slot));
        m_instances.try_emplace("initialize", std::move(instance), m_step.get());
    }
    for (const auto &[name, recipe] : m_recipes) {
        if (m_instances.count(name))
            continue;
        auto instance = recipe->Instantiate(Bindings(Inputs(*recipe, false)));
        if (!instance)
            return false;
        m_instances.try_emplace(name, std::move(instance), m_step.get());
    }
    return true;
}
JoltGpuControlWork PhysicsParticipant::PrepareInitialize()
{
    auto self = shared_from_this();
    if (IsRetired() || m_ready)
        return {};
    if (!BindInstances())
        return {nullptr, [self] { return self->PrepareInitialize(); }};
    std::vector<rhi::ComputeBufferUpdate> updates;
    for (const auto &[name, payload] : m_initialControls)
        updates.push_back({m_controlStorage.at(name), 0, *payload});
    const auto &initialize = m_instances.at("initialize");
    if (initialize.usesStep)
        updates.push_back({m_step, 0, Bytes(PhysicsParticipantStep{})});
    if (initialize.usesIndices) {
        const PhysicsParticipantIndices values{
            0u, 0u, 0u, 0u, m_initialCount, uint32_t(JoltGpuParticipantStage::Initialize)};
        updates.push_back({m_indices, 0, Bytes(values)});
    }
    JoltGpuControlWork work{initialize.work->PrepareBatch(std::move(updates)), [self] {
                                self->AcceptControls(self->m_initialControls);
                                self->m_initialControls.clear();
                                if (!self->IsRetired())
                                    self->m_ready = true;
                                return JoltGpuControlWork{};
                            }}; // Completion boundary: allocation or accepted dispatch is not readiness.
    return work;
}
JoltGpuControlWork PhysicsParticipant::PrepareStage(JoltGpuParticipantStage stage, const JPH::GPU::StepParams &params,
                                                    float fixedDeltaTime, uint32_t substep, uint32_t iteration,
                                                    const Prefix &prefix)
{
    const auto name = StageName(stage);
    const auto found = m_instances.find(name);
    const auto *bound = found != m_instances.end() ? &found->second : nullptr;
    const PhysicsParticipantStep timing{params.gravityDt.w, fixedDeltaTime};
    const bool uploadStep = bound && bound->usesStep &&
                            (!m_uploadedStep || m_uploadedStep->substepDeltaTime != timing.substepDeltaTime ||
                             m_uploadedStep->fixedDeltaTime != timing.fixedDeltaTime);
    std::vector<rhi::ComputeBufferUpdate> updates;
    if (uploadStep)
        updates.push_back({m_step, 0, Bytes(timing)});
    if (bound && bound->usesIndices) {
        const PhysicsParticipantIndices values{
            substep,         params.solve.y, stage == JoltGpuParticipantStage::Exchange ? iteration : 0u,
            params.counts.w, prefix.count,   uint32_t(stage)};
        updates.push_back({m_indices, 0, Bytes(values)});
    }
    if (stage == JoltGpuParticipantStage::Predict && substep == 0)
        for (const auto &[control, value] : prefix.controls)
            updates.push_back({m_controlStorage.at(control), 0, *value});
    if (!bound && updates.empty())
        return {};
    auto packet = bound ? bound->work->PrepareBatch(std::move(updates))
                        : rhi::PrepareComputeBatch(m_recipes.at("initialize")->GetHost(), std::move(updates), {});
    JoltGpuControlWork work{std::move(packet), {}, JoltGpuControlWork::ResumeAfter::Acceptance};
    if (bound && (uploadStep || bound->writesStep)) {
        // The resident value changes only when the packet enters the device queue.
        // A writable project binding can replace it before the next stage.
        const auto resident = bound->writesStep ? std::nullopt : std::optional(timing);
        work.continueWork = [self = shared_from_this(), resident] {
            self->m_uploadedStep = resident;
            return JoltGpuControlWork{};
        };
    }
    return work;
}

decltype(JoltGpuTickControl::prepareParticipant)
CapturePhysicsParticipants(const std::vector<std::weak_ptr<PhysicsParticipant>> &participants)
{
    auto owners = std::make_shared<std::vector<PhysicsParticipant::Prefix>>();
    for (const auto &entry : participants)
        if (const auto participant = entry.lock(); participant && participant->IsActive())
            owners->push_back(participant->CapturePrefix());
    if (owners->empty())
        return {};
    return [owners](JoltGpuParticipantStage stage, const JPH::GPU::StepParams &params, float fixedDeltaTime,
                    uint32_t substep, uint32_t iteration, std::function<void()> done) {
        return StageChain(owners, 0, stage, params, fixedDeltaTime, substep, iteration, std::move(done));
    };
}
} // namespace infernux
