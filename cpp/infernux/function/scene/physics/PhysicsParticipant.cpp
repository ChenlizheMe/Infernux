#include "PhysicsParticipant.h"
#include "../Component.h"
#include "../GameObject.h"
#include "../Rigidbody.h"
#include "../RuntimeWorld.h"
#include "../Scene.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <core/math/Intervals.h>
#include <cstring>
#include <set>
#include <stdexcept>

namespace infernux
{
namespace
{
const RuntimeWorld *WorldOf(const Component *component)
{
    const auto *owner = component ? component->GetGameObject() : nullptr;
    const auto *scene = owner ? owner->GetScene() : nullptr;
    return scene ? scene->GetRuntimeWorld() : nullptr;
}

const char *StageName(PhysicsParticipantStage stage)
{
    switch (stage) {
    case PhysicsParticipantStage::Predict:
        return "predict";
    case PhysicsParticipantStage::Prepare:
        return "prepare";
    case PhysicsParticipantStage::Exchange:
        return "exchange";
    case PhysicsParticipantStage::Accept:
        return "accept";
    case PhysicsParticipantStage::Initialize:
        return "initialize";
    }
    throw std::logic_error("Unknown participant stage");
}
PhysicsParticipantIndices StageIndices(PhysicsParticipantStage stage, const PhysicsParticipantStageInputs &inputs,
                                       uint32_t count)
{
    return {inputs.substep,    inputs.substeps, stage == PhysicsParticipantStage::Exchange ? inputs.iteration : 0u,
            inputs.iterations, count,           static_cast<uint32_t>(stage)};
}

std::vector<rhi::ComputeWorkInput> Bindings(const std::map<std::string, std::shared_ptr<rhi::ComputeBuffer>> &values)
{
    std::vector<rhi::ComputeWorkInput> result;
    for (const auto &[name, buffer] : values)
        result.push_back({name, buffer});
    return result;
}
const rhi::ComputeWorkPort *FindPort(const rhi::ComputeWorkRecipe &recipe, const std::string &name, bool input)
{
    for (const auto &port : recipe.GetPorts())
        if (port.input == input && port.name == name)
            return &port;
    return nullptr;
}
template <class Value> std::vector<uint8_t> Bytes(const Value &value)
{
    std::vector<uint8_t> result(sizeof(Value));
    std::memcpy(result.data(), &value, sizeof(Value));
    return result;
}
} // namespace

PhysicsParticipant::PhysicsParticipant(const Component &component, Recipes recipes, Controls controls, uint32_t count,
                                       const std::vector<Rigidbody *> &rigidbodies,
                                       const ExchangeStates &exchangeStates)
    : m_component(&component), m_recipes(std::move(recipes)), m_controls(std::move(controls)), m_count(count),
      m_initialCount(count)
{
    static std::atomic<uint64_t> nextGeneration{1};
    m_generation = nextGeneration.fetch_add(1, std::memory_order_relaxed);
    const auto *world = WorldOf(&component);
    std::set<const Rigidbody *> unique;
    for (const auto *body : rigidbodies) {
        if (!world || WorldOf(body) != world || !unique.insert(body).second)
            throw std::invalid_argument(
                "Participant rigid bindings require distinct Rigidbody components in the same World");
        m_rigidbodies.emplace_back(body);
    }
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
    for (const auto stage : {PhysicsParticipantStage::Predict, PhysicsParticipantStage::Prepare,
                             PhysicsParticipantStage::Exchange, PhysicsParticipantStage::Accept})
        if (m_recipes.count(StageName(stage)))
            m_stages |= PhysicsParticipantStageBit(stage);
    for (auto &[name, value] : m_controls) {
        if (name == "step" || name == "indices" || !value.prototype ||
            !value.prototype->GetHost().SharesServicesWith(host) ||
            value.initial.size() != value.prototype->GetDesc().GetByteSize())
            throw std::invalid_argument("Participant controls need complete matching author values");
        auto payload = std::make_shared<const std::vector<uint8_t>>(std::move(value.initial));
        m_initialControls.emplace(name, payload);
        m_pendingControls.emplace(name, std::move(payload));
    }
    if (!m_rigidbodies.empty())
        ValidateRigidPorts();
    bool state = false;
    for (const auto &port : initialize->second->GetPorts())
        if (!port.input) {
            if (port.name == "step" || port.name == "indices" || m_controls.count(port.name))
                throw std::invalid_argument("Participant state and runtime input names must be distinct");
            state = true;
        }
    if (!state)
        throw std::invalid_argument("Participant initialize must declare persistent outputs");
    DeclareExchangeStates(exchangeStates);
    for (const auto &[name, recipe] : m_recipes)
        for (const auto &port : recipe->GetPorts()) {
            if (!port.input || port.name == "step" || port.name == "indices" || m_controls.count(port.name) ||
                (name == "exchange" &&
                 (m_exchangeStates.count(port.name) || (!m_rigidbodies.empty() && port.name == "rigid_motion"))))
                continue;
            const auto *output = FindPort(*initialize->second, port.name, false);
            if (name == "initialize" || !output || port.range.offset != output->range.offset ||
                port.range.byteSize > output->range.byteSize)
                throw std::invalid_argument("Participant state input exceeds its named initialized output");
        }
}

void PhysicsParticipant::DeclareExchangeStates(const ExchangeStates &states)
{
    if (states.empty())
        return;
    const auto exchange = m_recipes.find("exchange");
    if (exchange == m_recipes.end())
        throw std::invalid_argument("Shared participant state requires Exchange Work");
    const auto &recipe = *exchange->second;
    const auto *world = WorldOf(m_component.GetComponent());
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> owners;
    std::vector<uint64_t> identities(recipe.GetSlots().size());
    std::map<uint32_t, intervals::Set> grants;
    for (const auto &[name, source] : states) {
        const auto &[peer, output] = source;
        if (!peer || peer->IsRetired() || !world || WorldOf(peer->m_component.GetComponent()) != world)
            throw std::invalid_argument("Shared Exchange state requires a live participant in the same World");
        const auto &initialize = *peer->m_recipes.at("initialize");
        const auto *from = FindPort(initialize, output, false);
        const auto *to = FindPort(recipe, name, true);
        if (!from || !to || name == "step" || name == "indices" || m_controls.count(name) ||
            FindPort(*m_recipes.at("initialize"), name, false) || (!m_rigidbodies.empty() && name == "rigid_motion"))
            throw std::invalid_argument("Exchange must bind a distinct input to a named peer state");
        const auto &slot = recipe.GetSlots().at(to->range.slot);
        if (!slot.Accepts(initialize.GetSlots().at(from->range.slot).allocation) ||
            to->range.offset < from->range.offset ||
            to->range.offset + to->range.byteSize > from->range.offset + from->range.byteSize)
            throw std::invalid_argument("Exchange input must match its peer allocation and named state range");
        const auto key = std::make_pair(peer->m_generation, from->range.slot);
        const auto identity = owners.try_emplace(key, owners.size() + 1).first->second;
        auto &assigned = identities[to->range.slot];
        if (assigned && assigned != identity)
            throw std::invalid_argument("One Exchange slot cannot bind different participant allocations");
        assigned = identity;
        intervals::Add(grants[to->range.slot], from->range.offset, from->range.offset + from->range.byteSize);
        m_exchangeStates.emplace(name, ExchangeState{peer, output, {}});
    }
    for (const auto &port : recipe.GetPorts())
        if (port.input && identities[port.range.slot] && !states.count(port.name))
            throw std::invalid_argument("Local and peer Exchange inputs cannot share one allocation slot");
    recipe.ValidateStorageAliases(identities);
    const auto requireGranted = [&](const rhi::ComputeWorkRange &range) {
        const auto found = grants.find(range.slot);
        if (found != grants.end() && !intervals::Contains(found->second, range.offset, range.offset + range.byteSize))
            throw std::invalid_argument("Exchange kernel accesses outside its named peer states");
    };
    for (const auto &dispatch : recipe.GetDispatches()) {
        for (const auto &binding : dispatch.bindings)
            requireGranted(binding.range);
        if (dispatch.indirectArguments)
            requireGranted(*dispatch.indirectArguments);
    }
}

void PhysicsParticipant::ValidateRigidPorts()
{
    const auto exchange = m_recipes.find("exchange");
    if (exchange == m_recipes.end() || m_controls.count("rigid_motion") || m_controls.count("rigid_impulse_delta"))
        throw std::invalid_argument(
            "Participant rigid targets require Exchange with dedicated motion and impulse ports");
    const uint64_t targets = m_rigidbodies.size();
    const auto &recipe = *exchange->second;
    for (const auto &[stage, declared] : m_recipes)
        for (const auto &port : declared->GetPorts()) {
            const bool motion = port.name == "rigid_motion";
            const bool response = port.name == "rigid_impulse_delta";
            if (!motion && !response)
                continue;
            const auto &slot = declared->GetSlots().at(port.range.slot);
            const uint64_t elements = targets * (motion ? 16u : 6u);
            if (stage != "exchange" || port.input != motion || slot.external || port.range.offset ||
                slot.allocation.scalarType != rhi::ComputeScalarType::Float32 || slot.allocation.lanes != 1 ||
                slot.allocation.elementCount != elements || port.range.byteSize != elements * sizeof(float))
                throw std::invalid_argument("Participant rigid exchange needs exact private f32[16N] / f32[6N] ports");
            auto &selected = motion ? m_rigidMotionSlot : m_rigidImpulseSlot;
            if (selected != UINT32_MAX)
                throw std::invalid_argument("Participant rigid exchange port is declared more than once");
            selected = port.range.slot;
        }
    if (m_rigidMotionSlot == UINT32_MAX || m_rigidImpulseSlot == UINT32_MAX || m_rigidMotionSlot == m_rigidImpulseSlot)
        throw std::invalid_argument("Participant rigid exchange requires distinct motion and impulse storage");
    for (const auto &port : recipe.GetPorts())
        if ((port.range.slot == m_rigidMotionSlot && port.name != "rigid_motion") ||
            (port.range.slot == m_rigidImpulseSlot && port.name != "rigid_impulse_delta"))
            throw std::invalid_argument("Participant rigid ports cannot alias other Work ports");

    std::vector<std::pair<uint64_t, uint64_t>> written;
    for (const auto &dispatch : recipe.GetDispatches()) {
        for (const auto &binding : dispatch.bindings)
            if (binding.range.slot == m_rigidMotionSlot && binding.access != rhi::ComputeBufferAccess::Read)
                throw std::invalid_argument("Participant rigid motion is read only");
        if (dispatch.indirectArguments)
            continue;
        for (const auto &fact : dispatch.completeWrites)
            if (dispatch.bindings.at(fact.binding).range.slot == m_rigidImpulseSlot)
                written.emplace_back(fact.offset, fact.offset + fact.byteSize);
    }
    std::sort(written.begin(), written.end());
    uint64_t covered = 0;
    for (const auto &[first, end] : written) {
        if (first > covered)
            break;
        covered = std::max(covered, end);
    }
    if (covered != targets * 6u * sizeof(float))
        throw std::invalid_argument(
            "Participant Exchange must freshly write every rigid impulse row on each invocation");
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
bool PhysicsParticipant::PublishResidentOutputs(uint64_t epoch, uint64_t step, bool isFixedStep)
{
    bool needsCopies = false;
    for (const auto &entry : m_outputs) {
        const auto capture = entry.lock();
        if (!capture || capture->IsDone())
            continue;
        const auto &source = m_state.at(capture->name);
        const auto bytes = capture->layout.GetByteSize();
        if (capture->sourceByteOffset || bytes != source->GetByteSize()) {
            needsCopies = true;
            continue;
        }
        if (!source->HasInitializedRange(0, bytes)) {
            capture->Fail("Participant output lacks an accepted complete writer");
            continue;
        }
        // Pin this allocation. A later writer preserves it through the normal
        // ComputeBuffer write barrier, including a writer bound by another participant.
        capture->value = source->Freeze();
        capture->sourceWrite = capture->value->GetLastWriteSubmission();
        capture->simulationEpoch = epoch;
        capture->fixedStepIndex = step;
        capture->isFixedStep = isFixedStep;
        capture->ready = true;
    }
    return needsCopies;
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
void PhysicsParticipant::ReleasePeerExchange()
{
    m_stages &= ~PhysicsParticipantStageBit(PhysicsParticipantStage::Exchange);
    m_instances.erase("exchange");
    m_recipes.erase("exchange");
    m_exchangeStates.clear();
    m_rigidbodies.clear();
    m_rigidMotion.reset();
}
PhysicsParticipant::Prefix PhysicsParticipant::CapturePrefix()
{
    const bool active = IsActive();
    bool peersActive = true;
    for (const auto &[name, state] : m_exchangeStates) {
        const auto peer = state.owner.lock();
        if (!peer || peer->IsRetired()) {
            ReleasePeerExchange();
            break;
        }
        if (active && !peer->IsActive())
            peersActive = false;
    }
    if (!active)
        return {};
    Prefix prefix{m_pendingControls, m_count, m_stages, {}};
    if (!peersActive)
        prefix.stages &= ~PhysicsParticipantStageBit(PhysicsParticipantStage::Exchange);
    if (m_rigidbodies.empty() || !(prefix.stages & PhysicsParticipantStageBit(PhysicsParticipantStage::Exchange)))
        return prefix;
    const auto *world = WorldOf(m_component.GetComponent());
    prefix.rigidBodyIds.reserve(m_rigidbodies.size());
    for (const auto &reference : m_rigidbodies) {
        const auto *body = dynamic_cast<Rigidbody *>(reference.GetComponent());
        if (!body || WorldOf(body) != world || !body->IsEnabled() || !body->GetGameObject()->IsActiveInHierarchy() ||
            body->IsKinematic() || body->GetBodyId() == UINT32_MAX) {
            prefix.rigidBodyIds.clear();
            prefix.stages &= ~PhysicsParticipantStageBit(PhysicsParticipantStage::Exchange);
            return prefix;
        }
        prefix.rigidBodyIds.push_back(body->GetBodyId());
    }
    prefix.stages |= PhysicsParticipantStageBit(PhysicsParticipantStage::Prepare);
    return prefix;
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
        else if (!initialize && !m_rigidbodies.empty() && port.name == "rigid_motion")
            result.emplace(port.name, m_rigidMotion);
        else if (const auto control = m_controlStorage.find(port.name); control != m_controlStorage.end())
            result.emplace(port.name, control->second);
        else if (const auto peer = m_exchangeStates.find(port.name); peer != m_exchangeStates.end())
            result.emplace(port.name, peer->second.buffer);
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
    if (m_batchRuntime.empty())
        m_batchRuntime.resize(4);
    auto &host = m_recipes.at("initialize")->GetHost();
    for (auto &[name, state] : m_exchangeStates) {
        const auto peer = state.owner.lock();
        if (!peer || peer->IsRetired()) {
            ReleasePeerExchange();
            break;
        }
        // Bindings refer to earlier registrations. World preparation initializes
        // those owners before admitting this participant's initialization.
        if (!peer->IsReady())
            return false;
        state.buffer = peer->m_state.at(state.output);
    }
    if (!m_rigidbodies.empty() && !m_rigidMotion) {
        const auto &layout = m_recipes.at("exchange")->GetSlots().at(m_rigidMotionSlot).allocation;
        m_rigidMotion = rhi::ComputeBuffer::TryCreatePrivate(host, layout);
        if (!m_rigidMotion)
            return false;
    }
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
            0u, 0u, 0u, 0u, m_initialCount, uint32_t(PhysicsParticipantStage::Initialize)};
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
std::optional<PhysicsParticipant::PreparedStage>
PhysicsParticipant::PrepareGpuStage(PhysicsParticipantStage stage, const PhysicsParticipantStageInputs &inputs,
                                    const Prefix &prefix, std::optional<uint32_t> runtimeSlot)
{
    PreparedStage result;
    result.host = &m_recipes.at("initialize")->GetHost();
    const auto found = m_instances.find(StageName(stage));
    const auto *bound = found != m_instances.end() ? &found->second : nullptr;
    const auto &timing = inputs.timing;
    const bool uploadStep = bound && bound->usesStep &&
                            (!m_uploadedStep || m_uploadedStep->substepDeltaTime != timing.substepDeltaTime ||
                             m_uploadedStep->fixedDeltaTime != timing.fixedDeltaTime);
    if (bound)
        bound->work->AppendDispatches(result.dispatches);
    auto step = m_step, indices = m_indices;
    BatchRuntime runtime;
    if (runtimeSlot && bound) {
        const auto &recipe = bound->work->GetRecipe();
        for (const auto &port : recipe.GetPorts()) {
            if (!port.input || (port.name != "step" && port.name != "indices"))
                continue;
            // Splitting a writable or otherwise named runtime allocation would
            // change an observable state/input identity. Keep the ordinary path.
            for (const auto &other : recipe.GetPorts())
                if (&other != &port && other.range.slot == port.range.slot)
                    return std::nullopt;
            const auto &original = bound->work->GetBuffers().at(port.range.slot);
            for (const auto &dispatch : result.dispatches)
                for (size_t i = 0; i < dispatch.buffers.size(); ++i)
                    if (dispatch.buffers[i] == original && dispatch.bufferAccesses[i] != rhi::ComputeBufferAccess::Read)
                        return std::nullopt;
        }
        // New snapshots remain local until acceptance. A declined/contracted
        // optional packet must not retain newly allocated residency debt.
        if (*runtimeSlot < m_batchRuntime.size())
            runtime = m_batchRuntime[*runtimeSlot];
        for (const auto usesIndices : {false, true}) {
            if (!(usesIndices ? bound->usesIndices : bound->usesStep))
                continue;
            auto &storage = usesIndices ? runtime.indices : runtime.step;
            const auto &original = usesIndices ? m_indices : m_step;
            if (!storage)
                storage = rhi::ComputeBuffer::TryCreatePrivate(*result.host, original->GetDesc());
            if (!storage)
                return std::nullopt;
            for (auto &dispatch : result.dispatches) {
                for (auto &buffer : dispatch.buffers)
                    if (buffer == original)
                        buffer = storage;
                if (dispatch.indirectArguments && dispatch.indirectArguments->buffer == original)
                    dispatch.indirectArguments->buffer = storage;
            }
            (usesIndices ? indices : step) = storage;
        }
    }
    if (bound && bound->usesStep && (runtimeSlot || uploadStep))
        result.updates.push_back({step, 0, Bytes(timing)});
    if (bound && bound->usesIndices)
        result.updates.push_back({indices, 0, Bytes(StageIndices(stage, inputs, prefix.count))});
    if (stage == PhysicsParticipantStage::Predict && inputs.substep == 0)
        for (const auto &[control, value] : prefix.controls)
            result.updates.push_back({m_controlStorage.at(control), 0, *value});
    if (bound && (runtimeSlot || uploadStep || bound->writesStep)) {
        const auto resident = runtimeSlot || bound->writesStep ? std::nullopt : std::optional(timing);
        const bool changeStep = bound->usesStep && (runtimeSlot || uploadStep || bound->writesStep);
        result.accepted = [self = shared_from_this(), resident, changeStep, runtimeSlot, runtime] {
            if (runtimeSlot) {
                if (*runtimeSlot >= self->m_batchRuntime.size())
                    self->m_batchRuntime.resize(size_t{*runtimeSlot} + 1);
                self->m_batchRuntime[*runtimeSlot] = runtime;
            }
            if (changeStep)
                self->m_uploadedStep = resident;
        };
    }
    return result;
}

std::optional<PhysicsParticipant::DeviceRigidExchange>
PhysicsParticipant::PrepareDeviceRigidExchange(const PhysicsParticipantStageInputs &inputs, const Prefix &prefix,
                                               uint32_t runtimeSlot)
{
    constexpr auto stage = PhysicsParticipantStage::Exchange;
    if (prefix.rigidBodyIds.empty() || !(prefix.stages & PhysicsParticipantStageBit(stage)))
        return std::nullopt;
    auto work = PrepareGpuStage(stage, inputs, prefix, runtimeSlot);
    if (!work)
        return std::nullopt;
    return DeviceRigidExchange{std::move(*work), m_rigidMotion,
                               m_instances.at("exchange").work->GetBuffers().at(m_rigidImpulseSlot)};
}

} // namespace infernux
