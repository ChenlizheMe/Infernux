#include "ParticleGpuSystemManager.h"

#include "ParticleGpuCollisionScene.h"
#include "ParticleGpuDrawRegistry.h"
#include "ParticleGpuMeshRenderer.h"

#include <core/log/InxLog.h>
#include <core/threading/FrameWorkOwner.h>
#include <core/threading/JobSystem.h>
#include <function/renderer/VulkanFrameSubmission.h>
#include <function/renderer/rhi/GpuRetirementQueue.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/vk/RenderGraph.h>
#include <function/renderer/vk/VkDeviceContext.h>
#include <function/renderer/vk/VkPipelineManager.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/renderer/vk/VulkanRhiDevice.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace infernux::particle
{

namespace
{

void SetError(std::string *error, std::string message)
{
    if (error)
        *error = std::move(message);
}

bool IsSpirv(const std::vector<uint32_t> &words)
{
    return words.size() >= 5 && words.front() == 0x07230203u;
}

struct StateGatherConstants
{
    uint32_t strideWords;
    uint32_t fieldWord;
    uint32_t fieldWords;
    uint32_t aliveSlot;
    uint32_t firstRow;
    uint32_t rowCount;
    uint32_t outputWord;
    uint32_t identityWord;
};

class StateGatherProgram
{
  public:
    static constexpr uint32_t WorkgroupSize = 64;
    static constexpr uint32_t IdentityWords = 3;
    static constexpr uint32_t SkipIdentities = UINT32_MAX;

    StateGatherProgram(rhi::Device &device, ShaderBytecode bytecode) : m_device(device)
    {
        rhi::BindingLayoutDesc layout;
        for (uint32_t binding = 0; binding < 5; ++binding)
            layout.entries[layout.entryCount++] = {binding, rhi::BindingType::StorageBuffer, rhi::ShaderStage::Compute,
                                                   1};
        m_layout = device.CreateBindingLayout(layout);
        const auto shader =
            device.CreateShaderModule(rhi::ShaderModuleDesc::FromSpirV(bytecode.words, bytecode.wordCount));
        if (m_layout.IsValid() && shader.IsValid()) {
            rhi::ComputePipelineDesc pipeline;
            pipeline.computeShader = shader;
            pipeline.bindingLayouts[0] = m_layout;
            pipeline.bindingLayoutCount = 1;
            pipeline.pushConstantBytes = sizeof(StateGatherConstants);
            m_pipeline = device.CreateComputePipeline(pipeline);
        }
        device.Release(shader);
    }

    ~StateGatherProgram()
    {
        m_device.Release(m_pipeline);
        m_device.Release(m_layout);
    }

    [[nodiscard]] bool IsValid() const noexcept
    {
        return m_pipeline.IsValid();
    }

    [[nodiscard]] rhi::BindGroupHandle Bind(const ParticleGpuRuntime &runtime, rhi::BufferHandle output,
                                            rhi::BufferHandle identities) const
    {
        rhi::BindGroupDesc group;
        group.layout = m_layout;
        const std::array buffers = {runtime.StateBuffer(), runtime.AliveIndexBuffer(runtime.AliveReadSlot()),
                                    runtime.AliveControlBuffer(), output, identities};
        for (uint32_t binding = 0; binding < buffers.size(); ++binding)
            group.buffers[group.bufferCount++] = {binding, rhi::BindingType::StorageBuffer, buffers[binding], 0, 0};
        return m_device.CreateBindGroup(group);
    }

    void Record(const rhi::ComputeCommandEncoder &encoder, rhi::BindGroupHandle group,
                const StateGatherConstants &constants) const
    {
        encoder.BindPipeline(m_pipeline);
        encoder.BindGroup(m_pipeline, 0, group);
        encoder.PushConstants(m_pipeline, sizeof(constants), &constants);
        encoder.Dispatch(1 + (constants.rowCount - 1) / WorkgroupSize);
    }

  private:
    rhi::Device &m_device;
    rhi::BindingLayoutHandle m_layout;
    rhi::ComputePipelineHandle m_pipeline;
};

struct StateGatherBindings
{
    explicit StateGatherBindings(rhi::Device &device) : device(device)
    {
    }
    ~StateGatherBindings()
    {
        for (const auto group : groups)
            device.Release(group);
    }

    rhi::Device &device;
    std::vector<rhi::BindGroupHandle> groups;
};

bool IsFinite(const GpuParticleTransforms &transforms) noexcept
{
    const auto finite = [](const auto &matrix) {
        return std::all_of(matrix.begin(), matrix.end(), [](float value) { return std::isfinite(value); });
    };
    return finite(transforms.emitterToWorld) && finite(transforms.worldToEmitter) &&
           finite(transforms.simulationToWorld) && finite(transforms.worldToSimulation);
}

struct alignas(16) PackedParticleMeshVertex
{
    std::array<float, 4> position{};
    std::array<float, 4> normal{};
    std::array<float, 4> tangent{};
    std::array<float, 4> color{};
    std::array<float, 4> uv{};
};

struct alignas(16) PackedParticleMeshPrimitive
{
    uint32_t first = 0;
    uint32_t second = 0;
    uint32_t thirdOrCdf = 0;
    uint32_t cdfOrPadding = 0;
};

struct alignas(16) PackedParticleSkinInfluence
{
    std::array<uint32_t, 4> bones{};
    std::array<float, 4> weights{};
};

uint32_t FloatBits(float value) noexcept
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

} // namespace

std::string_view GpuParticleStateGatherShaderSource() noexcept
{
    return R"glsl(#version 450
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) readonly buffer State { uint state[]; };
layout(std430, set = 0, binding = 1) readonly buffer LiveSlots { uint live_slots[]; };
layout(std430, set = 0, binding = 2) readonly buffer LiveControl { uint live_counts[2]; };
layout(std430, set = 0, binding = 3) writeonly buffer Samples { uint samples[]; };
layout(std430, set = 0, binding = 4) writeonly buffer Identities { uint identities[]; };
layout(push_constant) uniform Gather {
    uint stride_words;
    uint field_word;
    uint field_words;
    uint alive_slot;
    uint first_row;
    uint row_count;
    uint output_word;
    uint identity_word;
} pc;

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.row_count)
        return;

    uint source_row = pc.first_row + row;
    uint slot = 0xffffffffu;
    uint flags = 0u;
    uint source = 0u;
    if (source_row < live_counts[pc.alive_slot]) {
        slot = live_slots[source_row];
        source = slot * pc.stride_words;
        flags = state[source];
    }
    bool live = (flags & 1u) != 0u;
    if (pc.identity_word != 0xffffffffu) {
        uint identity = pc.identity_word + row * 3u;
        identities[identity] = live ? slot : 0xffffffffu;
        identities[identity + 1u] = live ? state[source + 1u] : 0u;
        identities[identity + 2u] = live ? flags : 0u;
    }
    uint dest = pc.output_word + row * pc.field_words;
    for (uint field = 0u; field < pc.field_words; ++field)
        samples[dest + field] = live ? state[source + pc.field_word + field] : 0u;
}
)glsl";
}

struct ParticleGpuSystemManager::Impl
{
    // Prewarm and seek may enqueue thousands of fixed simulation steps. Recording
    // several compiled graph executions into one command buffer erases the
    // sim/export ownership boundary and makes the following frame-slot fence
    // indistinguishable from a GPU hang after Play -> Stop -> Play. Record one
    // compiled execution per submission and drain the remaining exact steps
    // over subsequent frames.
    static constexpr uint32_t MaxQueuedSimulationStepsPerSubmission = 1;

    struct DiagnosticState
    {
        mutable std::mutex mutex;
        std::unordered_map<uint64_t, GpuParticleDiagnosticSnapshot> snapshots;
    };

    struct PendingDiagnostic
    {
        uint64_t requestId = 0;
        uint64_t graphInstanceId = 0;
        uint32_t remainingSampleFrames = 0;
        uint32_t stateSampleCount = 0;
        bool resetPending = true;
    };

    struct MeshResources
    {
        std::shared_ptr<rhi::BufferResource> vertices;
        std::shared_ptr<rhi::BufferResource> indices;
        std::shared_ptr<rhi::BufferResource> samplingTriangles;
        std::shared_ptr<rhi::BufferResource> skinInfluences;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t samplingTriangleCount = 0;
        uint32_t samplingEdgeCount = 0;
    };

    struct MeshUpload
    {
        enum Part : size_t
        {
            Vertices,
            Indices,
            Sampling,
            Skin,
            PartCount
        };

        struct Source
        {
            std::shared_ptr<const MeshGeometry> geometry;
            std::shared_ptr<const InxSkinnedMesh> skinned;
            std::vector<PackedParticleMeshVertex> vertices;
            std::vector<PackedParticleMeshPrimitive> samplingPrimitives;
            std::vector<PackedParticleSkinInfluence> skinInfluences;
            std::shared_ptr<FrameCpuStorageClaim> storage;
            std::string failure;
            uint64_t executionNanoseconds = 0;
            uint32_t samplingTriangleCount = 0;
            uint32_t samplingEdgeCount = 0;

            [[nodiscard]] uint64_t WorkspaceBytes() const
            {
                const uint64_t indices = geometry->indices.size();
                return uint64_t{geometry->vertices.size()} * sizeof(PackedParticleMeshVertex) +
                       (indices + indices / 3) * sizeof(PackedParticleMeshPrimitive) +
                       indices * sizeof(std::pair<uint32_t, uint32_t>) +
                       (skinned ? uint64_t{skinned->influences.size()} * sizeof(PackedParticleSkinInfluence) : 0);
            }
        };

        std::weak_ptr<InxMesh> owner;
        uint64_t generation = 0;
        uint64_t token = 0;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        uint32_t samplingTriangleCount = 0;
        uint32_t samplingEdgeCount = 0;
        std::shared_ptr<Source> source;
        JobHandle packJob;
        uint64_t packFrame = 0;
        uint64_t packGrant = 0;
        std::array<std::shared_ptr<vk::BufferUploadTicket>, PartCount> tickets;
        std::shared_ptr<MeshResources> resources;
        std::string failure;

        [[nodiscard]] size_t NextPart() const
        {
            for (size_t part = 0; part < tickets.size(); ++part) {
                const auto &ticket = tickets[part];
                if (ticket && ticket->GetSubmittedBytes() < ticket->GetSize())
                    return part;
            }
            return PartCount;
        }

        [[nodiscard]] bool IsCurrent() const
        {
            const auto mesh = owner.lock();
            return mesh && mesh->GetGeneration() == generation;
        }

        void Fail(std::string error)
        {
            source.reset();
            tickets = {};
            failure = std::move(error);
        }
    };

    struct Emitter
    {
        struct Output
        {
            uint64_t id = 0;
            std::string stableId;
            GpuParticleOutputType type = GpuParticleOutputType::Sprite;
            std::shared_ptr<InxMesh> mesh;
            std::shared_ptr<InxMaterial> material;
            std::shared_ptr<const ShaderProgramArtifact> shaderProgram;
            GpuBillboardMaterialState fallbackMaterial;
            ParticleOutputSemantics semantics;
            ParticleRibbonUvMode ribbonUvMode = ParticleRibbonUvMode::Stretch;
            float ribbonUvScale = 1.0f;
            uint32_t flipbookColumns = 1;
            uint32_t flipbookRows = 1;
            std::shared_ptr<ParticleGpuOutputRenderer> renderer;
        };

        uint64_t id = 0;
        uint64_t graphInstanceId = 0;
        uint64_t artifactRevision = 0;
        std::string stableId;
        bool statePreservedOnPublish = false;
        std::unique_ptr<ParticleGpuRuntime> runtime;
        std::unique_ptr<ParticleGpuBounds> bounds;
        std::shared_ptr<ParticleGpuRibbonTopology> ribbonTopology;
        std::shared_ptr<ParticleGpuMigrator> migration;
        std::shared_ptr<Emitter> migrationSource;
        GpuParticleEmitterProgram sourceProgram;
        std::vector<std::shared_ptr<InxTexture>> vectorFields;
        std::vector<uint64_t> vectorFieldGenerations;
        std::vector<uint64_t> observedVectorFieldGenerations;
        std::vector<std::shared_ptr<InxMesh>> meshes;
        std::vector<uint64_t> observedMeshGenerations;
        std::vector<uint32_t> billboardVertexShader;
        std::vector<uint32_t> billboardPickingFragmentShader;
        std::vector<uint32_t> billboardMotionVertexShader;
        std::vector<uint32_t> billboardMotionFragmentShader;
        std::vector<uint32_t> meshVertexShader;
        std::vector<uint32_t> meshShadowFragmentShader;
        std::vector<uint32_t> meshPickingFragmentShader;
        std::vector<uint32_t> meshMotionVertexShader;
        std::vector<uint32_t> meshMotionFragmentShader;
        std::vector<Output> outputs;
        bool hasFrameRequest = false;
        uint64_t lastFrameIndex = 0;
        uint32_t lastSpawnCount = 0;
        bool lastSimulate = false;
        bool lastRender = false;
        std::vector<GpuParticleFrameRequest> queuedFrameRequests;
        GpuParticleOffscreenPolicy lastOffscreenPolicy = GpuParticleOffscreenPolicy::AlwaysSimulate;
        GpuParticleBoundsMode lastBoundsMode = GpuParticleBoundsMode::Automatic;
    };

    struct GraphState
    {
        std::unique_ptr<vk::RenderGraph> graph;
        std::unordered_map<uint64_t, std::shared_ptr<ParticleGpuGraphSpawnDomain>> spawnDomains;
        std::vector<std::unique_ptr<ParticleRenderGraph>> schedulers;
        std::unordered_map<uint64_t, ParticleRenderGraph *> schedulerById;
        uint64_t generation = 0;
        uint32_t renderExportBatch = rhi::InvalidSubmissionBatchIndex;
        bool asyncRecordingActive = false;
        bool asyncPhasePartitioned = false;
    };

    struct StateCaptureRequest
    {
        std::weak_ptr<GpuParticleStateCapture> capture;
        uint32_t firstRow;
        uint32_t rowCount;
        std::vector<GpuParticleStateField> fields;
    };

    struct StateCaptureOutput
    {
        std::shared_ptr<rhi::ComputeBuffer> buffer;
        intervals::Set initialized;
        intervals::Set written;

        bool Prepare(rhi::ComputeHost &host, rhi::ComputeBufferDesc desc)
        {
            auto storage = rhi::ComputeBuffer::TryCreatePrivate(host, desc);
            if (!storage)
                return false;
            buffer = storage->Freeze();
            initialized.emplace_back(0, buffer->GetByteSize());
            written = initialized;
            return true;
        }

        void Publish(rhi::SubmissionTicket completion, rhi::QueueRole writer)
        {
            buffer->GetAllocation()->PublishGraphValue(completion, writer, std::move(initialized), std::move(written));
            buffer->PublishStorage();
        }
    };

    struct StateCaptureDispatch
    {
        StateGatherConstants constants;
        StateCaptureOutput output;
    };

    struct StateCaptureWork
    {
        explicit StateCaptureWork(rhi::Device &device) : bindings(device)
        {
        }

        StateGatherBindings bindings;
        std::shared_ptr<GpuParticleStateCapture> capture;
        std::shared_ptr<Emitter> emitter;
        std::vector<StateCaptureDispatch> fields;
        StateCaptureOutput identities;
        rhi::QueueRole writer = rhi::QueueRole::Count;

        void Publish(rhi::SubmissionTicket completion, FrameWorkOwner &owner, uint64_t estimatedNanoseconds)
        {
            if (writer == rhi::QueueRole::Count || capture->Done())
                return;
            for (size_t i = 0; i < fields.size(); ++i) {
                auto &output = fields[i].output;
                output.Publish(completion, writer);
                if (auto port = capture->ports[i].lock())
                    port->buffer = output.buffer;
            }
            identities.Publish(completion, writer);
            if (auto port = capture->ports.back().lock())
                port->buffer = identities.buffer;
            capture->ports.clear();
            capture->submission = completion;
            (void)owner.RecordAcceptedResource(FrameWorkResource::Gpu, estimatedNanoseconds);
        }
    };

    using EmitterMap = std::map<uint64_t, std::shared_ptr<Emitter>>;
    vk::VkDeviceContext *context = nullptr;
    vk::VkPipelineManager *pipelines = nullptr;
    vk::VkResourceManager *resources = nullptr;
    GpuRetirementQueue *deletionQueue = nullptr;
    ParticleGpuDrawRegistry *drawRegistry = nullptr;
    GpuBillboardTextureResolver textureResolver;
    GpuParticleVectorFieldTextureResolver vectorFieldTextureResolver;
    GpuParticleSkinnedMeshResolver skinnedMeshResolver;
    std::shared_ptr<const GpuParticleCullProgramStorage> cullProgram;
    std::shared_ptr<const GpuParticleSortProgramStorage> sortProgram;
    std::shared_ptr<const GpuParticleBoundsProgramStorage> boundsProgram;
    std::shared_ptr<const GpuParticleMigrationProgramStorage> migrationProgram;
    std::shared_ptr<const GpuParticleSpawnProgramStorage> spawnProgram;
    std::shared_ptr<const GpuParticleRibbonProgramStorage> ribbonTopologyProgram;
    std::shared_ptr<const GpuParticleRibbonRenderProgramStorage> ribbonRenderProgram;
    std::unique_ptr<StateGatherProgram> stateGatherProgram;
    std::unique_ptr<ParticleGpuCollisionScene> collisionScene;
    EmitterMap emitters;
    bool requiresCollisionScene = false;
    std::shared_ptr<GraphState> graphState;
    mutable uint64_t nextGraphGeneration = 1;
    mutable std::unordered_map<const InxMesh *, MeshUpload> meshUploads;
    FrameWorkOwner *meshUploadOwner = nullptr;
    FrameWorkOwner::SourceId meshUploadSource = 0;
    mutable uint64_t nextMeshUploadToken = 1;
    uint64_t lastMeshUploadToken = 0;
    uint64_t lastMeshPackToken = 0;
    uint64_t lastMeshPackNanoseconds = 0;
    std::shared_ptr<DiagnosticState> diagnosticState = std::make_shared<DiagnosticState>();
    std::vector<PendingDiagnostic> pendingDiagnostics;
    struct RecordedDiagnostic
    {
        uint64_t requestId;
        std::function<void()> complete;
    };
    std::vector<RecordedDiagnostic> recordedDiagnostics;
    std::vector<StateCaptureRequest> stateCaptures;
    std::vector<std::shared_ptr<StateCaptureWork>> preparedStateCaptures;
    uint64_t nextDiagnosticRequestId = 1;
    uint64_t nextCaptureRequestId = 1;

    void FailDiagnostics(uint64_t graphInstanceId, const char *reason) noexcept
    {
        for (const auto &request : stateCaptures)
            if (const auto capture = request.capture.lock();
                capture && (graphInstanceId == 0 || capture->graphInstanceId == graphInstanceId))
                capture->Cancel("GPU particle source changed before state capture was accepted");
        pendingDiagnostics.erase(std::remove_if(pendingDiagnostics.begin(), pendingDiagnostics.end(),
                                                [graphInstanceId](const auto &request) {
                                                    return graphInstanceId == 0 ||
                                                           request.graphInstanceId == graphInstanceId;
                                                }),
                                 pendingDiagnostics.end());

        std::scoped_lock lock(diagnosticState->mutex);
        for (auto &[requestId, snapshot] : diagnosticState->snapshots) {
            (void)requestId;
            if (snapshot.status != GpuParticleDiagnosticStatus::Pending ||
                (graphInstanceId != 0 && snapshot.graphInstanceId != graphInstanceId))
                continue;
            snapshot.status = GpuParticleDiagnosticStatus::Failed;
            snapshot.error = reason;
        }
    }

    void RecordStateCaptures(VkCommandBuffer commandBuffer, rhi::QueueRole queue)
    {
        if (preparedStateCaptures.empty())
            return;

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        auto &device = context->GetRhiDevice();
        vk::VulkanComputeCommandContext computeContext;
        const auto encoder = device.MakeComputeCommandEncoder(computeContext, commandBuffer);
        for (const auto &work : preparedStateCaptures) {
            auto &capture = *work->capture;
            const auto &runtime = *work->emitter->runtime;
            if (capture.Done() || runtime.NeedsBootstrap())
                continue;
            for (const auto &field : work->fields) {
                const auto output = field.output.buffer->GetBuffer();
                // Later dispatches only write their own field. Keep the shared
                // identity buffer out of their descriptor access ranges.
                const auto identities = field.constants.identityWord == StateGatherProgram::SkipIdentities
                                            ? output
                                            : work->identities.buffer->GetBuffer();
                const auto group = stateGatherProgram->Bind(runtime, output, identities);
                if (!group.IsValid()) {
                    capture.Cancel("GPU particle capture binding allocation failed");
                    break;
                }
                work->bindings.groups.push_back(group);
            }
            if (capture.Done())
                continue;
            for (size_t index = 0; index < work->fields.size(); ++index) {
                auto constants = work->fields[index].constants;
                constants.aliveSlot = runtime.AliveReadSlot();
                stateGatherProgram->Record(encoder, work->bindings.groups[index], constants);
            }
            const auto &scheduler = *graphState->schedulerById.at(capture.emitterId);
            capture.sourceFrame = scheduler.LastConsumedFrame();
            capture.sourceSubstep = scheduler.LastConsumedSubstep();
            work->writer = queue;
        }
    }

    [[nodiscard]] bool HasQueuedFrameRequests() const noexcept
    {
        return std::any_of(emitters.begin(), emitters.end(),
                           [](const auto &entry) { return !entry.second->queuedFrameRequests.empty(); });
    }

    [[nodiscard]] bool ArmNextQueuedFrameRequests()
    {
        struct Pending
        {
            std::shared_ptr<Emitter> emitter;
            ParticleRenderGraph *scheduler = nullptr;
            ParticleGpuGraphSpawnDomain *spawnDomain = nullptr;
        };
        std::vector<Pending> pending;
        for (const auto &[id, emitter] : emitters) {
            if (!emitter || emitter->queuedFrameRequests.empty())
                continue;
            const auto scheduler = graphState->schedulerById.find(id);
            const auto domain = graphState->spawnDomains.find(emitter->graphInstanceId);
            if (scheduler == graphState->schedulerById.end() || domain == graphState->spawnDomains.end() ||
                !scheduler->second->CanBeginFrame(emitter->queuedFrameRequests.front()))
                return false;
            pending.push_back({emitter, scheduler->second, domain->second.get()});
        }
        if (pending.empty())
            return false;
        std::vector<bool> previousAcceptance;
        previousAcceptance.reserve(pending.size());
        // Preflight every scheduler before mutating the spawn domain. BeginFrame
        // has no fallible work after CanBeginFrame, so this keeps the graph
        // mark out of a partially accepted batch even if the defensive check
        // below is ever reached.
        for (const auto &entry : pending) {
            const auto emitter = entry.emitter;
            previousAcceptance.push_back(emitter->lastSimulate);
            const auto &request = entry.emitter->queuedFrameRequests.front();
            if (!entry.spawnDomain->SetEmitterAcceptingBurstRequests(entry.emitter->sourceProgram.graphEmitterIndex,
                                                                     request.simulate)) {
                for (size_t rollback = 0; rollback + 1 < previousAcceptance.size(); ++rollback) {
                    (void)pending[rollback].spawnDomain->SetEmitterAcceptingBurstRequests(
                        pending[rollback].emitter->sourceProgram.graphEmitterIndex, previousAcceptance[rollback]);
                }
                return false;
            }
        }
        for (const auto &entry : pending) {
            if (!entry.scheduler->BeginFrame(entry.emitter->queuedFrameRequests.front())) {
                for (size_t rollback = 0; rollback < previousAcceptance.size(); ++rollback) {
                    (void)pending[rollback].spawnDomain->SetEmitterAcceptingBurstRequests(
                        pending[rollback].emitter->sourceProgram.graphEmitterIndex, previousAcceptance[rollback]);
                }
                return false;
            }
        }
        for (const auto &entry : pending)
            entry.spawnDomain->MarkFramePending();
        for (const auto &entry : pending)
            entry.emitter->queuedFrameRequests.erase(entry.emitter->queuedFrameRequests.begin());
        return true;
    }

    void ClearQueuedFrameRequests() noexcept
    {
        for (const auto &[id, emitter] : emitters) {
            (void)id;
            if (emitter)
                emitter->queuedFrameRequests.clear();
        }
    }

    void AbortAsyncRecording() noexcept
    {
        if (graphState)
            graphState->asyncRecordingActive = false;
    }

    [[nodiscard]] bool RecordPendingGraphExecution(VkCommandBuffer commandBuffer)
    {
        if (!graphState || !graphState->graph || commandBuffer == VK_NULL_HANDLE)
            return false;
        AbortAsyncRecording();
        auto &state = *graphState;
        if (!state.asyncPhasePartitioned) {
            state.graph->Execute(commandBuffer, rhi::QueueRole::Compute);
            return true;
        }
        const auto &plan = state.graph->GetSubmissionPlan();
        state.graph->BeginExecution();
        for (uint32_t batch = 0; batch < plan.batches.size(); ++batch) {
            if (plan.batches[batch].queue != rhi::QueueRole::Compute ||
                !state.graph->RecordSubmissionBatch(batch, commandBuffer)) {
                INXLOG_ERROR("GPU particle sync simulation rejected compiled batch ", batch);
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool RecordPendingUploads(VkCommandBuffer commandBuffer)
    {
        const bool collisionPending = requiresCollisionScene && collisionScene && collisionScene->HasPendingUpload();
        const bool runtimePending = std::any_of(emitters.begin(), emitters.end(), [](const auto &entry) {
            return entry.second->runtime->HasPendingUploads();
        });
        const bool domainPending =
            graphState && std::any_of(graphState->spawnDomains.begin(), graphState->spawnDomains.end(),
                                      [](const auto &entry) { return entry.second->HasPendingUploads(); });
        if (!collisionPending && !runtimePending && !domainPending)
            return true;
        vk::VulkanTransferCommandContext transferContext;
        const auto transfer = context->GetRhiDevice().MakeTransferCommandEncoder(transferContext, commandBuffer);
        // Parameters and playback requests also have GPU writers. Include
        // diagnostic transfers and earlier substeps before targeted overwrites.
        VkMemoryBarrier before{};
        before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                               VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        if (collisionPending && !collisionScene->RecordPendingUpload(transfer))
            return false;
        for (const auto &[id, emitter] : emitters) {
            (void)id;
            if (emitter->runtime->HasPendingUploads() && !emitter->runtime->RecordPendingUploads(transfer))
                return false;
        }
        if (graphState) {
            for (const auto &[id, domain] : graphState->spawnDomains) {
                (void)id;
                if (domain->HasPendingUploads() && !domain->RecordPendingUploads(transfer))
                    return false;
            }
        }
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
        return true;
    }

    void RecordDiagnostics(VkCommandBuffer commandBuffer)
    {
        if (pendingDiagnostics.empty() || !context || !deletionQueue || commandBuffer == VK_NULL_HANDLE)
            return;

        std::vector<PendingDiagnostic> requests;
        for (auto it = pendingDiagnostics.begin(); it != pendingDiagnostics.end();) {
            if (it->remainingSampleFrames != 0) {
                ++it;
                continue;
            }
            requests.push_back(*it);
            it = pendingDiagnostics.erase(it);
        }
        if (requests.empty())
            return;
        auto &device = context->GetRhiDevice();
        vk::VulkanTransferCommandContext transferContext;
        const auto transfer = device.MakeTransferCommandEncoder(transferContext, commandBuffer);

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                             nullptr, 0, nullptr);

        for (const auto &request : requests) {
            struct EmitterCapture
            {
                GpuParticleEmitterDiagnostic diagnostic;
                uint64_t counterOffset = 0;
                uint64_t counterBytes = 0;
                uint64_t boundsOffset = 0;
                uint64_t spawnMetadataOffset = 0;
                uint64_t spawnDomainStateOffset = 0;
                uint64_t contactCounterOffset = 0;
                uint64_t contactCounterBytes = 0;
                uint32_t contactRecordCapacity = 0;
                uint32_t contactWorkItemCapacity = 0;
                uint64_t stateOffset = 0;
                uint64_t gatherOffset = 0;
                uint64_t stateBytes = 0;
                uint32_t stateStride = 0;
                uint32_t stateSampleCount = 0;
            };
            std::vector<EmitterCapture> emitterCaptures;
            uint64_t stateReadbackBytes = 0;
            for (const auto &[id, emitter] : emitters) {
                if (!emitter || emitter->graphInstanceId != request.graphInstanceId || !emitter->runtime)
                    continue;
                EmitterCapture capture;
                capture.diagnostic.emitterId = id;
                capture.diagnostic.emitterIndex = emitter->sourceProgram.graphEmitterIndex;
                capture.diagnostic.capacity = emitter->runtime->Capacity();
                capture.diagnostic.boundsMode = emitter->lastBoundsMode;
                capture.diagnostic.eventOverflowCounts.resize(emitter->runtime->EventTypeCount());
                capture.diagnostic.eventEnqueueCounts.resize(emitter->runtime->EventTypeCount());
                capture.diagnostic.eventCompleteCounts.resize(emitter->runtime->EventTypeCount());
                capture.counterBytes = emitter->runtime->CounterBufferByteSize();
                if (emitter->runtime->HasContactRuntime()) {
                    capture.contactCounterBytes = sizeof(GpuParticleContactCounters);
                    capture.contactRecordCapacity = emitter->runtime->ContactTelemetry().contactRecordCapacity;
                    capture.contactWorkItemCapacity = emitter->runtime->ContactTelemetry().workItemCapacity;
                }
                if (request.stateSampleCount > 0 && !emitter->runtime->NeedsBootstrap()) {
                    capture.stateStride = emitter->runtime->StateStride();
                    capture.stateSampleCount = std::min(request.stateSampleCount, capture.diagnostic.capacity);
                    capture.stateBytes =
                        capture.stateSampleCount *
                        (uint64_t(capture.stateStride) + StateGatherProgram::IdentityWords * sizeof(uint32_t));
                    capture.gatherOffset = stateReadbackBytes;
                    stateReadbackBytes += capture.stateBytes;
                }
                emitterCaptures.push_back(capture);
            }
            std::sort(emitterCaptures.begin(), emitterCaptures.end(), [](const auto &lhs, const auto &rhs) {
                return lhs.diagnostic.emitterIndex < rhs.diagnostic.emitterIndex;
            });

            uint64_t totalBytes = 0;
            for (const auto &capture : emitterCaptures)
                totalBytes += capture.counterBytes + ParticleGpuBounds::BoundsBufferBytes +
                              sizeof(GpuParticleSpawnMetadata) + sizeof(uint32_t) * 4u + capture.contactCounterBytes +
                              capture.stateBytes;
            constexpr uint64_t MaxDiagnosticStateReadbackBytes = 16ull * 1024ull * 1024ull;
            if (stateReadbackBytes > MaxDiagnosticStateReadbackBytes) {
                std::scoped_lock lock(diagnosticState->mutex);
                auto &snapshot = diagnosticState->snapshots[request.requestId];
                snapshot.status = GpuParticleDiagnosticStatus::Failed;
                snapshot.error = "GPU particle state diagnostic exceeds the 16 MiB bounded readback budget";
                continue;
            }
            if (emitterCaptures.empty() || totalBytes == 0) {
                std::scoped_lock lock(diagnosticState->mutex);
                auto &snapshot = diagnosticState->snapshots[request.requestId];
                snapshot.status = GpuParticleDiagnosticStatus::Inactive;
                snapshot.error = "GPU particle graph is not resident";
                continue;
            }

            constexpr auto queues = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
            rhi::BufferDesc readbackDesc{totalBytes, rhi::BufferUsageFlags::TransferDestination,
                                         rhi::BufferMemory::Readback};
            readbackDesc.queueAccess = queues;
            const auto readbackHandle = device.CreateBuffer(readbackDesc);
            if (!readbackHandle.IsValid()) {
                std::scoped_lock lock(diagnosticState->mutex);
                auto &snapshot = diagnosticState->snapshots[request.requestId];
                snapshot.status = GpuParticleDiagnosticStatus::Failed;
                snapshot.error = "Failed to allocate GPU particle diagnostic readback buffer";
                continue;
            }
            auto readback = std::make_shared<rhi::BufferResource>(device, readbackHandle, totalBytes);

            std::shared_ptr<rhi::BufferResource> gathered;
            std::shared_ptr<StateGatherBindings> gatherBindings;
            if (stateReadbackBytes != 0) {
                rhi::BufferDesc gatherDesc{stateReadbackBytes,
                                           rhi::BufferUsageFlags::Storage | rhi::BufferUsageFlags::TransferSource,
                                           rhi::BufferMemory::DeviceLocal};
                gatherDesc.queueAccess = queues;
                const auto handle = device.CreateBuffer(gatherDesc);
                gathered = std::make_shared<rhi::BufferResource>(device, handle, stateReadbackBytes);
                gatherBindings = std::make_shared<StateGatherBindings>(device);
                if (handle.IsValid() && stateGatherProgram) {
                    for (const auto &capture : emitterCaptures) {
                        if (capture.stateSampleCount != 0) {
                            const auto &runtime = *emitters.at(capture.diagnostic.emitterId)->runtime;
                            gatherBindings->groups.push_back(stateGatherProgram->Bind(runtime, handle, handle));
                        }
                    }
                }
                if (gatherBindings->groups.empty() ||
                    std::any_of(gatherBindings->groups.begin(), gatherBindings->groups.end(),
                                [](auto group) { return !group.IsValid(); })) {
                    std::scoped_lock lock(diagnosticState->mutex);
                    auto &snapshot = diagnosticState->snapshots[request.requestId];
                    snapshot.status = GpuParticleDiagnosticStatus::Failed;
                    snapshot.error = "Failed to prepare GPU particle state gathering";
                    continue;
                }

                vk::VulkanComputeCommandContext computeContext;
                const auto compute = device.MakeComputeCommandEncoder(computeContext, commandBuffer);
                size_t groupIndex = 0;
                for (const auto &capture : emitterCaptures) {
                    if (capture.stateSampleCount == 0)
                        continue;
                    const auto &runtime = *emitters.at(capture.diagnostic.emitterId)->runtime;
                    const auto strideWords = static_cast<uint32_t>(capture.stateStride / sizeof(uint32_t));
                    const auto identityWord = static_cast<uint32_t>(capture.gatherOffset / sizeof(uint32_t));
                    stateGatherProgram->Record(
                        compute, gatherBindings->groups[groupIndex++],
                        {strideWords, 0, strideWords, runtime.AliveReadSlot(), 0, capture.stateSampleCount,
                         identityWord + capture.stateSampleCount * StateGatherProgram::IdentityWords, identityWord});
                }
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            }

            uint64_t offset = 0;
            const auto spawnDomain = graphState ? graphState->spawnDomains.find(request.graphInstanceId)
                                                : decltype(graphState->spawnDomains)::const_iterator{};
            for (auto &capture : emitterCaptures) {
                capture.counterOffset = offset;
                const auto emitter = emitters.find(capture.diagnostic.emitterId);
                transfer.CopyBuffer(emitter->second->runtime->CounterBuffer(), readbackHandle,
                                    {0, offset, capture.counterBytes});
                offset += capture.counterBytes;
                capture.boundsOffset = offset;
                transfer.CopyBuffer(emitter->second->bounds->BoundsBuffer(), readbackHandle,
                                    {0, offset, ParticleGpuBounds::BoundsBufferBytes});
                offset += ParticleGpuBounds::BoundsBufferBytes;
                capture.spawnMetadataOffset = offset;
                if (graphState && spawnDomain != graphState->spawnDomains.end()) {
                    transfer.CopyBuffer(spawnDomain->second->MetadataBuffer(), readbackHandle,
                                        {spawnDomain->second->MetadataOffset(capture.diagnostic.emitterIndex), offset,
                                         sizeof(GpuParticleSpawnMetadata)});
                }
                offset += sizeof(GpuParticleSpawnMetadata);
                capture.spawnDomainStateOffset = offset;
                if (graphState && spawnDomain != graphState->spawnDomains.end()) {
                    const uint64_t slotOffset = uint64_t(capture.diagnostic.emitterIndex) * sizeof(uint32_t);
                    transfer.CopyBuffer(spawnDomain->second->BurstRequestBuffer(), readbackHandle,
                                        {slotOffset, offset, sizeof(uint32_t)});
                    transfer.CopyBuffer(spawnDomain->second->ConsumingBuffer(), readbackHandle,
                                        {slotOffset, offset + sizeof(uint32_t), sizeof(uint32_t)});
                    transfer.CopyBuffer(spawnDomain->second->BurstRequestAcceptanceBuffer(), readbackHandle,
                                        {slotOffset, offset + sizeof(uint32_t) * 2u, sizeof(uint32_t)});
                    transfer.CopyBuffer(spawnDomain->second->EmitterPlayingStateBuffer(), readbackHandle,
                                        {slotOffset, offset + sizeof(uint32_t) * 3u, sizeof(uint32_t)});
                }
                offset += sizeof(uint32_t) * 4u;
                if (capture.contactCounterBytes > 0) {
                    capture.contactCounterOffset = offset;
                    transfer.CopyBuffer(emitter->second->runtime->ContactResources().counters, readbackHandle,
                                        {0, offset, capture.contactCounterBytes});
                    offset += capture.contactCounterBytes;
                }
                if (capture.stateBytes > 0) {
                    capture.stateOffset = offset;
                    transfer.CopyBuffer(gathered->GetBuffer(), readbackHandle,
                                        {capture.gatherOffset, offset, capture.stateBytes});
                    offset += capture.stateBytes;
                }
            }
            const auto state = diagnosticState;
            auto *readbackDevice = static_cast<rhi::Device *>(&device);
            recordedDiagnostics.push_back(
                {request.requestId, [state, readback, gathered, gatherBindings,
                                     emitterCaptures = std::move(emitterCaptures), request, readbackDevice]() mutable {
                     std::vector<uint8_t> bytes(static_cast<size_t>(readback->GetByteSize()));
                     GpuParticleDiagnosticSnapshot result;
                     result.requestId = request.requestId;
                     result.graphInstanceId = request.graphInstanceId;
                     if (!readbackDevice->ReadBuffer(readback->GetBuffer(), 0, bytes.data(), bytes.size())) {
                         result.status = GpuParticleDiagnosticStatus::Failed;
                         result.error = "Failed to read completed GPU particle diagnostic buffer";
                     } else {
                         result.status = GpuParticleDiagnosticStatus::Completed;
                         for (auto &capture : emitterCaptures) {
                             std::array<uint32_t, ParticleGpuRuntime::BaseCounterWordCount> counters{};
                             std::memcpy(counters.data(), bytes.data() + capture.counterOffset, sizeof(counters));
                             capture.diagnostic.freeCount = std::min(counters[0], capture.diagnostic.capacity);
                             capture.diagnostic.aliveCount = capture.diagnostic.capacity - capture.diagnostic.freeCount;
                             capture.diagnostic.visibleCount = counters[1];
                             capture.diagnostic.droppedCount = counters[2];
                             capture.diagnostic.initializedSpawnCount = counters[3];
                             capture.diagnostic.collisionHitCount = counters[4];
                             capture.diagnostic.collisionResponseCount = counters[5];
                             capture.diagnostic.collisionTriggerCount = counters[6];
                             capture.diagnostic.collisionEnterCount = counters[7];
                             capture.diagnostic.collisionStayCount = counters[8];
                             capture.diagnostic.collisionExitCount = counters[9];
                             std::memcpy(&capture.diagnostic.collisionMaxOutwardSpeed, &counters[10], sizeof(float));
                             std::memcpy(&capture.diagnostic.collisionMaxTangentSpeed, &counters[11], sizeof(float));
                             capture.diagnostic.collisionCandidateOverflowCount = counters[12];
                             if (!capture.diagnostic.eventOverflowCounts.empty()) {
                                 const size_t eventCount = capture.diagnostic.eventOverflowCounts.size();
                                 const size_t eventBytes = eventCount * sizeof(uint32_t);
                                 const auto *eventCounters =
                                     bytes.data() + capture.counterOffset +
                                     ParticleGpuRuntime::EventCounterWordOffset * sizeof(uint32_t);
                                 std::memcpy(capture.diagnostic.eventOverflowCounts.data(), eventCounters, eventBytes);
                                 std::memcpy(capture.diagnostic.eventEnqueueCounts.data(), eventCounters + eventBytes,
                                             eventBytes);
                                 std::memcpy(capture.diagnostic.eventCompleteCounts.data(),
                                             eventCounters + eventBytes * 2u, eventBytes);
                             }
                             std::array<uint32_t, 8> encodedBounds{};
                             std::memcpy(encodedBounds.data(), bytes.data() + capture.boundsOffset,
                                         sizeof(encodedBounds));
                             capture.diagnostic.boundsValid = encodedBounds[6] != 0u;
                             if (capture.diagnostic.boundsValid) {
                                 const auto decodeOrderedFloat = [](uint32_t ordered) {
                                     const uint32_t bits =
                                         (ordered & 0x80000000u) != 0u ? (ordered ^ 0x80000000u) : ~ordered;
                                     float value = 0.0f;
                                     std::memcpy(&value, &bits, sizeof(value));
                                     return value;
                                 };
                                 for (size_t axis = 0; axis < 3; ++axis) {
                                     capture.diagnostic.boundsLower[axis] = decodeOrderedFloat(encodedBounds[axis]);
                                     capture.diagnostic.boundsUpper[axis] = decodeOrderedFloat(encodedBounds[axis + 3]);
                                 }
                             }
                             GpuParticleSpawnMetadata spawnMetadata{};
                             std::memcpy(&spawnMetadata, bytes.data() + capture.spawnMetadataOffset,
                                         sizeof(spawnMetadata));
                             capture.diagnostic.preparedSpawnCount = spawnMetadata.count;
                             capture.diagnostic.preparedSpawnBaseId = spawnMetadata.baseId;
                             capture.diagnostic.preparedSpawnGeneration = spawnMetadata.generation;
                             capture.diagnostic.spawnOverflowCount = spawnMetadata.overflowCount;
                             capture.diagnostic.acceptedSpawnTotal = spawnMetadata.acceptedSpawnTotal;
                             std::array<uint32_t, 4> spawnDomainState{};
                             std::memcpy(spawnDomainState.data(), bytes.data() + capture.spawnDomainStateOffset,
                                         sizeof(spawnDomainState));
                             capture.diagnostic.queuedBurstCount = spawnDomainState[0];
                             capture.diagnostic.consumingBurstCount = spawnDomainState[1];
                             capture.diagnostic.acceptingBurstRequests = spawnDomainState[2] != 0u;
                             capture.diagnostic.gpuEmitterPlaying = spawnDomainState[3] != 0u;
                             if (capture.contactCounterBytes > 0) {
                                 GpuParticleContactCounters contactCounters{};
                                 std::memcpy(&contactCounters, bytes.data() + capture.contactCounterOffset,
                                             sizeof(contactCounters));
                                 capture.diagnostic.contactOverflowCount = contactCounters.contactOverflow;
                                 capture.diagnostic.contactWorkItemOverflowCount = contactCounters.workItemOverflow;
                                 capture.diagnostic.contactCurrentSimulationStep =
                                     contactCounters.currentSimulationStep;
                                 capture.diagnostic.contactResetSerial = contactCounters.resetSerial;
                                 capture.diagnostic.contactCurrentRecordCount =
                                     std::min(contactCounters.currentRecordCount, capture.contactRecordCapacity);
                                 capture.diagnostic.contactWorkItemCount =
                                     std::min(contactCounters.workItemCount, capture.contactWorkItemCapacity);
                                 capture.diagnostic.contactMaxPerParticle = contactCounters.maxContactsPerParticle;
                                 capture.diagnostic.multiContactParticleCount =
                                     contactCounters.multiContactParticleCount;
                                 capture.diagnostic.contactRetainedOrderHash = contactCounters.retainedOrderHash;
                                 capture.diagnostic.contactDroppedOrderHash = contactCounters.droppedOrderHash;
                                 capture.diagnostic.contactMinParticleIndex = contactCounters.minParticleIndex;
                                 capture.diagnostic.contactMaxParticleIndex = contactCounters.maxParticleIndex;
                             }
                             if (capture.stateBytes > 0 && capture.stateSampleCount > 0) {
                                 const size_t wordsPerState = capture.stateStride / sizeof(uint32_t);
                                 const auto *first = bytes.data() + capture.stateOffset;
                                 const auto *states = first + capture.stateSampleCount *
                                                                  StateGatherProgram::IdentityWords * sizeof(uint32_t);
                                 for (uint32_t row = 0; row < capture.stateSampleCount; ++row) {
                                     std::array<uint32_t, StateGatherProgram::IdentityWords> identity;
                                     std::memcpy(identity.data(), first + row * sizeof(identity), sizeof(identity));
                                     if ((identity[2] & 1u) == 0u)
                                         continue;
                                     GpuParticleEmitterDiagnostic::StateSample sample;
                                     sample.slotIndex = identity[0];
                                     sample.spawnGeneration = identity[1];
                                     sample.lifecycleFlags = identity[2];
                                     sample.words.resize(wordsPerState);
                                     std::memcpy(sample.words.data(), states + row * capture.stateStride,
                                                 capture.stateStride);
                                     capture.diagnostic.stateSamples.push_back(std::move(sample));
                                 }
                             }
                             result.emitters.push_back(capture.diagnostic);
                         }
                     }
                     std::scoped_lock lock(state->mutex);
                     const auto found = state->snapshots.find(request.requestId);
                     if (found != state->snapshots.end() &&
                         found->second.status == GpuParticleDiagnosticStatus::Pending)
                         found->second = std::move(result);
                 }});
        }
    }

    static void PackMeshUpload(MeshUpload::Source &payload)
    {
        const auto &sourceVertices = payload.geometry->vertices;
        const auto &sourceIndices = payload.geometry->indices;
        if (sourceVertices.empty() || sourceIndices.empty() || sourceIndices.size() % 3 != 0 ||
            sourceIndices.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
            std::any_of(sourceIndices.begin(), sourceIndices.end(),
                        [&](uint32_t index) { return index >= sourceVertices.size(); })) {
            payload.failure = "GPU particle Mesh output requires valid indexed geometry";
            return;
        }

        auto &vertices = payload.vertices;
        vertices.reserve(sourceVertices.size());
        for (const auto &source : sourceVertices) {
            PackedParticleMeshVertex vertex;
            vertex.position = {source.pos.x, source.pos.y, source.pos.z, 1.0f};
            vertex.normal = {source.normal.x, source.normal.y, source.normal.z, 0.0f};
            vertex.tangent = {source.tangent.x, source.tangent.y, source.tangent.z, source.tangent.w};
            vertex.color = {source.color.x, source.color.y, source.color.z, 1.0f};
            vertex.uv = {source.texCoord.x, source.texCoord.y, 0.0f, 0.0f};
            vertices.push_back(vertex);
        }
        auto &skinInfluences = payload.skinInfluences;
        if (const auto &skinned = payload.skinned; skinned) {
            if (skinned->influences.size() != sourceVertices.size()) {
                payload.failure = "GPU particle Skinned Mesh influence count does not match its vertices";
                return;
            }
            skinInfluences.reserve(skinned->influences.size());
            for (const auto &source : skinned->influences) {
                PackedParticleSkinInfluence influence;
                influence.bones = source.boneIndex;
                influence.weights = source.weight;
                skinInfluences.push_back(influence);
            }
        }
        auto &samplingPrimitives = payload.samplingPrimitives;
        samplingPrimitives.reserve(sourceIndices.size() + sourceIndices.size() / 3);
        double cumulativeArea = 0.0;
        for (size_t index = 0; index < sourceIndices.size(); index += 3) {
            const uint32_t first = sourceIndices[index];
            const uint32_t second = sourceIndices[index + 1];
            const uint32_t third = sourceIndices[index + 2];
            const glm::vec3 edgeA = sourceVertices[second].pos - sourceVertices[first].pos;
            const glm::vec3 edgeB = sourceVertices[third].pos - sourceVertices[first].pos;
            const double area = static_cast<double>(glm::length(glm::cross(edgeA, edgeB))) * 0.5;
            if (!std::isfinite(area) || area <= 1.0e-12)
                continue;
            cumulativeArea += area;
            samplingPrimitives.push_back({first, second, third, FloatBits(static_cast<float>(cumulativeArea))});
        }
        const size_t samplingTriangleCount = samplingPrimitives.size();
        if (samplingTriangleCount == 0 || !std::isfinite(cumulativeArea) || cumulativeArea <= 0.0) {
            payload.failure = "GPU particle Mesh sampling requires non-degenerate triangles";
            return;
        }
        const float inverseArea = static_cast<float>(1.0 / cumulativeArea);
        for (size_t index = 0; index < samplingTriangleCount; ++index) {
            float cumulative = 0.0f;
            std::memcpy(&cumulative, &samplingPrimitives[index].cdfOrPadding, sizeof(cumulative));
            samplingPrimitives[index].cdfOrPadding = FloatBits(cumulative * inverseArea);
        }
        samplingPrimitives[samplingTriangleCount - 1].cdfOrPadding = FloatBits(1.0f);

        std::vector<std::pair<uint32_t, uint32_t>> uniqueEdges;
        uniqueEdges.reserve(sourceIndices.size());
        for (size_t index = 0; index < sourceIndices.size(); index += 3) {
            const std::array<uint32_t, 3> triangle = {sourceIndices[index], sourceIndices[index + 1],
                                                      sourceIndices[index + 2]};
            for (size_t edge = 0; edge < triangle.size(); ++edge) {
                const uint32_t first = triangle[edge];
                const uint32_t second = triangle[(edge + 1) % triangle.size()];
                if (first != second)
                    uniqueEdges.emplace_back(std::min(first, second), std::max(first, second));
            }
        }
        std::sort(uniqueEdges.begin(), uniqueEdges.end());
        uniqueEdges.erase(std::unique(uniqueEdges.begin(), uniqueEdges.end()), uniqueEdges.end());
        double cumulativeLength = 0.0;
        for (const auto &[first, second] : uniqueEdges) {
            const double length =
                static_cast<double>(glm::length(sourceVertices[second].pos - sourceVertices[first].pos));
            if (!std::isfinite(length) || length <= 1.0e-12)
                continue;
            cumulativeLength += length;
            samplingPrimitives.push_back({first, second, FloatBits(static_cast<float>(cumulativeLength)), 0u});
        }
        const size_t samplingEdgeCount = samplingPrimitives.size() - samplingTriangleCount;
        if (samplingEdgeCount == 0 || !std::isfinite(cumulativeLength) || cumulativeLength <= 0.0) {
            payload.failure = "GPU particle Mesh edge sampling requires non-degenerate edges";
            return;
        }
        const float inverseLength = static_cast<float>(1.0 / cumulativeLength);
        for (size_t index = samplingTriangleCount; index < samplingPrimitives.size(); ++index) {
            float cumulative = 0.0f;
            std::memcpy(&cumulative, &samplingPrimitives[index].thirdOrCdf, sizeof(cumulative));
            samplingPrimitives[index].thirdOrCdf = FloatBits(cumulative * inverseLength);
        }
        samplingPrimitives.back().thirdOrCdf = FloatBits(1.0f);
        payload.samplingTriangleCount = static_cast<uint32_t>(samplingTriangleCount);
        payload.samplingEdgeCount = static_cast<uint32_t>(samplingEdgeCount);
    }

    static void RunMeshPack(MeshUpload::Source &payload)
    {
        const auto start = std::chrono::steady_clock::now();
        try {
            PackMeshUpload(payload);
        } catch (const std::exception &exception) {
            payload.failure = std::string("GPU particle Mesh packing failed: ") + exception.what();
        }
        payload.skinned.reset();
        if (payload.storage) {
            payload.storage->ReduceTo(payload.vertices.capacity() * sizeof(PackedParticleMeshVertex) +
                                      payload.samplingPrimitives.capacity() * sizeof(PackedParticleMeshPrimitive) +
                                      payload.skinInfluences.capacity() * sizeof(PackedParticleSkinInfluence));
        }
        payload.executionNanoseconds =
            (std::max)(uint64_t{1}, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                              std::chrono::steady_clock::now() - start)
                                                              .count()));
    }

    void PrepareMeshUploadTickets(MeshUpload &upload) const
    {
        const auto &payload = *upload.source;
        if (!payload.failure.empty()) {
            upload.Fail(payload.failure);
            return;
        }
        upload.vertexCount = static_cast<uint32_t>(payload.vertices.size());
        upload.indexCount = static_cast<uint32_t>(payload.geometry->indices.size());
        upload.samplingTriangleCount = payload.samplingTriangleCount;
        upload.samplingEdgeCount = payload.samplingEdgeCount;
        // Inline simulation runs on Graphics while large systems may use Compute.
        constexpr auto queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
        const auto makeTicket = [this](size_t bytes) {
            return resources->CreateBufferUpload(bytes, rhi::BufferUsageFlags::Storage, queueAccess);
        };
        upload.tickets[MeshUpload::Vertices] = makeTicket(payload.vertices.size() * sizeof(PackedParticleMeshVertex));
        upload.tickets[MeshUpload::Indices] = makeTicket(payload.geometry->indices.size() * sizeof(uint32_t));
        upload.tickets[MeshUpload::Sampling] =
            makeTicket(payload.samplingPrimitives.size() * sizeof(PackedParticleMeshPrimitive));
        if (!payload.skinInfluences.empty())
            upload.tickets[MeshUpload::Skin] =
                makeTicket(payload.skinInfluences.size() * sizeof(PackedParticleSkinInfluence));
    }

    bool AppendMeshUpload(MeshUpload &upload, size_t part, VkDeviceSize bytes) const
    {
        const auto &payload = *upload.source;
        const std::array<const void *, MeshUpload::PartCount> sources{
            payload.vertices.data(), payload.geometry->indices.data(), payload.samplingPrimitives.data(),
            payload.skinInfluences.data()};
        const auto &ticket = upload.tickets[part];
        const auto offset = ticket->GetSubmittedBytes();
        const auto *data = static_cast<const std::byte *>(sources[part]) + offset;
        constexpr auto unit = sizeof(uint32_t);
        const std::initializer_list<vk::BufferUploadTicket *> destinations{upload.tickets[MeshUpload::Indices].get(),
                                                                           upload.tickets[MeshUpload::Sampling].get(),
                                                                           upload.tickets[MeshUpload::Skin].get()};
        if (!resources->TryAppendBufferUpload(ticket, data, bytes, meshUploadOwner, bytes == unit,
                                              meshUploadOwner ? unit : 0,
                                              part == MeshUpload::Vertices && offset == 0 && meshUploadOwner
                                                  ? destinations
                                                  : std::initializer_list<vk::BufferUploadTicket *>{}))
            return false;
        if (upload.NextPart() == MeshUpload::PartCount)
            upload.source.reset();
        return true;
    }

    void PollMeshUploads()
    {
        resources->PollGpuUploads();
        for (auto it = meshUploads.begin(); it != meshUploads.end();) {
            auto &upload = it->second;
            const bool current = upload.IsCurrent();
            if (upload.packJob.IsValid()) {
                if (!upload.packJob.IsComplete()) {
                    if (!current && !upload.packJob.IsCancellationRequested())
                        upload.packJob.Cancel();
                    ++it;
                    continue;
                }
                upload.packJob = {};
                FinishMeshPack(upload, current);
            }
            if (current)
                ++it;
            else
                it = meshUploads.erase(it);
        }
    }

    void FinishMeshPack(MeshUpload &upload, bool current)
    {
        const auto elapsed = upload.source->executionNanoseconds;
        if (elapsed) {
            lastMeshPackNanoseconds = elapsed;
            meshUploadOwner->ObserveResource(FrameWorkResource::Cpu, elapsed, upload.packFrame, upload.packGrant);
        }
        if (!current)
            return;
        try {
            if (elapsed)
                PrepareMeshUploadTickets(upload);
            else
                upload.Fail("GPU particle Mesh packing was canceled");
        } catch (const std::exception &exception) {
            upload.Fail(std::string("GPU particle Mesh upload failed: ") + exception.what());
        }
    }

    bool StartMeshPack(MeshUpload &upload)
    {
        auto &owner = *meshUploadOwner;
        if (!owner.CanAdmitResource(FrameWorkResource::Cpu, lastMeshPackNanoseconds, true))
            return false;
        auto &payload = upload.source;
        payload->storage = owner.TryReserveCpuStorage(payload->WorkspaceBytes());
        if (!payload->storage)
            return false;
        // The worker owns only immutable input snapshots and its private output.
        if (JobSystem::IsAvailable() && !JobSystem::Get().IsInline()) {
            upload.packJob = JobSystem::Get().Schedule([payload] { RunMeshPack(*payload); }, JobDomain::Render,
                                                       JobPriority::Low, JobAffinity::WorkerOnly);
        } else {
            RunMeshPack(*payload);
        }
        upload.packFrame = owner.Opportunity();
        upload.packGrant = owner.RecordAcceptedResource(FrameWorkResource::Cpu, lastMeshPackNanoseconds);
        if (!upload.packJob.IsValid())
            FinishMeshPack(upload, true);
        return true;
    }

    [[nodiscard]] std::shared_ptr<MeshResources> ResolveMeshResources(const std::shared_ptr<InxMesh> &mesh,
                                                                      bool &uploadPending, std::string *error) const
    {
        if (!mesh) {
            SetError(error, "GPU particle Mesh resource binding requires a loaded Mesh asset");
            return {};
        }
        if (!resources) {
            SetError(error, "GPU particle Mesh resource service is unavailable");
            return {};
        }
        if (!meshUploadOwner) {
            for (auto it = meshUploads.begin(); it != meshUploads.end();) {
                if (it->second.IsCurrent())
                    ++it;
                else
                    it = meshUploads.erase(it);
            }
        }
        auto &upload = meshUploads[mesh.get()];
        if (!upload.IsCurrent() && !upload.packJob.IsValid()) {
            upload = {};
            upload.owner = mesh;
            upload.generation = mesh->GetGeneration();
            upload.token = nextMeshUploadToken++;
            try {
                upload.source = std::make_shared<MeshUpload::Source>();
                upload.source->geometry = mesh->GetGeometrySnapshot();
                upload.source->skinned = mesh->GetSkinnedData();
                if (meshUploadOwner) {
                    const auto bytes = upload.source->WorkspaceBytes();
                    const auto limit = meshUploadOwner->CpuStorageLimit();
                    if (bytes > limit) {
                        upload.Fail("GPU particle Mesh packing requires " + std::to_string(bytes) +
                                    " bytes, exceeding the CPU storage limit of " + std::to_string(limit) + " bytes");
                    } else {
                        meshUploadOwner->Wake(meshUploadSource);
                    }
                } else {
                    RunMeshPack(*upload.source);
                    PrepareMeshUploadTickets(upload);
                    for (size_t part = 0; part < upload.tickets.size(); ++part) {
                        if (const auto &ticket = upload.tickets[part]; ticket)
                            (void)AppendMeshUpload(upload, part, ticket->GetSize());
                    }
                }
            } catch (const std::exception &exception) {
                upload.Fail(std::string("GPU particle Mesh upload failed: ") + exception.what());
            }
        }
        if (!upload.failure.empty()) {
            SetError(error, upload.failure);
            return {};
        }
        if (upload.resources)
            return upload.resources;
        try {
            if (upload.source || !std::all_of(upload.tickets.begin(), upload.tickets.end(), [this](const auto &ticket) {
                    return !ticket || resources->TryPublishBufferUpload(ticket);
                })) {
                uploadPending = true;
                SetError(error, "GPU particle Mesh upload is pending");
                return {};
            }
            auto result = std::make_shared<MeshResources>();
            result->vertices = resources->GetPublishedRhiBuffer(upload.tickets[MeshUpload::Vertices]);
            result->indices = resources->GetPublishedRhiBuffer(upload.tickets[MeshUpload::Indices]);
            result->samplingTriangles = resources->GetPublishedRhiBuffer(upload.tickets[MeshUpload::Sampling]);
            if (upload.tickets[MeshUpload::Skin])
                result->skinInfluences = resources->GetPublishedRhiBuffer(upload.tickets[MeshUpload::Skin]);
            result->vertexCount = upload.vertexCount;
            result->indexCount = upload.indexCount;
            result->samplingTriangleCount = upload.samplingTriangleCount;
            result->samplingEdgeCount = upload.samplingEdgeCount;
            if (!result->vertices || !result->vertices->IsValid() || !result->indices || !result->indices->IsValid() ||
                !result->samplingTriangles || !result->samplingTriangles->IsValid() ||
                (upload.tickets[MeshUpload::Skin] && (!result->skinInfluences || !result->skinInfluences->IsValid())))
                throw std::runtime_error("upload produced invalid RHI buffers");
            upload.resources = result;
            return result;
        } catch (const std::exception &exception) {
            upload.Fail(std::string("GPU particle Mesh publication failed: ") + exception.what());
            SetError(error, upload.failure);
            return {};
        }
    }

    [[nodiscard]] bool PrepareMeshUploads(const GpuParticleEmitterProgram &program, bool &waiting,
                                          std::string *error) const
    {
        const auto prepare = [&](const std::shared_ptr<InxMesh> &mesh) {
            bool pending = false;
            if (ResolveMeshResources(mesh, pending, error))
                return true;
            waiting = pending;
            return pending;
        };
        for (const auto &interface : program.meshInterfaces) {
            if (!prepare(interface.mesh))
                return false;
        }
        for (const auto &output : program.outputs) {
            if (output.type == GpuParticleOutputType::Mesh && !prepare(output.mesh))
                return false;
        }
        return true;
    }

    static void CaptureMeshRevisions(const GpuParticleEmitterProgram &program,
                                     std::vector<std::shared_ptr<InxMesh>> &meshes, std::vector<uint64_t> &generations)
    {
        meshes.clear();
        generations.clear();
        std::unordered_set<const InxMesh *> seen;
        const auto append = [&](const std::shared_ptr<InxMesh> &mesh) {
            if (!mesh || !seen.insert(mesh.get()).second)
                return;
            meshes.push_back(mesh);
            generations.push_back(mesh->GetGeneration());
        };
        for (const auto &interface : program.meshInterfaces)
            append(interface.mesh);
        for (const auto &output : program.outputs)
            append(output.mesh);
    }

    [[nodiscard]] bool BuildRuntimeDesc(const GpuParticleEmitterProgram &program, GpuEmitterDesc &runtimeDesc,
                                        std::vector<std::shared_ptr<InxTexture>> &vectorFields,
                                        std::vector<uint64_t> &vectorFieldGenerations, bool &uploadPending,
                                        std::string *error, GpuParticleTextureRequest request) const
    {
        runtimeDesc.capacity = program.capacity;
        runtimeDesc.stateStride = program.stateStride;
        runtimeDesc.eventTypeCount = program.eventTypeCount;
        runtimeDesc.collisionEnabled = program.collisionEnabled;
        runtimeDesc.supportsFusedUpdateRendering = program.supportsFusedUpdateRendering;
        for (size_t index = 0; index < program.kernels.size(); ++index)
            runtimeDesc.kernels[index] = {program.kernels[index].data(), program.kernels[index].size()};
        runtimeDesc.continuation.capacity = program.continuationCapacity;
        if (program.continuationCapacity > 0) {
            runtimeDesc.continuation.recordStride = program.continuationRecordStride;
            runtimeDesc.continuation.laneCount = program.continuationLaneCount;
            runtimeDesc.continuation.joinCount = program.continuationJoinCount;
            runtimeDesc.continuation.program.prepare = {
                program.continuationKernels[static_cast<size_t>(GpuParticleContinuationKernelStage::Prepare)].data(),
                program.continuationKernels[static_cast<size_t>(GpuParticleContinuationKernelStage::Prepare)].size(),
            };
            runtimeDesc.continuation.program.classify = {
                program.continuationKernels[static_cast<size_t>(GpuParticleContinuationKernelStage::Classify)].data(),
                program.continuationKernels[static_cast<size_t>(GpuParticleContinuationKernelStage::Classify)].size(),
            };
            runtimeDesc.continuation.program.dispatch = {
                program.continuationKernels[static_cast<size_t>(GpuParticleContinuationKernelStage::Dispatch)].data(),
                program.continuationKernels[static_cast<size_t>(GpuParticleContinuationKernelStage::Dispatch)].size(),
            };
        }
        if (program.collisionEnabled) {
            if (!collisionScene || !collisionScene->IsValid()) {
                SetError(error, "GPU particle collision scene is not initialized");
                return false;
            }
            runtimeDesc.collisionSceneHeader = collisionScene->HeaderBuffer();
            runtimeDesc.collisionSceneColliders = collisionScene->ColliderBuffer();
            runtimeDesc.collisionSceneGridOffsets = collisionScene->GridOffsetBuffer();
            runtimeDesc.collisionSceneGridColliderIndices = collisionScene->GridColliderIndexBuffer();
            runtimeDesc.collisionSceneMeshVertices = collisionScene->MeshVertexBuffer();
            runtimeDesc.collisionSceneMeshIndices = collisionScene->MeshIndexBuffer();
            runtimeDesc.collisionSceneMeshBvhNodes = collisionScene->MeshBvhBuffer();
        }
        runtimeDesc.meshInterfaces.reserve(program.meshInterfaces.size());
        for (const auto &mesh : program.meshInterfaces) {
            const auto meshResources = ResolveMeshResources(mesh.mesh, uploadPending, error);
            if (!meshResources)
                return false;
            GpuMeshInterfaceDesc runtimeMesh;
            runtimeMesh.stableId = mesh.stableId;
            runtimeMesh.interfaceIndex = mesh.interfaceIndex;
            runtimeMesh.metadataOffsetWords = mesh.metadataOffsetWords;
            runtimeMesh.vertexBinding = mesh.vertexBinding;
            runtimeMesh.triangleBinding = mesh.triangleBinding;
            runtimeMesh.influenceBinding = mesh.influenceBinding;
            runtimeMesh.paletteBinding = mesh.paletteBinding;
            runtimeMesh.worldSpace = mesh.worldSpace;
            runtimeMesh.meshToSpace = mesh.meshToSpace;
            runtimeMesh.vertexCount = meshResources->vertexCount;
            runtimeMesh.triangleCount = meshResources->samplingTriangleCount;
            runtimeMesh.edgeCount = meshResources->samplingEdgeCount;
            runtimeMesh.vertices = meshResources->vertices->GetBuffer();
            runtimeMesh.triangles = meshResources->samplingTriangles->GetBuffer();
            runtimeMesh.vertexBufferBytes = meshResources->vertices->GetByteSize();
            runtimeMesh.triangleBufferBytes = meshResources->samplingTriangles->GetByteSize();
            if (mesh.skinnedRendererHandle) {
                if (!skinnedMeshResolver) {
                    SetError(error, "GPU particle SkinnedMeshRenderer resolver is unavailable");
                    return false;
                }
                const auto snapshot = skinnedMeshResolver(mesh.skinnedRendererHandle());
                if (!snapshot || !snapshot->mesh || snapshot->mesh.get() != mesh.mesh.get() || !snapshot->model ||
                    !snapshot->currentPalette || snapshot->currentPalette->empty() ||
                    snapshot->currentPalette->size() != snapshot->model->skeleton.bones.size() ||
                    !meshResources->skinInfluences || !meshResources->skinInfluences->IsValid()) {
                    SetError(error, "GPU particle SkinnedMeshRenderer source has no valid live skin pose");
                    return false;
                }
                runtimeMesh.worldSpace = true;
                runtimeMesh.meshToSpace = snapshot->sourceToWorld;
                runtimeMesh.boneCount = static_cast<uint32_t>(snapshot->currentPalette->size());
                runtimeMesh.poseRevision = snapshot->revision;
                runtimeMesh.influences = meshResources->skinInfluences->GetBuffer();
                runtimeMesh.influenceBufferBytes = meshResources->skinInfluences->GetByteSize();
                runtimeMesh.initialPalette.assign(snapshot->currentPalette->begin(), snapshot->currentPalette->end());
            }
            runtimeMesh.keepAlive = meshResources;
            runtimeDesc.meshInterfaces.push_back(std::move(runtimeMesh));
        }

        runtimeDesc.vectorFields.metadataBinding = program.vectorFields.metadataBinding;
        runtimeDesc.vectorFields.interfaceStrideWords = program.vectorFields.interfaceStrideWords;
        runtimeDesc.vectorFields.vectorFields.reserve(program.vectorFields.vectorFields.size());
        runtimeDesc.vectorFields.textureParameters.reserve(program.vectorFields.textureParameters.size());
        const size_t sampledTextureCount =
            program.vectorFields.vectorFields.size() + program.vectorFields.textureParameters.size();
        vectorFields.reserve(sampledTextureCount);
        vectorFieldGenerations.reserve(sampledTextureCount);
        std::unordered_set<std::string> stableIds;
        for (const auto &field : program.vectorFields.vectorFields) {
            const TextureSemantic expectedSemantic = field.kind == GpuVectorFieldDesc::Kind::SignedDistanceField
                                                         ? TextureSemantic::SignedDistanceField
                                                         : TextureSemantic::VectorField;
            if (field.stableId.empty() || !stableIds.insert(field.stableId).second || !field.texture ||
                field.texture->GetDimension() != TextureDimension::Texture3D ||
                field.texture->GetSemantic() != expectedSemantic || field.texture->GetGuid().empty() ||
                !vectorFieldTextureResolver) {
                SetError(error, "GPU particle volume bindings require unique identities and matching Texture3D assets");
                return false;
            }
            auto lease =
                vectorFieldTextureResolver(field.texture->GetGuid(), field.linearFiltering, field.repeat, request);
            if (lease.status != GpuBillboardTextureStatus::Ready || !lease.texture.IsValid() ||
                !lease.sampler.IsValid() || !lease.gpuView || !lease.gpuView->IsValid()) {
                uploadPending = lease.status == GpuBillboardTextureStatus::Pending;
                SetError(error, uploadPending ? "GPU particle volume texture upload is pending"
                                              : "GPU particle volume texture upload failed");
                return false;
            }
            GpuVectorFieldDesc runtimeField;
            runtimeField.kind = field.kind;
            runtimeField.interfaceIndex = field.interfaceIndex;
            runtimeField.textureBinding = field.textureBinding;
            runtimeField.worldSpace = field.worldSpace;
            runtimeField.fieldToSpace = field.fieldToSpace;
            runtimeField.vectorScale = field.vectorScale;
            runtimeField.texture = lease.texture;
            runtimeField.sampler = lease.sampler;
            runtimeField.keepAlive = std::move(lease.gpuView);
            runtimeDesc.vectorFields.vectorFields.push_back(std::move(runtimeField));
            vectorFields.push_back(field.texture);
            vectorFieldGenerations.push_back(field.texture->GetGeneration());
        }
        std::unordered_set<std::string> parameterIds;
        for (size_t index = 0; index < program.vectorFields.textureParameters.size(); ++index) {
            const auto &parameter = program.vectorFields.textureParameters[index];
            if (parameter.stableId.empty() || !parameterIds.insert(parameter.stableId).second ||
                parameter.resourceIndex != index || parameter.textureGuid.empty() || !textureResolver) {
                SetError(error, "GPU particle Texture2D parameter bindings require unique identities");
                return false;
            }
            auto lease = textureResolver(parameter.textureGuid, parameter.stableId, request);
            if (lease.status != GpuBillboardTextureStatus::Ready || !lease.texture.IsValid() ||
                !lease.sampler.IsValid() || !lease.gpuView || !lease.gpuView->IsValid()) {
                uploadPending = lease.status == GpuBillboardTextureStatus::Pending;
                SetError(error, uploadPending ? "GPU particle sampled texture upload is pending"
                                              : "GPU particle Texture2D parameter upload failed");
                return false;
            }
            GpuTexture2DParameterDesc runtimeParameter;
            runtimeParameter.resourceIndex = parameter.resourceIndex;
            runtimeParameter.parameterSlot = parameter.parameterSlot;
            runtimeParameter.textureBinding = parameter.textureBinding;
            runtimeParameter.texture = lease.texture;
            runtimeParameter.sampler = lease.sampler;
            runtimeParameter.keepAlive = std::move(lease.gpuView);
            runtimeDesc.vectorFields.textureParameters.push_back(std::move(runtimeParameter));
            vectorFields.push_back(parameter.texture);
            vectorFieldGenerations.push_back(parameter.texture ? parameter.texture->GetGeneration() : 0);
        }
        return true;
    }

    [[nodiscard]] std::shared_ptr<ParticleGpuOutputRenderer>
    CreateOutputRenderer(const Emitter &emitter, const GpuParticleOutputProgram &output, bool &uploadPending,
                         std::string *error) const
    {
        if (output.type == GpuParticleOutputType::Ribbon) {
            if (!emitter.ribbonTopology || !ribbonRenderProgram || !ribbonRenderProgram->IsValid()) {
                SetError(error, "GPU particle Ribbon resources are unavailable");
                return {};
            }
            auto renderer = std::make_shared<ParticleGpuRibbonRenderer>();
            GpuRibbonRendererDesc desc;
            desc.program = ribbonRenderProgram->View();
            desc.topology = emitter.ribbonTopology;
            desc.shaderProgram = output.shaderProgram;
            desc.material = output.material;
            desc.fallbackMaterial = output.fallbackMaterial;
            desc.semantics = output.semantics;
            desc.uvMode = output.ribbonUvMode;
            desc.uvScale = output.ribbonUvScale;
            desc.textureResolver = textureResolver;
            desc.deletionQueue = deletionQueue;
            return renderer->Create(context->GetRhiDevice(), desc) ? renderer : nullptr;
        }
        if (output.type == GpuParticleOutputType::Mesh) {
            const auto meshResources = ResolveMeshResources(output.mesh, uploadPending, error);
            if (!meshResources)
                return {};
            auto renderer = std::make_shared<ParticleGpuMeshRenderer>();
            GpuMeshRendererDesc desc;
            desc.vertexShader = {emitter.meshVertexShader.data(), emitter.meshVertexShader.size()};
            desc.shadowFragmentShader = {emitter.meshShadowFragmentShader.data(),
                                         emitter.meshShadowFragmentShader.size()};
            desc.pickingFragmentShader = {emitter.meshPickingFragmentShader.data(),
                                          emitter.meshPickingFragmentShader.size()};
            desc.motionVertexShader = {emitter.meshMotionVertexShader.data(), emitter.meshMotionVertexShader.size()};
            desc.motionFragmentShader = {emitter.meshMotionFragmentShader.data(),
                                         emitter.meshMotionFragmentShader.size()};
            desc.instances = emitter.runtime->InstanceBuffer();
            desc.renderIndices = emitter.runtime->RenderIndexBuffer();
            desc.mesh = output.mesh;
            desc.meshVertices = meshResources->vertices->GetBuffer();
            desc.meshIndices = meshResources->indices->GetBuffer();
            desc.indexCount = meshResources->indexCount;
            desc.meshBufferKeepAlive = meshResources;
            desc.shaderProgram = output.shaderProgram;
            desc.material = output.material;
            desc.fallbackMaterial = output.fallbackMaterial;
            desc.semantics = output.semantics;
            desc.textureResolver = textureResolver;
            desc.deletionQueue = deletionQueue;
            return renderer->Create(context->GetRhiDevice(), desc) ? renderer : nullptr;
        }
        auto renderer = std::make_shared<ParticleGpuBillboardRenderer>();
        GpuBillboardRendererDesc rendererDesc;
        rendererDesc.vertexShader = {emitter.billboardVertexShader.data(), emitter.billboardVertexShader.size()};
        rendererDesc.pickingFragmentShader = {emitter.billboardPickingFragmentShader.data(),
                                              emitter.billboardPickingFragmentShader.size()};
        rendererDesc.motionVertexShader = {emitter.billboardMotionVertexShader.data(),
                                           emitter.billboardMotionVertexShader.size()};
        rendererDesc.motionFragmentShader = {emitter.billboardMotionFragmentShader.data(),
                                             emitter.billboardMotionFragmentShader.size()};
        rendererDesc.shaderProgram = output.shaderProgram;
        rendererDesc.instances = emitter.runtime->InstanceBuffer();
        rendererDesc.renderIndices = emitter.runtime->RenderIndexBuffer();
        rendererDesc.material = output.material;
        rendererDesc.fallbackMaterial = output.fallbackMaterial;
        rendererDesc.semantics = output.semantics;
        rendererDesc.flipbookColumns = output.flipbookColumns;
        rendererDesc.flipbookRows = output.flipbookRows;
        rendererDesc.textureResolver = textureResolver;
        rendererDesc.deletionQueue = deletionQueue;
        return renderer->Create(context->GetRhiDevice(), rendererDesc) ? renderer : nullptr;
    }

    [[nodiscard]] bool PreflightEmitterProgram(const GpuParticleEmitterProgram &program,
                                               const std::shared_ptr<Emitter> &previous, std::string *error) const
    {
        if (program.id == 0 || program.graphInstanceId == 0 || program.stableId.empty() ||
            program.artifactRevision == 0 || program.capacity == 0 || program.stateStride == 0) {
            SetError(error,
                     "GPU particle program graph identity, emitter identity, revision, capacity, and state stride "
                     "must be valid");
            return false;
        }
        const auto &fusedKernel = program.kernels[static_cast<size_t>(GpuKernelStage::UpdateRenderingFused)];
        if (!std::all_of(program.kernels.begin(), program.kernels.end(),
                         [&](const auto &kernel) {
                             return (&kernel == &fusedKernel && !program.supportsFusedUpdateRendering) ||
                                    IsSpirv(kernel);
                         }) ||
            (program.supportsFusedUpdateRendering && !IsSpirv(fusedKernel))) {
            SetError(error, "GPU particle program contains invalid compute SPIR-V");
            return false;
        }
        const bool hasContinuationKernels =
            std::any_of(program.continuationKernels.begin(), program.continuationKernels.end(),
                        [](const auto &kernel) { return !kernel.empty(); });
        if ((program.continuationCapacity == 0) != !hasContinuationKernels ||
            program.continuationCapacity > ParticleGpuContinuationRuntime::MaximumCapacity ||
            (program.continuationCapacity > 0 &&
             (program.continuationRecordStride < sizeof(GpuParticleContinuationRecord) ||
              program.continuationRecordStride > ParticleGpuContinuationRuntime::MaximumRecordStride ||
              program.continuationRecordStride % 16 != 0)) ||
            (program.continuationCapacity > 0 &&
             (program.continuationLaneCount == 0 ||
              program.continuationLaneCount > ParticleGpuContinuationRuntime::MaximumLaneCount ||
              program.continuationJoinCount > ParticleGpuContinuationRuntime::MaximumJoinCount)) ||
            (program.continuationCapacity > 0 &&
             !std::all_of(program.continuationKernels.begin(), program.continuationKernels.end(),
                          [](const auto &kernel) { return IsSpirv(kernel); }))) {
            SetError(error,
                     "GPU particle continuation capacity, record stride, static lane/join counts and "
                     "Prepare/Classify/Dispatch SPIR-V must be supplied together and satisfy the bounded runtime "
                     "contract");
            return false;
        }
        if (program.outputs.empty()) {
            SetError(error, "GPU particle program requires at least one rendering output");
            return false;
        }

        const bool needsMesh = std::any_of(program.outputs.begin(), program.outputs.end(), [](const auto &output) {
            return output.type == GpuParticleOutputType::Mesh;
        });
        const bool needsRibbon = std::any_of(program.outputs.begin(), program.outputs.end(), [](const auto &output) {
            return output.type == GpuParticleOutputType::Ribbon;
        });
        if (needsMesh && (!IsSpirv(program.meshVertexShader) || !IsSpirv(program.meshShadowFragmentShader) ||
                          !IsSpirv(program.meshPickingFragmentShader) || !IsSpirv(program.meshMotionVertexShader) ||
                          !IsSpirv(program.meshMotionFragmentShader))) {
            SetError(error, "GPU particle program contains invalid mesh output SPIR-V");
            return false;
        }
        if (needsRibbon && (!ribbonTopologyProgram || !ribbonTopologyProgram->IsValid() || !ribbonRenderProgram ||
                            !ribbonRenderProgram->IsValid())) {
            SetError(error, "GPU particle Ribbon topology or render shaders are unavailable");
            return false;
        }

        std::unordered_set<uint64_t> outputIds;
        std::unordered_set<std::string> outputStableIds;
        outputIds.reserve(program.outputs.size());
        outputStableIds.reserve(program.outputs.size());
        for (const auto &output : program.outputs) {
            if (output.id == 0 || output.stableId.empty() || !outputIds.insert(output.id).second ||
                !outputStableIds.insert(output.stableId).second) {
                SetError(error, "GPU particle output identity must be valid and unique per emitter");
                return false;
            }
            if (!output.shaderProgram || !output.shaderProgram->IsValid() ||
                output.shaderProgram->domain != ShaderProgramDomain::ParticleSprite) {
                SetError(error, "GPU particle surface outputs require a valid linked ParticleSprite shader");
                return false;
            }
            if (output.semantics.castShadows && output.type != GpuParticleOutputType::Mesh) {
                SetError(error, "only static mesh particle outputs can cast shadows");
                return false;
            }
            if (!output.semantics.IsValid()) {
                SetError(error, "GPU particle output '" + output.stableId + "' has invalid rendering semantics");
                return false;
            }
            if (output.semantics.receiveSceneLighting &&
                !output.shaderProgram->FindVariant(ShaderCompileTarget::ForwardPlus)) {
                SetError(error, "GPU particle output '" + output.stableId +
                                    "' requires a linked Particle Forward+ shader variant when Receive Scene "
                                    "Lighting is enabled");
                return false;
            }
            if (output.type == GpuParticleOutputType::Mesh && (!output.mesh || output.semantics.softParticles)) {
                SetError(error, "GPU particle mesh output '" + output.stableId +
                                    "' requires a loaded mesh and currently does not support soft fading");
                return false;
            }
            if (output.type == GpuParticleOutputType::Ribbon &&
                (output.semantics.castShadows || !std::isfinite(output.ribbonUvScale) ||
                 output.ribbonUvScale <= 0.0f)) {
                SetError(error, "GPU particle Ribbon output '" + output.stableId +
                                    "' cannot cast shadows and requires a positive finite UV scale");
                return false;
            }
            if (output.type != GpuParticleOutputType::Ribbon && output.semantics.sortMode != ParticleSortMode::None &&
                (!sortProgram || !sortProgram->IsValid())) {
                SetError(error, "GPU particle sorting kernels are unavailable");
                return false;
            }
        }
        if (!cullProgram || !cullProgram->IsValid()) {
            SetError(error, "GPU particle view-culling kernels are unavailable");
            return false;
        }
        if (program.preserveState &&
            (!previous || previous->id != program.id || previous->graphInstanceId != program.graphInstanceId ||
             previous->stableId != program.stableId)) {
            SetError(error, "GPU particle state preservation requires the same live emitter identity");
            return false;
        }
        return true;
    }

    [[nodiscard]] std::shared_ptr<Emitter> CreateEmitter(const GpuParticleEmitterProgram &program,
                                                         const std::shared_ptr<Emitter> &previous, bool &uploadPending,
                                                         std::string *error) const
    {
        // ApplyGraphs preflights every candidate before any private runtime is
        // built. Keep validation in that single pass, including preserveState.
        const bool needsRibbon = std::any_of(program.outputs.begin(), program.outputs.end(), [](const auto &output) {
            return output.type == GpuParticleOutputType::Ribbon;
        });

        auto emitter = std::make_shared<Emitter>();
        emitter->id = program.id;
        emitter->graphInstanceId = program.graphInstanceId;
        emitter->artifactRevision = program.artifactRevision;
        emitter->stableId = program.stableId;
        emitter->statePreservedOnPublish = program.preserveState;
        emitter->billboardVertexShader = program.billboardVertexShader;
        emitter->billboardPickingFragmentShader = program.billboardPickingFragmentShader;
        emitter->billboardMotionVertexShader = program.billboardMotionVertexShader;
        emitter->billboardMotionFragmentShader = program.billboardMotionFragmentShader;
        emitter->meshVertexShader = program.meshVertexShader;
        emitter->meshShadowFragmentShader = program.meshShadowFragmentShader;
        emitter->meshPickingFragmentShader = program.meshPickingFragmentShader;
        emitter->meshMotionVertexShader = program.meshMotionVertexShader;
        emitter->meshMotionFragmentShader = program.meshMotionFragmentShader;
        emitter->runtime = std::make_unique<ParticleGpuRuntime>();

        GpuEmitterDesc runtimeDesc;
        if (!BuildRuntimeDesc(program, runtimeDesc, emitter->vectorFields, emitter->vectorFieldGenerations,
                              uploadPending, error, GpuParticleTextureRequest::Prepare))
            return {};
        emitter->observedVectorFieldGenerations = emitter->vectorFieldGenerations;
        CaptureMeshRevisions(program, emitter->meshes, emitter->observedMeshGenerations);
        emitter->sourceProgram = program;
        auto &device = context->GetRhiDevice();
        const bool requiresMigration = program.preserveState && program.migration.has_value();
        const bool runtimeCreated = requiresMigration ? emitter->runtime->Create(device, runtimeDesc)
                                    : program.preserveState
                                        ? emitter->runtime->CreateCompatible(device, runtimeDesc, *previous->runtime)
                                        : emitter->runtime->Create(device, runtimeDesc);
        if (!runtimeCreated) {
            SetError(error, program.preserveState
                                ? "GPU particle state ABI is incompatible with the requested hot reload"
                                : "failed to create GPU particle simulation runtime");
            return {};
        }
        if (requiresMigration) {
            const auto &migration = *program.migration;
            if (!migrationProgram || !migrationProgram->IsValid() ||
                migration.sourceStride != previous->runtime->StateStride() ||
                migration.destinationStride != emitter->runtime->StateStride()) {
                SetError(error, "GPU particle migration descriptor does not match the live state ABI");
                return {};
            }
            GpuParticleMigrationDesc migrationDesc;
            migrationDesc.sourceCapacity = previous->runtime->Capacity();
            migrationDesc.destinationCapacity = emitter->runtime->Capacity();
            migrationDesc.sourceStride = migration.sourceStride;
            migrationDesc.destinationStride = migration.destinationStride;
            migrationDesc.sourceStates = previous->runtime->StateBuffer();
            migrationDesc.sourceCounters = previous->runtime->CounterBuffer();
            migrationDesc.sourceCounterByteSize = previous->runtime->CounterBufferByteSize();
            migrationDesc.destinationStates = emitter->runtime->StateBuffer();
            migrationDesc.destinationFreeList = emitter->runtime->FreeListBuffer();
            migrationDesc.destinationCounters = emitter->runtime->CounterBuffer();
            migrationDesc.destinationAliveIndices = emitter->runtime->AliveIndexBuffer(0);
            migrationDesc.destinationAliveControl = emitter->runtime->AliveControlBuffer();
            migrationDesc.copyRanges = migration.copyRanges;
            migrationDesc.defaultStateWords = migration.defaultStateWords;
            migrationDesc.program = migrationProgram->View();
            emitter->migration = std::make_shared<ParticleGpuMigrator>();
            if (!emitter->migration->Create(device, migrationDesc)) {
                SetError(error, "failed to create GPU particle state migrator");
                return {};
            }
            emitter->migrationSource = previous;
        } else if (program.preserveState && previous && previous->migration &&
                   emitter->runtime->SharesStateWith(*previous->runtime)) {
            // A compatible publication can arrive before a previously recorded
            // layout migration reaches the next frame boundary (Save followed
            // immediately by Play is the common case). The resident buffers are
            // shared, so the pending migration transaction must follow them;
            // dropping it would bootstrap the uninitialized destination state.
            emitter->migration = previous->migration;
            emitter->migrationSource = previous->migrationSource;
        }
        if (!boundsProgram || !boundsProgram->IsValid()) {
            SetError(error, "GPU particle bounds kernels are unavailable");
            return {};
        }
        emitter->bounds = std::make_unique<ParticleGpuBounds>();
        GpuParticleBoundsDesc boundsDesc;
        boundsDesc.capacity = emitter->runtime->Capacity();
        boundsDesc.visibility = emitter->runtime->VisibilityBuffer();
        boundsDesc.sourceIndices = emitter->runtime->RenderIndexBuffer();
        boundsDesc.sourceIndirectArguments = emitter->runtime->IndirectBuffer();
        boundsDesc.simulationControl = emitter->runtime->SimulationControlBuffer();
        boundsDesc.program = boundsProgram->View();
        if (!emitter->bounds->Create(device, boundsDesc)) {
            SetError(error, "failed to create GPU particle bounds reducer");
            return {};
        }
        if (needsRibbon) {
            GpuParticleRibbonDesc ribbonDesc;
            ribbonDesc.capacity = emitter->runtime->Capacity();
            ribbonDesc.instances = emitter->runtime->InstanceBuffer();
            ribbonDesc.sourceIndices = emitter->runtime->RenderIndexBuffer();
            ribbonDesc.sourceIndirectArguments = emitter->runtime->IndirectBuffer();
            ribbonDesc.simulationControl = emitter->runtime->SimulationControlBuffer();
            ribbonDesc.program = ribbonTopologyProgram->View();
            emitter->ribbonTopology = std::make_shared<ParticleGpuRibbonTopology>();
            if (!emitter->ribbonTopology->Create(device, ribbonDesc)) {
                SetError(error, "failed to create GPU-resident Ribbon topology");
                return {};
            }
        }

        emitter->outputs.reserve(program.outputs.size());
        for (const auto &output : program.outputs) {
            auto renderer = CreateOutputRenderer(*emitter, output, uploadPending, error);
            if (!renderer) {
                if (!error || error->empty())
                    SetError(error,
                             "failed to create GPU particle output renderer for output '" + output.stableId + "'");
                return {};
            }
            Emitter::Output emitterOutput;
            emitterOutput.id = output.id;
            emitterOutput.stableId = output.stableId;
            emitterOutput.type = output.type;
            emitterOutput.mesh = output.mesh;
            emitterOutput.material = output.material;
            emitterOutput.shaderProgram = output.shaderProgram;
            emitterOutput.fallbackMaterial = output.fallbackMaterial;
            emitterOutput.semantics = output.semantics;
            if (output.type == GpuParticleOutputType::Ribbon)
                emitterOutput.semantics.sortMode = ParticleSortMode::None;
            emitterOutput.ribbonUvMode = output.ribbonUvMode;
            emitterOutput.ribbonUvScale = output.ribbonUvScale;
            emitterOutput.flipbookColumns = output.flipbookColumns;
            emitterOutput.flipbookRows = output.flipbookRows;
            emitterOutput.renderer = std::move(renderer);
            emitter->outputs.push_back(std::move(emitterOutput));
        }
        return emitter;
    }

    [[nodiscard]] bool RefreshResources(Emitter &emitter)
    {
        if (emitter.vectorFields.empty() && emitter.meshes.empty())
            return true;
        std::vector<uint64_t> currentVectorFieldGenerations;
        currentVectorFieldGenerations.reserve(emitter.vectorFields.size());
        bool textureChanged = emitter.vectorFields.size() != emitter.observedVectorFieldGenerations.size();
        for (size_t index = 0; index < emitter.vectorFields.size(); ++index) {
            const uint64_t generation = emitter.vectorFields[index] ? emitter.vectorFields[index]->GetGeneration() : 0;
            currentVectorFieldGenerations.push_back(generation);
            textureChanged = textureChanged || index >= emitter.observedVectorFieldGenerations.size() ||
                             generation != emitter.observedVectorFieldGenerations[index];
        }
        std::vector<uint64_t> currentMeshGenerations;
        currentMeshGenerations.reserve(emitter.meshes.size());
        bool meshChanged = emitter.meshes.size() != emitter.observedMeshGenerations.size();
        for (size_t index = 0; index < emitter.meshes.size(); ++index) {
            const uint64_t generation = emitter.meshes[index] ? emitter.meshes[index]->GetGeneration() : 0;
            currentMeshGenerations.push_back(generation);
            meshChanged = meshChanged || index >= emitter.observedMeshGenerations.size() ||
                          generation != emitter.observedMeshGenerations[index];
        }
        if (!textureChanged && !meshChanged)
            return true;

        std::string error;
        GpuEmitterDesc runtimeDesc;
        std::vector<std::shared_ptr<InxTexture>> vectorFields;
        std::vector<uint64_t> vectorFieldGenerations;
        auto replacement = std::make_unique<ParticleGpuRuntime>();
        bool uploadPending = false;
        const bool meshesReady =
            !meshChanged || (PrepareMeshUploads(emitter.sourceProgram, uploadPending, &error) && !uploadPending);
        const bool descriptorValid =
            meshesReady && BuildRuntimeDesc(emitter.sourceProgram, runtimeDesc, vectorFields, vectorFieldGenerations,
                                            uploadPending, &error, GpuParticleTextureRequest::Poll);
        const bool replacementCreated =
            descriptorValid && replacement->CreateCompatible(context->GetRhiDevice(), runtimeDesc, *emitter.runtime);
        const auto rememberHardFailure = [&]() {
            emitter.observedVectorFieldGenerations = currentVectorFieldGenerations;
            emitter.observedMeshGenerations = currentMeshGenerations;
        };
        if (!replacementCreated || !deletionQueue) {
            if (!uploadPending) {
                rememberHardFailure();
                INXLOG_WARN("GPU particle resource refresh kept last-known-good emitter '", emitter.stableId,
                            "': ", error.empty() ? "failed to create a compatible RHI revision" : error);
            }
            return false;
        }

        struct RendererReplacement
        {
            size_t outputIndex = 0;
            std::shared_ptr<ParticleGpuOutputRenderer> previous;
            std::shared_ptr<ParticleGpuOutputRenderer> next;
        };
        std::vector<RendererReplacement> renderers;
        if (meshChanged) {
            for (size_t index = 0; index < emitter.outputs.size(); ++index) {
                const auto &output = emitter.outputs[index];
                if (output.type != GpuParticleOutputType::Mesh)
                    continue;
                GpuParticleOutputProgram candidate;
                candidate.id = output.id;
                candidate.stableId = output.stableId;
                candidate.type = output.type;
                candidate.mesh = output.mesh;
                candidate.material = output.material;
                candidate.shaderProgram = output.shaderProgram;
                candidate.fallbackMaterial = output.fallbackMaterial;
                candidate.semantics = output.semantics;
                candidate.ribbonUvMode = output.ribbonUvMode;
                candidate.ribbonUvScale = output.ribbonUvScale;
                candidate.flipbookColumns = output.flipbookColumns;
                candidate.flipbookRows = output.flipbookRows;
                auto next = CreateOutputRenderer(emitter, candidate, uploadPending, &error);
                if (!next) {
                    if (!uploadPending) {
                        rememberHardFailure();
                        INXLOG_WARN("GPU particle Mesh output refresh kept last-known-good emitter '", emitter.stableId,
                                    "': ", error.empty() ? "renderer creation failed" : error);
                    }
                    return false;
                }
                renderers.push_back({index, output.renderer, std::move(next)});
            }
        }

        if (!emitter.runtime->AdoptCompatibleRevision(*replacement)) {
            rememberHardFailure();
            INXLOG_WARN("GPU particle resource refresh kept last-known-good emitter '", emitter.stableId,
                        "': compatible runtime adoption failed");
            return false;
        }
        for (auto &renderer : renderers)
            emitter.outputs[renderer.outputIndex].renderer = renderer.next;

        if (!renderers.empty() && (!drawRegistry || !drawRegistry->Replace(BuildDrawEntries(emitters)))) {
            for (auto &renderer : renderers)
                emitter.outputs[renderer.outputIndex].renderer = renderer.previous;
            const bool restored = emitter.runtime->AdoptCompatibleRevision(*replacement);
            rememberHardFailure();
            INXLOG_WARN("GPU particle resource refresh kept last-known-good emitter '", emitter.stableId,
                        "': draw registry publication failed", restored ? "" : " and runtime rollback failed");
            return false;
        }

        emitter.vectorFields = std::move(vectorFields);
        emitter.vectorFieldGenerations = vectorFieldGenerations;
        emitter.observedVectorFieldGenerations = std::move(vectorFieldGenerations);
        CaptureMeshRevisions(emitter.sourceProgram, emitter.meshes, emitter.observedMeshGenerations);

        std::vector<std::shared_ptr<ParticleGpuOutputRenderer>> retiredRenderers;
        retiredRenderers.reserve(renderers.size());
        for (auto &renderer : renderers)
            retiredRenderers.push_back(std::move(renderer.previous));
        auto retiredRuntime = std::shared_ptr<ParticleGpuRuntime>(std::move(replacement));
        deletionQueue->Retire(
            [retiredRuntime = std::move(retiredRuntime), retiredRenderers = std::move(retiredRenderers)]() mutable {
                retiredRenderers.clear();
                retiredRuntime.reset();
            });
        return true;
    }

    [[nodiscard]] std::shared_ptr<GraphState> BuildGraph(const EmitterMap &candidateEmitters, std::string *error) const
    {
        auto state = std::make_shared<GraphState>();
        state->generation = nextGraphGeneration++;
        state->graph = std::make_unique<vk::RenderGraph>();
        state->graph->Initialize(context, deletionQueue);
        auto *graph = state->graph.get();
        std::map<uint64_t, std::vector<std::shared_ptr<Emitter>>> emittersByGraph;
        std::unordered_map<uint64_t, ParticleGpuGraphSpawnDomain::GraphResources> spawnResources;
        std::unordered_set<uint64_t> retainedDomains;
        for (const auto &[id, emitter] : candidateEmitters) {
            (void)id;
            if (!emitter || emitter->graphInstanceId == 0) {
                SetError(error, "GPU particle graph contains an emitter without graph ownership");
                return {};
            }
            emittersByGraph[emitter->graphInstanceId].push_back(emitter);
        }
        for (auto &[graphInstanceId, graphEmitters] : emittersByGraph) {
            std::sort(graphEmitters.begin(), graphEmitters.end(), [](const auto &lhs, const auto &rhs) {
                return lhs->sourceProgram.graphEmitterIndex < rhs->sourceProgram.graphEmitterIndex;
            });
            for (uint32_t slot = 0; slot < graphEmitters.size(); ++slot) {
                if (graphEmitters[slot]->sourceProgram.graphEmitterIndex != slot) {
                    SetError(error, "GPU particle graph emitter indices must be dense and start at zero");
                    return {};
                }
            }
            std::shared_ptr<ParticleGpuGraphSpawnDomain> domain;
            const auto &parameterWords = graphEmitters.front()->sourceProgram.parameterWords;
            if (std::any_of(graphEmitters.begin(), graphEmitters.end(), [&](const auto &emitter) {
                    return emitter->sourceProgram.parameterWords != parameterWords;
                })) {
                SetError(error, "GPU particle graph emitters disagree on the shared parameter block");
                return {};
            }
            if (graphState) {
                const auto resident = graphState->spawnDomains.find(graphInstanceId);
                if (resident != graphState->spawnDomains.end() &&
                    resident->second->SlotCount() == graphEmitters.size() &&
                    std::all_of(graphEmitters.begin(), graphEmitters.end(), [&](const auto &emitter) {
                        const auto previous = emitters.find(emitter->id);
                        return previous != emitters.end() && previous->second == emitter;
                    })) {
                    domain = resident->second;
                    retainedDomains.insert(graphInstanceId);
                }
            }
            if (!domain) {
                domain = std::make_shared<ParticleGpuGraphSpawnDomain>();
                if (!spawnProgram || !spawnProgram->IsValid() ||
                    !domain->Create(context->GetRhiDevice(), graphInstanceId,
                                    static_cast<uint32_t>(graphEmitters.size()), spawnProgram->View(),
                                    parameterWords)) {
                    SetError(error, "failed to create the GPU particle graph spawn domain");
                    return {};
                }
                for (const auto &emitter : graphEmitters) {
                    if (!domain->RegisterEmitter(emitter->sourceProgram.graphEmitterIndex, *emitter->runtime)) {
                        SetError(error, "failed to bind an emitter to the GPU particle graph spawn domain");
                        return {};
                    }
                }
            }
            const std::string prefix = "GpuParticleGraph/" + std::to_string(graphInstanceId);
            if (!domain->Attach(*graph, prefix, spawnResources[graphInstanceId])) {
                SetError(error, "failed to attach the GPU particle graph spawn prepass");
                return {};
            }
            state->spawnDomains.emplace(graphInstanceId, std::move(domain));
        }
        state->schedulers.reserve(candidateEmitters.size());
        state->schedulerById.reserve(candidateEmitters.size());
        std::vector<std::pair<uint64_t, std::shared_ptr<Emitter>>> orderedEmitters(candidateEmitters.begin(),
                                                                                   candidateEmitters.end());
        std::sort(orderedEmitters.begin(), orderedEmitters.end(), [](const auto &lhs, const auto &rhs) {
            const auto &lhsEmitter = lhs.second;
            const auto &rhsEmitter = rhs.second;
            if (lhsEmitter->graphInstanceId != rhsEmitter->graphInstanceId)
                return lhsEmitter->graphInstanceId < rhsEmitter->graphInstanceId;
            if (lhsEmitter->sourceProgram.graphEmitterIndex != rhsEmitter->sourceProgram.graphEmitterIndex)
                return lhsEmitter->sourceProgram.graphEmitterIndex < rhsEmitter->sourceProgram.graphEmitterIndex;
            return lhs.first < rhs.first;
        });
        for (const auto &[id, emitter] : orderedEmitters) {
            auto scheduler = std::make_unique<ParticleRenderGraph>();
            const std::string prefix = "GpuParticle/" + std::to_string(id);
            const auto domain = state->spawnDomains.find(emitter->graphInstanceId);
            if (domain == state->spawnDomains.end() ||
                !scheduler->Attach(*state->graph, *emitter->runtime, *emitter->bounds, *domain->second,
                                   spawnResources.at(emitter->graphInstanceId),
                                   emitter->sourceProgram.graphEmitterIndex, prefix, emitter->migration.get(),
                                   emitter->ribbonTopology.get())) {
                SetError(error, "failed to attach GPU particle emitter to the simulation graph");
                return {};
            }
            if (retainedDomains.count(emitter->graphInstanceId) != 0) {
                const auto previous = graphState->schedulerById.find(id);
                if (previous == graphState->schedulerById.end() ||
                    !scheduler->PreserveSchedulingFrom(*previous->second)) {
                    SetError(error, "failed to preserve scheduling for an unchanged GPU particle emitter");
                    return {};
                }
            }
            state->schedulerById.emplace(id, scheduler.get());
            state->schedulers.push_back(std::move(scheduler));
        }
        // Do not add complete-bipartite RenderExport→SimulationTail edges
        // across graph instances. Spawn-domain versioning already requires an
        // earlier emitter's Rendering (after export) to finish before a later
        // sibling's Init. Linking every other instance's Update to every
        // RenderReset closes a cycle as soon as two multi-emitter graphs share
        // one compile. Mixed sim/export batches are reported as unsplit.
        if (!candidateEmitters.empty() && !state->graph->Compile()) {
            SetError(error, "failed to compile the GPU particle simulation graph");
            return {};
        }
        if (!state->schedulers.empty()) {
            uint32_t firstExportPass = UINT32_MAX;
            for (const auto &scheduler : state->schedulers)
                firstExportPass = (std::min)(firstExportPass, scheduler->RenderExportPassId());
            const auto &plan = state->graph->GetSubmissionPlan();
            for (const auto &batch : plan.batches) {
                if (std::find(batch.workItems.begin(), batch.workItems.end(), firstExportPass) !=
                    batch.workItems.end()) {
                    state->renderExportBatch = batch.index;
                    break;
                }
            }
            state->asyncPhasePartitioned = state->renderExportBatch != rhi::InvalidSubmissionBatchIndex;
            for (uint32_t batchIndex = state->renderExportBatch;
                 state->asyncPhasePartitioned && batchIndex < plan.batches.size(); ++batchIndex) {
                for (const uint32_t passId : plan.batches[batchIndex].workItems) {
                    if (std::any_of(state->schedulers.begin(), state->schedulers.end(),
                                    [&](const auto &scheduler) { return scheduler->IsSimulationPass(passId); })) {
                        state->asyncPhasePartitioned = false;
                        break;
                    }
                }
            }
        }
        return state;
    }

    [[nodiscard]] std::vector<GpuParticleDrawEntry> BuildDrawEntries(const EmitterMap &candidateEmitters) const
    {
        std::vector<GpuParticleDrawEntry> entries;
        size_t outputCount = 0;
        for (const auto &[id, emitter] : candidateEmitters) {
            (void)id;
            outputCount += emitter->outputs.size();
        }
        entries.reserve(outputCount);
        for (const auto &[id, emitter] : candidateEmitters) {
            (void)id;
            for (const auto &output : emitter->outputs) {
                GpuParticleDrawEntry entry;
                entry.id = output.id;
                entry.emitterId = emitter->id;
                entry.graphInstanceId = emitter->graphInstanceId;
                entry.emitterIndex = emitter->sourceProgram.graphEmitterIndex;
                entry.outputStableId = output.stableId;
                entry.ownerObjectId = emitter->sourceProgram.ownerObjectId;
                entry.ownerLayerMask = emitter->sourceProgram.ownerLayerMask;
                entry.capacity = emitter->runtime->Capacity();
                entry.instances = emitter->runtime->InstanceBuffer();
                entry.visibility = emitter->runtime->VisibilityBuffer();
                entry.renderIndices = output.renderer->RenderIndexBuffer();
                entry.indirectArguments = output.type == GpuParticleOutputType::Ribbon
                                              ? emitter->ribbonTopology->DrawIndirectBuffer()
                                              : emitter->runtime->IndirectBuffer();
                entry.bounds = emitter->bounds->BoundsBuffer();
                entry.simulationControl = emitter->runtime->SimulationControlBuffer();
                entry.renderer = output.renderer;
                entry.cullProgram = cullProgram;
                entry.cullMode = output.type == GpuParticleOutputType::Ribbon ? GpuParticleCullMode::RibbonSegments
                                                                              : GpuParticleCullMode::Instances;
                entry.sortProgram =
                    output.type == GpuParticleOutputType::Ribbon || output.semantics.sortMode == ParticleSortMode::None
                        ? nullptr
                        : sortProgram;
                entry.semantics = output.semantics;
                if (output.type == GpuParticleOutputType::Ribbon) {
                    entry.semantics.sortMode = ParticleSortMode::None;
                    entry.semantics.castShadows = false;
                }
                entries.push_back(std::move(entry));
            }
        }
        return entries;
    }

    void Retire(std::shared_ptr<GraphState> oldGraph, EmitterMap oldEmitters)
    {
        if (oldEmitters.empty())
            return;
        if (deletionQueue) {
            deletionQueue->Retire([oldGraph = std::move(oldGraph), oldEmitters = std::move(oldEmitters)]() mutable {
                oldGraph.reset();
                oldEmitters.clear();
            });
        }
    }

    void RetireCompletedMigrations()
    {
        if (!graphState)
            return;
        std::vector<uint64_t> completedIds;
        for (const auto &[id, emitter] : emitters) {
            (void)emitter;
            const auto scheduler = graphState->schedulerById.find(id);
            if (scheduler != graphState->schedulerById.end() && scheduler->second->HasCompletedMigration())
                completedIds.push_back(id);
        }
        if (completedIds.empty())
            return;

        struct RetiredMigration
        {
            std::shared_ptr<ParticleGpuMigrator> migrator;
            std::shared_ptr<Emitter> source;
        };
        std::vector<RetiredMigration> retired;
        retired.reserve(completedIds.size());
        for (const uint64_t id : completedIds) {
            auto &emitter = emitters.at(id);
            retired.push_back({std::move(emitter->migration), std::move(emitter->migrationSource)});
        }

        auto replacementGraph = BuildGraph(emitters, nullptr);
        if (!replacementGraph) {
            for (size_t index = 0; index < completedIds.size(); ++index) {
                auto &emitter = emitters.at(completedIds[index]);
                emitter->migration = std::move(retired[index].migrator);
                emitter->migrationSource = std::move(retired[index].source);
            }
            return;
        }

        for (const uint64_t id : completedIds) {
            const auto scheduler = graphState->schedulerById.find(id);
            if (scheduler != graphState->schedulerById.end())
                (void)scheduler->second->ConsumeMigrationCompletion();
        }
        auto retiredGraph = std::move(graphState);
        graphState = std::move(replacementGraph);
        EmitterMap graphLifetime = emitters;
        if (deletionQueue) {
            deletionQueue->Retire([graph = std::move(retiredGraph), emitters = std::move(graphLifetime),
                                   migrations = std::move(retired)]() mutable {
                graph.reset();
                emitters.clear();
                migrations.clear();
            });
        }
    }
};

ParticleGpuSystemManager::ParticleGpuSystemManager() : m_impl(std::make_unique<Impl>())
{
}

ParticleGpuSystemManager::~ParticleGpuSystemManager()
{
    Shutdown();
}

bool ParticleGpuSystemManager::Initialize(
    vk::VkDeviceContext &context, vk::VkPipelineManager &pipelines, vk::VkResourceManager &resources,
    GpuRetirementQueue &deletionQueue, ParticleGpuDrawRegistry &drawRegistry,
    GpuBillboardTextureResolver textureResolver, GpuParticleVectorFieldTextureResolver vectorFieldTextureResolver,
    GpuParticleSkinnedMeshResolver skinnedMeshResolver, const GpuParticleSortProgram &sortProgram,
    const GpuParticleCullProgram &cullProgram, const GpuParticleBoundsProgram &boundsProgram,
    const GpuParticleMigrationProgram &migrationProgram, const GpuParticleSpawnProgram &spawnProgram,
    const GpuParticleRibbonProgram &ribbonTopologyProgram, const GpuParticleRibbonRenderProgram &ribbonRenderProgram,
    ShaderBytecode stateGatherShader)
{
    if (!m_impl || m_impl->context || !context.IsValid() || !boundsProgram.IsValid() || !migrationProgram.IsValid() ||
        !spawnProgram.IsValid() || ribbonTopologyProgram.IsValid() != ribbonRenderProgram.IsValid()) {
        INXLOG_ERROR("ParticleGpuSystemManager initialization contract rejected: impl=", m_impl != nullptr,
                     " already_initialized=", m_impl && m_impl->context != nullptr, " context=", context.IsValid(),
                     " bounds=", boundsProgram.IsValid(), " migration=", migrationProgram.IsValid(),
                     " spawn=", spawnProgram.IsValid(), " ribbon_topology=", ribbonTopologyProgram.IsValid(),
                     " ribbon_render=", ribbonRenderProgram.IsValid());
        return false;
    }
    m_impl->context = &context;
    m_impl->pipelines = &pipelines;
    m_impl->resources = &resources;
    m_impl->deletionQueue = &deletionQueue;
    m_impl->drawRegistry = &drawRegistry;
    m_impl->textureResolver = std::move(textureResolver);
    m_impl->vectorFieldTextureResolver = std::move(vectorFieldTextureResolver);
    m_impl->skinnedMeshResolver = std::move(skinnedMeshResolver);
    m_impl->collisionScene = std::make_unique<ParticleGpuCollisionScene>();
    if (!m_impl->collisionScene->Create(context.GetRhiDevice(), ParticleGpuCollisionScene::DefaultCapacity)) {
        INXLOG_ERROR("ParticleGpuSystemManager failed to create the collision scene");
        Shutdown();
        return false;
    }
    auto boundsStorage = std::make_shared<GpuParticleBoundsProgramStorage>();
    if (!boundsStorage->Assign(boundsProgram)) {
        INXLOG_ERROR("ParticleGpuSystemManager failed to retain the bounds program");
        Shutdown();
        return false;
    }
    m_impl->boundsProgram = std::move(boundsStorage);
    auto migrationStorage = std::make_shared<GpuParticleMigrationProgramStorage>();
    if (!migrationStorage->Assign(migrationProgram)) {
        INXLOG_ERROR("ParticleGpuSystemManager failed to retain the migration program");
        Shutdown();
        return false;
    }
    m_impl->migrationProgram = std::move(migrationStorage);
    auto spawnStorage = std::make_shared<GpuParticleSpawnProgramStorage>();
    if (!spawnStorage->Assign(spawnProgram)) {
        INXLOG_ERROR("ParticleGpuSystemManager failed to retain the graph spawn program");
        Shutdown();
        return false;
    }
    m_impl->spawnProgram = std::move(spawnStorage);
    if (stateGatherShader.wordCount != 0) {
        m_impl->stateGatherProgram = std::make_unique<StateGatherProgram>(context.GetRhiDevice(), stateGatherShader);
        if (!m_impl->stateGatherProgram->IsValid()) {
            INXLOG_ERROR("ParticleGpuSystemManager failed to create the state gather program");
            Shutdown();
            return false;
        }
    }
    if (ribbonTopologyProgram.IsValid()) {
        auto ribbonTopologyStorage = std::make_shared<GpuParticleRibbonProgramStorage>();
        auto ribbonRenderStorage = std::make_shared<GpuParticleRibbonRenderProgramStorage>();
        if (!ribbonTopologyStorage->Assign(ribbonTopologyProgram) ||
            !ribbonRenderStorage->Assign(ribbonRenderProgram)) {
            INXLOG_ERROR("ParticleGpuSystemManager failed to retain the ribbon programs");
            Shutdown();
            return false;
        }
        m_impl->ribbonTopologyProgram = std::move(ribbonTopologyStorage);
        m_impl->ribbonRenderProgram = std::move(ribbonRenderStorage);
    }
    if (cullProgram.IsValid()) {
        auto storage = std::make_shared<GpuParticleCullProgramStorage>();
        if (!storage->Assign(cullProgram)) {
            INXLOG_ERROR("ParticleGpuSystemManager failed to retain the cull program");
            Shutdown();
            return false;
        }
        m_impl->cullProgram = std::move(storage);
    }
    if (sortProgram.IsValid()) {
        auto storage = std::make_shared<GpuParticleSortProgramStorage>();
        if (!storage->Assign(sortProgram)) {
            INXLOG_ERROR("ParticleGpuSystemManager failed to retain the sort program");
            Shutdown();
            return false;
        }
        m_impl->sortProgram = std::move(storage);
    }
    std::string graphError;
    m_impl->graphState = m_impl->BuildGraph({}, &graphError);
    if (!m_impl->graphState) {
        INXLOG_ERROR("ParticleGpuSystemManager failed to create the initial graph: ", graphError);
        Shutdown();
        return false;
    }
    return true;
}

void ParticleGpuSystemManager::Shutdown() noexcept
{
    if (!m_impl)
        return;
    if (m_impl->meshUploadOwner) {
        m_impl->meshUploadOwner->RemoveSource(m_impl->meshUploadSource);
        m_impl->meshUploadOwner = nullptr;
        m_impl->meshUploadSource = 0;
    }
    if (m_impl->drawRegistry)
        m_impl->drawRegistry->Clear();
    m_impl->FailDiagnostics(0, "GPU particle manager shut down before diagnostic recording completed");
    m_impl->recordedDiagnostics.clear();
    m_impl->preparedStateCaptures.clear();
    m_impl->stateCaptures.clear();
    m_impl->graphState.reset();
    m_impl->emitters.clear();
    m_impl->requiresCollisionScene = false;
    for (auto &[key, upload] : m_impl->meshUploads)
        upload.packJob.Cancel();
    m_impl->meshUploads.clear();
    m_impl->collisionScene.reset();
    m_impl->drawRegistry = nullptr;
    m_impl->textureResolver = {};
    m_impl->vectorFieldTextureResolver = {};
    m_impl->skinnedMeshResolver = {};
    m_impl->cullProgram.reset();
    m_impl->sortProgram.reset();
    m_impl->boundsProgram.reset();
    m_impl->migrationProgram.reset();
    m_impl->spawnProgram.reset();
    m_impl->ribbonTopologyProgram.reset();
    m_impl->ribbonRenderProgram.reset();
    m_impl->stateGatherProgram.reset();
    m_impl->deletionQueue = nullptr;
    m_impl->pipelines = nullptr;
    m_impl->resources = nullptr;
    m_impl->context = nullptr;
}

void ParticleGpuSystemManager::RegisterMeshUploadWork(FrameWorkOwner &owner)
{
    auto &impl = *m_impl;
    if (impl.meshUploadOwner)
        return;

    FrameWorkSource source;
    source.observe = [&impl] { impl.PollMeshUploads(); };
    source.refresh = source.observe;
    source.describe = [&impl, &owner] {
        FrameWorkFacts facts;
        const auto order = [](uint64_t token, uint64_t cursor) { return std::pair(token <= cursor, token); };
        for (const auto &[key, upload] : impl.meshUploads) {
            if (!upload.source) {
                facts.pending |= std::any_of(upload.tickets.begin(), upload.tickets.end(),
                                             [](const auto &ticket) { return ticket && ticket->IsPending(); });
                continue;
            }
            facts.pending = true;
            if (upload.packJob.IsValid()) {
                facts.workerBusy = true;
                continue;
            }
            if (!upload.tickets[Impl::MeshUpload::Vertices]) {
                if ((!facts.host.ready ||
                     order(upload.token, impl.lastMeshPackToken) < order(facts.host.token, impl.lastMeshPackToken)) &&
                    owner.CanAdmitResource(FrameWorkResource::Cpu, impl.lastMeshPackNanoseconds, true) &&
                    owner.CanReserveCpuStorage(upload.source->WorkspaceBytes()))
                    facts.host = {upload.token, 0, true, false};
                continue;
            }
            const auto &ticket = upload.tickets[upload.NextPart()];
            if (impl.resources->PlanBufferUploadBlock(*ticket, owner, sizeof(uint32_t)) &&
                (!facts.gpu.ready ||
                 order(upload.token, impl.lastMeshUploadToken) < order(facts.gpu.token, impl.lastMeshUploadToken)))
                facts.gpu = {upload.token, 0, true, false};
        }
        if (facts.workerBusy)
            facts.host = {};
        return facts;
    };
    source.advance = [&impl, &owner](bool gpu, uint64_t token) {
        for (auto &[key, upload] : impl.meshUploads) {
            if (upload.token != token || !upload.source)
                continue;
            try {
                if (!gpu) {
                    impl.lastMeshPackToken = token;
                    return impl.StartMeshPack(upload);
                }
                impl.lastMeshUploadToken = token;
                const auto part = upload.NextPart();
                const auto bytes =
                    impl.resources->PlanBufferUploadBlock(*upload.tickets[part], owner, sizeof(uint32_t));
                return bytes && impl.AppendMeshUpload(upload, part, bytes);
            } catch (const std::exception &exception) {
                upload.Fail(std::string("GPU particle Mesh upload failed: ") + exception.what());
                return true;
            }
        }
        return false;
    };
    impl.meshUploadSource = owner.AddSource(std::move(source));
    impl.meshUploadOwner = &owner;
}

GpuParticlePublicationStatus ParticleGpuSystemManager::ApplyGraph(const GpuParticleGraphProgram &program,
                                                                  std::string *error)
{
    return ApplyGraphs(std::vector<GpuParticleGraphProgram>{program}, error);
}

GpuParticlePublicationStatus ParticleGpuSystemManager::ApplyGraphs(const std::vector<GpuParticleGraphProgram> &programs,
                                                                   std::string *error)
{
    if (error)
        error->clear();
    if (!m_impl || !m_impl->context || !m_impl->drawRegistry) {
        SetError(error, "GPU particle manager is not initialized");
        return GpuParticlePublicationStatus::Failed;
    }
    if (programs.empty()) {
        SetError(error, "GPU particle graph publication batch cannot be empty");
        return GpuParticlePublicationStatus::Failed;
    }

    Impl::EmitterMap candidates = m_impl->emitters;
    std::unordered_set<uint64_t> batchIds;
    std::unordered_set<uint64_t> graphInstanceIds;
    std::size_t operationCount = 0;
    for (const auto &program : programs)
        operationCount += program.emitters.size() + program.removeEmitterIds.size();
    batchIds.reserve(operationCount);
    graphInstanceIds.reserve(programs.size());

    for (const auto &program : programs) {
        if (program.graphInstanceId == 0 || !graphInstanceIds.insert(program.graphInstanceId).second) {
            SetError(error, "GPU particle publication batch contains an invalid or duplicate graph instance id");
            return GpuParticlePublicationStatus::Failed;
        }
        if (program.emitters.empty() && program.removeEmitterIds.empty()) {
            SetError(error, "GPU particle update batch cannot be empty");
            return GpuParticlePublicationStatus::Failed;
        }
        for (const uint64_t id : program.removeEmitterIds) {
            if (id == 0 || !batchIds.insert(id).second) {
                SetError(error, "GPU particle update batch contains invalid or duplicate removal ids");
                return GpuParticlePublicationStatus::Failed;
            }
            const auto existing = m_impl->emitters.find(id);
            if (existing != m_impl->emitters.end() && existing->second->graphInstanceId != program.graphInstanceId) {
                SetError(error, "GPU particle graph cannot remove another graph's emitter");
                return GpuParticlePublicationStatus::Failed;
            }
            candidates.erase(id);
        }
        for (const auto &emitterProgram : program.emitters) {
            if (emitterProgram.graphInstanceId != program.graphInstanceId) {
                SetError(error, "GPU particle update batch mixes multiple graph instances");
                return GpuParticlePublicationStatus::Failed;
            }
            if (!batchIds.insert(emitterProgram.id).second) {
                SetError(error, "GPU particle update batch contains duplicate or conflicting emitter ids");
                return GpuParticlePublicationStatus::Failed;
            }
            const auto previous = m_impl->emitters.find(emitterProgram.id);
            if (previous != m_impl->emitters.end() && previous->second->graphInstanceId != program.graphInstanceId) {
                SetError(error, "GPU particle emitter id is already owned by another graph");
                return GpuParticlePublicationStatus::Failed;
            }
            if (!m_impl->PreflightEmitterProgram(
                    emitterProgram,
                    previous != m_impl->emitters.end() ? previous->second : std::shared_ptr<Impl::Emitter>{}, error))
                return GpuParticlePublicationStatus::Failed;
        }
    }

    // Admit all mesh dependencies before constructing any candidate runtimes.
    // The next host opportunity can advance both uploads while the live graph remains intact.
    bool waitingForMesh = false;
    for (const auto &program : programs) {
        for (const auto &emitterProgram : program.emitters) {
            if (!m_impl->PrepareMeshUploads(emitterProgram, waitingForMesh, error))
                return GpuParticlePublicationStatus::Failed;
        }
    }
    if (waitingForMesh)
        return GpuParticlePublicationStatus::Pending;

    for (const auto &program : programs) {
        for (const auto &emitterProgram : program.emitters) {
            const auto previous = m_impl->emitters.find(emitterProgram.id);
            bool uploadPending = false;
            auto emitter = m_impl->CreateEmitter(emitterProgram,
                                                 previous != m_impl->emitters.end() ? previous->second
                                                                                    : std::shared_ptr<Impl::Emitter>{},
                                                 uploadPending, error);
            if (!emitter)
                return uploadPending ? GpuParticlePublicationStatus::Pending : GpuParticlePublicationStatus::Failed;
            candidates[emitterProgram.id] = std::move(emitter);
        }
    }

    auto candidateGraph = m_impl->BuildGraph(candidates, error);
    if (!candidateGraph)
        return GpuParticlePublicationStatus::Failed;
    if (!m_impl->drawRegistry->Replace(m_impl->BuildDrawEntries(candidates))) {
        SetError(error, "failed to publish GPU particle draw entries");
        return GpuParticlePublicationStatus::Failed;
    }

    auto oldGraph = std::move(m_impl->graphState);
    auto oldEmitters = std::move(m_impl->emitters);
    for (const auto &program : programs) {
        m_impl->FailDiagnostics(program.graphInstanceId,
                                "GPU particle graph changed before diagnostic recording completed");
    }
    m_impl->graphState = std::move(candidateGraph);
    m_impl->emitters = std::move(candidates);
    m_impl->requiresCollisionScene =
        std::any_of(m_impl->emitters.begin(), m_impl->emitters.end(),
                    [](const auto &entry) { return entry.second && entry.second->sourceProgram.collisionEnabled; });
    m_impl->Retire(std::move(oldGraph), std::move(oldEmitters));
    return GpuParticlePublicationStatus::Published;
}

bool ParticleGpuSystemManager::UpdateGraphParameters(uint64_t graphInstanceId,
                                                     const std::vector<GpuParticleParameterUpdate> &updates,
                                                     std::string *error)
{
    if (error)
        error->clear();
    if (!m_impl || !m_impl->context || graphInstanceId == 0 || updates.empty()) {
        SetError(error, "GPU particle parameter update is invalid");
        return false;
    }
    std::vector<std::shared_ptr<Impl::Emitter>> targets;
    for (const auto &[id, emitter] : m_impl->emitters) {
        (void)id;
        if (emitter && emitter->graphInstanceId == graphInstanceId)
            targets.push_back(emitter);
    }
    if (targets.empty()) {
        SetError(error, "GPU particle graph instance is not active");
        return false;
    }
    const size_t wordCount = targets.front()->sourceProgram.parameterWords.size();
    if (std::any_of(targets.begin(), targets.end(),
                    [&](const auto &emitter) {
                        return !emitter->runtime || emitter->sourceProgram.parameterWords.size() != wordCount;
                    }) ||
        std::any_of(updates.begin(), updates.end(), [&](const auto &update) {
            return uint64_t(update.wordOffset) + update.words.size() > wordCount;
        })) {
        SetError(error, "GPU particle parameter layout does not match the active graph");
        return false;
    }
    if (!m_impl->graphState) {
        SetError(error, "GPU particle graph state is unavailable");
        return false;
    }
    const auto domain = m_impl->graphState->spawnDomains.find(graphInstanceId);
    if (domain == m_impl->graphState->spawnDomains.end() || !domain->second->UpdateParameters(updates)) {
        SetError(error, "GPU particle shared parameter upload failed");
        return false;
    }
    for (const auto &emitter : targets)
        for (const auto &update : updates)
            std::copy(update.words.begin(), update.words.end(),
                      emitter->sourceProgram.parameterWords.begin() + update.wordOffset);
    return true;
}

void ParticleGpuSystemManager::NotifySubmission(bool submitted) noexcept
{
    if (!m_impl)
        return;
    if (!submitted)
        for (const auto &work : m_impl->preparedStateCaptures)
            work->capture->Cancel("GPU particle state capture submission was not accepted");
    m_impl->preparedStateCaptures.clear();
    for (auto &diagnostic : m_impl->recordedDiagnostics) {
        if (submitted) {
            m_impl->deletionQueue->Retire(std::move(diagnostic.complete));
        } else {
            std::scoped_lock lock(m_impl->diagnosticState->mutex);
            auto &snapshot = m_impl->diagnosticState->snapshots.at(diagnostic.requestId);
            snapshot.status = GpuParticleDiagnosticStatus::Failed;
            snapshot.error = "GPU particle diagnostic submission was not accepted";
        }
    }
    m_impl->recordedDiagnostics.clear();
    if (m_impl->collisionScene)
        m_impl->collisionScene->NotifySubmission(submitted);
    for (const auto &[id, emitter] : m_impl->emitters) {
        (void)id;
        emitter->runtime->NotifySubmission(submitted);
    }
    if (m_impl->graphState) {
        for (const auto &[id, domain] : m_impl->graphState->spawnDomains) {
            (void)id;
            domain->NotifySubmission(submitted);
        }
    }
}

bool ParticleGpuSystemManager::PublishCollisionScene(const GpuParticleCollisionSceneSnapshot &snapshot,
                                                     std::string *error)
{
    if (!m_impl || !m_impl->collisionScene) {
        SetError(error, "GPU particle collision scene is not initialized");
        return false;
    }
    return m_impl->collisionScene->Publish(snapshot, error);
}

uint64_t ParticleGpuSystemManager::CollisionSceneRevision() const noexcept
{
    return m_impl && m_impl->collisionScene ? m_impl->collisionScene->PublishedRevision() : 0;
}

uint32_t ParticleGpuSystemManager::CollisionSceneColliderCount() const noexcept
{
    return m_impl && m_impl->collisionScene ? m_impl->collisionScene->PublishedColliderCount() : 0;
}

bool ParticleGpuSystemManager::RequiresCollisionScene() const noexcept
{
    return m_impl && m_impl->requiresCollisionScene;
}

void ParticleGpuSystemManager::InvalidateTextureAssets(const std::string &textureGuid)
{
    if (!m_impl || textureGuid.empty())
        return;
    std::unordered_set<InxMaterial *> invalidated;
    for (const auto &[id, emitter] : m_impl->emitters) {
        for (const auto &output : emitter->outputs) {
            const auto &material = output.material;
            if (!material || material->IsDeleted() || invalidated.count(material.get()))
                continue;
            for (const auto &[name, property] : material->GetAllProperties()) {
                if (property.type == MaterialPropertyType::Texture2D &&
                    std::get<std::string>(property.value) == textureGuid) {
                    material->InvalidateTextureAssets(textureGuid, false);
                    invalidated.insert(material.get());
                    break;
                }
            }
        }
    }
}

bool ParticleGpuSystemManager::RefreshMaterialProgram(const std::shared_ptr<InxMaterial> &material,
                                                      std::shared_ptr<const ShaderProgramArtifact> shaderProgram,
                                                      std::string *error)
{
    if (error)
        error->clear();
    if (!m_impl || !m_impl->context || !m_impl->drawRegistry) {
        SetError(error, "GPU particle manager is not initialized");
        return false;
    }
    if (!material) {
        SetError(error, "GPU particle material refresh requires a live material");
        return false;
    }
    const bool referenced = std::any_of(m_impl->emitters.begin(), m_impl->emitters.end(), [&](const auto &entry) {
        return std::any_of(entry.second->outputs.begin(), entry.second->outputs.end(),
                           [&](const auto &output) { return output.material.get() == material.get(); });
    });
    if (!referenced)
        return true;
    if (shaderProgram && (!shaderProgram->IsValid() || shaderProgram->domain != ShaderProgramDomain::ParticleSprite)) {
        SetError(error, "GPU particle material must resolve to a valid linked ParticleSprite shader program");
        return false;
    }

    struct Replacement
    {
        Impl::Emitter::Output *output = nullptr;
        std::shared_ptr<const ShaderProgramArtifact> previousProgram;
        std::shared_ptr<ParticleGpuOutputRenderer> previousRenderer;
        std::shared_ptr<ParticleGpuOutputRenderer> nextRenderer;
    };
    std::vector<Replacement> replacements;
    for (const auto &[id, emitter] : m_impl->emitters) {
        (void)id;
        for (auto &output : emitter->outputs) {
            if (output.material.get() != material.get())
                continue;
            const bool sameProgram =
                output.shaderProgram && shaderProgram && output.shaderProgram->key == shaderProgram->key;
            if (sameProgram)
                continue;
            if (!shaderProgram || !shaderProgram->IsValid() ||
                shaderProgram->domain != ShaderProgramDomain::ParticleSprite) {
                SetError(error, "GPU particle surface output requires a valid linked ParticleSprite shader program");
                return false;
            }
            if (output.semantics.receiveSceneLighting &&
                !shaderProgram->FindVariant(ShaderCompileTarget::ForwardPlus)) {
                SetError(error, "GPU particle output '" + output.stableId +
                                    "' requires a linked Particle Forward+ shader variant when Receive Scene "
                                    "Lighting is enabled");
                return false;
            }
            GpuParticleOutputProgram candidate;
            candidate.id = output.id;
            candidate.stableId = output.stableId;
            candidate.type = output.type;
            candidate.mesh = output.mesh;
            candidate.material = material;
            candidate.shaderProgram = shaderProgram;
            candidate.fallbackMaterial = output.fallbackMaterial;
            candidate.semantics = output.semantics;
            candidate.flipbookColumns = output.flipbookColumns;
            candidate.flipbookRows = output.flipbookRows;
            candidate.ribbonUvMode = output.ribbonUvMode;
            candidate.ribbonUvScale = output.ribbonUvScale;
            bool uploadPending = false;
            auto renderer = m_impl->CreateOutputRenderer(*emitter, candidate, uploadPending, error);
            if (!renderer) {
                SetError(error, "failed to refresh GPU particle material for output '" + output.stableId + "'");
                return false;
            }
            replacements.push_back({&output, output.shaderProgram, output.renderer, std::move(renderer)});
        }
    }
    if (replacements.empty())
        return true;

    for (auto &replacement : replacements) {
        replacement.output->shaderProgram = shaderProgram;
        replacement.output->renderer = replacement.nextRenderer;
    }
    if (!m_impl->drawRegistry->Replace(m_impl->BuildDrawEntries(m_impl->emitters))) {
        for (auto &replacement : replacements) {
            replacement.output->shaderProgram = std::move(replacement.previousProgram);
            replacement.output->renderer = std::move(replacement.previousRenderer);
        }
        SetError(error, "failed to publish refreshed GPU particle material draw entries");
        return false;
    }

    std::vector<std::shared_ptr<ParticleGpuOutputRenderer>> retired;
    retired.reserve(replacements.size());
    for (auto &replacement : replacements)
        retired.push_back(std::move(replacement.previousRenderer));
    if (m_impl->deletionQueue) {
        m_impl->deletionQueue->Retire([retired = std::move(retired)]() mutable { retired.clear(); });
    }
    return true;
}

void ParticleGpuSystemManager::Clear()
{
    if (!m_impl || !m_impl->context || m_impl->emitters.empty())
        return;
    auto candidateGraph = m_impl->BuildGraph({}, nullptr);
    if (!candidateGraph)
        return;
    if (m_impl->drawRegistry && !m_impl->drawRegistry->Replace({}))
        return;
    auto oldGraph = std::move(m_impl->graphState);
    auto oldEmitters = std::move(m_impl->emitters);
    m_impl->FailDiagnostics(0, "GPU particle graph was cleared before diagnostic recording completed");
    m_impl->graphState = std::move(candidateGraph);
    m_impl->emitters.clear();
    m_impl->requiresCollisionScene = false;
    m_impl->Retire(std::move(oldGraph), std::move(oldEmitters));
}

bool ParticleGpuSystemManager::BeginFrame(uint64_t id, const GpuParticleFrameRequest &request,
                                          const GpuParticleTransforms &transforms)
{
    if (!m_impl)
        return false;
    const auto emitter = m_impl->emitters.find(id);
    if (emitter == m_impl->emitters.end())
        return false;
    return BeginFrameBatch(emitter->second->graphInstanceId,
                           {{id, {}, request, transforms, emitter->second->sourceProgram.ownerLayerMask}});
}

bool ParticleGpuSystemManager::BeginFrameBatch(uint64_t graphInstanceId,
                                               const std::vector<GpuParticleBatchFrameItem> &items)
{
    if (!m_impl || !m_impl->graphState || graphInstanceId == 0 || items.empty())
        return false;

    // Migration callbacks mark completion while recording the command buffer,
    // before that command buffer has a submission ticket. Rebuilding the graph
    // from RecordAsyncExport/Execute therefore allowed the migration source,
    // descriptors, and pipelines to enter the retirement queue against the
    // previous GPU epoch. A rapid ParticleGraph save could destroy those
    // resources before the freshly recorded prime submission executed. The
    // next frame boundary runs after the prior submission has been published,
    // so retirement can safely inherit its completion epoch here.
    m_impl->RetireCompletedMigrations();

    struct PreparedItem
    {
        const GpuParticleBatchFrameItem *item = nullptr;
        std::shared_ptr<Impl::Emitter> emitter;
        ParticleRenderGraph *scheduler = nullptr;
        std::vector<GpuParticleFrameRequest> sequence;
    };
    std::vector<PreparedItem> prepared;
    prepared.reserve(items.size());
    std::unordered_set<uint64_t> emitterIds;
    emitterIds.reserve(items.size());
    const uint64_t frameIndex = items.front().request.frameIndex;
    const bool resetCollisionDiagnostics = std::any_of(
        m_impl->pendingDiagnostics.begin(), m_impl->pendingDiagnostics.end(), [graphInstanceId](const auto &request) {
            return request.graphInstanceId == graphInstanceId && request.resetPending;
        });
    const bool collectCollisionDiagnostics = std::any_of(
        m_impl->pendingDiagnostics.begin(), m_impl->pendingDiagnostics.end(), [graphInstanceId](const auto &request) {
            return request.graphInstanceId == graphInstanceId && !request.resetPending &&
                   request.remainingSampleFrames != 0;
        });
    const auto spawnDomain = m_impl->graphState->spawnDomains.find(graphInstanceId);
    if (spawnDomain == m_impl->graphState->spawnDomains.end())
        return false;

    for (const auto &item : items) {
        const auto emitter = m_impl->emitters.find(item.emitterId);
        const auto scheduler = m_impl->graphState->schedulerById.find(item.emitterId);
        if (item.emitterId == 0 || !emitterIds.insert(item.emitterId).second || item.request.frameIndex != frameIndex ||
            !IsFinite(item.transforms) || emitter == m_impl->emitters.end() ||
            scheduler == m_impl->graphState->schedulerById.end() ||
            emitter->second->graphInstanceId != graphInstanceId || !emitter->second->queuedFrameRequests.empty() ||
            item.prerollRequests.size() > 4096u)
            return false;
        PreparedItem decoded{&item, emitter->second, scheduler->second, item.prerollRequests};
        decoded.sequence.push_back(item.request);
        decoded.sequence.back().collectCollisionDiagnostics = collectCollisionDiagnostics;
        decoded.sequence.back().resetCollisionDiagnostics = resetCollisionDiagnostics;
        for (size_t substep = 0; substep < decoded.sequence.size(); ++substep) {
            const auto &request = decoded.sequence[substep];
            if (request.frameIndex != frameIndex || request.substepIndex != substep ||
                !ParticleRenderGraph::IsFrameRequestValid(request))
                return false;
        }
        // A pending reset is part of the next real frame, not a synthetic
        // frame of its own. Keeping bootstrap, reset-aware spawn preparation,
        // and the resumed simulation in one RenderGraph execution guarantees
        // that their resource transitions are described by one compiled pass
        // chain. Splitting reset into an execution followed by ContinueExecution
        // let CPU-side recording state advance before the bootstrap dispatch
        // had run, which could feed stale indirect arguments to the resumed
        // update after Play -> Stop -> Play.
        prepared.push_back(std::move(decoded));
    }

    for (const auto &entry : prepared)
        (void)m_impl->RefreshResources(*entry.emitter);

    // This is the transaction boundary for the batch: all schedulers are
    // preflighted before any one of them is armed, and the graph-level spawn
    // token is published only after every BeginFrame succeeds.
    for (const auto &entry : prepared) {
        if (!entry.scheduler->CanBeginFrame(entry.sequence.front()))
            return false;
    }
    for (const auto &entry : prepared) {
        std::vector<GpuSkinnedMeshFrameData> skinnedSources;
        for (const auto &mesh : entry.emitter->sourceProgram.meshInterfaces) {
            if (!mesh.skinnedRendererHandle)
                continue;
            if (!m_impl->skinnedMeshResolver)
                return false;
            const auto snapshot = m_impl->skinnedMeshResolver(mesh.skinnedRendererHandle());
            // A scene object may be destroyed between authoring and this frame.
            // Retain the last valid pose until the graph parameter is changed;
            // never invalidate a resident bind group mid-frame.
            if (!snapshot || !snapshot->mesh || snapshot->mesh.get() != mesh.mesh.get() || !snapshot->currentPalette ||
                snapshot->currentPalette->empty())
                continue;
            skinnedSources.push_back(
                {mesh.interfaceIndex, snapshot->revision, snapshot->sourceToWorld, snapshot->currentPalette});
        }
        if (!entry.emitter->runtime->UpdateSkinnedMeshSources(skinnedSources))
            return false;
        if (!entry.emitter->runtime->UpdateTransforms(entry.item->transforms))
            return false;
    }
    std::vector<bool> previousAcceptance;
    previousAcceptance.reserve(prepared.size());
    for (const auto &entry : prepared) {
        previousAcceptance.push_back(entry.emitter->lastSimulate);
        if (!spawnDomain->second->SetEmitterAcceptingBurstRequests(entry.emitter->sourceProgram.graphEmitterIndex,
                                                                   entry.sequence.front().simulate)) {
            for (size_t rollback = 0; rollback < previousAcceptance.size() - 1; ++rollback) {
                (void)spawnDomain->second->SetEmitterAcceptingBurstRequests(
                    prepared[rollback].emitter->sourceProgram.graphEmitterIndex, previousAcceptance[rollback]);
            }
            return false;
        }
    }
    for (const auto &entry : prepared) {
        if (!entry.scheduler->BeginFrame(entry.sequence.front())) {
            for (size_t rollback = 0; rollback < prepared.size(); ++rollback) {
                (void)spawnDomain->second->SetEmitterAcceptingBurstRequests(
                    prepared[rollback].emitter->sourceProgram.graphEmitterIndex, previousAcceptance[rollback]);
            }
            return false;
        }
    }
    spawnDomain->second->MarkFramePending();
    bool ownerLayersChanged = false;
    for (const auto &entry : prepared) {
        entry.emitter->queuedFrameRequests.assign(entry.sequence.begin() + 1, entry.sequence.end());
        entry.emitter->hasFrameRequest = true;
        entry.emitter->lastFrameIndex = entry.item->request.frameIndex;
        entry.emitter->lastSpawnCount = entry.item->request.spawnCount;
        entry.emitter->lastSimulate = entry.item->request.simulate;
        entry.emitter->lastRender = entry.item->request.render;
        entry.emitter->lastOffscreenPolicy = entry.item->request.offscreenPolicy;
        entry.emitter->lastBoundsMode = entry.item->request.boundsMode;
        auto &ownerLayerMask = entry.emitter->sourceProgram.ownerLayerMask;
        if (ownerLayerMask != entry.item->ownerLayerMask) {
            ownerLayerMask = entry.item->ownerLayerMask;
            ownerLayersChanged = true;
        }
    }
    // Layer edits publish only draw metadata. The resident simulation, state
    // buffers, scheduler and emitter identities remain unchanged.
    if (ownerLayersChanged && !m_impl->drawRegistry->Replace(m_impl->BuildDrawEntries(m_impl->emitters)))
        throw std::logic_error("Resident particle layer update produced invalid draw entries");
    if (resetCollisionDiagnostics) {
        for (auto &request : m_impl->pendingDiagnostics) {
            if (request.graphInstanceId == graphInstanceId && request.resetPending)
                request.resetPending = false;
        }
    } else if (collectCollisionDiagnostics) {
        for (auto &request : m_impl->pendingDiagnostics) {
            if (request.graphInstanceId == graphInstanceId && request.remainingSampleFrames != 0)
                --request.remainingSampleFrames;
        }
    }
    return true;
}

bool ParticleGpuSystemManager::SetEmitterPlaying(uint64_t id, bool playing)
{
    if (!m_impl || !m_impl->graphState)
        return false;
    const auto emitter = m_impl->emitters.find(id);
    if (emitter == m_impl->emitters.end())
        return false;
    const auto domain = m_impl->graphState->spawnDomains.find(emitter->second->graphInstanceId);
    return domain != m_impl->graphState->spawnDomains.end() &&
           domain->second->SetEmitterPlaying(emitter->second->sourceProgram.graphEmitterIndex, playing);
}

bool ParticleGpuSystemManager::Reset(uint64_t id)
{
    if (!m_impl || !m_impl->graphState)
        return false;
    const auto scheduler = m_impl->graphState->schedulerById.find(id);
    if (scheduler == m_impl->graphState->schedulerById.end())
        return false;
    const auto emitter = m_impl->emitters.find(id);
    if (emitter == m_impl->emitters.end())
        return false;
    const auto domain = m_impl->graphState->spawnDomains.find(emitter->second->graphInstanceId);
    if (domain == m_impl->graphState->spawnDomains.end() ||
        !domain->second->SetEmitterAcceptingBurstRequests(emitter->second->sourceProgram.graphEmitterIndex, false))
        return false;
    emitter->second->lastSimulate = false;
    emitter->second->queuedFrameRequests.clear();
    scheduler->second->Reset();
    domain->second->MarkFramePending();
    m_impl->FailDiagnostics(emitter->second->graphInstanceId,
                            "GPU particle graph was reset before diagnostic recording completed");
    return true;
}

void ParticleGpuSystemManager::ResetAll()
{
    if (!m_impl || !m_impl->graphState)
        return;
    for (const auto &[id, emitter] : m_impl->emitters) {
        (void)emitter;
        (void)Reset(id);
    }
}

void ParticleGpuSystemManager::Execute(VkCommandBuffer commandBuffer)
{
    if (!m_impl || !m_impl->graphState || !m_impl->graphState->graph || commandBuffer == VK_NULL_HANDLE)
        return;
    if (!m_impl->RecordPendingUploads(commandBuffer))
        throw std::runtime_error("GPU particle input upload failed");
    bool hasPendingEmitter = std::any_of(m_impl->graphState->schedulers.begin(), m_impl->graphState->schedulers.end(),
                                         [](const auto &scheduler) { return scheduler->HasPendingFrame(); });
    uint32_t recordedSteps = 0;
    while (hasPendingEmitter && recordedSteps < Impl::MaxQueuedSimulationStepsPerSubmission) {
        if (!m_impl->RecordPendingGraphExecution(commandBuffer))
            break;
        ++recordedSteps;
        if (!m_impl->HasQueuedFrameRequests())
            break;
        if (!m_impl->ArmNextQueuedFrameRequests()) {
            INXLOG_ERROR("GPU particle preroll sequence could not arm its next fixed step");
            m_impl->ClearQueuedFrameRequests();
            break;
        }
        hasPendingEmitter = true;
        if (recordedSteps < Impl::MaxQueuedSimulationStepsPerSubmission && !m_impl->RecordPendingUploads(commandBuffer))
            throw std::runtime_error("GPU particle substep input upload failed");
    }
    m_impl->RecordDiagnostics(commandBuffer);
    m_impl->RecordStateCaptures(commandBuffer, rhi::QueueRole::Graphics);
}

void ParticleGpuSystemManager::AbortAsyncRecording() noexcept
{
    if (m_impl)
        m_impl->AbortAsyncRecording();
}

bool ParticleGpuSystemManager::CanRecordPartitioned() const noexcept
{
    if (!m_impl || !m_impl->context || !m_impl->context->HasIndependentComputeQueue() || !m_impl->graphState ||
        !m_impl->graphState->graph || !m_impl->graphState->asyncPhasePartitioned || m_impl->emitters.empty())
        return false;
    const auto &plan = m_impl->graphState->graph->GetSubmissionPlan();
    const uint32_t boundary = m_impl->graphState->renderExportBatch;
    if (boundary == rhi::InvalidSubmissionBatchIndex || boundary == 0 || boundary >= plan.batches.size())
        return false;
    return std::any_of(m_impl->graphState->schedulers.begin(), m_impl->graphState->schedulers.end(),
                       [](const auto &scheduler) { return scheduler && scheduler->HasPendingFrame(); });
}

bool ParticleGpuSystemManager::CanExecuteAsync() const noexcept
{
    return CanRecordPartitioned() && !m_impl->HasQueuedFrameRequests() &&
           std::all_of(m_impl->emitters.begin(), m_impl->emitters.end(), [](const auto &entry) {
               return !entry.second->hasFrameRequest ||
                      entry.second->lastOffscreenPolicy == GpuParticleOffscreenPolicy::AlwaysSimulate;
           });
}

bool ParticleGpuSystemManager::HasPendingGpuWork() const noexcept
{
    if (!m_impl || !m_impl->graphState || m_impl->emitters.empty())
        return false;

    if (m_impl->requiresCollisionScene && m_impl->collisionScene && m_impl->collisionScene->HasPendingUpload())
        return true;
    if (!m_impl->pendingDiagnostics.empty() || !m_impl->stateCaptures.empty() || m_impl->HasQueuedFrameRequests())
        return true;
    if (std::any_of(m_impl->emitters.begin(), m_impl->emitters.end(),
                    [](const auto &entry) { return entry.second->runtime->HasPendingUploads(); }))
        return true;
    if (std::any_of(m_impl->graphState->spawnDomains.begin(), m_impl->graphState->spawnDomains.end(),
                    [](const auto &entry) { return entry.second->HasPendingUploads(); }))
        return true;

    return std::any_of(m_impl->graphState->schedulers.begin(), m_impl->graphState->schedulers.end(),
                       [](const auto &scheduler) { return scheduler && scheduler->HasPendingFrame(); });
}

uint64_t ParticleGpuSystemManager::AsyncExecutionGeneration() const noexcept
{
    return m_impl && m_impl->graphState ? m_impl->graphState->generation : 0;
}

bool ParticleGpuSystemManager::RecordAsyncSimulation(VkCommandBuffer commandBuffer)
{
    if (!CanRecordPartitioned() || commandBuffer == VK_NULL_HANDLE)
        return false;
    if (m_impl->graphState->asyncRecordingActive)
        m_impl->AbortAsyncRecording();
    if (!m_impl->RecordPendingUploads(commandBuffer)) {
        INXLOG_ERROR("GPU particle input upload failed during async simulation");
        return false;
    }
    auto &state = *m_impl->graphState;
    state.graph->BeginExecution();
    for (uint32_t batch = 0; batch < state.renderExportBatch; ++batch) {
        if (state.graph->GetSubmissionPlan().batches[batch].queue != rhi::QueueRole::Compute ||
            !state.graph->RecordSubmissionBatch(batch, commandBuffer)) {
            INXLOG_ERROR("GPU particle async simulation rejected non-Compute batch ", batch);
            return false;
        }
    }
    state.asyncRecordingActive = true;
    return true;
}

bool ParticleGpuSystemManager::RecordAsyncExport(VkCommandBuffer commandBuffer)
{
    if (!m_impl || !m_impl->graphState || !m_impl->graphState->graph || commandBuffer == VK_NULL_HANDLE ||
        !m_impl->graphState->asyncRecordingActive)
        return false;
    auto &state = *m_impl->graphState;
    const auto &plan = state.graph->GetSubmissionPlan();
    bool recorded = true;
    for (uint32_t batch = state.renderExportBatch; batch < plan.batches.size(); ++batch) {
        if (plan.batches[batch].queue != rhi::QueueRole::Compute ||
            !state.graph->RecordSubmissionBatch(batch, commandBuffer)) {
            INXLOG_ERROR("GPU particle async export rejected non-Compute batch ", batch);
            recorded = false;
            break;
        }
    }
    state.asyncRecordingActive = false;
    if (!recorded)
        return false;
    m_impl->RecordDiagnostics(commandBuffer);
    m_impl->RecordStateCaptures(commandBuffer, rhi::QueueRole::Compute);
    if (m_impl->HasQueuedFrameRequests() && !m_impl->ArmNextQueuedFrameRequests()) {
        INXLOG_ERROR("GPU particle preroll sequence could not arm its next fixed step");
        m_impl->ClearQueuedFrameRequests();
    }
    return true;
}

bool ParticleGpuSystemManager::Contains(uint64_t id) const
{
    return m_impl && m_impl->emitters.find(id) != m_impl->emitters.end();
}

size_t ParticleGpuSystemManager::Size() const
{
    return m_impl ? m_impl->emitters.size() : 0;
}

GpuParticleTelemetrySnapshot ParticleGpuSystemManager::TelemetrySnapshot() const
{
    GpuParticleTelemetrySnapshot snapshot;
    if (!m_impl)
        return snapshot;

    snapshot.systemCount = m_impl->emitters.size();
    for (const auto &[id, emitter] : m_impl->emitters) {
        (void)id;
        snapshot.outputCount += emitter->outputs.size();
        snapshot.totalCapacity += emitter->sourceProgram.capacity;
        const auto continuation = emitter->runtime->ContinuationTelemetry();
        if (continuation.capacity > 0) {
            ++snapshot.continuationSystemCount;
            snapshot.totalContinuationCapacity += continuation.capacity;
            snapshot.maximumContinuationProgramGeneration =
                std::max(snapshot.maximumContinuationProgramGeneration, continuation.programGeneration);
            snapshot.continuationPrepareRecordCalls += continuation.prepareRecordCalls;
            snapshot.continuationClassifyRecordCalls += continuation.classifyRecordCalls;
            snapshot.continuationDispatchRecordCalls += continuation.dispatchRecordCalls;
            snapshot.continuationResetPendingCount += continuation.resetPending ? 1u : 0u;
        }
        const auto contacts = emitter->runtime->ContactTelemetry();
        if (contacts.particleCapacity > 0) {
            ++snapshot.contactRuntimeSystemCount;
            snapshot.totalContactRecordCapacity += contacts.contactRecordCapacity;
            snapshot.totalContactWorkItemCapacity += contacts.workItemCapacity;
            snapshot.totalContactResidentBytes += contacts.contactBytes + contacts.hashBytes +
                                                  contacts.particleIndexBytes + contacts.particleStateBytes +
                                                  contacts.workItemBytes + contacts.continuationSnapshotBytes +
                                                  sizeof(GpuParticleContactCounters) + sizeof(std::array<uint32_t, 4>);
            snapshot.contactPrepareRecordCalls += emitter->runtime->ContactPrepareRecordCalls();
            snapshot.contactSolveRecordCalls += emitter->runtime->ContactSolveRecordCalls();
        }
        if (emitter->hasFrameRequest)
            snapshot.lastScheduledFrame = std::max(snapshot.lastScheduledFrame, emitter->lastFrameIndex);
    }
    for (const auto &[id, emitter] : m_impl->emitters) {
        (void)id;
        if (!emitter->hasFrameRequest || emitter->lastFrameIndex != snapshot.lastScheduledFrame)
            continue;
        ++snapshot.scheduledSystemCount;
        snapshot.simulatingSystemCount += emitter->lastSimulate ? 1u : 0u;
        snapshot.renderingSystemCount += emitter->lastRender ? 1u : 0u;
        snapshot.requestedSpawnCount += emitter->lastSpawnCount;
    }
    if (m_impl->collisionScene) {
        snapshot.collisionSceneRevision = m_impl->collisionScene->PublishedRevision();
        snapshot.collisionSceneColliderCount = m_impl->collisionScene->PublishedColliderCount();
        snapshot.collisionSceneTopologyRevision = m_impl->collisionScene->PublishedTopologyRevision();
        snapshot.collisionSceneMeshVertexCount = m_impl->collisionScene->PublishedMeshVertexCount();
        snapshot.collisionSceneMeshIndexCount = m_impl->collisionScene->PublishedMeshIndexCount();
        snapshot.collisionSceneMeshBvhNodeCount = m_impl->collisionScene->PublishedMeshBvhNodeCount();
    }
    return snapshot;
}

uint64_t ParticleGpuSystemManager::ActiveArtifactRevision(uint64_t id) const
{
    if (!m_impl)
        return 0;
    const auto found = m_impl->emitters.find(id);
    return found != m_impl->emitters.end() ? found->second->artifactRevision : 0;
}

bool ParticleGpuSystemManager::ActiveStateWasPreserved(uint64_t id) const
{
    if (!m_impl)
        return false;
    const auto found = m_impl->emitters.find(id);
    return found != m_impl->emitters.end() && found->second->statePreservedOnPublish;
}

size_t ParticleGpuSystemManager::ActiveOutputCount(uint64_t id) const
{
    if (!m_impl)
        return 0;
    const auto found = m_impl->emitters.find(id);
    return found != m_impl->emitters.end() ? found->second->outputs.size() : 0;
}

uint64_t ParticleGpuSystemManager::ActiveVectorFieldGeneration(uint64_t id, uint32_t interfaceIndex) const
{
    if (!m_impl)
        return 0;
    const auto found = m_impl->emitters.find(id);
    if (found == m_impl->emitters.end() || interfaceIndex >= found->second->vectorFieldGenerations.size())
        return 0;
    return found->second->vectorFieldGenerations[interfaceIndex];
}

int32_t ParticleGpuSystemManager::ActiveOutputRenderQueue(uint64_t emitterId, uint64_t outputId) const
{
    if (!m_impl || outputId == 0)
        return -1;
    const auto emitter = m_impl->emitters.find(emitterId);
    if (emitter == m_impl->emitters.end())
        return -1;
    const auto output = std::find_if(emitter->second->outputs.begin(), emitter->second->outputs.end(),
                                     [outputId](const auto &candidate) { return candidate.id == outputId; });
    return output != emitter->second->outputs.end() ? output->renderer->RenderQueue() : -1;
}

std::optional<ParticleOutputSemantics> ParticleGpuSystemManager::ActiveOutputSemantics(uint64_t emitterId,
                                                                                       uint64_t outputId) const
{
    if (!m_impl || outputId == 0)
        return std::nullopt;
    const auto emitter = m_impl->emitters.find(emitterId);
    if (emitter == m_impl->emitters.end())
        return std::nullopt;
    const auto output = std::find_if(emitter->second->outputs.begin(), emitter->second->outputs.end(),
                                     [outputId](const auto &candidate) { return candidate.id == outputId; });
    return output != emitter->second->outputs.end() ? std::optional{output->semantics} : std::nullopt;
}

std::vector<std::shared_ptr<GpuParticleStatePort>>
ParticleGpuSystemManager::CaptureState(uint64_t graphInstanceId, uint64_t emitterId, uint64_t artifactRevision,
                                       uint32_t firstRow, uint32_t rowCount, std::vector<GpuParticleStateField> fields)
{
    const auto found = m_impl->emitters.find(emitterId);
    if (!m_impl->stateGatherProgram || found == m_impl->emitters.end() ||
        found->second->graphInstanceId != graphInstanceId || found->second->artifactRevision != artifactRevision)
        throw std::invalid_argument("GPU particle capture requires an active emitter with the requested layout");
    const auto stride = found->second->runtime->StateStride() / sizeof(uint32_t);
    const auto &limits = m_impl->context->GetDeviceProperties().limits;
    const uint64_t maxDispatchRows = uint64_t(limits.maxComputeWorkGroupCount[0]) * StateGatherProgram::WorkgroupSize;
    const uint64_t rowBytesPerWord = uint64_t(rowCount) * sizeof(uint32_t);
    if (rowCount == 0 || rowCount > maxDispatchRows || uint64_t(firstRow) + rowCount > UINT32_MAX || fields.empty() ||
        rowBytesPerWord * StateGatherProgram::IdentityWords > limits.maxStorageBufferRange)
        throw std::invalid_argument("GPU particle capture range does not fit the dispatch or state layout");
    for (const auto &field : fields) {
        if (field.wordCount == 0 || uint64_t(field.wordOffset) + field.wordCount > stride ||
            field.scalarType == rhi::ComputeScalarType::Int64 ||
            rowBytesPerWord * field.wordCount > limits.maxStorageBufferRange)
            throw std::invalid_argument("GPU particle capture field does not fit the state layout");
    }

    auto capture = std::make_shared<GpuParticleStateCapture>();
    capture->graphInstanceId = graphInstanceId;
    capture->emitterId = emitterId;
    capture->artifactRevision = artifactRevision;
    capture->requestId = m_impl->nextCaptureRequestId++;
    std::vector<std::shared_ptr<GpuParticleStatePort>> ports;
    ports.reserve(fields.size() + 1);
    capture->ports.reserve(ports.capacity());
    // Fields keep their request order, followed by the shared row identities.
    for (size_t i = 0; i <= fields.size(); ++i) {
        auto port = std::make_shared<GpuParticleStatePort>();
        port->capture = capture;
        capture->ports.push_back(port);
        ports.push_back(std::move(port));
    }
    m_impl->stateCaptures.push_back({capture, firstRow, rowCount, std::move(fields)});
    return ports;
}

void ParticleGpuSystemManager::PrepareStateCaptures(vk::VulkanFrameSubmission &frame, rhi::ComputeHost &host,
                                                    FrameWorkOwner &owner)
{
    m_impl->preparedStateCaptures.clear();
    auto &requests = m_impl->stateCaptures;
    requests.erase(std::remove_if(requests.begin(), requests.end(),
                                  [](const auto &request) {
                                      const auto capture = request.capture.lock();
                                      return !capture || capture->Done();
                                  }),
                   requests.end());
    auto &device = m_impl->context->GetRhiDevice();
    uint64_t preparedNanoseconds = 0;
    for (const auto &request : requests) {
        const auto capture = request.capture.lock();
        const auto found = m_impl->emitters.find(capture->emitterId);
        if (found == m_impl->emitters.end() || found->second->artifactRevision != capture->artifactRevision) {
            capture->Cancel("GPU particle capture layout is no longer active");
            continue;
        }
        const uint64_t identityWords = uint64_t(request.rowCount) * StateGatherProgram::IdentityWords;
        uint64_t estimatedNanoseconds = 2 * identityWords * sizeof(uint32_t);
        for (const auto &field : request.fields)
            estimatedNanoseconds += 10000 + 2 * uint64_t(request.rowCount) * field.wordCount * sizeof(uint32_t);
        if (estimatedNanoseconds > owner.ResourceState(FrameWorkResource::Gpu).limit) {
            capture->Cancel("GPU particle capture exceeds the host GPU allowance; select fewer rows or attributes");
            continue;
        }
        if (!owner.CanAdmitResource(FrameWorkResource::Gpu, preparedNanoseconds + estimatedNanoseconds))
            continue;
        auto work = std::make_shared<Impl::StateCaptureWork>(device);
        work->capture = capture;
        work->emitter = found->second;
        work->fields.reserve(request.fields.size());
        work->bindings.groups.reserve(request.fields.size());
        const auto stride = static_cast<uint32_t>(work->emitter->runtime->StateStride() / sizeof(uint32_t));
        for (const auto &field : request.fields) {
            Impl::StateCaptureDispatch dispatch{{stride, field.wordOffset, field.wordCount, 0, request.firstRow,
                                                 request.rowCount, 0,
                                                 work->fields.empty() ? 0 : StateGatherProgram::SkipIdentities},
                                                {}};
            if (!dispatch.output.Prepare(host, {uint64_t(request.rowCount) * field.wordCount, field.scalarType}))
                break;
            work->fields.push_back(std::move(dispatch));
        }
        if (work->fields.size() != request.fields.size() ||
            !work->identities.Prepare(host, {identityWords, rhi::ComputeScalarType::UInt32}))
            continue;
        frame.Retain(work);
        frame.OnAccepted([work, &owner, estimatedNanoseconds](rhi::SubmissionTicket ticket) {
            work->Publish(ticket, owner, estimatedNanoseconds);
        });
        m_impl->preparedStateCaptures.push_back(std::move(work));
        preparedNanoseconds += estimatedNanoseconds;
    }
}

uint64_t ParticleGpuSystemManager::RequestDiagnostics(uint64_t graphInstanceId, uint32_t sampleFrames,
                                                      uint32_t stateSampleCount)
{
    if (!m_impl)
        return 0;
    const uint64_t requestId = m_impl->nextDiagnosticRequestId++;
    GpuParticleDiagnosticSnapshot snapshot;
    snapshot.requestId = requestId;
    snapshot.graphInstanceId = graphInstanceId;
    const bool hasEmitter =
        graphInstanceId != 0 &&
        std::any_of(m_impl->emitters.begin(), m_impl->emitters.end(), [graphInstanceId](const auto &entry) {
            return entry.second && entry.second->graphInstanceId == graphInstanceId;
        });
    if (sampleFrames == 0 || sampleFrames > 4096u) {
        snapshot.status = GpuParticleDiagnosticStatus::Failed;
        snapshot.error = "GPU particle diagnostic sample frames must be between 1 and 4096";
    } else if (stateSampleCount > 64u) {
        snapshot.status = GpuParticleDiagnosticStatus::Failed;
        snapshot.error = "GPU particle diagnostic state sample count must be between 0 and 64";
    } else if (m_impl->pendingDiagnostics.size() >= 8) {
        snapshot.status = GpuParticleDiagnosticStatus::Failed;
        snapshot.error = "Too many GPU particle diagnostic requests are pending";
    } else if (std::any_of(
                   m_impl->pendingDiagnostics.begin(), m_impl->pendingDiagnostics.end(),
                   [graphInstanceId](const auto &request) { return request.graphInstanceId == graphInstanceId; })) {
        snapshot.status = GpuParticleDiagnosticStatus::Failed;
        snapshot.error = "A GPU particle diagnostic request is already pending for this graph";
    } else if (!hasEmitter) {
        snapshot.status = GpuParticleDiagnosticStatus::Inactive;
        snapshot.error = "GPU particle graph is not resident";
    } else if (!m_impl->context || !m_impl->deletionQueue) {
        snapshot.status = GpuParticleDiagnosticStatus::Failed;
        snapshot.error = "GPU particle diagnostics are unavailable";
    } else if (stateSampleCount != 0 && !m_impl->stateGatherProgram) {
        snapshot.status = GpuParticleDiagnosticStatus::Failed;
        snapshot.error = "GPU particle state gathering is unavailable";
    } else {
        snapshot.status = GpuParticleDiagnosticStatus::Pending;
        m_impl->pendingDiagnostics.push_back({requestId, graphInstanceId, sampleFrames, stateSampleCount, true});
    }
    std::scoped_lock lock(m_impl->diagnosticState->mutex);
    if (m_impl->diagnosticState->snapshots.size() >= 128) {
        auto oldest = m_impl->diagnosticState->snapshots.end();
        for (auto it = m_impl->diagnosticState->snapshots.begin(); it != m_impl->diagnosticState->snapshots.end();
             ++it) {
            if (it->second.status == GpuParticleDiagnosticStatus::Pending)
                continue;
            if (oldest == m_impl->diagnosticState->snapshots.end() || it->first < oldest->first)
                oldest = it;
        }
        if (oldest != m_impl->diagnosticState->snapshots.end())
            m_impl->diagnosticState->snapshots.erase(oldest);
    }
    m_impl->diagnosticState->snapshots[requestId] = std::move(snapshot);
    return requestId;
}

GpuParticleDiagnosticSnapshot ParticleGpuSystemManager::QueryDiagnostics(uint64_t requestId) const
{
    if (!m_impl || requestId == 0)
        return {};
    std::scoped_lock lock(m_impl->diagnosticState->mutex);
    const auto found = m_impl->diagnosticState->snapshots.find(requestId);
    return found != m_impl->diagnosticState->snapshots.end() ? found->second : GpuParticleDiagnosticSnapshot{};
}

} // namespace infernux::particle
