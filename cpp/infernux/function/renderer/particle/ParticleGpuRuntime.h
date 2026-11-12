#pragma once

#include "ParticleRuntimeApi.h"
#include <function/renderer/shader/ShaderProgramCode.h>

#include "ParticleGpuContactRuntime.h"
#include "ParticleGpuContinuationRuntime.h"
#include "ParticleRenderInstance.h"

#include <function/renderer/rhi/RhiCommand.h>
#include <function/renderer/rhi/RhiDevice.h>

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace infernux::particle
{

class ParticleComputePipeline;
using CompiledParticleShader = std::shared_ptr<const ShaderTargetCode>;

inline bool IsCompiledParticleShaderValid(const CompiledParticleShader &shader) noexcept
{
    return shader && shader->entries.size() == 1 && shader->entries[0].stage == rhi::ShaderStage::Compute &&
           !shader->entries[0].code.empty();
}

enum class GpuKernelStage : uint8_t
{
    Bootstrap,
    Init,
    Update,
    UpdateRenderingFused,
    ContactPrepare,
    ContactSolve,
    ContactDispatch,
    RenderReset,
    Rendering,
    Count,
};

enum class GpuParticleOffscreenPolicy : uint32_t
{
    AlwaysSimulate = 0,
    PauseWhenOffscreen = 1,
};

struct alignas(16) GpuParticleSimulationControl
{
    uint32_t anyViewVisible = 1;
    uint32_t simulationAllowed = 1;
    GpuParticleOffscreenPolicy policy = GpuParticleOffscreenPolicy::AlwaysSimulate;
    uint32_t reserved = 0;
};

struct ShaderBytecode
{
    const uint32_t *words = nullptr;
    size_t wordCount = 0;
    const char *wgsl = nullptr;
    size_t wgslByteSize = 0;

    [[nodiscard]] static ShaderBytecode FromSpirV(const uint32_t *sourceWords, size_t sourceWordCount) noexcept
    {
        return {sourceWords, sourceWordCount, nullptr, 0};
    }

    [[nodiscard]] static ShaderBytecode FromWgsl(const char *source, size_t byteSize) noexcept
    {
        return {nullptr, 0, source, byteSize};
    }

    [[nodiscard]] bool IsSpirV() const noexcept
    {
        return words != nullptr && wordCount >= 5 && words[0] == 0x07230203u && wgsl == nullptr && wgslByteSize == 0;
    }

    [[nodiscard]] bool IsWgsl() const noexcept
    {
        return wgsl != nullptr && wgslByteSize != 0 && words == nullptr && wordCount == 0;
    }

    [[nodiscard]] bool IsValid() const noexcept
    {
        return IsSpirV() || IsWgsl();
    }

    [[nodiscard]] rhi::ShaderModuleDesc ModuleDesc(bool writableStorage = true) const noexcept
    {
        return IsWgsl() ? rhi::ShaderModuleDesc::FromWgsl(wgsl, wgslByteSize, writableStorage)
                        : rhi::ShaderModuleDesc::FromSpirV(words, wordCount);
    }
};

/// Own one concrete target program. Copies preserve its format and original
/// byte length; a WGSL module is never stored in a SPIR-V word container.
class ShaderBytecodeStorage
{
  public:
    void Assign(ShaderBytecode source)
    {
        if (source.IsWgsl())
            m_code = std::string(source.wgsl, source.wgslByteSize);
        else
            m_code = std::vector<uint32_t>(source.words, source.words + source.wordCount);
    }

    [[nodiscard]] ShaderBytecode View() const noexcept
    {
        if (const auto *text = std::get_if<std::string>(&m_code))
            return ShaderBytecode::FromWgsl(text->data(), text->size());
        const auto &words = std::get<std::vector<uint32_t>>(m_code);
        return ShaderBytecode::FromSpirV(words.data(), words.size());
    }

  private:
    std::variant<std::vector<uint32_t>, std::string> m_code;
};

struct GpuMeshInterfaceDesc
{
    std::string stableId;
    uint32_t interfaceIndex = 0;
    uint32_t metadataOffsetWords = 0;
    uint32_t vertexBinding = 1;
    uint32_t triangleBinding = 2;
    uint32_t influenceBinding = 3;
    uint32_t paletteBinding = 4;
    bool worldSpace = false;
    std::array<float, 16> meshToSpace = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
    uint32_t edgeCount = 0;
    uint32_t boneCount = 0;
    uint64_t poseRevision = 0;
    rhi::BufferHandle vertices;
    rhi::BufferHandle triangles;
    rhi::BufferHandle influences;
    rhi::BufferHandle palette;
    uint64_t vertexBufferBytes = 0;
    uint64_t triangleBufferBytes = 0;
    uint64_t influenceBufferBytes = 0;
    std::vector<glm::mat4> initialPalette;
    std::shared_ptr<const void> keepAlive;
};

struct GpuSkinnedMeshFrameData
{
    uint32_t interfaceIndex = 0;
    uint64_t poseRevision = 0;
    std::array<float, 16> sourceToWorld{};
    std::shared_ptr<const std::vector<glm::mat4>> currentPalette;
};

struct GpuVectorFieldDesc
{
    enum class Kind : uint8_t
    {
        VectorField,
        SignedDistanceField,
    };

    Kind kind = Kind::VectorField;
    uint32_t interfaceIndex = 0;
    uint32_t textureBinding = 0;
    bool worldSpace = true;
    std::array<float, 16> fieldToSpace = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    float vectorScale = 1.0f;
    rhi::TextureViewHandle texture;
    rhi::SamplerHandle sampler;
    std::shared_ptr<const void> keepAlive;
};

struct GpuTexture2DParameterDesc
{
    uint32_t resourceIndex = 0;
    uint32_t parameterSlot = 0;
    uint32_t textureBinding = 0;
    rhi::TextureViewHandle texture;
    rhi::SamplerHandle sampler;
    std::shared_ptr<const void> keepAlive;
};

struct GpuVectorFieldLayoutDesc
{
    uint32_t metadataBinding = 0;
    uint32_t interfaceStrideWords = 32;
    std::vector<GpuVectorFieldDesc> vectorFields;
    std::vector<GpuTexture2DParameterDesc> textureParameters;
};

struct GpuEmitterDesc
{
    uint32_t capacity = 0;
    uint32_t stateStride = 0;
    uint32_t eventTypeCount = 0;
    bool collisionEnabled = false;
    bool supportsFusedUpdateRendering = false;
    std::array<ShaderBytecode, static_cast<size_t>(GpuKernelStage::Count)> kernels{};
    std::vector<GpuMeshInterfaceDesc> meshInterfaces;
    GpuVectorFieldLayoutDesc vectorFields;
    rhi::BufferHandle collisionSceneHeader;
    rhi::BufferHandle collisionSceneColliders;
    rhi::BufferHandle collisionSceneGridOffsets;
    rhi::BufferHandle collisionSceneGridColliderIndices;
    rhi::BufferHandle collisionSceneMeshVertices;
    rhi::BufferHandle collisionSceneMeshIndices;
    rhi::BufferHandle collisionSceneMeshBvhNodes;
    /// Zero capacity means that the compiled emitter contains no suspension
    /// points and therefore pays no continuation resource cost.
    GpuParticleContinuationDesc continuation;
};

struct alignas(16) GpuParticleTransforms
{
    std::array<float, 16> emitterToWorld{};
    std::array<float, 16> worldToEmitter{};
    std::array<float, 16> simulationToWorld{};
    std::array<float, 16> worldToSimulation{};
};

struct GpuParticlePushConstants
{
    uint32_t capacity = 0;
    uint32_t invocationCount = 0;
    uint32_t spawnBaseId = 0;
    uint32_t spawnGeneration = 0;
    uint32_t systemSeed = 0;
    uint32_t simulationStep = 0;
    float deltaTime = 0.0f;
    uint32_t diagnosticFlags = 0;
    uint32_t aliveReadSlot = 0;
    uint32_t aliveWriteSlot = 1;
    uint32_t useAliveList = 0;
    uint32_t reserved = 0;
};

class ParticleGpuRuntime
{
  public:
    static constexpr uint32_t WorkgroupSize = 256;
    static constexpr uint32_t RenderInstanceStride = sizeof(GpuParticleRenderInstance);
    static constexpr uint32_t VisibilityInstanceStride = sizeof(GpuParticleVisibilityInstance);
    static constexpr uint32_t BaseCounterWordCount = 13;
    static constexpr uint32_t EventCounterWordOffset = BaseCounterWordCount;

    INFERNUX_PARTICLE_RUNTIME_API ParticleGpuRuntime();
    INFERNUX_PARTICLE_RUNTIME_API ~ParticleGpuRuntime();

    ParticleGpuRuntime(const ParticleGpuRuntime &) = delete;
    ParticleGpuRuntime &operator=(const ParticleGpuRuntime &) = delete;
    ParticleGpuRuntime(ParticleGpuRuntime &&) = delete;
    ParticleGpuRuntime &operator=(ParticleGpuRuntime &&) = delete;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool Create(rhi::Device &device, const GpuEmitterDesc &desc);
    /// Build a new kernel/pipeline revision over the same GPU-resident state.
    /// Capacity and state ABI must match exactly.
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool CreateCompatible(rhi::Device &device, const GpuEmitterDesc &desc,
                                                                      const ParticleGpuRuntime &previous);
    /// Atomically install a compatible pipeline/data-interface revision while
    /// leaving this runtime object and its GPU-resident particle state stable.
    /// The replacement receives the retired resources and must remain alive
    /// until all in-flight frames that referenced them have completed.
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool AdoptCompatibleRevision(ParticleGpuRuntime &replacement) noexcept;
    INFERNUX_PARTICLE_RUNTIME_API void Destroy() noexcept;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsValid() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool SharesStateWith(const ParticleGpuRuntime &other) const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool NeedsBootstrap() const noexcept;
    INFERNUX_PARTICLE_RUNTIME_API void RequestBootstrap() noexcept;
    INFERNUX_PARTICLE_RUNTIME_API void MarkStateInitialized() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool UpdateTransforms(const GpuParticleTransforms &transforms);
    /// Refresh scene-owned skinned Mesh parameters without rebuilding the
    /// particle graph or re-uploading immutable geometry.
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool
    UpdateSkinnedMeshSources(const std::vector<GpuSkinnedMeshFrameData> &sources);
    /// Capture prepared CPU inputs before simulation; acknowledge the actual
    /// submission result separately so rejected recordings retain their data.
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool HasPendingUploads() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool RecordPendingUploads(const rhi::TransferCommandEncoder &encoder);
    INFERNUX_PARTICLE_RUNTIME_API void NotifySubmission(bool submitted) noexcept;

    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool RecordBootstrap(const rhi::ComputeCommandEncoder &encoder,
                                                                     uint32_t systemSeed,
                                                                     rhi::BindGroupHandle graphSpawnGroup);
    INFERNUX_PARTICLE_RUNTIME_API void
    RecordInitIndirect(const rhi::ComputeCommandEncoder &encoder, uint32_t cpuSpawnCount, uint32_t spawnBaseId,
                       uint32_t spawnGeneration, uint32_t systemSeed, uint32_t simulationStep, float deltaTime,
                       rhi::BindGroupHandle graphSpawnGroup, rhi::BufferHandle spawnMetadata,
                       uint64_t indirectOffset) const;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool RecordUpdate(const rhi::ComputeCommandEncoder &encoder,
                                                                  uint32_t systemSeed, uint32_t simulationStep,
                                                                  float deltaTime, rhi::BindGroupHandle graphSpawnGroup,
                                                                  bool collectCollisionDiagnostics = false) const;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool
    RecordUpdateRenderingFused(const rhi::ComputeCommandEncoder &encoder, uint32_t systemSeed, uint32_t simulationStep,
                               float deltaTime, rhi::BindGroupHandle graphSpawnGroup) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordContactPrepare(const rhi::ComputeCommandEncoder &encoder,
                                                            uint32_t simulationStep,
                                                            rhi::BindGroupHandle graphSpawnGroup,
                                                            bool resetAll = false) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordContactSolve(const rhi::ComputeCommandEncoder &encoder,
                                                          uint32_t simulationStep,
                                                          rhi::BindGroupHandle graphSpawnGroup) const;
    INFERNUX_PARTICLE_RUNTIME_API void RecordContactDispatch(const rhi::ComputeCommandEncoder &encoder,
                                                             uint32_t systemSeed, uint32_t simulationStep,
                                                             float deltaTime, rhi::BindGroupHandle graphSpawnGroup,
                                                             bool collectCollisionDiagnostics = false) const;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool RecordRenderReset(const rhi::ComputeCommandEncoder &encoder,
                                                                       rhi::BindGroupHandle graphSpawnGroup,
                                                                       bool resetCollisionDiagnostics = false,
                                                                       bool prepareAliveWrite = false) const;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool RecordRendering(const rhi::ComputeCommandEncoder &encoder,
                                                                     uint32_t systemSeed, uint32_t simulationStep,
                                                                     rhi::BindGroupHandle graphSpawnGroup) const;
    [[nodiscard]] bool SupportsFusedUpdateRendering() const noexcept
    {
        return m_supportsFusedUpdateRendering;
    }
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool HasContinuations() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API const GpuParticleContinuationResources &
    ContinuationResources() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API GpuParticleContinuationTelemetry ContinuationTelemetry() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool
    SharesContinuationStateWith(const ParticleGpuRuntime &other) const noexcept;
    INFERNUX_PARTICLE_RUNTIME_API void RequestContinuationReset() noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool
    RecordContinuationPrepare(const rhi::ComputeCommandEncoder &encoder, uint32_t simulationStep,
                              uint64_t elapsedTimeTicks);
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool
    RecordContinuationClassify(const rhi::ComputeCommandEncoder &encoder, uint32_t simulationStep,
                               uint64_t elapsedTimeTicks) const;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool
    RecordContinuationDispatch(const rhi::ComputeCommandEncoder &encoder, uint32_t simulationStep,
                               uint64_t elapsedTimeTicks, uint32_t systemSeed, float deltaTime,
                               rhi::BindGroupHandle graphSpawnGroup) const;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool HasContactRuntime() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API const GpuParticleContactResources &ContactResources() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API GpuParticleContactTelemetry ContactTelemetry() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool
    SharesContactStateWith(const ParticleGpuRuntime &other) const noexcept;
    [[nodiscard]] uint64_t ContactPrepareRecordCalls() const noexcept
    {
        return m_contactPrepareRecordCalls;
    }
    [[nodiscard]] uint64_t ContactSolveRecordCalls() const noexcept
    {
        return m_contactSolveRecordCalls;
    }
    [[nodiscard]] uint64_t ContactDispatchRecordCalls() const noexcept
    {
        return m_contactDispatchRecordCalls;
    }

    [[nodiscard]] uint32_t Capacity() const noexcept
    {
        return m_capacity;
    }
    [[nodiscard]] uint32_t StateStride() const noexcept
    {
        return m_stateStride;
    }
    [[nodiscard]] uint32_t EventTypeCount() const noexcept
    {
        return m_eventTypeCount;
    }
    [[nodiscard]] bool CollisionEnabled() const noexcept
    {
        return m_collisionEnabled;
    }
    [[nodiscard]] uint64_t CounterBufferByteSize() const noexcept
    {
        const uint64_t eventCounterWords = static_cast<uint64_t>(m_eventTypeCount) * 3u;
        const uint64_t totalWords = BaseCounterWordCount + eventCounterWords;
        return ((totalWords + 3u) / 4u) * 16u;
    }
    /// The exact Mesh storage bindings consumed by particle kernels. Their
    /// byte ranges are shared by descriptor creation and RenderGraph imports.
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API const std::vector<rhi::BufferBinding> &
    MeshBufferBindings() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle StateBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle FreeListBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle CounterBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle InstanceBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle VisibilityBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle IndirectBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle RenderIndexBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle TransformBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle SimulationControlBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle AliveIndexBuffer(uint32_t slot) const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle AliveDispatchBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API rhi::BufferHandle AliveControlBuffer() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API uint32_t AliveReadSlot() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API uint32_t AliveWriteSlot() const noexcept;
    [[nodiscard]] INFERNUX_PARTICLE_RUNTIME_API bool IsAliveListReady() const noexcept;
    INFERNUX_PARTICLE_RUNTIME_API void PublishAliveWrite() noexcept;
    [[nodiscard]] rhi::BindingLayoutHandle GraphSpawnLayout() const noexcept
    {
        return m_graphSpawnLayout;
    }

  private:
    struct ResidentState;
    struct DataInterfaceState;
    struct VectorFieldState;

    [[nodiscard]] bool CreateInternal(rhi::Device &device, const GpuEmitterDesc &desc,
                                      std::shared_ptr<ResidentState> residentState,
                                      const ParticleGpuContinuationRuntime *previousContinuation,
                                      const ParticleGpuContactRuntime *previousContacts);
    [[nodiscard]] bool UpdateVectorFieldMetadata(const GpuParticleTransforms &transforms);
    [[nodiscard]] bool UpdateMeshInterfaceMetadata(const GpuParticleTransforms &transforms);
    void PrepareUpload(rhi::BufferHandle buffer, const void *data, size_t byteSize);
    bool Record(const rhi::ComputeCommandEncoder &encoder, GpuKernelStage stage,
                const GpuParticlePushConstants &constants, uint32_t invocationCount,
                rhi::BindGroupHandle graphSpawnGroup, rhi::BufferHandle indirectArguments = {},
                uint64_t indirectOffset = 0) const;
    [[nodiscard]] static uint32_t GroupCount(uint32_t invocationCount) noexcept;
    [[nodiscard]] uint32_t BindingLayoutCount() const noexcept;

    rhi::Device *m_device = nullptr;
    uint32_t m_capacity = 0;
    uint32_t m_stateStride = 0;
    uint32_t m_eventTypeCount = 0;
    bool m_collisionEnabled = false;
    bool m_supportsFusedUpdateRendering = false;
    mutable uint64_t m_contactPrepareRecordCalls = 0;
    mutable uint64_t m_contactSolveRecordCalls = 0;
    mutable uint64_t m_contactDispatchRecordCalls = 0;
    std::shared_ptr<ResidentState> m_residentState;
    std::unique_ptr<DataInterfaceState> m_dataInterfaces;
    std::unique_ptr<VectorFieldState> m_vectorFields;
    std::unique_ptr<ParticleGpuContinuationRuntime> m_continuation;
    std::unique_ptr<ParticleGpuContactRuntime> m_contacts;
    rhi::BindingLayoutHandle m_emptyDataInterfaceLayout;
    rhi::BindGroupHandle m_emptyDataInterfaceGroup;
    rhi::BindingLayoutHandle m_graphSpawnLayout;
    rhi::BindingLayoutHandle m_layout;
    rhi::BindGroupHandle m_group;
    rhi::BindingLayoutHandle m_collisionSceneLayout;
    rhi::BindGroupHandle m_collisionSceneGroup;
    std::array<rhi::ComputePipelineHandle, static_cast<size_t>(GpuKernelStage::Count)> m_pipelines{};
    GpuParticleTransforms m_cachedTransforms{};
    bool m_hasCachedTransforms = false;
    struct PendingUpload
    {
        rhi::BufferHandle buffer;
        std::vector<uint8_t> bytes;
        uint64_t revision = 0;
        uint64_t recordedRevision = 0;
        uint64_t submittedRevision = 0;
    };
    std::vector<PendingUpload> m_pendingUploads;
};

static_assert(sizeof(GpuParticleTransforms) == 256);
static_assert(sizeof(GpuParticlePushConstants) == 48);
static_assert(sizeof(GpuParticleSimulationControl) == 16);

} // namespace infernux::particle
