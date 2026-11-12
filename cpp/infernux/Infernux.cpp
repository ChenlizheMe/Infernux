#include <function/renderer/gui/PreviewImage.h>
/**
 * @file Infernux.cpp
 * @brief Infernux — Core lifecycle, resources, renderer init, gizmos, material pipeline
 *
 * Editor camera control → InfernuxCamera.cpp
 * Scene picking / raycasting → ScenePicker.cpp
 */

#include "Infernux.h"
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/renderer/vk/VulkanBackendContext.h>
#include <function/resources/AssetImporter/PluginPageTextureMetadata.h>
#include <function/resources/InxTexture/TextureDecoder.h>
// Explicit includes for types now only forward-declared in InxRenderer.h
#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <function/audio/AudioClipLoader.h>
#include <function/audio/AudioEngine.h>
#include <function/audio/AudioSource.h>
#include <function/editor/SelectionOutline.h>
#include <function/renderer/EditorGizmos.h>
#include <function/renderer/GizmosDrawCallBuffer.h>
#include <function/renderer/SceneRenderGraph.h>
#include <function/renderer/ScriptableRenderContext.h>
#include <function/renderer/gui/EditorGuiLayoutReset.h>
#include <function/renderer/gui/InxGUIContext.h>
#include <function/renderer/gui/InxResourcePreviewer.h>
#include <function/renderer/gui/InxScreenUIRenderer.h>
#include <function/renderer/gui/MeshPreviewPreparation.h>
#include <function/renderer/vk/VkResourceManager.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetFormatRegistry.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/AssetRegistry/AssetSource.h>
#include <function/resources/InxFileLoader/InxDefaultLoader.hpp>
#include <function/resources/InxFileLoader/InxPythonScriptLoader.hpp>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxFileLoader/InxTextureLoader.hpp>
#include <function/resources/InxMaterial/MaterialLoader.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/InxTexture/TextureLoader.h>
#include <function/resources/PhysicMaterial/PhysicMaterialLoader.h>
#include <function/resources/RenderTexture/RenderTextureLoader.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/scene/Collider.h>
#include <function/scene/Component.h>
#include <function/scene/ComponentFactory.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/PrimitiveMeshes.h>
#include <function/scene/physics/PhysicsWorld.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <imgui_internal.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <platform/filesystem/DocumentStore.h>
#include <platform/filesystem/IoAffinity.h>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <core/config/InxPlatform.h>
#include <core/threading/JobSystem.h>
#include <function/scene/SkeletonInstance.h>
#include <function/scene/TransformECSStore.h>

namespace infernux
{

struct TimelineCubePreviewCandidate
{
    InxMesh mesh{"__timeline_cube__"};
    std::shared_ptr<PreviewGeometryUpload> geometry;
    std::vector<std::shared_ptr<InxMaterial>> materials;
    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::vec3 camera{0.0f};
    int size = 0;
    uint64_t hash = 0;
};

struct Infernux::TexturePreviewSource
{
    TextureDecoder::SourcePreviewPlan plan;
    FileInput::Revision revision;
};

namespace
{

std::mutex g_runtimeOwnerMutex;
Infernux *g_runtimeOwner = nullptr;

class ActiveFrameScope
{
  public:
    explicit ActiveFrameScope(bool &active) : m_active(active), m_previous(active)
    {
        m_active = true;
    }
    ~ActiveFrameScope()
    {
        m_active = m_previous;
    }

  private:
    bool &m_active;
    bool m_previous;
};

class FrameOpportunityScope
{
  public:
    explicit FrameOpportunityScope(FrameWorkOwner &owner) : m_owner(owner)
    {
    }
    ~FrameOpportunityScope()
    {
        m_owner.EndOpportunity();
    }

  private:
    FrameWorkOwner &m_owner;
};

using json = nlohmann::json;

struct AcceptanceClock
{
    bool enabled = false;
    float fixedDeltaSeconds = 0.0f;
    uint64_t pauseAfterFrame = 0;
};

AcceptanceClock ReadPlayerAcceptanceClock()
{
    const char *fixedDeltaText = std::getenv("_INFERNUX_PLAYER_FIXED_DELTA");
    const char *pauseFrameText = std::getenv("_INFERNUX_PLAYER_PAUSE_AFTER_FRAME");
    const bool hasFixedDelta = fixedDeltaText != nullptr && fixedDeltaText[0] != '\0';
    const bool hasPauseFrame = pauseFrameText != nullptr && pauseFrameText[0] != '\0';
    if (!hasFixedDelta && !hasPauseFrame)
        return {};
    if (std::getenv("_INFERNUX_PLAYER_DEBUG_BUILD") == nullptr ||
        std::string_view(std::getenv("_INFERNUX_PLAYER_DEBUG_BUILD")) != "1") {
        throw std::logic_error("Player acceptance clock is available only in a PlayerDebug build");
    }
    if (!hasFixedDelta || !hasPauseFrame) {
        throw std::invalid_argument("Player acceptance clock requires both _INFERNUX_PLAYER_FIXED_DELTA and "
                                    "_INFERNUX_PLAYER_PAUSE_AFTER_FRAME");
    }

    errno = 0;
    char *fixedDeltaEnd = nullptr;
    const float fixedDelta = std::strtof(fixedDeltaText, &fixedDeltaEnd);
    if (errno != 0 || fixedDeltaEnd == fixedDeltaText || *fixedDeltaEnd != '\0' || !std::isfinite(fixedDelta) ||
        fixedDelta <= 0.0f || fixedDelta > 0.25f) {
        throw std::invalid_argument("_INFERNUX_PLAYER_FIXED_DELTA must be finite and in (0, 0.25]");
    }

    uint64_t pauseAfterFrame = 0;
    const char *pauseFrameEnd = pauseFrameText + std::strlen(pauseFrameText);
    const auto [parsedEnd, parseError] = std::from_chars(pauseFrameText, pauseFrameEnd, pauseAfterFrame, 10);
    if (parseError != std::errc{} || parsedEnd != pauseFrameEnd || pauseAfterFrame == 0) {
        throw std::invalid_argument("_INFERNUX_PLAYER_PAUSE_AFTER_FRAME must be a positive integer");
    }
    return {true, fixedDelta, pauseAfterFrame};
}

bool IsDirectStructuredStage(const ShaderDescriptor &descriptor)
{
    return std::any_of(
        descriptor.capabilities.begin(), descriptor.capabilities.end(), [](const std::string &capability) {
            std::string normalized = capability;
            std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                           [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
            return normalized == "fullscreen" || normalized == "standalone";
        });
}

std::optional<uint64_t> ParseComponentId(std::string_view value)
{
    uint64_t id = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), id);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || id == 0)
        return std::nullopt;
    return id;
}

template <typename T> T *ResolveDependentComponent(std::string_view guid)
{
    const auto id = ParseComponentId(guid);
    return id ? dynamic_cast<T *>(Component::FindByComponentId(*id)) : nullptr;
}

void RegisterPhysicMaterialAssetCallback()
{
    AssetDependencyGraph::Instance().RegisterCallback(
        ResourceType::PhysicMaterial,
        [](const std::string &dependentGuid, const std::string & /*materialGuid*/, AssetEvent event) {
            auto *collider = ResolveDependentComponent<Collider>(dependentGuid);
            if (!collider)
                return;
            collider->OnPhysicMaterialAssetEvent(event);
        });
}

bool ParseModelEmbeddedMaterialSlot(const std::string &path, std::string &outModel, int &outSlot)
{
    constexpr const char kTok[] = "::submat:";
    const size_t pos = path.find(kTok);
    if (pos == std::string::npos)
        return false;
    outModel = path.substr(0, pos);
    const char *begin = path.data() + pos + sizeof(kTok) - 1;
    const char *end = path.data() + path.size();
    const auto parsed = std::from_chars(begin, end, outSlot);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        return false;
    return !outModel.empty() && outSlot >= 0;
}

} // namespace

struct LinkedShaderProgramLoadTicket::SourceIdentity
{
    uint64_t sourceStamp = 0;
    uint64_t sourceEnvironmentRevision = 0;
    std::shared_ptr<const InxResourceMeta> vertexMetadata;
    std::shared_ptr<const InxResourceMeta> fragmentMetadata;

    bool operator==(const SourceIdentity &other) const noexcept
    {
        return sourceStamp == other.sourceStamp && sourceEnvironmentRevision == other.sourceEnvironmentRevision &&
               vertexMetadata == other.vertexMetadata && fragmentMetadata == other.fragmentMetadata;
    }

    bool operator!=(const SourceIdentity &other) const noexcept
    {
        return !(*this == other);
    }
};

struct LinkedShaderProgramLoadTicket::State
{
    struct Work
    {
        ShaderStagePair stages;
        std::string vertexPath;
        std::string fragmentPath;
        std::string vertexSource;
        std::string fragmentSource;
        std::shared_ptr<const InxResourceMeta> vertexMetadata;
        std::shared_ptr<const InxResourceMeta> fragmentMetadata;
        std::shared_ptr<const SourceIdentity> identity;
        uint64_t sourceStamp = 0;
        bool directStructuredStage = false;
        bool skipScenePrewarm = false;
        ShaderDescriptor fragmentDescriptor;
        LinkedShaderProgramArtifactCompilation compilation;
        std::string error;
    };

    struct MaterialBinding
    {
        std::string guid;
        ShaderAssetReference vertex;
        ShaderAssetReference fragment;
    };
    std::vector<Work> work;
    std::vector<MaterialBinding> materials;
    std::shared_ptr<const InxShaderLoader::SourceDependencyPublication::Prepared> dependencies;
    const Infernux *owner = nullptr;
    uint64_t sourceEnvironmentRevision = 0;
    JobHandle job;
    std::exception_ptr failure;
    std::thread::id ownerThread;
    std::thread::id producerThread;
    std::atomic<bool> cancelRequested{false};
    bool committed = false;
    bool superseded = false;
};

bool LinkedShaderProgramLoadTicket::IsComplete() const noexcept
{
    return !m_state || !m_state->job.IsValid() || m_state->job.IsComplete();
}

bool LinkedShaderProgramLoadTicket::IsCommitted() const noexcept
{
    return m_state && m_state->committed;
}

bool LinkedShaderProgramLoadTicket::IsSuperseded() const noexcept
{
    return m_state && m_state->superseded;
}

bool LinkedShaderProgramLoadTicket::WasProducedOnWorker() const noexcept
{
    return IsComplete() && m_state && m_state->producerThread != std::thread::id{} &&
           m_state->producerThread != m_state->ownerThread;
}

bool LinkedShaderProgramLoadTicket::Cancel() noexcept
{
    if (!m_state)
        return false;
    m_state->cancelRequested.store(true, std::memory_order_release);
    return !m_state->job.IsValid() || m_state->job.Cancel();
}

// ----------------------------------
// Helper method for validation
// ----------------------------------

bool Infernux::CheckEngineValid(const char *operation) const
{
    if (m_isCleanedUp) {
        INXLOG_ERROR("Cannot ", operation, ": Engine has been cleaned up.");
        return false;
    }
    if (m_isCleaningUp) {
        INXLOG_ERROR("Cannot ", operation, ": Engine is cleaning up.");
        return false;
    }
    return true;
}

// ----------------------------------
// Resources handling
// ----------------------------------

AssetDatabase *Infernux::GetAssetDatabase() const
{
    // An uninitialized instance owns its local database, never another
    // runtime's registry. Initialization transfers this unique owner once.
    if (m_assetDatabase)
        return m_assetDatabase.get();
    return m_ownsRuntimeServices ? AssetRegistry::Instance().GetAssetDatabase() : nullptr;
}

std::vector<AssetRuntimeRecord> Infernux::GetAssetRuntimeRecords() const
{
    const auto &registry = AssetRegistry::Instance();
    std::unordered_map<std::string, AssetRuntimeRecord> records;
    for (const auto &published : registry.GetAllPublishedAssetVersions()) {
        auto &record = records[published.guid];
        record.guid = published.guid;
        record.type = published.type;
        record.runtimeVersion = published.runtimeVersion;
    }
    for (const auto &cpu : registry.GetAllAssetResidency()) {
        auto &record = records[cpu.guid];
        record.guid = cpu.guid;
        record.type = cpu.type;
        record.runtimeTypeName = cpu.runtimeTypeName;
        record.runtimeVersion = cpu.runtimeVersion;
        record.cpuResident = true;
        record.cpuBytes = cpu.cpuBytes;
        record.explicitCpuPinCount = cpu.explicitPinCount;
        record.externalCpuReferenceCount = cpu.externalReferenceCount;
        record.cpuEvictable = cpu.evictable;
    }
    if (const auto *renderer = GetRenderer()) {
        for (const auto &gpu : renderer->GetAssetGpuResidency()) {
            auto &record = records[gpu.guid];
            record.guid = gpu.guid;
            if (record.runtimeVersion == 0) {
                record.runtimeVersion = gpu.runtimeVersion;
                record.type = gpu.domain == GpuAssetDomain::Mesh ? ResourceType::Mesh : ResourceType::Texture;
            }
            if (gpu.runtimeVersion != record.runtimeVersion) {
                record.staleGpuBytes += gpu.residentBytes;
                ++record.staleGpuAllocationCount;
                record.gpuVersionSynchronized = false;
                continue;
            }
            if (gpu.pending)
                record.gpuPendingBytes += gpu.residentBytes;
            else
                record.gpuResidentBytes += gpu.residentBytes;
            ++record.gpuAllocationCount;
            record.gpuPinned = record.gpuPinned || gpu.pinned;
        }
    }

    std::vector<AssetRuntimeRecord> result;
    result.reserve(records.size());
    for (auto &[guid, record] : records) {
        (void)guid;
        result.push_back(std::move(record));
    }
    std::sort(result.begin(), result.end(), [](const auto &left, const auto &right) { return left.guid < right.guid; });
    return result;
}

std::shared_ptr<LinkedShaderProgramLoadTicket>
Infernux::BeginPrepareLinkedShaderPrograms(const std::vector<std::string> &materialGuids)
{
    auto ticket = std::make_shared<LinkedShaderProgramLoadTicket>();
    ticket->m_state = std::make_shared<LinkedShaderProgramLoadTicket::State>();
    auto state = ticket->m_state;
    state->ownerThread = std::this_thread::get_id();
    state->owner = this;
    state->sourceEnvironmentRevision = InxShaderLoader::GetSourceEnvironmentRevision();

    if (!m_renderer)
        return ticket;
    auto *adb = GetAssetDatabase();
    if (!adb)
        return ticket;

    auto resolvePath = [&](const ShaderAssetReference &reference, const char *stage) {
        if (!reference.guid.empty())
            return adb->GetPathFromGuid(reference.guid);
        if (!reference.pathHint.empty())
            return std::string{};
        return adb->FindShaderPathById(reference.shaderId, stage);
    };
    auto readSource = [&](const std::string &path, std::string &source) {
        std::vector<char> bytes;
        if (!adb->ReadFile(path, bytes) || bytes.empty())
            return false;
        if (bytes.back() == '\0')
            bytes.pop_back();
        source.assign(bytes.begin(), bytes.end());
        return true;
    };

    std::unordered_set<ShaderStagePair, ShaderStagePairHash> scheduled;
    auto &registry = AssetRegistry::Instance();
    for (const auto &guid : materialGuids) {
        auto material = registry.GetAsset<InxMaterial>(guid);
        if (!material)
            continue;

        state->materials.push_back({guid, material->GetVertShaderReference(), material->GetFragShaderReference()});
        const auto stages = material->GetShaderStages();
        if (!stages.IsValid() || !scheduled.insert(stages).second)
            continue;
        const auto cached = m_linkedShaderProgramCache.find(stages);
        LinkedShaderProgramLoadTicket::State::Work work;
        work.stages = stages;
        work.vertexPath = resolvePath(material->GetVertShaderReference(), "vertex");
        work.fragmentPath = resolvePath(material->GetFragShaderReference(), "fragment");
        if (asset_formats::FindShaderSourceKind(work.fragmentPath) ||
            asset_formats::FindShaderSourceKind(work.vertexPath))
            continue; // Static author programs are prepared by the asset loader, not the legacy stage linker.
        if (work.vertexPath.empty() || work.fragmentPath.empty() || !readSource(work.vertexPath, work.vertexSource) ||
            !readSource(work.fragmentPath, work.fragmentSource)) {
            m_linkedShaderPreparations.erase(stages);
            continue;
        }
        work.sourceStamp =
            ComputeShaderProgramRevision(work.vertexSource, work.fragmentSource, ShaderCompileTarget::Forward, 0);
        work.vertexMetadata = adb->GetMetaByPath(work.vertexPath);
        work.fragmentMetadata = adb->GetMetaByPath(work.fragmentPath);
        const LinkedShaderProgramLoadTicket::SourceIdentity identity{work.sourceStamp, state->sourceEnvironmentRevision,
                                                                     work.vertexMetadata, work.fragmentMetadata};
        auto &currentIdentity = m_linkedShaderPreparations[stages];
        if (!currentIdentity || *currentIdentity != identity)
            currentIdentity = std::make_shared<const LinkedShaderProgramLoadTicket::SourceIdentity>(identity);
        work.identity = currentIdentity;
        if (cached != m_linkedShaderProgramCache.end() &&
            cached->second.sourceEnvironmentRevision == state->sourceEnvironmentRevision) {
            if ((cached->second.sourceStamp == work.sourceStamp && cached->second.programKey.IsValid() &&
                 m_renderer->HasShaderProgramArtifact(cached->second.programKey)) ||
                cached->second.failedSourceStamp == work.sourceStamp) {
                continue;
            }
        }
        state->work.push_back(std::move(work));
    }

    if (state->work.empty())
        return ticket;

    auto compile = [state]() {
        state->producerThread = std::this_thread::get_id();
        const InxShaderLoader::SourceDiagnosticScope diagnostics;
        InxShaderLoader::SourceDependencyPublication dependencies;
        try {
            // One job compiles the scene's unique pairs serially. This keeps
            // compiler memory bounded on low-end machines while the shared
            // guard also protects editor hot reload and imported shaders.
            const InxShaderLoader::CompilationGuard compilationGuard;
            InxShaderLoader compiler(true, false, false, false, false, true, false, false, false, false);
            for (auto &work : state->work) {
                if (state->cancelRequested.load(std::memory_order_acquire))
                    break;
                const std::string vertexCompilePath =
                    InxShaderLoader::StageQualifiedVirtualPath(work.vertexPath, "vertex");
                const std::string fragmentCompilePath =
                    InxShaderLoader::StageQualifiedVirtualPath(work.fragmentPath, "fragment");
                const ShaderDescriptor vertexDescriptor =
                    compiler.ParseShaderSource(work.vertexSource, vertexCompilePath);
                work.fragmentDescriptor = compiler.ParseShaderSource(work.fragmentSource, fragmentCompilePath);
                if (!ShaderStageLinker::ShouldPrewarmSceneMaterial(vertexDescriptor, work.fragmentDescriptor)) {
                    work.skipScenePrewarm = true;
                    continue;
                }
                if (IsDirectStructuredStage(vertexDescriptor) || IsDirectStructuredStage(work.fragmentDescriptor)) {
                    work.directStructuredStage = true;
                    continue;
                }
                work.compilation = compiler.CompileLinkedProgramArtifact(work.vertexSource, vertexCompilePath,
                                                                         work.fragmentSource, fragmentCompilePath);
                if (!work.compilation.IsValid()) {
                    std::ostringstream diagnostics;
                    for (const auto &error : work.compilation.errors) {
                        if (diagnostics.tellp() > 0)
                            diagnostics << '\n';
                        diagnostics << error;
                    }
                    work.error = diagnostics.str();
                    if (work.error.empty())
                        work.error = "Linked shader program compilation failed";
                }
            }
        } catch (...) {
            state->failure = std::current_exception();
        }
        state->dependencies = dependencies.Detach();
    };

    if (JobSystem::IsAvailable()) {
        state->job = JobSystem::Get().Schedule(std::move(compile), JobDomain::Asset, JobPriority::High);
    } else {
        compile();
    }
    return ticket;
}

bool Infernux::TryCommitLinkedShaderPrograms(const std::shared_ptr<LinkedShaderProgramLoadTicket> &ticket)
{
    if (!ticket || !ticket->m_state)
        return false;
    auto &state = *ticket->m_state;
    if (std::this_thread::get_id() != state.ownerThread)
        throw std::logic_error("linked shader program publication must run on the ticket owner thread");
    if (!ticket->IsComplete())
        return false;
    if (state.committed)
        return true;
    if (state.cancelRequested.load(std::memory_order_acquire))
        return false;

    // Validate the whole preparation before touching GPU programs, render
    // metadata or the error cache. This is publication identity, not a second
    // filesystem scan/hash, and also retires old failed compilations.
    auto *database = GetAssetDatabase();
    bool current = !state.superseded && state.owner == this && !m_isCleaningUp && !m_isCleanedUp &&
                   state.sourceEnvironmentRevision == InxShaderLoader::GetSourceEnvironmentRevision();
    for (const auto &binding : state.materials) {
        const auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(binding.guid);
        current = current && material && material->GetVertShaderReference() == binding.vertex &&
                  material->GetFragShaderReference() == binding.fragment;
    }
    for (const auto &work : state.work) {
        const auto prepared = m_linkedShaderPreparations.find(work.stages);
        current = current && database && prepared != m_linkedShaderPreparations.end() &&
                  prepared->second == work.identity &&
                  database->GetMetaByPath(work.vertexPath) == work.vertexMetadata &&
                  database->GetMetaByPath(work.fragmentPath) == work.fragmentMetadata;
    }
    if (!current) {
        state.superseded = true;
        return false;
    }
    if (state.failure)
        std::rethrow_exception(state.failure);

    for (auto &work : state.work) {
        if (work.skipScenePrewarm)
            continue;
        if (work.directStructuredStage) {
            m_linkedShaderProgramCache.erase(work.stages);
            continue;
        }
        if (!work.error.empty()) {
            auto &entry = m_linkedShaderProgramCache[work.stages];
            entry.failedSourceStamp = work.sourceStamp;
            entry.lastError = work.error;
            entry.failureReported = true;
            entry.sourceEnvironmentRevision = state.sourceEnvironmentRevision;
            INXLOG_ERROR("Linked shader prewarm rejected '", work.stages.ToString(), "': ", work.error);
            continue;
        }

        ShaderProgramArtifact artifact = work.compilation.CreateRuntimeArtifact(work.stages);
        if (!ShaderStageLinker::ShouldPublishScenePrewarmArtifact(artifact))
            continue;
        if (!artifact.IsValid() || artifact.key.stages != work.stages ||
            !m_renderer->PublishShaderProgramArtifact(artifact)) {
            auto &entry = m_linkedShaderProgramCache[work.stages];
            entry.failedSourceStamp = work.sourceStamp;
            entry.lastError = "Renderer rejected the prewarmed linked shader program artifact";
            entry.failureReported = true;
            entry.sourceEnvironmentRevision = state.sourceEnvironmentRevision;
            INXLOG_ERROR("Linked shader prewarm publication failed for '", work.stages.ToString(), "'");
            continue;
        }

        m_linkedShaderProgramCache[work.stages] = LinkedShaderProgramCacheEntry{work.sourceStamp, artifact.key, 0, {}};
        m_linkedShaderProgramCache[work.stages].sourceEnvironmentRevision = state.sourceEnvironmentRevision;
        m_linkedShaderProgramCache[work.stages].stageLabels = {work.compilation.interfaceArtifact.vertex.shaderId,
                                                               work.compilation.interfaceArtifact.fragment.shaderId};
        const auto &fragment = work.fragmentDescriptor;
        m_renderer->StoreShaderRenderMeta(work.stages.fragmentKey, fragment.surfaceOptions.cullMode,
                                          fragment.depthWrite, fragment.depthTest, fragment.surfaceOptions.blendMode,
                                          fragment.renderQueue, fragment.passTag, fragment.stencil,
                                          fragment.surfaceOptions.alphaClip);
    }
    InxShaderLoader::SourceDependencyPublication::Publish(state.dependencies);
    state.committed = true;
    return true;
}

// ----------------------------------
// Lifecycle
// ----------------------------------

void Infernux::AcquireRuntimeServices()
{
    std::lock_guard lock(g_runtimeOwnerMutex);
    if (g_runtimeOwner != nullptr)
        throw std::logic_error("A runtime already owns the process services; clean it up before initializing another");
    if (AssetRegistry::Instance().IsInitialized())
        throw std::logic_error("The asset registry is already owned outside this runtime");
    g_runtimeOwner = this;
    m_ownsRuntimeServices = true;
}

void Infernux::StartRuntimeServices()
{
    INXLOG_SET_LEVEL(m_logLevel);
    // Standalone native hosts can provide the worker pool. Borrowing it does
    // not grant this runtime permission to shut it down.
    if (!JobSystem::IsAvailable()) {
        JobSystem::Initialize();
        m_ownsJobSystem = true;
    }
}

void Infernux::ReleaseRuntimeServices() noexcept
{
    std::lock_guard lock(g_runtimeOwnerMutex);
    g_runtimeOwner = nullptr;
    m_ownsRuntimeServices = false;
}

void Infernux::RegisterAssetCallbacks()
{
    RegisterPhysicMaterialAssetCallback();
    // ── Register unified asset event callbacks ──────────────────
    auto &graph = AssetDependencyGraph::Instance();

    graph.RegisterCallback(ResourceType::RenderTexture, [this](const std::string &dependentGuid,
                                                               const std::string &guid, AssetEvent event) {
        if (m_renderer &&
            m_renderer->InvalidateMaterialTextureAssets(dependentGuid, guid, event == AssetEvent::Deleted))
            return;
        uint64_t id = 0;
        const auto [end, error] =
            std::from_chars(dependentGuid.data(), dependentGuid.data() + dependentGuid.size(), id);
        if (error != std::errc{} || end != dependentGuid.data() + dependentGuid.size())
            return;
        auto *camera = dynamic_cast<Camera *>(Component::FindByComponentId(id));
        if (camera && camera->GetTargetTextureGuid() == guid)
            camera->OnTargetTextureAssetChanged(event == AssetEvent::Deleted);
    });

    auto resolveMaterial = [](const std::string &matGuid) -> std::shared_ptr<InxMaterial> {
        auto &registry = AssetRegistry::Instance();
        auto mat = registry.GetAssetType(matGuid) == ResourceType::Material ? registry.GetAsset<InxMaterial>(matGuid)
                                                                            : nullptr;
        if (mat)
            return mat;
        auto *adb = AssetRegistry::Instance().GetAssetDatabase();
        if (adb) {
            std::string matPath = adb->GetPathFromGuid(matGuid);
            if (!matPath.empty() && adb->GetResourceTypeForPath(matPath) == ResourceType::Material)
                mat = AssetRegistry::Instance().LoadAssetByPath<InxMaterial>(matPath, ResourceType::Material);
        }
        return mat;
    };

    graph.RegisterCallback(
        ResourceType::Texture,
        [this, resolveMaterial](const std::string &dependentGuid, const std::string &texGuid, AssetEvent event) {
            if (m_renderer &&
                m_renderer->InvalidateMaterialTextureAssets(dependentGuid, texGuid, event == AssetEvent::Deleted))
                return;
            auto mat = resolveMaterial(dependentGuid);
            if (mat) {
                if (event == AssetEvent::Deleted) {
                    for (const auto &[propName, prop] : mat->GetAllProperties()) {
                        if (prop.type != MaterialPropertyType::Texture2D)
                            continue;
                        const auto *val = std::get_if<std::string>(&prop.value);
                        if (!val || *val != texGuid)
                            continue;
                        INXLOG_INFO("AssetGraph: texture '", propName, "' is missing for material '", mat->GetName(),
                                    "'; preserving its GUID");
                    }
                }

                if (event == AssetEvent::Deleted || event == AssetEvent::Modified) {
                    mat->MarkPropertiesDirty();
                    if (auto *database = AssetRegistry::Instance().GetAssetDatabase()) {
                        const std::string materialPath = database->GetPathFromGuid(dependentGuid);
                        if (!materialPath.empty())
                            InvalidateMaterialPreviewTask(std::string("mat|") + materialPath);
                    }
                    INXLOG_INFO("AssetGraph: queued descriptor refresh for material '", mat->GetMaterialKey(),
                                "' (texture changed)");
                }
                return;
            }

            auto *renderer = ResolveDependentComponent<MeshRenderer>(dependentGuid);
            if (renderer)
                renderer->OnParameterTextureAssetEvent(texGuid, event);
        });

    graph.RegisterCallback(
        ResourceType::Material, [](const std::string &dependentGuid, const std::string &matGuid, AssetEvent event) {
            if (event != AssetEvent::Deleted && event != AssetEvent::Modified)
                return;
            auto *mr = ResolveDependentComponent<MeshRenderer>(dependentGuid);
            if (!mr)
                return;
            mr->OnMaterialAssetEvent(matGuid, event);
            INXLOG_INFO("AssetGraph: refreshed MeshRenderer material reference without changing GUID");
        });

    graph.RegisterCallback(ResourceType::Mesh,
                           [this](const std::string &dependentGuid, const std::string &meshGuid, AssetEvent event) {
                               if (auto *skeleton = ResolveDependentComponent<SkeletonInstance>(dependentGuid)) {
                                   skeleton->OnModelAssetEvent(meshGuid, event);
                                   return;
                               }
                               auto *mr = ResolveDependentComponent<MeshRenderer>(dependentGuid);
                               if (!mr)
                                   return;
                               mr->OnMeshAssetEvent(meshGuid, event);
                               if (event == AssetEvent::Deleted && m_renderer)
                                   m_renderer->InvalidateMeshCache(meshGuid);
                               if (event != AssetEvent::RuntimeModified)
                                   INXLOG_INFO("AssetGraph: refreshed MeshRenderer mesh state");
                           });

    graph.RegisterCallback(ResourceType::Audio,
                           [](const std::string &dependentGuid, const std::string &audioGuid, AssetEvent event) {
                               if (event != AssetEvent::Deleted && event != AssetEvent::Modified)
                                   return;
                               auto *source = ResolveDependentComponent<AudioSource>(dependentGuid);
                               if (!source)
                                   return;
                               source->OnAudioClipAssetEvent(audioGuid, event);
                               INXLOG_INFO("AssetGraph: refreshed AudioSource clip reference without changing GUID");
                           });

    graph.RegisterCallback(ResourceType::Shader, [this, resolveMaterial](const std::string &dependentGuid,
                                                                         const std::string & /*shaderGuid*/,
                                                                         AssetEvent event) {
        if (event != AssetEvent::Modified && event != AssetEvent::Deleted)
            return;
        auto mat = resolveMaterial(dependentGuid);
        if (!mat)
            return;
        if (m_renderer) {
            if (auto *ui = m_renderer->GetScreenUIRenderer())
                ui->InvalidateMaterialRejection(dependentGuid);
            if (event == AssetEvent::Deleted)
                m_renderer->InvalidateUIMaterialProgram({mat->GetVertShaderName(), mat->GetFragShaderName()});
        }
        mat->MarkPipelineDirty();
        INXLOG_INFO("AssetGraph: marked material '", mat->GetName(), "' pipeline dirty (shader changed)");
    });
}

void Infernux::RegisterAssetLoaders()
{
    auto &registry = AssetRegistry::Instance();
    registry.RegisterLoader(ResourceType::Material, std::make_unique<MaterialLoader>());
    registry.RegisterLoader(ResourceType::PhysicMaterial, std::make_unique<PhysicMaterialLoader>());
    registry.RegisterLoader(ResourceType::Texture, std::make_unique<TextureLoader>());
    registry.RegisterLoader(ResourceType::Mesh, std::make_unique<MeshLoader>());
    registry.RegisterLoader(ResourceType::Audio, std::make_unique<AudioClipLoader>());
    registry.RegisterLoader(ResourceType::Shader, std::make_unique<ShaderLoader>());
    registry.RegisterLoader(ResourceType::Script, std::make_unique<InxPythonScriptLoader>());
    for (const auto type : {ResourceType::DefaultText, ResourceType::DefaultBinary, ResourceType::RenderEffect,
                            ResourceType::ParticleGraph, ResourceType::DataAsset})
        registry.RegisterLoader(type, std::make_unique<InxDefaultLoader>(type));
    registry.RegisterLoader(ResourceType::RenderTexture,
                            std::make_unique<RenderTextureLoader>(
                                [this](const std::string &guid, const rhi::RenderTextureDesc &description) {
                                    if (m_renderer)
                                        m_renderer->ReconfigureImportedRenderTexture(guid, description);
                                }));
    registry.PopulateAssetDatabaseLoaders();
    m_assetLoadWorkSource = m_frameWork.AddSource(registry.GetAssetLoadWorkSource(m_frameWork));
    m_assetCatalogWorkSource = m_frameWork.AddSource(registry.GetAssetDatabase()->GetCatalogWorkSource(m_frameWork));
    m_documentWriteWorkSource = m_frameWork.AddSource(DocumentStore::BeginFrameWork(m_frameWork));
    m_previewIoWorkSource = m_frameWork.AddSource(GetPreviewIoWorkSource());
    PumpPreviewTasks();
}

Infernux::Infernux(std::string dllPath, RuntimeMode mode) : m_runtimeMode(mode), m_isCleanedUp(false)
{
    ComponentFactory::PublishSemanticTypes();
    (void)dllPath;
    INXLOG_DEBUG("Create Infernux.");
    m_assetDatabase = std::make_unique<AssetDatabase>();
}

void Infernux::CreateRenderer()
{
    INXLOG_DEBUG("Create Infernux Renderer.");
    m_gpuServices = std::make_unique<vk::VulkanBackendContext>();
    m_renderer = std::make_unique<InxRenderer>(*m_gpuServices);
    m_renderer->SetMaterialShaderDomainInspector(
        [this](const std::shared_ptr<InxMaterial> &material) { return InspectMaterialShaderDomain(material); });
    m_renderer->SetShaderProgramArtifactResolver(
        [this](const std::shared_ptr<InxMaterial> &material, std::optional<ShaderProgramDomain> expectedDomain) {
            const auto declaredDomain = InspectMaterialShaderDomain(material);
            if (declaredDomain && expectedDomain && *declaredDomain != *expectedDomain)
                throw ShaderProgramDomainMismatch(*expectedDomain, *declaredDomain);
            if (declaredDomain &&
                (*declaredDomain == ShaderProgramDomain::ScreenUI || *declaredDomain == ShaderProgramDomain::WorldUI)) {
                throw ShaderProgramSelectionError(
                    "Material shader domain mismatch before publication: expected " +
                    std::string(expectedDomain ? ShaderProgramDomainName(*expectedDomain) : "Mesh/ParticleSprite") +
                    ", got " + ShaderProgramDomainName(*declaredDomain));
            }
            const LinkedShaderProgramPreparation prepared = EnsureLinkedShaderProgramArtifact(material);
            if (prepared.usesLinkedArtifact && !prepared.success) {
                const auto stages = material->GetShaderStages();
                auto &entry = m_linkedShaderProgramCache[stages];
                if (!entry.failureReported) {
                    entry.failureReported = true;
                    INXLOG_ERROR("Material shader combination '", stages.ToString(), "' rejected: ", prepared.error,
                                 ". The previous complete GPU program is retained when available; otherwise the "
                                 "material is not drawn. Retry follows a selection or source change.");
                }
                throw ShaderProgramSelectionError(prepared.error);
            }
        });
    m_renderer->SetUIMaterialShaderValidator(
        [this](const std::shared_ptr<InxMaterial> &material, ShaderProgramDomain expectedDomain) {
            // A material GUID does not imply a custom UI program. Existing
            // Standard/Unlit and ordinary mesh materials still use the fixed
            // UI vertex format, texture descriptor and tint pipeline.
            std::optional<ShaderProgramDomain> declaredDomain;
            try {
                declaredDomain = InspectMaterialShaderDomain(material);
            } catch (const std::exception &error) {
                throw UIShaderMaterialError(error.what());
            }
            if (!declaredDomain ||
                (*declaredDomain != ShaderProgramDomain::ScreenUI && *declaredDomain != ShaderProgramDomain::WorldUI))
                return false;
            // Check the requested Screen/World domain *before* compiling or
            // publishing. A cross-domain first draw must not leave an
            // ownerless artifact or Forward program in VkShaderCache.
            if (*declaredDomain != expectedDomain)
                throw UIShaderMaterialError("UI material shader domain mismatch: expected " +
                                            std::string(ShaderProgramDomainName(expectedDomain)) + ", got " +
                                            std::string(ShaderProgramDomainName(*declaredDomain)));
            const LinkedShaderProgramPreparation prepared = EnsureLinkedShaderProgramArtifact(material, true);
            if (!prepared.usesLinkedArtifact || !prepared.success)
                throw UIShaderMaterialError(
                    "UI material '" + material->GetName() + "' shader publication failed: " +
                    (prepared.error.empty() ? std::string("linked UI program is unavailable") : prepared.error));
            return true;
        });
    m_renderer->SetShaderAssetResolver([this](const std::string &shaderId, const std::string &shaderType) {
        return EnsureShaderLoaded(shaderId, shaderType);
    });
    m_renderer->SetFrameBeginCallback([this](float deltaTime) { BeginRuntimeFrame(deltaTime); });
}

Infernux::~Infernux()
{
    Cleanup();
}

void Infernux::Run()
{
    RequireFrameInactive();
    if (!CheckEngineValid("run") || !m_isInitialized) {
        throw std::logic_error("Cannot run an uninitialized engine");
    }

    StartFrameDriver();
    m_runLoopActive.store(true, std::memory_order_release);
    struct RunLoopGuard
    {
        std::atomic<bool> &flag;
        ~RunLoopGuard()
        {
            flag.store(false, std::memory_order_release);
        }
    } runLoopGuard{m_runLoopActive};
    INXLOG_DEBUG("Run Infernux.");
    const AcceptanceClock acceptanceClock = ReadPlayerAcceptanceClock();
    if (m_runtimeMode == RuntimeMode::Headless) {
        auto previous = std::chrono::steady_clock::now();
        while (!m_exitRequested.load(std::memory_order_acquire)) {
            const auto now = std::chrono::steady_clock::now();
            const float deltaTime = std::chrono::duration<float>(now - previous).count();
            previous = now;
            TickFrame(deltaTime);

            std::unique_lock<std::mutex> lock(m_runMutex);
            m_runCv.wait_for(lock, std::chrono::milliseconds(1),
                             [this] { return m_exitRequested.load(std::memory_order_acquire); });
        }
        INXLOG_DEBUG("Headless loop ended.");
        return;
    }

    m_renderer->OverrideNextFrameDeltaTime(0.0f);
    bool acceptancePauseApplied = false;
    while (!m_exitRequested.load(std::memory_order_acquire) && m_renderer->GetUserEvent()) {
        if (acceptanceClock.enabled) {
            const auto &sceneManager = SceneManager::Instance();
            m_renderer->OverrideNextFrameDeltaTime(sceneManager.IsPaused() ? 0.0f : acceptanceClock.fixedDeltaSeconds);
        }
        FrameOpportunityScope workFrame(m_frameWork);
        FinishFrame(m_renderer->DrawFrame(/*pace=*/true));
        auto &sceneManager = SceneManager::Instance();
        if (acceptanceClock.enabled && !acceptancePauseApplied && sceneManager.IsPlaying() &&
            !sceneManager.IsPaused() && sceneManager.GetRuntimeFrameCount() >= acceptanceClock.pauseAfterFrame) {
            if (sceneManager.GetRuntimeFrameCount() != acceptanceClock.pauseAfterFrame) {
                throw std::logic_error("Player acceptance clock advanced past its requested capture frame");
            }
            sceneManager.Pause();
            acceptancePauseApplied = true;
            INXLOG_INFO("INFERNUX_PLAYER_ACCEPTANCE_PAUSED frame=", sceneManager.GetRuntimeFrameCount(),
                        " fixed_delta=", acceptanceClock.fixedDeltaSeconds);
        }

        // Periodically save layout when ImGui marks it dirty
        ImGuiIO &io = ImGui::GetIO();
        if (io.WantSaveIniSettings) {
            SaveImGuiLayout();
            io.WantSaveIniSettings = false;
        }
    }
    INXLOG_DEBUG("Main loop ended.");
    SaveImGuiLayout();
    // NOTE: Cleanup is no longer called here — Python controls the
    // shutdown order so it can stop background threads first.
    // ~Infernux() still calls Cleanup() as a safety net.
}

void Infernux::Tick(float deltaTime)
{
    RequireFrameInactive();
    if (!CheckEngineValid("tick") || !m_isInitialized) {
        throw std::logic_error("Cannot tick an uninitialized engine");
    }
    if (!std::isfinite(deltaTime) || deltaTime < 0.0f) {
        throw std::invalid_argument("delta_time must be finite and non-negative");
    }
    StartFrameDriver();
    TickFrame(deltaTime);
}

void Infernux::TickFrame(float deltaTime)
{
    ActiveFrameScope tickScope(m_tickActive);
    FrameOpportunityScope workFrame(m_frameWork);

    // Explicit stepping supplies simulation time, not elapsed loading time.
    // A scene replacement inside this frame can still reset its own clock.
    (void)SceneManager::Instance().ConsumeFrameDeltaTime(0.0f);

    if (m_runtimeMode == RuntimeMode::Graphical) {
        // Headless-style manual stepping must stay usable with a window open:
        // one Tick = one fully simulated AND rendered frame with an exact
        // delta time. The caller owns the cadence of explicit host ticks.
        if (!m_renderer->GetUserEvent()) {
            Exit();
            return;
        }
        m_renderer->OverrideNextFrameDeltaTime(deltaTime);
        FinishFrame(m_renderer->DrawFrame(/*pace=*/false));
        return;
    }

    auto &sceneManager = SceneManager::Instance();
    BeginRuntimeFrame(deltaTime);
    const float sceneDeltaTime = sceneManager.ConsumeFrameDeltaTime(deltaTime);
    TransformECSStore::Instance().BeginFrameCache();
    sceneManager.Update(sceneDeltaTime);
    sceneManager.LateUpdate(sceneDeltaTime);
    // Mirror the graphical DrawFrame simulation segment exactly (see
    // InxRenderer::DrawFrame). Headless previously skipped the world-matrix
    // sync and the FinalTransformResolve/AnimationTimeline barriers, so
    // transform reads between frames and Python systems registered on those
    // barriers observed a different world than the graphical runtime.
    // Graphics-only barriers (RenderExtraction, RenderGraph,
    // SnapshotPublication) stay renderer-owned by design.
    if (TransformECSStore::Instance().EndFrameCache())
        sceneManager.PublishPhysicsTransformsToRenderer();
    sceneManager.PublishAuthoredTransformsToPhysics();
    if (!sceneManager.HasPendingFixedStep()) {
        sceneManager.EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::FinalTransformResolve);
        sceneManager.EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::AnimationTimeline);
    }
    sceneManager.EndFrame();
    FinishFrame({true});
}

void Infernux::SetEndFrameCallback(std::function<void(bool)> callback)
{
    m_endFrameCallback = std::move(callback);
}

void Infernux::FinishFrame(const RenderFrameResult &result)
{
    // Run() is outside Tick's guard, and Renderer has already returned. Keep
    // this owner callback inside the same no-reentry/no-cleanup frame boundary.
    ActiveFrameScope frameScope(m_tickActive);
    if (result.submitted) {
        ++m_submittedRenderFrames;
        SceneManager::Instance().EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::RenderSubmitted);
    }
    if (result.presentationAcceptedNs) {
        m_lastPresentation = {m_lastPresentation.sequence + 1, m_frameWork.Opportunity(),
                              result.presentationAcceptedNs};
        SceneManager::Instance().EmitRuntimeFrameBarrier(SceneManager::RuntimeFrameBarrier::RenderPresented);
    }
    const bool foregroundReady = result.foregroundReady;
    if (m_frameWork.AdvanceLate([this, foregroundReady] { return BackgroundGpuAvailability(foregroundReady); }))
        if (m_renderer)
            m_renderer->RequestFullSpeedFrame();
    if (auto callback = m_endFrameCallback)
        callback(foregroundReady && !m_exitRequested.load(std::memory_order_acquire));
    m_frameWork.EndOpportunity();
}

void Infernux::SetPreSceneUpdateCallback(std::function<void(float)> callback)
{
    m_preSceneUpdateCallback = std::move(callback);
}

void Infernux::SetRuntimeFrameCallbacks(std::function<void(float)> beforeWork, std::function<void()> afterWork)
{
    m_runtimeBeforeWork = std::move(beforeWork);
    m_runtimeAfterWork = std::move(afterWork);
}

void Infernux::BeginRuntimeFrame(float deltaTime)
{
    m_frameWork.BeginOpportunity();
    PollGpuCompletions();
    if (m_textureWorkSource)
        m_frameWork.Wake(m_textureWorkSource);
    if (m_assetLoadWorkSource)
        m_frameWork.Wake(m_assetLoadWorkSource);
    if (m_assetCatalogWorkSource)
        m_frameWork.Wake(m_assetCatalogWorkSource);
    if (m_documentWriteWorkSource)
        m_frameWork.Wake(m_documentWriteWorkSource);
    if (m_previewIoWorkSource)
        m_frameWork.Wake(m_previewIoWorkSource);
    if (m_previewWorkSource && (m_hasPreviewPumpWork.exchange(false, std::memory_order_acq_rel) || m_timelineCubeDirty))
        m_frameWork.Wake(m_previewWorkSource);
    if (auto callback = m_runtimeBeforeWork)
        callback(deltaTime);
    if (auto callback = m_preSceneUpdateCallback)
        callback(deltaTime);
    if (m_frameWork.AdvanceEarly())
        if (m_renderer)
            m_renderer->RequestFullSpeedFrame();
    if (auto callback = m_runtimeAfterWork)
        callback();
}

FrameGpuAvailability Infernux::BackgroundGpuAvailability(bool foregroundReady)
{
    if (m_exitRequested.load(std::memory_order_acquire))
        return FrameGpuAvailability::ForegroundDeferred;
    if (!m_gpuServices)
        return foregroundReady ? FrameGpuAvailability::Ready : FrameGpuAvailability::ForegroundDeferred;
    m_gpuServices->CollectComputeResources();
    // A successful foreground submission owns the front of this opportunity.
    // Older in-flight frames are normal pipelining, not a requirement to drain
    // the entire graphics queue before admitting one bounded compute segment.
    const auto pressure = m_gpuServices->Queues().GetFramePressure();
    if (pressure.recordingFrames)
        return FrameGpuAvailability::ForegroundRecording;
    if (!foregroundReady)
        return pressure.olderPendingFrames ? FrameGpuAvailability::ForegroundBacklog
                                           : FrameGpuAvailability::ForegroundDeferred;
    if (m_gpuServices->Compute().GetPendingSubmissionCount())
        return FrameGpuAvailability::BackgroundPending;
    return FrameGpuAvailability::Ready;
}

void Infernux::Exit()
{
    INXLOG_DEBUG("Exit requested.");
    {
        std::lock_guard lock(m_runMutex);
        m_frameWork.SuspendDriver();
        m_exitRequested.store(true, std::memory_order_release);
    }
    m_runCv.notify_all();
}

std::unique_ptr<rhi::ComputeHost> Infernux::AcquireComputeHost()
{
    if (!HasGpuCompute())
        throw std::runtime_error("GPU compute requires initialized engine device services");
    return m_gpuServices->AcquireComputeHost();
}

bool Infernux::HasGpuCompute() const noexcept
{
    return m_isInitialized && !m_isCleaningUp && !m_isCleanedUp && m_gpuServices && m_gpuServices->IsInitialized();
}

std::shared_ptr<rhi::RenderTexture> Infernux::CreateRenderTexture(const rhi::RenderTextureDesc &description)
{
    if (!m_isInitialized || m_isCleaningUp || m_isCleanedUp || !m_renderer)
        throw std::runtime_error("RenderTexture requires an initialized graphical engine");
    return m_renderer->CreateRenderTexture(description);
}

void Infernux::RequireComputeHostsReleased() const
{
    if (m_gpuServices && m_gpuServices->GetComputeHostLeaseCount() != 0)
        throw std::runtime_error("Release compute plugin runtimes before cleaning up the engine (active leases=" +
                                 std::to_string(m_gpuServices->GetComputeHostLeaseCount()) + ")");
}

std::shared_ptr<rhi::RenderTexture> Infernux::LoadRenderTexture(const std::string &guid)
{
    if (!m_isInitialized || m_isCleaningUp || m_isCleanedUp || !m_renderer)
        throw std::runtime_error("RenderTexture requires an initialized graphical engine");
    return m_renderer->LoadRenderTexture(guid);
}

void Infernux::RequireFrameInactive() const
{
    if (m_runLoopActive.load(std::memory_order_acquire) || m_tickActive || (m_renderer && m_renderer->IsDrawingFrame()))
        throw std::logic_error("operation requires a stopped frame driver and no active frame callback");
}

void Infernux::StartFrameDriver()
{
    // Exit stops the current driver; Cleanup ends the engine lifetime. A new
    // explicit Run/Tick resumes the same services. Run's internal headless
    // ticks must not clear an exit requested by its callbacks or another thread.
    std::lock_guard lock(m_runMutex);
    m_exitRequested.store(false, std::memory_order_release);
    m_frameWork.Reopen();
}

void Infernux::RequireCleanupBoundary() const
{
    RequireFrameInactive();
    if (m_ownsRuntimeServices) {
        Scene::RequireNoWorldRetirement();
        const auto *database = AssetRegistry::Instance().GetAssetDatabase();
        if (database && database->IsDeliveringModelReimport())
            throw std::logic_error("Cannot clean up the engine from a model Apply publication callback");
    }
}

void Infernux::Cleanup()
{
    RequireCleanupBoundary();
    if (m_isCleanedUp || m_isCleaningUp)
        return;

    RequireComputeHostsReleased();

    m_isCleaningUp = true;
    m_frameWork.Stop();
    m_frameWork.ClearSources();
    m_previewWorkSource = 0;
    m_previewIoWorkSource = 0;
    m_textureWorkSource = 0;
    m_assetLoadWorkSource = 0;
    m_assetCatalogWorkSource = 0;
    m_documentWriteWorkSource = 0;
    m_preSceneUpdateCallback = nullptr;
    m_endFrameCallback = nullptr;
    m_runtimeBeforeWork = nullptr;
    m_runtimeAfterWork = nullptr;

    if (!m_ownsRuntimeServices) {
        // Construction alone creates no global lifetime. In particular an
        // unused instance must not stop another runtime's scenes or workers.
        m_renderer.reset();
        m_gpuServices.reset();
        m_assetDatabase.reset();
        m_isCleanedUp = true;
        m_isInitialized = false;
        m_isCleaningUp = false;
        return;
    }

    if (m_runtimeMode == RuntimeMode::Graphical) {
        if (m_isInitialized) {
            SaveImGuiLayout();
        }
    }
    DrainPreviewJobs();
    auto *assetDatabase = AssetRegistry::Instance().GetAssetDatabase();
    if (assetDatabase && assetDatabase->IsModelReimportPublishing())
        assetDatabase->CompletePendingModelReimport();

    // Destroy all scenes/GameObjects FIRST: Collider destructors release
    // their Jolt bodies (needs a live PhysicsWorld) and AudioSource
    // destructors detach from the AudioEngine. Singletons are intentionally
    // leaked, so this explicit call is the only place scene teardown happens.
    SceneManager::Instance().Shutdown();

    if (assetDatabase)
        assetDatabase->FlushDerivedIndex(/*waitForPendingScan=*/false);

    // Event callbacks may capture this engine lifetime. Drop them together
    // with all runtime dependency edges before another engine is initialized.
    AssetDependencyGraph::Instance().Clear();

    AudioEngine::Instance().Shutdown();
    PhysicsWorld::Instance().Shutdown();

    // CPU asset preparation holds loader pointers and must finish before the
    // renderer or headless path stops the engine-wide JobSystem.
    AssetRegistry::Instance().DrainPendingLoads();

    if (m_gpuServices)
        m_gpuServices->BeginShutdown();
    m_renderer.reset();

    // AssetRegistry owns all loaded assets + builtins.
    AssetRegistry::Instance().Shutdown();
    // Initialization can fail before transferring this database to the registry.
    // Its accepted import/write jobs still need the shared workers and store.
    m_assetDatabase.reset();

    // All document producers are now stopped. No accepted scene, material,
    // metadata, settings, or prefab write may outlive this engine lifetime.
    DocumentStore::EndFrameWork(m_frameWork);

    if (m_ownsJobSystem) {
        JobSystem::Shutdown();
        m_ownsJobSystem = false;
    }

    m_gpuServices.reset();

    m_isCleanedUp = true;
    m_isInitialized = false;
    m_isCleaningUp = false;
    INXLOG_DEBUG("Cleanup completed.");
#if INFERNUX_FILE_LOGGING
    INXLOG_FLUSH_FILE();
    INXLOG_SHUTDOWN();
#endif
    ReleaseRuntimeServices();
}

void Infernux::DrainPreviewJobs()
{
    JobHandle dispatcher;
    std::deque<PendingPreviewJob> unaccepted;
    {
        std::lock_guard<std::mutex> lock(m_previewJobMutex);
        m_acceptPreviewJobs = false;
        m_previewJobs.swap(unaccepted);
        dispatcher = m_previewDispatcherJob;
    }

    // Drop unaccepted input references outside the queue lock. Only the one
    // accepted callback must finish while the engine's services are still live.
    unaccepted = {};

    if (dispatcher.IsValid() && JobSystem::IsAvailable())
        JobSystem::Get().WaitPassive(dispatcher);
    ObservePreviewJob();

    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    m_texturePreviewCompletedQueue.clear();
    m_materialPreviewRequestQueue.clear();
    m_meshPreviewRequestQueue.clear();
    m_materialPreviewStates.clear();
    m_texturePreviewStates.clear();
    m_meshPreviewStates.clear();
    m_cubePreviewCandidate.reset();
    m_timelineCubeDirty = false;
    m_hasPendingPreviewUploads.store(false, std::memory_order_release);
    m_hasPreviewPumpWork.store(false, std::memory_order_release);
}

void Infernux::EnqueuePreviewTask(std::function<void()> fn, std::optional<std::weak_ptr<const void>> owner)
{
    if (!fn)
        throw std::invalid_argument("preview job requires a callable");
    EnqueuePreviewTask([task = std::move(fn)](auto) { task(); }, 0, std::move(owner));
}

void Infernux::EnqueuePreviewTask(std::function<void(std::shared_ptr<FrameCpuStorageClaim>)> fn, uint64_t storageBytes,
                                  std::optional<std::weak_ptr<const void>> owner)
{
    if (!JobSystem::IsAvailable())
        throw std::logic_error("preview jobs require an initialized JobSystem");

    {
        std::lock_guard<std::mutex> lock(m_previewJobMutex);
        if (!m_acceptPreviewJobs)
            throw std::runtime_error("preview jobs are no longer accepted during engine shutdown");
        m_previewJobs.push_back({std::move(fn), std::move(owner), storageBytes});
    }
    // Enqueue can also run on a worker. The owner wakes this source at the
    // next host boundary; workers never touch the frame coordinator.
}

void Infernux::ObservePreviewJob()
{
    if (!m_previewDispatcherJob.IsValid() || !m_previewDispatcherJob.IsComplete())
        return;
    m_frameWork.ObserveResource(FrameWorkResource::Io, m_previewExecutionNanoseconds, m_previewJobFrame,
                                m_previewJobGrant);
    m_previewDispatcherJob = {};
    m_previewJobFrame = m_previewJobGrant = 0;
}

std::deque<Infernux::PendingPreviewJob>::iterator Infernux::FindReadyPreviewJob()
{
    m_previewJobs.erase(
        std::remove_if(m_previewJobs.begin(), m_previewJobs.end(), [](const auto &job) { return job.IsRetired(); }),
        m_previewJobs.end());
    return std::find_if(m_previewJobs.begin(), m_previewJobs.end(), [this](const auto &job) {
        // A permanently oversized request runs only to report its capacity error.
        return !job.storageBytes || job.storageBytes > m_frameWork.CpuStorageLimit() ||
               m_frameWork.CanReserveCpuStorage(job.storageBytes);
    });
}

FrameWorkSource Infernux::GetPreviewIoWorkSource()
{
    FrameWorkSource source;
    source.interactive = true;
    source.observe = [this] { ObservePreviewJob(); };
    source.refresh = source.observe;
    source.describe = [this] {
        FrameWorkFacts facts;
        std::lock_guard<std::mutex> lock(m_previewJobMutex);
        const bool canStart = m_acceptPreviewJobs && !m_previewDispatcherJob.IsValid() &&
                              m_frameWork.CanAdmitResource(FrameWorkResource::Io, 0) &&
                              FindReadyPreviewJob() != m_previewJobs.end();
        facts.pending = !m_previewJobs.empty() || m_previewDispatcherJob.IsValid();
        facts.workerBusy = m_previewDispatcherJob.IsValid() && !m_previewDispatcherJob.IsComplete();
        facts.observe = facts.pending;
        facts.host.ready = canStart;
        facts.ioReady = facts.host.ready;
        return facts;
    };
    source.advance = [this](bool, uint64_t) {
        if (m_previewDispatcherJob.IsValid() || !m_frameWork.CanAdmitResource(FrameWorkResource::Io, 0))
            return false;
        PendingPreviewJob task;
        std::shared_ptr<FrameCpuStorageClaim> storage;
        {
            std::lock_guard<std::mutex> lock(m_previewJobMutex);
            const auto ready = FindReadyPreviewJob();
            if (!m_acceptPreviewJobs || ready == m_previewJobs.end())
                return false;
            if (ready->storageBytes && ready->storageBytes <= m_frameWork.CpuStorageLimit()) {
                storage = m_frameWork.TryReserveCpuStorage(ready->storageBytes);
                if (!storage)
                    return false;
            }
            task = std::move(*ready);
            m_previewJobs.erase(ready);
        }
        m_previewExecutionNanoseconds = 0;
        m_previewDispatcherJob = JobSystem::Get().Schedule(
            [this, run = std::move(task.run), storage = std::move(storage)]() mutable {
                const auto started = std::chrono::steady_clock::now();
                try {
                    run(std::move(storage));
                } catch (const std::exception &error) {
                    INXLOG_ERROR("Preview worker job failed: ", error.what());
                } catch (...) {
                    INXLOG_ERROR("Preview worker job failed with an unknown exception");
                }
                m_previewExecutionNanoseconds = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                        .count());
            },
            JobDomain::IO, JobPriority::Low, detail::IoJobAffinity);
        m_previewJobFrame = m_frameWork.Opportunity();
        m_previewJobGrant = m_frameWork.RecordAcceptedResource(FrameWorkResource::Io, 0);
        return true;
    };
    return source;
}

std::string Infernux::BuildPreviewTextureName(const std::string &resourceKey)
{
    const auto hv = std::hash<std::string>{}(resourceKey);
    return std::string("__cpp_preview_mat__") + std::to_string(static_cast<unsigned long long>(hv));
}

std::string Infernux::BuildTexturePreviewTextureName(const std::string &resourceKey)
{
    const auto hv = std::hash<std::string>{}(resourceKey);
    return std::string("__cpp_preview_tex__") + std::to_string(static_cast<unsigned long long>(hv));
}

std::string Infernux::BuildMeshPreviewTextureName(const std::string &resourceKey)
{
    const auto hv = std::hash<std::string>{}(resourceKey);
    return std::string("__cpp_preview_mesh__") + std::to_string(static_cast<unsigned long long>(hv));
}

// Canonicalize the path portion of a preview cache key so callers that spell the
// same file differently map to ONE cache entry. The C++ Project panel builds keys
// with absolute forward-slash paths (FromFsPath), while the Python inspector passes
// os.path.normpath() results (backslashes on Windows). Without this, "mat|D:/a/b.mat"
// and "mat|D:\a\b.mat" are distinct entries, so the inspector can never reuse the
// Project panel's rendered texture (the long-standing flaky-preview bug). Only the
// MAP KEY is normalized here — the real file path used for disk I/O is passed
// separately and left untouched. Non-ASCII (UTF-8) bytes are preserved.
/// Numeric ImGui handles are frame-scoped observations. Callers retain the stable
/// texture name and re-resolve the currently published descriptor before drawing.
static ImGuiTexturePublication LiveImGuiTexture(const InxRenderer *renderer, const std::string &texName)
{
    if (!renderer || texName.empty())
        return {};
    return renderer->GetImGuiTexturePublication(texName);
}

static uint64_t LiveImGuiTextureId(const InxRenderer *renderer, const std::string &texName)
{
    return LiveImGuiTexture(renderer, texName).textureId;
}

static std::string CanonicalizePreviewKey(const std::string &resourceKey)
{
    const size_t bar = resourceKey.find('|');
    const size_t start = (bar == std::string::npos) ? 0 : bar + 1;
    std::string out = resourceKey;
    for (size_t i = start; i < out.size(); ++i) {
        char &c = out[i];
        if (c == '\\') {
            c = '/';
        }
#ifdef _WIN32
        else if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a'); // Windows paths are case-insensitive
        }
#endif
    }
    return out;
}

static void ApplyLinearToSrgbPreviewInPlace(std::vector<unsigned char> &pixels)
{
    if (pixels.empty())
        return;

    uint8_t lut[256];
    for (int i = 0; i < 256; ++i) {
        const float linear = static_cast<float>(i) / 255.0f;
        const float encoded = linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        lut[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(encoded * 255.0f + 0.5f), 0, 255));
    }

    for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
        pixels[i + 0] = lut[pixels[i + 0]];
        pixels[i + 1] = lut[pixels[i + 1]];
        pixels[i + 2] = lut[pixels[i + 2]];
    }
}

static void ApplyTextureFormatPreviewInPlace(std::vector<unsigned char> &pixels, const std::string &format)
{
    if (format != "rgba4444")
        return;
    for (unsigned char &value : pixels) {
        const unsigned int nibble = (static_cast<unsigned int>(value) + 8U) / 17U;
        value = static_cast<unsigned char>((std::min)(15U, nibble) * 17U);
    }
}

static void ApplyNormalMapPreviewInPlace(std::vector<unsigned char> &pixels, int width, int height)
{
    if (pixels.empty() || width <= 0 || height <= 0)
        return;

    const std::vector<unsigned char> source = pixels;
    constexpr float kStrength = 2.0f;
    const auto sampleLuminance = [&source, width, height](int x, int y) {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        const size_t index = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
        return (static_cast<float>(source[index]) * 0.299f + static_cast<float>(source[index + 1]) * 0.587f +
                static_cast<float>(source[index + 2]) * 0.114f) /
               255.0f;
    };

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float nx = (sampleLuminance(x - 1, y) - sampleLuminance(x + 1, y)) * kStrength;
            float ny = (sampleLuminance(x, y - 1) - sampleLuminance(x, y + 1)) * kStrength;
            float nz = 1.0f;
            const float inverseLength = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inverseLength;
            ny *= inverseLength;
            nz *= inverseLength;

            const size_t index = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
            pixels[index] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255.0f);
            pixels[index + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255.0f);
            pixels[index + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255.0f);
            pixels[index + 3] = 255;
        }
    }
}

struct Infernux::MaterialPreviewRaster
{
    std::shared_ptr<FrameCpuStorageClaim> storage;
    std::shared_ptr<const TextureCpuData> pixels;
    PreviewPixelSummary summary;
    std::exception_ptr failure;
    std::atomic<bool> complete{false};
};

static constexpr int kMaterialPreviewSize = 200;
static constexpr uint64_t kMaterialPreviewRasterBytes = kMaterialPreviewSize * kMaterialPreviewSize * 4;

static bool MaterialPreviewInputsCurrent(const std::vector<std::pair<std::string, uint64_t>> &inputs)
{
    auto &graph = AssetDependencyGraph::Instance();
    return std::all_of(inputs.begin(), inputs.end(),
                       [&](const auto &input) { return graph.GetContentRevision(input.first) == input.second; });
}

static PreviewPixelSummary SummarizePreviewPixels(const std::vector<unsigned char> &pixels)
{
    if (pixels.empty())
        return {};

    PreviewPixelSummary summary{UINT64_C(1469598103934665603)};
    uint8_t minRgb = 255;
    uint8_t maxRgb = 0;
    for (size_t index = 0; index < pixels.size(); ++index) {
        const uint8_t value = pixels[index];
        summary.hash ^= value;
        summary.hash *= UINT64_C(1099511628211);
        const size_t channel = index & 3u;
        if (channel < 3u) {
            minRgb = (std::min)(minRgb, value);
            maxRgb = (std::max)(maxRgb, value);
        } else if (value != 0) {
            ++summary.nonTransparentPixelCount;
        }
    }
    summary.minRgb = minRgb;
    summary.maxRgb = maxRgb;
    return summary;
}

static PreviewPixelSummary SummarizePreviewPixels(const TextureCpuData &texture)
{
    // Documentation decodes use uncompressed RGBA8 or linear RGBA32F. Compute
    // diagnostics on the decode worker; never scan full-size images on the UI thread.
    if (texture.format != TextureFormat::Rgba32Float)
        return SummarizePreviewPixels(texture.bytes);
    PreviewPixelSummary summary{UINT64_C(1469598103934665603)};
    for (const auto value : texture.bytes) {
        summary.hash ^= value;
        summary.hash *= UINT64_C(1099511628211);
    }
    float low = 1.0f, high = 0.0f;
    for (size_t offset = 0; offset + sizeof(float) * 4 <= texture.bytes.size(); offset += sizeof(float) * 4) {
        std::array<float, 4> rgba;
        std::memcpy(rgba.data(), texture.bytes.data() + offset, sizeof(rgba));
        if (rgba[3] > 0.0f)
            ++summary.nonTransparentPixelCount;
        for (int channel = 0; channel < 3; ++channel) {
            if (std::isfinite(rgba[channel])) {
                low = (std::min)(low, rgba[channel]);
                high = (std::max)(high, rgba[channel]);
            }
        }
    }
    auto displayByte = [](float value) {
        value = std::clamp(value, 0.0f, 1.0f);
        const float srgb = value <= 0.0031308f ? 12.92f * value : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
        return static_cast<uint8_t>(std::lround(srgb * 255.0f));
    };
    summary.minRgb = displayByte(low);
    summary.maxRgb = displayByte(high);
    return summary;
}

uint64_t Infernux::QueryOrScheduleMaterialPreview(const std::string &resourceKey, const std::string &matFilePath,
                                                  const std::string &materialJson, uint64_t fileMtimeHint,
                                                  bool authoring)
{
    if (resourceKey.empty())
        return 0;

    const std::string key = CanonicalizePreviewKey(resourceKey);
    const uint64_t revision = materialJson.empty() ? GetPreviewDependencyRevision(matFilePath) : 0;
    std::lock_guard lock(m_previewResultMutex);
    auto &state = m_materialPreviewStates[key];
    if (state.textureName.empty())
        state.textureName = BuildPreviewTextureName(key);

    const auto textureId = LiveImGuiTextureId(m_renderer.get(), state.textureName);
    // A live editor document owns the shared image until its save hands it back to Project.
    if ((!authoring && state.authoring) || (matFilePath.empty() && materialJson.empty()))
        return textureId;
    state.authoring |= authoring;

    const bool fileChanged = state.latestMatFilePath != matFilePath &&
                             CanonicalizePreviewKey(state.latestMatFilePath) != CanonicalizePreviewKey(matFilePath);
    const bool documentChanged = state.latestMaterialJson != materialJson;
    if (fileChanged || documentChanged ||
        (state.submittedInputs && !MaterialPreviewInputsCurrent(*state.submittedInputs)) ||
        (materialJson.empty() &&
         (revision != state.lastDependencyRevision || (fileMtimeHint && fileMtimeHint != state.lastFileMtime))))
        BumpMaterialPreviewGeneration(state);

    if (fileChanged)
        state.latestMatFilePath = matFilePath;
    if (documentChanged)
        state.latestMaterialJson = materialJson;
    if (fileChanged || fileMtimeHint)
        state.lastFileMtime = fileMtimeHint;
    state.lastDependencyRevision = revision;
    if (!textureId && state.readyGeneration == state.generation)
        state.readyGeneration = 0;
    ScheduleDirtyMaterialPreview(key, state);
    return textureId;
}

void Infernux::BumpMaterialPreviewGeneration(MaterialPreviewState &state)
{
    ++state.generation;
    state.submittedInputs.reset();
    if (state.upload.version) {
        m_renderer->SupersedePendingImGuiTextureUploads(state.textureName);
        state.upload = {};
    }
}

bool Infernux::RefreshMaterialPreviewDependency(MaterialPreviewState &state)
{
    const auto revision = state.latestMaterialJson.empty() ? GetPreviewDependencyRevision(state.latestMatFilePath) : 0;
    if (revision == state.lastDependencyRevision &&
        (!state.submittedInputs || MaterialPreviewInputsCurrent(*state.submittedInputs)))
        return false;
    state.lastDependencyRevision = revision;
    BumpMaterialPreviewGeneration(state);
    return true;
}

void Infernux::ScheduleDirtyMaterialPreview(const std::string &key, MaterialPreviewState &state)
{
    if (state.pendingGeneration || state.renderGeneration == state.generation ||
        state.upload.generation == state.generation || state.readyGeneration >= state.generation ||
        state.failedGeneration == state.generation ||
        (state.latestMatFilePath.empty() && state.latestMaterialJson.empty()))
        return;
    m_materialPreviewRequestQueue.push_back({key, state.latestMatFilePath, state.generation, state.latestMaterialJson});
    state.pendingGeneration = state.generation;
    m_hasPreviewPumpWork.store(true, std::memory_order_release);
    if (m_renderer)
        m_renderer->RequestFullSpeedFrame();
}

Infernux::PreviewStep Infernux::DescribeMaterialPreviewRequest(const MaterialPreviewRequest &request) const
{
    const auto state = m_materialPreviewStates.find(request.resourceKey);
    if (state == m_materialPreviewStates.end() || state->second.pendingGeneration != request.generation ||
        state->second.generation != request.generation ||
        (request.materialJson.empty() &&
         GetPreviewDependencyRevision(request.matFilePath) != state->second.lastDependencyRevision))
        return PreviewStep::Host;
    if (!IsMaterialPreviewSourceCurrent(request))
        return PreviewStep::Host;
    if (request.material) {
        if (!request.raster || request.raster->complete.load(std::memory_order_acquire))
            return PreviewStep::Gpu;
        return !request.raster->storage && (kMaterialPreviewRasterBytes > m_frameWork.CpuStorageLimit() ||
                                            m_frameWork.CanReserveCpuStorage(kMaterialPreviewRasterBytes))
                   ? PreviewStep::Host
                   : PreviewStep::Wait;
    }
    return request.loadTicket && !request.loadTicket->IsComplete() ? PreviewStep::Wait : PreviewStep::Host;
}

bool Infernux::IsMaterialPreviewSourceCurrent(const MaterialPreviewRequest &request) const
{
    const auto &registry = AssetRegistry::Instance();
    if (request.loadTicket)
        return registry.IsAssetLoadContentCurrent(request.loadTicket);
    if (!request.assetGuid.empty() &&
        registry.GetAssetContentGeneration(request.assetGuid) != request.assetContentGeneration)
        return false;
    if (!request.material)
        return true;
    return request.material->GetVersion() == request.materialVersion &&
           MaterialPreviewInputsCurrent(request.inputRevisions);
}

bool Infernux::PrepareMaterialPreview(MaterialPreviewRequest &request, bool &progressed)
{
    auto *database = GetAssetDatabase();
    request.raster.reset();
    if (!request.materialJson.empty()) {
        request.material = MaterialPreviewer::BuildPreviewMaterialFromJson(request.materialJson, database);
    } else {
        std::string modelPath;
        int slot = 0;
        const bool embedded = ParseModelEmbeddedMaterialSlot(request.matFilePath, modelPath, slot);
        const std::string guid =
            database ? database->GetGuidFromCatalogPath(embedded ? modelPath : request.matFilePath) : std::string{};
        request.material.reset();
        if (guid.empty()) {
            // The source may have been removed while its thumbnail waited for a host turn.
            request.loadTicket.reset();
            return true;
        }

        const auto type = embedded ? ResourceType::Mesh : ResourceType::Material;
        if (!RequirePreviewAsset(request.loadTicket, guid, type, progressed))
            return false;

        auto &registry = AssetRegistry::Instance();
        if (embedded) {
            if (auto mesh = registry.GetAsset<InxMesh>(guid))
                request.material = mesh->CreateMaterialCopy(static_cast<uint32_t>(slot));
        } else if (auto material = registry.GetAsset<InxMaterial>(guid)) {
            request.material = MaterialPreviewer::ClonePreviewMaterial(material, database);
        }
        // Keeping the ticket here can pin geometry while a cold texture needs space.
        request.assetGuid = guid;
        request.assetContentGeneration = registry.GetAssetContentGeneration(guid);
        request.loadTicket.reset();
    }

    if (!request.material)
        return true;
    request.materialVersion = request.material->GetVersion();
    request.inputRevisions.clear();
    const auto captureRevision = [&](const std::string &guid) {
        if (!guid.empty())
            request.inputRevisions.emplace_back(guid, AssetDependencyGraph::Instance().GetContentRevision(guid));
    };
    const auto captureShader = [&](const ShaderAssetReference &shader, const char *stage) {
        if (!shader.guid.empty())
            captureRevision(shader.guid);
        else if (database && shader.pathHint.empty() && !shader.shaderId.empty())
            captureRevision(database->GetGuidFromCatalogPath(database->FindShaderPathById(shader.shaderId, stage)));
    };
    captureRevision(request.assetGuid);
    captureShader(request.material->GetVertShaderReference(), "vertex");
    captureShader(request.material->GetFragShaderReference(), "fragment");
    for (const auto &[name, property] : request.material->GetAllProperties()) {
        if (property.type == MaterialPropertyType::Texture2D)
            if (const auto *guid = std::get_if<std::string>(&property.value))
                captureRevision(*guid);
    }
    if (!m_renderer->CanPreviewMaterialOnMesh(request.material))
        request.raster = std::make_shared<MaterialPreviewRaster>();
    return true;
}

int Infernux::AdvanceMaterialPreview(bool gpu, int uploadBudget, bool &progressed)
{
    int consumed = 0;
    if (!gpu) {
        std::lock_guard lock(m_previewResultMutex);
        for (auto &[key, state] : m_materialPreviewStates) {
            if (consumed >= uploadBudget)
                break;
            if (!state.renderTicket || !state.renderTicket->IsComplete())
                continue;

            const auto generation = std::exchange(state.renderGeneration, 0);
            auto image = std::move(state.renderTicket);
            RefreshMaterialPreviewDependency(state);
            progressed = true;
            if (state.generation != generation) {
                ScheduleDirtyMaterialPreview(key, state);
                continue;
            }
            try {
                const auto textureId = m_renderer->PublishPreviewImageForImGui(state.textureName, image);
                if (textureId) {
                    state.readyGeneration = generation;
                    state.readyPixels.reset();
                    ++consumed;
                    continue;
                }
            } catch (const std::exception &error) {
                INXLOG_ERROR("Failed to publish GPU material preview: ", error.what());
            }
            state.failedGeneration = generation;
        }
        if (consumed >= uploadBudget)
            return consumed;
    }

    MaterialPreviewRequest request;
    {
        std::lock_guard lock(m_previewResultMutex);
        const auto step = gpu ? PreviewStep::Gpu : PreviewStep::Host;
        const auto next = std::find_if(
            m_materialPreviewRequestQueue.begin(), m_materialPreviewRequestQueue.end(),
            [this, step](const auto &candidate) { return DescribeMaterialPreviewRequest(candidate) == step; });
        if (next == m_materialPreviewRequestQueue.end())
            return consumed;

        request = std::move(*next);
        m_materialPreviewRequestQueue.erase(next);
        const auto found = m_materialPreviewStates.find(request.resourceKey);
        if (found == m_materialPreviewStates.end() || found->second.pendingGeneration != request.generation) {
            progressed = true;
            return consumed;
        }

        auto &state = found->second;
        if (RefreshMaterialPreviewDependency(state) || request.generation != state.generation) {
            request = {request.resourceKey, state.latestMatFilePath, state.generation, state.latestMaterialJson};
            state.pendingGeneration = request.generation;
            progressed = true;
        }
    }

    const auto defer = [&] {
        std::lock_guard lock(m_previewResultMutex);
        const auto state = m_materialPreviewStates.find(request.resourceKey);
        if (state != m_materialPreviewStates.end() && state->second.pendingGeneration == request.generation) {
            m_materialPreviewRequestQueue.push_back(std::move(request));
            m_hasPreviewPumpWork.store(true, std::memory_order_release);
        }
    };
    const auto finish = [&](std::shared_ptr<PreviewImage> image) {
        std::lock_guard lock(m_previewResultMutex);
        const auto state = m_materialPreviewStates.find(request.resourceKey);
        if (state == m_materialPreviewStates.end() || state->second.pendingGeneration != request.generation)
            return;
        auto &current = state->second;
        current.pendingGeneration = 0;
        if (image) {
            current.renderGeneration = request.generation;
            current.renderTicket = std::move(image);
            if (current.generation == request.generation)
                current.submittedInputs =
                    std::make_shared<const decltype(request.inputRevisions)>(std::move(request.inputRevisions));
            m_hasPreviewPumpWork.store(true, std::memory_order_release);
        } else {
            current.failedGeneration = request.generation;
        }
        ScheduleDirtyMaterialPreview(state->first, current);
        progressed = true;
    };

    try {
        if (!gpu) {
            if (!request.material || !IsMaterialPreviewSourceCurrent(request)) {
                const bool prepared = PrepareMaterialPreview(request, progressed);
                if (prepared && !request.material) {
                    finish(nullptr);
                    return consumed;
                }
                progressed |= prepared;
            } else {
                auto &raster = request.raster;
                if (kMaterialPreviewRasterBytes > m_frameWork.CpuStorageLimit())
                    throw std::length_error("Material preview raster exceeds host CPU storage capacity");
                raster->storage = m_frameWork.TryReserveCpuStorage(kMaterialPreviewRasterBytes);
                if (raster->storage) {
                    // The ordinary clone receives shader updates on the host.
                    // A private copy has no live shader registrations or GPU bindings.
                    auto frozen = std::make_shared<InxMaterial>(*request.material);
                    for (const auto &binding : request.material->GetBuffers())
                        frozen->SetBuffer(binding.first, nullptr);
                    EnqueuePreviewTask(
                        [weak = std::weak_ptr(raster), material = std::shared_ptr<const InxMaterial>(std::move(frozen)),
                         database = GetAssetDatabase()] {
                            if (const auto work = weak.lock()) {
                                try {
                                    InxTextureData image;
                                    image.width = image.height = kMaterialPreviewSize;
                                    MaterialPreviewer::RenderCpuPreview(*material, kMaterialPreviewSize, image.pixels,
                                                                        database);
                                    if (image.IsValid()) {
                                        work->pixels = TextureDecoder::TakePreview(std::move(image), false);
                                        work->summary = SummarizePreviewPixels(*work->pixels);
                                    }
                                } catch (...) {
                                    work->failure = std::current_exception();
                                }
                                work->complete.store(true, std::memory_order_release);
                            }
                        },
                        std::weak_ptr<const void>(raster));
                    progressed = true;
                }
            }
            defer();
            return consumed;
        }

        if (!request.material) {
            defer();
            return consumed;
        }
        if (m_renderer->CanPreviewMaterialOnMesh(request.material) == static_cast<bool>(request.raster)) {
            request.material.reset();
            request.raster.reset();
            defer();
            progressed = true;
            return consumed;
        }
        if (!request.raster) {
            auto image = m_renderer->BeginMaterialPreviewGPU(request.material, kMaterialPreviewSize, progressed);
            if (!image)
                defer();
            else {
                finish(std::move(image));
                ++consumed;
            }
            return consumed;
        }

        auto &raster = *request.raster;
        if (raster.failure)
            std::rethrow_exception(raster.failure);
        if (!raster.pixels) {
            finish(nullptr);
            return consumed;
        }
        const auto &summary = raster.summary;
        std::lock_guard lock(m_previewResultMutex);
        const auto found = m_materialPreviewStates.find(request.resourceKey);
        if (found == m_materialPreviewStates.end() || found->second.pendingGeneration != request.generation)
            return consumed;
        auto &state = found->second;
        RefreshMaterialPreviewDependency(state);
        if (state.generation != request.generation) {
            state.pendingGeneration = 0;
            ScheduleDirtyMaterialPreview(found->first, state);
            progressed = true;
            return consumed;
        }
        auto inputs = std::make_shared<const decltype(request.inputRevisions)>(std::move(request.inputRevisions));
        const auto canPublish = [this, key = request.resourceKey, generation = request.generation, inputs] {
            std::lock_guard stateLock(m_previewResultMutex);
            const auto current = m_materialPreviewStates.find(key);
            return current != m_materialPreviewStates.end() && current->second.generation == generation &&
                   MaterialPreviewInputsCurrent(*inputs);
        };
        const auto uploadVersion =
            m_renderer->SubmitTextureForImGui(state.textureName, std::move(raster.pixels), rhi::FilterMode::Linear,
                                              false, canPublish, std::move(raster.storage));
        state.pendingGeneration = 0;
        if (!uploadVersion) {
            state.failedGeneration = request.generation;
            progressed = true;
            return consumed;
        }
        state.submittedInputs = std::move(inputs);
        state.upload = {uploadVersion, request.generation, kMaterialPreviewSize, kMaterialPreviewSize, summary};
        m_hasPendingPreviewUploads.store(true, std::memory_order_release);
        m_hasPreviewPumpWork.store(true, std::memory_order_release);
        progressed = true;
        ++consumed;
    } catch (const std::exception &error) {
        INXLOG_ERROR("Material preview rejected for '", request.resourceKey, "': ", error.what());
        finish(nullptr);
    }
    return consumed;
}

bool Infernux::CommitPublishedPreviewTextures()
{
    if (!m_hasPendingPreviewUploads.exchange(false, std::memory_order_acq_rel))
        return false;

    bool progressed = false;
    bool hasUnpublishedUploads = false;
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    const auto commitUpload = [this, &progressed, &hasUnpublishedUploads](auto &state) {
        const auto &upload = state.upload;
        if (!upload.version)
            return false;
        const auto published = m_renderer->GetImGuiTextureVersion(state.textureName);
        if (published == upload.version) {
            state.readyGeneration = upload.generation;
            state.readyPixels = upload.summary;
        } else if (m_renderer->GetFailedImGuiTextureVersion(state.textureName) >= upload.version) {
            state.failedGeneration = upload.generation;
        } else if (published < upload.version) {
            hasUnpublishedUploads = true;
            return false;
        }
        state.upload = {};
        progressed = true;
        return true;
    };
    for (auto &[key, state] : m_materialPreviewStates) {
        if (state.upload.version && RefreshMaterialPreviewDependency(state)) {
            ScheduleDirtyMaterialPreview(key, state);
            progressed = true;
        } else if (commitUpload(state)) {
            ScheduleDirtyMaterialPreview(key, state);
        }
    }
    for (auto &[key, state] : m_texturePreviewStates) {
        (void)key;
        commitUpload(state);
    }
    if (hasUnpublishedUploads) {
        m_hasPendingPreviewUploads.store(true, std::memory_order_release);
        m_hasPreviewPumpWork.store(true, std::memory_order_release);
    }
    return progressed;
}

void Infernux::PumpPreviewTasks()
{
    if (!m_renderer)
        return;
    if (!m_previewWorkSource) {
        FrameWorkSource source;
        source.interactive = true;
        source.observe = [this] {
            ObserveSubmittedPreviews();
            PollGpuCompletions();
        };
        source.refresh = source.observe;
        source.describe = [this] { return DescribePreviewWork(); };
        source.advance = [this](bool gpu, uint64_t lane) { return AdvancePreviewTasks(gpu, lane); };
        m_previewWorkSource = m_frameWork.AddSource(std::move(source));
    }
    m_frameWork.Wake(m_previewWorkSource);
}

FrameWorkFacts Infernux::DescribePreviewWork()
{
    FrameWorkFacts facts;
    facts.gpuBusy = m_renderer->HasPendingBackgroundGpuWork();
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    const auto completed = [](const auto &entry) {
        return entry.second.renderTicket && entry.second.renderTicket->IsComplete();
    };
    const auto readiness = [](const auto &queue, const auto &describe) {
        bool host = false;
        bool gpu = false;
        for (const auto &request : queue) {
            const auto step = describe(request);
            host |= step == PreviewStep::Host;
            gpu |= step == PreviewStep::Gpu;
            if (host && gpu)
                break;
        }
        return std::pair{host, gpu};
    };
    const auto [materialPrepareReady, materialGpuReady] = readiness(
        m_materialPreviewRequestQueue, [this](const auto &request) { return DescribeMaterialPreviewRequest(request); });
    const auto [meshPrepareReady, meshGpuReady] = readiness(
        m_meshPreviewRequestQueue, [this](const auto &request) { return DescribeMeshPreviewRequest(request); });
    const bool materialHostReady =
        materialPrepareReady || std::any_of(m_materialPreviewStates.begin(), m_materialPreviewStates.end(), completed);
    const bool meshHostReady =
        meshPrepareReady || std::any_of(m_meshPreviewStates.begin(), m_meshPreviewStates.end(), completed);
    const bool hostReady[] = {
        std::any_of(m_texturePreviewStates.begin(), m_texturePreviewStates.end(),
                    [](const auto &entry) { return entry.second.importedGpuPending; }),
        materialHostReady,
        false,
        meshHostReady,
        false,
    };
    for (uint64_t offset = 0; offset < std::size(hostReady); ++offset) {
        const auto lane = (m_nextPreviewHostLane + offset) % std::size(hostReady);
        if (hostReady[lane]) {
            facts.host = {lane, 0, true, false};
            break;
        }
    }
    const bool materialRenderPending = std::any_of(m_materialPreviewStates.begin(), m_materialPreviewStates.end(),
                                                   [](const auto &entry) { return bool(entry.second.renderTicket); });
    const bool ready[] = {
        false,
        !materialRenderPending && materialGpuReady,
        !m_texturePreviewCompletedQueue.empty(),
        meshGpuReady,
        m_timelineCubeDirty,
    };
    for (uint64_t offset = 0; offset < std::size(ready); ++offset) {
        const auto lane = (m_nextPreviewLane + offset) % std::size(ready);
        if (ready[lane]) {
            facts.gpu = {lane, 0, true, false};
            break;
        }
    }
    const auto waiting = [](const auto &entry) { return entry.second.HasPendingWork(); };
    facts.pending = facts.host.ready || facts.gpu.ready ||
                    std::any_of(m_texturePreviewStates.begin(), m_texturePreviewStates.end(), waiting) ||
                    std::any_of(m_materialPreviewStates.begin(), m_materialPreviewStates.end(),
                                [](const auto &entry) { return entry.second.HasPendingWork(); }) ||
                    std::any_of(m_meshPreviewStates.begin(), m_meshPreviewStates.end(), [](const auto &entry) {
                        return entry.second.pendingGeneration || entry.second.renderTicket;
                    });
    facts.observe = facts.pending || facts.gpuBusy || m_hasPendingPreviewUploads.load(std::memory_order_acquire);
    return facts;
}

bool Infernux::AdvancePreviewTasks(bool gpu, uint64_t lane)
{
    if (gpu)
        m_nextPreviewLane = (lane + 1) % 5;
    else
        m_nextPreviewHostLane = (lane + 1) % 5;
    bool progressed = false;

    // Registered project textures never enter the CPU thumbnail pipeline.
    // A pump-disabled query records only its GUID; the first normal preview
    // pump resolves the immutable runtime GPU publication and exposes that
    // same view to ImGui.
    struct ImportedPreviewPublication
    {
        std::string key;
        std::string textureName;
        std::string guid;
    };
    std::vector<ImportedPreviewPublication> importedPublications;
    // Resolving a preview prepares CPU data or binds a completed GPU view.
    // The renderer's texture source owns the actual GPU upload separately.
    if (!gpu && lane == 0) {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        for (const auto &[key, state] : m_texturePreviewStates) {
            if (state.importedGpuPending && !state.importedTextureGuid.empty())
                importedPublications.push_back({key, state.textureName, state.importedTextureGuid});
        }
    }
    std::sort(importedPublications.begin(), importedPublications.end(),
              [](const auto &left, const auto &right) { return left.key < right.key; });
    const auto nextImported = std::find_if(importedPublications.begin(), importedPublications.end(),
                                           [this](const auto &item) { return item.key > m_lastImportedPreviewKey; });
    if (nextImported != importedPublications.end())
        std::rotate(importedPublications.begin(), nextImported, importedPublications.end());
    for (const auto &publication : importedPublications) {
        m_lastImportedPreviewKey = publication.key;
        auto &registry = AssetRegistry::Instance();
        const auto content = registry.GetAssetContentGeneration(publication.guid);
        const uint64_t textureId =
            m_renderer->QueryImportedTextureForImGui(publication.textureName, publication.guid, progressed);
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        const auto stateIt = m_texturePreviewStates.find(publication.key);
        if (stateIt == m_texturePreviewStates.end() || stateIt->second.importedTextureGuid != publication.guid)
            break;
        if (textureId == 0) {
            m_hasPreviewPumpWork.store(true, std::memory_order_release);
            m_renderer->RequestFullSpeedFrame();
            break;
        }
        stateIt->second.readyGeneration = stateIt->second.generation;
        stateIt->second.readyPixels.reset();
        stateIt->second.importedGpuPending = false;
        stateIt->second.importedContentGeneration = content;
        stateIt->second.importedRuntimeVersion = registry.GetAssetVersion(publication.guid);
        progressed = true;
        break;
    }

    int uploadBudget = 1; // One advance selected by the common frame owner.

    if (lane == 1)
        uploadBudget -= AdvanceMaterialPreview(gpu, uploadBudget, progressed);

    // ── Process completed texture uploads ────────────────────────
    if (lane == 2) {
        std::lock_guard lock(m_previewResultMutex);
        while (!m_texturePreviewCompletedQueue.empty() && uploadBudget > 0) {
            TexturePreviewCompleted completed = std::move(m_texturePreviewCompletedQueue.front());
            m_texturePreviewCompletedQueue.pop_front();
            progressed = true; // Consume this distinct completion, including a retired generation.

            const auto found = m_texturePreviewStates.find(completed.resourceKey);
            if (found == m_texturePreviewStates.end() || found->second.decodeGeneration != completed.generation)
                continue;
            auto &state = found->second;
            state.decodeGeneration = 0;
            state.fileRequest.reset();
            if (state.generation != completed.generation)
                continue;

            if (!completed.pixels) {
                state.failedGeneration = completed.generation;
                continue;
            }

            const auto canPublish = [this, key = completed.resourceKey, generation = completed.generation] {
                std::lock_guard stateLock(m_previewResultMutex);
                const auto current = m_texturePreviewStates.find(key);
                return current != m_texturePreviewStates.end() && current->second.generation == generation;
            };
            try {
                const auto &mip = completed.pixels->mipLevels.front();
                const auto width = static_cast<int>(mip.width);
                const auto height = static_cast<int>(mip.height);
                const auto version =
                    completed.displayEncoding
                        ? m_renderer->SubmitDocumentTextureForImGui(state.textureName, std::move(completed.pixels),
                                                                    canPublish, std::move(completed.storage))
                        : m_renderer->SubmitTextureForImGui(state.textureName, std::move(completed.pixels),
                                                            completed.nearest ? rhi::FilterMode::Nearest
                                                                              : rhi::FilterMode::Linear,
                                                            false, canPublish, std::move(completed.storage));
                if (!version) {
                    state.failedGeneration = completed.generation;
                    continue;
                }
                state.upload = {version, completed.generation, width, height, completed.summary};
                m_hasPendingPreviewUploads.store(true, std::memory_order_release);
                m_hasPreviewPumpWork.store(true, std::memory_order_release);
                --uploadBudget;
            } catch (const std::exception &error) {
                INXLOG_ERROR("Failed to submit image preview texture: ", error.what());
                state.failedGeneration = completed.generation;
            }
        }

        if (!m_texturePreviewCompletedQueue.empty())
            m_hasPreviewPumpWork.store(true, std::memory_order_release);
    }

    // ── Mesh preview render + direct GPU publication ─────────────────
    if (lane == 3) {
        constexpr int kMeshPreviewSize = 256;
        if (!gpu) {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            for (auto &[key, state] : m_meshPreviewStates) {
                if (uploadBudget <= 0)
                    break;
                if (!state.renderTicket || !state.renderTicket->IsComplete())
                    continue;
                const auto generation = std::exchange(state.renderGeneration, 0);
                auto image = std::move(state.renderTicket);
                RefreshMeshPreviewDependency(state);
                progressed = true;
                if (state.generation != generation) {
                    ScheduleDirtyMeshPreview(key, state);
                    continue;
                }
                try {
                    if (m_renderer->PublishPreviewImageForImGui(state.textureName, image)) {
                        state.readyGeneration = generation;
                        --uploadBudget;
                        continue;
                    }
                } catch (const std::exception &error) {
                    INXLOG_ERROR("Failed to publish GPU mesh preview: ", error.what());
                }
                state.failedGeneration = generation;
            }
        }

        bool renderBusy = false;
        MeshPreviewRequest request;
        {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            renderBusy = std::any_of(m_meshPreviewStates.begin(), m_meshPreviewStates.end(),
                                     [](const auto &entry) { return bool(entry.second.renderTicket); });
            if (uploadBudget > 0) {
                const auto step = gpu ? PreviewStep::Gpu : PreviewStep::Host;
                const auto next = std::find_if(
                    m_meshPreviewRequestQueue.begin(), m_meshPreviewRequestQueue.end(),
                    [this, step](const auto &candidate) { return DescribeMeshPreviewRequest(candidate) == step; });
                if (next != m_meshPreviewRequestQueue.end()) {
                    request = std::move(*next);
                    m_meshPreviewRequestQueue.erase(next);
                }
            }
        }
        if (!request.resourceKey.empty()) {
            const auto refreshRequest = [&]() {
                std::lock_guard<std::mutex> lock(m_previewResultMutex);
                const auto state = m_meshPreviewStates.find(request.resourceKey);
                if (state == m_meshPreviewStates.end() || state->second.pendingGeneration != request.generation) {
                    progressed = true; // Retire the popped request; do not requeue it.
                    return false;
                }
                progressed |= RefreshMeshPreviewDependency(state->second);
                if (request.generation != state->second.generation ||
                    request.meshFilePath != state->second.meshFilePath) {
                    // A private candidate belongs to its original inputs. Drop
                    // it on change; never relabel old bytes/geometry as latest.
                    request.geometry.reset();
                    request = {request.resourceKey, state->second.meshFilePath, state->second.generation};
                    state->second.pendingGeneration = request.generation;
                    progressed = true;
                }
                return true;
            };
            const auto deferRequest = [&]() {
                std::lock_guard<std::mutex> lock(m_previewResultMutex);
                const auto state = m_meshPreviewStates.find(request.resourceKey);
                if (state != m_meshPreviewStates.end() && state->second.pendingGeneration == request.generation) {
                    m_meshPreviewRequestQueue.push_back(std::move(request));
                    m_hasPreviewPumpWork.store(true, std::memory_order_release);
                }
            };
            try {
                if (refreshRequest()) {
                    std::shared_ptr<InxMesh> mesh;
                    std::vector<std::shared_ptr<InxMaterial>> materials;
                    const bool ready = PrepareMeshPreview(request, gpu, mesh, materials, progressed);
                    const auto prepared = request.preparation;
                    if (refreshRequest()) {
                        if (!ready || prepared != request.preparation || renderBusy) {
                            deferRequest();
                        } else if (!mesh) {
                            progressed |= FinishMeshPreviewPreparation(request, nullptr);
                        } else {
                            bool materialsChanged = false;
                            auto ticket = m_renderer->BeginMeshPreviewGPU(
                                *mesh, materials, kMeshPreviewSize, request.geometry, progressed, materialsChanged);
                            if (!ticket) {
                                if (materialsChanged) {
                                    request.geometry.reset();
                                    request.preparation.reset();
                                    progressed = true;
                                }
                                deferRequest();
                            } else {
                                progressed |= FinishMeshPreviewPreparation(request, std::move(ticket));
                            }
                        }
                    }
                }
            } catch (const std::exception &error) {
                INXLOG_ERROR("Mesh preview preparation failed for '", request.meshFilePath, "': ", error.what());
                progressed |= FinishMeshPreviewPreparation(request, nullptr);
            }
        }
    }

    // A submitted preview still needs later frames for true GPU completion and
    // publication. Do not enter the editor's idle cadence between those
    // stages, including completed tickets deferred by this frame's budget.
    bool gpuPublicationPending = m_hasPendingPreviewUploads.load(std::memory_order_acquire);
    {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        gpuPublicationPending = gpuPublicationPending ||
                                std::any_of(m_materialPreviewStates.begin(), m_materialPreviewStates.end(),
                                            [](const auto &entry) { return bool(entry.second.renderTicket); }) ||
                                std::any_of(m_meshPreviewStates.begin(), m_meshPreviewStates.end(),
                                            [](const auto &entry) { return bool(entry.second.renderTicket); });
        if (!m_meshPreviewRequestQueue.empty())
            m_hasPreviewPumpWork.store(true, std::memory_order_release);
    }
    if (gpuPublicationPending) {
        m_hasPreviewPumpWork.store(true, std::memory_order_release);
        m_renderer->RequestFullSpeedFrame();
    }

    if (lane == 4)
        progressed |= PumpTimelineCubePreviewIfDirty();
    return progressed;
}

void Infernux::PollGpuCompletions()
{
    if (!m_renderer)
        return;
    // Retiring foreground GPU work is part of advancing the renderer itself.
    m_renderer->PollGpuCompletions();
    FrameWorkOwner::Scope scope(m_frameWork);
    if (CommitPublishedPreviewTextures())
        m_frameWork.ReportProgress();
}

bool Infernux::IsGpuSubmissionCompleted(rhi::SubmissionTicket ticket) const noexcept
{
    if (!m_gpuServices || !ticket.IsValid() || ticket.queue >= rhi::QueueRole::Count)
        return false;
    const auto &queues = m_gpuServices->Queues();
    return ticket.device == queues.GetDeviceId() && ticket.serial <= queues.GetCompletedSerial(ticket.queue);
}

uint64_t Infernux::GetMaterialPreviewTextureId(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_materialPreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (it == m_materialPreviewStates.end())
        return 0;
    return LiveImGuiTextureId(m_renderer.get(), it->second.textureName);
}

bool Infernux::IsMaterialPreviewReady(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_materialPreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (it == m_materialPreviewStates.end())
        return false;
    const auto &state = it->second;
    return state.generation != 0 && state.readyGeneration == state.generation &&
           LiveImGuiTextureId(m_renderer.get(), state.textureName) != 0;
}

std::vector<Infernux::PreviewTaskSnapshot> Infernux::GetPreviewTaskSnapshots() const
{
    std::unordered_map<uint64_t, uint32_t> imguiTextureUseCounts;
    if (const ImDrawData *drawData = ImGui::GetDrawData(); drawData && drawData->Valid) {
        for (int listIndex = 0; listIndex < drawData->CmdListsCount; ++listIndex) {
            const ImDrawList *drawList = drawData->CmdLists[listIndex];
            if (!drawList)
                continue;
            for (const ImDrawCmd &command : drawList->CmdBuffer) {
                const ImTextureID textureId = command.GetTexID();
                uint64_t value = 0;
                static_assert(sizeof(textureId) <= sizeof(value));
                std::memcpy(&value, &textureId, sizeof(textureId));
                if (value != 0)
                    ++imguiTextureUseCounts[value];
            }
        }
    }

    std::vector<PreviewTaskSnapshot> snapshots;
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    snapshots.reserve(m_materialPreviewStates.size() + m_texturePreviewStates.size() + m_meshPreviewStates.size());

    const auto finish = [this, &imguiTextureUseCounts](PreviewTaskSnapshot &snapshot,
                                                       const PreviewPixelSummary *pixels = nullptr) {
        const auto published = LiveImGuiTexture(m_renderer.get(), snapshot.textureName);
        snapshot.textureId = published.textureId;
        snapshot.readyWidth = published.width;
        snapshot.readyHeight = published.height;
        snapshot.publishedUploadVersion = published.uploadVersion;
        snapshot.imguiDrawCommandCount = imguiTextureUseCounts[snapshot.textureId];
        if (m_renderer)
            snapshot.failedUploadVersion = m_renderer->GetFailedImGuiTextureVersion(snapshot.textureName);
        if (!published.textureId)
            snapshot.readyGeneration = 0;
        if (snapshot.readyGeneration && pixels) {
            snapshot.pixelGeneration = snapshot.readyGeneration;
            snapshot.pixelHash = pixels->hash;
            snapshot.nonTransparentPixelCount = pixels->nonTransparentPixelCount;
            snapshot.minRgb = pixels->minRgb;
            snapshot.maxRgb = pixels->maxRgb;
        }
    };

    for (const auto &[key, state] : m_materialPreviewStates) {
        PreviewTaskSnapshot snapshot;
        snapshot.kind = "material";
        snapshot.resourceKey = key;
        snapshot.textureName = state.textureName;
        snapshot.generation = state.generation;
        snapshot.readyGeneration = state.readyGeneration;
        snapshot.pendingUploadVersion = state.upload.version;
        snapshot.pendingPreviewGeneration = state.upload.generation;
        snapshot.inFlight = state.HasPendingWork();
        snapshot.authoring = state.authoring;
        snapshot.hasRenderTicket = static_cast<bool>(state.renderTicket);
        snapshot.renderTicketDone = state.renderTicket && state.renderTicket->IsComplete();
        snapshot.pendingWidth = state.upload.width;
        snapshot.pendingHeight = state.upload.height;
        finish(snapshot, state.readyPixels ? &*state.readyPixels : nullptr);
        snapshots.push_back(std::move(snapshot));
    }

    for (const auto &[key, state] : m_texturePreviewStates) {
        PreviewTaskSnapshot snapshot;
        snapshot.kind = "texture";
        snapshot.resourceKey = key;
        snapshot.textureName = state.textureName;
        snapshot.generation = state.generation;
        snapshot.readyGeneration = state.readyGeneration;
        snapshot.pendingUploadVersion = state.upload.version;
        snapshot.pendingPreviewGeneration = state.upload.generation;
        snapshot.inFlight = state.HasPendingWork();
        snapshot.authoring = state.authoring;
        snapshot.pendingWidth = state.upload.width;
        snapshot.pendingHeight = state.upload.height;
        finish(snapshot, state.readyPixels ? &*state.readyPixels : nullptr);
        snapshots.push_back(std::move(snapshot));
    }

    for (const auto &[key, state] : m_meshPreviewStates) {
        PreviewTaskSnapshot snapshot;
        snapshot.kind = "mesh";
        snapshot.resourceKey = key;
        snapshot.textureName = state.textureName;
        snapshot.generation = state.generation;
        snapshot.readyGeneration = state.readyGeneration;
        snapshot.inFlight = state.pendingGeneration || state.renderTicket;
        snapshot.pendingPreviewGeneration = state.renderGeneration;
        snapshot.hasRenderTicket = static_cast<bool>(state.renderTicket);
        snapshot.renderTicketDone = state.renderTicket && state.renderTicket->IsComplete();
        finish(snapshot);
        snapshots.push_back(std::move(snapshot));
    }

    return snapshots;
}

uint64_t Infernux::GetTexturePreviewTextureId(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_texturePreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (it == m_texturePreviewStates.end())
        return 0;
    return LiveImGuiTextureId(m_renderer.get(), it->second.textureName);
}

uint64_t Infernux::GetMeshPreviewTextureId(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_meshPreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (it == m_meshPreviewStates.end())
        return 0;
    return LiveImGuiTextureId(m_renderer.get(), it->second.textureName);
}

std::pair<int, int> Infernux::GetTexturePreviewSize(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_texturePreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (it == m_texturePreviewStates.end())
        return {0, 0};
    const auto published = LiveImGuiTexture(m_renderer.get(), it->second.textureName);
    return {published.width, published.height};
}

void Infernux::InvalidateMaterialPreviewTask(const std::string &resourceKey)
{
    if (resourceKey.empty())
        return;

    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_materialPreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (it != m_materialPreviewStates.end())
        BumpMaterialPreviewGeneration(it->second);
}

void Infernux::InvalidateTexturePreviewTask(const std::string &resourceKey)
{
    if (resourceKey.empty())
        return;

    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_texturePreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (it == m_texturePreviewStates.end())
        return;

    // Keep the last imported preview visible while its replacement is prepared.
    it->second.generation++;
    it->second.lastContentStamp = 0;
    RetireTexturePreviewDecode(it->first, it->second);
    it->second.upload = {};
    if (!it->second.importedTextureGuid.empty()) {
        it->second.importedGpuPending = true;
        m_hasPreviewPumpWork.store(true, std::memory_order_release);
        m_renderer->RequestFullSpeedFrame();
    }
}

void Infernux::ReleaseTexturePreviewTask(const std::string &resourceKey)
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    const auto found = m_texturePreviewStates.find(CanonicalizePreviewKey(resourceKey));
    if (found == m_texturePreviewStates.end())
        return;
    auto &state = found->second;
    if (state.generation == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("Texture preview generation overflow");
    const uint64_t retiredGeneration = state.generation + 1;
    if (m_renderer && !state.textureName.empty())
        m_renderer->RemoveImGuiTexture(state.textureName);
    // Keep the epoch when a streaming slot is reused. Both an old worker
    // result and an already submitted GPU upload must remain retired.
    RetireTexturePreviewDecode(found->first, state);
    state = TexturePreviewState{};
    state.generation = retiredGeneration;
}

void Infernux::ReleaseMaterialPreviewTask(const std::string &resourceKey)
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    const std::string key = CanonicalizePreviewKey(resourceKey);
    const auto found = m_materialPreviewStates.find(key);
    if (found == m_materialPreviewStates.end())
        return;
    auto &state = found->second;
    if (state.generation == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("Material preview generation overflow");
    const uint64_t retiredGeneration = state.generation + 1;
    if (m_renderer && !state.textureName.empty())
        m_renderer->RemoveImGuiTexture(state.textureName);
    state = MaterialPreviewState{};
    state.generation = retiredGeneration;
    m_materialPreviewRequestQueue.erase(
        std::remove_if(m_materialPreviewRequestQueue.begin(), m_materialPreviewRequestQueue.end(),
                       [&key](const auto &request) { return request.resourceKey == key; }),
        m_materialPreviewRequestQueue.end());
}

void Infernux::ReleaseAssetPreviewTasks(const std::string &assetPath)
{
    if (assetPath.empty())
        return;
    const std::string path = CanonicalizePreviewKey("asset|" + assetPath).substr(6);
    const auto belongsToSource = [&path](const std::string &key) {
        const size_t separator = key.find('|');
        if (separator == std::string::npos)
            return false;
        const size_t start = separator + 1;
        if (key.compare(start, path.size(), path) != 0)
            return false;
        const size_t end = start + path.size();
        return end == key.size() || key.compare(end, 2, "::") == 0;
    };
    std::vector<std::string> materials;
    std::vector<std::string> textures;
    {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        for (const auto &[key, state] : m_materialPreviewStates) {
            if (belongsToSource(state.latestMatFilePath.empty()
                                    ? key
                                    : CanonicalizePreviewKey("material|" + state.latestMatFilePath)))
                materials.push_back(key);
        }
        for (const auto &[key, state] : m_texturePreviewStates) {
            if (belongsToSource(key))
                textures.push_back(key);
        }
        for (auto &[key, state] : m_meshPreviewStates) {
            if (!belongsToSource(CanonicalizePreviewKey("mesh|" + state.meshFilePath)))
                continue;
            if (state.generation == std::numeric_limits<uint64_t>::max())
                throw std::overflow_error("Mesh preview generation overflow");
            const uint64_t retiredGeneration = state.generation + 1;
            if (m_renderer && !state.textureName.empty())
                m_renderer->RemoveImGuiTexture(state.textureName);
            state = MeshPreviewState{};
            state.generation = retiredGeneration;
        }
        m_meshPreviewRequestQueue.erase(
            std::remove_if(m_meshPreviewRequestQueue.begin(), m_meshPreviewRequestQueue.end(),
                           [this](const auto &request) {
                               const auto state = m_meshPreviewStates.find(request.resourceKey);
                               return state == m_meshPreviewStates.end() ||
                                      state->second.pendingGeneration != request.generation;
                           }),
            m_meshPreviewRequestQueue.end());
    }
    for (const auto &key : materials)
        ReleaseMaterialPreviewTask(key);
    for (const auto &key : textures)
        ReleaseTexturePreviewTask(key);
}

void Infernux::ReleasePreviewAuthoring(const std::string &resourceKey)
{
    if (resourceKey.empty())
        return;

    const std::string key = CanonicalizePreviewKey(resourceKey);
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    if (auto material = m_materialPreviewStates.find(key); material != m_materialPreviewStates.end()) {
        material->second.authoring = false;
        material->second.lastFileMtime = 0;
    }
    if (auto texture = m_texturePreviewStates.find(key); texture != m_texturePreviewStates.end()) {
        texture->second.authoring = false;
        texture->second.lastContentStamp = 0;
    }
}

std::tuple<uint64_t, int, int>
Infernux::QueryOrScheduleTexturePreview(const std::string &resourceKey, const std::string &textureFilePath,
                                        uint64_t contentStampHint, bool nearest, bool srgb, int maxSize,
                                        const std::string &textureFormat, const std::string &textureType,
                                        bool authoring, bool pump, bool useImportedTexture)
{
    if (resourceKey.empty() || textureFilePath.empty())
        return {0, 0, 0};

    if (pump)
        PumpPreviewTasks();

    const std::string key = CanonicalizePreviewKey(resourceKey);

    // Asset texture previews, including sprite slicing, present the
    // final imported GPU publication. This keeps BC formats, sRGB conversion,
    // mip policy and runtime sampling identical across Project, Inspector,
    // sprite authoring and scene rendering.
    // Document illustrations use source pixels, independently of game asset
    // compression, channel conversion and sampler settings.
    if (auto *database = GetAssetDatabase(); useImportedTexture && database && m_renderer) {
        const std::string guid = database->GetGuidFromPath(textureFilePath);
        const auto metadata = guid.empty() ? nullptr : database->GetMetaByGuid(guid);
        // Only documentation-policy imports are faithful source illustrations.
        // Uninstalled packages and images referenced outside plugin_pages use
        // the lossless CPU decoder below, not an arbitrary BC game texture.
        const bool documentPreview = key.compare(0, 9, "document|") == 0;
        if (!guid.empty() &&
            (!documentPreview ||
             (metadata && IsPluginPageTexture(*metadata, textureFilePath, database->GetProjectRoot()) &&
              HasCurrentPluginPageTextureMetadata(*metadata, textureFilePath, database->GetProjectRoot())))) {
            const std::string textureName = BuildTexturePreviewTextureName(key);
            // Queries record demand; the admitted owner action resolves and
            // publishes the imported texture. UI cannot initiate uploads here.
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            auto &state = m_texturePreviewStates[key];
            const auto published = LiveImGuiTexture(m_renderer.get(), textureName);
            if (!authoring && state.authoring)
                return {published.textureId, published.width, published.height};
            const auto &registry = AssetRegistry::Instance();
            const auto content = registry.GetAssetContentGeneration(guid);
            const auto version = registry.GetAssetVersion(guid);
            const bool differentSource = state.importedTextureGuid != guid;
            state.textureName = textureName;
            const bool changed = differentSource || state.importedContentGeneration != content ||
                                 state.importedRuntimeVersion != version ||
                                 (contentStampHint != 0 && state.lastContentStamp != contentStampHint);
            if (differentSource) {
                ++state.generation;
                RetireTexturePreviewDecode(key, state);
                state.upload = {};
                state.sourcePath.clear();
                state.importedContentGeneration = 0;
                state.importedRuntimeVersion = 0;
            }
            state.importedTextureGuid = guid;
            state.authoring = state.authoring || authoring;
            if (contentStampHint != 0)
                state.lastContentStamp = contentStampHint;
            if (changed || !published.textureId || state.importedGpuPending) {
                if (!differentSource && !state.importedGpuPending)
                    ++state.generation;
                state.importedGpuPending = true;
                m_hasPreviewPumpWork.store(true, std::memory_order_release);
                m_renderer->RequestFullSpeedFrame();
            }
            return {published.textureId, published.width, published.height};
        }
    }

    std::shared_ptr<const TexturePreviewRequest> request;
    ImGuiTexturePublication published;

    {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        auto &state = m_texturePreviewStates[key];
        if (state.textureName.empty())
            state.textureName = BuildTexturePreviewTextureName(key);

        published = LiveImGuiTexture(m_renderer.get(), state.textureName);
        if (!authoring && state.authoring)
            return {published.textureId, published.width, published.height};
        state.authoring |= authoring;

        auto sourcePath = CanonicalizePreviewKey(textureFilePath);
        const bool sourceChanged = state.sourcePath != sourcePath;
        const int sanitizedMaxSize = maxSize == 0 ? 65'536 : std::clamp(maxSize, 1, 65'536);
        if (sourceChanged || (contentStampHint && contentStampHint != state.lastContentStamp) ||
            nearest != state.nearest || srgb != state.srgb || sanitizedMaxSize != state.maxSize ||
            textureFormat != state.textureFormat || textureType != state.textureType)
            ++state.generation;
        state.sourcePath = std::move(sourcePath);
        state.importedTextureGuid.clear();
        state.importedGpuPending = false;
        if (sourceChanged || contentStampHint)
            state.lastContentStamp = contentStampHint;
        state.nearest = nearest;
        state.srgb = srgb;
        state.maxSize = sanitizedMaxSize;
        state.textureFormat = textureFormat;
        state.textureType = textureType;
        if (state.decodeGeneration && state.decodeGeneration != state.generation)
            RetireTexturePreviewDecode(key, state);
        const bool upToDate = published.textureId && state.readyGeneration == state.generation;
        if (!upToDate && state.failedGeneration != state.generation && !state.HasPendingWork()) {
            request = std::make_shared<const TexturePreviewRequest>(TexturePreviewRequest{
                key, textureFilePath, state.generation, nearest, srgb, sanitizedMaxSize, textureFormat, textureType});
            state.decodeGeneration = state.generation;
            state.fileRequest = request;
        }
    }

    if (request)
        QueueTextureFilePreview(request);
    return {published.textureId, published.width, published.height};
}

void Infernux::QueueTextureFilePreview(const std::shared_ptr<const TexturePreviewRequest> &request)
{
    constexpr int componentIconSize = 64;
    const auto &key = request->resourceKey;
    const int decodeSize =
        key.compare(0, 9, "compicon|") == 0 ? (std::min)(componentIconSize, request->maxSize) : request->maxSize;
    const bool document = key.compare(0, 9, "document|") == 0;
    const bool fullImage = document || key.compare(0, 11, "spriteedit|") == 0;
    const bool normalMap = !document && request->textureType == "normal_map";
    const std::weak_ptr weak(request);
    EnqueuePreviewTask(
        [this, weak, decodeSize, document, fullImage, normalMap] {
            const auto req = weak.lock();
            if (!req)
                return;
            try {
                TexturePreviewSource preview;
                {
                    AssetSource source(req->textureFilePath);
                    if (source.Size() > static_cast<uint64_t>((std::numeric_limits<int>::max)()))
                        throw std::length_error("texture preview source exceeds decoder limits");
                    preview.revision = source.OpenedFile()->InspectRevision();
                    if (preview.revision.data.size != source.Size())
                        throw std::runtime_error("texture preview source changed before planning");
                    preview.plan = TextureDecoder::PlanSourcePreview(source, fullImage ? 0 : decodeSize, document);
                }
                if (preview.plan.bytes) {
                    const auto transformBytes =
                        normalMap ? uint64_t(preview.plan.previewWidth) * preview.plan.previewHeight * 8 : 0;
                    EnqueuePreviewTask(
                        [this, weak, decodeSize, preview](std::shared_ptr<FrameCpuStorageClaim> storage) {
                            if (auto request = weak.lock()) {
                                const auto input = *request;
                                request.reset();
                                DecodeTextureFilePreview(input, decodeSize, &preview, std::move(storage),
                                                         [&weak] { return weak.expired(); });
                            }
                        },
                        std::max(preview.plan.bytes, transformBytes), weak);
                } else {
                    DecodeTextureFilePreview(*req, decodeSize);
                }
            } catch (const std::exception &error) {
                INXLOG_WARN("Texture preview preparation failed for ", req->textureFilePath, ": ", error.what());
                TexturePreviewCompleted completed;
                completed.resourceKey = req->resourceKey;
                completed.generation = req->generation;
                CompleteTexturePreview(std::move(completed));
            }
        },
        weak);
}

void Infernux::DecodeTextureFilePreview(const TexturePreviewRequest &req, int decodeSize,
                                        const TexturePreviewSource *source,
                                        std::shared_ptr<FrameCpuStorageClaim> storage,
                                        const std::function<bool()> &retired)
{
    TexturePreviewCompleted completed;
    completed.resourceKey = req.resourceKey;
    completed.generation = req.generation;
    completed.nearest = req.nearest;
    completed.displayEncoding = req.resourceKey.compare(0, 9, "document|") == 0;
    completed.storage = std::move(storage);
    try {
        InxTextureData pixels;
        if (source) {
            if (!completed.storage)
                throw std::length_error("texture preview exceeds host CPU storage capacity");
            AssetSource input(req.textureFilePath);
            auto &file = *input.OpenedFile();
            if (input.Size() != source->revision.data.size || !(file.InspectRevision() == source->revision))
                throw std::runtime_error("texture preview source changed before decode");
            pixels = TextureDecoder::DecodeSourcePreview(input, source->plan, retired);
            if (!(file.InspectRevision() == source->revision) ||
                !(FileInput(ToFsPath(req.textureFilePath)).InspectRevision() == source->revision))
                throw std::runtime_error("texture preview source changed during decode");
        }
        if (completed.displayEncoding && !source) {
            InxResourceMeta settings;
            settings.AddMetadata("texture_type", std::string("ui"));
            settings.AddMetadata("texture_compression", std::string("none"));
            settings.AddMetadata("texture_format", std::string("auto"));
            settings.AddMetadata("max_size", 0);
            settings.AddMetadata("srgb", true);
            settings.AddMetadata("generate_mipmaps", false);
            completed.pixels = TextureDecoder::Decode(req.textureFilePath, settings, true);
        } else if (completed.displayEncoding) {
            completed.pixels = TextureDecoder::TakePreview(std::move(pixels), true, source->plan.format);
        } else {
            if (!source) {
                pixels = InxTextureLoader::LoadFromFile(req.textureFilePath, "", decodeSize);
                if (pixels.IsValid() && req.resourceKey.compare(0, 11, "spriteedit|") != 0)
                    TextureDecoder::ResizeRgba8Preview(pixels, static_cast<uint32_t>(decodeSize));
            }
            if (pixels.IsValid()) {
                if (req.textureType == "normal_map")
                    ApplyNormalMapPreviewInPlace(pixels.pixels, pixels.width, pixels.height);
                else if (!req.srgb)
                    ApplyLinearToSrgbPreviewInPlace(pixels.pixels);
                ApplyTextureFormatPreviewInPlace(pixels.pixels, req.textureFormat);
                completed.pixels = TextureDecoder::TakePreview(std::move(pixels), false);
            }
        }
        if (completed.pixels) {
            completed.summary = SummarizePreviewPixels(*completed.pixels);
            if (completed.storage)
                completed.storage->ReduceTo(sizeof(TextureCpuData) + completed.pixels->bytes.capacity() +
                                            completed.pixels->mipLevels.capacity() * sizeof(TextureMipLevel));
        }
    } catch (const std::exception &error) {
        if (!retired || !retired())
            INXLOG_WARN("Texture preview decode failed for ", req.textureFilePath, ": ", error.what());
    }
    CompleteTexturePreview(std::move(completed));
}

void Infernux::CompleteTexturePreview(TexturePreviewCompleted completed)
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    const auto found = m_texturePreviewStates.find(completed.resourceKey);
    if (found == m_texturePreviewStates.end() || found->second.decodeGeneration != completed.generation)
        return;
    m_texturePreviewCompletedQueue.push_back(std::move(completed));
    m_hasPreviewPumpWork.store(true, std::memory_order_release);
}

void Infernux::RetireTexturePreviewDecode(const std::string &key, TexturePreviewState &state)
{
    const auto generation = std::exchange(state.decodeGeneration, 0);
    state.fileRequest.reset();
    if (!generation)
        return;
    auto &completed = m_texturePreviewCompletedQueue;
    completed.erase(std::remove_if(completed.begin(), completed.end(),
                                   [&](const auto &result) {
                                       return result.generation == generation && result.resourceKey == key;
                                   }),
                    completed.end());
}

bool Infernux::ScheduleTexturePreviewFromMemory(const std::string &resourceKey, std::vector<unsigned char> imageData,
                                                uint64_t stamp, bool nearest)
{
    if (resourceKey.empty() || imageData.empty())
        return false;

    const std::string key = CanonicalizePreviewKey(resourceKey);
    uint64_t gen = 0;
    {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        auto &state = m_texturePreviewStates[key];
        if (state.textureName.empty())
            state.textureName = BuildTexturePreviewTextureName(key);

        const bool sameSource = state.sourcePath.empty() && state.importedTextureGuid.empty() &&
                                (!stamp || stamp == state.lastContentStamp) && nearest == state.nearest;
        if (state.decodeGeneration || state.upload.version)
            return sameSource;
        if (!sameSource || !state.generation)
            ++state.generation;
        state.sourcePath.clear();
        state.importedTextureGuid.clear();
        state.importedGpuPending = false;
        if (!sameSource || stamp)
            state.lastContentStamp = stamp;
        state.nearest = nearest;

        if (state.readyGeneration == state.generation && LiveImGuiTextureId(m_renderer.get(), state.textureName))
            return true;
        if (state.failedGeneration == state.generation)
            return false;
        state.decodeGeneration = state.generation;
        gen = state.generation;
    }

    auto dataCopy = std::make_shared<std::vector<unsigned char>>(std::move(imageData));
    const std::string keyCopy = key;
    const uint64_t genCopy = gen;
    const bool nearestCopy = nearest;

    EnqueuePreviewTask([this, keyCopy, dataCopy, genCopy, nearestCopy]() {
        TexturePreviewCompleted completed;
        completed.resourceKey = keyCopy;
        completed.generation = genCopy;
        completed.nearest = nearestCopy;

        try {
            auto texData = InxTextureLoader::LoadFromMemory(dataCopy->data(), dataCopy->size());
            if (texData.IsValid()) {
                completed.pixels = TextureDecoder::TakePreview(std::move(texData), false);
                completed.summary = SummarizePreviewPixels(*completed.pixels);
            }
        } catch (const std::exception &error) {
            INXLOG_WARN("In-memory texture preview decode failed: ", error.what());
        }

        CompleteTexturePreview(std::move(completed));
    });

    return true;
}

uint64_t Infernux::GetPreviewDependencyRevision(const std::string &path) const
{
    const auto *database = GetAssetDatabase();
    if (!database || path.empty())
        return 0;
    // Catalog paths avoid filesystem IO in the GUI. Embedded previews follow the source model.
    const auto guid = database->GetGuidFromCatalogPath(path.substr(0, path.find("::")));
    return guid.empty() ? 0 : AssetDependencyGraph::Instance().GetContentRevision(guid);
}

uint64_t Infernux::QueryOrScheduleMeshPreview(const std::string &resourceKey, const std::string &meshFilePath,
                                              uint64_t fileMtimeHint)
{
    if (resourceKey.empty() || meshFilePath.empty())
        return 0;

    const std::string key = CanonicalizePreviewKey(resourceKey);
    const auto dependencyRevision = GetPreviewDependencyRevision(meshFilePath);
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto &state = m_meshPreviewStates[key];
    if (state.textureName.empty())
        state.textureName = BuildMeshPreviewTextureName(key);

    const bool sourceChanged = state.meshFilePath != meshFilePath &&
                               CanonicalizePreviewKey(state.meshFilePath) != CanonicalizePreviewKey(meshFilePath);
    if (sourceChanged || dependencyRevision != state.lastDependencyRevision ||
        (fileMtimeHint != 0 && fileMtimeHint != state.lastFileMtime)) {
        if (sourceChanged)
            state.meshFilePath = meshFilePath;
        state.lastFileMtime = fileMtimeHint;
        state.lastDependencyRevision = dependencyRevision;
        ++state.generation;
    }

    // ── Already up-to-date? ─────────────────────────────────────
    const auto textureId = LiveImGuiTextureId(m_renderer.get(), state.textureName);
    if (state.readyGeneration == state.generation && textureId != 0)
        return textureId;
    if (state.failedGeneration == state.generation)
        return textureId;
    if (state.readyGeneration == state.generation)
        state.readyGeneration = 0;

    // ── Schedule render if not already in flight ────────────────
    ScheduleDirtyMeshPreview(key, state);

    // Stale-return: keep showing old preview while new one renders (no flicker).
    return textureId;
}

void Infernux::ScheduleDirtyMeshPreview(const std::string &key, MeshPreviewState &state)
{
    if (state.pendingGeneration || state.renderGeneration == state.generation ||
        state.readyGeneration >= state.generation || state.failedGeneration == state.generation)
        return;
    m_meshPreviewRequestQueue.push_back({key, state.meshFilePath, state.generation});
    state.pendingGeneration = state.generation;
    m_hasPreviewPumpWork.store(true, std::memory_order_release);
    if (m_renderer)
        m_renderer->RequestFullSpeedFrame();
}

bool Infernux::PumpTimelineCubePreviewIfDirty()
{
    if (!m_renderer || !m_timelineCubeDirty)
        return false;
    if (m_pendingCubePreviewHash == m_lastCubePreviewHash) {
        m_timelineCubeDirty = false;
        return true;
    }
    // Keep the latest request dirty while the GPU preview submission is still
    // in flight. The pending fields are overwritten by newer UI samples, so
    // completion naturally renders the newest state without building a queue.
    bool progressed = false;
    const bool submitted = ExecuteTimelineCubePreviewRender(
        m_pendingCubePx, m_pendingCubePy, m_pendingCubePz, m_pendingCubeRx, m_pendingCubeRy, m_pendingCubeRz,
        m_pendingCubeSx, m_pendingCubeSy, m_pendingCubeSz, m_pendingCubeCamYaw, m_pendingCubeCamPitch,
        m_pendingCubeCamDist, m_pendingCubeSize, m_pendingCubePreviewHash, progressed);
    m_timelineCubeDirty = !submitted || m_pendingCubePreviewHash != m_lastCubePreviewHash;
    return progressed;
}

uint64_t Infernux::RenderModelAnimationPreview(const std::shared_ptr<InxMesh> &mesh, const std::string &take,
                                               float seconds, int size, uint64_t dependencyRevision)
{
    return m_renderer ? m_renderer->RenderModelAnimationPreview(mesh, take, seconds, size, dependencyRevision) : 0;
}

bool Infernux::ExecuteTimelineCubePreviewRender(float px, float py, float pz, float rx, float ry, float rz, float sx,
                                                float sy, float sz, float camYaw, float camPitch, float camDistance,
                                                int size, uint64_t hash, bool &progressed)
{
    if (!m_renderer || size <= 0)
        return false;

    // ── Build the transformed cube + a fullscreen grid quad as one mesh ──
    if (!m_cubePreviewCandidate) {
        // Freeze a complete request while its geometry uploads. New UI samples
        // cannot relabel or restart this accepted continuation.
        auto makeVert = [](const glm::vec3 &p, const glm::vec3 &n) {
            Vertex v{};
            v.pos = p;
            v.normal = n;
            v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            v.color = glm::vec3(1.0f);
            v.texCoord = glm::vec2(0.0f);
            return v;
        };

        // Persistent dedicated preview materials (built once → cached pipeline reused).
        auto &registry = AssetRegistry::Instance();
        if (!m_cubePreviewCubeMat) {
            auto defaultLit = registry.GetBuiltinMaterial("DefaultLit");
            if (!defaultLit)
                return false;
            m_cubePreviewCubeMat = defaultLit->Clone();
            if (!m_cubePreviewCubeMat)
                return false;
            m_cubePreviewCubeMat->SetColor("base_color", glm::vec4(0.62f, 0.63f, 0.66f, 1.0f));
            m_cubePreviewCubeMat->SetFloat("metallic", 0.0f);
            m_cubePreviewCubeMat->SetFloat("smoothness", 0.35f);
        }
        if (!m_cubePreviewFloorMat) {
            auto floorSrc = registry.GetBuiltinMaterial("DefaultLit");
            if (floorSrc) {
                m_cubePreviewFloorMat = floorSrc->Clone();
                if (m_cubePreviewFloorMat)
                    m_cubePreviewFloorMat->SetColor("base_color", glm::vec4(0.20f, 0.21f, 0.23f, 1.0f));
            }
        }
        auto cubeMat = m_cubePreviewCubeMat;
        auto floorMat = m_cubePreviewFloorMat;

        std::vector<Vertex> verts;
        std::vector<uint32_t> indices;
        std::vector<SubMesh> subs;
        std::vector<std::shared_ptr<InxMaterial>> materials;

        // Cube (slot 0, drawn first / opaque): unit cube with the transform baked in.
        {
            const glm::quat q = glm::quat(glm::vec3(glm::radians(rx), glm::radians(ry), glm::radians(rz)));
            const glm::mat3 R = glm::mat3_cast(q);
            const glm::vec3 scl(sx, sy, sz);
            const glm::vec3 off(px, 0.5f * sy + py, pz);

            struct Face
            {
                glm::vec3 n;
                glm::vec3 p[4];
            };
            const Face faces[6] = {
                {{0, 0, 1}, {{-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}}},
                {{0, 0, -1}, {{0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}}},
                {{1, 0, 0}, {{0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}}},
                {{-1, 0, 0}, {{-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}}},
                {{0, 1, 0}, {{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}}},
                {{0, -1, 0}, {{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}}},
            };
            const uint32_t cubeIdxStart = static_cast<uint32_t>(indices.size());
            for (const auto &f : faces) {
                const glm::vec3 nW = glm::normalize(R * f.n);
                const uint32_t base = static_cast<uint32_t>(verts.size());
                for (int k = 0; k < 4; ++k)
                    verts.push_back(makeVert(R * (f.p[k] * scl) + off, nW));
                const uint32_t fi[6] = {base, base + 1, base + 2, base, base + 2, base + 3};
                for (uint32_t i : fi)
                    indices.push_back(i);
            }
            SubMesh sm;
            sm.indexStart = cubeIdxStart;
            sm.indexCount = static_cast<uint32_t>(indices.size()) - cubeIdxStart;
            sm.vertexStart = 0;
            sm.vertexCount = static_cast<uint32_t>(verts.size());
            sm.materialSlot = 0;
            subs.push_back(sm);
            materials.push_back(cubeMat);
        }

        // Simple floor plane (DefaultLit) — avoids the fullscreen GridMaterial pass
        // which is far too heavy for a per-frame interactive preview.
        if (floorMat) {
            if (!m_cubePreviewFloorBuilt) {
                const glm::vec3 n(0.0f, 1.0f, 0.0f);
                const float half = 12.0f;
                m_cubePreviewFloorVerts.clear();
                m_cubePreviewFloorIndices.clear();
                m_cubePreviewFloorVerts.push_back(makeVert({-half, 0.0f, -half}, n));
                m_cubePreviewFloorVerts.push_back(makeVert({half, 0.0f, -half}, n));
                m_cubePreviewFloorVerts.push_back(makeVert({half, 0.0f, half}, n));
                m_cubePreviewFloorVerts.push_back(makeVert({-half, 0.0f, half}, n));
                m_cubePreviewFloorIndices = {0, 1, 2, 0, 2, 3};
                m_cubePreviewFloorBuilt = true;
            }
            const uint32_t fBase = static_cast<uint32_t>(verts.size());
            const uint32_t fIdxStart = static_cast<uint32_t>(indices.size());
            verts.insert(verts.end(), m_cubePreviewFloorVerts.begin(), m_cubePreviewFloorVerts.end());
            for (uint32_t fi : m_cubePreviewFloorIndices)
                indices.push_back(fi + fBase);
            SubMesh sm;
            sm.indexStart = fIdxStart;
            sm.indexCount = static_cast<uint32_t>(m_cubePreviewFloorIndices.size());
            sm.vertexStart = fBase;
            sm.vertexCount = static_cast<uint32_t>(m_cubePreviewFloorVerts.size());
            sm.materialSlot = 1;
            subs.push_back(sm);
            materials.push_back(floorMat);
        }

        auto candidate = std::make_unique<TimelineCubePreviewCandidate>();
        candidate->mesh.SetData(std::move(verts), std::move(indices), std::move(subs));

        const glm::vec3 target(0.0f, 0.35f, 0.0f);
        const float dist = std::max(1.5f, camDistance);
        const float cp = glm::cos(camPitch);
        const glm::vec3 dir(cp * glm::sin(camYaw), glm::sin(camPitch), cp * glm::cos(camYaw));
        const glm::vec3 camPos = target + dir * dist;
        glm::mat4 view = glm::lookAt(camPos, target, glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(35.0f), 1.0f, 100.0f, 0.05f);
        proj[1][1] *= -1.0f;

        candidate->materials = std::move(materials);
        candidate->view = view;
        candidate->projection = proj;
        candidate->camera = camPos;
        candidate->size = size;
        candidate->hash = hash;
        m_cubePreviewCandidate = std::move(candidate);
        progressed = true;
    }
    auto &candidate = *m_cubePreviewCandidate;
    const uint64_t texId = m_renderer->RenderMeshPreviewGPUImGuiCamera(
        candidate.mesh, candidate.materials, candidate.size, candidate.view, candidate.projection, candidate.camera,
        candidate.geometry, progressed, false);
    if (texId != 0) {
        m_cubePreviewTexId = texId;
        m_lastCubePreviewHash = candidate.hash;
        m_cubePreviewCandidate.reset();
    }
    return texId != 0;
}

uint64_t Infernux::RenderTimelineCubePreview(float px, float py, float pz, float rx, float ry, float rz, float sx,
                                             float sy, float sz, float camYaw, float camPitch, float camDistance,
                                             int size)
{
    if (!m_renderer || size <= 0)
        return m_cubePreviewTexId;

    auto qi = [](float f) -> long long { return static_cast<long long>(f * 1000.0f + (f >= 0.0f ? 0.5f : -0.5f)); };
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&hash](long long v) { hash = (hash ^ static_cast<uint64_t>(v)) * 1099511628211ull; };
    mix(qi(px));
    mix(qi(py));
    mix(qi(pz));
    mix(qi(rx));
    mix(qi(ry));
    mix(qi(rz));
    mix(qi(sx));
    mix(qi(sy));
    mix(qi(sz));
    mix(qi(camYaw));
    mix(qi(camPitch));
    mix(qi(camDistance));
    mix(size);

    // The shared GPUMeshPreview recreates its display target on size/format
    // changes (e.g. another preview consumer requested a different size), which
    // invalidates every previously returned descriptor id. Never hand a stale
    // id back to the UI — it would be bound as a freed VkDescriptorSet.
    if (m_cubePreviewTexId != 0 && m_cubePreviewTexId != m_renderer->GetMeshPreviewDisplayTextureId()) {
        m_cubePreviewTexId = 0;
        m_lastCubePreviewHash = 0;
    }

    m_pendingCubePx = px;
    m_pendingCubePy = py;
    m_pendingCubePz = pz;
    m_pendingCubeRx = rx;
    m_pendingCubeRy = ry;
    m_pendingCubeRz = rz;
    m_pendingCubeSx = sx;
    m_pendingCubeSy = sy;
    m_pendingCubeSz = sz;
    m_pendingCubeCamYaw = camYaw;
    m_pendingCubeCamPitch = camPitch;
    m_pendingCubeCamDist = camDistance;
    m_pendingCubeSize = size;
    m_pendingCubePreviewHash = hash;
    m_timelineCubeDirty = m_cubePreviewTexId == 0 || hash != m_lastCubePreviewHash;
    if (!m_timelineCubeDirty)
        m_cubePreviewCandidate.reset();

    return m_cubePreviewTexId;
}

// ----------------------------------
// Renderer initialization
// ----------------------------------

void Infernux::InitRenderer(int width, int height, const std::string &projectPath,
                            const std::string &builtinResourcePath, std::function<void()> prepareSettings)
{
    if (m_runtimeMode != RuntimeMode::Graphical) {
        throw std::logic_error("init_renderer is unavailable in headless mode");
    }
    if (!CheckEngineValid("initialize renderer")) {
        throw std::logic_error("Renderer is not available");
    }
    if (m_isInitialized) {
        throw std::logic_error("Engine is already initialized");
    }

    AcquireRuntimeServices();
    if (prepareSettings)
        prepareSettings();
    StartRuntimeServices();
    CreateRenderer();

    using StartupClock = std::chrono::steady_clock;
    m_startupPhaseTimingsMs.clear();
    const auto startupBegin = StartupClock::now();
    auto startupPhase = [this](const char *name, const StartupClock::time_point begin) {
        m_startupPhaseTimingsMs[name] = std::chrono::duration<double, std::milli>(StartupClock::now() - begin).count();
    };

    // Open the project log before creating the native window/swapchain so
    // startup geometry and initialization failures are not lost on Hub launch.
    // Debug / RelWithDebInfo: truncate on startup and write through.
    // Release: retain only the last 100 lines and dump them on exit.
#if INFERNUX_FILE_LOGGING
    {
        auto logsDir = ToFsPath(JoinPath({projectPath, "Logs"}));
        std::filesystem::create_directories(logsDir);
        auto logFile = logsDir / "engine.log";
#if INFERNUX_DEFERRED_FILE_LOGGING
        INXLOG_SET_DEFERRED_FILE(FromFsPath(logFile), 100);
#else
        INXLOG_SET_FILE(FromFsPath(logFile));
#endif
    }
#endif

    auto phaseBegin = StartupClock::now();
    m_renderer->Init(width, height, m_metadata);
    m_textureWorkSource = m_renderer->RegisterFrameWork(m_frameWork);
    startupPhase("renderer_init", phaseBegin);
    if (!m_renderer->PumpStartupEvents())
        throw std::runtime_error("Startup cancelled");

    // Wire SceneManager to renderer so Play()/Stop() directly bypass idle
    // sleep without relying on the Python callback chain timing.
    {
        auto *renderer = m_renderer.get();
        SceneManager::Instance().SetPlayStateChangedCallback([renderer](bool playing) {
            if (renderer)
                renderer->SetPlayModeRendering(playing);
        });
    }

    INXLOG_DEBUG("Load shaders.");
    std::string defaultShaderPath = JoinPath({builtinResourcePath, "shaders"});
    std::string assetsPath = JoinPath({projectPath, "Assets"});
    if (m_assetDatabase) {
        // Register the built-in shader search path for ShaderInfo imports.
        InxShaderLoader::AddShaderSearchPath(defaultShaderPath);

        phaseBegin = StartupClock::now();
        m_assetDatabase->Initialize(projectPath);

        // ── Transfer AssetDatabase ownership to AssetRegistry ──────
        auto &registry = AssetRegistry::Instance();
        registry.Initialize(std::move(m_assetDatabase));

        RegisterAssetLoaders();

        // Register the builtin resource directory as an extra scan root
        // so that Library/Resources assets (materials, etc.) get GUIDs.
        if (!builtinResourcePath.empty())
            registry.GetAssetDatabase()->AddReadOnlyScanRoot(builtinResourcePath);
        // Package Runtime/Editor sources are first-class project assets. Their
        // paths may move, while serialized references and Player staging are
        // always resolved through the GUID catalog.
        registry.GetAssetDatabase()->AddScanRoot(JoinPath({projectPath, "Packages"}));

        const char *playerModeFlag = std::getenv("_INFERNUX_PLAYER_MODE");
        const bool playerMode = playerModeFlag != nullptr && playerModeFlag[0] == '1' && playerModeFlag[1] == '\0';
        InxShaderLoader::SetProjectShaderSearchPaths(
            playerMode ? std::vector<std::string>{JoinPath({projectPath, "Library", "Artifacts"})}
                       : std::vector<std::string>{assetsPath, JoinPath({projectPath, "Packages"})});
        const std::string runtimeAssetCatalog = JoinPath({projectPath, "Library", "RuntimeAssetRecords.json"});
        std::error_code runtimeCatalogError;
        const bool hasRuntimeCatalog =
            std::filesystem::is_regular_file(ToFsPath(runtimeAssetCatalog), runtimeCatalogError) &&
            !runtimeCatalogError;
        if (playerMode && hasRuntimeCatalog) {
            // A cooked Player already has an immutable catalog. Walking Assets
            // and the engine resource tree here re-imports every builtin
            // texture/icon on launch and is the dominant enter-game stall.
            // Builtin and project shaders resolve on demand from their catalog entries.
            registry.GetAssetDatabase()->InstallRuntimeAssetCatalog(runtimeAssetCatalog, true);
        } else {
            const bool restoredCachedCatalog = registry.GetAssetDatabase()->RestoreCachedCatalog();
            if (restoredCachedCatalog) {
                // The cached catalog is sufficient for scene/resource lookup.
                // Validate source fingerprints and import changed files on the
                // JobSystem after the editor becomes interactive.
                registry.GetAssetDatabase()->BeginRefresh();
            } else {
                registry.GetAssetDatabase()->Refresh();
            }
            if (hasRuntimeCatalog)
                registry.GetAssetDatabase()->InstallRuntimeAssetCatalog(runtimeAssetCatalog);
        }
        startupPhase("asset_catalog", phaseBegin);
        if (!m_renderer->PumpStartupEvents())
            throw std::runtime_error("Startup cancelled");

        RegisterAssetCallbacks();

        // ── Load and register shaders via AssetRegistry ─────────────
        phaseBegin = StartupClock::now();
        InitializeBootstrapShaders();
        if (!m_renderer->PumpStartupEvents())
            throw std::runtime_error("Startup cancelled");

        // The procedural sky is renderer-owned: no authored material GUID
        // creates a dependency edge that can prewarm it for a Player. Its two
        // Standalone main() stages are compiled independently, so publish both
        // before the material system creates the builtin sky material.
        static constexpr std::array<std::pair<std::string_view, std::string_view>, 2> proceduralSkyShaderStages{{
            {"Skybox Procedural", "vertex"},
            {"Skybox Procedural", "fragment"},
        }};
        for (const auto &[shaderId, shaderType] : proceduralSkyShaderStages) {
            if (!EnsureShaderLoaded(std::string(shaderId), std::string(shaderType))) {
                INXLOG_ERROR("Required procedural sky shader is unavailable: '", shaderId, "' (", shaderType, ")");
            }
        }
        if (!playerMode) {
            // Editor overlays are present from the first Scene frame. Keep the
            // rest of the shader catalog lazy, but publish this small mandatory
            // set before their built-in materials can create fallback pipelines.
            // A late publication cannot repair a cached fallback descriptor
            // generation without an explicit material invalidation.
            static constexpr std::array<std::pair<std::string_view, std::string_view>, 10> editorOverlayShaderStages{{
                {"Gizmo", "vertex"},
                {"Gizmo", "fragment"},
                {"Grid", "vertex"},
                {"Grid", "fragment"},
                {"Gizmo Icon", "vertex"},
                {"Gizmo Icon", "fragment"},
                {"Outline Mask", "vertex"},
                {"Outline Mask", "fragment"},
                {"Outline Composite", "vertex"},
                {"Outline Composite", "fragment"},
            }};
            for (const auto &[shaderId, shaderType] : editorOverlayShaderStages) {
                if (!EnsureShaderLoaded(std::string(shaderId), std::string(shaderType))) {
                    INXLOG_ERROR("Editor overlay shader is unavailable: '", shaderId, "' (", shaderType, ")");
                }
            }
        }
        startupPhase("startup_shaders", phaseBegin);
        if (!m_renderer->PumpStartupEvents())
            throw std::runtime_error("Startup cancelled");
    }

    INXLOG_DEBUG("Prepare pipeline.");
    phaseBegin = StartupClock::now();
    m_renderer->PreparePipeline();
    startupPhase("prepare_pipeline", phaseBegin);
    // Material programs require the shared descriptor layouts and the device
    // program cache established by PreparePipeline. A resident source compiler
    // can prepare code earlier; GPU publication belongs after these owners.
    if (!projectPath.empty()) {
        phaseBegin = StartupClock::now();
        const auto defaultMaterial = InxMaterial::CreateDefaultLit();
        const auto defaultProgram = EnsureLinkedShaderProgramArtifact(defaultMaterial);
        if (!defaultProgram.usesLinkedArtifact || !defaultProgram.success)
            throw std::runtime_error("Default material program is unavailable: " + defaultProgram.error);
        startupPhase("default_material_program", phaseBegin);
    }
    if (!m_renderer->PumpStartupEvents())
        throw std::runtime_error("Startup cancelled");

    // Editor layout is project-local machine state. Keep this native path in
    // exact agreement with engine.user_data.get_project_editor_layout_root;
    // Player processes do not own or persist editor docking state.
    phaseBegin = StartupClock::now();
    const char *playerModeFlag = std::getenv("_INFERNUX_PLAYER_MODE");
    const bool playerMode = playerModeFlag != nullptr && playerModeFlag[0] == '1' && playerModeFlag[1] == '\0';
    if (!playerMode) {
        const std::filesystem::path layoutDir = ToFsPath(JoinPath({projectPath, "Cache", "Editor", "Layout"}));
        std::filesystem::create_directories(layoutDir);
        m_imguiIniPath = layoutDir / "imgui.ini";
        m_imguiLayoutMetadataPath = layoutDir / "imgui-layout.json";
    }
    // Disable ImGui auto-save (it uses fopen which can't handle Unicode
    // paths on Windows). We manually load/save with std::fstream instead.
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    LoadImGuiLayout();
    startupPhase("layout", phaseBegin);

    // Initialize physics world (Jolt)
    phaseBegin = StartupClock::now();
    PhysicsWorld::Instance().Initialize();
    startupPhase("physics", phaseBegin);

    // Initialize audio engine (SDL3 audio)
    phaseBegin = StartupClock::now();
    if (!AudioEngine::Instance().Initialize()) {
        INXLOG_WARN("Audio engine failed to initialize. Audio features will be unavailable.");
    }
    startupPhase("audio", phaseBegin);
    startupPhase("total", startupBegin);
    m_frameWork.Reopen();
    m_isInitialized = true;
}

void Infernux::InitHeadless(const std::string &projectPath, const std::string &builtinResourcePath,
                            std::function<void()> prepareSettings, bool gpu)
{
    if (m_runtimeMode != RuntimeMode::Headless) {
        throw std::logic_error("init_headless requires RuntimeMode::Headless");
    }
    if (!CheckEngineValid("initialize headless runtime")) {
        throw std::logic_error("Engine is not available");
    }
    if (m_isInitialized) {
        throw std::logic_error("Engine is already initialized");
    }

    AcquireRuntimeServices();
    if (prepareSettings)
        prepareSettings();
    StartRuntimeServices();

    // Explicit cold preparation, before any runtime budget or first scene tick.
    // CPU-only startup never creates a Vulkan instance, device or queue.
    if (gpu) {
        vk::DeviceConfig config;
        config.presentation = false;
#if INFERNUX_VULKAN_VALIDATION_LAYERS
        config.enableValidationLayers = true;
#endif
        m_gpuServices = std::make_unique<vk::VulkanBackendContext>();
        if (!m_gpuServices->InitializeInstance(config) || !m_gpuServices->InitializeDevice(VK_NULL_HANDLE, 0))
            throw std::runtime_error("Failed to initialize headless Vulkan compute services");
    }

#if INFERNUX_FILE_LOGGING
    {
        auto logsDir = ToFsPath(JoinPath({projectPath, "Logs"}));
        std::filesystem::create_directories(logsDir);
        auto logFile = logsDir / "engine.log";
#if INFERNUX_DEFERRED_FILE_LOGGING
        INXLOG_SET_DEFERRED_FILE(FromFsPath(logFile), 100);
#else
        INXLOG_SET_FILE(FromFsPath(logFile));
#endif
    }
#endif

    m_assetDatabase->Initialize(projectPath);
    auto &registry = AssetRegistry::Instance();
    registry.Initialize(std::move(m_assetDatabase));
    RegisterAssetLoaders();
    if (!builtinResourcePath.empty()) {
        InxShaderLoader::AddShaderSearchPath(JoinPath({builtinResourcePath, "shaders"}));
        registry.GetAssetDatabase()->AddReadOnlyScanRoot(builtinResourcePath);
    }
    registry.GetAssetDatabase()->AddScanRoot(JoinPath({projectPath, "Packages"}));

    // Asset catalog resolution must match the graphical runtime so a cooked
    // project simulates against the same GUID catalog in both modes. Headless
    // keeps the synchronous full Refresh in the non-player path because
    // deterministic batch runs must not race a background catalog validation.
    {
        const char *playerModeFlag = std::getenv("_INFERNUX_PLAYER_MODE");
        const bool playerMode = playerModeFlag != nullptr && playerModeFlag[0] == '1' && playerModeFlag[1] == '\0';
        InxShaderLoader::SetProjectShaderSearchPaths(
            playerMode
                ? std::vector<std::string>{JoinPath({projectPath, "Library", "Artifacts"})}
                : std::vector<std::string>{JoinPath({projectPath, "Assets"}), JoinPath({projectPath, "Packages"})});
        const std::string runtimeAssetCatalog = JoinPath({projectPath, "Library", "RuntimeAssetRecords.json"});
        std::error_code runtimeCatalogError;
        const bool hasRuntimeCatalog =
            std::filesystem::is_regular_file(ToFsPath(runtimeAssetCatalog), runtimeCatalogError) &&
            !runtimeCatalogError;
        if (playerMode && hasRuntimeCatalog) {
            registry.GetAssetDatabase()->InstallRuntimeAssetCatalog(runtimeAssetCatalog, true);
        } else {
            registry.GetAssetDatabase()->Refresh();
            if (hasRuntimeCatalog)
                registry.GetAssetDatabase()->InstallRuntimeAssetCatalog(runtimeAssetCatalog);
        }
    }

    RegisterAssetCallbacks();

    PhysicsWorld::Instance().Initialize();
    if (SceneManager::Instance().GetActiveScene() == nullptr) {
        SceneManager::Instance().CreateScene("Headless Scene");
    }

    m_frameWork.Reopen();
    m_isInitialized = true;
    INXLOG_INFO("Headless runtime initialized without renderer, window, GUI, or audio device.");
}

// ----------------------------------
// Editor Gizmos
// ----------------------------------

void Infernux::SetSelectionOutline(uint64_t objectId)
{
    if (m_isCleanedUp || !m_renderer) {
        return;
    }
    SetSelectionOutlines(objectId == 0 ? std::vector<uint64_t>{} : std::vector<uint64_t>{objectId});
}

void Infernux::ClearSelectionOutline()
{
    if (m_isCleanedUp || !m_renderer) {
        return;
    }
    m_selectedObjectId = 0;
    m_renderer->SetSelectionState(0, {});
}

void Infernux::SetSelectionOutlines(const std::vector<uint64_t> &objectIds)
{
    if (m_isCleanedUp || !m_renderer) {
        return;
    }

    if (objectIds.empty()) {
        m_selectedObjectId = 0;
        m_renderer->SetSelectionState(0, {});
        return;
    }

    std::vector<uint64_t> expandedIds = ExpandSelectionOutlineIds(SceneManager::Instance(), objectIds);
    const uint64_t primaryObjectId = objectIds.empty() ? 0 : objectIds.back();
    m_selectedObjectId = primaryObjectId;
    m_renderer->SetSelectionState(primaryObjectId, expandedIds);
}

// ----------------------------------
// Material Pipeline
// ----------------------------------

void Infernux::RegisterShaderToRenderer(const ShaderAsset &asset)
{
    if (!m_renderer || !asset.HasVariant(ShaderCompileTarget::Forward))
        return;

    const char *stage = asset.shaderType == "vertex" ? "vertex" : "fragment";
    auto runtimeId = [&](ShaderCompileTarget target) {
        switch (target) {
        case ShaderCompileTarget::Forward:
            return asset.shaderId;
        case ShaderCompileTarget::GBuffer:
            return asset.shaderId + "/gbuffer";
        case ShaderCompileTarget::Shadow:
            return asset.shaderId + "/shadow";
        case ShaderCompileTarget::Depth:
            return asset.shaderId + "/depth";
        case ShaderCompileTarget::Picking:
            return asset.shaderId + "/picking";
        case ShaderCompileTarget::Motion:
            return asset.shaderId + "/motion";
        case ShaderCompileTarget::ForwardPlus:
            return asset.shaderId + "/forward_plus";
        case ShaderCompileTarget::Normal:
            return asset.shaderId + "/normal";
        case ShaderCompileTarget::BaseColor:
            return asset.shaderId + "/base_color";
        case ShaderCompileTarget::Count:
            break;
        }
        return std::string();
    };

    for (const auto &variant : asset.variants) {
        if (!variant.IsValid())
            continue;
        const std::string id = runtimeId(variant.target);
        if (id.empty())
            continue;
        m_renderer->LoadShader(id.c_str(), variant.spirv, stage);
        if (variant.target != ShaderCompileTarget::Forward)
            INXLOG_INFO("Registered ", ShaderCompileTargetName(variant.target), " ", stage, " variant '", id, "'");
    }

    // Render-state metadata (fragment shaders only)
    if (asset.shaderType == "fragment") {
        const auto &rm = asset.renderMeta;
        m_renderer->StoreShaderRenderMeta(asset.shaderId, rm.cullMode, rm.depthWrite, rm.depthTest, rm.blend, rm.queue,
                                          rm.passTag, rm.stencil, rm.alphaClip);
    }
}

void Infernux::InitializeBootstrapShaders()
{
    // Required presentation shaders are independent of project author assets.
    // The initial scene's material programs resolve lazily from the catalog.
    static constexpr std::string_view vertexSource = R"(
struct CameraData {
    column_major float4x4 model;
    column_major float4x4 view;
    column_major float4x4 projection;
    column_major float4x4 previous_view_projection;
    column_major float4x4 inverse_view_projection;
    float4 projection_parameters;
    float4 depth_parameters;
};
struct InstanceData { column_major float4x4 model; };
[[vk::binding(5, 1)]] ConstantBuffer<CameraData> camera;
[[vk::binding(1, 2)]] StructuredBuffer<InstanceData> instances;

[shader("vertex")]
float4 main(float3 position : POSITION, uint instance : SV_VulkanInstanceID) : SV_Position
{
    float4 clip = mul(camera.projection, mul(camera.view, mul(instances[instance].model, float4(position, 1.0))));
    // This native camera buffer still stores the Vulkan viewport projection.
    // Convert it to the compiler's canonical clip coordinates at this boundary.
    clip.y = -clip.y;
    return clip;
}
)";
    static constexpr std::string_view fragmentSource = R"(
[shader("fragment")]
float4 main() : SV_Target0 { return float4(1.0); }
)";
    auto &compiler = SlangCompiler::Get();
    const auto vertex =
        compiler.Compile({"inx_bootstrap_vertex", "Infernux/BootstrapVertex.slang", std::string(vertexSource)}, {},
                         {ShaderCodeTarget::SpirV});
    const auto fragment =
        compiler.Compile({"inx_bootstrap_fragment", "Infernux/BootstrapFragment.slang", std::string(fragmentSource)},
                         {}, {ShaderCodeTarget::SpirV});
    m_renderer->LoadShader("default", vertex.targets[0].entries[0].code, "vertex");
    m_renderer->LoadShader("default", fragment.targets[0].entries[0].code, "fragment");
}

bool Infernux::EnsureShaderLoaded(const std::string &shaderId, const std::string &shaderType)
{
    if (m_renderer->HasShader(shaderId, shaderType)) {
        return true;
    }

    INXLOG_DEBUG("Infernux::EnsureShaderLoaded: shader '", shaderId, "' (", shaderType,
                 ") not loaded, trying to find and load it");

    auto *adb = GetAssetDatabase();
    if (!adb) {
        INXLOG_WARN("Infernux::EnsureShaderLoaded: no AssetDatabase available");
        return false;
    }
    std::string shaderPath = adb->FindShaderPathById(shaderId, shaderType);
    if (shaderPath.empty()) {
        INXLOG_WARN("Infernux::EnsureShaderLoaded: could not find shader file for '", shaderId, "' (", shaderType, ")");
        return false;
    }

    INXLOG_DEBUG("Infernux::EnsureShaderLoaded: found shader at '", shaderPath, "', loading...");

    // This path is runtime publication, not an editor hot-reload operation.
    // Loading through AssetRegistry keeps stage identity in imported metadata
    // for both loose Editor shaders and packed Player GLSL.
    auto &registry = AssetRegistry::Instance();
    const std::string guid = adb->GetGuidFromPath(shaderPath);
    if (guid.empty()) {
        INXLOG_ERROR("Infernux::EnsureShaderLoaded: shader is not imported: ", shaderPath);
        return false;
    }
    auto shaderAsset = registry.LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
    if (!shaderAsset || shaderAsset->shaderId != shaderId || shaderAsset->shaderType != shaderType) {
        INXLOG_ERROR("Infernux::EnsureShaderLoaded: imported shader identity/stage mismatch for '", shaderPath,
                     "' (expected ", shaderId, "/", shaderType, ")");
        return false;
    }
    RegisterShaderToRenderer(*shaderAsset);
    return m_renderer->HasShader(shaderId, shaderType);
}

Infernux::LinkedShaderProgramPreparation
Infernux::EnsureLinkedShaderProgramArtifact(const std::shared_ptr<InxMaterial> &material, bool requireCurrentSource)
{
    if (!material)
        return {};

    auto *adb = GetAssetDatabase();
    if (!adb)
        return {};

    if (material->GetGuid().empty())
        AssetRegistry::Instance().RegisterRuntimeMaterial(material);

    auto resolvePath = [&](const ShaderAssetReference &reference, const char *stage) {
        if (!reference.guid.empty()) {
            // A missing GUID remains a missing reference. Never fall through
            // to path_hint or shader_id and silently bind a different asset.
            return adb->GetPathFromGuid(reference.guid);
        }

        // path_hint is display/diagnostic metadata only. A GUID-less, pathless
        // shader_id is the explicit symbolic built-in shader case; it is not
        // an AssetDatabase dependency and may use the program registry.
        if (!reference.pathHint.empty())
            return std::string{};
        return adb->FindShaderPathById(reference.shaderId, stage);
    };

    const std::string fragmentPath = resolvePath(material->GetFragShaderReference(), "fragment");
    const auto stages = material->GetShaderStages();
    const std::string vertexPath = resolvePath(material->GetVertShaderReference(), "vertex");
    return EnsureLinkedShaderProgramArtifact(stages, vertexPath, fragmentPath, requireCurrentSource);
}

Infernux::LinkedShaderProgramPreparation Infernux::EnsureLinkedShaderProgramArtifact(const ShaderStagePair &stages,
                                                                                     const std::string &vertexPath,
                                                                                     const std::string &fragmentPath,
                                                                                     bool requireCurrentSource)
{
    LinkedShaderProgramPreparation result;
    if (!m_renderer || !stages.IsValid())
        return result;

    if (asset_formats::FindShaderSourceKind(fragmentPath))
        return EnsureShaderAuthorMaterial(stages, vertexPath, fragmentPath);

    if (asset_formats::FindShaderSourceKind(vertexPath)) {
        const std::string error = "Python vertex shader '" + vertexPath +
                                  "' requires a Python mesh fragment; selected fragment: " + fragmentPath;
        auto &entry = m_linkedShaderProgramCache[stages];
        if (entry.lastError != error)
            entry.failureReported = false;
        entry.lastError = error;
        return {true, false, error};
    }

    const auto cached = m_linkedShaderProgramCache.find(stages);
    const auto publishPrepared = [&]() -> LinkedShaderProgramPreparation {
        auto &entry = cached->second;
        if (!m_renderer->PublishShaderProgramArtifact(*entry.preparedArtifact))
            return {true, false, "Renderer rejected the prepared UI shader candidate"};
        entry.preparedArtifact.reset();
        return {true, true, {}};
    };
    if (cached != m_linkedShaderProgramCache.end() && !requireCurrentSource) {
        result.usesLinkedArtifact = true;
        if (cached->second.failedSourceStamp != 0) {
            result.success = false;
            result.error = cached->second.lastError;
            return result;
        }
        if (cached->second.preparedArtifact)
            return publishPrepared();
        if (cached->second.sourceStamp != 0 && cached->second.programKey.IsValid() &&
            m_renderer->HasShaderProgramArtifact(cached->second.programKey)) {
            return result;
        }
    }

    if (vertexPath.empty() || fragmentPath.empty()) {
        if (requireCurrentSource)
            return {true, false, "UI material shader GUID does not resolve to both imported stages"};
        return result;
    }

    auto *adb = GetAssetDatabase();
    if (!adb)
        return result;

    auto readSource = [&](const std::string &path, std::string &source) {
        std::vector<char> bytes;
        if (!adb->ReadFile(path, bytes) || bytes.empty())
            return false;
        if (bytes.back() == '\0')
            bytes.pop_back();
        source.assign(bytes.begin(), bytes.end());
        return true;
    };

    std::string vertexSource;
    std::string fragmentSource;
    if (!readSource(vertexPath, vertexSource) || !readSource(fragmentPath, fragmentSource)) {
        if (requireCurrentSource)
            return {true, false, "UI material shader source could not be read from its imported GUID"};
        return result;
    }

    const uint64_t sourceStamp =
        ComputeShaderProgramRevision(vertexSource, fragmentSource, ShaderCompileTarget::Forward, 0);
    if (requireCurrentSource && cached != m_linkedShaderProgramCache.end()) {
        result.usesLinkedArtifact = true;
        if (cached->second.failedSourceStamp == sourceStamp)
            return {true, false, cached->second.lastError};
        if (cached->second.sourceStamp == sourceStamp && cached->second.preparedArtifact)
            return publishPrepared();
        if (cached->second.sourceStamp == sourceStamp && cached->second.programKey.IsValid() &&
            m_renderer->HasShaderProgramArtifact(cached->second.programKey))
            return result;
    }
    // A synchronous compile now owns publication for this pair, including a
    // rejected candidate; an older scene worker must not publish afterward.
    m_linkedShaderPreparations.erase(stages);
    const auto sourceEnvironmentRevision = InxShaderLoader::GetSourceEnvironmentRevision();
    auto rememberFailure = [&](const std::string &error) {
        auto &entry = m_linkedShaderProgramCache[stages];
        entry.failedSourceStamp = sourceStamp;
        entry.lastError = error;
        entry.failureReported = false;
        entry.sourceEnvironmentRevision = sourceEnvironmentRevision;
    };

    InxShaderLoader compiler(true, false, false, false, false, true, false, false, false, false);
    const std::string vertexCompilePath = InxShaderLoader::StageQualifiedVirtualPath(vertexPath, "vertex");
    const std::string fragmentCompilePath = InxShaderLoader::StageQualifiedVirtualPath(fragmentPath, "fragment");
    const ShaderDescriptor vertexDescriptor = compiler.ParseShaderSource(vertexSource, vertexCompilePath);
    const ShaderDescriptor fragmentDescriptor = compiler.ParseShaderSource(fragmentSource, fragmentCompilePath);

    if (IsDirectStructuredStage(vertexDescriptor) || IsDirectStructuredStage(fragmentDescriptor)) {
        // Explicit main() stages use the standalone importer, not the linker.
        // Publish cold stages here too: only the default sky is preloaded at startup.
        const auto publishStage = [&](const std::string &path, const std::string &id, const char *stage) {
            if (m_renderer->HasShader(id, stage))
                return true;
            auto *database = GetAssetDatabase();
            auto &registry = AssetRegistry::Instance();
            const auto guid = database->GetGuidFromCatalogPath(path);
            const auto asset = registry.LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
            if (!asset || asset->shaderId != id || asset->shaderType != stage)
                return false;
            RegisterShaderToRenderer(*asset);
            return m_renderer->HasShader(id, stage);
        };
        if (!publishStage(vertexPath, vertexDescriptor.shaderId, "vertex") ||
            !publishStage(fragmentPath, fragmentDescriptor.shaderId, "fragment")) {
            const std::string error = "Standalone material shader stages could not be published";
            rememberFailure(error);
            return {true, false, error};
        }
        m_linkedShaderProgramCache.erase(stages);
        return {};
    }

    result.usesLinkedArtifact = true;
    auto compilation =
        compiler.CompileLinkedProgramArtifact(vertexSource, vertexCompilePath, fragmentSource, fragmentCompilePath);
    if (!compilation.IsValid()) {
        result.success = false;
        std::ostringstream diagnostics;
        for (const auto &error : compilation.errors) {
            if (diagnostics.tellp() > 0)
                diagnostics << '\n';
            diagnostics << error;
        }
        result.error = diagnostics.str();
        if (result.error.empty())
            result.error = "Linked shader program compilation failed";
        rememberFailure(result.error);
        return result;
    }

    ShaderProgramArtifact artifact = compilation.CreateRuntimeArtifact(stages);
    if (!artifact.IsValid() || artifact.key.stages != stages) {
        result.success = false;
        result.error = "Linked shader compiler produced an invalid or mismatched program artifact";
        rememberFailure(result.error);
        return result;
    }

    if (!m_renderer->PublishShaderProgramArtifact(artifact)) {
        result.success = false;
        result.error = "Renderer rejected the linked shader program artifact; last-known-good remains active";
        rememberFailure(result.error);
        return result;
    }

    if (!compilation.pendingTargets.empty()) {
        std::ostringstream pending;
        for (const auto target : compilation.pendingTargets) {
            if (pending.tellp() > 0)
                pending << ", ";
            pending << ShaderCompileTargetName(target);
        }
        INXLOG_DEBUG("Linked shader program '", artifact.key.stages.ToString(),
                     "' has planned targets awaiting backend generation: ", pending.str());
    }

    m_linkedShaderProgramCache[stages] = LinkedShaderProgramCacheEntry{sourceStamp, artifact.key, 0, {}};
    m_linkedShaderProgramCache[stages].sourceEnvironmentRevision = sourceEnvironmentRevision;
    m_linkedShaderProgramCache[stages].stageLabels = {vertexDescriptor.shaderId, fragmentDescriptor.shaderId};
    m_renderer->StoreShaderRenderMeta(
        stages.fragmentKey, fragmentDescriptor.surfaceOptions.cullMode, fragmentDescriptor.depthWrite,
        fragmentDescriptor.depthTest, fragmentDescriptor.surfaceOptions.blendMode, fragmentDescriptor.renderQueue,
        fragmentDescriptor.passTag, fragmentDescriptor.stencil, fragmentDescriptor.surfaceOptions.alphaClip);
    return result;
}

std::optional<ShaderProgramDomain>
Infernux::InspectMaterialShaderDomain(const std::shared_ptr<InxMaterial> &material) const
{
    auto *database = GetAssetDatabase();
    if (!database || !material)
        throw std::runtime_error("Material shader classification requires a material and AssetDatabase");

    unsigned int stageCount = 0;
    const auto stageFlags = [&](const ShaderAssetReference &reference, const char *stage) {
        auto guid = reference.guid;
        std::shared_ptr<const InxResourceMeta> metadata;
        if (!guid.empty())
            metadata = database->GetMetaByGuid(guid);
        else if (reference.pathHint.empty() && !reference.shaderId.empty()) {
            const auto path = database->FindShaderPathById(reference.shaderId, stage);
            guid = database->GetGuidFromCatalogPath(path);
            metadata = database->GetMetaByGuid(guid);
        }
        if (!metadata) {
            if (!reference.guid.empty())
                throw ShaderProgramSelectionError("Material shader GUID cannot be resolved: " + reference.guid);
            return 0u;
        }
        ++stageCount;
        if (metadata->HasKey("shader_source_kind")) {
            const auto asset = AssetRegistry::Instance().LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
            if (!asset)
                throw ShaderProgramSelectionError("Material author program could not be loaded: " + guid);
            if (!asset->programDomain)
                throw ShaderProgramSelectionError("Material stage requires a graphics author program: " +
                                                  asset->filePath);
            switch (*asset->programDomain) {
            case ShaderProgramDomain::Mesh:
                return 0u;
            case ShaderProgramDomain::ScreenUI:
                return 1u;
            case ShaderProgramDomain::WorldUI:
                return 2u;
            case ShaderProgramDomain::ParticleSprite:
                return 4u;
            case ShaderProgramDomain::Fullscreen:
                return 8u;
            default:
                throw std::logic_error("Graphics author program has no geometry domain");
            }
        }
        const auto capabilities = json::parse(metadata->GetDataAs<std::string>("shader_capabilities"));
        unsigned int flags = 0;
        for (std::string capability : capabilities.get<std::vector<std::string>>()) {
            std::transform(capability.begin(), capability.end(), capability.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (capability == "screenui")
                flags |= 1u;
            else if (capability == "worldui")
                flags |= 2u;
            else if (capability == "particlesprite")
                flags |= 4u;
            else if (capability == "fullscreen")
                flags |= 8u;
        }
        return flags;
    };
    const unsigned int vertexFlags = stageFlags(material->GetVertShaderReference(), "vertex");
    const unsigned int fragmentFlags = stageFlags(material->GetFragShaderReference(), "fragment");
    const unsigned int flags = vertexFlags | fragmentFlags;
    if ((flags & (flags - 1u)) != 0)
        throw ShaderProgramSelectionError("Material shader stages declare conflicting geometry domains");
    if (flags == 1u)
        return ShaderProgramDomain::ScreenUI;
    if (flags == 2u)
        return ShaderProgramDomain::WorldUI;
    if (flags == 4u)
        return ShaderProgramDomain::ParticleSprite;
    if (flags == 8u)
        return ShaderProgramDomain::Fullscreen;
    if (stageCount == 2)
        return ShaderProgramDomain::Mesh;
    return std::nullopt;
}

bool Infernux::RefreshMaterialPipeline(std::shared_ptr<InxMaterial> material)
{
    INXLOG_DEBUG("Infernux::RefreshMaterialPipeline called");
    if (!CheckEngineValid("refresh material pipeline") || !m_renderer) {
        INXLOG_ERROR("Infernux::RefreshMaterialPipeline: engine or renderer invalid");
        return false;
    }

    auto *adb = GetAssetDatabase();
    if (!adb) {
        INXLOG_ERROR("Infernux::RefreshMaterialPipeline: no AssetDatabase available");
        return false;
    }

    if (!material) {
        INXLOG_ERROR("Infernux::RefreshMaterialPipeline: material is null");
        return false;
    }

    // Get shader names from material
    const std::string &vertName = material->GetVertShaderName();
    const std::string &fragName = material->GetFragShaderName();

    const auto stages = material->GetShaderStages();
    try {
        const auto declared = InspectMaterialShaderDomain(material);
        if (declared && (*declared == ShaderProgramDomain::ScreenUI || *declared == ShaderProgramDomain::WorldUI)) {
            // Inspector/bootstrap may refresh a UI material before it is ever
            // drawn. The draw owns publication, pipeline creation and release.
            m_renderer->InvalidateUIMaterialProgram(stages);
            return true;
        }
    } catch (const std::exception &error) {
        m_renderer->InvalidateUIMaterialProgram(stages);
        INXLOG_ERROR("Infernux::RefreshMaterialPipeline: ", error.what());
        return false;
    }

    const LinkedShaderProgramPreparation linkedProgram = EnsureLinkedShaderProgramArtifact(material);
    if (linkedProgram.usesLinkedArtifact) {
        if (!linkedProgram.success)
            INXLOG_ERROR("Infernux::RefreshMaterialPipeline: ", linkedProgram.error);
        // Refresh still runs on failure so an already-published last-known-good
        // artifact can remain visible.
        return m_renderer->RefreshMaterialPipeline(material);
    }

    // Ensure shaders are loaded before refreshing pipeline
    if (!vertName.empty()) {
        EnsureShaderLoaded(vertName, "vertex");
    }
    if (!fragName.empty()) {
        EnsureShaderLoaded(fragName, "fragment");
    }

    INXLOG_DEBUG("Infernux::RefreshMaterialPipeline: calling renderer");
    return m_renderer->RefreshMaterialPipeline(material);
}

bool Infernux::IsShaderLoaded(const std::string &shaderId, const std::string &shaderType) const
{
    if (shaderType != "vertex" && shaderType != "fragment")
        throw std::invalid_argument("Shader type must be vertex or fragment");
    if (!m_renderer)
        return false;
    if (m_renderer->HasShader(shaderId, shaderType))
        return true;
    auto *database = GetAssetDatabase();
    std::string guid;
    bool assetSelector = false;
    if (database) {
        if (database->GetMetaByGuid(shaderId))
            guid = shaderId;
        else
            guid = database->GetGuidFromPath(shaderId);
        assetSelector = !guid.empty();
        if (guid.empty())
            guid = database->GetGuidFromCatalogPath(database->FindShaderPathById(shaderId, shaderType));
    }
    if (!guid.empty()) {
        // A query observes prepared assets; it never loads or compiles one.
        if (database->GetMetaByGuid(guid)->GetResourceType() != ResourceType::Shader)
            return false;
        const auto asset = AssetRegistry::Instance().GetAsset<ShaderAsset>(guid);
        if (asset && (assetSelector || asset->shaderId == shaderId) && asset->shaderType == shaderType) {
            if (asset->materialProgram && m_renderer->HasShaderProgramArtifact(asset->materialProgram->key))
                return true;
            if (m_renderer->HasShader(asset->shaderId, shaderType))
                return true;
        }
    }
    for (const auto &[stages, entry] : m_linkedShaderProgramCache) {
        const std::string &stageId = shaderType == "vertex" ? stages.vertexKey : stages.fragmentKey;
        const auto &label = entry.stageLabels[shaderType == "vertex" ? 0 : 1];
        if ((assetSelector ? stageId == guid : label == shaderId) && entry.programKey.IsValid() &&
            m_renderer->HasShaderProgramArtifact(entry.programKey))
            return true;
    }
    return false;
}

std::string Infernux::ReloadShaderDependencies(const std::string &shaderPath, const std::string &previousSourcePath)
{
    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    const auto ext = FromFsPath(ToFsPath(shaderPath).extension());
    const InxShaderLoader::CompilationGuard compileGuard;
    // Query compiler subscriptions before rebuilding declaration maps.
    // Path subscriptions handle removal/rename; declaration subscriptions
    // handle newly created imports that were missing in a rejected root.
    std::string declarationId;
    std::vector<char> bytes;
    if (!adb->GetGuidFromPath(shaderPath).empty() && adb->ReadFile(shaderPath, bytes) && !bytes.empty()) {
        if (bytes.back() == '\0')
            bytes.pop_back();
        const auto info = ParseShaderInfo(std::string(bytes.begin(), bytes.end()));
        if (!info.name.empty())
            declarationId = ext == ".shadingmodel" ? "shadingmodel/" + info.name : info.name;
    }
    auto roots = InxShaderLoader::GetDependentStageSources(shaderPath, declarationId);
    if (!previousSourcePath.empty()) {
        const auto previousRoots = InxShaderLoader::GetDependentStageSources(previousSourcePath);
        roots.insert(roots.end(), previousRoots.begin(), previousRoots.end());
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    }
    InxShaderLoader::InvalidateDirectoryCache();
    InxShaderLoader::InvalidateTemplateCache();
    return ReloadShaderSourceBatch(roots);
}

std::string Infernux::ReloadShaderSourceBatch(const std::vector<std::string> &roots)
{
    m_shaderReloadSources = roots;
    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    InxShaderLoader::SourceDependencyPublication dependencyPublication;
    struct PreparedStage
    {
        std::string guid;
        std::shared_ptr<ShaderAsset> asset;
    };
    struct PreparedProgram
    {
        ShaderProgramArtifact artifact;
        ShaderDescriptor fragment;
        uint64_t sourceStamp;
        bool ui;
        std::array<std::string, 2> stageLabels;
    };
    std::vector<PreparedStage> preparedStages;
    std::vector<PreparedProgram> preparedPrograms;
    std::vector<ShaderStagePair> retiredPairs;
    std::unordered_set<std::string> changedStageIds;
    std::vector<std::string> linkedRootGuids;
    const auto materials = registry.GetAllMaterials();
    const auto affected = [&](const ShaderStagePair &stages) {
        return changedStageIds.count(stages.vertexKey) || changedStageIds.count(stages.fragmentKey);
    };
    std::ostringstream errors;
    const auto reject = [&](const std::string &source, const std::string &error) {
        if (errors.tellp() > 0)
            errors << '\n';
        errors << PortablePathFilename(source) << ": " << error;
    };
    const auto readSource = [&](const std::string &sourcePath, std::string &source) {
        std::vector<char> content;
        if (sourcePath.empty() || !adb->ReadFile(sourcePath, content) || content.empty())
            return false;
        if (content.back() == '\0')
            content.pop_back();
        source.assign(content.begin(), content.end());
        return true;
    };
    InxShaderLoader compiler(true, false, false, false, false, true, false, false, false, false);
    try {
        for (const auto &root : roots) {
            const auto guid = adb->GetGuidFromPath(root);
            if (guid.empty())
                continue; // A retired root has no live source to republish.
            std::string source;
            if (!readSource(root, source)) {
                reject(root, "Dependent shader source cannot be read");
                continue;
            }
            const auto descriptor = compiler.ParseShaderSource(source, root);
            if (!descriptor.errors.empty() || descriptor.shaderId.empty()) {
                std::ostringstream message;
                message << "Dependent ShaderInfo is invalid";
                for (const auto &error : descriptor.errors)
                    message << '\n' << error;
                reject(root, message.str());
                continue;
            }
            changedStageIds.insert(guid);
            // Symbolic built-in references intentionally carry no asset GUID.
            changedStageIds.insert(descriptor.shaderId);
            if (IsDirectStructuredStage(descriptor)) {
                auto *loader = registry.GetLoader(ResourceType::Shader);
                if (!loader)
                    throw std::logic_error("Dependent shader publication requires the registered ShaderLoader");
                auto candidate = loader->Load(root, guid, adb).Get<ShaderAsset>();
                if (!candidate || !candidate->HasVariant(ShaderCompileTarget::Forward))
                    reject(root, InxShaderLoader::GetLastCompileError().empty()
                                     ? "Standalone dependency compilation failed"
                                     : InxShaderLoader::GetLastCompileError());
                else
                    preparedStages.push_back({guid, std::move(candidate)});
            } else
                linkedRootGuids.push_back(guid);
        }

        std::unordered_map<ShaderStagePair, std::shared_ptr<InxMaterial>, ShaderStagePairHash> owners;
        std::unordered_set<ShaderStagePair, ShaderStagePairHash> affectedPairs;
        for (const auto &material : materials) {
            if (!material || material->IsDeleted())
                continue;
            const auto stages = material->GetShaderStages();
            if (affected(stages)) {
                affectedPairs.insert(stages);
                owners.try_emplace(stages, material);
            }
        }
        for (const auto &[stages, entry] : m_linkedShaderProgramCache)
            if (affected(stages) && owners.find(stages) == owners.end())
                retiredPairs.push_back(stages);
        std::vector<ShaderStagePair> orderedPairs(affectedPairs.begin(), affectedPairs.end());
        std::sort(orderedPairs.begin(), orderedPairs.end(), [](const auto &left, const auto &right) {
            return std::tie(left.vertexKey, left.fragmentKey) < std::tie(right.vertexKey, right.fragmentKey);
        });
        for (const auto &stages : orderedPairs) {
            const auto owner = owners.find(stages);
            const auto stagePath = [&](const std::string &id, const char *stage, bool vertex) {
                if (owner != owners.end()) {
                    const auto &reference =
                        vertex ? owner->second->GetVertShaderReference() : owner->second->GetFragShaderReference();
                    if (!reference.guid.empty())
                        return adb->GetPathFromGuid(reference.guid);
                    if (!reference.pathHint.empty())
                        return std::string();
                }
                return adb->FindShaderPathById(id, stage);
            };
            const auto vertexPath = stagePath(stages.vertexKey, "vertex", true);
            const auto fragmentPath = stagePath(stages.fragmentKey, "fragment", false);
            if (asset_formats::FindShaderSourceKind(vertexPath) || asset_formats::FindShaderSourceKind(fragmentPath))
                continue;
            std::string vertexSource, fragmentSource;
            if (!readSource(vertexPath, vertexSource) || !readSource(fragmentPath, fragmentSource)) {
                reject(fragmentPath, "Both dependent material stage GUIDs must resolve to readable sources");
                continue;
            }
            const auto vertex = compiler.ParseShaderSource(vertexSource, vertexPath);
            const auto fragment = compiler.ParseShaderSource(fragmentSource, fragmentPath);
            if (IsDirectStructuredStage(vertex) || IsDirectStructuredStage(fragment))
                continue;
            auto compilation = compiler.CompileLinkedProgramArtifact(
                vertexSource, InxShaderLoader::StageQualifiedVirtualPath(vertexPath, "vertex"), fragmentSource,
                InxShaderLoader::StageQualifiedVirtualPath(fragmentPath, "fragment"));
            auto artifact = compilation.CreateRuntimeArtifact(stages);
            if (!compilation.IsValid() || !artifact.IsValid() || artifact.key.stages != stages) {
                std::ostringstream message;
                for (const auto &error : compilation.errors)
                    message << error << '\n';
                reject(fragmentPath,
                       message.str().empty() ? "Dependent linked program compilation failed" : message.str());
                continue;
            }
            preparedPrograms.push_back(
                {std::move(artifact),
                 fragment,
                 ComputeShaderProgramRevision(vertexSource, fragmentSource, ShaderCompileTarget::Forward, 0),
                 ShaderStageLinker::IsUIStagePair(vertex, fragment),
                 {vertex.shaderId, fragment.shaderId}});
        }
    } catch (const std::runtime_error &rejected) {
        return rejected.what(); // Declaration-map/source rejection, owned by this source event.
    }
    if (errors.tellp() > 0)
        return errors.str();

    // No live renderer, material or resource state changes until every
    // affected root has produced its candidate. Each pair compiles once.
    std::vector<ShaderProgramArtifact> runtimePrograms;
    for (const auto &program : preparedPrograms)
        if (!program.ui)
            runtimePrograms.push_back(program.artifact);
    if (!m_renderer->PublishShaderProgramArtifacts(runtimePrograms))
        return "Renderer rejected the dependent shader batch; running programs remain active";
    for (const auto &stages : retiredPairs) {
        const auto &entry = m_linkedShaderProgramCache.at(stages);
        if (entry.programKey.IsValid())
            m_renderer->RetireShaderProgramArtifact(entry.programKey);
        m_linkedShaderProgramCache.erase(stages);
    }
    for (const auto &guid : linkedRootGuids)
        registry.InvalidateAsset(guid);
    for (const auto &stage : preparedStages) {
        m_renderer->InvalidateShaderCache(stage.asset->shaderId, stage.asset->shaderType);
        RegisterShaderToRenderer(*stage.asset);
        registry.PublishShader(stage.guid, stage.asset);
    }
    for (auto &program : preparedPrograms) {
        const auto stages = program.artifact.key.stages;
        LinkedShaderProgramCacheEntry entry{program.sourceStamp, program.artifact.key, 0, {}};
        entry.sourceEnvironmentRevision = InxShaderLoader::GetSourceEnvironmentRevision();
        entry.stageLabels = std::move(program.stageLabels);
        if (program.ui) {
            m_renderer->InvalidateUIMaterialProgram(stages);
            entry.preparedArtifact = std::make_shared<const ShaderProgramArtifact>(std::move(program.artifact));
        }
        m_linkedShaderProgramCache[stages] = std::move(entry);
        const auto &fragment = program.fragment;
        m_renderer->StoreShaderRenderMeta(stages.fragmentKey, fragment.surfaceOptions.cullMode, fragment.depthWrite,
                                          fragment.depthTest, fragment.surfaceOptions.blendMode, fragment.renderQueue,
                                          fragment.passTag, fragment.stencil, fragment.surfaceOptions.alphaClip);
    }
    dependencyPublication.Commit();
    for (const auto &material : materials) {
        if (!material || material->IsDeleted())
            continue;
        const auto stages = material->GetShaderStages();
        if (!affected(stages))
            continue;
        const auto cached = m_linkedShaderProgramCache.find(stages);
        if (cached != m_linkedShaderProgramCache.end() && cached->second.preparedArtifact)
            continue; // The first actual UI draw owns GPU publication.
        m_renderer->RefreshMaterialPipeline(material);
    }
    return "";
}

std::string Infernux::ReloadShaderRuntime(const std::string &shaderPath, const std::string &previousShaderId,
                                          const std::string &previousSourcePath)
{
    m_shaderReloadSources = {shaderPath};
    m_shaderReloadDiagnostics.clear();
    if (asset_formats::FindShaderSourceKind(shaderPath)) {
        if (!CheckEngineValid("reload shader"))
            return "Engine invalid";
        return ReloadShaderAuthorSource(shaderPath);
    }
    const InxShaderLoader::SourceDiagnosticScope sourceDiagnostics;
    // INXLOG_INFO("Infernux::ReloadShaderRuntime called: ", shaderPath);
    if (!CheckEngineValid("reload shader") || !m_renderer) {
        INXLOG_ERROR("Infernux::ReloadShaderRuntime: engine or renderer invalid");
        return "Engine or renderer invalid";
    }

    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    if (!adb) {
        INXLOG_ERROR("Infernux::ReloadShaderRuntime: no AssetDatabase available");
        return "No AssetDatabase available";
    }

    std::filesystem::path path = ToFsPath(shaderPath);
    std::string ext = FromFsPath(path.extension());

    const auto previousExt = FromFsPath(ToFsPath(previousSourcePath).extension());
    if (ext == ".glsl" || ext == ".shadingmodel" || previousExt == ".glsl" || previousExt == ".shadingmodel")
        return ReloadShaderDependencies(shaderPath, previousSourcePath);

    if (ext != ".vert" && ext != ".frag") {
        INXLOG_ERROR("Infernux::ReloadShaderRuntime: unsupported shader extension: ", ext);
        return "Unsupported shader extension: " + ext;
    }

    const std::string guid = adb->GetGuidFromPath(shaderPath);

    // Invalidate shader-id map cache for this directory so shading models
    // and imports added/modified since the last compile are discovered.
    InxShaderLoader::InvalidateDirectoryCache(FromFsPath(ToFsPath(shaderPath).parent_path()));
    InxShaderLoader::InvalidateTemplateCache();

    if (guid.empty()) {
        INXLOG_ERROR("Infernux::ReloadShaderRuntime: shader is not imported: ", shaderPath);
        return "Shader asset is not imported: " + shaderPath;
    }

    std::vector<char> sourceBytes;
    if (!adb->ReadFile(shaderPath, sourceBytes) || sourceBytes.empty())
        return "Failed to read shader source: " + shaderPath;
    if (sourceBytes.back() == '\0')
        sourceBytes.pop_back();
    const std::string source(sourceBytes.begin(), sourceBytes.end());

    InxShaderLoader sourceParser(true, false, false, false, false, true, false, false, false, false);
    const ShaderDescriptor changedDescriptor = sourceParser.ParseShaderSource(source, shaderPath);
    if (!changedDescriptor.errors.empty()) {
        std::ostringstream diagnostics;
        diagnostics << "ShaderInfo validation failed:";
        for (const auto &error : changedDescriptor.errors)
            diagnostics << '\n' << error;
        return diagnostics.str();
    }

    {
        const std::string changedShaderId = changedDescriptor.shaderId;
        if (changedShaderId.empty())
            return "ShaderInfo Name is required for runtime reload";
        if (!previousShaderId.empty() && changedShaderId != previousShaderId) {
            // Rename the label of this GUID, not every material with the same
            // display name. Validate all referencing stage pairs before any
            // live material moves to the new program namespace.
            const bool vertexChanged = ext == ".vert";
            const auto materials = registry.GetAllMaterials();
            struct Migration
            {
                std::shared_ptr<InxMaterial> material;
                ShaderAssetReference reference;
                ShaderStagePair stages;
            };
            struct PreparedProgram
            {
                ShaderProgramArtifact artifact;
                ShaderDescriptor fragment;
                uint64_t sourceStamp;
                bool ui;
                std::array<std::string, 2> stageLabels;
            };
            std::vector<Migration> migrations;
            std::unordered_map<ShaderStagePair, PreparedProgram, ShaderStagePairHash> programs;
            const auto stagePath = [&](const ShaderAssetReference &reference, const char *stage) {
                return !reference.guid.empty() ? adb->GetPathFromGuid(reference.guid)
                                               : adb->FindShaderPathById(reference.shaderId, stage);
            };
            const auto readStage = [&](const std::string &path, std::string &text) {
                std::vector<char> bytes;
                if (path.empty() || !adb->ReadFile(path, bytes) || bytes.empty())
                    return false;
                if (bytes.back() == '\0')
                    bytes.pop_back();
                text.assign(bytes.begin(), bytes.end());
                return true;
            };
            for (const auto &material : materials) {
                if (!material)
                    continue;
                auto reference =
                    vertexChanged ? material->GetVertShaderReference() : material->GetFragShaderReference();
                if (reference.guid != guid)
                    continue;
                reference.shaderId = changedShaderId;
                reference.pathHint = shaderPath;
                const auto stages = material->GetShaderStages();
                migrations.push_back({material, reference, stages});
                if (IsDirectStructuredStage(changedDescriptor) || programs.count(stages))
                    continue;
                const std::string vertexPath =
                    vertexChanged ? shaderPath : stagePath(material->GetVertShaderReference(), "vertex");
                const std::string fragmentPath =
                    vertexChanged ? stagePath(material->GetFragShaderReference(), "fragment") : shaderPath;
                if (asset_formats::FindShaderSourceKind(vertexPath) ||
                    asset_formats::FindShaderSourceKind(fragmentPath))
                    continue;
                std::string vertexSource, fragmentSource;
                if (!readStage(vertexPath, vertexSource) || !readStage(fragmentPath, fragmentSource))
                    return "Shader rename requires both referenced stage GUIDs to resolve to readable sources";
                auto compiled = sourceParser.CompileLinkedProgramArtifact(
                    vertexSource, InxShaderLoader::StageQualifiedVirtualPath(vertexPath, "vertex"), fragmentSource,
                    InxShaderLoader::StageQualifiedVirtualPath(fragmentPath, "fragment"));
                if (!compiled.IsValid()) {
                    std::ostringstream errors;
                    for (const auto &error : compiled.errors)
                        errors << error << '\n';
                    return errors.str().empty() ? "Shader rename stage-pair compilation failed" : errors.str();
                }
                auto artifact = compiled.CreateRuntimeArtifact(stages);
                if (!artifact.IsValid() || artifact.key.stages != stages)
                    return "Shader rename produced a mismatched linked program";
                const auto vertex = sourceParser.ParseShaderSource(vertexSource, vertexPath);
                const auto fragment = sourceParser.ParseShaderSource(fragmentSource, fragmentPath);
                programs.emplace(stages, PreparedProgram{std::move(artifact),
                                                         fragment,
                                                         ComputeShaderProgramRevision(vertexSource, fragmentSource,
                                                                                      ShaderCompileTarget::Forward, 0),
                                                         ShaderStageLinker::IsUIStagePair(vertex, fragment),
                                                         {vertex.shaderId, fragment.shaderId}});
            }
            registry.InvalidateAsset(guid);
            if (IsDirectStructuredStage(changedDescriptor)) {
                auto shader = registry.LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
                if (!shader || !shader->HasVariant(ShaderCompileTarget::Forward))
                    return InxShaderLoader::GetLastCompileError().empty()
                               ? "Renamed standalone shader compilation failed"
                               : InxShaderLoader::GetLastCompileError();
                m_renderer->InvalidateShaderCache(changedShaderId, shader->shaderType);
                RegisterShaderToRenderer(*shader);
            }
            for (const auto &[stages, program] : programs) {
                // UI draws own publication and lifetime. Compilation above is
                // a preflight only; it must not introduce an ownerless UI artifact.
                if (!program.ui && !m_renderer->PublishShaderProgramArtifact(program.artifact))
                    return "Renderer rejected the renamed linked shader program";
            }
            for (const auto &[stages, program] : programs) {
                if (program.ui)
                    continue;
                m_linkedShaderProgramCache[stages] = {program.sourceStamp, program.artifact.key, 0, {}};
                m_linkedShaderProgramCache[stages].sourceEnvironmentRevision =
                    InxShaderLoader::GetSourceEnvironmentRevision();
                m_linkedShaderProgramCache[stages].stageLabels = program.stageLabels;
                const auto &fragment = program.fragment;
                m_renderer->StoreShaderRenderMeta(
                    stages.fragmentKey, fragment.surfaceOptions.cullMode, fragment.depthWrite, fragment.depthTest,
                    fragment.surfaceOptions.blendMode, fragment.renderQueue, fragment.passTag, fragment.stencil,
                    fragment.surfaceOptions.alphaClip);
            }
            for (const auto &migration : migrations) {
                if (vertexChanged)
                    migration.material->SetVertShaderReference(migration.reference);
                else
                    migration.material->SetFragShaderReference(migration.reference);
                const auto program = programs.find(migration.stages);
                if (program != programs.end() && program->second.ui)
                    m_renderer->InvalidateUIMaterialProgram(migration.stages);
                else
                    m_renderer->RefreshMaterialPipeline(migration.material);
            }
            // Asset-backed programs keep the same key through a label change.
            // Only an unreferenced standalone stage has a name-owned module.
            const bool oldNameReferenced = std::any_of(materials.begin(), materials.end(), [&](const auto &material) {
                return material && (vertexChanged ? material->GetVertShaderName() : material->GetFragShaderName()) ==
                                       previousShaderId;
            });
            if (!oldNameReferenced)
                m_renderer->InvalidateShaderCache(previousShaderId, vertexChanged ? "vertex" : "fragment");
            return "";
        }

        return ReloadShaderSourceBatch({shaderPath});
    }
}

void Infernux::ReloadTexture(const std::string &texturePath)
{
    if (!CheckEngineValid("reload texture") || !m_renderer) {
        INXLOG_ERROR("Infernux::ReloadTexture: engine or renderer invalid");
        return;
    }

    // Boundary adapter: this is the only place where the texture hot-reload
    // path crosses from the file-system domain (paths, from file watchers)
    // into the renderer domain (GUID-only). Resolve here or bail out.
    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    std::string guid;
    if (adb) {
        guid = adb->GetGuidFromPath(texturePath);
        if (guid.empty()) {
            // Unregistered texture (e.g. just created) — register to mint a GUID.
            guid = adb->ImportAsset(texturePath).guid;
        }
    }

    if (guid.empty()) {
        INXLOG_WARN("Infernux::ReloadTexture: could not resolve GUID for '", texturePath,
                    "' — skipping GPU cache invalidation (GUID-only contract).");
        return;
    }

    // Retire the CPU publication without decoding the processed texture on the
    // render/UI owner thread. Material descriptors retain their last-known-good
    // GPU publication while ResolveTextureAsset asynchronously loads and
    // uploads the new revision.
    registry.InvalidateAsset(guid);

    // GUID-only invalidation (path fallback removed by design).
    m_renderer->InvalidateTextureCache(guid);

    // Fire graph notification so dependent materials get their pipelines
    // invalidated via the Texture Modified callback.
    {
        auto dependents = AssetDependencyGraph::Instance().GetDependents(guid);
        AssetDependencyGraph::Instance().NotifyEvent(guid, ResourceType::Texture, AssetEvent::Modified);
    }
}

void Infernux::ReloadMesh(const std::string &meshPath)
{
    if (!CheckEngineValid("reload mesh")) {
        INXLOG_ERROR("Infernux::ReloadMesh: engine invalid");
        return;
    }

    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    std::string guid;
    if (adb)
        guid = adb->GetGuidFromPath(meshPath);

    if (guid.empty()) {
        INXLOG_WARN("Infernux::ReloadMesh: could not resolve GUID for '", meshPath, "'");
        return;
    }

    const bool handled = registry.IsLoaded(guid) || registry.GetPendingMeshReload(guid)
                             ? registry.ReloadAsset(guid)
                             : registry.NotifySourceImported(guid, ResourceType::Mesh);
    if (!handled)
        AssetDependencyGraph::Instance().NotifyEvent(guid, ResourceType::Mesh, AssetEvent::Modified);
    if (m_renderer)
        m_renderer->RequestFullSpeedFrame();
}

void Infernux::ReloadAudio(const std::string &audioPath)
{
    INXLOG_INFO("Infernux::ReloadAudio called: ", audioPath);

    if (!CheckEngineValid("reload audio")) {
        INXLOG_ERROR("Infernux::ReloadAudio: engine invalid");
        return;
    }

    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    std::string guid;
    if (adb)
        guid = adb->GetGuidFromPath(audioPath);

    if (guid.empty()) {
        INXLOG_WARN("Infernux::ReloadAudio: could not resolve GUID for '", audioPath, "'");
        return;
    }

    if (registry.IsLoaded(guid))
        registry.ReloadAsset(guid);

    AssetDependencyGraph::Instance().NotifyEvent(guid, ResourceType::Audio, AssetEvent::Modified);

    INXLOG_INFO("Infernux::ReloadAudio: done for '", audioPath, "'");
}

// ----------------------------------
// Debug
// ----------------------------------

void Infernux::SetLogLevel(LogLevel engineLevel)
{
    m_logLevel = engineLevel;
    if (m_ownsRuntimeServices)
        INXLOG_SET_LEVEL(engineLevel);
}

// ----------------------------------
// ImGui layout save / load (Unicode-safe)
// ----------------------------------

void Infernux::ResetImGuiLayout()
{
    // Clear ImGui's in-memory ini state (windows, docking, tables)
    ResetEditorGuiLayout();
    // Delete the persisted ini file so the reset survives a restart
    if (!m_imguiIniPath.empty() && std::filesystem::exists(m_imguiIniPath)) {
        std::filesystem::remove(m_imguiIniPath);
    }
    if (!m_imguiLayoutMetadataPath.empty() && std::filesystem::exists(m_imguiLayoutMetadataPath)) {
        std::filesystem::remove(m_imguiLayoutMetadataPath);
    }
}

void Infernux::SelectDockedWindow(const std::string &windowId, bool allowDuringModal)
{
    auto *renderer = GetRenderer();
    if (renderer == nullptr) {
        return;
    }
    renderer->QueueDockTabSelection(windowId.c_str(), allowDuringModal);
}

uint64_t Infernux::QueueSyntheticKeyInput(int scancode, bool pressed, bool repeat)
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->QueueSyntheticKeyInput(scancode, pressed, repeat) : 0;
}

uint64_t Infernux::QueueSyntheticMouseButtonInput(int button, bool pressed, float x, float y)
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->QueueSyntheticMouseButtonInput(button, pressed, x, y) : 0;
}

uint64_t Infernux::QueueSyntheticMouseMotionInput(float x, float y, float deltaX, float deltaY)
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->QueueSyntheticMouseMotionInput(x, y, deltaX, deltaY) : 0;
}

uint64_t Infernux::QueueSyntheticMouseWheelInput(float horizontal, float vertical)
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->QueueSyntheticMouseWheelInput(horizontal, vertical) : 0;
}

uint64_t Infernux::QueueSyntheticTextInput(const std::string &text)
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->QueueSyntheticTextInput(text) : 0;
}

uint64_t Infernux::QueueSyntheticCloseRequest()
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->QueueSyntheticCloseRequest() : 0;
}

uint64_t Infernux::GetLastProcessedSyntheticInputSequence() const
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->GetLastProcessedSyntheticInputSequence() : 0;
}

size_t Infernux::GetPendingSyntheticInputCount() const
{
    auto *renderer = GetRenderer();
    return renderer ? renderer->GetPendingSyntheticInputCount() : 0;
}

void Infernux::LoadImGuiLayout()
{
    if (!std::filesystem::exists(m_imguiIniPath))
        return;
    if (!std::filesystem::exists(m_imguiLayoutMetadataPath)) {
        INXLOG_INFO("Ignoring unversioned ImGui layout; a current DPI-aware layout will be written on save");
        return;
    }

    std::ifstream metadataStream(m_imguiLayoutMetadataPath, std::ios::binary);
    if (!metadataStream.is_open())
        throw std::runtime_error("Cannot read ImGui layout metadata");

    json metadata;
    try {
        metadataStream >> metadata;
    } catch (const json::exception &error) {
        INXLOG_WARN("Ignoring invalid ImGui layout metadata: ", error.what());
        return;
    }
    if (!metadata.is_object() || metadata.value("schema", 0) != 1 || !metadata.contains("display_scale") ||
        !metadata["display_scale"].is_number()) {
        INXLOG_WARN("Ignoring ImGui layout with an unsupported metadata contract");
        return;
    }

    const float storedScale = metadata["display_scale"].get<float>();
    // This legacy metadata key stores the authored-to-window layout multiplier.
    // Correcting Retina's old double scaling changes it (e.g. 2 -> 1), so those
    // layouts are rejected below. Windows pixel density is 1 and its existing
    // correctly scaled user layouts remain compatible.
    if (!std::isfinite(storedScale) || storedScale <= 0.0f) {
        INXLOG_WARN("Ignoring ImGui layout with an invalid display scale");
        return;
    }
    auto *renderer = GetRenderer();
    if (renderer == nullptr)
        throw std::logic_error("Cannot load ImGui layout without an initialized renderer");
    const float currentScale = renderer->GetDisplayScale();
    if (std::abs(storedScale - currentScale) >= 0.01f) {
        INXLOG_INFO("Ignoring ImGui layout authored at display scale ", storedScale, "; current display scale is ",
                    currentScale);
        return;
    }

    // std::ifstream(std::filesystem::path) uses wchar_t on Windows,
    // so paths with Chinese / non-ASCII characters are handled properly.
    std::ifstream ifs(m_imguiIniPath, std::ios::binary | std::ios::ate);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot read persisted ImGui layout");
    auto size = ifs.tellg();
    if (size <= 0) {
        INXLOG_WARN("Ignoring empty ImGui layout");
        return;
    }
    ifs.seekg(0);
    std::string data(static_cast<size_t>(size), '\0');
    ifs.read(data.data(), size);
    ImGui::LoadIniSettingsFromMemory(data.c_str(), data.size());
}

void Infernux::SaveImGuiLayout()
{
    if (m_imguiIniPath.empty() || m_imguiLayoutMetadataPath.empty())
        return;
    size_t dataSize = 0;
    const char *data = ImGui::SaveIniSettingsToMemory(&dataSize);
    if (!data || dataSize == 0)
        return;
    auto *renderer = GetRenderer();
    if (renderer == nullptr)
        throw std::logic_error("Cannot save ImGui layout without an initialized renderer");
    const float displayScale = renderer->GetDisplayScale();
    std::filesystem::create_directories(m_imguiIniPath.parent_path());
    std::ofstream ofs(m_imguiIniPath, std::ios::binary);
    if (!ofs.is_open())
        throw std::runtime_error("Cannot write persisted ImGui layout");
    ofs.write(data, static_cast<std::streamsize>(dataSize));
    if (!ofs.good())
        throw std::runtime_error("Failed while writing persisted ImGui layout");

    std::ofstream metadataStream(m_imguiLayoutMetadataPath, std::ios::binary);
    if (!metadataStream.is_open())
        throw std::runtime_error("Cannot write ImGui layout metadata");
    metadataStream << json{{"schema", 1}, {"display_scale", displayScale}}.dump(2) << '\n';
    if (!metadataStream.good())
        throw std::runtime_error("Failed while writing ImGui layout metadata");
}

} // namespace infernux
