#include "Infernux.h"

#include <core/serialization/BinaryCodec.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderAuthorMaterial.h>
#include <function/resources/ShaderAsset/ShaderAuthorMetadata.h>
#include <function/resources/ShaderAsset/ShaderAuthorProgramCache.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>

#include <algorithm>
#include <exception>
#include <thread>
#include <unordered_set>

namespace infernux
{
namespace
{
std::optional<std::string> ReadAuthorText(const AssetDatabase &database, const std::string &path)
{
    std::vector<char> bytes;
    if (!database.ReadFile(path, bytes))
        return std::nullopt;
    return std::string(bytes.begin(), bytes.end());
}

// The owner freezes raw inputs and runs the Python frontend on a miss. The
// worker receives either immutable target code or fully lowered Slang modules.
struct FrozenAuthorProgram
{
    std::string selection;
    std::vector<ShaderCacheInput> inputs;
    ShaderAuthorSource generated;
    ShaderAuthorProgramRecord record;

    void Compile(const std::string &cacheDirectory)
    {
        if (record.compilation)
            return;
        record.compilation =
            std::make_shared<const ShaderCompilation>(SlangCompiler::Get().CompileAuthorProgram(generated));
        StoreShaderAuthorProgram(cacheDirectory, selection, inputs, record);
    }
};

FrozenAuthorProgram FreezeAuthorProgram(const AssetDatabase &database, const ShaderSourceModule &source,
                                        const ShaderSourceModule &vertex = {})
{
    FrozenAuthorProgram frozen;
    const auto vertexGuid = vertex.path.empty() ? std::string{} : database.GetGuidFromPath(vertex.path);
    frozen.selection =
        ShaderAuthorProgramSelection(database.GetGuidFromPath(source.path), vertexGuid, RuntimeShaderAuthorTarget());
    const auto capture = [&](const std::string &path, const std::string &text) {
        frozen.inputs.push_back(MakeShaderCacheInput(ShaderAuthorInputId(database, path), text, path));
    };
    capture(source.path, source.source);
    if (!vertex.path.empty())
        capture(vertex.path, vertex.source);
    if (auto program = FindShaderAuthorProgram(database, frozen.selection, frozen.inputs, &frozen.inputs)) {
        frozen.record = std::move(*program);
        return frozen;
    }
    const auto read = [&](const std::string &path) {
        auto text = ReadAuthorText(database, path);
        if (text)
            capture(path, *text);
        return text;
    };
    frozen.generated = SlangCompiler::Get().PrepareAuthorSource(source, read, ShaderAuthorPhase::Program, vertex);
    frozen.record.schema = frozen.generated.schema;
    for (const auto &library : frozen.generated.libraries) {
        if (asset_formats::FindShaderSourceKind(library.path)) {
            auto guid = database.GetGuidFromPath(library.path);
            if (!guid.empty())
                frozen.record.dependencies.push_back(std::move(guid));
        }
    }
    return frozen;
}
} // namespace

struct LinkedShaderProgramLoadTicket::SourceIdentity
{
    uint64_t sourceStamp = binary_codec::Fnv1a64Basis;
    std::vector<std::pair<std::string, std::shared_ptr<const InxResourceMeta>>> inputs;

    bool operator==(const SourceIdentity &other) const noexcept
    {
        return sourceStamp == other.sourceStamp && inputs == other.inputs;
    }
};

struct LinkedShaderProgramLoadTicket::State
{
    struct Asset
    {
        std::string guid;
        ShaderSourceModule source;
        FrozenAuthorProgram author;
        std::shared_ptr<ShaderAsset> compiled;
    };
    struct Work
    {
        ShaderStagePair stages;
        std::optional<FrozenAuthorProgram> linked;
        std::shared_ptr<Asset> fragment;
        std::shared_ptr<Asset> vertex;
        ShaderRenderMeta renderMeta;
        std::shared_ptr<const SourceIdentity> identity;
        ShaderProgramArtifact program;
        std::string error;
        std::vector<ShaderDiagnostic> diagnostics;
    };
    struct MaterialBinding
    {
        std::string guid;
        ShaderAssetReference vertex;
        ShaderAssetReference fragment;
    };
    std::vector<Work> work;
    std::vector<std::shared_ptr<Asset>> assets;
    std::vector<std::pair<ShaderStagePair, std::shared_ptr<const SourceIdentity>>> sources;
    std::vector<MaterialBinding> materials;
    std::string cacheDirectory;
    const Infernux *owner = nullptr;
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

void Infernux::RetireShaderPreparations(const std::vector<std::string> &sourcePaths)
{
    const std::unordered_set<std::string> changed(sourcePaths.begin(), sourcePaths.end());
    for (auto entry = m_linkedShaderPreparations.begin(); entry != m_linkedShaderPreparations.end();) {
        const auto &inputs = entry->second->inputs;
        if (std::any_of(inputs.begin(), inputs.end(), [&](const auto &input) { return changed.count(input.first); }))
            entry = m_linkedShaderPreparations.erase(entry);
        else
            ++entry;
    }
}

std::shared_ptr<LinkedShaderProgramLoadTicket>
Infernux::BeginPrepareLinkedShaderPrograms(const std::vector<std::string> &materialGuids)
{
    auto ticket = std::make_shared<LinkedShaderProgramLoadTicket>();
    ticket->m_state = std::make_shared<LinkedShaderProgramLoadTicket::State>();
    auto state = ticket->m_state;
    state->ownerThread = std::this_thread::get_id();
    state->owner = this;
    auto *database = GetAssetDatabase();
    if (!m_renderer || !database)
        return ticket;
    state->cacheDirectory = ProjectCacheDirectory(database->GetProjectRoot());

    const auto pathFor = [database](const ShaderAssetReference &reference, const char *stage) {
        if (!reference.guid.empty())
            return database->GetPathFromGuid(reference.guid);
        return reference.pathHint.empty() ? database->FindShaderPathById(reference.shaderId, stage) : std::string{};
    };
    std::unordered_map<std::string, std::shared_ptr<LinkedShaderProgramLoadTicket::State::Asset>> snapshots;
    const auto freeze = [&](const ShaderSourceModule &source) {
        const auto guid = database->GetGuidFromPath(source.path);
        const auto found = snapshots.find(guid);
        if (found != snapshots.end())
            return found->second;
        auto asset = std::make_shared<LinkedShaderProgramLoadTicket::State::Asset>();
        asset->guid = guid;
        asset->source = {"asset_" + guid, source.path, source.source};
        asset->author = FreezeAuthorProgram(*database, asset->source);
        snapshots.emplace(guid, asset);
        return asset;
    };
    std::unordered_set<std::string> scheduledAssets;
    const auto scheduleAsset = [&](const auto &asset) {
        if (scheduledAssets.insert(asset->guid).second)
            state->assets.push_back(asset);
    };
    std::unordered_set<ShaderStagePair, ShaderStagePairHash> scheduled;
    auto &registry = AssetRegistry::Instance();
    for (const auto &guid : materialGuids) {
        const auto material = registry.GetAsset<InxMaterial>(guid);
        if (!material)
            continue;
        state->materials.push_back({guid, material->GetVertShaderReference(), material->GetFragShaderReference()});
        const auto stages = material->GetShaderStages();
        if (!stages.IsValid() || !scheduled.insert(stages).second)
            continue;
        LinkedShaderProgramLoadTicket::State::Work work;
        work.stages = stages;
        LinkedShaderProgramLoadTicket::SourceIdentity identity;
        const auto remember = [&](const std::string &path) {
            if (!path.empty())
                identity.inputs.emplace_back(path, database->GetMetaByPath(path));
        };
        ShaderSourceModule fragment;
        ShaderSourceModule vertex;
        try {
            const auto fragmentPath = pathFor(material->GetFragShaderReference(), "fragment");
            const auto vertexPath =
                stages.vertexKey.empty() ? std::string{} : pathFor(material->GetVertShaderReference(), "vertex");
            remember(fragmentPath);
            remember(vertexPath);
            auto fragmentText = ReadAuthorText(*database, fragmentPath);
            if (!fragmentText)
                throw std::invalid_argument("Material fragment source is unavailable: " + fragmentPath);
            fragment = {"asset_" + database->GetGuidFromPath(fragmentPath), fragmentPath, std::move(*fragmentText)};
            const auto vertexMetadata = vertexPath.empty() ? nullptr : database->GetMetaByPath(vertexPath);
            const bool standardVertex =
                stages.vertexKey == kStandardVertexShaderId ||
                (vertexMetadata && vertexMetadata->GetDataAs<std::string>("shader_id") == kStandardVertexShaderId);
            if (!vertexPath.empty() && !standardVertex) {
                auto source = ReadAuthorText(*database, vertexPath);
                if (!source)
                    throw std::invalid_argument("Material vertex source is unavailable: " + vertexPath);
                vertex = {"vertex_" + stages.vertexKey, vertexPath, std::move(*source)};
                work.linked = FreezeAuthorProgram(
                    *database, {"material_" + stages.fragmentKey, fragmentPath, fragment.source}, vertex);
            } else {
                work.fragment = freeze(fragment);
            }
            const auto &author = work.linked ? *work.linked : work.fragment->author;
            const auto schema = nlohmann::json::parse(author.record.schema);
            const auto domain = schema.at("domain").get<std::string>();
            // UI draws own the device publication and release of their
            // programs. Scene preload prepares no UI pipeline or reference.
            if (domain == "screen_ui" || domain == "world_ui")
                continue;
            if (domain != "mesh" && domain != "particle")
                throw std::invalid_argument("Material fragment must declare a mesh or particle shader: " +
                                            fragmentPath);
            if (domain == "particle" && !stages.vertexKey.empty())
                throw std::invalid_argument(
                    "Particle geometry is provided by its output; remove the material vertex selection: " +
                    fragmentPath);
            work.renderMeta = ReadShaderAuthorRenderMeta(schema.at("info"));
            for (const auto &input : author.inputs)
                identity.sourceStamp = binary_codec::Fnv1a64(input.id, identity.sourceStamp ^ input.revision);
            for (const auto &dependency : author.record.dependencies)
                remember(database->GetPathFromGuid(dependency));
        } catch (const ShaderCompilationError &error) {
            work.error = error.what();
            work.diagnostics = error.diagnostics;
        } catch (const std::exception &error) {
            work.error = error.what();
        }
        auto &current = m_linkedShaderPreparations[stages];
        if (!current || !(*current == identity))
            current = std::make_shared<const LinkedShaderProgramLoadTicket::SourceIdentity>(std::move(identity));
        work.identity = current;
        // Cached programs still participate in publication freshness. A later
        // edit must supersede a ticket even when it needed no compiler work.
        state->sources.emplace_back(stages, current);
        const auto cached = m_linkedShaderProgramCache.find(stages);
        if (work.error.empty() && cached != m_linkedShaderProgramCache.end() &&
            cached->second.sourceStamp == current->sourceStamp &&
            m_renderer->HasShaderProgramArtifact(cached->second.programKey))
            continue;
        if (work.error.empty()) {
            try {
                if (work.linked) {
                    work.fragment = freeze(fragment);
                    work.vertex = freeze(vertex);
                    scheduleAsset(work.vertex);
                }
                scheduleAsset(work.fragment);
            } catch (const ShaderCompilationError &error) {
                work.error = error.what();
                work.diagnostics = error.diagnostics;
            } catch (const std::exception &error) {
                work.error = error.what();
            }
        }
        state->work.push_back(std::move(work));
    }
    if (state->work.empty())
        return ticket;
    const auto compile = [state] {
        state->producerThread = std::this_thread::get_id();
        try {
            for (const auto &asset : state->assets) {
                if (state->cancelRequested.load(std::memory_order_acquire))
                    return;
                asset->author.Compile(state->cacheDirectory);
                asset->compiled =
                    ShaderLoader::RestoreAuthorAsset(asset->source, asset->guid, std::move(asset->author.record));
            }
            for (auto &work : state->work) {
                if (state->cancelRequested.load(std::memory_order_acquire))
                    break;
                if (!work.error.empty())
                    continue;
                try {
                    if (work.linked) {
                        work.linked->Compile(state->cacheDirectory);
                        const auto &record = work.linked->record;
                        work.program = CreateShaderAuthorMaterial(work.stages.fragmentKey, record.compilation,
                                                                  record.schema, work.stages.vertexKey);
                    } else {
                        work.program = *work.fragment->compiled->materialProgram;
                    }
                    work.program.key.stages = work.stages;
                    auto preparation = std::make_shared<ShaderMaterialPreparation>();
                    preparation->program = std::make_shared<const ShaderProgramArtifact>(work.program);
                    preparation->vertex = work.vertex ? work.vertex->compiled->program : nullptr;
                    work.fragment->compiled->materialPrograms->entries[work.stages.vertexKey] = std::move(preparation);
                } catch (const ShaderCompilationError &error) {
                    work.error = error.what();
                    work.diagnostics = error.diagnostics;
                } catch (const std::exception &error) {
                    work.error = error.what();
                }
            }
        } catch (...) {
            state->failure = std::current_exception();
        }
    };
    if (JobSystem::IsAvailable() && !JobSystem::Get().IsInline())
        state->job = JobSystem::Get().Schedule(compile, JobDomain::Asset, JobPriority::Normal, JobAffinity::WorkerOnly);
    else
        compile();
    return ticket;
}

bool Infernux::TryCommitLinkedShaderPrograms(const std::shared_ptr<LinkedShaderProgramLoadTicket> &ticket)
{
    if (!ticket || !ticket->m_state)
        return false;
    auto &state = *ticket->m_state;
    if (std::this_thread::get_id() != state.ownerThread)
        throw std::logic_error("Shader program publication must run on its ticket owner thread");
    if (!ticket->IsComplete())
        return false;
    if (state.committed)
        return true;
    if (state.cancelRequested.load(std::memory_order_acquire))
        return false;
    auto *database = GetAssetDatabase();
    bool current = !state.superseded && state.owner == this && !m_isCleaningUp && !m_isCleanedUp;
    for (const auto &binding : state.materials) {
        const auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(binding.guid);
        current = current && material && material->GetVertShaderReference() == binding.vertex &&
                  material->GetFragShaderReference() == binding.fragment;
    }
    for (const auto &[stages, identity] : state.sources) {
        const auto prepared = m_linkedShaderPreparations.find(stages);
        current = current && database && prepared != m_linkedShaderPreparations.end() && prepared->second == identity;
        for (const auto &[path, metadata] : identity->inputs)
            current = current && database->GetMetaByPath(path) == metadata;
    }
    if (!current) {
        state.superseded = true;
        return false;
    }
    if (state.failure)
        std::rethrow_exception(state.failure);
    std::vector<ShaderProgramArtifact> programs;
    std::unordered_map<ShaderStagePair, const ShaderProgramArtifact *, ShaderStagePairHash> publications;
    for (const auto &work : state.work) {
        if (!work.error.empty()) {
            if (!work.diagnostics.empty())
                throw ShaderCompilationError(work.diagnostics);
            throw std::runtime_error(work.error);
        }
        programs.push_back(work.program);
        publications.emplace(work.stages, &work.program);
    }
    if (!programs.empty() && !m_renderer->PublishShaderProgramArtifacts(programs))
        throw std::runtime_error("Renderer rejected the prepared shader program batch");
    auto &registry = AssetRegistry::Instance();
    for (const auto &asset : state.assets)
        registry.PublishShader(asset->guid, asset->compiled);
    for (const auto &binding : state.materials) {
        const auto material = registry.GetAsset<InxMaterial>(binding.guid);
        const auto program = publications.find(material->GetShaderStages());
        if (program != publications.end())
            material->SynchronizeShaderPropertyDefaults(*program->second);
    }
    for (const auto &work : state.work) {
        auto &entry = m_linkedShaderProgramCache[work.stages];
        entry.sourceStamp = work.identity->sourceStamp;
        entry.programKey = work.program.key;
        entry.lastError.clear();
        const auto &meta = work.renderMeta;
        m_renderer->StoreShaderRenderMeta(work.stages.fragmentKey, meta.cullMode, meta.depthWrite, meta.depthTest,
                                          meta.blend, meta.queue, meta.passTag, meta.stencil, meta.alphaClip);
    }
    state.committed = true;
    return true;
}

} // namespace infernux
