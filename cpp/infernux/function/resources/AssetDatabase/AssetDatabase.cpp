#include "AssetDatabase.h"
#include "BuiltinSceneIconMetadata.h"

#include <function/resources/AssetFormatRegistry.h>

#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetImporter/ConcreteImporters.h>
#include <function/resources/AssetImporter/PluginPageTextureMetadata.h>
#include <function/resources/AssetRegistry/ArtifactFileReader.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifact.h>
#include <function/resources/InxMesh/MeshImportSettings.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <function/resources/InxTexture/TextureArtifact.h>
#include <function/resources/RenderTexture/RenderTextureArtifact.h>

#include <core/log/InxLog.h>
#include <platform/filesystem/DocumentStore.h>
#include <platform/filesystem/InxPack.h>
#include <platform/filesystem/InxPath.h>
#include <platform/filesystem/IoAffinity.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

namespace infernux
{

namespace
{
constexpr size_t kOwnerMergeEntryBudget = 256;
constexpr auto kOwnerMergeTimeBudget = std::chrono::milliseconds(2);

std::string Utf8Diagnostic(const char *message)
{
    // Parser errors can quote binary source bytes; diagnostics must remain valid JSON and Python text.
    const auto encoded = nlohmann::json(message).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    return nlohmann::json::parse(encoded).get<std::string>();
}

void ApplyDocumentationTextureMovePolicy(AssetDatabase &database, AssetMutationResult &result,
                                         const PathWriteScope &writeScope)
{
    if (result.resourceType != ResourceType::Texture)
        return;
    const auto metadata = database.GetMetaByGuid(result.guid);
    if (!metadata || HasCurrentPluginPageTextureMetadata(*metadata, result.path, database.GetProjectRoot()))
        return;
    // Relocation keeps the GUID, but entering a documentation directory
    // changes the import policy even when the source bytes are unchanged.
    const auto imported = database.ReimportAsset(result.path, nullptr, writeScope);
    if (!imported.succeeded) {
        result.succeeded = false;
        result.errorCode = AssetMutationErrorCode::RuntimeApplyFailed;
        result.error = imported.error;
    }
    result.queryGeneration = database.GetQueryGeneration();
}

bool IsCanonicalAssetGuid(std::string_view value)
{
    return value.size() == 32 && std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return std::isdigit(character) != 0 || (character >= 'a' && character <= 'f');
           });
}

template <typename Snapshot>
std::function<std::string(const std::string &, bool)> ModelTextureResolver(std::shared_ptr<Snapshot> snapshot)
{
    return [snapshot = std::move(snapshot)](const std::string &path, bool linear) -> std::string {
        const auto found = snapshot->pathToGuid.find(FilesystemPathKey(path));
        if (found == snapshot->pathToGuid.end())
            throw std::invalid_argument("model texture is not a registered project asset: " + path);
        const auto metadata = snapshot->metas.find(found->second);
        if (metadata == snapshot->metas.end() || metadata->second->GetResourceType() != ResourceType::Texture)
            throw std::invalid_argument("model texture path does not identify a Texture asset: " + path);
        if (linear && (!metadata->second->HasKey("srgb") || metadata->second->template GetDataAs<bool>("srgb")))
            throw std::invalid_argument(
                "model data texture requires linear sampling; disable sRGB in its import settings: " + path);
        return found->second;
    };
}

// A model sidecar is the sole authority for its imported Texture identities.
void ApplyModelTextureSettings(InxResourceMeta &model, const std::string &guid, const nlohmann::json &settings)
{
#if defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    (void)model;
    (void)guid;
    (void)settings;
    throw std::logic_error("Model Texture authoring is unavailable in a minimal Player host");
#else
    if (!settings.is_object())
        throw std::invalid_argument("model Texture settings must be an object");
    InxResourceMeta defaults;
    TextureImporter{}.EnsureDefaultSettings(defaults);
    const auto fields = defaults.SerializeDocument().at("metadata");
    auto records = nlohmann::json::parse(model.GetDataAs<std::string>("model_textures"));
    for (auto &record : records) {
        if (record.at("guid") != guid)
            continue;
        auto &entries = record.at("metadata").at("metadata");
        for (const auto &[key, value] : settings.items()) {
            if (key == "sprite_frames" && value.is_array()) {
                entries[key] = {{"type", "json_array"}, {"value", value}};
                continue;
            }
            if (!fields.contains(key))
                throw std::invalid_argument("unknown Texture import setting: " + key);
            const auto type = fields.at(key).at("type").get<std::string>();
            if ((type == "string" && !value.is_string()) || (type == "bool" && !value.is_boolean()) ||
                (type == "int" && !value.is_number_integer()))
                throw std::invalid_argument("invalid Texture import setting type: " + key);
            entries[key] = {{"type", type}, {"value", value}};
        }
        model.AddMetadata("model_textures", records.dump());
        return;
    }
    throw std::invalid_argument("model no longer owns Texture: " + guid);
#endif
}

// Stored catalog paths already have canonical backing paths. Virtual suffixes
// are identities, not filenames to probe or case-fold on the filesystem.
std::string IndexedAssetPathKey(const std::string &path)
{
    const auto suffix = path.find("::");
    return suffix == std::string::npos ? FoldFilesystemPathCase(path)
                                       : FoldFilesystemPathCase(path.substr(0, suffix)) + path.substr(suffix);
}

template <typename Visit> void VisitModelSubAssets(const InxResourceMeta &owner, Visit visit)
{
    for (const auto *table : {"model_textures", "model_animations"}) {
        if (!owner.HasKey(table))
            continue;
        const auto records = nlohmann::json::parse(owner.GetDataAs<std::string>(table));
        for (const auto &record : records)
            visit(record);
    }
}

std::shared_ptr<InxResourceMeta> ModelSubAssetMetadata(const nlohmann::json &record, const std::string &owner,
                                                       const std::string &sourcePath)
{
    auto metadata = std::make_shared<InxResourceMeta>();
    metadata->DeserializeDocument(record.at("metadata"));
    const auto guid = metadata->GetGuid();
    if (!IsCanonicalAssetGuid(guid) || guid == owner)
        throw std::invalid_argument("invalid model sub-asset identity: " + guid);
    const auto suffix =
        metadata->HasKey("import_document") ? "::subanim:" + record.at("id").get<std::string>() : "::subtex:" + guid;
    metadata->AddMetadata("file_path", sourcePath + suffix);
    metadata->AddMetadata("import_owner_guid", owner);
    return metadata;
}

// Full scans rebuild all derived entries once; individual imports update only
// their owner through UpdateModelSubAssets.
template <typename Paths, typename Metas, typename States, typename Results>
void ExpandModelSubAssets(Paths &guidToPath, Paths &pathToGuid, Metas &metas, States &fileStates, Results &results)
{
    std::vector<std::string> owners;
    for (auto it = metas.begin(); it != metas.end();) {
        if (it->second->HasKey("import_owner_guid")) {
            const auto path = guidToPath.find(it->first);
            if (path != guidToPath.end()) {
                const auto key = IndexedAssetPathKey(path->second);
                pathToGuid.erase(key);
                fileStates.erase(key);
                guidToPath.erase(path);
            }
            results.erase(it->first);
            it = metas.erase(it);
        } else {
            if (it->second->HasKey("model_textures") || it->second->HasKey("model_animations"))
                owners.push_back(it->first);
            ++it;
        }
    }
    for (const auto &owner : owners) {
        const auto sourcePath = ResolveFilesystemPath(guidToPath.at(owner));
        const auto sourceState = fileStates.at(IndexedAssetPathKey(sourcePath));
        VisitModelSubAssets(*metas.at(owner), [&](const nlohmann::json &record) {
            auto metadata = ModelSubAssetMetadata(record, owner, sourcePath);
            const auto guid = metadata->GetGuid();
            const auto path = metadata->template GetDataAs<std::string>("file_path");
            const auto key = IndexedAssetPathKey(path);
            if (metas.find(guid) != metas.end())
                throw std::invalid_argument("model sub-asset identity collides with another asset: " + guid);
            guidToPath.emplace(guid, path);
            pathToGuid.emplace(key, guid);
            metas.emplace(guid, std::move(metadata));
            auto childState = sourceState;
            childState.readOnly = true;
            fileStates.emplace(key, childState);
            if (const auto result = results.find(owner); result != results.end())
                results.emplace(guid, result->second);
        });
    }
}

void ValidateImportedDependencyIdentities(const ImportArtifact &artifact, const std::string &sourcePath)
{
    if (!artifact.dependenciesAuthoritative)
        return;

    // This is the publication boundary for the durable dependency graph.
    // Asset identity is GUID-only: paths and path_hint values are mutable
    // presentation metadata and must never be accepted as graph identities.
    for (const auto &dependency : artifact.dependencies) {
        if (dependency == artifact.metadata.GetGuid()) {
            throw std::runtime_error("Importer for '" + sourcePath + "' produced a self-dependency");
        }
        if (!IsCanonicalAssetGuid(dependency)) {
            throw std::runtime_error("Importer for '" + sourcePath + "' produced a non-GUID dependency identity '" +
                                     dependency + "'");
        }
    }
}

void ResolveImportedDependencyPathHints(ImportArtifact &artifact,
                                        const std::unordered_map<std::string, std::string> &pathToGuid,
                                        const std::string &sourcePath)
{
    for (const auto &[path, guid] : artifact.resolvedTextureSources) {
        const auto found = pathToGuid.find(FilesystemPathKey(path));
        if (found == pathToGuid.end() || found->second != guid)
            throw std::runtime_error("model texture identity changed during import: " + path);
    }
    if (artifact.dependencyPathHints.empty())
        return;

    std::unordered_set<std::string> dependencies(artifact.dependencies.begin(), artifact.dependencies.end());
    for (const auto &path : artifact.dependencyPathHints) {
        const auto it = pathToGuid.find(FilesystemPathKey(path));
        if (it == pathToGuid.end() || !IsCanonicalAssetGuid(it->second))
            throw std::runtime_error("Importer for '" + sourcePath + "' could not resolve model dependency path '" +
                                     path + "' to a project asset GUID");
        dependencies.insert(it->second);
    }

    artifact.dependencies.assign(dependencies.begin(), dependencies.end());
    std::sort(artifact.dependencies.begin(), artifact.dependencies.end());
    artifact.dependenciesAuthoritative = true;
    artifact.dependencyPathHints.clear();
}

template <typename Resolve> void ValidateModelMaterialTargets(const ImportArtifact &artifact, Resolve resolve)
{
    if (artifact.metadata.GetResourceType() != ResourceType::Mesh)
        return;
    const auto settings = MeshImportSettings::Read(artifact.metadata);
    if (settings.materialImportMode == "none")
        return;
    for (const auto &[sourceId, guid] : settings.materialRemaps.items()) {
        const auto material = resolve(guid.get<std::string>());
        if (!material || material->GetResourceType() != ResourceType::Material)
            throw std::invalid_argument("model material remap requires a registered Material: " + sourceId);
    }
}

InxResourceMeta LoadMetadataDocument(const std::string &path, AtomicFileState *consumedState = nullptr,
                                     std::string *consumedContent = nullptr)
{
    auto snapshot = ReadTextFileSnapshot(path);
    const nlohmann::json document = nlohmann::json::parse(snapshot.content);
    InxResourceMeta metadata;
    metadata.DeserializeDocument(document);
    if (consumedState)
        *consumedState = snapshot.state;
    if (consumedContent)
        *consumedContent = std::move(snapshot.content);
    return metadata;
}

std::string CopiedSkeletonOwner(const ImportRequest &request)
{
    if (request.resourceType != ResourceType::Mesh)
        return {};
    const auto settings = MeshImportSettings::Read(request.metadata);
    if (settings.skeletonDefinitionMode != "copy")
        return {};
    if (!IsCanonicalAssetGuid(settings.skeletonDefinitionGuid) || settings.skeletonDefinitionGuid == request.guid)
        throw std::invalid_argument("copied skeleton definition must reference another mesh asset GUID");
    return settings.skeletonDefinitionGuid;
}

std::string SkeletonSourceHash(const InxResourceMeta *metadata)
{
    if (!metadata || metadata->GetResourceType() != ResourceType::Mesh || !metadata->HasKey("content_hash"))
        throw std::invalid_argument("copied skeleton definition asset is unavailable");
    return metadata->GetDataAs<std::string>("content_hash");
}

// A single Apply pins the selected file across other catalog commits. A refresh
// owns its catalog transaction, so shared inputs can open on first worker use.
class SkeletonDefinitionCapture
{
  public:
    SkeletonDefinitionCapture(std::string sourceHash, const std::string &artifactPath, bool pinFile)
        : m_path(ToFsPath(artifactPath)), m_sourceHash(std::move(sourceHash))
    {
        if (pinFile)
            m_input = std::make_unique<FileInput>(m_path);
    }

    std::shared_ptr<const InxSkinnedMesh> Read()
    {
        std::call_once(m_once, [this] {
            auto input = std::move(m_input);
            try {
                if (!input)
                    input = std::make_unique<FileInput>(m_path);
                const auto workspace = SkinnedMeshArtifact::ReadWorkspace(*input);
                const auto bytes = (std::min<uint64_t>)(input->Size(), ArtifactFileReader::BlockBytes);
                std::vector<char> block(static_cast<size_t>(bytes));
                m_mesh = SkinnedMeshArtifact::Read(*input, m_sourceHash, block, workspace);
            } catch (...) {
                m_failure = std::current_exception();
            }
        });
        if (m_failure)
            std::rethrow_exception(m_failure);
        return m_mesh;
    }

  private:
    std::filesystem::path m_path;
    std::unique_ptr<FileInput> m_input;
    std::string m_sourceHash;
    std::once_flag m_once;
    std::shared_ptr<const InxSkinnedMesh> m_mesh;
    std::exception_ptr m_failure;
};

ImportRequest::SkeletonDefinitionReader CaptureSkeletonDefinition(std::string sourceHash,
                                                                  const std::string &artifactPath, bool pinFile)
{
    auto capture = std::make_shared<SkeletonDefinitionCapture>(std::move(sourceHash), artifactPath, pinFile);
    return [capture = std::move(capture)] { return capture->Read(); };
}

std::string
RuntimeArtifactRelativePath(const std::string &guid, ResourceType type,
                            ImportArtifact::RuntimeArtifactKind kind = ImportArtifact::RuntimeArtifactKind::Primary)
{
    if (type != ResourceType::Mesh && type != ResourceType::Texture && type != ResourceType::ParticleGraph &&
        type != ResourceType::RenderTexture)
        return {};
    if (guid.empty() || !std::all_of(guid.begin(), guid.end(), [](unsigned char character) {
            return std::isalnum(character) != 0 || character == '-' || character == '_';
        }))
        throw std::invalid_argument("runtime artifact GUID contains invalid path characters");
    if (type == ResourceType::Mesh) {
        if (kind == ImportArtifact::RuntimeArtifactKind::Primary)
            return "Library/Artifacts/Mesh/" + guid + ".inxmesh";
        if (kind == ImportArtifact::RuntimeArtifactKind::SkinnedMesh)
            return "Library/Artifacts/SkinnedMesh/" + guid + ".inxskin";
        throw std::invalid_argument("unsupported Mesh runtime artifact kind");
    }
    if (kind != ImportArtifact::RuntimeArtifactKind::Primary)
        throw std::invalid_argument("non-Mesh assets only support a primary runtime artifact");
    if (type == ResourceType::Texture)
        return "Library/Artifacts/Texture/" + guid + ".inxtex";
    if (type == ResourceType::ParticleGraph)
        return "Library/Artifacts/Particle/" + guid + ".inxparticle";
    if (type == ResourceType::RenderTexture)
        return "Library/Artifacts/RenderTexture/" + guid + ".inxrtex";
    return {};
}

bool RequiresRuntimeCpuArtifact(ResourceType type)
{
    return type == ResourceType::Mesh || type == ResourceType::Texture || type == ResourceType::RenderTexture;
}

std::string ShaderPathIndexKey(const InxResourceMeta &metadata)
{
    if (!metadata.HasKey("type") || !metadata.HasKey("shader_id"))
        return {};
    const auto type = metadata.GetDataAs<std::string>("type");
    if (type != "vertex" && type != "fragment")
        return {};
    return type + ":" + metadata.GetDataAs<std::string>("shader_id");
}

std::ifstream OpenRuntimeCatalogDocument(const std::filesystem::path &path)
{
    std::ifstream stream(NativeFilePath(path), std::ios::binary);
    if (!stream)
        throw std::runtime_error("runtime catalog document cannot be opened: " + FromFsPath(path));
    return stream;
}

nlohmann::json ReadRuntimeCatalogDocument(const std::filesystem::path &path)
{
    auto stream = OpenRuntimeCatalogDocument(path);
    return nlohmann::json::parse(stream);
}

bool HasCurrentRuntimeArtifactHeader(const std::filesystem::path &path, ResourceType type,
                                     ImportArtifact::RuntimeArtifactKind kind)
{
    const auto nativePath = NativeFilePath(path);
    std::error_code error;
    if (!std::filesystem::is_regular_file(nativePath, error))
        return false;
    std::ifstream stream(nativePath, std::ios::binary);
    if (!stream)
        return false;
    char buffer[32]{};
    stream.read(buffer, sizeof(buffer));
    const std::string_view header(buffer, static_cast<size_t>(stream.gcount()));
    if (type == ResourceType::RenderTexture)
        return kind == ImportArtifact::RuntimeArtifactKind::Primary && RenderTextureArtifact::HasCurrentHeader(header);
    if (type == ResourceType::Texture)
        return kind == ImportArtifact::RuntimeArtifactKind::Primary && TextureArtifact::HasCurrentHeader(header);
    if (type == ResourceType::Mesh)
        return kind == ImportArtifact::RuntimeArtifactKind::Primary ? MeshArtifact::HasCurrentHeader(header)
                                                                    : SkinnedMeshArtifact::HasCurrentHeader(header);
    return false;
}

bool HasReusableRuntimeArtifact(const AssetIndexEntry &entry, ResourceType type,
                                const std::filesystem::path &projectRoot)
{
    if (entry.contentHash.empty())
        return false;
    const std::string expected = RuntimeArtifactRelativePath(entry.guid, type);
    if (expected.empty())
        return entry.artifactPath.empty();
    if (type == ResourceType::ParticleGraph) {
        // Python publishes the GUID-addressed AOT after import. Native refresh
        // stays reusable before that file exists, then requires the current
        // path once AssetIndex records it.
        if (entry.artifactPath.empty())
            return true;
        if (entry.artifactPath != expected)
            return false;
        std::error_code error;
        const auto primaryPath = projectRoot / std::filesystem::u8path(expected);
        return std::filesystem::is_regular_file(primaryPath, error) && !error;
    }
    if (entry.artifactPath != expected)
        return false;
    std::error_code error;
    const auto primaryPath = projectRoot / std::filesystem::u8path(expected);
    if (!std::filesystem::is_regular_file(primaryPath, error) || error ||
        !HasCurrentRuntimeArtifactHeader(primaryPath, type, ImportArtifact::RuntimeArtifactKind::Primary))
        return false;
    if (type != ResourceType::Mesh)
        return true;
    const std::string skinned =
        RuntimeArtifactRelativePath(entry.guid, ResourceType::Mesh, ImportArtifact::RuntimeArtifactKind::SkinnedMesh);
    error.clear();
    const auto skinnedPath = projectRoot / std::filesystem::u8path(skinned);
    if (!std::filesystem::is_regular_file(skinnedPath, error) || error ||
        !HasCurrentRuntimeArtifactHeader(skinnedPath, type, ImportArtifact::RuntimeArtifactKind::SkinnedMesh))
        return false;
    // Embedded textures have no source file to scan. Their owner is the only
    // importer capable of rebuilding a missing or obsolete child artifact.
    if (entry.metadata.HasKey("model_textures")) {
        const auto textures = nlohmann::json::parse(entry.metadata.GetDataAs<std::string>("model_textures"));
        for (const auto &texture : textures) {
            const auto path = projectRoot / std::filesystem::u8path(RuntimeArtifactRelativePath(
                                                texture.at("guid").get<std::string>(), ResourceType::Texture));
            if (!HasCurrentRuntimeArtifactHeader(path, ResourceType::Texture,
                                                 ImportArtifact::RuntimeArtifactKind::Primary))
                return false;
        }
    }
    return true;
}

size_t CountSourceAssets(const AssetIndex &index)
{
    return static_cast<size_t>(std::count_if(index.Entries().begin(), index.Entries().end(), [](const auto &item) {
        return !item.second.metadata.HasKey("import_owner_guid");
    }));
}

std::vector<DocumentTransactionEntry>
TakeRuntimeArtifactWrites(std::vector<ImportArtifact::RuntimeCpuArtifact> &artifacts, const std::string &guid,
                          ResourceType requestedType, const std::string &projectRoot)
{
    bool hasPrimary = false;
    bool hasSkinnedMesh = false;
    std::unordered_set<std::string> ownedTextures;
    std::vector<DocumentTransactionEntry> writes;
    writes.reserve(artifacts.size());
    for (auto &artifact : artifacts) {
        if (!artifact.guid.empty()) {
            if (requestedType != ResourceType::Mesh || artifact.resourceType != ResourceType::Texture ||
                artifact.kind != ImportArtifact::RuntimeArtifactKind::Primary || artifact.bytes.empty() ||
                !IsCanonicalAssetGuid(artifact.guid) || artifact.guid == guid ||
                !ownedTextures.insert(artifact.guid).second)
                throw std::logic_error("Importer produced an invalid owned Texture artifact");
            writes.push_back({FromFsPath(ToFsPath(projectRoot) / std::filesystem::u8path(RuntimeArtifactRelativePath(
                                                                     artifact.guid, ResourceType::Texture))),
                              std::move(artifact.bytes)});
            continue;
        }
        if (artifact.resourceType != requestedType || artifact.bytes.empty())
            throw std::logic_error("Importer produced an invalid runtime CPU artifact");
        switch (artifact.kind) {
        case ImportArtifact::RuntimeArtifactKind::Primary:
            if (hasPrimary)
                throw std::logic_error("Importer produced duplicate primary runtime CPU artifacts");
            hasPrimary = true;
            break;
        case ImportArtifact::RuntimeArtifactKind::SkinnedMesh:
            if (requestedType != ResourceType::Mesh || hasSkinnedMesh)
                throw std::logic_error("Importer produced an invalid duplicate skinned Mesh artifact");
            hasSkinnedMesh = true;
            break;
        }
        const std::string relative = RuntimeArtifactRelativePath(guid, requestedType, artifact.kind);
        writes.push_back(
            {FromFsPath(ToFsPath(projectRoot) / std::filesystem::u8path(relative)), std::move(artifact.bytes)});
    }
    if (RequiresRuntimeCpuArtifact(requestedType) && !hasPrimary)
        throw std::logic_error("Importer completed without its required primary runtime CPU artifact");
    if (requestedType == ResourceType::Mesh && !hasSkinnedMesh)
        throw std::logic_error("Mesh importer completed without its required skinned companion artifact");
    return writes;
}

bool HasOwnerMergeBudget(const FrameWorkOwner *owner, const std::chrono::steady_clock::time_point started,
                         size_t processed)
{
    return processed < kOwnerMergeEntryBudget &&
           (processed == 0 || (std::chrono::steady_clock::now() - started < kOwnerMergeTimeBudget &&
                               (!owner || !owner->IsActive() || owner->Remaining(true) > 0)));
}

template <typename Callback> class ScopeExit final
{
  public:
    explicit ScopeExit(Callback callback) : m_callback(std::move(callback))
    {
    }
    ScopeExit(const ScopeExit &) = delete;
    ScopeExit &operator=(const ScopeExit &) = delete;
    ~ScopeExit()
    {
        if (m_active)
            m_callback();
    }
    void Release() noexcept
    {
        m_active = false;
    }

  private:
    Callback m_callback;
    bool m_active = true;
};

template <typename Callback> ScopeExit<Callback> MakeScopeExit(Callback callback)
{
    return ScopeExit<Callback>(std::move(callback));
}

uint64_t Fnv1a64(const std::string &value, uint64_t seed)
{
    uint64_t hash = seed;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string MakeReadOnlyGuid(const std::string &identityKey)
{
    const std::string namespaced = "infernux-builtin:" + identityKey;
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << Fnv1a64(namespaced, 14695981039346656037ULL)
           << std::setw(16) << Fnv1a64(namespaced, 1099511628211ULL);
    return stream.str();
}

bool ReadFingerprint(const std::filesystem::path &path, AssetFileFingerprint &out)
{
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error)
        return false;
    const uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size > std::numeric_limits<uint64_t>::max())
        return false;
    const auto modified = std::filesystem::last_write_time(path, error);
    if (error)
        return false;
    out.size = static_cast<uint64_t>(size);
    out.modifiedNs = static_cast<int64_t>(modified.time_since_epoch().count());
    return true;
}

bool ReadFingerprint(const std::filesystem::directory_entry &entry, AssetFileFingerprint &out)
{
    std::error_code error;
    const uintmax_t size = entry.file_size(error);
    if (error || size > std::numeric_limits<uint64_t>::max())
        return false;
    const auto modified = entry.last_write_time(error);
    if (error)
        return false;
    out.size = static_cast<uint64_t>(size);
    out.modifiedNs = static_cast<int64_t>(modified.time_since_epoch().count());
    return true;
}

void RequireUnchangedFingerprint(const std::string &path, const AssetFileFingerprint &expected)
{
    AssetFileFingerprint current;
    if (!ReadFingerprint(ToFsPath(path), current) || current != expected)
        throw std::runtime_error("asset source changed during asynchronous refresh: " + path);
}

} // namespace

const std::vector<AssetCatalogEntry> &AssetCatalogSnapshot::GetDirectory(const std::string &normalizedDirectory) const
{
    static const std::vector<AssetCatalogEntry> empty;
    const auto found = m_directories.find(normalizedDirectory);
    return found != m_directories.end() ? found->second : empty;
}

AssetDatabase::AssetDatabase()
{
    m_querySnapshot = std::make_shared<const QuerySnapshot>();
    // Loaders are populated later by AssetRegistry::PopulateAssetDatabaseLoaders()
    // after all IAssetLoader plug-ins have been registered.
}

AssetDatabase::~AssetDatabase()
{
    WaitForCatalogWork(m_refreshWork);
    WaitForCatalogWork(m_applyWork);
    WaitForCatalogWork(m_runtimeCatalogWork);
}

AssetRegistry *AssetDatabase::BoundRegistry() const
{
    auto &registry = AssetRegistry::Instance();
    return registry.GetAssetDatabase() == this ? &registry : nullptr;
}

std::string AssetDatabase::GetRuntimeArtifactPath(const std::string &guid, ResourceType type) const
{
    const std::string relative = RuntimeArtifactRelativePath(guid, type);
    if (relative.empty() || m_projectRoot.empty())
        return {};
    return FromFsPath(ToFsPath(RuntimeArtifactRoot(guid)) / std::filesystem::u8path(relative));
}

std::string AssetDatabase::GetSkinnedMeshArtifactPath(const std::string &guid) const
{
    const std::string relative =
        RuntimeArtifactRelativePath(guid, ResourceType::Mesh, ImportArtifact::RuntimeArtifactKind::SkinnedMesh);
    if (m_projectRoot.empty())
        return {};
    return FromFsPath(ToFsPath(RuntimeArtifactRoot(guid)) / std::filesystem::u8path(relative));
}

std::string AssetDatabase::RuntimeArtifactRoot(const std::string &guid) const
{
    if (m_cookedRuntimeCatalogInstalled) {
        const auto snapshot = LoadQuerySnapshot();
        if (const auto &roots = snapshot->runtimeArtifactRoots) {
            const auto root = roots->find(guid);
            if (root != roots->end())
                return root->second;
        }
    }
    return m_projectRoot;
}

bool AssetDatabase::HasCurrentMeshArtifacts(const std::string &guid) const
{
    const auto primary = ToFsPath(GetRuntimeArtifactPath(guid, ResourceType::Mesh));
    const auto skinned = ToFsPath(GetSkinnedMeshArtifactPath(guid));
    return HasCurrentRuntimeArtifactHeader(primary, ResourceType::Mesh, ImportArtifact::RuntimeArtifactKind::Primary) &&
           HasCurrentRuntimeArtifactHeader(skinned, ResourceType::Mesh,
                                           ImportArtifact::RuntimeArtifactKind::SkinnedMesh);
}

bool AssetDatabase::IsModelReimportPending() const noexcept
{
    return m_pendingModelReimport && m_pendingModelReimport->phase != PendingModelReimport::Phase::Complete;
}

bool AssetDatabase::IsModelReimportPublishing() const noexcept
{
    return m_pendingModelReimport && (m_pendingModelReimport->phase == PendingModelReimport::Phase::Write ||
                                      m_pendingModelReimport->phase == PendingModelReimport::Phase::Publish);
}

bool AssetDatabase::IsDeliveringModelReimport() const noexcept
{
    return m_pendingModelReimport && m_pendingModelReimport->phase == PendingModelReimport::Phase::Publish;
}

bool AssetDatabase::IsStagingRuntimeArtifact(const std::string &guid, ResourceType type) const noexcept
{
    if (type == ResourceType::Mesh && m_pendingRefreshCommit &&
        m_pendingRefreshCommit->phase != PendingRefreshCommit::Phase::ReadyToFinalize)
        return true;
    return IsModelReimportPublishing() && !IsDeliveringModelReimport() &&
           m_pendingModelReimport->artifactGuids.count(guid) != 0;
}

void AssetDatabase::CompleteRuntimeArtifactPublication()
{
    AssertMutationThread("CompleteRuntimeArtifactPublication");
    if (IsModelReimportPublishing())
        CompletePendingModelReimport();
    else if (IsRefreshPending())
        CompletePendingRefresh();
}

void AssetDatabase::AssertMutationThread(const char *operation) const
{
    if (!m_initialized)
        throw std::logic_error(std::string("AssetDatabase::") + operation + " requires Initialize()");
    if (std::this_thread::get_id() != m_ownerThread)
        throw std::logic_error(std::string("AssetDatabase::") + operation + " must run on its owner thread");
}

void AssetDatabase::AssertNoPendingCommit(const char *operation) const
{
    if (m_pendingRefreshCommit)
        throw std::logic_error(std::string("AssetDatabase::") + operation +
                               " is unavailable while a refresh commit is pending");
    if (IsModelReimportPublishing())
        throw std::logic_error(std::string("AssetDatabase::") + operation +
                               " is unavailable while a model Apply commit is pending");
}

bool AssetDatabase::IsOwnerThread() const
{
    return m_initialized && std::this_thread::get_id() == m_ownerThread;
}

bool AssetDatabase::CanReadWorkingSet() const
{
    return IsOwnerThread() && !m_pendingRefreshCommit;
}

AssetDatabase::WorkingSet AssetDatabase::TakeWorkingSet()
{
    WorkingSet result;
    result.guidToPath.swap(m_guidToPath);
    result.pathToGuid.swap(m_pathToGuid);
    result.metas.swap(m_metas);
    result.fileStates.swap(m_fileStates);
    result.importResults.swap(m_importResults);
    result.assetIndex = std::move(m_assetIndex);
    m_assetIndex = AssetIndex{};
    result.assetIndexDirty = m_assetIndexDirty;
    m_assetIndexDirty = false;
    return result;
}

void AssetDatabase::InstallWorkingSet(WorkingSet workingSet)
{
    m_guidToPath = std::move(workingSet.guidToPath);
    m_pathToGuid = std::move(workingSet.pathToGuid);
    m_metas = std::move(workingSet.metas);
    m_fileStates = std::move(workingSet.fileStates);
    m_importResults = std::move(workingSet.importResults);
    m_assetIndex = std::move(workingSet.assetIndex);
    m_assetIndexDirty = workingSet.assetIndexDirty;
}

std::shared_ptr<const AssetDatabase::QuerySnapshot> AssetDatabase::LoadQuerySnapshot() const
{
    return std::atomic_load_explicit(&m_querySnapshot, std::memory_order_acquire);
}

void AssetDatabase::PublishQuerySnapshot(bool includeCatalog)
{
    ExpandModelSubAssets(m_guidToPath, m_pathToGuid, m_metas, m_fileStates, m_importResults);
    InstallQuerySnapshot(BuildQuerySnapshotArtifact(m_guidToPath, m_pathToGuid, m_metas, m_fileStates, m_importResults,
                                                    m_queryGeneration + 1, includeCatalog));
}

void AssetDatabase::PublishModelSubAssetEvents(const std::shared_ptr<const QuerySnapshot> &previous,
                                               const std::unordered_set<std::string> *changedGuids)
{
    auto *registry = BoundRegistry();
    auto &graph = *m_dependencyGraph;
    const auto publish = [&](const std::string &guid) {
        const auto current = m_metas.find(guid);
        if (current != m_metas.end() && current->second->HasKey("import_document")) {
            const auto owner = current->second->GetDataAs<std::string>("import_owner_guid");
            if (!graph.HasDependency(guid, owner))
                graph.SetAssetDependencies(guid, {owner});
        }
        if (!previous)
            return;
        const auto old = previous->metas.find(guid);
        if (old == previous->metas.end()) {
            if (current != m_metas.end() && current->second->HasKey("import_owner_guid"))
                graph.NotifyEvent(guid, current->second->GetResourceType(), AssetEvent::Modified);
            return;
        }
        const auto &metadata = old->second;
        if (!metadata->HasKey("import_owner_guid"))
            return;
        if (current == m_metas.end()) {
            if (registry)
                registry->RemoveAsset(guid);
            graph.NotifyEvent(guid, metadata->GetResourceType(), AssetEvent::Deleted);
            graph.RemoveAsset(guid);
        } else if (metadata != current->second) {
            auto before = metadata->SerializeDocument();
            auto after = current->second->SerializeDocument();
            if (before == after)
                return;
            const bool moved = previous->guidToPath.at(guid) != m_guidToPath.at(guid);
            if (moved) {
                before["metadata"].erase("file_path");
                after["metadata"].erase("file_path");
            }
            const auto event = moved && before == after ? AssetEvent::Moved : AssetEvent::Modified;
            if (registry) {
                registry->UpdateLoadedAssetPath(guid, m_guidToPath.at(guid));
                if (event == AssetEvent::Modified && current->second->GetResourceType() == ResourceType::Texture)
                    registry->ReloadAsset(guid);
            }
            graph.NotifyEvent(guid, metadata->GetResourceType(), event);
        }
    };
    if (changedGuids) {
        for (const auto &guid : *changedGuids)
            publish(guid);
        return;
    }
    for (const auto &[guid, metadata] : m_metas)
        if (metadata->HasKey("import_owner_guid"))
            publish(guid);
    if (previous)
        for (const auto &[guid, metadata] : previous->metas)
            if (metadata->HasKey("import_owner_guid") && m_metas.find(guid) == m_metas.end())
                publish(guid);
}

void AssetDatabase::UpdateModelSubAssets(const std::string &owner, const QuerySnapshot &previous,
                                         QuerySnapshot &snapshot, std::unordered_set<std::string> &changedGuids)
{
    std::unordered_map<std::string, std::shared_ptr<InxResourceMeta>> children;
    std::string sourcePath;
    if (const auto current = m_metas.find(owner); current != m_metas.end()) {
        sourcePath = ResolveFilesystemPath(m_guidToPath.at(owner));
        VisitModelSubAssets(*current->second, [&](const nlohmann::json &record) {
            auto metadata = ModelSubAssetMetadata(record, owner, sourcePath);
            const auto guid = metadata->GetGuid();
            const auto existing = m_metas.find(guid);
            if ((existing != m_metas.end() &&
                 (!existing->second->HasKey("import_owner_guid") ||
                  existing->second->GetDataAs<std::string>("import_owner_guid") != owner)) ||
                !children.emplace(guid, metadata).second)
                throw std::invalid_argument("model sub-asset identity collides with another asset: " + guid);
            if (existing != m_metas.end() && existing->second->SerializeDocument() == metadata->SerializeDocument())
                children.at(guid) = existing->second;
        });
    }
    if (const auto old = previous.metas.find(owner); old != previous.metas.end()) {
        VisitModelSubAssets(*old->second, [&](const nlohmann::json &record) {
            const auto guid = record.at("guid").get<std::string>();
            const auto key = IndexedAssetPathKey(previous.guidToPath.at(guid));
            if (const auto mapping = m_pathToGuid.find(key); mapping != m_pathToGuid.end() && mapping->second == guid) {
                m_pathToGuid.erase(mapping);
                m_fileStates.erase(key);
            }
            if (const auto mapping = snapshot.pathToGuid.find(key);
                mapping != snapshot.pathToGuid.end() && mapping->second == guid)
                snapshot.pathToGuid.erase(mapping);
            if (children.find(guid) == children.end()) {
                m_guidToPath.erase(guid);
                m_metas.erase(guid);
                m_importResults.erase(guid);
                snapshot.guidToPath.erase(guid);
                snapshot.metas.erase(guid);
            }
            changedGuids.insert(guid);
        });
    }
    if (children.empty())
        return;
    auto state = m_fileStates.at(IndexedAssetPathKey(sourcePath));
    state.readOnly = true;
    for (auto &[guid, metadata] : children) {
        const auto path = metadata->GetDataAs<std::string>("file_path");
        const auto key = IndexedAssetPathKey(path);
        m_guidToPath[guid] = path;
        m_pathToGuid[key] = guid;
        m_metas[guid] = metadata;
        m_fileStates[key] = state;
        if (const auto result = m_importResults.find(owner); result != m_importResults.end())
            m_importResults[guid] = result->second;
        snapshot.guidToPath[guid] = path;
        snapshot.pathToGuid[key] = guid;
        snapshot.metas[guid] = metadata;
        changedGuids.insert(guid);
    }
}

void AssetDatabase::PublishRefreshAssetEvents(const WorkingSet &previous)
{
    struct Change
    {
        std::string guid;
        ResourceType type;
        AssetEvent event;
        bool moved = false;
    };
    std::vector<Change> changes;
    for (const auto &[guid, metadata] : previous.metas) {
        if (!metadata->HasKey("import_owner_guid") && m_metas.find(guid) == m_metas.end())
            changes.push_back({guid, metadata->GetResourceType(), AssetEvent::Deleted});
    }
    for (const auto &[guid, metadata] : m_metas) {
        if (metadata->HasKey("import_owner_guid"))
            continue; // Model-owned resources have their own publication below.
        const auto result = m_importResults.find(guid);
        if (result != m_importResults.end() && !result->second.succeeded)
            continue; // An unsuccessful import does not replace the last valid runtime payload.
        const auto old = previous.metas.find(guid);
        if (old == previous.metas.end()) {
            changes.push_back({guid, metadata->GetResourceType(), AssetEvent::Modified});
            continue;
        }
        const auto &path = m_guidToPath.at(guid);
        const auto &oldPath = previous.guidToPath.at(guid);
        const bool moved = path != oldPath;
        const auto oldFile = previous.fileStates.find(FilesystemPathKey(oldPath));
        const auto file = m_fileStates.find(FilesystemPathKey(path));
        const auto oldResult = previous.importResults.find(guid);
        bool modified = oldFile == previous.fileStates.end() || file == m_fileStates.end() ||
                        oldFile->second.source != file->second.source ||
                        (oldResult != previous.importResults.end() && !oldResult->second.succeeded);
        if (!modified && old->second != metadata) {
            // Source observations live in the local index; only authored settings
            // can change the runtime contract without a source change. A path-only
            // move must retain the payload and its published runtime version.
            auto oldSettings = old->second->SerializeDocumentPortable(m_projectRoot);
            auto settings = metadata->SerializeDocumentPortable(m_projectRoot);
            oldSettings["metadata"].erase("file_path");
            settings["metadata"].erase("file_path");
            modified = oldSettings != settings;
        }
        if (modified || moved)
            changes.push_back(
                {guid, metadata->GetResourceType(), modified ? AssetEvent::Modified : AssetEvent::Moved, moved});
    }
    std::sort(changes.begin(), changes.end(),
              [](const Change &left, const Change &right) { return left.guid < right.guid; });
    auto *registry = BoundRegistry();
    // Install every CPU cache change before notifying consumers, which may load
    // other changed assets through the already-published catalog.
    if (registry) {
        for (const auto &change : changes) {
            if (change.event == AssetEvent::Deleted) {
                registry->RemoveAsset(change.guid);
                continue;
            }
            if (change.moved)
                registry->UpdateLoadedAssetPath(change.guid, m_guidToPath.at(change.guid));
            if (change.event == AssetEvent::Modified && change.type != ResourceType::Mesh) {
                const bool loaded = registry->IsLoaded(change.guid);
                if (!registry->ReloadAsset(change.guid) && loaded)
                    throw std::runtime_error("Could not publish refreshed runtime asset: " + change.guid);
            }
        }
    }
    auto &graph = *m_dependencyGraph;
    for (const auto &change : changes) {
        if (registry && change.event == AssetEvent::Modified && change.type == ResourceType::Mesh &&
            registry->NotifySourceImported(change.guid, change.type))
            continue;
        graph.NotifyEvent(change.guid, change.type, change.event);
        if (change.event == AssetEvent::Deleted)
            graph.RemoveAsset(change.guid);
    }
}

std::unordered_set<std::string> AssetDatabase::PublishQuerySnapshotForPaths(const std::vector<std::string> &paths)
{
    const auto previous = LoadQuerySnapshot();
    if (!previous || !previous->catalog || paths.empty()) {
        PublishQuerySnapshot();
        std::unordered_set<std::string> changed;
        for (const auto &[guid, metadata] : m_metas)
            if (metadata->HasKey("import_owner_guid"))
                changed.insert(guid);
        if (previous)
            for (const auto &[guid, metadata] : previous->metas)
                if (metadata->HasKey("import_owner_guid"))
                    changed.insert(guid);
        return changed;
    }

    auto snapshot = std::make_shared<QuerySnapshot>(*previous);
    snapshot->generation = m_queryGeneration + 1;
    auto catalog = std::make_shared<AssetCatalogSnapshot>(*previous->catalog);
    catalog->m_generation = snapshot->generation;

    std::unordered_set<std::string> touchedDirectories;
    std::unordered_set<std::string> touchedPaths;
    std::unordered_set<std::string> touchedGuids;
    std::unordered_set<std::string> modelOwners;
    touchedPaths.reserve(paths.size());
    for (const std::string &path : paths) {
        const std::string normalizedPath = FilesystemPathKey(path);
        if (normalizedPath.empty() || !touchedPaths.insert(normalizedPath).second)
            continue;

        const size_t separator = normalizedPath.find_last_of('/');
        touchedDirectories.insert(separator == std::string::npos ? std::string{} : normalizedPath.substr(0, separator));

        const auto oldMapping = snapshot->pathToGuid.find(normalizedPath);
        if (oldMapping != snapshot->pathToGuid.end()) {
            const std::string oldGuid = oldMapping->second;
            touchedGuids.insert(oldGuid);
            const auto &oldMetadata = snapshot->metas.at(oldGuid);
            if (oldMetadata->HasKey("model_textures") || oldMetadata->HasKey("model_animations"))
                modelOwners.insert(oldGuid);
            snapshot->pathToGuid.erase(oldMapping);
            if (m_guidToPath.find(oldGuid) == m_guidToPath.end()) {
                snapshot->guidToPath.erase(oldGuid);
                snapshot->metas.erase(oldGuid);
                snapshot->importErrors.erase(oldGuid);
            }
        }

        const auto currentMapping = m_pathToGuid.find(normalizedPath);
        if (currentMapping == m_pathToGuid.end())
            continue;
        const std::string &guid = currentMapping->second;
        touchedGuids.insert(guid);
        const auto currentPath = m_guidToPath.find(guid);
        const auto metadata = m_metas.find(guid);
        if (currentPath == m_guidToPath.end() || metadata == m_metas.end() || !metadata->second)
            throw std::logic_error("Incremental query publication found an incomplete asset mapping");
        snapshot->pathToGuid[normalizedPath] = guid;
        snapshot->guidToPath[guid] = currentPath->second;
        snapshot->metas[guid] = metadata->second;
        snapshot->importErrors.erase(guid);
        const auto importResult = m_importResults.find(guid);
        if (importResult != m_importResults.end() && !importResult->second.succeeded)
            snapshot->importErrors[guid] = importResult->second.error;
        if (metadata->second->HasKey("model_textures") || metadata->second->HasKey("model_animations"))
            modelOwners.insert(guid);
    }

    std::unordered_set<std::string> changedSubAssets;
    for (const auto &owner : modelOwners)
        UpdateModelSubAssets(owner, *previous, *snapshot, changedSubAssets);

    for (const std::string &directory : touchedDirectories) {
        auto &entries = catalog->m_directories[directory];
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [&touchedGuids](const AssetCatalogEntry &entry) {
                                         return touchedGuids.find(entry.guid) != touchedGuids.end();
                                     }),
                      entries.end());
    }

    for (const std::string &normalizedPath : touchedPaths) {
        const auto mapping = m_pathToGuid.find(normalizedPath);
        if (mapping == m_pathToGuid.end())
            continue;
        const std::string &guid = mapping->second;
        const auto currentPath = m_guidToPath.find(guid);
        const auto metadata = m_metas.find(guid);
        const auto fileState = m_fileStates.find(normalizedPath);
        if (currentPath == m_guidToPath.end() || metadata == m_metas.end() || !metadata->second ||
            fileState == m_fileStates.end())
            throw std::logic_error("Incremental catalog publication found an incomplete asset mapping");

        AssetCatalogEntry entry;
        entry.guid = guid;
        entry.path = currentPath->second;
        entry.name = FromFsPath(ToFsPath(entry.path).filename());
        entry.sortKey = entry.name;
        for (char &character : entry.sortKey) {
            if (character >= 'A' && character <= 'Z')
                character = static_cast<char>(character + ('a' - 'A'));
        }
        entry.resourceType = metadata->second->GetResourceType();
        entry.source = fileState->second.source;
        const auto importError = snapshot->importErrors.find(guid);
        if (importError != snapshot->importErrors.end())
            entry.importError = importError->second;
        const size_t separator = normalizedPath.find_last_of('/');
        const std::string parent = separator == std::string::npos ? std::string{} : normalizedPath.substr(0, separator);
        catalog->m_directories[parent].push_back(std::move(entry));
        touchedDirectories.insert(parent);
    }

    for (const std::string &directory : touchedDirectories) {
        auto found = catalog->m_directories.find(directory);
        if (found == catalog->m_directories.end())
            continue;
        auto &entries = found->second;
        if (entries.empty()) {
            catalog->m_directories.erase(found);
            continue;
        }
        std::sort(entries.begin(), entries.end(), [](const AssetCatalogEntry &left, const AssetCatalogEntry &right) {
            if (left.sortKey != right.sortKey)
                return left.sortKey < right.sortKey;
            return left.path < right.path;
        });
    }

    snapshot->catalog = std::move(catalog);
    BuildShaderPathIndex(*snapshot);
    InstallQuerySnapshot(std::move(snapshot));
    return changedSubAssets;
}

void AssetDatabase::InstallQuerySnapshot(std::shared_ptr<QuerySnapshot> snapshot) noexcept
{
    if (m_cookedRuntimeCatalogInstalled && !snapshot->runtimeArtifactRoots)
        snapshot->runtimeArtifactRoots = LoadQuerySnapshot()->runtimeArtifactRoots;
    m_queryGeneration = snapshot->generation;
    std::atomic_store_explicit(&m_querySnapshot, std::shared_ptr<const QuerySnapshot>(std::move(snapshot)),
                               std::memory_order_release);
}

void AssetDatabase::ConfigureBlenderImport(const std::string &executable, const std::string &exportScript)
{
    if (m_initialized) {
        AssertMutationThread("ConfigureBlenderImport");
        AssertNoPendingCommit("ConfigureBlenderImport");
    }
    if (executable.empty() != exportScript.empty())
        throw std::invalid_argument("Blender import requires both an executable and an export script");
    for (const auto &path : {executable, exportScript})
        if (!path.empty() && !ToFsPath(path).is_absolute())
            throw std::invalid_argument("Blender import tool requires an absolute path: " + path);
    m_blenderExecutable = executable;
    m_blenderExportScript = exportScript;
}

void AssetDatabase::Initialize(const std::string &projectRoot)
{
    if (m_initialized)
        throw std::logic_error("AssetDatabase::Initialize may only be called once");
    m_projectRoot = ResolveFilesystemPath(projectRoot);
    if (m_projectRoot.empty())
        throw std::invalid_argument("AssetDatabase project root cannot be empty");
    const std::filesystem::path assetsPath = ToFsPath(m_projectRoot) / "Assets";
    std::error_code directoryError;
    std::filesystem::create_directories(assetsPath, directoryError);
    if (directoryError)
        throw std::runtime_error("Failed to create project Assets directory: " + directoryError.message());
    m_assetsRoot = FromFsPath(assetsPath);
    m_assetIndexPath = FromFsPath(ToFsPath(m_projectRoot) / "Library" / "AssetIndex.json");
    m_assetStartupCachePath = FromFsPath(ToFsPath(m_projectRoot) / "Library" / "AssetIndex.startup-cache.json");
    m_assetTransactionJournalPath = FromFsPath(ToFsPath(m_projectRoot) / "Library" / "AssetRefresh.transaction");
    if (DocumentTransaction::Recover(m_projectRoot, m_assetTransactionJournalPath))
        INXLOG_WARN("AssetDatabase recovered an interrupted metadata transaction");

    // Cooked Player hosts never import mutable authoring sources.
#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    m_importerRegistry.Register(std::make_unique<TextureImporter>());
    m_importerRegistry.Register(std::make_unique<ShaderImporter>());
    m_importerRegistry.Register(std::make_unique<MaterialImporter>());
    m_importerRegistry.Register(std::make_unique<PhysicMaterialImporter>());
    m_importerRegistry.Register(std::make_unique<RenderEffectImporter>());
    m_importerRegistry.Register(std::make_unique<ParticleGraphImporter>());
    m_importerRegistry.Register(std::make_unique<DataAssetImporter>());
    m_importerRegistry.Register(std::make_unique<RenderTextureImporter>());
    m_importerRegistry.Register(std::make_unique<ScriptImporter>());
    m_importerRegistry.Register(std::make_unique<AudioImporter>());
    m_importerRegistry.Register(std::make_unique<ModelImporter>());
    m_importerRegistry.Register(std::make_unique<PrefabImporter>());
#endif

    m_ownerThread = std::this_thread::get_id();
    m_initialized = true;
    PublishQuerySnapshot();
}

void AssetDatabase::InitializeRuntime(const std::string &projectRoot)
{
    if (m_initialized)
        throw std::logic_error("AssetDatabase::InitializeRuntime may only be called once");
    m_projectRoot = ResolveFilesystemPath(projectRoot);
    if (m_projectRoot.empty())
        throw std::invalid_argument("AssetDatabase runtime project root cannot be empty");
    m_assetsRoot = FromFsPath(ToFsPath(m_projectRoot) / "Assets");
    m_assetIndexPath = FromFsPath(ToFsPath(m_projectRoot) / "Library" / "AssetIndex.json");
    m_assetStartupCachePath = FromFsPath(ToFsPath(m_projectRoot) / "Library" / "AssetIndex.startup-cache.json");
    m_assetTransactionJournalPath = FromFsPath(ToFsPath(m_projectRoot) / "Library" / "AssetRefresh.transaction");
    m_ownerThread = std::this_thread::get_id();
    m_initialized = true;
    InstallQuerySnapshot(BuildQuerySnapshotArtifact(m_guidToPath, m_pathToGuid, m_metas, m_fileStates, m_importResults,
                                                    m_queryGeneration + 1, false));
}

void AssetDatabase::AddScanRoot(const std::string &path)
{
    AssertMutationThread("AddScanRoot");
    AssertNoPendingCommit("AddScanRoot");
    if (m_pendingAssetScan)
        throw std::logic_error("AssetDatabase scan roots cannot change during an asynchronous refresh");
    auto norm = ResolveFilesystemPath(path);
    for (const auto &existing : m_extraScanRoots) {
        if (existing == norm)
            return;
    }
    m_extraScanRoots.push_back(std::move(norm));
}

void AssetDatabase::AddReadOnlyScanRoot(const std::string &path)
{
    AssertMutationThread("AddReadOnlyScanRoot");
    AddScanRoot(path);
    m_readOnlyScanRoots.insert(ResolveFilesystemPath(path));
}

bool AssetDatabase::RestoreCachedCatalog()
{
    AssertMutationThread("RestoreCachedCatalog");
    AssertNoPendingCommit("RestoreCachedCatalog");
    if (m_pendingAssetScan)
        throw std::logic_error("AssetDatabase cached catalog cannot be restored during an asynchronous refresh");

    const bool profileStartup = [] {
        const char *value = std::getenv("INFERNUX_PROFILE_STARTUP");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    const auto restoreStarted = std::chrono::steady_clock::now();
    auto previousPhase = restoreStarted;
    auto reportPhase = [&](const char *name) {
        if (!profileStartup)
            return;
        const auto now = std::chrono::steady_clock::now();
        INXLOG_INFO("AssetDatabase cached restore ", name, "=",
                    std::chrono::duration<double, std::milli>(now - previousPhase).count(), "ms");
        previousPhase = now;
    };

    AssetIndex cached;
    try {
        if (!cached.Load(m_assetStartupCachePath, FilesystemPathKey(m_projectRoot))) {
            INXLOG_INFO("AssetDatabase: no committed startup catalog is available");
            return false;
        }
    } catch (const std::exception &exception) {
        INXLOG_WARN("AssetDatabase: discarded cached startup catalog: ", exception.what());
        return false;
    }
    reportPhase("load_index");

    WorkingSet restored;
    restored.assetIndex = cached;
    restored.fileStates.reserve(cached.Size());
    std::unordered_map<std::string, std::vector<std::string>> dependencies;
    dependencies.reserve(cached.Size());

    for (const auto &[normalizedPath, entry] : cached.Entries()) {
        if (entry.guid.empty() || normalizedPath.empty())
            return false;

        std::string path = normalizedPath;
        if (entry.metadata.HasKey("file_path")) {
            try {
                const std::string indexedPath = entry.metadata.GetDataAs<std::string>("file_path");
                if (!indexedPath.empty())
                    path = indexedPath;
            } catch (const std::exception &) {
                return false;
            }
        }
        if (!HasCurrentBuiltinSceneIconMetadata(entry.metadata, path, entry.readOnly) ||
            !HasCurrentPluginPageTextureMetadata(entry.metadata, path, m_projectRoot))
            return false;

        restored.guidToPath.emplace(entry.guid, path);
        restored.pathToGuid.emplace(normalizedPath, entry.guid);
        restored.metas.emplace(entry.guid, std::make_shared<InxResourceMeta>(entry.metadata));
        restored.fileStates.emplace(normalizedPath, CachedFileState{entry.source, entry.meta, entry.readOnly});
        restored.importResults.emplace(entry.guid, ImportResultState{entry.importSucceeded, entry.importError});
        dependencies.emplace(entry.guid, entry.dependencies);
    }
    reportPhase("rebuild_working_set");

    const uint64_t queryGeneration = m_queryGeneration + 1;
    auto query = BuildQuerySnapshotArtifact(restored.guidToPath, restored.pathToGuid, restored.metas,
                                            restored.fileStates, restored.importResults, queryGeneration);
    reportPhase("build_query_snapshot");
    const uint64_t dependencyGeneration = m_dependencyGraph->GetAssetGeneration() + 1;
    auto dependencySnapshot = AssetDependencyGraph::BuildAssetSnapshot(dependencies, dependencyGeneration);
    reportPhase("build_dependency_snapshot");

    (void)TakeWorkingSet();
    InstallWorkingSet(std::move(restored));
    m_dependencyGraph->InstallAssetSnapshot(std::move(dependencySnapshot));
    InstallQuerySnapshot(std::move(query));
    reportPhase("publish");
    m_assetIndexDirty = false;
    INXLOG_INFO("AssetDatabase: restored ", cached.Size(), " cached editor asset identities from ",
                m_assetStartupCachePath);
    return true;
}

void AssetDatabase::InstallRuntimeAssetCatalog(const std::string &catalogPath, bool trustedPackage)
{
    AssertMutationThread("InstallRuntimeAssetCatalog");
    AssertNoPendingCommit("InstallRuntimeAssetCatalog");

    const bool profileStartup = [] {
        const char *value = std::getenv("INFERNUX_PROFILE_STARTUP");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    auto previousPhase = std::chrono::steady_clock::now();
    auto reportPhase = [&](const char *name) {
        if (!profileStartup)
            return;
        const auto now = std::chrono::steady_clock::now();
        INXLOG_INFO("AssetDatabase runtime catalog ", name, "=",
                    std::chrono::duration<double, std::milli>(now - previousPhase).count(), "ms");
        previousPhase = now;
    };

    RuntimeCatalogPreparation preparation{CaptureRuntimeWorkingSet(), {}, m_queryGeneration, {}, {}};
    const auto document = ReadRuntimeCatalogDocument(ToFsPath(catalogPath));
    reportPhase("parse");
    const auto entryCount = PrepareRuntimeCatalog(preparation, document, m_projectRoot, m_readOnlyScanRoots,
                                                  trustedPackage, nullptr, reportPhase);
    InstallPreparedRuntimeCatalog(preparation);
    m_cookedRuntimeCatalogInstalled = trustedPackage;
    reportPhase("publish_snapshot");
    INXLOG_INFO("AssetDatabase: installed ", entryCount, " cooked Player asset identities");
}

AssetDatabase::WorkingSet AssetDatabase::CaptureRuntimeWorkingSet() const
{
    return {m_guidToPath, m_pathToGuid, m_metas, m_fileStates, m_importResults, {}, false};
}

size_t AssetDatabase::PrepareRuntimeCatalog(RuntimeCatalogPreparation &preparation, const nlohmann::json &document,
                                            const std::string &contentRoot,
                                            const std::unordered_set<std::string> &builtinRoots, bool trustedPackage,
                                            const std::unordered_set<std::string> *newGuids,
                                            const std::function<void(const char *)> &reportPhase)
{
    if (document.value("$schema", std::string{}) != "infernux.runtime_asset_records")
        throw std::runtime_error("runtime asset catalog has an unsupported schema");
    const auto entries = document.find("entries");
    if (entries == document.end() || !entries->is_array())
        throw std::runtime_error("runtime asset catalog has no entry array");

    auto &working = preparation.workingSet;
    auto &guidToPath = working.guidToPath;
    auto &pathToGuid = working.pathToGuid;
    auto &metas = working.metas;
    auto &fileStates = working.fileStates;
    auto &importResults = working.importResults;
    const size_t count = newGuids ? newGuids->size() : entries->size();
    const size_t expectedIdentityCount = guidToPath.size() + count;
    guidToPath.reserve(expectedIdentityCount);
    pathToGuid.reserve(pathToGuid.size() + count);
    metas.reserve(metas.size() + count);
    importResults.reserve(importResults.size() + count);
    fileStates.reserve(fileStates.size() + count);

    const auto contentBase = ToFsPath(contentRoot);
    size_t prepared = 0;
    for (const auto &entry : *entries) {
        if (!entry.is_object())
            throw std::runtime_error("runtime asset catalog entry is not an object");
        const std::string guid = entry.value("guid", std::string{});
        if (newGuids && !newGuids->count(guid))
            continue;
        const std::string relativePath = entry.value("runtime_path", std::string{});
        const auto metadataDocument = entry.find("metadata");
        if (guid.empty() || relativePath.empty() || metadataDocument == entry.end() || !metadataDocument->is_object())
            throw std::runtime_error("runtime asset catalog entry identity is incomplete");

        const std::filesystem::path relative = std::filesystem::u8path(relativePath);
        if (relative.is_absolute() || relative.empty())
            throw std::runtime_error("runtime asset catalog path must be project-relative: " + relativePath);
        for (const auto &part : relative) {
            if (part == "..")
                throw std::runtime_error("runtime asset catalog path escapes the project: " + relativePath);
        }

        std::filesystem::path resolvedAssetPath = contentBase / relative;
        const auto runtimeArtifacts = entry.find("runtime_artifacts");
        if (runtimeArtifacts != entry.end() && runtimeArtifacts->is_array()) {
            const std::string primaryArtifactId = entry.value("primary_runtime_artifact_id", std::string{});
            constexpr std::string_view builtinPrefix = "infernux/resources/";
            bool resolvedPrimaryArtifact = false;
            for (const auto &artifact : *runtimeArtifacts) {
                if (!artifact.is_object())
                    continue;
                const std::string artifactId = artifact.value("runtime_artifact_id", std::string{});
                if (!primaryArtifactId.empty() && artifactId != primaryArtifactId)
                    continue;
                const std::string package = artifact.value("package", std::string{});
                const std::string runtimePath = artifact.value("runtime_path", std::string{});
                if (package == "Content.inxpkg") {
                    const std::filesystem::path contentRelative = std::filesystem::u8path(runtimePath);
                    if (contentRelative.empty() || contentRelative.is_absolute())
                        throw std::runtime_error("runtime Content asset path is invalid: " + runtimePath);
                    for (const auto &part : contentRelative) {
                        if (part == "..")
                            throw std::runtime_error("runtime Content asset path escapes the project: " + runtimePath);
                    }
                    const std::filesystem::path candidate = contentBase / contentRelative;
                    if (!trustedPackage) {
                        std::error_code fileError;
                        if (!std::filesystem::is_regular_file(candidate, fileError) || fileError)
                            throw std::runtime_error(
                                "runtime Content asset was not extracted below the project root: " + runtimePath);
                    }
                    resolvedAssetPath = candidate;
                    resolvedPrimaryArtifact = true;
                    break;
                }
                if (package != "Runtime.inxrt" || runtimePath.rfind(builtinPrefix, 0) != 0)
                    continue;

                const std::filesystem::path builtinRelative =
                    std::filesystem::u8path(runtimePath.substr(builtinPrefix.size()));
                if (builtinRelative.empty() || builtinRelative.is_absolute())
                    throw std::runtime_error("runtime built-in asset path is invalid: " + runtimePath);
                for (const auto &part : builtinRelative) {
                    if (part == "..")
                        throw std::runtime_error("runtime built-in asset path escapes its resource root: " +
                                                 runtimePath);
                }

                bool found = false;
                for (const auto &root : builtinRoots) {
                    const std::filesystem::path candidate = ToFsPath(root) / builtinRelative;
                    if (!trustedPackage) {
                        std::error_code fileError;
                        if (!std::filesystem::is_regular_file(candidate, fileError) || fileError)
                            continue;
                    }
                    resolvedAssetPath = candidate;
                    found = true;
                    break;
                }
                if (!found) {
                    throw std::runtime_error("runtime built-in asset was not extracted below a read-only resource "
                                             "root: " +
                                             runtimePath);
                }
                resolvedPrimaryArtifact = true;
                break;
            }
            if (!primaryArtifactId.empty() && !resolvedPrimaryArtifact)
                throw std::runtime_error("runtime asset catalog primary artifact is unavailable: " + primaryArtifactId);
        }

        const std::string path = FromFsPath(resolvedAssetPath.lexically_normal());
        // Cooked files are regular entries below already-resolved package
        // roots. Installing their identities requires no per-file disk probe.
        const std::string normalizedPath = trustedPackage ? LexicalFilesystemPathKey(path) : FilesystemPathKey(path);
        InxResourceMeta metadata;
        metadata.DeserializeDocument(*metadataDocument);
        if (metadata.GetGuid() != guid)
            throw std::runtime_error("runtime asset catalog metadata GUID mismatch: " + guid);
        if (trustedPackage)
            metadata.AddMetadata("file_path", path);
        else
            metadata.UpdateFilePath(path);
        metadata.AddMetadata("read_only", true);

        const auto oldPathGuid = pathToGuid.find(normalizedPath);
        const auto oldGuidPath = guidToPath.find(guid);
        if (newGuids && (oldPathGuid != pathToGuid.end() || oldGuidPath != guidToPath.end()))
            throw std::runtime_error("runtime content conflicts with a published asset identity: " + guid);
        if (oldPathGuid != pathToGuid.end() && oldPathGuid->second != guid) {
            guidToPath.erase(oldPathGuid->second);
            metas.erase(oldPathGuid->second);
            importResults.erase(oldPathGuid->second);
        }
        if (oldGuidPath != guidToPath.end()) {
            const std::string oldKey =
                trustedPackage ? LexicalFilesystemPathKey(oldGuidPath->second) : FilesystemPathKey(oldGuidPath->second);
            pathToGuid.erase(oldKey);
            fileStates.erase(oldKey);
        }

        guidToPath[guid] = path;
        pathToGuid[normalizedPath] = guid;
        metas[guid] = std::make_shared<InxResourceMeta>(std::move(metadata));
        importResults[guid] = {true, {}};

        CachedFileState fileState;
        fileState.readOnly = true;
        // The Player launcher validates the immutable package hash before it
        // exposes the extraction root. Re-statting and hashing every cooked
        // file here made startup proportional to project size and duplicated
        // that trust decision. Editor/test callers retain strict validation.
        if (!trustedPackage)
            (void)ReadFingerprint(ToFsPath(path), fileState.source);
        fileStates[normalizedPath] = fileState;
        ++prepared;
    }
    if (prepared != count)
        throw std::runtime_error("runtime content catalog is missing selected asset identities");
    if (reportPhase)
        reportPhase("prepare_entries");

    // Player has no Project/FileManager panel. Keep GUID/path/metadata queries,
    // but do not expand source-model children over cooked artifact identities.
    preparation.query = BuildQuerySnapshotArtifact(guidToPath, pathToGuid, metas, fileStates, importResults,
                                                   preparation.baseGeneration + 1, false);
    return prepared;
}

void AssetDatabase::InstallPreparedRuntimeCatalog(RuntimeCatalogPreparation &preparation) noexcept
{
    auto &working = preparation.workingSet;
    m_guidToPath.swap(working.guidToPath);
    m_pathToGuid.swap(working.pathToGuid);
    m_metas.swap(working.metas);
    m_importResults.swap(working.importResults);
    m_fileStates.swap(working.fileStates);
    InstallQuerySnapshot(std::move(preparation.query));
}

void AssetDatabase::BeginRuntimeAssetCatalog(const std::string &packageRoot, const std::string &contentRoot,
                                             const std::vector<std::string> &allowedRoots)
{
    AssertMutationThread("BeginRuntimeAssetCatalog");
    AssertNoPendingCommit("BeginRuntimeAssetCatalog");
    ObserveCatalogWork(m_runtimeCatalogWork);
    if (!m_cookedRuntimeCatalogInstalled || !m_frameWork || m_runtimeCatalog || !m_runtimeCatalogWork.Complete())
        throw std::logic_error("runtime content preparation requires an idle cooked Player catalog and a host");
    if (!ToFsPath(packageRoot).is_absolute() || !ToFsPath(contentRoot).is_absolute())
        throw std::invalid_argument("runtime content requires resolved package and output directories");

    auto preparation = std::make_shared<RuntimeCatalogPreparation>();
    preparation->workingSet = CaptureRuntimeWorkingSet();
    preparation->baseGeneration = m_queryGeneration;
    QueueCatalogJob(
        m_runtimeCatalogWork, CatalogIoStage::RuntimeCatalog,
        [preparation, packageRoot, contentRoot, allowedRoots, builtinRoots = m_readOnlyScanRoots,
         previous = LoadQuerySnapshot()] {
            try {
                const auto package = ToFsPath(packageRoot);
                const auto destination = ToFsPath(contentRoot);
                inxpack::Reader catalog(package / "AssetCatalog.inxcat");
                preparation->documents["catalog"] = nlohmann::json::parse(catalog.Read("RuntimeAssetCatalog.json"));
                auto packageIndex = OpenRuntimeCatalogDocument(package / "PackageIndex.inxmanifest");
                preparation->documents["package_index"] = std::string(std::istreambuf_iterator<char>(packageIndex), {});
                // The asset worker already owns the host IO grant. Cold Player
                // startup uses parallel extraction before there is a host owner.
                const auto content = inxpack::Extract(package / "Content.inxpkg", destination, allowedRoots, false);
                const auto catalogManifest = catalog.GetManifest();
                preparation->documents["content_identity"] = {inxpack::HashToHex(content.archiveHash),
                                                              content.archiveBytes};
                preparation->documents["catalog_identity"] = {inxpack::HashToHex(catalogManifest.archiveHash),
                                                              catalogManifest.archiveBytes};
                auto files = nlohmann::json::object();
                for (const auto &entry : content.entries)
                    files[entry.path] = entry.rawBytes;
                preparation->documents["content_files"] = std::move(files);
                auto &records = preparation->documents["records"];
                records = ReadRuntimeCatalogDocument(destination / "Library" / "RuntimeAssetRecords.json");
                preparation->documents["types"] =
                    ReadRuntimeCatalogDocument(destination / "Library" / "RuntimeTypeRegistry.json");
                std::unordered_set<std::string> newGuids;
                for (const auto &entry : records.at("entries")) {
                    const auto guid = entry.at("guid").get<std::string>();
                    if (!previous->guidToPath.count(guid))
                        newGuids.insert(guid);
                }
                PrepareRuntimeCatalog(*preparation, records, contentRoot, builtinRoots, true, &newGuids);
                auto roots = previous->runtimeArtifactRoots
                                 ? std::make_shared<RuntimeArtifactRoots>(*previous->runtimeArtifactRoots)
                                 : std::make_shared<RuntimeArtifactRoots>();
                auto shaderPaths = previous->shaderPaths;
                for (const auto &guid : newGuids) {
                    roots->emplace(guid, contentRoot);
                    auto key = ShaderPathIndexKey(*preparation->workingSet.metas.at(guid));
                    if (!key.empty() &&
                        !shaderPaths.try_emplace(key, preparation->workingSet.guidToPath.at(guid)).second)
                        throw std::runtime_error("runtime content conflicts with a published shader: " + key);
                }
                preparation->query->runtimeArtifactRoots = std::move(roots);
                preparation->query->shaderPaths = std::move(shaderPaths);
                preparation->documents["asset_guids"] = std::move(newGuids);
            } catch (...) {
                preparation->failure = std::current_exception();
            }
        });
    m_runtimeCatalog = std::move(preparation);
}

bool AssetDatabase::IsRuntimeAssetCatalogReady()
{
    AssertMutationThread("IsRuntimeAssetCatalogReady");
    ObserveCatalogWork(m_runtimeCatalogWork);
    if (!m_runtimeCatalog || !m_runtimeCatalogWork.Complete())
        return false;
    if (m_runtimeCatalog->failure)
        std::rethrow_exception(m_runtimeCatalog->failure);
    AssertNoPendingCommit("IsRuntimeAssetCatalogReady");
    if (m_runtimeCatalog->baseGeneration != m_queryGeneration)
        throw std::runtime_error("runtime asset catalog changed while new content was being prepared");
    return true;
}

const nlohmann::json *AssetDatabase::GetPreparedRuntimeCatalogDocuments()
{
    return IsRuntimeAssetCatalogReady() ? &m_runtimeCatalog->documents : nullptr;
}

void AssetDatabase::PublishRuntimeAssetCatalog()
{
    if (!IsRuntimeAssetCatalogReady())
        throw std::logic_error("runtime content has not finished preparing");
    InstallPreparedRuntimeCatalog(*m_runtimeCatalog);
    m_runtimeCatalog.reset();
}

void AssetDatabase::DiscardRuntimeAssetCatalog()
{
    AssertMutationThread("DiscardRuntimeAssetCatalog");
    m_runtimeCatalogWork.DropUnsubmitted();
    m_runtimeCatalog.reset();
}

AssetDatabase::AssetScanRequest AssetDatabase::CaptureScanRequest() const
{
    AssetScanRequest request;
    request.assetIndexPath = m_assetIndexPath;
    request.normalizedProjectRoot = FilesystemPathKey(m_projectRoot);
    request.expectedQueryGeneration = m_queryGeneration;
    if (!m_assetsRoot.empty())
        request.roots.push_back({m_assetsRoot, false});
    for (const auto &extra : m_extraScanRoots) {
        request.roots.push_back(
            {extra, m_readOnlyScanRoots.find(FromFsPath(ToFsPath(extra))) != m_readOnlyScanRoots.end()});
    }
    return request;
}

AssetDatabase::AssetScanArtifact AssetDatabase::BuildScanArtifact(const AssetScanRequest &request)
{
    const auto started = std::chrono::steady_clock::now();
    AssetScanArtifact artifact;
    artifact.producerThread = std::this_thread::get_id();
    try {
        const bool loadedIndex = artifact.index.Load(request.assetIndexPath, request.normalizedProjectRoot);
        (void)loadedIndex;
    } catch (const std::exception &error) {
        artifact.diagnostics.push_back(std::string("discarded invalid AssetIndex: ") + error.what());
        artifact.index.Reset(request.normalizedProjectRoot);
    }

    artifact.files.reserve(artifact.index.Size());
    const std::string normalizedIndexPath = FilesystemPathKey(request.assetIndexPath);
    std::unordered_set<std::string> observedPaths;
    std::unordered_map<std::string, AssetFileFingerprint> metadataFingerprints;
    metadataFingerprints.reserve(artifact.index.Size());
    for (const auto &scanRoot : request.roots) {
        const std::filesystem::path rootPath = ToFsPath(scanRoot.path);
        if (!std::filesystem::exists(rootPath)) {
            artifact.diagnostics.push_back("scan root does not exist: " + scanRoot.path);
            continue;
        }

        for (auto iterator = std::filesystem::recursive_directory_iterator(rootPath);
             iterator != std::filesystem::recursive_directory_iterator(); ++iterator) {
            const auto &entry = *iterator;
            const std::filesystem::path filePath = entry.path();
            if (IsIgnoredImportPath(filePath)) {
                if (entry.is_directory())
                    iterator.disable_recursion_pending();
                continue;
            }
            std::error_code typeError;
            if (!entry.is_regular_file(typeError) || typeError)
                continue;

            const std::string path = FromFsPath(filePath);
            if (filePath.extension() == ".tmp") {
                if (!scanRoot.readOnly)
                    artifact.orphanedTempFiles.push_back(path);
                continue;
            }
            if (filePath.extension() == ".meta") {
                if (!scanRoot.readOnly) {
                    AssetFileFingerprint fingerprint;
                    if (ReadFingerprint(entry, fingerprint)) {
                        std::filesystem::path sourcePath = filePath;
                        sourcePath.replace_extension();
                        metadataFingerprints[LexicalFilesystemPathKey(FromFsPath(sourcePath))] = fingerprint;
                    }
                }
                continue;
            }

            const std::string normalizedPath = FilesystemPathKey(path);
            if (normalizedPath == normalizedIndexPath)
                continue;
            if (!observedPaths.insert(normalizedPath).second)
                throw std::runtime_error("asset appears in overlapping scan roots: " + path);

            AssetScanFile scanned;
            scanned.path = path;
            scanned.normalizedPath = normalizedPath;
            scanned.readOnly = scanRoot.readOnly;
            if (!ReadFingerprint(entry, scanned.source))
                continue;
            std::string relativePath;
            scanned.identityKey = TryMakeRelativeFilesystemPath(path, scanRoot.path, relativePath)
                                      ? relativePath
                                      : FromFsPath(filePath.filename());
            artifact.files.push_back(std::move(scanned));
        }
    }

    for (auto &file : artifact.files) {
        if (file.readOnly)
            continue;
        const auto metadata = metadataFingerprints.find(LexicalFilesystemPathKey(file.path));
        if (metadata != metadataFingerprints.end())
            file.meta = metadata->second;
    }

    std::sort(artifact.files.begin(), artifact.files.end(), [](const AssetScanFile &left, const AssetScanFile &right) {
        return left.normalizedPath < right.normalizedPath;
    });
    std::sort(artifact.orphanedTempFiles.begin(), artifact.orphanedTempFiles.end());
    artifact.scanMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return artifact;
}

void AssetDatabase::QueueCatalogWork(CatalogWork &work, CatalogIoStage stage, uint32_t count, CatalogWork::Factory make)
{
    work = {stage, count, 0, count ? std::move(make) : CatalogWork::Factory{}, {}};
    if (!m_frameWork)
        SubmitCatalogWork(work, false);
}

void AssetDatabase::QueueCatalogJob(CatalogWork &work, CatalogIoStage stage, JobSystem::JobFn job)
{
    QueueCatalogWork(work, stage, 1, [job = std::move(job)](uint32_t) mutable { return std::move(job); });
}

bool AssetDatabase::CanSubmitCatalogStage(CatalogIoStage stage) const
{
    const auto &jobs = JobSystem::Get();
    const auto workers = (std::max)(jobs.GetWorkerCount(), 1u);
    const auto limit = jobs.GetDomainConcurrency(JobDomain::Asset);
    const auto slots = limit ? (std::min)(workers, limit) : workers;
    return m_refreshWork.active.size() + m_applyWork.active.size() + m_runtimeCatalogWork.active.size() < slots &&
           m_frameWork->CanAdmitResource(FrameWorkResource::Io, m_catalogIoCosts[static_cast<size_t>(stage)], true);
}

bool AssetDatabase::CanSubmitCatalogWork(const CatalogWork &work) const
{
    return work.submitted < work.count && CanSubmitCatalogStage(work.stage);
}

bool AssetDatabase::SubmitCatalogWork(CatalogWork &work, bool managed)
{
    const auto remaining = work.count - work.submitted;
    if (!remaining || (managed && !CanSubmitCatalogWork(work)))
        return false;
    const auto count = managed ? 1u : remaining;
    work.active.reserve(work.active.size() + count);
    for (uint32_t index = 0; index < count; ++index) {
        auto elapsed = std::make_shared<uint64_t>(0);
        auto &jobs = JobSystem::Get();
        auto job = jobs.Schedule(
            [task = work.make(work.submitted), elapsed] {
                const auto started = std::chrono::steady_clock::now();
                const auto measure = MakeScopeExit([&] {
                    *elapsed = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                            .count());
                });
                task();
            },
            JobDomain::Asset, managed ? JobPriority::Low : JobPriority::Normal,
            jobs.IsInline() ? JobAffinity::AnyThread : detail::IoJobAffinity);
        CatalogWork::Receipt receipt{std::move(job), std::move(elapsed)};
        if (managed) {
            receipt.frame = m_frameWork->Opportunity();
            receipt.grant = m_frameWork->RecordAcceptedResource(FrameWorkResource::Io,
                                                                m_catalogIoCosts[static_cast<size_t>(work.stage)]);
        }
        work.active.push_back(std::move(receipt));
        ++work.submitted;
    }
    if (work.submitted == work.count)
        work.make = {};
    return true;
}

void AssetDatabase::ObserveCatalogWork(CatalogWork &work) noexcept
{
    const auto remaining = std::remove_if(work.active.begin(), work.active.end(), [&](const auto &receipt) {
        if (!receipt.job.IsComplete())
            return false;
        const auto actual = *receipt.elapsed;
        if (actual)
            m_catalogIoCosts[static_cast<size_t>(work.stage)] = actual;
        if (m_frameWork)
            m_frameWork->ObserveResource(FrameWorkResource::Io, actual, receipt.frame, receipt.grant);
        return true;
    });
    work.active.erase(remaining, work.active.end());
}

void AssetDatabase::WaitForCatalogWork(CatalogWork &work) noexcept
{
    for (const auto &receipt : work.active) {
        if (!receipt.job.IsComplete()) {
            if (JobSystem::IsAvailable())
                JobSystem::Get().WaitPassiveCompletion(receipt.job);
            else
                INXLOG_ERROR("AssetDatabase pending work outlived JobSystem");
        }
    }
    ObserveCatalogWork(work);
}

FrameWorkSource AssetDatabase::GetCatalogWorkSource(FrameWorkOwner &owner)
{
    m_frameWork = &owner;
    FrameWorkSource source;
    source.observe = source.refresh = [this] {
        ObserveCatalogWork(m_refreshWork);
        ObserveCatalogWork(m_applyWork);
        ObserveCatalogWork(m_runtimeCatalogWork);
    };
    source.describe = [this] {
        FrameWorkFacts facts;
        const bool refresh = m_pendingAssetScan || m_pendingRefreshCommit;
        const bool advanceRefresh =
            refresh && (m_refreshWork.Complete() ? !m_frameWork->IsLate() : CanSubmitCatalogWork(m_refreshWork));
        const bool advanceApply = CanAdvanceModelReimport();
        const bool advanceRuntimeCatalog = CanSubmitCatalogWork(m_runtimeCatalogWork);
        facts.pending =
            refresh || !m_refreshWork.Complete() || IsModelReimportPending() || !m_runtimeCatalogWork.Complete();
        facts.workerBusy =
            !m_refreshWork.active.empty() || !m_applyWork.active.empty() || !m_runtimeCatalogWork.active.empty();
        facts.observe = facts.workerBusy;
        facts.ioReady = advanceRefresh || advanceApply || advanceRuntimeCatalog;
        if (advanceRefresh || advanceApply) {
            const bool apply = advanceApply && (m_preferApply || !advanceRefresh);
            facts.host = {static_cast<uint64_t>(apply), 0, true, false};
        } else if (advanceRuntimeCatalog) {
            facts.host = {2, 0, true, false};
        }
        return facts;
    };
    source.advance = [this](bool, uint64_t token) {
        if (token == 2)
            return SubmitCatalogWork(m_runtimeCatalogWork, true);
        const bool apply = token != 0;
        m_preferApply = !apply;
        if (apply)
            return AdvanceModelReimport(true);
        try {
            return AdvanceRefresh(true);
        } catch (...) {
            m_refreshFailure = std::current_exception();
            m_pendingAssetScan.reset();
            m_pendingRefreshCommit.reset();
            m_refreshWork.DropUnsubmitted();
            return true;
        }
    };
    return source;
}

void AssetDatabase::BeginRefresh()
{
    AssertMutationThread("BeginRefresh");
    AssertNoPendingCommit("BeginRefresh");
    if (m_pendingAssetScan || !m_refreshWork.Complete() || m_refreshFailure)
        throw std::logic_error("AssetDatabase refresh is already pending");
    if (!JobSystem::IsAvailable())
        throw std::logic_error("AssetDatabase asynchronous refresh requires JobSystem");

    AssetScanRequest request = CaptureScanRequest();
    auto state = std::make_shared<PendingAssetScan>(request.expectedQueryGeneration);
    m_pendingAssetScan = state;
    try {
        QueueCatalogJob(m_refreshWork, CatalogIoStage::Scan, [state, request = std::move(request)] {
            try {
                state->artifact = BuildScanArtifact(request);
            } catch (...) {
                state->failure = std::current_exception();
            }
        });
    } catch (...) {
        m_pendingAssetScan.reset();
        m_refreshWork.DropUnsubmitted();
        throw;
    }
}

bool AssetDatabase::TryCommitRefresh()
{
    AssertMutationThread("TryCommitRefresh");
    RethrowRefreshFailure();
    if (m_frameWork && m_frameWork->IsActive())
        return !IsRefreshPending();
    (void)AdvanceRefresh(false);
    return !IsRefreshPending();
}

void AssetDatabase::RethrowRefreshFailure()
{
    if (auto failure = std::exchange(m_refreshFailure, {}))
        std::rethrow_exception(failure);
}

bool AssetDatabase::AdvanceRefresh(bool managed)
{
    ObserveCatalogWork(m_refreshWork);
    if (!m_refreshWork.Complete())
        return SubmitCatalogWork(m_refreshWork, managed);
    if (!m_pendingAssetScan && !m_pendingRefreshCommit)
        return false;
    (void)CommitReadyRefresh();
    return true;
}

bool AssetDatabase::CommitReadyRefresh()
{
    if (m_pendingRefreshCommit) {
        const auto state = m_pendingRefreshCommit;
        try {
            if (state->phase == PendingRefreshCommit::Phase::MetadataPrepare) {
                state->phase = PendingRefreshCommit::Phase::MetadataMerge;
            }
            if (state->phase == PendingRefreshCommit::Phase::MetadataMerge) {
                (void)ContinuePendingMetadataMerge(state);
                return false;
            }
            if (state->phase == PendingRefreshCommit::Phase::MetadataPublish) {
                (void)ContinuePendingMetadataPublish(state);
                return false;
            }
            if (state->phase == PendingRefreshCommit::Phase::ImportPrepare) {
                (void)ContinuePendingImportPrepare(state);
                return false;
            }
            if (state->phase == PendingRefreshCommit::Phase::Import) {
                for (const auto index : state->importWave)
                    state->workerImports[index].complete = true;
                if (BeginNextImportWave(state))
                    return false;
                state->phase = PendingRefreshCommit::Phase::ImportMerge;
            }
            if (state->phase == PendingRefreshCommit::Phase::ImportMerge) {
                (void)ContinuePendingImportMerge(state);
                return false;
            }
            if (state->phase == PendingRefreshCommit::Phase::MetadataWrite) {
                if (state->metadataWriteFailure)
                    std::rethrow_exception(state->metadataWriteFailure);
                m_lastRefreshStagedBytes = state->stagedBytes;
                m_lastRefreshStagingMilliseconds = state->stagingMilliseconds;
                m_lastRefreshJournalBytes = state->journalBytes;
                m_lastRefreshJournalSerializeMilliseconds = state->journalSerializeMilliseconds;
                m_lastRefreshJournalWriteMilliseconds = state->journalWriteMilliseconds;
                m_lastRefreshJournalApplyMilliseconds = state->journalApplyMilliseconds;
                m_lastRefreshMetadataWriteMilliseconds =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              state->metadataWriteStarted)
                        .count();
                state->phase = PendingRefreshCommit::Phase::FileStateMerge;
                return false;
            }
            if (state->phase == PendingRefreshCommit::Phase::FileStateMerge) {
                (void)ContinuePendingFileStateMerge(state);
                return false;
            }
            if (state->phase == PendingRefreshCommit::Phase::ReadyToBuildIndex)
                BeginPendingIndexBuild(state);
            if (state->phase == PendingRefreshCommit::Phase::IndexBuild) {
                if (!m_refreshWork.Complete())
                    return false;
                if (state->indexBuildFailure)
                    std::rethrow_exception(state->indexBuildFailure);
                if (state->indexRebuildRequired) {
                    if (!state->builtAssetIndex)
                        throw std::logic_error("AssetIndex worker completed without an artifact");
                    state->stagedWorkingSet.assetIndex = std::move(*state->builtAssetIndex);
                    state->stagedWorkingSet.assetIndexDirty = false;
                    m_lastRefreshIndexBuildMilliseconds = state->indexBuildMilliseconds;
                    m_lastRefreshIndexSaveMilliseconds = state->indexSaveMilliseconds;
                    m_lastRefreshIndexBuildOnWorker = state->indexProducerThread != m_ownerThread;
                }
                if (!state->builtQuerySnapshot)
                    throw std::logic_error("query snapshot worker completed without an artifact");
                if (!state->builtDependencySnapshot)
                    throw std::logic_error("dependency snapshot worker completed without an artifact");
                m_lastRefreshQueryBuildMilliseconds = state->queryBuildMilliseconds;
                m_lastRefreshQueryBuildOnWorker = state->queryProducerThread != m_ownerThread;
                m_lastRefreshDependencyBuildMilliseconds = state->dependencyBuildMilliseconds;
                m_lastRefreshDependencyBuildOnWorker = state->dependencyProducerThread != m_ownerThread;
                state->phase = PendingRefreshCommit::Phase::ReadyToFinalize;
            }
            FinalizePendingRefreshCommit(state);
        } catch (...) {
            m_pendingRefreshCommit.reset();
            m_refreshWork.DropUnsubmitted();
            throw;
        }
        m_pendingRefreshCommit.reset();
        return true;
    }

    const auto state = m_pendingAssetScan;
    if (!state)
        throw std::logic_error("AssetDatabase has no pending refresh");

    m_pendingAssetScan.reset();
    if (state->failure)
        std::rethrow_exception(state->failure);
    if (!state->artifact)
        throw std::logic_error("AssetDatabase scan completed without an artifact");
    if (state->expectedQueryGeneration != m_queryGeneration) {
        BeginRefresh();
        return false;
    }
    return CommitScanArtifact(std::move(*state->artifact), state->expectedQueryGeneration);
}

bool AssetDatabase::IsRefreshPending() const
{
    AssertMutationThread("IsRefreshPending");
    return m_pendingAssetScan || m_pendingRefreshCommit || !m_refreshWork.Complete() || m_refreshFailure;
}

void AssetDatabase::Refresh()
{
    AssertMutationThread("Refresh");
    BeginRefresh();
    CompletePendingRefresh();
}

void AssetDatabase::CompletePendingRefresh()
{
    AssertMutationThread("CompletePendingRefresh");
    if (!IsRefreshPending())
        throw std::logic_error("AssetDatabase has no pending refresh");
    while (IsRefreshPending()) {
        RethrowRefreshFailure();
        (void)AdvanceRefresh(false);
        WaitForCatalogWork(m_refreshWork);
    }
}

void AssetDatabase::PrepareMetadata(WorkerMetadataPrepare &item)
{
    item.producerThread = std::this_thread::get_id();
    try {
        if (!item.loader)
            throw std::logic_error("metadata preparation requires a loader");

        const std::string metaPath = InxResourceMeta::GetMetaFilePath(item.file.path);
        const bool sidecarExists = std::filesystem::is_regular_file(ToFsPath(metaPath));
        if (!item.file.readOnly)
            item.expectedMetadata = AtomicFileState{};
        InxResourceMeta metadata;
        InxResourceMeta previousMetadata;
        bool previousMetadataLoaded = false;
        std::string preservedGuid = item.fallbackGuid;

        // Read-only package roots never consume sidecars. They are derived
        // project state and may have been produced by a different engine.
        if (sidecarExists && !item.file.readOnly) {
            InxResourceMeta loadedMetadata =
                LoadMetadataDocument(metaPath, &*item.expectedMetadata, &item.metadataSource);
            previousMetadata = std::move(loadedMetadata);
            previousMetadataLoaded = true;
            preservedGuid = previousMetadata.GetGuid();
            if (preservedGuid.empty())
                throw std::runtime_error("asset metadata sidecar has no GUID: " + metaPath);
        }

        AssetSource source(item.file.path);
        item.loader->CreateMeta(source, metadata);
        metadata.AddMetadata("file_path", InxResourceMeta::NormalizeFilePath(item.file.path));
        item.createdGuid = metadata.GetGuid();
        item.inheritedGuid = !previousMetadataLoaded && !item.file.readOnly && !preservedGuid.empty();

        if (previousMetadataLoaded)
            metadata.InheritSettings(previousMetadata);
        if (!preservedGuid.empty()) {
            metadata.AddMetadata("guid", preservedGuid);
        } else if (item.file.readOnly) {
            metadata.AddMetadata("guid", MakeReadOnlyGuid(item.file.identityKey));
            metadata.AddMetadata("read_only", true);
        }
        if (metadata.GetResourceType() == ResourceType::Mesh && !previousMetadataLoaded)
            MeshImportSettings::InitializeDefaults(metadata);

        if (metadata.GetGuid().empty())
            throw std::runtime_error("metadata preparation produced an empty GUID");
        ApplyBuiltinSceneIconMetadata(metadata, item.file.path, item.file.readOnly);
        RequireUnchangedFingerprint(item.file.path, item.file.source);
        item.metadata = std::make_shared<InxResourceMeta>(std::move(metadata));
    } catch (const std::exception &exception) {
        item.error = Utf8Diagnostic(exception.what());
    } catch (...) {
        item.error = "metadata preparation raised a non-standard exception";
    }
}

void AssetDatabase::PublishMissingMetadata(WorkerMetadataPrepare &item, const std::string &projectRoot)
{
    if (item.file.readOnly || (item.expectedMetadata && item.expectedMetadata->exists))
        return;
    try {
        std::string content = item.metadata->SerializeDocumentPortable(projectRoot).dump(4) + "\n";
        DocumentWriteOptions options;
        options.mode = FileWriteMode::Create;
        options.writeScope = item.writeScope;
        const auto ticket =
            DocumentStore::Instance().Submit(InxResourceMeta::GetMetaFilePath(item.file.path), content, options);
        ticket->Wait();
        const auto publication = ticket->GetPublication().value();
        if (publication == FilePublication::Written) {
            item.expectedMetadata = ticket->GetCommittedFileState().value();
            item.metadataSource = std::move(content);
            item.inheritedGuid = false;
        } else {
            // Reuse the original preparation order: fresh source facts, then
            // the winner's complete authored settings and GUID. A losing Mesh
            // candidate must not retain its own prefilled default settings.
            PrepareMetadata(item);
            if (!item.error.empty())
                return;
            if (!item.expectedMetadata->exists)
                throw std::runtime_error("metadata creation winner disappeared before adoption: " + item.file.path);
        }
        item.file.meta = {item.expectedMetadata->size, item.expectedMetadata->modifiedNs};
        RequireUnchangedFingerprint(item.file.path, item.file.source);
    } catch (const std::exception &exception) {
        item.error = Utf8Diagnostic(exception.what());
    } catch (...) {
        item.error = "metadata identity publication raised a non-standard exception";
    }
}

bool AssetDatabase::CommitScanArtifact(AssetScanArtifact artifact, uint64_t expectedQueryGeneration)
{
    const auto commitStarted = std::chrono::steady_clock::now();
    if (m_queryGeneration != expectedQueryGeneration)
        throw std::logic_error("AssetDatabase scan artifact is stale after a newer owner-thread mutation");

    m_lastRefreshReusedCount = 0;
    m_lastRefreshImportedCount = 0;
    m_lastRefreshImportedPaths.clear();
    m_lastRefreshImporterTaskCount = 0;
    m_lastRefreshMetadataTaskCount = 0;
    m_lastRefreshWorkerMetadataCount = 0;
    m_lastRefreshWorkerImporterCount = 0;
    m_lastRefreshScannedCount = artifact.files.size();
    m_lastRefreshScanMilliseconds = artifact.scanMilliseconds;
    m_lastRefreshScanOnWorker = artifact.producerThread != m_ownerThread;
    m_lastRefreshRestoreMilliseconds = 0.0;
    m_lastRefreshImportMilliseconds = 0.0;
    m_lastRefreshPrepareMilliseconds = 0.0;
    m_lastRefreshFinalizeMilliseconds = 0.0;
    m_lastRefreshOwnerMergeMaxSliceMilliseconds = 0.0;
    m_lastRefreshOwnerMergeSliceCount = 0;
    m_lastRefreshMetadataWriteMilliseconds = 0.0;
    m_lastRefreshStagedBytes = 0;
    m_lastRefreshStagingMilliseconds = 0.0;
    m_lastRefreshJournalBytes = 0;
    m_lastRefreshJournalSerializeMilliseconds = 0.0;
    m_lastRefreshJournalWriteMilliseconds = 0.0;
    m_lastRefreshJournalApplyMilliseconds = 0.0;
    m_lastRefreshIndexBuildMilliseconds = 0.0;
    m_lastRefreshIndexSaveMilliseconds = 0.0;
    m_lastRefreshIndexBuildOnWorker = false;
    m_lastRefreshQueryBuildMilliseconds = 0.0;
    m_lastRefreshQueryBuildOnWorker = false;
    m_lastRefreshDependencyBuildMilliseconds = 0.0;
    m_lastRefreshDependencyBuildOnWorker = false;
    m_lastRefreshPublishMilliseconds = 0.0;
    for (const auto &diagnostic : artifact.diagnostics)
        INXLOG_WARN("AssetDatabase: ", diagnostic);
    for (const auto &tempPath : artifact.orphanedTempFiles) {
        std::error_code error;
        std::filesystem::remove(ToFsPath(tempPath), error);
    }

    bool unchanged = !m_assetIndexDirty && artifact.files.size() == CountSourceAssets(artifact.index) &&
                     m_guidToPath.size() == artifact.index.Size() && m_pathToGuid.size() == artifact.index.Size() &&
                     m_fileStates.size() == artifact.index.Size();
    for (const auto &file : artifact.files) {
        if (!unchanged)
            break;
        const AssetIndexEntry *indexed = artifact.index.Find(file.normalizedPath);
        const auto pathMapping = m_pathToGuid.find(file.normalizedPath);
        const auto fileState = m_fileStates.find(file.normalizedPath);
        if (!indexed || indexed->resourceType != GetResourceTypeForPath(file.path) || indexed->source != file.source ||
            indexed->meta != file.meta || indexed->readOnly != file.readOnly || !indexed->importSucceeded ||
            !HasReusableRuntimeArtifact(*indexed, indexed->resourceType, ToFsPath(m_projectRoot)) ||
            (!file.readOnly && file.meta.size == 0) || pathMapping == m_pathToGuid.end() ||
            pathMapping->second != indexed->guid || fileState == m_fileStates.end() ||
            fileState->second.source != file.source || fileState->second.meta != file.meta ||
            fileState->second.readOnly != file.readOnly ||
            !HasCurrentBuiltinSceneIconMetadata(indexed->metadata, file.path, file.readOnly) ||
            !HasCurrentPluginPageTextureMetadata(indexed->metadata, file.path, m_projectRoot)) {
            unchanged = false;
            break;
        }
        const auto guidMapping = m_guidToPath.find(indexed->guid);
        const auto importResult = m_importResults.find(indexed->guid);
        if (guidMapping == m_guidToPath.end() || guidMapping->second != file.path ||
            m_metas.find(indexed->guid) == m_metas.end() || importResult == m_importResults.end() ||
            !importResult->second.succeeded) {
            unchanged = false;
        }
    }
    if (unchanged) {
        m_assetIndex = std::move(artifact.index);
        m_lastRefreshReusedCount = artifact.files.size();
        m_lastRefreshCommitMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitStarted).count();
        m_lastRefreshPrepareMilliseconds = m_lastRefreshCommitMilliseconds;
        INXLOG_INFO("AssetDatabase.Refresh completed without owner-state changes. Total assets: ", m_guidToPath.size(),
                    ", scanned: ", m_lastRefreshScannedCount, ", scan_ms: ", m_lastRefreshScanMilliseconds,
                    ", commit_ms: ", m_lastRefreshCommitMilliseconds);
        return true;
    }

    WorkingSet previousWorkingSet = TakeWorkingSet();
    auto restorePreviousWorkingSet = MakeScopeExit([this, &previousWorkingSet] {
        (void)TakeWorkingSet();
        InstallWorkingSet(std::move(previousWorkingSet));
    });
    m_assetIndex = std::move(artifact.index);

    m_guidToPath.clear();
    m_pathToGuid.clear();
    m_metas.clear();
    m_importResults.clear();
    m_fileStates.clear();
    m_fileStates.reserve(artifact.files.size());
    for (const auto &file : artifact.files)
        m_fileStates.emplace(file.normalizedPath, CachedFileState{file.source, file.meta, file.readOnly});

    std::vector<WorkerMetadataPrepare> workerMetadata;
    std::unordered_map<std::string, std::vector<std::string>> restoredDependencies;
    restoredDependencies.reserve(artifact.files.size());
    const auto updateScannedMapping = [this](const std::string &guid, const AssetScanFile &file) {
        if (guid.empty())
            throw std::logic_error("AssetDatabase scan produced an empty GUID");
        m_guidToPath[guid] = file.path;
        m_pathToGuid[file.normalizedPath] = guid;
    };
    const auto restoreStarted = std::chrono::steady_clock::now();
    for (const auto &file : artifact.files) {
        const ResourceType type = GetResourceTypeForPath(file.path);
        if (type == ResourceType::Meta)
            continue;

        const AssetIndexEntry *indexed = m_assetIndex.Find(file.normalizedPath);
        if (indexed && indexed->resourceType == type && indexed->source == file.source && indexed->meta == file.meta &&
            indexed->readOnly == file.readOnly && indexed->importSucceeded &&
            HasReusableRuntimeArtifact(*indexed, type, ToFsPath(m_projectRoot)) &&
            (file.readOnly || file.meta.size > 0) &&
            HasCurrentBuiltinSceneIconMetadata(indexed->metadata, file.path, file.readOnly) &&
            HasCurrentPluginPageTextureMetadata(indexed->metadata, file.path, m_projectRoot)) {
            m_metas[indexed->guid] = std::make_shared<InxResourceMeta>(indexed->metadata);
            updateScannedMapping(indexed->guid, file);
            restoredDependencies.emplace(indexed->guid, indexed->dependencies);
            m_importResults[indexed->guid] = {true, {}};
            ++m_lastRefreshReusedCount;
            continue;
        }

        WorkerMetadataPrepare item;
        item.file = file;
        item.resourceType = type;
        const auto loader = m_loaders.find(type);
        if (loader == m_loaders.end() || !loader->second)
            throw std::logic_error("AssetDatabase metadata preparation has no loader");
        item.loader = loader->second;
        if (!file.readOnly) {
            // The live catalog owns registered identities. A derived index can
            // disappear while a sidecar is being rebuilt; losing that cache
            // must not create a new GUID for an existing path.
            const auto registered = previousWorkingSet.pathToGuid.find(file.normalizedPath);
            if (registered != previousWorkingSet.pathToGuid.end())
                item.fallbackGuid = registered->second;
            else if (indexed)
                item.fallbackGuid = indexed->guid;
        }
        workerMetadata.push_back(std::move(item));
        ++m_lastRefreshImportedCount;
        m_lastRefreshImportedPaths.push_back(file.path);
    }
    m_lastRefreshRestoreMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - restoreStarted).count();

    auto state = std::make_shared<PendingRefreshCommit>();
    state->commitStarted = commitStarted;
    state->scanArtifact = std::move(artifact);
    state->workerMetadata = std::move(workerMetadata);
    std::vector<std::string> writePaths{m_assetIndexPath, m_assetStartupCachePath};
    for (const auto &item : state->workerMetadata)
        if (!item.file.readOnly) {
            writePaths.push_back(item.file.path);
            writePaths.push_back(InxResourceMeta::GetMetaFilePath(item.file.path));
        }
    state->writeScope = PathWriteScope::BeginWrite(writePaths);
    for (auto &item : state->workerMetadata)
        item.writeScope = state->writeScope;
    m_lastRefreshMetadataTaskCount = state->workerMetadata.size();
    state->restoredDependencies = std::move(restoredDependencies);
    state->stagedWorkingSet = TakeWorkingSet();
    InstallWorkingSet(std::move(previousWorkingSet));
    restorePreviousWorkingSet.Release();

    if (state->workerMetadata.size() > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("AssetDatabase metadata batch exceeds JobSystem capacity");
    m_pendingRefreshCommit = state;
    try {
        QueueCatalogWork(
            m_refreshWork, CatalogIoStage::Metadata, static_cast<uint32_t>(state->workerMetadata.size()),
            [state](uint32_t index) { return [state, index] { PrepareMetadata(state->workerMetadata[index]); }; });
    } catch (...) {
        m_pendingRefreshCommit.reset();
        m_refreshWork.DropUnsubmitted();
        throw;
    }
    m_lastRefreshPrepareMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - commitStarted).count();

    return false;
}

bool AssetDatabase::ContinuePendingMetadataMerge(const std::shared_ptr<PendingRefreshCommit> &state)
{
    if (!state || state->phase != PendingRefreshCommit::Phase::MetadataMerge)
        throw std::logic_error("AssetDatabase metadata merge phase has invalid state");

    const auto ownerStarted = std::chrono::steady_clock::now();
    size_t processed = 0;
    auto &workingSet = state->stagedWorkingSet;
    if (state->metadataMergeCursor == 0 && state->pendingImports.empty())
        state->pendingImports.reserve(state->workerMetadata.size());

    // Resolve authored identities for the complete batch before assigning a
    // missing sidecar's inherited path identity. An external move carries its
    // GUID in the sidecar; a new source at the old path is a different asset.
    // Keep both passes inside the existing owner-thread merge budget.
    while (state->metadataIdentityCursor < state->workerMetadata.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        const auto &item = state->workerMetadata[state->metadataIdentityCursor++];
        if (!item.error.empty())
            throw std::runtime_error("Metadata preparation failed for '" + item.file.path + "': " + item.error);
        if (!item.metadata)
            throw std::logic_error("Metadata worker completed without an artifact");
        if (!item.inheritedGuid) {
            const std::string guid = item.metadata->GetGuid();
            const auto [existing, inserted] = state->authoredGuidPaths.emplace(guid, item.file.path);
            if (!inserted && existing->second != item.file.path)
                throw std::runtime_error("Duplicate asset GUID produced during refresh: " + guid);
        }
        ++processed;
    }
    while (state->metadataIdentityCursor == state->workerMetadata.size() &&
           state->metadataMergeCursor < state->workerMetadata.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        const auto metadataIndex = state->metadataMergeCursor++;
        auto &item = state->workerMetadata[metadataIndex];
        if (item.producerThread != m_ownerThread)
            ++m_lastRefreshWorkerMetadataCount;
        if (!item.error.empty())
            throw std::runtime_error("Metadata preparation failed for '" + item.file.path + "': " + item.error);
        if (!item.metadata)
            throw std::logic_error("Metadata worker completed without an artifact");

        if (item.inheritedGuid) {
            const std::string inherited = item.metadata->GetGuid();
            const auto authored = state->authoredGuidPaths.find(inherited);
            const auto restored = workingSet.guidToPath.find(inherited);
            if ((authored != state->authoredGuidPaths.end() && authored->second != item.file.path) ||
                (restored != workingSet.guidToPath.end() && restored->second != item.file.path)) {
                if (item.createdGuid.empty())
                    throw std::logic_error("Metadata worker produced no identity for a replacement asset");
                item.metadata->AddMetadata("guid", item.createdGuid);
            }
        }
        const std::string guid = item.metadata->GetGuid();
        if (guid.empty())
            throw std::logic_error("Metadata worker produced an empty GUID");
        const auto existingGuid = workingSet.guidToPath.find(guid);
        if (existingGuid != workingSet.guidToPath.end() && existingGuid->second != item.file.path)
            throw std::runtime_error("Duplicate asset GUID produced during refresh: " + guid);
        const auto existingPath = workingSet.pathToGuid.find(item.file.normalizedPath);
        if (existingPath != workingSet.pathToGuid.end() && existingPath->second != guid)
            throw std::runtime_error("Asset path produced multiple GUIDs during refresh: " + item.file.path);

        workingSet.metas[guid] = item.metadata;
        workingSet.guidToPath[guid] = item.file.path;
        workingSet.pathToGuid[item.file.normalizedPath] = guid;
        if (!item.file.readOnly && item.expectedMetadata && !item.expectedMetadata->exists)
            state->metadataPublications.push_back(metadataIndex);
        state->pendingImports.push_back({guid, item.file.path, item.file.normalizedPath, item.file.source,
                                         item.file.meta, item.file.readOnly, !item.file.readOnly, item.expectedMetadata,
                                         std::move(item.metadataSource)});
        ++processed;
    }

    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ownerStarted).count();
    m_lastRefreshPrepareMilliseconds += elapsed;
    state->ownerMergeMaxSliceMilliseconds = std::max(state->ownerMergeMaxSliceMilliseconds, elapsed);
    ++state->ownerMergeSliceCount;
    if (state->metadataMergeCursor != state->workerMetadata.size())
        return false;

    if (state->metadataPublications.empty()) {
        state->phase = PendingRefreshCommit::Phase::ImportPrepare;
        return false;
    }
    const auto projectRoot = m_projectRoot;
    QueueCatalogWork(m_refreshWork, CatalogIoStage::Sidecar, static_cast<uint32_t>(state->metadataPublications.size()),
                     [state, projectRoot](uint32_t index) {
                         return [state, projectRoot, index] {
                             PublishMissingMetadata(state->workerMetadata[state->metadataPublications[index]],
                                                    projectRoot);
                         };
                     });
    state->phase = PendingRefreshCommit::Phase::MetadataPublish;
    return false;
}

bool AssetDatabase::ContinuePendingMetadataPublish(const std::shared_ptr<PendingRefreshCommit> &state)
{
    if (!state || state->phase != PendingRefreshCommit::Phase::MetadataPublish)
        throw std::logic_error("AssetDatabase metadata publication phase has invalid state");
    const auto ownerStarted = std::chrono::steady_clock::now();
    size_t processed = 0;
    auto &workingSet = state->stagedWorkingSet;

    // Remove changed candidates together before checking adopted identities.
    // The staged maps have not escaped to importers or a published catalog.
    while (state->metadataPublicationEraseCursor < state->metadataPublications.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        const auto index = state->metadataPublications[state->metadataPublicationEraseCursor++];
        const auto &item = state->workerMetadata[index];
        const auto &asset = state->pendingImports[index];
        if (!item.error.empty())
            throw std::runtime_error("Metadata identity publication failed for '" + item.file.path +
                                     "': " + item.error);
        if (item.metadata->GetGuid() != asset.guid) {
            workingSet.metas.erase(asset.guid);
            workingSet.guidToPath.erase(asset.guid);
            workingSet.pathToGuid.erase(asset.normalizedPath);
        }
        ++processed;
    }
    while (state->metadataPublicationEraseCursor == state->metadataPublications.size() &&
           state->metadataPublicationMergeCursor < state->metadataPublications.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        const auto index = state->metadataPublications[state->metadataPublicationMergeCursor++];
        auto &item = state->workerMetadata[index];
        auto &asset = state->pendingImports[index];
        const std::string guid = item.metadata->GetGuid();
        const auto registered = workingSet.guidToPath.find(guid);
        if (registered != workingSet.guidToPath.end() && registered->second != item.file.path)
            throw std::runtime_error("Duplicate asset GUID adopted during refresh: " + guid);
        workingSet.metas[guid] = item.metadata;
        workingSet.guidToPath[guid] = item.file.path;
        workingSet.pathToGuid[item.file.normalizedPath] = guid;
        asset.guid = guid;
        asset.meta = item.file.meta;
        asset.expectedMetadata = item.expectedMetadata;
        asset.metadataSource = std::move(item.metadataSource);
        ++processed;
    }
    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ownerStarted).count();
    m_lastRefreshPrepareMilliseconds += elapsed;
    state->ownerMergeMaxSliceMilliseconds = std::max(state->ownerMergeMaxSliceMilliseconds, elapsed);
    ++state->ownerMergeSliceCount;
    if (state->metadataPublicationMergeCursor != state->metadataPublications.size())
        return false;
    state->phase = PendingRefreshCommit::Phase::ImportPrepare;
    return false;
}

bool AssetDatabase::ContinuePendingImportPrepare(const std::shared_ptr<PendingRefreshCommit> &state)
{
    if (!state || state->phase != PendingRefreshCommit::Phase::ImportPrepare)
        throw std::logic_error("AssetDatabase import preparation phase has invalid state");
    const auto ownerStarted = std::chrono::steady_clock::now();
    size_t processed = 0;
    auto &workingSet = state->stagedWorkingSet;
    if (state->metadataMergeCursor == state->workerMetadata.size() && state->importRequestCursor == 0 &&
        !state->pendingImports.empty()) {
        state->workerImports.reserve(state->pendingImports.size());
    }
    while (state->metadataMergeCursor == state->workerMetadata.size() &&
           state->importRequestCursor < state->pendingImports.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        const size_t assetIndex = state->importRequestCursor++;
        const auto &asset = state->pendingImports[assetIndex];
        const std::string extension = FromFsPath(ToFsPath(asset.path).extension());
        AssetImporter *importer = m_importerRegistry.GetImporterForExtension(extension);
        if (!importer) {
            workingSet.importResults[asset.guid] = {true, {}};
            ++processed;
            continue;
        }
        const auto metadata = workingSet.metas.find(asset.guid);
        if (metadata == workingSet.metas.end() || !metadata->second)
            throw std::logic_error("AssetDatabase worker importer has no metadata snapshot");

        WorkerImport item;
        item.assetIndex = assetIndex;
        item.importer = importer;
        item.request = MakeImportRequestBase(asset.guid, asset.path, false, *metadata->second);
        item.expectedSource = asset.source;
        state->workerImports.push_back(std::move(item));
        ++processed;
    }

    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ownerStarted).count();
    m_lastRefreshPrepareMilliseconds += elapsed;
    state->ownerMergeMaxSliceMilliseconds = std::max(state->ownerMergeMaxSliceMilliseconds, elapsed);
    ++state->ownerMergeSliceCount;
    if (state->metadataMergeCursor != state->workerMetadata.size() ||
        state->importRequestCursor != state->pendingImports.size())
        return false;

    m_lastRefreshImporterTaskCount = state->workerImports.size();
    if (state->workerImports.size() > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error("AssetDatabase importer batch exceeds JobSystem capacity");

    // One immutable catalog for the batch, including freshly assigned GUIDs
    // whose sidecars have not yet been written. Workers never query live state.
    std::shared_ptr<QuerySnapshot> textureCatalog;
    std::unordered_map<std::string, size_t> importByGuid;
    for (size_t index = 0; index < state->workerImports.size(); ++index)
        importByGuid.emplace(state->workerImports[index].request.guid, index);
    std::unordered_map<std::string, ImportRequest::SkeletonDefinitionReader> reusedDefinitions;
    for (auto &item : state->workerImports) {
        if (item.request.resourceType != ResourceType::Mesh)
            continue;
        if (!textureCatalog) {
            textureCatalog = std::make_shared<QuerySnapshot>();
            textureCatalog->pathToGuid = workingSet.pathToGuid;
            textureCatalog->metas.insert(workingSet.metas.begin(), workingSet.metas.end());
        }
        item.request.resolveTextureGuid = ModelTextureResolver(textureCatalog);
        try {
            const auto owner = CopiedSkeletonOwner(item.request);
            if (owner.empty())
                continue;
            const auto metadata = workingSet.metas.find(owner);
            const auto sourceHash =
                SkeletonSourceHash(metadata == workingSet.metas.end() ? nullptr : metadata->second.get());
            if (const auto dependency = importByGuid.find(owner); dependency != importByGuid.end()) {
                item.skeletonDependency = dependency->second;
                state->workerImports[dependency->second].request.retainSkeletonDefinition = true;
            } else {
                const auto result = workingSet.importResults.find(owner);
                if (result != workingSet.importResults.end() && !result->second.succeeded)
                    throw std::invalid_argument("copied skeleton definition import failed: " + owner);
                auto &definition = reusedDefinitions[owner];
                if (!definition)
                    definition = CaptureSkeletonDefinition(sourceHash, GetSkinnedMeshArtifactPath(owner), false);
                item.request.skeletonDefinition = definition;
            }
        } catch (const std::exception &exception) {
            item.error = Utf8Diagnostic(exception.what());
        }
    }

    state->phase = PendingRefreshCommit::Phase::Import;
    state->importStarted = std::chrono::steady_clock::now();
    (void)BeginNextImportWave(state);
    return true;
}

bool AssetDatabase::BeginNextImportWave(const std::shared_ptr<PendingRefreshCommit> &state)
{
    state->importWave.clear();
    for (size_t index = 0; index < state->workerImports.size(); ++index) {
        auto &item = state->workerImports[index];
        if (item.scheduled || item.complete)
            continue;
        if (item.skeletonDependency && !state->workerImports[*item.skeletonDependency].complete)
            continue;
        item.scheduled = true;
        state->importWave.push_back(index);
    }
    if (state->importWave.empty()) {
        // Every runnable predecessor has finished. Remaining nodes are in, or
        // blocked by, a cycle; they never consume stale on-disk artifacts.
        for (auto &item : state->workerImports) {
            if (!item.complete) {
                item.error = "copied skeleton import dependency cycle: " + item.request.guid;
                item.complete = true;
                item.producerThread = m_ownerThread;
            }
        }
        return false;
    }
    QueueCatalogWork(m_refreshWork, CatalogIoStage::Cook, static_cast<uint32_t>(state->importWave.size()),
                     [state](uint32_t waveIndex) {
                         const auto index = state->importWave[waveIndex];
                         return [state, index] {
                             auto &item = state->workerImports[index];
                             item.producerThread = std::this_thread::get_id();
                             try {
                                 if (!item.error.empty())
                                     return;
                                 if (item.skeletonDependency) {
                                     const auto &dependency = state->workerImports[*item.skeletonDependency];
                                     if (!dependency.error.empty() || !dependency.artifact ||
                                         !dependency.artifact->skeletonDefinition)
                                         throw std::invalid_argument("copied skeleton definition import failed: " +
                                                                     dependency.request.guid);
                                     item.request.skeletonDefinition =
                                         [definition = dependency.artifact->skeletonDefinition] { return definition; };
                                 }
                                 item.artifact = item.importer->Import(item.request);
                                 ValidateImportedDependencyIdentities(*item.artifact, item.request.sourcePath);
                                 RequireUnchangedFingerprint(item.request.sourcePath, item.expectedSource);
                             } catch (const std::exception &exception) {
                                 item.error = Utf8Diagnostic(exception.what());
                             } catch (...) {
                                 item.error = "Importer raised a non-standard exception";
                             }
                             item.request.skeletonDefinition = {};
                         };
                     });
    return true;
}

bool AssetDatabase::ContinuePendingImportMerge(const std::shared_ptr<PendingRefreshCommit> &state)
{
    if (!state || state->phase != PendingRefreshCommit::Phase::ImportMerge)
        throw std::logic_error("AssetDatabase import merge phase has invalid state");

    const auto ownerStarted = std::chrono::steady_clock::now();
    size_t processed = 0;
    auto &workingSet = state->stagedWorkingSet;
    if (!state->importMergeInitialized) {
        state->metadataWrites.reserve(state->pendingImports.size() * 3);
        state->committedDependencies = state->restoredDependencies;
        state->importMergeInitialized = true;
    }

    while (state->importResultMergeCursor < state->workerImports.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        auto &item = state->workerImports[state->importResultMergeCursor++];
        const auto &asset = state->pendingImports[item.assetIndex];
        if (item.producerThread != m_ownerThread)
            ++m_lastRefreshWorkerImporterCount;
        try {
            if (!item.error.empty())
                throw std::runtime_error(item.error);
            if (!item.artifact)
                throw std::logic_error("Worker importer completed without an artifact");
            item.artifact->skeletonDefinition.reset();
            ResolveImportedDependencyPathHints(*item.artifact, workingSet.pathToGuid, asset.path);
            ValidateImportedDependencyIdentities(*item.artifact, asset.path);
            ValidateModelMaterialTargets(*item.artifact, [&workingSet](const std::string &guid) {
                const auto found = workingSet.metas.find(guid);
                return found == workingSet.metas.end() ? nullptr : found->second;
            });
            auto runtimeArtifactWrites = TakeRuntimeArtifactWrites(item.artifact->runtimeCpuArtifacts, asset.guid,
                                                                   item.request.resourceType, m_projectRoot);
            std::vector<std::string> artifactPaths;
            for (const auto &write : runtimeArtifactWrites)
                artifactPaths.push_back(write.path);
            state->artifactWriteScopes.push_back(PathWriteScope::BeginWrite(artifactPaths));
            workingSet.metas[asset.guid] = std::make_shared<InxResourceMeta>(std::move(item.artifact->metadata));
            workingSet.importResults[asset.guid] = {true, {}};
            for (auto &runtimeArtifactWrite : runtimeArtifactWrites)
                state->metadataWrites.push_back(std::move(runtimeArtifactWrite));
        } catch (const std::exception &exception) {
            const auto error = Utf8Diagnostic(exception.what());
            workingSet.importResults[asset.guid] = {false, error};
            INXLOG_ERROR("Asset import failed for '", asset.path, "': ", error);
        }
        ++processed;
    }

    while (state->importResultMergeCursor == state->workerImports.size() &&
           state->metadataWritePrepareCursor < state->pendingImports.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        auto &asset = state->pendingImports[state->metadataWritePrepareCursor++];
        state->committedDependencies.try_emplace(asset.guid, std::vector<std::string>{});
        if (asset.persistMetadata) {
            const auto metadata = workingSet.metas.find(asset.guid);
            if (metadata == workingSet.metas.end() || !metadata->second) {
                workingSet.importResults[asset.guid] = {false, "Pending import has no metadata to persist"};
            } else {
                const std::string metaPath = InxResourceMeta::GetMetaFilePath(asset.path);
                std::string content = metadata->second->SerializeDocumentPortable(m_projectRoot).dump(4) + "\n";
                asset.persistMetadata = content != asset.metadataSource;
                if (asset.persistMetadata) {
                    state->metadataWrites.push_back({metaPath, std::move(content), asset.expectedMetadata});
                } else {
                    const auto &consumed = asset.expectedMetadata.value();
                    asset.meta = {consumed.size, consumed.modifiedNs};
                }
            }
        }
        ++processed;
    }

    while (state->metadataWritePrepareCursor == state->pendingImports.size() &&
           state->dependencyMergeCursor < state->workerImports.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        const auto &item = state->workerImports[state->dependencyMergeCursor++];
        const auto &asset = state->pendingImports[item.assetIndex];
        const auto result = workingSet.importResults.find(asset.guid);
        if (result != workingSet.importResults.end() && result->second.succeeded && item.artifact &&
            item.artifact->dependenciesAuthoritative) {
            state->committedDependencies[asset.guid] = item.artifact->dependencies;
        }
        ++processed;
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(now - ownerStarted).count();
    state->ownerFinalizeMilliseconds += elapsed;
    state->ownerMergeMaxSliceMilliseconds = std::max(state->ownerMergeMaxSliceMilliseconds, elapsed);
    ++state->ownerMergeSliceCount;
    if (state->importResultMergeCursor != state->workerImports.size() ||
        state->metadataWritePrepareCursor != state->pendingImports.size() ||
        state->dependencyMergeCursor != state->workerImports.size())
        return false;

    m_lastRefreshImportMilliseconds = std::chrono::duration<double, std::milli>(now - state->importStarted).count();
    state->metadataWriteStarted = now;
    state->phase = state->metadataWrites.empty() ? PendingRefreshCommit::Phase::FileStateMerge
                                                 : PendingRefreshCommit::Phase::MetadataWrite;
    if (!state->metadataWrites.empty()) {
        const std::string projectRoot = m_projectRoot;
        const std::string journalPath = m_assetTransactionJournalPath;
        const std::string assetIndexPath = m_assetIndexPath;
        QueueCatalogJob(
            m_refreshWork, CatalogIoStage::Persist,
            [state, projectRoot, journalPath, assetIndexPath, writes = std::move(state->metadataWrites)]() mutable {
                try {
                    for (const auto &asset : state->pendingImports)
                        RequireUnchangedFingerprint(asset.path, asset.source);
                    const DocumentTransactionStats stats = DocumentTransaction::Commit(
                        projectRoot, journalPath, std::move(writes), {assetIndexPath}, state->writeScope);
                    state->stagedBytes = stats.preparedBytes;
                    state->stagingMilliseconds = stats.prepareMilliseconds;
                    state->journalBytes = stats.journalBytes;
                    state->journalSerializeMilliseconds = stats.serializeMilliseconds;
                    state->journalWriteMilliseconds = stats.journalWriteMilliseconds;
                    state->journalApplyMilliseconds = stats.applyMilliseconds;
                    state->committedMetadataFingerprints.clear();
                    state->committedMetadataFingerprints.reserve(stats.committedFileStates.size());
                    for (const auto &file : stats.committedFileStates) {
                        if (file.path.size() > 5 && file.path.compare(file.path.size() - 5, 5, ".meta") == 0)
                            state->committedMetadataFingerprints.emplace(
                                file.path.substr(0, file.path.size() - 5),
                                AssetFileFingerprint{file.size, file.modifiedNs});
                    }
                } catch (...) {
                    state->metadataWriteFailure = std::current_exception();
                }
            });
    }
    return true;
}

bool AssetDatabase::ContinuePendingFileStateMerge(const std::shared_ptr<PendingRefreshCommit> &state)
{
    if (!state || state->phase != PendingRefreshCommit::Phase::FileStateMerge)
        throw std::logic_error("AssetDatabase file-state merge phase has invalid state");

    const auto ownerStarted = std::chrono::steady_clock::now();
    size_t processed = 0;
    while (state->fileStateMergeCursor < state->pendingImports.size() &&
           HasOwnerMergeBudget(m_frameWork, ownerStarted, processed)) {
        const auto &asset = state->pendingImports[state->fileStateMergeCursor++];
        AssetFileFingerprint meta = asset.meta;
        if (asset.persistMetadata) {
            const auto committed = state->committedMetadataFingerprints.find(asset.normalizedPath);
            if (committed == state->committedMetadataFingerprints.end())
                throw std::logic_error("metadata transaction did not return a committed fingerprint");
            meta = committed->second;
        }
        state->stagedWorkingSet.fileStates[asset.normalizedPath] = {asset.source, meta, asset.readOnly};
        ++processed;
    }

    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ownerStarted).count();
    state->ownerFinalizeMilliseconds += elapsed;
    state->ownerMergeMaxSliceMilliseconds = std::max(state->ownerMergeMaxSliceMilliseconds, elapsed);
    ++state->ownerMergeSliceCount;
    if (state->fileStateMergeCursor != state->pendingImports.size())
        return false;
    state->phase = PendingRefreshCommit::Phase::ReadyToBuildIndex;
    return true;
}

AssetIndex AssetDatabase::BuildDerivedIndexArtifact(
    const WorkingSet &workingSet, const std::unordered_map<std::string, std::vector<std::string>> &dependenciesByGuid,
    const std::string &normalizedProjectRoot)
{
    AssetIndex index;
    index.Reset(normalizedProjectRoot);
    for (const auto &[normalizedPath, guid] : workingSet.pathToGuid) {
        const auto path = workingSet.guidToPath.find(guid);
        if (path == workingSet.guidToPath.end())
            throw std::logic_error("AssetIndex build found an incomplete GUID/path mapping");
        const auto metadata = workingSet.metas.find(guid);
        if (metadata == workingSet.metas.end() || !metadata->second)
            throw std::logic_error("AssetIndex build found a missing metadata artifact");
        const auto fileState = workingSet.fileStates.find(normalizedPath);
        if (fileState == workingSet.fileStates.end())
            throw std::logic_error("AssetIndex build found a missing file-state artifact");

        AssetIndexEntry entry;
        entry.normalizedPath = normalizedPath;
        entry.guid = guid;
        entry.resourceType = metadata->second->GetResourceType();
        const std::string artifactRelativePath = RuntimeArtifactRelativePath(guid, entry.resourceType);
        if (!artifactRelativePath.empty()) {
            std::error_code artifactError;
            const auto artifactPath = ToFsPath(normalizedProjectRoot) / std::filesystem::u8path(artifactRelativePath);
            if (std::filesystem::is_regular_file(artifactPath, artifactError) && !artifactError)
                entry.artifactPath = artifactRelativePath;
        }
        entry.source = fileState->second.source;
        entry.meta = fileState->second.meta;
        entry.readOnly = fileState->second.readOnly;
        entry.metadata = *metadata->second;
        if (entry.metadata.HasKey("content_hash"))
            entry.contentHash = entry.metadata.GetDataAs<std::string>("content_hash");
        const auto dependencies = dependenciesByGuid.find(guid);
        if (dependencies != dependenciesByGuid.end())
            entry.dependencies = dependencies->second;
        if (entry.metadata.HasKey("import_document"))
            entry.dependencies = {entry.metadata.GetDataAs<std::string>("import_owner_guid")};
        const auto importResult = workingSet.importResults.find(guid);
        if (importResult != workingSet.importResults.end()) {
            entry.importSucceeded = importResult->second.succeeded;
            entry.importError = importResult->second.error;
        }
        if (entry.importSucceeded && entry.contentHash.empty())
            throw std::logic_error("AssetIndex build found a successful import without a content hash");
        index.Upsert(std::move(entry));
    }
    return index;
}

std::shared_ptr<AssetDatabase::QuerySnapshot> AssetDatabase::BuildQuerySnapshotArtifact(
    const std::unordered_map<std::string, std::string> &guidToPath,
    const std::unordered_map<std::string, std::string> &pathToGuid,
    const std::unordered_map<std::string, std::shared_ptr<InxResourceMeta>> &metas,
    const std::unordered_map<std::string, CachedFileState> &fileStates,
    const std::unordered_map<std::string, ImportResultState> &importResults, uint64_t generation, bool includeCatalog)
{
    auto snapshot = std::make_shared<QuerySnapshot>();
    snapshot->generation = generation;
    snapshot->guidToPath = guidToPath;
    snapshot->pathToGuid = pathToGuid;
    snapshot->metas.reserve(metas.size());
    for (const auto &[guid, metadata] : metas) {
        if (!metadata)
            throw std::logic_error("Query snapshot build found null metadata");
        snapshot->metas.emplace(guid, metadata);
    }
    BuildShaderPathIndex(*snapshot);
    for (const auto &[guid, result] : importResults)
        if (!result.succeeded)
            snapshot->importErrors.emplace(guid, result.error);

    auto catalog = std::make_shared<AssetCatalogSnapshot>();
    catalog->m_generation = generation;
    if (!includeCatalog) {
        snapshot->catalog = std::move(catalog);
        return snapshot;
    }
    catalog->m_directories.reserve(pathToGuid.size());
    for (const auto &[normalizedPath, guid] : pathToGuid) {
        const auto mappedPath = guidToPath.find(guid);
        if (mappedPath == guidToPath.end())
            throw std::logic_error("Query snapshot path map references an unknown GUID");
        const auto metadata = metas.find(guid);
        if (metadata == metas.end() || !metadata->second)
            throw std::logic_error("Query snapshot path map references missing metadata");
        if (metadata->second->HasKey("import_owner_guid"))
            continue; // Project lists these below their model, not as loose source files.
        const auto fileState = fileStates.find(normalizedPath);
        if (fileState == fileStates.end())
            throw std::logic_error("Query snapshot path map references missing file state");

        AssetCatalogEntry entry;
        entry.guid = guid;
        entry.path = mappedPath->second;
        entry.name = FromFsPath(ToFsPath(entry.path).filename());
        entry.sortKey = entry.name;
        for (char &character : entry.sortKey) {
            if (character >= 'A' && character <= 'Z')
                character = static_cast<char>(character + ('a' - 'A'));
        }
        entry.resourceType = metadata->second->GetResourceType();
        entry.source = fileState->second.source;
        const auto importError = snapshot->importErrors.find(guid);
        if (importError != snapshot->importErrors.end())
            entry.importError = importError->second;
        const size_t separator = normalizedPath.find_last_of('/');
        const std::string parent = separator == std::string::npos ? std::string{} : normalizedPath.substr(0, separator);
        catalog->m_directories[parent].push_back(std::move(entry));
    }
    for (auto &[directory, entries] : catalog->m_directories) {
        (void)directory;
        std::sort(entries.begin(), entries.end(), [](const AssetCatalogEntry &left, const AssetCatalogEntry &right) {
            if (left.sortKey != right.sortKey)
                return left.sortKey < right.sortKey;
            return left.path < right.path;
        });
    }
    snapshot->catalog = std::move(catalog);
    return snapshot;
}

void AssetDatabase::BeginPendingIndexBuild(const std::shared_ptr<PendingRefreshCommit> &state)
{
    if (!state || state->phase != PendingRefreshCommit::Phase::ReadyToBuildIndex)
        throw std::logic_error("AssetDatabase index build phase has invalid state");

    auto &working = state->stagedWorkingSet;
    ExpandModelSubAssets(working.guidToPath, working.pathToGuid, working.metas, working.fileStates,
                         working.importResults);
    for (const auto &[guid, metadata] : working.metas)
        if (metadata->HasKey("import_document"))
            state->committedDependencies[guid] = {metadata->GetDataAs<std::string>("import_owner_guid")};

    const bool reusedLoadedIndex =
        state->pendingImports.empty() && m_lastRefreshReusedCount == state->scanArtifact.files.size() &&
        state->stagedWorkingSet.guidToPath.size() == state->stagedWorkingSet.assetIndex.Size() &&
        state->scanArtifact.files.size() == CountSourceAssets(state->stagedWorkingSet.assetIndex);
    state->indexRebuildRequired = !reusedLoadedIndex;
    const std::string normalizedProjectRoot = FilesystemPathKey(m_projectRoot);
    const std::string assetIndexPath = m_assetIndexPath;
    const std::string assetStartupCachePath = m_assetStartupCachePath;
    const uint64_t queryGeneration = m_queryGeneration + 1;
    state->expectedDependencyGeneration = m_dependencyGraph->GetAssetGeneration();
    const uint64_t dependencyGeneration = state->expectedDependencyGeneration + 1;
    state->phase = PendingRefreshCommit::Phase::IndexBuild;
    QueueCatalogJob(
        m_refreshWork, CatalogIoStage::Index,
        [state, normalizedProjectRoot, assetIndexPath, assetStartupCachePath, queryGeneration, dependencyGeneration] {
            try {
                if (state->indexRebuildRequired) {
                    state->indexProducerThread = std::this_thread::get_id();
                    const auto buildStarted = std::chrono::steady_clock::now();
                    AssetIndex index = BuildDerivedIndexArtifact(state->stagedWorkingSet, state->committedDependencies,
                                                                 normalizedProjectRoot);
                    state->indexBuildMilliseconds =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStarted)
                            .count();
                    const auto saveStarted = std::chrono::steady_clock::now();
                    index.Save(assetIndexPath, state->writeScope);
                    index.Save(assetStartupCachePath, state->writeScope);
                    state->indexSaveMilliseconds =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - saveStarted)
                            .count();
                    state->builtAssetIndex = std::move(index);
                } else {
                    // A valid live index may predate the startup-cache feature.
                    // Publish its immutable snapshot asynchronously so a later
                    // metadata transaction can invalidate AssetIndex.json without
                    // turning the next launch into a full source-tree scan.
                    const auto saveStarted = std::chrono::steady_clock::now();
                    state->stagedWorkingSet.assetIndex.Save(assetStartupCachePath, state->writeScope);
                    state->indexSaveMilliseconds =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - saveStarted)
                            .count();
                }

                state->queryProducerThread = std::this_thread::get_id();
                const auto queryStarted = std::chrono::steady_clock::now();
                state->builtQuerySnapshot =
                    BuildQuerySnapshotArtifact(state->stagedWorkingSet.guidToPath, state->stagedWorkingSet.pathToGuid,
                                               state->stagedWorkingSet.metas, state->stagedWorkingSet.fileStates,
                                               state->stagedWorkingSet.importResults, queryGeneration);
                state->queryBuildMilliseconds =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - queryStarted).count();

                state->dependencyProducerThread = std::this_thread::get_id();
                const auto dependencyStarted = std::chrono::steady_clock::now();
                state->builtDependencySnapshot =
                    AssetDependencyGraph::BuildAssetSnapshot(state->committedDependencies, dependencyGeneration);
                state->dependencyBuildMilliseconds =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dependencyStarted)
                        .count();
            } catch (...) {
                state->indexBuildFailure = std::current_exception();
            }
        });
}

void AssetDatabase::FinalizePendingRefreshCommit(const std::shared_ptr<PendingRefreshCommit> &state)
{
    if (!state)
        throw std::invalid_argument("AssetDatabase refresh commit state cannot be null");
    if (state->phase != PendingRefreshCommit::Phase::ReadyToFinalize)
        throw std::logic_error("AssetDatabase refresh commit is not ready to finalize");
    if (!state->builtQuerySnapshot || state->builtQuerySnapshot->generation != m_queryGeneration + 1 ||
        !state->builtQuerySnapshot->catalog ||
        state->builtQuerySnapshot->catalog->GetGeneration() != state->builtQuerySnapshot->generation)
        throw std::logic_error("AssetDatabase refresh query artifact has an invalid generation");
    if (!state->builtDependencySnapshot ||
        state->builtDependencySnapshot->GetGeneration() != state->expectedDependencyGeneration + 1 ||
        m_dependencyGraph->GetAssetGeneration() != state->expectedDependencyGeneration)
        throw std::logic_error("AssetDatabase refresh dependency artifact is stale");
    const auto finalizeStarted = std::chrono::steady_clock::now();
    const auto previousSnapshot = LoadQuerySnapshot();
    WorkingSet previousWorkingSet = TakeWorkingSet();
    InstallWorkingSet(std::move(state->stagedWorkingSet));
    auto restorePreviousWorkingSet = MakeScopeExit([this, &previousWorkingSet] {
        (void)TakeWorkingSet();
        InstallWorkingSet(std::move(previousWorkingSet));
    });

    m_dependencyGraph->InstallAssetSnapshot(std::move(state->builtDependencySnapshot));
    m_assetIndexDirty = false;
    const auto publishStarted = std::chrono::steady_clock::now();
    InstallQuerySnapshot(std::move(state->builtQuerySnapshot));
    m_lastRefreshPublishMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - publishStarted).count();
    m_lastRefreshCommitMilliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state->commitStarted).count();
    m_lastRefreshOwnerMergeMaxSliceMilliseconds = state->ownerMergeMaxSliceMilliseconds;
    m_lastRefreshOwnerMergeSliceCount = state->ownerMergeSliceCount;
    m_lastRefreshFinalizeMilliseconds =
        state->ownerFinalizeMilliseconds +
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - finalizeStarted).count();
    restorePreviousWorkingSet.Release();
    PublishRefreshAssetEvents(previousWorkingSet);
    PublishModelSubAssetEvents(previousSnapshot);
    for (auto &item : state->workerMetadata)
        item.writeScope.Close();
    state->artifactWriteScopes.clear();
    state->writeScope.Close();
}

bool AssetDatabase::IsReadOnlyPath(const std::string &normalizedPath) const
{
    for (const auto &root : m_readOnlyScanRoots) {
        if (IsFilesystemPathWithin(normalizedPath, root))
            return true;
    }
    return false;
}

void AssetDatabase::RebuildDerivedIndex()
{
    WorkingSet workingSet = TakeWorkingSet();
    auto restoreWorkingSet = MakeScopeExit([this, &workingSet] { InstallWorkingSet(std::move(workingSet)); });
    std::vector<std::string> indexedGuids;
    indexedGuids.reserve(workingSet.guidToPath.size());
    for (const auto &[guid, path] : workingSet.guidToPath) {
        (void)path;
        indexedGuids.push_back(guid);
    }
    const auto dependencySets = m_dependencyGraph->GetDependenciesBatch(indexedGuids);
    std::unordered_map<std::string, std::vector<std::string>> dependenciesByGuid;
    dependenciesByGuid.reserve(dependencySets.size());
    for (const auto &[guid, dependencies] : dependencySets)
        dependenciesByGuid.emplace(guid, std::vector<std::string>(dependencies.begin(), dependencies.end()));
    workingSet.assetIndex = BuildDerivedIndexArtifact(workingSet, dependenciesByGuid, FilesystemPathKey(m_projectRoot));
    InstallWorkingSet(std::move(workingSet));
    restoreWorkingSet.Release();
}

void AssetDatabase::FlushDerivedIndex(bool waitForPendingScan)
{
    AssertMutationThread("FlushDerivedIndex");
    if (!waitForPendingScan && (IsRefreshPending() || IsModelReimportPublishing())) {
        // Shutdown path: the derived index is a startup cache, and the next
        // launch revalidates it against the real asset state anyway. Blocking
        // engine teardown on a full background project scan (worst case:
        // quitting right after startup) trades seconds of exit latency for a
        // cache write we can simply skip.
        INXLOG_INFO("FlushDerivedIndex skipped: asset catalog work is still pending at shutdown");
        return;
    }
    CompleteRuntimeArtifactPublication();
    if (!m_assetIndexDirty)
        return;
    const auto writeScope = PathWriteScope::BeginWrite({m_assetIndexPath, m_assetStartupCachePath});
    RebuildDerivedIndex();
    m_assetIndex.Save(m_assetIndexPath, writeScope);
    m_assetIndex.Save(m_assetStartupCachePath, writeScope);
    m_assetIndexDirty = false;
}

AssetMutationResult AssetDatabase::ImportAsset(const std::string &requestedPath, const PathWriteScope &parentScope)
{
    AssertMutationThread("ImportAsset");
    AssertNoPendingCommit("ImportAsset");
    // Derive sidecars and writer ranges from the same actual source spelling.
    const auto path = ResolveFilesystemPath(requestedPath);
    AssetMutationResult result;
    result.operation = "import";
    result.path = path;
    result.resourceType = GetResourceTypeForPath(path);
    std::filesystem::path fsPath = ToFsPath(path);
    if (!std::filesystem::exists(fsPath) || !std::filesystem::is_regular_file(fsPath)) {
        result.errorCode = AssetMutationErrorCode::InvalidPath;
        result.error = "asset path is not a regular file";
        return result;
    }

    if (IsIgnoredImportPath(fsPath)) {
        result.errorCode = AssetMutationErrorCode::UnsupportedType;
        result.error = "version control files, bytecode and cache paths are not importable assets";
        return result;
    }

    if (IsMetaFile(fsPath)) {
        result.errorCode = AssetMutationErrorCode::UnsupportedType;
        result.error = "metadata sidecars cannot be imported as assets";
        return result;
    }

    const ResourceType type = result.resourceType;
    if (type == ResourceType::Meta) {
        result.errorCode = AssetMutationErrorCode::UnsupportedType;
        result.error = "metadata resources cannot be imported directly";
        return result;
    }

    const std::string normalizedPath = FilesystemPathKey(path);
    const bool readOnly = IsReadOnlyPath(normalizedPath);
    std::optional<AtomicFileState> expectedMetadata;
    const auto writeScope = PathWriteScope::BeginWrite({path, InxResourceMeta::GetMetaFilePath(path)}, parentScope);
    const auto &authorization = parentScope.IsOpen() ? parentScope : writeScope;
    std::string guid =
        CreateOrLoadMetadata(path, type, readOnly, !readOnly, normalizedPath, &expectedMetadata, authorization);
    if (guid.empty()) {
        result.errorCode = AssetMutationErrorCode::ImportFailed;
        result.error = "asset metadata could not be created or loaded";
        return result;
    }
    result.guid = guid;

    UpdateMapping(guid, path);
    // Python owns script compilation, dependency analysis, and live class
    // publication. ScriptImporter has no artifact or dependency output, so
    // routing a new script through the generic importer only rewrites the
    // metadata sidecar that CreateOrLoadMetadata just persisted and commits a
    // heavyweight filesystem transaction on the editor thread.
    const bool imported = type == ResourceType::Script
                              ? (m_importResults[guid] = {true, {}}, true)
                              : RunImporter(guid, path, false, !readOnly, nullptr, nullptr,
                                            expectedMetadata ? &*expectedMetadata : nullptr, authorization);
    if (!imported) {
        result.errorCode = AssetMutationErrorCode::ImportFailed;
        result.error = m_importResults.at(guid).error;
    }
    // Failed files retain their authored identity, as during Refresh, but have no runtime publication.
    UpdateCachedFileState(path, readOnly);
    m_assetIndexDirty = true;
    const auto previous = LoadQuerySnapshot();
    const auto changedSubAssets = PublishQuerySnapshotForPaths({path});
    result.succeeded = imported;
    result.databaseCommitted = true;
    result.changed = true;
    result.queryGeneration = GetQueryGeneration();
    // Import may restore a previously deleted GUID. Incoming dependency edges
    // intentionally survive deletion, so publish availability only after the
    // catalog and importer outputs have committed. Existing runtime references
    // can then resolve their cache without any Project-panel-specific repair.
    if (imported) {
        PublishModelSubAssetEvents(previous, &changedSubAssets);
        auto *registry = BoundRegistry();
        if (!registry || !registry->NotifySourceImported(guid, type))
            m_dependencyGraph->NotifyEvent(guid, type, AssetEvent::Modified);
    }
    return result;
}

AssetMutationResult AssetDatabase::ReimportAsset(const std::string &path, const nlohmann::json &settings,
                                                 const PathWriteScope &parentScope)
{
    AssertMutationThread("ReimportAsset");
    AssertNoPendingCommit("ReimportAsset");
    AssetMutationResult result;
    WorkerMetadataPrepare candidate;
    const auto selected = GetMetaByPath(path);
    if (selected && selected->HasKey("import_document")) {
        result.errorCode = AssetMutationErrorCode::InvalidPath;
        result.error = "model animation clips are edited through their model import settings";
        return result;
    }
    const bool ownedTexture = selected && selected->HasKey("import_owner_guid");
    if (ownedTexture && settings.is_null()) {
        result.errorCode = AssetMutationErrorCode::InvalidPath;
        result.error = "model Texture reimport requires settings; reimport its owner to refresh source content";
        return result;
    }
    const auto sourcePath = ownedTexture ? GetPathFromGuid(selected->GetDataAs<std::string>("import_owner_guid"))
                                         : ResolveFilesystemPath(path);
    const auto writeScope =
        PathWriteScope::BeginWrite({sourcePath, InxResourceMeta::GetMetaFilePath(sourcePath)}, parentScope);
    const auto &authorization = parentScope.IsOpen() ? parentScope : writeScope;
    if (!PrepareReimportInput(sourcePath, candidate, result) ||
        !PrepareReimportMetadata(candidate, ownedTexture ? nlohmann::json{} : settings, result))
        return result;
    if (ownedTexture)
        ApplyModelTextureSettings(*candidate.metadata, selected->GetGuid(), settings);
    if (!RunImporter(result.guid, sourcePath, true, true, &*candidate.metadata, &candidate.file.source,
                     &candidate.expectedMetadata.value(), authorization)) {
        result.errorCode = AssetMutationErrorCode::ImportFailed;
        result.error = m_importResults.at(result.guid).error;
        ReportAssetLoadFailure(result.guid, result.error);
        result.queryGeneration = GetQueryGeneration();
        return result;
    }
    FinishReimport(result);
    return result;
}

bool AssetDatabase::PrepareReimportInput(const std::string &path, WorkerMetadataPrepare &candidate,
                                         AssetMutationResult &result)
{
    result.operation = "reimport";
    result.path = path;
    result.resourceType = GetResourceTypeForPath(path);
    const std::filesystem::path fsPath = ToFsPath(path);
    if (IsIgnoredImportPath(fsPath)) {
        result.errorCode = AssetMutationErrorCode::UnsupportedType;
        result.error = "version control files, bytecode and cache paths are not importable assets";
        return false;
    }
    if (!std::filesystem::is_regular_file(fsPath) || IsMetaFile(fsPath) || result.resourceType == ResourceType::Meta) {
        result.errorCode = AssetMutationErrorCode::InvalidPath;
        result.error = "registered asset path is not a supported regular file";
        return false;
    }

    const std::string guid = GetGuidFromPath(path);
    if (guid.empty()) {
        result.errorCode = AssetMutationErrorCode::NotFound;
        result.error = "asset is not registered";
        return false;
    }
    result.guid = guid;

    // Prepare a candidate without publishing sidecars or touching live metadata.
    // The importer transaction owns the single durable publication point.
    candidate.file.path = path;
    candidate.file.normalizedPath = FilesystemPathKey(path);
    candidate.file.readOnly = IsReadOnlyPath(candidate.file.normalizedPath);
    if (candidate.file.readOnly) {
        result.errorCode = AssetMutationErrorCode::InvalidPath;
        result.error = "read-only assets cannot be reimported";
        return false;
    }
    candidate.resourceType = result.resourceType;
    candidate.fallbackGuid = guid;
    const auto loader = m_loaders.find(result.resourceType);
    if (loader == m_loaders.end() || !loader->second)
        throw std::logic_error("Registered asset has no metadata loader");
    candidate.loader = loader->second;
    if (!ReadFingerprint(fsPath, candidate.file.source)) {
        result.errorCode = AssetMutationErrorCode::InvalidPath;
        result.error = "could not read source fingerprint";
        return false;
    }
    return true;
}

bool AssetDatabase::PrepareReimportMetadata(WorkerMetadataPrepare &candidate, const nlohmann::json &settings,
                                            AssetMutationResult &result)
{
    PrepareMetadata(candidate);
    if (!candidate.error.empty() || !candidate.metadata || candidate.metadata->GetGuid() != result.guid) {
        result.errorCode = AssetMutationErrorCode::ImportFailed;
        result.error =
            candidate.error.empty() ? "metadata preparation did not preserve the registered GUID" : candidate.error;
        return false;
    }
    if (!settings.is_null()) {
        if (result.resourceType != ResourceType::Mesh || !settings.is_object())
            throw std::invalid_argument("import settings require a model and an object");
        MeshImportSettings::ApplyPatch(*candidate.metadata, settings);
    }
    return true;
}

void AssetDatabase::FinishReimport(AssetMutationResult &result, const CachedFileState *committedFiles,
                                   const std::shared_ptr<AssetLoadTicket> &repairTicket)
{
    if (committedFiles)
        m_fileStates[LexicalFilesystemPathKey(result.path)] = *committedFiles;
    else
        UpdateCachedFileState(result.path, false);
    m_assetIndexDirty = true;
    const auto previous = LoadQuerySnapshot();
    const auto changedSubAssets = PublishQuerySnapshotForPaths({result.path});
    PublishModelSubAssetEvents(previous, &changedSubAssets);
    auto *registry = BoundRegistry();
    const bool meshPublication =
        registry && registry->NotifySourceImported(result.guid, result.resourceType, repairTicket);
    // Python owns ordinary script dependency and live class publication.
    if (result.resourceType != ResourceType::Script && !meshPublication)
        m_dependencyGraph->NotifyEvent(result.guid, result.resourceType, AssetEvent::Modified);
    result.succeeded = true;
    result.databaseCommitted = true;
    result.changed = true;
    result.queryGeneration = GetQueryGeneration();
}

void AssetDatabase::BeginModelReimport(const std::string &path, const nlohmann::json &settings,
                                       ModelReimportPublished onPublished)
{
    AssertMutationThread("BeginModelReimport");
    AssertNoPendingCommit("BeginModelReimport");
    ObserveCatalogWork(m_applyWork);
    if (m_pendingModelReimport && m_pendingModelReimport->IsDiscarded() && m_applyWork.Complete())
        m_pendingModelReimport.reset();
    if (m_finishedModelApply && m_finishedModelApply->abandoned)
        m_finishedModelApply.reset();
    if (m_pendingModelReimport || m_finishedModelApply)
        throw std::logic_error("A model Apply is already in progress");
    const auto selected = GetMetaByPath(path);
    if (selected && selected->HasKey("import_document"))
        throw std::invalid_argument("model animation clips are edited through their model import settings");
    const auto textureGuid = selected && selected->HasKey("import_owner_guid") ? selected->GetGuid() : std::string{};
    const auto sourcePath = textureGuid.empty()
                                ? ResolveFilesystemPath(path)
                                : GetPathFromGuid(selected->GetDataAs<std::string>("import_owner_guid"));
    if (GetResourceTypeForPath(sourcePath) != ResourceType::Mesh || !settings.is_object())
        throw std::invalid_argument("Model Apply requires a model path and settings object");

    m_lastModelReimportWorkerMilliseconds = 0.0;
    m_lastModelReimportPrepareMilliseconds = 0.0;
    m_lastModelReimportPersistenceMilliseconds = 0.0;
    m_lastModelReimportLivePublicationMilliseconds = 0.0;
    StartModelReimport(sourcePath, textureGuid, settings, std::move(onPublished));
}

bool AssetDatabase::CanBeginMeshArtifactRepair() const
{
    return !m_pendingModelReimport && !IsRefreshPending();
}

void AssetDatabase::BeginMeshArtifactRepair(const std::shared_ptr<AssetLoadTicket> &ticket)
{
    const auto path = GetPathFromGuid(ticket->GetGuid());
    if (!m_importerRegistry.GetImporterForExtension(FromFsPath(ToFsPath(path).extension())))
        throw std::runtime_error("Cooked mesh artifacts are unavailable for " + path);
    StartModelReimport(path, {}, nlohmann::json::object(), {}, ticket);
}

void AssetDatabase::StartModelReimport(const std::string &sourcePath, const std::string &textureGuid,
                                       const nlohmann::json &settings, ModelReimportPublished onPublished,
                                       std::shared_ptr<AssetLoadTicket> repairTicket)
{
    if (!JobSystem::IsAvailable())
        throw std::logic_error("Model import requires the engine JobSystem");
    auto pending = std::make_shared<PendingModelReimport>();
    pending->onPublished = std::move(onPublished);
    pending->repairTicket = std::move(repairTicket);
    pending->writeScope =
        PathWriteScope::BeginWrite({sourcePath, InxResourceMeta::GetMetaFilePath(sourcePath), m_assetIndexPath});
    if (!PrepareReimportInput(sourcePath, pending->metadata, pending->result))
        throw std::runtime_error(pending->result.error);
    auto &worker = pending->worker;
    worker.importer = m_importerRegistry.GetImporterForExtension(FromFsPath(ToFsPath(sourcePath).extension()));
    if (!worker.importer)
        throw std::logic_error("Model Apply has no registered importer");
    // Capture any external definition against the settings to be applied by
    // the worker, while the owner catalog and artifacts are still stable.
    InxResourceMeta requestMetadata = *m_metas.at(pending->result.guid);
    if (textureGuid.empty())
        MeshImportSettings::ApplyPatch(requestMetadata, settings);
    worker.request = MakeImportRequest(pending->result.guid, sourcePath, true, requestMetadata);
    worker.expectedSource = pending->metadata.file.source;
    pending->settings = settings;
    pending->metadataExists =
        ReadFingerprint(ToFsPath(InxResourceMeta::GetMetaFilePath(sourcePath)), pending->metadataFingerprint);
    QueueCatalogJob(m_applyWork, CatalogIoStage::Apply, [pending, textureGuid] {
        auto &worker = pending->worker;
        worker.producerThread = std::this_thread::get_id();
        try {
            if (!PrepareReimportMetadata(pending->metadata, textureGuid.empty() ? pending->settings : nlohmann::json{},
                                         pending->result))
                throw std::runtime_error(pending->result.error);
            if (!textureGuid.empty())
                ApplyModelTextureSettings(*pending->metadata.metadata, textureGuid, pending->settings);
            worker.request.metadata = std::move(*pending->metadata.metadata);
            const auto importStarted = std::chrono::steady_clock::now();
            worker.artifact = worker.importer->Reimport(worker.request);
            pending->workerMilliseconds =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - importStarted).count();
        } catch (const std::exception &exception) {
            worker.error = Utf8Diagnostic(exception.what());
        } catch (...) {
            worker.error = "Importer raised a non-standard exception";
        }
        worker.request.skeletonDefinition = {};
        pending->FinishWorker();
    });
    m_pendingModelReimport = std::move(pending);
}

std::optional<AssetMutationResult> AssetDatabase::TryCommitModelReimport()
{
    AssertMutationThread("TryCommitModelReimport");
    if (!m_finishedModelApply) {
        if (!m_pendingModelReimport || m_pendingModelReimport->repairTicket)
            throw std::logic_error("No model Apply is in progress");
        ObserveCatalogWork(m_applyWork);
        if (!m_frameWork || !m_frameWork->IsActive())
            AdvanceModelReimport(false);
        if (!m_finishedModelApply)
            return std::nullopt;
    }
    auto result = std::move(m_finishedModelApply->result);
    m_finishedModelApply.reset();
    return result;
}

std::optional<AssetMutationResult> AssetDatabase::CompletePendingModelReimport()
{
    AssertMutationThread("CompletePendingModelReimport");
    if (IsDeliveringModelReimport())
        throw std::logic_error("Model Apply cannot be awaited from its publication callback");
    if (const auto pending = m_pendingModelReimport) {
        if (IsRefreshPending())
            CompletePendingRefresh();
        while (pending->phase != PendingModelReimport::Phase::Complete) {
            WaitForCatalogWork(m_applyWork);
            AdvanceModelReimport(false);
        }
        if (pending->repairTicket && !pending->result.succeeded)
            return pending->result;
    }
    if (m_finishedModelApply && !m_finishedModelApply->discarded)
        return m_finishedModelApply->result;
    return std::nullopt;
}

bool AssetDatabase::CanAdvanceModelReimport() const
{
    const auto &pending = m_pendingModelReimport;
    if (!IsModelReimportPending() || IsDeliveringModelReimport())
        return false;
    if (!m_applyWork.Complete())
        return CanSubmitCatalogWork(m_applyWork);
    if (m_frameWork->IsLate())
        return false;
    return pending->phase == PendingModelReimport::Phase::Write || pending->IsDiscarded() ||
           !pending->worker.error.empty() || (!IsRefreshPending() && CanSubmitCatalogStage(CatalogIoStage::Persist));
}

bool AssetDatabase::AdvanceModelReimport(bool managed)
{
    const auto pending = m_pendingModelReimport;
    if (!IsModelReimportPending() || IsDeliveringModelReimport())
        return false;
    auto &worker = pending->worker;
    auto &result = pending->result;
    try {
        if (!m_applyWork.Complete())
            return SubmitCatalogWork(m_applyWork, managed);
        if (managed && m_frameWork->IsLate())
            return false;
        if (pending->IsDiscarded())
            throw std::runtime_error("Model Apply was discarded before publication");
        if (!worker.error.empty())
            throw std::runtime_error(worker.error);
        if (pending->phase == PendingModelReimport::Phase::Write) {
            PublishModelReimport(*pending);
        } else {
            // Refresh may have seen an external edit. Observe its published catalog before accepting a write.
            if (IsRefreshPending() || (managed && !CanSubmitCatalogStage(CatalogIoStage::Persist)))
                return false;
            const auto started = std::chrono::steady_clock::now();
            if (GetGuidFromCatalogPath(result.path) != result.guid)
                throw std::runtime_error("Model identity changed while Apply was in progress");
            ValidateImportPublication(worker.request, worker.artifact.value());
            pending->artifactGuids.insert(result.guid);
            for (const auto &output : worker.artifact->runtimeCpuArtifacts)
                if (!output.guid.empty())
                    pending->artifactGuids.insert(output.guid);
            pending->prepareMilliseconds =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            pending->phase = PendingModelReimport::Phase::Write;
            QueueCatalogJob(m_applyWork, CatalogIoStage::Persist, [this, pending] { WriteModelReimport(*pending); });
            if (m_frameWork)
                SubmitCatalogWork(m_applyWork, managed);
            return true;
        }
    } catch (const std::exception &exception) {
        m_applyWork.DropUnsubmitted();
        result.succeeded = false;
        result.errorCode = result.databaseCommitted ? AssetMutationErrorCode::RuntimeApplyFailed
                                                    : AssetMutationErrorCode::ImportFailed;
        result.error = Utf8Diagnostic(exception.what());
        INXLOG_ERROR("Model import failed for '", result.path, "': ", result.error);
    }
    pending->phase = PendingModelReimport::Phase::Complete;
    pending->writeScope.Close();
    pending->onPublished = {};
    if (pending->repairTicket) {
        BoundRegistry()->FinishMeshArtifactRepair(*pending->repairTicket, result);
    } else {
        m_lastModelReimportWorkerMilliseconds = pending->workerMilliseconds;
        m_lastModelReimportPrepareMilliseconds = pending->prepareMilliseconds;
        m_lastModelReimportPersistenceMilliseconds = pending->persistenceMilliseconds;
        m_finishedModelApply = FinishedModelApply{std::move(result), pending->IsDiscarded(), pending->abandoned};
    }
    m_pendingModelReimport.reset();
    return true;
}

void AssetDatabase::WriteModelReimport(PendingModelReimport &pending)
{
    auto &worker = pending.worker;
    const auto &path = worker.request.sourcePath;
    try {
        const auto started = std::chrono::steady_clock::now();
        RequireUnchangedFingerprint(path, worker.expectedSource);
        const auto metaPath = InxResourceMeta::GetMetaFilePath(path);
        AssetFileFingerprint metadata;
        const bool metadataExists = ReadFingerprint(ToFsPath(metaPath), metadata);
        if (metadataExists != pending.metadataExists || (metadataExists && metadata != pending.metadataFingerprint))
            throw std::runtime_error("Model import settings changed outside this Apply");
        auto writes = BuildImportWrites(worker.request, *worker.artifact, &pending.metadata.expectedMetadata.value());
        const auto writing = std::chrono::steady_clock::now();
        pending.prepareMilliseconds += std::chrono::duration<double, std::milli>(writing - started).count();
        const auto files = PersistImportWrites(worker.request, std::move(writes), pending.writeScope);
        pending.persistenceMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - writing).count();
        const auto key = LexicalFilesystemPathKey(metaPath);
        const auto committed = std::find_if(
            files.begin(), files.end(), [&](const auto &file) { return LexicalFilesystemPathKey(file.path) == key; });
        if (committed == files.end())
            throw std::logic_error("Model Apply did not persist its metadata");
        pending.committedFiles = {worker.expectedSource, {committed->size, committed->modifiedNs}, false};
    } catch (const std::exception &exception) {
        worker.error = Utf8Diagnostic(exception.what());
    } catch (...) {
        worker.error = "Model Apply persistence raised a non-standard exception";
    }
}

void AssetDatabase::PublishModelReimport(PendingModelReimport &pending)
{
    const auto started = std::chrono::steady_clock::now();
    auto &result = pending.result;
    uint64_t previousVersion = 0;
    std::optional<float> previousScale;
    if (pending.onPublished) {
        if (const auto *registry = BoundRegistry()) {
            if (const auto mesh = registry->GetAsset<InxMesh>(result.guid)) {
                previousVersion = registry->GetAssetVersion(result.guid);
                previousScale = mesh->GetImportedScale();
            }
        }
    }
    result.databaseCommitted = true;
    InstallImportArtifact(result.guid, std::move(*pending.worker.artifact));
    // Listeners now see the new snapshot and may acquire its runtime artifacts.
    pending.phase = PendingModelReimport::Phase::Publish;
    FinishReimport(result, &pending.committedFiles, pending.repairTicket);
    if (!pending.repairTicket)
        m_lastModelReimportLivePublicationMilliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (const auto published = std::exchange(pending.onPublished, {}))
        published(result, previousVersion, previousScale);
}

bool AssetDatabase::DiscardModelReimport()
{
    AssertMutationThread("DiscardModelReimport");
    if (m_finishedModelApply) {
        m_finishedModelApply->abandoned = true;
        return false;
    }
    const auto &pending = m_pendingModelReimport;
    if (!pending || pending->repairTicket)
        return false;
    pending->abandoned = true;
    if (pending->phase == PendingModelReimport::Phase::Complete || IsDeliveringModelReimport() ||
        (pending->phase == PendingModelReimport::Phase::Write && m_applyWork.submitted))
        return false;
    pending->Discard();
    if (!m_applyWork.submitted) {
        m_applyWork.DropUnsubmitted();
        pending->FinishWorker();
    }
    return true;
}

std::vector<std::string> AssetDatabase::GetDeletionWritePaths(const std::vector<std::string> &roots) const
{
    std::vector<std::string> paths;
    std::vector<std::string> rootKeys;
    for (const auto &root : roots) {
        rootKeys.push_back(FilesystemPathKey(root));
        paths.push_back(root);
        paths.push_back(InxResourceMeta::GetMetaFilePath(root));
    }
    const auto snapshot = LoadQuerySnapshot();
    for (const auto &[guid, path] : snapshot->guidToPath) {
        const auto backing = path.substr(0, path.find("::"));
        const auto key = LexicalFilesystemPathKey(backing);
        if (std::none_of(rootKeys.begin(), rootKeys.end(),
                         [&](const auto &root) { return IsFilesystemPathKeyWithin(key, root); }))
            continue;
        paths.push_back(backing);
        paths.push_back(InxResourceMeta::GetMetaFilePath(backing));
        const auto type = snapshot->metas.at(guid)->GetResourceType();
        const auto artifact = GetRuntimeArtifactPath(guid, type);
        if (!artifact.empty())
            paths.push_back(artifact);
        if (type == ResourceType::Mesh)
            paths.push_back(GetSkinnedMeshArtifactPath(guid));
    }
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    return paths;
}

AssetMutationResult AssetDatabase::DeleteAsset(const std::string &requestedPath, const PathWriteScope &parentScope)
{
    AssertMutationThread("DeleteAsset");
    AssertNoPendingCommit("DeleteAsset");
    std::string guid = GetGuidFromPath(requestedPath);
    const auto path = guid.empty() ? ResolveFilesystemPath(requestedPath) : GetPathFromGuid(guid);
    ResourceType type = GetResourceTypeForPath(path);
    AssetMutationResult result;
    result.operation = "delete";
    result.guid = guid;
    result.path = path;
    result.resourceType = type;

    const auto deletionPaths = GetDeletionWritePaths({path});
    std::vector<std::string> delegated, independent;
    for (const auto &target : deletionPaths)
        (parentScope.Covers(target) ? delegated : independent).push_back(target);
    const auto owned = PathWriteScope::BeginMutation(delegated, {}, parentScope);
    const auto derived = PathWriteScope::BeginMutation(independent);

    if (!guid.empty()) {
        const auto sourceKey = FilesystemPathKey(path);
        const auto metaKey = FilesystemPathKey(InxResourceMeta::GetMetaFilePath(path));
        for (const auto &artifactPath : deletionPaths) {
            const auto key = FilesystemPathKey(artifactPath);
            if (key == sourceKey || key == metaKey)
                continue;
            std::error_code artifactError;
            std::filesystem::remove(ToFsPath(artifactPath), artifactError);
            if (artifactError)
                throw std::runtime_error("Failed to remove runtime CPU artifact: " + artifactError.message());
        }
    }

    // Retire catalog and runtime state only after the sidecar deletion
    // succeeds. Notifications must observe a missing GUID, not reload the
    // just-evicted payload through the old catalog.
    DeleteMetadata(path);
    if (!guid.empty()) {
        RemoveMappingByGuid(guid);
    } else {
        RemoveMappingByPath(path);
    }
    if (!guid.empty())
        m_importResults.erase(guid);
    m_fileStates.erase(FilesystemPathKey(path));
    m_assetIndexDirty = true;
    if (!guid.empty()) {
        if (auto *registry = BoundRegistry())
            registry->RemoveAsset(guid);
        m_dependencyGraph->RemoveAsset(guid);
    }
    const auto previous = LoadQuerySnapshot();
    const auto changedSubAssets = PublishQuerySnapshotForPaths({path});
    result.succeeded = true;
    result.databaseCommitted = true;
    result.changed = !guid.empty();
    result.queryGeneration = GetQueryGeneration();
    PublishModelSubAssetEvents(previous, &changedSubAssets);
    if (!guid.empty())
        m_dependencyGraph->NotifyEvent(guid, type, AssetEvent::Deleted);
    return result;
}

AssetMutationResult AssetDatabase::MoveAsset(const std::string &oldPath, const std::string &newPath,
                                             const PathWriteScope &writeScope)
{
    auto results = MoveAssetsBatch({{oldPath, newPath}}, writeScope);
    auto result = std::move(results.front());
    result.operation = "move";
    return result;
}

std::vector<AssetMutationResult>
AssetDatabase::MoveAssetsBatch(const std::vector<std::pair<std::string, std::string>> &moves,
                               const PathWriteScope &parentScope)
{
    AssertMutationThread("MoveAssetsBatch");
    AssertNoPendingCommit("MoveAssetsBatch");
    if (moves.empty())
        return {};

    struct PreparedMove
    {
        std::string oldPath;
        std::string newPath;
        std::string oldKey;
        std::string newKey;
        std::string guid;
        ResourceType resourceType = ResourceType::DefaultText;
        bool reclassifySource = false;
    };

    auto failure = [](const std::string &oldPath, const std::string &newPath, AssetMutationErrorCode code,
                      const std::string &message) {
        AssetMutationResult result;
        result.operation = "move_batch";
        result.previousPath = oldPath;
        result.path = newPath;
        result.errorCode = code;
        result.error = message;
        return std::vector<AssetMutationResult>{std::move(result)};
    };

    std::vector<PreparedMove> prepared;
    prepared.reserve(moves.size());
    std::unordered_set<std::string> sourceKeys;
    std::unordered_set<std::string> destinationKeys;
    for (const auto &[oldPath, newPath] : moves) {
        const std::string oldKey = FilesystemPathKey(oldPath);
        const std::string newKey = FilesystemPathKey(newPath);
        if (oldKey.empty() || newKey.empty() ||
            NormalizeFilesystemPathLexically(oldPath) == NormalizeFilesystemPathLexically(newPath))
            return failure(oldPath, newPath, AssetMutationErrorCode::InvalidPath,
                           "asset relocation requires two different paths");
        if (!sourceKeys.insert(oldKey).second || !destinationKeys.insert(newKey).second)
            return failure(oldPath, newPath, AssetMutationErrorCode::InvalidPath,
                           "asset relocation contains duplicate source or destination paths");
        const std::string guid = GetGuidFromPath(oldPath);
        if (guid.empty())
            return failure(oldPath, newPath, AssetMutationErrorCode::NotFound,
                           "asset relocation source has no registered GUID");
        std::error_code destinationError;
        if (!std::filesystem::exists(ToFsPath(newPath), destinationError))
            return failure(oldPath, newPath,
                           destinationError ? AssetMutationErrorCode::InvalidPath : AssetMutationErrorCode::NotFound,
                           destinationError
                               ? "asset relocation destination cannot be inspected: " + destinationError.message()
                               : "asset relocation destination must exist before notification");
        const auto destination = ResolveFilesystemPath(newPath);
        prepared.push_back(
            {GetPathFromGuid(guid), destination, oldKey, newKey, guid, GetResourceTypeForPath(destination)});
    }

    const auto isSourceOnly = [](ResourceType type) {
        return type == ResourceType::Script || type == ResourceType::DefaultText || type == ResourceType::DefaultBinary;
    };
    for (PreparedMove &move : prepared) {
        if (move.oldKey != move.newKey && sourceKeys.count(move.newKey) != 0)
            return failure(move.oldPath, move.newPath, AssetMutationErrorCode::InvalidPath,
                           "asset relocation destinations cannot overlap another source in the same batch");
        const auto occupied = m_pathToGuid.find(move.newKey);
        if (occupied != m_pathToGuid.end() && occupied->second != move.guid)
            return failure(move.oldPath, move.newPath, AssetMutationErrorCode::InvalidPath,
                           "asset relocation destination is occupied by another GUID");
        const ResourceType previousType = m_metas.at(move.guid)->GetResourceType();
        if (previousType != move.resourceType) {
            // Raw sources and scripts have no native import artifacts to
            // migrate. Rebuild their metadata for the completed rename.
            if (!isSourceOnly(previousType) || !isSourceOnly(move.resourceType))
                return failure(move.oldPath, move.newPath, AssetMutationErrorCode::UnsupportedType,
                               "asset relocation cannot change its imported resource type");
            move.reclassifySource = true;
        }
    }

    std::vector<std::string> writePaths;
    for (const auto &move : prepared)
        for (const auto &path : {move.oldPath, move.newPath}) {
            writePaths.push_back(path);
            writePaths.push_back(InxResourceMeta::GetMetaFilePath(path));
        }
    const auto writeScope = PathWriteScope::BeginMutation(writePaths, {}, parentScope);
    const auto guidToPathBefore = m_guidToPath;
    const auto pathToGuidBefore = m_pathToGuid;
    const auto metasBefore = m_metas;
    const auto fileStatesBefore = m_fileStates;
    const bool assetIndexDirtyBefore = m_assetIndexDirty;
    const auto previous = LoadQuerySnapshot();
    std::unordered_set<std::string> changedSubAssets;
    std::vector<const PreparedMove *> metadataMoved;
    metadataMoved.reserve(prepared.size());
    try {
        for (const PreparedMove &move : prepared) {
            std::optional<InxResourceMeta> replacement;
            if (move.reclassifySource) {
                AssetSource source(move.newPath);
                replacement.emplace();
                m_loaders.at(move.resourceType)->CreateMeta(source, *replacement);
                replacement->AddMetadata("guid", move.guid);
            }
            MoveMetadata(move.oldPath, move.newPath, writeScope, replacement ? &*replacement : nullptr);
            metadataMoved.push_back(&move);
            UpdateMapping(move.guid, move.newPath);
            if (move.oldKey != move.newKey)
                RemoveMappingByPath(move.oldPath);
            m_fileStates.erase(move.oldKey);
            UpdateCachedFileState(move.newPath, IsReadOnlyPath(move.newKey));
        }
        m_assetIndexDirty = true;
        std::vector<std::string> changedPaths;
        changedPaths.reserve(prepared.size() * 2);
        for (const PreparedMove &move : prepared) {
            changedPaths.push_back(move.oldPath);
            changedPaths.push_back(move.newPath);
        }
        changedSubAssets = PublishQuerySnapshotForPaths(changedPaths);
    } catch (const std::exception &exc) {
        std::string rollbackFailures;
        for (auto it = metadataMoved.rbegin(); it != metadataMoved.rend(); ++it) {
            try {
                const InxResourceMeta *original = (*it)->reclassifySource ? metasBefore.at((*it)->guid).get() : nullptr;
                MoveMetadata((*it)->newPath, (*it)->oldPath, writeScope, original);
            } catch (const std::exception &rollbackError) {
                if (!rollbackFailures.empty())
                    rollbackFailures += "; ";
                rollbackFailures += (*it)->newPath + ": " + rollbackError.what();
            }
        }
        m_guidToPath = guidToPathBefore;
        m_pathToGuid = pathToGuidBefore;
        m_metas = metasBefore;
        m_fileStates = fileStatesBefore;
        m_assetIndexDirty = assetIndexDirtyBefore || !rollbackFailures.empty();
        if (!rollbackFailures.empty())
            INXLOG_ERROR("Asset relocation failed: ", exc.what(),
                         "; metadata rollback also failed: ", rollbackFailures);
        return failure(
            prepared.front().oldPath, prepared.front().newPath, AssetMutationErrorCode::RuntimeApplyFailed,
            std::string("asset relocation batch failed: ") + exc.what() +
                (rollbackFailures.empty() ? std::string{} : "; metadata rollback failed: " + rollbackFailures));
    }

    std::vector<AssetMutationResult> results;
    results.reserve(prepared.size());
    if (auto *registry = BoundRegistry()) {
        for (const PreparedMove &move : prepared) {
            if (move.reclassifySource)
                registry->InvalidateAsset(move.guid);
            else
                registry->UpdateLoadedAssetPath(move.guid, m_guidToPath.at(move.guid));
        }
    }
    PublishModelSubAssetEvents(previous, &changedSubAssets);
    for (const PreparedMove &move : prepared) {
        m_dependencyGraph->NotifyEvent(move.guid, move.resourceType, AssetEvent::Moved);
        AssetMutationResult result;
        result.succeeded = true;
        result.databaseCommitted = true;
        result.changed = true;
        result.operation = "move_batch";
        result.guid = move.guid;
        result.path = move.newPath;
        result.previousPath = move.oldPath;
        result.resourceType = move.resourceType;
        result.queryGeneration = GetQueryGeneration();
        ApplyDocumentationTextureMovePolicy(*this, result, writeScope);
        results.push_back(std::move(result));
    }
    return results;
}

bool AssetDatabase::ContainsGuid(const std::string &guid) const
{
    if (CanReadWorkingSet())
        return m_guidToPath.find(guid) != m_guidToPath.end();
    const auto snapshot = LoadQuerySnapshot();
    return snapshot->guidToPath.find(guid) != snapshot->guidToPath.end();
}

bool AssetDatabase::ContainsPath(const std::string &path) const
{
    std::string norm = AssetPathKey(path);
    if (CanReadWorkingSet())
        return m_pathToGuid.find(norm) != m_pathToGuid.end();
    const auto snapshot = LoadQuerySnapshot();
    return snapshot->pathToGuid.find(norm) != snapshot->pathToGuid.end();
}

std::string AssetDatabase::GetGuidFromPath(const std::string &path) const
{
    const std::string normalized = AssetPathKey(path);
    if (CanReadWorkingSet()) {
        const auto it = m_pathToGuid.find(normalized);
        return it != m_pathToGuid.end() ? it->second : "";
    }
    const auto snapshot = LoadQuerySnapshot();
    const auto it = snapshot->pathToGuid.find(normalized);
    return it != snapshot->pathToGuid.end() ? it->second : "";
}

std::string AssetDatabase::GetGuidFromCatalogPath(const std::string &path) const
{
    const size_t suffix = path.find("::");
    std::string normalized = LexicalFilesystemPathKey(path.substr(0, suffix));
    if (suffix != std::string::npos)
        normalized += path.substr(suffix);
    const auto snapshot = LoadQuerySnapshot();
    const auto found = snapshot->pathToGuid.find(normalized);
    return found != snapshot->pathToGuid.end() ? found->second : "";
}

std::string AssetDatabase::GetPathFromGuid(const std::string &guid) const
{
    if (CanReadWorkingSet()) {
        const auto it = m_guidToPath.find(guid);
        return it != m_guidToPath.end() ? it->second : "";
    }
    const auto snapshot = LoadQuerySnapshot();
    const auto it = snapshot->guidToPath.find(guid);
    return it != snapshot->guidToPath.end() ? it->second : "";
}

std::shared_ptr<const InxResourceMeta> AssetDatabase::GetMetaByGuid(const std::string &guid) const
{
    if (CanReadWorkingSet()) {
        const auto it = m_metas.find(guid);
        return it != m_metas.end() ? it->second : nullptr;
    }
    const auto snapshot = LoadQuerySnapshot();
    const auto it = snapshot->metas.find(guid);
    return it != snapshot->metas.end() ? it->second : nullptr;
}

std::shared_ptr<const InxResourceMeta> AssetDatabase::GetMetaByPath(const std::string &path) const
{
    if (path.empty())
        return nullptr;

    const std::string normalized = AssetPathKey(path);
    if (CanReadWorkingSet()) {
        const auto pathIt = m_pathToGuid.find(normalized);
        if (pathIt == m_pathToGuid.end())
            return nullptr;
        const auto metaIt = m_metas.find(pathIt->second);
        return metaIt != m_metas.end() ? metaIt->second : nullptr;
    }
    const auto snapshot = LoadQuerySnapshot();
    const auto pathIt = snapshot->pathToGuid.find(normalized);
    if (pathIt == snapshot->pathToGuid.end())
        return nullptr;
    const auto metaIt = snapshot->metas.find(pathIt->second);
    return metaIt != snapshot->metas.end() ? metaIt->second : nullptr;
}

std::vector<std::string> AssetDatabase::GetAllGuids() const
{
    std::vector<std::string> result;
    if (CanReadWorkingSet()) {
        result.reserve(m_guidToPath.size());
        for (const auto &[guid, path] : m_guidToPath) {
            (void)path;
            result.push_back(guid);
        }
        return result;
    }
    const auto snapshot = LoadQuerySnapshot();
    result.reserve(snapshot->guidToPath.size());
    for (const auto &pair : snapshot->guidToPath) {
        result.push_back(pair.first);
    }
    return result;
}

std::vector<std::string> AssetDatabase::GetAllAssetPaths() const
{
    std::vector<std::string> result;
    if (CanReadWorkingSet()) {
        result.reserve(m_guidToPath.size());
        for (const auto &[guid, path] : m_guidToPath) {
            (void)guid;
            result.push_back(path);
        }
        return result;
    }
    const auto snapshot = LoadQuerySnapshot();
    result.reserve(snapshot->guidToPath.size());
    for (const auto &[guid, path] : snapshot->guidToPath) {
        (void)guid;
        result.push_back(path);
    }
    return result;
}

std::shared_ptr<const AssetCatalogSnapshot> AssetDatabase::GetCatalogSnapshot() const
{
    return LoadQuerySnapshot()->catalog;
}

std::vector<AssetCatalogEntry> AssetDatabase::GetDirectoryCatalog(const std::string &directory) const
{
    const auto catalog = GetCatalogSnapshot();
    if (!catalog)
        return {};
    return catalog->GetDirectory(LexicalFilesystemPathKey(directory));
}

std::string AssetDatabase::GetImportError(const std::string &guid) const
{
    const auto snapshot = LoadQuerySnapshot();
    const auto found = snapshot->importErrors.find(guid);
    return found == snapshot->importErrors.end() ? std::string{} : found->second;
}

std::vector<AssetCatalogEntry> AssetDatabase::GetImportFailures() const
{
    const auto snapshot = LoadQuerySnapshot();
    std::vector<AssetCatalogEntry> failures;
    if (snapshot->importErrors.empty())
        return failures;
    for (const auto &[directory, entries] : snapshot->catalog->GetDirectories()) {
        (void)directory;
        for (const auto &entry : entries)
            if (!entry.importError.empty())
                failures.push_back(entry);
    }
    std::sort(failures.begin(), failures.end(),
              [](const auto &left, const auto &right) { return left.path < right.path; });
    return failures;
}

void AssetDatabase::ReportAssetLoadFailure(const std::string &guid, const std::string &error)
{
    AssertMutationThread("ReportAssetLoadFailure");
    const auto path = m_guidToPath.find(guid);
    if (path == m_guidToPath.end() || error.empty())
        return;
    if (GetImportError(guid) == error)
        return;
    m_importResults[guid] = {false, error};
    if (const auto indexed = m_assetIndex.Find(FilesystemPathKey(path->second))) {
        auto entry = *indexed;
        entry.importSucceeded = false;
        entry.importError = error;
        m_assetIndex.Upsert(std::move(entry));
        m_assetIndexDirty = true;
    }
    PublishQuerySnapshot();
}

uint64_t AssetDatabase::GetQueryGeneration() const
{
    return LoadQuerySnapshot()->generation;
}

size_t AssetDatabase::GetAssetCount() const
{
    return LoadQuerySnapshot()->guidToPath.size();
}

bool AssetDatabase::IsAssetPath(const std::string &path) const
{
    return !m_assetsRoot.empty() && IsFilesystemPathWithin(path, m_assetsRoot);
}

bool AssetDatabase::IsIgnoredImportPath(const std::filesystem::path &path)
{
    if (IsAtomicTemporaryPath(path))
        return true;
    const auto lowercase = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        return value;
    };
    for (const auto &component : path) {
        const std::string name = lowercase(FromFsPath(component));
        if (name == "__pycache__" || name == ".git" || name == ".hg" || name == ".svn")
            return true;
    }

    const std::string extension = lowercase(FromFsPath(path.extension()));
    if (extension == ".pyc" || extension == ".pyo")
        return true;

    const std::string filename = lowercase(FromFsPath(path.filename()));
    if (filename == ".gitignore" || filename == ".gitattributes" || filename == ".gitmodules" || filename == ".gitkeep")
        return true;

    // Blender's normal save flow keeps numbered copies beside the source
    // (Model.blend1, Model.blend2, ...). They are file-history generations,
    // not importable assets. Registering one can move the live model GUID to
    // the backup when Blender atomically replaces Model.blend.
    const size_t marker = filename.rfind(".blend");
    if (marker != std::string::npos && marker + 6 < filename.size() &&
        std::all_of(filename.begin() + static_cast<std::ptrdiff_t>(marker + 6), filename.end(),
                    [](unsigned char character) { return std::isdigit(character) != 0; }))
        return true;
    return false;
}

bool AssetDatabase::IsMetaFile(const std::filesystem::path &path) const
{
    return FromFsPath(path.extension()) == ".meta";
}

void AssetDatabase::UpdateMapping(const std::string &guid, const std::string &path)
{
    if (guid.empty() || path.empty())
        return;

    const std::string resolvedPath = ResolveFilesystemPath(path);
    std::string norm = FilesystemPathKey(resolvedPath);

    const auto previousPath = m_guidToPath.find(guid);
    if (previousPath != m_guidToPath.end()) {
        for (auto it = m_pathToGuid.begin(); it != m_pathToGuid.end();) {
            if (it->second == guid && it->first != norm)
                it = m_pathToGuid.erase(it);
            else
                ++it;
        }
    }

    const auto previousGuid = m_pathToGuid.find(norm);
    if (previousGuid != m_pathToGuid.end() && previousGuid->second != guid) {
        const auto reverse = m_guidToPath.find(previousGuid->second);
        if (reverse != m_guidToPath.end() && FilesystemPathKey(reverse->second) == norm)
            m_guidToPath.erase(reverse);
    }

    m_guidToPath[guid] = resolvedPath;
    m_pathToGuid[norm] = guid;
}

void AssetDatabase::RemoveMappingByGuid(const std::string &guid)
{
    for (auto it = m_pathToGuid.begin(); it != m_pathToGuid.end();) {
        if (it->second == guid)
            it = m_pathToGuid.erase(it);
        else
            ++it;
    }
    m_guidToPath.erase(guid);
}

void AssetDatabase::RemoveMappingByPath(const std::string &path)
{
    std::string norm = FilesystemPathKey(path);
    auto it = m_pathToGuid.find(norm);
    if (it != m_pathToGuid.end()) {
        m_pathToGuid.erase(it);
    }
}

void AssetDatabase::UpdateCachedFileState(const std::string &path, bool readOnly)
{
    CachedFileState state;
    if (!ReadFingerprint(ToFsPath(path), state.source)) {
        m_fileStates.erase(FilesystemPathKey(path));
        return;
    }
    state.readOnly = readOnly;
    if (!readOnly)
        ReadFingerprint(ToFsPath(InxResourceMeta::GetMetaFilePath(path)), state.meta);
    m_fileStates[FilesystemPathKey(path)] = state;
}

bool AssetDatabase::RunImporter(const std::string &guid, const std::string &path, bool isReimport, bool persistMetadata,
                                const InxResourceMeta *candidateMetadata, const AssetFileFingerprint *expectedSource,
                                const AtomicFileState *expectedMetadata, const PathWriteScope &writeScope)
{
    if (guid.empty() || path.empty())
        throw std::invalid_argument("AssetDatabase importer request requires GUID and path");

    std::string ext = FromFsPath(ToFsPath(path).extension());
    AssetImporter *importer = m_importerRegistry.GetImporterForExtension(ext);
    if (!importer && !candidateMetadata) {
        m_importResults[guid] = {true, {}};
        return true;
    }

    const auto metaIt = m_metas.find(guid);
    if (metaIt == m_metas.end() || !metaIt->second)
        throw std::logic_error("AssetDatabase importer request has no metadata snapshot");

    std::string error;
    try {
        const ImportRequest request =
            MakeImportRequest(guid, path, isReimport, candidateMetadata ? *candidateMetadata : *metaIt->second);
        // Ordinary scripts publish only their source metadata here; their
        // compiler and dependency transaction are owned by Python.
        ImportArtifact artifact = (!importer || request.resourceType == ResourceType::Script)
                                      ? ImportArtifact(request.metadata)
                                      : (isReimport ? importer->Reimport(request) : importer->Import(request));
        if (expectedSource)
            RequireUnchangedFingerprint(path, *expectedSource);
        if (persistMetadata && !expectedMetadata)
            throw std::logic_error("Author metadata publication requires its captured input state");
        ValidateImportPublication(request, artifact);
        auto writes = BuildImportWrites(request, artifact, persistMetadata ? expectedMetadata : nullptr);
        PersistImportWrites(request, std::move(writes), writeScope);
        InstallImportArtifact(guid, std::move(artifact));
        return true;
    } catch (const std::exception &exception) {
        error = Utf8Diagnostic(exception.what());
    } catch (...) {
        error = "Importer raised a non-standard exception";
    }
    m_importResults[guid] = {false, error};
    INXLOG_ERROR("Asset import failed for '", path, "': ", error);
    return false;
}

ImportRequest AssetDatabase::MakeImportRequest(const std::string &guid, const std::string &path, bool isReimport,
                                               const InxResourceMeta &metadata) const
{
    auto request = MakeImportRequestBase(guid, path, isReimport, metadata);
    if (request.resourceType == ResourceType::Mesh)
        request.resolveTextureGuid = ModelTextureResolver(LoadQuerySnapshot());
    const auto owner = CopiedSkeletonOwner(request);
    if (!owner.empty()) {
        const auto definitionMeta = m_metas.find(owner);
        const auto sourceHash =
            SkeletonSourceHash(definitionMeta == m_metas.end() ? nullptr : definitionMeta->second.get());
        const auto result = m_importResults.find(owner);
        if (result != m_importResults.end() && !result->second.succeeded)
            throw std::invalid_argument("copied skeleton definition import failed: " + owner);
        request.skeletonDefinition = CaptureSkeletonDefinition(sourceHash, GetSkinnedMeshArtifactPath(owner), true);
    }
    return request;
}

ImportRequest AssetDatabase::MakeImportRequestBase(const std::string &guid, const std::string &path, bool isReimport,
                                                   const InxResourceMeta &metadata) const
{
    ImportRequest request;
    request.sourcePath = path;
    request.projectRoot = m_projectRoot;
    request.blenderExecutable = m_blenderExecutable;
    request.blenderExportScript = m_blenderExportScript;
    request.guid = guid;
    request.resourceType = GetResourceTypeForPath(path);
    request.metadata = metadata;
    request.metadata.AddMetadata("file_path", InxResourceMeta::NormalizeFilePath(path));
    request.isReimport = isReimport;
    return request;
}

void AssetDatabase::ValidateImportPublication(const ImportRequest &request, ImportArtifact &artifact) const
{
    if (m_metas.find(request.guid) == m_metas.end())
        throw std::logic_error("Import publication has no registered metadata");
    ResolveImportedDependencyPathHints(artifact, m_pathToGuid, request.sourcePath);
    ValidateImportedDependencyIdentities(artifact, request.sourcePath);
    ValidateModelMaterialTargets(artifact, [this](const std::string &dependency) { return GetMetaByGuid(dependency); });
}

std::vector<DocumentTransactionEntry> AssetDatabase::BuildImportWrites(const ImportRequest &request,
                                                                       ImportArtifact &artifact,
                                                                       const AtomicFileState *expectedMetadata)
{
    std::vector<DocumentTransactionEntry> writes;
    writes.reserve(1 + artifact.runtimeCpuArtifacts.size());
    if (expectedMetadata) {
        writes.push_back({InxResourceMeta::GetMetaFilePath(request.sourcePath),
                          artifact.metadata.SerializeDocumentPortable(request.projectRoot).dump(4) + "\n",
                          *expectedMetadata});
    }
    auto runtimeArtifactWrites = TakeRuntimeArtifactWrites(artifact.runtimeCpuArtifacts, request.guid,
                                                           request.resourceType, request.projectRoot);
    for (auto &runtimeArtifactWrite : runtimeArtifactWrites)
        writes.push_back(std::move(runtimeArtifactWrite));
    return writes;
}

std::vector<DocumentTransactionFileState>
AssetDatabase::PersistImportWrites(const ImportRequest &request, std::vector<DocumentTransactionEntry> writes,
                                   const PathWriteScope &writeScope) const
{
    if (writes.empty())
        return {};
    if (IsFilesystemPathWithin(request.sourcePath, m_projectRoot)) {
        return DocumentTransaction::Commit(m_projectRoot, m_assetTransactionJournalPath, std::move(writes),
                                           {m_assetIndexPath}, writeScope)
            .committedFileStates;
    }

    std::vector<PathWriteScope> outputScopes;
    for (const auto &write : writes)
        outputScopes.push_back(
            PathWriteScope::BeginWrite({write.path}, writeScope.Covers(write.path) ? writeScope : PathWriteScope{}));
    outputScopes.push_back(PathWriteScope::BeginWrite(
        {m_assetIndexPath}, writeScope.Covers(m_assetIndexPath) ? writeScope : PathWriteScope{}));
    std::vector<DocumentTransactionFileState> files;
    files.reserve(writes.size());
    for (auto &write : writes) {
        const std::filesystem::path parent = ToFsPath(write.path).parent_path();
        if (!parent.empty()) {
            std::error_code directoryError;
            std::filesystem::create_directories(NativeFilePath(parent), directoryError);
            if (directoryError)
                throw std::runtime_error("Failed to create external import target directory: " +
                                         directoryError.message());
        }
        DocumentWriteOptions options;
        options.expectedFileState = write.expectedFileState;
        if (writeScope.Covers(write.path))
            options.writeScope = writeScope;
        auto ticket = DocumentStore::Instance().Submit(write.path, std::move(write.content), options);
        ticket->Wait();
        const auto state = ticket->GetCommittedFileState().value();
        files.push_back({std::move(write.path), state.size, state.modifiedNs});
    }
    std::error_code indexError;
    std::filesystem::remove(ToFsPath(m_assetIndexPath), indexError);
    if (indexError)
        throw std::runtime_error("Failed to invalidate AssetIndex after external import: " + indexError.message());
    return files;
}

void AssetDatabase::InstallImportArtifact(const std::string &guid, ImportArtifact artifact)
{
    if (artifact.dependenciesAuthoritative) {
        const std::unordered_set<std::string> dependencies(artifact.dependencies.begin(), artifact.dependencies.end());
        m_dependencyGraph->SetAssetDependencies(guid, dependencies);
    }
    m_metas.at(guid) = std::make_shared<InxResourceMeta>(std::move(artifact.metadata));
    m_importResults[guid] = {true, {}};
}

// ============================================================================
// Resource management
// ============================================================================

bool AssetDatabase::ReadFile(const std::string &filePath, std::vector<char> &content) const
{
    std::ifstream file(ToFsPath(filePath), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        INXLOG_ERROR("Failed to open file: ", filePath);
        content.clear();
        return false;
    }

    try {
        file.seekg(0, std::ios::end);
        if (file.fail()) {
            INXLOG_ERROR("Failed to seek to end of file: ", filePath);
            content.clear();
            return false;
        }

        std::streampos fileSize = file.tellg();
        if (fileSize == std::streampos(-1)) {
            INXLOG_ERROR("Failed to get file size: ", filePath);
            content.clear();
            return false;
        }

        file.seekg(0, std::ios::beg);
        if (file.fail()) {
            INXLOG_ERROR("Failed to seek to beginning of file: ", filePath);
            content.clear();
            return false;
        }

        content.reserve(static_cast<size_t>(fileSize));
        content.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        if (file.bad() || file.fail()) {
            INXLOG_ERROR("Error occurred while reading file: ", filePath);
            content.clear();
            return false;
        }

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("Exception while reading file: ", filePath, " - ", e.what());
        content.clear();
        return false;
    }
}

ResourceType AssetDatabase::GetResourcesType(const std::string &extensionName) const
{
    std::string ext = extensionName;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".vert" || ext == ".frag") {
        return ResourceType::Shader;
    }
    if (ext == ".mat") {
        return ResourceType::Material;
    }
    if (ext == ".physicmaterial") {
        return ResourceType::PhysicMaterial;
    }
    if (ext == ".effect" || ext == ".effectgroup") {
        return ResourceType::RenderEffect;
    }
    if (ext == ".particlegraph") {
        return ResourceType::ParticleGraph;
    }
    if (ext == ".inxdata") {
        return ResourceType::DataAsset;
    }
    if (ext == ".rendertexture") {
        return ResourceType::RenderTexture;
    }
    if (ext == ".meta") {
        return ResourceType::Meta;
    }
    if (ext == ".py") {
        return ResourceType::Script;
    }
    if (asset_formats::Contains(asset_formats::kTextureExtensions, ext)) {
        return ResourceType::Texture;
    }
    if (asset_formats::Contains(asset_formats::kAudioExtensions, ext)) {
        return ResourceType::Audio;
    }
    if (asset_formats::Contains(asset_formats::kMeshExtensions, ext)) {
        return ResourceType::Mesh;
    }
    if (asset_formats::Contains(asset_formats::kBinaryExtensions, ext)) {
        return ResourceType::DefaultBinary;
    }
    return ResourceType::DefaultText;
}

ResourceType AssetDatabase::GetResourceTypeForPath(const std::string &filePath) const
{
    std::filesystem::path path = ToFsPath(filePath);
    std::string ext = FromFsPath(path.extension());
    return GetResourcesType(ext);
}

std::string AssetDatabase::CreateOrLoadMetadata(const std::string &filePath, ResourceType type, bool readOnly,
                                                bool persistMetadata, const std::string &identityKey,
                                                std::optional<AtomicFileState> *expectedMetadata,
                                                const PathWriteScope &writeScope)
{
    if (filePath.empty()) {
        INXLOG_ERROR("Received empty filePath!");
        return "";
    }

    auto loader = m_loaders.find(type);
    if (loader == m_loaders.end()) {
        INXLOG_ERROR("Resource type not supported: ", static_cast<int>(type));
        return "";
    }

    WorkerMetadataPrepare candidate;
    candidate.writeScope = writeScope;
    candidate.file.path = filePath;
    candidate.file.normalizedPath = FilesystemPathKey(filePath);
    candidate.file.identityKey = identityKey;
    candidate.file.readOnly = readOnly;
    candidate.resourceType = type;
    candidate.loader = loader->second;
    candidate.fallbackGuid = GetGuidFromPath(filePath);
    if (!ReadFingerprint(ToFsPath(filePath), candidate.file.source)) {
        INXLOG_ERROR("Failed to inspect source for resource registration: ", filePath);
        return "";
    }
    PrepareMetadata(candidate);
    if (!candidate.error.empty()) {
        INXLOG_ERROR("Failed to prepare resource metadata: ", filePath, ": ", candidate.error);
        return "";
    }
    const std::string &preservedGuid = candidate.fallbackGuid;
    const auto validateIdentity = [&](const std::string &guid) {
        const auto registered = m_guidToPath.find(guid);
        if (registered != m_guidToPath.end() && FilesystemPathKey(registered->second) != FilesystemPathKey(filePath))
            throw std::runtime_error("Asset GUID is already registered at another path: " + guid + " ('" +
                                     registered->second + "', '" + filePath + "')");
        if (!preservedGuid.empty() && guid != preservedGuid)
            throw std::runtime_error("Asset metadata changed the registered GUID at path: " + filePath);
    };
    validateIdentity(candidate.metadata->GetGuid());
    if (!readOnly && persistMetadata) {
        PublishMissingMetadata(candidate, m_projectRoot);
        if (!candidate.error.empty()) {
            INXLOG_ERROR("Failed to publish resource metadata: ", filePath, ": ", candidate.error);
            return "";
        }
    }
    const auto &metaFile = *candidate.metadata;
    const std::string guid = metaFile.GetGuid();
    validateIdentity(guid);
    const std::string metaFilePath = InxResourceMeta::GetMetaFilePath(filePath);
    AtomicFileState metadataState = candidate.expectedMetadata.value_or(AtomicFileState{});

    // Validate ownership before the first durable write or live catalog edit.
    if (!readOnly && persistMetadata) {
        auto content = metaFile.SerializeDocumentPortable(m_projectRoot).dump(4) + "\n";
        if (content != candidate.metadataSource) {
            DocumentWriteOptions options;
            options.expectedFileState = metadataState;
            options.writeScope = writeScope;
            const auto ticket = DocumentStore::Instance().Submit(metaFilePath, std::move(content), options);
            ticket->Wait();
            metadataState = ticket->GetCommittedFileState().value();
        }
    }
    if (expectedMetadata && !readOnly)
        *expectedMetadata = metadataState;
    m_metas[guid] = candidate.metadata;
    UpdateMapping(guid, filePath);

    return guid;
}

void AssetDatabase::DeleteMetadata(const std::string &path)
{
    namespace fs = std::filesystem;

    std::string metaPath = InxResourceMeta::GetMetaFilePath(path);
    auto metaFsPath = ToFsPath(metaPath);
    if (fs::exists(metaFsPath)) {
        fs::remove(metaFsPath);
    }
    const auto pathIt = m_pathToGuid.find(FilesystemPathKey(path));
    if (pathIt != m_pathToGuid.end())
        m_metas.erase(pathIt->second);
}

void AssetDatabase::MoveMetadata(const std::string &oldPath, const std::string &newPath,
                                 const PathWriteScope &writeScope, const InxResourceMeta *replacement)
{
    namespace fs = std::filesystem;
    const auto guid = m_pathToGuid.at(FilesystemPathKey(oldPath));
    const auto current = m_metas.at(guid);
    const std::string oldMetaPath = InxResourceMeta::GetMetaFilePath(oldPath);
    const std::string newMetaPath = InxResourceMeta::GetMetaFilePath(newPath);
    const bool sameMetadata = FilesystemPathKey(oldMetaPath) == FilesystemPathKey(newMetaPath);
    const bool oldExists = fs::exists(ToFsPath(oldMetaPath));
    AtomicFileState destinationState;
    InxResourceMeta meta = *current;
    if (oldExists) {
        AtomicFileState sourceState;
        meta = LoadMetadataDocument(oldMetaPath, &sourceState);
        if (meta.GetGuid() != guid || meta.GetResourceType() != current->GetResourceType())
            throw std::runtime_error("asset relocation source metadata has a different identity: " + oldPath);
        if (!sameMetadata && fs::exists(ToFsPath(newMetaPath)))
            throw std::runtime_error("asset relocation destination metadata already exists: " + newPath);
        if (sameMetadata) {
            if (ToFsPath(oldMetaPath).filename() != ToFsPath(newMetaPath).filename())
                fs::rename(ToFsPath(oldMetaPath), ToFsPath(newMetaPath));
            destinationState = sourceState;
        }
    } else if (fs::exists(ToFsPath(newMetaPath))) {
        // A directory relocation has already moved the sidecar on disk.
        meta = LoadMetadataDocument(newMetaPath, &destinationState);
        if (meta.GetGuid() != guid || meta.GetResourceType() != current->GetResourceType())
            throw std::runtime_error("asset relocation destination metadata has a different identity: " + newPath);
    }
    if (replacement)
        meta = *replacement;
    meta.UpdateFilePath(newPath);
    DocumentWriteOptions options;
    options.expectedFileState = destinationState;
    options.writeScope = writeScope;
    DocumentStore::Instance()
        .Submit(newMetaPath, meta.SerializeDocumentPortable(m_projectRoot).dump(4) + "\n", options)
        ->Wait();
    if (oldExists && !sameMetadata) {
        std::error_code error;
        if (!fs::remove(ToFsPath(oldMetaPath), error) || error) {
            std::error_code cleanupError;
            fs::remove(ToFsPath(newMetaPath), cleanupError);
            throw std::runtime_error("Failed to remove old asset metadata: " + oldMetaPath);
        }
    }
    // Portable sidecars omit derived import state (hashes, model children,
    // artifact descriptions). Relocation changes location, not that state.
    auto movedMeta = std::make_shared<InxResourceMeta>(replacement ? *replacement : *current);
    movedMeta->UpdateFilePath(newPath);
    m_metas[guid] = std::move(movedMeta);
}

std::string AssetDatabase::FindShaderPathById(const std::string &shaderId, const std::string &shaderType) const
{
    const char *stage;
    if (shaderType == "vertex" || shaderType == ".vert" || shaderType == "vert") {
        stage = "vertex";
    } else if (shaderType == "fragment" || shaderType == ".frag" || shaderType == "frag") {
        stage = "fragment";
    } else {
        return "";
    }

    const auto snapshot = LoadQuerySnapshot();
    const auto found = snapshot->shaderPaths.find(std::string(stage) + ":" + shaderId);
    return found == snapshot->shaderPaths.end() ? std::string{} : found->second;
}

void AssetDatabase::BuildShaderPathIndex(QuerySnapshot &snapshot)
{
    snapshot.shaderPaths.clear();
    for (const auto &[guid, meta] : snapshot.metas) {
        if (!meta)
            continue;
        auto key = ShaderPathIndexKey(*meta);
        if (key.empty())
            continue;
        const auto path = snapshot.guidToPath.find(guid);
        if (path != snapshot.guidToPath.end())
            snapshot.shaderPaths.try_emplace(std::move(key), path->second);
    }
}

} // namespace infernux
