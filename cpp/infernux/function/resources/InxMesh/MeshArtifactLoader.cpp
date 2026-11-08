#include "MeshLoader.h"

#include "InxMesh.h"
#include "MeshArtifact.h"
#include "MeshImportSettings.h"

#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/ArtifactFileReader.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <platform/filesystem/InxPath.h>

#if !defined(INFERNUX_RUNTIME_MINIMAL_HOST)
#include "ModelSourceIO.h"
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#endif

#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <vector>

namespace infernux
{
void MeshLoader::OnRelocated(const RuntimeAssetPayload &payload, const std::string &filePath)
{
    const auto mesh = payload.Get<InxMesh>();
    mesh->SetFilePath(filePath);
    mesh->SetName(FromFsPath(ToFsPath(filePath).stem()));
}

#if defined(INFERNUX_RUNTIME_MINIMAL_HOST)
void MeshLoader::CreateMeta(AssetSource &source, InxResourceMeta &metadata) const
{
    throw std::logic_error("Mesh source import is unavailable in a minimal Player host");
}
#endif

namespace
{
struct MeshArtifactSource
{
    std::string primaryPath, skinnedPath, sourceHash, guid, filePath;
    bool cpuReadable;
    float importedScale;
    std::shared_ptr<const InxResourceMeta> metadata;
};

struct ArtifactLayout
{
    uint64_t workspace;
    uint64_t fileBytes;
};

template <typename Artifact> ArtifactLayout DescribeArtifactLayout(const std::string &path)
{
    FileInput file(ToFsPath(path));
    return {Artifact::ReadWorkspace(file), file.Size()};
}

class MeshArtifactDecode
{
  public:
    explicit MeshArtifactDecode(MeshArtifactSource source)
        : m_source(std::move(source)), m_primaryLayout(DescribeArtifactLayout<MeshArtifact>(m_source.primaryPath)),
          m_companionLayout(DescribeArtifactLayout<SkinnedMeshArtifact>(m_source.skinnedPath)),
          m_blockBytes(static_cast<size_t>(
              (std::min<uint64_t>)(ArtifactFileReader::BlockBytes,
                                   (std::max)(m_primaryLayout.fileBytes, m_companionLayout.fileBytes))))
    {
        const auto primary = m_primaryLayout.workspace;
        const auto companion = m_companionLayout.workspace;
        const uint64_t overhead = sizeof(*this) + m_blockBytes + m_source.primaryPath.capacity() +
                                  m_source.skinnedPath.capacity() + m_source.sourceHash.capacity() +
                                  3 * (m_source.guid.size() + m_source.filePath.size() + 64);
        const auto maximum = (std::numeric_limits<uint64_t>::max)();
        if (companion > maximum - overhead || primary > maximum - overhead - companion)
            throw std::length_error("model decode workspace exceeds the host byte range");
        m_workspace = primary + companion + overhead;
    }

    [[nodiscard]] uint64_t Workspace() const noexcept
    {
        return m_workspace;
    }

    [[nodiscard]] WorkerAssetLoadPlan::Read NextRead() const noexcept
    {
        using Kind = WorkerAssetLoadPlan::ReadKind;
        if (m_mesh)
            return {m_companionLayout.fileBytes, Kind::SkeletonArtifact};
        return {m_primaryLayout.fileBytes, Kind::GeometryArtifact};
    }

    RuntimeAssetPayload Advance()
    {
        if (!m_mesh) {
            FileInput primary(ToFsPath(m_source.primaryPath));
            m_companion.emplace(ToFsPath(m_source.skinnedPath));
            m_block.resize(m_blockBytes);
            m_mesh = MeshArtifact::Read(primary, m_source.sourceHash, m_block, m_primaryLayout.workspace);
            return {};
        }

        auto skinned =
            SkinnedMeshArtifact::Read(*m_companion, m_source.sourceHash, m_block, m_companionLayout.workspace);
        m_companion.reset();
        if (skinned) {
            skinned->guid = m_source.guid;
            skinned->sourcePath = m_source.filePath;
        }
        m_mesh->SetCpuReadable(m_source.cpuReadable);
        m_mesh->SetImportedScale(m_source.importedScale);
        m_mesh->SetImportMetadata(std::move(m_source.metadata));
        m_mesh->SetSkinnedData(std::move(skinned));
        m_mesh->SetGuid(m_source.guid);
        m_mesh->SetFilePath(m_source.filePath);
        return std::move(m_mesh);
    }

  private:
    MeshArtifactSource m_source;
    ArtifactLayout m_primaryLayout;
    ArtifactLayout m_companionLayout;
    std::optional<FileInput> m_companion;
    size_t m_blockBytes;
    uint64_t m_workspace;
    std::vector<char> m_block;
    std::shared_ptr<InxMesh> m_mesh;
};

MeshArtifactSource DescribeMeshLoad(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    if (!adb)
        throw std::invalid_argument("MeshLoader requires an AssetDatabase");
    auto metadata = adb->GetMetaByGuid(guid);
    if (!metadata)
        throw std::invalid_argument("MeshLoader could not resolve metadata for GUID: " + guid);
    if (!metadata->HasKey("content_hash"))
        throw std::invalid_argument("MeshLoader metadata has no source content hash");

    auto primary = adb->GetRuntimeArtifactPath(guid, ResourceType::Mesh);
    if (primary.empty())
        throw std::runtime_error("MeshLoader requires a current imported .inxmesh artifact; reimport '" + filePath +
                                 "'");
    return {std::move(primary),
            adb->GetSkinnedMeshArtifactPath(guid),
            metadata->GetDataAs<std::string>("content_hash"),
            guid,
            filePath,
            MeshImportSettings::Read(*metadata).isReadable,
            metadata->GetDataAs<float>("effective_scale"),
            std::move(metadata)};
}
} // namespace

RuntimeAssetPayload MeshLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    MeshArtifactDecode decode(DescribeMeshLoad(filePath, guid, adb));
    RuntimeAssetPayload payload;
    while (!payload)
        payload = decode.Advance();
    return payload;
}

WorkerAssetLoadPlan MeshLoader::PlanWorkerLoad(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    auto decode = std::make_shared<MeshArtifactDecode>(DescribeMeshLoad(filePath, guid, adb));
    return {decode->Workspace(),
            [decode](uint64_t, WorkerAssetLoadPlan::Read &next) {
                auto payload = decode->Advance();
                if (!payload)
                    next = decode->NextRead();
                return payload;
            },
            decode->NextRead()};
}

size_t MeshLoader::EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const
{
    const auto mesh = payload.Get<InxMesh>();
    if (!mesh)
        throw std::invalid_argument("MeshLoader cannot estimate an empty runtime payload");
    return mesh->GetRuntimeMemoryBytes();
}

std::set<std::string> MeshLoader::ScanExternalTexturePaths(const std::string &filePath)
{
    std::set<std::string> paths;
#if defined(INFERNUX_RUNTIME_MINIMAL_HOST)
    // A cooked Player consumes GUID dependencies published by the importer.
    // It never parses mutable model sources and therefore must not carry
    // Assimp or its generated configuration into the runtime host.
    (void)filePath;
    return paths;
#else
    if (filePath.empty())
        return paths;

    const auto sourcePath = ToFsPath(filePath);
    if (!std::filesystem::is_regular_file(sourcePath))
        return paths;

    // Model import is the authoring boundary where mutable paths may be
    // resolved to authoritative GUIDs.  Keep runtime loaders GUID-only.
    Assimp::Importer importer;
    const aiScene *scene = ReadModelSource(
        importer, filePath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    if (!scene || !scene->mMaterials)
        return paths;

    const std::filesystem::path sourceDirectory = sourcePath.parent_path();
    for (unsigned int materialIndex = 0; materialIndex < scene->mNumMaterials; ++materialIndex) {
        const aiMaterial *material = scene->mMaterials[materialIndex];
        if (!material)
            continue;
        // Scan every texture semantic.  The runtime material may choose only
        // a subset today, but Cook must not silently omit a source texture
        // that belongs to a composite model asset.
        for (unsigned int textureType = aiTextureType_NONE + 1; textureType <= aiTextureType_UNKNOWN; ++textureType) {
            const auto semantic = static_cast<aiTextureType>(textureType);
            const unsigned int count = material->GetTextureCount(semantic);
            for (unsigned int textureIndex = 0; textureIndex < count; ++textureIndex) {
                aiString texturePath;
                if (material->GetTexture(semantic, textureIndex, &texturePath) != AI_SUCCESS)
                    continue;
                const std::string authoredPath = texturePath.C_Str();
                // Embedded blobs are part of the source container.  They do
                // not have a project GUID until the importer gains explicit
                // embedded-texture extraction; do not invent one from the
                // literal '*N' token.
                if (authoredPath.empty() || authoredPath.front() == '*')
                    continue;

                // Blender writes project-relative external paths with a
                // leading "//".  Strip that authoring marker before asking
                // std::filesystem to classify the path; on Windows the raw
                // spelling would otherwise look like a UNC path.
                const bool blenderRelative = authoredPath.rfind("//", 0) == 0;
                std::string normalizedPath = blenderRelative ? authoredPath.substr(2) : authoredPath;
                // Assimp preserves Windows-authored OBJ/MTL separators even
                // when the same source is imported on Linux.  Treat both
                // separators as path separators at this authoring boundary;
                // runtime identities remain canonical engine paths.
                std::replace(normalizedPath.begin(), normalizedPath.end(), '\\', '/');
                std::filesystem::path candidate = std::filesystem::u8path(normalizedPath);
                if (candidate.is_relative())
                    candidate = sourceDirectory / candidate;
                // Keep filesystem identity/normalization in the shared InxPath
                // boundary.  Do not let an importer create its own lexical
                // spelling that can diverge from AssetDatabase keys.
                candidate = ToFsPath(NormalizeFilesystemPathLexically(FromFsPath(candidate)));
                if (!std::filesystem::is_regular_file(candidate))
                    continue;
                paths.insert(FromFsPath(candidate));
            }
        }
    }
    return paths;
#endif
}

std::set<std::string> MeshLoader::ScanDependencies(const std::string &filePath, AssetDatabase *adb)
{
    std::set<std::string> dependencies;
    if (!adb)
        return dependencies;
    const auto metadata = adb->GetMetaByGuid(adb->GetGuidFromPath(filePath));
    if (metadata) {
        const auto settings = MeshImportSettings::Read(*metadata);
        if (settings.materialImportMode == "none")
            return dependencies;
        for (const auto &[source, guid] : settings.materialRemaps.items())
            dependencies.insert(guid.get<std::string>());
    }
    for (const auto &path : ScanExternalTexturePaths(filePath)) {
        const std::string guid = adb->GetGuidFromPath(path);
        if (!guid.empty())
            dependencies.insert(guid);
    }
    return dependencies;
}

} // namespace infernux
