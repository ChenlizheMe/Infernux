#include "MeshLoader.h"

#include "InxMesh.h"
#include "MeshArtifact.h"

#include <core/log/InxLog.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <platform/filesystem/InxPath.h>

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <filesystem>
#include <fstream>
#include <set>

namespace infernux
{
#if defined(INFERNUX_RUNTIME_MINIMAL_HOST)
void MeshLoader::CreateMeta(const char *, size_t, const std::string &, InxResourceMeta &) const
{
    throw std::logic_error("Mesh source import is unavailable in a minimal Player host");
}
#endif

namespace
{
std::string ReadArtifactBytes(const std::string &path, std::string_view label)
{
    std::ifstream file(ToFsPath(path), std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("failed to open " + std::string(label));
    const auto size = file.tellg();
    if (size <= 0)
        throw std::runtime_error(std::string(label) + " is empty");
    std::string bytes(static_cast<size_t>(size), '\0');
    file.seekg(0);
    if (!file.read(bytes.data(), size))
        throw std::runtime_error("failed to read " + std::string(label));
    return bytes;
}
} // namespace

RuntimeAssetPayload MeshLoader::Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb)
{
    if (!adb)
        throw std::invalid_argument("MeshLoader requires an AssetDatabase");
    auto metadata = adb->GetMetaByGuid(guid);
    if (!metadata)
        throw std::invalid_argument("MeshLoader could not resolve metadata for GUID: " + guid);
    if (!metadata->HasKey("content_hash"))
        throw std::invalid_argument("MeshLoader metadata has no source content hash");

    const std::string sourceHash = metadata->GetDataAs<std::string>("content_hash");
    const std::string artifactPath = adb->GetRuntimeArtifactPath(guid, ResourceType::Mesh);
    const std::string skinnedArtifactPath = adb->GetSkinnedMeshArtifactPath(guid);
    if (artifactPath.empty() || !std::filesystem::is_regular_file(ToFsPath(artifactPath)))
        throw std::runtime_error("MeshLoader requires a current imported .inxmesh artifact; reimport '" + filePath +
                                 "'");

    auto mesh = MeshArtifact::Deserialize(ReadArtifactBytes(artifactPath, "Mesh artifact"), sourceHash);
    auto skinned = SkinnedMeshArtifact::Deserialize(
        ReadArtifactBytes(skinnedArtifactPath, "skinned Mesh companion artifact"), sourceHash);
    if (skinned) {
        skinned->guid = guid;
        skinned->sourcePath = filePath;
    }
    mesh->SetSkinnedData(std::move(skinned));
    mesh->SetGuid(guid);
    mesh->SetFilePath(filePath);
    return mesh;
}

bool MeshLoader::Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                        AssetDatabase *adb)
{
    INXLOG_INFO("MeshLoader::Reload: '", filePath, "'");
    auto freshData = Load(filePath, guid, adb);
    if (!freshData)
        return false;
    auto loaded = freshData.Get<InxMesh>();
    auto target = existing.Get<InxMesh>();
    if (!target)
        return false;

    target->SetName(loaded->GetName());
    target->SetFilePath(loaded->GetFilePath());
    target->SetData(std::vector<Vertex>(loaded->GetVertices()), std::vector<uint32_t>(loaded->GetIndices()),
                    std::vector<SubMesh>(loaded->GetSubMeshes()));
    target->SetMaterialSlotNames(std::vector<std::string>(loaded->GetMaterialSlotNames()));
    target->SetMaterialSlotData(std::vector<MaterialSlotData>(loaded->GetMaterialSlotData()));
    target->SetNodeNames(std::vector<std::string>(loaded->GetNodeNames()));
    target->SetSkinnedData(loaded->GetSkinnedData());
    INXLOG_INFO("MeshLoader::Reload: updated '", target->GetName(), "' in-place");
    return true;
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
    if (filePath.empty())
        return paths;

    const auto sourcePath = ToFsPath(filePath);
    if (!std::filesystem::is_regular_file(sourcePath))
        return paths;

    // Model import is the authoring boundary where mutable paths may be
    // resolved to authoritative GUIDs.  Keep runtime loaders GUID-only.
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(
        FromFsPath(sourcePath), aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
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
                std::filesystem::path candidate =
                    std::filesystem::u8path(blenderRelative ? authoredPath.substr(2) : authoredPath);
                if (candidate.is_relative())
                    candidate = sourceDirectory / candidate;
                candidate = candidate.lexically_normal();
                if (!std::filesystem::is_regular_file(candidate))
                    continue;
                paths.insert(FromFsPath(candidate));
            }
        }
    }
    return paths;
}

std::set<std::string> MeshLoader::ScanDependencies(const std::string &filePath, AssetDatabase *adb)
{
    std::set<std::string> dependencies;
    if (!adb)
        return dependencies;
    for (const auto &path : ScanExternalTexturePaths(filePath)) {
        const std::string guid = adb->GetGuidFromPath(path);
        if (!guid.empty())
            dependencies.insert(guid);
    }
    return dependencies;
}

} // namespace infernux
