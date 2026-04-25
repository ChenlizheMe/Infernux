#include "SkinnedModelCache.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <core/log/InxLog.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace infernux
{

namespace
{

static glm::mat4 AiToGlm(const aiMatrix4x4 &m)
{
    return glm::mat4(m.a1, m.b1, m.c1, m.d1, m.a2, m.b2, m.c2, m.d2, m.a3, m.b3, m.c3, m.d3, m.a4, m.b4, m.c4, m.d4);
}

static glm::vec3 AiToGlm(const aiVector3D &v)
{
    return glm::vec3(v.x, v.y, v.z);
}

static glm::quat AiToGlm(const aiQuaternion &q)
{
    return glm::quat(q.w, q.x, q.y, q.z);
}

static bool ReadFileBytes(const std::string &filePath, std::vector<char> &out)
{
    auto fsPath = ToFsPath(filePath);
    if (!std::filesystem::exists(fsPath))
        return false;

    std::ifstream file(fsPath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return false;

    auto fileSize = file.tellg();
    if (fileSize <= 0)
        return false;

    out.resize(static_cast<size_t>(fileSize));
    file.seekg(0);
    file.read(out.data(), fileSize);
    return true;
}

static std::string ResolveSourcePath(const std::string &sourceGuid, const std::string &sourcePath)
{
    auto *adb = AssetRegistry::Instance().GetAssetDatabase();
    if (adb && !sourceGuid.empty()) {
        const std::string resolved = adb->GetPathFromGuid(sourceGuid);
        if (!resolved.empty() && std::filesystem::exists(ToFsPath(resolved)))
            return resolved;
    }

    if (!sourcePath.empty() && std::filesystem::exists(ToFsPath(sourcePath)))
        return sourcePath;
    return sourcePath;
}

static float ReadScaleFactor(const std::string &sourceGuid, const std::string &sourcePath)
{
    auto *adb = AssetRegistry::Instance().GetAssetDatabase();
    if (!adb)
        return 0.01f;

    const InxResourceMeta *meta = nullptr;
    if (!sourceGuid.empty())
        meta = adb->GetMetaByGuid(sourceGuid);
    if (!meta && !sourcePath.empty())
        meta = adb->GetMetaByPath(sourcePath);
    if (meta && meta->HasKey("scale_factor"))
        return meta->GetDataAs<float>("scale_factor");
    return 0.01f;
}

static unsigned int BuildAssimpFlags()
{
    return aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | aiProcess_FlipUVs |
           aiProcess_JoinIdenticalVertices | aiProcess_SortByPType | aiProcess_ValidateDataStructure |
           aiProcess_ImproveCacheLocality;
}

static void CollectNodes(const aiNode *node, int parent, const glm::mat4 &parentGlobal, InxSkinnedMesh &out)
{
    const int index = static_cast<int>(out.nodes.size());
    SkinnedRuntimeNode rn;
    rn.name = node->mName.C_Str();
    rn.parent = parent;
    rn.bindLocal = AiToGlm(node->mTransformation);
    rn.bindGlobal = parentGlobal * rn.bindLocal;
    out.nodeByName[rn.name] = index;
    out.nodes.push_back(rn);

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
        CollectNodes(node->mChildren[i], index, rn.bindGlobal, out);
}

static void CollectMeshNodes(const aiNode *node, InxSkinnedMesh &model, std::vector<std::pair<uint32_t, int>> &out)
{
    int nodeIndex = -1;
    auto it = model.nodeByName.find(node->mName.C_Str());
    if (it != model.nodeByName.end())
        nodeIndex = it->second;

    for (unsigned int i = 0; i < node->mNumMeshes; ++i)
        out.push_back({node->mMeshes[i], nodeIndex});

    for (unsigned int i = 0; i < node->mNumChildren; ++i)
        CollectMeshNodes(node->mChildren[i], model, out);
}

static void AddInfluence(SkinInfluence &inf, uint32_t boneIndex, float weight)
{
    if (weight <= 0.0f)
        return;

    for (uint32_t i = 0; i < kMaxSkinInfluences; ++i) {
        if (inf.weight[i] <= 0.0f) {
            inf.boneIndex[i] = boneIndex;
            inf.weight[i] = weight;
            return;
        }
    }

    uint32_t lightest = 0;
    for (uint32_t i = 1; i < kMaxSkinInfluences; ++i)
        if (inf.weight[i] < inf.weight[lightest])
            lightest = i;
    if (weight > inf.weight[lightest]) {
        inf.boneIndex[lightest] = boneIndex;
        inf.weight[lightest] = weight;
    }
}

static uint32_t GetOrCreateBone(InxSkinnedMesh &model, const aiBone *bone)
{
    const std::string name = bone->mName.C_Str();
    auto it = model.boneByName.find(name);
    if (it != model.boneByName.end())
        return it->second;

    SkinnedRuntimeBone rb;
    rb.name = name;
    rb.inverseBind = AiToGlm(bone->mOffsetMatrix);
    auto nodeIt = model.nodeByName.find(name);
    if (nodeIt != model.nodeByName.end())
        rb.nodeIndex = nodeIt->second;

    const uint32_t idx = static_cast<uint32_t>(model.bones.size());
    model.boneByName[name] = idx;
    model.bones.push_back(rb);
    return idx;
}

static std::string CacheKey(const std::string &sourceGuid, const std::string &sourcePath)
{
    return !sourceGuid.empty() ? sourceGuid : sourcePath;
}

} // namespace

SkinnedModelCache &SkinnedModelCache::Instance()
{
    static SkinnedModelCache cache;
    return cache;
}

std::shared_ptr<InxSkinnedMesh> SkinnedModelCache::Load(const std::string &sourceGuid, const std::string &sourcePath)
{
    const std::string resolvedPath = ResolveSourcePath(sourceGuid, sourcePath);
    const std::string key = CacheKey(sourceGuid, resolvedPath);
    if (!key.empty()) {
        auto it = m_cache.find(key);
        if (it != m_cache.end())
            return it->second;
    }

    auto model = ImportModel(sourceGuid, resolvedPath);
    if (model && !key.empty())
        m_cache[key] = model;
    return model;
}

void SkinnedModelCache::Invalidate(const std::string &sourceGuid, const std::string &sourcePath)
{
    const std::string resolvedPath = ResolveSourcePath(sourceGuid, sourcePath);
    const std::string key = CacheKey(sourceGuid, resolvedPath);
    if (!key.empty())
        m_cache.erase(key);
}

void SkinnedModelCache::Clear()
{
    m_cache.clear();
}

std::shared_ptr<InxSkinnedMesh> SkinnedModelCache::ImportModel(const std::string &sourceGuid,
                                                               const std::string &sourcePath)
{
    std::vector<char> bytes;
    if (!ReadFileBytes(sourcePath, bytes)) {
        INXLOG_WARN("SkinnedModelCache: could not open source model '", sourcePath, "'");
        return nullptr;
    }

    auto fsPath = ToFsPath(sourcePath);
    std::string ext = fsPath.extension().string();
    if (!ext.empty() && ext[0] == '.')
        ext = ext.substr(1);

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFileFromMemory(bytes.data(), bytes.size(), BuildAssimpFlags(), ext.c_str());
    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        INXLOG_WARN("SkinnedModelCache: Assimp failed for '", sourcePath, "': ", importer.GetErrorString());
        return nullptr;
    }

    auto model = std::make_shared<InxSkinnedMesh>();
    model->sourcePath = sourcePath;
    model->guid = sourceGuid;
    model->scaleFactor = ReadScaleFactor(sourceGuid, sourcePath);

    CollectNodes(scene->mRootNode, -1, glm::mat4(1.0f), *model);

    std::vector<std::pair<uint32_t, int>> meshNodes;
    CollectMeshNodes(scene->mRootNode, *model, meshNodes);

    std::unordered_map<unsigned int, uint32_t> aiMatToSlot;
    uint32_t currentVertexOffset = 0;
    uint32_t currentIndexOffset = 0;

    for (const auto &[meshIndex, nodeIndex] : meshNodes) {
        if (meshIndex >= scene->mNumMeshes)
            continue;
        const aiMesh *aiM = scene->mMeshes[meshIndex];
        if (!(aiM->mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
            continue;

        const uint32_t vertexStart = currentVertexOffset;
        const uint32_t indexStart = currentIndexOffset;
        const bool hasNormals = aiM->HasNormals();
        const bool hasTangents = aiM->HasTangentsAndBitangents();
        const bool hasUVs = aiM->HasTextureCoords(0);
        const bool hasColors = aiM->HasVertexColors(0);

        for (unsigned int v = 0; v < aiM->mNumVertices; ++v) {
            Vertex vert{};
            vert.pos = AiToGlm(aiM->mVertices[v]);
            vert.normal = hasNormals ? glm::normalize(AiToGlm(aiM->mNormals[v])) : glm::vec3(0.0f, 1.0f, 0.0f);
            if (hasTangents)
                vert.tangent = glm::vec4(glm::normalize(AiToGlm(aiM->mTangents[v])), 1.0f);
            else
                vert.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            if (hasUVs)
                vert.texCoord = glm::vec2(aiM->mTextureCoords[0][v].x, aiM->mTextureCoords[0][v].y);
            if (hasColors)
                vert.color = glm::vec3(aiM->mColors[0][v].r, aiM->mColors[0][v].g, aiM->mColors[0][v].b);
            else
                vert.color = glm::vec3(1.0f);
            model->baseVertices.push_back(vert);
            model->influences.push_back({});
        }

        for (unsigned int b = 0; b < aiM->mNumBones; ++b) {
            const aiBone *bone = aiM->mBones[b];
            const uint32_t boneIndex = GetOrCreateBone(*model, bone);
            for (unsigned int wi = 0; wi < bone->mNumWeights; ++wi) {
                const aiVertexWeight &w = bone->mWeights[wi];
                const uint32_t globalVertex = vertexStart + w.mVertexId;
                if (globalVertex < model->influences.size())
                    AddInfluence(model->influences[globalVertex], boneIndex, w.mWeight);
            }
        }

        for (unsigned int f = 0; f < aiM->mNumFaces; ++f) {
            const aiFace &face = aiM->mFaces[f];
            for (unsigned int idx = 0; idx < face.mNumIndices; ++idx) {
                model->indices.push_back(face.mIndices[idx] + vertexStart);
                ++currentIndexOffset;
            }
        }

        uint32_t slot = 0;
        auto slotIt = aiMatToSlot.find(aiM->mMaterialIndex);
        if (slotIt != aiMatToSlot.end()) {
            slot = slotIt->second;
        } else {
            slot = static_cast<uint32_t>(aiMatToSlot.size());
            aiMatToSlot[aiM->mMaterialIndex] = slot;
        }

        SubMesh sub;
        sub.indexStart = indexStart;
        sub.indexCount = currentIndexOffset - indexStart;
        sub.vertexStart = vertexStart;
        sub.vertexCount = aiM->mNumVertices;
        sub.materialSlot = slot;
        sub.nodeGroup = nodeIndex >= 0 ? static_cast<uint32_t>(nodeIndex) : 0;
        sub.name = aiM->mName.C_Str();
        model->subMeshes.push_back(std::move(sub));
        currentVertexOffset += aiM->mNumVertices;
    }

    model->NormalizeInfluences();

    for (unsigned int ai = 0; ai < scene->mNumAnimations; ++ai) {
        const aiAnimation *src = scene->mAnimations[ai];
        SkinnedRuntimeAnimation anim;
        anim.name = src->mName.C_Str();
        anim.durationTicks = src->mDuration;
        anim.ticksPerSecond = src->mTicksPerSecond > 0.0 ? src->mTicksPerSecond : 25.0;
        anim.tracks.reserve(src->mNumChannels);

        for (unsigned int ci = 0; ci < src->mNumChannels; ++ci) {
            const aiNodeAnim *ch = src->mChannels[ci];
            SkinnedRuntimeTrack track;
            track.nodeName = ch->mNodeName.C_Str();
            track.positions.reserve(ch->mNumPositionKeys);
            track.rotations.reserve(ch->mNumRotationKeys);
            track.scales.reserve(ch->mNumScalingKeys);
            for (unsigned int k = 0; k < ch->mNumPositionKeys; ++k)
                track.positions.push_back({ch->mPositionKeys[k].mTime, AiToGlm(ch->mPositionKeys[k].mValue)});
            for (unsigned int k = 0; k < ch->mNumRotationKeys; ++k)
                track.rotations.push_back({ch->mRotationKeys[k].mTime, AiToGlm(ch->mRotationKeys[k].mValue)});
            for (unsigned int k = 0; k < ch->mNumScalingKeys; ++k)
                track.scales.push_back({ch->mScalingKeys[k].mTime, AiToGlm(ch->mScalingKeys[k].mValue)});

            anim.trackByNode[track.nodeName] = anim.tracks.size();
            anim.tracks.push_back(std::move(track));
        }
        model->animations.push_back(std::move(anim));
    }

    INXLOG_INFO("SkinnedModelCache: imported '", FromFsPath(fsPath.filename()), "' — ", model->baseVertices.size(),
                " verts, ", model->bones.size(), " bones, ", model->animations.size(), " anim(s)");
    return model;
}

} // namespace infernux
