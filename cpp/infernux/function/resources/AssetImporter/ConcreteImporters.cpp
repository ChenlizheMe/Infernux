#include "ConcreteImporters.h"

#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>

#include <core/log/InxLog.h>
#include <platform/filesystem/InxPath.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <assimp/material.h>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace infernux
{

namespace
{

glm::mat4 AiToGlmImport(const aiMatrix4x4 &m)
{
    return glm::mat4(m.a1, m.b1, m.c1, m.d1, m.a2, m.b2, m.c2, m.d2, m.a3, m.b3, m.c3, m.d3, m.a4, m.b4, m.c4, m.d4);
}

struct CollectedMeshImport
{
    uint32_t meshIndex;
    glm::mat4 worldTransform;
    uint32_t nodeGroup;
};

static void CollectMeshesImport(const aiNode *node, const glm::mat4 &parentTransform,
                                std::vector<CollectedMeshImport> &outMeshes, std::vector<std::string> &outNodeNames)
{
    glm::mat4 nodeTransform = parentTransform * AiToGlmImport(node->mTransformation);

    if (node->mNumMeshes > 0) {
        uint32_t group = static_cast<uint32_t>(outNodeNames.size());
        outNodeNames.push_back(node->mName.C_Str());
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            outMeshes.push_back({node->mMeshes[i], nodeTransform, group});
        }
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        CollectMeshesImport(node->mChildren[i], nodeTransform, outMeshes, outNodeNames);
    }
}

static nlohmann::json SerializeAiMaterialForImport(const aiMaterial *mat)
{
    nlohmann::json j;
    aiString aiName;
    mat->Get(AI_MATKEY_NAME, aiName);
    std::string name = aiName.C_Str();
    if (name.empty())
        name = "Material";
    j["name"] = name;

    aiColor4D base(1.f, 1.f, 1.f, 1.f);
    if (mat->Get(AI_MATKEY_BASE_COLOR, base) != AI_SUCCESS) {
        if (mat->Get(AI_MATKEY_COLOR_DIFFUSE, base) != AI_SUCCESS) {
            aiColor3D d3(1.f, 1.f, 1.f);
            if (mat->Get(AI_MATKEY_COLOR_DIFFUSE, d3) == AI_SUCCESS) {
                base.r = d3.r;
                base.g = d3.g;
                base.b = d3.b;
                base.a = 1.f;
            }
        }
    }

    float opacity = 1.f;
    if (mat->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
        base.a = std::clamp(opacity, 0.f, 1.f);
    }

    j["baseColor"] = nlohmann::json::array({base.r, base.g, base.b, base.a});

    float metallic = 0.f;
    mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
    j["metallic"] = static_cast<double>(std::clamp(metallic, 0.f, 1.f));

    float rough = 0.5f;
    if (mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, rough) != AI_SUCCESS) {
        float shininess = 0.f;
        if (mat->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS && shininess > 1e-4f) {
            rough = 1.f - std::clamp(std::sqrt(shininess / 128.f), 0.f, 1.f);
        }
    }
    rough = std::clamp(rough, 0.f, 1.f);
    j["smoothness"] = static_cast<double>(1.f - rough);

    std::string texPath;
    if (mat->GetTextureCount(aiTextureType_BASE_COLOR) > 0) {
        aiString p;
        if (mat->GetTexture(aiTextureType_BASE_COLOR, 0, &p) == AI_SUCCESS)
            texPath = p.C_Str();
    }
    if (texPath.empty() && mat->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
        aiString p;
        if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &p) == AI_SUCCESS)
            texPath = p.C_Str();
    }
    j["albedoTexturePath"] = texPath;

    return j;
}

static void BuildMeshOrderedImportMaterials(const aiScene *scene, nlohmann::json &importMaterialsJson,
                                            std::vector<std::string> &slotNamesOut)
{
    importMaterialsJson = nlohmann::json::array();
    slotNamesOut.clear();
    if (!scene || !scene->mRootNode)
        return;

    std::vector<CollectedMeshImport> collectedMeshes;
    std::vector<std::string> nodeNames;
    collectedMeshes.reserve(scene->mNumMeshes);
    CollectMeshesImport(scene->mRootNode, glm::mat4(1.0f), collectedMeshes, nodeNames);

    std::unordered_map<unsigned int, uint32_t> aiMatToSlot;

    for (const auto &cm : collectedMeshes) {
        const aiMesh *aiM = scene->mMeshes[cm.meshIndex];
        if (!(aiM->mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
            continue;

        auto it = aiMatToSlot.find(aiM->mMaterialIndex);
        if (it != aiMatToSlot.end())
            continue;

        const uint32_t slot = static_cast<uint32_t>(importMaterialsJson.size());
        aiMatToSlot[aiM->mMaterialIndex] = slot;

        if (aiM->mMaterialIndex < scene->mNumMaterials) {
            importMaterialsJson.push_back(SerializeAiMaterialForImport(scene->mMaterials[aiM->mMaterialIndex]));
            const std::string &nm = importMaterialsJson.back()["name"].get_ref<const std::string &>();
            slotNamesOut.push_back(nm);
        }
    }
}

} // namespace

void MaterialImporter::ScanDependencies(const ImportContext &ctx)
{
    if (ctx.guid.empty())
        return;

    auto &graph = AssetDependencyGraph::Instance();
    std::unordered_set<std::string> deps;

    // Parse .mat JSON
    nlohmann::json root;
    try {
        std::ifstream file(ToFsPath(ctx.sourcePath));
        if (!file.is_open())
            return;
        file >> root;
    } catch (const std::exception &e) {
        INXLOG_WARN("MaterialImporter: failed to parse '", ctx.sourcePath, "': ", e.what());
        return;
    } catch (...) {
        INXLOG_WARN("MaterialImporter: unknown error parsing '", ctx.sourcePath, "'");
        return;
    }

    // Shader dependencies (vertex + fragment paths)
    auto shadersIt = root.find("shaders");
    if (shadersIt != root.end() && shadersIt->is_object()) {
        for (const auto &key : {"vertex", "fragment"}) {
            auto it = shadersIt->find(key);
            if (it == shadersIt->end() || !it->is_string())
                continue;
            std::string shaderPath = it->get<std::string>();
            if (shaderPath.empty())
                continue;
            // Resolve path → GUID via AssetDatabase
            std::string depGuid;
            if (m_assetDb)
                depGuid = m_assetDb->GetGuidFromPath(shaderPath);
            if (!depGuid.empty())
                deps.insert(depGuid);
        }
    }

    // Texture dependencies (properties with type == 6 == Texture2D)
    auto propsIt = root.find("properties");
    if (propsIt != root.end() && propsIt->is_object()) {
        for (auto &[propName, propVal] : propsIt->items()) {
            if (!propVal.is_object())
                continue;
            auto typeIt = propVal.find("type");
            if (typeIt == propVal.end() || !typeIt->is_number_integer())
                continue;
            int ptype = typeIt->get<int>();
            if (ptype != 6) // 6 == Texture2D
                continue;
            auto guidIt = propVal.find("guid");
            if (guidIt != propVal.end() && guidIt->is_string()) {
                std::string texGuid = guidIt->get<std::string>();
                if (!texGuid.empty())
                    deps.insert(texGuid);
            }
        }
    }

    // Bulk-set (clears old deps, registers new)
    graph.SetDependencies(ctx.guid, deps);

    if (!deps.empty()) {
        INXLOG_DEBUG("MaterialImporter: material '", ctx.sourcePath, "' (", ctx.guid, ") depends on ", deps.size(),
                     " asset(s)");
    }
}

// ============================================================================
// ModelImporter — scan model file with Assimp and extract metadata into .meta
// ============================================================================

bool ModelImporter::Import(const ImportContext &ctx)
{
    if (!ctx.meta)
        return false;

    EnsureDefaultSettings(*ctx.meta);

    // Quick-validate the source file with a lightweight Assimp parse.
    // We only need the scene structure, not full post-processing.
    std::filesystem::path sourcePath = ToFsPath(ctx.sourcePath);
    if (!std::filesystem::exists(sourcePath)) {
        INXLOG_ERROR("ModelImporter: source file not found: ", ctx.sourcePath);
        return false;
    }

    Assimp::Importer importer;

    std::ifstream file(sourcePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        INXLOG_ERROR("ModelImporter: failed to open source file: ", ctx.sourcePath);
        return false;
    }

    std::streamsize fileSize = file.tellg();
    if (fileSize <= 0) {
        INXLOG_ERROR("ModelImporter: source file is empty or unreadable: ", ctx.sourcePath);
        return false;
    }

    std::vector<char> fileData(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    if (!file.read(fileData.data(), fileSize)) {
        INXLOG_ERROR("ModelImporter: failed to read source file: ", ctx.sourcePath);
        return false;
    }

    // Minimal flags: triangulate so we can count real triangle-indices,
    // but skip heavy post-processing (that happens at load time in MeshLoader).
    std::string ext = sourcePath.extension().string();
    if (!ext.empty() && ext[0] == '.')
        ext.erase(0, 1);
    const aiScene *scene = importer.ReadFileFromMemory(fileData.data(), static_cast<size_t>(fileData.size()),
                                                       aiProcess_Triangulate | aiProcess_SortByPType, ext.c_str());

    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        INXLOG_ERROR("ModelImporter: Assimp validation failed for '", ctx.sourcePath, "': ", importer.GetErrorString());
        return false;
    }

    // ── Collect metadata ────────────────────────────────────────────────

    uint32_t totalVertices = 0;
    uint32_t totalIndices = 0;
    uint32_t meshCount = scene->mNumMeshes;

    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh *aiM = scene->mMeshes[i];
        if (!(aiM->mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
            continue;
        totalVertices += aiM->mNumVertices;
        for (unsigned int f = 0; f < aiM->mNumFaces; ++f)
            totalIndices += aiM->mFaces[f].mNumIndices;
    }

    // Material slots + FBX/Assimp surface data — **same encounter order as MeshLoader**
    // (not raw scene->mMaterials[] order, which can disagree with merged mesh slots).
    nlohmann::json importMaterialsJson;
    std::vector<std::string> materialSlots;
    BuildMeshOrderedImportMaterials(scene, importMaterialsJson, materialSlots);

    // ── Write metadata to .meta ─────────────────────────────────────────

    ctx.meta->AddMetadata("mesh_count", static_cast<int>(meshCount));
    ctx.meta->AddMetadata("vertex_count", static_cast<int>(totalVertices));
    ctx.meta->AddMetadata("index_count", static_cast<int>(totalIndices));
    ctx.meta->AddMetadata("material_slot_count", static_cast<int>(materialSlots.size()));

    // Store material slot names as a comma-separated string for .meta
    // (InxResourceMeta uses std::any; a string is the simplest portable choice)
    std::string slotsStr;
    for (size_t i = 0; i < materialSlots.size(); ++i) {
        if (i > 0)
            slotsStr += ',';
        slotsStr += materialSlots[i];
    }
    ctx.meta->AddMetadata("material_slots", slotsStr);

    // Serialized JSON array: name, baseColor, metallic, smoothness, albedoTexturePath (relative to model file)
    ctx.meta->AddMetadata("import_materials", importMaterialsJson.dump());

    INXLOG_INFO("ModelImporter: imported '", FromFsPath(sourcePath.filename()), "' — ", meshCount, " mesh(es), ",
                totalVertices, " verts, ", totalIndices, " indices, ", materialSlots.size(), " material slot(s)");

    return true;
}

} // namespace infernux
