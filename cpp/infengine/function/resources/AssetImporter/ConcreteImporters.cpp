#include "ConcreteImporters.h"

#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetDependencyGraph.h>

#include <core/log/InfLog.h>
#include <platform/filesystem/InfPath.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace infengine
{

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
        INFLOG_WARN("MaterialImporter: failed to parse '", ctx.sourcePath, "': ", e.what());
        return;
    } catch (...) {
        INFLOG_WARN("MaterialImporter: unknown error parsing '", ctx.sourcePath, "'");
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
            // v2: texture GUID stored under "guid" key
            // v1 (legacy): texture path stored under "value" key, needs path→GUID
            auto guidIt = propVal.find("guid");
            if (guidIt != propVal.end() && guidIt->is_string()) {
                std::string texGuid = guidIt->get<std::string>();
                if (!texGuid.empty())
                    deps.insert(texGuid);
            } else {
                // Legacy v1 fallback: resolve path → GUID
                auto valIt = propVal.find("value");
                if (valIt == propVal.end() || !valIt->is_string())
                    continue;
                std::string texPath = valIt->get<std::string>();
                if (texPath.empty() || texPath == "white" || texPath == "black" || texPath == "normal")
                    continue;
                std::string depGuid;
                if (m_assetDb)
                    depGuid = m_assetDb->GetGuidFromPath(texPath);
                if (!depGuid.empty())
                    deps.insert(depGuid);
            }
        }
    }

    // Bulk-set (clears old deps, registers new)
    graph.SetDependencies(ctx.guid, deps);

    if (!deps.empty()) {
        INFLOG_DEBUG("MaterialImporter: material '", ctx.sourcePath, "' (", ctx.guid, ") depends on ", deps.size(),
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
        INFLOG_ERROR("ModelImporter: source file not found: ", ctx.sourcePath);
        return false;
    }

    Assimp::Importer importer;

    std::ifstream file(sourcePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        INFLOG_ERROR("ModelImporter: failed to open source file: ", ctx.sourcePath);
        return false;
    }

    std::streamsize fileSize = file.tellg();
    if (fileSize <= 0) {
        INFLOG_ERROR("ModelImporter: source file is empty or unreadable: ", ctx.sourcePath);
        return false;
    }

    std::vector<char> fileData(static_cast<size_t>(fileSize));
    file.seekg(0, std::ios::beg);
    if (!file.read(fileData.data(), fileSize)) {
        INFLOG_ERROR("ModelImporter: failed to read source file: ", ctx.sourcePath);
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
        INFLOG_ERROR("ModelImporter: Assimp validation failed for '", ctx.sourcePath, "': ", importer.GetErrorString());
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

    // Extract unique material names
    std::vector<std::string> materialSlots;
    materialSlots.reserve(scene->mNumMaterials);
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        aiString aiName;
        scene->mMaterials[i]->Get(AI_MATKEY_NAME, aiName);
        std::string name = aiName.C_Str();
        if (name.empty())
            name = "Material_" + std::to_string(i);
        materialSlots.push_back(std::move(name));
    }

    // ── Write metadata to .meta ─────────────────────────────────────────

    ctx.meta->AddMetadata("mesh_count", static_cast<int>(meshCount));
    ctx.meta->AddMetadata("vertex_count", static_cast<int>(totalVertices));
    ctx.meta->AddMetadata("index_count", static_cast<int>(totalIndices));
    ctx.meta->AddMetadata("material_slot_count", static_cast<int>(materialSlots.size()));

    // Store material slot names as a comma-separated string for .meta
    // (InfResourceMeta uses std::any; a string is the simplest portable choice)
    std::string slotsStr;
    for (size_t i = 0; i < materialSlots.size(); ++i) {
        if (i > 0)
            slotsStr += ',';
        slotsStr += materialSlots[i];
    }
    ctx.meta->AddMetadata("material_slots", slotsStr);

    INFLOG_INFO("ModelImporter: imported '", FromFsPath(sourcePath.filename()), "' — ", meshCount, " mesh(es), ",
                totalVertices, " verts, ", totalIndices, " indices, ", materialSlots.size(), " material slot(s)");

    return true;
}

} // namespace infengine
