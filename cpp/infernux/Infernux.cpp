/**
 * @file Infernux.cpp
 * @brief Infernux — Core lifecycle, resources, renderer init, gizmos, material pipeline
 *
 * Editor camera control → InfernuxCamera.cpp
 * Scene picking / raycasting → ScenePicker.cpp
 */

#include "Infernux.h"
// Explicit includes for types now only forward-declared in InxRenderer.h
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <function/audio/AudioClipLoader.h>
#include <function/audio/AudioEngine.h>
#include <function/renderer/EditorGizmos.h>
#include <function/renderer/GizmosDrawCallBuffer.h>
#include <function/renderer/SceneRenderGraph.h>
#include <function/renderer/ScriptableRenderContext.h>
#include <function/renderer/gui/InxGUIContext.h>
#include <function/renderer/gui/InxResourcePreviewer.h>
#include <function/renderer/gui/InxScreenUIRenderer.h>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxFileLoader/InxDefaultLoader.hpp>
#include <function/resources/InxFileLoader/InxPythonScriptLoader.hpp>
#include <function/resources/InxFileLoader/InxShaderLoader.hpp>
#include <function/resources/InxFileLoader/InxTextureLoader.hpp>
#include <function/resources/InxMaterial/MaterialLoader.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/InxTexture/TextureLoader.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <function/scene/Component.h>
#include <function/scene/MeshRenderer.h>
#include <function/scene/PrimitiveMeshes.h>
#include <function/scene/SceneRenderer.h>
#include <function/scene/physics/PhysicsWorld.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <imgui_internal.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>

#include <core/config/InxPlatform.h>
#ifdef INX_PLATFORM_WINDOWS
#include <ShlObj.h> // SHGetFolderPathW for Documents path
#endif

namespace infernux
{

namespace
{

using json = nlohmann::json;

bool ParseModelEmbeddedMaterialSlot(const std::string &path, std::string &outModel, int &outSlot)
{
    constexpr const char kTok[] = "::submat:";
    const size_t pos = path.find(kTok);
    if (pos == std::string::npos)
        return false;
    outModel = path.substr(0, pos);
    try {
        outSlot = std::stoi(path.substr(pos + sizeof(kTok) - 1));
    } catch (...) {
        return false;
    }
    return !outModel.empty() && outSlot >= 0;
}

struct PrefabPreviewAggregate
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> subMeshes;
    std::vector<std::shared_ptr<InxMaterial>> materials;
};

glm::quat PreviewEulerYXZToQuat(const glm::vec3 &eulerDeg)
{
    glm::vec3 r = glm::radians(eulerDeg);
    float cx = std::cos(r.x * 0.5f), sx = std::sin(r.x * 0.5f);
    float cy = std::cos(r.y * 0.5f), sy = std::sin(r.y * 0.5f);
    float cz = std::cos(r.z * 0.5f), sz = std::sin(r.z * 0.5f);

    glm::quat q;
    q.w = cy * cx * cz + sy * sx * sz;
    q.x = cy * sx * cz + sy * cx * sz;
    q.y = sy * cx * cz - cy * sx * sz;
    q.z = cy * cx * sz - sy * sx * cz;
    return q;
}

glm::vec3 ReadVec3(const json &j, const char *key, const glm::vec3 &fallback)
{
    auto it = j.find(key);
    if (it == j.end() || !it->is_array() || it->size() != 3)
        return fallback;
    return glm::vec3((*it)[0].get<float>(), (*it)[1].get<float>(), (*it)[2].get<float>());
}

glm::mat4 ReadNodeLocalMatrix(const json &node)
{
    auto it = node.find("transform");
    if (it == node.end() || !it->is_object())
        return glm::mat4(1.0f);

    const glm::vec3 position = ReadVec3(*it, "position", glm::vec3(0.0f));
    const glm::vec3 rotation = ReadVec3(*it, "rotation", glm::vec3(0.0f));
    const glm::vec3 scale = ReadVec3(*it, "scale", glm::vec3(1.0f));

    return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(PreviewEulerYXZToQuat(rotation)) *
           glm::scale(glm::mat4(1.0f), scale);
}

bool GetPreviewPrimitiveMeshData(const std::string &name, const std::vector<Vertex> *&vertices,
                                 const std::vector<uint32_t> *&indices)
{
    vertices = nullptr;
    indices = nullptr;

    if (name == "Cube") {
        vertices = &PrimitiveMeshes::GetCubeVertices();
        indices = &PrimitiveMeshes::GetCubeIndices();
    } else if (name == "Quad") {
        vertices = &PrimitiveMeshes::GetQuadVertices();
        indices = &PrimitiveMeshes::GetQuadIndices();
    } else if (name == "Sphere") {
        vertices = &PrimitiveMeshes::GetSphereVertices();
        indices = &PrimitiveMeshes::GetSphereIndices();
    } else if (name == "Capsule") {
        vertices = &PrimitiveMeshes::GetCapsuleVertices();
        indices = &PrimitiveMeshes::GetCapsuleIndices();
    } else if (name == "Cylinder") {
        vertices = &PrimitiveMeshes::GetCylinderVertices();
        indices = &PrimitiveMeshes::GetCylinderIndices();
    } else if (name == "Plane") {
        vertices = &PrimitiveMeshes::GetPlaneVertices();
        indices = &PrimitiveMeshes::GetPlaneIndices();
    }

    return vertices != nullptr && indices != nullptr;
}

glm::vec3 NormalizeOrFallback(const glm::vec3 &value, const glm::vec3 &fallback)
{
    const float lenSq = glm::dot(value, value);
    if (lenSq > 1e-10f)
        return glm::normalize(value);

    const float fallbackLenSq = glm::dot(fallback, fallback);
    if (fallbackLenSq > 1e-10f)
        return glm::normalize(fallback);

    return glm::vec3(0.0f, 1.0f, 0.0f);
}

void ComputeBoundsFromVertices(const std::vector<Vertex> &vertices, glm::vec3 &outMin, glm::vec3 &outMax)
{
    constexpr float kInf = std::numeric_limits<float>::max();
    outMin = glm::vec3(kInf);
    outMax = glm::vec3(-kInf);
    for (const auto &v : vertices) {
        outMin = glm::min(outMin, v.pos);
        outMax = glm::max(outMax, v.pos);
    }
    if (vertices.empty()) {
        outMin = glm::vec3(0.0f);
        outMax = glm::vec3(0.0f);
    }
}

void ComputeBoundsFromIndexRange(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                                 uint32_t indexStart, uint32_t indexCount, glm::vec3 &outMin, glm::vec3 &outMax)
{
    constexpr float kInf = std::numeric_limits<float>::max();
    outMin = glm::vec3(kInf);
    outMax = glm::vec3(-kInf);

    for (uint32_t i = 0; i < indexCount; ++i) {
        const uint32_t index = indices[indexStart + i];
        if (index >= vertices.size())
            continue;
        outMin = glm::min(outMin, vertices[index].pos);
        outMax = glm::max(outMax, vertices[index].pos);
    }

    if (indexCount == 0 || outMin.x == kInf) {
        outMin = glm::vec3(0.0f);
        outMax = glm::vec3(0.0f);
    }
}

std::shared_ptr<InxMaterial> BuildPreviewMaterialFromSlotData(const MaterialSlotData *slotData,
                                                              const std::shared_ptr<InxMaterial> &defaultMat)
{
    if (!defaultMat)
        return nullptr;
    if (!slotData)
        return defaultMat;

    auto mat = defaultMat->Clone();
    if (!mat)
        return defaultMat;

    mat->SetColor("baseColor", slotData->baseColor);
    mat->SetColor("emissionColor", slotData->emissionColor);
    mat->SetFloat("metallic", slotData->metallic);
    mat->SetFloat("smoothness", slotData->smoothness);
    return mat;
}

std::shared_ptr<InxMaterial> ResolvePrefabPreviewMaterial(const json &componentJson, uint32_t materialSlot,
                                                          const std::shared_ptr<InxMesh> &assetMesh,
                                                          const std::shared_ptr<InxMaterial> &defaultMat,
                                                          const std::shared_ptr<InxMaterial> &errorMat)
{
    auto &registry = AssetRegistry::Instance();
    auto matsIt = componentJson.find("materials");
    if (matsIt != componentJson.end() && matsIt->is_array() && materialSlot < matsIt->size()) {
        const auto &slotJson = (*matsIt)[materialSlot];
        if (slotJson.is_string()) {
            const std::string guid = slotJson.get<std::string>();
            if (!guid.empty()) {
                auto mat = registry.GetAsset<InxMaterial>(guid);
                if (!mat)
                    mat = registry.LoadAsset<InxMaterial>(guid, ResourceType::Material);
                if (mat) {
                    if (!mat->IsDeleted())
                        return mat;
                    return errorMat ? errorMat : defaultMat;
                }
            }
        }
    }

    const MaterialSlotData *slotData = nullptr;
    if (assetMesh && materialSlot < assetMesh->GetMaterialSlotData().size())
        slotData = &assetMesh->GetMaterialSlotData()[materialSlot];
    return BuildPreviewMaterialFromSlotData(slotData, defaultMat);
}

bool AppendPrefabMeshComponent(const json &componentJson, const glm::mat4 &worldMatrix,
                               PrefabPreviewAggregate &aggregate, const std::shared_ptr<InxMaterial> &defaultMat,
                               const std::shared_ptr<InxMaterial> &errorMat)
{
    if (!componentJson.is_object())
        return false;
    if (componentJson.value("type", std::string()) != "MeshRenderer")
        return false;
    if (!componentJson.value("enabled", true))
        return false;

    std::shared_ptr<InxMesh> assetMesh;
    std::vector<Vertex> inlineVertices;
    std::vector<uint32_t> inlineIndices;
    std::vector<SubMesh> inlineSubMeshes;

    const std::vector<Vertex> *srcVertices = nullptr;
    const std::vector<uint32_t> *srcIndices = nullptr;
    const std::vector<SubMesh> *srcSubMeshes = nullptr;

    auto meshGuidIt = componentJson.find("meshAssetGuid");
    if (meshGuidIt != componentJson.end() && meshGuidIt->is_string()) {
        const std::string meshGuid = meshGuidIt->get<std::string>();
        if (!meshGuid.empty()) {
            auto &registry = AssetRegistry::Instance();
            assetMesh = registry.GetAsset<InxMesh>(meshGuid);
            if (!assetMesh)
                assetMesh = registry.LoadAsset<InxMesh>(meshGuid, ResourceType::Mesh);
            if (assetMesh) {
                srcVertices = &assetMesh->GetVertices();
                srcIndices = &assetMesh->GetIndices();
                srcSubMeshes = &assetMesh->GetSubMeshes();
            }
        }
    }

    if ((!srcVertices || !srcIndices || !srcSubMeshes) && componentJson.value("useInlineMesh", false)) {
        const std::string inlineName = componentJson.value("inlineMeshName", std::string());
        if (componentJson.value("inlineMeshBuiltin", false)) {
            const std::vector<Vertex> *builtinVertices = nullptr;
            const std::vector<uint32_t> *builtinIndices = nullptr;
            if (GetPreviewPrimitiveMeshData(inlineName, builtinVertices, builtinIndices)) {
                inlineVertices.assign(builtinVertices->begin(), builtinVertices->end());
                inlineIndices.assign(builtinIndices->begin(), builtinIndices->end());
            }
        } else {
            auto vertsIt = componentJson.find("inlineVertices");
            if (vertsIt != componentJson.end() && vertsIt->is_array()) {
                inlineVertices.reserve(vertsIt->size());
                for (const auto &vertexJson : *vertsIt) {
                    Vertex vertex{};
                    vertex.pos = ReadVec3(vertexJson, "pos", glm::vec3(0.0f));
                    vertex.normal = ReadVec3(vertexJson, "normal", glm::vec3(0.0f, 1.0f, 0.0f));
                    vertex.color = ReadVec3(vertexJson, "color", glm::vec3(1.0f));

                    auto tangentIt = vertexJson.find("tangent");
                    if (tangentIt != vertexJson.end() && tangentIt->is_array() && tangentIt->size() == 4) {
                        vertex.tangent = glm::vec4((*tangentIt)[0].get<float>(), (*tangentIt)[1].get<float>(),
                                                   (*tangentIt)[2].get<float>(), (*tangentIt)[3].get<float>());
                    } else {
                        vertex.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                    }

                    auto uvIt = vertexJson.find("texCoord");
                    if (uvIt != vertexJson.end() && uvIt->is_array() && uvIt->size() == 2) {
                        vertex.texCoord = glm::vec2((*uvIt)[0].get<float>(), (*uvIt)[1].get<float>());
                    }

                    inlineVertices.push_back(vertex);
                }
            }

            auto indicesIt = componentJson.find("inlineIndices");
            if (indicesIt != componentJson.end() && indicesIt->is_array()) {
                inlineIndices.reserve(indicesIt->size());
                for (const auto &indexJson : *indicesIt)
                    inlineIndices.push_back(indexJson.get<uint32_t>());
            }
        }

        if (!inlineVertices.empty() && !inlineIndices.empty()) {
            SubMesh inlineSubMesh;
            inlineSubMesh.indexStart = 0;
            inlineSubMesh.indexCount = static_cast<uint32_t>(inlineIndices.size());
            inlineSubMesh.vertexStart = 0;
            inlineSubMesh.vertexCount = static_cast<uint32_t>(inlineVertices.size());
            inlineSubMesh.materialSlot = 0;
            inlineSubMesh.nodeGroup = 0;
            inlineSubMesh.name = inlineName;
            ComputeBoundsFromVertices(inlineVertices, inlineSubMesh.boundsMin, inlineSubMesh.boundsMax);
            inlineSubMeshes.push_back(std::move(inlineSubMesh));

            srcVertices = &inlineVertices;
            srcIndices = &inlineIndices;
            srcSubMeshes = &inlineSubMeshes;
        }
    }

    if (!srcVertices || !srcIndices || !srcSubMeshes || srcVertices->empty() || srcIndices->empty())
        return false;

    const int32_t submeshFilter = componentJson.value("submeshIndex", -1);
    const int32_t nodeGroupFilter = componentJson.value("nodeGroup", -1);

    std::vector<const SubMesh *> selectedSubMeshes;
    selectedSubMeshes.reserve(srcSubMeshes->size());
    for (size_t subMeshIndex = 0; subMeshIndex < srcSubMeshes->size(); ++subMeshIndex) {
        const SubMesh &subMesh = (*srcSubMeshes)[subMeshIndex];
        if (subMesh.indexCount == 0)
            continue;
        if (submeshFilter >= 0 && static_cast<int32_t>(subMeshIndex) != submeshFilter)
            continue;
        if (nodeGroupFilter >= 0 && static_cast<int32_t>(subMesh.nodeGroup) != nodeGroupFilter)
            continue;
        selectedSubMeshes.push_back(&subMesh);
    }

    if (selectedSubMeshes.empty())
        return false;

    const glm::vec3 pivotOffset = ReadVec3(componentJson, "meshPivotOffset", glm::vec3(0.0f));
    const glm::mat3 world3x3(worldMatrix);
    glm::mat3 normalMatrix(1.0f);
    const float determinant = glm::determinant(world3x3);
    if (std::abs(determinant) > 1e-8f)
        normalMatrix = glm::transpose(glm::inverse(world3x3));
    const float tangentHandedness = determinant < 0.0f ? -1.0f : 1.0f;

    const uint32_t vertexBase = static_cast<uint32_t>(aggregate.vertices.size());
    aggregate.vertices.reserve(aggregate.vertices.size() + srcVertices->size());
    for (const auto &srcVertex : *srcVertices) {
        Vertex vertex = srcVertex;
        vertex.pos = glm::vec3(worldMatrix * glm::vec4(srcVertex.pos + pivotOffset, 1.0f));
        vertex.normal = NormalizeOrFallback(normalMatrix * srcVertex.normal, srcVertex.normal);
        vertex.tangent =
            glm::vec4(NormalizeOrFallback(normalMatrix * glm::vec3(srcVertex.tangent), glm::vec3(srcVertex.tangent)),
                      srcVertex.tangent.w * tangentHandedness);
        aggregate.vertices.push_back(vertex);
    }

    for (const SubMesh *subMesh : selectedSubMeshes) {
        SubMesh previewSubMesh;
        previewSubMesh.indexStart = static_cast<uint32_t>(aggregate.indices.size());
        previewSubMesh.vertexStart = vertexBase;
        previewSubMesh.vertexCount = static_cast<uint32_t>(srcVertices->size());
        previewSubMesh.materialSlot = static_cast<uint32_t>(aggregate.materials.size());
        previewSubMesh.nodeGroup = subMesh->nodeGroup;
        previewSubMesh.name = subMesh->name;

        const uint32_t indexEnd = subMesh->indexStart + subMesh->indexCount;
        aggregate.indices.reserve(aggregate.indices.size() + subMesh->indexCount);
        for (uint32_t index = subMesh->indexStart; index < indexEnd; ++index)
            aggregate.indices.push_back((*srcIndices)[index] + vertexBase);

        previewSubMesh.indexCount = static_cast<uint32_t>(aggregate.indices.size()) - previewSubMesh.indexStart;
        ComputeBoundsFromIndexRange(aggregate.vertices, aggregate.indices, previewSubMesh.indexStart,
                                    previewSubMesh.indexCount, previewSubMesh.boundsMin, previewSubMesh.boundsMax);

        aggregate.materials.push_back(
            ResolvePrefabPreviewMaterial(componentJson, subMesh->materialSlot, assetMesh, defaultMat, errorMat));
        aggregate.subMeshes.push_back(std::move(previewSubMesh));
    }

    return true;
}

void AppendPrefabNodePreview(const json &nodeJson, const glm::mat4 &parentWorld, bool parentActive,
                             PrefabPreviewAggregate &aggregate, const std::shared_ptr<InxMaterial> &defaultMat,
                             const std::shared_ptr<InxMaterial> &errorMat)
{
    if (!nodeJson.is_object())
        return;

    const bool isActive = parentActive && nodeJson.value("active", true);
    if (!isActive)
        return;

    const glm::mat4 worldMatrix = parentWorld * ReadNodeLocalMatrix(nodeJson);

    auto componentsIt = nodeJson.find("components");
    if (componentsIt != nodeJson.end() && componentsIt->is_array()) {
        for (const auto &componentJson : *componentsIt)
            AppendPrefabMeshComponent(componentJson, worldMatrix, aggregate, defaultMat, errorMat);
    }

    auto childrenIt = nodeJson.find("children");
    if (childrenIt != nodeJson.end() && childrenIt->is_array()) {
        for (const auto &childJson : *childrenIt)
            AppendPrefabNodePreview(childJson, worldMatrix, isActive, aggregate, defaultMat, errorMat);
    }
}

bool BuildPrefabPreviewMesh(const std::string &prefabFilePath, std::shared_ptr<InxMesh> &outMesh,
                            std::vector<std::shared_ptr<InxMaterial>> &outMaterials)
{
    std::ifstream input(ToFsPath(prefabFilePath), std::ios::binary);
    if (!input.is_open())
        return false;

    json prefabJson = json::parse(input, nullptr, false);
    if (prefabJson.is_discarded())
        return false;

    auto rootIt = prefabJson.find("root_object");
    if (rootIt == prefabJson.end() || !rootIt->is_object())
        return false;

    auto &registry = AssetRegistry::Instance();
    auto defaultMat = registry.GetBuiltinMaterial("DefaultLit");
    auto errorMat = registry.GetBuiltinMaterial("ErrorMaterial");

    PrefabPreviewAggregate aggregate;
    AppendPrefabNodePreview(*rootIt, glm::mat4(1.0f), true, aggregate, defaultMat, errorMat);

    if (aggregate.vertices.empty() || aggregate.indices.empty() || aggregate.subMeshes.empty())
        return false;

    auto mesh = std::make_shared<InxMesh>(ToFsPath(prefabFilePath).stem().string());
    mesh->SetFilePath(prefabFilePath);
    mesh->SetData(std::move(aggregate.vertices), std::move(aggregate.indices), std::move(aggregate.subMeshes));

    outMesh = std::move(mesh);
    outMaterials = std::move(aggregate.materials);
    return true;
}

std::vector<std::shared_ptr<InxMaterial>> BuildDefaultPreviewMaterialsForMesh(const InxMesh &mesh)
{
    auto defaultMat = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    std::vector<std::shared_ptr<InxMaterial>> materials;
    if (!defaultMat)
        return materials;

    uint32_t maxSlot = 0;
    for (const auto &subMesh : mesh.GetSubMeshes())
        maxSlot = std::max(maxSlot, subMesh.materialSlot + 1);

    const auto &slotData = mesh.GetMaterialSlotData();
    materials.reserve(maxSlot);
    for (uint32_t slot = 0; slot < maxSlot; ++slot) {
        const MaterialSlotData *data = slot < slotData.size() ? &slotData[slot] : nullptr;
        materials.push_back(BuildPreviewMaterialFromSlotData(data, defaultMat));
    }
    return materials;
}

bool IsPrefabPreviewPath(const std::string &filePath)
{
    std::string ext = ToFsPath(filePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return ext == ".prefab";
}

} // namespace

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

void Infernux::ModifyResources(const std::string &filePath)
{
    if (!CheckEngineValid("modify resources"))
        return;
    auto *adb = GetAssetDatabase();
    if (adb)
        adb->ModifyResource(filePath);
}

void Infernux::DeleteResources(const std::string &filePath)
{
    if (!CheckEngineValid("delete resources"))
        return;
    auto *adb = GetAssetDatabase();
    if (adb)
        adb->DeleteResource(filePath);
}

void Infernux::MoveResources(const std::string &oldFilePath, const std::string &newFilePath)
{
    if (!CheckEngineValid("move resources"))
        return;
    auto *adb = GetAssetDatabase();
    if (adb)
        adb->MoveResource(oldFilePath, newFilePath);
}

AssetDatabase *Infernux::GetAssetDatabase() const
{
    // After InitRenderer, ownership is transferred to AssetRegistry
    auto *adb = AssetRegistry::Instance().GetAssetDatabase();
    return adb ? adb : m_assetDatabase.get();
}

// ----------------------------------
// Lifecycle
// ----------------------------------

Infernux::Infernux(std::string dllPath) : m_isCleanedUp(false)
{
    INXLOG_DEBUG("Create Infernux.");
    m_assetDatabase = std::make_unique<AssetDatabase>();

    INXLOG_DEBUG("Create Infernux Renderer.");
    m_renderer = std::make_unique<InxRenderer>();

    InitPreviewTaskSystem(1);
}

Infernux::~Infernux()
{
    INXLOG_DEBUG("Infernux destructor called.");
    Cleanup();
}

void Infernux::Run()
{
    if (!CheckEngineValid("run") || !m_renderer) {
        INXLOG_ERROR("Cannot run: Renderer is not initialized.");
        return;
    }

    INXLOG_DEBUG("Run Infernux.");
    while (m_renderer && m_renderer->GetUserEvent()) {
        try {
            m_renderer->DrawFrame();
        } catch (const std::exception &ex) {
            INXLOG_ERROR("Exception in DrawFrame: {}", ex.what());
        } catch (...) {
            INXLOG_ERROR("Unknown exception in DrawFrame!");
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

void Infernux::Exit()
{
    INXLOG_DEBUG("Exit requested.");
    // Set exit flag to make the main loop exit
    // The actual exit happens when GetUserEvent() returns false
}

void Infernux::Cleanup()
{
    if (m_isCleanedUp) {
        INXLOG_DEBUG("Already cleaned up, skipping.");
        return;
    }

    m_isCleaningUp = true;

    ShutdownPreviewTaskSystem();

    SaveImGuiLayout();
    AudioEngine::Instance().Shutdown();
    PhysicsWorld::Instance().Shutdown();

    m_renderer.reset();

    // AssetRegistry owns all loaded assets + builtins.
    AssetRegistry::Instance().Shutdown();

    m_assetDatabase.reset();
    m_extLoader.reset();

    INXLOG_DEBUG("Cleanup completed.");
#if INFERNUX_FILE_LOGGING
    INXLOG_FLUSH_FILE();
#endif

    m_isCleanedUp = true;
    m_isCleaningUp = false;
}

void Infernux::InitPreviewTaskSystem(uint32_t workerCount)
{
    if (m_previewTaskSystemInitialized)
        return;

    const uint32_t count = (workerCount == 0) ? 1u : workerCount;
    m_previewStopRequested = false;
    m_previewWorkers.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        m_previewWorkers.emplace_back([this]() {
            for (;;) {
                PreviewTaskItem item;
                {
                    std::unique_lock<std::mutex> lock(m_previewTaskMutex);
                    m_previewTaskCv.wait(lock,
                                         [this]() { return m_previewStopRequested || !m_previewTaskQueue.empty(); });

                    if (m_previewStopRequested && m_previewTaskQueue.empty())
                        return;

                    item = std::move(m_previewTaskQueue.front());
                    m_previewTaskQueue.pop();
                }

                if (item.fn)
                    item.fn();
            }
        });
    }

    m_previewTaskSystemInitialized = true;
}

void Infernux::ShutdownPreviewTaskSystem()
{
    if (!m_previewTaskSystemInitialized)
        return;

    {
        std::lock_guard<std::mutex> lock(m_previewTaskMutex);
        m_previewStopRequested = true;
    }
    m_previewTaskCv.notify_all();

    for (auto &worker : m_previewWorkers) {
        if (worker.joinable())
            worker.join();
    }
    m_previewWorkers.clear();

    {
        std::lock_guard<std::mutex> lock(m_previewTaskMutex);
        std::queue<PreviewTaskItem> empty;
        m_previewTaskQueue.swap(empty);
    }
    {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        std::queue<MaterialPreviewCompleted> empty;
        m_previewCompletedQueue.swap(empty);
        std::queue<TexturePreviewCompleted> emptyTex;
        m_texturePreviewCompletedQueue.swap(emptyTex);
        std::queue<MaterialPreviewRequest> emptyReq;
        m_previewRequestQueue.swap(emptyReq);
        m_materialPreviewStates.clear();
        m_texturePreviewStates.clear();
    }

    m_previewTaskSystemInitialized = false;
}

void Infernux::EnqueuePreviewTask(std::function<void()> fn)
{
    if (!fn)
        return;

    {
        std::lock_guard<std::mutex> lock(m_previewTaskMutex);
        m_previewTaskQueue.push(PreviewTaskItem{std::move(fn)});
    }
    m_previewTaskCv.notify_one();
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

static void DownsampleNearestRgba(const std::vector<unsigned char> &src, int srcW, int srcH, int maxPx,
                                  std::vector<unsigned char> &dst, int &dstW, int &dstH)
{
    if (srcW <= 0 || srcH <= 0 || src.empty()) {
        dst.clear();
        dstW = 0;
        dstH = 0;
        return;
    }

    if (maxPx <= 0 || (srcW <= maxPx && srcH <= maxPx)) {
        dst = src;
        dstW = srcW;
        dstH = srcH;
        return;
    }

    const float scale = static_cast<float>(maxPx) / static_cast<float>(std::max(srcW, srcH));
    dstW = std::max(1, static_cast<int>(srcW * scale));
    dstH = std::max(1, static_cast<int>(srcH * scale));
    dst.resize(static_cast<size_t>(dstW) * static_cast<size_t>(dstH) * 4u);

    const int rowStride = srcW * 4;
    for (int dy = 0; dy < dstH; ++dy) {
        const int sy = std::min(static_cast<int>((dy + 0.5f) * srcH / dstH), srcH - 1);
        const int rowOff = sy * rowStride;
        for (int dx = 0; dx < dstW; ++dx) {
            const int sx = std::min(static_cast<int>((dx + 0.5f) * srcW / dstW), srcW - 1);
            const int srcIdx = rowOff + sx * 4;
            const int dstIdx = (dy * dstW + dx) * 4;
            dst[dstIdx + 0] = src[srcIdx + 0];
            dst[dstIdx + 1] = src[srcIdx + 1];
            dst[dstIdx + 2] = src[srcIdx + 2];
            dst[dstIdx + 3] = src[srcIdx + 3];
        }
    }
}

static void ApplySrgbPreviewInPlace(std::vector<unsigned char> &pixels)
{
    if (pixels.empty())
        return;

    uint8_t lut[256];
    for (int i = 0; i < 256; ++i) {
        const float v = std::pow(static_cast<float>(i) / 255.0f, 1.0f / 2.2f);
        lut[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(v * 255.0f + 0.5f), 0, 255));
    }

    for (size_t i = 0; i + 3 < pixels.size(); i += 4) {
        pixels[i + 0] = lut[pixels[i + 0]];
        pixels[i + 1] = lut[pixels[i + 1]];
        pixels[i + 2] = lut[pixels[i + 2]];
    }
}

bool Infernux::ScheduleMaterialPreviewTask(const std::string &resourceKey, const std::string &matFilePath,
                                           uint64_t stamp)
{
    // Legacy wrapper — delegates to the unified query function with mtime hint.
    QueryOrScheduleMaterialPreview(resourceKey, matFilePath, "", stamp);
    return true;
}

uint64_t Infernux::QueryOrScheduleMaterialPreview(const std::string &resourceKey, const std::string &matFilePath,
                                                  const std::string &materialJson, uint64_t fileMtimeHint)
{
    if (resourceKey.empty())
        return 0;

    if (!m_previewTaskSystemInitialized)
        InitPreviewTaskSystem(1);

    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto &state = m_materialPreviewStates[resourceKey];
    if (state.textureName.empty())
        state.textureName = BuildPreviewTextureName(resourceKey);

    // ── Detect content changes ──────────────────────────────────
    std::string renderJson; // JSON to use if we schedule a render

    if (!materialJson.empty()) {
        const uint64_t h = std::hash<std::string>{}(materialJson);
        if (h != state.lastJsonHash) {
            state.lastJsonHash = h;
            state.generation++;
        }
        renderJson = materialJson; // prefer JSON for rendering
    }

    if (fileMtimeHint != 0 && fileMtimeHint != state.lastFileMtime) {
        state.lastFileMtime = fileMtimeHint;
        // Only bump generation from mtime if no JSON was provided in this call
        // (avoids double-bump when both are present).
        if (materialJson.empty())
            state.generation++;
    }

    // ── Already up-to-date? ─────────────────────────────────────
    if (state.readyGeneration == state.generation && state.textureId != 0)
        return state.textureId;

    // ── Schedule render if not already in flight ────────────────
    if (!state.inFlight && state.readyGeneration < state.generation) {
        state.inFlight = true;
        m_previewRequestQueue.push(MaterialPreviewRequest{resourceKey, matFilePath, state.generation, renderJson});
    }

    // Stale-return: keep showing old preview while new one renders (no flicker).
    return state.textureId;
}

bool Infernux::ScheduleTexturePreviewTask(const std::string &resourceKey, const std::string &textureFilePath,
                                          uint64_t stamp, bool nearest, bool srgb)
{
    // Legacy wrapper — delegates to the unified query function with content stamp hint.
    QueryOrScheduleTexturePreview(resourceKey, textureFilePath, stamp, nearest, srgb, false);
    return true;
}

void Infernux::PumpPreviewTasks()
{
    if (!m_renderer)
        return;

    // ── Per-frame guard ──────────────────────────────────────────
    // Multiple call-sites (ProjectPanel::PreRender, Python inspector,
    // Python texture queries) may invoke PumpPreviewTasks within the
    // same ImGui frame.  Only the first call per frame does real work.
    const int currentFrame = ImGui::GetFrameCount();
    if (m_lastPumpFrame == currentFrame)
        return;
    m_lastPumpFrame = currentFrame;

    // Global per-frame budget for GPU uploads (each UploadTextureForImGui
    // does BeginSingleTimeCommands + fence wait ~1-2 ms).  We cap total
    // uploads (material + texture) to keep the frame under budget.
    // Raised to 3 so startup batch can upload 2 materials + 1 texture.
    constexpr int kMaxUploadsPerFrame = 3;
    int uploadBudget = kMaxUploadsPerFrame;

    // ── Material render + upload (inline, single-phase) ──────────
    // Material rendering is synchronous on the main thread (needs Vulkan
    // GPU context), so there is no benefit in a 2-phase render→completed→
    // upload pipeline — that only added 1 frame of latency per material.
    // We now render AND upload in the same frame.
    //
    // Batch mode  (queueSize ≥ 2): render+upload up to 2 per frame,
    //   no cooldown.  This covers bootstrap prewarm and multi-material
    //   invalidation — N materials finish in ~⌈N/2⌉ frames.
    // Interactive (queueSize < 2): render+upload 1 with 300 ms cooldown
    //   to avoid stalling the UI during continuous slider dragging.
    {
        size_t queueSize = 0;
        {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            queueSize = m_previewRequestQueue.size();
        }

        if (queueSize > 0 && uploadBudget > 0) {
            const bool batchMode = (queueSize >= 2);
            constexpr int kMaterialCooldownMs = 300;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastMaterialRenderTime);

            int maxRenders = 0;
            if (batchMode) {
                // Startup / batch: render multiple, no cooldown.
                maxRenders = std::min(uploadBudget, std::min(static_cast<int>(queueSize), 2));
            } else if (elapsed.count() >= kMaterialCooldownMs) {
                // Interactive: single render with cooldown.
                maxRenders = std::min(uploadBudget, 1);
            }

            for (int renderIdx = 0; renderIdx < maxRenders; ++renderIdx) {
                MaterialPreviewRequest req;
                {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    if (m_previewRequestQueue.empty())
                        break;
                    req = std::move(m_previewRequestQueue.front());
                    m_previewRequestQueue.pop();
                }
                if (req.resourceKey.empty())
                    continue;

                // ── Render to pixels (synchronous GPU, ~5-10 ms) ─────
                std::vector<unsigned char> pixels;
                AssetDatabase *adb = GetAssetDatabase();
                bool ok = false;
                if (!req.materialJson.empty()) {
                    // Render from in-memory JSON snapshot (Inspector live edits).
                    ok = MaterialPreviewer::RenderFromJson(req.materialJson, 256, pixels, adb, m_renderer.get());
                } else {
                    std::string embedModel;
                    int embedSlot = -1;
                    if (ParseModelEmbeddedMaterialSlot(req.matFilePath, embedModel, embedSlot)) {
                        ok = MaterialPreviewer::RenderModelEmbeddedMaterialToPixels(
                            embedModel, static_cast<uint32_t>(embedSlot), 256, pixels, adb, m_renderer.get());
                    } else {
                        // Render from disk file (ProjectPanel thumbnails).
                        ok = MaterialPreviewer::RenderToPixels(req.matFilePath, 256, pixels, adb, m_renderer.get());
                    }
                }

                if (!ok || pixels.empty()) {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    auto it = m_materialPreviewStates.find(req.resourceKey);
                    if (it != m_materialPreviewStates.end()) {
                        it->second.inFlight = false;
                    }
                    continue;
                }

                // ── Upload to ImGui (synchronous GPU, ~1-2 ms) ───────
                std::string texName;
                {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    auto it = m_materialPreviewStates.find(req.resourceKey);
                    if (it == m_materialPreviewStates.end())
                        continue;
                    it->second.inFlight = false;
                    if (it->second.textureName.empty())
                        it->second.textureName = BuildPreviewTextureName(req.resourceKey);
                    texName = it->second.textureName;
                }
                if (texName.empty())
                    continue;

                const uint64_t texId =
                    m_renderer->UploadTextureForImGui(texName, pixels.data(), 256, 256, VK_FILTER_LINEAR);

                {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    auto it = m_materialPreviewStates.find(req.resourceKey);
                    if (it != m_materialPreviewStates.end() && texId != 0) {
                        it->second.textureId = texId;
                        it->second.readyGeneration = req.generation;
                        it->second.readySize = 256;
                    }
                }
                --uploadBudget;
                m_lastMaterialRenderTime = now;
            }
        }
    }

    // ── Process completed texture uploads ────────────────────────
    {
        std::queue<TexturePreviewCompleted> texLocal;
        {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            texLocal.swap(m_texturePreviewCompletedQueue);
        }

        while (!texLocal.empty() && uploadBudget > 0) {
            TexturePreviewCompleted completed = std::move(texLocal.front());
            texLocal.pop();

            TexturePreviewState stateSnapshot;
            {
                std::lock_guard<std::mutex> lock(m_previewResultMutex);
                auto it = m_texturePreviewStates.find(completed.resourceKey);
                if (it == m_texturePreviewStates.end())
                    continue;

                it->second.inFlight = false;
                if (it->second.textureName.empty())
                    it->second.textureName = BuildTexturePreviewTextureName(completed.resourceKey);
                stateSnapshot = it->second;
            }

            if (!completed.success || completed.pixels.empty() || completed.width <= 0 || completed.height <= 0)
                continue;

            if (stateSnapshot.textureName.empty())
                continue;

            const uint64_t texId = m_renderer->UploadTextureForImGui(
                stateSnapshot.textureName, completed.pixels.data(), completed.width, completed.height,
                completed.nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);

            {
                std::lock_guard<std::mutex> lock(m_previewResultMutex);
                auto it = m_texturePreviewStates.find(completed.resourceKey);
                if (it == m_texturePreviewStates.end())
                    continue;

                if (texId != 0) {
                    it->second.textureId = texId;
                    it->second.readyGeneration = completed.generation;
                    it->second.readyWidth = completed.width;
                    it->second.readyHeight = completed.height;
                }
            }
            --uploadBudget;
        }

        // Put unconsumed items back for next frame.
        if (!texLocal.empty()) {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            while (!texLocal.empty()) {
                m_texturePreviewCompletedQueue.push(std::move(texLocal.front()));
                texLocal.pop();
            }
        }
    }

    // ── Mesh preview render + upload (inline, single-phase) ─────
    // Same approach as material previews: synchronous GPU render on
    // main thread.  Mesh loading goes through AssetRegistry which
    // caches already-loaded meshes, so cost is mainly the GPU render.
    {
        size_t queueSize = 0;
        {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            queueSize = m_meshPreviewRequestQueue.size();
        }

        if (queueSize > 0 && uploadBudget > 0) {
            int maxRenders = std::min(uploadBudget, std::min(static_cast<int>(queueSize), 2));

            for (int renderIdx = 0; renderIdx < maxRenders; ++renderIdx) {
                MeshPreviewRequest req;
                {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    if (m_meshPreviewRequestQueue.empty())
                        break;
                    req = std::move(m_meshPreviewRequestQueue.front());
                    m_meshPreviewRequestQueue.pop();
                }
                if (req.resourceKey.empty())
                    continue;

                std::shared_ptr<InxMesh> mesh;
                std::vector<std::shared_ptr<InxMaterial>> materials;

                if (IsPrefabPreviewPath(req.meshFilePath)) {
                    if (!BuildPrefabPreviewMesh(req.meshFilePath, mesh, materials)) {
                        std::lock_guard<std::mutex> lock(m_previewResultMutex);
                        auto it = m_meshPreviewStates.find(req.resourceKey);
                        if (it != m_meshPreviewStates.end())
                            it->second.inFlight = false;
                        continue;
                    }
                } else {
                    mesh = AssetRegistry::Instance().LoadAssetByPath<InxMesh>(req.meshFilePath, ResourceType::Mesh);
                    if (mesh)
                        materials = BuildDefaultPreviewMaterialsForMesh(*mesh);
                }

                if (!mesh) {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    auto it = m_meshPreviewStates.find(req.resourceKey);
                    if (it != m_meshPreviewStates.end())
                        it->second.inFlight = false;
                    continue;
                }

                // ── GPU render (synchronous, ~5-15 ms) ──────────────
                constexpr int kMeshPreviewSize = 256;
                std::vector<unsigned char> pixels;
                bool ok = m_renderer->RenderMeshPreviewGPU(*mesh, materials, kMeshPreviewSize, pixels);

                if (!ok || pixels.empty()) {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    auto it = m_meshPreviewStates.find(req.resourceKey);
                    if (it != m_meshPreviewStates.end())
                        it->second.inFlight = false;
                    continue;
                }

                // ── Upload to ImGui (synchronous GPU, ~1-2 ms) ──────
                std::string texName;
                {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    auto it = m_meshPreviewStates.find(req.resourceKey);
                    if (it == m_meshPreviewStates.end())
                        continue;
                    it->second.inFlight = false;
                    if (it->second.textureName.empty())
                        it->second.textureName = BuildMeshPreviewTextureName(req.resourceKey);
                    texName = it->second.textureName;
                }
                if (texName.empty())
                    continue;

                const uint64_t texId = m_renderer->UploadTextureForImGui(texName, pixels.data(), kMeshPreviewSize,
                                                                         kMeshPreviewSize, VK_FILTER_LINEAR);

                {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    auto it = m_meshPreviewStates.find(req.resourceKey);
                    if (it != m_meshPreviewStates.end() && texId != 0) {
                        it->second.textureId = texId;
                        it->second.readyGeneration = req.generation;
                        it->second.readySize = kMeshPreviewSize;
                    }
                }
                --uploadBudget;
            }
        }
    }
}

void Infernux::FlushAllMaterialPreviews()
{
    if (!m_renderer)
        return;

    // Drain the entire material request queue synchronously.
    // This is intended for bootstrap prewarm only — no budget, no cooldown.
    for (;;) {
        MaterialPreviewRequest req;
        {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            if (m_previewRequestQueue.empty())
                break;
            req = std::move(m_previewRequestQueue.front());
            m_previewRequestQueue.pop();
        }
        if (req.resourceKey.empty())
            continue;

        // Render to pixels
        std::vector<unsigned char> pixels;
        AssetDatabase *adb = GetAssetDatabase();
        bool ok = false;
        if (!req.materialJson.empty()) {
            ok = MaterialPreviewer::RenderFromJson(req.materialJson, 256, pixels, adb, m_renderer.get());
        } else {
            std::string embedModel;
            int embedSlot = -1;
            if (ParseModelEmbeddedMaterialSlot(req.matFilePath, embedModel, embedSlot)) {
                ok = MaterialPreviewer::RenderModelEmbeddedMaterialToPixels(
                    embedModel, static_cast<uint32_t>(embedSlot), 256, pixels, adb, m_renderer.get());
            } else {
                ok = MaterialPreviewer::RenderToPixels(req.matFilePath, 256, pixels, adb, m_renderer.get());
            }
        }

        if (!ok || pixels.empty()) {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            auto it = m_materialPreviewStates.find(req.resourceKey);
            if (it != m_materialPreviewStates.end()) {
                it->second.inFlight = false;
            }
            continue;
        }

        // Upload to ImGui
        std::string texName;
        {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            auto it = m_materialPreviewStates.find(req.resourceKey);
            if (it == m_materialPreviewStates.end())
                continue;
            it->second.inFlight = false;
            if (it->second.textureName.empty())
                it->second.textureName = BuildPreviewTextureName(req.resourceKey);
            texName = it->second.textureName;
        }
        if (texName.empty())
            continue;

        const uint64_t texId = m_renderer->UploadTextureForImGui(texName, pixels.data(), 256, 256, VK_FILTER_LINEAR);

        {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            auto it = m_materialPreviewStates.find(req.resourceKey);
            if (it != m_materialPreviewStates.end() && texId != 0) {
                it->second.textureId = texId;
                it->second.readyGeneration = req.generation;
                it->second.readySize = 256;
            }
        }
    }

    m_lastMaterialRenderTime = std::chrono::steady_clock::now();
}

uint64_t Infernux::GetMaterialPreviewTextureId(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_materialPreviewStates.find(resourceKey);
    if (it == m_materialPreviewStates.end())
        return 0;
    return it->second.textureId;
}

uint64_t Infernux::GetTexturePreviewTextureId(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_texturePreviewStates.find(resourceKey);
    if (it == m_texturePreviewStates.end())
        return 0;
    return it->second.textureId;
}

std::pair<int, int> Infernux::GetTexturePreviewSize(const std::string &resourceKey) const
{
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_texturePreviewStates.find(resourceKey);
    if (it == m_texturePreviewStates.end())
        return {0, 0};
    return {it->second.readyWidth, it->second.readyHeight};
}

void Infernux::InvalidateMaterialPreviewTask(const std::string &resourceKey)
{
    if (resourceKey.empty())
        return;

    // Bump generation so next query re-renders.  Keep old textureId for
    // stale-return anti-flicker.  Reset content hashes so both sources
    // (JSON and mtime) re-evaluate on next call.
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_materialPreviewStates.find(resourceKey);
    if (it != m_materialPreviewStates.end()) {
        it->second.generation++;
        it->second.lastJsonHash = 0;
        it->second.lastFileMtime = 0;
        it->second.inFlight = false;
    }
}

void Infernux::InvalidateTexturePreviewTask(const std::string &resourceKey)
{
    if (resourceKey.empty())
        return;

    // Bump generation so next query re-renders.  Keep old textureId for
    // stale-return anti-flicker.  Reset content stamp so next call re-evaluates.
    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto it = m_texturePreviewStates.find(resourceKey);
    if (it != m_texturePreviewStates.end()) {
        it->second.generation++;
        it->second.lastContentStamp = 0;
        it->second.inFlight = false;
    }
}

std::tuple<uint64_t, int, int> Infernux::QueryOrScheduleTexturePreview(const std::string &resourceKey,
                                                                       const std::string &textureFilePath,
                                                                       uint64_t contentStampHint, bool nearest,
                                                                       bool srgb, bool pump)
{
    if (resourceKey.empty() || textureFilePath.empty())
        return {0, 0, 0};

    if (pump)
        PumpPreviewTasks();

    if (!m_previewTaskSystemInitialized)
        InitPreviewTaskSystem(1);

    bool shouldEnqueue = false;
    TexturePreviewRequest req;
    uint64_t texId = 0;
    int w = 0, h = 0;

    {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        auto &state = m_texturePreviewStates[resourceKey];
        if (state.textureName.empty())
            state.textureName = BuildTexturePreviewTextureName(resourceKey);

        // ── Detect content changes ──────────────────────────────
        if (contentStampHint != 0 && contentStampHint != state.lastContentStamp) {
            state.lastContentStamp = contentStampHint;
            state.generation++;
        }

        // Also bump generation if filter/srgb settings changed.
        if (state.textureId != 0 && (nearest != state.nearest || srgb != state.srgb)) {
            state.generation++;
        }
        state.nearest = nearest;
        state.srgb = srgb;

        // Stale-return: keep showing old preview while new one loads.
        texId = state.textureId;
        w = state.readyWidth;
        h = state.readyHeight;

        // Already up-to-date?
        if (state.readyGeneration == state.generation && state.textureId != 0)
            return {texId, w, h};

        // Schedule render if not already in flight.
        if (!state.inFlight && state.readyGeneration < state.generation) {
            state.inFlight = true;
            req = TexturePreviewRequest{resourceKey, textureFilePath, state.generation, nearest, srgb};
            shouldEnqueue = true;
        }
    }

        if (shouldEnqueue) {
            constexpr int kDefaultPreviewResolution = 256;
            // Inspector component icons (~16 logical px, often 2x framebuffer); keep GPU texture near
            // display density instead of a 256px atlas + heavy minification.
            constexpr int kComponentIconPreviewMaxDim = 64;
            EnqueuePreviewTask([this, req, kDefaultPreviewResolution, kComponentIconPreviewMaxDim]() {
                TexturePreviewCompleted completed;
                completed.resourceKey = req.resourceKey;
                completed.generation = req.generation;
                completed.nearest = req.nearest;

                auto texData = InxTextureLoader::LoadFromFile(req.textureFilePath);
                if (!texData.IsValid()) {
                    std::lock_guard<std::mutex> lock(m_previewResultMutex);
                    m_texturePreviewCompletedQueue.push(std::move(completed));
                    return;
                }

                std::vector<unsigned char> sampled;
                int outW = 0;
                int outH = 0;
                const int maxDim = (!req.resourceKey.empty() && req.resourceKey.compare(0, 9, "compicon|") == 0)
                                       ? kComponentIconPreviewMaxDim
                                       : kDefaultPreviewResolution;
                DownsampleNearestRgba(texData.pixels, texData.width, texData.height, maxDim, sampled, outW, outH);
            if (sampled.empty() || outW <= 0 || outH <= 0) {
                std::lock_guard<std::mutex> lock(m_previewResultMutex);
                m_texturePreviewCompletedQueue.push(std::move(completed));
                return;
            }

            if (req.srgb)
                ApplySrgbPreviewInPlace(sampled);

            completed.width = outW;
            completed.height = outH;
            completed.success = true;
            completed.pixels = std::move(sampled);

            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            m_texturePreviewCompletedQueue.push(std::move(completed));
        });
    }

    // Stale-return: keep showing old preview while new one loads (no flicker).
    return {texId, w, h};
}

bool Infernux::ScheduleTexturePreviewFromMemory(const std::string &resourceKey,
                                                const std::vector<unsigned char> &imageData, uint64_t stamp,
                                                bool nearest)
{
    if (resourceKey.empty() || imageData.empty())
        return false;

    if (!m_previewTaskSystemInitialized)
        InitPreviewTaskSystem(1);

    uint64_t gen = 0;
    {
        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        auto &state = m_texturePreviewStates[resourceKey];
        if (state.textureName.empty())
            state.textureName = BuildTexturePreviewTextureName(resourceKey);

        // Use caller's stamp as content-change hint.
        if (stamp != 0 && stamp != state.lastContentStamp) {
            state.lastContentStamp = stamp;
            state.generation++;
        }

        if (state.readyGeneration == state.generation && state.textureId != 0)
            return true;

        if (state.inFlight)
            return true; // already in-flight

        state.inFlight = true;
        state.nearest = nearest;
        gen = state.generation;
    }

    // Copy data so worker thread owns it.
    auto dataCopy = std::make_shared<std::vector<unsigned char>>(imageData);
    const std::string keyCopy = resourceKey;
    const uint64_t genCopy = gen;
    const bool nearestCopy = nearest;

    EnqueuePreviewTask([this, keyCopy, dataCopy, genCopy, nearestCopy]() {
        TexturePreviewCompleted completed;
        completed.resourceKey = keyCopy;
        completed.generation = genCopy;
        completed.nearest = nearestCopy;

        auto texData = InxTextureLoader::LoadFromMemory(dataCopy->data(), dataCopy->size());
        if (!texData.IsValid()) {
            std::lock_guard<std::mutex> lock(m_previewResultMutex);
            m_texturePreviewCompletedQueue.push(std::move(completed));
            return;
        }

        completed.width = texData.width;
        completed.height = texData.height;
        completed.success = true;
        completed.pixels = std::move(texData.pixels);

        std::lock_guard<std::mutex> lock(m_previewResultMutex);
        m_texturePreviewCompletedQueue.push(std::move(completed));
    });

    return true;
}

uint64_t Infernux::QueryOrScheduleMeshPreview(const std::string &resourceKey, const std::string &meshFilePath,
                                              uint64_t fileMtimeHint)
{
    if (resourceKey.empty() || meshFilePath.empty())
        return 0;

    if (!m_previewTaskSystemInitialized)
        InitPreviewTaskSystem(1);

    std::lock_guard<std::mutex> lock(m_previewResultMutex);
    auto &state = m_meshPreviewStates[resourceKey];
    if (state.textureName.empty())
        state.textureName = BuildMeshPreviewTextureName(resourceKey);
    state.meshFilePath = meshFilePath;

    // ── Detect content changes ──────────────────────────────────
    if (fileMtimeHint != 0 && fileMtimeHint != state.lastFileMtime) {
        state.lastFileMtime = fileMtimeHint;
        state.generation++;
    }

    // First request: generation was never bumped, force it so we render once.
    if (state.generation == 0)
        state.generation = 1;

    // ── Already up-to-date? ─────────────────────────────────────
    if (state.readyGeneration == state.generation && state.textureId != 0)
        return state.textureId;

    // ── Schedule render if not already in flight ────────────────
    if (!state.inFlight && state.readyGeneration < state.generation) {
        state.inFlight = true;
        m_meshPreviewRequestQueue.push(MeshPreviewRequest{resourceKey, meshFilePath, state.generation});
    }

    // Stale-return: keep showing old preview while new one renders (no flicker).
    return state.textureId;
}

bool Infernux::ScheduleMaterialSaveSnapshotTask(const std::string &key, const std::string &filePath,
                                                const std::string &jsonSnapshot)
{
    (void)key;
    if (filePath.empty())
        return false;

    if (!m_previewTaskSystemInitialized)
        InitPreviewTaskSystem(1);

    const std::string pathCopy = filePath;
    const std::string jsonCopy = jsonSnapshot;
    EnqueuePreviewTask([pathCopy, jsonCopy]() {
        try {
            std::ofstream out(ToFsPath(pathCopy), std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                INXLOG_WARN("ScheduleMaterialSaveSnapshotTask: cannot open file for write: ", pathCopy);
                return;
            }
            out.write(jsonCopy.data(), static_cast<std::streamsize>(jsonCopy.size()));
            out.flush();
        } catch (const std::exception &ex) {
            INXLOG_WARN("ScheduleMaterialSaveSnapshotTask failed for ", pathCopy, ": ", ex.what());
        } catch (...) {
            INXLOG_WARN("ScheduleMaterialSaveSnapshotTask failed for ", pathCopy, ": unknown exception");
        }
    });

    return true;
}

// ----------------------------------
// Renderer initialization
// ----------------------------------

void Infernux::InitRenderer(int width, int height, const std::string &projectPath,
                            const std::string &builtinResourcePath)
{
    if (!CheckEngineValid("initialize renderer") || !m_renderer) {
        INXLOG_ERROR("Cannot initialize renderer: Renderer is not available.");
        return;
    }

    m_renderer->Init(width, height, m_metadata);

    // Wire SceneManager to renderer so Play()/Stop() directly bypass idle
    // sleep without relying on the Python callback chain timing.
    {
        auto *renderer = m_renderer.get();
        SceneManager::Instance().SetPlayStateChangedCallback([renderer](bool playing) {
            if (renderer)
                renderer->SetPlayModeRendering(playing);
        });
    }
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

    INXLOG_DEBUG("Load shaders.");
    std::string defaultShaderPath = JoinPath({builtinResourcePath, "shaders"});
    std::string assetsPath = JoinPath({projectPath, "Assets"});
    if (m_assetDatabase) {
        // Register the builtin shader search path for @import resolution
        InxShaderLoader::AddShaderSearchPath(defaultShaderPath);

        m_assetDatabase->Initialize(projectPath);

        // ── Transfer AssetDatabase ownership to AssetRegistry ──────
        auto &registry = AssetRegistry::Instance();
        registry.Initialize(std::move(m_assetDatabase));

        // Register loader plug-ins for all asset types
        registry.RegisterLoader(ResourceType::Material, std::make_unique<MaterialLoader>());
        registry.RegisterLoader(ResourceType::Texture, std::make_unique<TextureLoader>());
        registry.RegisterLoader(ResourceType::Mesh, std::make_unique<MeshLoader>());
        registry.RegisterLoader(ResourceType::Audio, std::make_unique<AudioClipLoader>());
        registry.RegisterLoader(ResourceType::Shader, std::make_unique<ShaderLoader>());
        registry.RegisterLoader(ResourceType::Script, std::make_unique<InxPythonScriptLoader>());
        registry.RegisterLoader(ResourceType::DefaultText, std::make_unique<InxDefaultTextLoader>());
        registry.RegisterLoader(ResourceType::DefaultBinary, std::make_unique<InxDefaultBinaryLoader>());

        // Populate AssetDatabase's meta-loader table from registered loaders
        registry.PopulateAssetDatabaseLoaders();

        // Register the builtin resource directory as an extra scan root
        // so that Library/Resources assets (materials, etc.) get GUIDs.
        if (!builtinResourcePath.empty())
            registry.GetAssetDatabase()->AddScanRoot(builtinResourcePath);

        registry.GetAssetDatabase()->Refresh();

        // ── Load and register shaders via AssetRegistry ─────────────
        LoadAndRegisterShaders(defaultShaderPath, false);
        LoadAndRegisterShaders(assetsPath, true);

        // ── Register unified asset event callbacks ──────────────────
        auto &graph = AssetDependencyGraph::Instance();

        auto resolveMaterial = [](const std::string &matGuid) -> std::shared_ptr<InxMaterial> {
            auto mat = AssetRegistry::Instance().GetAsset<InxMaterial>(matGuid);
            if (mat)
                return mat;
            auto *adb = AssetRegistry::Instance().GetAssetDatabase();
            if (adb) {
                std::string matPath = adb->GetPathFromGuid(matGuid);
                if (!matPath.empty())
                    mat = AssetRegistry::Instance().LoadAssetByPath<InxMaterial>(matPath, ResourceType::Material);
            }
            return mat;
        };

        graph.RegisterCallback(ResourceType::Texture, [this, resolveMaterial](const std::string &dependentGuid,
                                                                              const std::string &texGuid,
                                                                              AssetEvent event) {
            auto mat = resolveMaterial(dependentGuid);
            if (!mat)
                return;

            if (event == AssetEvent::Deleted) {
                bool changed = false;
                for (const auto &[propName, prop] : mat->GetAllProperties()) {
                    if (prop.type != MaterialPropertyType::Texture2D)
                        continue;
                    const auto *val = std::get_if<std::string>(&prop.value);
                    if (!val || *val != texGuid)
                        continue;
                    mat->ClearTexture(propName);
                    changed = true;
                    INXLOG_INFO("AssetGraph: cleared texture '", propName, "' from material '", mat->GetName(), "'");
                }
                if (changed)
                    mat->SaveToFile();
            }

            if (event == AssetEvent::Deleted || event == AssetEvent::Modified) {
                if (m_renderer) {
                    std::string matName = mat->GetMaterialKey();
                    if (matName.empty())
                        matName = mat->GetName();
                    m_renderer->RemoveMaterialPipeline(matName);
                    mat->MarkPropertiesDirty();
                    INXLOG_INFO("AssetGraph: invalidated pipeline for material '", matName, "' (texture changed)");
                }
            }
        });

        graph.RegisterCallback(ResourceType::Material,
                               [](const std::string &dependentGuid, const std::string & /*matGuid*/, AssetEvent event) {
                                   if (event != AssetEvent::Deleted)
                                       return;
                                   uint64_t compId = 0;
                                   try {
                                       compId = std::stoull(dependentGuid);
                                   } catch (...) {
                                       return;
                                   }
                                   auto *comp = Component::FindByComponentId(compId);
                                   if (!comp)
                                       return;
                                   auto *mr = dynamic_cast<MeshRenderer *>(comp);
                                   if (!mr)
                                       return;
                                   auto fallback = AssetRegistry::Instance().GetBuiltinMaterial("ErrorMaterial");
                                   if (fallback)
                                       mr->SetMaterial(0, fallback);
                                   INXLOG_INFO("AssetGraph: reassigned MeshRenderer to error material");
                               });

        graph.RegisterCallback(ResourceType::Mesh, [](const std::string &dependentGuid,
                                                      const std::string & /*meshGuid*/, AssetEvent event) {
            uint64_t compId = 0;
            try {
                compId = std::stoull(dependentGuid);
            } catch (...) {
                return;
            }
            auto *comp = Component::FindByComponentId(compId);
            if (!comp)
                return;
            auto *mr = dynamic_cast<MeshRenderer *>(comp);
            if (!mr)
                return;
            mr->OnMeshAssetEvent(event);
            INXLOG_INFO("AssetGraph: refreshed MeshRenderer mesh state");
        });

        graph.RegisterCallback(ResourceType::Shader, [this, resolveMaterial](const std::string &dependentGuid,
                                                                             const std::string & /*shaderGuid*/,
                                                                             AssetEvent event) {
            if (event != AssetEvent::Modified && event != AssetEvent::Deleted)
                return;
            auto mat = resolveMaterial(dependentGuid);
            if (!mat)
                return;
            mat->MarkPipelineDirty();
            INXLOG_INFO("AssetGraph: marked material '", mat->GetName(), "' pipeline dirty (shader changed)");
        });
    }

    INXLOG_DEBUG("Prepare pipeline.");
    m_renderer->PreparePipeline();

    // Set ImGui ini file path to user's Documents folder for per-project
    // layout persistence (keeps project directory clean / not in VCS).
    // We use std::filesystem::path throughout (wide-char on Windows) so
    // paths with non-ASCII characters (e.g. Chinese usernames) work.
    {
        std::filesystem::path layoutDir;
#ifdef INX_PLATFORM_WINDOWS
        wchar_t docsPath[MAX_PATH] = {};
        if (SHGetFolderPathW(nullptr, CSIDL_PERSONAL, nullptr, SHGFP_TYPE_CURRENT, docsPath) == S_OK) {
            std::filesystem::path projFs = ToFsPath(projectPath);
            std::filesystem::path projectNameFs = projFs.filename();
            layoutDir = std::filesystem::path(docsPath) / L"Infernux" / projectNameFs;
        }
#else
        const char *home = std::getenv("HOME");
        if (home) {
            std::filesystem::path projFs = ToFsPath(projectPath);
            std::filesystem::path projectNameFs = projFs.filename();
            layoutDir = std::filesystem::path(home) / ".config" / "Infernux" / projectNameFs;
        }
#endif
        if (layoutDir.empty()) {
            layoutDir = ToFsPath(projectPath);
        }
        std::filesystem::create_directories(layoutDir);
        m_imguiIniPath = layoutDir / "imgui.ini";
    }
    // Disable ImGui auto-save (it uses fopen which can't handle Unicode
    // paths on Windows). We manually load/save with std::fstream instead.
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    LoadImGuiLayout();

    // Initialize physics world (Jolt)
    PhysicsWorld::Instance().Initialize();

    // Initialize audio engine (SDL3 audio)
    if (!AudioEngine::Instance().Initialize()) {
        INXLOG_WARN("Audio engine failed to initialize. Audio features will be unavailable.");
    }
}

// ----------------------------------
// Mesh geometry extraction helper (shared by SetSelectionOutline / SetSelectionOutlines)
// ----------------------------------

static bool ExtractMeshGeometry(MeshRenderer *renderer, std::vector<glm::vec3> &positions,
                                std::vector<glm::vec3> &normals, std::vector<uint32_t> &indices)
{
    positions.clear();
    normals.clear();
    indices.clear();

    if (renderer->HasInlineMesh()) {
        const auto &verts = renderer->GetInlineVertices();
        positions.reserve(verts.size());
        normals.reserve(verts.size());
        for (const auto &v : verts) {
            positions.push_back(v.pos);
            normals.push_back(v.normal);
        }
        indices = renderer->GetInlineIndices();
    } else if (renderer->HasMeshAsset()) {
        auto mesh = renderer->GetMeshAssetRef().Get();
        if (!mesh || mesh->GetVertices().empty() || mesh->GetIndices().empty())
            return false;

        const auto &meshVertices = mesh->GetVertices();
        const auto &meshIndices = mesh->GetIndices();
        int32_t nodeGroup = renderer->GetNodeGroup();

        if (nodeGroup >= 0) {
            std::unordered_map<uint32_t, uint32_t> vertexRemap;
            for (const auto &sub : mesh->GetSubMeshes()) {
                if (static_cast<int32_t>(sub.nodeGroup) != nodeGroup)
                    continue;
                for (uint32_t i = 0; i < sub.indexCount; ++i) {
                    uint32_t origIdx = meshIndices[sub.indexStart + i];
                    auto it = vertexRemap.find(origIdx);
                    if (it == vertexRemap.end()) {
                        uint32_t newIdx = static_cast<uint32_t>(positions.size());
                        vertexRemap[origIdx] = newIdx;
                        positions.push_back(meshVertices[origIdx].pos);
                        normals.push_back(meshVertices[origIdx].normal);
                        indices.push_back(newIdx);
                    } else {
                        indices.push_back(it->second);
                    }
                }
            }
        } else {
            positions.reserve(meshVertices.size());
            normals.reserve(meshVertices.size());
            for (const auto &v : meshVertices) {
                positions.push_back(v.pos);
                normals.push_back(v.normal);
            }
            indices = meshIndices;
        }
    } else {
        return false;
    }

    return !positions.empty() && !indices.empty();
}

// ----------------------------------
// Editor Gizmos
// ----------------------------------

void Infernux::SetSelectionOutline(uint64_t objectId)
{
    if (m_isCleanedUp || !m_renderer) {
        return;
    }

    m_cachedOutlineIds.clear();
    m_selectedObjectId = objectId;
    m_renderer->SetSelectedObjectId(objectId);

    auto &gizmos = m_renderer->GetEditorGizmos();

    if (objectId == 0) {
        gizmos.ClearSelectionOutline();
        return;
    }

    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (!scene) {
        gizmos.ClearSelectionOutline();
        return;
    }

    GameObject *obj = scene->FindByID(objectId);
    if (!obj || !obj->IsActiveInHierarchy()) {
        gizmos.ClearSelectionOutline();
        return;
    }

    MeshRenderer *renderer = obj->GetComponent<MeshRenderer>();
    if (!renderer || !renderer->IsEnabled()) {
        gizmos.ClearSelectionOutline();
        return;
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<uint32_t> indices;

    if (!ExtractMeshGeometry(renderer, positions, normals, indices)) {
        gizmos.ClearSelectionOutline();
        return;
    }

    glm::mat4 worldMatrix = obj->GetTransform()->GetWorldMatrix();
    gizmos.SetSelectionOutline(positions, normals, indices, worldMatrix);
}

void Infernux::ClearSelectionOutline()
{
    if (m_isCleanedUp || !m_renderer) {
        return;
    }
    m_cachedOutlineIds.clear();
    m_selectedObjectId = 0;
    m_renderer->SetSelectedObjectId(0);
    m_renderer->GetEditorGizmos().ClearSelectionOutline();
}

void Infernux::SetSelectionOutlines(const std::vector<uint64_t> &objectIds)
{
    if (m_isCleanedUp || !m_renderer) {
        return;
    }

    // Fast path: skip expensive mesh extraction when the ID set is unchanged.
    if (objectIds == m_cachedOutlineIds) {
        return;
    }
    m_cachedOutlineIds = objectIds;

    auto &gizmos = m_renderer->GetEditorGizmos();

    if (objectIds.empty()) {
        gizmos.ClearSelectionOutline();
        return;
    }

    if (objectIds.size() == 1) {
        SetSelectionOutline(objectIds[0]);
        return;
    }

    Scene *scene = SceneManager::Instance().GetActiveScene();
    if (!scene) {
        gizmos.ClearSelectionOutline();
        return;
    }

    std::vector<glm::vec3> mergedPositions;
    std::vector<glm::vec3> mergedNormals;
    std::vector<uint32_t> mergedIndices;

    for (uint64_t objId : objectIds) {
        GameObject *obj = scene->FindByID(objId);
        if (!obj || !obj->IsActiveInHierarchy())
            continue;

        MeshRenderer *renderer = obj->GetComponent<MeshRenderer>();
        if (!renderer || !renderer->IsEnabled())
            continue;

        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;
        std::vector<uint32_t> indices;

        if (!ExtractMeshGeometry(renderer, positions, normals, indices))
            continue;

        glm::mat4 worldMatrix = obj->GetTransform()->GetWorldMatrix();
        glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(worldMatrix)));

        uint32_t baseIndex = static_cast<uint32_t>(mergedPositions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            glm::vec4 wp = worldMatrix * glm::vec4(positions[i], 1.0f);
            mergedPositions.push_back(glm::vec3(wp));
            glm::vec3 wn = glm::normalize(normalMatrix * normals[i]);
            mergedNormals.push_back(wn);
        }
        for (uint32_t idx : indices) {
            mergedIndices.push_back(idx + baseIndex);
        }
    }

    if (mergedPositions.empty() || mergedIndices.empty()) {
        gizmos.ClearSelectionOutline();
        return;
    }

    gizmos.SetSelectionOutline(mergedPositions, mergedNormals, mergedIndices, glm::mat4(1.0f));

    // Store the first valid ID for frame-by-frame matrix updates
    m_selectedObjectId = objectIds[0];
    m_renderer->SetSelectedObjectId(objectIds[0]);
}

// ----------------------------------
// Material Pipeline
// ----------------------------------

void Infernux::RegisterShaderToRenderer(const ShaderAsset &asset)
{
    if (!m_renderer || asset.spirvForward.empty())
        return;

    m_renderer->LoadShader(asset.shaderId.c_str(), asset.spirvForward,
                           asset.shaderType == "vertex" ? "vertex" : "fragment");

    // Shadow vertex variant
    if (asset.shaderType == "vertex" && !asset.spirvShadowVertex.empty()) {
        std::string shadowId = asset.shaderId + "/shadow";
        m_renderer->LoadShader(shadowId.c_str(), asset.spirvShadowVertex, "vertex");
        INXLOG_INFO("Registered shadow vertex variant '", shadowId, "'");
    }

    // Shadow fragment variant
    if (asset.shaderType == "fragment" && !asset.spirvShadow.empty()) {
        std::string shadowId = asset.shaderId + "/shadow";
        m_renderer->LoadShader(shadowId.c_str(), asset.spirvShadow, "fragment");
        INXLOG_INFO("Registered shadow fragment variant '", shadowId, "'");
    }

    // GBuffer fragment variant
    if (asset.shaderType == "fragment" && !asset.spirvGBuffer.empty()) {
        std::string gbufferId = asset.shaderId + "/gbuffer";
        m_renderer->LoadShader(gbufferId.c_str(), asset.spirvGBuffer, "fragment");
        INXLOG_INFO("Registered GBuffer variant '", gbufferId, "'");
    }

    // Render-state metadata (fragment shaders only)
    if (asset.shaderType == "fragment") {
        const auto &rm = asset.renderMeta;
        m_renderer->StoreShaderRenderMeta(asset.shaderId, rm.cullMode, rm.depthWrite, rm.depthTest, rm.blend, rm.queue,
                                          rm.passTag, rm.stencil, rm.alphaClip);
    }
}

void Infernux::LoadAndRegisterShaders(const std::string &dir, bool recursive)
{
    namespace fs = std::filesystem;
    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    if (!adb || !m_renderer)
        return;

    fs::path dirPath = ToFsPath(dir);
    if (!fs::exists(dirPath))
        return;

    std::unordered_set<std::string> loadedShaderKeys;
    std::vector<char> defaultVertCode;
    std::vector<char> defaultFragCode;

    auto processEntry = [&](const fs::directory_entry &entry) {
        if (!entry.is_regular_file())
            return;
        fs::path file = entry.path();
        std::string ext = file.extension().string();

        if (ext != ".vert" && ext != ".frag")
            return;

        std::string filePath = FromFsPath(file);

        // Always register to ensure meta + GUID↔path mappings are cached
        std::string guid = adb->RegisterResource(filePath, ResourceType::Shader);
        if (guid.empty())
            return;

        // Get shader_id from meta
        std::string shaderId;
        const InxResourceMeta *meta = adb->GetMetaByGuid(guid);
        if (meta && meta->HasKey("shader_id")) {
            shaderId = meta->GetDataAs<std::string>("shader_id");
        }
        if (shaderId.empty()) {
            shaderId = FromFsPath(file.stem());
        }

        std::string shaderKey = shaderId + "_" + ext;

        // Skip duplicates
        if (loadedShaderKeys.count(shaderKey))
            return;

        // For recursive (asset) shaders, skip if already loaded in renderer
        if (recursive && m_renderer->HasShader(shaderId, ext == ".vert" ? "vertex" : "fragment")) {
            loadedShaderKeys.insert(shaderKey);
            return;
        }

        loadedShaderKeys.insert(shaderKey);

        // Load via AssetRegistry (compiles the shader)
        auto shaderAsset = registry.LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
        if (!shaderAsset || shaderAsset->spirvForward.empty())
            return;

        // Register all variants with the renderer
        RegisterShaderToRenderer(*shaderAsset);

        INXLOG_DEBUG("Loaded shader '", shaderId, "' (", ext, ") from ", filePath);

        // Track built-in fallback shaders used for the renderer's default program.
        if (!recursive) {
            if (shaderId == "standard" && ext == ".vert")
                defaultVertCode = shaderAsset->spirvForward;
            else if (shaderId == "unlit" && ext == ".frag")
                defaultFragCode = shaderAsset->spirvForward;
        }
    };

    if (recursive) {
        for (const auto &entry : fs::recursive_directory_iterator(dirPath))
            processEntry(entry);
    } else {
        for (const auto &entry : fs::directory_iterator(dirPath))
            processEntry(entry);
    }

    // Register fallback shaders (non-recursive = builtin shaders only)
    if (!recursive) {
        if (!defaultVertCode.empty()) {
            m_renderer->LoadShader("default", defaultVertCode, "vertex");
            INXLOG_INFO("Registered 'standard' as default vertex shader");
        }
        if (!defaultFragCode.empty()) {
            m_renderer->LoadShader("default", defaultFragCode, "fragment");
            INXLOG_INFO("Registered 'unlit' as default fragment shader");
        }
    }
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

    return ReloadShader(shaderPath).empty();
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

std::string Infernux::ReloadShader(const std::string &shaderPath)
{
    INXLOG_INFO("Infernux::ReloadShader called: ", shaderPath);
    if (!CheckEngineValid("reload shader") || !m_renderer) {
        INXLOG_ERROR("Infernux::ReloadShader: engine or renderer invalid");
        return "Engine or renderer invalid";
    }

    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    if (!adb) {
        INXLOG_ERROR("Infernux::ReloadShader: no AssetDatabase available");
        return "No AssetDatabase available";
    }

    std::filesystem::path path = ToFsPath(shaderPath);
    std::string ext = path.extension().string();

    if (ext != ".vert" && ext != ".frag") {
        INXLOG_ERROR("Infernux::ReloadShader: unsupported shader extension: ", ext);
        return "Unsupported shader extension: " + ext;
    }

    // Get or create GUID for this shader
    std::string guid = adb->GetGuidFromPath(shaderPath);
    std::string oldShaderId;

    if (!guid.empty()) {
        // Get old shader_id before recompilation
        const InxResourceMeta *existingMeta = adb->GetMetaByGuid(guid);
        if (existingMeta && existingMeta->HasKey("shader_id")) {
            oldShaderId = existingMeta->GetDataAs<std::string>("shader_id");
        }
    }

    // Re-register meta (updates shader_id annotations from source)
    adb->ModifyResource(shaderPath);

    // Invalidate shader-id map cache for this directory so shading models
    // and imports added/modified since the last compile are discovered.
    InxShaderLoader::InvalidateDirectoryCache(FromFsPath(ToFsPath(shaderPath).parent_path()));
    InxShaderLoader::InvalidateTemplateCache();

    if (guid.empty()) {
        guid = adb->RegisterResource(shaderPath, ResourceType::Shader);
    } else {
        // Refresh the guid from the updated meta
        guid = adb->GetGuidFromPath(shaderPath);
    }

    if (guid.empty()) {
        INXLOG_ERROR("Infernux::ReloadShader: failed to register resource: ", shaderPath);
        return "Failed to register resource: " + shaderPath;
    }

    // Invalidate the AssetRegistry cache so the shader gets recompiled
    registry.InvalidateAsset(guid);

    // Reload via AssetRegistry → ShaderLoader
    auto shaderAsset = registry.LoadAsset<ShaderAsset>(guid, ResourceType::Shader);
    if (!shaderAsset || shaderAsset->spirvForward.empty()) {
        INXLOG_ERROR("Infernux::ReloadShader: compilation failed for: ", shaderPath);
        std::string compileErr = InxShaderLoader::s_lastCompileError;
        if (!compileErr.empty())
            return compileErr;
        return "Shader compilation failed (no compiled data)";
    }

    // Invalidate renderer caches BEFORE loading new shader code
    m_renderer->InvalidateShaderCache(shaderAsset->shaderId);
    if (!oldShaderId.empty() && oldShaderId != shaderAsset->shaderId) {
        m_renderer->InvalidateShaderCache(oldShaderId);
        INXLOG_INFO("Infernux::ReloadShader: shader_id changed from '", oldShaderId, "' to '", shaderAsset->shaderId,
                    "'");
    }

    // Also invalidate variant caches
    m_renderer->InvalidateShaderCache(shaderAsset->shaderId + "/shadow");
    m_renderer->InvalidateShaderCache(shaderAsset->shaderId + "/gbuffer");

    // Register all SPIR-V variants with the renderer
    RegisterShaderToRenderer(*shaderAsset);

    INXLOG_INFO("Infernux::ReloadShader: reloaded shader '", shaderAsset->shaderId, "' from ", shaderPath);

    // Refresh all materials using this shader
    m_renderer->RefreshMaterialsUsingShader(shaderAsset->shaderId);

    // If shader_id changed, update materials that referenced the old name
    if (!oldShaderId.empty() && oldShaderId != shaderAsset->shaderId) {
        auto materials = registry.GetAllMaterials();
        for (auto &material : materials) {
            if (!material)
                continue;
            if (material->GetFragShaderName() == oldShaderId) {
                material->SetFragShader(shaderAsset->shaderId);
                INXLOG_INFO("Infernux::ReloadShader: updated material '", material->GetName(), "' frag shader from '",
                            oldShaderId, "' to '", shaderAsset->shaderId, "'");
            }
            if (material->GetVertShaderName() == oldShaderId) {
                material->SetVertShader(shaderAsset->shaderId);
                INXLOG_INFO("Infernux::ReloadShader: updated material '", material->GetName(), "' vert shader from '",
                            oldShaderId, "' to '", shaderAsset->shaderId, "'");
            }
        }
        m_renderer->RefreshMaterialsUsingShader(shaderAsset->shaderId);
    }

    return ""; // success
}

void Infernux::ReloadTexture(const std::string &texturePath)
{
    INXLOG_INFO("Infernux::ReloadTexture called: ", texturePath);

    if (!CheckEngineValid("reload texture") || !m_renderer) {
        INXLOG_ERROR("Infernux::ReloadTexture: engine or renderer invalid");
        return;
    }

    // Resolve path → GUID so that InvalidateTextureCache can match GUID-based cache keys
    auto &registry = AssetRegistry::Instance();
    auto *adb = registry.GetAssetDatabase();
    std::string guid;
    if (adb)
        guid = adb->GetGuidFromPath(texturePath);

    if (guid.empty()) {
        INXLOG_WARN("Infernux::ReloadTexture: could not resolve GUID for '", texturePath, "'");
    }

    // Reload InxTexture metadata in AssetRegistry (import settings may have changed)
    if (!guid.empty() && registry.IsLoaded(guid)) {
        registry.ReloadAsset(guid);

        // Log the reloaded import settings for diagnostics
        auto infTex = registry.LoadAsset<InxTexture>(guid, ResourceType::Texture);
        if (infTex) {
            INXLOG_INFO(
                "Infernux::ReloadTexture: InxTexture reloaded — IsLinear=", infTex->IsLinear() ? "true" : "false",
                ", GenerateMipmaps=", infTex->GenerateMipmaps() ? "true" : "false", ", GUID=", guid);
        }
    }

    // InvalidateTextureCache accepts both GUID and path — prefer GUID
    m_renderer->InvalidateTextureCache(!guid.empty() ? guid : texturePath);

    // Fire graph notification so dependent materials get their pipelines
    // invalidated via the Texture Modified callback.
    if (!guid.empty()) {
        auto dependents = AssetDependencyGraph::Instance().GetDependents(guid);
        INXLOG_INFO("Infernux::ReloadTexture: NotifyEvent guid=", guid, " dependents=", dependents.size());
        AssetDependencyGraph::Instance().NotifyEvent(guid, ResourceType::Texture, AssetEvent::Modified);
    }

    INXLOG_INFO("Infernux::ReloadTexture: done for '", texturePath, "'");
}

void Infernux::ReloadMesh(const std::string &meshPath)
{
    INXLOG_INFO("Infernux::ReloadMesh called: ", meshPath);

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

    if (registry.IsLoaded(guid))
        registry.ReloadAsset(guid);

    auto dependents = AssetDependencyGraph::Instance().GetDependents(guid);
    INXLOG_INFO("Infernux::ReloadMesh: NotifyEvent guid=", guid, " dependents=", dependents.size());
    AssetDependencyGraph::Instance().NotifyEvent(guid, ResourceType::Mesh, AssetEvent::Modified);

    INXLOG_INFO("Infernux::ReloadMesh: done for '", meshPath, "'");
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
    INXLOG_SET_LEVEL(engineLevel);
    m_logLevel = engineLevel;
}

// ----------------------------------
// ImGui layout save / load (Unicode-safe)
// ----------------------------------

void Infernux::ResetImGuiLayout()
{
    // Clear ImGui's in-memory ini state (windows, docking, tables)
    ImGui::ClearIniSettings();
    // Delete the persisted ini file so the reset survives a restart
    if (!m_imguiIniPath.empty() && std::filesystem::exists(m_imguiIniPath)) {
        std::filesystem::remove(m_imguiIniPath);
    }
}

void Infernux::SelectDockedWindow(const std::string &windowId)
{
    auto *renderer = GetRenderer();
    if (renderer == nullptr) {
        return;
    }
    renderer->QueueDockTabSelection(windowId.c_str());
}

void Infernux::LoadImGuiLayout()
{
    if (!std::filesystem::exists(m_imguiIniPath))
        return;
    // std::ifstream(std::filesystem::path) uses wchar_t on Windows,
    // so paths with Chinese / non-ASCII characters are handled properly.
    std::ifstream ifs(m_imguiIniPath, std::ios::binary | std::ios::ate);
    if (!ifs.is_open())
        return;
    auto size = ifs.tellg();
    if (size <= 0)
        return;
    ifs.seekg(0);
    std::string data(static_cast<size_t>(size), '\0');
    ifs.read(data.data(), size);
    ImGui::LoadIniSettingsFromMemory(data.c_str(), data.size());
}

void Infernux::SaveImGuiLayout()
{
    if (m_imguiIniPath.empty())
        return;
    size_t dataSize = 0;
    const char *data = ImGui::SaveIniSettingsToMemory(&dataSize);
    if (!data || dataSize == 0)
        return;
    std::filesystem::create_directories(m_imguiIniPath.parent_path());
    std::ofstream ofs(m_imguiIniPath, std::ios::binary);
    if (ofs.is_open())
        ofs.write(data, static_cast<std::streamsize>(dataSize));
}

} // namespace infernux