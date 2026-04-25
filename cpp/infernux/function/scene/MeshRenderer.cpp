#include "MeshRenderer.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "SceneManager.h"
#include <core/log/InxLog.h>
#include <cstring>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/scene/PrimitiveMeshes.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>

using json = nlohmann::json;

namespace infernux
{

namespace
{

bool MeshDataEquals(const std::shared_ptr<InxMesh> &mesh, const std::vector<Vertex> &vertices,
                    const std::vector<uint32_t> &indices)
{
    if (!mesh)
        return false;
    if (mesh->GetVertices().size() != vertices.size() || mesh->GetIndices().size() != indices.size())
        return false;

    const bool sameVertices = vertices.empty() || std::memcmp(mesh->GetVertices().data(), vertices.data(),
                                                              vertices.size() * sizeof(Vertex)) == 0;
    const bool sameIndices = indices.empty() || std::memcmp(mesh->GetIndices().data(), indices.data(),
                                                            indices.size() * sizeof(uint32_t)) == 0;
    return sameVertices && sameIndices;
}

std::string FindMatchingMeshAssetGuid(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                                      const std::string &preferredName)
{
    if (vertices.empty() || indices.empty())
        return {};

    auto &registry = AssetRegistry::Instance();
    auto *assetDb = registry.GetAssetDatabase();
    if (!assetDb)
        return {};

    auto tryFind = [&](bool requirePreferredName) -> std::string {
        for (const auto &guid : assetDb->GetAllResourceGuids()) {
            const auto *meta = assetDb->GetMetaByGuid(guid);
            if (!meta || meta->GetResourceType() != ResourceType::Mesh)
                continue;

            if (requirePreferredName) {
                if (preferredName.empty() || meta->GetResourceName() != preferredName)
                    continue;
            }

            auto mesh = registry.GetAsset<InxMesh>(guid);
            if (!mesh)
                mesh = registry.LoadAsset<InxMesh>(guid, ResourceType::Mesh);
            if (!MeshDataEquals(mesh, vertices, indices))
                continue;
            return guid;
        }
        return {};
    };

    if (!preferredName.empty()) {
        if (auto guid = tryFind(true); !guid.empty())
            return guid;
    }

    return tryFind(false);
}

bool GetBuiltinPrimitiveMeshData(const std::string &name, const std::vector<Vertex> *&vertices,
                                 const std::vector<uint32_t> *&indices)
{
    vertices = nullptr;
    indices = nullptr;

    if (name == "Cube") {
        vertices = &PrimitiveMeshes::GetCubeVertices();
        indices = &PrimitiveMeshes::GetCubeIndices();
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

bool MatchesBuiltinPrimitiveMesh(const std::string &name, const std::vector<Vertex> &vertices,
                                 const std::vector<uint32_t> &indices)
{
    const std::vector<Vertex> *builtinVertices = nullptr;
    const std::vector<uint32_t> *builtinIndices = nullptr;
    if (!GetBuiltinPrimitiveMeshData(name, builtinVertices, builtinIndices))
        return false;

    if (builtinVertices->size() != vertices.size() || builtinIndices->size() != indices.size())
        return false;

    const bool sameVertices = builtinVertices->empty() || std::memcmp(builtinVertices->data(), vertices.data(),
                                                                      vertices.size() * sizeof(Vertex)) == 0;
    const bool sameIndices = builtinIndices->empty() || std::memcmp(builtinIndices->data(), indices.data(),
                                                                    indices.size() * sizeof(uint32_t)) == 0;
    return sameVertices && sameIndices;
}

void RestoreBuiltinPrimitiveMesh(const std::string &name, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices)
{
    const std::vector<Vertex> *builtinVertices = nullptr;
    const std::vector<uint32_t> *builtinIndices = nullptr;
    if (!GetBuiltinPrimitiveMeshData(name, builtinVertices, builtinIndices))
        return;

    vertices.assign(builtinVertices->begin(), builtinVertices->end());
    indices.assign(builtinIndices->begin(), builtinIndices->end());
}

constexpr const char *kEmbeddedMaterialToken = "::submat:";

bool ParseEmbeddedMaterialSource(const std::string &sourcePath, std::string &outModelPath, int &outSlot)
{
    const size_t pos = sourcePath.find(kEmbeddedMaterialToken);
    if (pos == std::string::npos)
        return false;

    outModelPath = sourcePath.substr(0, pos);
    try {
        outSlot = std::stoi(sourcePath.substr(pos + std::strlen(kEmbeddedMaterialToken)));
    } catch (...) {
        return false;
    }
    return !outModelPath.empty() && outSlot >= 0;
}

std::shared_ptr<InxMaterial> BuildEmbeddedMaterialFromSource(const std::string &sourcePath)
{
    std::string modelPath;
    int slot = -1;
    if (!ParseEmbeddedMaterialSource(sourcePath, modelPath, slot))
        return nullptr;

    auto mesh = AssetRegistry::Instance().LoadAssetByPath<InxMesh>(modelPath, ResourceType::Mesh);
    if (!mesh)
        return nullptr;

    const auto &slotData = mesh->GetMaterialSlotData();
    if (slot < 0 || slot >= static_cast<int>(slotData.size()))
        return nullptr;

    auto defaultMat = AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
    if (!defaultMat)
        return nullptr;

    auto mat = defaultMat->Clone();
    const auto &sd = slotData[static_cast<size_t>(slot)];
    mat->SetColor("baseColor", sd.baseColor);
    mat->SetColor("emissionColor", sd.emissionColor);
    mat->SetFloat("metallic", sd.metallic);
    mat->SetFloat("smoothness", sd.smoothness);

    const auto &slotNames = mesh->GetMaterialSlotNames();
    if (slot < static_cast<int>(slotNames.size()) && !slotNames[static_cast<size_t>(slot)].empty())
        mat->SetName(slotNames[static_cast<size_t>(slot)]);
    else
        mat->SetName("EmbeddedMaterial_" + std::to_string(slot));

    mat->SetFilePath(sourcePath);
    return mat;
}

void NotifyRenderableStateChanged(MeshRenderer *renderer)
{
    if (renderer)
        SceneManager::Instance().NotifyMeshRendererChanged(renderer);
}

} // namespace

INFERNUX_REGISTER_COMPONENT("MeshRenderer", MeshRenderer)

MeshRenderer::~MeshRenderer()
{
    // Remove all dependency edges from this instance in the unified graph
    AssetDependencyGraph::Instance().ClearDependenciesOf(GetInstanceGuid());
    // Safety net: ensure we're removed from the registry even if
    // OnDisable wasn't called (e.g. direct destruction during scene teardown).
    SceneManager::Instance().UnregisterMeshRenderer(this);
}

void MeshRenderer::OnEnable()
{
    // Only register with the global renderer list if this object belongs to
    // the active scene.  Objects living in utility scenes (e.g. the prefab
    // template cache) must not pollute the active scene's render list.
    if (auto *go = GetGameObject())
        if (go->GetScene() != SceneManager::Instance().GetActiveScene())
            return;
    SceneManager::Instance().RegisterMeshRenderer(this);
}

void MeshRenderer::OnDisable()
{
    SceneManager::Instance().UnregisterMeshRenderer(this);
}

void MeshRenderer::SetMesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices)
{
    if (m_meshAsset.HasGuid())
        AssetDependencyGraph::Instance().RemoveDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;
    m_inlineVertices = std::move(vertices);
    m_inlineIndices = std::move(indices);
    m_useInlineMesh = true;
    m_meshAsset.Clear();
    m_meshBufferDirty = true;
    ComputeLocalBoundsFromInlineVertices();
    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetSharedPrimitiveMesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                                          const std::string &primitiveName)
{
    if (m_meshAsset.HasGuid())
        AssetDependencyGraph::Instance().RemoveDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_sharedVertices = &vertices;
    m_sharedIndices = &indices;
    m_useInlineMesh = true;
    m_inlineMeshName = primitiveName;
    m_meshAsset.Clear();
    m_meshBufferDirty = true;

    // Cache bounds per primitive type (keyed by static vertex data address).
    // Avoids iterating all vertices for every identical primitive.
    static std::unordered_map<const void *, std::pair<glm::vec3, glm::vec3>> s_boundsCache;
    auto it = s_boundsCache.find(&vertices);
    if (it != s_boundsCache.end()) {
        m_localBoundsMin = it->second.first;
        m_localBoundsMax = it->second.second;
    } else {
        ComputeLocalBoundsFromInlineVertices();
        s_boundsCache[&vertices] = {m_localBoundsMin, m_localBoundsMax};
    }

    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMeshAsset(const std::string &guid, std::shared_ptr<InxMesh> mesh)
{
    auto &graph = AssetDependencyGraph::Instance();
    if (m_meshAsset.HasGuid() && m_meshAsset.GetGuid() != guid)
        graph.RemoveDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_meshAsset = AssetRef<InxMesh>(guid, std::move(mesh));
    m_useInlineMesh = false;
    m_meshBufferDirty = true;
    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;

    if (!guid.empty())
        graph.AddDependency(GetInstanceGuid(), guid);

    auto m = m_meshAsset.Get();
    if (m) {
        if (m_nodeGroup >= 0)
            UpdateBoundsForNodeGroup(m);
        else
            SetLocalBounds(m->GetBoundsMin(), m->GetBoundsMax());
    }

    SyncMaterialSlotsToMesh();
    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMeshAssetGuid(const std::string &guid)
{
    auto &graph = AssetDependencyGraph::Instance();
    if (m_meshAsset.HasGuid() && m_meshAsset.GetGuid() != guid)
        graph.RemoveDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_meshAsset.SetGuid(guid);
    m_useInlineMesh = false;
    m_meshBufferDirty = true;
    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;

    if (!guid.empty())
        graph.AddDependency(GetInstanceGuid(), guid);

    NotifyRenderableStateChanged(this);
}

void MeshRenderer::ClearMeshAsset()
{
    if (m_meshAsset.HasGuid())
        AssetDependencyGraph::Instance().RemoveDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_meshAsset.Clear();
    m_meshBufferDirty = true;
    m_useInlineMesh = false;
    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;
    m_localBoundsMin = glm::vec3(-0.5f);
    m_localBoundsMax = glm::vec3(0.5f);
    NotifyRenderableStateChanged(this);
}

void MeshRenderer::OnMeshAssetEvent(AssetEvent event)
{
    if (!m_meshAsset.HasGuid())
        return;

    if (event == AssetEvent::Deleted) {
        ClearMeshAsset();
        return;
    }

    AssetRegistry::Instance().Resolve(m_meshAsset, ResourceType::Mesh);
    auto mesh = m_meshAsset.Get();
    if (!mesh)
        return;

    if (m_nodeGroup >= 0)
        UpdateBoundsForNodeGroup(mesh);
    else
        SetLocalBounds(mesh->GetBoundsMin(), mesh->GetBoundsMax());
    SyncMaterialSlotsToMesh();
    MarkMeshBufferDirty();
    NotifyRenderableStateChanged(this);
}

bool MeshRenderer::ConsumeMeshBufferDirty()
{
    bool d = m_meshBufferDirty;
    m_meshBufferDirty = false;
    return d;
}

void MeshRenderer::SetMaterial(uint32_t slot, std::shared_ptr<InxMaterial> material)
{
    if (slot >= m_materials.size())
        m_materials.resize(slot + 1);

    auto &ref = m_materials[slot];
    auto oldMat = ref.Get();
    if (oldMat == material)
        return;

    auto &graph = AssetDependencyGraph::Instance();
    if (oldMat && !oldMat->GetGuid().empty())
        graph.RemoveDependency(GetInstanceGuid(), oldMat->GetGuid());

    std::string guid = material ? material->GetGuid() : "";
    ref = AssetRef<InxMaterial>(guid, std::move(material));

    auto newMat = ref.Get();
    if (newMat && !newMat->GetGuid().empty())
        graph.AddDependency(GetInstanceGuid(), newMat->GetGuid());

    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMaterial(uint32_t slot, const std::string &guid)
{
    if (slot >= m_materials.size())
        m_materials.resize(slot + 1);

    auto &ref = m_materials[slot];
    auto oldMat = ref.Get();
    auto &graph = AssetDependencyGraph::Instance();
    if (oldMat && !oldMat->GetGuid().empty())
        graph.RemoveDependency(GetInstanceGuid(), oldMat->GetGuid());

    ref.SetGuid(guid);
    AssetRegistry::Instance().Resolve(ref, ResourceType::Material);

    auto newMat = ref.Get();
    if (newMat && !newMat->GetGuid().empty())
        graph.AddDependency(GetInstanceGuid(), newMat->GetGuid());

    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMaterials(const std::vector<std::string> &guids)
{
    // Clear old dependency edges
    auto &graph = AssetDependencyGraph::Instance();
    for (auto &ref : m_materials) {
        auto mat = ref.Get();
        if (mat && !mat->GetGuid().empty())
            graph.RemoveDependency(GetInstanceGuid(), mat->GetGuid());
    }

    m_materials.resize(guids.size());
    for (uint32_t i = 0; i < guids.size(); ++i) {
        m_materials[i].SetGuid(guids[i]);
        AssetRegistry::Instance().Resolve(m_materials[i], ResourceType::Material);

        auto newMat = m_materials[i].Get();
        if (newMat && !newMat->GetGuid().empty())
            graph.AddDependency(GetInstanceGuid(), newMat->GetGuid());
    }

    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMaterialSlotCount(uint32_t count)
{
    if (count == m_materials.size())
        return;

    // Remove dependency edges for slots being removed
    auto &graph = AssetDependencyGraph::Instance();
    for (uint32_t i = count; i < m_materials.size(); ++i) {
        auto mat = m_materials[i].Get();
        if (mat && !mat->GetGuid().empty())
            graph.RemoveDependency(GetInstanceGuid(), mat->GetGuid());
    }
    m_materials.resize(count);
    NotifyRenderableStateChanged(this);
}

std::shared_ptr<InxMaterial> MeshRenderer::GetMaterial(uint32_t slot) const
{
    if (slot >= m_materials.size())
        return nullptr;
    return m_materials[slot].Get();
}

std::shared_ptr<InxMaterial> MeshRenderer::GetEffectiveMaterial(uint32_t slot) const
{
    auto mat = GetMaterial(slot);
    if (mat) {
        if (!mat->IsDeleted())
            return mat;
        auto &registry = AssetRegistry::Instance();
        auto err = registry.GetBuiltinMaterial("ErrorMaterial");
        return err ? err : registry.GetBuiltinMaterial("DefaultLit");
    }
    return AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
}

std::string MeshRenderer::GetMaterialGuid(uint32_t slot) const
{
    if (slot >= m_materials.size())
        return "";
    return m_materials[slot].GetGuid();
}

std::vector<std::string> MeshRenderer::GetMaterialGuids() const
{
    std::vector<std::string> guids;
    guids.reserve(m_materials.size());
    for (const auto &ref : m_materials)
        guids.push_back(ref.GetGuid());
    return guids;
}

void MeshRenderer::SyncMaterialSlotsToMesh()
{
    if (!HasMeshAsset())
        return;
    auto mesh = m_meshAsset.Get();
    if (!mesh)
        return;

    // Single-submesh mode: only 1 material slot needed
    if (m_submeshIndex >= 0) {
        if (m_materials.size() < 1)
            m_materials.resize(1);
        return;
    }

    // Node-group mode: count unique material slots within this node group
    if (m_nodeGroup >= 0) {
        std::set<uint32_t> uniqueSlots;
        for (const auto &sub : mesh->GetSubMeshes()) {
            if (static_cast<int32_t>(sub.nodeGroup) == m_nodeGroup)
                uniqueSlots.insert(sub.materialSlot);
        }
        uint32_t needed = std::max(static_cast<uint32_t>(uniqueSlots.size()), 1u);
        if (m_materials.size() != needed)
            SetMaterialSlotCount(needed);
        return;
    }

    uint32_t needed = std::max(mesh->GetMaterialSlotCount(), 1u);
    if (m_materials.size() != needed)
        SetMaterialSlotCount(needed);
}

void MeshRenderer::ComputeLocalBoundsFromInlineVertices()
{
    const auto &verts = GetInlineVertices();
    if (verts.empty()) {
        m_localBoundsMin = glm::vec3(-0.5f);
        m_localBoundsMax = glm::vec3(0.5f);
        return;
    }

    glm::vec3 bmin(std::numeric_limits<float>::max());
    glm::vec3 bmax(std::numeric_limits<float>::lowest());
    for (const auto &v : verts) {
        bmin = glm::min(bmin, v.pos);
        bmax = glm::max(bmax, v.pos);
    }
    m_localBoundsMin = bmin;
    m_localBoundsMax = bmax;
}

void MeshRenderer::SetNodeGroup(int32_t group)
{
    m_nodeGroup = group;
    if (HasMeshAsset()) {
        auto mesh = m_meshAsset.Get();
        if (mesh) {
            if (m_nodeGroup >= 0)
                UpdateBoundsForNodeGroup(mesh);
            else
                SetLocalBounds(mesh->GetBoundsMin(), mesh->GetBoundsMax());
            SyncMaterialSlotsToMesh();
        }
    }
    NotifyRenderableStateChanged(this);
}

void MeshRenderer::UpdateBoundsForNodeGroup(const std::shared_ptr<InxMesh> &mesh)
{
    constexpr float INF = std::numeric_limits<float>::max();
    glm::vec3 bmin(INF);
    glm::vec3 bmax(-INF);
    bool found = false;
    for (const auto &sub : mesh->GetSubMeshes()) {
        if (static_cast<int32_t>(sub.nodeGroup) == m_nodeGroup) {
            bmin = glm::min(bmin, sub.boundsMin);
            bmax = glm::max(bmax, sub.boundsMax);
            found = true;
        }
    }
    if (found)
        SetLocalBounds(bmin, bmax);
}

void MeshRenderer::GetWorldBounds(glm::vec3 &outMin, glm::vec3 &outMax) const
{
    if (!m_gameObject) {
        outMin = m_localBoundsMin;
        outMax = m_localBoundsMax;
        return;
    }

    const Transform *transform = m_gameObject->GetTransform();
    const glm::mat4 &worldMatrix = transform->GetWorldMatrix();
    ComputeWorldBounds(worldMatrix, outMin, outMax);
}

void MeshRenderer::ComputeWorldBounds(const glm::mat4 &worldMatrix, glm::vec3 &outMin, glm::vec3 &outMax) const
{
    // Arvo's AABB transform method: O(9) multiply-adds instead of
    // 8 × mat4×vec4 (O(32) multiplies).  For each output axis, we
    // compute the transformed center ± transformed half-extent.
    const glm::vec3 center = (m_localBoundsMin + m_localBoundsMax) * 0.5f;
    const glm::vec3 extent = (m_localBoundsMax - m_localBoundsMin) * 0.5f;

    glm::vec3 newCenter, newExtent;
    for (int i = 0; i < 3; ++i) {
        newCenter[i] = worldMatrix[3][i]; // translation
        newExtent[i] = 0.0f;
        for (int j = 0; j < 3; ++j) {
            float e = worldMatrix[j][i]; // column-major: M[col][row]
            newCenter[i] += e * center[j];
            newExtent[i] += std::abs(e) * extent[j];
        }
    }

    outMin = newCenter - newExtent;
    outMax = newCenter + newExtent;
}

std::string MeshRenderer::Serialize() const
{
    json j;
    j["schema_version"] = 4;
    j["type"] = GetTypeName();
    j["enabled"] = IsEnabled();
    j["component_id"] = GetComponentID();

    // Mesh reference
    j["meshId"] = m_mesh.meshId;

    const bool builtinPrimitive =
        m_useInlineMesh && !m_inlineMeshName.empty() &&
        MatchesBuiltinPrimitiveMesh(m_inlineMeshName, GetInlineVertices(), GetInlineIndices());
    const std::string matchedInlineMeshGuid =
        (!HasMeshAsset() && m_useInlineMesh && !builtinPrimitive)
            ? FindMatchingMeshAssetGuid(GetInlineVertices(), GetInlineIndices(), m_inlineMeshName)
            : std::string();
    const std::string serializedMeshGuid = HasMeshAsset() ? m_meshAsset.GetGuid() : matchedInlineMeshGuid;

    // Mesh asset GUID (for model-file meshes managed by AssetRegistry)
    if (!serializedMeshGuid.empty()) {
        j["meshAssetGuid"] = serializedMeshGuid;
    }

    // Materials:
    // - v3 stored GUID-or-null entries only.
    // - v4 also stores inline/runtime material snapshots for guid-less slots
    //   (used by FBX embedded materials and other non-asset overrides).
    json materialsJson = json::array();
    for (const auto &ref : m_materials) {
        const auto &guid = ref.GetGuid();
        if (!guid.empty()) {
            materialsJson.push_back(guid);
            continue;
        }

        auto mat = ref.Get();
        if (!mat) {
            materialsJson.push_back(nullptr);
            continue;
        }

        json slotJson = json::object();
        const std::string &sourcePath = mat->GetFilePath();
        if (!sourcePath.empty())
            slotJson["source_path"] = sourcePath;

        const std::string matJson = mat->Serialize();
        if (!matJson.empty())
            slotJson["material_json"] = matJson;

        materialsJson.push_back(slotJson.empty() ? json(nullptr) : slotJson);
    }
    j["materials"] = materialsJson;

    // Rendering flags
    j["castShadows"] = m_castShadows;
    j["receivesShadows"] = m_receiveShadows;

    // Submesh filter
    if (m_submeshIndex >= 0) {
        j["submeshIndex"] = m_submeshIndex;
    }

    // Node group filter
    if (m_nodeGroup >= 0) {
        j["nodeGroup"] = m_nodeGroup;
    }

    // Mesh pivot offset (for submesh centering)
    if (m_meshPivotOffset != glm::vec3(0.0f)) {
        j["meshPivotOffset"] = {m_meshPivotOffset.x, m_meshPivotOffset.y, m_meshPivotOffset.z};
    }

    // Bounds
    j["boundsMin"] = {m_localBoundsMin.x, m_localBoundsMin.y, m_localBoundsMin.z};
    j["boundsMax"] = {m_localBoundsMax.x, m_localBoundsMax.y, m_localBoundsMax.z};

    // Inline mesh data (for primitives and procedural geometry)
    const bool serializeInlineMesh = m_useInlineMesh && serializedMeshGuid.empty();
    j["useInlineMesh"] = serializeInlineMesh;
    if (!m_inlineMeshName.empty()) {
        j["inlineMeshName"] = m_inlineMeshName;
    }
    if (serializeInlineMesh) {
        if (builtinPrimitive) {
            j["inlineMeshBuiltin"] = true;
        } else {
            json verticesJson = json::array();
            for (const auto &v : GetInlineVertices()) {
                json vj;
                vj["pos"] = {v.pos.x, v.pos.y, v.pos.z};
                vj["normal"] = {v.normal.x, v.normal.y, v.normal.z};
                vj["tangent"] = {v.tangent.x, v.tangent.y, v.tangent.z, v.tangent.w};
                vj["color"] = {v.color.x, v.color.y, v.color.z};
                vj["texCoord"] = {v.texCoord.x, v.texCoord.y};
                verticesJson.push_back(vj);
            }
            j["inlineVertices"] = verticesJson;

            json indicesJson = json::array();
            for (uint32_t idx : GetInlineIndices()) {
                indicesJson.push_back(idx);
            }
            j["inlineIndices"] = indicesJson;
        }
    }

    return j.dump(2);
}

bool MeshRenderer::Deserialize(const std::string &jsonStr)
{
    try {
        json j = json::parse(jsonStr);

        // Call base class deserialize
        Component::Deserialize(jsonStr);

        // Mesh reference
        if (j.contains("meshId")) {
            m_mesh.meshId = j["meshId"].get<uint64_t>();
        }

        // Mesh asset GUID (model-file meshes managed by AssetRegistry)
        if (j.contains("meshAssetGuid") && j["meshAssetGuid"].is_string()) {
            std::string meshGuid = j["meshAssetGuid"].get<std::string>();
            if (!meshGuid.empty()) {
                SetMeshAssetGuid(meshGuid);
                AssetRegistry::Instance().Resolve(m_meshAsset, ResourceType::Mesh);
                auto mesh = m_meshAsset.Get();
                if (mesh)
                    SetLocalBounds(mesh->GetBoundsMin(), mesh->GetBoundsMax());
            }
        }

        // ================================================================
        // Materials
        // v3: GUID-or-null entries
        // v4: also supports object entries with source_path / material_json
        // ================================================================
        auto &graph = AssetDependencyGraph::Instance();
        for (auto &ref : m_materials) {
            auto mat = ref.Get();
            if (mat && !mat->GetGuid().empty())
                graph.RemoveDependency(GetInstanceGuid(), mat->GetGuid());
        }
        if (j.contains("materials") && j["materials"].is_array()) {
            const auto &matsJson = j["materials"];
            m_materials.assign(matsJson.size(), AssetRef<InxMaterial>{});
            for (size_t i = 0; i < matsJson.size(); ++i) {
                const auto &slotJson = matsJson[i];
                if (slotJson.is_null())
                    continue;

                if (slotJson.is_string()) {
                    std::string guid = slotJson.get<std::string>();
                    if (!guid.empty()) {
                        m_materials[i].SetGuid(guid);
                        AssetRegistry::Instance().Resolve(m_materials[i], ResourceType::Material);
                        auto mat = m_materials[i].Get();
                        if (mat && !mat->GetGuid().empty())
                            graph.AddDependency(GetInstanceGuid(), mat->GetGuid());
                    }
                    continue;
                }

                if (!slotJson.is_object())
                    continue;

                const std::string guid = slotJson.value("guid", std::string());
                if (!guid.empty()) {
                    m_materials[i].SetGuid(guid);
                    AssetRegistry::Instance().Resolve(m_materials[i], ResourceType::Material);
                    auto mat = m_materials[i].Get();
                    if (mat && !mat->GetGuid().empty())
                        graph.AddDependency(GetInstanceGuid(), mat->GetGuid());
                    continue;
                }

                const std::string sourcePath = slotJson.value("source_path", std::string());
                const std::string materialJson = slotJson.value("material_json", std::string());

                std::shared_ptr<InxMaterial> restoredMat;
                if (!materialJson.empty()) {
                    restoredMat = std::make_shared<InxMaterial>();
                    if (!restoredMat->Deserialize(materialJson))
                        restoredMat.reset();
                }
                if (!restoredMat && !sourcePath.empty())
                    restoredMat = BuildEmbeddedMaterialFromSource(sourcePath);

                if (restoredMat) {
                    if (!sourcePath.empty())
                        restoredMat->SetFilePath(sourcePath);
                    m_materials[i] = AssetRef<InxMaterial>(std::string(), std::move(restoredMat));
                }
            }
        } else {
            m_materials.clear();
        }

        // Read nodeGroup BEFORE SyncMaterialSlotsToMesh so slot counting is nodeGroup-aware
        if (j.contains("nodeGroup")) {
            m_nodeGroup = j["nodeGroup"].get<int32_t>();
        }

        // Sync slot count to mesh submesh count
        SyncMaterialSlotsToMesh();

        // Rendering flags
        if (j.contains("castShadows")) {
            m_castShadows = j["castShadows"].get<bool>();
        }
        if (j.contains("receivesShadows")) {
            m_receiveShadows = j["receivesShadows"].get<bool>();
        }
        if (j.contains("submeshIndex")) {
            m_submeshIndex = j["submeshIndex"].get<int32_t>();
        }
        if (j.contains("meshPivotOffset") && j["meshPivotOffset"].is_array() && j["meshPivotOffset"].size() == 3) {
            m_meshPivotOffset.x = j["meshPivotOffset"][0].get<float>();
            m_meshPivotOffset.y = j["meshPivotOffset"][1].get<float>();
            m_meshPivotOffset.z = j["meshPivotOffset"][2].get<float>();
        }

        // Bounds
        if (j.contains("boundsMin") && j["boundsMin"].is_array() && j["boundsMin"].size() == 3) {
            m_localBoundsMin.x = j["boundsMin"][0].get<float>();
            m_localBoundsMin.y = j["boundsMin"][1].get<float>();
            m_localBoundsMin.z = j["boundsMin"][2].get<float>();
        }
        if (j.contains("boundsMax") && j["boundsMax"].is_array() && j["boundsMax"].size() == 3) {
            m_localBoundsMax.x = j["boundsMax"][0].get<float>();
            m_localBoundsMax.y = j["boundsMax"][1].get<float>();
            m_localBoundsMax.z = j["boundsMax"][2].get<float>();
        }

        // Inline mesh data (for primitives like cubes)
        m_useInlineMesh = j.value("useInlineMesh", false);
        m_inlineMeshName = j.value("inlineMeshName", std::string());
        m_inlineVertices.clear();
        m_inlineIndices.clear();
        m_sharedVertices = nullptr;
        m_sharedIndices = nullptr;

        if (m_useInlineMesh) {
            const bool isBuiltinPrimitive = j.value("inlineMeshBuiltin", false);
            if (isBuiltinPrimitive) {
                RestoreBuiltinPrimitiveMesh(m_inlineMeshName, m_inlineVertices, m_inlineIndices);
            } else if (j.contains("inlineVertices") && j["inlineVertices"].is_array()) {
                for (const auto &vj : j["inlineVertices"]) {
                    Vertex v;
                    if (vj.contains("pos") && vj["pos"].is_array() && vj["pos"].size() == 3) {
                        v.pos.x = vj["pos"][0].get<float>();
                        v.pos.y = vj["pos"][1].get<float>();
                        v.pos.z = vj["pos"][2].get<float>();
                    }
                    if (vj.contains("normal") && vj["normal"].is_array() && vj["normal"].size() == 3) {
                        v.normal.x = vj["normal"][0].get<float>();
                        v.normal.y = vj["normal"][1].get<float>();
                        v.normal.z = vj["normal"][2].get<float>();
                    }
                    if (vj.contains("tangent") && vj["tangent"].is_array() && vj["tangent"].size() == 4) {
                        v.tangent.x = vj["tangent"][0].get<float>();
                        v.tangent.y = vj["tangent"][1].get<float>();
                        v.tangent.z = vj["tangent"][2].get<float>();
                        v.tangent.w = vj["tangent"][3].get<float>();
                    }
                    if (vj.contains("color") && vj["color"].is_array() && vj["color"].size() == 3) {
                        v.color.x = vj["color"][0].get<float>();
                        v.color.y = vj["color"][1].get<float>();
                        v.color.z = vj["color"][2].get<float>();
                    }
                    if (vj.contains("texCoord") && vj["texCoord"].is_array() && vj["texCoord"].size() == 2) {
                        v.texCoord.x = vj["texCoord"][0].get<float>();
                        v.texCoord.y = vj["texCoord"][1].get<float>();
                    }
                    m_inlineVertices.push_back(v);
                }
            }
            if (j.contains("inlineIndices") && j["inlineIndices"].is_array()) {
                for (const auto &idx : j["inlineIndices"]) {
                    m_inlineIndices.push_back(idx.get<uint32_t>());
                }
            }

            if (!m_inlineVertices.empty()) {
                ComputeLocalBoundsFromInlineVertices();
            }
        }

        return true;
    } catch (const std::exception &e) {
        INXLOG_ERROR("MeshRenderer::Deserialize failed: ", e.what());
        return false;
    }
}

std::unique_ptr<Component> MeshRenderer::Clone() const
{
    auto clone = std::make_unique<MeshRenderer>();
    // Base fields
    clone->m_enabled = m_enabled;
    clone->m_executionOrder = m_executionOrder;
    // Mesh ref
    clone->m_mesh = m_mesh;
    clone->m_meshAsset = m_meshAsset;
    clone->m_meshBufferDirty = false;
    // Inline mesh
    clone->m_useInlineMesh = m_useInlineMesh;
    clone->m_inlineMeshName = m_inlineMeshName;
    clone->m_sharedVertices = m_sharedVertices;
    clone->m_sharedIndices = m_sharedIndices;
    if (!m_sharedVertices) {
        clone->m_inlineVertices = m_inlineVertices;
        clone->m_inlineIndices = m_inlineIndices;
    }
    // Materials
    clone->m_materials = m_materials;
    // Rendering flags
    clone->m_castShadows = m_castShadows;
    clone->m_receiveShadows = m_receiveShadows;
    // Submesh / node group
    clone->m_submeshIndex = m_submeshIndex;
    clone->m_nodeGroup = m_nodeGroup;
    clone->m_meshPivotOffset = m_meshPivotOffset;
    // Bounds
    clone->m_localBoundsMin = m_localBoundsMin;
    clone->m_localBoundsMax = m_localBoundsMax;
    return clone;
}

} // namespace infernux
