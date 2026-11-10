#include "MeshPreviewPreparation.h"
#include <algorithm>
#include <cctype>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/scene/PrimitiveMeshes.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <platform/filesystem/InxPath.h>
#include <stdexcept>

namespace infernux
{
bool RequirePreviewAsset(std::shared_ptr<AssetLoadTicket> &ticket, const std::string &guid, ResourceType type,
                         bool &progressed)
{
    auto &registry = AssetRegistry::Instance();
    if (ticket &&
        (!registry.IsAssetLoadContentCurrent(ticket) || (ticket->IsCommitted() && !registry.IsLoaded(guid)))) {
        ticket.reset();
        progressed = true;
    }
    if (!ticket) {
        ticket = registry.BeginLoadAsset(guid, type);
        progressed = true;
    }
    try {
        const bool committed = ticket->IsCommitted();
        const bool ready = registry.TryCommitAssetLoad(ticket, true);
        progressed |= ready && !committed;
        return ready;
    } catch (const AssetLoadSuperseded &) {
        ticket.reset();
        progressed = true;
        return false;
    }
}

bool PreviewAssetLoads::Require(const std::string &guid, ResourceType type, bool &progressed)
{
    const auto *database = AssetRegistry::Instance().GetAssetDatabase();
    // Missing durable references retain the normal preview fallback. A path
    // hint must never resurrect a deleted GUID or introduce a synchronous load.
    if (guid.empty() || !database || !database->GetMetaByGuid(guid))
        return true;
    return RequirePreviewAsset(m_tickets[guid], guid, type, progressed);
}

bool PreviewAssetLoads::IsCurrent() const
{
    auto &registry = AssetRegistry::Instance();
    return std::all_of(m_tickets.begin(), m_tickets.end(),
                       [&](const auto &entry) { return registry.IsAssetLoadContentCurrent(entry.second); });
}

std::shared_ptr<InxMaterial> GetModelPreviewMaterial(const InxMesh *mesh, uint32_t slot)
{
    auto &registry = AssetRegistry::Instance();
    if (!mesh || slot >= mesh->GetMaterialSlotData().size())
        return registry.GetBuiltinMaterial("DefaultLit");
    const auto &guid = mesh->GetMaterialSlotData()[slot].materialGuid;
    if (guid.empty()) {
        if (auto material = mesh->CreateMaterialCopy(slot))
            return material;
        return registry.GetBuiltinMaterial("DefaultLit");
    }
    auto material = registry.GetAsset<InxMaterial>(guid);
    return material && !material->IsDeleted() ? material : registry.GetBuiltinMaterial("ErrorMaterial");
}

bool TryPrepareModelPreviewMaterials(const InxMesh &mesh, const std::vector<SubMesh> &subMeshes,
                                     PreviewAssetLoads &loads, std::vector<std::shared_ptr<InxMaterial>> &materials,
                                     bool &progressed)
{
    uint32_t slots = 0;
    for (const auto &submesh : subMeshes) {
        if (!submesh.indexCount)
            continue;
        if (submesh.materialSlot == (std::numeric_limits<uint32_t>::max)())
            throw std::length_error("Mesh preview material slot exceeds the supported range");
        slots = std::max(slots, submesh.materialSlot + 1);
    }
    std::vector<bool> used(slots, false);
    for (const auto &submesh : subMeshes)
        if (submesh.indexCount)
            used[submesh.materialSlot] = true;
    bool ready = true;
    for (uint32_t slot = 0; slot < slots && slot < mesh.GetMaterialSlotData().size(); ++slot)
        if (used[slot])
            ready = loads.Require(mesh.GetMaterialSlotData()[slot].materialGuid, ResourceType::Material, progressed) &&
                    ready;
    if (!ready)
        return false;
    materials.assign(slots, nullptr);
    for (uint32_t slot = 0; slot < slots; ++slot)
        if (used[slot])
            materials[slot] = GetModelPreviewMaterial(&mesh, slot);
    return true;
}

namespace
{
using json = nlohmann::json;

glm::vec3 ReadVec3(const json &document, const char *key, const glm::vec3 &fallback)
{
    const auto value = document.find(key);
    return value == document.end() || !value->is_array() || value->size() != 3
               ? fallback
               : glm::vec3((*value)[0].get<float>(), (*value)[1].get<float>(), (*value)[2].get<float>());
}

glm::mat4 NodeMatrix(const json &node)
{
    const auto value = node.find("transform");
    if (value == node.end() || !value->is_object())
        return glm::mat4(1.0f);
    const auto rotation = glm::radians(ReadVec3(*value, "rotation", glm::vec3(0.0f)));
    const glm::quat q = glm::angleAxis(rotation.y, glm::vec3(0, 1, 0)) *
                        glm::angleAxis(rotation.x, glm::vec3(1, 0, 0)) * glm::angleAxis(rotation.z, glm::vec3(0, 0, 1));
    return glm::translate(glm::mat4(1.0f), ReadVec3(*value, "position", glm::vec3(0.0f))) * glm::mat4_cast(q) *
           glm::scale(glm::mat4(1.0f), ReadVec3(*value, "scale", glm::vec3(1.0f)));
}

std::shared_ptr<const MeshGeometry> InlineGeometry(const json &data)
{
    if (!data.value("useInlineMesh", false))
        return {};
    auto geometry = std::make_shared<MeshGeometry>();
    const auto name = data.value("inlineMeshName", std::string());
    if (data.value("inlineMeshBuiltin", false)) {
        const std::vector<Vertex> *vertices = nullptr;
        const std::vector<uint32_t> *indices = nullptr;
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
        if (vertices) {
            geometry->vertices = *vertices;
            geometry->indices = *indices;
        }
    } else {
        const auto vertices = data.find("inlineVertices");
        if (vertices != data.end() && vertices->is_array()) {
            geometry->vertices.reserve(vertices->size());
            for (const auto &value : *vertices) {
                Vertex vertex{};
                vertex.pos = ReadVec3(value, "pos", glm::vec3(0.0f));
                vertex.normal = ReadVec3(value, "normal", glm::vec3(0, 1, 0));
                vertex.color = ReadVec3(value, "color", glm::vec3(1.0f));
                const auto tangent = value.find("tangent");
                vertex.tangent = tangent != value.end() && tangent->is_array() && tangent->size() == 4
                                     ? glm::vec4((*tangent)[0].get<float>(), (*tangent)[1].get<float>(),
                                                 (*tangent)[2].get<float>(), (*tangent)[3].get<float>())
                                     : glm::vec4(1, 0, 0, 1);
                const auto uv = value.find("texCoord");
                if (uv != value.end() && uv->is_array() && uv->size() == 2)
                    vertex.texCoord = glm::vec2((*uv)[0].get<float>(), (*uv)[1].get<float>());
                geometry->vertices.push_back(vertex);
            }
        }
        const auto indices = data.find("inlineIndices");
        if (indices != data.end() && indices->is_array())
            geometry->indices = indices->get<std::vector<uint32_t>>();
    }
    if (geometry->vertices.empty() || geometry->indices.empty())
        return {};
    geometry->vertexCount = static_cast<uint32_t>(geometry->vertices.size());
    geometry->indexCount = static_cast<uint32_t>(geometry->indices.size());
    SubMesh submesh;
    submesh.indexCount = geometry->indexCount;
    submesh.vertexCount = geometry->vertexCount;
    submesh.name = name;
    geometry->subMeshes.push_back(std::move(submesh));
    return geometry;
}

void ParseNode(const json &node, const glm::mat4 &parentWorld, std::vector<PrefabPreviewPart> &parts)
{
    if (!node.is_object() || !node.value("active", true))
        return;
    const auto world = parentWorld * NodeMatrix(node);
    const auto components = node.find("components");
    if (components != node.end() && components->is_array()) {
        for (const auto &component : *components) {
            if (!component.is_object() || !component.value("enabled", true))
                continue;
            auto type = component.value("type_id", std::string());
            const auto separator = type.find_last_of(".:");
            if (separator != std::string::npos)
                type.erase(0, separator + 1);
            if (type != "MeshRenderer" && type != "SkinnedMeshRenderer" && type != "LineRenderer")
                continue;
            const auto data = component.find("data");
            if (data == component.end() || !data->is_object())
                continue;
            PrefabPreviewPart part;
            part.world = world;
            part.pivot = ReadVec3(*data, "meshPivotOffset", glm::vec3(0.0f));
            part.submesh = data->value("submeshIndex", -1);
            part.nodeGroup = data->value("nodeGroup", -1);
            part.meshGuid = data->value("meshAssetGuid", std::string());
            part.geometry = InlineGeometry(*data);
            const auto materials = data->find("materials");
            if (materials != data->end() && materials->is_array()) {
                for (const auto &material : *materials)
                    part.materials.push_back(material.is_string() ? material.get<std::string>() : std::string());
            }
            parts.push_back(std::move(part));
        }
    }
    const auto children = node.find("children");
    if (children != node.end() && children->is_array())
        for (const auto &child : *children)
            ParseNode(child, world, parts);
}

glm::vec3 NormalizeOrFallback(const glm::vec3 &value, const glm::vec3 &fallback)
{
    if (glm::dot(value, value) > 1e-10f)
        return glm::normalize(value);
    return glm::dot(fallback, fallback) > 1e-10f ? glm::normalize(fallback) : glm::vec3(0, 1, 0);
}
} // namespace

bool IsPrefabPreviewPath(const std::string &path)
{
    auto extension = FromFsPath(ToFsPath(path).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".prefab";
}

std::vector<PrefabPreviewPart> ParsePrefabPreview(std::string_view source)
{
    const auto document = json::parse(source);
    const auto root = document.find("root_object");
    if (root == document.end() || !root->is_object())
        throw std::invalid_argument("Prefab preview requires a root_object");
    std::vector<PrefabPreviewPart> parts;
    ParseNode(*root, glm::mat4(1.0f), parts);
    return parts;
}

PrefabPreviewLayout MeasurePrefabPreviewGeometry(const std::vector<PrefabPreviewPart> &parts)
{
    uint64_t vertices = 0, indices = 0, subMeshes = 0;
    for (const auto &part : parts) {
        if (!part.geometry)
            continue;
        const auto &source = *part.geometry;
        const auto firstSubMesh = subMeshes;
        for (size_t index = 0; index < source.subMeshes.size(); ++index) {
            const auto &submesh = source.subMeshes[index];
            if (part.Includes(index, submesh)) {
                indices += submesh.indexCount;
                ++subMeshes;
            }
        }
        if (subMeshes != firstSubMesh)
            vertices += source.vertices.size();
    }
    constexpr auto maxCount = (std::numeric_limits<uint32_t>::max)();
    if (vertices > maxCount || indices > maxCount || subMeshes > maxCount)
        throw std::length_error("Prefab preview exceeds the mesh index range");
    return {static_cast<uint32_t>(vertices), static_cast<uint32_t>(indices), static_cast<uint32_t>(subMeshes)};
}

uint64_t PrefabPreviewLayout::StorageBytes() const noexcept
{
    return sizeof(InxMesh) + 2 * sizeof(MeshGeometry) + uint64_t(vertexCount) * sizeof(Vertex) +
           uint64_t(indexCount) * sizeof(uint32_t) + uint64_t(subMeshCount) * sizeof(SubMesh);
}

std::shared_ptr<InxMesh> BuildPrefabPreviewGeometry(const std::vector<PrefabPreviewPart> &parts,
                                                    const PrefabPreviewLayout &layout)
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> submeshes;
    vertices.reserve(layout.vertexCount);
    indices.reserve(layout.indexCount);
    submeshes.reserve(layout.subMeshCount);
    for (const auto &part : parts) {
        if (!part.geometry)
            continue;
        const auto &source = *part.geometry;
        const auto base = static_cast<uint32_t>(vertices.size());
        bool appendedVertices = false;
        for (size_t subMeshIndex = 0; subMeshIndex < source.subMeshes.size(); ++subMeshIndex) {
            const auto &value = source.subMeshes[subMeshIndex];
            if (!part.Includes(subMeshIndex, value))
                continue;
            if (!appendedVertices) {
                const glm::mat3 world(part.world);
                const float determinant = glm::determinant(world);
                const auto normal =
                    std::abs(determinant) > 1e-8f ? glm::transpose(glm::inverse(world)) : glm::mat3(1.0f);
                for (auto vertex : source.vertices) {
                    vertex.pos = glm::vec3(part.world * glm::vec4(vertex.pos + part.pivot, 1));
                    vertex.normal = NormalizeOrFallback(normal * vertex.normal, vertex.normal);
                    vertex.tangent =
                        glm::vec4(NormalizeOrFallback(normal * glm::vec3(vertex.tangent), glm::vec3(vertex.tangent)),
                                  vertex.tangent.w * (determinant < 0 ? -1.0f : 1.0f));
                    vertices.push_back(vertex);
                }
                appendedVertices = true;
            }
            SubMesh submesh;
            submesh.indexCount = value.indexCount;
            submesh.indexStart = static_cast<uint32_t>(indices.size());
            submesh.vertexStart = base;
            submesh.vertexCount = static_cast<uint32_t>(source.vertices.size());
            // Material preparation enumerates the same selected submeshes.
            submesh.materialSlot = static_cast<uint32_t>(submeshes.size());
            submesh.nodeGroup = value.nodeGroup;
            submesh.boundsMin = glm::vec3(std::numeric_limits<float>::max());
            submesh.boundsMax = -submesh.boundsMin;
            for (uint32_t offset = 0; offset < value.indexCount; ++offset) {
                const auto index = source.indices.at(static_cast<size_t>(value.indexStart) + offset);
                // Source documents and imported streams share the same index validity.
                const auto &vertex = vertices.at(static_cast<size_t>(base) + index);
                indices.push_back(base + index);
                submesh.boundsMin = glm::min(submesh.boundsMin, vertex.pos);
                submesh.boundsMax = glm::max(submesh.boundsMax, vertex.pos);
            }
            submeshes.push_back(std::move(submesh));
        }
    }
    auto mesh = std::make_shared<InxMesh>();
    mesh->SetData(std::move(vertices), std::move(indices), std::move(submeshes));
    return mesh;
}
} // namespace infernux
