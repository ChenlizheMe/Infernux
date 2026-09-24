#include "MeshRenderer.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "GameObject.h"
#include "MeshCollider.h"
#include "SceneManager.h"
#include "function/renderer/rhi/RhiComputeBuffer.h"
#include <algorithm>
#include <cmath>
#include <core/log/InxLog.h>
#include <cstring>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/MaterialDocumentValidation.h>
#include <function/scene/PrimitiveMeshes.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <type_traits>

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

std::vector<std::string> ResolveModelNodePathBySubresourceId(const std::string &meshGuid,
                                                             const std::string &subresourceId,
                                                             std::vector<std::string> serializedPath)
{
    if (subresourceId.empty())
        return serializedPath;

    auto *assetDb = AssetRegistry::Instance().GetAssetDatabase();
    const auto meta = assetDb ? assetDb->GetMetaByGuid(meshGuid) : nullptr;
    if (!meta || !meta->HasKey("model_meshes"))
        throw std::invalid_argument("Model mesh identity manifest is unavailable");

    const auto manifest = nlohmann::json::parse(meta->GetDataAs<std::string>("model_meshes"));
    if (!manifest.is_array())
        throw std::invalid_argument("Model mesh identity manifest must be an array");

    std::vector<std::string> resolvedPath;
    size_t matches = 0;
    for (const auto &entry : manifest) {
        if (!entry.is_object() || entry.value("subresource_id", std::string{}) != subresourceId)
            continue;
        if (!entry.contains("path") || !entry["path"].is_array())
            throw std::invalid_argument("Model mesh identity has no node path");
        resolvedPath = entry["path"].get<std::vector<std::string>>();
        ++matches;
    }
    if (matches != 1 || resolvedPath.empty())
        throw std::invalid_argument(matches == 0 ? "Model mesh identity no longer exists"
                                                 : "Model mesh identity is ambiguous");
    return resolvedPath;
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
        for (const auto &guid : assetDb->GetAllGuids()) {
            const auto meta = assetDb->GetMetaByGuid(guid);
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
    } else if (name == "Quad") {
        vertices = &PrimitiveMeshes::GetQuadVertices();
        indices = &PrimitiveMeshes::GetQuadIndices();
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

json SerializeRendererParameter(const MaterialProperty &property)
{
    json result = {{"type", static_cast<int>(property.type)}};
    switch (property.type) {
    case MaterialPropertyType::Float:
        result["value"] = std::get<float>(property.value);
        break;
    case MaterialPropertyType::Float2: {
        const auto value = std::get<glm::vec2>(property.value);
        result["value"] = {value.x, value.y};
        break;
    }
    case MaterialPropertyType::Float3: {
        const auto value = std::get<glm::vec3>(property.value);
        result["value"] = {value.x, value.y, value.z};
        break;
    }
    case MaterialPropertyType::Float4:
    case MaterialPropertyType::Color: {
        const auto value = std::get<glm::vec4>(property.value);
        result["value"] = {value.x, value.y, value.z, value.w};
        break;
    }
    case MaterialPropertyType::Int:
        result["value"] = std::get<int>(property.value);
        break;
    case MaterialPropertyType::Mat4: {
        const auto value = std::get<glm::mat4>(property.value);
        result["value"] = json::array();
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                result["value"].push_back(value[column][row]);
        break;
    }
    case MaterialPropertyType::Texture2D:
        result["guid"] = std::get<std::string>(property.value);
        break;
    }
    return result;
}

MaterialProperty DeserializeRendererParameter(const std::string &name, const json &document)
{
    if (!document.is_object() || !document.contains("type") || !document["type"].is_number_integer())
        throw std::invalid_argument("renderer parameter '" + name + "' requires an integer type");
    const int typeValue = document["type"].get<int>();
    if (typeValue < static_cast<int>(MaterialPropertyType::Float) ||
        typeValue > static_cast<int>(MaterialPropertyType::Color))
        throw std::invalid_argument("renderer parameter '" + name + "' has an invalid type");

    MaterialProperty property{name, static_cast<MaterialPropertyType>(typeValue), 0.0f};
    const auto requireFiniteArray = [&](size_t size) -> const json & {
        if (document.size() != 2 || !document.contains("value") || !document["value"].is_array() ||
            document["value"].size() != size)
            throw std::invalid_argument("renderer parameter '" + name + "' has an invalid vector value");
        for (const auto &item : document["value"]) {
            if (!item.is_number() || !std::isfinite(item.get<double>()))
                throw std::invalid_argument("renderer parameter '" + name + "' requires finite numbers");
        }
        return document["value"];
    };

    switch (property.type) {
    case MaterialPropertyType::Float:
        if (document.size() != 2 || !document.contains("value") || !document["value"].is_number() ||
            !std::isfinite(document["value"].get<double>()))
            throw std::invalid_argument("renderer float parameter '" + name + "' requires one finite number");
        property.value = document["value"].get<float>();
        break;
    case MaterialPropertyType::Float2: {
        const auto &value = requireFiniteArray(2);
        property.value = glm::vec2(value[0].get<float>(), value[1].get<float>());
        break;
    }
    case MaterialPropertyType::Float3: {
        const auto &value = requireFiniteArray(3);
        property.value = glm::vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        break;
    }
    case MaterialPropertyType::Float4:
    case MaterialPropertyType::Color: {
        const auto &value = requireFiniteArray(4);
        property.value =
            glm::vec4(value[0].get<float>(), value[1].get<float>(), value[2].get<float>(), value[3].get<float>());
        break;
    }
    case MaterialPropertyType::Int:
        if (document.size() != 2 || !document.contains("value") || !document["value"].is_number_integer())
            throw std::invalid_argument("renderer int parameter '" + name + "' requires one integer");
        property.value = document["value"].get<int>();
        break;
    case MaterialPropertyType::Mat4: {
        const auto &value = requireFiniteArray(16);
        glm::mat4 matrix{};
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                matrix[column][row] = value[column * 4 + row].get<float>();
        property.value = matrix;
        break;
    }
    case MaterialPropertyType::Texture2D:
        if (document.size() != 2 || !document.contains("guid") || !document["guid"].is_string())
            throw std::invalid_argument("renderer texture parameter '" + name + "' requires one GUID");
        property.value = document["guid"].get<std::string>();
        break;
    }
    return property;
}

void NotifyRenderableStateChanged(MeshRenderer *renderer)
{
    if (renderer)
        SceneManager::Instance().NotifyMeshRendererChanged(renderer);
}

void NotifyCollisionGeometryChanged(MeshRenderer *renderer)
{
    NotifyRenderableStateChanged(renderer);
    auto *gameObject = renderer ? renderer->GetGameObject() : nullptr;
    if (!gameObject)
        return;
    for (auto *collider : gameObject->GetComponents<MeshCollider>()) {
        if (collider)
            collider->OnMeshGeometryChanged();
    }
}

} // namespace

void RecalculateMeshNormals(std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
{
    if (indices.size() % 3 != 0)
        throw std::invalid_argument("Normal generation requires complete triangles");
    for (auto &vertex : vertices)
        vertex.normal = glm::vec3(0.0f);
    for (size_t i = 0; i < indices.size(); i += 3) {
        const uint32_t ia = indices[i];
        const uint32_t ib = indices[i + 1];
        const uint32_t ic = indices[i + 2];
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
            throw std::invalid_argument("Triangle index lies outside the vertex stream");
        const glm::vec3 face = glm::cross(vertices[ib].pos - vertices[ia].pos, vertices[ic].pos - vertices[ia].pos);
        vertices[ia].normal += face;
        vertices[ib].normal += face;
        vertices[ic].normal += face;
    }
    for (auto &vertex : vertices) {
        const float length = glm::length(vertex.normal);
        if (length > 1.0e-12f)
            vertex.normal /= length;
    }
}

void RecalculateMeshTangents(std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
{
    if (indices.size() % 3 != 0)
        throw std::invalid_argument("Tangent generation requires complete triangles");
    std::vector<glm::vec3> tangents(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitangents(vertices.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < indices.size(); i += 3) {
        const uint32_t ia = indices[i];
        const uint32_t ib = indices[i + 1];
        const uint32_t ic = indices[i + 2];
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
            throw std::invalid_argument("Triangle index lies outside the vertex stream");
        const glm::vec3 edge1 = vertices[ib].pos - vertices[ia].pos;
        const glm::vec3 edge2 = vertices[ic].pos - vertices[ia].pos;
        const glm::vec2 uv1 = vertices[ib].texCoord - vertices[ia].texCoord;
        const glm::vec2 uv2 = vertices[ic].texCoord - vertices[ia].texCoord;
        const float determinant = uv1.x * uv2.y - uv1.y * uv2.x;
        if (std::abs(determinant) <= 1.0e-12f)
            continue;
        const float inverse = 1.0f / determinant;
        const glm::vec3 tangent = (edge1 * uv2.y - edge2 * uv1.y) * inverse;
        const glm::vec3 bitangent = (edge2 * uv1.x - edge1 * uv2.x) * inverse;
        for (const uint32_t index : {ia, ib, ic}) {
            tangents[index] += tangent;
            bitangents[index] += bitangent;
        }
    }
    for (size_t i = 0; i < vertices.size(); ++i) {
        // Authoring APIs preserve the authored normal values, but tangent
        // construction operates on a normalized frame just like the shader
        // path does.  Projecting against an unnormalised normal leaves a
        // visible lighting seam after procedural mesh updates.
        const glm::vec3 authoredNormal = vertices[i].normal;
        const float normalLength = glm::length(authoredNormal);
        const glm::vec3 normal = normalLength > 1.0e-12f ? authoredNormal / normalLength : glm::vec3(0.0f);
        glm::vec3 tangent = tangents[i] - normal * glm::dot(normal, tangents[i]);
        float length = glm::length(tangent);
        if (length <= 1.0e-12f) {
            tangent = std::abs(normal.x) > std::abs(normal.z) ? glm::vec3(-normal.y, normal.x, 0.0f)
                                                              : glm::vec3(0.0f, -normal.z, normal.y);
            length = glm::length(tangent);
        }
        if (length <= 1.0e-12f) {
            vertices[i].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
            continue;
        }
        tangent /= length;
        const float handedness = glm::dot(glm::cross(normal, tangent), bitangents[i]) < 0.0f ? -1.0f : 1.0f;
        vertices[i].tangent = glm::vec4(tangent, handedness);
    }
}

namespace
{
SemanticTypeDescriptor DescribeMeshRenderer()
{
    SemanticTypeDescriptor type;
    type.typeGuid = "native:infernux.MeshRenderer";
    type.readableId = "infernux.component.mesh_renderer";
    type.owner = "engine:native";
    type.origin = "native";
    type.displayName = "MeshRenderer";
    type.runtimeProfiles = {"editor", "player", "headless"};
    const auto add = [&](const char *name, const char *stored, const char *kind, json initial,
                         bool hidden = false) -> json & {
        type.fields.push_back({std::string("MeshRenderer.") + name,
                               std::string("FieldType.") + kind,
                               false,
                               {{"field_id", name},
                                {"serialized_name", stored},
                                {"serialized", true},
                                {"hidden", hidden},
                                {"nullable", false},
                                {"storage_kind", "native_property"},
                                {"default", std::move(initial)}}});
        return type.fields.back().attributes;
    };
    add("casts_shadows", "castShadows", "BOOL", true)["tooltip"] = "Whether this renderer casts shadows";
    add("receives_shadows", "receivesShadows", "BOOL", true)["tooltip"] = "Whether this renderer receives shadows";
    auto &submesh = add("submesh_index", "submeshIndex", "INT", -1, true);
    submesh["range"] = {-1, std::numeric_limits<int32_t>::max()};
    submesh["setter_owns_document_shape"] = true; // -1 omits the serialized selector.
    add("mesh_pivot_offset", "meshPivotOffset", "VEC3", {0.0, 0.0, 0.0}, true);
    return type;
}

const bool registeredMeshRenderer = ComponentFactory::Register(
    "MeshRenderer", [] { return std::make_unique<MeshRenderer>(); }, MeshRenderer::ValidateSerializedDocument,
    MeshRenderer::GetTypeConstraints(), DescribeMeshRenderer);
} // namespace

MeshRenderer::~MeshRenderer()
{
    // Remove all dependency edges from this instance in the unified graph
    AssetDependencyGraph::Instance().ClearRuntimeDependenciesOf(GetInstanceGuid());
    // Safety net: ensure we're removed from the registry even if
    // OnDisable wasn't called (e.g. direct destruction during scene teardown).
    SceneManager::Instance().UnregisterMeshRenderer(this);
}

void MeshRenderer::OnEnable()
{
    // Only runtime-resident scenes contribute to the global renderer list.
    // Utility scenes stay isolated, while DontDestroyOnLoad renderers can be
    // disabled and re-enabled without disappearing from rendering.
    if (auto *go = GetGameObject())
        if (!SceneManager::Instance().IsRuntimeScene(go->GetScene()))
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
        AssetDependencyGraph::Instance().RemoveRuntimeDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;
    m_vertexBuffer.reset();
    m_inlineVertices = std::move(vertices);
    m_inlineIndices = std::move(indices);
    m_useInlineMesh = true;
    m_meshAsset.Clear();
    m_modelNodePath.clear();
    m_modelSubresourceId.clear();
    ++m_inlineMeshVersion;
    m_meshBufferDirty = true;
    ComputeLocalBoundsFromInlineVertices();
    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::SetProceduralMesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices)
{
    const bool wasSharedPrimitive = HasSharedInlineMesh();
    const bool wasDrawable = m_useInlineMesh && !GetInlineVertices().empty() && !GetInlineIndices().empty();
    if (m_meshAsset.HasGuid())
        AssetDependencyGraph::Instance().RemoveRuntimeDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;
    m_vertexBuffer.reset();
    m_inlineVertices = std::move(vertices);
    m_inlineIndices = std::move(indices);
    m_useInlineMesh = true;
    m_meshAsset.Clear();
    m_modelNodePath.clear();
    m_modelSubresourceId.clear();
    ++m_inlineMeshVersion;
    m_meshBufferDirty = true;
    ComputeLocalBoundsFromInlineVertices();
    const bool isDrawable = !m_inlineVertices.empty() && !m_inlineIndices.empty();
    if (wasDrawable != isDrawable || wasSharedPrimitive)
        SceneManager::Instance().NotifyMeshRendererChanged(this);
    else
        SceneManager::Instance().NotifyMeshRendererGeometryChanged(this);
}

void MeshRenderer::SetSharedPrimitiveMesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                                          const std::string &primitiveName)
{
    if (m_meshAsset.HasGuid())
        AssetDependencyGraph::Instance().RemoveRuntimeDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_vertexBuffer.reset();
    m_sharedVertices = &vertices;
    m_sharedIndices = &indices;
    m_useInlineMesh = true;
    m_inlineMeshName = primitiveName;
    m_meshAsset.Clear();
    m_modelNodePath.clear();
    m_modelSubresourceId.clear();
    ++m_inlineMeshVersion;
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

    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::RecalculateInlineNormals()
{
    if (!HasInlineMesh())
        throw std::logic_error("Normal generation requires an inline mesh");
    if (m_vertexBuffer)
        throw std::logic_error("Resident meshes rebuild normals on the GPU");
    if (m_sharedVertices) {
        m_inlineVertices = *m_sharedVertices;
        m_inlineIndices = *m_sharedIndices;
        m_sharedVertices = nullptr;
        m_sharedIndices = nullptr;
    }
    RecalculateMeshNormals(m_inlineVertices, m_inlineIndices);
    ++m_inlineMeshVersion;
    m_meshBufferDirty = true;
    SceneManager::Instance().NotifyMeshRendererGeometryChanged(this);
}

void MeshRenderer::RecalculateInlineTangents()
{
    if (!HasInlineMesh())
        throw std::logic_error("Tangent generation requires an inline mesh");
    if (m_vertexBuffer)
        throw std::logic_error("Resident meshes rebuild tangents on the GPU");
    if (m_sharedVertices) {
        m_inlineVertices = *m_sharedVertices;
        m_inlineIndices = *m_sharedIndices;
        m_sharedVertices = nullptr;
        m_sharedIndices = nullptr;
    }
    RecalculateMeshTangents(m_inlineVertices, m_inlineIndices);
    ++m_inlineMeshVersion;
    m_meshBufferDirty = true;
    SceneManager::Instance().NotifyMeshRendererGeometryChanged(this);
}

void MeshRenderer::RecalculateInlineBounds()
{
    if (!HasInlineMesh())
        throw std::logic_error("Bounds generation requires an inline mesh");
    if (m_vertexBuffer)
        throw std::logic_error("Resident mesh bounds must be supplied without a GPU readback");
    ComputeLocalBoundsFromInlineVertices();
    SceneManager::Instance().NotifyMeshRendererGeometryChanged(this);
}

void MeshRenderer::SetVertexBuffer(std::shared_ptr<rhi::ComputeBuffer> buffer, const glm::vec3 &boundsMin,
                                   const glm::vec3 &boundsMax, bool worldSpace)
{
    if (!buffer)
        throw std::invalid_argument("Mesh vertex buffer must not be null");
    if (!HasInlineMesh() || GetInlineVertices().empty() || GetInlineIndices().empty())
        throw std::logic_error("Mesh vertex buffer requires an authored inline mesh");
    const auto &desc = buffer->GetDesc();
    const uint64_t requiredBytes = static_cast<uint64_t>(GetInlineVertices().size()) * sizeof(Vertex);
    const uint64_t availableBytes = buffer->GetByteSize();
    if (desc.scalarType != rhi::ComputeScalarType::Float32 || desc.lanes != 1 || availableBytes < requiredBytes ||
        availableBytes % sizeof(Vertex) != 0)
        throw std::invalid_argument(
            "Mesh vertex buffer must be canonical float32 Vertex storage with capacity >= vertex_count");
    if (!std::isfinite(boundsMin.x) || !std::isfinite(boundsMin.y) || !std::isfinite(boundsMin.z) ||
        !std::isfinite(boundsMax.x) || !std::isfinite(boundsMax.y) || !std::isfinite(boundsMax.z) ||
        glm::any(glm::greaterThan(boundsMin, boundsMax)))
        throw std::invalid_argument("Mesh vertex buffer bounds must be finite and ordered");
    m_vertexBuffer = std::move(buffer);
    m_vertexBufferWorldSpace = worldSpace;
    m_vertexBufferWorldBoundsAnchorInverse = worldSpace && m_gameObject && m_gameObject->GetTransform()
                                                 ? glm::inverse(m_gameObject->GetTransform()->GetWorldMatrix())
                                                 : glm::mat4(1.0f);
    SetLocalBounds(boundsMin, boundsMax);
    m_meshBufferDirty = true;
    SceneManager::Instance().NotifyMeshRendererGeometryChanged(this);
}

size_t MeshRenderer::GetVertexBufferCapacity() const noexcept
{
    return m_vertexBuffer ? static_cast<size_t>(m_vertexBuffer->GetByteSize() / sizeof(Vertex)) : 0;
}

void MeshRenderer::ClearVertexBuffer()
{
    if (!m_vertexBuffer)
        return;
    m_vertexBuffer.reset();
    m_vertexBufferWorldSpace = false;
    m_vertexBufferWorldBoundsAnchorInverse = glm::mat4(1.0f);
    ComputeLocalBoundsFromInlineVertices();
    m_meshBufferDirty = true;
    SceneManager::Instance().NotifyMeshRendererGeometryChanged(this);
}

void MeshRenderer::SetMeshAsset(const std::string &guid, std::shared_ptr<InxMesh> mesh)
{
    if (m_meshAsset.GetGuid() != guid) {
        m_modelNodePath.clear();
        m_modelSubresourceId.clear();
    }
    auto &graph = AssetDependencyGraph::Instance();
    if (m_meshAsset.HasGuid() && m_meshAsset.GetGuid() != guid)
        graph.RemoveRuntimeDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    const uint64_t runtimeVersion = guid.empty() ? 0 : AssetRegistry::Instance().GetAssetVersion(guid);
    m_meshAsset = AssetRef<InxMesh>(guid, std::move(mesh), runtimeVersion);
    m_useInlineMesh = false;
    m_meshBufferDirty = true;
    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;
    m_vertexBuffer.reset();

    if (!guid.empty())
        graph.AddRuntimeDependency(GetInstanceGuid(), guid);

    auto m = m_meshAsset.Get();
    ResolveModelNodeBinding();
    if (m)
        UpdateBoundsForMeshSelection(m);

    SyncMaterialSlotsToMesh();
    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::SetMeshAssetGuid(const std::string &guid)
{
    if (m_meshAsset.GetGuid() != guid) {
        m_modelNodePath.clear();
        m_modelSubresourceId.clear();
    }
    auto &graph = AssetDependencyGraph::Instance();
    if (m_meshAsset.HasGuid() && m_meshAsset.GetGuid() != guid)
        graph.RemoveRuntimeDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_meshAsset.SetGuid(guid);
    m_useInlineMesh = false;
    m_meshBufferDirty = true;
    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;
    m_vertexBuffer.reset();

    if (!guid.empty())
        graph.AddRuntimeDependency(GetInstanceGuid(), guid);

    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::ClearMeshAsset()
{
    m_modelNodePath.clear();
    m_modelSubresourceId.clear();
    if (m_meshAsset.HasGuid())
        AssetDependencyGraph::Instance().RemoveRuntimeDependency(GetInstanceGuid(), m_meshAsset.GetGuid());

    m_meshAsset.Clear();
    m_embeddedMaterialVersions.clear();
    m_meshBufferDirty = true;
    m_useInlineMesh = false;
    m_inlineVertices.clear();
    m_inlineIndices.clear();
    m_sharedVertices = nullptr;
    m_sharedIndices = nullptr;
    m_vertexBuffer.reset();
    m_localBoundsMin = glm::vec3(-0.5f);
    m_localBoundsMax = glm::vec3(0.5f);
    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::OnMeshAssetEvent(AssetEvent event)
{
    if (!m_meshAsset.HasGuid())
        return;

    if (event == AssetEvent::Deleted) {
        m_meshAsset.Invalidate();
        m_meshBufferDirty = true;
        m_useInlineMesh = false;
        m_inlineVertices.clear();
        m_inlineIndices.clear();
        m_sharedVertices = nullptr;
        m_sharedIndices = nullptr;
        m_vertexBuffer.reset();
        m_localBoundsMin = glm::vec3(-0.5f);
        m_localBoundsMax = glm::vec3(0.5f);
        NotifyCollisionGeometryChanged(this);
        return;
    }

    AssetRegistry::Instance().Resolve(m_meshAsset, ResourceType::Mesh);
    auto mesh = m_meshAsset.Get();
    if (!mesh)
        return;

    ResolveModelNodeBinding();
    UpdateBoundsForMeshSelection(mesh);
    SyncMaterialSlotsToMesh();
    MarkMeshBufferDirty();
    if (event == AssetEvent::RuntimeModified)
        NotifyRenderableStateChanged(this);
    else
        NotifyCollisionGeometryChanged(this);
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
    EnsureParameterSlot(slot);
    m_embeddedMaterialVersions.resize(m_materials.size());
    m_embeddedMaterialVersions[slot].reset();

    auto &ref = m_materials[slot];
    auto oldMat = ref.Get();
    const std::string oldGuid = ref.GetGuid();
    const std::string guid = material ? material->GetGuid() : "";
    if (oldMat == material && oldGuid == guid)
        return;

    auto &graph = AssetDependencyGraph::Instance();
    if (!oldGuid.empty())
        graph.RemoveRuntimeDependency(GetInstanceGuid(), oldGuid);

    const uint64_t runtimeVersion = guid.empty() ? 0 : AssetRegistry::Instance().GetAssetVersion(guid);
    ref = AssetRef<InxMaterial>(guid, std::move(material), runtimeVersion);

    if (!guid.empty())
        graph.AddRuntimeDependency(GetInstanceGuid(), guid);

    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMaterial(uint32_t slot, const std::string &guid)
{
    if (slot >= m_materials.size())
        m_materials.resize(slot + 1);
    EnsureParameterSlot(slot);
    m_embeddedMaterialVersions.resize(m_materials.size());
    m_embeddedMaterialVersions[slot].reset();

    auto &ref = m_materials[slot];
    const std::string oldGuid = ref.GetGuid();
    auto &graph = AssetDependencyGraph::Instance();
    if (!oldGuid.empty() && oldGuid != guid)
        graph.RemoveRuntimeDependency(GetInstanceGuid(), oldGuid);

    ref.SetGuid(guid);
    AssetRegistry::Instance().Resolve(ref, ResourceType::Material);

    if (!guid.empty())
        graph.AddRuntimeDependency(GetInstanceGuid(), guid);

    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMaterials(const std::vector<std::string> &guids)
{
    // Clear old dependency edges
    auto &graph = AssetDependencyGraph::Instance();
    for (auto &ref : m_materials) {
        if (ref.HasGuid())
            graph.RemoveRuntimeDependency(GetInstanceGuid(), ref.GetGuid());
    }

    m_materials.resize(guids.size());
    m_embeddedMaterialVersions.assign(guids.size(), std::nullopt);
    m_persistentParameters.resize(guids.size());
    m_runtimeParameters.resize(guids.size());
    m_parameterBlocks.resize(guids.size());
    for (uint32_t i = 0; i < guids.size(); ++i) {
        m_materials[i].SetGuid(guids[i]);
        AssetRegistry::Instance().Resolve(m_materials[i], ResourceType::Material);

        if (!guids[i].empty())
            graph.AddRuntimeDependency(GetInstanceGuid(), guids[i]);
    }

    RefreshParameterTextureDependencies();
    NotifyRenderableStateChanged(this);
}

void MeshRenderer::SetMaterialSlotCount(uint32_t count)
{
    if (count == m_materials.size())
        return;

    // Remove dependency edges for slots being removed
    auto &graph = AssetDependencyGraph::Instance();
    for (uint32_t i = count; i < m_materials.size(); ++i) {
        if (m_materials[i].HasGuid())
            graph.RemoveRuntimeDependency(GetInstanceGuid(), m_materials[i].GetGuid());
    }
    m_materials.resize(count);
    m_embeddedMaterialVersions.resize(count);
    m_persistentParameters.resize(count);
    m_runtimeParameters.resize(count);
    m_parameterBlocks.resize(count);
    RefreshParameterTextureDependencies();
    NotifyRenderableStateChanged(this);
}

void MeshRenderer::EnsureParameterSlot(uint32_t slot)
{
    const size_t required = static_cast<size_t>(slot) + 1;
    if (m_persistentParameters.size() < required)
        m_persistentParameters.resize(required);
    if (m_runtimeParameters.size() < required)
        m_runtimeParameters.resize(required);
    if (m_parameterBlocks.size() < required)
        m_parameterBlocks.resize(required);
}

void MeshRenderer::PublishParameterSlot(uint32_t slot)
{
    EnsureParameterSlot(slot);
    auto values = m_persistentParameters[slot];
    std::unordered_map<std::string, const RuntimeParameterEntry *> newest;
    for (const auto &[owner, layer] : m_runtimeParameters[slot]) {
        (void)owner;
        for (const auto &[name, entry] : layer) {
            const auto found = newest.find(name);
            if (found == newest.end() || found->second->writeRevision < entry.writeRevision)
                newest[name] = &entry;
        }
    }
    for (const auto &[name, entry] : newest)
        values[name] = entry->property;

    if (values.empty()) {
        m_parameterBlocks[slot].reset();
    } else {
        ++m_parameterRevision;
        if (m_parameterRevision == 0)
            ++m_parameterRevision;
        auto publication = std::make_shared<RendererParameterBlock>();
        publication->properties = std::move(values);
        publication->revision = m_parameterRevision;
        m_parameterBlocks[slot] = std::move(publication);
    }
    RefreshParameterTextureDependencies();
    SceneManager::Instance().NotifyMeshRendererContentChanged(this);
}

void MeshRenderer::RefreshParameterTextureDependencies()
{
    std::unordered_set<std::string> next;
    const auto collectProperty = [&next](const MaterialProperty &property) {
        if (property.type != MaterialPropertyType::Texture2D)
            return;
        const auto *guid = std::get_if<std::string>(&property.value);
        if (guid && !guid->empty() && *guid != "white" && *guid != "black" && *guid != "normal")
            next.insert(*guid);
    };
    for (const auto &slot : m_persistentParameters) {
        for (const auto &[name, property] : slot) {
            (void)name;
            collectProperty(property);
        }
    }
    for (const auto &owners : m_runtimeParameters) {
        for (const auto &[owner, layer] : owners) {
            (void)owner;
            for (const auto &[name, entry] : layer) {
                (void)name;
                collectProperty(entry.property);
            }
        }
    }

    auto &graph = AssetDependencyGraph::Instance();
    for (const auto &guid : m_parameterTextureDependencies) {
        if (next.count(guid) == 0)
            graph.RemoveRuntimeDependency(GetInstanceGuid(), guid);
    }
    for (const auto &guid : next) {
        if (m_parameterTextureDependencies.count(guid) == 0)
            graph.AddRuntimeDependency(GetInstanceGuid(), guid);
    }
    m_parameterTextureDependencies = std::move(next);
}

void MeshRenderer::SetParameter(uint32_t slot, const std::string &name, MaterialPropertyValue value, bool persistent,
                                const std::string &owner)
{
    if (slot != 0 && slot >= m_materials.size())
        throw std::out_of_range("renderer parameter material slot does not exist");
    auto material = GetEffectiveMaterial(slot);
    if (!material)
        throw std::logic_error("renderer parameter assignment requires an effective material");
    const MaterialProperty *declaration = material->GetProperty(name);
    if (!declaration)
        throw std::invalid_argument("material shader has no parameter named '" + name + "'");

    const bool typeMatches =
        (declaration->type == MaterialPropertyType::Float && std::holds_alternative<float>(value)) ||
        (declaration->type == MaterialPropertyType::Float2 && std::holds_alternative<glm::vec2>(value)) ||
        (declaration->type == MaterialPropertyType::Float3 && std::holds_alternative<glm::vec3>(value)) ||
        ((declaration->type == MaterialPropertyType::Float4 || declaration->type == MaterialPropertyType::Color) &&
         std::holds_alternative<glm::vec4>(value)) ||
        (declaration->type == MaterialPropertyType::Int && std::holds_alternative<int>(value)) ||
        (declaration->type == MaterialPropertyType::Mat4 && std::holds_alternative<glm::mat4>(value)) ||
        (declaration->type == MaterialPropertyType::Texture2D && std::holds_alternative<std::string>(value));
    if (!typeMatches)
        throw std::invalid_argument("renderer parameter '" + name + "' does not match the reflected shader type");

    if (declaration->type == MaterialPropertyType::Texture2D)
        value = InxMaterial::RequireTextureGuid(std::get<std::string>(value));

    EnsureParameterSlot(slot);
    MaterialProperty property{name, declaration->type, std::move(value), declaration->hdr, declaration->range};
    if (persistent) {
        m_persistentParameters[slot][name] = std::move(property);
    } else {
        if (owner.empty())
            throw std::invalid_argument("runtime renderer parameter owner cannot be empty");
        if (m_runtimeParameterWriteRevision == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("runtime renderer parameter write revision overflow");
        m_runtimeParameters[slot][owner][name] =
            RuntimeParameterEntry{std::move(property), ++m_runtimeParameterWriteRevision};
    }
    PublishParameterSlot(slot);
}

const MaterialProperty *MeshRenderer::GetParameter(uint32_t slot, const std::string &name, bool persistentOnly,
                                                   const std::string &owner) const
{
    if (!persistentOnly && slot < m_runtimeParameters.size()) {
        if (!owner.empty()) {
            const auto layer = m_runtimeParameters[slot].find(owner);
            if (layer != m_runtimeParameters[slot].end()) {
                const auto found = layer->second.find(name);
                if (found != layer->second.end())
                    return &found->second.property;
            }
        } else {
            const RuntimeParameterEntry *newest = nullptr;
            for (const auto &[layerOwner, layer] : m_runtimeParameters[slot]) {
                (void)layerOwner;
                const auto found = layer.find(name);
                if (found != layer.end() && (!newest || newest->writeRevision < found->second.writeRevision))
                    newest = &found->second;
            }
            if (newest)
                return &newest->property;
        }
    }
    if ((persistentOnly || owner.empty()) && slot < m_persistentParameters.size()) {
        const auto found = m_persistentParameters[slot].find(name);
        if (found != m_persistentParameters[slot].end())
            return &found->second;
    }
    return nullptr;
}

bool MeshRenderer::RemoveParameter(uint32_t slot, const std::string &name, bool persistent, const std::string &owner)
{
    if (persistent) {
        if (slot >= m_persistentParameters.size() || m_persistentParameters[slot].erase(name) == 0)
            return false;
    } else {
        if (owner.empty())
            throw std::invalid_argument("runtime renderer parameter owner cannot be empty");
        if (slot >= m_runtimeParameters.size())
            return false;
        auto layer = m_runtimeParameters[slot].find(owner);
        if (layer == m_runtimeParameters[slot].end() || layer->second.erase(name) == 0)
            return false;
        if (layer->second.empty())
            m_runtimeParameters[slot].erase(layer);
    }
    PublishParameterSlot(slot);
    return true;
}

void MeshRenderer::ClearParameters(uint32_t slot, bool persistent, const std::string &owner)
{
    if (persistent) {
        if (slot >= m_persistentParameters.size() || m_persistentParameters[slot].empty())
            return;
        m_persistentParameters[slot].clear();
    } else {
        if (owner.empty())
            throw std::invalid_argument("runtime renderer parameter owner cannot be empty");
        if (slot >= m_runtimeParameters.size() || m_runtimeParameters[slot].erase(owner) == 0)
            return;
    }
    PublishParameterSlot(slot);
}

std::shared_ptr<const RendererParameterBlock> MeshRenderer::GetParameterBlock(uint32_t slot) const
{
    return slot < m_parameterBlocks.size() ? m_parameterBlocks[slot] : nullptr;
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
    if (slot < m_materials.size() && m_materials[slot].HasGuid()) {
        auto &registry = AssetRegistry::Instance();
        auto err = registry.GetBuiltinMaterial("ErrorMaterial");
        return err ? err : registry.GetBuiltinMaterial("DefaultLit");
    }
    return AssetRegistry::Instance().GetBuiltinMaterial("DefaultLit");
}

void MeshRenderer::OnMaterialAssetEvent(const std::string &guid, AssetEvent event)
{
    if (guid.empty() || (event != AssetEvent::Deleted && event != AssetEvent::Modified))
        return;
    bool changed = false;
    for (auto &reference : m_materials) {
        if (reference.GetGuid() != guid)
            continue;
        if (event == AssetEvent::Deleted) {
            reference.Invalidate();
        } else {
            reference.MarkStale();
            AssetRegistry::Instance().Resolve(reference, ResourceType::Material);
        }
        changed = true;
    }
    if (changed)
        NotifyRenderableStateChanged(this);
}

void MeshRenderer::OnParameterTextureAssetEvent(const std::string &guid, AssetEvent event)
{
    if (guid.empty() || (event != AssetEvent::Deleted && event != AssetEvent::Modified))
        return;
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(m_parameterBlocks.size()); ++slot) {
        const auto &block = m_parameterBlocks[slot];
        if (!block)
            continue;
        const bool usesTexture =
            std::any_of(block->properties.begin(), block->properties.end(), [&guid](const auto &entry) {
                const auto &property = entry.second;
                if (property.type != MaterialPropertyType::Texture2D)
                    return false;
                const auto *value = std::get_if<std::string>(&property.value);
                return value && *value == guid;
            });
        if (usesTexture)
            PublishParameterSlot(slot);
    }
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
    if (IsModelNodeLocal() && m_nodeGroup < 0)
        return;
    if (!HasMeshAsset())
        return;
    auto mesh = m_meshAsset.Get();
    if (!mesh)
        return;

    // Single-submesh mode: only 1 material slot needed
    if (m_submeshIndex >= 0) {
        SetMaterialSlotCount(1);
        ApplyEmbeddedMaterialsFromMesh(mesh);
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
        ApplyEmbeddedMaterialsFromMesh(mesh);
        return;
    }

    uint32_t needed = std::max(mesh->GetMaterialSlotCount(), 1u);
    if (m_materials.size() != needed)
        SetMaterialSlotCount(needed);
    ApplyEmbeddedMaterialsFromMesh(mesh);
}

bool MeshRenderer::IsUnmodifiedEmbeddedMaterial(size_t slot) const
{
    if (slot >= m_embeddedMaterialVersions.size() || !m_embeddedMaterialVersions[slot])
        return false;
    const auto &source = *m_embeddedMaterialVersions[slot];
    if (!source.guid.empty())
        return m_materials[slot].GetGuid() == source.guid;
    if (m_materials[slot].HasGuid())
        return false;
    const auto material = m_materials[slot].Get();
    return material && material->GetAuthoredVersion() == source.authoredVersion;
}

void MeshRenderer::ApplyEmbeddedMaterialsFromMesh(const std::shared_ptr<InxMesh> &mesh)
{
    if (!mesh || m_materials.empty())
        return;
    m_embeddedMaterialVersions.resize(m_materials.size());
    const auto &slotData = mesh->GetMaterialSlotData();
    if (slotData.empty()) {
        for (size_t slot = 0; slot < m_materials.size(); ++slot) {
            if (IsUnmodifiedEmbeddedMaterial(slot))
                m_materials[slot].Clear();
        }
        m_embeddedMaterialVersions.assign(m_materials.size(), std::nullopt);
        return;
    }
    std::vector<uint32_t> sourceSlots;
    if (m_submeshIndex >= 0) {
        const auto &subMeshes = mesh->GetSubMeshes();
        if (static_cast<size_t>(m_submeshIndex) < subMeshes.size())
            sourceSlots.push_back(subMeshes[static_cast<size_t>(m_submeshIndex)].materialSlot);
    } else if (m_nodeGroup >= 0) {
        std::set<uint32_t> uniqueSlots;
        for (const auto &subMesh : mesh->GetSubMeshes()) {
            if (static_cast<int32_t>(subMesh.nodeGroup) == m_nodeGroup &&
                uniqueSlots.insert(subMesh.materialSlot).second)
                sourceSlots.push_back(subMesh.materialSlot);
        }
    } else {
        sourceSlots.resize(m_materials.size());
        for (uint32_t slot = 0; slot < sourceSlots.size(); ++slot)
            sourceSlots[slot] = slot;
    }
    if (sourceSlots.empty())
        sourceSlots.push_back(0);

    const size_t count = std::min(m_materials.size(), sourceSlots.size());
    for (size_t rendererSlot = 0; rendererSlot < count; ++rendererSlot) {
        auto &reference = m_materials[rendererSlot];
        const bool importedDefault = IsUnmodifiedEmbeddedMaterial(rendererSlot);
        if (!importedDefault && (reference.HasGuid() || reference.Get())) {
            m_embeddedMaterialVersions[rendererSlot].reset();
            continue;
        }
        const uint32_t sourceSlot = sourceSlots[rendererSlot];
        if (sourceSlot >= slotData.size()) {
            reference.Clear();
            m_embeddedMaterialVersions[rendererSlot].reset();
            continue;
        }
        const MaterialSlotData &data = slotData[sourceSlot];
        if (!data.materialGuid.empty()) {
            if (reference.GetGuid() != data.materialGuid)
                SetMaterial(static_cast<uint32_t>(rendererSlot), data.materialGuid);
            m_embeddedMaterialVersions[rendererSlot] = ImportedMaterialState{data.materialGuid, 0};
            continue;
        }
        // Geometry-only publications retain materials and their pipelines;
        // source surface changes must still publish even when colors match.
        if (importedDefault && !reference.HasGuid() && mesh->MatchesMaterialCopy(sourceSlot, *reference.Get()))
            continue;
        SetMaterial(static_cast<uint32_t>(rendererSlot), mesh->CreateMaterialCopy(sourceSlot));
        m_embeddedMaterialVersions[rendererSlot] = ImportedMaterialState{{}, reference.Get()->GetAuthoredVersion()};
    }
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
    if (m_nodeGroup == group)
        return;
    m_nodeGroup = group;
    if (HasMeshAsset()) {
        auto mesh = m_meshAsset.Get();
        if (mesh) {
            UpdateBoundsForMeshSelection(mesh);
            SyncMaterialSlotsToMesh();
        }
    }
    NotifyCollisionGeometryChanged(this);
}

std::shared_ptr<const MeshGeometry> MeshRenderer::GetAssetGeometry() const
{
    const auto mesh = m_meshAsset.Get();
    if (!mesh)
        return {};
    if (IsModelNodeLocal())
        return m_nodeGroup >= 0 ? mesh->GetModelSourceGeometry() : nullptr;
    return mesh->GetGeometrySnapshot();
}

void MeshRenderer::SetModelNodePath(std::vector<std::string> path)
{
    if (m_modelNodePath == path)
        return;
    const auto mesh = m_meshAsset.Get();
    if (!path.empty() && (!mesh || !mesh->GetModelSourceGeometry()))
        throw std::invalid_argument("Node-local rendering requires imported source geometry");
    m_modelNodePath = std::move(path);
    ResolveModelNodeBinding();
    if (mesh)
        UpdateBoundsForMeshSelection(mesh);
    SyncMaterialSlotsToMesh();
    MarkMeshBufferDirty();
    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::ResolveModelNodeBinding()
{
    if (!IsModelNodeLocal())
        return;
    m_nodeGroup = -1;
    const auto mesh = m_meshAsset.Get();
    if (!mesh)
        return;
    const auto &nodes = mesh->GetModelNodes();
    bool matched = false;
    for (size_t index = 0; index < nodes.size(); ++index) {
        if (nodes[index].nodeGroup < 0)
            continue;
        int32_t node = static_cast<int32_t>(index);
        size_t part = m_modelNodePath.size();
        while (node >= 0 && part > 0 && nodes[node].name == m_modelNodePath[part - 1]) {
            node = nodes[node].parentIndex;
            --part;
        }
        if (node >= 0 || part != 0)
            continue;
        if (matched) {
            m_nodeGroup = -1;
            break;
        }
        matched = true;
        m_nodeGroup = nodes[index].nodeGroup;
    }
    // A source reimport can temporarily invalidate a path while the editor's
    // owner-thread model reconciliation removes/creates the corresponding
    // instance node.  Keep the renderer unbound for that safe-point instead of
    // reporting a fatal asset error; a genuinely unresolved path remains
    // visible through the renderer's empty geometry state.
}

void MeshRenderer::SetSubmeshIndex(int32_t index)
{
    if (index < -1)
        throw std::invalid_argument("MeshRenderer submesh index must be -1 or non-negative");
    if (m_submeshIndex == index)
        return;
    m_submeshIndex = index;
    if (auto mesh = m_meshAsset.Get()) {
        UpdateBoundsForMeshSelection(mesh);
        SyncMaterialSlotsToMesh();
    }
    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::SetMeshPivotOffset(const glm::vec3 &offset)
{
    if (!std::isfinite(offset.x) || !std::isfinite(offset.y) || !std::isfinite(offset.z))
        throw std::invalid_argument("MeshRenderer mesh pivot must be finite");
    if (m_meshPivotOffset == offset)
        return;
    m_meshPivotOffset = offset;
    if (auto mesh = m_meshAsset.Get())
        UpdateBoundsForMeshSelection(mesh);
    NotifyCollisionGeometryChanged(this);
}

void MeshRenderer::UpdateBoundsForMeshSelection(const std::shared_ptr<InxMesh> &mesh)
{
    const auto geometry = GetAssetGeometry();
    if (!geometry)
        return;
    if (m_submeshIndex >= 0 && static_cast<size_t>(m_submeshIndex) < geometry->subMeshes.size()) {
        const auto &sub = geometry->subMeshes[static_cast<uint32_t>(m_submeshIndex)];
        SetLocalBounds(sub.boundsMin + m_meshPivotOffset, sub.boundsMax + m_meshPivotOffset);
        return;
    }
    if (m_nodeGroup < 0) {
        SetLocalBounds(mesh->GetBoundsMin(), mesh->GetBoundsMax());
        return;
    }
    constexpr float INF = std::numeric_limits<float>::max();
    glm::vec3 bmin(INF);
    glm::vec3 bmax(-INF);
    bool found = false;
    for (const auto &sub : geometry->subMeshes) {
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
    ComputeWorldBounds(ResolveBoundsWorldMatrix(transform->GetWorldMatrix()), outMin, outMax);
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

nlohmann::json MeshRenderer::SerializeDocument() const
{
    json j = Component::SerializeDocument();

    // Mesh reference
    j["meshId"] = m_mesh.meshId;

    const bool persistInlineMesh = ShouldSerializeInlineMeshData();
    const bool builtinPrimitive =
        persistInlineMesh && m_useInlineMesh && !m_inlineMeshName.empty() &&
        MatchesBuiltinPrimitiveMesh(m_inlineMeshName, GetInlineVertices(), GetInlineIndices());
    const std::string matchedInlineMeshGuid =
        (persistInlineMesh && !HasMeshAsset() && m_useInlineMesh && !builtinPrimitive)
            ? FindMatchingMeshAssetGuid(GetInlineVertices(), GetInlineIndices(), m_inlineMeshName)
            : std::string();
    const std::string serializedMeshGuid = HasMeshAsset() ? m_meshAsset.GetGuid() : matchedInlineMeshGuid;

    // Mesh asset GUID (for model-file meshes managed by AssetRegistry)
    if (!serializedMeshGuid.empty()) {
        j["meshAssetGuid"] = serializedMeshGuid;
    }

    // Materials:
    // GUID-backed slots remain compact strings. Runtime material snapshots are
    // embedded as typed documents instead of nested JSON text.
    json materialsJson = json::array();
    for (size_t slot = 0; slot < m_materials.size(); ++slot) {
        const auto &ref = m_materials[slot];
        if (IsUnmodifiedEmbeddedMaterial(slot)) {
            materialsJson.push_back(nullptr);
            continue;
        }
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
        slotJson["material"] = mat->SerializeDocument();

        materialsJson.push_back(slotJson.empty() ? json(nullptr) : slotJson);
    }
    j["materials"] = materialsJson;

    bool hasPersistentParameters = false;
    for (const auto &slot : m_persistentParameters)
        hasPersistentParameters = hasPersistentParameters || !slot.empty();
    if (hasPersistentParameters) {
        json parameterSlots = json::array();
        for (const auto &slot : m_persistentParameters) {
            json values = json::object();
            for (const auto &[name, property] : slot)
                values[name] = SerializeRendererParameter(property);
            parameterSlots.push_back(std::move(values));
        }
        j["parameterOverrides"] = std::move(parameterSlots);
    }

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
    if (IsModelNodeLocal())
        j["modelNodePath"] = m_modelNodePath;
    if (!m_modelSubresourceId.empty())
        j["modelSubresourceId"] = m_modelSubresourceId;

    // Mesh pivot offset (for submesh centering)
    if (m_meshPivotOffset != glm::vec3(0.0f)) {
        j["meshPivotOffset"] = {m_meshPivotOffset.x, m_meshPivotOffset.y, m_meshPivotOffset.z};
    }

    // Bounds
    j["boundsMin"] = {m_localBoundsMin.x, m_localBoundsMin.y, m_localBoundsMin.z};
    j["boundsMax"] = {m_localBoundsMax.x, m_localBoundsMax.y, m_localBoundsMax.z};

    // Inline mesh data (for primitives and procedural geometry)
    const bool serializeInlineMesh = persistInlineMesh && m_useInlineMesh && serializedMeshGuid.empty();
    j["useInlineMesh"] = serializeInlineMesh;
    if (persistInlineMesh && !m_inlineMeshName.empty()) {
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

    return j;
}

void MeshRenderer::ValidateSerializedDocument(const nlohmann::json &document)
{
    ValidateSerializedDocumentForType(document, "MeshRenderer");
}

void MeshRenderer::ValidateSerializedDocumentForType(const nlohmann::json &j, std::string_view expectedType)
{
    using namespace component_document_validation;
    std::vector<std::string_view> required = {"meshId",    "materials", "castShadows",  "receivesShadows",
                                              "boundsMin", "boundsMax", "useInlineMesh"};
    std::vector<std::string_view> optional = {"meshAssetGuid",   "submeshIndex",      "nodeGroup",
                                              "meshPivotOffset", "inlineMeshName",    "inlineMeshBuiltin",
                                              "inlineVertices",  "inlineIndices",     "parameterOverrides",
                                              "modelNodePath",   "modelSubresourceId"};
    if (expectedType == "SpriteRenderer") {
        required.insert(required.end(), {"frameId", "spriteColor", "flipX", "flipY"});
        optional.push_back("spriteGuid");
    } else if (expectedType == "LineRenderer") {
        required.insert(required.end(), {"positions", "widthCurve", "widthMultiplier", "colorGradient", "loop",
                                         "useWorldSpace", "alignment", "textureMode", "textureScale",
                                         "numCornerVertices", "numCapVertices", "shadowBias", "generateLightingData"});
    } else if (expectedType == "SkinnedMeshRenderer") {
        optional.push_back("activeTakeName");
    } else if (expectedType != "MeshRenderer") {
        throw std::invalid_argument("unsupported MeshRenderer document type: " + std::string(expectedType));
    }
    ValidateComponentDocumentFields(j, expectedType, required, optional);

    RequireUnsignedInteger(j, "meshId", expectedType);
    const auto &materials = j["materials"];
    if (!materials.is_array())
        throw std::invalid_argument(std::string(expectedType) + ".materials must be an array");
    for (size_t index = 0; index < materials.size(); ++index) {
        const auto &slot = materials[index];
        if (slot.is_null())
            continue;
        if (slot.is_string()) {
            if (slot.get_ref<const std::string &>().empty())
                throw std::invalid_argument(std::string(expectedType) + ".materials[" + std::to_string(index) +
                                            "] must use null instead of an empty GUID");
            continue;
        }
        if (!slot.is_object())
            throw std::invalid_argument(std::string(expectedType) + ".materials[" + std::to_string(index) +
                                        "] must be null, a GUID string, or an object");
        if (slot.size() != 1 || !slot.contains("material") || !slot["material"].is_object())
            throw std::invalid_argument(std::string(expectedType) + ".materials[" + std::to_string(index) +
                                        "] must contain only a material object");
        material_document_validation::ValidateMaterialDocument(
            slot["material"], std::string(expectedType) + ".materials[" + std::to_string(index) + "].material");
    }
    if (j.contains("parameterOverrides")) {
        const auto &slots = j["parameterOverrides"];
        if (!slots.is_array())
            throw std::invalid_argument(std::string(expectedType) + ".parameterOverrides must be an array");
        for (size_t slotIndex = 0; slotIndex < slots.size(); ++slotIndex) {
            if (!slots[slotIndex].is_object())
                throw std::invalid_argument(std::string(expectedType) + ".parameterOverrides slots must be objects");
            for (const auto &[name, value] : slots[slotIndex].items()) {
                if (name.empty())
                    throw std::invalid_argument(std::string(expectedType) + " parameter names must not be empty");
                (void)DeserializeRendererParameter(name, value);
            }
        }
    }

    RequireBoolean(j, "castShadows", expectedType);
    RequireBoolean(j, "receivesShadows", expectedType);
    RequireFiniteVector(j, "boundsMin", 3, expectedType);
    RequireFiniteVector(j, "boundsMax", 3, expectedType);
    RequireBoolean(j, "useInlineMesh", expectedType);
    for (size_t axis = 0; axis < 3; ++axis) {
        if (j["boundsMin"][axis].get<float>() > j["boundsMax"][axis].get<float>())
            throw std::invalid_argument(std::string(expectedType) + " bounds are inverted");
    }

    if (j.contains("meshAssetGuid") && RequireString(j, "meshAssetGuid", expectedType).empty())
        throw std::invalid_argument(std::string(expectedType) + ".meshAssetGuid must not be empty");
    if (j.contains("submeshIndex") && RequireInteger(j, "submeshIndex", expectedType) < 0)
        throw std::invalid_argument(std::string(expectedType) + ".submeshIndex must be non-negative");
    if (j.contains("nodeGroup") && RequireInteger(j, "nodeGroup", expectedType) < 0)
        throw std::invalid_argument(std::string(expectedType) + ".nodeGroup must be non-negative");
    if (j.contains("modelNodePath")) {
        const auto &path = j["modelNodePath"];
        if (!path.is_array() || path.empty() ||
            std::any_of(path.begin(), path.end(), [](const auto &item) { return !item.is_string(); }))
            throw std::invalid_argument("Model node path must be a non-empty string array");
        if (expectedType != "MeshRenderer" || !j.contains("meshAssetGuid") ||
            j["meshAssetGuid"].get<std::string>().empty())
            throw std::invalid_argument("Node-local geometry requires a static model node binding");
    }
    if (j.contains("modelSubresourceId") && RequireString(j, "modelSubresourceId", expectedType).empty())
        throw std::invalid_argument(std::string(expectedType) + ".modelSubresourceId must not be empty");
    if (j.contains("meshPivotOffset"))
        RequireFiniteVector(j, "meshPivotOffset", 3, expectedType);
    if (j.contains("inlineMeshName"))
        RequireString(j, "inlineMeshName", expectedType);
    if (j.contains("inlineMeshBuiltin"))
        RequireBoolean(j, "inlineMeshBuiltin", expectedType);

    const bool useInlineMesh = j["useInlineMesh"].get<bool>();
    const bool inlineBuiltin = j.value("inlineMeshBuiltin", false);
    if (useInlineMesh && j.contains("meshAssetGuid"))
        throw std::invalid_argument(std::string(expectedType) + " cannot combine inline mesh and meshAssetGuid");
    if (!useInlineMesh &&
        (j.contains("inlineMeshBuiltin") || j.contains("inlineVertices") || j.contains("inlineIndices")))
        throw std::invalid_argument(std::string(expectedType) + " has inline data while useInlineMesh is false");
    if (useInlineMesh && inlineBuiltin && (j.contains("inlineVertices") || j.contains("inlineIndices")))
        throw std::invalid_argument(std::string(expectedType) + " builtin inline mesh cannot contain raw data");
    if (useInlineMesh && !inlineBuiltin && (!j.contains("inlineVertices") || !j.contains("inlineIndices")))
        throw std::invalid_argument(std::string(expectedType) + " raw inline mesh requires vertices and indices");

    if (j.contains("inlineVertices")) {
        const auto &vertices = j["inlineVertices"];
        if (!vertices.is_array())
            throw std::invalid_argument(std::string(expectedType) + ".inlineVertices must be an array");
        for (size_t index = 0; index < vertices.size(); ++index) {
            const auto &vertex = vertices[index];
            if (!vertex.is_object() || vertex.size() != 5 || !vertex.contains("pos") || !vertex.contains("normal") ||
                !vertex.contains("tangent") || !vertex.contains("color") || !vertex.contains("texCoord"))
                throw std::invalid_argument(std::string(expectedType) + ".inlineVertices[" + std::to_string(index) +
                                            "] has invalid fields");
            const auto validateVector = [&](const char *field, size_t size) {
                const auto &value = vertex[field];
                if (!value.is_array() || value.size() != size)
                    throw std::invalid_argument(std::string(expectedType) + ".inlineVertices[" + std::to_string(index) +
                                                "]." + field + " has invalid length");
                for (const auto &item : value) {
                    if (!item.is_number())
                        throw std::invalid_argument("inline vertex attribute must contain numbers");
                    const double number = item.get<double>();
                    if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
                        throw std::invalid_argument("inline vertex attribute must contain finite floats");
                }
            };
            validateVector("pos", 3);
            validateVector("normal", 3);
            validateVector("tangent", 4);
            validateVector("color", 3);
            validateVector("texCoord", 2);
        }
    }
    if (j.contains("inlineIndices")) {
        const auto &indices = j["inlineIndices"];
        if (!indices.is_array())
            throw std::invalid_argument(std::string(expectedType) + ".inlineIndices must be an array");
        for (const auto &index : indices) {
            if (!index.is_number_unsigned() || index.get<uint64_t>() > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::string(expectedType) + ".inlineIndices contains an invalid index");
        }
    }

    if (expectedType == "SkinnedMeshRenderer" && j.contains("activeTakeName"))
        RequireString(j, "activeTakeName", expectedType);
    if (expectedType == "SpriteRenderer") {
        const auto &frameId = RequireString(j, "frameId", expectedType);
        if (!frameId.empty()) {
            const bool valid =
                frameId.size() == 32 && std::all_of(frameId.begin(), frameId.end(), [](unsigned char ch) {
                    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
                });
            if (!valid)
                throw std::invalid_argument(
                    "SpriteRenderer.frameId must be empty or a 32-character lowercase UUID hex string");
        }
        RequireFiniteVector(j, "spriteColor", 4, expectedType);
        RequireBoolean(j, "flipX", expectedType);
        RequireBoolean(j, "flipY", expectedType);
        if (j.contains("spriteGuid")) {
            if (RequireString(j, "spriteGuid", expectedType).empty())
                throw std::invalid_argument("SpriteRenderer.spriteGuid must not be empty");
            if (frameId.empty())
                throw std::invalid_argument(
                    "SpriteRenderer.frameId must identify a persisted SpriteFrame when spriteGuid is set");
        }
    }
    if (expectedType == "LineRenderer") {
        const auto &positions = j["positions"];
        if (!positions.is_array())
            throw std::invalid_argument("LineRenderer.positions must be an array");
        for (size_t index = 0; index < positions.size(); ++index) {
            if (!positions[index].is_array() || positions[index].size() != 3)
                throw std::invalid_argument("LineRenderer.positions entries must contain three numbers");
            for (const auto &coordinate : positions[index]) {
                if (!coordinate.is_number() || !std::isfinite(coordinate.get<double>()))
                    throw std::invalid_argument("LineRenderer.positions must contain finite numbers");
            }
        }
        const auto requireFiniteNonNegative = [&](const char *field) {
            if (!j[field].is_number())
                throw std::invalid_argument(std::string("LineRenderer.") + field + " must be finite and non-negative");
            const double value = j[field].get<double>();
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument(std::string("LineRenderer.") + field + " must be finite and non-negative");
        };
        requireFiniteNonNegative("widthMultiplier");
        requireFiniteNonNegative("shadowBias");
        RequireFiniteVector(j, "textureScale", 2, expectedType);
        RequireBoolean(j, "loop", expectedType);
        RequireBoolean(j, "useWorldSpace", expectedType);
        RequireBoolean(j, "generateLightingData", expectedType);
        const auto requireEnum = [&](const char *field, int maximum) {
            const int64_t value = RequireInteger(j, field, expectedType);
            if (value < 0 || value > maximum)
                throw std::invalid_argument(std::string("LineRenderer.") + field + " is outside its enum range");
        };
        requireEnum("alignment", 1);
        requireEnum("textureMode", 4);
        const auto requireRoundingCount = [&](const char *field) {
            const uint64_t value = RequireUnsignedInteger(j, field, expectedType);
            if (value > 1024u)
                throw std::invalid_argument(std::string("LineRenderer.") + field + " exceeds the supported limit");
        };
        requireRoundingCount("numCornerVertices");
        requireRoundingCount("numCapVertices");

        const auto &widthCurve = j["widthCurve"];
        if (!widthCurve.is_object() || widthCurve.size() != 3 || !widthCurve.contains("keys") ||
            !widthCurve.contains("preWrap") || !widthCurve.contains("postWrap") || !widthCurve["keys"].is_array() ||
            widthCurve["keys"].empty())
            throw std::invalid_argument("LineRenderer.widthCurve has invalid fields");
        double previousTime = -std::numeric_limits<double>::infinity();
        for (const auto &key : widthCurve["keys"]) {
            if (!key.is_object() || key.size() != 4 || !key.contains("time") || !key.contains("value") ||
                !key.contains("inTangent") || !key.contains("outTangent"))
                throw std::invalid_argument("LineRenderer.widthCurve contains an invalid key");
            for (const char *field : {"time", "value", "inTangent", "outTangent"}) {
                if (!key[field].is_number() || !std::isfinite(key[field].get<double>()))
                    throw std::invalid_argument("LineRenderer.widthCurve keys must contain finite numbers");
            }
            const double time = key["time"].get<double>();
            if (time <= previousTime || key["value"].get<double>() < 0.0)
                throw std::invalid_argument(
                    "LineRenderer.widthCurve key times must increase and values must be non-negative");
            previousTime = time;
        }
        const auto requireNestedEnum = [&](const nlohmann::json &object, const char *field, int maximum,
                                           const char *label) {
            if (!object[field].is_number_integer())
                throw std::invalid_argument(std::string(label) + " must be an integer");
            const int64_t value = object[field].get<int64_t>();
            if (value < 0 || value > maximum)
                throw std::invalid_argument(std::string(label) + " is outside its enum range");
        };
        requireNestedEnum(widthCurve, "preWrap", 2, "LineRenderer.widthCurve.preWrap");
        requireNestedEnum(widthCurve, "postWrap", 2, "LineRenderer.widthCurve.postWrap");

        const auto &gradient = j["colorGradient"];
        if (!gradient.is_object() || gradient.size() != 2 || !gradient.contains("keys") || !gradient.contains("mode") ||
            !gradient["keys"].is_array() || gradient["keys"].empty())
            throw std::invalid_argument("LineRenderer.colorGradient has invalid fields");
        previousTime = -std::numeric_limits<double>::infinity();
        for (const auto &key : gradient["keys"]) {
            if (!key.is_object() || key.size() != 2 || !key.contains("time") || !key.contains("color") ||
                !key["time"].is_number())
                throw std::invalid_argument("LineRenderer.colorGradient contains an invalid key");
            const double time = key["time"].get<double>();
            if (!std::isfinite(time) || time < 0.0 || time > 1.0 || time <= previousTime)
                throw std::invalid_argument(
                    "LineRenderer.colorGradient key times must strictly increase between zero and one");
            const auto &color = key["color"];
            if (!color.is_array() || color.size() != 4)
                throw std::invalid_argument("LineRenderer.colorGradient colors require four channels");
            for (const auto &channel : color) {
                if (!channel.is_number() || !std::isfinite(channel.get<double>()))
                    throw std::invalid_argument("LineRenderer.colorGradient colors must be finite");
            }
            previousTime = time;
        }
        requireNestedEnum(gradient, "mode", 2, "LineRenderer.colorGradient.mode");
    }
}

bool MeshRenderer::DeserializeDocument(const nlohmann::json &j)
{
    try {
        ValidateSerializedDocumentForType(j, GetTypeName());

        auto &registry = AssetRegistry::Instance();
        const std::string meshGuid = j.value("meshAssetGuid", std::string());
        std::shared_ptr<InxMesh> stagedMesh;
        if (!meshGuid.empty()) {
            stagedMesh = registry.LoadAsset<InxMesh>(meshGuid, ResourceType::Mesh);
        }
        const std::string stagedSubresourceId = j.value("modelSubresourceId", std::string{});
        const auto stagedPath = ResolveModelNodePathBySubresourceId(
            meshGuid, stagedSubresourceId, j.value("modelNodePath", std::vector<std::string>{}));
        const bool meshAssetResolved = static_cast<bool>(stagedMesh);
        if (!stagedPath.empty() && stagedMesh && !stagedMesh->GetModelSourceGeometry())
            throw std::invalid_argument("Model node binding requires node-local source geometry");
        if (stagedMesh && !stagedPath.empty())
            static_cast<void>(stagedMesh->RequireModelNode(stagedPath));

        const auto &materialsDocument = j["materials"];
        std::vector<AssetRef<InxMaterial>> stagedMaterials(materialsDocument.size());
        for (size_t index = 0; index < materialsDocument.size(); ++index) {
            const auto &slotDocument = materialsDocument[index];
            if (slotDocument.is_null())
                continue;
            if (slotDocument.is_string()) {
                const std::string guid = slotDocument.get<std::string>();
                auto material = registry.LoadAsset<InxMaterial>(guid, ResourceType::Material);
                if (material)
                    stagedMaterials[index] =
                        AssetRef<InxMaterial>(guid, std::move(material), registry.GetAssetVersion(guid));
                else
                    stagedMaterials[index] = AssetRef<InxMaterial>(guid);
                continue;
            }

            auto material = std::make_shared<InxMaterial>();
            if (!material->DeserializeDocument(slotDocument["material"]))
                throw std::invalid_argument("invalid embedded material document");
            stagedMaterials[index] = AssetRef<InxMaterial>(std::string(), std::move(material), 0);
        }
        std::vector<std::unordered_map<std::string, MaterialProperty>> stagedParameters;
        if (j.contains("parameterOverrides")) {
            const auto &parameterDocument = j["parameterOverrides"];
            stagedParameters.resize(parameterDocument.size());
            for (size_t slot = 0; slot < parameterDocument.size(); ++slot) {
                for (const auto &[name, value] : parameterDocument[slot].items())
                    stagedParameters[slot].emplace(name, DeserializeRendererParameter(name, value));
            }
        }

        if (!Component::DeserializeDocument(j))
            return false;

        // Mesh reference
        m_mesh.meshId = j["meshId"].get<uint64_t>();

        if (j.contains("nodeGroup"))
            m_nodeGroup = j["nodeGroup"].get<int32_t>();
        else
            m_nodeGroup = -1;

        m_modelNodePath.clear();
        m_modelSubresourceId.clear();
        m_submeshIndex = j.contains("submeshIndex") ? j["submeshIndex"].get<int32_t>() : -1;
        if (j.contains("meshPivotOffset")) {
            m_meshPivotOffset.x = j["meshPivotOffset"][0].get<float>();
            m_meshPivotOffset.y = j["meshPivotOffset"][1].get<float>();
            m_meshPivotOffset.z = j["meshPivotOffset"][2].get<float>();
        } else {
            m_meshPivotOffset = glm::vec3(0.0f);
        }

        // Mesh asset GUID (model-file meshes managed by AssetRegistry)
        if (stagedMesh)
            SetMeshAsset(meshGuid, std::move(stagedMesh));
        else if (!meshGuid.empty())
            SetMeshAssetGuid(meshGuid);
        else
            ClearMeshAsset();

        // Restore the view after assigning the source. Ordinary mesh changes
        // clear the previous model binding, while a scene document owns both.
        m_modelNodePath = stagedPath;
        m_modelSubresourceId = stagedSubresourceId;
        ResolveModelNodeBinding();
        if (const auto mesh = m_meshAsset.Get())
            UpdateBoundsForMeshSelection(mesh);

        // Materials are GUID strings, null slots, or typed runtime documents.
        auto &graph = AssetDependencyGraph::Instance();
        for (auto &ref : m_materials) {
            if (ref.HasGuid())
                graph.RemoveRuntimeDependency(GetInstanceGuid(), ref.GetGuid());
        }
        m_materials = std::move(stagedMaterials);
        m_embeddedMaterialVersions.assign(m_materials.size(), std::nullopt);
        for (const auto &reference : m_materials) {
            if (reference.HasGuid())
                graph.AddRuntimeDependency(GetInstanceGuid(), reference.GetGuid());
        }

        // Sync slot count to mesh submesh count
        SyncMaterialSlotsToMesh();

        for (size_t slot = 0; slot < stagedParameters.size(); ++slot) {
            auto material = GetEffectiveMaterial(static_cast<uint32_t>(slot));
            if (!material)
                throw std::invalid_argument("renderer parameter slot has no effective material");
            for (auto &[name, property] : stagedParameters[slot]) {
                const MaterialProperty *declaration = material->GetProperty(name);
                if (!declaration)
                    throw std::invalid_argument("material shader has no parameter named '" + name + "'");
                if (declaration->type != property.type)
                    throw std::invalid_argument("renderer parameter '" + name +
                                                "' does not match the reflected shader type");
                if (property.type == MaterialPropertyType::Texture2D)
                    property.value = InxMaterial::RequireTextureGuid(std::get<std::string>(property.value));
            }
        }

        m_persistentParameters = std::move(stagedParameters);
        m_runtimeParameters.clear();
        m_runtimeParameterWriteRevision = 0;
        m_parameterBlocks.clear();
        const size_t parameterSlotCount = (std::max)(m_materials.size(), m_persistentParameters.size());
        m_persistentParameters.resize(parameterSlotCount);
        m_runtimeParameters.resize(parameterSlotCount);
        m_parameterBlocks.resize(parameterSlotCount);
        for (uint32_t slot = 0; slot < static_cast<uint32_t>(parameterSlotCount); ++slot)
            PublishParameterSlot(slot);
        RefreshParameterTextureDependencies();

        // Rendering flags
        m_castShadows = j["castShadows"].get<bool>();
        m_receiveShadows = j["receivesShadows"].get<bool>();
        // A resident external mesh owns its imported bounds. Documents without
        // one (including inline geometry) use their required serialized bounds.
        if (!meshAssetResolved) {
            m_localBoundsMin = glm::vec3(j["boundsMin"][0].get<float>(), j["boundsMin"][1].get<float>(),
                                         j["boundsMin"][2].get<float>());
            m_localBoundsMax = glm::vec3(j["boundsMax"][0].get<float>(), j["boundsMax"][1].get<float>(),
                                         j["boundsMax"][2].get<float>());
        }

        // Inline mesh data (for primitives like cubes)
        m_useInlineMesh = j.value("useInlineMesh", false);
        m_inlineMeshName = j.value("inlineMeshName", std::string());
        m_inlineVertices.clear();
        m_inlineIndices.clear();
        m_sharedVertices = nullptr;
        m_sharedIndices = nullptr;
        m_vertexBuffer.reset();

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

        ++m_inlineMeshVersion;

        // Component deserialization is order-independent. A MeshCollider may
        // have been deserialized before the complete model binding was ready.
        NotifyCollisionGeometryChanged(this);

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
    clone->m_inlineMeshVersion = m_inlineMeshVersion;
    clone->m_sharedVertices = m_sharedVertices;
    clone->m_sharedIndices = m_sharedIndices;
    if (!m_sharedVertices) {
        clone->m_inlineVertices = m_inlineVertices;
        clone->m_inlineIndices = m_inlineIndices;
    }
    // Materials
    clone->m_materials = m_materials;
    clone->m_embeddedMaterialVersions = m_embeddedMaterialVersions;
    clone->m_persistentParameters = m_persistentParameters;
    clone->m_runtimeParameters.clear();
    clone->m_runtimeParameterWriteRevision = 0;
    clone->m_parameterBlocks.resize(clone->m_persistentParameters.size());
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(clone->m_persistentParameters.size()); ++slot)
        clone->PublishParameterSlot(slot);
    auto &graph = AssetDependencyGraph::Instance();
    if (clone->m_meshAsset.HasGuid())
        graph.AddRuntimeDependency(clone->GetInstanceGuid(), clone->m_meshAsset.GetGuid());
    for (const auto &reference : clone->m_materials) {
        if (reference.HasGuid())
            graph.AddRuntimeDependency(clone->GetInstanceGuid(), reference.GetGuid());
    }
    // Rendering flags
    clone->m_castShadows = m_castShadows;
    clone->m_receiveShadows = m_receiveShadows;
    // Submesh / node group
    clone->m_submeshIndex = m_submeshIndex;
    clone->m_nodeGroup = m_nodeGroup;
    clone->m_modelNodePath = m_modelNodePath;
    clone->m_modelSubresourceId = m_modelSubresourceId;
    clone->m_meshPivotOffset = m_meshPivotOffset;
    // Bounds
    clone->m_localBoundsMin = m_localBoundsMin;
    clone->m_localBoundsMax = m_localBoundsMax;
    return clone;
}

} // namespace infernux
