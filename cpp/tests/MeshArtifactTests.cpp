#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifact.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/scene/PrimitiveMeshes.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
template <typename Callback> void RequireInvalid(Callback callback)
{
    bool rejected = false;
    try {
        callback();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

bool NearlyEqual(float left, float right)
{
    return std::abs(left - right) < 1.0e-6f;
}
} // namespace

int main()
{
    std::shared_ptr<const infernux::MeshGeometry> retainedGeometry;
    {
        infernux::InxMesh mesh("mutable identity");
        infernux::Vertex oldVertex{};
        oldVertex.pos = {1.0f, 2.0f, 3.0f};
        infernux::SubMesh oldSubMesh;
        oldSubMesh.name = "old";
        mesh.SetData({oldVertex}, {0, 0, 0}, {oldSubMesh});
        retainedGeometry = mesh.GetGeometrySnapshot();
        infernux::InxMesh copy = mesh;
        oldVertex.pos = {9.0f, 8.0f, 7.0f};
        mesh.SetData({oldVertex}, {0}, {});
        assert(mesh.GetGeometrySnapshot() != retainedGeometry);
        assert(mesh.GetVertices().front().pos == oldVertex.pos);
        assert(mesh.GetBoundsMin() == oldVertex.pos);
        assert(copy.GetGeometrySnapshot() == retainedGeometry);
        assert(copy.GetVertices().front().pos == glm::vec3(1.0f, 2.0f, 3.0f));
        assert(retainedGeometry->indices.size() == 3);
        assert(retainedGeometry->subMeshes.front().name == "old");
        assert(retainedGeometry->boundsMax == glm::vec3(1.0f, 2.0f, 3.0f));
        mesh.SetData({}, {}, {});
        assert(mesh.GetVertices().empty());
        assert(mesh.GetBoundsMin() == glm::vec3(0.0f));
        assert(mesh.GetGeneration() == 3);
    }
    assert(retainedGeometry->vertices.front().pos.x == 1.0f);
    const std::weak_ptr<const infernux::MeshGeometry> retired = retainedGeometry;
    retainedGeometry.reset();
    assert(retired.expired());

    {
        infernux::InxMesh mesh("range update");
        std::vector<infernux::Vertex> vertices(6);
        for (size_t index = 0; index < vertices.size(); ++index)
            vertices[index].pos = glm::vec3(static_cast<float>(index));
        infernux::SubMesh left, right;
        left.vertexCount = 3;
        left.indexCount = 3;
        left.name = "left";
        right.vertexStart = 3;
        right.vertexCount = 3;
        right.indexStart = 3;
        right.indexCount = 3;
        right.materialSlot = 1;
        mesh.SetData(vertices, {0, 1, 2, 3, 4, 5}, {left, right});
        const auto before = mesh.GetGeometrySnapshot();
        auto replacement = vertices[1];
        replacement.pos = {-2.0f, 8.0f, 1.0f};
        replacement.normal = {1.0f, 0.0f, 0.0f};
        mesh.UpdateVertexRange(1, {replacement});
        assert(mesh.GetGeneration() == 2);
        assert(before->vertices[1].pos == glm::vec3(1.0f));
        assert(mesh.GetVertices()[1].normal == replacement.normal);
        assert(mesh.GetVertices()[0].pos == vertices[0].pos);
        assert(mesh.GetVertices()[5].pos == vertices[5].pos);
        assert(mesh.GetIndices() == before->indices);
        assert(mesh.GetBoundsMin() == glm::vec3(-2.0f, 0.0f, 0.0f));
        assert(mesh.GetBoundsMax() == glm::vec3(5.0f, 8.0f, 5.0f));
        assert(mesh.GetSubMesh(0).boundsMax == glm::vec3(2.0f, 8.0f, 2.0f));
        assert(mesh.GetSubMesh(1).boundsMin == glm::vec3(3.0f));
        assert(mesh.GetSubMesh(1).materialSlot == 1);
        const auto updated = mesh.GetGeometrySnapshot();
        RequireInvalid([&] { mesh.UpdateVertexRange(6, {replacement}); });
        RequireInvalid([&] { mesh.UpdateVertexRange(size_t(-1), {}); });
        mesh.UpdateVertexRange(6, {});
        assert(mesh.GetGeometrySnapshot() == updated);
        assert(mesh.GetGeneration() == 2);
    }

    const infernux::Vertex defaultVertex{};
    assert(defaultVertex.pos == glm::vec3(0.0f));
    assert(defaultVertex.normal == glm::vec3(0.0f, 1.0f, 0.0f));
    assert(defaultVertex.tangent == glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    assert(defaultVertex.color == glm::vec3(1.0f));
    assert(defaultVertex.texCoord == glm::vec2(0.0f));

    const auto &sphereVertices = infernux::PrimitiveMeshes::GetSphereVertices();
    const auto &sphereIndices = infernux::PrimitiveMeshes::GetSphereIndices();
    assert(sphereIndices.size() % 3 == 0);
    for (size_t triangle = 0; triangle < sphereIndices.size(); triangle += 3) {
        const float u0 = sphereVertices.at(sphereIndices[triangle]).texCoord.x;
        const float u1 = sphereVertices.at(sphereIndices[triangle + 1]).texCoord.x;
        const float u2 = sphereVertices.at(sphereIndices[triangle + 2]).texCoord.x;
        const float minU = std::min({u0, u1, u2});
        const float maxU = std::max({u0, u1, u2});
        assert(maxU - minU <= 0.500001f);
    }

    infernux::InxMesh source("artifact-probe");
    assert(source.GetGeneration() == 0);
    infernux::Vertex vertex{};
    vertex.pos = {1.0f, 2.0f, 3.0f};
    vertex.normal = {0.0f, 1.0f, 0.0f};
    vertex.tangent = {1.0f, 0.0f, 0.0f, -1.0f};
    vertex.color = {0.25f, 0.5f, 0.75f};
    vertex.texCoord = {0.125f, 0.875f};
    vertex.boneIndices = {1, 2, 3, 4};
    vertex.boneWeights = {0.4f, 0.3f, 0.2f, 0.1f};

    infernux::SubMesh subMesh;
    subMesh.indexCount = 3;
    subMesh.vertexCount = 1;
    subMesh.materialSlot = 2;
    subMesh.nodeGroup = 1;
    subMesh.boundsMin = vertex.pos;
    subMesh.boundsMax = vertex.pos;
    subMesh.name = "triangle";
    source.SetData({vertex}, {0, 0, 0}, {subMesh});
    assert(source.GetGeneration() == 1);
    source.SetData({vertex}, {0, 0, 0}, {subMesh});
    assert(source.GetGeneration() == 2);
    source.SetMaterialSlotNames({"surface"});
    infernux::MaterialSlotData material;
    material.baseColor = {0.1f, 0.2f, 0.3f, 0.4f};
    material.emissionColor = {0.5f, 0.6f, 0.7f, 0.8f};
    material.metallic = 0.9f;
    material.sourceId = "material/surface";
    material.materialGuid = "abcdabcdabcdabcdabcdabcdabcdabcd";
    material.smoothness = 0.65f;
    material.opacity = 0.4f;
    material.alphaMode = infernux::ModelAlphaMode::Mask;
    material.alphaCutoff = 0.37f;
    material.doubleSided = true;
    source.SetMaterialSlotData({material});
    source.SetNodeNames({"root", "child"});

    // Old native mesh sources remain byte-for-byte geometry-only payloads.
    constexpr const char *SourceHash = "0123456789abcdef";
    const auto geometryOnly = infernux::MeshArtifact::Serialize(source, SourceHash);
    auto geometryOnlyRestored = infernux::MeshArtifact::Deserialize(geometryOnly, SourceHash);
    assert(geometryOnlyRestored->GetModelNodes().empty());
    assert(infernux::MeshArtifact::Serialize(*geometryOnlyRestored, SourceHash) == geometryOnly);

    infernux::ImportedModelNode root, pivot, child;
    root.name = "Assembly";
    root.localTransform[3] = {2.0f, 3.0f, 4.0f, 1.0f};
    pivot.name = "Empty pivot";
    pivot.parentIndex = 0;
    pivot.localTransform[0][0] = -2.0f;
    child.name = "child";
    child.parentIndex = 1;
    child.nodeGroup = 1;
    child.localTransform[3] = {0.0f, 5.0f, 0.0f, 1.0f};
    source.SetModelNodes({root, pivot, child});
    auto invalidChild = child;
    invalidChild.parentIndex = 2;
    RequireInvalid([&] { source.SetModelNodes({root, pivot, invalidChild}); });
    invalidChild = child;
    invalidChild.nodeGroup = 2;
    RequireInvalid([&] { source.SetModelNodes({root, pivot, invalidChild}); });
    invalidChild = child;
    invalidChild.localTransform[0][0] = std::numeric_limits<float>::quiet_NaN();
    RequireInvalid([&] { source.SetModelNodes({root, pivot, invalidChild}); });
    RequireInvalid([&] { source.SetModelNodes({root, pivot, child, child}); });
    assert(source.GetModelNodes().size() == 3); // Invalid publication does not replace it.

    const std::string bytes = infernux::MeshArtifact::Serialize(source, SourceHash);
    auto restored = infernux::MeshArtifact::Deserialize(bytes, SourceHash);
    assert(restored->GetName() == "artifact-probe");
    assert(restored->GetVertexCount() == 1);
    assert(restored->GetIndexCount() == 3);
    assert(restored->GetSubMeshCount() == 1);
    assert(restored->GetSubMesh(0).name == "triangle");
    assert(restored->GetMaterialSlotNames() == std::vector<std::string>{"surface"});
    assert(restored->GetNodeNames() == std::vector<std::string>({"root", "child"}));
    const auto &nodes = restored->GetModelNodes();
    assert(nodes.size() == 3);
    assert(nodes[0].name == root.name && nodes[0].parentIndex == -1 && nodes[0].nodeGroup == -1);
    assert(nodes[1].name == pivot.name && nodes[1].parentIndex == 0 && nodes[1].nodeGroup == -1);
    assert(nodes[2].parentIndex == 1 && nodes[2].nodeGroup == 1);
    assert(nodes[0].localTransform == root.localTransform);
    assert(nodes[1].localTransform == pivot.localTransform);
    assert(nodes[2].localTransform == child.localTransform);
    assert(infernux::MeshArtifact::Serialize(*restored, SourceHash) == bytes);
    const auto &restoredVertex = restored->GetVertices().front();
    assert(NearlyEqual(restoredVertex.pos.x, 1.0f));
    assert(restoredVertex.boneIndices == glm::uvec4(1, 2, 3, 4));
    assert(NearlyEqual(restoredVertex.boneWeights.w, 0.1f));
    assert(NearlyEqual(restored->GetMaterialSlotData().front().metallic, 0.9f));
    assert(restored->GetMaterialSlotData().front().sourceId == material.sourceId);
    assert(restored->GetMaterialSlotData().front().materialGuid == material.materialGuid);
    assert(restored->GetMaterialSlotData().front().alphaMode == material.alphaMode);
    assert(restored->GetMaterialSlotData().front().alphaCutoff == material.alphaCutoff);
    assert(restored->GetMaterialSlotData().front().doubleSided == material.doubleSided);

    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(bytes, "different-source"); });

    std::string corrupted = bytes;
    corrupted[corrupted.size() / 2] ^= 0x5a;
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(corrupted, SourceHash); });
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(bytes.substr(0, bytes.size() - 1), SourceHash); });
    RequireInvalid([&] { (void)infernux::MeshArtifact::Serialize(source, {}); });

    source.SetGuid("original-model-guid");
    source.SetFilePath("Assets/original.obj");
    const auto authoredBytes = infernux::MeshArtifact::SerializeSource(source);
    auto authored = infernux::MeshArtifact::DeserializeSource(authoredBytes);
    assert(authored->GetGuid().empty());
    assert(authored->GetFilePath().empty());
    assert(authored->GetVertices().front().normal == vertex.normal);
    assert(authored->GetVertices().front().tangent == vertex.tangent);
    assert(authored->GetVertices().front().texCoord == vertex.texCoord);
    assert(authored->GetIndices() == source.GetIndices());
    assert(authored->GetMaterialSlotNames() == source.GetMaterialSlotNames());
    assert(infernux::MeshArtifact::SerializeSource(*authored) == authoredBytes);
    RequireInvalid([&] { (void)infernux::MeshArtifact::DeserializeSource(bytes); });
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(authoredBytes, SourceHash); });
    auto skinned = std::make_shared<infernux::InxSkinnedMesh>();
    skinned->baseVertices = source.GetVertices();
    skinned->indices = source.GetIndices();
    source.SetSkinnedData(skinned);
    RequireInvalid([&] { (void)infernux::MeshArtifact::SerializeSource(source); });

    // Imported assets store node-local geometry once. The ordinary mesh view
    // remains model-space for existing render, bounds and picking consumers.
    infernux::InxMesh localModel("local-model");
    localModel.SetNodeNames({"root", "child"});
    localModel.SetModelData({vertex}, {0, 0, 0}, {subMesh}, {root, pivot, child});
    assert(localModel.GetGeneration() == 1);
    assert(localModel.GetModelSourceGeometry()->vertices[0].pos == vertex.pos);
    assert(localModel.GetVertices()[0].pos == glm::vec3(0, 10, 7));
    assert(localModel.GetVertices()[0].normal == vertex.normal);
    assert(localModel.GetVertices()[0].tangent == glm::vec4(-1, 0, 0, 1));
    assert(localModel.GetSubMesh(0).boundsMin == localModel.GetVertices()[0].pos);
    const auto localBytes = infernux::MeshArtifact::SerializeSource(localModel);
    auto localRestored = infernux::MeshArtifact::DeserializeSource(localBytes);
    assert(localRestored->GetModelSourceGeometry()->vertices[0].pos == vertex.pos);
    assert(localRestored->GetVertices()[0].pos == localModel.GetVertices()[0].pos);
    assert(infernux::MeshArtifact::SerializeSource(*localRestored) == localBytes);
    auto bakedOnly = localModel;
    bakedOnly.SetData(localModel.GetVertices(), localModel.GetIndices(), localModel.GetSubMeshes());
    assert(!bakedOnly.GetModelSourceGeometry());
    // NOD1/NOD2 have the same metadata size: no duplicated vertex payload.
    const auto bakedBytes = infernux::MeshArtifact::SerializeSource(bakedOnly);
    assert(bakedBytes.size() == localBytes.size());
    assert(!infernux::MeshArtifact::DeserializeSource(bakedBytes)->GetModelSourceGeometry());

    const auto published = localModel.GetGeometrySnapshot();
    const auto localPublished = localModel.GetModelSourceGeometry();
    auto invalidSubMesh = subMesh;
    invalidSubMesh.vertexCount = 2;
    RequireInvalid([&] { localModel.SetModelData({vertex}, {0, 0, 0}, {invalidSubMesh}, {root, pivot, child}); });
    RequireInvalid([&] { localModel.SetModelData({vertex}, {1, 0, 0}, {subMesh}, {root, pivot, child}); });
    RequireInvalid([&] { localModel.SetModelData({vertex}, {}, {}, {root, pivot, child}); });
    RequireInvalid([&] { localModel.SetModelData({vertex}, {0, 0, 0}, {subMesh}, {root}); });
    assert(localModel.GetGeometrySnapshot() == published);
    assert(localModel.GetModelSourceGeometry() == localPublished);
    assert(localModel.GetGeneration() == 1);

    // Collapsed authored scale must not destroy the source or yield inverse
    // matrix NaNs. Editing node transforms re-derives from the same local data.
    auto flattened = pivot;
    flattened.localTransform[2][2] = 0.0f;
    localModel.SetModelNodes({root, flattened, child});
    assert(localModel.GetVertices()[0].pos == glm::vec3(0, 10, 4));
    assert(localModel.GetVertices()[0].normal == glm::vec3(0));
    assert(localModel.GetModelSourceGeometry()->vertices[0].pos == vertex.pos);
    assert(localModel.GetGeneration() == 2);
    localModel.SetModelNodes({root, pivot, child});
    assert(localModel.GetVertices()[0].pos == published->vertices[0].pos);
    assert(localPublished->vertices[0].pos == vertex.pos);

    infernux::InxMesh reloadTarget("before-reload");
    reloadTarget.SetGuid("persistent-asset-guid");
    reloadTarget.SetData({}, {}, {});
    reloadTarget.ReplaceImportedContent(localModel);
    assert(reloadTarget.GetGuid() == "persistent-asset-guid");
    assert(reloadTarget.GetGeneration() == 2);
    assert(reloadTarget.GetGeometrySnapshot() == localModel.GetGeometrySnapshot());
    assert(reloadTarget.GetModelSourceGeometry() == localModel.GetModelSourceGeometry());
    auto editedVertex = reloadTarget.GetVertices()[0];
    editedVertex.pos.x = 20.0f;
    reloadTarget.UpdateVertexRange(0, {editedVertex});
    assert(!reloadTarget.GetModelSourceGeometry());
    assert(localModel.GetVertices()[0].pos.x == 0.0f);

    std::cout << "Mesh artifact tests passed\n";
    return 0;
}
