#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifact.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <set>
#include <string>

#ifndef INFERNUX_SOURCE_DIR
#error "INFERNUX_SOURCE_DIR must be supplied by the CMake test target"
#endif

int main(int argc, char **argv)
{
    const std::filesystem::path sourceRoot = INFERNUX_SOURCE_DIR;
    const auto blendPath = sourceRoot / "external" / "assimp" / "test" / "models" / "BLEND" / "CubeHierarchy_248.blend";
    assert(std::filesystem::is_regular_file(blendPath));

    infernux::InxResourceMeta metadata;
    const auto imported =
        infernux::MeshLoader::ImportSourceDetailed(blendPath.string(), "0123456789abcdef0123456789abcdef", metadata);

    // A .blend is a composite source, but the loader must expose the same
    // engine-owned mesh contract as FBX/GLTF: binary-ready geometry, stable
    // submesh/material slots, and the authored node hierarchy.
    assert(imported.mesh);
    assert(imported.meshCount > 0);
    assert(imported.vertexCount > 0);
    assert(imported.indexCount > 0);
    assert(imported.mesh->GetSubMeshCount() == imported.meshCount);
    assert(!imported.mesh->GetNodeNames().empty());
    assert(imported.materialSlots.size() == imported.mesh->GetMaterialSlotNames().size());
    assert(imported.mesh->GetGuid() == "0123456789abcdef0123456789abcdef");

    // A real tree is distinct from the former flat list of mesh-node names.
    // Transform-only parents, nonuniform mirrored scale and local rotation
    // must survive import and the same binary artifact used by Player.
    const auto hierarchyPath = sourceRoot / "cpp/tests/fixtures/model_hierarchy.gltf";
    metadata.AddMetadata("scale_factor", 2.0f);
    const auto hierarchyImport = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(hierarchyPath),
                                                                            "hierarchy-test-guid", metadata);
    const auto hierarchy = hierarchyImport.mesh;
    assert(hierarchy);
    assert(hierarchyImport.meshCount == hierarchy->GetSubMeshCount());
    assert(hierarchyImport.vertexCount == hierarchy->GetVertexCount());
    assert(hierarchyImport.indexCount == hierarchy->GetIndexCount());
    assert(hierarchyImport.materialSlots == hierarchy->GetMaterialSlotNames());
    assert(hierarchy->GetModelSourceGeometry());
    const auto &nodes = hierarchy->GetModelNodes();
    const auto findNode = [](const auto &modelNodes, const std::string &name) -> size_t {
        for (size_t index = 0; index < modelNodes.size(); ++index)
            if (modelNodes[index].name == name)
                return index;
        assert(false && "expected imported node is missing");
        return 0;
    };
    const auto root = findNode(nodes, "Assembly");
    const auto pivot = findNode(nodes, "Empty pivot");
    const auto upper = findNode(nodes, "Upper");
    const auto lower = findNode(nodes, "Lower");
    assert(nodes[pivot].parentIndex == root);
    assert(nodes[upper].parentIndex == pivot && nodes[lower].parentIndex == pivot);
    assert(nodes[root].nodeGroup == -1 && nodes[pivot].nodeGroup == -1);
    assert(nodes[upper].nodeGroup >= 0 && nodes[lower].nodeGroup >= 0);
    assert(nodes[upper].nodeGroup != nodes[lower].nodeGroup);
    assert(glm::vec3(nodes[root].localTransform[3]) == glm::vec3(4, 6, 8));
    assert(glm::vec3(nodes[pivot].localTransform[3]) == glm::vec3(0, 10, 0));
    assert(nodes[pivot].localTransform[0][0] == -2.0f);
    assert(nodes[pivot].localTransform[1][1] == 3.0f);
    assert(std::abs(nodes[lower].localTransform[0][1] - 1.0f) < 1.e-5f);
    for (const auto &subMesh : hierarchy->GetSubMeshes()) {
        if (subMesh.nodeGroup != nodes[upper].nodeGroup)
            continue;
        // Existing combined geometry remains in model space, not transformed twice.
        assert(subMesh.boundsMin == glm::vec3(-4, 16, 8));
        assert(subMesh.boundsMax == glm::vec3(0, 22, 8));
    }
    const auto bytes = infernux::MeshArtifact::Serialize(*hierarchy, "hierarchy-source");
    const auto cooked = infernux::MeshArtifact::Deserialize(bytes, "hierarchy-source");
    assert(cooked->GetModelSourceGeometry());
    assert(cooked->GetModelSourceGeometry()->vertices.size() == hierarchy->GetVertexCount());
    for (size_t index = 0; index < hierarchy->GetVertexCount(); ++index) {
        assert(cooked->GetModelSourceGeometry()->vertices[index].pos ==
               hierarchy->GetModelSourceGeometry()->vertices[index].pos);
        assert(cooked->GetVertices()[index].pos == hierarchy->GetVertices()[index].pos);
    }
    assert(infernux::MeshArtifact::Serialize(*cooked, "hierarchy-source") == bytes);
    assert(cooked->GetModelNodes().size() == nodes.size());
    for (size_t index = 0; index < nodes.size(); ++index) {
        const auto &restored = cooked->GetModelNodes()[index];
        assert(restored.name == nodes[index].name);
        assert(restored.parentIndex == nodes[index].parentIndex);
        assert(restored.nodeGroup == nodes[index].nodeGroup);
        assert(restored.localTransform == nodes[index].localTransform);
    }

    // Author options must affect the actual imported geometry, not only meta.
    const auto duplicatePath = sourceRoot / "cpp/tests/fixtures/model_duplicate_vertices.obj";
    infernux::InxResourceMeta unweldedSettings;
    unweldedSettings.AddMetadata("weld_vertices", false);
    unweldedSettings.AddMetadata("optimize_mesh", false);
    const auto unwelded = infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(duplicatePath),
                                                                     "unwelded-guid", unweldedSettings);
    infernux::InxResourceMeta weldedSettings;
    weldedSettings.AddMetadata("weld_vertices", true);
    weldedSettings.AddMetadata("optimize_mesh", true);
    const auto welded =
        infernux::MeshLoader::ImportSourceDetailed(infernux::FromFsPath(duplicatePath), "welded-guid", weldedSettings);
    assert(unwelded.vertexCount == 6);
    assert(welded.vertexCount == 4);
    assert(unwelded.indexCount == 6 && welded.indexCount == 6);
    assert(unwelded.mesh->GetModelNodes().size() == welded.mesh->GetModelNodes().size());

    // Optional modern Blender-generated GLB supplied by an integration run.
    // This is additional evidence, never a replacement for the fixed fixture.
    if (argc > 1) {
        const auto modernImport = infernux::MeshLoader::ImportSourceDetailed(argv[1], "modern-blend-guid", metadata);
        const auto modern = modernImport.mesh;
        assert(modern);
        const auto &modernNodes = modern->GetModelNodes();
        const auto assembly = findNode(modernNodes, "Assembly");
        const auto hinge = findNode(modernNodes, "Hinge");
        assert(modernNodes[hinge].parentIndex == assembly);
        assert(modernNodes[findNode(modernNodes, "Upper")].parentIndex == hinge);
        assert(modernNodes[findNode(modernNodes, "Lower")].parentIndex == hinge);
        assert(modern->GetMaterialSlotCount() == 2);
        assert(modernImport.materialSlots == modern->GetMaterialSlotNames());
    }

    // Composite model sources must expose their regular external textures as
    // authoring paths so the AssetDatabase can publish GUID-only edges. The
    // OBJ fixture deliberately uses Windows-style relative texture tokens;
    // this keeps the importer contract exercised independently of a project
    // scan.
    const auto objPath = sourceRoot / "external" / "assimp" / "test" / "models" / "OBJ" / "spider.obj";
    const auto textures = infernux::MeshLoader::ScanExternalTexturePaths(objPath.string());
    assert(textures.size() == 5);
    for (const char *name :
         {"wal67ar_small.jpg", "wal69ar_small.jpg", "SpiderTex.jpg", "drkwood2.jpg", "engineflare1.jpg"}) {
        assert(textures.count(infernux::FromFsPath(objPath.parent_path() / name)) == 1);
    }
    return 0;
}
