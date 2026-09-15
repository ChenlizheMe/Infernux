#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxResource/InxResourceMeta.h>
#include <platform/filesystem/InxPath.h>

#include <cassert>
#include <filesystem>
#include <set>
#include <string>

#ifndef INFERNUX_SOURCE_DIR
#error "INFERNUX_SOURCE_DIR must be supplied by the CMake test target"
#endif

int main()
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
