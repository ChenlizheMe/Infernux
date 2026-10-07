#pragma once

#include <function/resources/InxMesh/ModelSourceIO.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxMesh/MeshImportSettings.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedModelImporter.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>

#include <array>
#include <cassert>
#include <chrono>
#include <fstream>
#include <future>

inline void TestModelSourceFileIO()
{
    using namespace infernux;
    const auto cwd = std::filesystem::current_path();
    const auto root = std::filesystem::temp_directory_path() /
                      ("infernux-model-io-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    assert(std::filesystem::create_directory(root));
    const std::array folders{root / ToFsPath("中文 模型🧩"), root / "independent"};
    for (size_t index = 0; index < folders.size(); ++index) {
        assert(std::filesystem::create_directory(folders[index]));
        const auto &folder = folders[index];
        const std::array<float, 9> positions{0, 0, 0, static_cast<float>(index + 2), 0, 0, 0, 1, 0};
        const std::array<uint16_t, 3> indices{0, 1, 2};
        {
            std::ofstream blob(folder / "triangle.bin", std::ios::binary);
            blob.write(reinterpret_cast<const char *>(positions.data()), sizeof(positions));
            blob.write(reinterpret_cast<const char *>(indices.data()), sizeof(indices));
            assert(blob.good());
        }
        std::ofstream(folder / "model.gltf") << R"({
            "asset":{"version":"2.0"}, "buffers":[{"uri":"triangle.bin","byteLength":42}],
            "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},
                           {"buffer":0,"byteOffset":36,"byteLength":6}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
                          "min":[0,0,0],"max":[3,1,0]},
                         {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
            "nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0})";
    }
    ModelSourceIO io(FromFsPath(folders[0] / "model.gltf"));
    assert(io.Exists("triangle.bin"));
    assert(!io.Exists("missing.bin"));
    assert(io.ComparePaths("triangle.bin", FromFsPath(folders[0] / "./triangle.bin").c_str()));
    assert(!io.ComparePaths("triangle.bin", FromFsPath(folders[1] / "triangle.bin").c_str()));
    assert(!io.Open("triangle.bin", "wb"));
    assert(!io.ChangeDirectory(FromFsPath(folders[1])));
    {
        std::unique_ptr<Assimp::IOStream> stream(io.Open("triangle.bin"));
        assert(stream && stream->FileSize() == 42);
        std::array<float, 9> positions{};
        assert(stream->Read(positions.data(), sizeof(float), positions.size()) == positions.size());
        assert(positions[3] == 2 && stream->Tell() == 36);
        assert(stream->Seek(static_cast<size_t>(-6), aiOrigin_END) == AI_SUCCESS);
        std::array<uint16_t, 3> indices{};
        assert(stream->Read(indices.data(), sizeof(uint16_t), 3) == 3);
        assert((indices == std::array<uint16_t, 3>{0, 1, 2}));
        assert(stream->Read(indices.data(), sizeof(uint16_t), 1) == 0);
        assert(stream->Seek(0, aiOrigin_SET) == AI_SUCCESS);
        assert(stream->Read(positions.data(), sizeof(float), 3) == 3);
    }
    const auto import = [](const auto &folder, float width) {
        InxResourceMeta metadata;
        MeshImportSettings::InitializeDefaults(metadata);
        const auto source = FromFsPath(folder / "model.gltf");
        const auto ordinary = MeshLoader::ImportSourceDetailed(source, "model", metadata);
        assert(ordinary.vertexCount == 3 && ordinary.indexCount == 3);
        const auto skinned = SkinnedModelImporter::ImportSource("model", source, 1.0f);
        assert(skinned->baseVertices.size() == 3 && skinned->indices.size() == 3);
        float maximumX = 0;
        for (const auto &vertex : skinned->baseVertices)
            maximumX = std::max(maximumX, vertex.pos.x);
        assert(maximumX == width);
    };
    // Simultaneous source files use identically named sidecars with different
    // geometry. No global chdir or shared resolver state may mix their data.
    auto first = std::async(std::launch::async, import, folders[0], 2.0f);
    auto second = std::async(std::launch::async, import, folders[1], 3.0f);
    first.get();
    second.get();
    std::ofstream(folders[0] / "model.obj") << "mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Hull\nf 1 2 3\n";
    std::ofstream(folders[0] / "model.mtl") << "newmtl Hull\nKd 0.2 0.4 0.6\nmap_Kd texture.png\n";
    std::ofstream(folders[0] / "texture.png") << "texture scan uses source paths, not image decoding";
    const auto textures = MeshLoader::ScanExternalTexturePaths(FromFsPath(folders[0] / "model.obj"));
    assert(textures.size() == 1);
    assert(FilesystemPathsEquivalent(*textures.begin(), FromFsPath(folders[0] / "texture.png")));
    assert(std::filesystem::current_path() == cwd);
    for (const auto &folder : folders) {
        for (const auto *name : {"triangle.bin", "model.gltf", "model.obj", "model.mtl", "texture.png"})
            std::filesystem::remove(folder / name);
        std::filesystem::remove(folder);
    }
    std::filesystem::remove(root);
}
