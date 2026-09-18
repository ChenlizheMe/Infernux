#pragma once

#include <function/resources/AssetRegistry/AssetRegistry.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace infernux
{

class InxMesh;
class InxSkinnedMesh;

struct MeshSourceImportResult
{
    struct TextureSource
    {
        uint32_t materialSlot;
        std::string path;
        uint32_t channel = 0; // ModelTexture index, shared with MaterialSlotData.
        int32_t embeddedIndex = -1;
    };
    struct EmbeddedImage
    {
        std::string key;
        std::string name;
        std::vector<unsigned char> bytes; // Encoded image, or RGBA8 when height != 0.
        uint32_t width = 0;
        uint32_t height = 0;
    };
    std::vector<TextureSource> textureSources;
    std::vector<EmbeddedImage> embeddedImages;
    std::shared_ptr<InxMesh> mesh;
    std::shared_ptr<InxSkinnedMesh> skinnedMesh;
    uint64_t meshCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    std::vector<std::string> materialSlots;
    std::vector<std::string> boneNames;
    std::vector<std::string> animationNames;
};

/**
 * @brief IAssetLoader implementation for 3D model assets (.fbx, .obj, .gltf, …).
 *
 * Source import uses Assimp for interchange models or MeshArtifact for native
 * .inxmesh sources. Native sources
 * preserve authored geometry without applying
 * Assimp conversion settings. Both produce the same imported binary
 * artifacts;
 * runtime Load reads those artifacts, not the original model.
 * Key design points:
 *   - Load() produces a new shared_ptr<InxMesh> from imported geometry.
 *   - Reload() replaces the geometry data
 * in-place so all AssetRef holders see updated data without re-resolving.
 *   - ScanDependencies() resolves external material textures to project GUIDs
 *     at the authoring boundary so
 * composite model sources are Cook-complete.
 *     Renderer-authored material bindings remain independent.
 */
class MeshLoader final : public IAssetLoader
{
  public:
    [[nodiscard]] static MeshSourceImportResult
    ImportSourceDetailed(const std::string &filePath, const std::string &guid, const InxResourceMeta &metadata);
    [[nodiscard]] static std::shared_ptr<InxMesh> ImportSource(const std::string &filePath, const std::string &guid,
                                                               const InxResourceMeta &metadata);

    /// Enumerate regular external texture files referenced by a composite
    /// model source. The AssetDatabase authoring boundary converts these
    /// paths to GUID dependencies before publication.
    [[nodiscard]] static std::set<std::string> ScanExternalTexturePaths(const std::string &filePath);

    RuntimeAssetPayload Load(const std::string &filePath, const std::string &guid, AssetDatabase *adb) override;
    [[nodiscard]] bool SupportsWorkerLoad() const noexcept override
    {
        return true;
    }

    bool Reload(const RuntimeAssetPayload &existing, const std::string &filePath, const std::string &guid,
                AssetDatabase *adb) override;
    [[nodiscard]] size_t EstimateRuntimeBytes(const RuntimeAssetPayload &payload) const override;

    std::set<std::string> ScanDependencies(const std::string &filePath, AssetDatabase *adb) override;

    void CreateMeta(const char *content, size_t contentSize, const std::string &filePath,
                    InxResourceMeta &metaData) const override;
};

} // namespace infernux
