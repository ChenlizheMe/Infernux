#pragma once

#include <memory>
#include <string>

struct aiScene;

namespace infernux
{

class InxSkinnedMesh;
struct MeshImportSettings;

class SkinnedModelImporter final
{
  public:
    static void ApplyAnimationClips(InxSkinnedMesh &model, const MeshImportSettings &settings);
    static void ApplyRigSettings(InxSkinnedMesh &model, const MeshImportSettings &settings,
                                 const InxSkinnedMesh *copiedDefinition = nullptr);
    static void ApplyAnimationSettings(InxSkinnedMesh &model, const MeshImportSettings &settings);
    [[nodiscard]] static bool HasSkinningData(const aiScene &scene, bool includeAnimations = true) noexcept;
    [[nodiscard]] static std::shared_ptr<InxSkinnedMesh>
    ConvertScene(const aiScene &scene, const std::string &sourceGuid, const std::string &sourcePath, float scaleFactor,
                 bool importAnimations = true, int maxBonesPerVertex = 4, float minBoneWeight = 0.0f,
                 bool synthesizeMissingTangents = true, bool importBlendShapes = true, bool bakeAxisConversion = false);
    [[nodiscard]] static std::shared_ptr<InxSkinnedMesh> ImportSource(const std::string &sourceGuid,
                                                                      const std::string &sourcePath, float scaleFactor);
};

} // namespace infernux
