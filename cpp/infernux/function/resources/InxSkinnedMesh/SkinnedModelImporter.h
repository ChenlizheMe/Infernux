#pragma once

#include <memory>
#include <string>

struct aiScene;

namespace infernux
{

class InxSkinnedMesh;

class SkinnedModelImporter final
{
  public:
    [[nodiscard]] static bool HasSkinningData(const aiScene &scene, bool includeAnimations = true) noexcept;
    [[nodiscard]] static std::shared_ptr<InxSkinnedMesh> ConvertScene(const aiScene &scene,
                                                                      const std::string &sourceGuid,
                                                                      const std::string &sourcePath, float scaleFactor,
                                                                      bool importAnimations = true,
                                                                      int maxBonesPerVertex = 4,
                                                                      float minBoneWeight = 0.0f);
    [[nodiscard]] static std::shared_ptr<InxSkinnedMesh> ImportSource(const std::string &sourceGuid,
                                                                      const std::string &sourcePath, float scaleFactor);
};

} // namespace infernux
