#pragma once

#include "InxSkinnedMesh.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace infernux
{

class SkinnedModelCache
{
  public:
    static SkinnedModelCache &Instance();

    [[nodiscard]] std::shared_ptr<InxSkinnedMesh> Load(const std::string &sourceGuid, const std::string &sourcePath);
    void Invalidate(const std::string &sourceGuid, const std::string &sourcePath);
    void Clear();

  private:
    SkinnedModelCache() = default;

    [[nodiscard]] std::shared_ptr<InxSkinnedMesh> ImportModel(const std::string &sourceGuid,
                                                              const std::string &sourcePath);

    std::unordered_map<std::string, std::shared_ptr<InxSkinnedMesh>> m_cache;
};

} // namespace infernux
