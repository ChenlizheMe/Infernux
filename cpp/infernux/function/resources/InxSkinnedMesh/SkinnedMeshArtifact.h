#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{

class InxSkinnedMesh;
class FileInput;

class SkinnedMeshArtifact final
{
  public:
    [[nodiscard]] static bool HasCurrentHeader(std::string_view bytes) noexcept;
    [[nodiscard]] static uint64_t ReadWorkspace(FileInput &file);
    [[nodiscard]] static std::string Serialize(const InxSkinnedMesh &mesh, std::string_view sourceContentHash);
    [[nodiscard]] static std::string SerializeEmpty(std::string_view sourceContentHash);
    [[nodiscard]] static std::shared_ptr<InxSkinnedMesh> Deserialize(std::string_view bytes,
                                                                     std::string_view expectedSourceContentHash);
    [[nodiscard]] static std::shared_ptr<InxSkinnedMesh> Read(FileInput &file,
                                                              std::string_view expectedSourceContentHash,
                                                              std::vector<char> &block,
                                                              uint64_t workspaceLimit = UINT64_MAX);
};

} // namespace infernux
