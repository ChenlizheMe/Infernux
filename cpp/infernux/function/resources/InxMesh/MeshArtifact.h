#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{

class InxMesh;
class FileInput;

class MeshArtifact final
{
  public:
    /// Standalone static-mesh authoring payload, sharing the cooked mesh body.
    /// Asset identity is assigned by import, not copied from the original model.
    [[nodiscard]] static std::string SerializeSource(const InxMesh &mesh);
    [[nodiscard]] static std::shared_ptr<InxMesh> DeserializeSource(std::string_view bytes);
    [[nodiscard]] static bool HasCurrentHeader(std::string_view bytes) noexcept;
    [[nodiscard]] static uint64_t ReadWorkspace(FileInput &file);
    [[nodiscard]] static std::string Serialize(const InxMesh &mesh, std::string_view sourceContentHash);
    [[nodiscard]] static std::shared_ptr<InxMesh> Deserialize(std::string_view bytes,
                                                              std::string_view expectedSourceContentHash);
    [[nodiscard]] static std::shared_ptr<InxMesh> Read(FileInput &file, std::string_view expectedSourceContentHash,
                                                       std::vector<char> &block, uint64_t workspaceLimit = UINT64_MAX);
};

} // namespace infernux
