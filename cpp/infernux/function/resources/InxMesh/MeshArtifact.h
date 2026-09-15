#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace infernux
{

class InxMesh;

class MeshArtifact final
{
  public:
    /// Standalone static-mesh authoring payload, using the same binary layout.
    /// Asset identity is assigned by import, not copied from the original model.
    [[nodiscard]] static std::string SerializeSource(const InxMesh &mesh);
    [[nodiscard]] static std::shared_ptr<InxMesh> DeserializeSource(std::string_view bytes);
    [[nodiscard]] static bool HasCurrentHeader(std::string_view bytes) noexcept;
    [[nodiscard]] static std::string Serialize(const InxMesh &mesh, std::string_view sourceContentHash);
    [[nodiscard]] static std::shared_ptr<InxMesh> Deserialize(std::string_view bytes,
                                                              std::string_view expectedSourceContentHash);
};

} // namespace infernux
