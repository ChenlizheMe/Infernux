#pragma once

#include <core/types/InxFwdType.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <glm/mat4x4.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux
{
class AssetLoadTicket;
class InxMaterial;

bool RequirePreviewAsset(std::shared_ptr<AssetLoadTicket> &ticket, const std::string &guid, ResourceType type,
                         bool &progressed);

// Owner-thread consumers share the Registry's original, revision-qualified
// tickets. This object owns no executor, file reader or asset cache.
class PreviewAssetLoads
{
  public:
    bool Require(const std::string &guid, ResourceType type, bool &progressed);
    [[nodiscard]] bool IsCurrent() const;
    void Clear()
    {
        m_tickets.clear();
    }

  private:
    std::unordered_map<std::string, std::shared_ptr<AssetLoadTicket>> m_tickets;
};

bool TryPrepareModelPreviewMaterials(const InxMesh &mesh, const std::vector<SubMesh> &subMeshes,
                                     PreviewAssetLoads &loads, std::vector<std::shared_ptr<InxMaterial>> &materials,
                                     bool &progressed);
std::shared_ptr<InxMaterial> GetModelPreviewMaterial(const InxMesh *mesh, uint32_t slot);

// Parsing produces values only. The owner fills imported geometry snapshots;
// the worker then reads immutable geometry, never a live Registry/Scene/Mesh.
struct PrefabPreviewPart
{
    glm::mat4 world{1.0f};
    glm::vec3 pivot{0.0f};
    std::string meshGuid;
    std::vector<std::string> materials;
    int32_t submesh = -1, nodeGroup = -1;
    std::shared_ptr<const MeshGeometry> geometry;

    [[nodiscard]] bool Includes(size_t index, const SubMesh &candidate) const noexcept
    {
        return candidate.indexCount && (submesh < 0 || static_cast<size_t>(submesh) == index) &&
               (nodeGroup < 0 || static_cast<uint32_t>(nodeGroup) == candidate.nodeGroup);
    }
};

struct PrefabPreviewLayout
{
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t subMeshCount = 0;
    [[nodiscard]] uint64_t StorageBytes() const noexcept;
};

bool IsPrefabPreviewPath(const std::string &path);
std::vector<PrefabPreviewPart> ParsePrefabPreview(std::string_view source);
PrefabPreviewLayout MeasurePrefabPreviewGeometry(const std::vector<PrefabPreviewPart> &parts);
std::shared_ptr<InxMesh> BuildPrefabPreviewGeometry(const std::vector<PrefabPreviewPart> &parts,
                                                    const PrefabPreviewLayout &layout);
} // namespace infernux
