#pragma once

#include <function/renderer/InxRenderStruct.h>
#include <function/resources/InxMaterial/MaterialProperty.h>

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace infernux
{

class InxSkinnedMesh;
class InxMaterial;

/**
 * @brief A contiguous range within a shared vertex/index buffer.
 *
 * Each SubMesh maps to exactly one material slot on the MeshRenderer.
 * Multiple SubMeshes share the parent InxMesh's vertex and index arrays
 * so a single GPU buffer upload serves the entire model.
 */
struct SubMesh
{
    uint32_t indexStart = 0;  ///< First index in the parent index array
    uint32_t indexCount = 0;  ///< Number of indices (must be a multiple of 3)
    uint32_t vertexStart = 0; ///< First vertex in the parent vertex array (base-vertex offset)
    uint32_t vertexCount = 0; ///< Number of vertices referenced by this submesh

    uint32_t materialSlot = 0; ///< Material slot index (0-based, maps to MeshRenderer.materials[])
    uint32_t nodeGroup = 0;    ///< Source node group (meshes under the same DCC node share the same group)

    glm::vec3 boundsMin{0.0f}; ///< Local-space AABB minimum
    glm::vec3 boundsMax{0.0f}; ///< Local-space AABB maximum

    std::string name; ///< Optional submesh name (from DCC tool, e.g. "Body", "Glass")
};

/**
 * @brief Per-slot material data extracted from a model file (FBX/OBJ/GLTF).
 *
 * Stores the material properties that Assimp reads from the source file
 * so that default materials can be created with correct colours rather
 * than falling back to plain white.
 */
enum class ModelAlphaMode : uint32_t
{
    Opaque,
    Mask,
    Blend,
};

enum class ModelTexture : uint32_t
{
    BaseColor,
    Normal,
    Metallic,
    Roughness,
    Occlusion,
    Emission,
    Count
};
inline constexpr size_t ModelTextureCount = static_cast<size_t>(ModelTexture::Count);

enum class MeshCompression : uint32_t
{
    Off,
    Low,
    Medium,
    High,
};

/// One imported blend-shape target in the same vertex domain as its owning
/// MeshGeometry.  Values are deltas rather than replacement attributes so the
/// target remains valid when the model hierarchy is baked into the combined
/// preview geometry. Empty normal/tangent streams mean that the source target
/// did not author those channels; they are never synthesized silently.
struct MeshMorphTarget
{
    std::string name;
    float defaultWeight = 0.0f;
    std::vector<glm::vec3> positionDeltas;
    std::vector<glm::vec3> normalDeltas;
    std::vector<glm::vec3> tangentDeltas;
};

struct MaterialSlotData
{
    glm::vec4 baseColor{1.0f, 1.0f, 1.0f, 1.0f};     ///< Diffuse / albedo colour (RGBA)
    glm::vec4 emissionColor{0.0f, 0.0f, 0.0f, 0.0f}; ///< Emission colour (RGBA)
    float metallic = 0.0f;
    float smoothness = 0.5f;
    float opacity = 1.0f;
    ModelAlphaMode alphaMode = ModelAlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
    // Import-local material identity: kind + unique authored source name.
    // Empty for unnamed/ambiguous materials; never substitute a slot index.
    std::string sourceId;
    std::string materialGuid;                                ///< Optional importer-level external material binding.
    std::array<std::string, ModelTextureCount> textureGuids; ///< Imported identities, never source paths.
    /// Source UV set selected independently by each texture binding.  The
    /// current runtime vertex contract exposes UV0 and UV1; import rejects
    /// larger indices instead of silently sampling UV0.
    std::array<uint8_t, ModelTextureCount> textureUvSets{};
    std::array<MaterialTextureSampler, ModelTextureCount> textureSamplers{};
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    bool packedMetallicRoughness = false; ///< glTF: metallic B, roughness G; otherwise scalar R.
};

/// Source hierarchy in parent-before-child order, including transform-only
/// nodes. Indices describe this import, not persistent subresource identities.
/// Geometry is still stored in model space; consumers must not apply these
/// transforms a second time to the combined vertex buffer.
struct ImportedModelNode
{
    std::string name;
    int32_t parentIndex = -1;
    int32_t nodeGroup = -1;         ///< -1 for nodes without geometry
    bool visible = true;            ///< Imported renderer visibility after parent inheritance
    glm::mat4 localTransform{1.0f}; ///< In engine units, relative to parent
};

/// One immutable geometry generation retained by consumers while in flight.
struct MeshGeometry
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<SubMesh> subMeshes;
    std::vector<MeshMorphTarget> morphTargets;
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    // Counts survive removal of CPU staging arrays.  Render/culling metadata
    // remains authoritative after a non-readable mesh becomes GPU resident.
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
};

/**
 * @brief Runtime mesh asset — the loaded, GPU-ready representation of a 3D model.
 *
 * InxMesh is the engine's canonical mesh data container, analogous to
 * Unity's Mesh or UE5's UStaticMesh.  It stores all geometry for a model
 * file (.fbx, .obj, .gltf, …) as a single pair of vertex/index arrays
 * partitioned into SubMeshes.
 *
 * Design decisions:
 *   - **Single buffer, multiple submeshes** — minimises GPU buffer count
 *     and allows one vkCmdBindVertexBuffers per model regardless of
 *     how many material slots it uses.
 *   - **Immutable geometry generations** — source import and cooked artifact
 *     loading publish through SetData.
 * Draw commands retain the generation
 *     they reference, independently of later replacement or asset retirement.
 *
 * - **Vertex layout matches the engine's `Vertex` struct** — Assimp data is converted once during loading; no runtime
 * format conversion.
 *
 * Ownership: managed by AssetRegistry via shared_ptr<InxMesh>.
 * MeshRenderers hold AssetRef<InxMesh> resolved through the registry.
 */
class InxMesh
{
  public:
    InxMesh() = default;
    explicit InxMesh(const std::string &name) : m_name(name)
    {
    }

    // ── Identification ───────────────────────────────────────────────────

    [[nodiscard]] const std::string &GetName() const
    {
        return m_name;
    }
    void SetName(const std::string &name)
    {
        m_name = name;
    }

    [[nodiscard]] const std::string &GetGuid() const
    {
        return m_guid;
    }
    void SetGuid(const std::string &guid)
    {
        m_guid = guid;
    }

    [[nodiscard]] const std::string &GetFilePath() const
    {
        return m_filePath;
    }
    void SetFilePath(const std::string &path)
    {
        m_filePath = path;
    }

    // ── Geometry data ────────────────────────────────────────────────────

    [[nodiscard]] const std::vector<Vertex> &GetVertices() const
    {
        return m_geometry->vertices;
    }
    [[nodiscard]] const std::vector<uint32_t> &GetIndices() const
    {
        return m_geometry->indices;
    }

    [[nodiscard]] uint32_t GetVertexCount() const
    {
        return m_geometry->vertexCount;
    }
    [[nodiscard]] uint32_t GetIndexCount() const
    {
        return m_geometry->indexCount;
    }

    /// CPU Read/Write is an import contract, not a request to silently read
    /// GPU memory back.  Internal upload/cook consumers may use the staging
    /// generation until ReleaseCpuGeometry() is called.
    void SetCpuReadable(bool readable) noexcept
    {
        m_cpuReadable = readable;
    }
    [[nodiscard]] bool IsCpuReadable() const noexcept
    {
        return m_cpuReadable;
    }
    [[nodiscard]] bool HasCpuGeometry() const noexcept
    {
        return m_geometry->vertices.size() == m_geometry->vertexCount &&
               m_geometry->indices.size() == m_geometry->indexCount;
    }
    void RequireCpuReadable(std::string_view operation) const;
    /// Drop only recreatable CPU vertex/index/morph staging. Submesh ranges,
    /// bounds, hierarchy and counts remain available. Returns bytes released.
    size_t ReleaseCpuGeometry();

    [[nodiscard]] const std::vector<MeshMorphTarget> &GetMorphTargets() const noexcept
    {
        return m_geometry->morphTargets;
    }

    // ── SubMesh access ───────────────────────────────────────────────────

    [[nodiscard]] const std::vector<SubMesh> &GetSubMeshes() const
    {
        return m_geometry->subMeshes;
    }
    [[nodiscard]] uint32_t GetSubMeshCount() const
    {
        return static_cast<uint32_t>(m_geometry->subMeshes.size());
    }
    [[nodiscard]] const SubMesh &GetSubMesh(uint32_t index) const
    {
        return m_geometry->subMeshes.at(index);
    }

    // ── Bounds ───────────────────────────────────────────────────────────

    [[nodiscard]] const glm::vec3 &GetBoundsMin() const
    {
        return m_geometry->boundsMin;
    }
    [[nodiscard]] const glm::vec3 &GetBoundsMax() const
    {
        return m_geometry->boundsMax;
    }

    [[nodiscard]] std::shared_ptr<const MeshGeometry> GetGeometrySnapshot() const
    {
        return m_geometry;
    }

    [[nodiscard]] uint64_t GetGeneration() const noexcept
    {
        return m_generation;
    }

    void SetIndexFormat(MeshIndexFormat format) noexcept
    {
        m_indexFormat = format;
    }
    [[nodiscard]] MeshIndexFormat GetIndexFormat() const noexcept
    {
        return m_indexFormat;
    }
    void SetCompression(MeshCompression compression) noexcept
    {
        m_compression = compression;
    }
    [[nodiscard]] MeshCompression GetCompression() const noexcept
    {
        return m_compression;
    }

    // ── Material slot names (extracted from model file) ──────────────────

    [[nodiscard]] const std::vector<std::string> &GetMaterialSlotNames() const
    {
        return m_materialSlotNames;
    }
    [[nodiscard]] uint32_t GetMaterialSlotCount() const
    {
        return static_cast<uint32_t>(m_materialSlotNames.size());
    }

    // ── Material slot data (extracted from model file) ───────────────

    [[nodiscard]] const std::vector<MaterialSlotData> &GetMaterialSlotData() const
    {
        return m_materialSlotData;
    }
    void SetMaterialSlotData(std::vector<MaterialSlotData> data)
    {
        m_materialSlotData = std::move(data);
    }
    /// Create a detached material from one imported source slot. The renderer
    /// and editor extraction use the same conversion; this does not save an asset.
    [[nodiscard]] std::shared_ptr<InxMaterial> CreateMaterialCopy(uint32_t slot) const;
    [[nodiscard]] std::vector<std::string> GetModelNodePath(size_t index) const;
    [[nodiscard]] int32_t RequireModelNode(const std::vector<std::string> &path) const;
    // Detached compact local geometry for previews/tools. Scene renderers share
    // the source asset and persist its GUID + node path instead of this copy.
    [[nodiscard]] std::shared_ptr<InxMesh> CreateModelNodeCopy(const std::vector<std::string> &path) const;
    [[nodiscard]] bool MatchesMaterialCopy(uint32_t slot, const InxMaterial &material) const;

    // ── Node group metadata (for per-object hierarchy) ────────────────

    [[nodiscard]] const std::vector<std::string> &GetNodeNames() const
    {
        return m_nodeNames;
    }
    [[nodiscard]] uint32_t GetNodeGroupCount() const
    {
        return static_cast<uint32_t>(m_nodeNames.size());
    }
    void SetNodeNames(std::vector<std::string> names)
    {
        m_nodeNames = std::move(names);
    }

    [[nodiscard]] const std::vector<ImportedModelNode> &GetModelNodes() const noexcept
    {
        return m_modelNodes;
    }
    void SetModelNodes(std::vector<ImportedModelNode> nodes);

    /// Node-local source geometry is authoritative for imported models. The
    /// ordinary geometry snapshot remains a derived, merged model-space view.
    [[nodiscard]] std::shared_ptr<const MeshGeometry> GetModelSourceGeometry() const noexcept
    {
        return m_modelSourceGeometry;
    }
    void SetModelData(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<SubMesh> subMeshes,
                      std::vector<ImportedModelNode> nodes, std::vector<MeshMorphTarget> morphTargets = {});
    void ReplaceImportedContent(const InxMesh &source);

    // ── Builder API (called by MeshLoader during import) ─────────────────

    /**
     * @brief Set the mesh geometry and submesh layout.
     *
     * Takes ownership of the data via move.  Recomputes the overall AABB
     * from the vertex positions.
     */
    void SetData(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<SubMesh> subMeshes,
                 std::vector<MeshMorphTarget> morphTargets = {});

    /// Replace a vertex range and publish a new geometry generation. Recompute
    /// overall/submesh bounds; topology and other vertex attributes are caller-owned.
    /// This CPU operation does not imply a partial GPU upload.
    void UpdateVertexRange(size_t first, const std::vector<Vertex> &vertices);

    /**
     * @brief Set material slot names extracted from the model file.
     *
     * The i-th name corresponds to materialSlot i.  The MeshRenderer
     * inspector uses these names as labels.
     */
    void SetMaterialSlotNames(std::vector<std::string> names)
    {
        m_materialSlotNames = std::move(names);
    }

    void SetSkinnedData(std::shared_ptr<const InxSkinnedMesh> skinnedData);
    [[nodiscard]] const std::shared_ptr<const InxSkinnedMesh> &GetSkinnedData() const noexcept
    {
        return m_skinnedData;
    }
    [[nodiscard]] bool HasSkinnedData() const noexcept
    {
        return static_cast<bool>(m_skinnedData);
    }

    [[nodiscard]] size_t GetRuntimeMemoryBytes() const noexcept;

  private:
    std::string m_name;
    std::string m_guid;
    std::string m_filePath;

    std::shared_ptr<const MeshGeometry> m_geometry = std::make_shared<const MeshGeometry>();
    std::shared_ptr<const MeshGeometry> m_modelSourceGeometry;

    std::vector<std::string> m_materialSlotNames;
    std::vector<MaterialSlotData> m_materialSlotData;
    std::vector<std::string> m_nodeNames; ///< Node names indexed by nodeGroup
    std::vector<ImportedModelNode> m_modelNodes;
    std::shared_ptr<const InxSkinnedMesh> m_skinnedData;
    MeshIndexFormat m_indexFormat = MeshIndexFormat::Auto;
    MeshCompression m_compression = MeshCompression::Off;
    bool m_cpuReadable = true;
    uint64_t m_generation = 0;
};

} // namespace infernux
