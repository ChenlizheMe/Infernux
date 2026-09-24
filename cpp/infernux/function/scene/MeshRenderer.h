#pragma once

#include "Component.h"
#include "function/renderer/InxRenderStruct.h"
#include "function/renderer/RendererParameterBlock.h"
#include <cstdint>
#include <function/resources/AssetDependencyGraph.h>
#include <function/resources/AssetRef.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

namespace rhi
{
class ComputeBuffer;
}

/**
 * @brief Reference to a mesh resource for rendering.
 *
 * This is a lightweight reference to mesh data stored elsewhere.
 * The actual vertex/index data is managed by the resource system.
 */
struct MeshRef
{
    uint64_t meshId = 0; // Resource ID for the mesh

    bool IsValid() const
    {
        return meshId != 0;
    }
};

/// Derive standard rendering attributes from authored triangle topology.
/// Equal positions are not welded; UV/hard-edge splits remain authoritative.
void RecalculateMeshNormals(std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices);
void RecalculateMeshTangents(std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices);

/**
 * @brief MeshRenderer component for rendering 3D meshes.
 *
 * Attach to a GameObject to make it render a mesh.
 * The renderer uses the Transform of the GameObject for positioning.
 *
 * Material reference uses a single AssetRef<InxMaterial> identified by GUID.
 * Serialization writes only "materialGuid"; deserialization resolves via
 * AssetRegistry to obtain the live InxMaterial pointer.
 */
class MeshRenderer : public Component
{
  public:
    [[nodiscard]] static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.allowMultiple = false;
        constraints.exclusiveGroups = {"renderer-owner"};
        return constraints;
    }

    MeshRenderer() = default;
    ~MeshRenderer() override;

    [[nodiscard]] const char *GetTypeName() const override
    {
        return "MeshRenderer";
    }

    // ========================================================================
    // Lifecycle — register/unregister with SceneManager component registry
    // ========================================================================

    void OnEnable() override;
    void OnDisable() override;

    // ========================================================================
    // Mesh
    // ========================================================================

    [[nodiscard]] const MeshRef &GetMesh() const
    {
        return m_mesh;
    }
    void SetMesh(const MeshRef &mesh)
    {
        m_mesh = mesh;
    }
    void SetMesh(uint64_t meshId)
    {
        m_mesh.meshId = meshId;
        m_useInlineMesh = false;
    }

    /// @brief Set mesh from inline vertex/index data (for primitives)
    void SetMesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices);

    /// Replace visual geometry without automatically recooking colliders.
    /// Existing drawable renderers use the render-content invalidation path.
    void SetProceduralMesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices);

    /// Rebuild CPU-resident inline attributes. Resident compute meshes derive
    /// these attributes on the GPU instead of forcing a readback.
    void RecalculateInlineNormals();
    void RecalculateInlineTangents();
    void RecalculateInlineBounds();

    /// @brief Set mesh from shared static primitive data (zero-copy).
    /// The referenced vectors must outlive this MeshRenderer.
    void SetSharedPrimitiveMesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices,
                                const std::string &primitiveName);

    /// @brief Get/set the display name for inline (primitive) meshes.
    [[nodiscard]] const std::string &GetInlineMeshName() const
    {
        return m_inlineMeshName;
    }
    void SetInlineMeshName(const std::string &name)
    {
        m_inlineMeshName = name;
    }

    // ========================================================================
    // Mesh asset reference (for model-file meshes managed by AssetRegistry)
    // ========================================================================

    /// @brief Set mesh from asset GUID + pre-resolved pointer
    void SetMeshAsset(const std::string &guid, std::shared_ptr<InxMesh> mesh);

    /// @brief Set mesh from asset GUID only (resolution deferred)
    void SetMeshAssetGuid(const std::string &guid);

    /// @brief Clear the asset-managed mesh reference.
    void ClearMeshAsset();

    /// @brief Handle asset graph notifications for the referenced mesh asset.
    void OnMeshAssetEvent(AssetEvent event);

    /// @brief Invalidate or reconnect material slots without changing GUIDs.
    void OnMaterialAssetEvent(const std::string &guid, AssetEvent event);

    /// Republish an effective renderer-local texture after its asset generation changes.
    void OnParameterTextureAssetEvent(const std::string &guid, AssetEvent event);

    /// @brief Get the mesh asset reference
    [[nodiscard]] const AssetRef<InxMesh> &GetMeshAssetRef() const
    {
        return m_meshAsset;
    }

    /// @brief Get the mesh asset GUID (empty if using inline or no asset)
    [[nodiscard]] const std::string &GetMeshAssetGuid() const
    {
        return m_meshAsset.GetGuid();
    }

    /// @brief Check if this renderer uses an asset-managed mesh
    [[nodiscard]] bool HasMeshAsset() const
    {
        return m_meshAsset.HasGuid();
    }

    /// @brief Mark the mesh GPU buffer as needing re-upload (after asset reload)
    void MarkMeshBufferDirty()
    {
        m_meshBufferDirty = true;
    }

    /// @brief Check and consume the dirty flag
    [[nodiscard]] bool ConsumeMeshBufferDirty();

    /// @brief Check if this renderer uses inline mesh data
    [[nodiscard]] bool HasInlineMesh() const
    {
        return m_useInlineMesh;
    }

    /// Shared inline meshes are immutable engine-owned primitive streams.
    /// Their pointer identity is stable for the process lifetime, so render
    /// extraction can retain the stream directly instead of snapshotting the
    /// same Cube/Quad data once per GameObject.
    [[nodiscard]] bool HasSharedInlineMesh() const
    {
        return m_useInlineMesh && m_sharedVertices != nullptr && m_sharedIndices != nullptr;
    }

    /// Stable cache identity for an engine-owned inline primitive.
    [[nodiscard]] std::string GetSharedInlineMeshGuid() const
    {
        return HasSharedInlineMesh() && !m_inlineMeshName.empty() ? "builtin-mesh:" + m_inlineMeshName : "";
    }

    /// @brief Get inline vertex data
    [[nodiscard]] const std::vector<Vertex> &GetInlineVertices() const
    {
        return m_sharedVertices ? *m_sharedVertices : m_inlineVertices;
    }

    /// @brief Get inline index data
    [[nodiscard]] const std::vector<uint32_t> &GetInlineIndices() const
    {
        return m_sharedIndices ? *m_sharedIndices : m_inlineIndices;
    }

    /// Monotonic identity for runtime geometry publication. Rendering uses
    /// this instead of hashing the complete vertex/index payload.
    [[nodiscard]] uint64_t GetInlineMeshVersion() const noexcept
    {
        return m_inlineMeshVersion;
    }

    /// Bind canonical interleaved Vertex storage owned by inx.buffer. The
    /// authored mesh continues to own topology and all initial attributes;
    /// compute may update the resident stream in place without CPU readback.
    void SetVertexBuffer(std::shared_ptr<rhi::ComputeBuffer> buffer, const glm::vec3 &boundsMin,
                         const glm::vec3 &boundsMax, bool worldSpace = false);
    void ClearVertexBuffer();
    [[nodiscard]] const std::shared_ptr<rhi::ComputeBuffer> &GetVertexBuffer() const noexcept
    {
        return m_vertexBuffer;
    }
    [[nodiscard]] bool HasVertexBuffer() const noexcept
    {
        return static_cast<bool>(m_vertexBuffer);
    }
    [[nodiscard]] bool IsVertexBufferWorldSpace() const noexcept
    {
        return m_vertexBuffer && m_vertexBufferWorldSpace;
    }
    /// Allocated canonical-Vertex slots in the resident stream.  The effective
    /// draw count remains the authored mesh's vertex count; spare slots allow
    /// compute workloads to keep one allocation while their active range
    /// changes within an explicitly chosen capacity.
    [[nodiscard]] size_t GetVertexBufferCapacity() const noexcept;

    // ========================================================================
    // Materials (multi-slot, submesh-indexed)
    // ========================================================================

    /// @brief Get the number of material slots.
    [[nodiscard]] uint32_t GetMaterialCount() const
    {
        return static_cast<uint32_t>(m_materials.size());
    }

    /// @brief Get material on a specific slot (nullptr if not assigned).
    [[nodiscard]] std::shared_ptr<InxMaterial> GetMaterial(uint32_t slot = 0) const;

    /// @brief Get the effective material for a slot (returns default if none).
    [[nodiscard]] virtual std::shared_ptr<InxMaterial> GetEffectiveMaterial(uint32_t slot = 0) const;

    /// @brief Get the GUID of the material at a specific slot.
    [[nodiscard]] std::string GetMaterialGuid(uint32_t slot = 0) const;

    /// @brief Get all material GUIDs.
    [[nodiscard]] std::vector<std::string> GetMaterialGuids() const;

    /// @brief Set a material on a specific slot by GUID (resolution deferred).
    void SetMaterial(uint32_t slot, const std::string &guid);

    /// @brief Set a material on a specific slot by pointer.
    void SetMaterial(uint32_t slot, std::shared_ptr<InxMaterial> material);

    /// @brief Bulk-set all materials from GUID list.
    void SetMaterials(const std::vector<std::string> &guids);

    /// @brief Resize the material slot array (new slots get empty refs).
    void SetMaterialSlotCount(uint32_t count);

    /// @brief Get all material AssetRefs.
    [[nodiscard]] const std::vector<AssetRef<InxMaterial>> &GetMaterialRefs() const
    {
        return m_materials;
    }

    /// @brief Synchronize material slot count to match mesh submesh count.
    void SyncMaterialSlotsToMesh();

    // ========================================================================
    // Per-renderer material parameters
    // ========================================================================

    /// Publish one parameter override for a material slot. The value type must
    /// exactly match the shader-reflected material property type. Persistent
    /// values are serialized; runtime values live only for this component
    /// lifetime and take precedence over persistent values.
    void SetParameter(uint32_t slot, const std::string &name, MaterialPropertyValue value, bool persistent = false,
                      const std::string &owner = "script");
    [[nodiscard]] const MaterialProperty *GetParameter(uint32_t slot, const std::string &name,
                                                       bool persistentOnly = false,
                                                       const std::string &owner = "") const;
    [[nodiscard]] bool RemoveParameter(uint32_t slot, const std::string &name, bool persistent = false,
                                       const std::string &owner = "script");
    void ClearParameters(uint32_t slot, bool persistent = false, const std::string &owner = "script");
    [[nodiscard]] std::shared_ptr<const RendererParameterBlock> GetParameterBlock(uint32_t slot = 0) const;

    // ========================================================================
    // Rendering flags
    // ========================================================================

    // ========================================================================
    // Submesh filtering
    // ========================================================================

    /// @brief Get the submesh index filter (-1 = render all submeshes).
    [[nodiscard]] int32_t GetSubmeshIndex() const
    {
        return m_submeshIndex;
    }

    /// @brief Set which submesh to render (-1 = all, >= 0 = specific submesh).
    void SetSubmeshIndex(int32_t index);

    /// @brief Get the mesh pivot offset (pre-transform to re-center submesh geometry).
    [[nodiscard]] const glm::vec3 &GetMeshPivotOffset() const
    {
        return m_meshPivotOffset;
    }

    /// @brief Set the mesh pivot offset (used to re-center submesh geometry around the transform).
    void SetMeshPivotOffset(const glm::vec3 &offset);

    /// @brief Get the node group filter (-1 = render all nodes, >= 0 = specific node group).
    [[nodiscard]] int32_t GetNodeGroup() const
    {
        return m_nodeGroup;
    }

    /// @brief Set which node group to render (-1 = all, >= 0 = specific node group).
    void SetNodeGroup(int32_t group);

    /// Imported hierarchy instances consume node-local geometry; ordinary mesh
    /// assignments retain the merged model-space view. Persisted with the binding.
    void SetModelNodePath(std::vector<std::string> path);
    [[nodiscard]] const std::vector<std::string> &GetModelNodePath() const noexcept
    {
        return m_modelNodePath;
    }
    /// Stable imported subresource identity used to reconcile DCC renames.
    void SetModelSubresourceId(std::string id)
    {
        m_modelSubresourceId = std::move(id);
    }
    [[nodiscard]] const std::string &GetModelSubresourceId() const noexcept
    {
        return m_modelSubresourceId;
    }
    [[nodiscard]] bool IsModelNodeLocal() const noexcept
    {
        return !m_modelNodePath.empty();
    }
    [[nodiscard]] std::shared_ptr<const MeshGeometry> GetAssetGeometry() const;

    [[nodiscard]] bool CastsShadows() const
    {
        return m_castShadows;
    }
    void SetCastShadows(bool cast)
    {
        m_castShadows = cast;
    }

    [[nodiscard]] bool ReceivesShadows() const
    {
        return m_receiveShadows;
    }
    void SetReceivesShadows(bool receive)
    {
        m_receiveShadows = receive;
    }

    // ========================================================================
    // Bounds (for culling)
    // ========================================================================

    /// @brief Get local-space bounding box (from mesh)
    [[nodiscard]] const glm::vec3 &GetLocalBoundsMin() const
    {
        return m_localBoundsMin;
    }
    [[nodiscard]] const glm::vec3 &GetLocalBoundsMax() const
    {
        return m_localBoundsMax;
    }

    /// @brief Set local bounds (usually from mesh loading)
    void SetLocalBounds(const glm::vec3 &min, const glm::vec3 &max)
    {
        m_localBoundsMin = min;
        m_localBoundsMax = max;
    }

    /// @brief Get world-space bounding box (transformed by GameObject)
    void GetWorldBounds(glm::vec3 &outMin, glm::vec3 &outMax) const;

    /// @brief Compute world bounds from a pre-computed world matrix (avoids double GetWorldMatrix)
    virtual void ComputeWorldBounds(const glm::mat4 &worldMatrix, glm::vec3 &outMin, glm::vec3 &outMax) const;

    /// @brief Resolve the matrix submitted to rendering for this renderer.
    /// Procedural renderers may override this when their authored geometry is
    /// already expressed in world space.
    [[nodiscard]] virtual glm::mat4 ResolveRenderWorldMatrix(const glm::mat4 &objectWorldMatrix) const
    {
        return IsVertexBufferWorldSpace() ? glm::mat4(1.0f) : objectWorldMatrix;
    }

    /// Resolve the transform applied to authored bounds. World-space resident
    /// vertices bypass the render transform, but their conservative bounds
    /// remain attached to the GameObject pose captured when the buffer was
    /// bound. This keeps culling, picking and editor tools on the same moving
    /// Transform anchor without reading vertex data back from the GPU.
    [[nodiscard]] virtual glm::mat4 ResolveBoundsWorldMatrix(const glm::mat4 &objectWorldMatrix) const
    {
        return IsVertexBufferWorldSpace() ? objectWorldMatrix * m_vertexBufferWorldBoundsAnchorInverse
                                          : objectWorldMatrix;
    }

    /// Refresh transform-dependent procedural data before render extraction.
    virtual void RefreshProceduralGeometry(const glm::mat4 &objectWorldMatrix)
    {
        (void)objectWorldMatrix;
    }

    /// @brief Recompute local bounds from inline vertex positions.
    void ComputeLocalBoundsFromInlineVertices();

    /// @brief Recompute local bounds for a specific node group.
    void UpdateBoundsForMeshSelection(const std::shared_ptr<InxMesh> &mesh);

    // ========================================================================
    // Serialization
    // ========================================================================

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    [[nodiscard]] std::unique_ptr<Component> Clone() const override;

  protected:
    static void ValidateSerializedDocumentForType(const nlohmann::json &document, std::string_view expectedType);

    /// Derived renderers may generate their inline mesh entirely from their
    /// authored fields and omit that cache from scene documents.
    [[nodiscard]] virtual bool ShouldSerializeInlineMeshData() const
    {
        return true;
    }

  private:
    /// Populate empty slots and refresh untouched imported defaults. Explicit
    /// assignments and authored material edits survive model reimport.
    void ApplyEmbeddedMaterialsFromMesh(const std::shared_ptr<InxMesh> &mesh);
    void ResolveModelNodeBinding();
    [[nodiscard]] bool IsUnmodifiedEmbeddedMaterial(size_t slot) const;

    MeshRef m_mesh;

    // Material slots — one per submesh, GUID-based, resolved via AssetRegistry
    struct RuntimeParameterEntry
    {
        MaterialProperty property;
        uint64_t writeRevision = 0;
    };
    using RuntimeParameterLayer = std::unordered_map<std::string, RuntimeParameterEntry>;
    using RuntimeParameterOwners = std::unordered_map<std::string, RuntimeParameterLayer>;

    std::vector<AssetRef<InxMaterial>> m_materials;
    // Imported defaults are derived data. An explicit assignment or a material
    // edit becomes an authored override; untouched defaults follow reimport.
    struct ImportedMaterialState
    {
        std::string guid;
        uint64_t authoredVersion = 0;
    };
    std::vector<std::optional<ImportedMaterialState>> m_embeddedMaterialVersions;
    std::vector<std::unordered_map<std::string, MaterialProperty>> m_persistentParameters;
    std::vector<RuntimeParameterOwners> m_runtimeParameters;
    std::vector<std::shared_ptr<const RendererParameterBlock>> m_parameterBlocks;
    std::unordered_set<std::string> m_parameterTextureDependencies;
    uint64_t m_runtimeParameterWriteRevision = 0;
    uint64_t m_parameterRevision = 0;

    void EnsureParameterSlot(uint32_t slot);
    void PublishParameterSlot(uint32_t slot);
    void RefreshParameterTextureDependencies();

    // Mesh asset reference (for model-file meshes managed by AssetRegistry)
    AssetRef<InxMesh> m_meshAsset;
    bool m_meshBufferDirty = false;

    // Inline mesh data (for primitives, not using resource system)
    std::vector<Vertex> m_inlineVertices;
    std::vector<uint32_t> m_inlineIndices;
    // Shared primitive mesh data (zero-copy pointer to static data)
    const std::vector<Vertex> *m_sharedVertices = nullptr;
    const std::vector<uint32_t> *m_sharedIndices = nullptr;
    bool m_useInlineMesh = false;
    std::string m_inlineMeshName; // display name for inline (primitive) meshes
    uint64_t m_inlineMeshVersion = 1;
    // Runtime-only GPU vertex storage is deliberately absent from scene
    // serialization and cloning. Scripts recreate it for each Play lifetime.
    std::shared_ptr<rhi::ComputeBuffer> m_vertexBuffer;
    bool m_vertexBufferWorldSpace = false;
    glm::mat4 m_vertexBufferWorldBoundsAnchorInverse{1.0f};

    int32_t m_submeshIndex = -1; // -1 = render all submeshes, >= 0 = single submesh
    int32_t m_nodeGroup = -1;    // -1 = render all node groups, >= 0 = specific node group
    std::vector<std::string> m_modelNodePath;
    std::string m_modelSubresourceId;
    glm::vec3 m_meshPivotOffset{0.0f}; // Pre-transform to re-center submesh geometry

    bool m_castShadows = true;
    bool m_receiveShadows = true;

    // Local-space bounding box
    glm::vec3 m_localBoundsMin{-0.5f};
    glm::vec3 m_localBoundsMax{0.5f};
};

} // namespace infernux
