#pragma once

#include <function/resources/AssetRegistry/ArtifactFileReader.h>
#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>

namespace infernux::mesh_decode_storage
{

// Allocation strides cover the native record layouts on the supported hosts.
// The cooked total is independent of the importer's vector capacities.
template <typename T> inline constexpr size_t SlotBytes = sizeof(T);
template <> inline constexpr size_t SlotBytes<InxMesh> = 512;
template <> inline constexpr size_t SlotBytes<MeshGeometry> = 192;
template <> inline constexpr size_t SlotBytes<Vertex> = 128;
template <> inline constexpr size_t SlotBytes<SubMesh> = 96;
template <> inline constexpr size_t SlotBytes<MeshMorphTarget> = 160;
template <> inline constexpr size_t SlotBytes<MaterialSlotData> = 640;
template <> inline constexpr size_t SlotBytes<ImportedModelNode> = 128;
template <> inline constexpr size_t SlotBytes<glm::vec3> = 16;
template <> inline constexpr size_t SlotBytes<glm::mat4> = 64;
template <> inline constexpr size_t SlotBytes<std::string> = 48;
template <> inline constexpr size_t SlotBytes<InxSkinnedMesh> = 1024;
template <> inline constexpr size_t SlotBytes<SkinInfluence> = 32;
template <> inline constexpr size_t SlotBytes<SkinnedRuntimeNode> = 192;
template <> inline constexpr size_t SlotBytes<SkinnedRuntimeBone> = 128;
template <> inline constexpr size_t SlotBytes<SkinnedRuntimeAnimation> = 448;
template <> inline constexpr size_t SlotBytes<SkinnedRuntimeTrack> = 128;
template <> inline constexpr size_t SlotBytes<SkinnedRuntimeFloatCurve> = 80;
template <> inline constexpr size_t SlotBytes<SkinnedRuntimeEvent> = 112;
template <> inline constexpr size_t SlotBytes<HumanoidValidationIssue> = 128;
template <> inline constexpr size_t SlotBytes<std::pair<double, glm::vec3>> = 32;
template <> inline constexpr size_t SlotBytes<std::pair<double, glm::quat>> = 32;
template <> inline constexpr size_t SlotBytes<std::pair<double, float>> = 16;
template <> inline constexpr size_t SlotBytes<std::pair<std::string, int>> = 56;

class DecodeWorkspace
{
  public:
    explicit DecodeWorkspace(uint64_t capacity = UINT64_MAX) : m_capacity(capacity), m_remaining(capacity)
    {
    }

    template <typename T> void Add(uint64_t count = 1)
    {
        static_assert(SlotBytes<T> >= sizeof(T));
        Charge(count, SlotBytes<T>);
    }
    void AddText(size_t length)
    {
        Charge(length + 32, 1);
    }
    void AddText(std::string_view value)
    {
        AddText(value.size());
    }
    void AddNameTable(size_t count)
    {
        // A reserved table has at most two buckets per entry plus its empty table.
        // Include two bucket pointers and the linked node on MSVC.
        Charge(count, 128);
        Charge(1, 128);
    }
    template <typename Table> void ReserveNameTable(Table &table, size_t count)
    {
        AddNameTable(count);
        table.reserve(count);
    }
    uint64_t Bytes() const
    {
        return m_capacity - m_remaining;
    }

  private:
    void Charge(uint64_t count, uint64_t stride)
    {
        if (count > m_remaining / stride)
            throw std::invalid_argument("model decode exceeds its cooked workspace");
        m_remaining -= count * stride;
    }

    uint64_t m_capacity;
    uint64_t m_remaining;
};

template <uint32_t MaximumStringBytes> class Reader : public binary_codec::Reader<MaximumStringBytes>
{
  public:
    using Input = binary_codec::Reader<MaximumStringBytes>;
    Reader(Input input, uint64_t workspace) : Input(std::move(input)), m_workspace(workspace)
    {
    }

    std::string ReadString(bool allowEmpty = true)
    {
        return Input::ReadString(allowEmpty, [this](uint32_t size) { m_workspace.AddText(size); });
    }

    template <typename T> uint32_t ReadArrayCount(uint32_t maximum, size_t minimumElementBytes = 1)
    {
        const uint32_t count = Input::ReadCount(maximum, minimumElementBytes);
        m_workspace.Add<T>(count);
        return count;
    }

    DecodeWorkspace &Workspace()
    {
        return m_workspace;
    }

  private:
    DecodeWorkspace m_workspace;
};

inline void AddGeometry(DecodeWorkspace &work, size_t vertices, size_t indices, const std::vector<SubMesh> &subMeshes,
                        const std::vector<MeshMorphTarget> &morphTargets)
{
    work.Add<Vertex>(vertices);
    work.Add<uint32_t>(indices);
    work.Add<SubMesh>(subMeshes.size());
    for (const auto &subMesh : subMeshes)
        work.AddText(subMesh.name);
    work.Add<MeshMorphTarget>(morphTargets.size());
    for (const auto &target : morphTargets) {
        work.AddText(target.name);
        work.Add<glm::vec3>(target.positionDeltas.size() + target.normalDeltas.size() + target.tangentDeltas.size());
    }
}

inline void AddNames(DecodeWorkspace &work, const std::vector<std::string> &names)
{
    work.Add<std::string>(names.size());
    for (const auto &name : names)
        work.AddText(name);
}

inline void AddSourceLocal(DecodeWorkspace &work, size_t vertices, size_t indices,
                           const std::vector<SubMesh> &subMeshes, const std::vector<MeshMorphTarget> &morphTargets,
                           const std::vector<std::string> &nodeNames, size_t nodeCount)
{
    // SetModelData retains source geometry while building its model-space view.
    work.Add<InxMesh>();
    work.Add<MeshGeometry>(2);
    AddGeometry(work, vertices, indices, subMeshes, morphTargets);
    AddNames(work, nodeNames);
    work.Add<glm::mat4>(nodeCount);
    work.Add<int32_t>(vertices + nodeNames.size());
}

inline uint64_t Mesh(const InxMesh &mesh, const MeshGeometry &geometry, bool sourceLocal, std::string_view sourceHash)
{
    DecodeWorkspace work;
    work.Add<InxMesh>();
    work.Add<MeshGeometry>(2);
    work.AddText(sourceHash);
    work.AddText(mesh.GetName());
    AddNames(work, mesh.GetMaterialSlotNames());
    AddNames(work, mesh.GetNodeNames());
    work.Add<MaterialSlotData>(mesh.GetMaterialSlotData().size());
    for (const auto &material : mesh.GetMaterialSlotData()) {
        work.AddText(material.sourceId);
        work.AddText(material.materialGuid);
        for (const auto &guid : material.textureGuids)
            work.AddText(guid);
    }
    work.Add<ImportedModelNode>(mesh.GetModelNodes().size());
    for (const auto &node : mesh.GetModelNodes())
        work.AddText(node.name);
    AddGeometry(work, geometry.vertices.size(), geometry.indices.size(), geometry.subMeshes, geometry.morphTargets);
    work.AddNameTable(geometry.morphTargets.size());
    if (sourceLocal)
        AddSourceLocal(work, geometry.vertices.size(), geometry.indices.size(), geometry.subMeshes,
                       geometry.morphTargets, mesh.GetNodeNames(), mesh.GetModelNodes().size());
    work.Add<uint8_t>(mesh.GetNodeNames().size());
    return work.Bytes();
}

inline uint64_t Skin(const InxSkinnedMesh &mesh, std::string_view sourceHash, std::string_view definitionGuid,
                     std::string_view definitionId)
{
    DecodeWorkspace work;
    work.Add<InxSkinnedMesh>();
    work.AddText(sourceHash);
    AddGeometry(work, mesh.baseVertices.size(), mesh.indices.size(), mesh.subMeshes, mesh.morphTargets);
    work.Add<SkinInfluence>(mesh.baseVertices.size());
    work.Add<SkinnedRuntimeNode>(mesh.skeleton.nodes.size());
    work.AddNameTable(mesh.skeleton.nodes.size());
    for (const auto &node : mesh.skeleton.nodes) {
        work.AddText(node.name);
        work.AddText(node.name);
    }
    work.Add<SkinnedRuntimeBone>(mesh.skeleton.bones.size());
    work.AddNameTable(mesh.skeleton.bones.size());
    for (const auto &bone : mesh.skeleton.bones) {
        work.AddText(bone.name);
        work.AddText(bone.name);
    }
    work.Add<SkinnedRuntimeAnimation>(mesh.animations.size());
    work.AddNameTable(mesh.animations.size()); // display names
    work.AddNameTable(mesh.animations.size()); // persistent IDs
    for (const auto &animation : mesh.animations) {
        work.AddText(animation.name);
        work.AddText(animation.id);
        work.AddText(animation.name);
        if (!animation.id.empty())
            work.AddText(animation.id);
        work.Add<int>(mesh.skeleton.nodes.size());
        work.Add<SkinnedRuntimeTrack>(animation.tracks.size());
        for (const auto &track : animation.tracks) {
            work.Add<std::pair<double, glm::vec3>>(track.positions.size() + track.scales.size());
            work.Add<std::pair<double, glm::quat>>(track.rotations.size());
        }
        work.AddText(animation.rootMotionReferencePose);
        work.Add<std::pair<double, glm::vec3>>(animation.rootMotionPositions.size());
        work.Add<std::pair<double, glm::quat>>(animation.rootMotionRotations.size());
        work.Add<SkinnedRuntimeFloatCurve>(animation.curves.size());
        work.AddNameTable(animation.curves.size());
        for (const auto &curve : animation.curves) {
            work.AddText(curve.name);
            work.AddText(curve.name);
            work.Add<std::pair<double, float>>(curve.keys.size());
        }
        work.Add<SkinnedRuntimeEvent>(animation.events.size());
        for (const auto &event : animation.events) {
            work.AddText(event.function);
            work.AddText(event.stringArgument);
        }
        AddNames(work, animation.boneMask);
        work.AddNameTable(animation.boneMask.size());
        for (const auto &name : animation.boneMask)
            work.AddText(name);
    }
    work.AddText(definitionGuid);
    work.AddText(definitionId);
    work.Add<int>(mesh.exposedSkeletonNodeIndices.size());
    work.Add<std::pair<std::string, int>>(mesh.humanoid.bones.size());
    work.AddNameTable(mesh.humanoid.bones.size());
    for (const auto &[slot, node] : mesh.humanoid.bones) {
        (void)node;
        work.AddText(slot);
        work.AddText(slot);
    }
    work.Add<uint8_t>(mesh.skeleton.nodes.size()); // shared rig index validation flags
    work.Add<HumanoidValidationIssue>(mesh.humanoid.issues.size());
    for (const auto &issue : mesh.humanoid.issues) {
        work.AddText(issue.code);
        work.AddText(issue.bone);
        work.AddText(issue.detail);
    }
    return work.Bytes();
}
} // namespace infernux::mesh_decode_storage
