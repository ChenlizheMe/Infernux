#include "SkinnedMeshArtifact.h"

#include "InxSkinnedMesh.h"
#include <function/resources/InxMesh/MeshGeometryCodec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace infernux
{
namespace
{
constexpr std::string_view Magic = "INXSKINAR";
constexpr uint32_t EndianMarker = 0x01020304U;
// Persistent schema identity. Obsolete Library artifacts are reimported; the
// runtime never branches into retired skinned-mesh layouts.
constexpr uint32_t ArtifactSchema = 0x314e4b53U; // SKN1
constexpr uint32_t MaximumVertices = 10'000'000U;
constexpr uint32_t MaximumIndices = 30'000'000U;
constexpr uint32_t MaximumObjects = 1'000'000U;
constexpr uint32_t MaximumKeys = 100'000'000U;
constexpr uint32_t MaximumStringBytes = 1024U * 1024U;
constexpr uint32_t MaximumHashBytes = 1024U;
constexpr uint64_t MaximumArtifactBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr uint32_t AnimationIdentities = 0x31444941U; // AID1
constexpr uint32_t AnimationRootMotion = 0x314d5241U; // ARM1
constexpr uint32_t AnimationExtras = 0x31584541U;     // AEX1
constexpr uint32_t MorphTargetsV1 = 0x3152504dU;      // MPR1
constexpr uint32_t RigDefinitionV2 = 0x32474952U;     // RIG2, includes resolved humanoid mapping/report
constexpr uint32_t GeometryEncoding = 0x334f4547U;    // GEO3, typed indices, compression and UV0/UV1

uint64_t Fnv1a64(std::string_view bytes)
{
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void AppendU32(std::string &output, uint32_t value)
{
    for (unsigned int shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void AppendU16(std::string &output, uint16_t value)
{
    output.push_back(static_cast<char>(value & 0xffU));
    output.push_back(static_cast<char>((value >> 8U) & 0xffU));
}

void AppendU64(std::string &output, uint64_t value)
{
    for (unsigned int shift = 0; shift < 64; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void AppendI32(std::string &output, int32_t value)
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU32(output, bits);
}

void AppendFloat(std::string &output, float value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("skinned Mesh artifact contains a non-finite float");
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU32(output, bits);
}

void AppendDouble(std::string &output, double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("skinned Mesh artifact contains a non-finite double");
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU64(output, bits);
}

void AppendString(std::string &output, std::string_view value, bool allowEmpty = true)
{
    if ((!allowEmpty && value.empty()) || value.size() > MaximumStringBytes)
        throw std::invalid_argument("skinned Mesh artifact contains an invalid string");
    AppendU32(output, static_cast<uint32_t>(value.size()));
    output.append(value);
}

void AppendCount(std::string &output, size_t count, uint32_t maximum)
{
    if (count > maximum)
        throw std::overflow_error("skinned Mesh artifact count exceeds its format limit");
    AppendU32(output, static_cast<uint32_t>(count));
}

void AppendVec2(std::string &output, const glm::vec2 &value)
{
    AppendFloat(output, value.x);
    AppendFloat(output, value.y);
}

void AppendVec3(std::string &output, const glm::vec3 &value)
{
    AppendFloat(output, value.x);
    AppendFloat(output, value.y);
    AppendFloat(output, value.z);
}

void AppendVec4(std::string &output, const glm::vec4 &value)
{
    AppendFloat(output, value.x);
    AppendFloat(output, value.y);
    AppendFloat(output, value.z);
    AppendFloat(output, value.w);
}

void AppendQuat(std::string &output, const glm::quat &value)
{
    AppendFloat(output, value.w);
    AppendFloat(output, value.x);
    AppendFloat(output, value.y);
    AppendFloat(output, value.z);
}

void AppendMat4(std::string &output, const glm::mat4 &value)
{
    for (glm::length_t column = 0; column < value.length(); ++column)
        AppendVec4(output, value[column]);
}

class Reader final
{
  public:
    explicit Reader(std::string_view bytes) : m_bytes(bytes)
    {
    }

    [[nodiscard]] uint32_t ReadU32()
    {
        Require(sizeof(uint32_t));
        uint32_t value = 0;
        for (unsigned int shift = 0; shift < 32; shift += 8)
            value |= static_cast<uint32_t>(static_cast<unsigned char>(m_bytes[m_cursor++])) << shift;
        return value;
    }

    [[nodiscard]] uint16_t ReadU16()
    {
        Require(sizeof(uint16_t));
        const uint16_t low = static_cast<unsigned char>(m_bytes[m_cursor++]);
        const uint16_t high = static_cast<unsigned char>(m_bytes[m_cursor++]);
        return static_cast<uint16_t>(low | (high << 8U));
    }

    [[nodiscard]] uint64_t ReadU64()
    {
        Require(sizeof(uint64_t));
        uint64_t value = 0;
        for (unsigned int shift = 0; shift < 64; shift += 8)
            value |= static_cast<uint64_t>(static_cast<unsigned char>(m_bytes[m_cursor++])) << shift;
        return value;
    }

    [[nodiscard]] int32_t ReadI32()
    {
        const uint32_t bits = ReadU32();
        int32_t value = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    [[nodiscard]] float ReadFloat()
    {
        const uint32_t bits = ReadU32();
        float value = 0.0f;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value))
            throw std::invalid_argument("skinned Mesh artifact contains a non-finite float");
        return value;
    }

    [[nodiscard]] double ReadDouble()
    {
        const uint64_t bits = ReadU64();
        double value = 0.0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value))
            throw std::invalid_argument("skinned Mesh artifact contains a non-finite double");
        return value;
    }

    [[nodiscard]] std::string ReadString(bool allowEmpty = true)
    {
        const uint32_t size = ReadU32();
        if ((!allowEmpty && size == 0) || size > MaximumStringBytes)
            throw std::invalid_argument("skinned Mesh artifact contains an invalid string");
        Require(size);
        std::string value(m_bytes.substr(m_cursor, size));
        m_cursor += size;
        return value;
    }

    [[nodiscard]] std::string_view ReadBytes(size_t size)
    {
        Require(size);
        const auto value = m_bytes.substr(m_cursor, size);
        m_cursor += size;
        return value;
    }

    [[nodiscard]] uint32_t ReadCount(uint32_t maximum)
    {
        const uint32_t count = ReadU32();
        if (count > maximum)
            throw std::invalid_argument("skinned Mesh artifact count exceeds its format limit");
        return count;
    }

    [[nodiscard]] bool AtEnd() const noexcept
    {
        return m_cursor == m_bytes.size();
    }

  private:
    void Require(size_t size) const
    {
        if (m_cursor > m_bytes.size() || size > m_bytes.size() - m_cursor)
            throw std::invalid_argument("skinned Mesh artifact is truncated");
    }

    std::string_view m_bytes;
    size_t m_cursor = 0;
};

glm::vec2 ReadVec2(Reader &reader)
{
    return {reader.ReadFloat(), reader.ReadFloat()};
}

glm::vec3 ReadVec3(Reader &reader)
{
    return {reader.ReadFloat(), reader.ReadFloat(), reader.ReadFloat()};
}

glm::vec4 ReadVec4(Reader &reader)
{
    return {reader.ReadFloat(), reader.ReadFloat(), reader.ReadFloat(), reader.ReadFloat()};
}

glm::quat ReadQuat(Reader &reader)
{
    const float w = reader.ReadFloat();
    const float x = reader.ReadFloat();
    const float y = reader.ReadFloat();
    const float z = reader.ReadFloat();
    const glm::quat value(w, x, y, z);
    const float length = glm::length(value);
    if (!std::isfinite(length) || length <= std::numeric_limits<float>::epsilon())
        throw std::invalid_argument("skinned Mesh artifact contains an invalid quaternion");
    return glm::normalize(value);
}

glm::mat4 ReadMat4(Reader &reader)
{
    glm::mat4 value(1.0f);
    for (glm::length_t column = 0; column < value.length(); ++column)
        value[column] = ReadVec4(reader);
    return value;
}

void AppendVertex(std::string &output, const Vertex &vertex, bool secondaryUv)
{
    AppendVec3(output, vertex.pos);
    AppendVec3(output, vertex.normal);
    AppendVec4(output, vertex.tangent);
    AppendVec3(output, vertex.color);
    AppendVec2(output, vertex.texCoord);
    if (secondaryUv)
        AppendVec2(output, vertex.texCoord1);
    for (glm::length_t component = 0; component < vertex.boneIndices.length(); ++component)
        AppendU32(output, vertex.boneIndices[component]);
    AppendVec4(output, vertex.boneWeights);
}

Vertex ReadVertex(Reader &reader, bool secondaryUv)
{
    Vertex vertex;
    vertex.pos = ReadVec3(reader);
    vertex.normal = ReadVec3(reader);
    vertex.tangent = ReadVec4(reader);
    vertex.color = ReadVec3(reader);
    vertex.texCoord = ReadVec2(reader);
    if (secondaryUv)
        vertex.texCoord1 = ReadVec2(reader);
    for (glm::length_t component = 0; component < vertex.boneIndices.length(); ++component)
        vertex.boneIndices[component] = reader.ReadU32();
    vertex.boneWeights = ReadVec4(reader);
    return vertex;
}

template <typename Value, typename AppendValue>
void AppendKeys(std::string &output, const std::vector<std::pair<double, Value>> &keys, AppendValue appendValue)
{
    AppendCount(output, keys.size(), MaximumKeys);
    double previousTime = -std::numeric_limits<double>::infinity();
    for (const auto &[time, value] : keys) {
        if (time < previousTime)
            throw std::invalid_argument("skinned Mesh animation keys are not ordered by time");
        AppendDouble(output, time);
        appendValue(output, value);
        previousTime = time;
    }
}

template <typename Value, typename ReadValue>
std::vector<std::pair<double, Value>> ReadKeys(Reader &reader, ReadValue readValue)
{
    std::vector<std::pair<double, Value>> keys;
    const uint32_t count = reader.ReadCount(MaximumKeys);
    keys.reserve(count);
    double previousTime = -std::numeric_limits<double>::infinity();
    for (uint32_t index = 0; index < count; ++index) {
        const double time = reader.ReadDouble();
        if (time < previousTime)
            throw std::invalid_argument("skinned Mesh animation keys are not ordered by time");
        keys.push_back({time, readValue(reader)});
        previousTime = time;
    }
    return keys;
}

} // namespace

std::string SkinnedMeshArtifact::Serialize(const InxSkinnedMesh &mesh, std::string_view sourceContentHash)
{
    if (!mesh.IsAssetPayloadValid() || mesh.influences.size() != mesh.baseVertices.size())
        throw std::invalid_argument("cannot serialize invalid skinned Mesh data");
    if (sourceContentHash.empty() || sourceContentHash.size() > MaximumHashBytes)
        throw std::invalid_argument("skinned Mesh artifact requires a bounded source content hash");
    std::string bytes(Magic);
    AppendU32(bytes, EndianMarker);
    AppendU32(bytes, ArtifactSchema);
    AppendString(bytes, sourceContentHash, false);
    AppendU32(bytes, 1);
    AppendFloat(bytes, mesh.scaleFactor);

    const auto indexFormat = ResolveMeshIndexFormat(mesh.indexFormat, mesh.baseVertices.size(), mesh.indices);
    AppendU32(bytes, GeometryEncoding);
    AppendU32(bytes, indexFormat == MeshIndexFormat::UInt16 ? 16U : 32U);
    AppendU32(bytes, static_cast<uint32_t>(mesh.compression));
    AppendCount(bytes, mesh.baseVertices.size(), MaximumVertices);
    if (mesh.compression == MeshCompression::Off) {
        for (const Vertex &vertex : mesh.baseVertices)
            AppendVertex(bytes, vertex, true);
    } else {
        const std::string encoded = mesh_geometry_codec::Encode(mesh.baseVertices, mesh.compression, true);
        AppendCount(bytes, encoded.size(), MaximumArtifactBytes);
        bytes.append(encoded);
    }

    AppendCount(bytes, mesh.indices.size(), MaximumIndices);
    for (const uint32_t index : mesh.indices) {
        if (index >= mesh.baseVertices.size())
            throw std::invalid_argument("skinned Mesh contains an out-of-range index");
        if (indexFormat == MeshIndexFormat::UInt16)
            AppendU16(bytes, static_cast<uint16_t>(index));
        else
            AppendU32(bytes, index);
    }

    AppendCount(bytes, mesh.subMeshes.size(), MaximumObjects);
    for (const SubMesh &subMesh : mesh.subMeshes) {
        if (subMesh.indexStart > mesh.indices.size() || subMesh.indexCount > mesh.indices.size() - subMesh.indexStart ||
            subMesh.vertexStart > mesh.baseVertices.size() ||
            subMesh.vertexCount > mesh.baseVertices.size() - subMesh.vertexStart)
            throw std::invalid_argument("skinned Mesh contains an invalid submesh range");
        AppendU32(bytes, subMesh.indexStart);
        AppendU32(bytes, subMesh.indexCount);
        AppendU32(bytes, subMesh.vertexStart);
        AppendU32(bytes, subMesh.vertexCount);
        AppendU32(bytes, subMesh.materialSlot);
        AppendU32(bytes, subMesh.nodeGroup);
        AppendVec3(bytes, subMesh.boundsMin);
        AppendVec3(bytes, subMesh.boundsMax);
        AppendString(bytes, subMesh.name);
    }

    AppendCount(bytes, mesh.skeleton.nodes.size(), MaximumObjects);
    std::unordered_set<std::string> nodeNames;
    for (size_t index = 0; index < mesh.skeleton.nodes.size(); ++index) {
        const auto &node = mesh.skeleton.nodes[index];
        if (node.name.empty() || !nodeNames.insert(node.name).second || node.parent < -1 ||
            (node.parent >= 0 && static_cast<size_t>(node.parent) >= index))
            throw std::invalid_argument("skinned Mesh contains an invalid node hierarchy");
        AppendString(bytes, node.name, false);
        AppendI32(bytes, node.parent);
        AppendMat4(bytes, node.bindLocal);
    }

    AppendCount(bytes, mesh.skeleton.bones.size(), MaximumObjects);
    std::unordered_set<std::string> boneNames;
    for (const auto &bone : mesh.skeleton.bones) {
        if (bone.name.empty() || !boneNames.insert(bone.name).second || bone.nodeIndex < -1 ||
            (bone.nodeIndex >= 0 && static_cast<size_t>(bone.nodeIndex) >= mesh.skeleton.nodes.size()))
            throw std::invalid_argument("skinned Mesh contains an invalid bone");
        AppendString(bytes, bone.name, false);
        AppendI32(bytes, bone.nodeIndex);
        AppendMat4(bytes, bone.inverseBind);
    }
    for (size_t vertexIndex = 0; vertexIndex < mesh.baseVertices.size(); ++vertexIndex) {
        const Vertex &vertex = mesh.baseVertices[vertexIndex];
        const SkinInfluence &influence = mesh.influences[vertexIndex];
        for (glm::length_t component = 0; component < vertex.boneWeights.length(); ++component) {
            const float weight = vertex.boneWeights[component];
            if (weight < 0.0f || (weight > 0.0f && vertex.boneIndices[component] >= mesh.skeleton.bones.size()))
                throw std::invalid_argument("skinned Mesh contains an invalid bone influence");
            if (vertex.boneIndices[component] != influence.boneIndex[component] ||
                std::abs(weight - influence.weight[component]) > 1e-6f)
                throw std::invalid_argument("skinned Mesh vertex and influence streams disagree");
        }
    }

    AppendCount(bytes, mesh.animations.size(), MaximumObjects);
    std::unordered_set<std::string> animationNames;
    for (const auto &animation : mesh.animations) {
        if (animation.name.empty() || !animationNames.insert(animation.name).second || animation.durationTicks < 0.0 ||
            animation.ticksPerSecond <= 0.0)
            throw std::invalid_argument("skinned Mesh contains an invalid animation");
        AppendString(bytes, animation.name, false);
        AppendDouble(bytes, animation.durationTicks);
        AppendDouble(bytes, animation.ticksPerSecond);
        AppendCount(bytes, animation.tracks.size(), MaximumObjects);
        std::unordered_set<int> trackNodes;
        for (const auto &track : animation.tracks) {
            if (track.nodeIndex < 0 || static_cast<size_t>(track.nodeIndex) >= mesh.skeleton.nodes.size() ||
                !trackNodes.insert(track.nodeIndex).second)
                throw std::invalid_argument("skinned Mesh contains an invalid animation track");
            AppendI32(bytes, track.nodeIndex);
            AppendKeys(bytes, track.positions,
                       [](std::string &output, const glm::vec3 &value) { AppendVec3(output, value); });
            AppendKeys(bytes, track.rotations,
                       [](std::string &output, const glm::quat &value) { AppendQuat(output, value); });
            AppendKeys(bytes, track.scales,
                       [](std::string &output, const glm::vec3 &value) { AppendVec3(output, value); });
        }
    }
    // AID1 extends the existing payload with stable animation identities.
    AppendU32(bytes, AnimationIdentities);
    AppendCount(bytes, mesh.animations.size(), MaximumObjects);
    std::unordered_set<std::string> clipIds;
    for (const auto &animation : mesh.animations) {
        if (!animation.id.empty() && !clipIds.insert(animation.id).second)
            throw std::invalid_argument("skinned Mesh contains duplicate animation ids");
        AppendString(bytes, animation.id);
    }
    // ARM1 stores import-time playback semantics and the extracted root-motion stream.
    AppendU32(bytes, AnimationRootMotion);
    AppendCount(bytes, mesh.animations.size(), MaximumObjects);
    for (const auto &animation : mesh.animations) {
        if (animation.rootMotionNodeIndex < -1 ||
            (animation.rootMotionNodeIndex >= 0 &&
             static_cast<size_t>(animation.rootMotionNodeIndex) >= mesh.skeleton.nodes.size()) ||
            (animation.rootMotionReferencePose != "bind_pose" && animation.rootMotionReferencePose != "first_frame"))
            throw std::invalid_argument("skinned Mesh contains invalid root-motion settings");
        AppendU32(bytes, animation.defaultLoop ? 1U : 0U);
        AppendI32(bytes, animation.rootMotionNodeIndex);
        AppendString(bytes, animation.rootMotionReferencePose, false);
        AppendKeys(bytes, animation.rootMotionPositions,
                   [](std::string &output, const glm::vec3 &value) { AppendVec3(output, value); });
        AppendKeys(bytes, animation.rootMotionRotations,
                   [](std::string &output, const glm::quat &value) { AppendQuat(output, value); });
    }
    // AEX1 persists import-authored float curves, events and exact-node masks.
    AppendU32(bytes, AnimationExtras);
    AppendCount(bytes, mesh.animations.size(), MaximumObjects);
    for (const auto &animation : mesh.animations) {
        AppendCount(bytes, animation.curves.size(), MaximumObjects);
        std::unordered_set<std::string> curveNames;
        for (const auto &curve : animation.curves) {
            if (curve.name.empty() || !curveNames.insert(curve.name).second ||
                (!curve.keys.empty() && (curve.keys.front().first < 0.0 || curve.keys.back().first > 1.0)))
                throw std::invalid_argument("skinned Mesh contains invalid animation curves");
            AppendString(bytes, curve.name, false);
            AppendKeys(bytes, curve.keys, [](std::string &output, float value) { AppendFloat(output, value); });
        }
        AppendCount(bytes, animation.events.size(), MaximumObjects);
        double previousEvent = -1.0;
        for (const auto &event : animation.events) {
            if (!std::isfinite(event.normalizedTime) || event.normalizedTime < previousEvent ||
                event.normalizedTime < 0.0 || event.normalizedTime > 1.0 || event.function.empty() ||
                !std::isfinite(event.numberArgument))
                throw std::invalid_argument("skinned Mesh contains invalid animation events");
            previousEvent = event.normalizedTime;
            AppendDouble(bytes, event.normalizedTime);
            AppendString(bytes, event.function, false);
            AppendString(bytes, event.stringArgument);
            AppendDouble(bytes, event.numberArgument);
        }
        AppendCount(bytes, animation.boneMask.size(), MaximumObjects);
        std::unordered_set<std::string> maskedBones;
        for (const auto &bone : animation.boneMask) {
            if (bone.empty() || mesh.skeleton.nodeByName.find(bone) == mesh.skeleton.nodeByName.end() ||
                !maskedBones.insert(bone).second)
                throw std::invalid_argument("skinned Mesh contains an invalid animation bone mask");
            AppendString(bytes, bone, false);
        }
    }
    AppendU32(bytes, MorphTargetsV1);
    AppendCount(bytes, mesh.morphTargets.size(), MaximumObjects);
    for (const auto &target : mesh.morphTargets) {
        AppendString(bytes, target.name, false);
        AppendFloat(bytes, target.defaultWeight);
        const auto appendDeltas = [&](const std::vector<glm::vec3> &values) {
            AppendCount(bytes, values.size(), MaximumVertices);
            for (const auto &value : values)
                AppendVec3(bytes, value);
        };
        appendDeltas(target.positionDeltas);
        appendDeltas(target.normalDeltas);
        appendDeltas(target.tangentDeltas);
    }
    // Definition identity and exposed attachment nodes are part of the cooked
    // rig payload.  GUID + local ID, never a source path, is the runtime key.
    // Programmatic assets have no import settings document. They still obtain
    // a deterministic in-memory definition from their resource GUID (or the
    // checked source hash for test-only transient assets); imported assets are
    // always populated by ApplyRigSettings above.
    const std::string_view definitionGuid = mesh.skeletonDefinitionGuid.empty()
                                                ? (mesh.guid.empty() ? sourceContentHash : std::string_view(mesh.guid))
                                                : std::string_view(mesh.skeletonDefinitionGuid);
    const std::string_view definitionId =
        mesh.skeletonDefinitionId.empty() ? "skeleton" : std::string_view(mesh.skeletonDefinitionId);
    const int rootNodeIndex = mesh.skeletonRootNodeIndex < 0 ? 0 : mesh.skeletonRootNodeIndex;
    if (definitionGuid.empty() || definitionId.empty() || rootNodeIndex < 0 ||
        static_cast<size_t>(rootNodeIndex) >= mesh.skeleton.nodes.size())
        throw std::invalid_argument("skinned Mesh contains an invalid skeleton definition");
    AppendU32(bytes, RigDefinitionV2);
    AppendString(bytes, definitionGuid, false);
    AppendString(bytes, definitionId, false);
    AppendI32(bytes, rootNodeIndex);
    AppendCount(bytes, mesh.exposedSkeletonNodeIndices.size(), MaximumObjects);
    std::unordered_set<int> exposedNodes;
    for (const int nodeIndex : mesh.exposedSkeletonNodeIndices) {
        if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= mesh.skeleton.nodes.size() ||
            !exposedNodes.insert(nodeIndex).second)
            throw std::invalid_argument("skinned Mesh contains an invalid exposed skeleton node");
        AppendI32(bytes, nodeIndex);
    }
    AppendU32(bytes, mesh.humanoid.enabled ? 1U : 0U);
    AppendU32(bytes, mesh.humanoid.requiredBonesValid ? 1U : 0U);
    AppendU32(bytes, mesh.humanoid.hierarchyValid ? 1U : 0U);
    AppendU32(bytes, mesh.humanoid.referencePoseValid ? 1U : 0U);
    AppendCount(bytes, mesh.humanoid.bones.size(), MaximumObjects);
    std::unordered_set<std::string> humanoidSlots;
    std::unordered_set<int> humanoidNodes;
    for (const auto &[slot, nodeIndex] : mesh.humanoid.bones) {
        if (slot.empty() || nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= mesh.skeleton.nodes.size() ||
            !humanoidSlots.insert(slot).second || !humanoidNodes.insert(nodeIndex).second)
            throw std::invalid_argument("skinned Mesh contains an invalid humanoid mapping");
        AppendString(bytes, slot, false);
        AppendI32(bytes, nodeIndex);
    }
    AppendCount(bytes, mesh.humanoid.issues.size(), MaximumObjects);
    for (const auto &issue : mesh.humanoid.issues) {
        if (issue.code.empty())
            throw std::invalid_argument("skinned Mesh contains an invalid humanoid report");
        AppendString(bytes, issue.code, false);
        AppendString(bytes, issue.bone);
        AppendString(bytes, issue.detail);
    }
    if (bytes.size() > MaximumArtifactBytes - sizeof(uint64_t))
        throw std::overflow_error("skinned Mesh artifact exceeds its size limit");
    AppendU64(bytes, Fnv1a64(bytes));
    return bytes;
}

std::string SkinnedMeshArtifact::SerializeEmpty(std::string_view sourceContentHash)
{
    if (sourceContentHash.empty() || sourceContentHash.size() > MaximumHashBytes)
        throw std::invalid_argument("skinned Mesh artifact requires a bounded source content hash");
    std::string bytes(Magic);
    AppendU32(bytes, EndianMarker);
    AppendU32(bytes, ArtifactSchema);
    AppendString(bytes, sourceContentHash, false);
    AppendU32(bytes, 0);
    AppendU64(bytes, Fnv1a64(bytes));
    return bytes;
}

std::shared_ptr<InxSkinnedMesh> SkinnedMeshArtifact::Deserialize(std::string_view bytes,
                                                                 std::string_view expectedSourceContentHash)
{
    if (expectedSourceContentHash.empty() || expectedSourceContentHash.size() > MaximumHashBytes)
        throw std::invalid_argument("skinned Mesh artifact requires an expected source content hash");
    if (bytes.size() > MaximumArtifactBytes || bytes.size() < Magic.size() + sizeof(uint32_t) + sizeof(uint64_t) ||
        bytes.substr(0, Magic.size()) != Magic)
        throw std::invalid_argument("skinned Mesh artifact has an invalid header");
    const size_t checksumOffset = bytes.size() - sizeof(uint64_t);
    Reader checksum(bytes.substr(checksumOffset));
    if (checksum.ReadU64() != Fnv1a64(bytes.substr(0, checksumOffset)))
        throw std::invalid_argument("skinned Mesh artifact checksum mismatch");

    Reader reader(bytes.substr(Magic.size(), checksumOffset - Magic.size()));
    if (reader.ReadU32() != EndianMarker)
        throw std::invalid_argument("skinned Mesh artifact has an invalid endian marker");
    if (reader.ReadU32() != ArtifactSchema)
        throw std::invalid_argument("skinned Mesh artifact has an unsupported schema");
    const std::string sourceHash = reader.ReadString(false);
    if (sourceHash != expectedSourceContentHash)
        throw std::invalid_argument("skinned Mesh artifact does not match the imported source content");
    const uint32_t hasPayload = reader.ReadU32();
    if (hasPayload == 0) {
        if (!reader.AtEnd())
            throw std::invalid_argument("empty skinned Mesh artifact contains trailing data");
        return {};
    }
    if (hasPayload != 1)
        throw std::invalid_argument("skinned Mesh artifact has an invalid payload marker");

    auto mesh = std::make_shared<InxSkinnedMesh>();
    mesh->scaleFactor = reader.ReadFloat();
    if (mesh->scaleFactor <= 0.0f)
        throw std::invalid_argument("skinned Mesh artifact has an invalid scale factor");

    if (reader.ReadU32() != GeometryEncoding)
        throw std::invalid_argument("skinned Mesh artifact has an invalid geometry encoding");
    const uint32_t indexBits = reader.ReadU32();
    if (indexBits != 16U && indexBits != 32U)
        throw std::invalid_argument("skinned Mesh artifact contains an unsupported index encoding");
    mesh->indexFormat = indexBits == 16U ? MeshIndexFormat::UInt16 : MeshIndexFormat::UInt32;
    const uint32_t encodedCompression = reader.ReadU32();
    if (encodedCompression > static_cast<uint32_t>(MeshCompression::High))
        throw std::invalid_argument("skinned Mesh artifact contains unsupported geometry compression");
    mesh->compression = static_cast<MeshCompression>(encodedCompression);
    const uint32_t vertexCount = reader.ReadCount(MaximumVertices);
    if (mesh->compression == MeshCompression::Off) {
        mesh->baseVertices.reserve(vertexCount);
        for (uint32_t index = 0; index < vertexCount; ++index)
            mesh->baseVertices.push_back(ReadVertex(reader, true));
    } else {
        const uint32_t encodedSize = reader.ReadCount(static_cast<uint32_t>(MaximumArtifactBytes));
        mesh->baseVertices =
            mesh_geometry_codec::Decode(reader.ReadBytes(encodedSize), vertexCount, mesh->compression, true);
    }
    mesh->influences.reserve(vertexCount);
    for (const Vertex &vertex : mesh->baseVertices) {
        SkinInfluence influence;
        for (uint32_t component = 0; component < kMaxSkinInfluences; ++component) {
            influence.boneIndex[component] = vertex.boneIndices[component];
            influence.weight[component] = vertex.boneWeights[component];
        }
        mesh->influences.push_back(influence);
    }

    const uint32_t indexCount = reader.ReadCount(MaximumIndices);
    mesh->indices.reserve(indexCount);
    for (uint32_t index = 0; index < indexCount; ++index) {
        const uint32_t vertexIndex = mesh->indexFormat == MeshIndexFormat::UInt16 ? reader.ReadU16() : reader.ReadU32();
        if (vertexIndex >= mesh->baseVertices.size())
            throw std::invalid_argument("skinned Mesh artifact contains an out-of-range index");
        mesh->indices.push_back(vertexIndex);
    }

    mesh->subMeshes.resize(reader.ReadCount(MaximumObjects));
    for (SubMesh &subMesh : mesh->subMeshes) {
        subMesh.indexStart = reader.ReadU32();
        subMesh.indexCount = reader.ReadU32();
        subMesh.vertexStart = reader.ReadU32();
        subMesh.vertexCount = reader.ReadU32();
        subMesh.materialSlot = reader.ReadU32();
        subMesh.nodeGroup = reader.ReadU32();
        subMesh.boundsMin = ReadVec3(reader);
        subMesh.boundsMax = ReadVec3(reader);
        subMesh.name = reader.ReadString();
        if (subMesh.indexStart > mesh->indices.size() ||
            subMesh.indexCount > mesh->indices.size() - subMesh.indexStart ||
            subMesh.vertexStart > mesh->baseVertices.size() ||
            subMesh.vertexCount > mesh->baseVertices.size() - subMesh.vertexStart)
            throw std::invalid_argument("skinned Mesh artifact contains an invalid submesh range");
    }

    const uint32_t nodeCount = reader.ReadCount(MaximumObjects);
    mesh->skeleton.nodes.reserve(nodeCount);
    for (uint32_t index = 0; index < nodeCount; ++index) {
        SkinnedRuntimeNode node;
        node.name = reader.ReadString(false);
        node.parent = reader.ReadI32();
        node.bindLocal = ReadMat4(reader);
        if (node.parent < -1 || (node.parent >= 0 && static_cast<size_t>(node.parent) >= index) ||
            !mesh->skeleton.nodeByName.emplace(node.name, static_cast<int>(index)).second)
            throw std::invalid_argument("skinned Mesh artifact contains an invalid node hierarchy");
        node.bindGlobal = node.parent >= 0
                              ? mesh->skeleton.nodes[static_cast<size_t>(node.parent)].bindGlobal * node.bindLocal
                              : node.bindLocal;
        mesh->skeleton.nodes.push_back(std::move(node));
    }

    const uint32_t boneCount = reader.ReadCount(MaximumObjects);
    mesh->skeleton.bones.reserve(boneCount);
    for (uint32_t index = 0; index < boneCount; ++index) {
        SkinnedRuntimeBone bone;
        bone.name = reader.ReadString(false);
        bone.nodeIndex = reader.ReadI32();
        bone.inverseBind = ReadMat4(reader);
        if (bone.nodeIndex < -1 ||
            (bone.nodeIndex >= 0 && static_cast<size_t>(bone.nodeIndex) >= mesh->skeleton.nodes.size()) ||
            !mesh->skeleton.boneByName.emplace(bone.name, static_cast<uint32_t>(index)).second)
            throw std::invalid_argument("skinned Mesh artifact contains an invalid bone");
        mesh->skeleton.bones.push_back(std::move(bone));
    }
    for (const Vertex &vertex : mesh->baseVertices) {
        for (glm::length_t component = 0; component < vertex.boneWeights.length(); ++component) {
            if (vertex.boneWeights[component] < 0.0f ||
                (vertex.boneWeights[component] > 0.0f && vertex.boneIndices[component] >= mesh->skeleton.bones.size()))
                throw std::invalid_argument("skinned Mesh artifact contains an invalid bone influence");
        }
    }

    const uint32_t animationCount = reader.ReadCount(MaximumObjects);
    mesh->animations.reserve(animationCount);
    std::unordered_set<std::string> animationNames;
    for (uint32_t animationIndex = 0; animationIndex < animationCount; ++animationIndex) {
        SkinnedRuntimeAnimation animation;
        animation.name = reader.ReadString(false);
        animation.durationTicks = reader.ReadDouble();
        animation.ticksPerSecond = reader.ReadDouble();
        if (!animationNames.insert(animation.name).second || animation.durationTicks < 0.0 ||
            animation.ticksPerSecond <= 0.0)
            throw std::invalid_argument("skinned Mesh artifact contains an invalid animation timing");
        const uint32_t trackCount = reader.ReadCount(MaximumObjects);
        animation.tracks.reserve(trackCount);
        animation.trackByNodeIndex.assign(mesh->skeleton.nodes.size(), -1);
        for (uint32_t trackIndex = 0; trackIndex < trackCount; ++trackIndex) {
            SkinnedRuntimeTrack track;
            track.nodeIndex = reader.ReadI32();
            if (track.nodeIndex < 0 || static_cast<size_t>(track.nodeIndex) >= mesh->skeleton.nodes.size() ||
                animation.trackByNodeIndex[static_cast<size_t>(track.nodeIndex)] >= 0)
                throw std::invalid_argument("skinned Mesh artifact contains an invalid animation track");
            animation.trackByNodeIndex[static_cast<size_t>(track.nodeIndex)] = static_cast<int>(trackIndex);
            track.positions = ReadKeys<glm::vec3>(reader, [](Reader &input) { return ReadVec3(input); });
            track.rotations = ReadKeys<glm::quat>(reader, [](Reader &input) { return ReadQuat(input); });
            track.scales = ReadKeys<glm::vec3>(reader, [](Reader &input) { return ReadVec3(input); });
            animation.tracks.push_back(std::move(track));
        }
        mesh->animations.push_back(std::move(animation));
    }
    if (reader.ReadU32() != AnimationIdentities || reader.ReadCount(MaximumObjects) != animationCount)
        throw std::invalid_argument("skinned Mesh artifact has an invalid animation identity table");
    std::unordered_set<std::string> clipIds;
    for (auto &animation : mesh->animations) {
        animation.id = reader.ReadString();
        if (!animation.id.empty() && !clipIds.insert(animation.id).second)
            throw std::invalid_argument("skinned Mesh artifact contains duplicate animation ids");
    }

    if (reader.ReadU32() != AnimationRootMotion || reader.ReadCount(MaximumObjects) != animationCount)
        throw std::invalid_argument("skinned Mesh artifact has an invalid root-motion table");
    for (auto &animation : mesh->animations) {
        const uint32_t defaultLoop = reader.ReadU32();
        if (defaultLoop > 1U)
            throw std::invalid_argument("skinned Mesh artifact has an invalid loop setting");
        animation.defaultLoop = defaultLoop != 0U;
        animation.rootMotionNodeIndex = reader.ReadI32();
        animation.rootMotionReferencePose = reader.ReadString(false);
        animation.rootMotionPositions = ReadKeys<glm::vec3>(reader, [](Reader &input) { return ReadVec3(input); });
        animation.rootMotionRotations = ReadKeys<glm::quat>(reader, [](Reader &input) { return ReadQuat(input); });
        if (animation.rootMotionNodeIndex < -1 ||
            (animation.rootMotionNodeIndex >= 0 &&
             static_cast<size_t>(animation.rootMotionNodeIndex) >= mesh->skeleton.nodes.size()) ||
            (animation.rootMotionReferencePose != "bind_pose" && animation.rootMotionReferencePose != "first_frame") ||
            (animation.rootMotionNodeIndex < 0 &&
             (!animation.rootMotionPositions.empty() || !animation.rootMotionRotations.empty())))
            throw std::invalid_argument("skinned Mesh artifact contains invalid root-motion settings");
    }

    if (reader.ReadU32() != AnimationExtras || reader.ReadCount(MaximumObjects) != animationCount)
        throw std::invalid_argument("skinned Mesh artifact has an invalid animation extras table");
    for (auto &animation : mesh->animations) {
        const uint32_t curveCount = reader.ReadCount(MaximumObjects);
        std::unordered_set<std::string> curveNames;
        animation.curves.reserve(curveCount);
        for (uint32_t index = 0; index < curveCount; ++index) {
            SkinnedRuntimeFloatCurve curve;
            curve.name = reader.ReadString(false);
            curve.keys = ReadKeys<float>(reader, [](Reader &input) { return input.ReadFloat(); });
            if (!curveNames.insert(curve.name).second ||
                (!curve.keys.empty() && (curve.keys.front().first < 0.0 || curve.keys.back().first > 1.0)))
                throw std::invalid_argument("skinned Mesh artifact contains invalid animation curves");
            animation.curves.push_back(std::move(curve));
        }
        const uint32_t eventCount = reader.ReadCount(MaximumObjects);
        animation.events.reserve(eventCount);
        double previousEvent = -1.0;
        for (uint32_t index = 0; index < eventCount; ++index) {
            SkinnedRuntimeEvent event;
            event.normalizedTime = reader.ReadDouble();
            event.function = reader.ReadString(false);
            event.stringArgument = reader.ReadString();
            event.numberArgument = reader.ReadDouble();
            if (!std::isfinite(event.normalizedTime) || event.normalizedTime < previousEvent ||
                event.normalizedTime < 0.0 || event.normalizedTime > 1.0 || !std::isfinite(event.numberArgument))
                throw std::invalid_argument("skinned Mesh artifact contains invalid animation events");
            previousEvent = event.normalizedTime;
            animation.events.push_back(std::move(event));
        }
        const uint32_t maskCount = reader.ReadCount(MaximumObjects);
        std::unordered_set<std::string> maskedBones;
        animation.boneMask.reserve(maskCount);
        for (uint32_t index = 0; index < maskCount; ++index) {
            auto bone = reader.ReadString(false);
            if (mesh->skeleton.nodeByName.find(bone) == mesh->skeleton.nodeByName.end() ||
                !maskedBones.insert(bone).second)
                throw std::invalid_argument("skinned Mesh artifact contains an invalid animation bone mask");
            animation.boneMask.push_back(std::move(bone));
        }
    }

    if (reader.ReadU32() != MorphTargetsV1)
        throw std::invalid_argument("skinned Mesh artifact has an invalid morph-target table");
    const uint32_t morphCount = reader.ReadCount(MaximumObjects);
    mesh->morphTargets.reserve(morphCount);
    for (uint32_t morphIndex = 0; morphIndex < morphCount; ++morphIndex) {
        MeshMorphTarget target;
        target.name = reader.ReadString();
        target.defaultWeight = reader.ReadFloat();
        const auto readDeltas = [&](std::vector<glm::vec3> &values) {
            const uint32_t count = reader.ReadCount(MaximumVertices);
            values.reserve(count);
            for (uint32_t index = 0; index < count; ++index)
                values.push_back(ReadVec3(reader));
        };
        readDeltas(target.positionDeltas);
        readDeltas(target.normalDeltas);
        readDeltas(target.tangentDeltas);
        mesh->morphTargets.push_back(std::move(target));
    }

    if (reader.ReadU32() != RigDefinitionV2)
        throw std::invalid_argument("skinned Mesh artifact has an invalid rig definition table");
    mesh->skeletonDefinitionGuid = reader.ReadString(false);
    mesh->skeletonDefinitionId = reader.ReadString(false);
    mesh->skeletonRootNodeIndex = reader.ReadI32();
    if (mesh->skeletonRootNodeIndex < 0 ||
        static_cast<size_t>(mesh->skeletonRootNodeIndex) >= mesh->skeleton.nodes.size())
        throw std::invalid_argument("skinned Mesh artifact contains an invalid rig root");
    const uint32_t exposedCount = reader.ReadCount(MaximumObjects);
    std::unordered_set<int> exposedNodes;
    mesh->exposedSkeletonNodeIndices.reserve(exposedCount);
    for (uint32_t index = 0; index < exposedCount; ++index) {
        const int nodeIndex = reader.ReadI32();
        if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= mesh->skeleton.nodes.size() ||
            !exposedNodes.insert(nodeIndex).second)
            throw std::invalid_argument("skinned Mesh artifact contains an invalid exposed skeleton node");
        mesh->exposedSkeletonNodeIndices.push_back(nodeIndex);
    }
    const uint32_t humanoidEnabled = reader.ReadU32();
    const uint32_t requiredBonesValid = reader.ReadU32();
    const uint32_t hierarchyValid = reader.ReadU32();
    const uint32_t referencePoseValid = reader.ReadU32();
    if (humanoidEnabled > 1U || requiredBonesValid > 1U || hierarchyValid > 1U || referencePoseValid > 1U)
        throw std::invalid_argument("skinned Mesh artifact contains invalid humanoid validity flags");
    mesh->humanoid.enabled = humanoidEnabled != 0U;
    mesh->humanoid.requiredBonesValid = requiredBonesValid != 0U;
    mesh->humanoid.hierarchyValid = hierarchyValid != 0U;
    mesh->humanoid.referencePoseValid = referencePoseValid != 0U;
    const uint32_t humanoidBoneCount = reader.ReadCount(MaximumObjects);
    std::unordered_set<std::string> humanoidSlots;
    std::unordered_set<int> humanoidNodes;
    mesh->humanoid.bones.reserve(humanoidBoneCount);
    for (uint32_t index = 0; index < humanoidBoneCount; ++index) {
        auto slot = reader.ReadString(false);
        const int nodeIndex = reader.ReadI32();
        if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= mesh->skeleton.nodes.size() ||
            !humanoidSlots.insert(slot).second || !humanoidNodes.insert(nodeIndex).second)
            throw std::invalid_argument("skinned Mesh artifact contains an invalid humanoid mapping");
        mesh->humanoid.bones.emplace_back(std::move(slot), nodeIndex);
    }
    const uint32_t issueCount = reader.ReadCount(MaximumObjects);
    mesh->humanoid.issues.reserve(issueCount);
    for (uint32_t index = 0; index < issueCount; ++index) {
        HumanoidValidationIssue issue;
        issue.code = reader.ReadString(false);
        issue.bone = reader.ReadString();
        issue.detail = reader.ReadString();
        mesh->humanoid.issues.push_back(std::move(issue));
    }
    if (!reader.AtEnd())
        throw std::invalid_argument("skinned Mesh artifact contains trailing data");
    if (!mesh->IsAssetPayloadValid())
        throw std::invalid_argument("skinned Mesh artifact contains neither geometry nor animation data");
    mesh->NormalizeInfluences();
    return mesh;
}

bool SkinnedMeshArtifact::HasCurrentHeader(std::string_view bytes) noexcept
{
    const size_t schemaOffset = Magic.size() + sizeof(uint32_t);
    return bytes.size() >= schemaOffset + sizeof(uint32_t) && bytes.substr(0, Magic.size()) == Magic &&
           static_cast<unsigned char>(bytes[Magic.size()]) == 0x04 &&
           static_cast<unsigned char>(bytes[Magic.size() + 1]) == 0x03 &&
           static_cast<unsigned char>(bytes[Magic.size() + 2]) == 0x02 &&
           static_cast<unsigned char>(bytes[Magic.size() + 3]) == 0x01 &&
           static_cast<unsigned char>(bytes[schemaOffset]) == 'S' &&
           static_cast<unsigned char>(bytes[schemaOffset + 1]) == 'K' &&
           static_cast<unsigned char>(bytes[schemaOffset + 2]) == 'N' &&
           static_cast<unsigned char>(bytes[schemaOffset + 3]) == '1';
}

} // namespace infernux
