#include "MeshArtifact.h"

#include "InxMesh.h"
#include "MeshGeometryCodec.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace infernux
{
namespace
{
constexpr std::string_view Magic = "INXMESHART";
constexpr std::string_view AuthoredSourceIdentity = "infernux.static-mesh.source";
constexpr uint32_t EndianMarker = 0x01020304U;
// Persistent schema identity. AssetDatabase uses this hard boundary to
// reimport obsolete Library artifacts instead of attempting format fallback.
constexpr uint32_t ArtifactSchema = 0x3148534dU;     // MSH1
constexpr uint32_t ModelNodesSource = 0x33444f4eU;   // NOD3, node-local geometry + inherited visibility
constexpr uint32_t ModelNodesBaked = 0x34444f4eU;    // NOD4, baked geometry + inherited visibility
constexpr uint32_t MaterialBindingsV1 = 0x3142544dU; // MTB1, source identity and external GUID per slot
constexpr uint32_t MaterialSurfaceV1 = 0x3153544dU;  // MTS1, source surface rendering per slot
constexpr uint32_t MaterialTextures = 0x3458544dU;   // MTX4, UV set + per-binding sampler
constexpr uint32_t MorphTargetsV1 = 0x3152504dU;     // MPR1, imported vertex-aligned blend-shape deltas
constexpr uint32_t GeometryEncoding = 0x334f4547U;   // GEO3, typed indices, compression and UV0/UV1
constexpr uint32_t MaximumElementCount = 100'000'000U;
constexpr uint32_t MaximumStringBytes = 16U * 1024U * 1024U;

uint64_t Fnv1a64(std::string_view bytes)
{
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void AppendU32(std::string &out, uint32_t value)
{
    for (unsigned shift = 0; shift < 32; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void AppendU16(std::string &out, uint16_t value)
{
    out.push_back(static_cast<char>(value & 0xffU));
    out.push_back(static_cast<char>((value >> 8U) & 0xffU));
}

void AppendU64(std::string &out, uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

void AppendFloat(std::string &out, float value)
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    AppendU32(out, bits);
}

void AppendString(std::string &out, std::string_view value)
{
    if (value.size() > MaximumStringBytes)
        throw std::overflow_error("mesh artifact string exceeds the format limit");
    AppendU32(out, static_cast<uint32_t>(value.size()));
    out.append(value);
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
        for (unsigned shift = 0; shift < 32; shift += 8)
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
        for (unsigned shift = 0; shift < 64; shift += 8)
            value |= static_cast<uint64_t>(static_cast<unsigned char>(m_bytes[m_cursor++])) << shift;
        return value;
    }

    [[nodiscard]] float ReadFloat()
    {
        const uint32_t bits = ReadU32();
        float value = 0.0f;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    [[nodiscard]] std::string ReadString()
    {
        const uint32_t size = ReadU32();
        if (size > MaximumStringBytes)
            throw std::invalid_argument("mesh artifact string exceeds the format limit");
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

    [[nodiscard]] uint32_t ReadCount()
    {
        const uint32_t count = ReadU32();
        if (count > MaximumElementCount)
            throw std::invalid_argument("mesh artifact element count exceeds the format limit");
        return count;
    }

    [[nodiscard]] size_t Cursor() const noexcept
    {
        return m_cursor;
    }

    [[nodiscard]] bool AtEnd() const noexcept
    {
        return m_cursor == m_bytes.size();
    }

  private:
    void Require(size_t size) const
    {
        if (size > m_bytes.size() - m_cursor)
            throw std::invalid_argument("mesh artifact is truncated");
    }

    std::string_view m_bytes;
    size_t m_cursor = 0;
};

void AppendVec2(std::string &out, const glm::vec2 &value)
{
    AppendFloat(out, value.x);
    AppendFloat(out, value.y);
}

void AppendVec3(std::string &out, const glm::vec3 &value)
{
    AppendFloat(out, value.x);
    AppendFloat(out, value.y);
    AppendFloat(out, value.z);
}

void AppendVec4(std::string &out, const glm::vec4 &value)
{
    AppendFloat(out, value.x);
    AppendFloat(out, value.y);
    AppendFloat(out, value.z);
    AppendFloat(out, value.w);
}

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

void AppendCount(std::string &out, size_t count)
{
    if (count > MaximumElementCount)
        throw std::overflow_error("mesh artifact element count exceeds the format limit");
    AppendU32(out, static_cast<uint32_t>(count));
}

} // namespace

std::string MeshArtifact::SerializeSource(const InxMesh &mesh)
{
    if (mesh.HasSkinnedData())
        throw std::invalid_argument("Static mesh source cannot discard skinned mesh data");
    return Serialize(mesh, AuthoredSourceIdentity);
}

std::shared_ptr<InxMesh> MeshArtifact::DeserializeSource(std::string_view bytes)
{
    return Deserialize(bytes, AuthoredSourceIdentity);
}

std::string MeshArtifact::Serialize(const InxMesh &mesh, std::string_view sourceContentHash)
{
    if (sourceContentHash.empty())
        throw std::invalid_argument("mesh artifact requires a source content hash");
    if (!mesh.HasCpuGeometry())
        throw std::runtime_error("mesh artifact serialization requires retained CPU geometry");

    std::string bytes(Magic);
    AppendU32(bytes, EndianMarker);
    AppendU32(bytes, ArtifactSchema);
    AppendString(bytes, sourceContentHash);
    AppendString(bytes, mesh.GetName());

    const auto sourceGeometry = mesh.GetModelSourceGeometry();
    const auto geometry = sourceGeometry ? sourceGeometry : mesh.GetGeometrySnapshot();
    const auto &vertices = geometry->vertices;
    const auto indexFormat = ResolveMeshIndexFormat(mesh.GetIndexFormat(), vertices.size(), geometry->indices);
    AppendU32(bytes, GeometryEncoding);
    AppendU32(bytes, indexFormat == MeshIndexFormat::UInt16 ? 16U : 32U);
    AppendU32(bytes, static_cast<uint32_t>(mesh.GetCompression()));
    AppendCount(bytes, vertices.size());
    if (mesh.GetCompression() == MeshCompression::Off) {
        for (const Vertex &vertex : vertices) {
            AppendVec3(bytes, vertex.pos);
            AppendVec3(bytes, vertex.normal);
            AppendVec4(bytes, vertex.tangent);
            AppendVec3(bytes, vertex.color);
            AppendVec2(bytes, vertex.texCoord);
            AppendVec2(bytes, vertex.texCoord1);
            for (glm::length_t component = 0; component < vertex.boneIndices.length(); ++component)
                AppendU32(bytes, vertex.boneIndices[component]);
            AppendVec4(bytes, vertex.boneWeights);
        }
    } else {
        const std::string encoded = mesh_geometry_codec::Encode(vertices, mesh.GetCompression(), true);
        AppendCount(bytes, encoded.size());
        bytes.append(encoded);
    }

    const auto &indices = geometry->indices;
    AppendCount(bytes, indices.size());
    for (uint32_t index : indices) {
        if (indexFormat == MeshIndexFormat::UInt16)
            AppendU16(bytes, static_cast<uint16_t>(index));
        else
            AppendU32(bytes, index);
    }

    const auto &subMeshes = geometry->subMeshes;
    AppendCount(bytes, subMeshes.size());
    for (const SubMesh &subMesh : subMeshes) {
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

    const auto &slotNames = mesh.GetMaterialSlotNames();
    AppendCount(bytes, slotNames.size());
    for (const auto &name : slotNames)
        AppendString(bytes, name);

    const auto &slotData = mesh.GetMaterialSlotData();
    AppendCount(bytes, slotData.size());
    for (const MaterialSlotData &material : slotData) {
        AppendVec4(bytes, material.baseColor);
        AppendVec4(bytes, material.emissionColor);
        AppendFloat(bytes, material.metallic);
        AppendFloat(bytes, material.smoothness);
        AppendFloat(bytes, material.opacity);
    }

    const auto &nodeNames = mesh.GetNodeNames();
    AppendCount(bytes, nodeNames.size());
    for (const auto &name : nodeNames)
        AppendString(bytes, name);

    const auto &nodes = mesh.GetModelNodes();
    if (!nodes.empty()) {
        AppendU32(bytes, sourceGeometry ? ModelNodesSource : ModelNodesBaked);
        AppendCount(bytes, nodes.size());
        for (const auto &node : nodes) {
            AppendString(bytes, node.name);
            AppendU32(bytes, static_cast<uint32_t>(node.parentIndex));
            AppendU32(bytes, static_cast<uint32_t>(node.nodeGroup));
            for (glm::length_t column = 0; column < 4; ++column)
                AppendVec4(bytes, node.localTransform[column]);
            AppendU32(bytes, node.visible ? 1U : 0U);
        }
    }

    if (!slotData.empty()) {
        AppendU32(bytes, MaterialBindingsV1);
        AppendCount(bytes, slotData.size());
        for (const auto &material : slotData) {
            AppendString(bytes, material.sourceId);
            AppendString(bytes, material.materialGuid);
        }
        AppendU32(bytes, MaterialSurfaceV1);
        AppendCount(bytes, slotData.size());
        for (const auto &material : slotData) {
            AppendU32(bytes, static_cast<uint32_t>(material.alphaMode));
            AppendFloat(bytes, material.alphaCutoff);
            AppendU32(bytes, material.doubleSided ? 1U : 0U);
        }
        AppendU32(bytes, MaterialTextures);
        AppendCount(bytes, slotData.size());
        for (const auto &material : slotData) {
            for (size_t index = 0; index < ModelTextureCount; ++index) {
                const auto &guid = material.textureGuids[index];
                AppendString(bytes, guid);
                if (material.textureUvSets[index] > 1)
                    throw std::invalid_argument("mesh artifact has invalid material UV set");
                AppendU32(bytes, material.textureUvSets[index]);
                const auto &sampler = material.textureSamplers[index];
                AppendU32(bytes, static_cast<uint32_t>(sampler.minFilter));
                AppendU32(bytes, static_cast<uint32_t>(sampler.magFilter));
                AppendU32(bytes, static_cast<uint32_t>(sampler.mipFilter));
                AppendU32(bytes, static_cast<uint32_t>(sampler.addressU));
                AppendU32(bytes, static_cast<uint32_t>(sampler.addressV));
                AppendU32(bytes, static_cast<uint32_t>(sampler.addressW));
            }
            AppendFloat(bytes, material.normalScale);
            AppendFloat(bytes, material.occlusionStrength);
            AppendU32(bytes, material.packedMetallicRoughness ? 1U : 0U);
        }
    }
    if (!geometry->morphTargets.empty()) {
        AppendU32(bytes, MorphTargetsV1);
        AppendCount(bytes, geometry->morphTargets.size());
        for (const auto &target : geometry->morphTargets) {
            AppendString(bytes, target.name);
            AppendFloat(bytes, target.defaultWeight);
            AppendCount(bytes, target.positionDeltas.size());
            for (const auto &delta : target.positionDeltas)
                AppendVec3(bytes, delta);
            AppendCount(bytes, target.normalDeltas.size());
            for (const auto &delta : target.normalDeltas)
                AppendVec3(bytes, delta);
            AppendCount(bytes, target.tangentDeltas.size());
            for (const auto &delta : target.tangentDeltas)
                AppendVec3(bytes, delta);
        }
    }
    AppendU64(bytes, Fnv1a64(bytes));
    return bytes;
}

std::shared_ptr<InxMesh> MeshArtifact::Deserialize(std::string_view bytes, std::string_view expectedSourceContentHash)
{
    if (bytes.size() < Magic.size() + sizeof(uint32_t) + sizeof(uint64_t) || bytes.substr(0, Magic.size()) != Magic)
        throw std::invalid_argument("mesh artifact has an invalid header");

    const size_t payloadSize = bytes.size() - sizeof(uint64_t);
    Reader checksumReader(bytes.substr(payloadSize));
    const uint64_t storedChecksum = checksumReader.ReadU64();
    if (storedChecksum != Fnv1a64(bytes.substr(0, payloadSize)))
        throw std::invalid_argument("mesh artifact checksum mismatch");

    Reader reader(bytes.substr(Magic.size(), payloadSize - Magic.size()));
    if (reader.ReadU32() != EndianMarker)
        throw std::invalid_argument("mesh artifact has an invalid endian marker");
    if (reader.ReadU32() != ArtifactSchema)
        throw std::invalid_argument("mesh artifact has an unsupported schema");
    const std::string sourceContentHash = reader.ReadString();
    if (sourceContentHash.empty() || sourceContentHash != expectedSourceContentHash)
        throw std::invalid_argument("mesh artifact does not match the imported source content");

    auto mesh = std::make_shared<InxMesh>(reader.ReadString());

    if (reader.ReadU32() != GeometryEncoding)
        throw std::invalid_argument("mesh artifact has an invalid geometry encoding");
    const uint32_t indexBits = reader.ReadU32();
    if (indexBits != 16U && indexBits != 32U)
        throw std::invalid_argument("mesh artifact contains an unsupported index encoding");
    const MeshIndexFormat indexFormat = indexBits == 16U ? MeshIndexFormat::UInt16 : MeshIndexFormat::UInt32;
    const uint32_t encodedCompression = reader.ReadU32();
    if (encodedCompression > static_cast<uint32_t>(MeshCompression::High))
        throw std::invalid_argument("mesh artifact contains an unsupported geometry compression");
    const auto compression = static_cast<MeshCompression>(encodedCompression);
    const uint32_t vertexCount = reader.ReadCount();
    mesh->SetIndexFormat(indexFormat);
    mesh->SetCompression(compression);
    std::vector<Vertex> vertices;
    if (compression == MeshCompression::Off) {
        vertices.resize(vertexCount);
        for (Vertex &vertex : vertices) {
            vertex.pos = ReadVec3(reader);
            vertex.normal = ReadVec3(reader);
            vertex.tangent = ReadVec4(reader);
            vertex.color = ReadVec3(reader);
            vertex.texCoord = ReadVec2(reader);
            vertex.texCoord1 = ReadVec2(reader);
            for (glm::length_t component = 0; component < vertex.boneIndices.length(); ++component)
                vertex.boneIndices[component] = reader.ReadU32();
            vertex.boneWeights = ReadVec4(reader);
        }
    } else {
        const uint32_t encodedSize = reader.ReadCount();
        vertices = mesh_geometry_codec::Decode(reader.ReadBytes(encodedSize), vertexCount, compression, true);
    }

    std::vector<uint32_t> indices(reader.ReadCount());
    for (uint32_t &index : indices) {
        index = indexFormat == MeshIndexFormat::UInt16 ? reader.ReadU16() : reader.ReadU32();
        if (index >= vertices.size())
            throw std::invalid_argument("mesh artifact contains an out-of-range vertex index");
    }

    std::vector<SubMesh> subMeshes(reader.ReadCount());
    for (SubMesh &subMesh : subMeshes) {
        subMesh.indexStart = reader.ReadU32();
        subMesh.indexCount = reader.ReadU32();
        subMesh.vertexStart = reader.ReadU32();
        subMesh.vertexCount = reader.ReadU32();
        subMesh.materialSlot = reader.ReadU32();
        subMesh.nodeGroup = reader.ReadU32();
        subMesh.boundsMin = ReadVec3(reader);
        subMesh.boundsMax = ReadVec3(reader);
        subMesh.name = reader.ReadString();
        if (subMesh.indexStart > indices.size() || subMesh.indexCount > indices.size() - subMesh.indexStart ||
            subMesh.vertexStart > vertices.size() || subMesh.vertexCount > vertices.size() - subMesh.vertexStart)
            throw std::invalid_argument("mesh artifact contains an invalid submesh range");
    }

    std::vector<std::string> slotNames(reader.ReadCount());
    for (auto &name : slotNames)
        name = reader.ReadString();

    std::vector<MaterialSlotData> slotData(reader.ReadCount());
    for (MaterialSlotData &material : slotData) {
        material.baseColor = ReadVec4(reader);
        material.emissionColor = ReadVec4(reader);
        material.metallic = reader.ReadFloat();
        material.smoothness = reader.ReadFloat();
        material.opacity = reader.ReadFloat();
    }

    std::vector<std::string> nodeNames(reader.ReadCount());
    for (auto &name : nodeNames)
        name = reader.ReadString();
    std::vector<ImportedModelNode> nodes;
    bool sourceLocal = false;
    bool readNodes = false;
    bool readBindings = false;
    bool readSurface = false;
    bool readTextures = false;
    bool readMorphTargets = false;
    std::vector<MeshMorphTarget> morphTargets;
    while (!reader.AtEnd()) {
        const uint32_t nodeFormat = reader.ReadU32();
        if (nodeFormat == MorphTargetsV1) {
            if (readMorphTargets)
                throw std::invalid_argument("mesh artifact repeats its morph-target section");
            readMorphTargets = true;
            morphTargets.resize(reader.ReadCount());
            std::unordered_set<std::string> names;
            for (auto &target : morphTargets) {
                target.name = reader.ReadString();
                target.defaultWeight = reader.ReadFloat();
                if (target.name.empty() || !names.insert(target.name).second)
                    throw std::invalid_argument("mesh artifact contains invalid morph-target names");
                const auto readDeltas = [&](std::vector<glm::vec3> &values, bool required) {
                    const uint32_t count = reader.ReadCount();
                    if ((required && count != vertices.size()) || (!required && count != 0 && count != vertices.size()))
                        throw std::invalid_argument("mesh artifact morph target does not match its vertex domain");
                    values.reserve(count);
                    for (uint32_t index = 0; index < count; ++index)
                        values.push_back(ReadVec3(reader));
                };
                readDeltas(target.positionDeltas, true);
                readDeltas(target.normalDeltas, false);
                readDeltas(target.tangentDeltas, false);
            }
            continue;
        }
        if (nodeFormat == MaterialTextures) {
            if (readTextures || reader.ReadCount() != slotData.size())
                throw std::invalid_argument("mesh artifact has invalid material textures");
            readTextures = true;
            for (auto &material : slotData) {
                for (size_t index = 0; index < ModelTextureCount; ++index) {
                    material.textureGuids[index] = reader.ReadString();
                    const auto uvSet = reader.ReadU32();
                    if (uvSet > 1)
                        throw std::invalid_argument("mesh artifact has invalid material UV set");
                    material.textureUvSets[index] = static_cast<uint8_t>(uvSet);
                    auto &sampler = material.textureSamplers[index];
                    const auto minFilter = reader.ReadU32(), magFilter = reader.ReadU32();
                    const auto mipFilter = reader.ReadU32(), addressU = reader.ReadU32();
                    const auto addressV = reader.ReadU32(), addressW = reader.ReadU32();
                    if (minFilter > static_cast<uint32_t>(MaterialSamplerFilter::Linear) ||
                        magFilter > static_cast<uint32_t>(MaterialSamplerFilter::Linear) ||
                        mipFilter > static_cast<uint32_t>(MaterialSamplerFilter::Linear) ||
                        addressU > static_cast<uint32_t>(MaterialSamplerAddress::Mirror) ||
                        addressV > static_cast<uint32_t>(MaterialSamplerAddress::Mirror) ||
                        addressW > static_cast<uint32_t>(MaterialSamplerAddress::Mirror))
                        throw std::invalid_argument("mesh artifact has invalid material sampler");
                    sampler.minFilter = static_cast<MaterialSamplerFilter>(minFilter);
                    sampler.magFilter = static_cast<MaterialSamplerFilter>(magFilter);
                    sampler.mipFilter = static_cast<MaterialSamplerFilter>(mipFilter);
                    sampler.addressU = static_cast<MaterialSamplerAddress>(addressU);
                    sampler.addressV = static_cast<MaterialSamplerAddress>(addressV);
                    sampler.addressW = static_cast<MaterialSamplerAddress>(addressW);
                }
                material.normalScale = reader.ReadFloat();
                material.occlusionStrength = reader.ReadFloat();
                const auto packed = reader.ReadU32();
                if (packed > 1U)
                    throw std::invalid_argument("mesh artifact has invalid packed texture flag");
                material.packedMetallicRoughness = packed != 0;
            }
            continue;
        }
        if (nodeFormat == MaterialSurfaceV1) {
            if (readSurface || reader.ReadCount() != slotData.size())
                throw std::invalid_argument("mesh artifact has invalid material surfaces");
            readSurface = true;
            for (auto &material : slotData) {
                const auto mode = reader.ReadU32();
                if (mode > static_cast<uint32_t>(ModelAlphaMode::Blend))
                    throw std::invalid_argument("mesh artifact has invalid material alpha mode");
                material.alphaMode = static_cast<ModelAlphaMode>(mode);
                material.alphaCutoff = reader.ReadFloat();
                const auto sided = reader.ReadU32();
                if (sided > 1U)
                    throw std::invalid_argument("mesh artifact has invalid material double-sided flag");
                material.doubleSided = sided != 0;
            }
            continue;
        }
        if (nodeFormat == MaterialBindingsV1) {
            if (readBindings || reader.ReadCount() != slotData.size())
                throw std::invalid_argument("mesh artifact has invalid material bindings");
            readBindings = true;
            for (auto &material : slotData) {
                material.sourceId = reader.ReadString();
                material.materialGuid = reader.ReadString();
            }
            continue;
        }
        if (readNodes)
            throw std::invalid_argument("mesh artifact repeats its model hierarchy section");
        readNodes = true;
        sourceLocal = nodeFormat == ModelNodesSource;
        if (nodeFormat != ModelNodesSource && nodeFormat != ModelNodesBaked)
            throw std::invalid_argument("mesh artifact has an unsupported model hierarchy section");
        nodes.resize(reader.ReadCount());
        const auto readNodeIndex = [&reader]() -> int32_t {
            const uint32_t value = reader.ReadU32();
            if (value == std::numeric_limits<uint32_t>::max())
                return -1;
            if (value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
                throw std::invalid_argument("mesh artifact contains an invalid model node index");
            return static_cast<int32_t>(value);
        };
        for (auto &node : nodes) {
            node.name = reader.ReadString();
            node.parentIndex = readNodeIndex();
            node.nodeGroup = readNodeIndex();
            for (glm::length_t column = 0; column < 4; ++column)
                node.localTransform[column] = ReadVec4(reader);
            const uint32_t visible = reader.ReadU32();
            if (visible > 1U)
                throw std::invalid_argument("mesh artifact contains an invalid model visibility flag");
            node.visible = visible != 0;
        }
    }
    if (!reader.AtEnd())
        throw std::invalid_argument("mesh artifact contains trailing data");
    if (!slotData.empty() && (!readBindings || !readSurface || !readTextures))
        throw std::invalid_argument("mesh artifact is missing current material sections");

    mesh->SetMaterialSlotNames(std::move(slotNames));
    mesh->SetMaterialSlotData(std::move(slotData));
    mesh->SetNodeNames(std::move(nodeNames));
    if (sourceLocal) {
        mesh->SetModelData(std::move(vertices), std::move(indices), std::move(subMeshes), std::move(nodes),
                           std::move(morphTargets));
    } else {
        mesh->SetData(std::move(vertices), std::move(indices), std::move(subMeshes), std::move(morphTargets));
        mesh->SetModelNodes(std::move(nodes));
    }
    return mesh;
}

bool MeshArtifact::HasCurrentHeader(std::string_view bytes) noexcept
{
    const size_t schemaOffset = Magic.size() + sizeof(uint32_t);
    return bytes.size() >= schemaOffset + sizeof(uint32_t) && bytes.substr(0, Magic.size()) == Magic &&
           static_cast<unsigned char>(bytes[Magic.size()]) == 0x04 &&
           static_cast<unsigned char>(bytes[Magic.size() + 1]) == 0x03 &&
           static_cast<unsigned char>(bytes[Magic.size() + 2]) == 0x02 &&
           static_cast<unsigned char>(bytes[Magic.size() + 3]) == 0x01 &&
           static_cast<unsigned char>(bytes[schemaOffset]) == 'M' &&
           static_cast<unsigned char>(bytes[schemaOffset + 1]) == 'S' &&
           static_cast<unsigned char>(bytes[schemaOffset + 2]) == 'H' &&
           static_cast<unsigned char>(bytes[schemaOffset + 3]) == '1';
}

} // namespace infernux
