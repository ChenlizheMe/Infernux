#include "MeshArtifact.h"

#include "InxMesh.h"
#include "MeshDecodeStorage.h"
#include "MeshGeometryCodec.h"

#include <algorithm>
#include <core/serialization/BinaryCodec.h>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infernux
{
namespace
{
constexpr std::string_view Magic = "INXMESHART";
constexpr std::string_view AuthoredSourceIdentity = "infernux.static-mesh.source";
constexpr uint32_t SourceSchema = 0x3148534dU; // MSH1, authored mesh
// Persistent schema identity. AssetDatabase uses this hard boundary to
// reimport obsolete Library artifacts instead of attempting format fallback.
constexpr uint32_t ArtifactSchema = 0x3248534dU;     // MSH2 with decode workspace
constexpr uint32_t ModelNodesSource = 0x33444f4eU;   // NOD3, node-local geometry + inherited visibility
constexpr uint32_t ModelNodesBaked = 0x34444f4eU;    // NOD4, baked geometry + inherited visibility
constexpr uint32_t MaterialBindingsV1 = 0x3142544dU; // MTB1, source identity and external GUID per slot
constexpr uint32_t MaterialSurfaceV1 = 0x3153544dU;  // MTS1, source surface rendering per slot
constexpr uint32_t MaterialTextures = 0x3458544dU;   // MTX4, UV set + per-binding sampler
constexpr uint32_t MorphTargetsV1 = 0x3152504dU;     // MPR1, imported vertex-aligned blend-shape deltas
constexpr uint32_t GeometryEncoding = 0x334f4547U;   // GEO3, typed indices, compression and UV0/UV1
constexpr uint32_t MaximumElementCount = 100'000'000U;
constexpr uint32_t MaximumStringBytes = 16U * 1024U * 1024U;

using binary_codec::Fnv1a64;

using binary_codec::AppendU32;

using binary_codec::AppendU16;

using binary_codec::AppendU64;

using binary_codec::AppendFloat;

void AppendString(std::string &out, std::string_view value)
{
    if (value.size() > MaximumStringBytes)
        throw std::overflow_error("mesh artifact string exceeds the format limit");
    AppendU32(out, static_cast<uint32_t>(value.size()));
    out.append(value);
}

using Input = binary_codec::Reader<MaximumStringBytes>;
using Reader = mesh_decode_storage::Reader<MaximumStringBytes>;

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

std::string WriteMesh(const InxMesh &mesh, std::string_view sourceContentHash, bool cooked)
{
    if (sourceContentHash.empty())
        throw std::invalid_argument("mesh artifact requires a source content hash");
    if (!mesh.HasCpuGeometry())
        throw std::runtime_error("mesh artifact serialization requires retained CPU geometry");

    const auto sourceGeometry = mesh.GetModelSourceGeometry();
    const auto geometry = sourceGeometry ? sourceGeometry : mesh.GetGeometrySnapshot();
    const bool sourceLocal = sourceGeometry && !mesh.GetModelNodes().empty();
    std::string bytes;
    if (cooked) {
        artifact_layout::AppendHeader(bytes, Magic, ArtifactSchema,
                                      mesh_decode_storage::Mesh(mesh, *geometry, sourceLocal, sourceContentHash));
    } else {
        artifact_layout::AppendSignature(bytes, Magic, SourceSchema);
    }
    AppendString(bytes, sourceContentHash);
    AppendString(bytes, mesh.GetName());

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
        const std::string encoded = mesh_geometry_codec::Encode(vertices, mesh.GetCompression());
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

std::shared_ptr<InxMesh> ReadMeshPayload(Reader &reader, std::string_view expectedSourceContentHash)
{
    auto &work = reader.Workspace();
    const std::string sourceContentHash = reader.ReadString();
    if (sourceContentHash.empty() || sourceContentHash != expectedSourceContentHash)
        throw std::invalid_argument("mesh artifact source content hash does not match metadata");

    work.Add<InxMesh>();
    work.Add<MeshGeometry>(2);
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
    const uint32_t vertexCount = reader.ReadArrayCount<Vertex>(MaximumElementCount);
    mesh->SetIndexFormat(indexFormat);
    mesh->SetCompression(compression);
    auto vertices = mesh_geometry_codec::ReadVertices(reader, vertexCount, compression, MaximumElementCount);

    std::vector<uint32_t> indices(reader.ReadArrayCount<uint32_t>(MaximumElementCount, indexBits / 8u));
    for (uint32_t &index : indices) {
        index = indexFormat == MeshIndexFormat::UInt16 ? reader.ReadU16() : reader.ReadU32();
        if (index >= vertices.size())
            throw std::invalid_argument("mesh artifact contains an out-of-range vertex index");
    }

    std::vector<SubMesh> subMeshes(reader.ReadArrayCount<SubMesh>(MaximumElementCount, 52u));
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

    std::vector<std::string> slotNames(reader.ReadArrayCount<std::string>(MaximumElementCount, 4u));
    for (auto &name : slotNames)
        name = reader.ReadString();

    std::vector<MaterialSlotData> slotData(reader.ReadArrayCount<MaterialSlotData>(MaximumElementCount, 44u));
    for (MaterialSlotData &material : slotData) {
        material.baseColor = ReadVec4(reader);
        material.emissionColor = ReadVec4(reader);
        material.metallic = reader.ReadFloat();
        material.smoothness = reader.ReadFloat();
        material.opacity = reader.ReadFloat();
    }

    std::vector<std::string> nodeNames(reader.ReadArrayCount<std::string>(MaximumElementCount, 4u));
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
            morphTargets.resize(reader.ReadArrayCount<MeshMorphTarget>(MaximumElementCount, 20u));
            for (auto &target : morphTargets) {
                target.name = reader.ReadString();
                target.defaultWeight = reader.ReadFloat();
                const auto readDeltas = [&](std::vector<glm::vec3> &values, bool required) {
                    const uint32_t count = reader.ReadArrayCount<glm::vec3>(MaximumElementCount, 12u);
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
        nodes.resize(reader.ReadArrayCount<ImportedModelNode>(MaximumElementCount, 80u));
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

    work.Add<uint8_t>(nodeNames.size());
    work.AddNameTable(morphTargets.size());
    if (sourceLocal) {
        mesh_decode_storage::AddSourceLocal(work, vertices.size(), indices.size(), subMeshes, morphTargets, nodeNames,
                                            nodes.size());
    }
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

std::shared_ptr<InxMesh> ReadCookedMesh(Input input, std::string_view expectedSourceContentHash,
                                        uint64_t workspaceLimit = UINT64_MAX)
{
    const auto workspace = artifact_layout::ReadHeader(input, Magic, ArtifactSchema, workspaceLimit);
    Reader reader(std::move(input), workspace);
    return ReadMeshPayload(reader, expectedSourceContentHash);
}
} // namespace

std::string MeshArtifact::SerializeSource(const InxMesh &mesh)
{
    if (mesh.HasSkinnedData())
        throw std::invalid_argument("Static mesh source cannot discard skinned mesh data");
    return WriteMesh(mesh, AuthoredSourceIdentity, false);
}

std::shared_ptr<InxMesh> MeshArtifact::DeserializeSource(std::string_view bytes)
{
    Input input(binary_codec::CheckedPayload(bytes));
    if (!artifact_layout::ReadSignature(input, Magic, SourceSchema))
        throw std::invalid_argument("invalid authored mesh layout");
    Reader reader(std::move(input), UINT64_MAX);
    return ReadMeshPayload(reader, AuthoredSourceIdentity);
}

std::string MeshArtifact::Serialize(const InxMesh &mesh, std::string_view sourceContentHash)
{
    return WriteMesh(mesh, sourceContentHash, true);
}

std::shared_ptr<InxMesh> MeshArtifact::Deserialize(std::string_view bytes, std::string_view expectedSourceContentHash)
{
    return ReadCookedMesh(Input(binary_codec::CheckedPayload(bytes)), expectedSourceContentHash);
}

std::shared_ptr<InxMesh> MeshArtifact::Read(FileInput &file, std::string_view expectedSourceContentHash,
                                            std::vector<char> &block, uint64_t workspaceLimit)
{
    ArtifactFileReader input(file, (std::numeric_limits<size_t>::max)(), block);
    auto mesh = ReadCookedMesh(input.Begin<Input>(), expectedSourceContentHash, workspaceLimit);
    input.Verify();
    return mesh;
}

bool MeshArtifact::HasCurrentHeader(std::string_view bytes) noexcept
{
    return artifact_layout::HasHeader(bytes, Magic, ArtifactSchema);
}

uint64_t MeshArtifact::ReadWorkspace(FileInput &file)
{
    return artifact_layout::ReadWorkspace(file, Magic, ArtifactSchema);
}

} // namespace infernux
