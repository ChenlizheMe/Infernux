#include <function/resources/InxMesh/InxMesh.h>
#include <function/resources/InxMesh/MeshArtifact.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/scene/PrimitiveMeshes.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
template <typename Callback> void RequireInvalid(Callback callback)
{
    bool rejected = false;
    try {
        callback();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

bool NearlyEqual(float left, float right)
{
    return std::abs(left - right) < 1.0e-6f;
}

uint64_t Fnv1a64(std::string_view bytes)
{
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void RewriteChecksum(std::string &bytes)
{
    const size_t offset = bytes.size() - sizeof(uint64_t);
    const uint64_t checksum = Fnv1a64(std::string_view(bytes).substr(0, offset));
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes[offset + shift / 8] = static_cast<char>((checksum >> shift) & 0xffU);
}

void WriteU32(std::string &bytes, size_t offset, uint32_t value)
{
    assert(offset <= bytes.size() && sizeof(value) <= bytes.size() - offset);
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes[offset + shift / 8] = static_cast<char>((value >> shift) & 0xffU);
}

uint32_t ReadU32(const std::string &bytes, size_t offset)
{
    assert(offset <= bytes.size() && sizeof(uint32_t) <= bytes.size() - offset);
    uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + shift / 8])) << shift;
    return value;
}

void WriteFloat(std::string &bytes, size_t offset, float value)
{
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    WriteU32(bytes, offset, bits);
}

size_t DowngradeGeo3ToGeo2(std::string &bytes, size_t vertexCount)
{
    const size_t marker = bytes.find("GEO3");
    assert(marker != std::string::npos);
    constexpr size_t NewVertexBytes = 25U * sizeof(uint32_t);
    constexpr size_t SecondaryUvOffset = 15U * sizeof(uint32_t);
    const size_t vertexStart = marker + 4U * sizeof(uint32_t);
    for (size_t index = vertexCount; index-- > 0;)
        bytes.erase(vertexStart + index * NewVertexBytes + SecondaryUvOffset, 2U * sizeof(float));
    WriteU32(bytes, marker, 0x324f4547U);
    return marker;
}

void DowngradeMaterialTexturesV3(std::string &bytes, bool toV1)
{
    const size_t marker = bytes.find("MTX4");
    assert(marker != std::string::npos);
    const uint32_t materialCount = ReadU32(bytes, marker + sizeof(uint32_t));
    size_t cursor = marker + 2U * sizeof(uint32_t);
    std::string replacement;
    for (uint32_t material = 0; material < materialCount; ++material) {
        std::array<std::string, infernux::ModelTextureCount> encodedGuids;
        for (auto &encoded : encodedGuids) {
            const size_t start = cursor;
            const uint32_t length = ReadU32(bytes, cursor);
            cursor += sizeof(uint32_t) + length;
            encoded = bytes.substr(start, sizeof(uint32_t) + length);
            cursor += 7U * sizeof(uint32_t); // V4 UV set + sampler state.
        }
        if (toV1) {
            replacement += encodedGuids.front();
        } else {
            for (const auto &encoded : encodedGuids)
                replacement += encoded;
            replacement.append(bytes, cursor, 2U * sizeof(float) + sizeof(uint32_t));
        }
        cursor += 2U * sizeof(float) + sizeof(uint32_t);
    }
    bytes.replace(marker + 2U * sizeof(uint32_t), cursor - (marker + 2U * sizeof(uint32_t)), replacement);
    WriteU32(bytes, marker, toV1 ? 0x3158544dU : 0x3258544dU);
    RewriteChecksum(bytes);
}

void DowngradeMaterialTexturesV4ToV3(std::string &bytes)
{
    const size_t marker = bytes.find("MTX4");
    assert(marker != std::string::npos);
    const uint32_t materialCount = ReadU32(bytes, marker + sizeof(uint32_t));
    size_t cursor = marker + 2U * sizeof(uint32_t);
    std::string replacement;
    for (uint32_t material = 0; material < materialCount; ++material) {
        for (size_t binding = 0; binding < infernux::ModelTextureCount; ++binding) {
            const uint32_t length = ReadU32(bytes, cursor);
            const size_t guidBytes = sizeof(uint32_t) + length;
            replacement.append(bytes, cursor, guidBytes + sizeof(uint32_t)); // GUID + UV set.
            cursor += guidBytes + 7U * sizeof(uint32_t);                     // Skip UV set + six sampler fields.
        }
        replacement.append(bytes, cursor, 2U * sizeof(float) + sizeof(uint32_t));
        cursor += 2U * sizeof(float) + sizeof(uint32_t);
    }
    bytes.replace(marker + 2U * sizeof(uint32_t), cursor - (marker + 2U * sizeof(uint32_t)), replacement);
    WriteU32(bytes, marker, 0x3358544dU);
    RewriteChecksum(bytes);
}
} // namespace

int main()
{
    std::shared_ptr<const infernux::MeshGeometry> retainedGeometry;
    {
        infernux::InxMesh mesh("mutable identity");
        infernux::Vertex oldVertex{};
        oldVertex.pos = {1.0f, 2.0f, 3.0f};
        infernux::SubMesh oldSubMesh;
        oldSubMesh.name = "old";
        mesh.SetData({oldVertex}, {0, 0, 0}, {oldSubMesh});
        retainedGeometry = mesh.GetGeometrySnapshot();
        infernux::InxMesh copy = mesh;
        oldVertex.pos = {9.0f, 8.0f, 7.0f};
        mesh.SetData({oldVertex}, {0}, {});
        assert(mesh.GetGeometrySnapshot() != retainedGeometry);
        assert(mesh.GetVertices().front().pos == oldVertex.pos);
        assert(mesh.GetBoundsMin() == oldVertex.pos);
        assert(copy.GetGeometrySnapshot() == retainedGeometry);
        assert(copy.GetVertices().front().pos == glm::vec3(1.0f, 2.0f, 3.0f));
        assert(retainedGeometry->indices.size() == 3);
        assert(retainedGeometry->subMeshes.front().name == "old");
        assert(retainedGeometry->boundsMax == glm::vec3(1.0f, 2.0f, 3.0f));
        mesh.SetData({}, {}, {});
        assert(mesh.GetVertices().empty());
        assert(mesh.GetBoundsMin() == glm::vec3(0.0f));
        assert(mesh.GetGeneration() == 3);
    }
    assert(retainedGeometry->vertices.front().pos.x == 1.0f);
    const std::weak_ptr<const infernux::MeshGeometry> retired = retainedGeometry;
    retainedGeometry.reset();
    assert(retired.expired());

    {
        infernux::InxMesh mesh("range update");
        std::vector<infernux::Vertex> vertices(6);
        for (size_t index = 0; index < vertices.size(); ++index)
            vertices[index].pos = glm::vec3(static_cast<float>(index));
        infernux::SubMesh left, right;
        left.vertexCount = 3;
        left.indexCount = 3;
        left.name = "left";
        right.vertexStart = 3;
        right.vertexCount = 3;
        right.indexStart = 3;
        right.indexCount = 3;
        right.materialSlot = 1;
        mesh.SetData(vertices, {0, 1, 2, 3, 4, 5}, {left, right});
        const auto before = mesh.GetGeometrySnapshot();
        auto replacement = vertices[1];
        replacement.pos = {-2.0f, 8.0f, 1.0f};
        replacement.normal = {1.0f, 0.0f, 0.0f};
        mesh.UpdateVertexRange(1, {replacement});
        assert(mesh.GetGeneration() == 2);
        assert(before->vertices[1].pos == glm::vec3(1.0f));
        assert(mesh.GetVertices()[1].normal == replacement.normal);
        assert(mesh.GetVertices()[0].pos == vertices[0].pos);
        assert(mesh.GetVertices()[5].pos == vertices[5].pos);
        assert(mesh.GetIndices() == before->indices);
        assert(mesh.GetBoundsMin() == glm::vec3(-2.0f, 0.0f, 0.0f));
        assert(mesh.GetBoundsMax() == glm::vec3(5.0f, 8.0f, 5.0f));
        assert(mesh.GetSubMesh(0).boundsMax == glm::vec3(2.0f, 8.0f, 2.0f));
        assert(mesh.GetSubMesh(1).boundsMin == glm::vec3(3.0f));
        assert(mesh.GetSubMesh(1).materialSlot == 1);
        const auto updated = mesh.GetGeometrySnapshot();
        RequireInvalid([&] { mesh.UpdateVertexRange(6, {replacement}); });
        RequireInvalid([&] { mesh.UpdateVertexRange(size_t(-1), {}); });
        mesh.UpdateVertexRange(6, {});
        assert(mesh.GetGeometrySnapshot() == updated);
        assert(mesh.GetGeneration() == 2);
    }

    const infernux::Vertex defaultVertex{};
    assert(defaultVertex.pos == glm::vec3(0.0f));
    assert(defaultVertex.normal == glm::vec3(0.0f, 1.0f, 0.0f));
    assert(defaultVertex.tangent == glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    assert(defaultVertex.color == glm::vec3(1.0f));
    assert(defaultVertex.texCoord == glm::vec2(0.0f));
    assert(defaultVertex.texCoord1 == glm::vec2(0.0f));

    const auto &sphereVertices = infernux::PrimitiveMeshes::GetSphereVertices();
    const auto &sphereIndices = infernux::PrimitiveMeshes::GetSphereIndices();
    assert(sphereIndices.size() % 3 == 0);
    for (size_t triangle = 0; triangle < sphereIndices.size(); triangle += 3) {
        const float u0 = sphereVertices.at(sphereIndices[triangle]).texCoord.x;
        const float u1 = sphereVertices.at(sphereIndices[triangle + 1]).texCoord.x;
        const float u2 = sphereVertices.at(sphereIndices[triangle + 2]).texCoord.x;
        const float minU = std::min({u0, u1, u2});
        const float maxU = std::max({u0, u1, u2});
        assert(maxU - minU <= 0.500001f);
    }

    infernux::InxMesh source("artifact-probe");
    assert(source.GetGeneration() == 0);
    infernux::Vertex vertex{};
    vertex.pos = {1.0f, 2.0f, 3.0f};
    vertex.normal = {0.0f, 1.0f, 0.0f};
    vertex.tangent = {1.0f, 0.0f, 0.0f, -1.0f};
    vertex.color = {0.25f, 0.5f, 0.75f};
    vertex.texCoord = {0.125f, 0.875f};
    vertex.texCoord1 = {0.75f, 0.25f};
    vertex.boneIndices = {1, 2, 3, 4};
    vertex.boneWeights = {0.4f, 0.3f, 0.2f, 0.1f};

    infernux::SubMesh subMesh;
    subMesh.indexCount = 3;
    subMesh.vertexCount = 1;
    subMesh.materialSlot = 2;
    subMesh.nodeGroup = 1;
    subMesh.boundsMin = vertex.pos;
    subMesh.boundsMax = vertex.pos;
    subMesh.name = "triangle";
    source.SetData({vertex}, {0, 0, 0}, {subMesh});
    assert(source.GetGeneration() == 1);
    source.SetData({vertex}, {0, 0, 0}, {subMesh});
    assert(source.GetGeneration() == 2);
    source.SetMaterialSlotNames({"surface"});
    infernux::MaterialSlotData material;
    material.baseColor = {0.1f, 0.2f, 0.3f, 0.4f};
    material.emissionColor = {0.5f, 0.6f, 0.7f, 0.8f};
    material.metallic = 0.9f;
    material.sourceId = "material/surface";
    material.materialGuid = "abcdabcdabcdabcdabcdabcdabcdabcd";
    material.smoothness = 0.65f;
    material.opacity = 0.4f;
    material.alphaMode = infernux::ModelAlphaMode::Mask;
    material.alphaCutoff = 0.37f;
    material.doubleSided = true;
    material.textureGuids[static_cast<size_t>(infernux::ModelTexture::BaseColor)] = "11111111111111111111111111111111";
    material.textureUvSets[static_cast<size_t>(infernux::ModelTexture::BaseColor)] = 1;
    auto &baseSampler = material.textureSamplers[static_cast<size_t>(infernux::ModelTexture::BaseColor)];
    baseSampler.minFilter = infernux::MaterialSamplerFilter::Nearest;
    baseSampler.magFilter = infernux::MaterialSamplerFilter::Linear;
    baseSampler.mipFilter = infernux::MaterialSamplerFilter::Nearest;
    baseSampler.addressU = infernux::MaterialSamplerAddress::Clamp;
    baseSampler.addressV = infernux::MaterialSamplerAddress::Mirror;
    source.SetMaterialSlotData({material});
    source.SetNodeNames({"root", "child"});

    // Meshes without a source hierarchy remain valid current geometry-only payloads.
    constexpr const char *SourceHash = "0123456789abcdef";
    const auto geometryOnly = infernux::MeshArtifact::Serialize(source, SourceHash);
    assert(infernux::MeshArtifact::HasCurrentHeader(std::string_view(geometryOnly).substr(0, 32)));
    std::string obsoleteHeader = geometryOnly;
    obsoleteHeader.erase(std::string_view("INXMESHART").size() + sizeof(uint32_t), sizeof(uint32_t));
    RewriteChecksum(obsoleteHeader);
    assert(!infernux::MeshArtifact::HasCurrentHeader(std::string_view(obsoleteHeader).substr(0, 32)));
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(obsoleteHeader, SourceHash); });
    auto geometryOnlyRestored = infernux::MeshArtifact::Deserialize(geometryOnly, SourceHash);
    assert(geometryOnlyRestored->GetModelNodes().empty());
    assert(geometryOnlyRestored->GetIndexFormat() == infernux::MeshIndexFormat::UInt16);
    assert(geometryOnlyRestored->GetVertices().front().texCoord1 == vertex.texCoord1);
    assert(infernux::MeshArtifact::Serialize(*geometryOnlyRestored, SourceHash) == geometryOnly);

    // Auto uses a real 16-bit Cook stream when the vertex domain permits it;
    // explicit UInt32 remains wider, while forced UInt16 rejects rather than
    // truncating an oversized source index.
    auto force32 = source;
    force32.SetIndexFormat(infernux::MeshIndexFormat::UInt32);
    const auto uint32Bytes = infernux::MeshArtifact::Serialize(force32, SourceHash);
    assert(uint32Bytes.size() == geometryOnly.size() + source.GetIndexCount() * sizeof(uint16_t));
    assert(infernux::MeshArtifact::Deserialize(uint32Bytes, SourceHash)->GetIndexFormat() ==
           infernux::MeshIndexFormat::UInt32);
    std::string legacyUInt32 = uint32Bytes;
    const auto geometryMarker = DowngradeGeo3ToGeo2(legacyUInt32, source.GetVertexCount());
    legacyUInt32.erase(geometryMarker, 12); // Old artifacts begin directly with vertex count.
    RewriteChecksum(legacyUInt32);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(legacyUInt32, SourceHash); });
    std::string geo1UInt32 = uint32Bytes;
    const auto geo1Marker = DowngradeGeo3ToGeo2(geo1UInt32, source.GetVertexCount());
    WriteU32(geo1UInt32, geo1Marker, 0x314f4547U);
    geo1UInt32.erase(geo1Marker + 8, 4); // GEO1 had no compression field.
    RewriteChecksum(geo1UInt32);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(geo1UInt32, SourceHash); });
    infernux::InxMesh tooLarge("uint16-overflow");
    std::vector<infernux::Vertex> largeVertices(65537);
    infernux::SubMesh largeSub;
    largeSub.vertexCount = static_cast<uint32_t>(largeVertices.size());
    largeSub.indexCount = 3;
    tooLarge.SetData(std::move(largeVertices), {0, 65536, 0}, {largeSub});
    tooLarge.SetIndexFormat(infernux::MeshIndexFormat::UInt16);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Serialize(tooLarge, SourceHash); });
    tooLarge.SetIndexFormat(infernux::MeshIndexFormat::Auto);
    const auto largeBytes = infernux::MeshArtifact::Serialize(tooLarge, SourceHash);
    const auto largeRoundTrip = infernux::MeshArtifact::Deserialize(largeBytes, SourceHash);
    assert(largeRoundTrip->GetIndexFormat() == infernux::MeshIndexFormat::UInt32);
    assert(largeRoundTrip->GetIndices()[1] == 65536U);

    // Geometry compression is semantic quantization in the cooked vertex
    // stream, not a metadata flag or a generic archive wrapper. Every level
    // has a distinct byte budget and a measured positional error envelope.
    infernux::InxMesh compressionProbe("compression-probe");
    std::vector<infernux::Vertex> compressionVertices(1000);
    std::vector<uint32_t> compressionIndices(999);
    for (size_t i = 0; i < compressionVertices.size(); ++i) {
        auto &sample = compressionVertices[i];
        const float t = static_cast<float>(i) / 999.0f;
        sample.pos = {t * 123.0f - 31.0f, std::sin(t * 9.0f) * 7.0f, std::cos(t * 5.0f) * 3.0f};
        sample.normal = glm::normalize(glm::vec3(t + 0.1f, 1.0f, 0.5f - t));
        sample.tangent = glm::vec4(glm::normalize(glm::vec3(1.0f - t, 0.2f, t + 0.1f)), i & 1 ? -1.0f : 1.0f);
        sample.color = {t * 2.0f, 0.25f + t, 1.5f - t};
        sample.texCoord = {-2.0f + t * 4.0f, 5.0f - t * 3.0f};
        sample.boneIndices = {0, 7, 19, 31};
        sample.boneWeights = {0.4f, 0.3f, 0.2f, 0.1f};
        if (i < compressionIndices.size())
            compressionIndices[i] = static_cast<uint32_t>(i);
    }
    infernux::SubMesh compressionSub;
    compressionSub.vertexCount = static_cast<uint32_t>(compressionVertices.size());
    compressionSub.indexCount = static_cast<uint32_t>(compressionIndices.size());
    compressionProbe.SetData(compressionVertices, compressionIndices, {compressionSub});
    std::array<size_t, 4> compressedSizes{};
    std::array<float, 4> maximumPositionError{};
    for (const auto level : {infernux::MeshCompression::Off, infernux::MeshCompression::Low,
                             infernux::MeshCompression::Medium, infernux::MeshCompression::High}) {
        compressionProbe.SetCompression(level);
        const auto encoded = infernux::MeshArtifact::Serialize(compressionProbe, SourceHash);
        const auto decoded = infernux::MeshArtifact::Deserialize(encoded, SourceHash);
        const size_t slot = static_cast<size_t>(level);
        compressedSizes[slot] = encoded.size();
        assert(decoded->GetCompression() == level);
        assert(decoded->GetIndices() == compressionIndices);
        for (size_t i = 0; i < compressionVertices.size(); ++i) {
            maximumPositionError[slot] = std::max(
                maximumPositionError[slot], glm::length(decoded->GetVertices()[i].pos - compressionVertices[i].pos));
            assert(decoded->GetVertices()[i].boneIndices == compressionVertices[i].boneIndices);
            if (level != infernux::MeshCompression::Off) {
                assert(std::abs(glm::length(decoded->GetVertices()[i].normal) - 1.0f) < 1.0e-5f);
                assert(std::abs(glm::length(glm::vec3(decoded->GetVertices()[i].tangent)) - 1.0f) < 1.0e-5f);
                assert(decoded->GetVertices()[i].tangent.w == compressionVertices[i].tangent.w);
            }
        }
    }
    assert(compressedSizes[0] > compressedSizes[1] && compressedSizes[1] > compressedSizes[2] &&
           compressedSizes[2] > compressedSizes[3]);
    assert(maximumPositionError[1] < 0.001f);
    assert(maximumPositionError[2] < 0.01f);
    assert(maximumPositionError[3] < 0.2f);

    // normal_mode=None / tangent_mode=None uses strict zero channels in the
    // fixed Vertex layout. GEO3 must preserve absence instead of fabricating
    // a basis or rejecting a legal imported model.
    infernux::InxMesh missingBasis("missing-basis");
    auto missingBasisVertex = defaultVertex;
    missingBasisVertex.normal = glm::vec3(0.0f);
    missingBasisVertex.tangent = glm::vec4(0.0f);
    infernux::SubMesh missingBasisSub;
    missingBasisSub.vertexCount = 1;
    missingBasisSub.indexCount = 1;
    missingBasis.SetData({missingBasisVertex}, {0}, {missingBasisSub});
    missingBasis.SetCompression(infernux::MeshCompression::High);
    const auto missingBasisBytes = infernux::MeshArtifact::Serialize(missingBasis, SourceHash);
    const auto missingBasisRoundTrip = infernux::MeshArtifact::Deserialize(missingBasisBytes, SourceHash);
    assert(missingBasisRoundTrip->GetVertices()[0].normal == glm::vec3(0.0f));
    assert(missingBasisRoundTrip->GetVertices()[0].tangent == glm::vec4(0.0f));

    // Finite float endpoints whose span exceeds FLT_MAX must still quantize
    // through double precision without producing infinity.
    infernux::InxMesh extremeRange("extreme-range");
    auto extremeLeft = defaultVertex;
    auto extremeRight = defaultVertex;
    extremeLeft.pos.x = -std::numeric_limits<float>::max();
    extremeRight.pos.x = std::numeric_limits<float>::max();
    infernux::SubMesh extremeSub;
    extremeSub.vertexCount = 2;
    extremeSub.indexCount = 2;
    extremeRange.SetData({extremeLeft, extremeRight}, {0, 1}, {extremeSub});
    extremeRange.SetCompression(infernux::MeshCompression::High);
    const auto extremeBytes = infernux::MeshArtifact::Serialize(extremeRange, SourceHash);
    const auto extremeDecoded = infernux::MeshArtifact::Deserialize(extremeBytes, SourceHash);
    assert(std::isfinite(extremeDecoded->GetVertices()[0].pos.x));
    assert(std::isfinite(extremeDecoded->GetVertices()[1].pos.x));
    assert(extremeDecoded->GetVertices()[0].pos.x < 0.0f);
    assert(extremeDecoded->GetVertices()[1].pos.x > 0.0f);

    // A valid checksum cannot bless hostile GEO3 range metadata. The encoded
    // stream starts after marker/index/compression/vertex-count/byte-count.
    compressionProbe.SetCompression(infernux::MeshCompression::High);
    std::string reversedRange = infernux::MeshArtifact::Serialize(compressionProbe, SourceHash);
    const auto compressedMarker = reversedRange.find("GEO3");
    assert(compressedMarker != std::string::npos);
    const size_t firstRange = compressedMarker + 5U * sizeof(uint32_t);
    WriteFloat(reversedRange, firstRange, 2.0f);
    WriteFloat(reversedRange, firstRange + sizeof(float), -2.0f);
    RewriteChecksum(reversedRange);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(reversedRange, SourceHash); });

    infernux::ImportedModelNode root, pivot, child;
    root.name = "Assembly";
    root.localTransform[3] = {2.0f, 3.0f, 4.0f, 1.0f};
    pivot.name = "Empty pivot";
    pivot.parentIndex = 0;
    pivot.localTransform[0][0] = -2.0f;
    child.name = "child";
    child.parentIndex = 1;
    child.nodeGroup = 1;
    child.visible = false;
    child.localTransform[3] = {0.0f, 5.0f, 0.0f, 1.0f};
    source.SetModelNodes({root, pivot, child});
    auto invalidChild = child;
    invalidChild.parentIndex = 2;
    RequireInvalid([&] { source.SetModelNodes({root, pivot, invalidChild}); });
    invalidChild = child;
    invalidChild.nodeGroup = 2;
    RequireInvalid([&] { source.SetModelNodes({root, pivot, invalidChild}); });
    invalidChild = child;
    invalidChild.localTransform[0][0] = std::numeric_limits<float>::quiet_NaN();
    RequireInvalid([&] { source.SetModelNodes({root, pivot, invalidChild}); });
    RequireInvalid([&] { source.SetModelNodes({root, pivot, child, child}); });
    assert(source.GetModelNodes().size() == 3); // Invalid publication does not replace it.

    const std::string bytes = infernux::MeshArtifact::Serialize(source, SourceHash);
    auto restored = infernux::MeshArtifact::Deserialize(bytes, SourceHash);
    assert(restored->GetName() == "artifact-probe");
    assert(restored->GetVertexCount() == 1);
    assert(restored->GetIndexCount() == 3);
    assert(restored->GetSubMeshCount() == 1);
    assert(restored->GetSubMesh(0).name == "triangle");
    assert(restored->GetMaterialSlotNames() == std::vector<std::string>{"surface"});
    assert(restored->GetNodeNames() == std::vector<std::string>({"root", "child"}));
    const auto &nodes = restored->GetModelNodes();
    assert(nodes.size() == 3);
    assert(nodes[0].name == root.name && nodes[0].parentIndex == -1 && nodes[0].nodeGroup == -1);
    assert(nodes[1].name == pivot.name && nodes[1].parentIndex == 0 && nodes[1].nodeGroup == -1);
    assert(nodes[2].parentIndex == 1 && nodes[2].nodeGroup == 1);
    assert(nodes[0].visible && nodes[1].visible && !nodes[2].visible);
    assert(nodes[0].localTransform == root.localTransform);
    assert(nodes[1].localTransform == pivot.localTransform);
    assert(nodes[2].localTransform == child.localTransform);
    assert(infernux::MeshArtifact::Serialize(*restored, SourceHash) == bytes);
    const auto &restoredVertex = restored->GetVertices().front();
    assert(NearlyEqual(restoredVertex.pos.x, 1.0f));
    assert(restoredVertex.boneIndices == glm::uvec4(1, 2, 3, 4));
    assert(NearlyEqual(restoredVertex.boneWeights.w, 0.1f));
    assert(NearlyEqual(restored->GetMaterialSlotData().front().metallic, 0.9f));
    assert(restored->GetMaterialSlotData().front().sourceId == material.sourceId);
    assert(restored->GetMaterialSlotData().front().materialGuid == material.materialGuid);
    assert(restored->GetMaterialSlotData().front().alphaMode == material.alphaMode);
    assert(restored->GetMaterialSlotData().front().alphaCutoff == material.alphaCutoff);
    assert(restored->GetMaterialSlotData().front().doubleSided == material.doubleSided);
    assert(restored->GetMaterialSlotData().front().textureGuids == material.textureGuids);
    assert(restored->GetMaterialSlotData().front().textureUvSets == material.textureUvSets);
    assert(restored->GetMaterialSlotData().front().textureSamplers == material.textureSamplers);

    std::string legacyTexturesV3 = bytes;
    DowngradeMaterialTexturesV4ToV3(legacyTexturesV3);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(legacyTexturesV3, SourceHash); });

    std::string legacyTexturesV2 = bytes;
    DowngradeMaterialTexturesV3(legacyTexturesV2, false);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(legacyTexturesV2, SourceHash); });

    std::string legacyTexturesV1 = bytes;
    DowngradeMaterialTexturesV3(legacyTexturesV1, true);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(legacyTexturesV1, SourceHash); });

    std::string invalidUvSet = bytes;
    const size_t textureMarker = invalidUvSet.find("MTX4");
    assert(textureMarker != std::string::npos);
    const size_t firstGuid = textureMarker + 2U * sizeof(uint32_t);
    const size_t firstUvSet = firstGuid + sizeof(uint32_t) + ReadU32(invalidUvSet, firstGuid);
    WriteU32(invalidUvSet, firstUvSet, 2U);
    RewriteChecksum(invalidUvSet);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(invalidUvSet, SourceHash); });

    std::string invalidSampler = bytes;
    const size_t samplerMarker = invalidSampler.find("MTX4");
    const size_t samplerGuid = samplerMarker + 2U * sizeof(uint32_t);
    const size_t firstSampler =
        samplerGuid + sizeof(uint32_t) + ReadU32(invalidSampler, samplerGuid) + sizeof(uint32_t);
    WriteU32(invalidSampler, firstSampler, 3U);
    RewriteChecksum(invalidSampler);
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(invalidSampler, SourceHash); });

    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(bytes, "different-source"); });

    std::string corrupted = bytes;
    corrupted[corrupted.size() / 2] ^= 0x5a;
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(corrupted, SourceHash); });
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(bytes.substr(0, bytes.size() - 1), SourceHash); });
    RequireInvalid([&] { (void)infernux::MeshArtifact::Serialize(source, {}); });

    source.SetGuid("original-model-guid");
    source.SetFilePath("Assets/original.obj");
    const auto authoredBytes = infernux::MeshArtifact::SerializeSource(source);
    auto authored = infernux::MeshArtifact::DeserializeSource(authoredBytes);
    assert(authored->GetGuid().empty());
    assert(authored->GetFilePath().empty());
    assert(authored->GetVertices().front().normal == vertex.normal);
    assert(authored->GetVertices().front().tangent == vertex.tangent);
    assert(authored->GetVertices().front().texCoord == vertex.texCoord);
    assert(authored->GetIndices() == source.GetIndices());
    assert(authored->GetMaterialSlotNames() == source.GetMaterialSlotNames());
    assert(infernux::MeshArtifact::SerializeSource(*authored) == authoredBytes);
    RequireInvalid([&] { (void)infernux::MeshArtifact::DeserializeSource(bytes); });
    RequireInvalid([&] { (void)infernux::MeshArtifact::Deserialize(authoredBytes, SourceHash); });
    auto skinned = std::make_shared<infernux::InxSkinnedMesh>();
    skinned->baseVertices = source.GetVertices();
    skinned->indices = source.GetIndices();
    source.SetSkinnedData(skinned);
    RequireInvalid([&] { (void)infernux::MeshArtifact::SerializeSource(source); });

    // Imported assets store node-local geometry once. The ordinary mesh view
    // remains model-space for existing render, bounds and picking consumers.
    infernux::InxMesh localModel("local-model");
    localModel.SetNodeNames({"root", "child"});
    infernux::MeshMorphTarget morph;
    morph.name = "Raise";
    morph.defaultWeight = 0.25f;
    morph.positionDeltas = {{1.0f, 2.0f, 3.0f}};
    morph.normalDeltas = {{0.0f, 0.25f, 0.0f}};
    morph.tangentDeltas = {{0.0f, 0.0f, 0.5f}};
    localModel.SetModelData({vertex}, {0, 0, 0}, {subMesh}, {root, pivot, child}, {morph});
    assert(localModel.GetGeneration() == 1);
    assert(localModel.GetModelSourceGeometry()->vertices[0].pos == vertex.pos);
    assert(localModel.GetVertices()[0].pos == glm::vec3(0, 10, 7));
    assert(localModel.GetVertices()[0].normal == vertex.normal);
    assert(localModel.GetVertices()[0].tangent == glm::vec4(-1, 0, 0, 1));
    assert(localModel.GetModelSourceGeometry()->morphTargets.size() == 1);
    assert(localModel.GetModelSourceGeometry()->morphTargets[0].name == morph.name);
    assert(localModel.GetModelSourceGeometry()->morphTargets[0].positionDeltas == morph.positionDeltas);
    assert(localModel.GetMorphTargets().size() == 1);
    assert(localModel.GetMorphTargets()[0].positionDeltas[0] == glm::vec3(-2, 2, 3));
    assert(localModel.GetSubMesh(0).boundsMin == localModel.GetVertices()[0].pos);
    const auto localBytes = infernux::MeshArtifact::SerializeSource(localModel);
    auto localRestored = infernux::MeshArtifact::DeserializeSource(localBytes);
    assert(localRestored->GetModelSourceGeometry()->vertices[0].pos == vertex.pos);
    assert(localRestored->GetVertices()[0].pos == localModel.GetVertices()[0].pos);
    assert(localRestored->GetModelSourceGeometry()->morphTargets.size() == 1);
    assert(localRestored->GetModelSourceGeometry()->morphTargets[0].positionDeltas == morph.positionDeltas);
    assert(localRestored->GetMorphTargets().size() == localModel.GetMorphTargets().size());
    assert(localRestored->GetMorphTargets()[0].positionDeltas == localModel.GetMorphTargets()[0].positionDeltas);
    assert(infernux::MeshArtifact::SerializeSource(*localRestored) == localBytes);
    auto bakedOnly = localModel;
    bakedOnly.SetData(localModel.GetVertices(), localModel.GetIndices(), localModel.GetSubMeshes(),
                      localModel.GetMorphTargets());
    assert(!bakedOnly.GetModelSourceGeometry());
    // Baked/local hierarchy sections carry the same visibility metadata: no
    // duplicated vertex payload and no format-dependent authoring behavior.
    const auto bakedBytes = infernux::MeshArtifact::SerializeSource(bakedOnly);
    assert(bakedBytes.size() == localBytes.size());
    assert(!infernux::MeshArtifact::DeserializeSource(bakedBytes)->GetModelSourceGeometry());

    // Read/Write-disabled meshes preserve render metadata while dropping both
    // combined and source-local CPU geometry. Counts stay authoritative and
    // scripting gets a clear error instead of an implicit GPU readback.
    auto releaseProbe = localModel;
    releaseProbe.SetCpuReadable(false);
    const auto releaseVertexCount = releaseProbe.GetVertexCount();
    const auto releaseIndexCount = releaseProbe.GetIndexCount();
    const auto releaseBoundsMin = releaseProbe.GetBoundsMin();
    const auto releaseSubMeshes = releaseProbe.GetSubMeshCount();
    const size_t bytesBeforeRelease = releaseProbe.GetRuntimeMemoryBytes();
    const size_t releasedBytes = releaseProbe.ReleaseCpuGeometry();
    assert(releasedBytes > 0 && releaseProbe.GetRuntimeMemoryBytes() + releasedBytes == bytesBeforeRelease);
    assert(!releaseProbe.HasCpuGeometry() && releaseProbe.GetModelSourceGeometry());
    assert(releaseProbe.GetModelSourceGeometry()->vertices.empty());
    assert(releaseProbe.GetModelSourceGeometry()->vertexCount == releaseVertexCount);
    assert(releaseProbe.GetVertexCount() == releaseVertexCount);
    assert(releaseProbe.GetIndexCount() == releaseIndexCount);
    assert(releaseProbe.GetSubMeshCount() == releaseSubMeshes);
    assert(releaseProbe.GetBoundsMin() == releaseBoundsMin);
    std::cout << "Mesh CPU release bytes: before=" << bytesBeforeRelease
              << " after=" << releaseProbe.GetRuntimeMemoryBytes() << " released=" << releasedBytes << '\n';
    bool readRejected = false;
    try {
        releaseProbe.RequireCpuReadable("Mesh.vertex_buffer");
    } catch (const std::runtime_error &error) {
        readRejected = std::string(error.what()).find("Read/Write") != std::string::npos;
    }
    assert(readRejected);
    bool nodeCopyRejected = false;
    try {
        (void)releaseProbe.CreateModelNodeCopy({"root", "pivot", "child"});
    } catch (const std::runtime_error &error) {
        nodeCopyRejected = std::string(error.what()).find("released mesh CPU geometry") != std::string::npos;
    }
    assert(nodeCopyRejected);

    // Keep a representative flat render mesh measurement separate from the
    // hierarchy contract above. This is the payload a packaged Player can
    // discard immediately after its authoritative GPU upload.
    std::vector<infernux::Vertex> memoryVertices(4096);
    std::vector<uint32_t> memoryIndices(12288, 0);
    infernux::SubMesh memorySubMesh;
    memorySubMesh.vertexStart = 0;
    memorySubMesh.vertexCount = static_cast<uint32_t>(memoryVertices.size());
    memorySubMesh.indexStart = 0;
    memorySubMesh.indexCount = static_cast<uint32_t>(memoryIndices.size());
    infernux::InxMesh memoryProbe;
    memoryProbe.SetData(std::move(memoryVertices), std::move(memoryIndices), {memorySubMesh});
    memoryProbe.SetCpuReadable(false);
    const size_t flatBytesBeforeRelease = memoryProbe.GetRuntimeMemoryBytes();
    const size_t flatReleasedBytes = memoryProbe.ReleaseCpuGeometry();
    assert(memoryProbe.GetVertexCount() == 4096);
    assert(memoryProbe.GetIndexCount() == 12288);
    assert(memoryProbe.GetSubMeshCount() == 1);
    assert(!memoryProbe.HasCpuGeometry());
    assert(flatReleasedBytes > 0 && memoryProbe.GetRuntimeMemoryBytes() + flatReleasedBytes == flatBytesBeforeRelease);
    std::cout << "Flat Player mesh CPU release bytes: before=" << flatBytesBeforeRelease
              << " after=" << memoryProbe.GetRuntimeMemoryBytes() << " released=" << flatReleasedBytes << '\n';

    const auto published = localModel.GetGeometrySnapshot();
    const auto localPublished = localModel.GetModelSourceGeometry();
    auto invalidSubMesh = subMesh;
    invalidSubMesh.vertexCount = 2;
    RequireInvalid([&] { localModel.SetModelData({vertex}, {0, 0, 0}, {invalidSubMesh}, {root, pivot, child}); });
    RequireInvalid([&] { localModel.SetModelData({vertex}, {1, 0, 0}, {subMesh}, {root, pivot, child}); });
    RequireInvalid([&] { localModel.SetModelData({vertex}, {}, {}, {root, pivot, child}); });
    RequireInvalid([&] { localModel.SetModelData({vertex}, {0, 0, 0}, {subMesh}, {root}); });
    auto invalidMorph = morph;
    invalidMorph.positionDeltas.clear();
    RequireInvalid(
        [&] { localModel.SetModelData({vertex}, {0, 0, 0}, {subMesh}, {root, pivot, child}, {invalidMorph}); });
    assert(localModel.GetGeometrySnapshot() == published);
    assert(localModel.GetModelSourceGeometry() == localPublished);
    assert(localModel.GetGeneration() == 1);

    // Collapsed authored scale must not destroy the source or yield inverse
    // matrix NaNs. Editing node transforms re-derives from the same local data.
    auto flattened = pivot;
    flattened.localTransform[2][2] = 0.0f;
    localModel.SetModelNodes({root, flattened, child});
    assert(localModel.GetVertices()[0].pos == glm::vec3(0, 10, 4));
    assert(localModel.GetVertices()[0].normal == glm::vec3(0));
    assert(localModel.GetModelSourceGeometry()->vertices[0].pos == vertex.pos);
    assert(localModel.GetGeneration() == 2);
    localModel.SetModelNodes({root, pivot, child});
    assert(localModel.GetVertices()[0].pos == published->vertices[0].pos);
    assert(localPublished->vertices[0].pos == vertex.pos);

    infernux::InxMesh reloadTarget("before-reload");
    reloadTarget.SetGuid("persistent-asset-guid");
    reloadTarget.SetData({}, {}, {});
    reloadTarget.ReplaceImportedContent(localModel);
    assert(reloadTarget.GetGuid() == "persistent-asset-guid");
    assert(reloadTarget.GetGeneration() == 2);
    assert(reloadTarget.GetGeometrySnapshot() == localModel.GetGeometrySnapshot());
    assert(reloadTarget.GetModelSourceGeometry() == localModel.GetModelSourceGeometry());
    auto editedVertex = reloadTarget.GetVertices()[0];
    editedVertex.pos.x = 20.0f;
    reloadTarget.UpdateVertexRange(0, {editedVertex});
    assert(!reloadTarget.GetModelSourceGeometry());
    assert(localModel.GetVertices()[0].pos.x == 0.0f);

    std::cout << "Mesh artifact tests passed\n";
    return 0;
}
