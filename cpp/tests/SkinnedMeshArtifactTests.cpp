// This executable validates production Release artifacts as well as Debug builds.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <function/scene/SkinPoseHistory.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
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
    assert(bytes.size() >= sizeof(uint64_t));
    const size_t offset = bytes.size() - sizeof(uint64_t);
    const uint64_t checksum = Fnv1a64(std::string_view(bytes).substr(0, offset));
    for (unsigned int shift = 0; shift < 64; shift += 8)
        bytes[offset + shift / 8] = static_cast<char>((checksum >> shift) & 0xffU);
}

void WriteU32(std::string &bytes, size_t offset, uint32_t value)
{
    assert(offset <= bytes.size() && sizeof(value) <= bytes.size() - offset);
    for (unsigned int shift = 0; shift < 32; shift += 8)
        bytes[offset + shift / 8] = static_cast<char>((value >> shift) & 0xffU);
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

bool NearlyEqual(float left, float right)
{
    return std::abs(left - right) < 1.0e-6f;
}
} // namespace

int main()
{
    infernux::InxSkinnedMesh source;
    source.scaleFactor = 0.01f;

    infernux::Vertex vertex;
    vertex.pos = {1.0f, 2.0f, 3.0f};
    vertex.normal = {0.0f, 1.0f, 0.0f};
    vertex.tangent = {1.0f, 0.0f, 0.0f, 1.0f};
    vertex.color = {0.2f, 0.4f, 0.6f};
    vertex.texCoord = {0.25f, 0.75f};
    vertex.texCoord1 = {0.75f, 0.25f};
    source.baseVertices.push_back(vertex);

    infernux::SkinInfluence influence;
    influence.boneIndex[0] = 0;
    influence.weight[0] = 1.0f;
    source.influences.push_back(influence);
    source.indices = {0, 0, 0};

    infernux::SubMesh subMesh;
    subMesh.indexCount = 3;
    subMesh.vertexCount = 1;
    subMesh.boundsMin = vertex.pos;
    subMesh.boundsMax = vertex.pos;
    subMesh.name = "body";
    source.subMeshes.push_back(subMesh);

    infernux::SkinnedRuntimeNode node;
    node.name = "Root";
    source.skeleton.nodeByName.emplace(node.name, 0);
    source.skeleton.nodes.push_back(node);

    infernux::SkinnedRuntimeBone bone;
    bone.name = "Root";
    bone.nodeIndex = 0;
    source.skeleton.boneByName.emplace(bone.name, 0);
    source.skeleton.bones.push_back(bone);

    infernux::SkinnedRuntimeTrack track;
    track.nodeIndex = 0;
    track.positions = {{0.0, {0.0f, 0.0f, 0.0f}}, {10.0, {2.0f, 0.0f, 0.0f}}};
    track.rotations = {{0.0, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)}, {10.0, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)}};
    track.scales = {{0.0, {1.0f, 1.0f, 1.0f}}, {10.0, {1.0f, 1.0f, 1.0f}}};
    infernux::SkinnedRuntimeAnimation animation;
    animation.name = "Move";
    animation.durationTicks = 10.0;
    animation.ticksPerSecond = 20.0;
    animation.trackByNodeIndex = {0};
    animation.tracks.push_back(track);
    source.animations.push_back(animation);
    source.NormalizeInfluences();

    infernux::MeshMorphTarget morph;
    morph.name = "Blink";
    morph.defaultWeight = 0.35f;
    morph.positionDeltas = {{0.0f, 0.1f, 0.0f}};
    morph.normalDeltas = {{0.0f, 0.0f, 0.2f}};
    source.morphTargets.push_back(morph);

    constexpr std::string_view SourceHash = "0123456789abcdef";
    const std::string bytes = infernux::SkinnedMeshArtifact::Serialize(source, SourceHash);
    assert(infernux::SkinnedMeshArtifact::HasCurrentHeader(std::string_view(bytes).substr(0, 32)));
    std::string obsoleteHeader = bytes;
    obsoleteHeader.erase(std::string_view("INXSKINAR").size() + sizeof(uint32_t), sizeof(uint32_t));
    RewriteChecksum(obsoleteHeader);
    assert(!infernux::SkinnedMeshArtifact::HasCurrentHeader(std::string_view(obsoleteHeader).substr(0, 32)));
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(obsoleteHeader, SourceHash); });
    auto restored = infernux::SkinnedMeshArtifact::Deserialize(bytes, SourceHash);
    assert(restored);
    assert(restored->baseVertices.size() == 1);
    assert(restored->indices == std::vector<uint32_t>({0, 0, 0}));
    assert(restored->skeleton.nodes.size() == 1 && restored->skeleton.nodeByName.at("Root") == 0);
    assert(restored->skeleton.bones.size() == 1 && restored->skeleton.boneByName.at("Root") == 0);
    assert(restored->animations.size() == 1);
    assert(restored->animations.front().trackByNodeIndex.at(0) == 0);
    assert(restored->baseVertices.front().texCoord1 == vertex.texCoord1);
    assert(restored->morphTargets.size() == 1);
    assert(restored->morphTargets[0].name == morph.name);
    assert(restored->morphTargets[0].defaultWeight == morph.defaultWeight);
    assert(restored->morphTargets[0].positionDeltas == morph.positionDeltas);
    assert(restored->morphTargets[0].normalDeltas == morph.normalDeltas);
    assert(restored->indexFormat == infernux::MeshIndexFormat::UInt16);

    // Humanoid mapping/report is part of the sole current RIG2 tail and is
    // readable without consulting editor metadata. Invalid auto mappings are
    // intentionally publishable so the Inspector can explain how to fix them.
    auto humanoidSource = source;
    humanoidSource.humanoid.enabled = true;
    humanoidSource.humanoid.requiredBonesValid = false;
    humanoidSource.humanoid.hierarchyValid = false;
    humanoidSource.humanoid.referencePoseValid = false;
    humanoidSource.humanoid.bones = {{"hips", 0}};
    humanoidSource.humanoid.issues = {{"missing_required_bone", "head", "required humanoid bone is unmapped"}};
    const auto humanoidBytes = infernux::SkinnedMeshArtifact::Serialize(humanoidSource, SourceHash);
    const auto humanoidRestored = infernux::SkinnedMeshArtifact::Deserialize(humanoidBytes, SourceHash);
    assert(humanoidRestored->humanoid.enabled && !humanoidRestored->humanoid.IsValid());
    assert((humanoidRestored->humanoid.bones == std::vector<std::pair<std::string, int>>({{"hips", 0}})));
    assert(humanoidRestored->humanoid.issues.size() == 1);
    assert(humanoidRestored->humanoid.issues.front().code == "missing_required_bone");

    auto force32 = source;
    force32.indexFormat = infernux::MeshIndexFormat::UInt32;
    const auto uint32Bytes = infernux::SkinnedMeshArtifact::Serialize(force32, SourceHash);
    assert(uint32Bytes.size() == bytes.size() + source.indices.size() * sizeof(uint16_t));
    assert(infernux::SkinnedMeshArtifact::Deserialize(uint32Bytes, SourceHash)->indexFormat ==
           infernux::MeshIndexFormat::UInt32);
    std::string legacyUInt32 = uint32Bytes;
    const auto geometryMarker = DowngradeGeo3ToGeo2(legacyUInt32, source.baseVertices.size());
    legacyUInt32.erase(geometryMarker, 12);
    RewriteChecksum(legacyUInt32);
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(legacyUInt32, SourceHash); });
    std::string geo1UInt32 = uint32Bytes;
    const auto geo1Marker = DowngradeGeo3ToGeo2(geo1UInt32, source.baseVertices.size());
    WriteU32(geo1UInt32, geo1Marker, 0x314f4547U);
    geo1UInt32.erase(geo1Marker + 8, 4);
    RewriteChecksum(geo1UInt32);
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(geo1UInt32, SourceHash); });
    auto tooLarge = source;
    tooLarge.baseVertices.assign(65537, source.baseVertices.front());
    tooLarge.influences.assign(65537, source.influences.front());
    tooLarge.indices = {0, 65536, 0};
    tooLarge.subMeshes.front().vertexCount = 65537;
    tooLarge.subMeshes.front().indexCount = 3;
    tooLarge.morphTargets.clear();
    tooLarge.indexFormat = infernux::MeshIndexFormat::UInt16;
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Serialize(tooLarge, SourceHash); });
    tooLarge.indexFormat = infernux::MeshIndexFormat::Auto;
    const auto largeBytes = infernux::SkinnedMeshArtifact::Serialize(tooLarge, SourceHash);
    const auto largeRoundTrip = infernux::SkinnedMeshArtifact::Deserialize(largeBytes, SourceHash);
    assert(largeRoundTrip->indexFormat == infernux::MeshIndexFormat::UInt32);
    assert(largeRoundTrip->indices[1] == 65536U);
    auto compressedSkin = source;
    compressedSkin.compression = infernux::MeshCompression::Medium;
    const auto compressedSkinBytes = infernux::SkinnedMeshArtifact::Serialize(compressedSkin, SourceHash);
    const auto compressedSkinRoundTrip = infernux::SkinnedMeshArtifact::Deserialize(compressedSkinBytes, SourceHash);
    assert(compressedSkinRoundTrip->compression == infernux::MeshCompression::Medium);
    assert(glm::length(compressedSkinRoundTrip->baseVertices[0].pos - source.baseVertices[0].pos) < 0.01f);
    assert(compressedSkinRoundTrip->baseVertices[0].boneIndices == source.baseVertices[0].boneIndices);

    auto compressedMissingBasis = source;
    compressedMissingBasis.compression = infernux::MeshCompression::High;
    compressedMissingBasis.baseVertices[0].normal = glm::vec3(0.0f);
    compressedMissingBasis.baseVertices[0].tangent = glm::vec4(0.0f);
    const auto compressedMissingBasisBytes =
        infernux::SkinnedMeshArtifact::Serialize(compressedMissingBasis, SourceHash);
    const auto compressedMissingBasisRoundTrip =
        infernux::SkinnedMeshArtifact::Deserialize(compressedMissingBasisBytes, SourceHash);
    assert(compressedMissingBasisRoundTrip->baseVertices[0].normal == glm::vec3(0.0f));
    assert(compressedMissingBasisRoundTrip->baseVertices[0].tangent == glm::vec4(0.0f));

    // MPR1 uses an optional tail section instead of changing the Vertex ABI.
    // Checksum-correct hostile payloads must still fail at the publication
    // boundary, and the same rule must reject invalid live model data.
    const size_t morphOffset = bytes.rfind("MPR1");
    assert(morphOffset != std::string::npos);
    const size_t nameLengthOffset = morphOffset + 8;
    const size_t nameOffset = nameLengthOffset + sizeof(uint32_t);
    const size_t defaultWeightOffset = nameOffset + morph.name.size();
    const size_t firstPositionDeltaOffset = defaultWeightOffset + sizeof(float) + sizeof(uint32_t);
    for (const size_t corruptOffset : {defaultWeightOffset, firstPositionDeltaOffset}) {
        std::string nonFinite = bytes;
        WriteU32(nonFinite, corruptOffset, 0x7fc00000U);
        RewriteChecksum(nonFinite);
        RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(nonFinite, SourceHash); });
    }
    std::string emptyName = bytes;
    emptyName.erase(nameOffset, morph.name.size());
    WriteU32(emptyName, nameLengthOffset, 0U);
    RewriteChecksum(emptyName);
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(emptyName, SourceHash); });

    auto invalidMorphModel = source;
    invalidMorphModel.morphTargets[0].defaultWeight = std::numeric_limits<float>::quiet_NaN();
    assert(!invalidMorphModel.HasValidMorphTargets());
    assert(!invalidMorphModel.IsAssetPayloadValid());
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Serialize(invalidMorphModel, SourceHash); });
    invalidMorphModel = source;
    invalidMorphModel.morphTargets[0].positionDeltas[0].x = std::numeric_limits<float>::infinity();
    assert(!invalidMorphModel.HasValidMorphTargets());
    assert(!invalidMorphModel.IsAssetPayloadValid());
    invalidMorphModel = source;
    invalidMorphModel.morphTargets[0].name.clear();
    assert(!invalidMorphModel.HasValidMorphTargets());
    assert(!invalidMorphModel.IsAssetPayloadValid());
    assert(NearlyEqual(restored->animations.front().DurationSeconds(), 0.5f));
    assert(restored->BuildGpuBonePalette({"Move", 0.25f}).size() == 1);
    assert(restored->GetRuntimeMemoryBytes() > sizeof(infernux::InxSkinnedMesh));

    // Authoring reads the shared InxMesh bounds before the first Animator
    // update, while rendering and culling consume the skinned bind palette.
    // Both publications must use the same converted unit space exactly once.
    // A mismatch here makes a correctly framed character jump in scale as
    // soon as Play publishes its first pose.
    const auto bindPalette = restored->BuildGpuBonePalette({});
    glm::vec3 bindBoundsMin;
    glm::vec3 bindBoundsMax;
    assert(restored->ComputeSkinnedBounds(bindPalette, bindBoundsMin, bindBoundsMax));
    const auto sampledBindVertices = restored->SampleVertices({});
    assert(sampledBindVertices.size() == 1);
    assert(glm::length(sampledBindVertices.front().pos - bindBoundsMin) < 1.e-6f);
    assert(glm::length(sampledBindVertices.front().pos - bindBoundsMax) < 1.e-6f);
    assert(NearlyEqual(bindBoundsMin.x, source.baseVertices.front().pos.x * source.scaleFactor));
    assert(NearlyEqual(bindBoundsMin.y, source.baseVertices.front().pos.y * source.scaleFactor));
    assert(NearlyEqual(bindBoundsMin.z, source.baseVertices.front().pos.z * source.scaleFactor));

    // A separate animation-only model with the same skeleton must drive the
    // render model. This is the standard Mixamo "With Skin" model + "Without
    // Skin" animation workflow.
    infernux::InxSkinnedMesh animationOnly;
    animationOnly.guid = "animation-source";
    animationOnly.skeleton = source.skeleton;
    animationOnly.skeleton.bones.clear();
    animationOnly.skeleton.boneByName.clear();
    infernux::SkinnedRuntimeAnimation externalAnimation = animation;
    externalAnimation.name = "ExternalMove";
    externalAnimation.tracks.front().positions.back().second = {4.0f, 0.0f, 0.0f};
    animationOnly.animations.push_back(externalAnimation);
    assert(!animationOnly.IsValid());
    assert(animationOnly.IsAnimationSource());
    std::string compatibilityReason;
    assert(source.skeleton.IsAnimationCompatible(animationOnly.skeleton, externalAnimation, &compatibilityReason));
    assert(compatibilityReason.empty());
    const auto externalPalette = restored->BuildGpuBonePalette({"ExternalMove", 0.25f}, &animationOnly);
    assert(externalPalette.size() == 1);
    assert(NearlyEqual(externalPalette.front()[3].x, 0.02f));
    glm::vec3 animatedBoundsMin;
    glm::vec3 animatedBoundsMax;
    assert(restored->ComputeSkinnedBounds(externalPalette, animatedBoundsMin, animatedBoundsMax));
    assert(NearlyEqual(animatedBoundsMin.x, 0.03f) && NearlyEqual(animatedBoundsMax.x, 0.03f));
    assert(NearlyEqual(animatedBoundsMin.y, 0.02f) && NearlyEqual(animatedBoundsMax.y, 0.02f));
    assert(NearlyEqual(animatedBoundsMin.z, 0.03f) && NearlyEqual(animatedBoundsMax.z, 0.03f));

    const std::string animationBytes = infernux::SkinnedMeshArtifact::Serialize(animationOnly, SourceHash);
    const auto restoredAnimation = infernux::SkinnedMeshArtifact::Deserialize(animationBytes, SourceHash);
    assert(restoredAnimation && restoredAnimation->IsAnimationSource() && !restoredAnimation->IsValid());
    assert(restoredAnimation->animations.front().name == "ExternalMove");

    // An entirely renamed rig is not guessed from geometry. Symmetric limbs
    // and fingers require an explicit Avatar/Rig-style mapping.
    infernux::Skeleton renamedSkeleton = animationOnly.skeleton;
    renamedSkeleton.nodeByName.clear();
    renamedSkeleton.nodes.front().name = "CompletelyDifferentJointName";
    renamedSkeleton.nodeByName.emplace(renamedSkeleton.nodes.front().name, 0);
    infernux::SkinnedRuntimeAnimation renamedAnimation = externalAnimation;
    assert(!source.skeleton.IsAnimationCompatible(renamedSkeleton, renamedAnimation, &compatibilityReason));
    assert(compatibilityReason.find("exactly named target joint") != std::string::npos);

    // Two independently named and proportioned humanoid skeletons retarget by
    // their published slots only. This is the authoritative cross-skeleton
    // path consumed by SkinnedMeshRenderer/SkeletalAnimator; no node-name or
    // path heuristic is available to make this test pass.
    constexpr std::array<std::string_view, 15> requiredHumanoidSlots = {
        "hips",
        "spine",
        "head",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_upper_leg",
        "left_lower_leg",
        "left_foot",
        "right_upper_leg",
        "right_lower_leg",
        "right_foot",
    };
    const auto makeHumanoid = [&](std::string_view prefix, float segmentLength, bool renderable) {
        infernux::InxSkinnedMesh model;
        model.guid = std::string(prefix) + "-guid";
        model.scaleFactor = 1.0f;
        const std::array<int, 15> parents = {-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13};
        for (size_t index = 0; index < requiredHumanoidSlots.size(); ++index) {
            infernux::SkinnedRuntimeNode humanoidNode;
            humanoidNode.name = std::string(prefix) + "_joint_" + std::to_string(index);
            humanoidNode.parent = parents[index];
            if (humanoidNode.parent >= 0)
                humanoidNode.bindLocal[3] = glm::vec4(0.0f, segmentLength, 0.0f, 1.0f);
            humanoidNode.bindGlobal =
                humanoidNode.parent >= 0
                    ? model.skeleton.nodes[static_cast<size_t>(humanoidNode.parent)].bindGlobal * humanoidNode.bindLocal
                    : humanoidNode.bindLocal;
            model.skeleton.nodeByName.emplace(humanoidNode.name, static_cast<int>(index));
            model.skeleton.nodes.push_back(humanoidNode);
            model.humanoid.bones.emplace_back(std::string(requiredHumanoidSlots[index]), static_cast<int>(index));
            if (renderable) {
                infernux::SkinnedRuntimeBone humanoidBone;
                humanoidBone.name = humanoidNode.name;
                humanoidBone.nodeIndex = static_cast<int>(index);
                humanoidBone.inverseBind = glm::inverse(humanoidNode.bindGlobal);
                model.skeleton.boneByName.emplace(humanoidBone.name,
                                                  static_cast<uint32_t>(model.skeleton.bones.size()));
                model.skeleton.bones.push_back(humanoidBone);
            }
        }
        model.humanoid.enabled = true;
        model.humanoid.requiredBonesValid = true;
        model.humanoid.hierarchyValid = true;
        model.humanoid.referencePoseValid = true;
        return model;
    };
    auto humanoidTarget = makeHumanoid("Target", 2.0f, true);
    auto humanoidAnimationSource = makeHumanoid("Source", 0.75f, false);
    infernux::SkinnedRuntimeAnimation humanoidWave;
    humanoidWave.name = "Wave";
    humanoidWave.id = "wave-stable-id";
    humanoidWave.durationTicks = 1.0;
    humanoidWave.ticksPerSecond = 1.0;
    infernux::SkinnedRuntimeTrack humanoidArmTrack;
    humanoidArmTrack.nodeIndex = 3;
    humanoidArmTrack.rotations = {
        {0.0, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)},
        {1.0, glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f))},
    };
    humanoidWave.tracks.push_back(humanoidArmTrack);
    humanoidWave.trackByNodeIndex.assign(requiredHumanoidSlots.size(), -1);
    humanoidWave.trackByNodeIndex[3] = 0;
    humanoidAnimationSource.animations.push_back(humanoidWave);

    const auto humanoidMap = humanoidTarget.BuildRetargetMap(humanoidAnimationSource, humanoidWave);
    assert(humanoidMap.humanoidSlots);
    assert(humanoidMap.mappedNodes == requiredHumanoidSlots.size());
    assert(humanoidMap.exactNameMatches == 0);
    assert(humanoidMap.targetToSourceNode[3] == 3);
    assert(humanoidTarget.skeleton.nodes[3].name != humanoidAnimationSource.skeleton.nodes[3].name);
    assert(humanoidTarget.IsAnimationCompatible(humanoidAnimationSource, humanoidWave, &compatibilityReason));
    assert(compatibilityReason.empty());
    const infernux::SkinnedSampleRequest humanoidRequest{"wave-stable-id", 1.0f, false};
    const auto humanoidPalette = humanoidTarget.GetOrBuildGpuBonePalette(humanoidRequest, &humanoidAnimationSource);
    const auto humanoidPaletteCached =
        humanoidTarget.GetOrBuildGpuBonePalette(humanoidRequest, &humanoidAnimationSource);
    assert(humanoidPalette == humanoidPaletteCached);
    assert(humanoidPalette->size() == requiredHumanoidSlots.size());
    assert(std::abs((*humanoidPalette)[3][0][0]) < 1.0e-5f);
    assert(std::abs(std::abs((*humanoidPalette)[3][0][1]) - 1.0f) < 1.0e-5f);

    auto incompleteHumanoid = humanoidAnimationSource;
    incompleteHumanoid.humanoid.bones.erase(incompleteHumanoid.humanoid.bones.begin() + 2);
    assert(!humanoidTarget.IsAnimationCompatible(incompleteHumanoid, humanoidWave, &compatibilityReason));
    assert(compatibilityReason.find("head") != std::string::npos);
    auto genericAnimationSource = humanoidAnimationSource;
    genericAnimationSource.humanoid = {};
    assert(!humanoidTarget.IsAnimationCompatible(genericAnimationSource, humanoidWave, &compatibilityReason));
    assert(compatibilityReason.find("both assets") != std::string::npos);
    RequireInvalid([&] { (void)humanoidTarget.BuildGpuBonePalette(humanoidRequest, &genericAnimationSource); });

    // Exact joint identity wins over geometric guessing. Symmetric branches
    // cannot be safely disambiguated by bind-pose position when exporters use
    // different coordinate systems or proportions.
    infernux::Skeleton targetFork;
    infernux::Skeleton sourceFork;
    for (int index = 0; index < 3; ++index) {
        infernux::SkinnedRuntimeNode targetNode;
        infernux::SkinnedRuntimeNode sourceNode;
        targetNode.parent = sourceNode.parent = index == 0 ? -1 : 0;
        targetNode.name = index == 0 ? "TargetRoot" : (index == 1 ? "Left" : "Right");
        sourceNode.name = index == 0 ? "SourceRoot" : (index == 1 ? "Right" : "Left");
        if (index > 0) {
            const float x = index == 1 ? -1.0f : 1.0f;
            targetNode.bindLocal[3] = sourceNode.bindLocal[3] = glm::vec4(x, 1.0f, 0.0f, 1.0f);
            targetNode.bindGlobal = targetNode.bindLocal;
            sourceNode.bindGlobal = sourceNode.bindLocal;
        }
        targetFork.nodeByName.emplace(targetNode.name, index);
        sourceFork.nodeByName.emplace(sourceNode.name, index);
        targetFork.nodes.push_back(targetNode);
        sourceFork.nodes.push_back(sourceNode);
    }
    infernux::SkinnedRuntimeAnimation forkAnimation;
    forkAnimation.name = "Fork";
    forkAnimation.durationTicks = 1.0;
    forkAnimation.ticksPerSecond = 1.0;
    infernux::SkinnedRuntimeTrack forkTrack;
    forkTrack.nodeIndex = 1;
    forkTrack.rotations = {{0.0, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)}};
    forkAnimation.tracks.push_back(forkTrack);
    forkAnimation.trackByNodeIndex = {-1, 0, -1};
    const infernux::SkeletonRetargetMap forkMap = targetFork.BuildRetargetMap(sourceFork, forkAnimation);
    assert(forkMap.targetToSourceNode == std::vector<int>({-1, 2, 1}));
    assert(forkMap.exactNameMatches == 2);
    assert(forkMap.mappedAnimatedNodes == 1);
    assert(!forkMap.identicalTopology);
    assert(!targetFork.IsAnimationCompatible(sourceFork, forkAnimation, &compatibilityReason));
    assert(compatibilityReason.find("same mapped hierarchy") != std::string::npos);

    // Assimp expands FBX pivots into helper nodes in animation files. The
    // helper's channel must drive an exactly named deform joint even though
    // the render-model hierarchy does not contain that helper node.
    infernux::InxSkinnedMesh pivotTarget;
    pivotTarget.scaleFactor = 1.0f;
    infernux::SkinnedRuntimeNode targetRig;
    targetRig.name = "Rig";
    pivotTarget.skeleton.nodeByName.emplace(targetRig.name, 0);
    pivotTarget.skeleton.nodes.push_back(targetRig);
    infernux::SkinnedRuntimeNode targetJoint;
    targetJoint.name = "Joint";
    targetJoint.parent = 0;
    targetJoint.bindLocal[3] = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    targetJoint.bindGlobal = targetJoint.bindLocal;
    pivotTarget.skeleton.nodeByName.emplace(targetJoint.name, 1);
    pivotTarget.skeleton.nodes.push_back(targetJoint);
    infernux::SkinnedRuntimeBone targetBone;
    targetBone.name = targetJoint.name;
    targetBone.nodeIndex = 1;
    targetBone.inverseBind = glm::inverse(targetJoint.bindGlobal);
    pivotTarget.skeleton.boneByName.emplace(targetBone.name, 0);
    pivotTarget.skeleton.bones.push_back(targetBone);

    infernux::InxSkinnedMesh pivotSource;
    pivotSource.scaleFactor = 1.0f;
    pivotSource.skeleton.nodeByName.emplace(targetRig.name, 0);
    pivotSource.skeleton.nodes.push_back(targetRig);
    infernux::SkinnedRuntimeNode pivotHelper;
    pivotHelper.name = "Joint_$AssimpFbx$_Rotation";
    pivotHelper.parent = 0;
    pivotSource.skeleton.nodeByName.emplace(pivotHelper.name, 1);
    pivotSource.skeleton.nodes.push_back(pivotHelper);
    infernux::SkinnedRuntimeNode sourceJoint = targetJoint;
    sourceJoint.parent = 1;
    pivotSource.skeleton.nodeByName.emplace(sourceJoint.name, 2);
    pivotSource.skeleton.nodes.push_back(sourceJoint);
    infernux::SkinnedRuntimeAnimation pivotAnimation;
    pivotAnimation.name = "PivotRotation";
    pivotAnimation.durationTicks = 1.0;
    pivotAnimation.ticksPerSecond = 1.0;
    infernux::SkinnedRuntimeTrack pivotTrack;
    pivotTrack.nodeIndex = 1;
    constexpr float kSqrtHalf = 0.70710678118f;
    pivotTrack.rotations = {
        {0.0, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)},
        {1.0, glm::quat(kSqrtHalf, 0.0f, 0.0f, kSqrtHalf)},
    };
    pivotAnimation.trackByNodeIndex = {-1, 0, -1};
    pivotAnimation.tracks.push_back(pivotTrack);
    pivotSource.animations.push_back(pivotAnimation);

    const auto pivotMap = pivotTarget.skeleton.BuildRetargetMap(pivotSource.skeleton, pivotAnimation);
    assert(pivotMap.targetToSourceNode == std::vector<int>({-1, 2}));
    assert(pivotMap.exactNameMatches == 1);
    assert(pivotMap.mappedAnimationTracks == 1);
    assert(pivotMap.missingTargetDeformJoints == 0);
    assert(pivotMap.topologyDifferences == 0);
    assert(pivotTarget.skeleton.IsAnimationCompatible(pivotSource.skeleton, pivotAnimation, &compatibilityReason));
    assert(compatibilityReason.empty());
    const auto pivotPalette = pivotTarget.BuildGpuBonePalette({"PivotRotation", 1.0f, false}, &pivotSource);
    assert(pivotPalette.size() == 1);
    assert(std::abs(pivotPalette.front()[0][0]) < 1.0e-5f);
    assert(std::abs(std::abs(pivotPalette.front()[0][1]) - 1.0f) < 1.0e-5f);

    infernux::Skeleton relatedTarget = source.skeleton;
    infernux::SkinnedRuntimeNode extraNode;
    extraNode.name = "AccessoryJoint";
    extraNode.parent = 0;
    relatedTarget.nodeByName.emplace(extraNode.name, 1);
    relatedTarget.nodes.push_back(extraNode);
    infernux::SkinnedRuntimeBone extraBone;
    extraBone.name = extraNode.name;
    extraBone.nodeIndex = 1;
    relatedTarget.boneByName.emplace(extraBone.name, 1);
    relatedTarget.bones.push_back(extraBone);
    assert(!relatedTarget.IsAnimationCompatible(animationOnly.skeleton, externalAnimation, &compatibilityReason));
    assert(compatibilityReason.find("exact joint identities") != std::string::npos);

    infernux::SkinPoseHistory poseHistory;
    auto firstPose = restored->GetOrBuildGpuBonePalette({"Move", 0.0f});
    poseHistory.Publish(firstPose, true);
    const auto firstSnapshot = poseHistory.Acquire();
    assert(firstSnapshot && firstSnapshot->IsValid());
    assert(firstSnapshot->current == firstPose);
    assert(firstSnapshot->previous == firstPose);

    auto secondPose = restored->GetOrBuildGpuBonePalette({"Move", 0.25f});
    poseHistory.Publish(secondPose, false);
    const auto secondSnapshot = poseHistory.Acquire();
    assert(secondSnapshot && secondSnapshot->IsValid());
    assert(secondSnapshot->current == secondPose);
    assert(secondSnapshot->previous == firstPose);
    assert(secondSnapshot->revision > firstSnapshot->revision);

    auto differentSkeleton = std::make_shared<const infernux::SkinPoseHistory::Palette>(2, glm::mat4(1.0f));
    poseHistory.Publish(differentSkeleton, false);
    const auto resizedSnapshot = poseHistory.Acquire();
    assert(resizedSnapshot && resizedSnapshot->IsValid());
    assert(resizedSnapshot->current == differentSkeleton);
    assert(resizedSnapshot->previous == differentSkeleton);
    assert(resizedSnapshot->revision > secondSnapshot->revision);

    poseHistory.Reset();
    const auto resetSnapshot = poseHistory.Acquire();
    assert(resetSnapshot && !resetSnapshot->IsValid());
    assert(!resetSnapshot->current);
    assert(!resetSnapshot->previous);
    assert(resetSnapshot->revision > resizedSnapshot->revision);

    const std::string empty = infernux::SkinnedMeshArtifact::SerializeEmpty(SourceHash);
    assert(!infernux::SkinnedMeshArtifact::Deserialize(empty, SourceHash));

    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(bytes, "different-source"); });
    std::string corrupted = bytes;
    corrupted[corrupted.size() / 2] ^= 0x5a;
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(corrupted, SourceHash); });
    RequireInvalid(
        [&] { (void)infernux::SkinnedMeshArtifact::Deserialize(bytes.substr(0, bytes.size() - 1), SourceHash); });

    std::string trailing = bytes;
    trailing.insert(trailing.end() - static_cast<std::ptrdiff_t>(sizeof(uint64_t)), 'x');
    RewriteChecksum(trailing);
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(trailing, SourceHash); });

    // The current tail tables are mandatory; a checksum-correct truncated
    // artifact must not be interpreted as an older layout.
    const size_t identityOffset = bytes.rfind("AID1");
    assert(identityOffset != std::string::npos);
    std::string legacy = bytes.substr(0, identityOffset);
    legacy.resize(legacy.size() + sizeof(uint64_t));
    RewriteChecksum(legacy);
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Deserialize(legacy, SourceHash); });

    auto identified = source;
    identified.animations.front().id = std::string(32, 'a');
    const auto identifiedBytes = infernux::SkinnedMeshArtifact::Serialize(identified, SourceHash);
    const auto identifiedModel = infernux::SkinnedMeshArtifact::Deserialize(identifiedBytes, SourceHash);
    assert(identifiedModel->FindAnimation(std::string(32, 'a'))->name == "Move");
    auto duplicate = identified.animations.front();
    duplicate.name = "Other";
    identified.animations.push_back(duplicate);
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Serialize(identified, SourceHash); });

    source.influences.front().weight[0] = 0.5f;
    RequireInvalid([&] { (void)infernux::SkinnedMeshArtifact::Serialize(source, SourceHash); });

    std::cout << "Skinned Mesh artifact tests passed\n";
    return 0;
}
