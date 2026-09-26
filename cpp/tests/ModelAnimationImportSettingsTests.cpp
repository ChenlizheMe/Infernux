#include <function/resources/InxMesh/MeshImportSettings.h>
#include <function/resources/InxSkinnedMesh/InxSkinnedMesh.h>
#include <function/resources/InxSkinnedMesh/SkinnedMeshArtifact.h>
#include <function/resources/InxSkinnedMesh/SkinnedModelImporter.h>

#include <glm/gtc/quaternion.hpp>

#include <cassert>
#include <stdexcept>
#include <string>

namespace
{
infernux::InxSkinnedMesh MakeAnimationModel()
{
    infernux::InxSkinnedMesh model;
    infernux::SkinnedRuntimeNode root;
    root.name = "Root";
    model.skeleton.nodes.push_back(root);
    model.skeleton.nodeByName.emplace(root.name, 0);

    infernux::SkinnedRuntimeAnimation animation;
    animation.name = "Processed";
    animation.id = "processed";
    animation.durationTicks = 2.0;
    animation.ticksPerSecond = 1.0;
    animation.trackByNodeIndex = {0};
    infernux::SkinnedRuntimeTrack track;
    track.nodeIndex = 0;
    track.positions = {{0.0, glm::vec3(0.0f)}, {1.0, glm::vec3(1.0f, 1.0f, 0.0f)}, {2.0, glm::vec3(2.0f, 0.0f, 0.0f)}};
    track.rotations = {{0.0, glm::quat(1.0f, 0.0f, 0.0f, 0.0f)},
                       {2.0, glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f))}};
    track.scales = {{0.0, glm::vec3(1.0f)}, {1.0, glm::vec3(1.5f)}, {2.0, glm::vec3(2.0f)}};
    animation.tracks.push_back(std::move(track));
    model.animations.push_back(std::move(animation));
    return model;
}
} // namespace

int main()
{
    auto sampled = MakeAnimationModel();
    sampled.guid = "model-guid";
    infernux::MeshImportSettings settings;
    settings.animationSampleRate = 4.0f;
    infernux::SkinnedModelImporter::ApplyAnimationSettings(sampled, settings);
    const auto &sampledTrack = sampled.animations[0].tracks[0];
    assert(sampledTrack.positions.size() == 9);
    assert(sampledTrack.rotations.size() == 9);
    assert(sampledTrack.scales.size() == 9);
    assert(sampledTrack.positions[3].first == 0.75);
    assert(glm::length(sampledTrack.positions[3].second - glm::vec3(0.75f, 0.75f, 0.0f)) < 1e-6f);

    auto definition = MakeAnimationModel();
    definition.guid = "definition-guid";
    settings = {};
    settings.skeletonDefinitionId = "rig/main";
    infernux::SkinnedModelImporter::ApplyRigSettings(definition, settings);
    const auto definitionBytes = infernux::SkinnedMeshArtifact::Serialize(definition, "definition-source");
    const auto definitionLoaded = infernux::SkinnedMeshArtifact::Deserialize(definitionBytes, "definition-source");

    auto rigged = MakeAnimationModel();
    rigged.guid = "model-guid";
    infernux::SkinnedRuntimeNode prunable;
    prunable.name = "Unused";
    prunable.parent = 0;
    rigged.skeleton.nodes.push_back(prunable);
    rigged.skeleton.nodeByName.emplace(prunable.name, 1);
    rigged.animations[0].trackByNodeIndex.push_back(-1);
    settings = {};
    settings.skeletonDefinitionMode = "copy";
    settings.optimizeBoneHierarchy = true;
    settings.exposedBones = nlohmann::json::array({"Root"});
    settings.skeletonDefinitionGuid = "definition-guid";
    settings.skeletonDefinitionId = "rig/main";
    infernux::SkinnedModelImporter::ApplyRigSettings(rigged, settings, definitionLoaded.get());
    assert(rigged.skeleton.nodes.size() == 1 && rigged.skeletonRootNodeIndex == 0);
    assert(rigged.skeletonDefinitionGuid == "definition-guid" && rigged.skeletonDefinitionId == "rig/main");
    assert(rigged.exposedSkeletonNodeIndices == std::vector<int>{0});
    const auto riggedBytes = infernux::SkinnedMeshArtifact::Serialize(rigged, "rigged");
    const auto riggedLoaded = infernux::SkinnedMeshArtifact::Deserialize(riggedBytes, "rigged");
    assert(riggedLoaded->skeletonDefinitionGuid == "definition-guid");
    assert(riggedLoaded->skeletonDefinitionId == "rig/main");
    assert(riggedLoaded->exposedSkeletonNodeIndices == std::vector<int>{0});

    auto incompatible = MakeAnimationModel();
    incompatible.guid = "incompatible-guid";
    incompatible.skeleton.nodes[0].name = "Other";
    incompatible.skeleton.nodeByName.clear();
    incompatible.skeleton.nodeByName.emplace("Other", 0);
    bool rejectedCopy = false;
    try {
        infernux::SkinnedModelImporter::ApplyRigSettings(incompatible, settings, definitionLoaded.get());
    } catch (const std::invalid_argument &) {
        rejectedCopy = true;
    }
    assert(rejectedCopy && incompatible.skeletonDefinitionGuid.empty());

    auto compressed = MakeAnimationModel();
    settings.animationPositionError = 0.25f;
    settings.animationRotationError = 0.01f;
    settings.animationScaleError = 0.001f;
    infernux::SkinnedModelImporter::ApplyAnimationSettings(compressed, settings);
    const auto &compressedTrack = compressed.animations[0].tracks[0];
    assert(compressedTrack.positions.size() == 3);
    assert(compressedTrack.positions[1].first == 1.0);
    assert(compressedTrack.rotations.size() == 2);
    assert(compressedTrack.scales.size() == 2);

    const auto bytes = infernux::SkinnedMeshArtifact::Serialize(compressed, "processed");
    const auto loaded = infernux::SkinnedMeshArtifact::Deserialize(bytes, "processed");
    assert(loaded->animations[0].tracks[0].positions.size() == 3);
    assert(loaded->animations[0].tracks[0].rotations.size() == 2);
    assert(loaded->animations[0].tracks[0].scales.size() == 2);

    const auto keyCountBeforeFailure = compressed.animations[0].tracks[0].positions.size();
    settings.animationSampleRate = 241.0f;
    bool rejected = false;
    try {
        infernux::SkinnedModelImporter::ApplyAnimationSettings(compressed, settings);
    } catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()).find("animation_sample_rate") != std::string::npos;
    }
    assert(rejected);
    assert(compressed.animations[0].tracks[0].positions.size() == keyCountBeforeFailure);

    auto zeroDuration = MakeAnimationModel();
    zeroDuration.animations[0].durationTicks = 0.0;
    settings.animationSampleRate = 30.0f;
    rejected = false;
    try {
        infernux::SkinnedModelImporter::ApplyAnimationSettings(zeroDuration, settings);
    } catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()).find("positive duration") != std::string::npos;
    }
    assert(rejected);
    assert(zeroDuration.animations[0].tracks[0].positions.size() == 3);

    auto rooted = MakeAnimationModel();
    rooted.scaleFactor = 1.0f;
    rooted.skeleton.nodes[0].bindLocal[3] = glm::vec4(10.0f, 0.0f, 0.0f, 1.0f);
    settings = {};
    settings.animationLoopTime = false;
    settings.animationApplyRootMotion = true;
    settings.animationReferencePose = "bind_pose";
    infernux::SkinnedModelImporter::ApplyAnimationSettings(rooted, settings);
    const auto &rootAnimation = rooted.animations[0];
    assert(!rootAnimation.defaultLoop);
    assert(rootAnimation.rootMotionNodeIndex == 0);
    assert(rootAnimation.rootMotionReferencePose == "bind_pose");
    assert(rootAnimation.rootMotionPositions.size() == 3);
    assert(rootAnimation.tracks[0].positions.size() == 1);
    assert(glm::length(rootAnimation.tracks[0].positions[0].second - glm::vec3(10.0f, 0.0f, 0.0f)) < 1e-6f);
    const auto linearDelta = rooted.SampleRootMotionDelta("Processed", 0.0f, 2.0f, false);
    assert(glm::length(linearDelta.translation - glm::vec3(2.0f, 0.0f, 0.0f)) < 1e-6f);
    assert(std::abs(glm::degrees(glm::angle(linearDelta.rotation)) - 90.0f) < 1e-4f);
    const auto wrappedDelta = rooted.SampleRootMotionDelta("Processed", 1.5f, 2.5f, true);
    assert(glm::length(wrappedDelta.translation - glm::vec3(1.0f, 0.0f, 0.0f)) < 1e-6f);

    const auto rootedBytes = infernux::SkinnedMeshArtifact::Serialize(rooted, "rooted");
    const auto rootedLoaded = infernux::SkinnedMeshArtifact::Deserialize(rootedBytes, "rooted");
    assert(!rootedLoaded->animations[0].defaultLoop);
    assert(rootedLoaded->animations[0].rootMotionNodeIndex == 0);
    assert(rootedLoaded->animations[0].rootMotionPositions.size() == 3);

    auto firstFrame = MakeAnimationModel();
    settings.animationLoopTime = true;
    settings.animationReferencePose = "first_frame";
    infernux::SkinnedModelImporter::ApplyAnimationSettings(firstFrame, settings);
    assert(firstFrame.animations[0].defaultLoop);
    assert(firstFrame.animations[0].tracks[0].positions.size() == 1);
    assert(glm::length(firstFrame.animations[0].tracks[0].positions[0].second) < 1e-6f);

    auto enriched = MakeAnimationModel();
    settings = {};
    settings.animationClipExtras = nlohmann::json::array({{
        {"clip_id", "processed"},
        {"curves",
         nlohmann::json::array({{{"name", "FootPlant"},
                                 {"keys", nlohmann::json::array({{{"time_normalized", 0.0}, {"value", 0.0}},
                                                                 {{"time_normalized", 1.0}, {"value", 1.0}}})}}})},
        {"events",
         nlohmann::json::array(
             {{{"time_normalized", 0.5}, {"function", "footstep"}, {"string_arg", "L"}, {"number_arg", 2.0}}})},
        {"bone_mask", nlohmann::json::array({"Root"})},
    }});
    infernux::SkinnedModelImporter::ApplyAnimationSettings(enriched, settings);
    assert(enriched.animations[0].curves.size() == 1);
    assert(enriched.animations[0].curves[0].name == "FootPlant");
    assert(enriched.animations[0].events.size() == 1);
    assert(enriched.animations[0].events[0].function == "footstep");
    assert(enriched.animations[0].boneMask == std::vector<std::string>{"Root"});
    const auto enrichedBytes = infernux::SkinnedMeshArtifact::Serialize(enriched, "extras");
    const auto enrichedLoaded = infernux::SkinnedMeshArtifact::Deserialize(enrichedBytes, "extras");
    assert(enrichedLoaded->animations[0].curves[0].keys.size() == 2);
    assert(enrichedLoaded->animations[0].events[0].stringArgument == "L");
    assert(enrichedLoaded->animations[0].boneMask[0] == "Root");

    auto exactMask = MakeAnimationModel();
    infernux::SkinnedRuntimeNode sibling;
    sibling.name = "Sibling";
    exactMask.skeleton.nodes.push_back(sibling);
    exactMask.skeleton.nodeByName.emplace(sibling.name, 1);
    infernux::SkinnedRuntimeBone rootBone;
    rootBone.name = "Root";
    rootBone.nodeIndex = 0;
    exactMask.skeleton.boneByName.emplace(rootBone.name, 0);
    exactMask.skeleton.bones.push_back(rootBone);
    infernux::SkinnedRuntimeBone siblingBone;
    siblingBone.name = "Sibling";
    siblingBone.nodeIndex = 1;
    exactMask.skeleton.boneByName.emplace(siblingBone.name, 1);
    exactMask.skeleton.bones.push_back(siblingBone);
    exactMask.animations[0].trackByNodeIndex.push_back(1);
    infernux::SkinnedRuntimeTrack siblingTrack;
    siblingTrack.nodeIndex = 1;
    siblingTrack.positions = {{0.0, glm::vec3(0.0f)}, {2.0, glm::vec3(0.0f, 4.0f, 0.0f)}};
    exactMask.animations[0].tracks.push_back(std::move(siblingTrack));
    settings.animationClipExtras[0]["bone_mask"] = nlohmann::json::array({"Root"});
    infernux::SkinnedModelImporter::ApplyAnimationSettings(exactMask, settings);
    infernux::PoseStackLayer maskedLayer;
    maskedLayer.takeName = "Processed";
    maskedLayer.timeSeconds = 2.0f;
    maskedLayer.loop = false;
    maskedLayer.boneMask = exactMask.animations[0].boneMask;
    const auto maskedPose = exactMask.BuildBoneMatricesFromPoseStack({maskedLayer});
    assert(glm::length(glm::vec3(maskedPose[0][3]) - glm::vec3(2.0f, 0.0f, 0.0f)) < 1e-6f);
    assert(glm::length(glm::vec3(maskedPose[1][3])) < 1e-6f);

    auto rejectedExtras = MakeAnimationModel();
    settings.animationClipExtras[0]["bone_mask"] = nlohmann::json::array({"Missing"});
    rejected = false;
    try {
        infernux::SkinnedModelImporter::ApplyAnimationSettings(rejectedExtras, settings);
    } catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()).find("missing skeleton node") != std::string::npos;
    }
    assert(rejected);
    assert(rejectedExtras.animations[0].curves.empty());
    return 0;
}
