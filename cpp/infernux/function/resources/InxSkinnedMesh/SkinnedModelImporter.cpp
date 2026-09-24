#include "SkinnedModelImporter.h"
#include <function/resources/InxMesh/MeshImportSettings.h>

#include "InxSkinnedMesh.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <platform/filesystem/InxPath.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{
namespace
{
glm::mat4 AiToGlm(const aiMatrix4x4 &matrix)
{
    return glm::mat4(matrix.a1, matrix.b1, matrix.c1, matrix.d1, matrix.a2, matrix.b2, matrix.c2, matrix.d2, matrix.a3,
                     matrix.b3, matrix.c3, matrix.d3, matrix.a4, matrix.b4, matrix.c4, matrix.d4);
}

glm::vec3 AiToGlm(const aiVector3D &vector)
{
    return {vector.x, vector.y, vector.z};
}

glm::quat AiToGlm(const aiQuaternion &quaternion)
{
    return {quaternion.w, quaternion.x, quaternion.y, quaternion.z};
}

bool IsFinite(const glm::vec2 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y);
}

bool IsFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool IsFinite(const glm::quat &value)
{
    return std::isfinite(value.w) && std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool IsFinite(const glm::mat4 &value)
{
    for (glm::length_t column = 0; column < value.length(); ++column) {
        for (glm::length_t row = 0; row < value[column].length(); ++row) {
            if (!std::isfinite(value[column][row]))
                return false;
        }
    }
    return true;
}

glm::vec3 NormalizeOr(const glm::vec3 &value, const glm::vec3 &fallback)
{
    if (!IsFinite(value))
        return fallback;
    const float lengthSquared = glm::dot(value, value);
    if (!std::isfinite(lengthSquared) || lengthSquared <= 1e-12f)
        return fallback;
    return value * glm::inversesqrt(lengthSquared);
}

glm::quat NormalizeOrIdentity(const glm::quat &value)
{
    if (!IsFinite(value))
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    const float lengthSquared = glm::dot(value, value);
    if (!std::isfinite(lengthSquared) || lengthSquared <= 1e-12f)
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    return value * glm::inversesqrt(lengthSquared);
}

unsigned int BuildAssimpFlags()
{
    return aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | aiProcess_FlipUVs |
           aiProcess_JoinIdenticalVertices | aiProcess_SortByPType | aiProcess_ValidateDataStructure |
           aiProcess_ImproveCacheLocality;
}

void CollectNodes(const aiNode &node, int parent, const glm::mat4 &parentGlobal, InxSkinnedMesh &output)
{
    const int index = static_cast<int>(output.skeleton.nodes.size());
    SkinnedRuntimeNode runtimeNode;
    runtimeNode.name = node.mName.C_Str();
    runtimeNode.parent = parent;
    runtimeNode.bindLocal = AiToGlm(node.mTransformation);
    runtimeNode.bindGlobal = parentGlobal * runtimeNode.bindLocal;
    if (!IsFinite(runtimeNode.bindLocal) || !IsFinite(runtimeNode.bindGlobal))
        throw std::runtime_error("Skinned model contains a non-finite node transform: " + runtimeNode.name);
    if (runtimeNode.name.empty() || !output.skeleton.nodeByName.emplace(runtimeNode.name, index).second)
        throw std::runtime_error("Skinned model contains an empty or duplicate node name");
    output.skeleton.nodes.push_back(runtimeNode);

    for (unsigned int childIndex = 0; childIndex < node.mNumChildren; ++childIndex) {
        if (!node.mChildren[childIndex])
            throw std::runtime_error("Skinned model scene contains a null node child");
        CollectNodes(*node.mChildren[childIndex], index, runtimeNode.bindGlobal, output);
    }
}

void CollectMeshNodes(const aiNode &node, const InxSkinnedMesh &model, std::vector<std::pair<uint32_t, int>> &output)
{
    int nodeIndex = -1;
    const auto found = model.skeleton.nodeByName.find(node.mName.C_Str());
    if (found != model.skeleton.nodeByName.end())
        nodeIndex = found->second;
    for (unsigned int meshIndex = 0; meshIndex < node.mNumMeshes; ++meshIndex)
        output.push_back({node.mMeshes[meshIndex], nodeIndex});
    for (unsigned int childIndex = 0; childIndex < node.mNumChildren; ++childIndex)
        CollectMeshNodes(*node.mChildren[childIndex], model, output);
}

void AddInfluence(SkinInfluence &influence, uint32_t boneIndex, float weight, uint32_t limit = kMaxSkinInfluences)
{
    if (!std::isfinite(weight) || weight <= 0.0f)
        return;
    for (uint32_t index = 0; index < limit; ++index) {
        if (influence.weight[index] <= 0.0f) {
            influence.boneIndex[index] = boneIndex;
            influence.weight[index] = weight;
            return;
        }
    }
    uint32_t lightest = 0;
    for (uint32_t index = 1; index < limit; ++index) {
        if (influence.weight[index] < influence.weight[lightest])
            lightest = index;
    }
    if (weight > influence.weight[lightest]) {
        influence.boneIndex[lightest] = boneIndex;
        influence.weight[lightest] = weight;
    }
}

bool MatricesNearlyEqual(const glm::mat4 &lhs, const glm::mat4 &rhs, float epsilon = 1e-3f)
{
    for (glm::length_t column = 0; column < 4; ++column) {
        for (glm::length_t row = 0; row < 4; ++row) {
            if (std::abs(lhs[column][row] - rhs[column][row]) > epsilon)
                return false;
        }
    }
    return true;
}

uint32_t GetOrCreateBone(InxSkinnedMesh &model, const aiBone &source, const glm::mat4 &inverseBind, uint32_t meshIndex)
{
    const std::string sourceName = source.mName.C_Str();
    if (sourceName.empty())
        throw std::runtime_error("Skinned model contains a bone without a name");
    if (!IsFinite(inverseBind))
        throw std::runtime_error("Skinned model contains a non-finite inverse bind transform for bone: " + sourceName);

    const auto node = model.skeleton.nodeByName.find(sourceName);
    const int nodeIndex = node != model.skeleton.nodeByName.end() ? node->second : -1;
    const std::string meshAliasPrefix = sourceName + "__mesh_";
    for (uint32_t index = 0; index < model.skeleton.bones.size(); ++index) {
        const auto &existing = model.skeleton.bones[index];
        const bool sameSourceBone =
            existing.name == sourceName || existing.name.compare(0, meshAliasPrefix.size(), meshAliasPrefix) == 0;
        if (sameSourceBone && existing.nodeIndex == nodeIndex && MatricesNearlyEqual(existing.inverseBind, inverseBind))
            return index;
    }

    std::string runtimeName = sourceName;
    if (model.skeleton.boneByName.find(runtimeName) != model.skeleton.boneByName.end()) {
        runtimeName += "__mesh_" + std::to_string(meshIndex);
        for (uint32_t suffix = 1; model.skeleton.boneByName.find(runtimeName) != model.skeleton.boneByName.end();
             ++suffix)
            runtimeName = sourceName + "__mesh_" + std::to_string(meshIndex) + "_" + std::to_string(suffix);
    }

    SkinnedRuntimeBone bone;
    bone.name = runtimeName;
    bone.nodeIndex = nodeIndex;
    bone.inverseBind = inverseBind;
    const uint32_t index = static_cast<uint32_t>(model.skeleton.bones.size());
    model.skeleton.boneByName.emplace(runtimeName, index);
    model.skeleton.bones.push_back(std::move(bone));
    return index;
}

uint32_t GetOrCreateMeshNodeFallbackBone(InxSkinnedMesh &model, int nodeIndex, const glm::mat4 &inverseBind)
{
    const std::string name = "__mesh_node_fallback_" + std::to_string(nodeIndex);
    const auto found = model.skeleton.boneByName.find(name);
    if (found != model.skeleton.boneByName.end())
        return found->second;
    SkinnedRuntimeBone bone;
    bone.name = name;
    bone.nodeIndex = nodeIndex;
    bone.inverseBind = inverseBind;
    const uint32_t index = static_cast<uint32_t>(model.skeleton.bones.size());
    model.skeleton.boneByName.emplace(name, index);
    model.skeleton.bones.push_back(std::move(bone));
    return index;
}

bool HasInfluence(const SkinInfluence &influence)
{
    for (const float weight : influence.weight) {
        if (weight > 0.0f)
            return true;
    }
    return false;
}

glm::mat4 ComposeTrs(const glm::vec3 &translation, const glm::quat &rotation, const glm::vec3 &scale)
{
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

bool StrictDecomposeTrs(const glm::mat4 &matrix, glm::vec3 &translation, glm::quat &rotation, glm::vec3 &scale)
{
    if (!IsFinite(matrix) || std::abs(matrix[0][3]) > 1.e-5f || std::abs(matrix[1][3]) > 1.e-5f ||
        std::abs(matrix[2][3]) > 1.e-5f || std::abs(matrix[3][3] - 1.0f) > 1.e-5f)
        return false;
    translation = glm::vec3(matrix[3]);
    glm::vec3 axes[] = {glm::vec3(matrix[0]), glm::vec3(matrix[1]), glm::vec3(matrix[2])};
    scale = {glm::length(axes[0]), glm::length(axes[1]), glm::length(axes[2])};
    if (!IsFinite(scale) || scale.x <= 1.e-8f || scale.y <= 1.e-8f || scale.z <= 1.e-8f)
        return false;
    for (size_t axis = 0; axis < 3; ++axis)
        axes[axis] /= scale[static_cast<glm::length_t>(axis)];
    if (std::abs(glm::dot(axes[0], axes[1])) > 2.e-4f || std::abs(glm::dot(axes[0], axes[2])) > 2.e-4f ||
        std::abs(glm::dot(axes[1], axes[2])) > 2.e-4f)
        return false;
    if (glm::dot(glm::cross(axes[0], axes[1]), axes[2]) < 0.0f) {
        axes[0] = -axes[0];
        scale.x = -scale.x;
    }
    const glm::mat3 rotationMatrix(axes[0], axes[1], axes[2]);
    if (std::abs(glm::determinant(rotationMatrix) - 1.0f) > 2.e-4f)
        return false;
    rotation = NormalizeOrIdentity(glm::quat_cast(rotationMatrix));
    const glm::mat4 rebuilt = ComposeTrs(translation, rotation, scale);
    float largest = 1.0f;
    float error = 0.0f;
    for (glm::length_t column = 0; column < 4; ++column)
        for (glm::length_t row = 0; row < 4; ++row) {
            largest = std::max(largest, std::abs(matrix[column][row]));
            error = std::max(error, std::abs(matrix[column][row] - rebuilt[column][row]));
        }
    return error <= largest * 2.e-4f;
}

template <typename T, typename Interpolate>
T SampleImportedKeys(const std::vector<std::pair<double, T>> &keys, double time, const T &fallback,
                     Interpolate interpolate)
{
    if (keys.empty())
        return fallback;
    const auto upper = std::upper_bound(keys.begin(), keys.end(), time,
                                        [](double value, const auto &key) { return value < key.first; });
    if (upper == keys.begin())
        return upper->second;
    if (upper == keys.end())
        return keys.back().second;
    const auto &lower = *(upper - 1);
    const double span = upper->first - lower.first;
    if (!(span > 0.0))
        throw std::runtime_error("source-root animation keys must use strictly increasing times");
    return interpolate(lower.second, upper->second, static_cast<float>((time - lower.first) / span));
}

SkinnedRuntimeTrack BakeSourceRootAnimationTrack(const SkinnedRuntimeTrack &source, const glm::mat4 &sourceRootBasis,
                                                 double durationTicks, std::string_view animationName)
{
    glm::vec3 bindTranslation, bindScale;
    glm::quat bindRotation;
    if (!StrictDecomposeTrs(sourceRootBasis, bindTranslation, bindRotation, bindScale))
        throw std::runtime_error("Bake Axis Conversion requires a finite, invertible TRS source root in animation '" +
                                 std::string(animationName) + "'");
    const auto validateTimes = [durationTicks, animationName](const auto &keys, std::string_view channel) {
        double previous = -std::numeric_limits<double>::infinity();
        for (const auto &[time, _] : keys) {
            if (!std::isfinite(time) || time < -1.e-9 || time <= previous ||
                (durationTicks > 0.0 && time > durationTicks + 1.e-9))
                throw std::runtime_error("Bake Axis Conversion requires finite, ordered " + std::string(channel) +
                                         " keys inside animation '" + std::string(animationName) + "'");
            previous = time;
        }
    };
    validateTimes(source.positions, "position");
    validateTimes(source.rotations, "rotation");
    validateTimes(source.scales, "scale");
    std::vector<double> times;
    times.reserve(source.positions.size() + source.rotations.size() + source.scales.size() + 2);
    times.push_back(0.0);
    if (durationTicks > 0.0)
        times.push_back(durationTicks);
    for (const auto &[time, _] : source.positions)
        times.push_back(time);
    for (const auto &[time, _] : source.rotations)
        times.push_back(time);
    for (const auto &[time, _] : source.scales)
        times.push_back(time);
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end(),
                            [](double left, double right) { return std::abs(left - right) <= 1.e-9; }),
                times.end());
    if (times.empty())
        throw std::runtime_error("Bake Axis Conversion source-root animation has no finite sample domain");

    const glm::mat4 inverseRootBasis = glm::inverse(sourceRootBasis);
    SkinnedRuntimeTrack baked;
    baked.nodeIndex = source.nodeIndex;
    baked.positions.reserve(times.size());
    baked.rotations.reserve(times.size());
    baked.scales.reserve(times.size());
    glm::quat previousRotation(1.0f, 0.0f, 0.0f, 0.0f);
    bool hasPreviousRotation = false;
    for (const double time : times) {
        const glm::vec3 translation = SampleImportedKeys(
            source.positions, time, bindTranslation,
            [](const glm::vec3 &left, const glm::vec3 &right, float alpha) { return glm::mix(left, right, alpha); });
        const glm::quat sourceRotation = SampleImportedKeys(
            source.rotations, time, bindRotation, [](const glm::quat &left, const glm::quat &right, float alpha) {
                return NormalizeOrIdentity(glm::slerp(left, right, alpha));
            });
        const glm::vec3 sourceScale = SampleImportedKeys(
            source.scales, time, bindScale,
            [](const glm::vec3 &left, const glm::vec3 &right, float alpha) { return glm::mix(left, right, alpha); });
        const glm::mat4 converted = ComposeTrs(translation, sourceRotation, sourceScale) * inverseRootBasis;
        glm::vec3 bakedTranslation, bakedScale;
        glm::quat bakedRotation;
        if (!StrictDecomposeTrs(converted, bakedTranslation, bakedRotation, bakedScale))
            throw std::runtime_error("Bake Axis Conversion produced a non-TRS source-root sample in animation '" +
                                     std::string(animationName) + "' at tick " + std::to_string(time));
        if (hasPreviousRotation && glm::dot(previousRotation, bakedRotation) < 0.0f)
            bakedRotation = -bakedRotation;
        previousRotation = bakedRotation;
        hasPreviousRotation = true;
        baked.positions.emplace_back(time, bakedTranslation);
        baked.rotations.emplace_back(time, bakedRotation);
        baked.scales.emplace_back(time, bakedScale);
    }
    return baked;
}

struct HumanoidSlotSpec
{
    const char *name;
    bool required;
    std::vector<std::string_view> aliases;
};

const std::vector<HumanoidSlotSpec> &HumanoidSlots()
{
    static const std::vector<HumanoidSlotSpec> slots = {
        {"hips", true, {"hips", "pelvis"}},
        {"spine", true, {"spine", "spine1"}},
        {"chest", false, {"chest", "spine2"}},
        {"upper_chest", false, {"upperchest", "spine3"}},
        {"neck", false, {"neck"}},
        {"head", true, {"head"}},
        {"left_shoulder", false, {"leftshoulder", "shoulderl", "lshoulder"}},
        {"left_upper_arm", true, {"leftupperarm", "upperarml", "lupperarm", "leftarm"}},
        {"left_lower_arm", true, {"leftlowerarm", "lowerarml", "lforearm", "leftforearm"}},
        {"left_hand", true, {"lefthand", "handl", "lhand"}},
        {"right_shoulder", false, {"rightshoulder", "shoulderr", "rshoulder"}},
        {"right_upper_arm", true, {"rightupperarm", "upperarmr", "rupperarm", "rightarm"}},
        {"right_lower_arm", true, {"rightlowerarm", "lowerarmr", "rforearm", "rightforearm"}},
        {"right_hand", true, {"righthand", "handr", "rhand"}},
        {"left_upper_leg", true, {"leftupperleg", "leftupleg", "upperlegl", "lthigh", "leftthigh"}},
        {"left_lower_leg", true, {"leftlowerleg", "lowerlegl", "lcalf", "leftleg"}},
        {"left_foot", true, {"leftfoot", "footl", "lfoot"}},
        {"left_toes", false, {"lefttoes", "lefttoebase", "toel"}},
        {"right_upper_leg", true, {"rightupperleg", "rightupleg", "upperlegr", "rthigh", "rightthigh"}},
        {"right_lower_leg", true, {"rightlowerleg", "lowerlegr", "rcalf", "rightleg"}},
        {"right_foot", true, {"rightfoot", "footr", "rfoot"}},
        {"right_toes", false, {"righttoes", "righttoebase", "toer"}},
    };
    return slots;
}

std::string NormalizeHumanoidNodeName(std::string_view name)
{
    const size_t separator = name.find_last_of(":|/");
    if (separator != std::string_view::npos)
        name.remove_prefix(separator + 1);
    std::string normalized;
    for (const unsigned char character : name)
        if (std::isalnum(character))
            normalized.push_back(static_cast<char>(std::tolower(character)));
    constexpr std::string_view mixamo = "mixamorig";
    if (normalized.rfind(mixamo, 0) == 0)
        normalized.erase(0, mixamo.size());
    return normalized;
}

bool IsDescendantOf(const Skeleton &skeleton, int node, int ancestor)
{
    for (; node >= 0; node = skeleton.nodes[static_cast<size_t>(node)].parent)
        if (node == ancestor)
            return true;
    return false;
}

void BuildHumanoidRigDefinition(InxSkinnedMesh &model, const MeshImportSettings &settings, int rootIndex)
{
    model.humanoid = {};
    if (settings.rigType != "humanoid")
        return;
    MeshImportSettings::RequireHumanoidBoneOverrides(settings.humanoidBoneOverrides);
    model.humanoid.enabled = true;

    std::map<std::string, int> mapped;
    std::unordered_set<int> usedNodes;
    for (const auto &slot : HumanoidSlots()) {
        int selected = -1;
        if (settings.humanoidBoneOverrides.contains(slot.name)) {
            const std::string identity = settings.humanoidBoneOverrides.at(slot.name).get<std::string>();
            const auto found = model.skeleton.nodeByName.find(identity);
            if (found == model.skeleton.nodeByName.end() || !IsDescendantOf(model.skeleton, found->second, rootIndex))
                throw std::invalid_argument("humanoid override references a missing node under the rig root: " +
                                            std::string(slot.name) + " -> " + identity);
            selected = found->second;
        } else {
            // Aliases are preference ordered: an exact canonical name wins
            // over looser DCC conventions such as Spine1/Spine2.
            for (const auto alias : slot.aliases) {
                std::vector<int> candidates;
                for (size_t index = 0; index < model.skeleton.nodes.size(); ++index) {
                    if (IsDescendantOf(model.skeleton, static_cast<int>(index), rootIndex) &&
                        NormalizeHumanoidNodeName(model.skeleton.nodes[index].name) == alias)
                        candidates.push_back(static_cast<int>(index));
                }
                if (candidates.size() == 1) {
                    selected = candidates.front();
                    break;
                }
                if (candidates.size() > 1) {
                    model.humanoid.issues.push_back(
                        {"ambiguous_auto_mapping", slot.name, "multiple source nodes match this humanoid slot"});
                    break;
                }
            }
        }
        if (selected >= 0 && !usedNodes.insert(selected).second) {
            model.humanoid.issues.push_back(
                {"duplicate_node", slot.name, "a source node cannot fill more than one humanoid slot"});
            selected = -1;
        }
        if (selected >= 0)
            mapped.emplace(slot.name, selected);
        else if (slot.required)
            model.humanoid.issues.push_back({"missing_required_bone", slot.name, "required humanoid bone is unmapped"});
    }

    model.humanoid.requiredBonesValid =
        std::all_of(HumanoidSlots().begin(), HumanoidSlots().end(),
                    [&](const auto &slot) { return !slot.required || mapped.find(slot.name) != mapped.end(); });
    model.humanoid.hierarchyValid = model.humanoid.requiredBonesValid;
    const auto requireAncestor = [&](const char *ancestor, const char *child) {
        const auto parentNode = mapped.find(ancestor), childNode = mapped.find(child);
        if (parentNode == mapped.end() || childNode == mapped.end())
            return;
        if (!IsDescendantOf(model.skeleton, childNode->second, parentNode->second) ||
            childNode->second == parentNode->second) {
            model.humanoid.hierarchyValid = false;
            model.humanoid.issues.push_back(
                {"invalid_hierarchy", child, std::string(child) + " must descend from " + ancestor});
        }
    };
    requireAncestor("hips", "spine");
    requireAncestor("spine", "head");
    requireAncestor("spine", "left_upper_arm");
    requireAncestor("left_upper_arm", "left_lower_arm");
    requireAncestor("left_lower_arm", "left_hand");
    requireAncestor("spine", "right_upper_arm");
    requireAncestor("right_upper_arm", "right_lower_arm");
    requireAncestor("right_lower_arm", "right_hand");
    requireAncestor("hips", "left_upper_leg");
    requireAncestor("left_upper_leg", "left_lower_leg");
    requireAncestor("left_lower_leg", "left_foot");
    requireAncestor("hips", "right_upper_leg");
    requireAncestor("right_upper_leg", "right_lower_leg");
    requireAncestor("right_lower_leg", "right_foot");

    model.humanoid.referencePoseValid = model.humanoid.requiredBonesValid;
    const auto requireSegment = [&](const char *from, const char *to) {
        const auto a = mapped.find(from), b = mapped.find(to);
        if (a == mapped.end() || b == mapped.end())
            return;
        const auto &left = model.skeleton.nodes[static_cast<size_t>(a->second)].bindGlobal;
        const auto &right = model.skeleton.nodes[static_cast<size_t>(b->second)].bindGlobal;
        if (!IsFinite(left) || !IsFinite(right) || std::abs(glm::determinant(left)) <= 1.e-8f ||
            std::abs(glm::determinant(right)) <= 1.e-8f || glm::length(glm::vec3(right[3] - left[3])) <= 1.e-6f) {
            model.humanoid.referencePoseValid = false;
            model.humanoid.issues.push_back(
                {"invalid_reference_pose", to,
                 std::string(from) + " and " + to + " require finite, invertible, distinct bind transforms"});
        }
    };
    requireSegment("hips", "spine");
    requireSegment("spine", "head");
    requireSegment("left_upper_arm", "left_lower_arm");
    requireSegment("left_lower_arm", "left_hand");
    requireSegment("right_upper_arm", "right_lower_arm");
    requireSegment("right_lower_arm", "right_hand");
    requireSegment("left_upper_leg", "left_lower_leg");
    requireSegment("left_lower_leg", "left_foot");
    requireSegment("right_upper_leg", "right_lower_leg");
    requireSegment("right_lower_leg", "right_foot");
    for (const auto &[name, node] : mapped)
        model.humanoid.bones.emplace_back(name, node);
}
} // namespace

bool SkinnedModelImporter::HasSkinningData(const aiScene &scene, bool includeAnimations) noexcept
{
    if (includeAnimations && scene.mNumAnimations > 0)
        return true;
    for (unsigned int index = 0; index < scene.mNumMeshes; ++index) {
        if (scene.mMeshes[index] && scene.mMeshes[index]->mNumBones > 0)
            return true;
    }
    return false;
}

std::shared_ptr<InxSkinnedMesh> SkinnedModelImporter::ConvertScene(const aiScene &scene, const std::string &sourceGuid,
                                                                   const std::string &sourcePath, float scaleFactor,
                                                                   bool importAnimations, int maxBonesPerVertex,
                                                                   float minBoneWeight, bool synthesizeMissingTangents,
                                                                   bool importBlendShapes, bool bakeAxisConversion)
{
    if (!scene.mRootNode)
        throw std::invalid_argument("Skinned model scene has no root node");
    if (!std::isfinite(scaleFactor) || scaleFactor <= 0.0f)
        throw std::invalid_argument("Skinned model scale factor must be finite and positive");
    if (maxBonesPerVertex < 1 || maxBonesPerVertex > static_cast<int>(kMaxSkinInfluences) ||
        !std::isfinite(minBoneWeight) || minBoneWeight < 0.0f || minBoneWeight > 1.0f)
        throw std::invalid_argument("Skinned model requires 1-4 influences and a minimum weight in [0, 1]");

    auto model = std::make_shared<InxSkinnedMesh>();
    model->sourcePath = sourcePath;
    model->guid = sourceGuid;
    model->scaleFactor = scaleFactor;
    CollectNodes(*scene.mRootNode, -1, glm::mat4(1.0f), *model);
    const glm::mat4 sourceRootBasis =
        model->skeleton.nodes.empty() ? glm::mat4(1.0f) : model->skeleton.nodes.front().bindLocal;
    if (bakeAxisConversion && !model->skeleton.nodes.empty()) {
        model->skeleton.nodes.front().bindLocal = glm::mat4(1.0f);
        model->skeleton.nodes.front().bindGlobal = glm::mat4(1.0f);
        for (size_t index = 1; index < model->skeleton.nodes.size(); ++index) {
            auto &node = model->skeleton.nodes[index];
            if (node.parent == 0)
                node.bindLocal = sourceRootBasis * node.bindLocal;
        }
    }

    std::vector<std::pair<uint32_t, int>> meshNodes;
    CollectMeshNodes(*scene.mRootNode, *model, meshNodes);
    std::unordered_map<unsigned int, uint32_t> materialSlots;
    std::unordered_map<int, uint32_t> nodeGroups;
    std::unordered_map<std::string, size_t> morphTargetByName;
    uint32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
    for (const auto &[meshIndex, nodeIndex] : meshNodes) {
        if (meshIndex >= scene.mNumMeshes || !scene.mMeshes[meshIndex])
            throw std::runtime_error("Skinned model node references an invalid mesh");
        const aiMesh &sourceMesh = *scene.mMeshes[meshIndex];
        if (!(sourceMesh.mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
            continue;

        const uint32_t vertexStart = vertexOffset;
        const uint32_t indexStart = indexOffset;
        const glm::mat4 meshToModel =
            nodeIndex >= 0 ? model->skeleton.nodes[static_cast<size_t>(nodeIndex)].bindGlobal : glm::mat4(1.0f);
        const glm::mat4 sourceBasis = bakeAxisConversion && nodeIndex == 0 ? sourceRootBasis : glm::mat4(1.0f);
        const glm::mat4 sourceMeshToModel = meshToModel * sourceBasis;
        if (!IsFinite(sourceMeshToModel) || std::abs(glm::determinant(glm::mat3(sourceMeshToModel))) <= 1e-12f)
            throw std::runtime_error("Skinned model mesh node has a non-invertible bind transform: " +
                                     std::string(sourceMesh.mName.C_Str()));
        const glm::mat4 modelToMesh = glm::inverse(sourceMeshToModel);
        const glm::mat3 normalToModel = glm::inverseTranspose(glm::mat3(sourceMeshToModel));
        const glm::mat3 tangentToModel(sourceMeshToModel);
        const float orientation = glm::determinant(tangentToModel) < 0.0f ? -1.0f : 1.0f;
        const bool hasNormals = sourceMesh.HasNormals();
        const bool hasTangents = hasNormals && sourceMesh.HasTangentsAndBitangents();
        const bool hasUvs = sourceMesh.HasTextureCoords(0);
        const bool hasColors = sourceMesh.HasVertexColors(0);
        for (unsigned int vertexIndex = 0; vertexIndex < sourceMesh.mNumVertices; ++vertexIndex) {
            Vertex vertex{};
            vertex.pos = glm::vec3(sourceMeshToModel * glm::vec4(AiToGlm(sourceMesh.mVertices[vertexIndex]), 1.0f));
            if (!IsFinite(vertex.pos))
                throw std::runtime_error("Skinned model contains a non-finite vertex position");
            vertex.normal = hasNormals ? NormalizeOr(normalToModel * AiToGlm(sourceMesh.mNormals[vertexIndex]),
                                                     glm::vec3(0.0f, 1.0f, 0.0f))
                                       : glm::vec3(0.0f);
            if (hasTangents) {
                const auto tangent = AiToGlm(sourceMesh.mTangents[vertexIndex]);
                const auto bitangent = AiToGlm(sourceMesh.mBitangents[vertexIndex]);
                const float handedness =
                    glm::dot(glm::cross(AiToGlm(sourceMesh.mNormals[vertexIndex]), tangent), bitangent) < 0.0f ? -1.0f
                                                                                                               : 1.0f;
                vertex.tangent = glm::vec4(NormalizeOr(tangentToModel * tangent, glm::vec3(1.0f, 0.0f, 0.0f)),
                                           handedness * orientation);
            } else if (synthesizeMissingTangents && hasNormals) {
                // Static and skinned consumers of one imported model must
                // publish the same complete vertex basis.  A stable tangent
                // keeps the default normal-map path defined when the source
                // supplies normals but no UV/tangent channel.
                const glm::vec3 normal = vertex.normal;
                const glm::vec3 reference =
                    std::abs(normal.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                vertex.tangent = glm::vec4(glm::normalize(glm::cross(reference, normal)), 1.0f);
            } else {
                vertex.tangent = glm::vec4(0.0f);
            }
            if (hasUvs) {
                vertex.texCoord = {sourceMesh.mTextureCoords[0][vertexIndex].x,
                                   sourceMesh.mTextureCoords[0][vertexIndex].y};
                if (!IsFinite(vertex.texCoord))
                    vertex.texCoord = glm::vec2(0.0f);
            }
            if (sourceMesh.HasTextureCoords(1)) {
                vertex.texCoord1 = {sourceMesh.mTextureCoords[1][vertexIndex].x,
                                    sourceMesh.mTextureCoords[1][vertexIndex].y};
                if (!IsFinite(vertex.texCoord1))
                    throw std::runtime_error("Skinned model contains non-finite secondary UV coordinates");
            }
            vertex.color = hasColors
                               ? glm::vec3(sourceMesh.mColors[0][vertexIndex].r, sourceMesh.mColors[0][vertexIndex].g,
                                           sourceMesh.mColors[0][vertexIndex].b)
                               : glm::vec3(1.0f);
            if (!IsFinite(vertex.color))
                vertex.color = glm::vec3(1.0f);
            model->baseVertices.push_back(vertex);
            model->influences.push_back({});
        }

        for (auto &target : model->morphTargets) {
            target.positionDeltas.resize(model->baseVertices.size(), glm::vec3(0.0f));
            if (!target.normalDeltas.empty())
                target.normalDeltas.resize(model->baseVertices.size(), glm::vec3(0.0f));
            if (!target.tangentDeltas.empty())
                target.tangentDeltas.resize(model->baseVertices.size(), glm::vec3(0.0f));
        }
        if (importBlendShapes) {
            std::unordered_set<std::string> meshTargetNames;
            for (unsigned int morphIndex = 0; morphIndex < sourceMesh.mNumAnimMeshes; ++morphIndex) {
                const aiAnimMesh *sourceTarget = sourceMesh.mAnimMeshes[morphIndex];
                if (!sourceTarget || sourceTarget->mNumVertices != sourceMesh.mNumVertices)
                    throw std::runtime_error("Skinned model blend shape has an invalid vertex domain");
                if (!std::isfinite(sourceTarget->mWeight))
                    throw std::runtime_error("Skinned model blend shape has a non-finite default weight");
                std::string targetName = sourceTarget->mName.C_Str();
                if (targetName.empty())
                    targetName = std::string(sourceMesh.mName.C_Str()) + "/Morph_" + std::to_string(morphIndex);
                if (!meshTargetNames.insert(targetName).second)
                    throw std::runtime_error("Skinned model mesh contains duplicate blend shape names: " + targetName);
                auto [found, inserted] = morphTargetByName.emplace(targetName, model->morphTargets.size());
                if (inserted) {
                    MeshMorphTarget target;
                    target.name = targetName;
                    target.defaultWeight = static_cast<float>(sourceTarget->mWeight);
                    target.positionDeltas.resize(model->baseVertices.size(), glm::vec3(0.0f));
                    model->morphTargets.push_back(std::move(target));
                } else if (std::abs(model->morphTargets[found->second].defaultWeight -
                                    static_cast<float>(sourceTarget->mWeight)) > 1.e-6f) {
                    throw std::runtime_error("Skinned model blend shape parts disagree on the default weight: " +
                                             targetName);
                }
                auto &target = model->morphTargets[found->second];
                if (sourceTarget->mNormals && target.normalDeltas.empty())
                    target.normalDeltas.resize(model->baseVertices.size(), glm::vec3(0.0f));
                if (sourceTarget->mTangents && target.tangentDeltas.empty())
                    target.tangentDeltas.resize(model->baseVertices.size(), glm::vec3(0.0f));
                for (unsigned int vertexIndex = 0; vertexIndex < sourceMesh.mNumVertices; ++vertexIndex) {
                    const size_t outputIndex = static_cast<size_t>(vertexStart) + vertexIndex;
                    if (sourceTarget->mVertices) {
                        const glm::vec3 delta =
                            AiToGlm(sourceTarget->mVertices[vertexIndex]) - AiToGlm(sourceMesh.mVertices[vertexIndex]);
                        target.positionDeltas[outputIndex] = glm::mat3(sourceMeshToModel) * delta;
                    }
                    if (sourceTarget->mNormals) {
                        const glm::vec3 base =
                            sourceMesh.mNormals ? AiToGlm(sourceMesh.mNormals[vertexIndex]) : glm::vec3(0.0f);
                        const glm::vec3 sourceValue = AiToGlm(sourceTarget->mNormals[vertexIndex]);
                        target.normalDeltas[outputIndex] = NormalizeOr(normalToModel * sourceValue, glm::vec3(0.0f)) -
                                                           model->baseVertices[outputIndex].normal;
                    }
                    if (sourceTarget->mTangents) {
                        const glm::vec3 sourceValue = AiToGlm(sourceTarget->mTangents[vertexIndex]);
                        target.tangentDeltas[outputIndex] = NormalizeOr(tangentToModel * sourceValue, glm::vec3(0.0f)) -
                                                            glm::vec3(model->baseVertices[outputIndex].tangent);
                    }
                }
            }
        }

        std::vector<bool> weightedVertices(sourceMesh.mNumVertices, false);
        for (unsigned int boneIndex = 0; boneIndex < sourceMesh.mNumBones; ++boneIndex) {
            if (!sourceMesh.mBones[boneIndex])
                throw std::runtime_error("Skinned model mesh contains a null bone");
            const aiBone &sourceBone = *sourceMesh.mBones[boneIndex];
            glm::mat4 inverseBind = AiToGlm(sourceBone.mOffsetMatrix) * modelToMesh;
            const auto boneNode = model->skeleton.nodeByName.find(sourceBone.mName.C_Str());
            if (bakeAxisConversion && boneNode != model->skeleton.nodeByName.end() && boneNode->second == 0)
                inverseBind = sourceRootBasis * inverseBind;
            const uint32_t runtimeBone = GetOrCreateBone(*model, sourceBone, inverseBind, meshIndex);
            for (unsigned int weightIndex = 0; weightIndex < sourceBone.mNumWeights; ++weightIndex) {
                const aiVertexWeight &weight = sourceBone.mWeights[weightIndex];
                if (weight.mVertexId >= sourceMesh.mNumVertices)
                    throw std::runtime_error("Skinned model bone weight references an invalid vertex");
                if (!std::isfinite(weight.mWeight) || weight.mWeight < 0.0f)
                    throw std::runtime_error("Skinned model contains a negative or non-finite bone weight");
                weightedVertices[weight.mVertexId] = weightedVertices[weight.mVertexId] || weight.mWeight > 0.0f;
                if (weight.mWeight >= minBoneWeight)
                    AddInfluence(model->influences[vertexStart + weight.mVertexId], runtimeBone, weight.mWeight,
                                 static_cast<uint32_t>(maxBonesPerVertex));
            }
        }

        if (nodeIndex >= 0) {
            uint32_t fallbackBone = 0;
            bool createdFallback = false;
            for (unsigned int vertexIndex = 0; vertexIndex < sourceMesh.mNumVertices; ++vertexIndex) {
                SkinInfluence &influence = model->influences[vertexStart + vertexIndex];
                if (!HasInfluence(influence)) {
                    if (weightedVertices[vertexIndex])
                        throw std::runtime_error("min_bone_weight removes every influence of mesh '" +
                                                 std::string(sourceMesh.mName.C_Str()) + "' vertex " +
                                                 std::to_string(vertexIndex) + "; lower the threshold");
                    if (!createdFallback) {
                        fallbackBone = GetOrCreateMeshNodeFallbackBone(*model, nodeIndex, modelToMesh);
                        createdFallback = true;
                    }
                    AddInfluence(influence, fallbackBone, 1.0f);
                }
            }
        }

        for (unsigned int faceIndex = 0; faceIndex < sourceMesh.mNumFaces; ++faceIndex) {
            const aiFace &face = sourceMesh.mFaces[faceIndex];
            if (face.mNumIndices != 3)
                throw std::runtime_error("Skinned model contains a non-triangle face after triangulation");
            for (unsigned int component = 0; component < face.mNumIndices; ++component) {
                if (face.mIndices[component] >= sourceMesh.mNumVertices)
                    throw std::runtime_error("Skinned model face references an invalid vertex");
                model->indices.push_back(face.mIndices[component] + vertexStart);
                ++indexOffset;
            }
        }

        const auto [slot, inserted] =
            materialSlots.try_emplace(sourceMesh.mMaterialIndex, static_cast<uint32_t>(materialSlots.size()));
        (void)inserted;
        SubMesh subMesh;
        subMesh.indexStart = indexStart;
        subMesh.indexCount = indexOffset - indexStart;
        subMesh.vertexStart = vertexStart;
        subMesh.vertexCount = sourceMesh.mNumVertices;
        subMesh.materialSlot = slot->second;
        // InxMesh assigns compact node groups only to nodes that own meshes.
        // Keep that exact contract here: the full skeleton node index is not a
        // renderer group and may include hundreds of intervening bone nodes.
        const auto [nodeGroup, groupInserted] =
            nodeGroups.try_emplace(nodeIndex, static_cast<uint32_t>(nodeGroups.size()));
        (void)groupInserted;
        subMesh.nodeGroup = nodeGroup->second;
        subMesh.name = sourceMesh.mName.C_Str();
        if (subMesh.vertexCount > 0) {
            subMesh.boundsMin = glm::vec3(std::numeric_limits<float>::max());
            subMesh.boundsMax = glm::vec3(std::numeric_limits<float>::lowest());
            for (uint32_t index = subMesh.vertexStart; index < subMesh.vertexStart + subMesh.vertexCount; ++index) {
                subMesh.boundsMin = glm::min(subMesh.boundsMin, model->baseVertices[index].pos);
                subMesh.boundsMax = glm::max(subMesh.boundsMax, model->baseVertices[index].pos);
            }
        }
        model->subMeshes.push_back(std::move(subMesh));
        vertexOffset += sourceMesh.mNumVertices;
    }
    model->NormalizeInfluences();

    std::unordered_set<std::string> animationNames;
    for (unsigned int animationIndex = 0; importAnimations && animationIndex < scene.mNumAnimations; ++animationIndex) {
        if (!scene.mAnimations[animationIndex])
            throw std::runtime_error("Skinned model scene contains a null animation");
        const aiAnimation &sourceAnimation = *scene.mAnimations[animationIndex];
        SkinnedRuntimeAnimation animation;
        animation.name = sourceAnimation.mName.C_Str();
        if (animation.name.empty())
            animation.name = "Anim_" + std::to_string(animationIndex);
        if (!animationNames.insert(animation.name).second)
            throw std::runtime_error("Skinned model contains duplicate animation names: " + animation.name);
        animation.id = "source-";
        for (const unsigned char byte : animation.name) {
            constexpr const char *hex = "0123456789abcdef";
            animation.id += hex[byte >> 4];
            animation.id += hex[byte & 15];
        }
        animation.durationTicks = std::isfinite(sourceAnimation.mDuration) && sourceAnimation.mDuration >= 0.0
                                      ? sourceAnimation.mDuration
                                      : 0.0;
        animation.ticksPerSecond =
            std::isfinite(sourceAnimation.mTicksPerSecond) && sourceAnimation.mTicksPerSecond > 0.0
                ? sourceAnimation.mTicksPerSecond
                : 25.0;
        animation.tracks.reserve(sourceAnimation.mNumChannels);
        animation.trackByNodeIndex.assign(model->skeleton.nodes.size(), -1);
        for (unsigned int channelIndex = 0; channelIndex < sourceAnimation.mNumChannels; ++channelIndex) {
            if (!sourceAnimation.mChannels[channelIndex])
                throw std::runtime_error("Skinned model animation contains a null channel");
            const aiNodeAnim &channel = *sourceAnimation.mChannels[channelIndex];
            SkinnedRuntimeTrack track;
            const std::string channelNodeName = channel.mNodeName.C_Str();
            const auto sourceNode = model->skeleton.nodeByName.find(channelNodeName);
            if (channelNodeName.empty() || sourceNode == model->skeleton.nodeByName.end())
                throw std::runtime_error("Skinned model animation targets a node outside its source skeleton");
            track.nodeIndex = sourceNode->second;
            if (animation.trackByNodeIndex[static_cast<size_t>(track.nodeIndex)] >= 0)
                throw std::runtime_error("Skinned model animation contains an invalid duplicate node track");
            track.positions.reserve(channel.mNumPositionKeys);
            track.rotations.reserve(channel.mNumRotationKeys);
            track.scales.reserve(channel.mNumScalingKeys);
            for (unsigned int key = 0; key < channel.mNumPositionKeys; ++key) {
                const auto value = AiToGlm(channel.mPositionKeys[key].mValue);
                if (std::isfinite(channel.mPositionKeys[key].mTime) && IsFinite(value))
                    track.positions.push_back({channel.mPositionKeys[key].mTime, value});
            }
            for (unsigned int key = 0; key < channel.mNumRotationKeys; ++key) {
                const auto value = AiToGlm(channel.mRotationKeys[key].mValue);
                if (std::isfinite(channel.mRotationKeys[key].mTime) && IsFinite(value))
                    track.rotations.push_back({channel.mRotationKeys[key].mTime, NormalizeOrIdentity(value)});
            }
            for (unsigned int key = 0; key < channel.mNumScalingKeys; ++key) {
                const auto value = AiToGlm(channel.mScalingKeys[key].mValue);
                if (std::isfinite(channel.mScalingKeys[key].mTime) && IsFinite(value))
                    track.scales.push_back({channel.mScalingKeys[key].mTime, value});
            }
            if (bakeAxisConversion && track.nodeIndex == 0) {
                if (track.positions.size() != channel.mNumPositionKeys ||
                    track.rotations.size() != channel.mNumRotationKeys ||
                    track.scales.size() != channel.mNumScalingKeys)
                    throw std::runtime_error("Bake Axis Conversion rejects non-finite source-root animation keys in '" +
                                             animation.name + "'");
                track = BakeSourceRootAnimationTrack(track, sourceRootBasis, animation.durationTicks, animation.name);
            }
            animation.trackByNodeIndex[static_cast<size_t>(track.nodeIndex)] =
                static_cast<int>(animation.tracks.size());
            animation.tracks.push_back(std::move(track));
        }
        model->animations.push_back(std::move(animation));
    }
    if (!model->IsAssetPayloadValid())
        throw std::runtime_error("Skinned model conversion produced neither renderable geometry nor animation data");
    return model;
}

namespace
{
template <typename T, typename Interpolate>
std::vector<std::pair<double, T>> SliceKeys(const std::vector<std::pair<double, T>> &keys, double start, double end,
                                            Interpolate interpolate)
{
    if (keys.empty())
        return {};
    if (!std::is_sorted(keys.begin(), keys.end(), [](const auto &a, const auto &b) { return a.first < b.first; }))
        throw std::invalid_argument("animation clip source keys must be ordered by time");
    const auto sample = [&](double time) {
        auto upper =
            std::upper_bound(keys.begin(), keys.end(), time, [](double t, const auto &key) { return t < key.first; });
        if (upper == keys.begin())
            return upper->second;
        if (upper == keys.end())
            return keys.back().second;
        const auto &lower = *(upper - 1);
        return interpolate(lower.second, upper->second,
                           static_cast<float>((time - lower.first) / (upper->first - lower.first)));
    };
    std::vector<std::pair<double, T>> sliced{{0.0, sample(start)}};
    for (const auto &[time, value] : keys)
        if (time > start && time < end)
            sliced.emplace_back(time - start, value);
    sliced.emplace_back(end - start, sample(end));
    return sliced;
}

SkinnedNodePose LocalBindPose(const SkinnedRuntimeNode &node)
{
    SkinnedNodePose pose;
    pose.translation = glm::vec3(node.bindLocal[3]);
    pose.scale = {glm::length(glm::vec3(node.bindLocal[0])), glm::length(glm::vec3(node.bindLocal[1])),
                  glm::length(glm::vec3(node.bindLocal[2]))};
    glm::mat3 rotation(1.0f);
    for (glm::length_t axis = 0; axis < 3; ++axis)
        if (pose.scale[axis] > 1e-8f)
            rotation[axis] = glm::vec3(node.bindLocal[axis]) / pose.scale[axis];
    pose.rotation = NormalizeOrIdentity(glm::quat_cast(rotation));
    return pose;
}

template <typename T, typename Interpolate>
T SampleKeys(const std::vector<std::pair<double, T>> &keys, double time, Interpolate interpolate)
{
    if (keys.empty())
        throw std::invalid_argument("animation sampling requires at least one source key");
    if (!std::is_sorted(keys.begin(), keys.end(), [](const auto &a, const auto &b) { return a.first < b.first; }))
        throw std::invalid_argument("animation sampling source keys must be ordered by time");
    const auto upper = std::upper_bound(keys.begin(), keys.end(), time,
                                        [](double value, const auto &key) { return value < key.first; });
    if (upper == keys.begin())
        return upper->second;
    if (upper == keys.end())
        return keys.back().second;
    const auto &lower = *(upper - 1);
    const double span = upper->first - lower.first;
    if (!(span > 0.0))
        throw std::invalid_argument("animation sampling source keys must use strictly increasing times");
    return interpolate(lower.second, upper->second, static_cast<float>((time - lower.first) / span));
}

template <typename T, typename Interpolate>
std::vector<std::pair<double, T>> ResampleKeys(const std::vector<std::pair<double, T>> &keys, double durationTicks,
                                               double ticksPerSecond, float sampleRate, Interpolate interpolate)
{
    if (keys.size() <= 1 || sampleRate <= 0.0f)
        return keys;
    if (!(durationTicks > 0.0))
        throw std::invalid_argument("animation sampling requires positive duration for a multi-key track");
    const double durationSeconds = durationTicks / ticksPerSecond;
    const double intervalsValue = std::ceil(durationSeconds * static_cast<double>(sampleRate));
    constexpr size_t MaximumSampleIntervals = 1000000;
    if (!std::isfinite(intervalsValue) || intervalsValue > static_cast<double>(MaximumSampleIntervals))
        throw std::invalid_argument("animation sampling exceeds the supported one-million interval limit");
    const size_t intervals = std::max<size_t>(1, static_cast<size_t>(intervalsValue));
    std::vector<std::pair<double, T>> result;
    result.reserve(intervals + 1);
    for (size_t index = 0; index <= intervals; ++index) {
        const double time = index == intervals
                                ? durationTicks
                                : std::min(durationTicks, static_cast<double>(index) * ticksPerSecond / sampleRate);
        result.emplace_back(time, SampleKeys(keys, time, interpolate));
    }
    return result;
}

template <typename T, typename Interpolate, typename Error>
std::vector<std::pair<double, T>> CompressKeys(const std::vector<std::pair<double, T>> &keys, float tolerance,
                                               Interpolate interpolate, Error error)
{
    if (keys.size() <= 2 || tolerance <= 0.0f)
        return keys;
    std::vector<bool> keep(keys.size(), false);
    keep.front() = keep.back() = true;
    std::vector<std::pair<size_t, size_t>> ranges{{0, keys.size() - 1}};
    while (!ranges.empty()) {
        const auto [first, last] = ranges.back();
        ranges.pop_back();
        const double span = keys[last].first - keys[first].first;
        if (!(span > 0.0))
            throw std::invalid_argument("animation compression requires strictly increasing key times");
        float largestError = tolerance;
        size_t split = last;
        for (size_t index = first + 1; index < last; ++index) {
            const float alpha = static_cast<float>((keys[index].first - keys[first].first) / span);
            const T estimate = interpolate(keys[first].second, keys[last].second, alpha);
            const float candidate = error(keys[index].second, estimate);
            if (candidate > largestError) {
                largestError = candidate;
                split = index;
            }
        }
        if (split == last)
            continue;
        keep[split] = true;
        ranges.emplace_back(first, split);
        ranges.emplace_back(split, last);
    }
    std::vector<std::pair<double, T>> result;
    result.reserve(keys.size());
    for (size_t index = 0; index < keys.size(); ++index)
        if (keep[index])
            result.push_back(keys[index]);
    return result;
}
} // namespace

void SkinnedModelImporter::ApplyAnimationClips(InxSkinnedMesh &model, const MeshImportSettings &settings)
{
    if (!settings.importAnimations || !settings.customAnimationClips)
        return;
    MeshImportSettings::RequireAnimationClips(settings.animationClips);
    std::vector<SkinnedRuntimeAnimation> clips;
    for (const auto &spec : settings.animationClips) {
        const auto sourceName = spec.at("source_take").get<std::string>();
        const auto *source = model.FindAnimation(sourceName);
        if (!source)
            throw std::invalid_argument("animation clip source take is missing: " + sourceName);
        const double start = spec.at("start").get<double>() * source->ticksPerSecond;
        const double end = spec.at("end").get<double>() * source->ticksPerSecond;
        if (end > source->durationTicks + 1e-6 * std::max(1.0, source->durationTicks))
            throw std::invalid_argument("animation clip end exceeds source take duration: " + sourceName);
        SkinnedRuntimeAnimation clip = *source;
        clip.name = spec.at("name").get<std::string>();
        clip.id = spec.at("id").get<std::string>();
        clip.durationTicks = end - start;
        for (auto &track : clip.tracks) {
            const auto linear = [](const glm::vec3 &a, const glm::vec3 &b, float t) { return glm::mix(a, b, t); };
            track.positions = SliceKeys(track.positions, start, end, linear);
            track.scales = SliceKeys(track.scales, start, end, linear);
            track.rotations =
                SliceKeys(track.rotations, start, end, [](const glm::quat &a, const glm::quat &b, float t) {
                    return glm::normalize(glm::slerp(a, b, t));
                });
        }
        clips.push_back(std::move(clip));
    }
    model.animations = std::move(clips);
}

void SkinnedModelImporter::ApplyRigSettings(InxSkinnedMesh &model, const MeshImportSettings &settings,
                                            const InxSkinnedMesh *copiedDefinition)
{
    MeshImportSettings::RequireSkeletonDefinitionMode(settings.skeletonDefinitionMode);
    MeshImportSettings::RequireRigNodeName(settings.rigRootNode, "rig_root_node");
    MeshImportSettings::RequireRigNodeName(settings.skeletonDefinitionGuid, "skeleton_definition_guid");
    MeshImportSettings::RequireRigNodeName(settings.skeletonDefinitionId, "skeleton_definition_id");
    MeshImportSettings::RequireExposedBones(settings.exposedBones);
    if (settings.skeletonDefinitionId.empty())
        throw std::invalid_argument("skeleton_definition_id must not be empty");
    if (!model.skeleton.IsValid())
        throw std::invalid_argument("rig import requires a valid source skeleton");

    std::string rootIdentity = settings.rigRootNode;
    if (settings.skeletonDefinitionMode == "copy") {
        if (!settings.rigRootNode.empty() || settings.skeletonDefinitionGuid.empty() || !copiedDefinition ||
            copiedDefinition->skeletonDefinitionGuid != settings.skeletonDefinitionGuid ||
            copiedDefinition->skeletonDefinitionId != settings.skeletonDefinitionId ||
            copiedDefinition->skeletonRootNodeIndex < 0 ||
            static_cast<size_t>(copiedDefinition->skeletonRootNodeIndex) >= copiedDefinition->skeleton.nodes.size())
            throw std::invalid_argument(
                "Copy skeleton definition requires a resolvable GUID and stable subresource id");
        rootIdentity =
            copiedDefinition->skeleton.nodes[static_cast<size_t>(copiedDefinition->skeletonRootNodeIndex)].name;
        const auto sourceRoot = model.skeleton.nodeByName.find(rootIdentity);
        if (sourceRoot == model.skeleton.nodeByName.end())
            throw std::invalid_argument("copied skeleton definition root is missing from this model: " + rootIdentity);
        if (model.skeleton.bones.size() != copiedDefinition->skeleton.bones.size())
            throw std::invalid_argument("copied skeleton definition has an incompatible deformation bone count");
        const auto ancestry = [](const Skeleton &skeleton, int index) {
            std::vector<std::string> names;
            for (; index >= 0; index = skeleton.nodes[static_cast<size_t>(index)].parent)
                names.push_back(skeleton.nodes[static_cast<size_t>(index)].name);
            return names;
        };
        for (const auto &bone : copiedDefinition->skeleton.bones) {
            const auto localBone = model.skeleton.boneByName.find(bone.name);
            if (localBone == model.skeleton.boneByName.end() || bone.nodeIndex < 0 ||
                localBone->second >= model.skeleton.bones.size() ||
                model.skeleton.bones[localBone->second].nodeIndex < 0 ||
                ancestry(copiedDefinition->skeleton, bone.nodeIndex) !=
                    ancestry(model.skeleton, model.skeleton.bones[localBone->second].nodeIndex))
                throw std::invalid_argument("copied skeleton definition has an incompatible bone hierarchy: " +
                                            bone.name);
        }
    } else if (copiedDefinition || !settings.skeletonDefinitionGuid.empty()) {
        throw std::invalid_argument("Create skeleton definition must not reference another model");
    }
    const auto root = rootIdentity.empty() ? model.skeleton.nodes.begin()
                                           : std::find_if(model.skeleton.nodes.begin(), model.skeleton.nodes.end(),
                                                          [&](const auto &node) { return node.name == rootIdentity; });
    if (root == model.skeleton.nodes.end())
        throw std::invalid_argument("rig_root_node does not identify a source skeleton node: " + rootIdentity);
    const int rootIndex = static_cast<int>(std::distance(model.skeleton.nodes.begin(), root));
    const auto isDescendant = [&](int node) {
        for (; node >= 0; node = model.skeleton.nodes[static_cast<size_t>(node)].parent)
            if (node == rootIndex)
                return true;
        return false;
    };

    std::vector<int> exposed;
    exposed.reserve(settings.exposedBones.size());
    for (const auto &identity : settings.exposedBones) {
        const auto found = model.skeleton.nodeByName.find(identity.get<std::string>());
        if (found == model.skeleton.nodeByName.end() || !isDescendant(found->second))
            throw std::invalid_argument("exposed_bones references a node outside the selected rig root: " +
                                        identity.get<std::string>());
        exposed.push_back(found->second);
    }

    model.skeletonDefinitionGuid =
        settings.skeletonDefinitionMode == "copy" ? settings.skeletonDefinitionGuid : model.guid;
    model.skeletonDefinitionId = settings.skeletonDefinitionId;
    model.skeletonRootNodeIndex = rootIndex;
    model.exposedSkeletonNodeIndices = exposed;
    BuildHumanoidRigDefinition(model, settings, rootIndex);
    if (!settings.optimizeBoneHierarchy)
        return;

    // Pruning is deliberately conservative: deformation joints, every
    // animated channel, author-exposed attachment and their ancestry survive.
    // This preserves bind poses and track semantics instead of treating an
    // optimization toggle as permission to remove script-visible sockets.
    std::vector<bool> keep(model.skeleton.nodes.size(), false);
    const auto keepAncestors = [&](int node) {
        for (; node >= 0; node = model.skeleton.nodes[static_cast<size_t>(node)].parent) {
            keep[static_cast<size_t>(node)] = true;
            if (node == rootIndex)
                return;
        }
        throw std::invalid_argument("rig root does not contain a referenced deformation or animation node");
    };
    keepAncestors(rootIndex);
    for (const auto &bone : model.skeleton.bones) {
        if (bone.nodeIndex < 0)
            throw std::invalid_argument("bone pruning requires every deformation bone to resolve to a skeleton node");
        keepAncestors(bone.nodeIndex);
    }
    for (const auto &animation : model.animations)
        for (const auto &track : animation.tracks)
            keepAncestors(track.nodeIndex);
    for (const int node : exposed)
        keepAncestors(node);
    // Clip masks are authored node references too.  They are applied after
    // this pass alongside sampling/compression, so retain their targets now.
    for (const auto &extra : settings.animationClipExtras)
        for (const auto &identity : extra.at("bone_mask")) {
            const auto found = model.skeleton.nodeByName.find(identity.get<std::string>());
            if (found == model.skeleton.nodeByName.end())
                throw std::invalid_argument("animation bone mask references a missing skeleton node: " +
                                            identity.get<std::string>());
            keepAncestors(found->second);
        }

    std::vector<int> remap(model.skeleton.nodes.size(), -1);
    Skeleton pruned;
    for (size_t oldIndex = 0; oldIndex < model.skeleton.nodes.size(); ++oldIndex) {
        if (!keep[oldIndex])
            continue;
        auto node = model.skeleton.nodes[oldIndex];
        node.parent = node.parent < 0 ? -1 : remap[static_cast<size_t>(node.parent)];
        if (node.parent < -1)
            throw std::logic_error("rig pruning lost a required ancestor");
        node.bindGlobal = node.parent >= 0 ? pruned.nodes[static_cast<size_t>(node.parent)].bindGlobal * node.bindLocal
                                           : node.bindLocal;
        const int newIndex = static_cast<int>(pruned.nodes.size());
        pruned.nodeByName.emplace(node.name, newIndex);
        pruned.nodes.push_back(std::move(node));
        remap[oldIndex] = newIndex;
    }
    for (auto bone : model.skeleton.bones) {
        bone.nodeIndex = remap[static_cast<size_t>(bone.nodeIndex)];
        pruned.boneByName.emplace(bone.name, static_cast<uint32_t>(pruned.bones.size()));
        pruned.bones.push_back(std::move(bone));
    }
    for (auto &animation : model.animations) {
        for (auto &track : animation.tracks)
            track.nodeIndex = remap[static_cast<size_t>(track.nodeIndex)];
        animation.trackByNodeIndex.assign(pruned.nodes.size(), -1);
        for (size_t index = 0; index < animation.tracks.size(); ++index)
            animation.trackByNodeIndex[static_cast<size_t>(animation.tracks[index].nodeIndex)] =
                static_cast<int>(index);
    }
    for (auto &index : model.exposedSkeletonNodeIndices)
        index = remap[static_cast<size_t>(index)];
    for (auto &[_, index] : model.humanoid.bones)
        index = remap[static_cast<size_t>(index)];
    model.skeletonRootNodeIndex = remap[static_cast<size_t>(rootIndex)];
    model.skeleton = std::move(pruned);
}

void SkinnedModelImporter::ApplyAnimationSettings(InxSkinnedMesh &model, const MeshImportSettings &settings)
{
    for (const auto &field : MeshImportSettings::Scalars)
        if (std::string_view(field.name).rfind("animation_", 0) == 0)
            MeshImportSettings::RequireScalar(field, settings.*(field.member));
    MeshImportSettings::RequireAnimationReferencePose(settings.animationReferencePose);
    MeshImportSettings::RequireAnimationClipExtras(settings.animationClipExtras);

    std::unordered_map<std::string, const nlohmann::json *> extrasByClip;
    for (const auto &extra : settings.animationClipExtras)
        extrasByClip.emplace(extra.at("clip_id").get<std::string>(), &extra);
    for (const auto &entry : extrasByClip) {
        const auto &clipId = entry.first;
        const auto *extra = entry.second;
        const auto animation = std::find_if(model.animations.begin(), model.animations.end(),
                                            [&](const auto &candidate) { return candidate.id == clipId; });
        if (animation == model.animations.end())
            throw std::invalid_argument("animation_clip_extras references a missing clip: " + clipId);
        for (const auto &bone : extra->at("bone_mask")) {
            const auto boneName = bone.get<std::string>();
            if (model.skeleton.nodeByName.find(boneName) == model.skeleton.nodeByName.end())
                throw std::invalid_argument("animation bone mask references a missing skeleton node: " + boneName);
        }
    }

    const auto linear = [](const glm::vec3 &a, const glm::vec3 &b, float amount) { return glm::mix(a, b, amount); };
    const auto spherical = [](const glm::quat &a, const glm::quat &b, float amount) {
        return glm::normalize(glm::slerp(a, b, amount));
    };
    const auto vectorError = [](const glm::vec3 &actual, const glm::vec3 &estimate) {
        return glm::length(actual - estimate);
    };
    const auto rotationError = [](const glm::quat &actual, const glm::quat &estimate) {
        const float alignment =
            std::clamp(std::abs(glm::dot(glm::normalize(actual), glm::normalize(estimate))), 0.0f, 1.0f);
        return glm::degrees(2.0f * std::acos(alignment));
    };

    auto processed = model.animations;
    for (auto &animation : processed) {
        if (!std::isfinite(animation.durationTicks) || animation.durationTicks < 0.0 ||
            !std::isfinite(animation.ticksPerSecond) || animation.ticksPerSecond <= 0.0)
            throw std::invalid_argument("animation settings require a finite duration and sample clock");
        for (auto &track : animation.tracks) {
            track.positions = ResampleKeys(track.positions, animation.durationTicks, animation.ticksPerSecond,
                                           settings.animationSampleRate, linear);
            track.rotations = ResampleKeys(track.rotations, animation.durationTicks, animation.ticksPerSecond,
                                           settings.animationSampleRate, spherical);
            track.scales = ResampleKeys(track.scales, animation.durationTicks, animation.ticksPerSecond,
                                        settings.animationSampleRate, linear);
            track.positions = CompressKeys(track.positions, settings.animationPositionError, linear, vectorError);
            track.rotations = CompressKeys(track.rotations, settings.animationRotationError, spherical, rotationError);
            track.scales = CompressKeys(track.scales, settings.animationScaleError, linear, vectorError);
        }
        animation.defaultLoop = settings.animationLoopTime;
        animation.rootMotionReferencePose = settings.animationReferencePose;
        animation.rootMotionNodeIndex = -1;
        animation.rootMotionPositions.clear();
        animation.rootMotionRotations.clear();
        animation.curves.clear();
        animation.events.clear();
        animation.boneMask.clear();
        if (const auto found = extrasByClip.find(animation.id); found != extrasByClip.end()) {
            const auto &extra = *found->second;
            for (const auto &curveDocument : extra.at("curves")) {
                SkinnedRuntimeFloatCurve curve;
                curve.name = curveDocument.at("name").get<std::string>();
                for (const auto &key : curveDocument.at("keys"))
                    curve.keys.emplace_back(key.at("time_normalized").get<double>(), key.at("value").get<float>());
                animation.curves.push_back(std::move(curve));
            }
            for (const auto &eventDocument : extra.at("events")) {
                SkinnedRuntimeEvent event;
                event.normalizedTime = eventDocument.at("time_normalized").get<double>();
                event.function = eventDocument.at("function").get<std::string>();
                event.stringArgument = eventDocument.at("string_arg").get<std::string>();
                event.numberArgument = eventDocument.at("number_arg").get<double>();
                animation.events.push_back(std::move(event));
            }
            for (const auto &bone : extra.at("bone_mask"))
                animation.boneMask.push_back(bone.get<std::string>());
        }
        if (!settings.animationApplyRootMotion)
            continue;

        // Pick the highest animated node. Exporters commonly insert an
        // unanimated scene root above Hips; choosing the first animated node
        // by hierarchy depth preserves that authored root motion without a
        // format-specific bone-name heuristic.
        int rootTrackIndex = -1;
        int rootDepth = std::numeric_limits<int>::max();
        for (size_t trackIndex = 0; trackIndex < animation.tracks.size(); ++trackIndex) {
            const auto &track = animation.tracks[trackIndex];
            if (track.positions.empty() && track.rotations.empty())
                continue;
            int depth = 0;
            for (int node = track.nodeIndex; node >= 0; node = model.skeleton.nodes[static_cast<size_t>(node)].parent)
                ++depth;
            if (depth < rootDepth) {
                rootDepth = depth;
                rootTrackIndex = static_cast<int>(trackIndex);
            }
        }
        if (rootTrackIndex < 0)
            throw std::invalid_argument("root motion import requires an animated skeleton node");
        auto &rootTrack = animation.tracks[static_cast<size_t>(rootTrackIndex)];
        animation.rootMotionNodeIndex = rootTrack.nodeIndex;
        animation.rootMotionPositions = rootTrack.positions;
        animation.rootMotionRotations = rootTrack.rotations;
        const auto bind = LocalBindPose(model.skeleton.nodes[static_cast<size_t>(rootTrack.nodeIndex)]);
        const glm::vec3 referenceTranslation =
            settings.animationReferencePose == "first_frame" && !rootTrack.positions.empty()
                ? SampleKeys(rootTrack.positions, 0.0, linear)
                : bind.translation;
        const glm::quat referenceRotation =
            settings.animationReferencePose == "first_frame" && !rootTrack.rotations.empty()
                ? SampleKeys(rootTrack.rotations, 0.0, spherical)
                : bind.rotation;
        if (!rootTrack.positions.empty())
            rootTrack.positions = {{0.0, referenceTranslation}};
        if (!rootTrack.rotations.empty())
            rootTrack.rotations = {{0.0, referenceRotation}};
    }
    model.animations = std::move(processed);
}

std::shared_ptr<InxSkinnedMesh> SkinnedModelImporter::ImportSource(const std::string &sourceGuid,
                                                                   const std::string &sourcePath, float scaleFactor)
{
    const auto filePath = ToFsPath(sourcePath);
    if (!std::filesystem::is_regular_file(filePath))
        throw std::runtime_error("Skinned model source file not found: " + sourcePath);
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Skinned model source file cannot be opened: " + sourcePath);
    const auto size = file.tellg();
    if (size <= 0)
        throw std::runtime_error("Skinned model source file is empty: " + sourcePath);
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(bytes.data(), size))
        throw std::runtime_error("Skinned model source file cannot be read: " + sourcePath);

    std::string extension = FromFsPath(filePath.extension());
    if (!extension.empty() && extension.front() == '.')
        extension.erase(extension.begin());
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    Assimp::Importer importer;
    const aiScene *scene =
        importer.ReadFileFromMemory(bytes.data(), bytes.size(), BuildAssimpFlags(), extension.c_str());
    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode)
        throw std::runtime_error("Skinned model Assimp import failed for '" + sourcePath +
                                 "': " + importer.GetErrorString());
    return ConvertScene(*scene, sourceGuid, sourcePath, scaleFactor);
}

} // namespace infernux
