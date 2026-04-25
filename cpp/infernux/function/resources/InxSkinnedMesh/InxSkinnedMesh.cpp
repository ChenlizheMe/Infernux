#include "InxSkinnedMesh.h"

#include <core/config/MathConstants.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <cmath>

namespace infernux
{

namespace
{

static glm::mat4 MakeTRS(const glm::vec3 &t, const glm::quat &r, const glm::vec3 &s)
{
    return glm::translate(glm::mat4(1.0f), t) * glm::toMat4(r) * glm::scale(glm::mat4(1.0f), s);
}

template <typename T> static size_t FindKeySpan(const std::vector<std::pair<double, T>> &keys, double t)
{
    if (keys.size() < 2)
        return 0;
    for (size_t i = 0; i + 1 < keys.size(); ++i)
        if (t < keys[i + 1].first)
            return i;
    return keys.size() - 2;
}

static glm::vec3 SampleVec3(const std::vector<std::pair<double, glm::vec3>> &keys, double t, const glm::vec3 &fallback)
{
    if (keys.empty())
        return fallback;
    if (keys.size() == 1)
        return keys[0].second;
    const size_t i = FindKeySpan(keys, t);
    const auto &a = keys[i];
    const auto &b = keys[i + 1];
    const double span = std::max(b.first - a.first, 1e-8);
    const float f = static_cast<float>((t - a.first) / span);
    return glm::mix(a.second, b.second, glm::clamp(f, 0.0f, 1.0f));
}

static glm::quat SampleQuat(const std::vector<std::pair<double, glm::quat>> &keys, double t, const glm::quat &fallback)
{
    if (keys.empty())
        return fallback;
    if (keys.size() == 1)
        return keys[0].second;
    const size_t i = FindKeySpan(keys, t);
    const auto &a = keys[i];
    const auto &b = keys[i + 1];
    const double span = std::max(b.first - a.first, 1e-8);
    const float f = static_cast<float>((t - a.first) / span);
    return glm::normalize(glm::slerp(a.second, b.second, glm::clamp(f, 0.0f, 1.0f)));
}

static void DecomposeTRS(const glm::mat4 &m, glm::vec3 &t, glm::quat &r, glm::vec3 &s)
{
    t = glm::vec3(m[3]);
    s = glm::vec3(glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])), glm::length(glm::vec3(m[2])));
    glm::mat3 rot(1.0f);
    if (s.x > kEpsilon)
        rot[0] = glm::vec3(m[0]) / s.x;
    if (s.y > kEpsilon)
        rot[1] = glm::vec3(m[1]) / s.y;
    if (s.z > kEpsilon)
        rot[2] = glm::vec3(m[2]) / s.z;
    r = glm::normalize(glm::quat_cast(rot));
}

static SkinnedNodePose BindNodePose(const SkinnedRuntimeNode &node)
{
    SkinnedNodePose pose;
    DecomposeTRS(node.bindLocal, pose.translation, pose.rotation, pose.scale);
    return pose;
}

static double ToAnimationTicks(const SkinnedRuntimeAnimation *anim, float seconds)
{
    if (!anim)
        return 0.0;
    double tTicks = static_cast<double>(seconds) * anim->ticksPerSecond;
    if (anim->durationTicks > 0.0)
        tTicks = std::fmod(tTicks, anim->durationTicks);
    return tTicks;
}

static SkinnedNodePose BlendNodePose(const SkinnedNodePose &a, const SkinnedNodePose &b, float weight)
{
    const float w = glm::clamp(weight, 0.0f, 1.0f);
    SkinnedNodePose pose;
    pose.translation = glm::mix(a.translation, b.translation, w);
    pose.rotation = glm::normalize(glm::slerp(a.rotation, b.rotation, w));
    pose.scale = glm::mix(a.scale, b.scale, w);
    return pose;
}

} // namespace

float SkinnedRuntimeAnimation::DurationSeconds() const
{
    if (durationTicks <= 0.0 || ticksPerSecond <= 0.0)
        return 0.0f;
    return static_cast<float>(durationTicks / ticksPerSecond);
}

const SkinnedRuntimeAnimation *InxSkinnedMesh::FindAnimation(const std::string &takeName) const
{
    if (animations.empty())
        return nullptr;
    if (!takeName.empty()) {
        for (const auto &anim : animations)
            if (anim.name == takeName)
                return &anim;
    }
    return &animations.front();
}

float InxSkinnedMesh::GetAnimationDurationSeconds(const std::string &takeName) const
{
    const SkinnedRuntimeAnimation *anim = FindAnimation(takeName);
    return anim ? anim->DurationSeconds() : 0.0f;
}

void InxSkinnedMesh::NormalizeInfluences()
{
    for (size_t vi = 0; vi < influences.size(); ++vi) {
        auto &inf = influences[vi];
        float total = 0.0f;
        for (float w : inf.weight)
            total += w;
        if (total <= kEpsilon) {
            if (!bones.empty()) {
                inf.boneIndex[0] = 0;
                inf.weight[0] = 1.0f;
            }
        } else {
            for (float &w : inf.weight)
                w /= total;
        }

        if (vi < baseVertices.size()) {
            baseVertices[vi].boneIndices =
                glm::uvec4(inf.boneIndex[0], inf.boneIndex[1], inf.boneIndex[2], inf.boneIndex[3]);
            baseVertices[vi].boneWeights = glm::vec4(inf.weight[0], inf.weight[1], inf.weight[2], inf.weight[3]);
        }
    }
}

SkinnedNodePose InxSkinnedMesh::SampleNodePose(const SkinnedRuntimeAnimation *anim, const SkinnedRuntimeNode &node,
                                               double tTicks) const
{
    SkinnedNodePose pose = BindNodePose(node);
    if (!anim)
        return pose;

    auto trIt = anim->trackByNode.find(node.name);
    if (trIt == anim->trackByNode.end())
        return pose;

    const SkinnedRuntimeTrack &track = anim->tracks[trIt->second];
    pose.translation = SampleVec3(track.positions, tTicks, pose.translation);
    pose.rotation = SampleQuat(track.rotations, tTicks, pose.rotation);
    pose.scale = SampleVec3(track.scales, tTicks, pose.scale);
    return pose;
}

std::vector<glm::mat4> InxSkinnedMesh::BuildBoneMatrices(const SkinnedSampleRequest &request) const
{
    const SkinnedRuntimeAnimation *anim = FindAnimation(request.takeName);
    const SkinnedRuntimeAnimation *blendAnim = (request.blendWeight > 0.0f && !request.blendTakeName.empty())
                                                  ? FindAnimation(request.blendTakeName)
                                                  : nullptr;
    const double tTicks = ToAnimationTicks(anim, request.timeSeconds);
    const double blendTicks = ToAnimationTicks(blendAnim, request.blendTimeSeconds);
    const float w = (blendAnim && blendAnim != anim) ? glm::clamp(request.blendWeight, 0.0f, 1.0f) : 0.0f;

    std::vector<glm::mat4> globals(nodes.size(), glm::mat4(1.0f));
    for (size_t ni = 0; ni < nodes.size(); ++ni) {
        const SkinnedRuntimeNode &node = nodes[ni];
        SkinnedNodePose pose = SampleNodePose(anim, node, tTicks);
        if (w > 0.0f) {
            SkinnedNodePose target = SampleNodePose(blendAnim, node, blendTicks);
            pose = BlendNodePose(pose, target, w);
        }

        glm::mat4 local = MakeTRS(pose.translation, pose.rotation, pose.scale);
        globals[ni] = (node.parent >= 0) ? globals[static_cast<size_t>(node.parent)] * local : local;
    }

    std::vector<glm::mat4> boneMatrices(bones.size(), glm::mat4(1.0f));
    for (size_t bi = 0; bi < bones.size(); ++bi) {
        const SkinnedRuntimeBone &bone = bones[bi];
        if (bone.nodeIndex >= 0 && static_cast<size_t>(bone.nodeIndex) < globals.size())
            boneMatrices[bi] = globals[static_cast<size_t>(bone.nodeIndex)] * bone.inverseBind;
    }
    return boneMatrices;
}

std::vector<glm::mat4> InxSkinnedMesh::BuildGpuBonePalette(const SkinnedSampleRequest &request) const
{
    std::vector<glm::mat4> palette = BuildBoneMatrices(request);
    const glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor));
    for (glm::mat4 &m : palette)
        m = scale * m;
    return palette;
}

std::vector<Vertex> InxSkinnedMesh::SampleVertices(const SkinnedSampleRequest &request) const
{
    std::vector<Vertex> outVertices = baseVertices;
    if (outVertices.empty())
        return outVertices;

    if (bones.empty()) {
        for (auto &v : outVertices)
            v.pos *= scaleFactor;
        return outVertices;
    }

    const std::vector<glm::mat4> boneMatrices = BuildBoneMatrices(request);

    for (size_t vi = 0; vi < outVertices.size() && vi < influences.size(); ++vi) {
        const Vertex &base = baseVertices[vi];
        const SkinInfluence &inf = influences[vi];

        glm::vec4 p(0.0f);
        glm::vec3 n(0.0f);
        glm::vec3 tangent(0.0f);
        float total = 0.0f;

        for (uint32_t i = 0; i < kMaxSkinInfluences; ++i) {
            const float w = inf.weight[i];
            const uint32_t bi = inf.boneIndex[i];
            if (w <= 0.0f || bi >= boneMatrices.size())
                continue;
            const glm::mat4 &m = boneMatrices[bi];
            p += w * (m * glm::vec4(base.pos, 1.0f));
            n += w * (glm::mat3(m) * base.normal);
            tangent += w * (glm::mat3(m) * glm::vec3(base.tangent));
            total += w;
        }

        if (total > kEpsilon) {
            outVertices[vi].pos = glm::vec3(p) * scaleFactor;
            if (glm::length2(n) > kEpsilon)
                outVertices[vi].normal = glm::normalize(n);
            if (glm::length2(tangent) > kEpsilon)
                outVertices[vi].tangent = glm::vec4(glm::normalize(tangent), base.tangent.w);
        } else {
            outVertices[vi].pos = base.pos * scaleFactor;
        }
    }
    return outVertices;
}

} // namespace infernux
