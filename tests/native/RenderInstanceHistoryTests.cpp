#include <function/renderer/RenderInstanceHistory.h>

#include <cassert>
#include <cstdint>

using infernux::PackGPUObjectId;
using infernux::RenderDomain;
using infernux::RenderDrawIdentity;
using infernux::RenderInstanceHistory;
using infernux::UnpackGPUObjectId;

namespace
{
bool MatrixEquals(const glm::mat4 &lhs, const glm::mat4 &rhs)
{
    for (glm::length_t column = 0; column < 4; ++column) {
        if (lhs[column] != rhs[column])
            return false;
    }
    return true;
}
} // namespace

int main()
{
    constexpr uint64_t objectId = 0xfedcba9876543210ull;
    assert(UnpackGPUObjectId(PackGPUObjectId(objectId)) == objectId);

    const RenderDrawIdentity first{101, 0, RenderDomain::SceneGeometry};
    const RenderDrawIdentity second{202, 0, RenderDomain::SceneGeometry};
    const glm::mat4 modelA(1.0f);
    glm::mat4 modelB(1.0f);
    modelB[3].x = 4.0f;
    glm::mat4 modelC(1.0f);
    modelC[3].x = 9.0f;

    RenderInstanceHistory history;
    history.BeginFrame();
    assert(!history.Previous(first, 0));
    history.Record(first, modelA, {}, 0);
    history.CommitFrame();

    history.BeginFrame();
    const auto *continuous = history.Previous(first, 0);
    assert(continuous && MatrixEquals(continuous->model, modelA));
    history.Record(first, modelB, {}, 0);

    // A second pass in the same logical frame sees the same accepted history.
    const auto *repeated = history.Previous(first, 0);
    assert(repeated && MatrixEquals(repeated->model, modelA));
    history.Record(first, modelC, {}, 0);

    // Independent identities never share transform history.
    assert(!history.Previous(second, 0));
    history.Record(second, modelC, {}, 0);
    history.CommitFrame();

    // Missing a frame suppresses stale motion when the primitive reappears.
    history.BeginFrame();
    const auto *secondFrame = history.Previous(second, 0);
    assert(secondFrame && MatrixEquals(secondFrame->model, modelC));
    history.Record(second, modelC, {}, 0);
    history.CommitFrame();
    history.BeginFrame();
    assert(!history.Previous(first, 0));
    history.Record(first, modelC, {}, 0);

    // Invalid identities do not enter the accepted history.
    history.Record({}, modelB, {}, 0);
    history.CommitFrame();
    assert(!history.Previous({}, 0));

    history.Clear();
    assert(!history.Previous(first, 0) && !history.Previous(second, 0));

    // Device generations advance independently of a camera. Only an accepted
    // rendering sample advances that view's previous pose, including skips.
    auto deviceA = std::make_shared<infernux::RenderPoseValue>();
    deviceA->sourceIdentity = std::make_shared<const uint8_t>(0);
    deviceA->generation = 1;
    deviceA->firstModel = 3;
    deviceA->firstBounds = 5;
    const infernux::RenderPoseBinding poseA{deviceA, 2, 7};
    history.BeginFrame();
    assert(!history.Previous(first, 0, poseA));
    history.Record(first, modelA, {}, 0, poseA);
    history.CommitFrame();
    auto deviceB = std::make_shared<infernux::RenderPoseValue>(*deviceA);
    deviceB->generation = 4;
    deviceB->firstModel = 10;
    const infernux::RenderPoseBinding poseB{deviceB, 2, 7};
    history.BeginFrame();
    assert(history.Previous(first, 0, poseB)->devicePose.value == deviceA);
    history.Record(first, modelB, {}, 0, poseB);
    // Abandon this recording and start another. No physics tick or abandoned
    // render recording is permitted to replace the last accepted sample.
    history.BeginFrame();
    assert(history.Previous(first, 0, poseB)->devicePose.value == deviceA);
    assert(!history.Previous(first, 0));
    assert(!history.Previous(first, 0, {deviceB, 1, 7}));
    assert(!history.Previous(first, 0, {deviceB, 2, 8}));
    auto unrelated = std::make_shared<infernux::RenderPoseValue>(*deviceB);
    unrelated->sourceIdentity = std::make_shared<const uint8_t>(0);
    assert(!history.Previous(first, 0, {unrelated, 2, 7}));

    infernux::GPUInstanceAuxData auxiliary;
    infernux::ApplyDevicePose(auxiliary, poseB, &poseA);
    assert(auxiliary.flags == infernux::kGPUInstanceAuxFlagDevicePose);
    assert(auxiliary.poseIndices == glm::uvec4(12, 5, 7, 0));
    history.Record(first, modelC, {}, 0, poseB);
    history.CommitFrame();
    history.BeginFrame();
    assert(history.Previous(first, 0, poseB)->devicePose.value == deviceB);
    RenderInstanceHistory otherView;
    otherView.BeginFrame();
    assert(!otherView.Previous(first, 0, poseB));

    return 0;
}
