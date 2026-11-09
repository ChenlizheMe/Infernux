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

    return 0;
}
