#include <function/renderer/gui/WorldUIOcclusionPlan.h>

#include <cassert>

using namespace infernux;

int main()
{
    const auto ordinary = BuildWorldUIOcclusionPlan({{}, {}, {}});
    assert(ordinary.uniqueIgnoredObjectIds.empty());
    assert(ordinary.runs.size() == 1);
    assert(ordinary.runs[0] == (WorldUIOcclusionRun{0, 3, 0, false}));

    // Repeated A after B retains one alternate depth for A, while the UI
    // remains in the original transparent draw order.
    const auto interleaved = BuildWorldUIOcclusionPlan({
        {11, false},
        {11, false},
        {22, false},
        {11, false},
        {},
        {},
        {22, false},
        {0, true},
    });
    assert((interleaved.uniqueIgnoredObjectIds == std::vector<uint64_t>{11, 22}));
    assert(interleaved.runs.size() == 6);
    assert(interleaved.runs[0] == (WorldUIOcclusionRun{0, 2, 11, false}));
    assert(interleaved.runs[1] == (WorldUIOcclusionRun{2, 3, 22, false}));
    assert(interleaved.runs[2] == (WorldUIOcclusionRun{3, 4, 11, false}));
    assert(interleaved.runs[3] == (WorldUIOcclusionRun{4, 6, 0, false}));
    assert(interleaved.runs[4] == (WorldUIOcclusionRun{6, 7, 22, false}));
    assert(interleaved.runs[5] == (WorldUIOcclusionRun{7, 8, 0, true}));
}
