#include <function/renderer/RenderGraphSchedule.h>

#include <cassert>
#include <iostream>

int main()
{
    using namespace infernux;
    RenderGraphIdentitySource identity;
    const GraphResourceHandle oldValue{identity.Current(), 0, 1};
    const GraphResourceHandle newValue{identity.Current(), 0, 2};
    const GraphResourceHandle output{identity.Current(), 1, 1};
    std::vector<GraphSchedulePass> passes(4);
    passes[0].writes = {oldValue};
    passes[1].reads = {{oldValue, false}};
    passes[1].writes = {newValue};
    passes[1].externalWrite = true;
    passes[2].reads = {{oldValue}};
    passes[2].writes = {output};
    passes[3].writes = {{identity.Current(), 2, 1}};
    const auto branch = CompileRenderGraphSchedule(passes, output);
    // A later-declared reader must see the old version before the same storage
    // is cleared, while an unreachable branch must never execute.
    assert((branch.order == std::vector<uint32_t>{0, 2, 1}));
    assert(branch.retention[3] == GraphPassRetention::Unreachable);

    passes[2].reads.clear();
    const auto clear = CompileRenderGraphSchedule(passes, newValue);
    assert((clear.order == std::vector<uint32_t>{1}));
    assert(clear.retention[0] == GraphPassRetention::Unreachable);
    passes[3].sideEffect = true;
    const auto sideEffect = CompileRenderGraphSchedule(passes, newValue);
    assert((sideEffect.order == std::vector<uint32_t>{1, 3}));

    passes[2].reads = {{oldValue}};
    passes[0].reads = {{output}};
    const auto cycle = CompileRenderGraphSchedule(passes, output);
    assert(cycle.order.empty());
    assert((cycle.unresolved == std::vector<uint32_t>{0, 1, 2}));
    std::cout << "PASS shared graph schedule: version branches, clears, dead passes, side effects and cycles\n";
}
