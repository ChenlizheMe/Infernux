// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/island.py and _src/sleep.py.

#include "model.slang"
#include "contact.slang"
#include "constraint_data.slang"
#include "awake_dispatch.slang"

struct TreeActivity
{
    Atomic<uint> wakeParent;
    Atomic<uint> islandParent;
    Atomic<uint> wakeAge;
    Atomic<uint> quietAge;
};

// Header: tree, body, DOF and joint counts.
// Layout: tree table, body-to-tree table, DOF table and dependency count.
// Tree: first body, body count, first DOF and DOF count, followed by first joint,
// joint count and sleep eligibility. DOF records hold tree and velocity scale.
[[vk::binding(0, 0)]] StructuredBuffer<uint4> sleepModel;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint2> treeSleep; // quiet steps, sleeping group root or NoIndex
[[vk::binding(2, 0)]] RWStructuredBuffer<TreeActivity> activity;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> qvel;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> qacc;
[[vk::binding(5, 0)]] StructuredBuffer<float> appliedForce;
[[vk::binding(6, 0)]] StructuredBuffer<SpatialMotion> appliedWrench;
[[vk::binding(7, 0)]] StructuredBuffer<uint> bodyMoved;
[[vk::binding(8, 0)]] StructuredBuffer<uint4> sleepSettings; // quiet steps required, tolerance (float bits)
[[vk::binding(9, 0)]] StructuredBuffer<Contact> contacts;
[[vk::binding(10, 0)]] RWStructuredBuffer<Atomic<uint>> contactCounts;
[[vk::binding(11, 0)]] StructuredBuffer<uint4> constraintSettings;
[[vk::binding(12, 0)]] StructuredBuffer<uint4> colliderTopology;
[[vk::binding(13, 0)]] RWStructuredBuffer<Atomic<uint>> rowCounts;
[[vk::binding(14, 0)]] RWStructuredBuffer<ConstraintRow> rows;
[[vk::binding(15, 0)]] RWStructuredBuffer<ConstraintColumn> columns;
[[vk::binding(16, 0)]] StructuredBuffer<uint2> treeDependencies;
[[vk::binding(17, 0)]] StructuredBuffer<uint> dependencyActive;
[[vk::binding(18, 0)]] RWStructuredBuffer<Atomic<uint>> awakeCounts; // total DOFs and joints across the batch
[[vk::binding(19, 0)]] RWStructuredBuffer<uint2> awakeDofs;
[[vk::binding(20, 0)]] RWStructuredBuffer<uint2> awakeJoints;
[[vk::binding(21, 0)]] RWStructuredBuffer<uint4> awakeDispatches;
[[vk::binding(22, 0)]] RWStructuredBuffer<float> qaccWarmstart;
[[vk::binding(23, 0)]] StructuredBuffer<uint4> solverDispatchLayout;
[[vk::binding(24, 0)]] RWStructuredBuffer<uint4> solverDispatches;

uint4 treeRange(uint tree)
{
    return sleepModel[sleepModel[1].x + 2 * tree];
}

uint4 treePolicy(uint tree)
{
    return sleepModel[sleepModel[1].x + 2 * tree + 1];
}

uint parentOf<let Wake : bool>(uint base, uint tree)
{
    if (Wake)
        return activity[base + tree].wakeParent.load();
    return activity[base + tree].islandParent.load();
}

uint rootOf<let Wake : bool>(uint base, uint tree)
{
    uint parent = parentOf<Wake>(base, tree);
    while (parent != tree)
    {
        const uint ancestor = parentOf<Wake>(base, parent);
        if (ancestor < parent)
        {
            if (Wake)
                activity[base + tree].wakeParent.min(ancestor);
            else
                activity[base + tree].islandParent.min(ancestor);
        }
        tree = parent;
        parent = ancestor;
    }
    return tree;
}

void linkTrees<let Wake : bool>(uint base, uint left, uint right)
{
    while (true)
    {
        const uint a = rootOf<Wake>(base, left);
        const uint b = rootOf<Wake>(base, right);
        if (a == b)
            return;
        const uint low = min(a, b);
        const uint high = max(a, b);
        uint observed;
        if (Wake)
            observed = activity[base + high].wakeParent.compareExchange(high, low);
        else
            observed = activity[base + high].islandParent.compareExchange(high, low);
        if (observed == high)
            return;
    }
}

bool hasTreeActivity(uint world, uint tree, float tolerance, uint4 counts)
{
    if (treePolicy(tree).z == 0)
        return true;
    const uint4 range = treeRange(tree);
    for (uint i = 0; i < range.y; ++i)
    {
        const uint index = world * counts.y + range.x + i;
        const SpatialMotion wrench = appliedWrench[index];
        if (bodyMoved[index] != 0 || any(wrench.angular.xyz != 0.0) || any(wrench.linear.xyz != 0.0))
            return true;
    }
    const uint dofBase = world * counts.z;
    const uint dofTable = sleepModel[1].z;
    for (uint i = 0; i < range.w; ++i)
    {
        const uint dof = range.z + i;
        const float speed = qvel[dofBase + dof];
        const float scale = asfloat(sleepModel[dofTable + dof].y);
        if (appliedForce[dofBase + dof] != 0.0 ||
            (tolerance > 0.0 ? abs(scale * speed) >= tolerance : speed != 0.0))
            return true;
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void begin_tree_activity(uint3 id : SV_DispatchThreadID)
{
    const uint count = sleepModel[0].x;
    if (id.x >= count)
        return;
    const uint index = id.y * count + id.x;
    activity[index].wakeParent.store(id.x);
    activity[index].islandParent.store(id.x);
    activity[index].wakeAge.store(NoIndex);
    activity[index].quietAge.store(NoIndex);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void link_sleeping_groups(uint3 id : SV_DispatchThreadID)
{
    const uint count = sleepModel[0].x;
    if (id.x >= count)
        return;
    const uint base = id.y * count;
    const uint root = treeSleep[base + id.x].y;
    if (root != NoIndex && root != id.x)
        linkTrees<true>(base, id.x, root);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void link_contact_wakes(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = constraintSettings[0];
    if (id.x >= min(contactCounts[id.y].load(), limits.y))
        return;
    const Contact contact = contacts[id.y * limits.y + id.x];
    if ((contact.geometry.w & ContactConstraint) == 0)
        return;
    const uint4 counts = sleepModel[0];
    const uint bodies = sleepModel[1].y;
    const uint base = id.y * counts.x;
    const uint firstBody = colliderTopology[contact.geometry.x].x;
    const uint secondBody = colliderTopology[contact.geometry.y].x;
    const uint first = sleepModel[bodies + firstBody].x;
    const uint second = sleepModel[bodies + secondBody].x;
    if (first != NoIndex && second != NoIndex)
        linkTrees<true>(base, first, second);
    else if (first != NoIndex && bodyMoved[id.y * counts.y + secondBody] != 0)
        activity[base + first].wakeAge.store(0u);
    else if (second != NoIndex && bodyMoved[id.y * counts.y + firstBody] != 0)
        activity[base + second].wakeAge.store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void link_constraint_islands(uint3 id : SV_DispatchThreadID)
{
    const uint4 limits = constraintSettings[0];
    const uint count = rowCounts[2 * id.y].load();
    if (id.x >= count || count > limits.z || rowCounts[2 * id.y + 1].load() > limits.w)
        return;
    const ConstraintRow row = rows[id.y * limits.z + id.x];
    const uint base = id.y * sleepModel[0].x;
    const uint dofTable = sleepModel[1].z;
    const uint firstColumn = id.y * limits.w + row.layout.x;
    uint first = NoIndex;
    for (uint i = 0; i < row.layout.y; ++i)
    {
        const uint tree = sleepModel[dofTable + columns[firstColumn + i].dof].x;
        if (first == NoIndex)
            first = tree;
        else if (tree != first)
        {
            linkTrees<true>(base, first, tree);
            linkTrees<false>(base, first, tree);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void link_tree_dependencies(uint3 id : SV_DispatchThreadID)
{
    const uint count = sleepModel[1].w;
    if (id.x >= count)
        return;
    const uint enabled = dependencyActive[id.y * count + id.x];
    if (enabled == 0u)
        return;
    const uint2 trees = treeDependencies[id.x];
    const uint base = id.y * sleepModel[0].x;
    linkTrees<true>(base, trees.x, trees.y);
    if ((enabled & 2u) != 0)
        linkTrees<false>(base, trees.x, trees.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collect_tree_wakes(uint3 id : SV_DispatchThreadID)
{
    const uint4 counts = sleepModel[0];
    if (id.x >= counts.x)
        return;
    const uint base = id.y * counts.x;
    const uint index = base + id.x;
    const uint2 sleep = treeSleep[index];
    uint age = activity[index].wakeAge.load();
    if (sleep.y == NoIndex)
        age = min(age, sleep.x);
    else if (hasTreeActivity(id.y, id.x, 0.0, counts))
        age = 0;
    if (age != NoIndex)
        activity[base + rootOf<true>(base, id.x)].wakeAge.min(age);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void clear_awake_work(uint3 id : SV_DispatchThreadID)
{
    awakeCounts[0].store(0u);
    awakeCounts[1].store(0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void publish_awake_work(uint3 id : SV_DispatchThreadID)
{
    const uint4 counts = sleepModel[0];
    if (id.x >= counts.x)
        return;
    const uint base = id.y * counts.x;
    const uint index = base + id.x;
    const uint age = activity[base + rootOf<true>(base, id.x)].wakeAge.load();
    if (age == NoIndex)
        return;
    if (treeSleep[index].y != NoIndex)
        treeSleep[index] = uint2(age, NoIndex);
    const uint4 range = treeRange(id.x);
    const uint2 joint = treePolicy(id.x).xy;
    const uint dofStart = awakeCounts[0].add(range.w);
    const uint jointStart = awakeCounts[1].add(joint.y);
    for (uint i = 0; i < range.w; ++i)
        awakeDofs[dofStart + i] = uint2(range.z + i, id.y);
    for (uint i = 0; i < joint.y; ++i)
        awakeJoints[jointStart + i] = uint2(joint.x + i, id.y);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void emit_awake_dispatches(uint3 id : SV_DispatchThreadID)
{
    awakeDispatches[0] = awakeDispatch(awakeCounts[0].load());
    awakeDispatches[1] = awakeDispatch(awakeCounts[1].load());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void emit_awake_solver_dispatches(uint3 id : SV_DispatchThreadID)
{
    uint count, stride;
    solverDispatchLayout.GetDimensions(count, stride);
    if (id.x >= count)
        return;
    uint4 dispatch = solverDispatchLayout[id.x];
    if (awakeCounts[0].load() == 0)
        dispatch.x = 0;
    solverDispatches[id.x] = dispatch;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void measure_tree_quiet(uint3 id : SV_DispatchThreadID)
{
    const uint4 counts = sleepModel[0];
    if (id.x >= counts.x)
        return;
    const uint base = id.y * counts.x;
    const uint index = base + id.x;
    const uint2 sleep = treeSleep[index];
    if (sleep.y != NoIndex)
        return;
    const uint2 options = sleepSettings[id.y].xy;
    const uint age = hasTreeActivity(id.y, id.x, asfloat(options.y), counts) ? 0 : min(sleep.x + 1, options.x);
    treeSleep[index].x = age;
    activity[base + rootOf<false>(base, id.x)].quietAge.min(age);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_tree_sleep(uint3 id : SV_DispatchThreadID)
{
    const uint4 counts = sleepModel[0];
    if (id.x >= counts.x)
        return;
    const uint base = id.y * counts.x;
    const uint index = base + id.x;
    const bool wasSleeping = treeSleep[index].y != NoIndex;
    if (!wasSleeping)
    {
        const uint root = rootOf<false>(base, id.x);
        if (activity[base + root].quietAge.load() < sleepSettings[id.y].x)
            return;
        treeSleep[index].y = root;
    }
    const uint4 range = treeRange(id.x);
    const uint dofBase = id.y * counts.z + range.z;
    for (uint i = 0; i < range.w; ++i)
    {
        qacc[dofBase + i] = 0.0;
        if (wasSleeping)
            qaccWarmstart[dofBase + i] = 0.0;
        else
            qvel[dofBase + i] = 0.0;
    }
}
