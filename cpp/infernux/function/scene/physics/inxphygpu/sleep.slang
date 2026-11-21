// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/island.py and _src/sleep.py.

#include "model.slang"
#include "contact.slang"
#include "constraint_data.slang"
#include "linear_dispatch.slang"
#include "prescribed_motion.slang"
#include "scene_motion.slang"

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
[[vk::binding(18, 0)]] RWStructuredBuffer<Atomic<uint>> awakeCounts; // total DOFs, joints and worlds
[[vk::binding(19, 0)]] RWStructuredBuffer<uint2> awakeDofs;
[[vk::binding(20, 0)]] RWStructuredBuffer<uint2> awakeJoints;
[[vk::binding(21, 0)]] RWStructuredBuffer<uint4> awakeDispatches;
[[vk::binding(22, 0)]] RWStructuredBuffer<float> qaccWarmstart;
[[vk::binding(23, 0)]] StructuredBuffer<uint4> solverDispatchLayout;
[[vk::binding(24, 0)]] RWStructuredBuffer<uint4> solverDispatches;
[[vk::binding(25, 0)]] RWStructuredBuffer<Atomic<uint>> awakeWorldFlags;
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> awakeWorlds;
[[vk::binding(27, 0)]] StructuredBuffer<ModelDimensions> sceneDimensions;
[[vk::binding(28, 0)]] StructuredBuffer<BodyModel> sceneBodies;
[[vk::binding(29, 0)]] StructuredBuffer<uint> sceneFixedSteps;
[[vk::binding(30, 0)]] StructuredBuffer<PrescribedMotion> scenePrescribed;
[[vk::binding(31, 0)]] StructuredBuffer<uint4> sceneJointModels;
[[vk::binding(32, 0)]] RWStructuredBuffer<SceneBodyActivity> activityCheckpoint;
[[vk::binding(33, 0)]] StructuredBuffer<uint> sceneStepStatus;
[[vk::binding(34, 0)]] RWStructuredBuffer<SceneBodyActivity> sceneActivity;
[[vk::binding(35, 0)]] StructuredBuffer<float2> sceneSleepSettings;
[[vk::binding(36, 0)]] StructuredBuffer<float> sceneCommandedForce;
[[vk::binding(37, 0)]] StructuredBuffer<float> sceneTimestep;
[[vk::binding(38, 0)]] StructuredBuffer<float> scenePositions;
[[vk::binding(40, 0)]] RWStructuredBuffer<float> sceneSmoothForce;

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
    if (id.x == 0)
        awakeWorldFlags[id.y].store(0u);
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
    awakeCounts[2].store(0u);
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
    const uint root = rootOf<true>(base, id.x);
    const uint age = activity[base + root].wakeAge.load();
    if (age == NoIndex)
        return;
    if (root == id.x && awakeWorldFlags[id.y].load() == 0u)
    {
        if (awakeWorldFlags[id.y].exchange(1u) == 0u)
            awakeWorlds[awakeCounts[2].add(1u)] = id.y;
    }
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
    awakeDispatches[0] = linearDispatch<64>(awakeCounts[0].load());
    awakeDispatches[1] = linearDispatch<64>(awakeCounts[1].load());
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
    dispatch.y = awakeCounts[2].load();
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

// Scene bodies are independent free trees. Their active contact and joint
// edges use the same device union/find as articulated tree wakes.
[shader("compute")]
[numthreads(64, 1, 1)]
void begin_scene_wakes(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = sceneDimensions[0];
    if (id.x >= size.state.z)
        return;
    const uint index = id.y * size.state.z + id.x;
    const uint2 dofs = sceneBodies[id.x].dofs.xy;
    SceneBodyActivity state = sceneActivity[index];
    bool awake = dofs.y != 0 && (state.group == NoIndex || state.jointEdited != 0u);
    if (state.jointEdited != 0u)
    {
        state.marker = 0u;
        state.jointEdited = 0u;
        sceneActivity[index] = state;
    }
    for (uint dof = 0; dof < dofs.y && !awake; ++dof)
        awake = appliedForce[id.y * size.state.y + dofs.x + dof] != 0.0;
    activity[index].wakeParent.store(min(id.x, state.group));
    activity[index].islandParent.store(id.x);
    activity[index].wakeAge.store(awake ? 0u : NoIndex);
    activity[index].quietAge.store(0u); // Counts supports until wake resolution.
}

bool sceneSupportMoves(uint world, uint body, ModelDimensions size)
{
    const uint step = sceneFixedSteps[world] + 1u;
    if (sceneActivity[world * size.state.z + body].marker == step)
        return true;
    const uint row = prescribedRow(sceneBodies[body], size.layout.y);
    if (row == NoIndex)
        return false;
    const PrescribedMotion motion = scenePrescribed[world * prescribedStride(size, sceneBodies[0]) + row];
    return motion.step == step && (any(motion.linear.xyz != 0.0) || any(motion.angular.xyz != 0.0));
}

void linkScenePair<let CountContacts : bool>(uint world, uint2 bodies)
{
    const ModelDimensions size = sceneDimensions[0];
    const uint base = world * size.state.z;
    const bool a = sceneBodies[bodies.x].dofs.y != 0;
    const bool b = sceneBodies[bodies.y].dofs.y != 0;
    if (CountContacts)
    {
        if (a)
            activity[base + bodies.x].quietAge.add(1u);
        if (b)
            activity[base + bodies.y].quietAge.add(1u);
    }
    const bool sleepingA = a && sceneActivity[base + bodies.x].group != NoIndex;
    const bool sleepingB = b && sceneActivity[base + bodies.y].group != NoIndex;
    if (a && b)
    {
        linkTrees<false>(base, bodies.x, bodies.y);
        if (sleepingA || sleepingB)
            linkTrees<true>(base, bodies.x, bodies.y);
    }
    else if (a || b)
    {
        const uint movable = a ? bodies.x : bodies.y;
        const uint support = a ? bodies.y : bodies.x;
        if ((sleepingA || sleepingB) && sceneSupportMoves(world, support, size))
            activity[base + movable].wakeAge.store(0u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void link_scene_contact_wakes(uint3 id : SV_DispatchThreadID)
{
    const uint capacity = constraintSettings[0].y;
    if (id.x >= min(contactCounts[id.y].load(), capacity))
        return;
    const Contact contact = contacts[id.y * capacity + id.x];
    if ((contact.geometry.w & ContactConstraint) != 0)
        linkScenePair<true>(id.y, uint2(colliderTopology[contact.geometry.x].x, colliderTopology[contact.geometry.y].x));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void link_scene_joint_wakes(uint3 id : SV_DispatchThreadID)
{
    if (id.x < sceneJointModels[0].x)
        linkScenePair<false>(id.y, sceneJointModels[1 + SceneJointWords * id.x].yz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void collect_scene_wakes(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= sceneDimensions[0].state.z)
        return;
    const uint base = id.y * sceneDimensions[0].state.z;
    const uint index = base + id.x;
    const SceneBodyActivity state = sceneActivity[index];
    const bool lostSupport = state.group != NoIndex && activity[index].quietAge.load() < state.supports;
    if (lostSupport || activity[index].wakeAge.load() == 0u)
    {
        const uint root = rootOf<true>(base, id.x);
        if (lostSupport || root != id.x)
            activity[base + root].wakeAge.store(0u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void resolve_scene_wakes(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= sceneDimensions[0].state.z)
        return;
    const uint base = id.y * sceneDimensions[0].state.z;
    const uint index = base + id.x;
    SceneBodyActivity state = sceneActivity[index];
    state.supports = activity[index].quietAge.exchange(NoIndex);
    if (state.group != NoIndex && activity[base + rootOf<true>(base, id.x)].wakeAge.load() == 0u)
    {
        state.group = NoIndex;
        state.marker = 0;
    }
    sceneActivity[index] = state;
    if (state.group != NoIndex)
    {
        const uint2 dofs = sceneBodies[id.x].dofs.xy;
        const uint first = id.y * sceneDimensions[0].state.y + dofs.x;
        for (uint dof = 0; dof < dofs.y; ++dof)
        {
            qacc[first + dof] = 0.0;
            qaccWarmstart[first + dof] = 0.0;
            sceneSmoothForce[first + dof] = 0.0;
        }
    }
}

[[vk::binding(39, 0)]] StructuredBuffer<JointModel> sceneFreeJoints;

[shader("compute")]
[numthreads(64, 1, 1)]
void measure_scene_quiet(uint3 id : SV_DispatchThreadID)
{
    const ModelDimensions size = sceneDimensions[0];
    if (id.x >= size.state.z)
        return;
    const BodyModel body = sceneBodies[id.x];
    const uint base = id.y * size.state.z;
    const uint index = base + id.x;
    const SceneBodyActivity state = sceneActivity[index];
    if (body.dofs.y == 0 || state.group != NoIndex)
        return;
    const JointModel joint = sceneFreeJoints[body.hierarchy.y];
    const uint v = id.y * size.state.y + joint.indices.z;
    const uint q = id.y * size.state.x + joint.indices.y + 3u;
    const float4 rotation = float4(scenePositions[q], scenePositions[q + 1], scenePositions[q + 2], scenePositions[q + 3]);
    const float3 spin = rotateVector(float3(qvel[v + 3], qvel[v + 4], qvel[v + 5]), rotation);
    const float3 offset = rotateVector(body.inertialPosition.xyz, rotation);
    const float3 speed = float3(qvel[v], qvel[v + 1], qvel[v + 2]) + cross(spin, offset);
    const float bound = length(speed) + body.subtreeMass.z * length(spin);
    const float2 options = sceneSleepSettings[id.y];
    bool quiet = bound <= options.y;
    for (uint dof = 0; dof < 6 && quiet; ++dof)
        quiet = appliedForce[v + dof] == 0.0 && sceneCommandedForce[v + dof] == 0.0;
    const uint duration = quiet ? asuint(min(options.x, asfloat(state.marker) + sceneTimestep[id.y])) : 0u;
    sceneActivity[index].marker = duration;
    activity[base + rootOf<false>(base, id.x)].quietAge.min(duration);
}

void finishSceneSleep<let HasStatus : bool>(uint3 id)
{
    const ModelDimensions size = sceneDimensions[0];
    if (id.x >= size.state.z || (HasStatus && sceneStepStatus[id.y] != 0))
        return;
    const BodyModel body = sceneBodies[id.x];
    const uint base = id.y * size.state.z;
    const uint index = base + id.x;
    if (body.dofs.y == 0 || sceneActivity[index].group != NoIndex)
        return;
    const uint root = rootOf<false>(base, id.x);
    if (asfloat(activity[base + root].quietAge.load()) < sceneSleepSettings[id.y].x)
        return;
    sceneActivity[index].group = root;
    const uint v = id.y * size.state.y + body.dofs.x;
    for (uint dof = 0; dof < 6; ++dof)
    {
        qvel[v + dof] = 0.0;
        qaccWarmstart[v + dof] = 0.0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_scene_sleep(uint3 id : SV_DispatchThreadID)
{
    finishSceneSleep<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finish_scene_contact_sleep(uint3 id : SV_DispatchThreadID)
{
    finishSceneSleep<true>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void checkpoint_scene_activity(uint3 id : SV_DispatchThreadID)
{
    const uint count = sceneDimensions[0].state.z;
    if (id.x < count)
    {
        const uint row = id.y * count + id.x;
        activityCheckpoint[row] = sceneActivity[row];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restore_scene_activity(uint3 id : SV_DispatchThreadID)
{
    const uint count = sceneDimensions[0].state.z;
    if (id.x < count && sceneStepStatus[id.y] != 0)
    {
        const uint row = id.y * count + id.x;
        sceneActivity[row] = activityCheckpoint[row];
    }
}
