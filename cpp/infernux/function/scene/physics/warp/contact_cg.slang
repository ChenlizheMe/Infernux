// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/solver.py preconditioned CG.

#include "contact_solve_common.slang"
#include "constraint_transpose.slang"
#include "convex_status.slang"

[[vk::binding(25, 0)]] StructuredBuffer<uint> candidateCount;
[[vk::binding(26, 0)]] StructuredBuffer<uint> contactCounts;
[[vk::binding(27, 0)]] RWStructuredBuffer<uint> contactStepStatus;
[[vk::binding(28, 0)]] RWStructuredBuffer<uint4> contactSolveDispatches;
[[vk::binding(30, 0)]] StructuredBuffer<uint> narrowphaseStatus;

groupshared uint activeRowMaximum[CgWidth];
groupshared uint transposePrefix[2][CgWidth];

[shader("compute")]
[numthreads(64, 1, 1)]
void prepare_contact_solve_dispatches(uint lane : SV_GroupIndex)
{
    uint worldCount, stride;
    worlds.GetDimensions(worldCount, stride);
    uint activeRows = 0;
    for (uint world = lane; world < worldCount; world += CgWidth)
    {
        const uint4 progress = worlds[world].progress;
        if (progress.y == Running)
            activeRows = max(activeRows, progress.z);
    }
    activeRowMaximum[lane] = activeRows;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = CgWidth / 2; width > 0; width >>= 1)
    {
        if (lane < width)
            activeRowMaximum[lane] = max(activeRowMaximum[lane], activeRowMaximum[lane + width]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
    {
        const uint rows = (activeRowMaximum[0] + CgWidth - 1) / CgWidth;
        const uint dofs = (settings[1].y + CgWidth - 1) / CgWidth;
        const uint batch = rows > 0 ? worldCount : 0;
        const uint3 massGroups = cgOptions[2].xyz;
        contactSolveDispatches[0] = uint4(1, batch, 1, 0);
        contactSolveDispatches[1] = uint4(dofs, batch, 1, 0);
        contactSolveDispatches[2] = uint4(rows, batch, 1, 0);
        contactSolveDispatches[3] = uint4(max(rows, dofs), batch, 1, 0);
        contactSolveDispatches[4] = uint4(massGroups.x, batch, 1, 0);
        contactSolveDispatches[5] = uint4(massGroups.y, batch, 1, 0);
        contactSolveDispatches[6] = uint4(massGroups.z, batch, 1, 0);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void finish_contact_step(uint3 id : SV_DispatchThreadID)
{
    const uint progress = worlds[id.y].progress.y;
    const uint candidates = candidateCount[0] > cgOptions[1].x ? 1u : 0u;
    const uint contactsOverflow = contactCounts[id.y] > settings[0].y ? 2u : 0u;
    const uint constraintsOverflow = progress == CapacityExceeded ? 4u : 0u;
    const uint invalid = progress == InvalidFactor || progress == Running ? 8u : 0u;
    const uint workspaceOverflow = (narrowphaseStatus[id.y] & ConvexCapacityFlags) != 0 ? 16u : 0u;
    // A bounded solve may integrate its best accepted iterate. Capacity failure cannot.
    const uint earlier = cgOptions[1].y > 1u ? contactStepStatus[id.y] : 0u;
    contactStepStatus[id.y] = earlier | candidates | contactsOverflow | constraintsOverflow | invalid | workspaceOverflow;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void begin_contact_cg(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    const uint nv = settings[1].y;
    const uint4 limit = settings[0];
    const uint rowCount = rowCounts[2 * world].load();
    const uint nonzeros = rowCounts[2 * world + 1].load();
    const uint massBase = world * cgOptions[0].z;
    float trace = 0.0;
    for (uint i = lane; i < nv; i += CgWidth)
        trace += massMatrix[massBase + massTopology[nv + 2 + 2 * massTopology[i]]];
    trace = reduceCg(float4(trace, 0, 0, 0), lane).x;
    if (lane == 0)
    {
        CgWorld value;
        uint status = rowCount > limit.z || nonzeros > limit.w ? CapacityExceeded : Running;
        if (status == Running && (rowCount == 0 || nv == 0))
            status = Converged;
        value.progress = uint4(0, status, rowCount, 0);
        value.search = float4(0, 0, 0, 1.0 / max(CgMinValue, trace));
        value.gauss = float2(0.0);
        worlds[world] = value;
    }
}

float multiplyMass<let Ray : bool>(uint world, uint dof)
{
    const uint nv = settings[1].y;
    const uint dofBase = world * nv;
    const uint massBase = world * cgOptions[0].z;
    float sum = 0.0;
    for (uint i = massTopology[dof]; i < massTopology[dof + 1]; ++i)
    {
        const uint index = nv + 1 + 2 * i;
        const uint column = dofBase + massTopology[index];
        const float value = Ray ? directionHistory[column].z : acceleration[column];
        sum += massMatrix[massBase + massTopology[index + 1]] * value;
    }
    return sum;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_initial_mass_product(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint nv = settings[1].y;
    if (id.x < nv && worlds[id.y].progress.y == Running)
    {
        const uint i = id.y * nv + id.x;
        massProducts[i] = float2(multiplyMass<false>(id.y, id.x), 0.0);
    }
}

void distributeTransposeEntries<let Scatter : bool>(uint3 id)
{
    id = solverDispatchId(id);
    const uint4 progress = worlds[id.y].progress;
    if (progress.y != Running || id.x >= progress.z)
        return;

    const uint4 limits = settings[0];
    const uint cursorBase = transposeCursorBase(id.y, settings[1].y);
    const uint entryBase = id.y * limits.w;
    const ConstraintRow row = rows[id.y * limits.z + id.x];
    const uint columnBase = entryBase + row.layout.x;
    for (uint i = 0; i < row.layout.y; ++i)
    {
        const ConstraintColumn column = columns[columnBase + i];
        if (column.value != 0.0)
        {
            const uint index = transposeAddress[cursorBase + column.dof].add(1u);
            if (Scatter)
                transposeEntries[entryBase + index] = {id.x, column.value};
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void count_cg_transpose(uint3 id : SV_DispatchThreadID)
{
    distributeTransposeEntries<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void prefix_cg_transpose(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    if (broadcastCg(worlds[world].progress.y, lane) != Running)
        return;

    const uint nv = settings[1].y;
    const uint offsetBase = transposeOffsetBase(world, nv);
    const uint cursorBase = transposeCursorBase(world, nv);
    uint total = 0;
    for (uint first = 0; first < nv; first += CgWidth)
    {
        const uint dof = first + lane;
        const uint count = dof < nv ? transposeAddress[cursorBase + dof].load() : 0u;
        transposePrefix[0][lane] = count;
        GroupMemoryBarrierWithGroupSync();
        uint read = 0;
        for (uint stride = 1; stride < CgWidth; stride *= 2)
        {
            const uint earlier = lane < stride ? 0u : transposePrefix[read][lane - stride];
            transposePrefix[1 - read][lane] = transposePrefix[read][lane] + earlier;
            GroupMemoryBarrierWithGroupSync();
            read = 1 - read;
        }
        const uint start = total + transposePrefix[read][lane] - count;
        total += transposePrefix[read][CgWidth - 1];
        GroupMemoryBarrierWithGroupSync();
        if (dof < nv)
        {
            transposeAddress[offsetBase + dof].store(start);
            transposeAddress[cursorBase + dof].store(start);
        }
    }
    if (lane == 0)
        transposeAddress[offsetBase + nv].store(total);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_cg_transpose(uint3 id : SV_DispatchThreadID)
{
    distributeTransposeEntries<true>(id);
}

bool precedesTransposeEntry(TransposedConstraint a, TransposedConstraint b)
{
    return a.row < b.row || (a.row == b.row && asuint(a.value) < asuint(b.value));
}

void sortTransposeEntries(uint start, uint end)
{
    const uint count = end - start;
    uint gap = 1;
    while (gap < count / 3)
        gap = 3 * gap + 1;
    for (; gap != 0; gap /= 3)
    {
        for (uint i = start + gap; i < end; ++i)
        {
            const TransposedConstraint value = transposeEntries[i];
            uint at = i;
            while (at - start >= gap)
            {
                const TransposedConstraint earlier = transposeEntries[at - gap];
                if (!precedesTransposeEntry(value, earlier))
                    break;
                transposeEntries[at] = earlier;
                at -= gap;
            }
            transposeEntries[at] = value;
        }
    }
}

void projectCgGradient<let First : bool>(uint3 id)
{
    id = solverDispatchId(id);
    const uint nv = settings[1].y;
    if (id.x >= nv)
        return;
    const uint index = id.y * nv + id.x;
    float result = 0.0;
    if (worlds[id.y].progress.y == Running)
    {
        const uint4 limits = settings[0];
        const uint entryBase = id.y * limits.w;
        const uint offset = transposeOffsetBase(id.y, nv) + id.x;
        const uint start = entryBase + transposeAddress[offset].load();
        const uint end = entryBase + transposeAddress[offset + 1].load();
        // Canonicalize the integer scatter once per substep. Each lane owns its entire column.
        if (First)
            sortTransposeEntries(start, end);

        const uint rowBase = id.y * limits.z;
        precise float force = 0.0;
        for (uint i = start; i < end; ++i)
        {
            const TransposedConstraint entry = transposeEntries[i];
            force += entry.value * constraintState[rowBase + entry.row].force;
        }
        result = massProducts[index].x - smoothForce[index] - force;
    }
    gradient[index] = result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_project_first_gradient(uint3 id : SV_DispatchThreadID)
{
    projectCgGradient<true>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_project_gradient(uint3 id : SV_DispatchThreadID)
{
    projectCgGradient<false>(id);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_search_direction(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;
    const uint nv = settings[1].y;
    const uint base = world * nv;
    float4 partial = float4(0.0);
    for (uint i = lane; i < nv; i += CgWidth)
    {
        const float grad = gradient[base + i];
        const float scaled = preconditionedGradient[base + i];
        const float2 prev = directionHistory[base + i].xy;
        partial += float4(grad * (scaled - prev.y), prev.x * prev.y, grad * grad, grad * scaled);
    }
    const float4 sums = reduceCg(partial, lane);
    const float scale = state.search.w;
    const float tolerance = asfloat(settings[2 + world].z);
    const bool initial = state.progress.x == 0;
    const bool done = initial ? max(0.0, 0.5 * scale * sums.w) < tolerance :
                      state.search.x == 0.0 || scale * sqrt(sums.z) < tolerance ||
                      (state.search.y > 0.0 && scale * state.search.y < tolerance);
    const bool exhausted = state.progress.x >= cgOptions[0].x;
    if (broadcastCg(done || exhausted ? 1u : 0u, lane) != 0u)
    {
        if (lane == 0)
        {
            state.progress.y = done ? Converged : IterationLimit;
            worlds[world] = state;
        }
        return;
    }

    const float beta = initial ? 0.0 : max(0.0, sums.x / max(CgMinValue, sums.y));
    float directionNorm = 0.0;
    for (uint i = lane; i < nv; i += CgWidth)
    {
        const uint index = base + i;
        const float scaled = preconditionedGradient[index];
        const float direction = initial ? -scaled : -scaled + beta * directionHistory[index].z;
        directionHistory[index] = float4(gradient[index], scaled, direction, 0.0);
        directionNorm += direction * direction;
    }
    directionNorm = reduceCg(float4(directionNorm, 0, 0, 0), lane).x;
    if (lane == 0)
    {
        state.search.z = directionNorm;
        worlds[world] = state;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_ray_constraint_product(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint4 progress = worlds[id.y].progress;
    if (progress.y != Running || id.x >= progress.z)
        return;
    const uint4 limit = settings[0];
    const uint index = id.y * limit.z + id.x;
    const ConstraintRow row = rows[index];
    const uint base = id.y * limit.w + row.layout.x;
    const uint dofBase = id.y * settings[1].y;
    float value = 0.0;
    for (uint i = 0; i < row.layout.y; ++i)
    {
        const ConstraintColumn column = columns[base + i];
        value += column.value * directionHistory[dofBase + column.dof].z;
    }
    constraintState[index].ray = value;
}

float2 computeMassRay(uint world, uint lane)
{
    const uint nv = settings[1].y;
    float2 gauss = float2(0.0);
    for (uint i = lane; i < nv; i += CgWidth)
    {
        const uint index = world * nv + i;
        const float ma = massProducts[index].x;
        const float mv = multiplyMass<true>(world, i);
        massProducts[index] = float2(ma, mv);
        gauss += directionHistory[index].z * float2(ma - smoothForce[index], mv);
    }
    return reduceCg(float4(gauss, 0, 0), lane).xy;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_ray_mass_and_gauss(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;
    const float2 gauss = computeMassRay(world, lane);
    if (lane == 0)
    {
        state.gauss = gauss;
        worlds[world] = state;
    }
}

static const uint ConeSatisfied = 0;
static const uint ConeQuadratic = 1;
static const uint ConeMiddle = 2;

uint coneRegion(float normal, float tangent, float mu)
{
    if (normal >= mu * tangent)
        return ConeSatisfied;
    return mu * normal + tangent <= 0.0 || tangent == 0.0 ? ConeQuadratic : ConeMiddle;
}

float frictionRayCostChange(float start, float delta, float friction, float radius, float diagonal)
{
    const float end = start + delta;
    const bool previousQuadratic = -radius < start && start < radius;
    const bool nextQuadratic = -radius < end && end < radius;
    if (previousQuadratic && nextQuadratic)
        return 0.5 * diagonal * delta * (start + end);
    if (start <= -radius && end <= -radius)
        return -friction * delta;
    if (start >= radius && end >= radius)
        return friction * delta;
    if (previousQuadratic)
    {
        const float side = end >= radius ? 1.0 : -1.0;
        const float gap = radius - side * start;
        return side * friction * delta - 0.5 * diagonal * gap * gap;
    }
    if (nextQuadratic)
    {
        const float magnitude = abs(end);
        const float gap = radius - magnitude;
        if (start * end >= 0.0)
            return (start >= radius ? friction : -friction) * delta + 0.5 * diagonal * gap * gap;
        return -(friction * (abs(start) - radius) + 0.5 * diagonal * gap * (radius + magnitude));
    }
    return (end >= radius ? friction : -friction) * (end + start);
}

float4 cgRowRay(uint index, uint contactBase, bool elliptic, float inverseFrictionRatio, float alpha)
{
    const ConstraintRow row = rows[index];
    const uint kind = row.kind;
    if (kind == EqualityRow)
        return float4(0.0);
    const float start = constraintState[index].residual;
    const float direction = constraintState[index].ray;
    const float diagonal = row.diagonal;
    const float delta = alpha * direction;
    const float end = start + delta;
    if (isFrictionConstraint(kind))
    {
        const float friction = row.frictionLoss;
        const float radius = friction / diagonal;
        const ConstraintEvaluation before = evaluateFriction(start, diagonal, friction);
        const ConstraintEvaluation after = evaluateFriction(end, diagonal, friction);
        return float4(frictionRayCostChange(start, delta, friction, radius, diagonal), -after.force * direction,
                      after.state == QuadraticConstraint ? diagonal * direction * direction : 0.0, before.cost);
    }
    if (elliptic && kind == ContactRow)
    {
        const Contact contact = contacts[contactBase + row.layout.z];
        if (contact.geometry.z > 1)
        {
            if (row.layout.w != 0)
                return float4(0.0);
            const float mu = contact.friction.x * inverseFrictionRatio;
            float3 quadratic = diagonal * float3(0.5 * start * start, start * direction, direction * direction);
            float3 tangent = float3(0.0);
            float2 tangentStart = float2(0.0);
            for (uint j = 1; j < contact.geometry.z; ++j)
            {
                const float u = constraintState[index + j].residual;
                const float v = constraintState[index + j].ray;
                const float t = u + alpha * v;
                const float d = rows[index + j].diagonal;
                const float muj = contactFriction(contact, j - 1);
                quadratic += d * float3(0.5 * u * u, u * v, v * v);
                const float tv = muj * t;
                const float vv = muj * v;
                const float uu = muj * u;
                tangent += float3(tv * tv, tv * vv, vv * vv);
                tangentStart += float2(uu * uu, uu * vv);
            }
            const float dm = diagonal / max(CgMinValue, mu * mu * (1.0 + mu * mu));
            const float n0 = mu * start;
            const float n = mu * end;
            const float t0 = sqrt(tangentStart.x);
            const float t = sqrt(tangent.x);
            const uint previous = coneRegion(n0, t0, mu);
            const uint next = coneRegion(n, t, mu);
            const float previousGap = n0 - mu * t0;
            const float oldCost = previous == ConeQuadratic ? quadratic.x :
                                  previous == ConeMiddle ? 0.5 * dm * previousGap * previousGap : 0.0;
            const float quadraticDelta = alpha * (quadratic.y + 0.5 * alpha * quadratic.z);
            if (next == ConeSatisfied)
                return float4(-oldCost, 0.0, 0.0, oldCost);
            if (next == ConeQuadratic)
            {
                float change = quadraticDelta;
                if (previous == ConeMiddle)
                {
                    const float boundary = mu * n0 + t0;
                    change += 0.5 * dm * boundary * boundary;
                }
                else if (previous == ConeSatisfied)
                    change = 0.5 * dm * (1.0 + mu * mu) * (n * n + tangent.x);
                return float4(change, quadratic.y + alpha * quadratic.z, quadratic.z, oldCost);
            }

            const float tSlope = tangent.y / t;
            const float tCurvature = max(0.0, tangent.z - tSlope * tSlope) / t;
            const float gap = n - mu * t;
            const float gapSlope = mu * (direction - tSlope);
            float change = 0.5 * dm * gap * gap;
            if (previous == ConeMiddle)
            {
                const float tDelta = alpha * (2.0 * tangentStart.y + alpha * tangent.z) / (t + t0);
                const float gapDelta = mu * (delta - tDelta);
                change = 0.5 * dm * gapDelta * (2.0 * previousGap + gapDelta);
            }
            else if (previous == ConeQuadratic)
            {
                const float boundary = mu * n + t;
                change = quadraticDelta - 0.5 * dm * boundary * boundary;
            }
            return float4(change, dm * gap * gapSlope,
                          dm * (gapSlope * gapSlope - gap * mu * tCurvature), oldCost);
        }
    }

    const float before = min(start, 0.0);
    const float after = min(end, 0.0);
    const float displacement = start < 0.0 && end < 0.0 ? delta : after - before;
    return diagonal * float4(0.5 * displacement * (after + before), after * direction,
                             end < 0.0 ? direction * direction : 0.0, 0.5 * before * before);
}

float2 equalityRayTerms(uint world, uint rowCount, uint lane)
{
    const uint base = world * settings[0].z;
    float2 sum = float2(0.0);
    for (uint i = lane; i < rowCount; i += CgWidth)
    {
        const uint index = base + i;
        const ConstraintRow row = rows[index];
        if (row.kind == EqualityRow)
        {
            const float direction = constraintState[index].ray;
            sum += (row.diagonal * direction) * float2(constraintState[index].residual, direction);
        }
    }
    return reduceCg(float4(sum, 0.0, 0.0), lane).xy;
}

float4 evaluateCgRay(CgWorld state, uint world, uint lane, float alpha)
{
    const uint4 limit = settings[0];
    const bool elliptic = (settings[1].w & EllipticCone) != 0;
    const float ratio = asfloat(settings[2 + world].y);
    float4 sum = float4(0.0);
    for (uint row = lane; row < state.progress.z; row += CgWidth)
        sum += cgRowRay(world * limit.z + row, world * limit.y, elliptic, ratio, alpha);
    sum = reduceCg(sum, lane);
    sum.xyz += float3(alpha * (state.gauss.x + 0.5 * alpha * state.gauss.y),
                      state.gauss.x + alpha * state.gauss.y, state.gauss.y);
    return sum;
}

struct RayPoint
{
    float alpha;
    float3 value; // change in cost, derivative, curvature
};

bool tightenRay(inout RayPoint bound, RayPoint candidate)
{
    const float here = bound.value.y;
    const float next = candidate.value.y;
    const bool closer = (here < next && next < 0.0) || (here > next && next > 0.0);
    if (closer)
        bound = candidate;
    return closer;
}

bool acceptRay(inout RayPoint best, RayPoint candidate, float tolerance)
{
    const bool acceptable = abs(candidate.value.y) < tolerance && candidate.value.x < 0.0;
    if (acceptable && candidate.value.x < best.value.x)
        best = candidate;
    return acceptable;
}

float newtonAlpha(RayPoint point)
{
    return point.alpha - point.value.y / max(CgMinValue, point.value.z);
}

void searchContactRay(uint world, uint lane, CgWorld state)
{
    // Equality is quadratic at every alpha, so its ray coefficients can be evaluated once.
    CgWorld ray = state;
    ray.gauss += equalityRayTerms(world, state.progress.z, lane);
    const float4 initial = evaluateCgRay(ray, world, lane, 0.0);
    const float noise = AlphaNoise * (sqrt(2.0 * max(0.0, initial.w) * max(0.0, initial.z - ray.gauss.y)) +
                                     abs(ray.gauss.x));
    const float2 tolerances = asfloat(settings[2 + world].zw);
    const float tolerance = max(noise, tolerances.x * tolerances.y * sqrt(state.search.z) / state.search.w);
    RayPoint best = {0.0, float3(0.0)};
    bool finished = abs(initial.y) <= noise;
    if (broadcastCg(finished ? 1u : 0u, lane) == 0u)
    {
        const RayPoint origin = {0.0, initial.xyz};
        const float alpha = newtonAlpha(origin);
        const RayPoint trial = {alpha, evaluateCgRay(ray, world, lane, alpha).xyz};
        finished = acceptRay(best, trial, tolerance);
        if (trial.value.x < best.value.x)
            best = trial;
        RayPoint lo = trial.value.y < origin.value.y ? trial : origin;
        RayPoint hi = trial.value.y < origin.value.y ? origin : trial;
        const uint iterationLimit = broadcastCg(cgOptions[0].y, lane);
        for (uint i = 0; i < iterationLimit; ++i)
        {
            if (broadcastCg(finished ? 1u : 0u, lane) != 0u)
                break;
            const float left = newtonAlpha(lo);
            const float right = newtonAlpha(hi);
            const float mid = 0.5 * (lo.alpha + hi.alpha);
            const RayPoint leftPoint = {left, evaluateCgRay(ray, world, lane, left).xyz};
            const RayPoint rightPoint = {right, evaluateCgRay(ray, world, lane, right).xyz};
            const RayPoint midPoint = {mid, evaluateCgRay(ray, world, lane, mid).xyz};
            RayPoint accepted = {0.0, float3(0.0)};
            const bool acceptLeft = acceptRay(accepted, leftPoint, tolerance);
            const bool acceptRight = acceptRay(accepted, rightPoint, tolerance);
            const bool acceptMid = acceptRay(accepted, midPoint, tolerance);
            if (acceptLeft || acceptRight || acceptMid)
            {
                best = accepted;
                finished = true;
            }
            else
            {
                const bool loLeft = tightenRay(lo, leftPoint);
                const bool loMid = tightenRay(lo, midPoint);
                const bool loRight = tightenRay(lo, rightPoint);
                const bool overshoot = hi.value.y < 0.0 && rightPoint.value.y > 0.0;
                const bool hiRight = tightenRay(hi, rightPoint);
                if (overshoot)
                    hi = rightPoint;
                const bool hiMid = tightenRay(hi, midPoint);
                const bool hiLeft = tightenRay(hi, leftPoint);
                finished = !(loLeft || loMid || loRight || overshoot || hiRight || hiMid || hiLeft);
                if (lo.value.x < best.value.x)
                    best = lo;
                if (hi.value.x < best.value.x)
                    best = hi;
            }
        }
    }
    if (lane == 0)
    {
        state.progress.x += 1;
        if (!finished)
            state.progress.w |= LineSearchLimit;
        state.search.xy = float2(best.alpha, -best.value.x);
        worlds[world] = state;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_line_search(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const CgWorld state = worlds[group.y];
    if (broadcastCg(state.progress.y, lane) == Running)
        searchContactRay(group.y, lane, state);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_search_step(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const uint world = solverDispatchId(group).y;
    CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;
    state.gauss = computeMassRay(world, lane);
    searchContactRay(world, lane, state);
}

static const ConstraintEvaluation SatisfiedRow = {0.0, 0.0, SatisfiedConstraint};

ConstraintEvaluation evaluateScalarRow(ConstraintRow row, float residual)
{
    if (row.kind == EqualityRow)
        return evaluateQuadratic(residual, row.diagonal);
    if (isFrictionConstraint(row.kind))
        return evaluateFriction(residual, row.diagonal, row.frictionLoss);
    return residual >= 0.0 ? SatisfiedRow : evaluateQuadratic(residual, row.diagonal);
}

float rowResidual<let Initial : bool>(uint world, uint index, ConstraintRow row, float alpha)
{
    if (!Initial)
    {
        const ConstraintSolveState old = constraintState[index];
        return old.residual + alpha * old.ray;
    }
    const uint base = world * settings[0].w + row.layout.x;
    const uint dofBase = world * settings[1].y;
    float residual = -row.state.w;
    for (uint col = 0; col < row.layout.y; ++col)
    {
        const ConstraintColumn entry = columns[base + col];
        residual += entry.value * acceleration[dofBase + entry.dof];
    }
    return residual;
}

void writeEvaluatedRow<let Initial : bool>(uint index, float residual, ConstraintEvaluation value)
{
    const float ray = Initial ? 0.0 : constraintState[index].ray;
    constraintState[index] = {residual, ray, value.force, value.state};
}

void updateConstraintRows<let Initial : bool>(uint world, uint index, float alpha)
{
    const ConstraintRow row = rows[index];
    if (row.kind != ContactRow || (settings[1].w & EllipticCone) == 0)
    {
        const float residual = rowResidual<Initial>(world, index, row, alpha);
        writeEvaluatedRow<Initial>(index, residual, evaluateScalarRow(row, residual));
        return;
    }
    // One invocation owns all components of an elliptic cone. This avoids
    // reading a neighbour's residual while it is advancing the same search ray.
    if (row.layout.w != 0u)
        return;
    const Contact contact = contacts[world * settings[0].y + row.layout.z];
    const uint count = contact.geometry.z;
    const float first = rowResidual<Initial>(world, index, row, alpha);
    if (count == 1u)
    {
        writeEvaluatedRow<Initial>(index, first, evaluateScalarRow(row, first));
        return;
    }

    float residuals[6];
    residuals[0] = first;
    float tangentSquared = 0.0;
    for (uint ordinal = 1; ordinal < count; ++ordinal)
    {
        const uint component = index + ordinal;
        const float residual = rowResidual<Initial>(world, component, rows[component], alpha);
        residuals[ordinal] = residual;
        const float value = residual * contactFriction(contact, ordinal - 1u);
        tangentSquared += value * value;
    }
    const float mu = contact.friction.x * asfloat(settings[2 + world].y);
    const float normal = first * mu;
    const float tangent = sqrt(tangentSquared);
    if (normal >= mu * tangent)
    {
        for (uint ordinal = 0; ordinal < count; ++ordinal)
            writeEvaluatedRow<Initial>(index + ordinal, residuals[ordinal], SatisfiedRow);
    }
    else if (mu * normal + tangent <= 0.0 || tangent == 0.0)
    {
        for (uint ordinal = 0; ordinal < count; ++ordinal)
            writeEvaluatedRow<Initial>(index + ordinal, residuals[ordinal],
                                       evaluateQuadratic(residuals[ordinal], rows[index + ordinal].diagonal));
    }
    else
    {
        const float dm = row.diagonal / max(CgMinValue, mu * mu * (1.0 + mu * mu));
        const float gap = normal - mu * tangent;
        const float force = -dm * gap * mu;
        writeEvaluatedRow<Initial>(index, first, {force, 0.5 * dm * gap * gap, EllipticMiddleState});
        for (uint ordinal = 1; ordinal < count; ++ordinal)
        {
            const float friction = contactFriction(contact, ordinal - 1u);
            const float componentForce =
                -force * residuals[ordinal] * friction * friction / max(CgMinValue, tangent);
            writeEvaluatedRow<Initial>(index + ordinal, residuals[ordinal],
                                       {componentForce, 0.0, EllipticMiddleState});
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_initialize_constraint_residuals(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint4 progress = worlds[id.y].progress;
    if (progress.y == Running && id.x < progress.z)
        updateConstraintRows<true>(id.y, id.y * settings[0].z + id.x, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_apply_step(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const CgWorld state = worlds[id.y];
    if (state.progress.y != Running)
        return;
    const float alpha = state.search.x;
    const uint nv = settings[1].y;
    if (id.x < nv)
    {
        const uint dof = id.y * nv + id.x;
        const float2 m = massProducts[dof];
        acceleration[dof] += alpha * directionHistory[dof].z;
        massProducts[dof] = float2(m.x + alpha * m.y, m.y);
    }
    if (id.x < state.progress.z)
        updateConstraintRows<false>(id.y, id.y * settings[0].z + id.x, alpha);
}
