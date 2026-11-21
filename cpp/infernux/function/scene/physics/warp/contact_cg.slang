// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/solver.py preconditioned CG.

#include "contact_solve_common.slang"
#include "convex_status.slang"

[[vk::binding(24, 0)]] StructuredBuffer<uint> awakeWorldFlags;
[[vk::binding(25, 0)]] StructuredBuffer<uint> candidateCount;
[[vk::binding(26, 0)]] StructuredBuffer<uint> contactCounts;
[[vk::binding(27, 0)]] RWStructuredBuffer<uint> contactStepStatus;
[[vk::binding(28, 0)]] RWStructuredBuffer<uint4> contactSolveDispatches;
[[vk::binding(30, 0)]] StructuredBuffer<uint> narrowphaseStatus;

groupshared uint activeRowMaximum[CgWidth];

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
void clear_sleeping_solver_outputs(uint3 id : SV_DispatchThreadID)
{
    if (awakeWorldFlags[id.y] != 0)
        return;
    const uint nv = settings[1].y;
    if (id.x < nv)
        constraintForce[id.y * nv + id.x].store(0u);
    if (id.x == 0)
    {
        const CgWorld result = {uint4(0, Converged, 0, 0), float4(0.0), float2(0.0)};
        worlds[id.y] = result;
    }
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

void addConstraintForce(uint index, float value)
{
    if (value == 0.0)
        return;
    uint previous = constraintForce[index].load();
    for (;;)
    {
        const uint requested = asuint(asfloat(previous) + value);
        const uint observed = constraintForce[index].compareExchange(previous, requested);
        if (observed == previous)
            return;
        previous = observed;
    }
}

void syncCgProjection()
{
    __target_switch
    {
    case wgsl: __intrinsic_asm "storageBarrier()";
    default: DeviceMemoryBarrierWithGroupSync();
    }
}

bool projectCgForce<let Publish : bool>(uint world, uint lane)
{
    const uint4 progress = worlds[world].progress;
    const uint status = broadcastCg(progress.y, lane);
    if (status >= CapacityExceeded || (!Publish && status != Running))
        return false;
    const uint nv = settings[1].y;
    const uint dofBase = world * nv;
    for (uint i = lane; i < nv; i += CgWidth)
        constraintForce[dofBase + i].store(0u);
    syncCgProjection();

    const uint4 limit = settings[0];
    const uint rowBase = world * limit.z;
    const uint columnBase = world * limit.w;
    for (uint i = lane; i < progress.z; i += CgWidth)
    {
        const uint index = rowBase + i;
        const ConstraintRow row = rows[index];
        const uint base = columnBase + row.layout.x;
        const float force = evaluation[index].force;
        for (uint j = 0; j < row.layout.y; ++j)
        {
            const ConstraintColumn column = columns[base + j];
            addConstraintForce(dofBase + column.dof, column.value * force);
        }
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_project_iteration(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    projectCgForce<false>(group.y, lane);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_project_gradient(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    const bool projected = projectCgForce<false>(world, lane);
    // The one workgroup for this world owns both the projection and its gradient.
    syncCgProjection();
    const uint nv = settings[1].y;
    const uint dofBase = world * nv;
    for (uint i = lane; i < nv; i += CgWidth)
    {
        const uint index = dofBase + i;
        gradient[index] = projected ?
            massProducts[index].x - smoothForce[index] - asfloat(constraintForce[index].load()) : 0.0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_publish_force(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    projectCgForce<true>(group.y, lane);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_gradient(uint3 id : SV_DispatchThreadID)
{
    id = solverDispatchId(id);
    const uint nv = settings[1].y;
    if (id.x >= nv)
        return;
    const uint index = id.y * nv + id.x;
    gradient[index] = worlds[id.y].progress.y == Running ?
                      massProducts[index].x - smoothForce[index] - asfloat(constraintForce[index].load()) : 0.0;
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
    rowRay[index] = value;
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
    gauss = reduceCg(float4(gauss, 0, 0), lane).xy;
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
    const float start = residuals[index];
    const float direction = rowRay[index];
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
                const float u = residuals[index + j];
                const float v = rowRay[index + j];
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
            const float direction = rowRay[index];
            sum += (row.diagonal * direction) * float2(residuals[index], direction);
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

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_line_search(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    group = solverDispatchId(group);
    const uint world = group.y;
    CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;

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
    {
        const uint row = id.y * settings[0].z + id.x;
        residuals[row] += alpha * rowRay[row];
    }
}
