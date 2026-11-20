// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Slang adaptation of MuJoCo Warp _src/solver.py preconditioned CG.

#include "contact.slang"
#include "constraint_data.slang"

static const uint CgWidth = 64;
static const float CgMinValue = 1.0e-15;
static const float AlphaNoise = 9.5367431640625e-7;
static const uint Running = 0;
static const uint Converged = 1;
static const uint IterationLimit = 2;
static const uint CapacityExceeded = 3;
static const uint LineSearchLimit = 1;

struct CgWorld
{
    uint4 progress; // accepted/attempted iterations, status, active rows, warning bits
    float4 search; // alpha, improvement, squared direction norm, inverse inertia trace
    float2 gauss; // smooth cost slope, curvature
};

[[vk::binding(0, 0)]] StructuredBuffer<uint4> settings;
[[vk::binding(1, 0)]] StructuredBuffer<uint4> cgOptions; // iteration limit, line limit, packed mass length
[[vk::binding(2, 0)]] RWStructuredBuffer<Atomic<uint>> rowCounts;
[[vk::binding(3, 0)]] StructuredBuffer<ConstraintRow> rows;
[[vk::binding(4, 0)]] StructuredBuffer<ConstraintColumn> columns;
[[vk::binding(5, 0)]] StructuredBuffer<Contact> contacts;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> residuals;
[[vk::binding(7, 0)]] StructuredBuffer<ConstraintEvaluation> evaluation;
[[vk::binding(8, 0)]] RWStructuredBuffer<float> acceleration;
[[vk::binding(9, 0)]] StructuredBuffer<float> smoothForce;
[[vk::binding(10, 0)]] StructuredBuffer<float> massMatrix;
// DOF row offsets followed by pairs of (DOF, packed matrix offset), diagonal first.
[[vk::binding(11, 0)]] StructuredBuffer<uint> massTopology;
[[vk::binding(12, 0)]] RWStructuredBuffer<float> gradient;
[[vk::binding(13, 0)]] StructuredBuffer<float> preconditionedGradient;
[[vk::binding(14, 0)]] RWStructuredBuffer<Atomic<uint>> constraintForce;
[[vk::binding(15, 0)]] RWStructuredBuffer<float4> directionHistory; // previous gradient, preconditioned, direction
[[vk::binding(16, 0)]] RWStructuredBuffer<CgWorld> worlds;
[[vk::binding(17, 0)]] RWStructuredBuffer<float2> massProducts; // M * acceleration, M * direction
[[vk::binding(18, 0)]] RWStructuredBuffer<float> rowRay;

groupshared float4 cgReduction[CgWidth];
groupshared uint uniformStatus;

float4 reduceCg(float4 partial, uint lane)
{
    cgReduction[lane] = partial;
    GroupMemoryBarrierWithGroupSync();
    for (uint width = CgWidth / 2; width > 0; width >>= 1)
    {
        if (lane < width)
            cgReduction[lane] += cgReduction[lane + width];
        GroupMemoryBarrierWithGroupSync();
    }
    const float4 sum = cgReduction[0];
    GroupMemoryBarrierWithGroupSync();
    return sum;
}

uint broadcastCg(uint value, uint lane)
{
    if (lane == 0)
        uniformStatus = value;
    GroupMemoryBarrierWithGroupSync();
    const uint result = workgroupUniformLoad(uniformStatus);
    GroupMemoryBarrierWithGroupSync();
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void begin_contact_cg(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
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

void projectCgForce<let Publish : bool>(uint world, uint lane)
{
    const uint4 progress = worlds[world].progress;
    const uint status = broadcastCg(progress.y, lane);
    if (status == CapacityExceeded || (!Publish && status != Running))
        return;
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
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_project_iteration(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    projectCgForce<false>(group.y, lane);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_publish_force(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    projectCgForce<true>(group.y, lane);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cg_gradient(uint3 id : SV_DispatchThreadID)
{
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

float4 cgRowRay(uint index, uint contactBase, bool elliptic, float inverseFrictionRatio, float alpha)
{
    const ConstraintRow row = rows[index];
    const float start = residuals[index];
    const float direction = rowRay[index];
    const float diagonal = row.weight.x;
    const float delta = alpha * direction;
    const float end = start + delta;
    if (elliptic)
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
                const float d = rows[index + j].weight.x;
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
    const uint world = group.y;
    CgWorld state = worlds[world];
    if (broadcastCg(state.progress.y, lane) != Running)
        return;

    const float4 initial = evaluateCgRay(state, world, lane, 0.0);
    const float noise = AlphaNoise * (sqrt(2.0 * max(0.0, initial.w) * max(0.0, initial.z - state.gauss.y)) +
                                     abs(state.gauss.x));
    const float2 tolerances = asfloat(settings[2 + world].zw);
    const float tolerance = max(noise, tolerances.x * tolerances.y * sqrt(state.search.z) / state.search.w);
    RayPoint best = {0.0, float3(0.0)};
    bool finished = abs(initial.y) <= noise;
    if (broadcastCg(finished ? 1u : 0u, lane) == 0u)
    {
        const RayPoint origin = {0.0, initial.xyz};
        const float alpha = newtonAlpha(origin);
        const RayPoint trial = {alpha, evaluateCgRay(state, world, lane, alpha).xyz};
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
            const RayPoint leftPoint = {left, evaluateCgRay(state, world, lane, left).xyz};
            const RayPoint rightPoint = {right, evaluateCgRay(state, world, lane, right).xyz};
            const RayPoint midPoint = {mid, evaluateCgRay(state, world, lane, mid).xyz};
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
