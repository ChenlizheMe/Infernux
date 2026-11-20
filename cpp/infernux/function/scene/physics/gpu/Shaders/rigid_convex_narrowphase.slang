// Discrete convex contacts share the analytic pass's candidate ordinals and
// contact output. Only candidates that require support mapping enter the job queue.
#include "rigid_convex_manifold.slang"
#include "rigid_joints.slang"
#include "component_union.slang"
ConstantBuffer<StepParams> step_params;
ConstantBuffer<StageWork> stage_work;
StructuredBuffer<ShapeModel> shape_models;
StructuredBuffer<BodyState> body_states;
RWStructuredBuffer<ContactPatch> contact_patches;
RWStructuredBuffer<Contact> contacts;
StructuredBuffer<uint4> convex_geometry;
RWStructuredBuffer<uint4> convex_scratch;
StructuredBuffer<BodyModel> body_models;
RWStructuredBuffer<SweepBody> sweep_bodies;
RWStructuredBuffer<Atomic<uint>> sweep_choice;
RWStructuredBuffer<Atomic<uint>> convex_jobs;
RWStructuredBuffer<uint4> convex_dispatch_args;
// One record per world's substep. Remaining seconds and advance fraction
// are FP32 bits, followed by consecutive impacts at the current time.
RWStructuredBuffer<uint4> sweep_clock;

uint cvx_step_slots()
{
    uint words,stride; convex_scratch.GetDimensions(words,stride);
    return words>=4u?min(32u,(words-4u)/CvxSlotWords):0u;
}
void cvx_initialize_jobs(uint index)
{
    uint words,stride; convex_scratch.GetDimensions(words,stride);
    if(index>=words) return;
    uint4 value=uint4(0); uint slots=cvx_step_slots();
    if(index==0u) value=uint4(slots==0u?QueryNumericalFailure:QueryPending,0,0,0);
    if(index==1u) value=uint4(stage_work.range.x,stage_work.range.x<step_params.solve.w?min(stage_work.range.y,step_params.solve.w-stage_work.range.x):0u,slots,step_params.selection.x);
    if(index>=4u && (index-4u)%CvxSlotWords==48u) value=uint4(stage_work.range.x+(index-4u)/CvxSlotWords,0,0,0);
    convex_scratch[index]=value;
}

[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_init_jobs(uint3 tid:SV_DispatchThreadID)
{
    if (tid.x < stage_work.range.x) convex_scratch[tid.x] = uint4(0u);
    if (tid.x < stage_work.range.y) convex_jobs[tid.x].store(0u);
}

[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_queue_jobs(uint3 tid:SV_DispatchThreadID)
{
    const uint ordinal = stage_work.range.x + tid.x;
    const uint2 header = contact_patches[0].leaves.xy;
    if (tid.x >= stage_work.range.y || ordinal >= header.y) return;
    const uint2 leaves = contact_patches[header.x + ordinal].leaves.xy;
    if (all(leaves < step_params.counts.zz) &&
        shape_models[leaves.x].meta.x <= ShapeCapsule && shape_models[leaves.y].meta.x <= ShapeCapsule) return;
    // Execution order does not affect the solver's canonical order: each lane
    // writes the contact records belonging to the original candidate ordinal.
    const uint index = convex_jobs[0u].add(1u);
    convex_jobs[1u + index].store(ordinal);
}

void cvx_dispatch_jobs(uint lanes)
{
    convex_dispatch_args[0u] = uint4((lanes + ConvexJobGroupSize - 1u) / ConvexJobGroupSize, 1u, 1u, 0u);
}

[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_publish_jobs(uint lane:SV_GroupIndex)
{
    const uint count = convex_jobs[0u].load();
    const uint slots = min(cvx_step_slots(), count);
    if (lane < slots) convex_scratch[4u + lane * CvxSlotWords + 48u] = uint4(lane, 0u, 0u, 0u);
    if (lane == 0u)
    {
        convex_scratch[1u] = uint4(0u, count, slots, 0u);
        cvx_dispatch_jobs(slots);
    }
}

uint cvx_job_ordinal(uint b)
{
    return convex_jobs[1u + convex_scratch[b + 48u].x].load();
}

void cvx_step_next(uint b,uint slots)
{
    uint4 owner=convex_scratch[b+48u];
    owner.x += slots;
    owner.yz = uint2(0u);
    owner.w = uint(owner.x >= convex_scratch[1u].y);
    convex_scratch[b+48u] = owner;
}
void cvx_step_start(uint b,uint slots)
{
    uint4 owner=convex_scratch[b+48u],job=convex_scratch[1];
    if(owner.x-job.x>=job.y) { convex_scratch[b+48u].w=1u; return; }
    uint patchIndex=contact_patches[0].leaves.x+cvx_job_ordinal(b);
    ContactPatch patch=contact_patches[patchIndex];
    if(patch.leaves.x>=step_params.counts.z || patch.leaves.y>=step_params.counts.z) { cvx_fail(convex_scratch,b,40); convex_scratch[b+48u].w=2; return; }
    ShapeModel a=shape_models[patch.leaves.x],p=shape_models[patch.leaves.y];
    CvxOperand oa=cvx_leaf_operand(a,body_states[a.meta.y]),op=cvx_leaf_operand(p,body_states[p.meta.y]);
    if(cvx_triangle_backfacing(oa,op.center.xyz) || cvx_triangle_backfacing(op,oa.center.xyz)) {
        cvx_step_next(b,slots); return;
    }
    float scale;
    if(!cvx_pair_scale(convex_geometry,oa,op,scale)) { cvx_fail(convex_scratch,b,7u); convex_scratch[b+48u].w=2u; return; }
    cvx_begin(convex_scratch,b,oa,op,oa.center.xyz,scale,1e-4f,false);
    convex_scratch[b+48u].y=1u;
}
void cvx_step_metric_complete(uint b,uint slots)
{
    uint ordinal=cvx_job_ordinal(b);
    ContactPatch patch=contact_patches[contact_patches[0].leaves.x+ordinal];
    ShapeModel a=shape_models[patch.leaves.x],p=shape_models[patch.leaves.y];
    bool sensor=jph_sensor(a)||jph_sensor(p);
    float band=sensor?0.0f:step_params.contact.x;
    if(cvx_f(convex_scratch,b,7).w>band) { cvx_step_next(b,slots); return; }
    if(sensor) { cvx_manifold_primary(convex_scratch,b); convex_scratch[b+49u]=uint4(0); convex_scratch[b+48u].y=3u; return; }
    BodyState sa=body_states[a.meta.y],sp=body_states[p.meta.y];
    CvxOperand oa=cvx_leaf_operand(a,sa),op=cvx_leaf_operand(p,sp);
    float3 normal=cvx_f(convex_scratch,b,7).xyz;
    if(cvx_is_triangle(oa))
        normal=-cvx_triangle_fix_normal(convex_geometry,oa,cvx_f(convex_scratch,b,8).xyz,-normal,
                                         sp.linearVelocity.xyz-sa.linearVelocity.xyz);
    else if(cvx_is_triangle(op))
        normal=cvx_triangle_fix_normal(convex_geometry,op,cvx_f(convex_scratch,b,9).xyz,normal,
                                        sa.linearVelocity.xyz-sp.linearVelocity.xyz);
    float4 metric=cvx_f(convex_scratch,b,7); metric.xyz=normal; cvx_put(convex_scratch,b,7,metric);
    cvx_manifold_begin(convex_scratch,b,max(band,1e-4f),body_states[a.meta.y].positionCOM.xyz);
    convex_scratch[b+48u].y=2u;
}
void cvx_step_contact(uint b,uint slots)
{
    uint ordinal=cvx_job_ordinal(b),point=convex_scratch[b+49u].x;
    uint patchIndex=contact_patches[0].leaves.x+ordinal;
    ContactPatch patch=contact_patches[patchIndex];
    ShapeModel a=shape_models[patch.leaves.x],p=shape_models[patch.leaves.y];
    BodyState sa=body_states[a.meta.y],sb=body_states[p.meta.y];
    bool sensor=jph_sensor(a)||jph_sensor(p); uint count=convex_scratch[b+36u].w;
    Contact result; result.meta=uint4(a.meta.y,p.meta.y,0,0);
    result.normalDepth=result.localAnchorA=result.localAnchorB=result.impulse=float4(0);
    if(point<count) {
        float3 pa=cvx_f(convex_scratch,b,CvxManifoldOut+3u*point).xyz,pb=cvx_f(convex_scratch,b,CvxManifoldOut+3u*point+1u).xyz;
        float3 normal=cvx_f(convex_scratch,b,7).xyz;
        uint feature=convex_scratch[b+CvxManifoldOut+3u*point+2u].x;
        result.meta.zw=uint2(sensor?0u:1u,(a.meta.x<<28u)|(p.meta.x<<24u)|(feature&0x00ffffffu));
        result.normalDepth=float4(normal,dot(pa-pb,normal));
        result.localAnchorA=float4(jph_inverse_rotate(sa.rotation,pa-sa.positionCOM.xyz),0);
        result.localAnchorB=float4(jph_inverse_rotate(sb.rotation,pb-sb.positionCOM.xyz),0);
        if(!all(isfinite(result.normalDepth)) || !all(isfinite(result.localAnchorA)) || !all(isfinite(result.localAnchorB))) {
            cvx_fail(convex_scratch,b,41); convex_scratch[b+48u].w=2; return;
        }
    }
    contacts[ordinal*MaxContactPointsPerPair+point]=result;
    ++convex_scratch[b+49u].x;
    if(point+1u==MaxContactPointsPerPair) {
        patch.leaves.z=(sensor?PatchSensor:PatchActive)|(cvx_f(convex_scratch,b,7).w<=0?PatchOverlapping:0u);
        patch.leaves.w=count; contact_patches[patchIndex]=patch; cvx_step_next(b,slots);
    }
}
[shader("compute")]
[numthreads(ConvexJobGroupSize,1,1)]
void narrowphase_advance_jobs(uint3 tid:SV_DispatchThreadID)
{
    uint slots=convex_scratch[1u].z; if(tid.x>=slots) return;
    uint b=4u+tid.x*CvxSlotWords;
    // Independent lanes keep their cursors in the shared arena. A packet can
    // grant multiple 32-operation quanta without additional dispatches.
    const uint work_units = stage_work.range.x * ConvexJobAdvanceQuantum;
    for(uint unit=0u;unit<work_units && convex_scratch[b+48u].w==0u;++unit) {
        uint phase=convex_scratch[b+48u].y;
        if(phase==0u) cvx_step_start(b,slots);
        else if(phase==1u) {
            cvx_advance(convex_geometry,convex_scratch,b,1u);
            if(convex_scratch[b].x==CvxSuccess) cvx_step_metric_complete(b,slots);
        } else if(phase==2u) {
            cvx_manifold_advance(convex_geometry,convex_scratch,b,1u);
            if(convex_scratch[b].x==CvxSuccess) { convex_scratch[b+49u]=uint4(0); convex_scratch[b+48u].y=3u; }
        } else cvx_step_contact(b,slots);
        if(phase!=0u && convex_scratch[b].x==CvxFailure) convex_scratch[b+48u].w=2u;
    }
}
void cvx_step_progress()
{
    uint slots=cvx_step_slots(),complete=0; uint4 summary=uint4(QuerySuccess,0,0,0);
    for(uint i=0u;i<slots;++i) {
        uint b=4u+i*CvxSlotWords; uint4 owner=convex_scratch[b+48u];
        if(owner.w==2u) { summary=uint4(QueryNumericalFailure,complete,owner.x,convex_scratch[b].w); break; }
        if(owner.w==1u) ++complete;
        else summary.x=QueryPending;
    }
    if(slots==0u) summary=uint4(QueryNumericalFailure,0,0,42);
    summary.y=complete; convex_scratch[0]=summary;
}

[shader("compute")]
[numthreads(64,1,1)]
void narrowphase_job_progress(uint3 tid:SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    const uint slots = convex_scratch[1u].z;
    uint pending_lanes = 0u, failed_ordinal = 0xffffffffu, reason = 0u;
    for (uint i = 0u; i < slots; ++i)
    {
        const uint b = 4u + i * CvxSlotWords;
        const uint state = convex_scratch[b + 48u].w;
        if (state == 0u) pending_lanes = i + 1u;
        if (state == 2u)
        {
            const uint ordinal = cvx_job_ordinal(b);
            if (ordinal < failed_ordinal)
            {
                failed_ordinal = ordinal;
                reason = convex_scratch[b].w;
            }
        }
    }
    const bool failed = failed_ordinal != 0xffffffffu;
    convex_scratch[0u] = uint4(failed ? QueryNumericalFailure : pending_lanes != 0u ? QueryPending : QuerySuccess,
                              convex_scratch[1u].y, failed_ordinal, reason);
    cvx_dispatch_jobs(failed ? 0u : pending_lanes);
}

void sweep_record_linear_hit(uint base, float fraction, float3 normal, float3 ra, float3 rb, float normalTravel)
{
    if (normalTravel <= 0.0f) return;
    uint4 identity = convex_scratch[base + 49u];
    SweepBody a = sweep_bodies[identity.z - step_params.selection.y];
    SweepBody b = sweep_bodies[identity.w - step_params.selection.y];
    float3 relativeVelocity = a.linearVelocity.xyz + cross(a.angularVelocity.xyz, ra)
                            - b.linearVelocity.xyz - cross(b.angularVelocity.xyz, rb);
    float approachSpeed = dot(normal, relativeVelocity);
    if (approachSpeed <= 0.0f) return;
    // Repeating an impulse below the cast's spatial resolution cannot change
    // the accepted linear motion in this interval.
    if (stage_work.range.z == SweepContinuation && approachSpeed * asfloat(sweep_clock[0].x) <= CvxCastTolerance)
        return;
    sweep_choice[3u * step_params.selectedCounts.x + 1u].min(asuint(fraction));
}

void sweep_sphere_pair(uint base, CvxOperand a, CvxOperand b, float3 bodyB, float3 direction, float distance)
{
    float3 centers = b.center.xyz - a.center.xyz;
    float reach = a.geometry.w + b.geometry.w;
    float gap = jph_capsule_length(centers) - reach;
    float travel = 0.0f;
    if (gap > 0.0f)
    {
        float along = dot(centers, direction);
        float3 perpendicular = centers - along * direction;
        float discriminant = reach * reach - dot(perpendicular, perpendicular);
        if (along <= 0.0f || discriminant <= 0.0f) return;
        // Stable smaller root of the relative ray's sphere intersection.
        travel = gap * (gap + 2.0f * reach) / (along + sqrt(discriminant));
        if (travel > distance) return;
    }
    float3 offset = centers - direction * travel;
    float3 normal = jph_capsule_max_abs(offset) > 0.0f ? jph_capsule_unit(offset) : direction;
    float3 ra = a.center.xyz + a.geometry.w * normal;
    float3 rb = b.center.xyz - bodyB - b.geometry.w * normal;
    sweep_record_linear_hit(base, clamp(travel / distance, 0.0f, 1.0f), normal, ra, rb,
                                   dot(normal, direction) * distance);
}

void cvx_sweep_start(uint base, uint slots)
{
    uint ordinal = convex_scratch[base + 48u].x;
    uint4 job = convex_scratch[1];
    if (ordinal - job.x >= job.y)
    {
        convex_scratch[base + 48u].w = 1u;
        return;
    }
    uint2 leaves = contact_patches[contact_patches[0].leaves.x + ordinal].leaves.xy;
    ShapeModel a = shape_models[leaves.x], b = shape_models[leaves.y];
    SweepBody ma = sweep_bodies[a.meta.y - step_params.selection.y];
    SweepBody mb = sweep_bodies[b.meta.y - step_params.selection.y];
    bool twoDynamic = ma.displacement.w != 0.0f && mb.displacement.w != 0.0f;
    float radiusA = body_models[a.meta.y].invInertiaDiagonal.w;
    float radiusB = body_models[b.meta.y].invInertiaDiagonal.w;
    // Dynamic pairs are driven by the opted-in endpoint. With two such
    // endpoints the smaller radius gives the conservative travel threshold.
    // A newly applied impact can drive a discrete target into another pair
    // during this same interval, without changing its authored CCD mode.
    bool eligibleA = radiusA < 0.0f || ma.positionCOM.w != 0.0f;
    bool eligibleB = radiusB < 0.0f || mb.positionCOM.w != 0.0f;
    bool preferB = radiusB < 0.0f && (radiusA >= 0.0f || radiusB > radiusA);
    bool swap = twoDynamic ? eligibleB && (!eligibleA || preferB)
                           : ma.displacement.w <= 0.0f && ma.positionCOM.w == 0.0f;
    ShapeModel driver = swap ? b : a, target = swap ? a : b;
    SweepBody from = swap ? mb : ma, to = swap ? ma : mb;
    BodyState sa = jph_sweep_state(from), sb = jph_sweep_state(to);
    sb.positionCOM.xyz -= sa.positionCOM.xyz;
    sa.positionCOM.xyz = float3(0.0f);
    CvxOperand oa = cvx_leaf_operand(driver, sa), ob = cvx_leaf_operand(target, sb);
    float3 relative = from.displacement.xyz - to.displacement.xyz;
    float distance = length(relative);
    float radius = abs(swap ? radiusB : radiusA);
    float minimumTravel = stage_work.range.z == SweepContinuation ? CvxCastTolerance
                                                                  : radius * step_params.restitution.y;
    if (distance <= minimumTravel || cvx_triangle_backfacing(ob, oa.center.xyz))
    {
        cvx_step_next(base, slots);
        return;
    }
    convex_scratch[base + 49u] = uint4(swap ? leaves.yx : leaves, driver.meta.y, target.meta.y);
    float3 direction = relative / distance;
    if (oa.meta.x == CvxSphere && ob.meta.x == CvxSphere)
    {
        sweep_sphere_pair(base, oa, ob, sb.positionCOM.xyz, direction, distance);
        cvx_step_next(base, slots);
        return;
    }
    cvx_cast_begin(convex_geometry, convex_scratch, base, oa, ob, direction, distance);
    convex_scratch[base + 48u].y = 1u;
}

void cvx_sweep_hit(uint base)
{
    uint4 identity = convex_scratch[base + 49u];
    SweepBody a = sweep_bodies[identity.z - step_params.selection.y];
    SweepBody b = sweep_bodies[identity.w - step_params.selection.y];
    float4 sweep = cvx_f(convex_scratch, base, CvxCastMotion);
    float4 metric = cvx_f(convex_scratch, base, 7u);
    float fraction = clamp(cvx_cast_distance(convex_scratch, base) / sweep.w, 0.0f, 1.0f);
    float3 origin = cvx_cast_origin(convex_scratch, base);
    float3 pa = cvx_f(convex_scratch, base, 8u).xyz;
    float3 pb = cvx_f(convex_scratch, base, 9u).xyz;
    CvxOperand target = cvx_operand(convex_scratch, base, 1u);
    float3 normal = metric.xyz;
    if (cvx_is_triangle(target))
        normal = cvx_triangle_fix_normal(convex_geometry, target, pb, normal, sweep.xyz);
    float3 ra = pa + origin + (b.displacement.xyz - a.displacement.xyz) * fraction;
    float3 rb = pb + origin - (b.positionCOM.xyz - a.positionCOM.xyz);
    sweep_record_linear_hit(base, fraction, normal, ra, rb, dot(normal, sweep.xyz) * sweep.w);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_advance_jobs(uint3 tid : SV_DispatchThreadID)
{
    uint slots = cvx_step_slots();
    if (tid.x >= slots) return;
    uint base = 4u + tid.x * CvxSlotWords;
    for (uint unit = 0u; unit < 32u && convex_scratch[base + 48u].w == 0u; ++unit)
    {
        if (convex_scratch[base + 48u].y == 0u)
        {
            cvx_sweep_start(base, slots);
            continue;
        }
        uint status = cvx_cast_unit(convex_geometry, convex_scratch, base);
        if (status == CvxCastPending) continue;
        if (status == CvxCastError)
            convex_scratch[base + 48u].w = 2u;
        else
        {
            if (status == CvxCastHit) cvx_sweep_hit(base);
            cvx_step_next(base, slots);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_initialize_jobs(uint3 tid : SV_DispatchThreadID)
{
    cvx_initialize_jobs(tid.x);
    uint count = step_params.selectedCounts.x;
    // Island parents, row heads and cursors share the event workspace with the earliest cast time.
    if (tid.x < count)
    {
        sweep_choice[tid.x].store(tid.x);
        sweep_choice[count + tid.x].store(0xffffffffu);
        sweep_choice[2u * count + tid.x].store(0xffffffffu);
        sweep_choice[3u * count + 2u + tid.x].store(0u);
        sweep_choice[4u * count + 2u + tid.x].store(0u);
        uint words, stride;
        sweep_choice.GetDimensions(words, stride);
        for (uint row = 5u * count + 3u + tid.x; row < words; row += count)
            sweep_choice[row].store(0xffffffffu);
    }
    if (tid.x == 0u)
    {
        sweep_choice[3u * count].store(0u);
        sweep_choice[3u * count + 1u].store(0xffffffffu);
        sweep_choice[5u * count + 2u].store(0u);
        if (stage_work.range.z == SweepInitial)
            sweep_clock[0] = uint4(asuint(step_params.gravityDt.w), 0u, 0u, 0u);
    }
}

uint4 sweep_groups(uint count)
{
    return uint4((min(SweepMotionChunk, count) + 63u) / 64u, 1u, 1u, 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_event_contact_progress(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    cvx_step_progress();
    bool ready = convex_scratch[0].x == QuerySuccess;
    uint bodies = ready ? step_params.selectedCounts.x : 0u;
    convex_dispatch_args[SweepLinkDispatch] = uint4(ready ? 1u : 0u, 1u, 1u, 0u);
    convex_dispatch_args[SweepFlattenDispatch] = sweep_groups(bodies);
    convex_dispatch_args[SweepSolveDispatch] = uint4(bodies / 64u + uint(bodies % 64u != 0u), 1u, 1u, 0u);
}

void sweep_publish_arguments(bool move, bool report, bool integrate)
{
    uint bodies = step_params.selectedCounts.x;
    convex_dispatch_args[SweepMotionDispatch] = sweep_groups(move ? bodies : 0u);
    convex_dispatch_args[SweepReportDispatch] = sweep_groups(report ? step_params.solve.w : 0u);
    convex_dispatch_args[SweepIntegrateDispatch] = sweep_groups(integrate ? bodies : 0u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_job_progress(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    // A deferred or failed candidate producer also publishes zero live rows.
    // It is not an empty successful cast, and must not integrate anything.
    if (contact_patches[0].leaves.z != 0u)
    {
        convex_scratch[0] = uint4(SweepCandidatesIncomplete, 0u, 0u, 0u);
        sweep_publish_arguments(false, false, false);
        return;
    }
    if (step_params.solve.w == 0u)
    {
        convex_scratch[0] = uint4(QuerySuccess, cvx_step_slots(), SweepNoImpact, 0u);
        sweep_publish_arguments(false, false, true);
        return;
    }
    cvx_step_progress();
    uint4 result = convex_scratch[0];
    if (result.x != QuerySuccess)
    {
        sweep_publish_arguments(false, false, false);
        return;
    }

    uint body_count = step_params.selectedCounts.x;
    uint impact = sweep_choice[3u * body_count + 1u].load();
    if (impact == 0xffffffffu)
    {
        convex_scratch[0].z = SweepNoImpact;
        sweep_publish_arguments(false, false, true);
        return;
    }

    uint4 clock = sweep_clock[0];
    float previous = asfloat(clock.x);
    float remaining = previous * (1.0f - asfloat(impact));
    uint repeats = remaining < previous ? 0u : clock.z + 1u;
    // Repeated zero-time impacts must settle within the same physical limit
    // as before. The division avoids a wrapped limit for a large body domain.
    if (repeats > 64u && (repeats - 65u) / 4u >= body_count)
    {
        convex_scratch[0] = uint4(QueryNumericalFailure, result.y, 0u, SweepUnsettled);
        sweep_publish_arguments(false, false, false);
        return;
    }

    sweep_clock[0] = uint4(asuint(remaining), impact, repeats, 0u);
    convex_scratch[0].z = remaining > 0.0f ? SweepMoreMotion : SweepLastImpact;
    convex_scratch[0].w = stage_work.range.w != 0u ? step_params.solve.w : 0u;
    sweep_publish_arguments(true, false, false);
}

// A root may have received an external impulse before its first local row is linked.
static const uint SweepImpactedRoot = 0xfffffffeu;

// Every pass shortens a chain by at least a factor of 32. Host derives the
// pass count from the body range, and emits complete passes in queue order.
[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_flatten_parents(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint local = stage_work.range.x + tid.x;
    uint root = sweep_choice[local].load();
    if (root != local)
    {
        for (uint hop = 0u; hop < 32u; ++hop)
        {
            uint parent = sweep_choice[root].load();
            if (parent == root) break;
            root = parent;
        }
        sweep_choice[local].store(root);
    }
    if (stage_work.range.z != 0u && sweep_bodies[local].angularResponse.w != 0.0f)
        sweep_choice[step_params.selectedCounts.x + root].store(SweepImpactedRoot);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_move_to_event(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint local = stage_work.range.x + tid.x;
    sweep_bodies[local].positionCOM.xyz += asfloat(sweep_clock[0].y) * sweep_bodies[local].displacement.xyz;
}

BodyModel sweep_event_model(uint body)
{
    BodyModel model = body_models[body];
    // The sweep captured the velocity of any temporarily driven static leaf.
    if (!jph_dynamic(model)) model.meta.z = MotionKinematic;
    return model;
}

bool sweep_event_record(uint ordinal, uint2 leaves, JphPairMetric metric)
{
    if (metric.separation > CvxCastTolerance) return false;
    ShapeModel a = shape_models[leaves.x], b = shape_models[leaves.y];
    uint2 bodies = uint2(a.meta.y, b.meta.y);
    uint2 local = bodies - step_params.selection.y;
    BodyState sa = jph_sweep_state(sweep_bodies[local.x]);
    BodyState sb = jph_sweep_state(sweep_bodies[local.y]);
    float3 armA = 0.5f * (metric.pointA + metric.pointB);
    float3 armB = armA - (sb.positionCOM.xyz - sa.positionCOM.xyz);
    float speed = jph_jv(sweep_event_model(bodies.x), sa, armA, sweep_event_model(bodies.y), sb, armB, metric.normal);
    float bounce = speed > step_params.restitution.x
        ? jph_combine_material(a.material.y, a.surface.w, b.material.y, b.surface.w) : 0.0f;
    Contact row;
    row.meta = uint4(bodies, 1u, metric.feature);
    row.normalDepth = float4(metric.normal, -metric.separation);
    row.localAnchorA = float4(armA, 0.0f);
    row.localAnchorB = float4(armB, jph_combine_material(a.material.x, a.surface.z, b.material.x, b.surface.z));
    row.impulse = float4(0.0f, 0.0f, 0.0f, -bounce * speed);
    contacts[ordinal] = row;
    return jph_dynamic(body_models[bodies.x]) && jph_dynamic(body_models[bodies.y]);
}

void sweep_event_finish_row(uint base, uint slots, JphPairMetric metric)
{
    uint ordinal = convex_scratch[base + 48u].x;
    uint2 leaves = contact_patches[contact_patches[0].leaves.x + ordinal].leaves.xy;
    if (metric.status != 1u)
    {
        cvx_fail(convex_scratch, base, 65u);
        convex_scratch[base + 48u].w = 2u;
        return;
    }
    if (!sweep_event_record(ordinal, leaves, metric))
    {
        cvx_step_next(base, slots);
        return;
    }
    convex_scratch[base + 50u].xy = contacts[ordinal].meta.xy - step_params.selection.y;
    convex_scratch[base + 48u].y = 2u;
}

void sweep_event_start_row(uint base, uint slots)
{
    uint ordinal = convex_scratch[base + 48u].x;
    uint4 job = convex_scratch[1];
    if (ordinal - job.x >= job.y)
    {
        convex_scratch[base + 48u].w = 1u;
        return;
    }
    uint patchIndex = contact_patches[0].leaves.x + ordinal;
    ContactPatch patch = contact_patches[patchIndex];
    patch.leaves.zw = uint2(0u);
    contact_patches[patchIndex] = patch;
    contacts[ordinal].meta.z = 0u;
    ShapeModel a = shape_models[patch.leaves.x], b = shape_models[patch.leaves.y];
    BodyState sa = jph_sweep_state(sweep_bodies[a.meta.y - step_params.selection.y]);
    BodyState sb = jph_sweep_state(sweep_bodies[b.meta.y - step_params.selection.y]);
    sb.positionCOM.xyz -= sa.positionCOM.xyz;
    sa.positionCOM.xyz = float3(0.0f);
    CvxOperand oa = cvx_leaf_operand(a, sa), ob = cvx_leaf_operand(b, sb);
    if (a.meta.x <= ShapeCapsule && b.meta.x <= ShapeCapsule)
    {
        sweep_event_finish_row(base, slots,
            jph_pair_metric(a.meta.x, oa.center.xyz, oa.rotation, oa.geometry,
                            b.meta.x, ob.center.xyz, ob.rotation, ob.geometry));
        return;
    }
    if (cvx_triangle_backfacing(oa, ob.center.xyz) || cvx_triangle_backfacing(ob, oa.center.xyz))
    {
        cvx_step_next(base, slots);
        return;
    }
    float scale;
    if (!cvx_pair_scale(convex_geometry, oa, ob, scale))
    {
        cvx_fail(convex_scratch, base, 7u);
        convex_scratch[base + 48u].w = 2u;
        return;
    }
    cvx_begin(convex_scratch, base, oa, ob, oa.center.xyz, scale, CvxCastTolerance, false);
    convex_scratch[base + 48u].y = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_event_contacts(uint3 tid : SV_DispatchThreadID)
{
    uint slots = cvx_step_slots();
    if (tid.x >= slots) return;
    uint base = 4u + tid.x * CvxSlotWords;
    for (uint unit = 0u; unit < 32u && convex_scratch[base + 48u].w == 0u; ++unit)
    {
        uint phase = convex_scratch[base + 48u].y;
        if (phase == 0u)
            sweep_event_start_row(base, slots);
        else if (phase == 2u)
        {
            uint2 edge = jph_join_components(sweep_choice, convex_scratch[base + 50u].xy, 1u);
            if (edge.x == edge.y) cvx_step_next(base, slots);
            else convex_scratch[base + 50u].xy = edge;
        }
        else
        {
            cvx_advance(convex_geometry, convex_scratch, base, 1u);
            uint status = convex_scratch[base].x;
            if (status == CvxFailure) convex_scratch[base + 48u].w = 2u;
            else if (status == CvxSuccess)
            {
                JphPairMetric metric = jph_empty_pair_metric();
                float4 plane = cvx_f(convex_scratch, base, 7u);
                metric.normal = plane.xyz;
                metric.separation = plane.w;
                metric.pointA = cvx_f(convex_scratch, base, 8u).xyz;
                metric.pointB = cvx_f(convex_scratch, base, 9u).xyz;
                metric.status = 1u;
                sweep_event_finish_row(base, slots, metric);
            }
        }
    }
}

// Walk the broadphase's stable ordinal order once. The remaining work is
// independent per dynamic component and never scans unrelated contacts.
[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_link_event_islands(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint count = step_params.selectedCounts.x;
    uint first = stage_work.range.x;
    uint live = step_params.solve.w;
    if (first >= live) return;
    uint length = min(stage_work.range.y, live - first);
    for (uint i = 0u; i < length; ++i)
    {
        uint row = live - 1u - (first + i);
        uint4 meta = contacts[row].meta;
        if (meta.z == 0u) continue;
        uint body = jph_dynamic(body_models[meta.x]) ? meta.x : meta.y;
        uint root = sweep_choice[body - step_params.selection.y].load();
        uint next = sweep_choice[count + root].load();
        if (next == SweepImpactedRoot) next = 0xffffffffu;
        sweep_choice[5u * count + 3u + row].store(next);
        sweep_choice[count + root].store(row);
        sweep_choice[2u * count + root].store(row);
        sweep_choice[5u * count + 2u].add(1u);
        if (next == 0xffffffffu) sweep_choice[3u * count].add(1u);
    }
}

bool sweep_event_joint_enabled(JointDirectory joint)
{
    return (joint.bodies.w & JointEnabled) != 0u && jph_resident(body_models[joint.bodies.y]) &&
           (joint.bodies.x == InvalidQueryBodySlot || jph_resident(body_models[joint.bodies.x]));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_join_event_joints(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    for (uint i = 0u; i < stage_work.range.y; ++i)
    {
        JointDirectory joint = jph_joint_load_directory(step_params.selection.w + stage_work.range.x + i);
        if (!sweep_event_joint_enabled(joint) || joint.bodies.x == InvalidQueryBodySlot ||
            !jph_dynamic(body_models[joint.bodies.x]) || !jph_dynamic(body_models[joint.bodies.y]))
            continue;
        uint2 edge = joint.bodies.xy - step_params.selection.y;
        do { edge = jph_join_components(sweep_choice, edge, 32u); } while (edge.x != edge.y);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_link_event_joints(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    uint count = step_params.selectedCounts.x;
    for (uint i = 0u; i < stage_work.range.y; ++i)
    {
        uint index = step_params.selectedCounts.z - 1u - (stage_work.range.x + i);
        JointDirectory joint = jph_joint_load_directory(step_params.selection.w + index);
        if (!sweep_event_joint_enabled(joint)) continue;
        uint body = joint.bodies.y;
        if (!jph_dynamic(body_models[body]))
        {
            body = joint.bodies.x;
            if (body == InvalidQueryBodySlot || !jph_dynamic(body_models[body])) continue;
        }
        uint root = sweep_choice[body - step_params.selection.y].load();
        uint head = sweep_choice[count + root].load();
        if (head == 0xffffffffu) continue;
        if (head == SweepImpactedRoot)
        {
            head = 0xffffffffu;
            sweep_choice[3u * count].add(1u);
        }
        uint row = step_params.solve.w + index;
        sweep_choice[5u * count + 3u + row].store(head);
        sweep_choice[count + root].store(row);
        sweep_choice[2u * count + root].store(row);
    }
}

BodyState sweep_event_velocities(SweepBody motion)
{
    BodyState state = jph_sweep_state(motion);
    state.linearVelocity.xyz += motion.linearResponse.xyz;
    state.angularVelocity.xyz += motion.angularResponse.xyz;
    return state;
}

void sweep_event_response(uint local, SweepBody motion, BodyState state, bool impact)
{
    motion.linearResponse.xyz = state.linearVelocity.xyz - motion.linearVelocity.xyz;
    motion.angularResponse.xyz = state.angularVelocity.xyz - motion.angularVelocity.xyz;
    motion.angularResponse.w = max(motion.angularResponse.w, float(impact));
    sweep_bodies[local] = motion;
}

float sweep_solve_event_row(uint index, float relaxation)
{
    Contact row = contacts[index];
    uint2 local = row.meta.xy - step_params.selection.y;
    SweepBody a = sweep_bodies[local.x], b = sweep_bodies[local.y];
    BodyState sa = sweep_event_velocities(a), sb = sweep_event_velocities(b);
    BodyModel ma = sweep_event_model(row.meta.x), mb = sweep_event_model(row.meta.y);
    float3 n = row.normalDepth.xyz, ra = row.localAnchorA.xyz, rb = row.localAnchorB.xyz;
    float3 incoming = jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb);
    float effective = jph_effective_mass(ma, sa, ra, mb, sb, rb, n);
    float lambda = max(0.0f, row.impulse.x + relaxation * effective * (dot(incoming, n) - row.impulse.w));
    jph_apply_impulse(ma, sa, ra, mb, sb, rb, n, lambda - row.impulse.x);
    float2 tangents = row.impulse.yz;
    if (row.localAnchorB.w != 0.0f)
        jph_solve_friction(ma, sa, ra, mb, sb, rb, n, row.localAnchorB.w * lambda, tangents);
    row.impulse.xyz = float3(lambda, tangents);
    contacts[index] = row;
    if (jph_dynamic(ma)) sweep_event_response(local.x, a, sa, lambda > 0.0f);
    if (jph_dynamic(mb)) sweep_event_response(local.y, b, sb, lambda > 0.0f);
    return length(jph_point_velocity(ma, sa, ra) - jph_point_velocity(mb, sb, rb) - incoming);
}

float sweep_solve_event_joint(uint index)
{
    JointDirectory joint = jph_joint_load_directory(step_params.selection.w + index);
    uint2 local = joint.bodies.xy - step_params.selection.y;
    BodyModel ma = jph_joint_world_model();
    BodyState sa = jph_joint_world_state();
    SweepBody a = (SweepBody)0;
    if (joint.bodies.x != InvalidQueryBodySlot)
    {
        ma = sweep_event_model(joint.bodies.x);
        a = sweep_bodies[local.x];
        sa = sweep_event_velocities(a);
    }
    BodyModel mb = sweep_event_model(joint.bodies.y);
    SweepBody b = sweep_bodies[local.y];
    BodyState sb = sweep_event_velocities(b);
    float3 oldA = sa.linearVelocity.xyz, oldB = sb.linearVelocity.xyz;
    float3 spinA = sa.angularVelocity.xyz, spinB = sb.angularVelocity.xyz;
    // Keep the ordinary joint's impulse accumulator and per-step motor cap.
#define INX_SWEEP_JOINT(KIND, AUTHOR, STATE) \
    case KIND: { \
        AUTHOR author; STATE state; \
        jph_joint_load_author(joint, author); \
        jph_joint_load_state(joint, state); \
        jph_joint_solve_velocity(author, state, ma, sa, mb, sb, step_params.gravityDt.w); \
        jph_joint_store_state(joint, state); \
        if (state.flags.y != JointStatusSuccess) return -1.0f; \
        break; \
    }
    switch (joint.bodies.z)
    {
        INX_SWEEP_JOINT(JointFixed, JointFixedAuthor, JointFixedState)
        INX_SWEEP_JOINT(JointHinge, JointHingeAuthor, JointHingeState)
        INX_SWEEP_JOINT(JointSlider, JointSliderAuthor, JointSliderState)
        INX_SWEEP_JOINT(JointDistance, JointDistanceAuthor, JointDistanceState)
        INX_SWEEP_JOINT(JointSwingTwist, JointSwingAuthor, JointSwingState)
        INX_SWEEP_JOINT(JointSixDOF, JointSixAuthor, JointSixState)
    }
#undef INX_SWEEP_JOINT
    float changeA = max(length(sa.linearVelocity.xyz - oldA), length(sa.angularVelocity.xyz - spinA));
    float changeB = max(length(sb.linearVelocity.xyz - oldB), length(sb.angularVelocity.xyz - spinB));
    if (jph_dynamic(ma)) sweep_event_response(local.x, a, sa, changeA > 0.0f);
    if (jph_dynamic(mb)) sweep_event_response(local.y, b, sb, changeB > 0.0f);
    return max(changeA, changeB);
}

interface SweepEventRowSolver
{
    float solve(uint row, float relaxation);
}

struct SweepContactRows : SweepEventRowSolver
{
    float solve(uint row, float relaxation) { return sweep_solve_event_row(row, relaxation); }
}

struct SweepContactAndJointRows : SweepEventRowSolver
{
    float solve(uint row, float relaxation)
    {
        return row >= step_params.solve.w ? sweep_solve_event_joint(row - step_params.solve.w)
                                         : sweep_solve_event_row(row, relaxation);
    }
}

void sweep_solve_event_island<Solver : SweepEventRowSolver>(Solver solver, uint root)
{
    uint count = step_params.selectedCounts.x;
    if (root >= count) return;
    uint cursor = sweep_choice[2u * count + root].load();
    if (cursor == 0xffffffffu) return;
    uint head = sweep_choice[count + root].load();
    uint nextBase = 5u * count + 3u;
    uint rounds = sweep_choice[4u * count + 2u + root].load();
    float change = asfloat(sweep_choice[3u * count + 2u + root].load());
    float relaxation = sweep_choice[nextBase + head].load() == 0xffffffffu ? 1.0f : 1.65f;
    float tolerance = CvxCastTolerance / step_params.gravityDt.w;
    // Retain the iterator and residual at a real row budget boundary.
    for (uint unit = 0u; unit < 128u; ++unit)
    {
        float delta = solver.solve(cursor, relaxation);
        if (delta < 0.0f || !isfinite(delta))
        {
            sweep_choice[3u * count + 1u].min(root);
            break;
        }
        change = max(change, delta);
        cursor = sweep_choice[nextBase + cursor].load();
        if (cursor != 0xffffffffu) continue;
        if (change <= tolerance)
        {
            sweep_choice[3u * count].add(0xffffffffu);
            break;
        }
        if (++rounds == 4096u)
        {
            sweep_choice[3u * count + 1u].min(root);
            break;
        }
        change = 0.0f;
        cursor = head;
    }
    sweep_choice[2u * count + root].store(cursor);
    sweep_choice[3u * count + 2u + root].store(asuint(change));
    sweep_choice[4u * count + 2u + root].store(rounds);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_solve_event_islands(uint3 tid : SV_DispatchThreadID)
{
    SweepContactRows solver;
    sweep_solve_event_island(solver, tid.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_solve_event_joint_islands(uint3 tid : SV_DispatchThreadID)
{
    SweepContactAndJointRows solver;
    sweep_solve_event_island(solver, tid.x);
}

void sweep_complete_event()
{
    uint count = step_params.selectedCounts.x;
    uint failed = sweep_choice[3u * count + 1u].load();
    uint pending = sweep_choice[3u * count].load();
    uint rows = sweep_choice[5u * count + 2u].load();
    if (failed != 0xffffffffu)
        convex_scratch[0] = uint4(QueryNumericalFailure, pending, failed, 66u);
    else if (pending != 0u)
        convex_scratch[0] = uint4(QueryPending, pending, rows, SweepSolverReceipt);
    else if (rows == 0u)
        convex_scratch[0] = uint4(QueryNumericalFailure, 0u, 0u, SweepMissingContact);
    else
        convex_scratch[0] = uint4(QuerySuccess, 0u, rows, SweepSolverReceipt);
    bool ready = convex_scratch[0].x == QuerySuccess;
    sweep_publish_arguments(ready, ready, ready && asfloat(sweep_clock[0].x) == 0.0f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_event_progress(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0u) sweep_complete_event();
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_linked_event_progress(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) return;
    if (convex_scratch[0].x == QuerySuccess)
        sweep_complete_event();
    else
        sweep_publish_arguments(false, false, false);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_report_event(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint ordinal = stage_work.range.x + tid.x;
    if (ordinal >= step_params.solve.w) return;
    Contact row = contacts[ordinal];
    if (row.meta.z == 0u || row.impulse.x <= 0.0f) return;
    BodyState sa = jph_sweep_state(sweep_bodies[row.meta.x - step_params.selection.y]);
    BodyState sb = jph_sweep_state(sweep_bodies[row.meta.y - step_params.selection.y]);
    float3 ra = row.localAnchorA.xyz, rb = row.localAnchorB.xyz;
    row.localAnchorA = float4(sa.positionCOM.xyz + ra, 0.0f);
    row.localAnchorB = float4(jph_point_velocity(sweep_event_model(row.meta.y), sb, rb)
                            - jph_point_velocity(sweep_event_model(row.meta.x), sa, ra), 0.0f);
    float3 t1, t2;
    jph_tangents(row.normalDepth.xyz, t1, t2);
    row.impulse = float4(row.normalDepth.xyz * row.impulse.x + t1 * row.impulse.y + t2 * row.impulse.z, 0.0f);
    contacts[ordinal] = row;
    uint patchIndex = contact_patches[0].leaves.x + ordinal;
    contact_patches[patchIndex].leaves.zw = uint2(PatchActive, 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sweep_resume_motion(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= stage_work.range.y) return;
    uint local = stage_work.range.x + tid.x;
    float remaining = asfloat(sweep_clock[0].x);
    SweepBody motion = sweep_bodies[local];
    BodyModel model = body_models[step_params.selection.y + local];
    motion.positionCOM.w = max(motion.positionCOM.w, motion.angularResponse.w);
    motion.linearVelocity.xyz += motion.linearResponse.xyz;
    motion.angularVelocity.xyz += motion.angularResponse.xyz;
    if (jph_dynamic(model))
    {
        motion.linearVelocity.xyz = jph_clamp_speed(motion.linearVelocity.xyz, model.damping.z);
        motion.angularVelocity.xyz = jph_clamp_speed(motion.angularVelocity.xyz, model.damping.w);
    }
    float3 displacement = remaining * motion.linearVelocity.xyz;
    motion.displacement = float4(displacement, jph_sweep_role(model, displacement, 0.0f));
    motion.linearResponse = float4(0.0f, 0.0f, 0.0f, 1.0f);
    motion.angularResponse = float4(0.0f);
    sweep_bodies[local] = motion;
}
