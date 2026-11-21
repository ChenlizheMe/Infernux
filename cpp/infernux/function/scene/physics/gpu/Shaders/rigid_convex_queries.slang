// Included after the ordinary analytic query functions. The same immutable
// operands and original collectors remain authoritative; only work is sliced.
#ifndef JPH_GPU_RIGID_CONVEX_QUERIES_SLANG
#define JPH_GPU_RIGID_CONVEX_QUERIES_SLANG

uint cvx_query_header() { return 5u*query_params.counts.x; }
uint cvx_query_slots()
{
    uint words,stride; query_control.GetDimensions(words,stride);
    uint prefix=cvx_query_header();
    return words>=prefix+4u?min(32u,(words-prefix-4u)/CvxSlotWords):0u;
}
bool cvx_query_deferred(QueryInput request)
{
    return request.meta.x==QueryOverlap || request.meta.x==QueryShapeCast;
}
CvxOperand cvx_query_request(QueryInput request)
{
    CvxOperand o; o.meta=uint4(request.meta.z,0,0,0); o.center=float4(request.positionMaximum.xyz,0);
    o.rotation=request.rotation; o.geometry=request.geometry; o.geometryReference=request.geometryReference;
    return o;
}
CvxOperand cvx_query_target(uint leaf,float3 origin)
{
    ShapeModel shape=shape_models[leaf];
    BodyState state=body_states[shape.meta.y-query_params.selection.y];
    state.positionCOM.xyz-=origin;
    return cvx_leaf_operand(shape,state);
}
// Match native defaults before starting this leaf's metric/cast. A cast's
// triangle backface is determined by its original direction, not the moving COM.
bool cvx_query_backfacing(QueryInput request,CvxOperand target)
{
    if(!cvx_is_triangle(target)) return false;
    if(request.meta.x==QueryShapeCast)
        return dot(cvx_triangle_normal(target),jph_capsule_unit(request.direction.xyz))>0;
    return cvx_triangle_backfacing(target,request.positionMaximum.xyz);
}
void cvx_query_finish(uint b,uint status)
{
    uint index=query_control[b+48u].x;
    QueryCastState state=query_load_cast_state(index); state.cursor.y=2u; state.cursor.w=status;
    query_store_cast_state(index,state);
    QuerySpan span; span.range=uint4(0,state.cursor.z,status,0); query_store_span(index,span);
    query_control[b+48u].x+=cvx_query_slots(); query_control[b+48u].yz=uint2(0);
}
void cvx_query_failure(uint b,uint reason)
{
    query_control[b+59u]=uint4(query_control[b+48u].x,reason,0,0);
    cvx_fail(query_control,b,reason); query_control[b+48u].w=2u;
    cvx_query_finish(b,QueryNumericalFailure);
}
void cvx_query_next_leaf(uint b)
{
    uint index=query_control[b+48u].x;
    QueryCastState state=query_load_cast_state(index); ++state.cursor.x; state.cursor.y=0u;
    state.progress.xyz=float3(0,3.402823466e38f,0); state.iteration.yzw=uint3(0);
    query_store_cast_state(index,state); query_control[b+48u].y=1u;
}
QueryHit cvx_query_saved_hit(uint b)
{
    QueryHit hit; hit.meta=query_control[b+60u]; hit.positionDistance=cvx_f(query_control,b,61u);
    hit.normalFraction=cvx_f(query_control,b,62u); hit.pointADepth=cvx_f(query_control,b,63u); return hit;
}
void cvx_query_collect_begin(uint b,QueryHit hit)
{
    if(hit.meta.y==0u) { cvx_query_next_leaf(b); return; }
    uint index=query_control[b+48u].x; QueryInput request=query_load_input(index);
    if(!query_cast_valid_hit(hit)) { cvx_query_failure(b,60u); return; }
    QueryCastState state=query_load_cast_state(index); uint begin=index*query_params.sizes.y;
    if(request.meta.y==QueryClosest) {
        query_collect(request,hit,begin,state.cursor.z);
        if(request.meta.x==QueryShapeCast) state.progress.w=query_load_hit(begin).positionDistance.w;
        query_store_cast_state(index,state); cvx_query_next_leaf(b); return;
    }
    if(request.meta.y==QueryAny) {
        query_store_hit(begin,hit); state.cursor.z=1u; query_store_cast_state(index,state); cvx_query_finish(b,QuerySuccess); return;
    }
    if(state.cursor.z>=query_params.sizes.y) { cvx_query_failure(b,61u); return; }
    query_control[b+60u]=hit.meta; cvx_put(query_control,b,61,hit.positionDistance);
    cvx_put(query_control,b,62,hit.normalFraction); cvx_put(query_control,b,63,hit.pointADepth);
    query_control[b+49u]=uint4(0,state.cursor.z,0,0);
    query_control[b+48u].y=request.meta.y==QueryUniqueCollider?7u:8u;
}
void cvx_query_collect_unit(uint b)
{
    uint index=query_control[b+48u].x; QueryInput request=query_load_input(index);
    QueryCastState state=query_load_cast_state(index); uint begin=index*query_params.sizes.y;
    QueryHit hit=cvx_query_saved_hit(b); uint4 cursor=query_control[b+49u];
    if(query_control[b+48u].y==7u) {
        if(cursor.x==state.cursor.z) { query_control[b+48u].y=8u; return; }
        QueryHit previous=query_load_hit(begin+cursor.x);
        bool duplicate=request.meta.y==QueryUniqueCollider?all(shape_models[previous.meta.z].source.xy==shape_models[hit.meta.z].source.xy):previous.meta.x==hit.meta.x;
        if(duplicate) { cvx_query_next_leaf(b); return; }
        ++query_control[b+49u].x; return;
    }
    uint pos=cursor.y;
    if(pos>0u && query_less(request,hit,query_load_hit(begin+pos-1u))) {
        query_store_hit(begin+pos,query_load_hit(begin+pos-1u)); --query_control[b+49u].y; return;
    }
    query_store_hit(begin+pos,hit); ++state.cursor.z; query_store_cast_state(index,state); cvx_query_next_leaf(b);
}

// CylinderShape::CastRay/RayCylinder: full cylinder (the convex radius is an
// internal support convention, not a rounded ray volume). Scale coordinates.
void cvx_query_metric_begin(uint b,QueryInput request,CvxOperand target)
{
    CvxOperand a=cvx_query_request(request);
    float scale;
    if(!cvx_pair_scale(query_geometry,a,target,scale)) { cvx_query_failure(b,63u); return; }
    cvx_begin(query_control,b,a,target,a.center.xyz,scale,CvxCastTolerance,false);
    query_control[b+48u].y=2u;
}
void cvx_query_metric_complete(uint b,QueryInput request)
{
    uint index=query_control[b+48u].x; QueryCastState state=query_load_cast_state(index);
    float4 metric=cvx_f(query_control,b,7); float3 a=cvx_f(query_control,b,8).xyz,p=cvx_f(query_control,b,9).xyz;
    float separation=metric.w; QueryHit hit=query_miss();
    CvxOperand target=cvx_operand(query_control,b,1u);
    bool triangle=cvx_is_triangle(target);
    float3 normal=metric.xyz;
    if(request.meta.x==QueryShapeCast) {
        float3 direction=jph_capsule_unit(request.direction.xyz);
        // Triangle casts used the original direction for the back-face test.
        if((!triangle || (request.meta.w&QuerySweepPolicy)!=0u) && !query_cast_accept(request,normal,direction)) {
            cvx_query_next_leaf(b); return;
        }
        if(triangle) normal=cvx_triangle_fix_normal(query_geometry,target,p,normal,float3(0));
        float travel=cvx_cast_distance(query_control,b);
        float3 origin=cvx_cast_origin(query_control,b);
        a=request.positionMaximum.xyz+(origin+a);
        p=request.positionMaximum.xyz+(origin+p);
        hit.positionDistance=float4(p,travel);
        hit.normalFraction=float4(-normal,request.positionMaximum.w>0?travel/request.positionMaximum.w:0);
    } else {
        if(separation>0) { cvx_query_next_leaf(b); return; }
        hit.positionDistance=float4(p,max(separation,0.0f));
        if(triangle)
            normal=cvx_triangle_fix_normal(query_geometry,target,p,normal,float3(0));
        hit.normalFraction=float4(-normal,0);
    }
    hit.meta=uint4(InvalidQueryBodySlot,1,InvalidQueryBodySlot,0);
    hit.meta.xz=uint2(shape_models[state.cursor.x].meta.y,state.cursor.x);
    hit.pointADepth=float4(a,max(-separation,0.0f)); cvx_query_collect_begin(b,hit);
}
void cvx_query_visit(uint b,QueryInput request)
{
    uint index=query_control[b+48u].x; QueryCastState state=query_load_cast_state(index);
    uint leaf=state.cursor.x;
    if(leaf-query_params.selection.z>=query_params.selectedCounts.y) { cvx_query_finish(b,QuerySuccess); return; }
    ShapeModel shape=shape_models[leaf]; uint slot=shape.meta.y;
    bool admitted=slot>=query_params.counts.y && slot-query_params.counts.y<query_params.counts.z;
    if(admitted) {
        BodyModel model=body_models[slot]; admitted=jph_resident(model) && leaf>=model.meta.x && leaf-model.meta.x<model.meta.y && query_accepts(request,shape,slot);
    }
    if(admitted && (request.meta.w&QuerySweepPolicy)!=0u)
        admitted=!all(query_body_ids[query_cast_params.controls.y+leaf-query_params.selection.z].ids.xy==request.source.zw);
    if(admitted && (request.meta.w&QuerySweepPairPolicy)!=0u)
        admitted=!query_collider_suppressed(uint4(request.filter.zw,request.lifetime.xy),shape.source);
    if(!admitted) { cvx_query_next_leaf(b); return; }
    bool cast=request.meta.x==QueryShapeCast;
    CvxOperand target=cvx_query_target(leaf,cast?request.positionMaximum.xyz:float3(0.0f));
    if(!all(isfinite(target.center)) || !query_valid_quaternion(target.rotation)) { cvx_query_failure(b,65u); return; }
    if(cvx_query_backfacing(request,target)) { cvx_query_next_leaf(b); return; }
    bool analytic=request.meta.z<=ShapeCapsule && shape.meta.x<=ShapeCapsule;
    if(cast && analytic) {
        // One original CA step per worker unit, not an entire quantum nested
        // inside this quantum. State remains in its independent prefix.
        QueryHit hit; uint status;
        bool done=query_cast_pair_step(request,shape,body_states[slot-query_params.selection.y],state,jph_capsule_unit(request.direction.xyz),hit,status);
        query_store_cast_state(index,state);
        if(status!=QuerySuccess) { cvx_query_failure(b,67u); return; }
        if(done) { hit.meta.z=leaf; cvx_query_collect_begin(b,hit); }
    } else if(analytic) {
        uint status; QueryHit hit=query_leaf(request,shape,body_states[slot-query_params.selection.y],leaf,status);
        if(status!=QuerySuccess) { cvx_query_failure(b,68u); return; } cvx_query_collect_begin(b,hit);
    } else if(cast) {
        CvxOperand moving=cvx_query_request(request);
        moving.center.xyz=float3(0.0f);
        cvx_cast_begin(query_geometry,query_control,b,moving,target,
                       jph_capsule_unit(request.direction.xyz),min(request.positionMaximum.w,state.progress.w));
        query_control[b+48u].y=6u;
    } else cvx_query_metric_begin(b,request,target);
}
[shader("compute")]
[numthreads(64,1,1)]
void query_init_jobs(uint3 tid:SV_DispatchThreadID)
{
    uint words,stride; query_control.GetDimensions(words,stride); uint h=cvx_query_header();
    if(tid.x>=words-h) return;
    uint4 value=uint4(0); uint slots=cvx_query_slots();
    if(tid.x==0u) value=uint4(slots==0u?QueryNumericalFailure:QueryPending,0,0,0);
    if(tid.x==1u) value=uint4(query_params.counts.x,slots,0,0);
    if(tid.x>=4u && (tid.x-4u)%CvxSlotWords==48u) value=uint4((tid.x-4u)/CvxSlotWords,0,0,0);
    query_control[h+tid.x]=value;
}
[shader("compute")]
[numthreads(64,1,1)]
void query_advance_jobs(uint3 tid:SV_DispatchThreadID)
{
    uint slots=cvx_query_slots(); if(tid.x>=slots) return;
    uint b=cvx_query_header()+4u+tid.x*CvxSlotWords;
    for(uint unit=0u;unit<32u && query_control[b+48u].w==0u;++unit) {
        uint index=query_control[b+48u].x,phase=query_control[b+48u].y;
        if(index>=query_params.counts.x) { query_control[b+48u].w=1u; break; }
        if(phase==0u) {
            if(query_load_span(index).range.z!=QueryPending) { query_control[b+48u].x+=slots; continue; }
            query_control[b+48u].y=1u; continue;
        }
        QueryInput request=query_load_input(index);
        if(phase==1u) cvx_query_visit(b,request);
        else if(phase==2u) {
            cvx_advance(query_geometry,query_control,b,1u);
            if(query_control[b].x==CvxFailure) cvx_query_failure(b,query_control[b].w);
            else if(query_control[b].x==CvxSuccess) cvx_query_metric_complete(b,request);
        }
        else if(phase==6u) {
            uint result=cvx_cast_unit(query_geometry,query_control,b);
            if(result==CvxCastError) cvx_query_failure(b,query_control[b].w);
            else if(result==CvxCastMiss) cvx_query_next_leaf(b);
            else if(result==CvxCastHit) cvx_query_metric_complete(b,request);
        }
        else cvx_query_collect_unit(b);
    }
}
[shader("compute")]
[numthreads(64,1,1)]
void query_job_progress(uint3 tid:SV_DispatchThreadID)
{
    if(tid.x!=0u) return;
    uint slots=cvx_query_slots(),status=QuerySuccess;
    for(uint i=0u;i<slots;++i) {
        uint terminal=query_control[cvx_query_header()+4u+i*CvxSlotWords+48u].w;
        if(terminal==2u) { status=QueryNumericalFailure; break; }
        if(terminal!=1u) status=QueryPending;
    }
    if(slots==0u) status=QueryNumericalFailure;
    QuerySummary summary; summary.counts=uint4(0,status,query_params.counts.x,0); query_summary[0]=summary;
}
#endif
