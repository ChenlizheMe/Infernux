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
    return request.meta.x==QueryRay || request.meta.x==QueryOverlap || request.meta.x==QueryClosestPoint ||
           request.meta.x==QueryPenetration || request.meta.x==QueryShapeCast;
}
CvxOperand cvx_query_request(QueryInput request)
{
    CvxOperand o; o.meta=uint4(request.meta.z,0,0,0); o.center=float4(request.positionMaximum.xyz,0);
    o.rotation=request.rotation; o.geometry=request.geometry; o.geometryReference=request.geometryReference;
    if(request.meta.x==QueryClosestPoint) {
        o.meta.x=ShapeSphere; o.rotation=float4(0,0,0,1); o.geometry=float4(0); o.geometryReference=uint4(0);
    }
    return o;
}
CvxOperand cvx_query_target(QueryInput request,uint leaf,float3 origin=float3(0.0f))
{
    CvxOperand o;
    if(request.meta.x==QueryPenetration) {
        QueryPrimitive p=query_load_primitive(request.source.y); o.meta=p.meta; o.center=p.position;
        o.rotation=p.rotation; o.geometry=p.geometry; o.geometryReference=p.geometryReference;
    } else {
        ShapeModel shape=shape_models[leaf];
        BodyState state=body_states[shape.meta.y-query_params.selection.y];
        state.positionCOM.xyz-=origin;
        o=cvx_leaf_operand(shape,state);
    }
    return o;
}
// Match native defaults before starting this leaf's metric/cast. A cast's
// triangle backface is determined by its original direction, not the moving COM.
bool cvx_query_backfacing(QueryInput request,CvxOperand target)
{
    if(!cvx_is_triangle(target) || request.meta.x==QueryClosestPoint) return false;
    if(request.meta.x==QueryRay || request.meta.x==QueryShapeCast)
        return dot(cvx_triangle_normal(target),jph_capsule_unit(request.direction.xyz))>0;
    return (request.meta.x==QueryOverlap || request.meta.x==QueryPenetration)
           && cvx_triangle_backfacing(target,request.positionMaximum.xyz);
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
    uint index=query_control[b+48u].x; QueryInput request=query_load_input(index);
    if(request.meta.x==QueryPenetration) { cvx_query_finish(b,QuerySuccess); return; }
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
    if(request.meta.x==QueryPenetration) {
        query_store_hit(begin,hit); state.cursor.z=1u; query_store_cast_state(index,state); cvx_query_finish(b,QuerySuccess); return;
    }
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
    query_control[b+48u].y=(request.meta.y==QueryUniqueCollider || request.meta.y==QueryUniqueBody)?7u:8u;
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
bool cvx_query_cylinder(float3 origin,float3 direction,float radius,float height,float maximum,out float distance,out float3 normal)
{
    float scale=max(jph_capsule_max_abs(origin),max(radius,height));
    float3 p=origin/scale; float r=radius/scale,h=height/scale;
    float c=dot(p.xz,p.xz)-r*r,enter=0.0f; bool radial=true;
    if(c>0) {
        float a=dot(direction.xz,direction.xz),d=dot(p.xz,direction.xz);
        if(a==0 || d>=0) radial=false;
        else {
            float2 perpendicular=p.xz-direction.xz*(d/a); float disc=a*(r*r-dot(perpendicular,perpendicular));
            if(disc<0) radial=false;
            else enter=query_entering_distance(d,a,sqrt(disc),c,1.0f);
        }
    }
    distance=0; normal=float3(0,-1,0);
    if(!radial) return false;
    if(abs(p.y+enter*direction.y)>h) {
        if(direction.y==0) return false;
        enter=((direction.y<0?h:-h)-p.y)/direction.y;
        if(enter<0 || dot(p.xz+enter*direction.xz,p.xz+enter*direction.xz)>r*r) return false;
    }
    distance=enter*scale; if(distance>maximum || distance<0) return false;
    float3 at=p+enter*direction; float radialLength=length(at.xz);
    if(abs(radialLength-r)<abs(abs(at.y)-h)) normal=float3(at.x/radialLength,0,at.z/radialLength);
    else normal=float3(0,at.y>0?1.0f:-1.0f,0);
    return true;
}
QueryHit cvx_query_ray_hit(QueryInput request,uint leaf,CvxOperand target,float distance,float3 normal)
{
    QueryHit hit=query_miss(); hit.meta=uint4(shape_models[leaf].meta.y,1,leaf,0);
    hit.positionDistance=float4(request.positionMaximum.xyz+jph_capsule_unit(request.direction.xyz)*distance,distance);
    // Native triangle rays return the authored CCW face normal. The two
    // planes of the exact planar hull are not an outward-normal tie breaker.
    float3 worldNormal=cvx_is_triangle(target)?cvx_triangle_normal(target):jph_rotate(target.rotation,normal);
    hit.normalFraction=float4(worldNormal,request.positionMaximum.w>0?distance/request.positionMaximum.w:0);
    hit.pointADepth=float4(hit.positionDistance.xyz,0); return hit;
}
float2 cvx_query_triangle_project(float3 point,uint x,uint y,uint z,float2 shear)
{
    precise float2 projected=float2(point[x],point[y])-shear*point[z];
    return projected;
}
float cvx_query_triangle_edge(float2 a,float2 b)
{
    // Evaluate opposite windings in the same operand order, even on targets
    // that contract multiply-subtract, then restore the directed edge sign.
    bool reversed=b.x<a.x || (b.x==a.x && b.y<a.y);
    float2 lo=reversed?b:a,hi=reversed?a:b;
    precise float edge=lo.x*hi.y-lo.y*hi.x;
    return reversed?-edge:edge;
}
void cvx_query_triangle_ray(uint b,QueryInput request,uint leaf,CvxOperand target)
{
    uint count,first;
    if(!cvx_hull_header(query_geometry,target,count,first,true)) { cvx_query_failure(b,63u); return; }
    uint base=target.geometryReference.x;
    uint vertices=query_geometry[base+2u].z;
    if(vertices==0u) { cvx_query_failure(b,63u); return; }

    ShapeModel shape=shape_models[leaf];
    BodyState body=body_states[shape.meta.y-query_params.selection.y];
    float3 meshOriginCOM=float3(shape.localPositionCOM.w,shape.material.zw);
    float3 originInBody=jph_inverse_rotate(body.rotation,request.positionMaximum.xyz-body.positionCOM.xyz);
    float3 origin=jph_inverse_rotate(shape.localRotation,originInBody-meshOriginCOM);
    float3 directionInBody=jph_inverse_rotate(body.rotation,jph_capsule_unit(request.direction.xyz));
    float3 direction=jph_inverse_rotate(shape.localRotation,directionInBody);
    if(!all(isfinite(origin)) || !all(isfinite(direction))) { cvx_query_failure(b,62u); return; }

    // Project along the dominant ray axis. Adjacent leaves use the same ray
    // frame and project the same bit-identical mesh vertices.
    float3 d=abs(direction);
    uint z=d.x>=d.y && d.x>=d.z?0u:(d.y>=d.z?1u:2u);
    uint x=(z+1u)%3u,y=(z+2u)%3u;
    if(direction[z]==0.0f) { cvx_query_next_leaf(b); return; }
    float2 shear=float2(direction[x],direction[y])/direction[z];
    float3 a=asfloat(query_geometry[base+vertices]).xyz-origin;
    float3 p=asfloat(query_geometry[base+vertices+1u]).xyz-origin;
    float3 c=asfloat(query_geometry[base+vertices+2u]).xyz-origin;
    float2 pa=cvx_query_triangle_project(a,x,y,z,shear);
    float2 pb=cvx_query_triangle_project(p,x,y,z,shear);
    float2 pc=cvx_query_triangle_project(c,x,y,z,shear);
    float3 weights=float3(cvx_query_triangle_edge(pb,pc),cvx_query_triangle_edge(pc,pa),
                          cvx_query_triangle_edge(pa,pb));
    if(!all(isfinite(weights))) { cvx_query_failure(b,62u); return; }
    float area=weights.x+weights.y+weights.z;
    if(!isfinite(area)) { cvx_query_failure(b,62u); return; }
    if(area==0.0f || (any(weights<0.0f) && any(weights>0.0f))) { cvx_query_next_leaf(b); return; }

    float distance=dot(weights/area,float3(a[z],p[z],c[z]))/direction[z];
    if(!isfinite(distance)) { cvx_query_failure(b,62u); return; }
    if(distance<0.0f || distance>request.positionMaximum.w) { cvx_query_next_leaf(b); return; }
    cvx_query_collect_begin(b,cvx_query_ray_hit(request,leaf,target,distance,float3(0.0f)));
}
void cvx_query_ray_begin(uint b,QueryInput request,uint leaf,CvxOperand target)
{
    float3 origin=jph_inverse_rotate(target.rotation,request.positionMaximum.xyz-target.center.xyz);
    float3 direction=jph_inverse_rotate(target.rotation,jph_capsule_unit(request.direction.xyz));
    if(!all(isfinite(origin)) || !all(isfinite(direction))) { cvx_query_failure(b,62u); return; }
    if(target.meta.x==ShapeCylinder) {
        float distance; float3 normal;
        if(cvx_query_cylinder(origin,direction,target.geometry.x,target.geometry.y,request.positionMaximum.w,distance,normal))
            cvx_query_collect_begin(b,cvx_query_ray_hit(request,leaf,target,distance,normal));
        else cvx_query_next_leaf(b);
        return;
    }
    uint count,first;
    if(!cvx_hull_header(query_geometry,target,count,first,true)) { cvx_query_failure(b,63u); return; }
    uint4 header=query_geometry[target.geometryReference.x];
    cvx_put(query_control,b,50,float4(origin,0)); cvx_put(query_control,b,51,float4(direction,0));
    cvx_put(query_control,b,52,float4(0,request.positionMaximum.w,3.402823466e38f,0));
    query_control[b+49u]=uint4(0,0,0,0); query_control[b+48u].y=3u;
    if(header.w==2u) {
        uint planes=query_geometry[target.geometryReference.x+1u].z;
        float4 plane=asfloat(query_geometry[target.geometryReference.x+planes]); float projection=dot(plane.xyz,direction);
        if(abs(projection*request.positionMaximum.w)<1e-12f) { cvx_query_next_leaf(b); return; }
        float distance=-(dot(plane.xyz,origin)+plane.w)/projection;
        if(distance<0 || distance>request.positionMaximum.w) { cvx_query_next_leaf(b); return; }
        cvx_put(query_control,b,52,float4(distance,distance,3.402823466e38f,0)); query_control[b+48u].y=4u;
    }
}
void cvx_query_ray_unit(uint b)
{
    uint index=query_control[b+48u].x; QueryInput request=query_load_input(index);
    QueryCastState state=query_load_cast_state(index); CvxOperand target=cvx_query_target(request,state.cursor.x);
    uint base=target.geometryReference.x; uint4 header=query_geometry[base],layout=query_geometry[base+1u];
    uint4 cursor=query_control[b+49u]; float3 origin=cvx_f(query_control,b,50).xyz,direction=cvx_f(query_control,b,51).xyz;
    float4 range=cvx_f(query_control,b,52); uint phase=query_control[b+48u].y;
    if(phase==3u) {
        if(cursor.x==header.w) { query_control[b+49u].x=0; query_control[b+48u].y=5u; return; }
        float4 plane=asfloat(query_geometry[base+layout.z+cursor.x]); float d=dot(plane.xyz,origin)+plane.w,projection=dot(plane.xyz,direction);
        if(abs(projection*request.positionMaximum.w)>=1e-12f) {
            float distance=-d/projection;
            if(projection<0) range.x=max(range.x,distance); else range.y=min(range.y,distance);
        } else if(d>0) { cvx_query_next_leaf(b); return; }
        if(range.x>range.y || range.y<0) { cvx_query_next_leaf(b); return; }
        cvx_put(query_control,b,52,range); ++query_control[b+49u].x; return;
    }
    if(phase==4u) {
        uint4 face=query_geometry[base+layout.w];
        if(cursor.x==face.y) { query_control[b+49u].x=0; query_control[b+48u].y=5u; return; }
        uint loops=query_geometry[base+2u].x;
        uint previous=face.x+(cursor.x==0u?face.y-1u:cursor.x-1u),current=face.x+cursor.x;
        uint ai=query_geometry[base+loops+previous/4u][previous%4u],bi=query_geometry[base+loops+current/4u][current%4u];
        float3 a=asfloat(query_geometry[base+layout.x+ai]).xyz,p=asfloat(query_geometry[base+layout.x+bi]).xyz;
        float3 planeNormal=asfloat(query_geometry[base+layout.z+face.z]).xyz;
        if(dot(cross(p-a,origin+range.x*direction-a),planeNormal)<0) { cvx_query_next_leaf(b); return; }
        ++query_control[b+49u].x; return;
    }
    // GetSurfaceNormal chooses the nearest native full plane, first on ties.
    if(cursor.x==header.w) {
        cvx_query_collect_begin(b,cvx_query_ray_hit(request,state.cursor.x,target,range.x,cvx_f(query_control,b,53).xyz)); return;
    }
    float4 plane=asfloat(query_geometry[base+layout.z+cursor.x]); float distance=abs(dot(plane.xyz,origin+range.x*direction)+plane.w);
    if(cursor.x==0u || distance<range.z) { range.z=distance; cvx_put(query_control,b,52,range); cvx_put(query_control,b,53,float4(plane.xyz,0)); }
    ++query_control[b+49u].x;
}

void cvx_query_metric_begin(uint b,QueryInput request,CvxOperand target)
{
    CvxOperand a=cvx_query_request(request);
    float scale;
    if(!cvx_pair_scale(query_geometry,a,target,scale)) { cvx_query_failure(b,63u); return; }
    cvx_begin(query_control,b,a,target,a.center.xyz,scale,CvxCastTolerance,request.meta.x==QueryClosestPoint);
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
        if((request.meta.x==QueryOverlap && separation>0) || (request.meta.x==QueryPenetration && separation>=0)) { cvx_query_next_leaf(b); return; }
        hit.positionDistance=float4(request.meta.x==QueryClosestPoint && separation<=0?request.positionMaximum.xyz:p,
            request.meta.x==QueryPenetration?0:max(separation,0.0f));
        // ClosestPoint stays two-sided and keeps its actual closest metric.
        if(triangle && request.meta.x!=QueryClosestPoint)
            normal=cvx_triangle_fix_normal(query_geometry,target,p,normal,float3(0));
        hit.normalFraction=float4(-normal,0);
    }
    hit.meta=uint4(InvalidQueryBodySlot,1,InvalidQueryBodySlot,0);
    if(request.meta.x!=QueryPenetration) hit.meta.xz=uint2(shape_models[state.cursor.x].meta.y,state.cursor.x);
    hit.pointADepth=float4(a,max(-separation,0.0f)); cvx_query_collect_begin(b,hit);
}
void cvx_query_visit(uint b,QueryInput request)
{
    uint index=query_control[b+48u].x; QueryCastState state=query_load_cast_state(index);
    if(request.meta.x==QueryPenetration) {
        CvxOperand target=cvx_query_target(request,0);
        if(cvx_query_backfacing(request,target)) { cvx_query_next_leaf(b); return; }
        if(request.meta.z<=ShapeCapsule && target.meta.x<=ShapeCapsule) {
            JphPairMetric metric=jph_pair_metric(request.meta.z,request.positionMaximum.xyz,request.rotation,request.geometry,
                target.meta.x,target.center.xyz,target.rotation,target.geometry);
            if(query_metric_status(metric)!=QuerySuccess) { cvx_query_failure(b,69u); return; }
            if(metric.separation>=0) { cvx_query_finish(b,QuerySuccess); return; }
            QueryHit hit=query_miss(); hit.meta.yw=uint2(1u,metric.feature);
            hit.positionDistance=float4(metric.pointB,0); hit.normalFraction=float4(-metric.normal,0);
            hit.pointADepth=float4(metric.pointA,-metric.separation); cvx_query_collect_begin(b,hit);
        } else cvx_query_metric_begin(b,request,target);
        return;
    }
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
    CvxOperand target=cvx_query_target(request,leaf,cast?request.positionMaximum.xyz:float3(0.0f));
    if(!all(isfinite(target.center)) || !query_valid_quaternion(target.rotation)) { cvx_query_failure(b,65u); return; }
    if(cvx_query_backfacing(request,target)) { cvx_query_next_leaf(b); return; }
    if(request.meta.x==QueryRay) {
        if(shape.meta.x<=ShapeCapsule) {
            uint status; QueryHit hit=query_leaf(request,shape,body_states[slot-query_params.selection.y],leaf,status);
            if(status!=QuerySuccess) { cvx_query_failure(b,66u); return; } cvx_query_collect_begin(b,hit);
        } else if(cvx_is_triangle(target)) cvx_query_triangle_ray(b,request,leaf,target);
        else cvx_query_ray_begin(b,request,leaf,target);
        return;
    }
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
        } else if(phase>=3u && phase<=5u) cvx_query_ray_unit(b);
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
