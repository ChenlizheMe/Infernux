// Private GPU convex metric core. Native Jolt support/core conventions;
// GJK closest-simplex and EPA expanding-polytope laws, with persistent work.
#ifndef JPH_GPU_RIGID_CONVEX_SLANG
#define JPH_GPU_RIGID_CONVEX_SLANG
#include "rigid_math.slang"
#include "../ConvexGeometryTypes.h"

// Word offsets match GENERIC-WORKSPACE. No invocation-local EPA arrays.
static const uint CvxSlotWords=ConvexJobSlotWords, CvxSimplex=ConvexJobSimplexFirst, CvxPoints=ConvexJobPointsFirst;
static const uint CvxFaces=ConvexJobFacesFirst, CvxHorizon=ConvexJobHorizonFirst, CvxTraversal=ConvexJobTraversalFirst;
static const uint CvxNewFaces=ConvexJobNewFacesFirst, CvxMasks=ConvexJobMasksFirst;
static const uint CvxPending=0u, CvxSuccess=1u, CvxFailure=2u;
static const uint CvxGjk=1u, CvxGjkPoint=2u, CvxClosest=3u, CvxReduce=4u;
static const uint CvxSupport=10u, CvxSeed=20u, CvxSeedPoint=21u;
static const uint CvxFaceInit=22u, CvxFaceMin=23u, CvxEpaPoint=24u;
static const uint CvxVisible=25u, CvxEdge=26u, CvxCreate=27u;
static const uint CvxSphere=ShapeSphere, CvxBox=ShapeBox, CvxCapsule=ShapeCapsule, CvxCylinder=ShapeCylinder, CvxHull=ShapeConvexHull;

struct CvxOperand
{
    uint4 meta;                 // type, ShapeModel flags,0,0
    float4 center;              // world xyz; w0
    float4 rotation;            // world xyzw
    float4 geometry;            // Existing primitive law; Cylinder r,h,margin,0; Hull margin,0,0,0
    uint4 geometryReference;    // absolute word offset,length,0,0
};
struct CvxWitness { float3 a; float3 b; float3 w; };

CvxOperand cvx_leaf_operand(ShapeModel shape, BodyState state)
{
    CvxOperand operand;
    operand.meta = uint4(shape.meta.x, shape.meta.w, 0u, 0u);
    operand.center = float4(state.positionCOM.xyz + jph_rotate(state.rotation, shape.localPositionCOM.xyz), 0);
    operand.rotation = jph_quat_mul(state.rotation, shape.localRotation);
    operand.geometry = shape.geometry;
    operand.geometryReference = shape.geometryReference;
    return operand;
}

Aabb cvx_leaf_bounds(StructuredBuffer<uint4> geometry, CvxOperand operand, float padding)
{
    float4 dimensions = operand.geometry;
    float3 center = operand.center.xyz;
    float4 rotation = operand.rotation;
    float3 extent;
    if (operand.meta.x == CvxSphere)
        extent = float3(dimensions.w);
    else if (operand.meta.x == CvxCapsule)
        extent = abs(jph_rotate(rotation, float3(0.0f, 1.0f, 0.0f))) * dimensions.y + dimensions.w;
    else if (operand.meta.x == CvxCylinder)
    {
        float3 axis = jph_rotate(rotation, float3(0, 1, 0));
        extent = abs(axis) * dimensions.y + dimensions.x * sqrt(max(float3(0), float3(1) - axis * axis));
    }
    else
    {
        float3 half = dimensions.xyz;
        if (operand.meta.x == CvxHull)
        {
            uint base = operand.geometryReference.x;
            float3 low = asfloat(geometry[base + 3u]).xyz;
            float3 high = asfloat(geometry[base + 4u]).xyz;
            center += jph_rotate(rotation, low * .5f + high * .5f);
            half = high * .5f - low * .5f;
        }
        extent = abs(jph_rotate(rotation, float3(1, 0, 0))) * half.x
               + abs(jph_rotate(rotation, float3(0, 1, 0))) * half.y
               + abs(jph_rotate(rotation, float3(0, 0, 1))) * half.z;
    }
    extent += padding;
    Aabb result;
    result.minimum = float4(center - extent, 0.0f);
    result.maximum = float4(center + extent, 0.0f);
    return result;
}

bool cvx_is_triangle(CvxOperand o) { return (o.meta.y & ShapeOneSidedTriangle) != 0u; }
float3 cvx_triangle_normal(CvxOperand o) { return jph_rotate(o.rotation, o.geometry.yzw); }
bool cvx_triangle_backfacing(CvxOperand triangle, float3 otherCenter)
{
    return cvx_is_triangle(triangle) && dot(cvx_triangle_normal(triangle), otherCenter-triangle.center.xyz)<0;
}

// Native ActiveEdges::FixNormal. The host maps the cooked edge bits to the
// exact three full-geometry vertices; winding is retained separately. Inputs
// and the result are world-space, with the penetration normal toward the triangle.
float3 cvx_triangle_fix_normal(StructuredBuffer<uint4> g, CvxOperand triangle,
                               float3 point, float3 normal, float3 movement)
{
    uint active=(triangle.meta.y>>ShapeTriangleActiveEdgeShift)&7u;
    if(active==7u) return normal;
    float3 face=-cvx_triangle_normal(triangle);
    float normalLength=length(normal);
    if(dot(movement,normal)<dot(movement,face)*normalLength) return normal;
    if(active==0u) return face;
    if(dot(face,normal)>0.999848f*normalLength) return normal;
    uint first=g[triangle.geometryReference.x+1u].x+triangle.geometryReference.x;
    float3 a=asfloat(g[first]).xyz,b=asfloat(g[first+1u]).xyz,c=asfloat(g[first+2u]).xyz;
    float3 p=jph_inverse_rotate(triangle.rotation,point-triangle.center.xyz);
    float3 e=b-a,f=c-a,q=p-a;
    // Barycentric ratios are scale invariant. Normalize the edge coordinates
    // before cross products so small valid cooked triangles retain their area.
    float scale=max(max(abs(e.x),max(abs(e.y),abs(e.z))),max(abs(f.x),max(abs(f.y),abs(f.z))));
    e/=scale; f/=scale; q/=scale;
    float3 area=cross(e,f);
    float denominator=dot(area,area);
    float v=dot(cross(q,f),area)/denominator,w=dot(cross(e,q),area)/denominator,u=1-v-w;
    uint edge;
    if(u>0.9999f) edge=5u;
    else if(v>0.9999f) edge=3u;
    else if(w>0.9999f) edge=6u;
    else if(u<0.0001f) edge=2u;
    else if(v<0.0001f) edge=4u;
    else if(w<0.0001f) edge=1u;
    else return face;
    return (active&edge)!=0u?normal:face;
}

float4 cvx_f(RWStructuredBuffer<uint4> s,uint b,uint i) { return asfloat(s[b+i]); }
void cvx_put(RWStructuredBuffer<uint4> s,uint b,uint i,float4 v) { s[b+i]=asuint(v); }
void cvx_fail(RWStructuredBuffer<uint4> s,uint b,uint reason)
{ uint4 m=s[b]; m.x=CvxFailure; m.w=reason; s[b]=m; }
float3 cvx_unit(float3 v)
{
    float scale=max(abs(v.x),max(abs(v.y),abs(v.z)));
    float3 n=v/scale;
    return n/sqrt(dot(n,n));
}
CvxOperand cvx_operand(RWStructuredBuffer<uint4> s,uint b,uint side)
{
    uint p=b+10u+5u*side; CvxOperand o;
    o.meta=s[p]; o.center=asfloat(s[p+1u]); o.rotation=asfloat(s[p+2u]);
    o.geometry=asfloat(s[p+3u]); o.geometryReference=s[p+4u]; return o;
}
CvxWitness cvx_point(RWStructuredBuffer<uint4> s,uint b,uint first,uint i)
{
    CvxWitness p; p.a=cvx_f(s,b,first+3u*i).xyz; p.b=cvx_f(s,b,first+3u*i+1u).xyz;
    p.w=cvx_f(s,b,first+3u*i+2u).xyz; return p;
}
void cvx_store_point(RWStructuredBuffer<uint4> s,uint b,uint first,uint i,CvxWitness p)
{
    cvx_put(s,b,first+3u*i,float4(p.a,0)); cvx_put(s,b,first+3u*i+1u,float4(p.b,0));
    cvx_put(s,b,first+3u*i+2u,float4(p.w,0));
}
float cvx_margin(CvxOperand o)
{
    return o.meta.x==CvxCylinder ? o.geometry.z : o.meta.x==CvxHull ? o.geometry.x : o.geometry.w;
}

// full=false: exact native ExcludeConvexRadius core. full=true: native Include,
// including Cylinder's full sharp cylinder, not core inflated by a sphere.
float3 cvx_inline(CvxOperand o,float3 n,bool full)
{
    uint kind=o.meta.x; float4 g=o.geometry; float3 p=float3(0);
    float3 u=cvx_unit(n);
    if(kind==CvxSphere) return full ? g.w*u : p;
    if(kind==CvxCapsule) { p.y=n.y>=0 ? g.y : -g.y; return p+(full?g.w*u:float3(0)); }
    if(kind==CvxBox) {
        float3 h=g.xyz-(full?0:g.w); p=float3(n.x>=0?h.x:-h.x,n.y>=0?h.y:-h.y,n.z>=0?h.z:-h.z);
        return p;
    }
    float radius=g.x-(full?0:g.z),height=g.y-(full?0:g.z);
    float radial=max(abs(n.x),abs(n.z));
    if(radial>0) { float2 d=n.xz/radial; p.xz=radius*d/sqrt(dot(d,d)); }
    p.y=n.y>=0?height:-height; return p;
}

// Geometry header validation is constant work; native host additionally proves
// complete topology/range ownership. Support itself scans ONE vertex per unit.
bool cvx_hull_header(StructuredBuffer<uint4> g,CvxOperand o,out uint count,out uint first,bool full)
{
    uint words,stride; g.GetDimensions(words,stride); uint4 r=o.geometryReference;
    count=0; first=0;
    if(r.y<5u || any(r.zw!=uint2(0)) || r.x>words || r.y>words-r.x) return false;
    uint4 h=g[r.x],v=g[r.x+1u],l=g[r.x+2u];
    first=full?v.x:v.y; count=h.z;
    bool rayRange=l.z==0u || (count==3u && h.w==2u && l.z>=5u && l.z<=r.y && r.y-l.z==3u);
    return h.x==CvxHull && h.y==r.y && count>0u && first>=5u && first<=r.y && count<=r.y-first && l.w==0u && rayRange;
}

bool cvx_operand_extent(StructuredBuffer<uint4> geometry, CvxOperand operand, out float extent)
{
    float4 shape = operand.geometry;
    extent = 0;
    if (operand.meta.x == CvxHull) {
        uint count, first;
        if (!cvx_hull_header(geometry, operand, count, first, true)) return false;
        uint base = operand.geometryReference.x;
        float3 low = asfloat(geometry[base + 3u]).xyz, high = asfloat(geometry[base + 4u]).xyz;
        extent = max(jph_capsule_max_abs(low), jph_capsule_max_abs(high));
    } else if (operand.meta.x == CvxCylinder)
        extent = max(shape.x, shape.y);
    else if (operand.meta.x == CvxCapsule)
        extent = shape.y + shape.w;
    else
        extent = max(jph_capsule_max_abs(shape.xyz), shape.w);
    return true;
}

bool cvx_pair_scale(StructuredBuffer<uint4> geometry, CvxOperand a, CvxOperand b, out float scale)
{
    scale = 0;
    float extentA, extentB;
    if (!cvx_operand_extent(geometry, a, extentA) || !cvx_operand_extent(geometry, b, extentB)) return false;
    scale = max(jph_capsule_max_abs(b.center.xyz - a.center.xyz), max(extentA, extentB));
    return true;
}

void cvx_support_begin(RWStructuredBuffer<uint4> s,uint b,float3 direction,uint next)
{
    cvx_put(s,b,4,float4(direction,0)); s[b+23u]=uint4(0,0,0,next);
    cvx_put(s,b,22,float4(0,0,0,-3.402823466e38f)); uint4 m=s[b]; m.y=CvxSupport; s[b]=m;
}
void cvx_support_unit(StructuredBuffer<uint4> geometry,RWStructuredBuffer<uint4> s,uint b)
{
    uint4 scan=s[b+23u]; CvxOperand o=cvx_operand(s,b,scan.x);
    float3 n=cvx_f(s,b,4).xyz*(scan.x==0u?1.0f:-1.0f);
    float3 local=jph_inverse_rotate(o.rotation,n); bool full=s[b].z!=0u; float3 p;
    if(o.meta.x==CvxHull) {
        uint count,first;
        if(!cvx_hull_header(geometry,o,count,first,full)) { cvx_fail(s,b,1); return; }
        float3 v=asfloat(geometry[o.geometryReference.x+first+scan.y]).xyz;
        float value=dot(v,local); float4 best=cvx_f(s,b,22);
        if(value>best.w) best=float4(v,value);
        ++scan.y; cvx_put(s,b,22,best);
        if(scan.y<count) { s[b+23u]=scan; return; }
        p=best.xyz;
    } else {
        if(o.meta.x>CvxCylinder || any(o.geometryReference!=uint4(0))) { cvx_fail(s,b,2); return; }
        p=cvx_inline(o,local,full);
    }
    float4 origin=cvx_f(s,b,3);
    p=(o.center.xyz-origin.xyz)/origin.w+jph_rotate(o.rotation,p/origin.w);
    if(!all(isfinite(p))) { cvx_fail(s,b,3); return; }
    cvx_put(s,b,20u+scan.x,float4(p,0));
    if(scan.x==0u) { scan.xy=uint2(1,0); cvx_put(s,b,22,float4(0,0,0,-3.402823466e38f)); s[b+23u]=scan; }
    else { uint4 m=s[b]; m.y=scan.w; s[b]=m; }
}
CvxWitness cvx_supported(RWStructuredBuffer<uint4> s,uint b)
{ CvxWitness p; p.a=cvx_f(s,b,20).xyz; p.b=cvx_f(s,b,21).xyz; p.w=p.a-p.b; return p; }

// Closest point on the affine feature selected by mask. Enumerating15 subsets
// costs15 retained units, not an unbounded local simplex solver or private array.
bool cvx_feature(RWStructuredBuffer<uint4> s,uint b,uint first,uint4 ids,uint count,out float4 lambda,out float3 point)
{
    float3 a=cvx_point(s,b,first,ids.x).w;
    lambda=float4(1,0,0,0); point=a;
    if(count==1u) return true;
    float3 e=cvx_point(s,b,first,ids.y).w-a; float ee=dot(e,e);
    if(count==2u) {
        if(ee==0) return false; float t=-dot(a,e)/ee; lambda=float4(1-t,t,0,0);
    } else {
        float3 f=cvx_point(s,b,first,ids.z).w-a; float3 n=cross(e,f);
        if(count==3u) {
            float nn=dot(n,n); if(nn==0) return false;
            float u=dot(cross(-a,f),n)/nn,v=dot(cross(e,-a),n)/nn;
            lambda=float4(1-u-v,u,v,0);
        } else {
            float3 h=cvx_point(s,b,first,ids.w).w-a; float det=dot(e,cross(f,h));
            if(det==0) return false;
            float u=dot(-a,cross(f,h))/det,v=dot(e,cross(-a,h))/det,w=dot(e,cross(f,-a))/det;
            lambda=float4(1-u-v-w,u,v,w);
        }
    }
    if(any(lambda<float4(0)) || !all(isfinite(lambda))) return false;
    point=float3(0);
    for(uint i=0u;i<count;++i) point+=lambda[i]*cvx_point(s,b,first,ids[i]).w;
    return all(isfinite(point));
}
void cvx_metric(RWStructuredBuffer<uint4> s,uint b,float3 normal,float separation,float3 a,float3 p)
{
    float4 origin=cvx_f(s,b,3);
    float4 result=float4(normal,separation*origin.w);
    float3 pa=origin.xyz+a*origin.w,pb=origin.xyz+p*origin.w;
    if(!all(isfinite(result)) || !all(isfinite(pa)) || !all(isfinite(pb))) { cvx_fail(s,b,4); return; }
    cvx_put(s,b,7,result); cvx_put(s,b,8,float4(pa,0)); cvx_put(s,b,9,float4(pb,0));
    uint4 m=s[b]; m.x=CvxSuccess; s[b]=m;
}
// Native GJK's collision/precision classification uses the closest Minkowski
// point, not subtraction of independently reconstructed A/B witnesses.
bool cvx_near_origin(RWStructuredBuffer<uint4> s,uint b,float lengthSquared)
{
    float maximum=0;
    for(uint i=0u;i<s[b+1u].x;++i) {
        float3 w=cvx_point(s,b,CvxSimplex,i).w; maximum=max(maximum,dot(w,w));
    }
    float tolerance=cvx_f(s,b,25).x;
    return lengthSquared<=max(tolerance*tolerance,1.1920928955078125e-7f*maximum);
}
void cvx_distance_done(RWStructuredBuffer<uint4> s,uint b)
{
    float3 a=float3(0),p=float3(0); float4 weights=cvx_f(s,b,6); uint count=s[b+1u].x;
    for(uint i=0u;i<count;++i) { CvxWitness v=cvx_point(s,b,CvxSimplex,i); a+=weights[i]*v.a; p+=weights[i]*v.b; }
    float4 closest=cvx_f(s,b,5); float tolerance=cvx_f(s,b,25).x;
    bool near=closest.w<=tolerance*tolerance;
    bool separated=s[b+24u].w==1u;
    if(s[b+24u].w==2u) {
        // An unfinished low-dimensional distance solve is not a separation
        // certificate. Only a feasible spanning feature in a proven affine
        // hull containing the origin can terminate as tolerance-touch.
        if(s[b+40u].y!=0u && count>=s[b+40u].x+1u && near) cvx_seed_touch_done(s,b,cvx_f(s,b,37).xyz);
        else cvx_fail(s,b,21u);
        return;
    }
    if(cvx_near_origin(s,b,closest.w) && !separated) { s[b].y=CvxSeed; return; }
    float3 v=-closest.xyz; float distance=sqrt(closest.w); float3 n;
    if(near) {
        // This plane was just certified by actual FULL support, so the full
        // shapes are separated even when their distance is below tolerance.
        // Keep full-simplex witnesses and avoid an EPA<->GJK restart cycle.
        float4 plane=cvx_f(s,b,39); n=plane.xyz;
        // The support plane is a lower bound for every point of the FULL
        // simplex. Subtracting the separately rounded A/B witnesses can
        // lose a positive gap when a CCD impact leaves the pair near touch.
        distance=max(plane.w,dot(n,v));
    } else n=cvx_unit(v);
    float scale=cvx_f(s,b,3).w;
    float ra=s[b].z==0u?cvx_margin(cvx_operand(s,b,0))/scale:0;
    float rb=s[b].z==0u?cvx_margin(cvx_operand(s,b,1))/scale:0;
    cvx_metric(s,b,n,distance-ra-rb,a+ra*n,p-rb*n);
}
void cvx_gjk_point(RWStructuredBuffer<uint4> s,uint b)
{
    CvxWitness p=cvx_supported(s,b); uint4 counts=s[b+1u];
    if(counts.x>0u) {
        float4 closest=cvx_f(s,b,5);
        if(s[b+24u].w==2u && any(closest.xyz!=float3(0))) {
            // This is a completed FULL support, so it can certify the current
            // candidate direction. A nonnegative projection does not prove inclusion.
            float3 normal=cvx_unit(-closest.xyz); float projection=dot(normal,p.w);
            if(!isfinite(projection)) { cvx_fail(s,b,18u); return; }
            if(projection<0) {
                cvx_put(s,b,39,float4(normal,-projection)); s[b+24u]=uint4(0,11u,0xffffffffu,1u);
            }
        }
    }
    if(counts.x>=4u) {
        if(s[b+24u].w==1u) cvx_distance_done(s,b);
        else if(s[b+24u].w==2u) cvx_fail(s,b,20u);
        else s[b].y=CvxSeed;
        return;
    }
    cvx_store_point(s,b,CvxSimplex,counts.x,p); ++counts.x; s[b+1u]=counts;
    s[b+26u]=uint4(1,0,0,0); cvx_put(s,b,27,float4(0,0,0,3.402823466e38f)); s[b].y=CvxClosest;
}
void cvx_closest_unit(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 work=s[b+26u]; uint count=s[b+1u].x; uint mask=work.x;
    uint4 ids=uint4(0); uint n=0;
    for(uint i=0u;i<count;++i) if((mask&(1u<<i))!=0u) ids[n++]=i;
    float4 lambda; float3 point;
    if(cvx_feature(s,b,CvxSimplex,ids,n,lambda,point)) {
        float d2=dot(point,point); float4 best=cvx_f(s,b,27);
        if(d2<best.w) { cvx_put(s,b,27,float4(point,d2)); s[b+28u]=ids; cvx_put(s,b,29,lambda); work.y=n; }
    }
    ++work.x; s[b+26u]=work;
    if(work.x==(1u<<count)) s[b].y=CvxReduce;
}
void cvx_reduce(RWStructuredBuffer<uint4> s,uint b)
{
    uint n=s[b+26u].y; if(n==0u) { cvx_fail(s,b,5); return; }
    uint4 ids=s[b+28u]; float4 next=cvx_f(s,b,27),old=cvx_f(s,b,5);
    bool separated=s[b+24u].w==1u,lower=s[b+24u].w==2u; float tolerance=cvx_f(s,b,25).x;
    // A feasible full tetrahedron contains the origin (native set==0xf).
    // Never undo that origin proof merely because its rounded residual stalls.
    if(n==4u) {
        if(separated) { cvx_fail(s,b,20u); return; }
        next=float4(0);
    }
    bool originFeature=lower && s[b+40u].y!=0u &&
        (n>=s[b+40u].x+1u || all(next.xyz==float3(0))) && next.w<=tolerance*tolerance;
    if(n!=4u && !originFeature && !(next.w<old.w)) {
        // Native GetClosest failure undoes only this appended support. Old
        // triples, weights and closest have not been overwritten/compacted.
        --s[b+1u].x; cvx_distance_done(s,b); return;
    }
    for(uint i=0u;i<n;++i) cvx_store_point(s,b,CvxSimplex,i,cvx_point(s,b,CvxSimplex,ids[i]));
    s[b+1u].x=n; cvx_put(s,b,6,cvx_f(s,b,29)); cvx_put(s,b,5,next);
    if(lower) {
        bool feasibleSpan=s[b+40u].y!=0u && n>=s[b+40u].x+1u;
        bool zeroCombination=all(next.xyz==float3(0));
        if(s[b+40u].y!=0u && (feasibleSpan || zeroCombination) && next.w<=tolerance*tolerance) {
            cvx_seed_touch_done(s,b,cvx_f(s,b,37).xyz); return;
        }
    } else {
        if(n==4u && !separated) { s[b].y=CvxSeed; return; }
        if((cvx_near_origin(s,b,next.w) && !separated) || next.w<=tolerance*tolerance) { cvx_distance_done(s,b); return; }
    }
    // A proven lower-dimensional affine hull cannot contain a tetrahedron;
    // a real separation certificate also excludes four-point containment.
    if(n==4u) { cvx_fail(s,b,20u); return; }
    // Native relative improvement termination applies to this new valid
    // simplex, after near-origin/containment classification.
    if(old.w-next.w<=1.1920928955078125e-7f*old.w) { cvx_distance_done(s,b); return; }
    ++s[b+2u].y;
    if(s[b+2u].y>=128u) { cvx_fail(s,b,6); return; }
    cvx_support_begin(s,b,-next.xyz,CvxGjkPoint);
}

// Caller supplies positive finite scale from the pair's real bounds and centers.
// The entire2048-word slot must already have an accepted direct initialization.
void cvx_begin(RWStructuredBuffer<uint4> s,uint b,CvxOperand a,CvxOperand p,float3 origin,float scale,float tolerance,bool full=false)
{
    s[b]=uint4(CvxPending,CvxGjk,full?1u:0u,0); s[b+1u]=uint4(0); s[b+2u]=uint4(0);
    cvx_put(s,b,3,float4(origin,scale)); cvx_put(s,b,5,float4(0,0,0,3.402823466e38f));
    s[b+24u]=uint4(0); cvx_put(s,b,25,float4(tolerance/scale,1e-4f,3.402823466e38f,0));
    s[b+10u]=a.meta; cvx_put(s,b,11,a.center); cvx_put(s,b,12,a.rotation); cvx_put(s,b,13,a.geometry); s[b+14u]=a.geometryReference;
    s[b+15u]=p.meta; cvx_put(s,b,16,p.center); cvx_put(s,b,17,p.rotation); cvx_put(s,b,18,p.geometry); s[b+19u]=p.geometryReference;
    if(!(scale>0) || !isfinite(scale) || !all(isfinite(origin))) cvx_fail(s,b,7);
}

float3 cvx_perpendicular(float3 a)
{
    float3 v=abs(a); float3 axis=v.x<=v.y && v.x<=v.z?float3(1,0,0):v.y<=v.z?float3(0,1,0):float3(0,0,1);
    return cvx_unit(cross(a,axis));
}
void cvx_full_separation(RWStructuredBuffer<uint4> s,uint b,float3 normal,float gap,uint face)
{
    if(!all(isfinite(normal)) || !isfinite(gap) || !(gap>0)) { cvx_fail(s,b,18u); return; }
    cvx_put(s,b,39,float4(normal,gap));
    s[b+1u].x=0; s[b+24u]=uint4(0,11u,face,1u);
    float4 quality=cvx_f(s,b,25); quality.z=3.402823466e38f; quality.w=0; cvx_put(s,b,25,quality);
    cvx_put(s,b,5,float4(0,0,0,3.402823466e38f)); s[b].yz=uint2(CvxGjk,1u);
}
void cvx_seed_touch_done(RWStructuredBuffer<uint4> s,uint b,float3 n)
{
    // The support-width certificate established a genuinely lower-dimensional
    // full Minkowski shape. Its zero-distance GJK simplex already provides the
    // touching witnesses. A bootstrap triangle need not contain the origin
    // (a rectangle's selected three corners cover only half of its area).
    float3 a=float3(0),q=float3(0); float4 lambda=cvx_f(s,b,6);
    for(uint i=0u;i<s[b+1u].x;++i) {
        CvxWitness v=cvx_point(s,b,CvxSimplex,i); a+=lambda[i]*v.a; q+=lambda[i]*v.b;
    }
    float tolerance=cvx_f(s,b,25).x;
    if(cvx_f(s,b,5).w>tolerance*tolerance) { cvx_fail(s,b,21u); return; }
    s[b+24u]=uint4(1,10,0xffffffffu,0); cvx_put(s,b,27,float4(0));
    cvx_metric(s,b,n,0,a,q);
}
void cvx_lower_distance_begin(RWStructuredBuffer<uint4> s,uint b)
{
    s[b]=uint4(CvxPending,CvxGjk,1u,0); s[b+1u].x=0;
    s[b+24u]=uint4(0,12u,0xffffffffu,2u);
    cvx_put(s,b,5,float4(0,0,0,3.402823466e38f));
    float4 quality=cvx_f(s,b,25); quality.z=3.402823466e38f; quality.w=0; cvx_put(s,b,25,quality);
}
void cvx_seed_touch(RWStructuredBuffer<uint4> s,uint b,float3 n,uint rank)
{
    // Every completed +/- FULL bootstrap probe was nonnegative. Together with
    // its zero width this places the origin in the corresponding numerical
    // affine band (plane; two line normals; or three point axes). It does not
    // prove polygon/segment containment: the same full GJK must prove a
    // feasible spanning feature within absolute tolerance or find separation.
    cvx_put(s,b,37,float4(n,0)); s[b+40u]=uint4(rank,1u,0,0);
    cvx_lower_distance_begin(s,b);
}
// A support already in the seed triangle has exactly zero affine height.
// Preserve this identity before rounded cross/dot arithmetic can invent a
// fourth vertex. A distinct support keeps its actual signed height.
float cvx_seed_plane_height(float3 n,float3 a,float3 v,float3 c,float3 p)
{
    if(all(p==a) || all(p==v) || all(p==c)) return 0;
    return dot(n,p-a);
}
void cvx_seed_point(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 seed=s[b+30u]; CvxWitness p=cvx_supported(s,b);
    // CvxSeed switches to FULL before issuing every bootstrap support.
    // Test that completed support, not an arbitrary vertex's rounded offset.
    float3 direction=cvx_f(s,b,4).xyz; float projection=dot(direction,p.w);
    if(!isfinite(projection)) { cvx_fail(s,b,18u); return; }
    if(projection<0) {
        float length=sqrt(dot(direction,direction));
        cvx_full_separation(s,b,direction/length,-projection/length,0xffffffffu); return;
    }
    if(seed.x==0u) {
        cvx_store_point(s,b,CvxPoints,0,p); seed.x=1u; s[b+30u]=seed;
        cvx_support_begin(s,b,-cvx_f(s,b,4).xyz,CvxSeedPoint); return;
    }
    if(seed.x==1u) {
        cvx_store_point(s,b,CvxPoints,1,p); float3 axis=p.w-cvx_point(s,b,CvxPoints,0).w;
        if(all(axis==float3(0))) {
            ++seed.z;
            if(seed.z<3u) { seed.x=0; s[b+30u]=seed; cvx_support_begin(s,b,seed.z==1u?float3(0,1,0):float3(0,0,1),CvxSeedPoint); }
            else cvx_seed_touch(s,b,cvx_unit(cvx_f(s,b,4).xyz),0u);
            return;
        }
        axis=cvx_unit(axis); cvx_put(s,b,36,float4(axis,0));
        seed.x=2; seed.y=0; s[b+30u]=seed;
        cvx_support_begin(s,b,cvx_perpendicular(axis),CvxSeedPoint); return;
    }
    if(seed.x==2u) {
        cvx_store_point(s,b,CvxPoints,2,p); seed.x=3; s[b+30u]=seed;
        cvx_support_begin(s,b,-cvx_f(s,b,4).xyz,CvxSeedPoint); return;
    }
    if(seed.x==3u) {
        CvxWitness a=cvx_point(s,b,CvxPoints,0),v=cvx_point(s,b,CvxPoints,1),c=cvx_point(s,b,CvxPoints,2);
        float3 line=v.w-a.w,n=cross(line,c.w-a.w),other=cross(line,p.w-a.w);
        if(dot(other,other)>dot(n,n)) { c=p; n=other; cvx_store_point(s,b,CvxPoints,2,p); }
        if(dot(n,n)==0) {
            if(seed.y==0u) { seed.x=2; seed.y=1; s[b+30u]=seed; cvx_support_begin(s,b,cross(cvx_f(s,b,36).xyz,cvx_perpendicular(cvx_f(s,b,36).xyz)),CvxSeedPoint); }
            else cvx_seed_touch(s,b,cvx_unit(cvx_f(s,b,4).xyz),1u);
            return;
        }
        seed.x=4; s[b+30u]=seed; cvx_put(s,b,37,float4(cvx_unit(n),0));
        cvx_support_begin(s,b,cvx_unit(n),CvxSeedPoint); return;
    }
    if(seed.x==4u) {
        cvx_store_point(s,b,CvxPoints,3,p); seed.x=5; s[b+30u]=seed;
        cvx_support_begin(s,b,-cvx_f(s,b,37).xyz,CvxSeedPoint); return;
    }
    CvxWitness a=cvx_point(s,b,CvxPoints,0),v=cvx_point(s,b,CvxPoints,3);
    float3 edge=cvx_point(s,b,CvxPoints,1).w,corner=cvx_point(s,b,CvxPoints,2).w;
    float3 n=cvx_f(s,b,37).xyz;
    float d=cvx_seed_plane_height(n,a.w,edge,corner,v.w),other=cvx_seed_plane_height(n,a.w,edge,corner,p.w);
    if(abs(other)>abs(d)) { v=p; d=other; cvx_store_point(s,b,CvxPoints,3,p); }
    if(d==0) { cvx_seed_touch(s,b,n,2u); return; }
    float3 interior=float3(0);
    for(uint i=0u;i<4u;++i) interior+=cvx_point(s,b,CvxPoints,i).w*.25f;
    cvx_put(s,b,31,float4(interior,0)); s[b+1u].yz=uint2(4,4); s[b+32u]=uint4(0);
    s[b].y=CvxFaceInit;
}

// Faces use5 words as designed. All geometry quantities stay in pair-scaled
// coordinates. Active/eligible bits are independent; noninterior faces still
// participate in topology but cannot become an EPA distance certificate.
uint cvx_face_write(RWStructuredBuffer<uint4> s,uint b,uint index,uint3 ids)
{
    float3 a=cvx_point(s,b,CvxPoints,ids.x).w,v=cvx_point(s,b,CvxPoints,ids.y).w,c=cvx_point(s,b,CvxPoints,ids.z).w;
    float3 n=cross(v-a,c-a); float nn=dot(n,n);
    if(!isfinite(nn)) return 2u;
    if(nn==0) return 1u;
    if(dot(n,cvx_f(s,b,31).xyz-a)>0) { uint t=ids.y; ids.y=ids.z; ids.z=t; n=-n; float3 z=v; v=c; c=z; }
    float d=dot(n,a); float u=dot(cross(-a,c-a),n)/nn,w=dot(cross(v-a,-a),n)/nn;
    float3 lambda=float3(1-u-w,u,w);
    // Native EPA queues every origin-facing face during enclosure, even if
    // the origin's plane projection is outside this triangle.
    bool eligible=d<0 || all(lambda>=float3(-1e-3f));
    uint first=b+CvxFaces+5u*index;
    s[first]=uint4(ids,eligible?3u:1u); s[first+1u]=asuint(float4(n,d));
    s[first+2u]=asuint(float4(lambda,(d<0?-1.0f:1.0f)*d*d/nn));
    s[first+3u]=asuint(float4((a+v+c)/3.0f,0)); s[first+4u]=uint4(0);
    return all(isfinite(lambda))?0u:2u;
}
// Native EPA keeps its last valid face when a finite expansion reaches its
// bounded point/topology capacity or numerical precision. The saved witnesses
// are independent of face slots which expansion may have retired/reused.
void cvx_epa_stop(RWStructuredBuffer<uint4> s,uint b,uint reason,uint failure)
{
    if(s[b+24u].x==0u) { cvx_fail(s,b,failure); return; }
    s[b+24u].y=reason; s[b].x=CvxSuccess;
}
void cvx_face_scan(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 scan=s[b+32u]; uint count=s[b+1u].z;
    if(scan.x<count) {
        uint p=b+CvxFaces+5u*scan.x; uint4 face=s[p];
        if((face.w&3u)==3u) {
            float d=asfloat(s[p+2u]).w;
            if(scan.y==0xffffffffu || d<cvx_f(s,b,38).w) { scan.y=scan.x; cvx_put(s,b,38,float4(0,0,0,d)); }
        }
        ++scan.x; s[b+32u]=scan; return;
    }
    if(scan.y==0xffffffffu) { cvx_epa_stop(s,b,9u,10u); return; }
    if(s[b+24u].x!=0u && cvx_f(s,b,38).w>=cvx_f(s,b,25).z) { cvx_epa_stop(s,b,3u,10u); return; }
    float3 direction=asfloat(s[b+CvxFaces+5u*scan.y+1u]).xyz;
    cvx_support_begin(s,b,direction,CvxEpaPoint);
}
void cvx_epa_done(RWStructuredBuffer<uint4> s,uint b,uint faceIndex)
{
    uint first=b+CvxFaces+5u*faceIndex; uint3 ids=s[first].xyz;
    float4 nd=asfloat(s[first+1u]); float3 lambda=asfloat(s[first+2u]).xyz;
    float3 a=float3(0),p=float3(0);
    for(uint i=0u;i<3u;++i) { CvxWitness v=cvx_point(s,b,CvxPoints,ids[i]); a+=lambda[i]*v.a; p+=lambda[i]*v.b; }
    float length=sqrt(dot(nd.xyz,nd.xyz));
    cvx_metric(s,b,nd.xyz/length,-nd.w/length,a,p);
}
void cvx_epa_point(RWStructuredBuffer<uint4> s,uint b)
{
    uint chosen=s[b+32u].y; uint first=b+CvxFaces+5u*chosen;
    float4 nd=asfloat(s[first+1u]); float faceDistance=asfloat(s[first+2u]).w;
    CvxWitness p=cvx_supported(s,b); float projection=dot(nd.xyz,p.w),nn=dot(nd.xyz,nd.xyz);
    if(!isfinite(projection) || !isfinite(nn) || !(nn>0)) { cvx_fail(s,b,18u); return; }
    if(projection<0) {
        // A native shrunken core can exceed the full shape by float rounding.
        // Re-run the SAME GJK with full support to obtain a separation metric.
        float length=sqrt(nn);
        cvx_full_separation(s,b,nd.xyz/length,-projection/length,chosen); return;
    }
    float upper=projection*projection/nn;
    if(!isfinite(upper)) { cvx_fail(s,b,18u); return; }
    if(faceDistance>=0) {
        cvx_epa_done(s,b,chosen); if(s[b].x==CvxFailure) return;
        s[b].x=CvxPending; s[b+24u]=uint4(1u,0u,chosen,0u);
        float lower=sqrt(faceDistance),high=sqrt(upper),scale=cvx_f(s,b,3).w;
        float relative=faceDistance>0?(upper-faceDistance)/faceDistance:(upper==0?0:3.402823466e38f);
        cvx_put(s,b,27,float4(lower*scale,high*scale,(high-lower)*scale,relative));
        float4 quality=cvx_f(s,b,25); quality.z=min(quality.z,upper); quality.w=relative; cvx_put(s,b,25,quality);
        if(upper-faceDistance<=quality.y*faceDistance) { cvx_epa_stop(s,b,1u,18u); return; }
        if(dot(nd.xyz,p.w-asfloat(s[first+3u]).xyz)<=0) { cvx_epa_stop(s,b,2u,18u); return; }
    }
    uint point=s[b+1u].y;
    if(point>=128u) { cvx_epa_stop(s,b,4u,11u); return; }
    cvx_store_point(s,b,CvxPoints,point,p); ++s[b+1u].y;
    s[b+1u].w=0; s[b+32u].x=0; s[b+33u]=uint4(0); s[b].y=CvxVisible;
}
void cvx_visible(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+32u]; uint count=s[b+1u].z;
    if(cursor.x==count) { s[b+33u]=uint4(0); s[b].y=CvxCreate; return; }
    uint first=b+CvxFaces+5u*cursor.x; ++s[b+32u].x;
    uint4 face=s[first];
    if((face.w&1u)==0u) return;
    float3 n=asfloat(s[first+1u]).xyz,center=asfloat(s[first+3u]).xyz;
    float3 p=cvx_point(s,b,CvxPoints,s[b+1u].y-1u).w;
    if(dot(n,p-center)<=0) return;
    s[first].w=0; s[b+34u]=uint4(face.xyz,0); s[b+35u]=uint4(0,0xffffffffu,0,0);
    s[b].y=CvxEdge;
}
void cvx_edge_unit(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 face=s[b+34u],cursor=s[b+35u]; uint edge=face.w;
    uint a=face[edge],p=face[(edge+1u)%3u]; uint count=s[b+1u].w;
    bool finished=false;
    if(cursor.x<count) {
        uint4 h=s[b+CvxHorizon+cursor.x];
        if(h.z==0u && cursor.y==0xffffffffu) cursor.y=cursor.x;
        if(h.z!=0u && h.x==p && h.y==a) { s[b+CvxHorizon+cursor.x].z=0; finished=true; }
        else if(h.z!=0u && h.x==a && h.y==p) { cvx_epa_stop(s,b,7u,12u); return; }
        ++cursor.x;
    } else {
        uint slot=cursor.y==0xffffffffu?count:cursor.y;
        if(slot>=128u) { cvx_epa_stop(s,b,6u,13u); return; }
        s[b+CvxHorizon+slot]=uint4(a,p,1,0);
        if(slot==count) ++s[b+1u].w;
        finished=true;
    }
    if(finished) {
        ++face.w; s[b+34u]=face; cursor=uint4(0,0xffffffffu,0,0);
        if(face.w==3u) s[b].y=CvxVisible;
    }
    s[b+35u]=cursor;
}
void cvx_create_unit(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+33u]; uint count=s[b+1u].w;
    if(cursor.x==count) {
        ++s[b+2u].z; s[b+32u]=uint4(0,0xffffffffu,0,0); s[b].y=CvxFaceMin; return;
    }
    uint4 edge=s[b+CvxHorizon+cursor.x];
    if(edge.z==0u) { ++cursor.x; cursor.y=0; s[b+33u]=cursor; return; }
    if(cursor.y<s[b+1u].z && (s[b+CvxFaces+5u*cursor.y].w&1u)!=0u) { ++s[b+33u].y; return; }
    if(cursor.y>=256u) { cvx_epa_stop(s,b,5u,14u); return; }
    uint result=cvx_face_write(s,b,cursor.y,uint3(edge.x,edge.y,s[b+1u].y-1u));
    if(result!=0u) { if(result==1u) cvx_epa_stop(s,b,8u,15u); else cvx_fail(s,b,15u); return; }
    if(asfloat(s[b+CvxFaces+5u*cursor.y+1u]).w<0 && s[b+24u].x!=0u) { cvx_epa_stop(s,b,8u,15u); return; }
    s[b+1u].z=max(s[b+1u].z,cursor.y+1u); ++cursor.x; cursor.y=0; s[b+33u]=cursor;
}
void cvx_epa_unit(StructuredBuffer<uint4> geometry,RWStructuredBuffer<uint4> s,uint b)
{
    uint phase=s[b].y;
    if(phase==CvxSeed) {
        s[b].z=1; s[b+30u]=uint4(0); cvx_support_begin(s,b,float3(1,0,0),CvxSeedPoint);
    } else if(phase==CvxSeedPoint) cvx_seed_point(s,b);
    else if(phase==CvxFaceInit) {
        uint i=s[b+32u].x;
        uint3 ids=i==0?uint3(0,1,2):i==1?uint3(0,3,1):i==2?uint3(0,2,3):uint3(1,3,2);
        if(cvx_face_write(s,b,i,ids)!=0u) { cvx_fail(s,b,16); return; }
        ++s[b+32u].x;
        if(i==3u) { s[b+32u]=uint4(0,0xffffffffu,0,0); s[b].y=CvxFaceMin; }
    } else if(phase==CvxFaceMin) cvx_face_scan(s,b);
    else if(phase==CvxEpaPoint) cvx_epa_point(s,b);
    else if(phase==CvxVisible) cvx_visible(s,b);
    else if(phase==CvxEdge) cvx_edge_unit(s,b);
    else if(phase==CvxCreate) cvx_create_unit(s,b);
    else cvx_fail(s,b,17);
}
void cvx_advance(StructuredBuffer<uint4> geometry,RWStructuredBuffer<uint4> s,uint b,uint workUnits)
{
    for(uint unit=0u;unit<min(workUnits,32u) && s[b].x==CvxPending;++unit) {
        uint phase=s[b].y;
        if(s[b+2u].x==0xffffffffu) { cvx_fail(s,b,8); return; }
        ++s[b+2u].x;
        if(phase==CvxGjk) {
            float3 direction=(cvx_operand(s,b,1).center.xyz-cvx_operand(s,b,0).center.xyz)/cvx_f(s,b,3).w;
            if(all(direction==float3(0))) direction=float3(1,0,0); // Native GJK's initial, arbitrary search direction.
            cvx_support_begin(s,b,direction,CvxGjkPoint);
        } else if(phase==CvxSupport) cvx_support_unit(geometry,s,b);
        else if(phase==CvxGjkPoint) cvx_gjk_point(s,b);
        else if(phase==CvxClosest) cvx_closest_unit(s,b);
        else if(phase==CvxReduce) cvx_reduce(s,b);
        else cvx_epa_unit(geometry,s,b);
    }
}

static const uint CvxCastPending = 0u;
static const uint CvxCastHit = 1u;
static const uint CvxCastMiss = 2u;
static const uint CvxCastError = 3u;
static const uint CvxCastPlane = 128u;
static const float CvxCastTolerance = 1e-4f;
static const uint CvxCastOrigin = 54u;
static const uint CvxCastCursor = 55u;
static const uint CvxCastMotion = 56u;
static const uint CvxCastPlaneNormal = 57u;

// Separation is a witnessed distance; gap is the lower bound from a real
// separating support plane. Only the latter may advance a moving pair.
uint cvx_cast_progress(float separation, float gap, float closing, float limit,
                        inout float travel, inout uint restarted)
{
    float remaining = limit - travel;
    if (!isfinite(gap) || !isfinite(closing) || remaining < 0.0f) return CvxCastError;
    if (gap > 0.0f && (closing <= 0.0f || gap > closing * remaining)) return CvxCastMiss;
    if (separation <= 0.0f ||
        (restarted != 0u && gap <= 0.0f && separation <= CvxCastTolerance) ||
        (travel > 0.0f && separation <= CvxCastTolerance && closing >= 0.0f))
        return CvxCastHit;
    if (gap > 0.0f && closing > 0.0f) {
        float advance = gap / closing;
        float next = min(travel + advance, limit);
        if (!isfinite(advance) || !isfinite(next)) return CvxCastError;
        if (next > travel) {
            travel = next;
            return CvxCastPending;
        }
    }
    // One coordinate restart resolves a rounding disagreement without treating
    // a numerically stalled cast as a miss or advancing past its certificate.
    if (restarted != 0u) return CvxCastError;
    restarted = 1u;
    return CvxCastPending;
}

void cvx_cast_metric(StructuredBuffer<uint4> geometry, RWStructuredBuffer<uint4> scratch,
                     uint base, CvxOperand moving, CvxOperand target, float3 direction, float travel, uint restarted)
{
    moving.center = float4(direction * travel, 0.0f);
    float scale;
    if (!cvx_pair_scale(geometry, moving, target, scale)) {
        cvx_fail(scratch, base, 7u);
        return;
    }
    float3 origin = moving.center.xyz;
    if (restarted != 0u)
        origin += 0.5f * (target.center.xyz - origin);
    cvx_begin(scratch, base, moving, target, origin, scale, CvxCastTolerance, true);
}

void cvx_cast_begin(StructuredBuffer<uint4> geometry, RWStructuredBuffer<uint4> scratch,
                    uint base, CvxOperand moving, CvxOperand target, float3 direction, float limit)
{
    // Accumulate travel relative to A's initial center. Repeatedly adding
    // a small advance to a large world coordinate destroys the certificate
    // that the support point moved by that advance.
    cvx_put(scratch, base, CvxCastOrigin, moving.center);
    target.center.xyz -= moving.center.xyz;
    scratch[base + CvxCastCursor] = uint4(0u);
    cvx_put(scratch, base, CvxCastMotion, float4(direction, limit));
    cvx_cast_metric(geometry, scratch, base, moving, target, direction, 0.0f, 0u);
}

// The inputs start in a common frame. Direction and horizon describe A
// relative to B. Each invocation advances one metric or support unit.
uint cvx_cast_unit(StructuredBuffer<uint4> geometry, RWStructuredBuffer<uint4> scratch, uint base)
{
    uint4 work = scratch[base];
    if (work.x == CvxFailure) return CvxCastError;
    bool support = work.y == CvxSupport && scratch[base + 23u].w == CvxCastPlane;
    if (support)
        cvx_support_unit(geometry, scratch, base);
    else
        cvx_advance(geometry, scratch, base, 1u);
    work = scratch[base];
    if (work.x == CvxFailure) return CvxCastError;
    if (support ? work.y != CvxCastPlane : work.x != CvxSuccess) return CvxCastPending;

    float4 metric = cvx_f(scratch, base, 7u);
    if (metric.w > 0.0f && !support) {
        scratch[base].x = CvxPending;
        cvx_support_begin(scratch, base, metric.xyz, CvxCastPlane);
        return CvxCastPending;
    }
    float gap = metric.w;
    if (support) {
        gap = dot(metric.xyz, cvx_f(scratch, base, 21u).xyz - cvx_f(scratch, base, 20u).xyz)
              * cvx_f(scratch, base, 3u).w;
        if (work.z == 0u)
            gap -= cvx_margin(cvx_operand(scratch, base, 0u)) + cvx_margin(cvx_operand(scratch, base, 1u));
    }
    float4 motion = cvx_f(scratch, base, CvxCastMotion);
    uint4 cursor = scratch[base + CvxCastCursor];
    float previousTravel = asfloat(cursor.x);
    float travel = previousTravel;
    if (cursor.y == 0xffffffffu) {
        cvx_fail(scratch, base, 64u);
        return CvxCastError;
    }
    ++cursor.y;
    uint result = cvx_cast_progress(metric.w, gap, dot(metric.xyz, motion.xyz), motion.w, travel, cursor.z);
    if (result == CvxCastPending) {
        if (travel > previousTravel)
            cvx_put(scratch, base, CvxCastPlaneNormal, float4(metric.xyz, 0.0f));
        cursor.x = asuint(travel);
        scratch[base + CvxCastCursor] = cursor;
        cvx_cast_metric(geometry, scratch, base, cvx_operand(scratch, base, 0u), cvx_operand(scratch, base, 1u),
                        motion.xyz, travel, cursor.z);
        return CvxCastPending;
    }
    // The last advancing plane remains a support plane at the new distance.
    // Near tangency the closest-point witnesses can still be valid while
    // their tiny separation vector is too ill-conditioned for an impulse.
    if (result == CvxCastHit && travel > 0.0f)
        cvx_put(scratch, base, 7u, float4(cvx_f(scratch, base, CvxCastPlaneNormal).xyz, metric.w));
    if (result == CvxCastError) cvx_fail(scratch, base, 64u);
    return result;
}

float cvx_cast_distance(RWStructuredBuffer<uint4> scratch, uint base)
{
    return asfloat(scratch[base + CvxCastCursor].x);
}

// Add this offset to the metric witnesses to return to the caller's frame.
float3 cvx_cast_origin(RWStructuredBuffer<uint4> scratch, uint base)
{
    return cvx_f(scratch, base, CvxCastOrigin).xyz;
}

#endif
