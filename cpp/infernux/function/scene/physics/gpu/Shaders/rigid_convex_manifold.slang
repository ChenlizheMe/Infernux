// Persistent supporting-face/clipping/reduction. Jolt GetSupportingFace,
// ClipPoly and ManifoldBetweenTwoFaces conventions; no invocation-local polygon.
#ifndef JPH_GPU_RIGID_CONVEX_MANIFOLD_SLANG
#define JPH_GPU_RIGID_CONVEX_MANIFOLD_SLANG
#include "rigid_convex.slang"
static const uint CvxFaceSelect=40u,CvxFaceVertices=41u,CvxClipInit=42u,CvxClip=43u;
static const uint CvxProject=44u,CvxPrune=45u,CvxPublish=46u;
static const uint CvxFaceA=460u,CvxFaceB=492u,CvxClipA=524u,CvxClipB=588u;
static const uint CvxContacts=652u,CvxManifoldOut=844u;

void cvx_manifold_primary(RWStructuredBuffer<uint4> s,uint b)
{
    cvx_put(s,b,CvxManifoldOut,cvx_f(s,b,8)); cvx_put(s,b,CvxManifoldOut+1u,cvx_f(s,b,9));
    s[b+CvxManifoldOut+2u]=uint4(0); s[b+36u]=uint4(1,0,0,1); s[b].x=CvxSuccess;
}
void cvx_manifold_begin(RWStructuredBuffer<uint4> s,uint b,float maximumContactDistance,float3 bodyComA)
{
    if(s[b].x!=CvxSuccess) return;
    s[b].xy=uint2(CvxPending,CvxFaceSelect); s[b+39u]=uint4(0,0,0xffffffffu,0);
    s[b+41u]=uint4(0); cvx_put(s,b,40,float4(0,0,0,-3.402823466e38f));
    cvx_put(s,b,46,float4(maximumContactDistance/cvx_f(s,b,3).w,0,0,0));
    cvx_put(s,b,47,float4((bodyComA-cvx_f(s,b,3).xyz)/cvx_f(s,b,3).w,0));
}
void cvx_next_face(RWStructuredBuffer<uint4> s,uint b)
{
    uint side=s[b+39u].x;
    if(side==0u) { s[b+39u]=uint4(1,0,0xffffffffu,0); cvx_put(s,b,40,float4(0,0,0,-3.402823466e38f)); s[b].y=CvxFaceSelect; }
    else s[b].y=CvxClipInit;
}
void cvx_face_select(StructuredBuffer<uint4> g,RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+39u]; CvxOperand o=cvx_operand(s,b,cursor.x);
    float3 n=jph_inverse_rotate(o.rotation,cvx_f(s,b,7).xyz)*(cursor.x==0u?1.0f:-1.0f);
    uint count=0;
    if(o.meta.x==CvxHull) {
        uint vertices,first;
        if(!cvx_hull_header(g,o,vertices,first,true)) { cvx_fail(s,b,20); return; }
        uint4 header=g[o.geometryReference.x],offset=g[o.geometryReference.x+1u];
        if(cursor.y<header.w) {
            if(offset.z>=o.geometryReference.y || header.w>o.geometryReference.y-offset.z) { cvx_fail(s,b,21); return; }
            float4 plane=asfloat(g[o.geometryReference.x+offset.z+cursor.y]);
            float value=dot(plane.xyz,n)/sqrt(dot(plane.xyz,plane.xyz));
            if(value>cvx_f(s,b,40).w) { cursor.z=cursor.y; cvx_put(s,b,40,float4(n,value)); }
            ++cursor.y; s[b+39u]=cursor; return;
        }
        if(cursor.z==0xffffffffu || offset.w>=o.geometryReference.y || header.w>o.geometryReference.y-offset.w) { cvx_fail(s,b,22); return; }
        uint4 face=g[o.geometryReference.x+offset.w+cursor.z];
        // Exactly ConvexHullShape::GetSupportingFace's capacity/2 stride law.
        uint delta=(face.y+16u)/16u;
        count=delta==0u?0u:(face.y+delta-1u)/delta;
        s[b+45u]=uint4(face.x,delta,first,0);
    } else if(o.meta.x==CvxBox) count=4;
    else if(o.meta.x==CvxCylinder) count=dot(n.xz,n.xz)>n.y*n.y?2u:8u;
    else if(o.meta.x==CvxCapsule) {
        if(dot(n.xz,n.xz)>0 && abs(2.0f*o.geometry.y*n.y)<.02f*sqrt(dot(n,n))) count=2;
    } else if(o.meta.x!=CvxSphere) { cvx_fail(s,b,23); return; }
    if(count>32u) { cvx_fail(s,b,24); return; }
    s[b+41u][cursor.x]=count; s[b+39u].w=0;
    cvx_put(s,b,40,float4(n,cvx_f(s,b,40).w)); s[b].y=CvxFaceVertices;
    if(count==0u) cvx_next_face(s,b);
}
float3 cvx_cylinder_cap(uint i)
{
    const float k=.707106769f;
    return i==0?float3(0,1,1):i==1?float3(k,1,k):i==2?float3(1,1,0):i==3?float3(k,1,-k):
           i==4?float3(0,1,-1):i==5?float3(-k,1,-k):i==6?float3(-1,1,0):float3(-k,1,k);
}
void cvx_face_vertex(StructuredBuffer<uint4> g,RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+39u]; uint side=cursor.x,i=cursor.w,count=s[b+41u][side];
    CvxOperand o=cvx_operand(s,b,side); float3 toward=cvx_f(s,b,40).xyz,p=float3(0);
    if(o.meta.x==CvxHull) {
        uint4 topology=s[b+45u],indexInfo=g[o.geometryReference.x+2u];
        uint element=topology.x+i*topology.y;
        if(element>=indexInfo.y || indexInfo.x>=o.geometryReference.y || element/4u>=o.geometryReference.y-indexInfo.x) { cvx_fail(s,b,25); return; }
        uint index=g[o.geometryReference.x+indexInfo.x+element/4u][element%4u];
        if(index>=g[o.geometryReference.x].z) { cvx_fail(s,b,26); return; }
        p=asfloat(g[o.geometryReference.x+topology.z+index]).xyz;
    } else if(o.meta.x==CvxBox) {
        float3 v=abs(toward); uint axis=v.x>=v.y && v.x>=v.z?0u:v.y>=v.z?1u:2u;
        uint u=(axis+1u)%3u,w=(axis+2u)%3u; float sign=toward[axis]>=0?1.0f:-1.0f;
        p[axis]=sign*o.geometry[axis];
        p[u]=(i==0u || i==3u?-1.0f:1.0f)*o.geometry[u];
        p[w]=(i<2u?-1.0f:1.0f)*o.geometry[w]*sign;
    } else if(o.meta.x==CvxCapsule) {
        p=o.geometry.w*cvx_unit(float3(toward.x,0,toward.z)); p.y=i==0?o.geometry.y:-o.geometry.y;
    } else if(count==2u) {
        p.xz=o.geometry.x*cvx_unit(float3(toward.x,0,toward.z)).xz; p.y=i==0?o.geometry.y:-o.geometry.y;
    } else {
        // Native Cylinder cap's8 vertices and5-degree radial alignment law.
        float3 v=cvx_cylinder_cap(i); float3 input=-toward;
        v*=input.y<0?float3(o.geometry.x,o.geometry.y,o.geometry.x):float3(-o.geometry.x,-o.geometry.y,o.geometry.x);
        if(dot(input.xz,input.xz)>.00765427f*input.y*input.y) {
            float3 x=cvx_unit(float3(input.x,0,input.z)),z=float3(-x.z,0,x.x);
            p=x*v.x+float3(0,v.y,0)+z*v.z;
        } else p=v;
    }
    float4 origin=cvx_f(s,b,3);
    float3 world=(o.center.xyz-origin.xyz)/origin.w+jph_rotate(o.rotation,p/origin.w);
    if(!all(isfinite(world))) { cvx_fail(s,b,27); return; }
    cvx_put(s,b,(side==0u?CvxFaceA:CvxFaceB)+i,float4(world,0));
    ++s[b+39u].w; if(i+1u==count) cvx_next_face(s,b);
}
void cvx_clip_setup(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 faces=s[b+41u];
    if(min(faces.x,faces.y)<2u || max(faces.x,faces.y)<3u) { cvx_manifold_primary(s,b); return; }
    bool swap=faces.y<3u; uint reference=swap?CvxFaceB:CvxFaceA,incident=swap?CvxFaceA:CvxFaceB;
    uint refCount=swap?faces.y:faces.x,incCount=swap?faces.x:faces.y;
    float3 n=cvx_f(s,b,7).xyz*(swap?-1.0f:1.0f);
    float3 origin=cvx_f(s,b,reference).xyz,edge=cvx_f(s,b,reference+1u).xyz-origin;
    float3 plane=refCount>=3u?cross(edge,cvx_f(s,b,reference+2u).xyz-origin):cross(cross(edge,n),edge);
    float d=dot(n,plane);
    if(d==0) { cvx_manifold_primary(s,b); return; }
    cvx_put(s,b,43,float4(origin,d)); cvx_put(s,b,44,float4(plane,0));
    cvx_put(s,b,40,float4(n,0)); s[b+45u]=uint4(reference,incident,refCount,incCount);
    s[b+35u]=uint4(0,refCount==2u?1u:0u,swap?1u:0u,0);
    s[b+42u]=uint4(0,0,incCount,0); s[b].y=CvxClip;
}
void cvx_clip_append(RWStructuredBuffer<uint4> s,uint b,uint target,float3 point)
{
    uint count=s[b+42u].w;
    if(count>=64u) { cvx_fail(s,b,28); return; }
    cvx_put(s,b,target+count,float4(point,0)); ++s[b+42u].w;
}
void cvx_clip_unit(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+42u],regions=s[b+45u],mode=s[b+35u];
    uint source=cursor.x==0u?regions.y:(mode.x==0u?CvxClipA:CvxClipB);
    uint target=mode.x==0u?CvxClipB:CvxClipA; float3 n=cvx_f(s,b,40).xyz;
    float3 origin=cvx_f(s,b,regions.x+cursor.x).xyz;
    float3 edge=cvx_f(s,b,regions.x+(cursor.x+1u)%regions.z).xyz-origin;
    float3 normal=cross(n,edge);
    if(cursor.y==0u) {
        float3 last=cvx_f(s,b,source+cursor.z-1u).xyz;
        cvx_put(s,b,34,float4(last,dot(origin-last,normal)));
        if(mode.y==1u) {
            float3 p0=cvx_f(s,b,source).xyz,pn=cross(cvx_f(s,b,source+2u).xyz-p0,cvx_f(s,b,source+1u).xyz-p0);
            float nn=dot(pn,pn);
            if(nn==0) { cvx_manifold_primary(s,b); return; }
            float3 v1=origin+dot(pn,p0-origin)*pn/nn,v2=origin+edge;
            v2+=dot(pn,p0-v2)*pn/nn;
            cvx_put(s,b,32,float4(v1,0)); cvx_put(s,b,33,float4(v2,0));
        }
    }
    float4 last=cvx_f(s,b,34); float3 p=cvx_f(s,b,source+cursor.y).xyz;
    float distance=dot(origin-p,normal); bool inside=distance<0,previous=last.w<0;
    if(inside!=previous) {
        float3 delta=p-last.xyz; float den=dot(delta,normal);
        if(den!=0 || mode.y==1u) {
            float3 intersection=den!=0?last.xyz+(last.w/den)*delta:last.xyz;
            if(mode.y==1u) {
                float3 v1=cvx_f(s,b,32).xyz,v2=cvx_f(s,b,33).xyz,line=v2-v1;
                float along=dot(intersection-v1,line),ll=dot(line,line);
                if(along<0) intersection=v1; else if(along>ll) intersection=v2;
            }
            cvx_clip_append(s,b,target,intersection);
        } else inside=previous;
    }
    if(inside && mode.y==0u) cvx_clip_append(s,b,target,p);
    cvx_put(s,b,34,float4(p,distance)); ++s[b+42u].y;
    if(cursor.y+1u==cursor.z) {
        uint outCount=s[b+42u].w;
        if((mode.y==0u && outCount<3u) || outCount==0u) { cvx_manifold_primary(s,b); return; }
        ++cursor.x; mode.x^=1u; cursor.y=0; cursor.z=outCount; cursor.w=0;
        s[b+35u]=mode; s[b+42u]=cursor;
        if(mode.y==1u || cursor.x==regions.z) { s[b+36u]=uint4(0); s[b+42u].y=0; s[b].y=CvxProject; }
    }
}
void cvx_project_unit(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+42u],mode=s[b+35u]; uint source=mode.x==0?CvxClipA:CvxClipB;
    uint i=cursor.y;
    if(i==cursor.z) {
        if(s[b+36u].x==0u) cvx_manifold_primary(s,b);
        else { s[b+36u].yz=uint2(0,0); cvx_put(s,b,37,float4(-3.402823466e38f,0,0,0)); s[b+38u]=uint4(0xffffffffu); s[b].y=CvxPrune; }
        return;
    }
    float3 p=cvx_f(s,b,source+i).xyz,normal=cvx_f(s,b,40).xyz;
    float4 origin=cvx_f(s,b,43); float3 plane=cvx_f(s,b,44).xyz;
    float distance=dot(p-origin.xyz,plane)/origin.w;
    if(distance<cvx_f(s,b,46).x) {
        uint count=s[b+36u].x;
        if(count>=64u) { cvx_fail(s,b,29); return; }
        float3 q=p-distance*normal; float3 a=mode.z==0?q:p,v=mode.z==0?p:q;
        cvx_put(s,b,CvxContacts+3u*count,float4(a,0)); cvx_put(s,b,CvxContacts+3u*count+1u,float4(v,0));
        s[b+CvxContacts+3u*count+2u]=uint4(i,0,0,0); ++s[b+36u].x;
    }
    ++s[b+42u].y;
}
float3 cvx_projected_contact(RWStructuredBuffer<uint4> s,uint b,uint i)
{
    float3 p=cvx_f(s,b,CvxContacts+3u*i).xyz-cvx_f(s,b,47).xyz,n=cvx_f(s,b,7).xyz;
    return p-dot(p,n)*n;
}
void cvx_prune_unit(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+36u]; uint count=cursor.x,stage=cursor.y,i=cursor.z;
    if(count<=4u) { s[b+38u]=uint4(0,1,2,3); s[b+36u].w=count; s[b+36u].z=0; s[b].y=CvxPublish; return; }
    uint4 chosen=s[b+38u]; float4 best=cvx_f(s,b,37);
    if(i==count) {
        ++cursor.y; cursor.z=0;
        if(stage==2u) {
            s[b+39u]=uint4(chosen.x,chosen.z,chosen.y,chosen.w); s[b+36u].z=0;
            s[b+36u].w=2u+(chosen.z!=0xffffffffu?1u:0u)+(chosen.w!=0xffffffffu?1u:0u); s[b].y=CvxPublish;
        } else { cvx_put(s,b,37,float4(-3.402823466e38f,0,0,0)); s[b+36u]=cursor; }
        return;
    }
    float3 point=cvx_projected_contact(s,b,i); float value;
    if(stage<2u) {
        float3 delta=stage==0u?point:point-cvx_projected_contact(s,b,chosen.x);
        float3 separation=cvx_f(s,b,CvxContacts+3u*i).xyz-cvx_f(s,b,CvxContacts+3u*i+1u).xyz;
        float minimum=1e-6f/(cvx_f(s,b,3).w*cvx_f(s,b,3).w);
        value=max(minimum,dot(delta,delta))*max(minimum,dot(separation,separation));
        if((stage==0u || i!=chosen.x) && value>best.x) { best.x=value; chosen[stage]=i; }
    } else if(i!=chosen.x && i!=chosen.y) {
        float3 first=cvx_projected_contact(s,b,chosen.x),second=cvx_projected_contact(s,b,chosen.y);
        value=dot(cross(second-first,cvx_f(s,b,7).xyz),point-first);
        if(value<best.y) { best.y=value; chosen.z=i; } else if(value>best.z) { best.z=value; chosen.w=i; }
    }
    ++s[b+36u].z; s[b+38u]=chosen; cvx_put(s,b,37,best);
}
void cvx_publish_unit(RWStructuredBuffer<uint4> s,uint b)
{
    uint4 cursor=s[b+36u]; uint i=cursor.z; uint4 chosen=cursor.x<=4u?s[b+38u]:s[b+39u];
    uint output=i;
    if(cursor.x>4u) {
        // Compact missing side extrema without changing native polygon order.
        output=0;
        for(uint k=0u;k<i;++k) if(chosen[k]!=0xffffffffu) ++output;
    }
    if(chosen[i]!=0xffffffffu) {
        float4 origin=cvx_f(s,b,3);
        cvx_put(s,b,CvxManifoldOut+3u*output,float4(origin.xyz+origin.w*cvx_f(s,b,CvxContacts+3u*chosen[i]).xyz,0));
        cvx_put(s,b,CvxManifoldOut+3u*output+1u,float4(origin.xyz+origin.w*cvx_f(s,b,CvxContacts+3u*chosen[i]+1u).xyz,0));
        s[b+CvxManifoldOut+3u*output+2u]=s[b+CvxContacts+3u*chosen[i]+2u];
    }
    ++s[b+36u].z;
    if(i+1u==(cursor.x<=4u?cursor.w:4u)) s[b].x=CvxSuccess;
}
void cvx_manifold_advance(StructuredBuffer<uint4> geometry,RWStructuredBuffer<uint4> s,uint b,uint units)
{
    for(uint unit=0u;unit<min(units,32u) && s[b].x==CvxPending;++unit) {
        uint phase=s[b].y;
        if(phase==CvxFaceSelect) cvx_face_select(geometry,s,b);
        else if(phase==CvxFaceVertices) cvx_face_vertex(geometry,s,b);
        else if(phase==CvxClipInit) cvx_clip_setup(s,b);
        else if(phase==CvxClip) cvx_clip_unit(s,b);
        else if(phase==CvxProject) cvx_project_unit(s,b);
        else if(phase==CvxPrune) cvx_prune_unit(s,b);
        else if(phase==CvxPublish) cvx_publish_unit(s,b);
        else cvx_fail(s,b,30);
    }
}
#endif
