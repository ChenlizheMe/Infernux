// Private reduced-state capture and FK. Each real work unit is one link.
#include "rigid_math.slang"
#include "../StateFkTypes.h"
[[vk::binding(0,0)]] StructuredBuffer<BodyModel> sf_models;
[[vk::binding(1,0)]] StructuredBuffer<BodyState> sf_accepted;
[[vk::binding(2,0)]] StructuredBuffer<uint4> sf_joints;
[[vk::binding(3,0)]] StructuredBuffer<ArticulationLink> sf_links;
[[vk::binding(4,0)]] RWStructuredBuffer<BodyState> sf_roots;
[[vk::binding(5,0)]] RWStructuredBuffer<float4> sf_coordinates;
[[vk::binding(6,0)]] RWStructuredBuffer<StateFkTree> sf_trees;
[[vk::binding(7,0)]] RWStructuredBuffer<BodyState> sf_output;
[[vk::binding(8,0)]] ConstantBuffer<StateFkWork> sf_work;

#include "articulation_kinematics.slang"
void sf_fail(uint tree,uint status,uint reason) {
    StateFkTree t=sf_trees[tree];t.progress.x=status;t.progress.w=reason;sf_trees[tree]=t;
}
bool sf_range(uint first,uint count,uint total) { return first<=total && count<=total-first; }
void sf_load(JointDirectory d,out JointFixedAuthor a,out JointFixedState s) {
    a.bodies=d.bodies;a.identity=d.identity;uint p=d.layout.x;
#define SF_F(name) a.name=asfloat(sf_joints[p++]);
#define SF_U(name) a.name=sf_joints[p++];
    JPH_FIXED_AUTHOR_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
    p=d.layout.y;
#define SF_F(name) s.name=asfloat(sf_joints[p++]);
#define SF_U(name) s.name=sf_joints[p++];
    JPH_FIXED_STATE_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
}
void sf_load(JointDirectory d,out JointHingeAuthor a,out JointHingeState s) {
    a.bodies=d.bodies;a.identity=d.identity;uint p=d.layout.x;
#define SF_F(name) a.name=asfloat(sf_joints[p++]);
#define SF_U(name) a.name=sf_joints[p++];
    JPH_HINGE_AUTHOR_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
    p=d.layout.y;
#define SF_F(name) s.name=asfloat(sf_joints[p++]);
#define SF_U(name) s.name=sf_joints[p++];
    JPH_HINGE_STATE_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
}
void sf_load(JointDirectory d,out JointSliderAuthor a,out JointSliderState s) {
    a.bodies=d.bodies;a.identity=d.identity;uint p=d.layout.x;
#define SF_F(name) a.name=asfloat(sf_joints[p++]);
#define SF_U(name) a.name=sf_joints[p++];
    JPH_SLIDER_AUTHOR_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
    p=d.layout.y;
#define SF_F(name) s.name=asfloat(sf_joints[p++]);
#define SF_U(name) s.name=sf_joints[p++];
    JPH_SLIDER_STATE_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
}

bool sf_joint(uint tree,uint local,out ArticulationLink link,out SfJoint joint) {
    StateFkTree t=sf_trees[tree];link=sf_links[t.range.x+local];joint=(SfJoint)0;
    uint bodies,bs;sf_models.GetDimensions(bodies,bs);
    if(link.topology.x>=bodies || any(link.identity.zw!=uint2(t.owner.x,0))) {sf_fail(tree,StateFkInvalid,10u);return false;}
    BodyModel model=sf_models[link.topology.x];
    if(model.meta.w!=t.owner.x || !jph_resident(model)) {sf_fail(tree,StateFkInvalid,11u);return false;}
    if(local==0u) {
        if(link.topology.x!=t.owner.y || any(link.topology.yzw!=uint3(ArtNoParent))
           || any(link.identity.xy!=uint2(0))) {sf_fail(tree,StateFkInvalid,12u);return false;}
        return true;
    }
    if(link.topology.y>=local) {sf_fail(tree,StateFkInvalid,13u);return false;}
    ArticulationLink parent=sf_links[t.range.x+link.topology.y];
    uint words,stride;sf_joints.GetDimensions(words,stride);
    if(words==0u) {sf_fail(tree,StateFkInvalid,14u);return false;}
    uint4 header=sf_joints[0];
    if(header.x>(words-1u)/JointDirectoryWords || header.z>words || header.w!=0u
       || header.y<1u+header.x*JointDirectoryWords || header.y>header.z || link.topology.z>=header.x) {
        sf_fail(tree,StateFkInvalid,15u);return false;
    }
    uint base=1u+JointDirectoryWords*link.topology.z;
    JointDirectory d;d.bodies=sf_joints[base];d.identity=sf_joints[base+1u];d.layout=sf_joints[base+2u];
    uint aw=0,sw=0;
    if(d.bodies.z==JointFixed) {aw=JointFixedAuthorWords;sw=JointFixedStateWords;}
    else if(d.bodies.z==JointHinge) {aw=JointHingeAuthorWords;sw=JointHingeStateWords;}
    else if(d.bodies.z==JointSlider) {aw=JointSliderAuthorWords;sw=JointSliderStateWords;}
    else {sf_fail(tree,StateFkInvalid,16u);return false;}
    if(any(d.bodies.xyz!=uint3(parent.topology.x,link.topology.x,link.topology.w))
       || (d.bodies.w&JointEnabled)==0u || any(d.identity.xyz!=link.identity.xyz) || d.identity.w!=0u
       || any(d.layout.zw!=uint2(0)) || d.layout.x<1u+header.x*JointDirectoryWords
       || !sf_range(d.layout.x,aw,header.y) || d.layout.y<header.y || !sf_range(d.layout.y,sw,header.z)) {
        sf_fail(tree,StateFkInvalid,17u);return false;
    }
    uint4 flags=sf_joints[d.layout.y];
    if(flags.x!=1u || flags.y!=JointStatusSuccess) {sf_fail(tree,StateFkInvalid,18u);return false;}
    if(d.bodies.z==JointFixed) {
        JointFixedAuthor a;JointFixedState s;sf_load(d,a,s);
        joint.a1=s.anchor1.xyz*a.scale1.xyz+s.collapsed1.xyz-a.rootCOM1.xyz;
        joint.a2=s.anchor2.xyz*a.scale2.xyz+s.collapsed2.xyz-a.rootCOM2.xyz;joint.f1=s.frame1;joint.f2=s.frame2;
    } else if(d.bodies.z==JointHinge) {
        JointHingeAuthor a;JointHingeState s;sf_load(d,a,s);
        joint.a1=s.anchor1.xyz*a.scale1.xyz+s.collapsed1.xyz-a.rootCOM1.xyz;
        joint.a2=s.anchor2.xyz*a.scale2.xyz+s.collapsed2.xyz-a.rootCOM2.xyz;joint.f1=s.frame1;joint.f2=s.frame2;
    } else {
        JointSliderAuthor a;JointSliderState s;sf_load(d,a,s);
        joint.a1=s.anchor1.xyz*a.scale1.xyz+s.collapsed1.xyz-a.rootCOM1.xyz;
        joint.a2=s.anchor2.xyz*a.scale2.xyz+s.collapsed2.xyz-a.rootCOM2.xyz;joint.f1=s.frame1;joint.f2=s.frame2;
    }
    joint.kind=d.bodies.z;
    if(!all(isfinite(joint.a1)) || !all(isfinite(joint.a2)) || !sf_quaternion(joint.f1) || !sf_quaternion(joint.f2)) {
        sf_fail(tree,StateFkInvalid,19u);return false;
    }
    return true;
}

void sf_capture_link(uint tree,uint local) {
    StateFkTree t=sf_trees[tree];ArticulationLink link;SfJoint joint;if(!sf_joint(tree,local,link,joint))return;
    uint states,stride;sf_accepted.GetDimensions(states,stride);
    if(link.topology.x>=states) {sf_fail(tree,StateFkInvalid,20u);return;}
    BodyState child=sf_accepted[link.topology.x];
    if(!sf_state(child)) {sf_fail(tree,StateFkInvalid,21u);return;}
    if(local==0u) {sf_roots[tree]=child;sf_coordinates[t.range.z]=float4(0);return;}
    ArticulationLink pl=sf_links[t.range.x+link.topology.y];
    if(pl.topology.x>=states) {sf_fail(tree,StateFkInvalid,22u);return;}
    BodyState parent=sf_accepted[pl.topology.x];if(!sf_state(parent)){sf_fail(tree,StateFkInvalid,23u);return;}
    float4 frame=normalize(jph_quat_mul(parent.rotation,joint.f1));
    float3 axis=jph_rotate(frame,joint.kind==JointSlider?float3(1,0,0):float3(0,0,1));
    float2 coordinate=float2(0);
    if(joint.kind==JointHinge) {
        float4 relative=normalize(jph_quat_mul(jph_quat_mul(sf_conjugate(frame),child.rotation),joint.f2));
        if(relative.w<0 || (relative.w==0 && relative.z<0)) relative=-relative;
        if(relative.w==0 && relative.z==0) {sf_fail(tree,StateFkUnrepresentable,25u);return;}
        coordinate.x=2.0f*atan2(relative.z,relative.w);
        coordinate.y=dot(child.angularVelocity.xyz-parent.angularVelocity.xyz,axis);
    } else if(joint.kind==JointSlider) {
        float3 delta=child.positionCOM.xyz+jph_rotate(normalize(child.rotation),joint.a2)
            -parent.positionCOM.xyz-jph_rotate(normalize(parent.rotation),joint.a1);
        coordinate.x=dot(delta,axis);
        coordinate.y=dot(child.linearVelocity.xyz-parent.linearVelocity.xyz
            -cross(parent.angularVelocity.xyz,child.positionCOM.xyz-parent.positionCOM.xyz),axis);
    }
    BodyState predicted=sf_forward(parent,joint,coordinate);
    if(!sf_state(predicted) || !all(isfinite(coordinate))) {sf_fail(tree,StateFkNumericalFailure,24u);return;}
    float4 residual=normalize(jph_quat_mul(sf_conjugate(predicted.rotation),child.rotation));
    float angle=2.0f*atan2(length(residual.xyz),abs(residual.w));
    if(length(predicted.positionCOM.xyz-child.positionCOM.xyz)>sf_work.tolerance.x || angle>sf_work.tolerance.y
       || length(predicted.linearVelocity.xyz-child.linearVelocity.xyz)>sf_work.tolerance.z
       || length(predicted.angularVelocity.xyz-child.angularVelocity.xyz)>sf_work.tolerance.w) {
        sf_fail(tree,StateFkUnrepresentable,25u);return;
    }
    sf_coordinates[t.range.z+local]=float4(coordinate,0,0);
}
void sf_forward_link(uint tree,uint local) {
    StateFkTree t=sf_trees[tree];ArticulationLink link;SfJoint joint;if(!sf_joint(tree,local,link,joint))return;
    float4 coordinate=sf_coordinates[t.range.z+local];
    if(!all(isfinite(coordinate)) || any(coordinate.zw!=float2(0))
       || ((local==0u || joint.kind==JointFixed) && any(coordinate.xy!=float2(0)))) {
        sf_fail(tree,StateFkInvalid,30u);return;
    }
    BodyState state;
    if(local==0u) state=sf_roots[tree];
    else state=sf_forward(sf_output[t.range.w+link.topology.y],joint,coordinate.xy);
    if(!sf_state(state)) {sf_fail(tree,StateFkNumericalFailure,31u);return;}
    // Root is copied verbatim, preserving its real pose and quaternion sign.
    sf_output[t.range.w+local]=state;
}
void sf_begin(uint tree,uint phase) {
    uint count,stride;sf_trees.GetDimensions(count,stride);if(tree>=count)return;
    StateFkTree t=sf_trees[tree];t.progress=uint4(StateFkPending,phase,0,0);sf_trees[tree]=t;
    uint links,ls,coordinates,cs,roots,rs;sf_links.GetDimensions(links,ls);sf_coordinates.GetDimensions(coordinates,cs);sf_roots.GetDimensions(roots,rs);
    if(t.range.y==0 || !sf_range(t.range.x,t.range.y,links) || !sf_range(t.range.z,t.range.y,coordinates)
       || tree>=roots || any(t.owner.zw!=uint2(0)) || sf_work.range.z==0 || sf_work.range.z>ArtPacketMaxUnits
       || sf_work.range.w!=0 || !all(isfinite(sf_work.tolerance)) || any(sf_work.tolerance<=float4(0)))
        sf_fail(tree,StateFkInvalid,1u);
}
bool sf_next(uint tree,uint phase,out StateFkTree t) {
    t=(StateFkTree)0;
    uint count,stride;sf_trees.GetDimensions(count,stride);if(tree>=count)return false;
    if(sf_work.range.z==0 || sf_work.range.z>ArtPacketMaxUnits || sf_work.range.w!=0) {sf_fail(tree,StateFkInvalid,3u);return false;}
    t=sf_trees[tree];if(t.progress.x!=StateFkPending)return false;
    if(t.progress.y!=phase || t.progress.z>=t.range.y) {sf_fail(tree,StateFkInvalid,4u);return false;}
    return true;
}
void sf_finish_link(uint tree,inout StateFkTree t) {
    ++t.progress.z;if(t.progress.z==t.range.y)t.progress.xy=uint2(StateFkSuccess,StateFkDone);
    sf_trees[tree]=t;
}
[shader("compute")][numthreads(64,1,1)]
void articulation_capture_begin(uint3 id:SV_DispatchThreadID) {
    if(id.x<sf_work.range.y && id.x<=0xffffffffu-sf_work.range.x)sf_begin(sf_work.range.x+id.x,StateFkCapturePhase);
}
[shader("compute")][numthreads(64,1,1)]
void articulation_capture_advance(uint3 id:SV_DispatchThreadID) {
    if(id.x>=sf_work.range.y || id.x>0xffffffffu-sf_work.range.x)return;uint tree=sf_work.range.x+id.x;
    for(uint unit=0u;unit<sf_work.range.z;++unit) {
        StateFkTree t;if(!sf_next(tree,StateFkCapturePhase,t))return;
        sf_capture_link(tree,t.progress.z);if(sf_trees[tree].progress.x!=StateFkPending)return;
        sf_finish_link(tree,t);
    }
}
[shader("compute")][numthreads(64,1,1)]
void articulation_fk_begin(uint3 id:SV_DispatchThreadID) {
    if(id.x>=sf_work.range.y || id.x>0xffffffffu-sf_work.range.x)return;uint tree=sf_work.range.x+id.x;
    sf_begin(tree,StateFkForwardPhase);
    uint count,stride;sf_trees.GetDimensions(count,stride);if(tree>=count)return;
    StateFkTree t=sf_trees[tree];uint outputs,os;sf_output.GetDimensions(outputs,os);
    if(!sf_range(t.range.w,t.range.y,outputs))sf_fail(tree,StateFkInvalid,2u);
}
[shader("compute")][numthreads(64,1,1)]
void articulation_fk_advance(uint3 id:SV_DispatchThreadID) {
    if(id.x>=sf_work.range.y || id.x>0xffffffffu-sf_work.range.x)return;uint tree=sf_work.range.x+id.x;
    for(uint unit=0u;unit<sf_work.range.z;++unit) {
        StateFkTree t;if(!sf_next(tree,StateFkForwardPhase,t))return;
        sf_forward_link(tree,t.progress.z);if(sf_trees[tree].progress.x!=StateFkPending)return;
        sf_finish_link(tree,t);
    }
}
