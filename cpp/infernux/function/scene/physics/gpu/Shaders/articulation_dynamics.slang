// Private full-rank closed-tree dynamics foundation. No contact/owner admission.
#define ARTICULATION_DYNAMICS 1
#include "rigid_math.slang"
#include "../ArticulationDynamicsTypes.h"
[[vk::binding(0,0)]] StructuredBuffer<BodyModel> art_models;
[[vk::binding(1,0)]] RWStructuredBuffer<BodyState> art_states;
[[vk::binding(2,0)]] RWStructuredBuffer<uint4> art_joints;
[[vk::binding(3,0)]] StructuredBuffer<ArticulationLink> art_links;
[[vk::binding(4,0)]] RWStructuredBuffer<BodyState> dyn_roots;
[[vk::binding(5,0)]] RWStructuredBuffer<ArticulationTree> art_trees;
[[vk::binding(6,0)]] RWStructuredBuffer<uint4> art_scratch;
[[vk::binding(7,0)]] RWStructuredBuffer<float4> dyn_coordinates;
[[vk::binding(8,0)]] ConstantBuffer<ArticulationDynamicsWork> art_work;
#include "articulation_aba.slang"
#include "articulation_joint_load.slang"
#include "articulation_kinematics.slang"

// 21/22 actual pre-force w/v; 23/24 LOCAL edge dt*c; 25 staged q/qdot;
// 26 accepted rotor speed/armature; 27..30 staged BodyState (old pose until FK);
// root31 is the active job's real dt/tolerance witness, not a version field.
bool dyn_range(uint first,uint count,uint total) {return first<=total && count<=total-first;}
bool dyn_same(BodyState a,BodyState b) {
    return all(asuint(a.positionCOM)==asuint(b.positionCOM)) && all(asuint(a.rotation)==asuint(b.rotation))
        && all(asuint(a.linearVelocity)==asuint(b.linearVelocity)) && all(asuint(a.angularVelocity)==asuint(b.angularVelocity));
}
bool dyn_matches(BodyState a,BodyState b) {
    float4 rel=normalize(jph_quat_mul(sf_conjugate(normalize(a.rotation)),normalize(b.rotation)));
    return length(a.positionCOM.xyz-b.positionCOM.xyz)<=art_work.integration.y
        && 2.0f*atan2(length(rel.xyz),abs(rel.w))<=art_work.integration.z
        && length(a.linearVelocity.xyz-b.linearVelocity.xyz)<=art_work.integration.w
        && length(a.angularVelocity.xyz-b.angularVelocity.xyz)<=art_work.integration.w;
}
bool dyn_quanta() {return art_work.range.z>0u && art_work.range.z<=ArtPacketMaxUnits && art_work.range.w==0u;}
BodyState dyn_staged(uint b) {
    BodyState s;s.positionCOM=asfloat(art_scratch[b+27u]);s.rotation=asfloat(art_scratch[b+28u]);
    s.linearVelocity=asfloat(art_scratch[b+29u]);s.angularVelocity=asfloat(art_scratch[b+30u]);return s;
}
void dyn_stage(uint b,BodyState s) {
    art_scratch[b+27u]=asuint(s.positionCOM);art_scratch[b+28u]=asuint(s.rotation);
    art_scratch[b+29u]=asuint(s.linearVelocity);art_scratch[b+30u]=asuint(s.angularVelocity);
}
bool dyn_joint(uint tree,uint local,out ArticulationLink link,out JointDirectory d,out SfJoint g,
               out float armature,out float rotor_speed) {
    ArticulationTree t=art_trees[tree];link=art_links[t.range.x+local];d=(JointDirectory)0;g=(SfJoint)0;
    armature=rotor_speed=0;
    uint models,ms,states,ss;art_models.GetDimensions(models,ms);art_states.GetDimensions(states,ss);
    if(link.topology.x>=models || link.topology.x>=states || any(link.identity.zw!=uint2(t.rootMode.z,0))) {
        art_failure(tree,ArtInvalid,100u);return false;
    }
    BodyModel model=art_models[link.topology.x];bool fixedRoot=local==0u && t.rootMode.x==1u;
    if(model.meta.w!=t.rootMode.z || !jph_resident(model)) {art_failure(tree,ArtInvalid,101u);return false;}
    if(fixedRoot ? model.meta.z!=MotionStatic : (model.meta.z!=MotionDynamic || model.mass.y!=0 || model.mass.z!=0)) {
        art_failure(tree,ArtDynamicsUnrepresentable,102u);return false;
    }
    if(!fixedRoot && (!(model.mass.x>0) || !isfinite(model.mass.x) || !all(isfinite(model.invInertiaDiagonal))
        || !all(model.invInertiaDiagonal.xyz>float3(0)) || !sf_quaternion(model.inertiaRotation)
        || !all(isfinite(model.damping)) || any(model.damping<float4(0)))) {
        art_failure(tree,ArtDynamicsUnrepresentable,103u);return false;
    }
    if(local==0u) {
        if(link.topology.x!=t.rootMode.w || any(link.topology.yzw!=uint3(ArtNoParent)) || any(link.identity.xy!=uint2(0))) {
            art_failure(tree,ArtInvalid,104u);return false;
        }
        return true;
    }
    if(link.topology.y>=local) {art_failure(tree,ArtInvalid,105u);return false;}
    ArticulationLink parent=art_links[t.range.x+link.topology.y];
    uint words,stride;art_joints.GetDimensions(words,stride);
    if(words==0u) {art_failure(tree,ArtInvalid,106u);return false;}
    uint4 header=art_joints[0];
    if(header.x>(words-1u)/JointDirectoryWords || header.z>words || header.w!=0u
        || header.y<1u+header.x*JointDirectoryWords || header.y>header.z || link.topology.z>=header.x) {
        art_failure(tree,ArtInvalid,107u);return false;
    }
    d=art_directory(link.topology.z);uint aw=0u,sw=0u;
    if(d.bodies.z==JointFixed) {aw=JointFixedAuthorWords;sw=JointFixedStateWords;}
    else if(d.bodies.z==JointHinge) {aw=JointHingeAuthorWords;sw=JointHingeStateWords;}
    else if(d.bodies.z==JointSlider) {aw=JointSliderAuthorWords;sw=JointSliderStateWords;}
    else {art_failure(tree,ArtDynamicsUnrepresentable,108u);return false;}
    if(any(d.bodies.xyz!=uint3(parent.topology.x,link.topology.x,link.topology.w))
        || (d.bodies.w&JointEnabled)==0u || any(d.identity.xyz!=link.identity.xyz) || d.identity.w!=0u
        || any(d.layout.zw!=uint2(0)) || d.layout.x<1u+header.x*JointDirectoryWords
        || !dyn_range(d.layout.x,aw,header.y) || d.layout.y<header.y || !dyn_range(d.layout.y,sw,header.z)) {
        art_failure(tree,ArtInvalid,109u);return false;
    }
    uint4 flags=art_joints[d.layout.y];
    if(flags.x!=1u || flags.y!=JointStatusSuccess) {art_failure(tree,ArtDynamicsUnrepresentable,110u);return false;}
    if((d.bodies.w&(JointLimitsEnabled|JointMotorEnabled))!=0u || (flags.w&JointPoseDriveActive)!=0u) {
        art_failure(tree,ArtDynamicsUnrepresentable,111u);return false;
    }
    if(d.bodies.z==JointFixed) {
        JointFixedAuthor a;JointFixedState s;art_load(d,a,s);
        g.a1=s.anchor1.xyz*a.scale1.xyz+s.collapsed1.xyz-a.rootCOM1.xyz;
        g.a2=s.anchor2.xyz*a.scale2.xyz+s.collapsed2.xyz-a.rootCOM2.xyz;g.f1=s.frame1;g.f2=s.frame2;
    } else if(d.bodies.z==JointHinge) {
        JointHingeAuthor a;JointHingeState s;art_load(d,a,s);
        if(any(a.passive.xy!=float2(0)) || !(a.passive.z>=0) || !isfinite(a.passive.z)
            || !all(isfinite(s.rotor)) || s.rotor.w!=a.passive.z) {
            art_failure(tree,ArtDynamicsUnrepresentable,112u);return false;
        }
        armature=a.passive.z;rotor_speed=armature>0?s.rotor.x-s.rotor.y/s.rotor.w:0;
        if(!isfinite(rotor_speed)) {art_failure(tree,ArtNumericalFailure,113u);return false;}
        g.a1=s.anchor1.xyz*a.scale1.xyz+s.collapsed1.xyz-a.rootCOM1.xyz;
        g.a2=s.anchor2.xyz*a.scale2.xyz+s.collapsed2.xyz-a.rootCOM2.xyz;g.f1=s.frame1;g.f2=s.frame2;
    } else {
        JointSliderAuthor a;JointSliderState s;art_load(d,a,s);
        g.a1=s.anchor1.xyz*a.scale1.xyz+s.collapsed1.xyz-a.rootCOM1.xyz;
        g.a2=s.anchor2.xyz*a.scale2.xyz+s.collapsed2.xyz-a.rootCOM2.xyz;g.f1=s.frame1;g.f2=s.frame2;
    }
    g.kind=d.bodies.z;
    if(!all(isfinite(g.a1)) || !all(isfinite(g.a2)) || !sf_quaternion(g.f1) || !sf_quaternion(g.f2)) {
        art_failure(tree,ArtInvalid,114u);return false;
    }
    return true;
}
void dyn_preserve(uint tree,uint local) {
    ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    ArticulationLink link;JointDirectory d;SfJoint g;float armature,speed;
    if(!dyn_joint(tree,local,link,d,g,armature,speed))return;
    BodyState s=art_states[link.topology.x];float4 q=dyn_coordinates[t.range.w+local];
    if(!sf_state(s) || !all(isfinite(q)) || any(q.zw!=float2(0))
        || ((local==0u || g.kind==JointFixed) && any(q.xy!=float2(0)))) {
        art_failure(tree,ArtInvalid,115u);return;
    }
    if(local==0u) {
        if(!dyn_same(s,dyn_roots[t.rootMode.y])) {art_failure(tree,ArtDynamicsUnrepresentable,116u);return;}
        if(t.rootMode.x==1u && (any(s.linearVelocity.xyz!=float3(0)) || any(s.angularVelocity.xyz!=float3(0)))) {
            art_failure(tree,ArtDynamicsUnrepresentable,117u);return;
        }
    } else {
        ArticulationLink parent=art_links[t.range.x+link.topology.y];
        BodyState predicted=sf_forward(art_states[parent.topology.x],g,q.xy);
        if(!sf_state(predicted) || !dyn_matches(predicted,s)) {art_failure(tree,ArtDynamicsUnrepresentable,118u);return;}
    }
    for(uint word=0u;word<ArtScratchWords;++word)art_scratch[b+word]=uint4(0);
    art_scratch[b+21u]=asuint(s.angularVelocity);art_scratch[b+22u]=asuint(s.linearVelocity);
    art_scratch[b+26u]=asuint(float4(speed,armature,0,0));dyn_stage(b,s);
    if(local==0u)art_scratch[b+31u]=asuint(art_work.integration);
}
void dyn_initialize(uint tree,uint local) {
    ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    ArticulationLink link;JointDirectory d;SfJoint g;float armature,speed;
    if(!dyn_joint(tree,local,link,d,g,armature,speed))return;
    BodyState s=art_states[link.topology.x];BodyModel m=art_models[link.topology.x];
    if(!sf_state(s) || any(asuint(s.positionCOM)!=art_scratch[b+27u]) || any(asuint(s.rotation)!=art_scratch[b+28u])) {
        art_failure(tree,ArtInvalid,119u);return;
    }
    bool fixedRoot=local==0u && t.rootMode.x==1u;
    // The genuinely Static root does not read reciprocal mass/inertia.
    if(!fixedRoot) {
        float4 frame=normalize(jph_quat_mul(normalize(s.rotation),normalize(m.inertiaRotation)));
        for(uint principal=0u;principal<3u;++principal) {
            float3 axis=float3(0);axis[principal]=1;axis=jph_rotate(frame,axis);
            float inertia=1/m.invInertiaDiagonal[principal];
            for(uint i=0u;i<3u;++i)for(uint j=0u;j<3u;++j)art_mat_set(b,i,j,art_mat(b,i,j)+inertia*axis[i]*axis[j]);
        }
        for(uint i=3u;i<6u;++i)art_mat_set(b,i,i,1/m.mass.x);
        float3 dw=s.angularVelocity.xyz-asfloat(art_scratch[b+21u]).xyz;
        float3 dv=s.linearVelocity.xyz-asfloat(art_scratch[b+22u]).xyz;
        for(uint i=0u;i<6u;++i) {
            float momentum=0;for(uint j=0u;j<6u;++j)momentum+=art_mat(b,i,j)*art_component(dw,dv,j);
            if(!isfinite(momentum)) {art_failure(tree,ArtNumericalFailure,120u);return;}
            art_vec_set(b,9u,i,-momentum);
        }
    }
    if(local==0u)return;
    ArticulationLink pl=art_links[t.range.x+link.topology.y];uint pb=t.range.z+ArtScratchWords*link.topology.y;
    BodyState parent=art_states[pl.topology.x];float3 wp=asfloat(art_scratch[pb+21u]).xyz,wc=asfloat(art_scratch[b+21u]).xyz;
    float3 r=s.positionCOM.xyz-parent.positionCOM.xyz;art_scratch[b+15u]=asuint(float4(r,0));
    float4 q=dyn_coordinates[t.range.w+local];
    float3 axis=jph_rotate(normalize(jph_quat_mul(parent.rotation,g.f1)),g.kind==JointSlider?float3(1,0,0):float3(0,0,1));
    float3 u1=jph_rotate(normalize(parent.rotation),g.a1),u2=jph_rotate(normalize(s.rotation),g.a2);
    float3 sw=float3(0),sv=float3(0),cw=float3(0),cv;
    if(g.kind==JointHinge) {
        sw=axis;sv=-cross(axis,u2);cw=cross(wp,axis*q.y);
        cv=cross(wp,cross(wp,u1))-cross(wc,cross(wc,u2))-cross(cw,u2);
    } else if(g.kind==JointSlider) {
        sv=axis;cv=cross(wp,cross(wp,r))+2.0f*cross(wp,axis)*q.y;
    } else cv=cross(wp,cross(wp,r));
    float dt=art_work.integration.x;
    if(!all(isfinite(cw)) || !all(isfinite(cv))) {art_failure(tree,ArtNumericalFailure,121u);return;}
    for(uint i=0u;i<6u;++i) {
        art_vec_set(b,13u,i,art_component(sw,sv,i));
        art_vec_set(b,23u,i,dt*art_component(cw,cv,i));
    }
    float4 rotor=asfloat(art_scratch[b+26u]);
    if(rotor.y!=armature) {art_failure(tree,ArtInvalid,122u);return;}
    art_scratch[b+16u]=asuint(float4(0,armature*(rotor.x-q.y),armature,0));
}
void dyn_recovered(uint tree,uint local,float delta_qdot) {
    ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    float4 old=dyn_coordinates[t.range.w+local];float speed=old.y+delta_qdot;
    float4 rotor=asfloat(art_scratch[b+26u]);
    float4 candidate=float4(local==0u?0:old.x+art_work.integration.x*speed,local==0u?0:speed,
        delta_qdot,rotor.y*(rotor.x-speed));
    if(!all(isfinite(candidate))) {art_failure(tree,ArtNumericalFailure,123u);return;}
    art_scratch[b+25u]=asuint(candidate);
}
bool dyn_within_cap(BodyModel m,BodyState s) {
    return all(jph_clamp_speed(s.linearVelocity.xyz,m.damping.z)==s.linearVelocity.xyz)
        && all(jph_clamp_speed(s.angularVelocity.xyz,m.damping.w)==s.angularVelocity.xyz);
}
void dyn_candidate(uint tree,uint local) {
    ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    ArticulationLink link;JointDirectory d;SfJoint g;float armature,speed;
    if(!dyn_joint(tree,local,link,d,g,armature,speed))return;
    BodyState current=art_states[link.topology.x],candidate;float4 q=asfloat(art_scratch[b+25u]);
    if(local==0u) {
        current=dyn_roots[t.rootMode.y];
        current.angularVelocity.xyz+=asfloat(art_scratch[b+17u]).xyz;
        current.linearVelocity.xyz+=asfloat(art_scratch[b+18u]).xyz;candidate=current;
        if(t.rootMode.x==0u) {
            candidate.rotation=jph_rotation_step(current.rotation,art_work.integration.x*current.angularVelocity.xyz);
            candidate.positionCOM.xyz+=art_work.integration.x*current.linearVelocity.xyz;
        }
    } else {
        uint pb=t.range.z+ArtScratchWords*link.topology.y;ArticulationLink pl=art_links[t.range.x+link.topology.y];
        BodyState parent=art_states[pl.topology.x];
        parent.angularVelocity=asfloat(art_scratch[pb+21u]);parent.linearVelocity=asfloat(art_scratch[pb+22u]);
        float3 axis=jph_rotate(normalize(jph_quat_mul(parent.rotation,g.f1)),g.kind==JointSlider?float3(1,0,0):float3(0,0,1));
        float3 wr=g.kind==JointHinge?axis*q.y:float3(0);
        current.angularVelocity=float4(parent.angularVelocity.xyz+wr,0);
        current.linearVelocity=float4(parent.linearVelocity.xyz+cross(parent.angularVelocity.xyz,current.positionCOM.xyz-parent.positionCOM.xyz)
            -cross(wr,jph_rotate(normalize(current.rotation),g.a2))+(g.kind==JointSlider?axis*q.y:float3(0)),0);
        candidate=sf_forward(dyn_staged(pb),g,q.xy);
    }
    if(!sf_state(current) || !sf_state(candidate)) {art_failure(tree,ArtNumericalFailure,124u);return;}
    if(!(local==0u && t.rootMode.x==1u) && (!dyn_within_cap(art_models[link.topology.x],current)
        || !dyn_within_cap(art_models[link.topology.x],candidate))) {
        art_failure(tree,ArtDynamicsUnrepresentable,125u);return;
    }
    // Old-pose tangent velocity is separate from the integrated FK twist.
    art_scratch[b+21u]=asuint(current.angularVelocity);art_scratch[b+22u]=asuint(current.linearVelocity);
    dyn_stage(b,candidate);
}
void dyn_commit_velocity(uint tree,uint local) {
    ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    ArticulationLink link=art_links[t.range.x+local];BodyState s=art_states[link.topology.x];
    s.angularVelocity=asfloat(art_scratch[b+21u]);s.linearVelocity=asfloat(art_scratch[b+22u]);art_states[link.topology.x]=s;
}
void dyn_commit_pose(uint tree,uint local) {
    ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    ArticulationLink link=art_links[t.range.x+local];BodyState s=dyn_staged(b);
    art_states[link.topology.x]=s;if(local==0u)dyn_roots[t.rootMode.y]=s;
    float4 q=asfloat(art_scratch[b+25u]);dyn_coordinates[t.range.w+local]=float4(q.xy,0,0);
    if(local!=0u && link.topology.w==JointHinge) {
        JointDirectory d=art_directory(link.topology.z);JointHingeAuthor a;JointHingeState runtime;art_load(d,a,runtime);
        float4 rotor=asfloat(art_scratch[b+26u]);runtime.rotor=float4(rotor.x,q.w,rotor.x,rotor.y);
        // This slice has no motor/friction/damping; preserve actual rotor speed
        // in the existing x-y/a contract. Representation changes still require
        // the owner's existing preserve-speed warm reset before native reuse.
        runtime.axisImpulse.yzw=float3(0,0,q.w);uint p=d.layout.y;
#define DYN_STORE_F(name) art_joints[p++]=asuint(runtime.name);
#define DYN_STORE_U(name) art_joints[p++]=runtime.name;
        JPH_HINGE_STATE_FIELDS(DYN_STORE_F,DYN_STORE_U)
#undef DYN_STORE_F
#undef DYN_STORE_U
    }
}
bool dyn_tree(uint id,out uint tree) {
    tree=0;if(id>=art_work.range.y || id>0xffffffffu-art_work.range.x)return false;
    tree=art_work.range.x+id;uint count,stride;art_trees.GetDimensions(count,stride);return tree<count;
}
void dyn_begin(uint tree) {
    ArticulationTree t=art_trees[tree];t.progress=uint4(ArtPending,ArtDynPreserve,0,0);art_trees[tree]=t;
    uint links,ls,scratch,ss,coordinates,cs,roots,rs;art_links.GetDimensions(links,ls);art_scratch.GetDimensions(scratch,ss);
    dyn_coordinates.GetDimensions(coordinates,cs);dyn_roots.GetDimensions(roots,rs);
    if(!dyn_quanta() || !all(isfinite(art_work.integration)) || any(art_work.integration<=float4(0))
        || t.range.y==0u || !dyn_range(t.range.x,t.range.y,links) || t.range.z>scratch
        || t.range.y>(scratch-t.range.z)/ArtScratchWords || !dyn_range(t.range.w,t.range.y,coordinates)
        || t.rootMode.x>1u || t.rootMode.y>=roots)art_failure(tree,ArtInvalid,126u);
}
void dyn_transition(uint tree,uint status,uint from,uint to) {
    ArticulationTree t=art_trees[tree];
    if(t.progress.x!=status || t.progress.y!=from || t.progress.z!=t.range.y || !dyn_quanta()
        || any(art_scratch[t.range.z+31u]!=asuint(art_work.integration))) {art_failure(tree,ArtInvalid,127u);return;}
    t.progress=uint4(ArtPending,to,0,0);art_trees[tree]=t;
}
void dyn_advance(uint tree,uint entry) {
    if(!dyn_quanta()) {art_failure(tree,ArtInvalid,128u);return;}
    for(uint unit=0u;unit<art_work.range.z;++unit) {
        ArticulationTree t=art_trees[tree];if(t.progress.x!=ArtPending)return;
        uint phase=t.progress.y,local=t.progress.z;
        bool allowed=entry==0u?phase==ArtDynPreserve:entry==1u?(phase>=ArtDynInitialize && phase<=ArtDynCandidate):
            entry==2u?phase==ArtDynCommitVelocity:phase==ArtDynCommitPose;
        if(!allowed || local>=t.range.y) {
            art_failure(tree,ArtInvalid,129u);return;
        }
        if(!(phase==ArtDynPreserve && local==0u)) {
            if(any(art_scratch[t.range.z+31u]!=asuint(art_work.integration))) {art_failure(tree,ArtInvalid,129u);return;}
        }
        if(phase==ArtDynPreserve) {
            dyn_preserve(tree,local);if(art_trees[tree].progress.x!=ArtPending)return;
            if(++local==t.range.y)phase=ArtDynAwaitForces;
        } else if(phase==ArtDynInitialize) {
            dyn_initialize(tree,local);if(art_trees[tree].progress.x!=ArtPending)return;
            if(++local==t.range.y) {phase=t.range.y==1u?ArtDynRoot:ArtDynBackward;local=t.range.y-1u;}
        } else if(phase==ArtDynBackward) {
            art_backward_link(tree,local);if(art_trees[tree].progress.x!=ArtPending)return;
            if(--local==0u)phase=ArtDynRoot;
        } else if(phase==ArtDynRoot) {
            art_root_solve(tree);if(art_trees[tree].progress.x!=ArtPending)return;phase=ArtDynForward;local=0u;
        } else if(phase==ArtDynForward) {
            art_forward_link(tree,local);if(art_trees[tree].progress.x!=ArtPending)return;
            if(++local==t.range.y) {phase=ArtDynCandidate;local=0u;}
        } else if(phase==ArtDynCandidate) {
            dyn_candidate(tree,local);if(art_trees[tree].progress.x!=ArtPending)return;
            if(++local==t.range.y) {phase=ArtDynReady;t.progress.x=ArtSuccess;}
        } else if(phase==ArtDynCommitVelocity) {
            dyn_commit_velocity(tree,local);if(++local==t.range.y)phase=ArtDynAwaitIntegration;
        } else if(phase==ArtDynCommitPose) {
            dyn_commit_pose(tree,local);if(++local==t.range.y) {phase=ArtDynDone;t.progress.x=ArtSuccess;}
        }
        t.progress.yz=uint2(phase,local);art_trees[tree]=t;
        if(phase==ArtDynAwaitForces || phase==ArtDynReady || phase==ArtDynAwaitIntegration || phase==ArtDynDone)return;
    }
}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_begin(t);}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_preserve(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_advance(t,0u);}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_solve_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_transition(t,ArtPending,ArtDynAwaitForces,ArtDynInitialize);}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_solve(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_advance(t,1u);}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_velocity_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_transition(t,ArtSuccess,ArtDynReady,ArtDynCommitVelocity);}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_velocity(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_advance(t,2u);}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_pose_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_transition(t,ArtPending,ArtDynAwaitIntegration,ArtDynCommitPose);}
[shader("compute")][numthreads(64,1,1)]
void articulation_dynamics_pose(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t))dyn_advance(t,3u);}
