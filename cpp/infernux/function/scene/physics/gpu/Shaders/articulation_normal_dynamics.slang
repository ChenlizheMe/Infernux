// Numerical owner entries around the byte-for-byte frozen dynamics module.
#include "articulation_dynamics.slang"
#include "rigid_joints.slang"
#include "../ArticulationIntegrationTypes.h"

uint nd_control(uint byteAddress) {return art_scratch[byteAddress>>4u][(byteAddress>>2u)&3u];}
uint nd_flags(uint tree) {return nd_control(art_scratch[1u].w+4u*tree);}
bool nd_captured(uint tree) {return (nd_flags(tree)&ArtControlCaptured)!=0u;}
bool nd_eligible(uint tree) {return (nd_flags(tree)&(ArtControlCaptured|ArtControlEligible))==(ArtControlCaptured|ArtControlEligible);}
bool nd_candidate(uint tree) {return (nd_flags(tree)&ArtControlCandidate)!=0u;}
void nd_store(JointDirectory d,JointFixedState s) {
    uint p=d.layout.y;
#define ND_F(name) art_joints[p++]=asuint(s.name);
#define ND_U(name) art_joints[p++]=s.name;
    JPH_FIXED_STATE_FIELDS(ND_F,ND_U)
#undef ND_F
#undef ND_U
}
void nd_store(JointDirectory d,JointHingeState s) {
    uint p=d.layout.y;
#define ND_F(name) art_joints[p++]=asuint(s.name);
#define ND_U(name) art_joints[p++]=s.name;
    JPH_HINGE_STATE_FIELDS(ND_F,ND_U)
#undef ND_F
#undef ND_U
}
void nd_store(JointDirectory d,JointSliderState s) {
    uint p=d.layout.y;
#define ND_F(name) art_joints[p++]=asuint(s.name);
#define ND_U(name) art_joints[p++]=s.name;
    JPH_SLIDER_STATE_FIELDS(ND_F,ND_U)
#undef ND_F
#undef ND_U
}
float2 nd_basis(float3 a,float3 b,float3 value,out bool valid) {
    float aa=dot(a,a),ab=dot(a,b),bb=dot(b,b),det=aa*bb-ab*ab;
    valid=det>0.0f&&isfinite(det);
    if(!valid)return float2(0);
    float2 rhs=float2(dot(a,value),dot(b,value));
    float2 answer=float2(bb*rhs.x-ab*rhs.y,aa*rhs.y-ab*rhs.x)/det;
    valid=all(isfinite(answer));return answer;
}
void nd_reaction(uint tree,uint local) {
    ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    if(local==0u)return; // free-root LDLT overwrote its matrix; no incoming joint.
    float3 moment=float3(0),point=float3(0);
    for(uint i=0u;i<6u;++i) {
        float value=art_vec(b,9u,i);
        for(uint j=0u;j<6u;++j) {
            float delta=art_component(asfloat(art_scratch[b+17u]).xyz,asfloat(art_scratch[b+18u]).xyz,j);
            value+=art_mat(b,i,j)*delta;
        }
        if(i<3u)moment[i]=value;else point[i-3u]=value;
    }
    // Ibar*deltaV+pbar is the actual incoming impulse, including recursively
    // accumulated kappa and armature. Old-pose tangent minus Vfree is NOT it.
    ArticulationLink link;JointDirectory d;SfJoint g;float armature,speed;
    if(!dyn_joint(tree,local,link,d,g,armature,speed))return;
    ArticulationLink parent=art_links[t.range.x+link.topology.y];
    BodyState s1=art_states[parent.topology.x],s2=art_states[link.topology.x];
    // Match the actual native warm helper's arm interpretation exactly. Frozen
    // FK uses local unit quaternions but must retain original stored root bits.
    float3 r2=jph_rotate(s2.rotation,g.a2);
    float3 angular=moment-cross(r2,point),storedAngular=angular,storedPoint=point;
    float4 storedAxis=float4(0);bool valid=all(isfinite(moment))&&all(isfinite(point));
    float3 reconstructedMoment=moment,reconstructedPoint=point;
    if(g.kind==JointHinge) {
        JointHingeAuthor a;JointHingeState s;art_load(d,a,s);
        float3 j0,j1;float2 error;jph_joint_swing_basis(s1,s2,s,j0,j1,error);
        float3 axis=jph_joint_axis(s1,s);float rotorImpulse=asfloat(art_scratch[b+25u]).w;
        bool basisValid;float2 coefficients=nd_basis(j0,j1,angular-axis*rotorImpulse,basisValid);
        valid=valid&&basisValid;storedAngular=float3(coefficients,0);storedAxis.w=rotorImpulse;
        reconstructedMoment=j0*coefficients.x+j1*coefficients.y+axis*rotorImpulse+cross(r2,point);
    } else if(g.kind==JointSlider) {
        JointSliderAuthor a;JointSliderState s;art_load(d,a,s);float3 axis,n0,n1;jph_slider_basis(s1,s,axis,n0,n1);
        bool basisValid;float2 coefficients=nd_basis(n0,n1,point,basisValid);valid=valid&&basisValid;
        storedPoint=float3(coefficients,0);reconstructedPoint=n0*coefficients.x+n1*coefficients.y;
        reconstructedMoment=angular+cross(r2,reconstructedPoint);
    }
    // Frozen finite SI conversion bound, declared before any new device run.
    // Refusal changes no live State/root/q/history. No hidden reaction projection.
    if(!valid||!all(isfinite(storedAngular))||!all(isfinite(storedAxis))
        ||length(reconstructedPoint-point)>5e-5f||length(reconstructedMoment-moment)>5e-5f) {
        art_failure(tree,ArtDynamicsUnrepresentable,130u);return;
    }
    // All matrix consumers are finished. A later link needs only its own matrix.
    art_scratch[b]=asuint(float4(storedPoint,0));
    art_scratch[b+1u]=asuint(float4(storedAngular,0));art_scratch[b+2u]=asuint(storedAxis);
}
void nd_history(uint tree,uint local) {
    if(local==0u)return;ArticulationTree t=art_trees[tree];uint b=t.range.z+ArtScratchWords*local;
    ArticulationLink link=art_links[t.range.x+local];JointDirectory d=art_directory(link.topology.z);
    float4 point=asfloat(art_scratch[b]),angular=asfloat(art_scratch[b+1u]),axis=asfloat(art_scratch[b+2u]);
    if(link.topology.w==JointFixed) {
        JointFixedAuthor a;JointFixedState s;art_load(d,a,s);s.pointImpulse=point;s.angularImpulse=angular;nd_store(d,s);
    } else if(link.topology.w==JointHinge) {
        JointHingeAuthor a;JointHingeState s;art_load(d,a,s);s.pointImpulse=point;s.angularImpulse=angular;s.axisImpulse=axis;
        float4 old=asfloat(art_scratch[b+26u]);s.rotor=float4(old.x,axis.w,old.x,old.y);nd_store(d,s);
    } else {
        JointSliderAuthor a;JointSliderState s;art_load(d,a,s);s.pointImpulse=point;s.angularImpulse=angular;s.axisImpulse=float4(0);nd_store(d,s);
    }
}
void nd_owner_begin(uint tree,uint phase) {
    ArticulationTree t=art_trees[tree];
    if(t.progress.x!=ArtSuccess||t.progress.y!=ArtDynReady)return;
    t.progress=uint4(ArtPending,phase,0,0);art_trees[tree]=t;
}
bool nd_owner_finish_link(uint tree,inout ArticulationTree t) {
    if(++t.progress.z==t.range.y) {
        t.progress.xy=uint2(ArtSuccess,ArtDynReady);art_trees[tree]=t;return true;
    }
    return false;
}
void nd_reaction_advance(uint tree) {
    ArticulationTree t=art_trees[tree];if(t.progress.x!=ArtPending||t.progress.y!=ArtNormalReaction)return;
    for(uint unit=0u;unit<art_work.range.z;++unit) {
        nd_reaction(tree,t.progress.z);
        if(art_trees[tree].progress.x!=ArtPending)return;
        if(nd_owner_finish_link(tree,t))return;
    }
    art_trees[tree]=t;
}
void nd_history_advance(uint tree) {
    ArticulationTree t=art_trees[tree];if(t.progress.x!=ArtPending||t.progress.y!=ArtNormalHistory)return;
    for(uint unit=0u;unit<art_work.range.z;++unit) {
        nd_history(tree,t.progress.z);
        if(nd_owner_finish_link(tree,t))return;
    }
    art_trees[tree]=t;
}
[shader("compute")][numthreads(64,1,1)]
void normal_dynamics_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t)&&nd_captured(t))dyn_begin(t);}
[shader("compute")][numthreads(64,1,1)]
void normal_dynamics_preserve(uint3 id:SV_DispatchThreadID) {
    uint t;if(!dyn_tree(id.x,t)||!nd_captured(t))return;
    ArticulationTree tree=art_trees[t];
    if(tree.progress.x!=ArtPending||tree.progress.y!=ArtDynPreserve)return;
    for(uint unit=0u;unit<art_work.range.z;++unit) {
        uint local=tree.progress.z;dyn_preserve(t,local);
        if(art_trees[t].progress.x!=ArtPending)return;
        ArticulationLink link=art_links[tree.range.x+local];
        if(local!=0u&&link.topology.w==JointHinge) {
            uint4 selected=art_scratch[0u],storage=art_scratch[2u];
            float4 witness=asfloat(art_scratch[(storage.y>>4u)+link.topology.z-selected.z]);
            uint b=tree.range.z+ArtScratchWords*local;
            if(witness.z!=1.0f||witness.y!=asfloat(art_scratch[b+26u]).y) {
                art_failure(t,ArtDynamicsUnrepresentable,131u);return;
            }
            art_scratch[b+26u]=asuint(float4(witness.xy,0,0));
        }
        if(++tree.progress.z==tree.range.y) {
            tree.progress.y=ArtDynAwaitForces;art_trees[t]=tree;return;
        }
    }
    art_trees[t]=tree;
}
[shader("compute")][numthreads(64,1,1)]
void normal_dynamics_solve_begin(uint3 id:SV_DispatchThreadID) {
    uint t;if(!dyn_tree(id.x,t)||!nd_eligible(t))return;
    ArticulationTree tree=art_trees[t];
    if(tree.progress.x!=ArtPending||tree.progress.y!=ArtDynAwaitForces)return;
    dyn_transition(t,ArtPending,ArtDynAwaitForces,ArtDynInitialize);
}
[shader("compute")][numthreads(64,1,1)]
void normal_dynamics_solve(uint3 id:SV_DispatchThreadID) {
    uint t;if(!dyn_tree(id.x,t)||!nd_eligible(t))return;
    if(art_trees[t].progress.y<ArtDynInitialize||art_trees[t].progress.y>ArtDynCandidate)return;dyn_advance(t,1u);
}
[shader("compute")][numthreads(64,1,1)]
void normal_reaction_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t)&&nd_eligible(t))nd_owner_begin(t,ArtNormalReaction);}
[shader("compute")][numthreads(64,1,1)]
void normal_reaction_advance(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t)&&nd_eligible(t))nd_reaction_advance(t);}
[shader("compute")][numthreads(64,1,1)]
void normal_history_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t)&&nd_candidate(t))nd_owner_begin(t,ArtNormalHistory);}
[shader("compute")][numthreads(64,1,1)]
void normal_history_advance(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t)&&nd_candidate(t))nd_history_advance(t);}
[shader("compute")][numthreads(64,1,1)]
void normal_velocity_begin(uint3 id:SV_DispatchThreadID) {uint t;if(dyn_tree(id.x,t)&&nd_candidate(t))dyn_transition(t,ArtSuccess,ArtDynReady,ArtDynCommitVelocity);}
[shader("compute")][numthreads(64,1,1)]
void normal_velocity_advance(uint3 id:SV_DispatchThreadID) {
    uint t;if(!dyn_tree(id.x,t)||!nd_candidate(t))return;
    if(art_trees[t].progress.y==ArtDynAwaitIntegration)return;dyn_advance(t,2u);
}
[shader("compute")][numthreads(64,1,1)]
void normal_pose_begin(uint3 id:SV_DispatchThreadID) {
    uint t;if(dyn_tree(id.x,t)&&(nd_flags(t)&ArtControlAdmitted)!=0u)dyn_transition(t,ArtPending,ArtDynAwaitIntegration,ArtDynCommitPose);
}
[shader("compute")][numthreads(64,1,1)]
void normal_pose_advance(uint3 id:SV_DispatchThreadID) {
    uint t;if(!dyn_tree(id.x,t)||(nd_flags(t)&ArtControlAdmitted)==0u)return;
    if(art_trees[t].progress.y==ArtDynDone)return;dyn_advance(t,3u);
}
