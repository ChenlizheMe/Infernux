// Included by rigid_step after native selection and joint/schedule declarations.
#include "../ArticulationIntegrationTypes.h"
#include "../ArticulationDynamicsTypes.h"
#include "../StateFkTypes.h"
ConstantBuffer<ArticulationClassifyWork> articulation_params;
RWStructuredBuffer<Atomic<uint>> articulation_words;
StructuredBuffer<QueryRecordWord> articulation_control;
StructuredBuffer<ArticulationTree> articulation_trees;
StructuredBuffer<StateFkTree> articulation_capture_trees;
StructuredBuffer<QueryRecordWord> articulation_previous_control;
StructuredBuffer<uint4> articulation_transitions;
RWStructuredBuffer<float4> articulation_transition_witness;
RWStructuredBuffer<Atomic<uint>> articulation_summary;
RWStructuredBuffer<uint4> articulation_retirement;

uint ac_load(uint byteAddress) { return articulation_words[byteAddress>>2u].load(); }
void ac_store(uint byteAddress,uint value) { articulation_words[byteAddress>>2u].store(value); }
uint4 ac_header(uint word) {
    uint b=word*16u;return uint4(ac_load(b),ac_load(b+4u),ac_load(b+8u),ac_load(b+12u));
}
bool ac_contains(uint index,uint first,uint count) { return index>=first && index-first<count; }
uint ac_body_tree(uint row) {
    uint4 s=ac_header(0u),m=ac_header(1u);
    return ac_contains(row,s.x,s.y)?ac_load(m.y+4u*(row-s.x)):ArtControlInvalid;
}
uint ac_joint_tree(uint row) {
    uint4 s=ac_header(0u),m=ac_header(1u);
    return ac_contains(row,s.z,s.w)?ac_load(m.z+4u*(row-s.z)):ArtControlInvalid;
}
void ac_refuse(uint tree,uint reason) {
    if(tree!=ArtControlInvalid)articulation_words[(ac_header(2u).x>>2u)+tree].or(reason);
}
uint ac_read(uint byteAddress) {
    return articulation_control[byteAddress>>4u].value[(byteAddress>>2u)&3u];
}
bool jph_articulation_admitted(uint row,bool joint) {
    uint4 selected=articulation_control[0].value,maps=articulation_control[1].value;
    uint first=joint?selected.z:selected.x,count=joint?selected.w:selected.y;
    if(!ac_contains(row,first,count))return false;
    uint tree=ac_read((joint?maps.z:maps.y)+4u*(row-first));
    return tree!=ArtControlInvalid && (ac_read(maps.w+4u*tree)&ArtControlAdmitted)!=0u;
}

[numthreads(64,1,1)]
void articulation_capture_rotors(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;
    uint row=articulation_params.range.x+id.x;
    uint4 selected=ac_header(0u);
    if(!ac_contains(row,selected.z,selected.w)||ac_joint_tree(row)==ArtControlInvalid)return;
    JointDirectory d=jph_joint_load_directory(row);
    if(d.bodies.z!=JointHinge)return;
    JointHingeState state;jph_joint_load_state(d,state);
    // Cached old armature is essential before native Setup rewrites x/y.
    float speed=state.rotor.w>0.0f?state.rotor.x-state.rotor.y/state.rotor.w:0.0f;
    bool valid=state.rotor.w>=0.0f && all(isfinite(state.rotor)) && isfinite(speed);
    uint b=ac_header(2u).y+16u*(row-selected.z);
    uint4 value=asuint(float4(speed,state.rotor.w,valid?1.0f:0.0f,0));
    for(uint lane=0u;lane<4u;++lane)ac_store(b+4u*lane,value[lane]);
}
[numthreads(64,1,1)]
void articulation_classify_begin(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;
    uint tree=articulation_params.range.x+id.x;
    if(tree>=ac_header(1u).x)return;
    ac_store(ac_header(2u).x+4u*tree,articulation_params.range.w);
    uint address=ac_header(1u).w+4u*tree;
    // Previous representation/dirty/capture facts survive this substep begin.
    ac_store(address,ac_load(address)&~(ArtControlEligible|ArtControlAdmitted|ArtControlCandidate));
}
[numthreads(64,1,1)]
void articulation_classify_bodies(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;
    uint row=articulation_params.range.x+id.x,tree=ac_body_tree(row);
    if(tree==ArtControlInvalid)return;
    BodyModel m=body_models[row];
    if(m.meta.w!=articulation_params.range.z || !jph_resident(m)) {
        ac_refuse(tree,ArtRefusedBody);return;
    }
    uint motion=m.meta.z&~MotionNotResident;
    if(motion==MotionDynamic) {
        if(!(m.mass.x>0.0f)||m.mass.y!=0.0f||m.mass.z!=0.0f
            ||!all(m.invInertiaDiagonal.xyz>float3(0)))ac_refuse(tree,ArtRefusedBody);
        if(body_sleep[row].runtime.state.x==0u)ac_refuse(tree,ArtRefusedSleeping);
    } else if(motion!=MotionStatic)ac_refuse(tree,ArtRefusedBody);
    // Includes TemporaryStatic and the final actual prescribed-motion witness.
    BodyModel effective=jph_effective_model(m,kinematic_targets[row]);
    if((motion==MotionStatic && effective.meta.z!=MotionStatic)
        ||body_sleep[row].runtime.state.w!=0u)ac_refuse(tree,ArtRefusedPrescribed);
}
void ac_incident(uint row,uint internalTree,uint reason) {
    if(row==InvalidQueryBodySlot)return;
    uint tree=ac_body_tree(row);
    if(tree!=ArtControlInvalid && jph_dynamic(body_models[row]) && tree!=internalTree)ac_refuse(tree,reason);
}
[numthreads(64,1,1)]
void articulation_classify_joints(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;
    uint row=articulation_params.range.x+id.x,tree=ac_joint_tree(row);
    if(tree==ArtControlInvalid)return;
    JointDirectory d=jph_joint_load_directory(row);
    uint4 flags=joint_stream[d.layout.y].value;
    if(flags.x!=1u||flags.y!=JointStatusSuccess||(flags.w&JointPoseDriveActive)!=0u)
        ac_refuse(tree,ArtRefusedRuntimeJoint);
}
[numthreads(64,1,1)]
void articulation_classify_refs(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;
    bool patch=articulation_params.range.w!=0u;
    uint ordinal=articulation_params.range.x+id.x;
    uint count=patch?constraint_schedule[3u].value.w:articulation_params.references.y;
    if(ordinal>=count)return;
    uint base=patch?articulation_params.references.z:articulation_params.references.x;
    uint4 endpoints=constraint_schedule[base+2u*ordinal].value;
    uint4 state=constraint_schedule[base+2u*ordinal+1u].value;
    if(endpoints.x!=articulation_params.range.z||state.y!=1u||state.z!=0u)return;
    // sourceOrdinal is the canonical native directory row, never public A/B.
    uint internalTree=patch?ArtControlInvalid:ac_joint_tree(endpoints.w);
    uint reason=patch?ArtRefusedContact:ArtRefusedExternalJoint;
    ac_incident(endpoints.y,internalTree,reason);
    ac_incident(endpoints.z,internalTree,reason);
}
[numthreads(64,1,1)]
void articulation_classify_finalize(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;
    uint tree=articulation_params.range.x+id.x;
    if(tree>=ac_header(1u).x)return;
    uint reasons=ac_load(ac_header(2u).x+4u*tree);
    if(articulation_params.range.w!=0u) {
        reasons|=ArtRefusedVelocityDisabled;ac_store(ac_header(2u).x+4u*tree,reasons);
    }
    uint address=ac_header(1u).w+4u*tree;
    ac_store(address,(ac_load(address)&~ArtControlEligible)|(reasons==0u?ArtControlEligible:0u));
}

[numthreads(64,1,1)]
void articulation_dirty(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;uint tree=articulation_params.range.x+id.x;
    uint address=ac_header(1u).w+4u*tree;ac_store(address,ac_load(address)|ArtControlDirty);
}
[numthreads(64,1,1)]
void articulation_initialize_numeric(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;
    // Explicit scalar range is a complete store; it excludes the initialized prefix.
    articulation_words[articulation_params.range.x+id.x].store(0u);
}
[numthreads(64,1,1)]
void articulation_mark_candidate(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;uint tree=articulation_params.range.x+id.x;
    uint address=ac_header(1u).w+4u*tree,flags=ac_load(address);ArticulationTree t=articulation_trees[tree];
    bool ready=(flags&(ArtControlCaptured|ArtControlEligible))==(ArtControlCaptured|ArtControlEligible)
        &&t.progress.x==ArtSuccess&&t.progress.y==ArtDynReady&&t.progress.z==t.range.y;
    ac_store(address,(flags&~ArtControlCandidate)|(ready?ArtControlCandidate:0u));
}
[numthreads(64,1,1)]
void articulation_admit(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;uint tree=articulation_params.range.x+id.x;
    uint address=ac_header(1u).w+4u*tree,flags=ac_load(address);ArticulationTree t=articulation_trees[tree];
    bool committed=(flags&ArtControlCandidate)!=0u&&t.progress.x==ArtPending
        &&t.progress.y==ArtDynAwaitIntegration&&t.progress.z==t.range.y;
    ac_store(address,(flags&~ArtControlAdmitted)|(committed?ArtControlAdmitted:0u));
}
uint ac_previous_flags(uint tree) {
    uint address=articulation_previous_control[1u].value.w+4u*tree;
    return articulation_previous_control[address>>4u].value[(address>>2u)&3u];
}
[numthreads(64,1,1)]
void articulation_capture_transition(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;uint row=articulation_params.range.x+id.x;
    uint4 map=articulation_transitions[row];JointDirectory d=jph_joint_load_directory(map.y);
    float4 value=float4(0,0,1,(ac_previous_flags(map.x)&ArtControlPreviousReduced)!=0u?1.0f:0.0f);
    if(d.bodies.z==JointHinge) {
        JointHingeState s;jph_joint_load_state(d,s);
        value.y=s.rotor.w;value.x=s.rotor.w>0?s.rotor.x-s.rotor.y/s.rotor.w:0;
        value.z=s.rotor.w>=0&&all(isfinite(s.rotor))&&isfinite(value.x)?1.0f:0.0f;
    }
    articulation_transition_witness[row]=value;
}
void ac_reset_transition(JointDirectory d,JointFixedState s,float4 witness) {
    jph_joint_clear_impulses(s);jph_joint_store_state(d,s);
}
void ac_reset_transition(JointDirectory d,JointHingeState s,float4 witness) {
    jph_joint_clear_impulses(s);s.rotor=float4(witness.x,0,witness.x,s.rotor.w);jph_joint_store_state(d,s);
}
void ac_reset_transition(JointDirectory d,JointSliderState s,float4 witness) {
    jph_joint_clear_impulses(s);jph_joint_store_state(d,s);
}
[numthreads(64,1,1)]
void articulation_reset_transition(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;uint row=articulation_params.range.x+id.x;
    uint4 map=articulation_transitions[row];float4 witness=articulation_transition_witness[row];
    bool before=witness.w==1.0f;
    bool after=false;
    if(map.z!=ArtControlInvalid)after=(ac_load(ac_header(1u).w+4u*map.z)&ArtControlCandidate)!=0u;
    if(before==after)return;
    if(witness.z!=1.0f) {articulation_summary[0u].or(1u);return;}
    JointDirectory d=jph_joint_load_directory(map.y);
    if(d.bodies.z==JointFixed) {JointFixedState s;jph_joint_load_state(d,s);ac_reset_transition(d,s,witness);}
    else if(d.bodies.z==JointHinge) {JointHingeState s;jph_joint_load_state(d,s);ac_reset_transition(d,s,witness);}
    else if(d.bodies.z==JointSlider) {JointSliderState s;jph_joint_load_state(d,s);ac_reset_transition(d,s,witness);}
}
[numthreads(64,1,1)]
void articulation_clear_summary(uint3 id:SV_DispatchThreadID) {
    if(id.x<4u)articulation_summary[id.x].store(0u);
}
[numthreads(64,1,1)]
void articulation_numeric_summary(uint3 id:SV_DispatchThreadID) {
    if(id.x>=articulation_params.range.y)return;uint tree=articulation_params.range.x+id.x;
    uint flags=ac_load(ac_header(1u).w+4u*tree);ArticulationTree t=articulation_trees[tree];
    bool failure=false;
    if(articulation_params.range.w==0u) {
        if((flags&ArtControlCandidate)!=0u)failure=t.progress.x!=ArtPending
            ||t.progress.y!=ArtDynAwaitIntegration||t.progress.z!=t.range.y||(flags&ArtControlAdmitted)==0u;
    } else if((flags&ArtControlAdmitted)!=0u) {
        StateFkTree capture=articulation_capture_trees[tree];
        failure=t.progress.x!=ArtSuccess||t.progress.y!=ArtDynDone||capture.progress.x!=StateFkSuccess
            ||capture.progress.y!=StateFkDone||capture.progress.z!=capture.range.y;
    }
    if(failure)articulation_summary[0u].or(1u);
    if((flags&ArtControlCandidate)!=0u)articulation_summary[1u].add(1u);
    if((flags&ArtControlAdmitted)!=0u)articulation_summary[2u].add(1u);
    articulation_summary[3u].add(1u);
}

// Cold-generation rejection only. The old generation has no further numerical
// consumer: reuse its dead numeric tail for (oldTree,currentRow,World,status).
// Prefix and row/status bytes are one RAW view, not aliased bindings.
bool ac_retire_declined(JointDirectory d,JointFixedState s) {
    jph_joint_clear_impulses(s);jph_joint_store_state(d,s);return true;
}
bool ac_retire_declined(JointDirectory d,JointHingeState s) {
    if(!all(isfinite(s.rotor))||s.rotor.w<0.0f)return false;
    float speed=s.rotor.w>0.0f?s.rotor.x-s.rotor.y/s.rotor.w:0.0f;
    if(!isfinite(speed))return false;
    float armature=s.rotor.w;jph_joint_clear_impulses(s);
    s.rotor=float4(speed,0,speed,armature);jph_joint_store_state(d,s);return true;
}
bool ac_retire_declined(JointDirectory d,JointSliderState s) {
    jph_joint_clear_impulses(s);jph_joint_store_state(d,s);return true;
}
[numthreads(64,1,1)]
void articulation_retire_declined(uint3 id:SV_DispatchThreadID) {
    uint4 job=articulation_retirement[articulation_retirement[2u].z>>4u];
    if(id.x>=2048u||id.x>=job.y-min(job.w,job.y))return;uint row=job.w+id.x;
    uint address=job.x+row;uint4 record=articulation_retirement[address];
    uint flagsByte=articulation_retirement[1u].w+4u*record.x;
    uint flags=articulation_retirement[flagsByte>>4u][(flagsByte>>2u)&3u];record.w=0u;
    if((flags&ArtControlPreviousReduced)!=0u) {
        JointDirectory d=jph_joint_load_directory(record.y);bool valid=false;
        if(d.identity.z==record.z) {
            if(d.bodies.z==JointFixed) {JointFixedState s;jph_joint_load_state(d,s);valid=ac_retire_declined(d,s);}
            else if(d.bodies.z==JointHinge) {JointHingeState s;jph_joint_load_state(d,s);valid=ac_retire_declined(d,s);}
            else if(d.bodies.z==JointSlider) {JointSliderState s;jph_joint_load_state(d,s);valid=ac_retire_declined(d,s);}
        }
        record.w=valid?0u:1u;
    }
    articulation_retirement[address]=record;
}
[numthreads(64,1,1)]
void articulation_clear_retirement_summary(uint3 id:SV_DispatchThreadID) {
    if(id.x!=0u)return;uint header=articulation_retirement[2u].z>>4u;uint4 job=articulation_retirement[header];
    job.w=0u;articulation_retirement[header]=job;articulation_retirement[job.z]=uint4(0);
}

// Setup has already consumed x-y/w, possibly for only a native row prefix.
// Restore the real pre-Setup physical rotor witness exactly once; never decode
// the partially transformed native rotor representation a second time.
[numthreads(64,1,1)]
void articulation_retire_captured(uint3 id:SV_DispatchThreadID) {
    uint4 job=articulation_retirement[articulation_retirement[2u].z>>4u];
    if(id.x>=2048u||id.x>=job.y-min(job.w,job.y))return;
    uint row=job.w+id.x,address=job.x+row;uint4 record=articulation_retirement[address];
    float4 witness=articulation_transition_witness[row];record.w=0u;
    if(witness.w==1.0f) {
        JointDirectory d=jph_joint_load_directory(record.y);bool valid=witness.z==1.0f&&d.identity.z==record.z;
        if(valid) {
            if(d.bodies.z==JointFixed) {JointFixedState s;jph_joint_load_state(d,s);jph_joint_clear_impulses(s);jph_joint_store_state(d,s);}
            else if(d.bodies.z==JointHinge) {JointHingeState s;jph_joint_load_state(d,s);jph_joint_clear_impulses(s);s.rotor=float4(witness.x,0,witness.x,s.rotor.w);jph_joint_store_state(d,s);}
            else if(d.bodies.z==JointSlider) {JointSliderState s;jph_joint_load_state(d,s);jph_joint_clear_impulses(s);jph_joint_store_state(d,s);}
            else valid=false;
        }
        record.w=valid?0u:1u;
    }
    articulation_retirement[address]=record;
}
[numthreads(64,1,1)]
void articulation_advance_retirement(uint3 id:SV_DispatchThreadID) {
    if(id.x!=0u)return;uint header=articulation_retirement[2u].z>>4u;uint4 job=articulation_retirement[header];
    job.w+=min(2048u,job.y-min(job.w,job.y));articulation_retirement[header]=job;
}
[numthreads(64,1,1)]
void articulation_retirement_summary(uint3 id:SV_DispatchThreadID) {
    uint header=articulation_words[10u].load()>>2u;
    uint first=articulation_words[header].load(),count=articulation_words[header+1u].load();
    uint summary=4u*articulation_words[header+2u].load(),cursor=articulation_words[header+3u].load();
    if(id.x>=2048u||id.x>=count-min(cursor,count))return;uint row=cursor+id.x;
    uint status=articulation_words[4u*(first+row)+3u].load();
    if(status!=0u)articulation_words[summary].or(status);
    articulation_words[summary+3u].add(1u);
}
