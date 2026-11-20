// Included after rigid_step.slang's canonical state/model/constraint resources.
// This is control work on the same source; no dispatch submits or restarts time.
ConstantBuffer<SleepWorkParameters> sleep_params;
RWStructuredBuffer<Atomic<uint>> body_sleep_roots;
RWStructuredBuffer<Atomic<uint>> step_status;
RWStructuredBuffer<float4> sleep_aabbs;
RWStructuredBuffer<uint4> sleep_snapshot;
StructuredBuffer<uint> sleep_requests;
RWStructuredBuffer<uint4> sleep_results;

static const uint SleepBadState=1u;
static const uint SleepBadRoot=2u;
static const uint SleepInvalid=0xffffffffu;

void sleep_fail(uint code) { step_status[StepSleepStatus].or(code); }
bool sleep_work_item(uint lane,out uint item)
{
    item=0u;
    if(lane>=stage_work.range.y) return false;
    item=stage_work.range.x+lane;return true;
}
bool sleep_body(uint body)
{
    if(body<sleep_params.selected.y || body-sleep_params.selected.y>=sleep_params.selected.z
       || body>=sleep_params.selected.w) return false;
    return body_models[body].meta.w==sleep_params.selected.x;
}
uint sleep_root(uint body)
{
    uint local=body-sleep_params.selected.y;
    if(local>=sleep_params.selected.z) {sleep_fail(SleepBadRoot);return SleepInvalid;}
    uint root=constraint_schedule[sleep_params.layout.x+local].value.x;
    if(root>=sleep_params.selected.z) {sleep_fail(SleepBadRoot);return SleepInvalid;}
    return root;
}
bool sleep_active_endpoint(uint body)
{
    if(body==SleepInvalid || !sleep_body(body)) return false;
    if(!jph_resident(body_models[body])) return false;
    uint motion=body_models[body].meta.z&~MotionNotResident;
    return motion==MotionStatic?body_sleep[body].runtime.state.w!=0u:body_sleep[body].runtime.state.x!=0u;
}
void sleep_activate(uint body,bool resetActive)
{
    if(!sleep_body(body)) return;
    if(!jph_resident(body_models[body]) || !jph_nonstatic(body_models[body])) return;
    BodySleepRuntime state=body_sleep[body].runtime;
    if(state.state.x==0u || resetActive) {
        BodyState pose=body_states[body];float3 p0,p1,p2;
        sleep_test_points(pose,body_sleep[body].author.rootHalfExtent.xyz,p0,p1,p2);
        sleep_reset(state,p0,p1,p2);state.state.x=1u;
        body_sleep[body].runtime=state;
    }
}
void sleep_seed_pair(uint a,uint b)
{
    if(!sleep_active_endpoint(a) && !sleep_active_endpoint(b)) return;
    if(a!=SleepInvalid) if(sleep_body(a)) if(jph_dynamic(body_models[a])) {
        uint root=sleep_root(a);if(root!=SleepInvalid) body_sleep_roots[root].or(1u);
    }
    if(b!=SleepInvalid) if(sleep_body(b)) if(jph_dynamic(body_models[b])) {
        uint root=sleep_root(b);if(root!=SleepInvalid) body_sleep_roots[root].or(1u);
    }
}
[shader("compute")][numthreads(64,1,1)]
void sleep_clear_votes(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body)) return;
    uint local=body-sleep_params.selected.y;
    body_sleep_roots[local].store(0u);
}
[shader("compute")][numthreads(64,1,1)]
void sleep_seed_awake_bodies(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body) || !sleep_body(body)) return;
    if(jph_resident(body_models[body]) && jph_dynamic(body_models[body]) && body_sleep[body].runtime.state.x!=0u) {
        uint root=sleep_root(body);if(root!=SleepInvalid) body_sleep_roots[root].or(1u);
    }
}
[shader("compute")][numthreads(64,1,1)]
void sleep_seed_joint_endpoints(uint3 tid:SV_DispatchThreadID)
{
    uint index;if(!sleep_work_item(tid.x,index)) return;
    if(index<sleep_params.layout.z || index-sleep_params.layout.z>=sleep_params.layout.w) return;
    JointDirectory directory=jph_joint_load_directory(index);
    if(directory.identity.z!=sleep_params.selected.x || (directory.bodies.w&JointEnabled)==0u) return;
    uint a=directory.bodies.x,b=directory.bodies.y;
    if(!sleep_body(b)) return;
    if(!jph_resident(body_models[b])) return;
    if(a!=SleepInvalid) {if(!sleep_body(a)) return;if(!jph_resident(body_models[a])) return;}
    sleep_seed_pair(a,b);
}
[shader("compute")][numthreads(64,1,1)]
void sleep_seed_patch_endpoints(uint3 tid:SV_DispatchThreadID)
{
    uint index;if(!sleep_work_item(tid.x,index) || index>=sleep_params.layout.w) return;
    uint4 header = contact_patches[0].leaves;
    if (index >= header.y) return;
    uint4 patch=contact_patches[header.x+index].leaves;
    if((patch.z&PatchActive)==0u || (patch.z&PatchSensor)!=0u) return;
    sleep_seed_pair(shape_models[patch.x].meta.y,shape_models[patch.y].meta.y);
}
[shader("compute")][numthreads(64,1,1)]
void sleep_wake_components(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body) || !sleep_body(body)) return;
    if(!jph_resident(body_models[body]) || !jph_dynamic(body_models[body])) return;
    uint root=sleep_root(body);
    if(root!=SleepInvalid && (body_sleep_roots[root].load()&1u)!=0u) sleep_activate(body,false);
}
// Only emitted as a successor to a genuinely new accepted force-input writer.
// Historical accumulated forces are never polled as an activation cause.
[shader("compute")][numthreads(64,1,1)]
void sleep_wake_force_input(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body) || !sleep_body(body)) return;
    if(!jph_dynamic(body_models[body])) return;
    BodyForces force=body_forces[body];
    if(any(force.force.xyz!=float3(0)) || any(force.torque.xyz!=float3(0))) sleep_activate(body,true);
}
uint sleep_update_body(uint body)
{
    if(!sleep_body(body)) return SleepInvalid;
    BodyModel model=body_models[body];
    if(!jph_resident(model) || !jph_nonstatic(model)) return SleepInvalid;
    BodySleepRecord record=body_sleep[body];BodyState pose=body_states[body];
    if(!all(isfinite(pose.positionCOM)) || !all(isfinite(pose.rotation))
       || !all(isfinite(pose.linearVelocity)) || !all(isfinite(pose.angularVelocity))
       || dot(pose.rotation,pose.rotation)<=0 || any(record.runtime.state>uint4(1u))) {
        sleep_fail(SleepBadState);return SleepInvalid;
    }
    if(record.runtime.state.x==0u) {
        if(any(pose.linearVelocity.xyz!=float3(0)) || any(pose.angularVelocity.xyz!=float3(0))) sleep_fail(SleepBadState);
        return SleepInvalid;
    }
    // Retire forces for the final active set before Commit may put it to sleep.
    if(sleep_params.operation.z!=0u) {
        BodyForces cleared;cleared.force=cleared.torque=float4(0);
        body_forces[body]=cleared;
    }
    BodySleepParameters parameters;parameters.timing=sleep_params.timing;
    uint vote=sleep_update_vote(record.author,record.runtime,pose,parameters);
    if(vote==2u) {sleep_fail(SleepBadState);return SleepInvalid;}
    body_sleep[body].runtime=record.runtime;
    return vote;
}
void sleep_put_to_sleep(uint body,BodySleepRuntime state)
{
    BodyState pose=body_states[body];pose.linearVelocity=pose.angularVelocity=float4(0);
    body_states[body]=pose;state.state.x=0u;body_sleep[body].runtime=state;
}
void sleep_commit_body(uint body)
{
    if(!sleep_body(body) || sleep_params.operation.z==0u) return;
    if(step_status[StepSleepStatus].load()!=0u) return;
    BodyModel model=body_models[body];BodySleepRuntime state=body_sleep[body].runtime;
    if(!jph_resident(model) || !jph_nonstatic(model) || state.state.x==0u) return;
    bool eligible=state.state.z!=0u;
    if(jph_dynamic(model)) {uint root=sleep_root(body);eligible=root!=SleepInvalid && (body_sleep_roots[root].load()&2u)==0u;}
    if(eligible) sleep_put_to_sleep(body,state);
}
[shader("compute")][numthreads(64,1,1)]
void sleep_update_history(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body)) return;
    uint vote=sleep_update_body(body);
    if(vote==0u && jph_dynamic(body_models[body])) {
        uint root=sleep_root(body);if(root!=SleepInvalid) body_sleep_roots[root].or(2u);
    }
}
[shader("compute")][numthreads(64,1,1)]
void sleep_commit_components(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body)) return;
    sleep_commit_body(body);
}
// Admitted only for an accepted graph with no joints and no actual patches.
// A body's own vote decides whether it can sleep.
[shader("compute")][numthreads(64,1,1)]
void sleep_update_commit_isolated(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body)) return;
    uint vote=sleep_update_body(body);
    if(vote==1u && sleep_params.operation.z!=0u && step_status[StepSleepStatus].load()==0u)
        sleep_put_to_sleep(body,body_sleep[body].runtime);
}
[shader("compute")][numthreads(64,1,1)]
void sleep_retire_hinge_momentum(uint3 tid:SV_DispatchThreadID)
{
    uint index;if(!sleep_work_item(tid.x,index) || sleep_params.operation.z==0u) return;
    if(index<sleep_params.layout.z || index-sleep_params.layout.z>=sleep_params.layout.w) return;
    if(step_status[StepSleepStatus].load()!=0u) return;
    JointDirectory directory=jph_joint_load_directory(index);
    if(directory.identity.z!=sleep_params.selected.x || directory.bodies.z!=JointHinge) return;
    uint a=directory.bodies.x,b=directory.bodies.y;
    bool stopped=false;
    if(sleep_body(b)) stopped=jph_resident(body_models[b]) && jph_dynamic(body_models[b])
        && body_sleep[b].runtime.state.x==0u && body_sleep[b].runtime.state.z!=0u;
    if(a!=SleepInvalid) if(sleep_body(a))
        stopped=stopped || (jph_resident(body_models[a]) && jph_dynamic(body_models[a])
            && body_sleep[a].runtime.state.x==0u && body_sleep[a].runtime.state.z!=0u);
    if(stopped) {
        JointHingeState state;jph_joint_load_state(directory,state);
        state.rotor.xyz=float3(0);jph_joint_store_state(directory,state);
    }
}
void sleep_root_bounds(uint body,out float3 low,out float3 high)
{
    BodySleepAuthor author=body_sleep[body].author;BodyState pose=body_states[body];
    float3 center=pose.positionCOM.xyz+jph_rotate(pose.rotation,author.rootBoundsCenter.xyz);
    float3 x=abs(jph_rotate(pose.rotation,float3(author.rootHalfExtent.x,0,0)));
    float3 y=abs(jph_rotate(pose.rotation,float3(0,author.rootHalfExtent.y,0)));
    float3 z=abs(jph_rotate(pose.rotation,float3(0,0,author.rootHalfExtent.z)));
    low=center-x-y-z;high=center+x+y+z;
}
[shader("compute")][numthreads(64,1,1)]
void sleep_capture_old_bounds(uint3 tid:SV_DispatchThreadID)
{
    if(tid.x!=0u) return;uint body=sleep_params.operation.x;
    float3 low=float3(0),high=float3(0);
    BodyState pose=(BodyState)0;
    if(!sleep_body(body)) sleep_fail(SleepBadRoot);
    else {pose=body_states[body];sleep_root_bounds(body,low,high);if(!all(isfinite(low)) || !all(isfinite(high))) sleep_fail(SleepBadState);}
    sleep_aabbs[0]=float4(low,0);sleep_aabbs[1]=float4(high,0);
    sleep_aabbs[2]=pose.positionCOM;sleep_aabbs[3]=pose.rotation;
}

bool sleep_bounds_target(uint lane,out uint body)
{
    if(!sleep_work_item(lane,body) || !sleep_body(body)) return false;
    return jph_resident(body_models[body]) && jph_dynamic(body_models[body]) && body!=sleep_params.operation.x;
}
bool sleep_checked_bounds(uint body,out float3 low,out float3 high)
{
    sleep_root_bounds(body,low,high);
    if(all(isfinite(low)) && all(isfinite(high))) return true;
    sleep_fail(SleepBadState);return false;
}
bool sleep_current_bounds_hit(float3 low,float3 high,out bool hit)
{
    hit=false;
    uint source=sleep_params.operation.x;
    if(!sleep_body(source)) {sleep_fail(SleepBadRoot);return false;}
    float3 a,b;if(!sleep_checked_bounds(source,a,b)) return false;
    hit=!any(low>b) && !any(a>high);
    return true;
}

[shader("compute")][numthreads(64,1,1)]
void sleep_wake_bounds(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_bounds_target(tid.x,body)) return;
    if((sleep_params.operation.y&4u)!=0u) {
        if(!sleep_body(sleep_params.operation.x)) {sleep_fail(SleepBadRoot);return;}
        BodyState source=body_states[sleep_params.operation.x];
        float4 oldq=sleep_aabbs[3];
        if(all(source.positionCOM.xyz==sleep_aabbs[2].xyz)
           && (all(source.rotation==oldq) || all(source.rotation==-oldq))) return;
    }
    float3 low,high;if(!sleep_checked_bounds(body,low,high)) return;
    bool hit=false;
    if((sleep_params.operation.y&1u)!=0u)
        hit=!any(low>sleep_aabbs[1].xyz) && !any(sleep_aabbs[0].xyz>high);
    if((sleep_params.operation.y&2u)!=0u) {
        bool currentHit;
        if(!sleep_current_bounds_hit(low,high,currentHit)) return;
        hit=hit || currentHit;
    }
    if(hit) sleep_activate(body,true);
}

bool sleep_bounds_pair_excluded(uint source,uint target)
{
    uint first=0u,end=sleep_params.layout.w;
    while(first<end) {
        uint middle=first+(end-first)/2u;
        uint offset=sleep_params.layout.z+2u*middle;
        uint a=sleep_requests[offset],b=sleep_requests[offset+1u];
        if(a==source && b==target) return true;
        if(a<source || (a==source && b<target)) first=middle+1u;
        else end=middle;
    }
    return false;
}

[shader("compute")][numthreads(64,1,1)]
void sleep_wake_new_bounds(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_bounds_target(tid.x,body)) return;
    float3 low,high;if(!sleep_checked_bounds(body,low,high)) return;
    for(uint i=0u;i<sleep_params.operation.w;++i) {
        uint source=sleep_requests[sleep_params.operation.z+i];
        if(source==body) continue;
        if(!sleep_body(source)) {sleep_fail(SleepBadRoot);return;}
        float3 a,b;if(!sleep_checked_bounds(source,a,b)) return;
        if(!any(low>b) && !any(a>high) && !sleep_bounds_pair_excluded(source,body)) {
            sleep_activate(body,true);return;
        }
    }
}
[shader("compute")][numthreads(64,1,1)]
void sleep_export_status(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body) || !sleep_body(body)) return;
    uint4 state=body_sleep[body].runtime.state;
    uint motion=body_models[body].meta.z&~MotionNotResident;
    sleep_snapshot[body-sleep_params.selected.y]=uint4(motion==MotionStatic?0u:state.x,body_models[body].meta.z,state.y,0u);
}
[shader("compute")][numthreads(64,1,1)]
void query_sleep_status(uint3 tid:SV_DispatchThreadID)
{
    uint query;if(!sleep_work_item(tid.x,query) || query>=sleep_params.operation.w) return;
    uint relative=sleep_requests[query];uint4 result=uint4(0);
    if(relative>=sleep_params.selected.z) sleep_fail(SleepBadRoot);
    else result=sleep_snapshot[relative];
    sleep_results[query]=result;
}

// Source indices are immutable author intent, never a Host copy of motion.
// Four float4s retain the exact pre-preparation bound and pose. This is kept
// only until this Fixed's real prescribed-motion wake producer completes.
[shader("compute")][numthreads(64,1,1)]
void sleep_capture_prescribed_bounds(uint3 tid:SV_DispatchThreadID)
{
    uint index;if(!sleep_work_item(tid.x,index) || index>=sleep_params.operation.w) return;
    uint source=sleep_requests[index];float3 low=float3(0),high=float3(0);
    BodyState pose=(BodyState)0;
    if(!sleep_body(source)) sleep_fail(SleepBadRoot);
    else {
        pose=body_states[source];sleep_root_bounds(source,low,high);
        if(!all(isfinite(low)) || !all(isfinite(high)) || !all(isfinite(pose.rotation))) sleep_fail(SleepBadState);
    }
    sleep_aabbs[4u*index]=float4(low,0);sleep_aabbs[4u*index+1u]=float4(high,0);
    sleep_aabbs[4u*index+2u]=pose.positionCOM;sleep_aabbs[4u*index+3u]=pose.rotation;
}
[shader("compute")][numthreads(64,1,1)]
void sleep_wake_prescribed_bounds(uint3 tid:SV_DispatchThreadID)
{
    uint body;if(!sleep_work_item(tid.x,body) || !sleep_body(body)) return;
    if(!jph_resident(body_models[body]) || !jph_dynamic(body_models[body])) return;
    uint index=sleep_params.layout.z;
    if(index>=sleep_params.operation.w) {sleep_fail(SleepBadRoot);return;}
    uint source=sleep_requests[index];
    if(!sleep_body(source)) {sleep_fail(SleepBadRoot);return;}
    if(!jph_resident(body_models[source])) return;
    BodyState pose=body_states[source];float4 oldq=sleep_aabbs[4u*index+3u];
    uint motion=body_models[source].meta.z&~MotionNotResident;
    bool moving=false;
    if(motion==MotionStatic) moving=body_sleep[source].runtime.state.w!=0u;
    else if(motion==MotionKinematic) {
        bool changed=any(pose.positionCOM.xyz!=sleep_aabbs[4u*index+2u].xyz)
            || !(all(pose.rotation==oldq) || all(pose.rotation==-oldq));
        moving=changed || any(pose.linearVelocity.xyz!=float3(0)) || any(pose.angularVelocity.xyz!=float3(0));
    }
    if(!moving) return;
    float3 low,high,a,b;sleep_root_bounds(body,low,high);sleep_root_bounds(source,a,b);
    if(!all(isfinite(low)) || !all(isfinite(high)) || !all(isfinite(a)) || !all(isfinite(b))) {
        sleep_fail(SleepBadState);return;
    }
    bool hit=(!any(low>sleep_aabbs[4u*index+1u].xyz) && !any(sleep_aabbs[4u*index].xyz>high))
        || (!any(low>b) && !any(a>high));
    if(hit) sleep_activate(body,true);
}
