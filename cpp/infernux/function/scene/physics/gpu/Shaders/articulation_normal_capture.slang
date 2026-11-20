// Owner wrapper. Included frozen capture is unchanged; no FK projection of a
// refused live state. Active and staging root/q have separate actual owners.
#include "articulation_state_fk.slang"
#include "../ArticulationIntegrationTypes.h"
#include "../ArticulationDynamicsTypes.h"
[[vk::binding(9,0)]] StructuredBuffer<BodyState> normal_active_roots;
[[vk::binding(10,0)]] StructuredBuffer<float4> normal_active_coordinates;
[[vk::binding(11,0)]] RWStructuredBuffer<Atomic<uint>> normal_capture_control;
[[vk::binding(12,0)]] RWStructuredBuffer<BodyState> normal_resident_roots;
[[vk::binding(13,0)]] RWStructuredBuffer<float4> normal_resident_coordinates;
[[vk::binding(14,0)]] StructuredBuffer<ArticulationTree> normal_dynamics_trees;

uint nc_flags_byte() {return normal_capture_control[7u].load();}
uint nc_flags(uint tree) {return normal_capture_control[(nc_flags_byte()>>2u)+tree].load();}
void nc_flags(uint tree,uint flags) {normal_capture_control[(nc_flags_byte()>>2u)+tree].store(flags);}
bool nc_tree(uint id,out uint tree) {
    tree=0;if(id>=sf_work.range.y)return false;tree=sf_work.range.x+id;
    uint count,stride;sf_trees.GetDimensions(count,stride);return tree<count;
}

[shader("compute")][numthreads(64,1,1)]
void normal_capture_initialize(uint3 id:SV_DispatchThreadID) {
    if(id.x>=sf_work.range.y)return;uint index=sf_work.range.x+id.x;
    uint roots,rs,coordinates,cs;sf_roots.GetDimensions(roots,rs);sf_coordinates.GetDimensions(coordinates,cs);
    if(index<roots) {sf_roots[index]=(BodyState)0;normal_resident_roots[index]=(BodyState)0;}
    if(index<coordinates) {sf_coordinates[index]=float4(0);normal_resident_coordinates[index]=float4(0);}
}
[shader("compute")][numthreads(64,1,1)]
void normal_capture_begin(uint3 id:SV_DispatchThreadID) {
    uint tree;if(!nc_tree(id.x,tree))return;
    uint flags=nc_flags(tree);nc_flags(tree,flags&~ArtControlCaptured);
    bool capture=(flags&ArtControlDirty)!=0u||(flags&ArtControlPreviousReduced)==0u;
    sf_begin(tree,capture?StateFkCapturePhase:ArtNormalSeed);
}
[shader("compute")][numthreads(64,1,1)]
void normal_capture_advance(uint3 id:SV_DispatchThreadID) {
    uint tree;if(!nc_tree(id.x,tree))return;
    StateFkTree state=sf_trees[tree];
    if(state.progress.x!=StateFkPending||state.progress.y!=StateFkCapturePhase)return;
    // Invoke the unchanged frozen per-link helper, not an entry point from
    // another entry. Owner scheduling is bounded and root sign/bits survive.
    for(uint unit=0u;unit<sf_work.range.z;++unit) {
        StateFkTree t;if(!sf_next(tree,StateFkCapturePhase,t))break;
        sf_capture_link(tree,t.progress.z);if(sf_trees[tree].progress.x!=StateFkPending)break;
        sf_finish_link(tree,t);
    }
    if(sf_trees[tree].progress.x==StateFkSuccess)nc_flags(tree,nc_flags(tree)|ArtControlCaptured);
}
[shader("compute")][numthreads(64,1,1)]
void normal_seed_advance(uint3 id:SV_DispatchThreadID) {
    uint tree;if(!nc_tree(id.x,tree))return;
    StateFkTree t=sf_trees[tree];
    if(t.progress.x!=StateFkPending||t.progress.y!=ArtNormalSeed)return;
    for(uint unit=0u;unit<sf_work.range.z;++unit) {
        uint local=t.progress.z;
        if(local==0u)sf_roots[tree]=normal_active_roots[tree];
        sf_coordinates[t.range.z+local]=normal_active_coordinates[t.range.z+local];
        ++t.progress.z;
        if(t.progress.z==t.range.y) {
            t.progress.xy=uint2(StateFkSuccess,StateFkDone);sf_trees[tree]=t;
            nc_flags(tree,nc_flags(tree)|ArtControlCaptured);return;
        }
    }
    sf_trees[tree]=t;
}
[shader("compute")][numthreads(64,1,1)]
void normal_promote_begin(uint3 id:SV_DispatchThreadID) {
    uint tree;if(!nc_tree(id.x,tree))return;
    uint flags=nc_flags(tree);
    if((flags&ArtControlAdmitted)==0u) {
        nc_flags(tree,(flags&~(ArtControlPreviousReduced|ArtControlCaptured|ArtControlCandidate))|ArtControlDirty);
        StateFkTree t=sf_trees[tree];t.progress=uint4(StateFkSuccess,StateFkDone,t.range.y,0);sf_trees[tree]=t;return;
    }
    ArticulationTree dynamics=normal_dynamics_trees[tree];
    if(dynamics.progress.x!=ArtSuccess||dynamics.progress.y!=ArtDynDone) {
        // Accepted whole-tree commit has no ordinary refusal after velocity
        // publication. This is a structural owner failure, never a native retry.
        sf_fail(tree,StateFkInvalid,90u);return;
    }
    sf_begin(tree,ArtNormalPromote);
}
[shader("compute")][numthreads(64,1,1)]
void normal_promote_advance(uint3 id:SV_DispatchThreadID) {
    uint tree;if(!nc_tree(id.x,tree))return;StateFkTree t=sf_trees[tree];
    if(t.progress.x!=StateFkPending||t.progress.y!=ArtNormalPromote)return;
    for(uint unit=0u;unit<sf_work.range.z;++unit) {
        uint local=t.progress.z;
        if(local==0u)normal_resident_roots[tree]=sf_roots[tree];
        normal_resident_coordinates[t.range.z+local]=sf_coordinates[t.range.z+local];
        ++t.progress.z;
        if(t.progress.z==t.range.y) {
            t.progress.xy=uint2(StateFkSuccess,StateFkDone);sf_trees[tree]=t;
            nc_flags(tree,(nc_flags(tree)&~ArtControlDirty)|ArtControlPreviousReduced);return;
        }
    }
    sf_trees[tree]=t;
}
