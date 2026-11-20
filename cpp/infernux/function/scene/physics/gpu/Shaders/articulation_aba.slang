// Shared constant-size ABA; old response and dynamics use one implementation.
float art_get(uint b,uint word,uint lane) { return asfloat(art_scratch[b+word])[lane]; }
void art_set(uint b,uint word,uint lane,float value)
{
    uint4 v=art_scratch[b+word]; v[lane]=asuint(value); art_scratch[b+word]=v;
}
float art_vec(uint b,uint word,uint i) { return art_get(b,word+i/4u,i%4u); }
void art_vec_set(uint b,uint word,uint i,float v) { art_set(b,word+i/4u,i%4u,v); }
float art_mat(uint b,uint i,uint j) { uint k=i*6u+j; return art_get(b,k/4u,k%4u); }
void art_mat_set(uint b,uint i,uint j,float v) { uint k=i*6u+j; art_set(b,k/4u,k%4u,v); }
float art_component(float3 angular,float3 linear,uint i) { return i<3u?angular[i]:linear[i-3u]; }
float art_x(float3 r,uint i,uint j)
{
    if(i==j) return 1;
    if(i<3u || j>=3u) return 0;
    float3 axis=float3(0); axis[j]=1;
    return cross(axis,r)[i-3u];
}
float art_motion(uint parent,float3 r,uint i)
{
    float3 w=asfloat(art_scratch[parent+17u]).xyz,v=asfloat(art_scratch[parent+18u]).xyz;
    return art_component(w,v+cross(w,r),i);
}
void art_failure(uint tree,uint status,uint reason)
{
    ArticulationTree value=art_trees[tree]; value.progress.x=status; value.progress.w=reason; art_trees[tree]=value;
}
float art_recovery_base(uint parent,float3 r,uint b,uint i) {
    float value=art_motion(parent,r,i);
#ifdef ARTICULATION_DYNAMICS
    value+=art_vec(b,23u,i);
#endif
    return value;
}
#ifdef ARTICULATION_DYNAMICS
void dyn_recovered(uint tree,uint local,float delta_qdot);
#endif
void art_backward_link(uint tree,uint local)
{
    ArticulationTree t=art_trees[tree]; ArticulationLink link=art_links[t.range.x+local];
    uint b=t.range.z+ArtScratchWords*local,p=t.range.z+ArtScratchWords*link.topology.y;
    bool movable=link.topology.w!=JointFixed;
    float d=0,u=art_get(b,16u,1u);
    if(movable) {
        for(uint i=0u;i<6u;++i) {
            float value=0; for(uint j=0u;j<6u;++j) value+=art_mat(b,i,j)*art_vec(b,13u,j);
            art_vec_set(b,11u,i,value); d+=art_vec(b,13u,i)*value; u-=art_vec(b,13u,i)*art_vec(b,9u,i);
        }
        d+=art_get(b,16u,2u);
        if(!(d>0) || !isfinite(d) || !isfinite(u)) { art_failure(tree,ArtNumericalFailure,10u); return; }
        art_set(b,16u,0u,d); art_set(b,16u,1u,u);
    }
    float3 r=asfloat(art_scratch[b+15u]).xyz;
    // Constant6D Schur complement/transport. Each tree has one writer.
    for(uint i=0u;i<6u;++i) {
        float force=0;
        for(uint k=0u;k<6u;++k) {
            float value=art_vec(b,9u,k)+(movable?art_vec(b,11u,k)*u/d:0);
#ifdef ARTICULATION_DYNAMICS
            // Local edge kappa enters once, after elimination, through Ia.
            for(uint l=0u;l<6u;++l)
                value+=(art_mat(b,k,l)-(movable?art_vec(b,11u,k)*art_vec(b,11u,l)/d:0))*art_vec(b,23u,l);
#endif
            force+=art_x(r,k,i)*value;
        }
        force+=art_vec(p,9u,i);
        if(!isfinite(force)) { art_failure(tree,ArtNumericalFailure,11u); return; }
        art_vec_set(p,9u,i,force);
        for(uint j=i;j<6u;++j) {
            float value=0;
            for(uint k=0u;k<6u;++k) for(uint l=0u;l<6u;++l)
                value+=art_x(r,k,i)*(art_mat(b,k,l)-(movable?art_vec(b,11u,k)*art_vec(b,11u,l)/d:0))*art_x(r,l,j);
            value+=art_mat(p,i,j);
            if(!isfinite(value)) { art_failure(tree,ArtNumericalFailure,12u); return; }
            art_mat_set(p,i,j,value); art_mat_set(p,j,i,value);
        }
    }
}
void art_root_solve(uint tree)
{
    ArticulationTree t=art_trees[tree]; uint b=t.range.z;
    if(t.rootMode.x==1u) {
#ifndef ARTICULATION_DYNAMICS
        ArticulationOutput value=art_output[t.range.w];
        value.supportAngular=float4(art_vec(b,9u,0),art_vec(b,9u,1),art_vec(b,9u,2),0);
        value.supportLinear=float4(art_vec(b,9u,3),art_vec(b,9u,4),art_vec(b,9u,5),0);
        art_output[t.range.w]=value;
#endif
        return;
    }
    // Only a root6x6 block. In-place unit-lower L and diagonal D.
    for(uint k=0u;k<6u;++k) {
        float d=art_mat(b,k,k);
        for(uint j=0u;j<k;++j) d-=art_mat(b,k,j)*art_mat(b,k,j)*art_mat(b,j,j);
        if(!(d>0) || !isfinite(d)) { art_failure(tree,ArtNumericalFailure,13u); return; }
        art_mat_set(b,k,k,d);
        for(uint i=k+1u;i<6u;++i) {
            float value=art_mat(b,i,k);
            for(uint j=0u;j<k;++j) value-=art_mat(b,i,j)*art_mat(b,k,j)*art_mat(b,j,j);
            art_mat_set(b,i,k,value/d);
        }
    }
    for(uint i=0u;i<6u;++i) {
        float value=-art_vec(b,9u,i);
        for(uint j=0u;j<i;++j) value-=art_mat(b,i,j)*art_vec(b,19u,j);
        art_vec_set(b,19u,i,value);
    }
    for(uint i=0u;i<6u;++i) art_vec_set(b,19u,i,art_vec(b,19u,i)/art_mat(b,i,i));
    for(uint rev=0u;rev<6u;++rev) {
        uint i=5u-rev; float value=art_vec(b,19u,i);
        for(uint j=i+1u;j<6u;++j) value-=art_mat(b,j,i)*art_vec(b,19u,j);
        if(!isfinite(value)) { art_failure(tree,ArtNumericalFailure,14u); return; }
        art_vec_set(b,19u,i,value);
    }
    art_scratch[b+17u]=asuint(float4(art_vec(b,19u,0),art_vec(b,19u,1),art_vec(b,19u,2),0));
    art_scratch[b+18u]=asuint(float4(art_vec(b,19u,3),art_vec(b,19u,4),art_vec(b,19u,5),0));
}
void art_forward_link(uint tree,uint local)
{
    ArticulationTree t=art_trees[tree]; ArticulationLink link=art_links[t.range.x+local]; uint b=t.range.z+ArtScratchWords*local;
    float q=0;
    if(local!=0u) {
        uint p=t.range.z+ArtScratchWords*link.topology.y; float3 r=asfloat(art_scratch[b+15u]).xyz;
        if(link.topology.w!=JointFixed) {
            q=art_get(b,16u,1u);
            for(uint i=0u;i<6u;++i) q-=art_vec(b,11u,i)*art_recovery_base(p,r,b,i);
            q/=art_get(b,16u,0u);
        }
        float3 w=float3(0),v=float3(0);
        for(uint i=0u;i<6u;++i) {
            float value=art_recovery_base(p,r,b,i)+art_vec(b,13u,i)*q;
            if(!isfinite(value)) { art_failure(tree,ArtNumericalFailure,15u); return; }
            if(i<3u) w[i]=value; else v[i-3u]=value;
        }
        art_scratch[b+17u]=asuint(float4(w,0)); art_scratch[b+18u]=asuint(float4(v,0));
    }
#ifdef ARTICULATION_DYNAMICS
    dyn_recovered(tree,local,q);
#else
    ArticulationOutput output=art_output[t.range.w+local];
    output.angularJoint=float4(asfloat(art_scratch[b+17u]).xyz,q);
    output.linear=float4(asfloat(art_scratch[b+18u]).xyz,0);
    art_output[t.range.w+local]=output;
#endif
}
