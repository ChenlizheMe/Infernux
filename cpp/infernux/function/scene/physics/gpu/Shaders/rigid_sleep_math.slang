// GPU sleep geometry/time law: native Body/GetSleepTestPoints,
// Sphere::EncapsulatePoint and MotionProperties timer laws. No dispatch/solver.
#include "../SleepTypes.h"
#include "rigid_math.slang"

void sleep_test_points(BodyState body,float3 extent,out float3 p0,out float3 p1,out float3 p2)
{
    p0=body.positionCOM.xyz;
    // Exactly Vec3::GetLowestComponentIndex tie ordering, then Body.inl156.
    uint lowest=extent.x<extent.y?(extent.z<extent.x?2u:0u):(extent.z<extent.y?2u:1u);
    float3 axis1=lowest==0u?float3(0,extent.y,0):float3(extent.x,0,0);
    float3 axis2=lowest==2u?float3(0,extent.y,0):float3(0,0,extent.z);
    p1=p0+jph_rotate(body.rotation,axis1);
    p2=p0+jph_rotate(body.rotation,axis2);
}
void sleep_reset(inout BodySleepRuntime state,float3 p0,float3 p1,float3 p2)
{
    state.sphere0=float4(p0,0);state.sphere1=float4(p1,0);state.sphere2=float4(p2,0);
    state.timing.x=0u;state.state.y=1u;state.state.zw=uint2(0u);
}
void sleep_encapsulate(inout float4 sphere,float3 point)
{
    float3 delta=point-sphere.xyz;float d2=dot(delta,delta);
    if(d2>sphere.w*sphere.w) {
        float d=sqrt(d2);float radius=.5f*(sphere.w+d);
        sphere.xyz+=(radius-sphere.w)/d*delta;sphere.w=radius;
    }
}
// Return0/1 native CanSleep vote,2 invalid input/state. The caller must supply
// the actual simulated interval, never Host/Work time or a dispatch count.
uint sleep_update_vote(BodySleepAuthor author,inout BodySleepRuntime state,
                       BodyState body,BodySleepParameters parameters)
{
    float dt=parameters.timing.x,timeBefore=parameters.timing.y;
    float maxMovement=parameters.timing.z*timeBefore;
    if(state.state.y!=1u || !(dt>=0) || !(timeBefore>=0) || !(parameters.timing.z>0)
       || !all(isfinite(parameters.timing.xyz)) || !all(isfinite(author.rootHalfExtent.xyz))
       || any(author.rootHalfExtent.xyz<0) || !all(isfinite(body.positionCOM.xyz))
       || !all(isfinite(body.rotation))) return 2u;
    state.state.z=0u;
    if((author.policy.x&1u)==0u || (author.policy.x&2u)!=0u) return 0u;
    float3 p0,p1,p2;sleep_test_points(body,author.rootHalfExtent.xyz,p0,p1,p2);
    sleep_encapsulate(state.sphere0,p0);
    if(state.sphere0.w>maxMovement) {sleep_reset(state,p0,p1,p2);return 0u;}
    sleep_encapsulate(state.sphere1,p1);
    if(state.sphere1.w>maxMovement) {sleep_reset(state,p0,p1,p2);return 0u;}
    sleep_encapsulate(state.sphere2,p2);
    if(state.sphere2.w>maxMovement) {sleep_reset(state,p0,p1,p2);return 0u;}
    if(!all(isfinite(state.sphere0)) || !all(isfinite(state.sphere1))
       || !all(isfinite(state.sphere2))) return 2u;
    float quietTime=asfloat(state.timing.x)+dt;
    if(!isfinite(quietTime)) return 2u;
    state.timing.x=asuint(quietTime);
    state.state.z=quietTime>=timeBefore?1u:0u;
    return state.state.z;
}
