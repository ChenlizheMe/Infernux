// Shared unit-rotation interpretation; original root BodyState is never normalized in storage.
struct SfJoint { float3 a1;float3 a2;float4 f1;float4 f2;uint kind; };
float4 sf_conjugate(float4 q) { return float4(-q.xyz,q.w); }
bool sf_quaternion(float4 q) {
    return all(isfinite(q)) && abs(dot(q,q)-1.0f)<=1e-4f;
}
bool sf_state(BodyState s) {
    return sf_quaternion(s.rotation) && all(isfinite(s.positionCOM))
        && all(isfinite(s.linearVelocity)) && all(isfinite(s.angularVelocity))
        && s.positionCOM.w==0 && s.linearVelocity.w==0 && s.angularVelocity.w==0;
}
BodyState sf_forward(BodyState parent,SfJoint j,float2 coordinate) {
    BodyState child;float4 frame=normalize(jph_quat_mul(parent.rotation,j.f1));
    float4 turn=j.kind==JointHinge?float4(0,0,sin(.5f*coordinate.x),cos(.5f*coordinate.x)):float4(0,0,0,1);
    child.rotation=normalize(jph_quat_mul(jph_quat_mul(frame,turn),sf_conjugate(j.f2)));
    float3 axis=jph_rotate(frame,j.kind==JointSlider?float3(1,0,0):float3(0,0,1));
    float3 arm2=jph_rotate(child.rotation,j.a2);
    child.positionCOM=float4(parent.positionCOM.xyz+jph_rotate(normalize(parent.rotation),j.a1)-arm2
        +(j.kind==JointSlider?axis*coordinate.x:float3(0)),0);
    float3 relative_omega=j.kind==JointHinge?axis*coordinate.y:float3(0);
    child.angularVelocity=float4(parent.angularVelocity.xyz+relative_omega,0);
    child.linearVelocity=float4(parent.linearVelocity.xyz+cross(parent.angularVelocity.xyz,child.positionCOM.xyz-parent.positionCOM.xyz)
        -cross(relative_omega,arm2)+(j.kind==JointSlider?axis*coordinate.y:float3(0)),0);
    return child;
}
