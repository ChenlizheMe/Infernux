// Exact typed schema readers; resources supplied by each artifact.
void art_load(JointDirectory d,out JointFixedAuthor a,out JointFixedState s) {
    a.bodies=d.bodies;a.identity=d.identity;uint p=d.layout.x;
#define SF_F(name) a.name=asfloat(art_joints[p++]);
#define SF_U(name) a.name=art_joints[p++];
    JPH_FIXED_AUTHOR_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
    p=d.layout.y;
#define SF_F(name) s.name=asfloat(art_joints[p++]);
#define SF_U(name) s.name=art_joints[p++];
    JPH_FIXED_STATE_FIELDS(SF_F,SF_U)
#undef SF_F
#undef SF_U
}
JointDirectory art_directory(uint row)
{
    uint p=1u+JointDirectoryWords*row;
    JointDirectory d; d.bodies=art_joints[p]; d.identity=art_joints[p+1u]; d.layout=art_joints[p+2u]; return d;
}
void art_load(JointDirectory d,out JointHingeAuthor a,out JointHingeState state)
{
    a.bodies=d.bodies; a.identity=d.identity; uint p=d.layout.x;
#define ART_LOAD_F(name) a.name=asfloat(art_joints[p++]);
#define ART_LOAD_U(name) a.name=art_joints[p++];
    JPH_HINGE_AUTHOR_FIELDS(ART_LOAD_F,ART_LOAD_U)
#undef ART_LOAD_F
#undef ART_LOAD_U
    p=d.layout.y;
#define ART_LOAD_F(name) state.name=asfloat(art_joints[p++]);
#define ART_LOAD_U(name) state.name=art_joints[p++];
    JPH_HINGE_STATE_FIELDS(ART_LOAD_F,ART_LOAD_U)
#undef ART_LOAD_F
#undef ART_LOAD_U
}
void art_load(JointDirectory d,out JointSliderAuthor a,out JointSliderState state)
{
    a.bodies=d.bodies; a.identity=d.identity; uint p=d.layout.x;
#define ART_LOAD_F(name) a.name=asfloat(art_joints[p++]);
#define ART_LOAD_U(name) a.name=art_joints[p++];
    JPH_SLIDER_AUTHOR_FIELDS(ART_LOAD_F,ART_LOAD_U)
#undef ART_LOAD_F
#undef ART_LOAD_U
    p=d.layout.y;
#define ART_LOAD_F(name) state.name=asfloat(art_joints[p++]);
#define ART_LOAD_U(name) state.name=art_joints[p++];
    JPH_SLIDER_STATE_FIELDS(ART_LOAD_F,ART_LOAD_U)
#undef ART_LOAD_F
#undef ART_LOAD_U
}
