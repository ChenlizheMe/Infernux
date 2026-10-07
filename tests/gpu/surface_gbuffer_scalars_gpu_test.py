import argparse,json,tempfile
from pathlib import Path
import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel,InxMaterial,SceneManager,Vector3
from infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline
from infernux.renderstack.pipeline_compiler import compile_pipeline_definition
from infernux.renderstack.pipeline_dsl import Path as RenderPath, PipelineBuilder

SURFACE='''#version 450
ShaderInfo { Name "Tutorial Surface Scalar Contract" ShadingModel PBR CastShadows Off ReceiveShadows Off
 Properties { Float scalar0=0.375 Float scalar1=0.625 } }
void surface(out SurfaceData s) { s=InitSurfaceData(); s.shadingParam0=material.scalar0; s.shadingParam1=material.scalar1; }
'''
CONSUMER='''#version 450
ShaderInfo { Name "Tutorial Scalar GBuffer Consumer" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D albedoTex Texture2D materialTex Texture2D emissionTex }
 Inputs { Float2 inUV } Outputs { Float4 outColor } }
void main() {
 ivec2 p=ivec2(gl_FragCoord.xy);
 if(texelFetch(albedoTex,p,0).a<0.9) { outColor=vec4(0,0,0,1);return; }
 vec2 v=vec2(texelFetch(materialTex,p,0).a,texelFetch(emissionTex,p,0).a);
 if(all(equal(v,vec2(0.375,0.625)))) outColor=vec4(0,1,0,1);
 else if(all(equal(v,vec2(-2.25,4.5)))) outColor=vec4(1,0,0,1);
 else if(all(equal(v,vec2(0,0)))) outColor=vec4(0,0,1,1);
 else outColor=vec4(1,0,1,1);
}
'''
class ScalarPipeline(DefaultDeferredPipeline):
    name='Tutorial Canonical GBuffer Scalars'
    def __init__(self, route):
        super().__init__()
        self.route = route
    def define_topology(self,g):
        self.builds += 1
        if self.route == 'builtin':
            super().define_topology(g)
            albedo = g.get_texture('gbuffer_albedo')
            material = g.get_texture('gbuffer_material')
            emission = g.get_texture('gbuffer_emission')
        else:
            builder = PipelineBuilder()
            builder.frame(hdr=True, msaa=1)
            builder.opaque().deferred(fallback=RenderPath.FORWARD_PLUS)
            compile_pipeline_definition(builder.build(), g, pipeline=self)
            lighting, = [p for p in g._passes if p._action == 'fullscreen_quad' and p._shader_name == 'Deferred Lighting']
            albedo = g.get_texture(lighting._input_bindings['gAlbedo'])
            material = g.get_texture(lighting._input_bindings['gMaterial'])
            emission = g.get_texture(lighting._input_bindings['gEmission'])
        target = g.get_texture('color')
        p=g.add_pass('CheckScalarStorage').write_color(target)
        for key,texture in [('albedoTex',albedo),('materialTex',material),('emissionTex',emission)]:p.set_texture(key,texture)
        p.fullscreen_quad('Tutorial Scalar GBuffer Consumer')
        g.set_output(target)

parser = argparse.ArgumentParser()
parser.add_argument('--path', choices=('builtin', 'dsl'), default='builtin')
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix='infernux-gbuffer-scalar-contract-') as folder:
    project=Path(folder)
    for name in ('Assets','Packages','ProjectSettings'):(project/name).mkdir()
    for name,text in [('Surface.frag',SURFACE),('Consumer.frag',CONSUMER)]: (project/'Assets'/name).write_text(text,encoding='ascii')
    mesh=project/'Assets/Quad.obj'
    mesh.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n',encoding='ascii')
    frontend=inx.Engine();native=frontend.get_native_engine();console=ConsolePanel();pipeline=ScalarPipeline(args.path);pipeline.builds=0;pipeline.shadow_resolution=256
    proof={'scope':'Actual production Deferred GBuffer attachments preserve both model-defined scalars, including signed/HDR values, zeros and exact restore; no custom attachment formats.', 'path':args.path, 'phases':[]};failures=[];complete=False
    try:
        frontend.init_renderer(160,120,str(project));frontend.resize_game_render_target(160,120)
        native.set_scene_view_visible(False);native.set_editor_fps_cap(240);native.set_editor_idle_fps(0)
        database=frontend.get_asset_database()
        for path in (mesh,project/'Assets/Surface.frag',project/'Assets/Consumer.frag'):
            imported=AssetManager.import_asset(str(path),database=database);assert imported,imported.error
            if path==mesh:mesh_guid=imported.guid
        scene=SceneManager.instance().get_active_scene()
        for obj in scene.get_root_objects():scene.destroy_game_object(obj)
        camera=scene.create_game_object('Scalar Camera');camera.transform.position=Vector3(0,0,-5);camera.add_component('Camera')
        obj=scene.create_game_object('Scalar Quad');renderer=obj.add_component('MeshRenderer');renderer.set_mesh_asset_guid(mesh_guid)
        material=InxMaterial.create_default_lit();material.vert_shader_name='Standard';material.frag_shader_name='Tutorial Surface Scalar Contract';renderer.set_material(0,material)
        frontend.set_render_pipeline(pipeline);native.set_game_camera_enabled(True)
        frame=0;changed=0;phase=0;ticket=None;initial=None
        def after_draw():
            global frame,changed,phase,ticket,initial,complete
            try:
                frame+=1
                if ticket is None and frame>=changed+8:ticket=frontend.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels=ticket.result_numpy().copy().astype(np.float32);rgb=pixels[...,:3]
                    expected=((0,1,0),(1,0,0),(0,0,1),(0,1,0))[phase]
                    good=np.all(np.abs(rgb-np.asarray(expected))<.01,axis=-1);bad=(rgb[...,0]>.9)&(rgb[...,2]>.9)
                    record={'phase':('default_scalars','signed_hdr_scalars','zero_scalars','restored')[phase],'covered':int(good.sum()),'mismatch':int(bad.sum()),'builds':pipeline.builds}
                    assert good.sum()>100 and not bad.any() and pipeline.builds==1,record
                    issues = [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                    assert not issues, issues
                    if phase==0:initial=pixels
                    elif phase==3:np.testing.assert_array_equal(pixels,initial)
                    proof['phases'].append(record);print(record,flush=True)
                    if phase==3:complete=True;proof['passed']=True;native.exit();return
                    phase+=1
                    a,b=((-2.25,4.5),(0,0),(.375,.625))[phase-1]
                    material.set_float('scalar0',a);material.set_float('scalar1',b)
                    changed=frame;ticket=None
                if frame>110:raise AssertionError('GBuffer scalar GPU audit did not complete')
            except BaseException as exc:failures.append(exc);native.exit()
        native.set_post_draw_callback(after_draw);native.set_play_mode_rendering(True);native.run()
        if failures:raise failures[0]
        assert complete
    finally:frontend.set_render_pipeline(None);pipeline.dispose();native.cleanup()
    print(json.dumps(proof,indent=2),flush=True)
    print('PASS canonical GBuffer scalar ranges and exact restore',flush=True)
