import argparse,json,tempfile
from pathlib import Path
import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel,InxMaterial,SceneManager,Vector3
from infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline
from infernux.renderstack.pipeline_compiler import compile_pipeline_definition
from infernux.renderstack.pipeline_dsl import Path as RenderPath, PipelineBuilder

SURFACE = """import infernux.shader as shader


class ScalarSurface(shader.surface):
    shader_id = "tests/gbuffer/scalar_surface"
    cast_shadows = False
    receive_shadows = False
    scalar0 = shader.material.float(default=0.375)
    scalar1 = shader.material.float(default=0.625)

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(
            normal_ws=ctx.normal_ws, shading_param0=self.scalar0, shading_param1=self.scalar1,
        )
"""

CONSUMER = """import infernux.shader as shader
from infernux.shader.lib.surface import decode_model_scalars


class ScalarConsumer(shader.fullscreen):
    shader_id = "tests/gbuffer/scalar_consumer"
    hidden = True
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        packed = shader.load(self.surface_tex, pixel, 0)
        surface_id = packed.x
        if surface_id == 0:
            return shader.vec4(0, 0, 0, 1)
        record = self.surface_records[surface_id * 3]
        if surface_id > 255 or record.x != 1 or record.y == 0 or record.z != 0:
            return shader.vec4(1, 0, 1, 1)
        coverage_id = shader.float(surface_id) / 255.0
        scalars = decode_model_scalars(packed.y)
        if scalars.x == 0.375 and scalars.y == 0.625:
            return shader.vec4(0, 1, 0, coverage_id)
        if scalars.x == -2.25 and scalars.y == 4.5:
            return shader.vec4(1, 0, 0, coverage_id)
        if scalars.x == 0.0 and scalars.y == 0.0:
            return shader.vec4(0, 0, 1, coverage_id)
        return shader.vec4(1, 0, 1, 1)
"""


class ScalarPipeline(DefaultDeferredPipeline):
    name='Tutorial Canonical GBuffer Scalars'
    def __init__(self, route):
        super().__init__()
        self.route = route
    def define_topology(self,g):
        self.builds += 1
        if self.route == 'builtin':
            super().define_topology(g)
        else:
            builder = PipelineBuilder()
            builder.frame(hdr=True, msaa=1)
            builder.opaque().deferred(fallback=RenderPath.FORWARD_PLUS)
            compile_pipeline_definition(builder.build(), g, pipeline=self)
        lighting, = [p for p in g._passes if p._action == 'fullscreen_quad' and p._shader_name == 'infernux/internal/deferred_lighting']
        target = g.get_texture('color')
        p=g.add_pass('CheckScalarStorage').write_color(target)
        p.set_texture('surface_tex', g.get_texture(lighting._input_bindings['surface_tex']))
        p.set_buffer('surface_records', g.get_buffer(lighting._input_bindings['surface_records']))
        p.fullscreen_quad('tests/gbuffer/scalar_consumer')
        g.set_output(target)

parser = argparse.ArgumentParser()
parser.add_argument('--path', choices=('builtin', 'dsl'), default='builtin')
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix='infernux-gbuffer-scalar-contract-') as folder:
    project=Path(folder)
    for name in ('Assets','Packages','ProjectSettings'):(project/name).mkdir()
    for name,text in [('scalar_surface.frag.py',SURFACE),('scalar_consumer.frag.py',CONSUMER)]: (project/'Assets'/name).write_text(text,encoding='ascii')
    mesh=project/'Assets/Quad.obj'
    mesh.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n',encoding='ascii')
    frontend=inx.Engine();native=frontend.get_native_engine();console=ConsolePanel();pipeline=ScalarPipeline(args.path);pipeline.builds=0;pipeline.shadow_resolution=256
    proof={'scope':'Actual production Deferred GBuffer attachments preserve both model-defined scalars, including signed/HDR values, zeros and exact restore; no custom attachment formats.', 'path':args.path, 'phases':[]};failures=[];complete=False
    try:
        frontend.init_renderer(160,120,str(project));frontend.resize_game_render_target(160,120)
        native.set_scene_view_visible(False);native.set_editor_fps_cap(240);native.set_editor_idle_fps(0)
        database=frontend.get_asset_database()
        for path in (mesh,project/'Assets/scalar_surface.frag.py',project/'Assets/scalar_consumer.frag.py'):
            imported=AssetManager.import_asset(str(path),database=database);assert imported,imported.error
            if path==mesh:mesh_guid=imported.guid
        scene=SceneManager.instance().get_active_scene()
        for obj in scene.get_root_objects():scene.destroy_game_object(obj)
        camera=scene.create_game_object('Scalar Camera');camera.transform.position=Vector3(0,0,-5);camera.add_component('Camera')
        material=InxMaterial.create_default_lit();material.vert_shader_name='infernux/standard';material.frag_shader_name='tests/gbuffer/scalar_surface'
        def create_quad(name, x):
            obj = scene.create_game_object(name)
            obj.transform.position = Vector3(x, 0, 0)
            renderer = obj.add_component('MeshRenderer')
            renderer.set_mesh_asset_guid(mesh_guid)
            renderer.set_material(0, material)
            return obj

        left_quad = create_quad('Left Quad', -.7)
        create_quad('Right Quad', .7)
        frontend.set_render_pipeline(pipeline);native.set_game_camera_enabled(True)
        frame=0;changed=0;phase=0;ticket=None;initial=None
        def after_draw():
            global frame,changed,phase,ticket,initial,complete
            try:
                frame+=1
                if ticket is None and frame>=changed+8:ticket=frontend.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels=ticket.result_numpy().copy().astype(np.float32);rgb=pixels[...,:3]
                    expected=((0,1,0),(1,0,0),(0,0,1),(0,1,0),(0,1,0))[phase]
                    good=np.all(np.abs(rgb-np.asarray(expected))<.01,axis=-1);bad=(rgb[...,0]>.9)&(rgb[...,2]>.9)
                    record={'phase':('default_scalars','signed_hdr_scalars','zero_scalars','restored','recreated')[phase],'covered':int(good.sum()),'mismatch':int(bad.sum()),'builds':pipeline.builds}
                    assert good.sum()>100 and not bad.any() and pipeline.builds==1,record
                    surface_ids = np.unique(np.rint(pixels[good, 3] * 255).astype(int))
                    assert len(surface_ids) == 2 and surface_ids[0] > 0, surface_ids
                    record['surfaces'] = surface_ids.tolist()
                    if 0 < phase < 4:
                        np.testing.assert_array_equal(pixels[..., 3], initial[..., 3])
                    elif phase == 4:
                        old_ids = set(proof['phases'][0]['surfaces'])
                        new_ids = set(surface_ids)
                        assert len(new_ids & old_ids) == 1 and min(new_ids - old_ids) > max(old_ids)
                        np.testing.assert_array_equal(pixels[..., :3], initial[..., :3])
                        retained_id, = new_ids & old_ids
                        np.testing.assert_array_equal(
                            np.rint(pixels[..., 3] * 255) == retained_id,
                            np.rint(initial[..., 3] * 255) == retained_id,
                        )
                    issues = [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                    assert not issues, issues
                    if phase==0:initial=pixels
                    elif phase==3:np.testing.assert_array_equal(pixels,initial)
                    proof['phases'].append(record);print(record,flush=True)
                    if phase == 4:
                        complete = True
                        proof['passed'] = True
                        native.exit()
                        return
                    if phase == 3:
                        scene.destroy_game_object(left_quad)
                        create_quad('Left Quad', -.7)
                    else:
                        a,b=((-2.25,4.5),(0,0),(.375,.625))[phase]
                        material.set_float('scalar0',a);material.set_float('scalar1',b)
                    phase+=1
                    changed=frame;ticket=None
                if frame>110:raise AssertionError('GBuffer scalar GPU audit did not complete')
            except BaseException as exc:failures.append(exc);native.exit()
        native.set_post_draw_callback(after_draw);native.set_play_mode_rendering(True);native.run()
        if failures:raise failures[0]
        assert complete
    finally:frontend.set_render_pipeline(None);pipeline.dispose();native.cleanup()
    print(json.dumps(proof,indent=2),flush=True)
    print('PASS canonical GBuffer scalar ranges and exact restore',flush=True)
