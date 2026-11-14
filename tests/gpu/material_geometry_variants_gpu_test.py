"""Independent numeric GPU proof of generated deformation/cutout variants."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, SceneManager, Vector3

VERTEX = '''import infernux.shader as shader


class VariantShift(shader.vert):
    shader_id = "tests/variants/shift"
    shift_x = shader.material.float(default=0.4)
    shift_y = shader.material.float(default=0.2)

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        vertex.position_os += shader.vec3(self.shift_x, self.shift_y, 0.0)
        return vertex
'''
SURFACE = '''import infernux.shader as shader


class VariantCutout(shader.surface):
    shader_id = "tests/variants/cutout"
    shading_model = shader.ref("infernux/shading/unlit")
    cull_mode = "none"
    alpha_clip = 0.5
    depth_test = "less"
    clip_edge = shader.material.float(default=0.5)

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=shader.vec3(1.0, 0.0, 0.0),
                                  alpha=shader.step(self.clip_edge, ctx.uv0.x))
'''
CONSUMER = '''import infernux.shader as shader


class VariantCoverage(shader.fullscreen):
    shader_id = "tests/variants/coverage"
    hidden = True
    color_tex = shader.input("color_data", shader.Color2D)
    normal_tex = shader.input("normal", shader.Color2D)
    base_tex = shader.input("base", shader.Color2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    motion_tex = shader.input("motion", shader.Color2D)
    picking_tex = shader.input("picking", shader.UInt2D)
    expected_object = shader.parameter.float()
    secondary_object = shader.parameter.float()

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        color = shader.load(self.color_tex, pixel, 0).a > 0.9
        normal = shader.load(self.normal_tex, pixel, 0).a > 0.9
        base = shader.load(self.base_tex, pixel, 0).a > 0.9
        depth = shader.load(self.depth_tex, pixel, 0) > 0.0
        identity = shader.load(self.picking_tex, pixel, 0).rg
        picking = shader.all(identity == shader.uvec2(shader.uint(self.expected_object), 0)) or (self.secondary_object > 0.0 and shader.all(identity == shader.uvec2(shader.uint(self.secondary_object), 0)))
        moving = shader.length(shader.load(self.motion_tex, pixel, 0).rg) > 0.00001
        if color != normal or color != base or color != depth or color != picking or (not picking and shader.any(identity != shader.uvec2(0))) or (not color and moving):
            return shader.vec4(1.0, 0.0, 1.0, 1.0)
        if not color:
            return shader.vec4(0.0, 0.0, 0.0, 1.0)
        return shader.vec4(0.0, 1.0, 0.0, 1.0) if moving else shader.vec4(1.0, 0.0, 0.0, 1.0)
'''


class VariantsPipeline(inx.renderstack.RenderPipeline):
    name='Tutorial Vertex Surface Variant GPU'

    def define_topology(self, graph):
        self.builds+=1
        graph.set_msaa_samples(1)
        fmt=inx.rendergraph.Format
        depth=graph.create_texture('depth',format=fmt.D32_SFLOAT,samples=1)
        color=graph.create_texture('color_data',format=fmt.RGBA16_SFLOAT,samples=1)
        graph.add_pass('Color').write_color(color).write_depth(depth).set_clear(color=(0,0,0,0),depth=0.0).draw_renderers(queue_range=(0,2500))
        normal=graph.create_texture('normal',format=fmt.RGBA16_SFLOAT,samples=1)
        base=graph.create_texture('base',format=fmt.RGBA16_SFLOAT,samples=1)
        motion=graph.create_texture('motion',format=fmt.RG16_SFLOAT,samples=1)
        for name,texture,material_pass in (('Normal',normal,'normal'),('BaseColor',base,'base_color'),('Motion',motion,'motion')):
            graph.add_pass(name).read(depth).write_color(texture).set_clear(color=(0,0,0,0)).draw_renderers(queue_range=(0,2500),material_pass=material_pass)
        picking=graph.create_texture('picking',format=fmt.RG32_UINT,samples=1)
        picking_depth=graph.create_texture('picking_depth',format=fmt.D32_SFLOAT,samples=1)
        graph.add_pass('Picking').write_color(picking).write_depth(picking_depth).set_clear(color=(0,0,0,0),depth=0.0).draw_renderers(queue_range=(0,2500),material_pass='picking')
        target=graph.create_texture('color',camera_target=True)
        p=graph.add_pass('CompareVariants').write_color(target)
        for key,texture in (('color_tex',color),('normal_tex',normal),('base_tex',base),('depth_tex',depth),('motion_tex',motion),('picking_tex',picking)):
            p.set_texture(key,texture)
        p.set_param('expected_object',float(self.object_id)).fullscreen_quad('tests/variants/coverage')
        p.set_param('secondary_object',float(self.secondary_id))
        graph.screen_ui_overlay_section(resources={'color'})
        graph.set_output(target)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--instances',type=int,choices=(1,2),default=1)
    parser.add_argument('--skinned',action='store_true')
    parser.add_argument('--proof',type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='infernux-tutorial-variants-') as root:
        project=Path(root)
        for directory in ('Assets','Packages','ProjectSettings'):
            (project/directory).mkdir()
        mesh_path=project/('Assets/Probe.gltf' if args.skinned else 'Assets/Probe.obj')
        if args.skinned:
            from skinned_mesh_gpu_fixture import write_skinned_quad
            write_skinned_quad(mesh_path)
        else:
            mesh_path.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 -1\nf 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n',encoding='ascii')
        for file,content in (('Shift.vert.py',VERTEX),('Cutout.frag.py',SURFACE),('Consumer.frag.py',CONSUMER)):
            (project/'Assets'/file).write_text(content,encoding='ascii')
        frontend=inx.Engine()
        native=frontend.get_native_engine()
        console=ConsolePanel()
        pipeline=VariantsPipeline()
        pipeline.builds=0
        failures,completed=[],False
        proof={'scope':'Independent real GPU masks compare color, normal, base-color, depth and integer picking coverage for material vertex deformation and surface alpha clipping; motion vectors distinguish transform motion from post-hook local changes. No MCP Editor pixel access.', 'instances':args.instances, 'phases':[]}
        try:
            frontend.init_renderer(160,120,str(project))
            frontend.resize_game_render_target(160,120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database=frontend.get_asset_database()
            mesh=AssetManager.import_asset(str(mesh_path),database=database)
            assert mesh,mesh.error
            for file in ('Shift.vert.py','Cutout.frag.py','Consumer.frag.py'):
                result=AssetManager.import_asset(str(project/'Assets'/file),database=database)
                assert result,result.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera=scene.create_game_object('Variant Camera')
            camera.transform.position=Vector3(0,0,-5)
            camera.add_component('Camera')
            material=InxMaterial.create_default_unlit()
            material.vert_shader_name='tests/variants/shift'
            material.frag_shader_name='tests/variants/cutout'
            for key,value in (('shift_x',.4),('shift_y',.2),('clip_edge',.5)):
                material.set_float(key,value)
            objects=[]
            positions=(0,) if args.instances==1 else (-1,1)
            for index,x in enumerate(positions):
                obj=scene.create_game_object(f'Deformed Cutout {index}')
                obj.transform.position=Vector3(x,0,0)
                renderer=obj.add_component('SkinnedMeshRenderer' if args.skinned else 'MeshRenderer')
                if args.skinned:
                    renderer.set_source_model_guid(mesh.guid)
                    renderer.submit_animation_pose('Scale',0.,0.,loop=False)
                else:
                    renderer.set_mesh_asset_guid(mesh.guid)
                renderer.set_material(0,material)
                objects.append(obj)
            pipeline.object_id=objects[0].id
            pipeline.secondary_id=objects[1].id if args.instances==2 else 0
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            phases=('cutout','solid','cutout_restored','vertex_property_changed','object_motion','procedural_local_motion','static_restored')
            if args.skinned:
                phases+=('bone_motion','bone_stopped')
                proof['scope']+=' Imported one-joint animation covers moving bones and held poses after animation stops.'
            phase,frames,changed,ticket=0,0,0,None
            initial=None
            initial_pixels=None
            last_image=None

            def after_draw():
                nonlocal phase,frames,changed,ticket,completed,initial,initial_pixels,last_image
                try:
                    frames+=1
                    if ticket is None and frames>=changed+8:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32)
                        rgb=pixels[...,:3]
                        red=(rgb[...,0]>.9)&(rgb[...,1]<.05)&(rgb[...,2]<.05)
                        green=(rgb[...,1]>.9)&(rgb[...,0]<.05)&(rgb[...,2]<.05)
                        bad=(rgb[...,0]>.9)&(rgb[...,2]>.9)
                        coverage=red|green
                        record={'phase':phases[phase],'covered':int(coverage.sum()),'moving':int(green.sum()),'mismatch':int(bad.sum()),'builds':pipeline.builds}
                        record['game_draw_calls']=native.renderer_frame_snapshot['game_draw_call_count']
                        if coverage.any():
                            record['centroid_x']=float(np.where(coverage)[1].mean())
                        print(record,flush=True)
                        assert not bad.any() and coverage.sum()>100,record
                        assert pipeline.builds==1,record
                        assert not [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                        if phase==0:
                            initial=record
                            initial_pixels=pixels
                            assert record['centroid_x']>90,record
                        elif phase==1:
                            assert record['covered']>initial['covered']*1.8,record
                        elif phase in (2,6):
                            np.testing.assert_array_equal(pixels,initial_pixels)
                        elif phase==3:
                            assert record['centroid_x']<initial['centroid_x']-10,record
                        if phase in (4,7):
                            assert green.sum()>coverage.sum()*.9,record
                        else:
                            assert not green.any(),record
                        if phase==5:
                            assert last_image is not None and not np.array_equal(last_image,pixels),record
                        proof['phases'].append(record)
                        last_image=pixels
                        if phase==len(phases)-1:
                            proof['passed']=True
                            if args.proof:
                                args.proof.write_text(json.dumps(proof,indent=2),encoding='utf-8')
                            print(json.dumps(proof,indent=2),flush=True)
                            completed=True
                            native.exit()
                            return
                        phase+=1
                        if phase==1:
                            material.set_float('clip_edge',0.)
                        elif phase==2:
                            material.set_float('clip_edge',.5)
                        elif phase==3:
                            material.set_float('shift_x',-.4)
                        elif phase==6:
                            for obj,x in zip(objects,positions):
                                obj.transform.position=Vector3(x,0,0)
                            material.set_float('shift_x',.4)
                        changed,ticket=frames,None
                    if phase==4:
                        for index,obj in enumerate(objects):
                            position=obj.transform.position
                            obj.transform.position=Vector3(position.x+(.03 if index==0 else -.03),0,0)
                    elif phase==5:
                        material.set_float('shift_x',material.get_float('shift_x')+.03)
                    if phase==7:
                        for obj in objects:
                            seconds=(frames-changed)*.02
                            renderer=obj.get_component('SkinnedMeshRenderer')
                            renderer.submit_animation_pose('Scale',seconds-.01,(seconds-.01)/3.,loop=False)
                            renderer.submit_animation_pose('Scale',seconds,seconds/3.,loop=False)
                    if frames>250:
                        raise AssertionError('Variant GPU audit timed out')
                except BaseException as error:
                    failures.append(error)
                    native.exit()
            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert completed
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()


if __name__=='__main__':
    main()
