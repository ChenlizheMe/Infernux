"""Compare the actual Motion Blur shader to its input in the same GPU frame."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.engine.path_utils import resolved_path
from infernux.lib import ConsolePanel, SceneManager, Vector3
from infernux.renderstack.resource_bus import ResourceBus

ROOT=Path(resolved_path(__file__)).parents[2]
sys.path.append(str(ROOT/'tests/gpu'))
from skinned_mesh_gpu_fixture import write_skinned_quad

VERTEX = """import infernux.shader as shader

class Wave(shader.vert):
    shader_id = "tests/vertex/motion_wave"

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        vertex.position_os.y += shader.sin(vertex.position_os.x * 4.0) * 0.1
        return vertex
"""
SURFACE = """import infernux.shader as shader

class Pattern(shader.surface):
    shader_id = "tests/motion/pattern"
    shading_model = shader.ref("infernux/shading/unlit")
    cull_mode = "none"

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        stripe = shader.step(0.5, shader.frac(ctx.uv0.x * 12.0))
        color = shader.lerp(shader.vec3(0.05, 0.2, 0.5), shader.vec3(1.0, 0.7, 0.1), stripe)
        return shader.SurfaceData(albedo=color, normal_ws=ctx.normal_ws)
"""
COMPARE = """import infernux.shader as shader

class Compare(shader.fullscreen):
    shader_id = "tests/motion/compare"
    original_tex = shader.input("original", shader.Color2D)
    blurred_tex = shader.input("blurred", shader.Color2D)
    motion_tex = shader.input("motion", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        original = shader.load(self.original_tex, pixel, 0)
        blurred = shader.load(self.blurred_tex, pixel, 0)
        motion = shader.load(self.motion_tex, pixel, 0).xy
        changed = shader.any(shader.abs(original.rgb - blurred.rgb) > shader.vec3(0.005))
        moving = shader.length(motion) > 0.00001
        return shader.vec4(1.0 if changed else 0.0, 1.0 if moving else 0.0,
                           1.0 if original.a > 0.9 else 0.0, 1.0)
"""

class BlurPipeline(inx.renderstack.RenderPipeline):
    name='Tutorial Motion Blur Same Frame'
    def define_topology(self,graph):
        self.builds+=1
        graph.set_msaa_samples(1)
        fmt=inx.rendergraph.Format
        source=graph.create_texture('original',format=fmt.RGBA16_SFLOAT,samples=1)
        depth=graph.create_texture('depth',format=fmt.D32_SFLOAT,samples=1)
        motion=graph.create_texture('motion',format=fmt.RG16_SFLOAT,samples=1)
        graph.add_pass('Original').write_color(source).write_depth(depth).set_clear(color=(0,0,0,0),depth=0.0).draw_renderers(queue_range=(0,2500))
        graph.add_pass('Motion').read(depth).write_color(motion).set_clear(color=(0,0,0,0)).draw_renderers(queue_range=(0,2500),material_pass='motion')
        bus=ResourceBus()
        for name,handle in (('color',source),('depth',depth),('motion',motion)):
            bus.set(name,handle)
        effect=inx.renderstack.MotionBlurEffect()
        effect.intensity=1.
        effect.max_blur_pixels=32.
        effect.depth_rejection=1.
        effect.setup_passes(graph,bus)
        target=graph.create_texture('color',camera_target=True)
        graph.add_pass('CompareSameFrame').set_texture('original_tex',source).set_texture('blurred_tex',bus.get('color')).set_texture('motion_tex',motion).write_color(target).fullscreen_quad('tests/motion/compare')
        graph.screen_ui_overlay_section(resources={'color'})
        graph.set_output(target)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skinned',action='store_true')
    parser.add_argument('--proof',type=Path)
    args=parser.parse_args()
    proof={'package':inx.__file__,'skinned':args.skinned,'phases':[], 'scope':'Independent temporary project. Actual built-in Motion Blur output is compared with its source color in the same GPU frame. Two renderers share registered mesh and material. No live Editor pixels.'}
    with tempfile.TemporaryDirectory(prefix='infernux-motion-blur-tutorial-') as folder:
        project=Path(folder)
        for directory in ('Assets','Packages','ProjectSettings'):(project/directory).mkdir()
        mesh_path=project/('Assets/Animated.gltf' if args.skinned else 'Assets/Shared.obj')
        if args.skinned:write_skinned_quad(mesh_path)
        else:mesh_path.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 -1\nf 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n',encoding='ascii')
        for name,text in (('wave.vert.py',VERTEX),('pattern.frag.py',SURFACE),('compare.frag.py',COMPARE)):
            (project/'Assets'/name).write_text(text,encoding='utf-8')
        frontend=inx.Engine()
        native=frontend.get_native_engine()
        console=ConsolePanel()
        pipeline=BlurPipeline()
        pipeline.builds=0
        errors=[]
        completed=False
        try:
            frontend.init_renderer(320,240,str(project))
            frontend.resize_game_render_target(320,240)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database=frontend.get_asset_database()
            mesh=AssetManager.import_asset(str(mesh_path),database=database)
            assert mesh,mesh.error
            for name in ('wave.vert.py','pattern.frag.py','compare.frag.py'):
                result=AssetManager.import_asset(str(project/'Assets'/name),database=database)
                assert result,result.error
            scene=SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():scene.destroy_game_object(obj)
            camera=scene.create_game_object('Motion Camera')
            camera.transform.position=Vector3(0,0,-5)
            camera.add_component('Camera')
            material=inx.Material.from_shader(str(project/'Assets/pattern.frag.py'))
            material.vert_shader_name='tests/vertex/motion_wave'
            objects=[]
            for index,x in enumerate((-1.,1.)):
                obj=scene.create_game_object('Moving '+str(index))
                obj.transform.position=Vector3(x,0,0)
                renderer=obj.add_component('SkinnedMeshRenderer' if args.skinned else 'MeshRenderer')
                if args.skinned:
                    renderer.set_source_model_guid(mesh.guid)
                    renderer.submit_animation_pose('Scale',0.,0.,loop=False)
                else:renderer.set_mesh_asset_guid(mesh.guid)
                renderer.set_material(material)
                objects.append(obj)
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            phases=['fixed_transform_wave','transform_motion','held_transform']
            if args.skinned:phases+=['bone_motion','held_bones']
            phase=frames=changed=0
            ticket=None

            def after_draw():
                nonlocal phase,frames,changed,ticket,completed
                try:
                    frames+=1
                    if ticket is None and frames>=changed+10:
                        ticket=frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels=ticket.result_numpy().copy().astype(np.float32)
                        rgb=pixels[...,:3]
                        difference=rgb[...,0]>.9
                        motion=rgb[...,1]>.9
                        coverage=rgb[...,2]>.9
                        record={'phase':phases[phase],'blurred_pixels':int(difference.sum()),'moving_pixels':int(motion.sum()),'covered_pixels':int(coverage.sum()),'builds':pipeline.builds}
                        assert coverage.sum()>500 and pipeline.builds==1,record
                        if phases[phase] in ('transform_motion','bone_motion'):
                            assert motion.sum()>100 and difference.sum()>50,record
                        else:
                            assert not motion.any() and not difference.any(),record
                        assert not [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL')]
                        proof['phases'].append(record)
                        print('PASS '+json.dumps(record),flush=True)
                        if phase==len(phases)-1:
                            proof['passed']=True
                            if args.proof:
                                args.proof.write_text(json.dumps(proof,indent=2),encoding='utf-8')
                            completed=True
                            native.exit()
                            return
                        phase+=1
                        changed=frames
                        ticket=None
                    if phases[phase]=='transform_motion':
                        for index,obj in enumerate(objects):
                            pos=obj.transform.position
                            obj.transform.position=Vector3(pos.x+(.025 if index==0 else -.025),0,0)
                    if phases[phase]=='bone_motion':
                        for obj in objects:
                            seconds=(frames-changed)*.02
                            renderer=obj.get_component('SkinnedMeshRenderer')
                            renderer.submit_animation_pose('Scale',seconds,seconds/3.,loop=False)
                    if frames>240:raise AssertionError('Same-frame Motion Blur audit timed out')
                except BaseException as error:
                    errors.append(error)
                    native.exit()
            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if errors:raise errors[0]
            assert completed
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()

if __name__=='__main__':main()
