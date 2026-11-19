"""Actual tutorial transparent sorting, live blend changes, and opaque depth."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from renderer_fixture import GameFrames
from infernux import Engine
from infernux.core.assets import AssetManager
from infernux.lib import CameraClearFlags, ConsolePanel, InxMaterial, SceneManager, Vector3, vec4f
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline
from infernux.renderstack.forward_parameters import MSAASamples
from infernux.renderstack.default_forward_plus_pipeline import DefaultForwardPlusPipeline
from infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline

SHADER = '''import infernux.shader as shader
class Transparent(shader.surface):
    shader_id = "Tutorial Transparent"
    shading_model = shader.ref("infernux/shading/unlit")
    surface_type = "transparent"
    cull_mode = "none"
    blend = "MODE"
    cast_shadows = False
    receive_shadows = False
    base_color = shader.material.color(default=(1.0, 0.0, 0.0, 0.5))
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.base_color.rgb MULTIPLY, alpha=self.base_color.a)
'''

def encode(value):
    value = np.asarray(value)
    return np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1/2.4) - .055)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', choices=('forward','forward_plus','deferred'), default='forward')
    parser.add_argument('--samples',type=int,choices=(1,4),default=1)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='infernux-transparency-audit-') as root:
        project = Path(root)
        for directory in ('Assets','Packages','ProjectSettings'):
            (project/directory).mkdir()
        mesh = project/'Assets/Quad.obj'
        mesh.write_text('v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n'
                        'vn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n', encoding='ascii')
        shader = project/'Assets/Transparent.frag.py'
        def source(mode):
            text = SHADER.replace('MULTIPLY','* self.base_color.a' if mode=='Premultiplied' else '')
            if mode == 'Alpha':
                return text.replace('    blend = "MODE"\n','')
            if mode == 'Opaque':
                return text.replace('surface_type = "transparent"','surface_type = "opaque"').replace('MODE','off')
            return text.replace('MODE', {'Premultiplied':'premultiply','Additive':'additive','Alpha':'alpha'}[mode])
        shader.write_text(source('Alpha'),encoding='ascii')
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = {'forward':DefaultForwardPipeline,'forward_plus':DefaultForwardPlusPipeline,
                    'deferred':DefaultDeferredPipeline}[args.path]()
        failures, completed = [],False
        try:
            frontend.init_renderer(96,72,str(project))
            frontend.resize_game_render_target(96,72)
            engine.set_scene_view_visible(False)
            engine.set_editor_fps_cap(240)
            engine.set_editor_idle_fps(0)
            db = frontend.get_asset_database()
            imported = AssetManager.import_asset(str(mesh),database=db)
            assert imported, imported.error
            assert AssetManager.import_asset(str(shader),database=db)
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Camera')
            camera.transform.position = Vector3(0,0,-6)
            camera_component = camera.add_component('Camera')
            camera_component.background_color = vec4f(0,0,0,0)
            camera_component.clear_flags = CameraClearFlags.SolidColor
            renderers = []
            for name,z,rgb in (('Red',-.5,(1,0,0)),('Green',.5,(0,1,0))):
                obj = scene.create_game_object(name)
                obj.transform.position = Vector3(0,0,z)
                renderer = obj.add_component('MeshRenderer')
                renderer.set_mesh_asset_guid(imported.guid)
                material = InxMaterial()
                material.vert_shader_name = 'infernux/standard'
                material.frag_shader_name = 'Tutorial Transparent'
                material.set_color('base_color',*rgb,.5)
                renderer.set_material(0,material)
                renderers.append(renderer)
            blocker = scene.create_game_object('Opaque Blue')
            blocker.transform.position = Vector3(0,0,-1.0)
            blocker_renderer = blocker.add_component('MeshRenderer')
            blocker_renderer.set_mesh_asset_guid(imported.guid)
            blue = InxMaterial.create_default_unlit()
            blue.set_color('base_color',0,0,1,1)
            blocker_renderer.set_material(0,blue)
            blocker_renderer.enabled = False
            pipeline.msaa_samples = MSAASamples(args.samples)
            pipeline.shadow_resolution = 256
            frontend.set_render_pipeline(pipeline)
            engine.set_game_camera_enabled(True)
            phases = ('alpha_red_near','alpha_green_near','alpha_restored',
                      'premultiplied','additive','opaque_surface','opaque_occlusion','restored')
            phase,frame,changed,ticket = 0,0,0,None
            images = {}
            def edit(mode):
                shader.write_text(source(mode),encoding='ascii')
                result = AssetManager.reimport_asset(str(shader),database=db)
                assert result,result.error
            game_frames = GameFrames(engine)

            def after_draw():
                nonlocal phase,frame,changed,ticket,completed
                try:
                    if not game_frames.advance():
                        return
                    frame += 1
                    if ticket is None and frame >= changed+8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        images[phases[phase]] = pixels
                        center = pixels[32:40,44:52].mean(axis=(0,1))
                        print(args.path,args.samples,phases[phase],center.tolist(),flush=True)
                        for renderer in renderers:
                            state=renderer.get_material(0).serialize_document()['renderState']
                            assert state['renderQueue'] == (2000 if phase==5 else 3000),state
                            assert state['depthWriteEnable'] == (phase==5),state
                        errors = [e for e in console._get_visible_log_snapshot(2000)
                                  if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not errors,errors
                        expected = ((.5,.25,0),(.25,.5,0),(.5,.25,0),(.5,.25,0),
                                    (1,1,0),(1,0,0),(0,0,1),(.5,.25,0))[phase]
                        np.testing.assert_allclose(center[:3],encode(expected),atol=.005,
                                                   err_msg=f'{args.path}: {phases[phase]}')
                        # Deferred GBuffer alpha is opaque coverage; transparent
                        # source-over alpha is shared with both Forward paths.
                        opaque_alpha = 1.0 if args.path == 'deferred' else .5
                        expected_alpha = (.75, .75, .75, .75, 1., opaque_alpha, 1., .75)[phase]
                        np.testing.assert_allclose(center[3], expected_alpha, atol=.001)
                        expected_samples = 1 if args.path == 'deferred' else args.samples
                        assert engine.msaa_state['game_requested_samples'] == expected_samples
                        if phase == len(phases)-1:
                            for name in ('alpha_restored','premultiplied','restored'):
                                np.testing.assert_allclose(images[name],images['alpha_red_near'],atol=.005)
                            completed = True
                            print('PASS camera-relative sorting/live blends/opaque depth',flush=True)
                            engine.exit()
                            return
                        phase += 1
                        if phase == 1:
                            camera.transform.position = Vector3(0,0,6)
                            camera.transform.euler_angles = Vector3(0,180,0)
                        elif phase == 2:
                            camera.transform.position = Vector3(0,0,-6)
                            camera.transform.euler_angles = Vector3(0,0,0)
                        elif phase == 3:
                            edit('Premultiplied')
                        elif phase == 4:
                            edit('Additive')
                        elif phase == 5:
                            edit('Opaque')
                        elif phase == 6:
                            edit('Alpha')
                            blocker_renderer.enabled = True
                        else:
                            blocker_renderer.enabled = False
                        ticket,changed = None,frame
                    if frame > 210:
                        raise AssertionError('Transparent audit timeout')
                except BaseException as exc:
                    failures.append(exc)
                    engine.exit()
            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert completed
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            engine.cleanup()

if __name__=='__main__':
    main()
