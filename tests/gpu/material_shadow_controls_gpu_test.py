"""Actual fragment tutorial render-state branches, using real GPU readback."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
from infernux import Engine
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, LightShadows, SceneManager, Vector3
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline, MSAASamples
from infernux.renderstack.default_forward_plus_pipeline import DefaultForwardPlusPipeline
from infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline

CASTER = '''#version 450
ShaderInfo {
    Name "Tutorial Shadow Caster"
    ShadingModel Unlit
    Cull Off
    AlphaClip 0.5
    CastShadows CAST
    Properties { Color baseColor = [1.0, 0.0, 1.0, 1.0] }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    s.albedo = material.baseColor.rgb;
    s.alpha = ALPHA;
}
'''
RECEIVER = '''#version 450
ShaderInfo {
    Name "Tutorial Shadow Receiver"
    ShadingModel PBR
    Cull Off
    ReceiveShadows RECEIVE
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    s.albedo = vec3(0.5);
    s.smoothness = 0.0;
}
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', choices=('forward','forward_plus','deferred'), default='forward')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='infernux-shadow-audit-') as root:
        project = Path(root)
        for directory in ('Assets', 'Packages', 'ProjectSettings'):
            (project / directory).mkdir()
        quad = project / 'Assets/Quad.obj'
        quad.write_text('v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\n'
                        'vn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n', encoding='ascii')
        caster_file = project / 'Assets/Caster.frag'
        receiver_file = project / 'Assets/Receiver.frag'
        caster_file.write_text(CASTER.replace('CAST', 'On').replace('ALPHA', '1.0'), encoding='ascii')
        receiver_file.write_text(RECEIVER.replace('RECEIVE', 'On'), encoding='ascii')
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        failures, completed = [], False
        pipeline = {'forward': DefaultForwardPipeline,
                    'forward_plus': DefaultForwardPlusPipeline,
                    'deferred': DefaultDeferredPipeline}[args.path]()
        try:
            frontend.init_renderer(240, 180, str(project))
            frontend.resize_game_render_target(240, 180)
            engine.set_scene_view_visible(False)
            engine.set_editor_fps_cap(240)
            engine.set_editor_idle_fps(0)
            db = frontend.get_asset_database()
            imported = AssetManager.import_asset(str(quad), database=db)
            assert imported, imported.error
            for path in (caster_file, receiver_file):
                result = AssetManager.import_asset(str(path), database=db)
                assert result, result.error
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Camera')
            camera.transform.position = Vector3(0,0,-6)
            camera.add_component('Camera')
            sun = scene.create_game_object('Sun')
            sun.transform.euler_angles = Vector3(0,45,0)
            light = sun.add_component('Light')
            light.shadows = LightShadows.Hard
            light.intensity = 1.0
            plane = scene.create_game_object('Receiver')
            plane.transform.position = Vector3(1.5,0,0)
            plane.transform.local_scale = Vector3(3,4,1)
            receiver = plane.add_component('MeshRenderer')
            receiver.set_mesh_asset_guid(imported.guid)
            receive_material = InxMaterial.create_default_lit()
            receive_material.frag_shader_name = 'Tutorial Shadow Receiver'
            receiver.set_material(0, receive_material)
            # Shared mesh/material are eligible for native instance batching;
            # the second receiver keeps receiving while the first opts out.
            other_plane = scene.create_game_object('Shared Material Receiver')
            other_plane.transform.position = Vector3(-1.5,0,0)
            other_plane.transform.local_scale = Vector3(3,4,1)
            other_receiver = other_plane.add_component('MeshRenderer')
            other_receiver.set_mesh_asset_guid(imported.guid)
            other_receiver.set_material(0, receive_material)
            obj = scene.create_game_object('Caster')
            obj.transform.position = Vector3(-.5,0,-1.5)
            caster = obj.add_component('MeshRenderer')
            caster.set_mesh_asset_guid(imported.guid)
            cast_material = InxMaterial.create_default_unlit()
            cast_material.frag_shader_name = 'Tutorial Shadow Caster'
            cast_material.set_color('baseColor', 1.0, 0.0, 1.0, 1.0)
            caster.set_material(0, cast_material)
            caster.enabled = False
            other_obj = scene.create_game_object('Shared Material Caster')
            other_obj.transform.position = Vector3(-2.5,0,-1.5)
            other_caster = other_obj.add_component('MeshRenderer')
            other_caster.set_mesh_asset_guid(imported.guid)
            other_caster.set_material(0, cast_material)
            other_caster.enabled = False
            pipeline.msaa_samples = MSAASamples.OFF
            pipeline.shadow_resolution = 1024
            frontend.set_render_pipeline(pipeline)
            engine.set_game_camera_enabled(True)
            names = ('baseline','shadow','renderer_cast_off','renderer_receive_off',
                     'shader_cast_off','shader_receive_off','clipped','restored')
            images = {}
            phase, frame, changed, ticket = 0,0,0,None
            def edit(path, text):
                path.write_text(text, encoding='ascii')
                result = AssetManager.reimport_asset(str(path), database=db)
                assert result, result.error
            def after_draw():
                nonlocal phase, frame, changed, ticket, completed
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        name = names[phase]
                        images[name] = pixels
                        print(name, 'range', pixels[:,:,:3].min(), pixels[:,:,:3].max(), flush=True)
                        if phase:
                            diff = np.abs(pixels-images['baseline']).max(axis=-1)
                            print('changed', int((diff>.01).sum()), 'darkened',
                                  int(((images['baseline'][:,:,0]-pixels[:,:,0])>.08).sum()), flush=True)
                        errors = [e for e in console._get_visible_log_snapshot(2000)
                                  if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not errors, errors
                        if phase == len(names)-1:
                            visible = images['shadow'][:,:,0] > .95
                            visible &= images['shadow'][:,:,1] < .5
                            visible &= images['shadow'][:,:,2] > .95
                            assert visible.sum()>100, 'Caster color invisible'
                            shadow = images['baseline'][:,:,0]-images['shadow'][:,:,0] > .08
                            shadow &= ~visible
                            assert shadow.sum()>100, 'No separated shadow in fixture'
                            right = np.indices(shadow.shape)[1] >= shadow.shape[1]//2
                            assert (shadow & right).sum()>100
                            assert (shadow & ~right).sum()>100
                            for key in names[2:6]:
                                changed_shadow = shadow & right if key.startswith('renderer_') else shadow
                                np.testing.assert_allclose(images[key][changed_shadow], images['baseline'][changed_shadow], atol=.005,
                                                           err_msg=f'{args.path}: {key} did not disable shadow reception')
                                if key.startswith('renderer_'):
                                    np.testing.assert_allclose(images[key][shadow & ~right], images['shadow'][shadow & ~right],
                                                               atol=.005, err_msg='Per-instance switch leaked into shared material')
                            np.testing.assert_allclose(images['clipped'],images['baseline'],atol=.005)
                            np.testing.assert_allclose(images['restored'],images['shadow'],atol=.005)
                            print('PASS shadow casting/receiving/AlphaClip live roundtrip',flush=True)
                            completed = True
                            engine.exit()
                            return
                        phase += 1
                        if phase == 1:
                            caster.enabled = True
                            other_caster.enabled = True
                        elif phase == 2:
                            caster.casts_shadows = False
                        elif phase == 3:
                            caster.casts_shadows = True
                            receiver.receives_shadows = False
                        elif phase == 4:
                            receiver.receives_shadows = True
                            edit(caster_file, CASTER.replace('CAST','Off').replace('ALPHA','1.0'))
                        elif phase == 5:
                            edit(caster_file, CASTER.replace('CAST','On').replace('ALPHA','1.0'))
                            edit(receiver_file, RECEIVER.replace('RECEIVE','Off'))
                        elif phase == 6:
                            edit(receiver_file, RECEIVER.replace('RECEIVE','On'))
                            edit(caster_file, CASTER.replace('CAST','On').replace('ALPHA','0.0'))
                        else:
                            edit(caster_file, CASTER.replace('CAST','On').replace('ALPHA','1.0'))
                        ticket, changed = None, frame
                    if frame > 200:
                        raise AssertionError('Shadow audit timeout')
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

if __name__ == '__main__':
    main()
