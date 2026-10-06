"""Independent CPU-authored geometry reference for vertex deformation shadows."""
import json
from pathlib import Path
import tempfile
import argparse

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, LightShadows, SceneManager, Vector3
from deformed_tangent_frame_gpu_test import VERTEX, FRAGMENT

RECEIVER = '''#version 450
ShaderInfo { Name "Deformed Shadow Receiver" ShadingModel PBR Cull Off }
void surface(out SurfaceData s) {
    s = InitSurfaceData(); s.albedo = vec3(0.5); s.smoothness = 0.0;
}
'''

def main():
    from infernux.lib import _Infernux
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    proof = {'package': inx.__file__, 'native': _Infernux.__file__, 'phases': [], 'scope': 'Public vertex hook and alpha-clip shadow basis compared to separately authored equivalent CPU mesh. Unit, nonuniform and mirrored model scale, Forward/Forward+/Deferred, MSAA1/4.'}
    plans = [(inx.renderstack.DefaultForwardPipeline, 1), (inx.renderstack.DefaultForwardPipeline, 4),
             (inx.renderstack.DefaultForwardPlusPipeline, 1), (inx.renderstack.DefaultForwardPlusPipeline, 4),
             (inx.renderstack.DefaultDeferredPipeline, 1)]
    scales = [(1.,1.,1.), (2.,.5,1.), (-2.,.5,1.)]
    steps = ['baseline', 'gpu_hook', 'cpu_reference', 'restored']
    # Equal unreferenced envelope vertices keep CPU culling and shadow-fit
    # bounds identical while the referenced surface geometry changes.
    envelope = [(-2.,-2.,-2.), (2.,2.,2.)]
    mesh_positions = [np.array([(-.5,-.5,0),(.5,-.5,0),(.5,.5,0),(-.5,.5,0), *envelope], dtype=np.float32),
                      np.array([(-.1,-.56,-.42),(.7,-.08,-.06),(.1,.56,.42),(-.7,.08,.06), *envelope], dtype=np.float32)]
    mesh_normals = [np.tile(np.array((0,0,-1), dtype=np.float32),(6,1)),
                    np.tile(np.array((0,.6,-.8), dtype=np.float32),(6,1))]
    mesh_tangents = [np.tile(np.array((1,0,0,1), dtype=np.float32),(6,1)),
                     np.tile(np.array((.8,.48,.36,1), dtype=np.float32),(6,1))]
    mesh_uvs = np.array([(0,0),(1,0),(1,1),(0,1),(0,0),(0,0)], dtype=np.float32)
    mesh_indices = np.array([0,2,1,0,3,2], dtype=np.uint32)
    with tempfile.TemporaryDirectory(prefix='infernux-deformed-shadow-') as folder:
        project = Path(folder)
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (project / name).mkdir()
        quad = 'v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 -1\nf 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n'
        vertex = VERTEX.replace('Float handedness=1.0', 'Float handedness=1.0 Float deform=1.0')
        vertex = vertex.replace('v.position = rotation * v.position;', 'if (material.deform < 0.5) rotation = mat3(1.0);\n    v.position = rotation * v.position;')
        fragment = FRAGMENT.replace('ShadingModel Unlit Cull Off', 'ShadingModel Unlit Cull Off AlphaClip 0.5')
        fragment = fragment.replace('s.albedo = normalOK && tangentOK && basisOK && mappedOK ? vec3(0,1,0) : vec3(1,0,1);',
                                    's.alpha = normalOK && tangentOK && basisOK && mappedOK ? 1.0 : 0.0;\n    s.albedo = vec3(1,0,1);')
        for name, source in [('Quad.obj', quad), ('Probe.vert', vertex), ('Probe.frag', fragment), ('Receiver.frag', RECEIVER)]:
            (project / 'Assets' / name).write_text(source, encoding='ascii')
        frontend = inx.Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        pipeline = None
        failures, completed = [], False
        try:
            frontend.init_renderer(320, 240, str(project))
            frontend.resize_game_render_target(320, 240)
            engine.set_scene_view_visible(False)
            engine.set_editor_fps_cap(240)
            engine.set_editor_idle_fps(0)
            db = frontend.get_asset_database()
            imported = {}
            for name in ('Quad.obj', 'Probe.vert', 'Probe.frag', 'Receiver.frag'):
                result = AssetManager.import_asset(str(project / 'Assets' / name), database=db)
                assert result, result.error
                imported[name] = result.guid
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
            plane.transform.local_scale = Vector3(6,4,1)
            receiver = plane.add_component('MeshRenderer')
            receiver.set_mesh_asset_guid(imported['Quad.obj'])
            receiver_material = InxMaterial.create_default_lit()
            receiver_material.frag_shader_name = 'Deformed Shadow Receiver'
            receiver.set_material(0, receiver_material)
            obj = scene.create_game_object('Caster')
            obj.transform.position = Vector3(-.5,0,-1.5)
            caster = obj.add_component('MeshRenderer')
            material = InxMaterial.create_default_unlit()
            material.vert_shader_name = 'Deformed Tangent Probe'
            material.frag_shader_name = 'Deformed Tangent Consumer'
            caster.set_material(0, material)
            plan = scale_index = phase = frame = changed = 0
            ticket = None
            images = {}

            def set_pipeline():
                nonlocal pipeline
                if pipeline is not None:
                    pipeline.dispose()
                base, samples = plans[plan]
                pipeline = base()
                pipeline.shadow_resolution = 1024
                if base is not inx.renderstack.DefaultDeferredPipeline:
                    pipeline.msaa_samples = samples
                frontend.set_render_pipeline(pipeline)

            def apply():
                scale = scales[scale_index]
                obj.transform.local_scale = Vector3(*scale)
                for key, value in zip(('scaleX', 'scaleY', 'scaleZ'), scale):
                    material.set_float(key, value)
                caster.enabled = phase != 0
                choice = int(phase == 2)
                caster.set_inline_mesh_data(mesh_positions[choice], mesh_normals[choice], mesh_uvs,
                                            mesh_indices, 'CPU Shadow Reference' if choice else 'GPU Hook Shadow',
                                            mesh_tangents[choice])
                material.set_float('deform', 0. if phase == 2 else 1.)

            set_pipeline()
            apply()
            engine.set_game_camera_enabled(True)

            def after_draw():
                nonlocal plan, scale_index, phase, frame, changed, ticket, completed
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 10:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        images[steps[phase]] = pixels
                        issues = [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not issues, issues
                        if phase == 3:
                            rgb = images['cpu_reference'][...,:3]
                            visible = (rgb[...,0]>.95) & (rgb[...,1]<.5) & (rgb[...,2]>.95)
                            shadow = (images['baseline'][...,0] - rgb[...,0] > .08) & ~visible
                            record = {'pipeline': plans[plan][0].name, 'samples': plans[plan][1], 'scale': scales[scale_index], 'visible_pixels': int(visible.sum()), 'shadow_pixels': int(shadow.sum())}
                            assert visible.sum()>200 and shadow.sum()>100, record
                            gpu = images['gpu_hook'][...,:3]
                            gpu_visible = (gpu[...,0]>.95) & (gpu[...,1]<.5) & (gpu[...,2]>.95)
                            gpu_shadow = (images['baseline'][...,0] - gpu[...,0] > .08) & ~gpu_visible
                            record['visible_mismatch'] = int((visible != gpu_visible).sum())
                            record['shadow_union'] = int((shadow | gpu_shadow).sum())
                            record['shadow_intersection'] = int((shadow & gpu_shadow).sum())
                            record['shadow_mismatch'] = int((shadow != gpu_shadow).sum())
                            np.testing.assert_allclose(images['gpu_hook'], images['cpu_reference'], atol=.005, err_msg=str(record))
                            np.testing.assert_array_equal(images['gpu_hook'], images['restored'])
                            proof['phases'].append(record)
                            print('PASS ' + json.dumps(record), flush=True)
                            phase = 0
                            scale_index += 1
                            if scale_index == len(scales):
                                scale_index = 0
                                plan += 1
                                if plan == len(plans):
                                    completed = True
                                    engine.exit()
                                    return
                                set_pipeline()
                            images.clear()
                        else:
                            phase += 1
                        apply()
                        ticket, changed = None, frame
                    if frame > 1400:
                        raise AssertionError('Deformed shadow audit timeout')
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
            if pipeline is not None:
                pipeline.dispose()
            engine.cleanup()
            proof['passed'] = completed and not failures
            if args.proof:
                args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
