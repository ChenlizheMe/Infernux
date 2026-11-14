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

RECEIVER = '''import infernux.shader as shader


class ShadowReceiver(shader.surface):
    shader_id = "tests/deformation/shadow_receiver"
    cull_mode = "none"

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=shader.vec3(0.5), smoothness=0.0)
'''

def main():
    from infernux.lib import _Infernux
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--shading', choices=('unlit', 'pbr'), default='unlit')
    parser.add_argument('--skinned', action='store_true')
    args = parser.parse_args()
    proof = {'package': inx.__file__, 'native': _Infernux.__file__, 'shading': args.shading, 'phases': [], 'scope': 'Public vertex hook and alpha-clip shadow basis compared to separately authored equivalent CPU mesh. Unit, nonuniform and mirrored model scale, Forward/Forward+/Deferred, MSAA1/4.'}
    def visible_mask(rgb):
        if args.shading == 'pbr':
            return (rgb[...,0] > rgb[...,1]*1.2) & (rgb[...,1] > rgb[...,2]*1.2) & (rgb[...,0] > .1)
        return (rgb[...,0]>.95) & (rgb[...,1]<.5) & (rgb[...,2]>.95)
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
        vertex = VERTEX.replace('    handedness = shader.material.float(default=1.0)',
            '    handedness = shader.material.float(default=1.0)\n    deform = shader.material.float(default=1.0)')
        vertex = vertex.replace('        vertex.position_os = rotation @ vertex.position_os',
            '        if self.deform < 0.5:\n            rotation = shader.mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)\n        vertex.position_os = rotation @ vertex.position_os')
        fragment = FRAGMENT.replace('    cull_mode = "none"', '    cull_mode = "none"\n    alpha_clip = 0.5')
        fragment = fragment.replace('albedo = shader.vec3(0.0, 1.0, 0.0) if valid else shader.vec3(1.0, 0.0, 1.0)',
                                    'albedo = shader.vec3(1.0, 0.0, 1.0)')
        fragment = fragment.replace('return shader.SurfaceData(albedo=albedo)',
                                    'return shader.SurfaceData(albedo=albedo, alpha=1.0 if valid else 0.0)')
        if args.shading == 'pbr':
            fragment = fragment.replace('    shading_model = shader.ref("infernux/shading/unlit")\n', '')
            fragment = fragment.replace('albedo = shader.vec3(1.0, 0.0, 1.0)',
                                         'albedo = shader.vec3(0.5, 0.15, 0.04)')
            fragment = fragment.replace('albedo=albedo, alpha=',
                                         'albedo=albedo, smoothness=0.0, normal_ws=mapped_normal, alpha=')
        for name, source in [('Quad.obj', quad), ('Probe.vert.py', vertex), ('Probe.frag.py', fragment), ('Receiver.frag.py', RECEIVER)]:
            (project / 'Assets' / name).write_text(source, encoding='ascii')
        if args.skinned:
            from skinned_mesh_gpu_fixture import write_skinned_quad
            write_skinned_quad(project / 'Assets/Quad.gltf', envelope=True)
            # Match the hook's deliberately authored UV orientation in both
            # paths. The CPU mesh has already applied the bone reflection.
            vertex = vertex.replace('    deform = shader.material.float(default=1.0)', '    deform = shader.material.float(default=1.0)\n    reference_w = shader.material.float(default=1.0)')
            vertex = vertex.replace('self.handedness)', 'self.handedness * self.reference_w)')
            (project / 'Assets/Probe.vert.py').write_text(vertex, encoding='ascii')
            proof['scope'] += ' Imported bone scale animation after vertex hook compared with CPU positions and normal-mapped lighting.'
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
            for name in ('Quad.obj', 'Probe.vert.py', 'Probe.frag.py', 'Receiver.frag.py', *(('Quad.gltf',) if args.skinned else ())):
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
            receiver_material.frag_shader_name = 'tests/deformation/shadow_receiver'
            receiver.set_material(0, receiver_material)
            obj = scene.create_game_object('Caster')
            obj.transform.position = Vector3(-.5,0,-1.5)
            caster = obj.add_component('SkinnedMeshRenderer' if args.skinned else 'MeshRenderer')
            reference_obj = reference_renderer = None
            if args.skinned:
                caster.set_source_model_guid(imported['Quad.gltf'])
                assert caster.get_animation_take_names() == ['Scale']
                reference_obj = scene.create_game_object('CPU Skinned Reference')
                reference_obj.transform.position = obj.transform.position
                reference_renderer = reference_obj.add_component('MeshRenderer')
            material = InxMaterial.create_default_unlit()
            material.vert_shader_name = 'tests/deformation/tangent'
            material.frag_shader_name = 'tests/deformation/tangent_surface'
            caster.set_material(0, material)
            if reference_renderer is not None:
                reference_renderer.set_material(0, material)
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
                if not args.skinned:
                    obj.transform.local_scale = Vector3(*scale)
                for key, value in zip(('scale_x', 'scale_y', 'scale_z'), scale):
                    material.set_float(key, value)
                if args.skinned:
                    caster.enabled = phase in (1,3)
                    reference_renderer.enabled = phase == 2
                    pose_time = float(scale_index)
                    caster.submit_animation_pose('Scale', pose_time, pose_time/3., loop=False)
                    normals = mesh_normals[1] / np.array(scale, dtype=np.float32)
                    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                    tangents = mesh_tangents[1].copy()
                    tangents[:,:3] *= np.array(scale, dtype=np.float32)
                    tangents[:,:3] /= np.linalg.norm(tangents[:,:3], axis=1, keepdims=True)
                    tangents[:,3] *= np.sign(np.prod(scale))
                    reference_renderer.set_inline_mesh_data(mesh_positions[1] * np.array(scale, dtype=np.float32),
                        normals, mesh_uvs, mesh_indices, 'CPU Bone Scale Reference', tangents)
                    np.testing.assert_allclose(caster.get_world_bounds(), reference_renderer.get_world_bounds(), atol=1.e-5)
                else:
                    caster.enabled = phase != 0
                    choice = int(phase == 2)
                    caster.set_inline_mesh_data(mesh_positions[choice], mesh_normals[choice], mesh_uvs,
                                                mesh_indices, 'CPU Shadow Reference' if choice else 'GPU Hook Shadow',
                                                mesh_tangents[choice])
                material.set_float('deform', 0. if phase == 2 else 1.)
                if args.skinned:
                    material.set_float('reference_w', float(np.sign(np.prod(scale))) if phase == 2 else 1.)

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
                            visible = visible_mask(rgb)
                            shadow = (images['baseline'][...,0] - rgb[...,0] > .08) & ~visible
                            record = {'pipeline': plans[plan][0].name, 'samples': plans[plan][1], 'scale': scales[scale_index], 'visible_pixels': int(visible.sum()), 'shadow_pixels': int(shadow.sum())}
                            assert visible.sum()>200 and shadow.sum()>100, record
                            gpu = images['gpu_hook'][...,:3]
                            gpu_visible = visible_mask(gpu)
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
