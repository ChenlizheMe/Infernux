"""Reject non-mesh shaders before Vulkan layout creation, then recover."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.lib import ConsolePanel, InxMaterial, LightShadows, SceneManager, Vector3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=('Fullscreen', 'ParticleSprite'), required=True)
    parser.add_argument('--pipeline', choices=('forward', 'forward_plus', 'deferred'), default='forward')
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    wrong_pair = ('Fullscreen Triangle', 'Fullscreen Blit') if args.domain == 'Fullscreen' else ('Particle Sprite', 'Particle Unlit')
    proof = {'package': inx.__file__, 'domain': args.domain, 'pipeline': args.pipeline, 'phases': [],
             'scope': 'Independent temporary project GPU readbacks, cold rejection, valid Mesh recovery and rejected edit with previous valid pipeline. No live Editor pixels.'}
    failures = []
    completed = False
    with tempfile.TemporaryDirectory(prefix='infernux-mesh-domain-') as folder:
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (Path(folder) / name).mkdir()
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        try:
            frontend.init_renderer(160, 120, folder)
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            pipeline_type = {'forward': inx.renderstack.DefaultForwardPipeline,
                             'forward_plus': inx.renderstack.DefaultForwardPlusPipeline,
                             'deferred': inx.renderstack.DefaultDeferredPipeline}[args.pipeline]
            pipeline = pipeline_type()
            pipeline.shadow_resolution = 256
            if args.pipeline != 'deferred':
                pipeline.msaa_samples = 1
            frontend.set_render_pipeline(pipeline)
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Domain Camera')
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component('Camera')
            sun = scene.create_game_object('Domain Sun')
            sun.transform.euler_angles = Vector3(35, 25, 0)
            sun.add_component('Light').shadows = LightShadows.Hard
            owner = scene.create_game_object('Domain Quad')
            renderer = owner.add_component('MeshRenderer')
            renderer.set_inline_mesh_data(
                np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], dtype=np.float32),
                np.array([[0, 0, -1]] * 4, dtype=np.float32),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32), 'Domain Quad')
            material = InxMaterial.create_default_unlit()
            material.set_color('baseColor', 1., .1, .05, 1.)
            material.vert_shader_name, material.frag_shader_name = wrong_pair
            renderer.set_material(0, material)
            native.set_game_camera_enabled(True)
            phases = ('cold_rejection', 'mesh_recovery', 'rejected_edit', 'restored_mesh')
            phase = frames = changed = 0
            ticket = baseline = None

            def after_draw():
                nonlocal phase, frames, changed, ticket, baseline, completed
                try:
                    frames += 1
                    if ticket is None and frames >= changed + 10:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        entries = console._get_visible_log_snapshot(2000)
                        diagnostics = [e for e in entries if 'Material shader domain mismatch:' in e['message']]
                        expected = "uses domain '" + args.domain + "', but MeshRenderer geometry requires domain 'Mesh'"
                        assert len(diagnostics) == 1 and expected in diagnostics[0]['message'], diagnostics[:3]
                        assert diagnostics[0]['count'] == 1, diagnostics
                        unexpected = [e for e in entries if e['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')
                                      and e not in diagnostics]
                        assert not unexpected, unexpected[:5]
                        label = phases[phase]
                        coverage = (pixels[..., 0] > .8) & (pixels[..., 1] < .2) & (pixels[..., 2] < .2)
                        if label == 'cold_rejection':
                            # The editor's existing Error material marks a cold
                            # invalid selection; it must never draw the rejected ABI.
                            error_pixels = (pixels[..., 0] > .8) & (pixels[..., 1] < .1) & (pixels[..., 2] > .8)
                            assert error_pixels.sum() > 50 and coverage.sum() == 0, int(error_pixels.sum())
                        else:
                            assert coverage.sum() > 100, int(coverage.sum())
                            if baseline is None:
                                baseline = pixels
                            else:
                                np.testing.assert_array_equal(pixels, baseline)
                        proof['phases'].append({'phase': label, 'red_pixels': int(coverage.sum()),
                                                'domain_diagnostic_count': len(diagnostics)})
                        print('PASS ' + args.domain + ' ' + json.dumps(proof['phases'][-1]), flush=True)
                        if phase == len(phases) - 1:
                            proof['diagnostics'] = diagnostics
                            proof['passed'] = True
                            completed = True
                            native.exit()
                            return
                        phase += 1
                        pair = wrong_pair if phases[phase] == 'rejected_edit' else ('Standard', 'Unlit')
                        material.vert_shader_name, material.frag_shader_name = pair
                        changed = frames
                        ticket = None
                    if frames > 300:
                        raise AssertionError('Mesh shader domain GPU audit timed out')
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
            native.cleanup()
            if args.proof:
                proof.setdefault('passed', False)
                proof['failures'] = [str(error)[:3000] for error in failures]
                args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
