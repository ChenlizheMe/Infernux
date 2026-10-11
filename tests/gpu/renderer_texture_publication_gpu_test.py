"""Cold Renderer overrides and hot texture replacement without parameter edits."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3
from material_texture_defaults_gpu_test import DefaultProbePipeline, SHADER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof', type=Path, required=True)
    args = parser.parse_args()
    proof = dict(passed=False, phases=[], parameter_assignments=0, peak_pending=0,
                 scope='Actual GPU pixels: shared unchanged material, two cold per-renderer textures, '
                       'hot texture replacement, clear; no repeated parameter assignment during residency.')
    with tempfile.TemporaryDirectory(prefix='infernux-renderer-textures-') as folder:
        project = Path(folder)
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (project / name).mkdir()
        shader = project / 'Assets/RendererTexture.frag'
        shader.write_text(SHADER.replace('DEFAULT', 'white'), encoding='utf-8')
        texture_paths = [project / 'Assets/ColdRed.png', project / 'Assets/ColdBlue.png']
        for path, color in zip(texture_paths, [(255, 0, 0, 255), (0, 0, 255, 255)]):
            Image.new('RGBA', (1024, 1024), color).save(path)
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        pipeline = DefaultProbePipeline()
        console = ConsolePanel()
        failures = []
        complete = False
        try:
            frontend.init_renderer(192, 128, folder)
            frontend.resize_game_render_target(192, 128)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            database = frontend.get_asset_database()
            imported = AssetManager.import_asset(str(shader), database=database)
            assert imported, imported.error
            guids = []
            for path in texture_paths:
                imported = AssetManager.import_asset(str(path), database=database)
                assert imported, imported.error
                guids.append(imported.guid)
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('Texture Publication Camera')
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component(inx.Camera)
            material = inx.Material.create_unlit('Unchanged Shared Material')
            material.frag_shader_name = 'Declared Texture Default Probe'
            renderers = []
            for x in (-1, 1):
                cube = scene.create_primitive(inx.PrimitiveType.Cube)
                cube.transform.position = Vector3(x, 0, 0)
                renderer = cube.get_component(inx.MeshRenderer)
                renderer.set_material(0, material)
                renderers.append(renderer)
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            native.set_play_mode_rendering(True)
            phases = [
                ('base_white', ((1, 1, 1), (1, 1, 1)), 0),
                ('cold_overrides', ((1, 0, 0), (0, 0, 1)), 2),
                ('hot_green_override', ((0, 1, 0), (0, 0, 1)), 3),
                ('clear_returns_base', ((1, 1, 1), (1, 1, 1)), 3),
            ]
            phase = frame = changed = 0
            uploads_before = None
            material_version = material_document = None
            ticket = initial = None

            def after_draw():
                nonlocal phase, frame, changed, ticket, initial, complete
                nonlocal uploads_before, material_version, material_document
                try:
                    frame += 1
                    AssetManager.flush_pending_gpu_texture_reloads()
                    pending = native.pending_texture_cpu_load_count + native.pending_texture_gpu_upload_count
                    proof['peak_pending'] = max(proof['peak_pending'], pending)
                    ready = uploads_before is None or (
                        native.completed_texture_gpu_upload_count >= uploads_before + phases[phase][2] and pending == 0)
                    if ticket is None and frame >= changed + 12 and ready:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        measured = []
                        for side, expected in enumerate(phases[phase][1]):
                            half = pixels[:, side * 96:(side + 1) * 96]
                            covered = half[..., 3] > .9
                            assert covered.sum() > 100, (phases[phase][0], side, int(covered.sum()))
                            rgb = half[..., :3][covered]
                            np.testing.assert_allclose(rgb, np.broadcast_to(expected, rgb.shape), atol=.005)
                            measured.append(rgb.mean(axis=0).tolist())
                        issues = [entry for entry in console._get_visible_log_snapshot(1000)
                                  if entry['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                        assert not issues, issues
                        if material_version is not None:
                            assert material.native.get_version() == material_version
                            assert material.native.serialize_document() == material_document
                        proof['phases'].append(dict(phase=phases[phase][0], measured=measured,
                            uploads=native.completed_texture_gpu_upload_count, material_version=material.native.get_version()))
                        print('PASS', phases[phase][0], measured, flush=True)
                        if phase == 0:
                            initial = pixels
                            uploads_before = native.completed_texture_gpu_upload_count
                            material_version = material.native.get_version()
                            material_document = material.native.serialize_document()
                            for renderer, guid in zip(renderers, guids):
                                renderer.set_parameter('independentName', guid)
                                proof['parameter_assignments'] += 1
                        elif phase == 1:
                            Image.new('RGBA', (1024, 1024), (0, 255, 0, 255)).save(texture_paths[0])
                            imported = AssetManager.reimport_asset(str(texture_paths[0]), database=database)
                            assert imported and imported.guid == guids[0], imported.error
                        elif phase == 2:
                            for renderer in renderers:
                                renderer.clear_parameters()
                        else:
                            np.testing.assert_array_equal(pixels, initial)
                            assert proof['peak_pending'] > 0, proof
                            assert proof['parameter_assignments'] == 2
                            complete = True
                            native.exit()
                            return
                        phase += 1
                        changed, ticket = frame, None
                    if frame > 1000:
                        raise AssertionError(f'Texture publication did not finish: phase={phase}, '
                                             f'pending={pending}, uploads={native.completed_texture_gpu_upload_count}')
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()
    proof['passed'] = True
    args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print('PASS cold renderer textures and hot replacement without parameter reassignment', flush=True)


if __name__ == '__main__':
    main()
