"""Independent temporary project: exact tutorial material in both views and GPU preview."""
import argparse
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.engine.path_utils import resolved_path
from infernux.lib import ConsolePanel, LightShadows, SceneManager, Vector3

ROOT = Path(resolved_path(__file__)).parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', choices=('lit', 'painted'), required=True)
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    proof = {'passed': False, 'package': inx.__file__, 'surface': args.surface, 'phases': [],
             'scope': 'Independent disposable renderer project, Scene and Game GPU targets, material preview publication. No live Editor pixel access.'}
    with tempfile.TemporaryDirectory(prefix='infernux-tutorial-material-views-') as folder:
        project = Path(folder)
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (project / name).mkdir()
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        failures, complete = [], False
        stack = None
        try:
            frontend.init_renderer(960, 720, folder)
            frontend.resize_scene_render_target(960, 720)
            frontend.resize_game_render_target(960, 720)
            native.set_scene_view_visible(True)
            native.set_show_grid(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            native.editor_camera.restore_state(0, 2, -5, 0, 0, 0, 5.3851648, 0, 21.801409)
            database = frontend.get_asset_database()
            template = inx.Material.create_lit('Tutorial Shared')
            if args.surface == 'painted':
                source = re.findall(r'```glsl\n(.*?)```', (ROOT / 'docs/learn/fragment-materials.md').read_text(encoding='utf-8'), re.S)[0]
                shader = project / 'Assets/PaintedUnlit.frag'
                shader.write_text(source, encoding='utf-8')
                result = AssetManager.import_asset(str(shader), database=database)
                assert result, result.error
                template.frag_shader_name = 'Painted Unlit'
            path = project / 'Assets/TutorialShared.mat'
            path.write_text(json.dumps(template.serialize_document()), encoding='utf-8')
            result = AssetManager.import_asset(str(path), database=database)
            assert result, result.error
            material = inx.Material.load(str(path))
            assert material and material.guid == result.guid
            scene = SceneManager.instance().get_active_scene()
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            owner = scene.create_game_object('Tutorial Camera')
            owner.transform.position = Vector3(0, 1, -10)
            owner.add_component(inx.Camera)
            sun = scene.create_game_object('Directional Light')
            sun.transform.euler_angles = Vector3(50, -30, 0)
            sun.add_component(inx.Light).shadows = LightShadows.Hard
            objects = []
            for x in (-.8, .8):
                obj = scene.create_primitive(inx.PrimitiveType.Cube)
                obj.transform.position = Vector3(x, 0, 0)
                renderer = obj.get_component(inx.MeshRenderer)
                renderer.set_material(0, material)
                assert renderer.get_material(0).guid == material.guid
                objects.append(obj)
            native.set_game_camera_enabled(True)
            colors = ((.9, .1, .2, 1), (.2, .7, .1, 1))
            plans = [('red_implicit', 0), ('green_implicit', 1), ('red_explicit', 0),
                     ('green_explicit', 1), ('moved_green', 1), ('restored_green', 1)]
            material.set_color('baseColor', *colors[0])
            phase = frames = changed = 0
            tickets = None
            images, previews = {}, {}
            preview_key = 'mat|' + material.guid

            def after_draw():
                nonlocal phase, frames, changed, tickets, complete, stack
                try:
                    frames += 1
                    native.query_or_schedule_material_preview(preview_key, str(path), json.dumps(material.serialize_document()), 0, True)
                    native.pump_preview_tasks()
                    tasks = [s for s in native.preview_task_snapshots if s['kind'] == 'material' and s['resource_key'] == preview_key]
                    ready = tasks and tasks[0]['ready_generation'] == tasks[0]['generation'] and tasks[0]['texture_id'] != 0
                    if tickets is None and frames >= changed + 10 and ready:
                        tickets = [frontend.request_render_target_readback(False), frontend.request_render_target_readback(True)]
                    elif tickets is not None and all(t.done for t in tickets):
                        label, color_index = plans[phase]
                        record = {'phase': label, 'views': [], 'preview': tasks[0], 'material_guid': material.guid}
                        previews[label] = tasks[0]['pixel_hash']
                        assert tasks[0]['non_transparent_pixel_count'] > 500
                        for view, ticket in zip(('scene', 'game'), tickets):
                            pixels = ticket.result_numpy().copy().astype(np.float32)
                            rgb = pixels[..., :3]
                            dominant = (rgb[..., color_index] > .1) & (rgb[..., color_index] > rgb[..., 1-color_index] * 2) & (rgb[..., color_index] > rgb[..., 2] * 2)
                            middle = pixels.shape[1] // 2
                            assert dominant[:, :middle].sum() > 20 and dominant[:, middle:].sum() > 20, (view, label, int(dominant.sum()))
                            interior = dominant.copy()
                            for axis in (0, 1):
                                for offset in (-1, 1):
                                    interior &= np.roll(dominant, offset, axis=axis)
                            assert interior.sum() > 20, (view, label, 'No interior material pixels')
                            if args.surface == 'painted':
                                np.testing.assert_allclose(rgb[interior], np.broadcast_to(np.array(colors[color_index][:3]), rgb[interior].shape), atol=.005)
                            elif phase < 4:
                                assert float(np.ptp(rgb[interior, color_index])) > .01, (view, label, 'Lit faces have no brightness variation')
                            record['views'].append({'view': view, 'colored_pixels': int(dominant.sum()), 'left_pixels': int(dominant[:, :middle].sum()), 'right_pixels': int(dominant[:, middle:].sum()), 'interior_dominant_range': float(np.ptp(rgb[interior, color_index]))})
                            images[(label, view)] = pixels
                        frame = native.renderer_frame_snapshot
                        expected_graph = 'Default Forward' if phase < 2 else 'Pipeline+Stack'
                        assert frame['scene_render_graph_name'] == frame['game_render_graph_name'] == expected_graph, (label, frame['scene_render_graph_name'], frame['game_render_graph_name'])
                        if stack is not None:
                            assert stack.pipeline.name == 'Default Forward'
                        assert frame['scene_draw_call_count'] >= 2 and frame['game_draw_call_count'] >= 2
                        record['frame'] = frame
                        errors = [e for e in console._get_visible_log_snapshot(2000) if e['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                        assert not errors, errors[:5]
                        proof['phases'].append(record)
                        print('PASS', args.surface, label, record['views'], flush=True)
                        if phase == len(plans) - 1:
                            for view in ('scene', 'game'):
                                for a, b in (('red_implicit', 'red_explicit'), ('green_implicit', 'green_explicit'), ('green_explicit', 'restored_green')):
                                    np.testing.assert_allclose(images[(a, view)], images[(b, view)], atol=.005)
                                assert np.abs(images[('green_explicit', view)] - images[('moved_green', view)]).sum() > 1
                            assert previews['red_implicit'] != previews['green_implicit']
                            assert previews['red_implicit'] == previews['red_explicit']
                            assert previews['green_implicit'] == previews['green_explicit']
                            complete = True
                            native.exit()
                            return
                        phase += 1
                        material.set_color('baseColor', *colors[plans[phase][1]])
                        if phase == 2:
                            stack = scene.create_game_object('Explicit Default Forward').add_component(inx.RenderStack)
                            stack.pipeline_class_name = 'Default Forward'
                        if phase in (4, 5):
                            for index, obj in enumerate(objects):
                                obj.transform.position = Vector3((-.8, .8)[index], .5 if phase == 4 else 0, 0)
                        changed, tickets = frames, None
                    if frames > 400:
                        raise AssertionError(('Tutorial view/preview timeout', tasks))
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
            if stack is not None:
                stack.on_destroy()
            native.cleanup()
    proof['passed'] = True
    if args.proof:
        args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print('PASS Scene/Game/shared material/preview and empty RenderStack parity', flush=True)


if __name__ == '__main__':
    main()
