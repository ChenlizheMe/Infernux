"""Retiring a shadow atlas must republish every frame slot before drawing."""
import argparse
import json
from pathlib import Path
import tempfile

import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, LightShadows, RenderPipelineCallback, SceneManager, Vector3


class ShadowTopology(inx.renderstack.RenderPipeline):
    name = "Shadow Topology Transition"
    shadows_enabled = True

    def define(self, pipeline):
        pipeline.frame(hdr=True, msaa=1)
        if self.shadows_enabled:
            pipeline.shadows(resolution=256)
        pipeline.lighting(clustered=False)
        with pipeline.opaque() as opaque:
            opaque.otherwise().forward()
        pipeline.screen_ui()


class Host(RenderPipelineCallback):
    def __init__(self):
        super().__init__()
        self.stack = inx.RenderStack()
        self.pipeline = ShadowTopology()
        self.stack._pipeline = self.pipeline

    def render(self, context, camera):
        self.stack.render(context, camera)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proof', type=Path)
    args = parser.parse_args()
    proof = {'passed': False, 'package': inx.__file__, 'phases': []}
    with tempfile.TemporaryDirectory(prefix='infernux-shadow-topology-') as folder:
        project = Path(folder)
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (project / name).mkdir()
        mesh = project / 'Assets/Probe.obj'
        mesh.write_text('v -1 -1 0\nv 1 -1 0\nv 0 1 0\nvn 0 0 -1\nvn 0 0 1\nf 1//1 2//1 3//1\nf 3//2 2//2 1//2\n', encoding='ascii')
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        host = Host()
        failures = []
        try:
            frontend.init_renderer(64, 64, folder)
            frontend.resize_game_render_target(64, 64)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.0)
            native.set_editor_idle_fps(0.0)
            imported = AssetManager.import_asset(str(mesh), database=frontend.get_asset_database())
            assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            camera = scene.create_game_object('Camera')
            camera.transform.position = Vector3(0, 0, -4)
            camera.add_component('Camera')
            sun = scene.create_game_object('Sun').add_component('Light')
            sun.shadows = LightShadows.Hard
            renderer = scene.create_game_object('Probe').add_component('MeshRenderer')
            renderer.set_mesh_asset_guid(imported.guid)
            renderer.set_material(0, InxMaterial.create_default_lit())
            frontend.set_render_pipeline(host)
            native.set_game_camera_enabled(True)
            frames = 0
            phase = 0

            def after_draw():
                nonlocal frames, phase
                try:
                    frames += 1
                    errors = [e for e in console._get_visible_log_snapshot(2000)
                              if e['level'] in ('ERROR', 'FATAL')]
                    assert not errors, errors[:3]
                    if frames % 12:
                        return
                    state = host.stack._graph_state
                    assert not state.build_failed and state.description is not None
                    names = [p.name for p in state.description.passes]
                    assert ('ShadowCasterPass' in names) == host.pipeline.shadows_enabled, names
                    frame = native.renderer_frame_snapshot
                    assert frame['game_render_graph_current_executed'], frame
                    assert frame['game_draw_call_count'] > 0, frame
                    assert ('ShadowCasterPass' in frame['game_render_graph_pass_names']) == host.pipeline.shadows_enabled
                    proof['phases'].append({'shadows': host.pipeline.shadows_enabled,
                                            'revision': state.description.source_revision,
                                            'frames': frames, 'passes': names,
                                            'draws': frame['game_draw_call_count']})
                    print('PASS shadow topology', phase, host.pipeline.shadows_enabled, flush=True)
                    if phase == 4:
                        proof['passed'] = True
                        native.exit()
                        return
                    phase += 1
                    host.pipeline.shadows_enabled = not host.pipeline.shadows_enabled
                    host.stack.invalidate_graph()
                except BaseException as exc:
                    failures.append(exc)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert proof['passed']
        finally:
            frontend.set_render_pipeline(None)
            host.stack.on_destroy()
            native.cleanup()
            if args.proof:
                args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
