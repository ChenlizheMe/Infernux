"""Real Vulkan regression for live queue edits on an unchanged camera draw list."""
from __future__ import annotations

import tempfile
import json
import sys
from pathlib import Path

import numpy as np

from Infernux import Engine
from Infernux.core.assets import AssetManager
from Infernux.lib import AssetRegistry, InxMaterial, RenderPipelineCallback, SceneManager, Vector3
from Infernux.rendergraph import Format, RenderGraph


class SelectedQueuePipeline(RenderPipelineCallback):
    def __init__(self):
        super().__init__()
        graph = RenderGraph("MaterialQueueProbe")
        color = graph.create_texture("color", camera_target=True)
        depth = graph.create_texture("depth", format=Format.D32_SFLOAT)
        with graph.add_pass("Clear") as render_pass:
            render_pass.write_color(color)
            render_pass.write_depth(depth)
            render_pass.set_clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)
        with graph.add_pass("Queue1000") as render_pass:
            render_pass.write_color(color)
            render_pass.write_depth(depth)
            render_pass.draw_renderers(queue_range=(1000, 1099))
        graph.set_output(color)
        self.description = graph.build()

    def render(self, context, camera):
        if not context.render_compiled(camera, self.description.source_revision):
            context.render_with_graph(camera, self.description)

    def dispose(self):
        pass


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="infernux-material-queue-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        mesh_path = project / "Assets" / "QueueProbe.obj"
        mesh_path.write_text(
            "v -1 -1 0\nv 1 -1 0\nv 0 1 0\n"
            "vn 0 0 -1\nvn 0 0 1\n"
            "f 1//1 2//1 3//1\nf 3//2 2//2 1//2\n", encoding="ascii",
        )
        frontend = Engine()
        engine = frontend.get_native_engine()
        failures = []
        completed = False
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            frontend.resize_game_render_target(64, 64)
            imported = AssetManager.import_asset(str(mesh_path), database=frontend.get_asset_database())
            assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            camera = scene.create_game_object("QueueCamera")
            camera.transform.position = Vector3(0.0, 0.0, -4.0)
            camera.add_component("Camera")
            engine.set_game_camera_enabled(True)
            probe = scene.create_game_object("QueueProbe")
            probe.is_static = True
            renderer = probe.add_component("MeshRenderer")
            renderer.set_mesh_asset_guid(imported.guid)
            material = InxMaterial.create_default_unlit()
            material.set_render_queue(1000)
            material_path = project / "Assets" / "QueueProbe.mat"
            material_document = material.serialize_document()
            material_document['builtin'] = False
            material_document['renderStateOverrides'] = 64
            asset_mode = '--asset' in sys.argv
            if asset_mode:
                material_path.write_text(json.dumps(material_document), encoding='utf-8')
                material_import = AssetManager.import_asset(str(material_path), database=frontend.get_asset_database())
                assert material_import, material_import.error
                material = AssetRegistry.instance().load_material_by_guid(material_import.guid)
                renderer.set_material(0, material_import.guid)
            else:
                renderer.set_material(0, material)
            pipeline = SelectedQueuePipeline()
            engine.set_render_pipeline(pipeline)

            frame = 0
            stage = "initial"
            switched_frame = 0
            ticket = None
            first_pixels = None

            def set_queue(queue):
                if asset_mode:
                    material_document['renderState']['renderQueue'] = queue
                    material_path.write_text(json.dumps(material_document), encoding='utf-8')
                    result = AssetManager.reimport_asset(str(material_path), database=frontend.get_asset_database())
                    assert result, result.error
                    assert renderer.get_material(0).get_render_queue() == queue
                else:
                    material.set_render_queue(queue)

            def after_draw():
                nonlocal frame, stage, switched_frame, ticket, first_pixels, completed
                try:
                    frame += 1
                    if frame >= 8 and stage == "initial":
                        ticket = frontend.request_render_target_readback(True)
                        stage = "capture_initial"
                    elif stage == "capture_initial" and ticket.done:
                        first_pixels = ticket.result_numpy().copy()
                        assert np.any(first_pixels[..., :3] != first_pixels[0, 0, :3]), "initial probe is invisible"
                        set_queue(2000)
                        switched_frame = frame
                        stage = "hidden"
                    elif stage == "hidden" and frame >= switched_frame + 8:
                        ticket = frontend.request_render_target_readback(True)
                        stage = "capture_hidden"
                    elif stage == "capture_hidden" and ticket.done:
                        hidden_pixels = ticket.result_numpy()
                        assert np.all(hidden_pixels[..., :3] == hidden_pixels[0, 0, :3]), "queue edit left the probe in its old route"
                        set_queue(1000)
                        switched_frame = frame
                        stage = "restored"
                    elif stage == "restored" and frame >= switched_frame + 8:
                        ticket = frontend.request_render_target_readback(True)
                        stage = "capture_restored"
                    elif stage == "capture_restored" and ticket.done:
                        np.testing.assert_array_equal(ticket.result_numpy(), first_pixels)
                        completed = True
                        print(f"material queue GPU regression passed ({'asset reload' if asset_mode else 'live setter'}): 1000 visible -> 2000 hidden -> 1000 visible")
                        engine.exit()
                    if frame >= 180 and not completed:
                        raise AssertionError(f"material queue GPU test timed out in {stage}")
                except BaseException as exception:
                    failures.append(exception)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert completed
            return 0
        finally:
            engine.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
