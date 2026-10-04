"""A Deferred route must not shade depth written by a preceding Forward route."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from infernux import Engine
from infernux.core.assets import AssetManager
from infernux.lib import InxMaterial, SceneManager, Vector3
from infernux.renderstack import Path as RenderPath, Queue, RenderPipeline


class MixedRouteProbe(RenderPipeline):
    name = "Mixed Deferred Coverage Probe"

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def define(self, pipeline):
        pipeline.frame(hdr=True, msaa=self.samples)
        pipeline.lighting(clustered=True)
        with pipeline.opaque() as opaque:
            with opaque.layer("Mixed") as layer:
                layer.forward(Queue(1, 100))
                layer.deferred(Queue(101, 200), fallback=RenderPath.FORWARD_PLUS)
            opaque.otherwise().forward_plus()
        pipeline.screen_ui()


def assert_route_colors(pixels, label):
    rgb = pixels[..., :3].astype(np.float32)
    if np.issubdtype(pixels.dtype, np.integer):
        rgb /= np.iinfo(pixels.dtype).max
    red = (rgb[..., 0] > .8) & (rgb[..., 1] < .12) & (rgb[..., 2] < .12)
    green = (rgb[..., 0] < .12) & (rgb[..., 1] > .8) & (rgb[..., 2] < .12)
    magenta = (rgb[..., 0] > .8) & (rgb[..., 1] < .12) & (rgb[..., 2] > .8)
    counts = {"red": int(red.sum()), "green": int(green.sum()), "magenta": int(magenta.sum())}
    assert counts["red"] > 10 and counts["green"] > 10 and counts["magenta"] == 0, (label, counts, str(pixels.dtype), rgb.max(axis=(0, 1)))
    print(label, counts, flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-mixed-deferred-") as root:
        project = Path(root)
        assets = project / "Assets"
        assets.mkdir()
        (project / "ProjectSettings").mkdir()
        mesh = assets / "Probe.obj"
        mesh.write_text("v -1 -1 0\nv 1 -1 0\nv 0 1 0\nvn 0 0 -1\nvn 0 0 1\nf 1//1 2//1 3//1\nf 3//2 2//2 1//2\n", encoding="ascii")
        frontend = Engine()
        engine = frontend.get_native_engine()
        failures = []
        completed = False
        try:
            frontend.init_renderer(96, 64, str(project))
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            frontend.resize_game_render_target(96, 64)
            imported = AssetManager.import_asset(str(mesh), database=frontend.get_asset_database())
            assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            camera = scene.create_game_object("Camera")
            camera.transform.position = Vector3(0, 0, -4)
            camera.add_component("Camera")
            for name, x, queue, color in [("Forward Red", -.85, 50, (1., 0., 0., 1.)), ("Deferred Green", .85, 150, (0., 1., 0., 1.))]:
                obj = scene.create_game_object(name)
                obj.transform.position = Vector3(x, 0, 0)
                obj.transform.local_scale = Vector3(.6, .6, .6)
                renderer = obj.add_component("MeshRenderer")
                renderer.set_mesh_asset_guid(imported.guid)
                material = InxMaterial.create_default_unlit()
                material.set_color("baseColor", *color)
                material.set_render_queue(queue)
                renderer.set_material(0, material)
            frontend.set_render_pipeline(MixedRouteProbe(4))
            engine.set_game_camera_enabled(True)
            frame, switched_frame = 0, 0
            stage, ticket = "baseline", None

            def after_draw():
                nonlocal frame, switched_frame, stage, ticket, completed
                try:
                    frame += 1
                    if stage == "baseline" and frame >= 8:
                        ticket = frontend.request_render_target_readback(True)
                        stage = "baseline_readback"
                    elif stage == "baseline_readback" and ticket.done:
                        assert_route_colors(ticket.result_numpy(), "MSAA4 explicit Forward+ route")
                        frontend.set_render_pipeline(MixedRouteProbe(1))
                        switched_frame, stage = frame, "deferred"
                    elif stage == "deferred" and frame >= switched_frame + 8:
                        ticket = frontend.request_render_target_readback(True)
                        stage = "deferred_readback"
                    elif stage == "deferred_readback" and ticket.done:
                        assert_route_colors(ticket.result_numpy(), "MSAA1 mixed Forward/Deferred routes")
                        completed = True
                        engine.exit()
                    if frame > 150:
                        raise AssertionError("Mixed route readback did not complete")
                except BaseException as exc:
                    failures.append(exc)
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
