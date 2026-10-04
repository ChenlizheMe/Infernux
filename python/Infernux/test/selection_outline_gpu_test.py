"""Real Scene View coverage: no sky and a vertex shader without material properties."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import Infernux as inx
from Infernux import Engine
from Infernux.core.assets import AssetManager
from Infernux.gizmos import Gizmos
from Infernux.lib import ConsolePanel, InxMaterial, SceneManager, Vector3
from Infernux.renderstack import RenderPipeline


class NoSkyPipeline(RenderPipeline):
    name = "No Sky Outline Probe"

    def define(self, pipeline):
        pipeline.frame(hdr=False, msaa=1)
        with pipeline.opaque() as opaque:
            opaque.otherwise().forward()
        pipeline.screen_ui()


VERTEX = '''#version 450
ShaderInfo { Name "Outline Shift Probe" }
void vertex(inout VertexInput v) {
    v.position.xy += vec2(1.4, 0.6);
}
'''

VERTEX_PROPERTIES = '''#version 450
ShaderInfo {
    Name "Outline Shift Probe"
    Properties { Float xShift = 0.0 Float yShift = 0.0 }
}
void vertex(inout VertexInput v) {
    v.position.xy += vec2(material.xShift, material.yShift);
}
'''


class BackgroundGizmoProbe(inx.InxComponent):
    def on_draw_gizmos(self):
        Gizmos.color = (0.0, 1.0, 0.0)
        Gizmos.draw_line((-1.5, 1.5, 0.0), (0.5, 1.5, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=("background", "deformation", "multiple", "reload", "game-alpha", "gizmo", "properties"), required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="infernux-outline-gpu-") as root:
        project = Path(root)
        assets = project / "Assets"
        assets.mkdir()
        (project / "ProjectSettings").mkdir()
        (project / "Packages").mkdir()
        mesh = assets / "Probe.obj"
        mesh.write_text(
            "v -0.5 -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv -0.5 0.5 0\n"
            "vn 0 0 -1\nvn 0 0 1\n"
            "f 1//1 2//1 3//1\nf 1//1 3//1 4//1\n"
            "f 3//2 2//2 1//2\nf 4//2 3//2 1//2\n", encoding="ascii",
        )
        vertex = assets / "Shift.vert"
        vertex.write_text(VERTEX_PROPERTIES if args.check == "properties" else VERTEX, encoding="ascii")
        frontend = Engine()
        engine = frontend.get_native_engine()
        console = ConsolePanel()
        failures = []
        completed = False
        try:
            frontend.init_renderer(160, 120, str(project))
            frontend.resize_scene_render_target(160, 120)
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            engine.set_show_grid(False)
            engine.set_scene_view_visible(True)
            engine.editor_camera.restore_state(0, 0, -5, 0, 0, 0, 5, 0, 0)
            database = frontend.get_asset_database()
            imported = AssetManager.import_asset(str(mesh), database=database)
            assert imported, imported.error
            shader = AssetManager.import_asset(str(vertex), database=database)
            assert shader, shader.error
            scene = SceneManager.instance().get_active_scene()
            # Native renderer initialization creates sample geometry. Remove it
            # so occlusion by that cube/ground cannot distort this silhouette oracle.
            for root_object in scene.get_root_objects():
                scene.destroy_game_object(root_object)
            probe = scene.create_game_object("Shifted Probe")
            renderer = probe.add_component("MeshRenderer")
            renderer.set_mesh_asset_guid(imported.guid)
            material = InxMaterial.create_default_unlit()
            material.set_color("baseColor", 0.0, 0.0, 1.0, 1.0)
            material.vert_shader_name = "Outline Shift Probe"
            if args.check == "properties":
                material.set_float("xShift", 1.4)
                material.set_float("yShift", .6)
            renderer.set_material(0, material)
            selected_ids = [probe.id]
            if args.check == "multiple":
                other = scene.create_game_object("Second Shifted Probe")
                other.transform.position = Vector3(-2.5, -1.2, 0)
                other_renderer = other.add_component("MeshRenderer")
                other_renderer.set_mesh_asset_guid(imported.guid)
                other_renderer.set_material(0, material)
                selected_ids.append(other.id)
            if args.check == "game-alpha":
                camera = scene.create_game_object("Game Camera")
                camera.transform.position = Vector3(0, 0, -5)
                camera.add_component("Camera")
                frontend.resize_game_render_target(160, 120)
                engine.set_game_camera_enabled(True)
            frontend.set_render_pipeline(NoSkyPipeline())
            frame, changed = 0, 0
            stage, ticket, baseline = "baseline", None, None

            def after_draw():
                nonlocal frame, changed, stage, ticket, baseline, completed
                try:
                    frame += 1
                    if stage in ("baseline", "selected") and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(args.check == "game-alpha")
                        stage += "_readback"
                    elif stage.endswith("_readback") and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        if stage == "baseline_readback":
                            baseline = pixels
                            blue = (pixels[..., 2] > .9) & (pixels[..., 0] < .05)
                            print("Scene pixel maxima", pixels.max(axis=(0, 1)).tolist(), "blue", int(blue.sum()), flush=True)
                            if not blue.any():
                                print([e for e in console._get_visible_log_snapshot(1000) if e['level'] == 'ERROR'], flush=True)
                            assert blue.sum() > 100, ("Scene geometry missing", int(blue.sum()))
                            if args.check == "properties":
                                assert abs(np.where(blue)[1].mean() - 80) > 15, "Authored vertex material properties were not applied"
                            engine.set_selection_outlines(selected_ids)
                            if args.check == "gizmo":
                                scene.create_game_object("Background Gizmo").add_py_component(BackgroundGizmoProbe())
                            if args.check == "game-alpha":
                                background = np.all(pixels[..., :3] < .001, axis=-1)
                                assert background.sum() > 100
                                assert np.all(pixels[..., 3][background] == 0.0), "Scene opacity policy leaked into Game output"
                                print("Game output preserves transparent background alpha", flush=True)
                                completed = True
                                engine.exit()
                                return
                            stage = "selected"
                            changed = frame
                        else:
                            foreground = (baseline[..., 2] > .9) & (baseline[..., 0] < .05)
                            background = np.all(baseline[..., :3] < .001, axis=-1)
                            outline = (pixels[..., 0] > .7) & (pixels[..., 1] > .25) & (pixels[..., 2] < .1)
                            assert outline.sum() > 20, ("Outline absent", int(outline.sum()))
                            print("No-sky background alpha", float(baseline[..., 3][background].min()),
                                  "outline background pixels", int((outline & background).sum()), flush=True)
                            if args.check == "gizmo":
                                green = (pixels[..., 1] > .8) & (pixels[..., 0] < .05) & (pixels[..., 2] < .05)
                                gizmo_on_black = green & background
                                print("Gizmo pixels over black background", int(gizmo_on_black.sum()), flush=True)
                                assert gizmo_on_black.sum() > 20, "Gizmo was not drawn over empty background"
                                assert np.all(pixels[..., 3][gizmo_on_black] == 1.0), "Gizmo remains invisible through alpha"
                                completed = True
                                engine.exit()
                                return
                            if args.check == "background":
                                assert np.all(baseline[..., 3][background] == 1.0), "Scene View leaves background transparent"
                                assert np.all(pixels[..., 3][outline] == 1.0), "Outline is clipped by transparent background alpha"
                                assert (outline & background).sum() > 20
                            else:
                                y, x = np.where(foreground)
                                adjacent = np.zeros_like(foreground)
                                for dy in range(-4, 5):
                                    for dx in range(-4, 5):
                                        adjacent[np.clip(y + dy, 0, 119), np.clip(x + dx, 0, 159)] = True
                                misplaced = outline & ~adjacent
                                print("Outline outside rendered silhouette neighbourhood", int(misplaced.sum()),
                                      "of", int(outline.sum()), flush=True)
                                assert not misplaced.any(), "Selection mask ignores the material's vertex deformation"
                                if args.check == "multiple":
                                    for sign in (-1, 1):
                                        # Each independently transformed object needs its own outline.
                                        assert (outline & (np.indices(outline.shape)[1] * sign > (80 * sign))).sum() > 20
                                if args.check == "reload" and stage == "selected_readback":
                                    vertex.write_text(VERTEX.replace("vec2(1.4, 0.6)", "vec2(-1.4, -0.6)"), encoding="ascii")
                                    error = engine.reload_shader_runtime(str(vertex), "Outline Shift Probe")
                                    assert not error, error
                                    engine.set_selection_outlines([])
                                    stage = "baseline"
                                    args.check = "deformation"
                                    changed = frame
                                    return
                            completed = True
                            engine.exit()
                    if frame > 100:
                        raise AssertionError("Scene outline readback timed out")
                except BaseException as exception:
                    failures.append(exception)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.run()
            if failures:
                raise failures[0]
            assert completed
            errors = [entry for entry in console._get_visible_log_snapshot(1000)
                      if entry["level"] in ("ERROR", "FATAL", "WARN", "WARNING")]
            assert not errors, errors
        finally:
            engine.cleanup()


if __name__ == "__main__":
    main()
