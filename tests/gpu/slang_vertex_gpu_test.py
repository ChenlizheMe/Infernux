"""A standalone vertex asset drives every geometry pass, history and hot publication."""

from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3
from slang_surface_buffers_gpu_test import BuffersPipeline, CONSUMER, SURFACE


VERTEX = '''import infernux.shader as shader
class Translate(shader.vert):
    shader_id = "tests/vertex/translate"
    offset = shader.material.vec3()
    rate = shader.material.float()
    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        vertex.position_os += self.offset
        vertex.position_os.x += ctx.time * self.rate
        return vertex
'''


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-slang-vertex-") as directory:
        root = Path(directory)
        for name in ("Assets", "ProjectSettings"):
            (root / name).mkdir()
        vertex_path = root / "Assets/translate.vert.py"
        other_path = root / "Assets/other.vert.py"
        conflict_path = root / "Assets/conflict.vert.py"
        surface_path = root / "Assets/cutout.frag.py"
        consumer_path = root / "Assets/compare.frag.py"
        surface_source = SURFACE
        vertex_path.write_text(VERTEX, encoding="utf8")
        other_path.write_text(VERTEX.replace('tests/vertex/translate', 'tests/vertex/other').replace('ctx.time * self.rate', '0.4 + ctx.time * self.rate'), encoding="utf8")
        conflict_path.write_text(VERTEX.replace('tests/vertex/translate', 'tests/vertex/conflict')
                                 .replace('offset = shader.material.vec3()', 'offset = shader.material.vec3()\n    clip_edge = shader.material.vec3()'), encoding="utf8")
        surface_path.write_text(surface_source, encoding="utf8")
        consumer_path.write_text(CONSUMER, encoding="utf8")
        frontend, console, pipeline = inx.Engine(), ConsolePanel(), BuffersPipeline()
        native = frontend.get_native_engine()
        pipeline.builds, pipeline.consumer = 0, str(consumer_path)
        failures, complete = [], False
        try:
            frontend.init_renderer(160, 120, directory)
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            database = frontend.get_asset_database()
            vertex_guid = database.get_guid_from_path(str(vertex_path))
            other_guid = database.get_guid_from_path(str(other_path))
            assert vertex_guid and other_guid
            scene = SceneManager.instance().get_active_scene()
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            camera = scene.create_game_object("相机")
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component(inx.Camera)
            owner = scene.create_game_object("顶点变形")
            renderer = owner.add_component(inx.MeshRenderer)
            renderer.set_inline_mesh_data(
                np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
                np.array([[0, 0, -1]] * 4, dtype=np.float32),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32), "Vertex Quad")
            material = inx.Material.from_shader(str(surface_path))
            fragment_reference = material.native.frag_shader_reference
            material.vert_shader_name = "tests/vertex/translate"
            assert material.native.vert_shader_reference["guid"] == vertex_guid
            material.alpha_clip_enabled, material.alpha_clip_threshold = True, 0.5
            renderer.set_material(material)
            pipeline.object_id = owner.id
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            frame = changed = phase = 0
            ticket = baseline = None
            baseline_center = None
            labels = ("baseline", "parameter_offset", "parameter_restore", "time_motion", "motion_restore",
                      "vertex_hot_edit", "vertex_restore", "vertex_switch", "switch_restore",
                      "legacy_pair_rejected", "pair_recovered", "field_conflict_rejected", "field_conflict_recovered",
                      "first_invalid_suspended", "first_invalid_recovered")
            rejected = None

            def after_draw():
                nonlocal frame, changed, phase, ticket, baseline, baseline_center, complete, rejected
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        red = (pixels[..., 0] > 0.9) & (pixels[..., 1] < 0.05) & (pixels[..., 2] < 0.05)
                        green = (pixels[..., 1] > 0.9) & (pixels[..., 0] < 0.05) & (pixels[..., 2] < 0.05)
                        covered = red | green
                        bad = (pixels[..., 0] > 0.9) & (pixels[..., 2] > 0.9)
                        assert not bad.any(), (labels[phase], np.unique(pixels[bad, 1]))
                        if phase == 13:
                            assert not covered.any(), labels[phase]
                        else:
                            assert covered.sum() > 100, labels[phase]
                        assert pipeline.builds == 1
                        errors = [row for row in console._get_visible_log_snapshot(2000) if row["level"] in ("ERROR", "FATAL")]
                        if phase < 9:
                            assert not errors, errors
                        elif phase == 9:
                            assert any("requires a Python mesh fragment" in row["message"] for row in errors), errors
                        elif phase == 11:
                            assert any("disagree on material field" in row["message"] for row in errors), errors
                        center = float(np.nonzero(covered)[1].mean()) if covered.any() else 0.0
                        if phase == 0:
                            baseline, baseline_center = pixels, center
                        elif phase in (2, 4, 6, 8, 9, 10, 11, 12, 14):
                            np.testing.assert_array_equal(pixels, baseline)
                        elif phase in (1, 5, 7):
                            assert center > baseline_center + 3, (labels[phase], center, baseline_center)
                        if phase == 3:
                            assert green.sum() > covered.sum() * 0.9, (green.sum(), covered.sum())
                        else:
                            assert not green.any(), (labels[phase], green.sum())
                        print("VERTEX", labels[phase], "covered", int(covered.sum()), "moving", int(green.sum()), "center", center, flush=True)
                        phase += 1
                        if phase == len(labels):
                            complete = True
                            native.exit()
                            return
                        if phase in (1, 2):
                            material.set_vector3("offset", 0.5 if phase == 1 else 0, 0, 0)
                        elif phase in (3, 4):
                            material.set_float("rate", 0.3 if phase == 3 else 0)
                        elif phase in (5, 6):
                            vertex_path.write_text(VERTEX.replace("ctx.time * self.rate", "0.4 + ctx.time * self.rate") if phase == 5 else VERTEX, encoding="utf8")
                            assert inx.Shader.reload(str(vertex_path))
                        elif phase in (7, 8):
                            material.vert_shader_name = "tests/vertex/other" if phase == 7 else "tests/vertex/translate"
                        elif phase in (9, 10):
                            if phase == 9:
                                material.frag_shader_name = "Toon"
                            else:
                                material.native.frag_shader_reference = fragment_reference
                                material.alpha_clip_enabled, material.alpha_clip_threshold = True, 0.5
                        elif phase == 11:
                            material.vert_shader_name = "tests/vertex/conflict"
                        elif phase == 12:
                            conflict_path.write_text(conflict_path.read_text(encoding="utf8").replace(
                                'clip_edge = shader.material.vec3()', 'clip_edge = shader.material.float(default=0.5)'), encoding="utf8")
                            assert inx.Shader.reload(str(conflict_path))
                        elif phase == 13:
                            rejected = inx.Material.from_shader(str(surface_path))
                            rejected.vert_shader_name = "tests/vertex/translate"
                            rejected.frag_shader_name = "Toon"
                            renderer.set_material(rejected)
                        elif phase == 14:
                            rejected.native.frag_shader_reference = fragment_reference
                            rejected.alpha_clip_enabled, rejected.alpha_clip_threshold = True, 0.5
                        changed, ticket = frame, None
                    if frame > 300:
                        raise AssertionError(("Vertex GPU timeout", labels[phase]))
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert complete
            print("SLANG_VERTEX_OK", flush=True)
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()


if __name__ == "__main__":
    main()
