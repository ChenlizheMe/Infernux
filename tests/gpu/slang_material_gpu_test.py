"""A static Python mesh author reaches real instanced draws and hot publication."""
from pathlib import Path
import argparse
import json
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import AssetRegistry, ConsolePanel, RenderPipelineCallback, SceneManager, Vector3
from infernux.rendergraph import Format, RenderGraph


SOURCE = '''import infernux.shader as shader

class Painted(shader.frag):
    tint = shader.material.vec4(default=(0.2, 0.4, 0.8, 1.0))
    gain = shader.material.float(default=0.75, range=(0.0, 2.0), label="强度")
    transform = shader.material.mat4(default=((1.0, 0.25, 0.0, 0.0), (0.0, 1.0, 0.5, 0.0),
                                             (0.1, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        transformed = self.transform @ shader.vec4(self.tint.rgb, 0.0)
        return shader.vec4(transformed.rgb * self.gain, self.tint.a)
'''


class MeshPipeline(RenderPipelineCallback):
    def __init__(self, material_pass):
        super().__init__()
        graph = RenderGraph("Python mesh author")
        graph.set_msaa_samples(1)
        color = graph.create_texture("color", camera_target=True)
        depth = graph.create_texture("depth", format=Format.D32_SFLOAT)
        graph.add_pass("Clear").write_color(color).write_depth(depth).set_clear(color=(0, 0, 0, 1), depth=0.0)
        graph.add_pass("Meshes").write_color(color).write_depth(depth).draw_renderers(material_pass=material_pass)
        graph.set_output(color)
        self.description = graph.build()

    def render(self, context, camera):
        context.render_with_graph(camera, self.description)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-pass", choices=("forward", "forward_plus"), default="forward")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="infernux-slang-material-") as root:
        project = Path(root)
        for directory in ("Assets", "Packages", "ProjectSettings"):
            (project / directory).mkdir()
        path = project / "Assets/绘制.frag.py"
        path.write_text(SOURCE, encoding="utf-8")
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        failures, completed = [], False
        try:
            frontend.init_renderer(160, 120, root)
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            imported = AssetManager.import_asset(str(path), database=frontend.get_asset_database())
            assert imported, imported.error
            schema = inx.Shader.get_schema(str(path))
            assert schema["domain"] == "mesh"
            from infernux.engine.ui.inspector_shader_utils import parse_shader_properties
            fields = parse_shader_properties(str(path))
            assert fields[1]["range"] == [0.0, 2.0] and fields[1]["label"] == "强度"
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object("相机")
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component(inx.Camera)
            materials = []
            for index in range(2):
                obj = scene.create_game_object("网格" + str(index))
                obj.transform.position = Vector3(-0.75 + 1.5 * index, 0.35 * (1 - 2 * index), 0)
                renderer = obj.add_component(inx.MeshRenderer)
                renderer.set_inline_mesh_data(
                    np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32),
                    np.array([[0, 0, -1]] * 4, dtype=np.float32),
                    np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                    np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32), "Quad")
                material = inx.Material.from_shader(str(path), "材质" + str(index))
                material.cull_mode = 2
                if index:
                    material.set_vector4("tint", 0.8, 0.2, 0.3, 1.0)
                saved = project / f"Assets/材质{index}.mat"
                assert material.save(str(saved))
                imported = AssetManager.import_asset(str(saved), database=frontend.get_asset_database())
                assert imported, imported.error
                material = inx.Material.load(str(saved))
                assert material is not None and material.get_float("gain") == 0.75
                renderer.set_material(0, material)
                materials.append(material)
            matrix = np.array(((1, 0.25, 0, 0), (0, 1, 0.5, 0), (0.1, 0, 1, 0), (0, 0, 0, 1)), dtype=np.float32)
            np.testing.assert_array_equal(materials[0].get_matrix("transform"), matrix)
            post = project / "Assets/后处理.frag.py"
            post.write_text('''import infernux.shader as shader
class Fill(shader.fullscreen):
    def main(self) -> shader.vec4:
        return shader.vec4(1.0)
''', encoding="utf-8")
            imported_post = AssetManager.import_asset(str(post), database=frontend.get_asset_database())
            assert imported_post, imported_post.error
            bad_path = project / "Assets/错误渲染域.mat"
            bad_document = materials[0].to_dict()
            bad_document["shaders"]["fragment"] = {"guid": imported_post.guid, "shader_id": "后处理"}
            bad_path.write_text(json.dumps(bad_document, ensure_ascii=False), encoding="utf-8")
            imported_bad = AssetManager.import_asset(str(bad_path), database=frontend.get_asset_database())
            assert imported_bad, imported_bad.error
            try:
                inx.Material.load(str(bad_path))
            except ValueError as error:
                assert "mesh shader author" in str(error)
            else:
                raise AssertionError("A fullscreen author was accepted as a mesh material")
            assert AssetRegistry.instance().get_asset_runtime_type_name(imported_bad.guid) == ""
            frontend.set_render_pipeline(MeshPipeline(options.material_pass))
            native.set_game_camera_enabled(True)
            frame = changed = phase = 0
            ticket = baseline = None

            def after_draw():
                nonlocal frame, changed, phase, ticket, baseline, completed
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 6:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        # Inline meshes use the normal asynchronous upload lane.
                        # The initial numeric check starts after both are drawable.
                        if phase == 0 and any(np.count_nonzero(np.max(pixels[:, begin:begin + 80, :3], axis=-1) > 0.05) <= 50
                                              for begin in (0, 80)):
                            if frame > 80:
                                raise AssertionError("Initial mesh uploads did not become drawable")
                            changed, ticket = frame, None
                            return
                        for index, tint in enumerate(((0.2, 0.4, 0.8, 0), (0.8, 0.2, 0.3, 0))):
                            transform = matrix.T if phase >= 2 and index == 0 else matrix
                            gain = 0.25 if phase >= 1 and index == 0 else 0.75
                            linear = (transform @ np.array(tint))[:3] * gain * (0.5 if phase == 3 else 1.0)
                            half = pixels[:, index * 80:(index + 1) * 80, :3]
                            visible = half[np.max(half, axis=-1) > 0.05]
                            assert len(visible) > 50, console._get_visible_log_snapshot(500)
                            rows = np.where(np.max(half, axis=-1) > 0.05)[0]
                            assert rows.mean() < 59.5 if index == 0 else rows.mean() > 59.5
                            np.testing.assert_allclose(visible, np.broadcast_to(linear, visible.shape), atol=0.006)
                        errors = [item for item in console._get_visible_log_snapshot(3000) if item["level"] in ("ERROR", "FATAL")]
                        # The Project preview may also attempt the deliberately
                        # invalid document. Only that exact rejection is expected.
                        errors = [item for item in errors if not (
                            "Material preview rejected" in item["message"] and "错误渲染域.mat" in item["message"]
                            and "Material requires a compiled mesh shader author" in item["message"])]
                        assert not errors, errors
                        assert AssetRegistry.instance().get_asset_runtime_type_name(imported_bad.guid) == ""
                        print("MESH_AUTHOR", phase, "passed", flush=True)
                        if phase == 0:
                            baseline = pixels
                        elif phase == 1:
                            np.testing.assert_array_equal(pixels[:, 80:], baseline[:, 80:])
                        if phase == 4:
                            completed = True
                            native.exit()
                            return
                        phase += 1
                        if phase == 1:
                            materials[0].set_float("gain", 0.25)
                        elif phase == 2:
                            materials[0].set_matrix("transform", matrix.T)
                        else:
                            source = SOURCE.replace("transformed.rgb * self.gain", "transformed.rgb * self.gain * 0.5") if phase == 3 else SOURCE
                            path.write_text(source, encoding="utf-8")
                            assert inx.Shader.reload(str(path))
                        changed, ticket = frame, None
                    if frame > 160:
                        raise AssertionError("Mesh author audit did not finish")
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
        print("PASS static Python mesh material", flush=True)


if __name__ == "__main__":
    main()
