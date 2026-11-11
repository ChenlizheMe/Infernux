"""Project-authored fullscreen programs execute through the real RenderGraph."""
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, GraphParameterBlockUpdate, SceneManager, Vector3
from infernux.renderstack import RenderPipeline
from infernux.rendergraph import Format
from infernux.debug import DebugConsole


class AuthorPipeline(RenderPipeline):
    name = "Python Shader Pipeline"

    def define_topology(self, graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        source = graph.create_texture("source", format=Format.RGBA16_SFLOAT, samples=1)
        color = graph.create_texture("color", camera_target=True)
        graph.add_pass("Fill").write_color(source).fullscreen_quad(str(self.fill))
        graph.add_pass("Grayscale").write_color(color).set_texture("image", source).bind_parameter_block(
            "grayscale", {"strength": 0.0},
        ).fullscreen_quad(str(self.gray))
        graph.screen_ui_overlay_section(resources={"color"})
        graph.set_output(color)

    def render_camera(self, context, camera, culling):
        if not context.is_graph_revision_current(self._standalone_desc.source_revision):
            assert context.apply_graph(self._standalone_desc)
        update = GraphParameterBlockUpdate()
        update.id = "grayscale"
        update.revision = self.phase + 1
        update.values = [("strength", min(self.phase, 2) * 0.5)]
        context.update_parameter_blocks([update])
        context.submit_culling(culling)


def run(project):
    for directory in ("Assets/Shaders", "Packages", "ProjectSettings"):
        (project / directory).mkdir(parents=True, exist_ok=True)
    fill = project / "Assets/Shaders/填色.frag.py"
    gray = project / "Assets/Shaders/灰度.frag.py"
    palette = project / "Assets/Shaders/palette.shaderlib.py"
    palette.write_text('''import infernux.shader as shader

def tweak(color: shader.vec4) -> shader.vec4:
    return shader.vec4(color.rgb * 1.0, color.a)
''', encoding="utf-8")
    unused = project / "Assets/Shaders/unused.shaderlib.py"
    unused.write_text(palette.read_text(encoding="utf-8"), encoding="utf-8")
    fill.write_text('''import infernux.shader as shader
from .palette import tweak

class Fill(shader.fullscreen):
    tint = shader.parameter.color(default=(0.2, 0.4, 0.8, 1.0))
    transform = shader.parameter.mat3(default=((1.0, 0.25, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.5)))
    enabled = shader.parameter.bool(default=True)
    def main(self, uv: shader.vec2) -> shader.vec4:
        rgb = self.transform @ self.tint.rgb
        return tweak(shader.vec4(rgb, self.tint.a)) if self.enabled else shader.vec4(0.0)
''', encoding="utf-8")
    gray.write_text('''import infernux.shader as shader
from infernux.shader.lib.color import luminance
from .palette import tweak

class Grayscale(shader.fullscreen):
    image = shader.input("scene", shader.Color2D, domain="scene_linear_hdr")
    strength = shader.parameter.float(default=0.75, range=(0.0, 1.0))
    def main(self, uv: shader.vec2) -> shader.vec4:
        color = tweak(shader.sample(self.image, uv))
        return shader.vec4(shader.lerp(color.rgb, shader.vec3(luminance(color.rgb)), self.strength), color.a)
''', encoding="utf-8")
    frontend = inx.Engine()
    native = frontend.get_native_engine()
    pipeline = AuthorPipeline()
    pipeline.fill, pipeline.gray = fill, gray
    pipeline.builds = pipeline.phase = 0
    console = ConsolePanel()
    completed, failures = False, []
    try:
        frontend.init_renderer(64, 48, str(project))
        frontend.resize_game_render_target(64, 48)
        native.set_scene_view_visible(False)
        native.set_editor_fps_cap(240)
        native.set_editor_idle_fps(0)
        database = native.get_asset_database()
        for path in (palette, unused, fill, gray):
            imported = database.import_asset(str(path))
            assert imported, imported.error
        # Draft checking uses the asset IO and compiler, without publication.
        assert inx.Shader.check(str(fill))["valid"]
        draft = fill.read_text(encoding="utf-8").replace("self.tint.rgb", "self.missing.rgb")
        checked = inx.Shader.check(str(fill), source=draft)
        assert not checked["valid"] and checked["diagnostics"][0]["phase"] == "author"
        assert checked["diagnostics"][0]["path"] == str(fill)
        assert inx.Shader.check(str(fill))["valid"]
        library_schema = inx.Shader.get_schema(str(palette))
        assert library_schema["domain"] == "library"
        assert library_schema["functions"][0]["name"] == "tweak"
        scene = SceneManager.instance().get_active_scene()
        for obj in scene.get_root_objects():
            scene.destroy_game_object(obj)
        camera = scene.create_game_object("着色器测试相机")
        camera.transform.position = Vector3(0, 0, -5)
        camera.add_component("Camera")
        frontend.set_render_pipeline(pipeline)
        native.set_game_camera_enabled(True)
        frame, changed, ticket = 0, 0, None
        broken_source = None

        def after_draw():
            nonlocal frame, changed, ticket, completed, broken_source
            try:
                frame += 1
                if ticket is None and frame >= changed + 6:
                    ticket = frontend.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy().copy().astype(np.float32)
                    original = np.array([0.3, 0.4, 0.4], dtype=np.float32)
                    gray_value = np.dot(original, [0.2126, 0.7152, 0.0722])
                    strength = min(pipeline.phase, 2) * 0.5
                    if pipeline.phase == 3:
                        gray_value *= 0.5
                    elif pipeline.phase == 6:
                        gray_value *= 0.25
                    linear = original * (1 - strength) + gray_value * strength
                    expected = np.r_[np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055), 1.0]
                    print("AUTHOR_FULLSCREEN", pipeline.phase, pixels[24, 32].tolist(), "builds", pipeline.builds, flush=True)
                    np.testing.assert_allclose(pixels, np.broadcast_to(expected, pixels.shape), atol=0.005)
                    assert pipeline.builds == 1
                    errors = [item for item in console._get_visible_log_snapshot(3000) if item["level"] in ("ERROR", "FATAL")]
                    python_errors = DebugConsole.instance().get_filtered_entries(
                        show_logs=False, show_warnings=False, show_errors=True,
                    )
                    messages = [item["message"] for item in errors] + [item.message for item in python_errors]
                    if pipeline.phase == 5:
                        assert messages and all("missing" in message for message in messages), messages
                    else:
                        assert not messages, messages
                    if pipeline.phase == 7:
                        completed = True
                        native.exit()
                        return
                    pipeline.phase += 1
                    if pipeline.phase == 3:
                        text = gray.read_text(encoding="utf-8")
                        gray.write_text(text.replace("shader.vec3(luminance(color.rgb))", "shader.vec3(0.5 * luminance(color.rgb))"), encoding="utf-8")
                        assert inx.Shader.reload(str(gray))
                    elif pipeline.phase == 4:
                        text = gray.read_text(encoding="utf-8")
                        gray.write_text(text.replace("shader.vec3(0.5 * luminance(color.rgb))", "shader.vec3(luminance(color.rgb))"), encoding="utf-8")
                        assert inx.Shader.reload(str(gray))
                    elif pipeline.phase == 5:
                        registry = AssetManager._get_registry()
                        roots = [(database.get_guid_from_path(str(path)), path) for path in (fill, gray)]
                        versions = {guid: registry.get_asset_version(guid) for guid, _ in roots}
                        assert inx.Shader.reload(str(unused))
                        assert {guid: registry.get_asset_version(guid) for guid, _ in roots} == versions
                        bad = max(roots, key=lambda row: row[0])[1]
                        original_source = bad.read_text(encoding="utf-8")
                        broken_source = (bad, original_source)
                        bad.write_text(original_source.replace("return ", "return self.missing + ", 1), encoding="utf-8")
                        palette.write_text(palette.read_text(encoding="utf-8").replace(" * 1.0,", " * 0.5,"), encoding="utf-8")
                        try:
                            inx.Shader.reload(str(palette))
                        except RuntimeError as error:
                            assert "missing" in str(error)
                        else:
                            raise AssertionError("An invalid consumer must reject the whole library publication")
                        assert {guid: registry.get_asset_version(guid) for guid, _ in roots} == versions
                    elif pipeline.phase == 6:
                        bad, original_source = broken_source
                        bad.write_text(original_source, encoding="utf-8")
                        assert inx.Shader.reload(str(palette))
                    elif pipeline.phase == 7:
                        palette.write_text(palette.read_text(encoding="utf-8").replace(" * 0.5,", " * 1.0,"), encoding="utf-8")
                        assert inx.Shader.reload(str(palette))
                    changed, ticket = frame, None
                if frame > 130:
                    raise AssertionError("Author fullscreen readback timed out")
            except BaseException as error:
                failures.append(error)
                native.exit()

        native.set_post_draw_callback(after_draw)
        native.run()
        if failures:
            raise failures[0]
        assert completed
        print("AUTHOR_RENDERGRAPH_FULLSCREEN_OK", flush=True)
    finally:
        frontend.set_render_pipeline(None)
        pipeline.dispose()
        native.cleanup()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="infernux-slang-fullscreen-") as folder:
        run(Path(folder))
