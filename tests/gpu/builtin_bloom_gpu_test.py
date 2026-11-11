"""Built-in Slang Bloom: compare the whole image and live effect edits to CPU filtering."""
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, RenderPipelineCallback, SceneManager
from infernux.renderstack import EffectSlot, RenderPipeline, RenderStack


def sample(image, height, width, offset=(0, 0)):
    sy, sx = image.shape[:2]
    x = (np.arange(width) + 0.5) * sx / width - 0.5 + offset[0]
    y = (np.arange(height) + 0.5) * sy / height - 0.5 + offset[1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    tx, ty = (x - x0)[None, :, None], (y - y0)[:, None, None]
    a, b = np.clip(x0, 0, sx - 1), np.clip(x0 + 1, 0, sx - 1)
    c, d = np.clip(y0, 0, sy - 1), np.clip(y0 + 1, 0, sy - 1)
    return ((image[c[:, None], a] * (1 - tx) + image[c[:, None], b] * tx) * (1 - ty)
            + (image[d[:, None], a] * (1 - tx) + image[d[:, None], b] * tx) * ty)


def reference(intensity, tint, scatter):
    x, y = np.meshgrid((np.arange(64) + 0.5) / 64, (np.arange(48) + 0.5) / 48)
    source = np.stack((0.02 + 0.04 * x, np.full_like(x, 0.03), np.full_like(x, 0.04)), axis=-1)
    source += ((abs(x - 0.5) < 0.1) & (abs(y - 0.5) < 0.25))[..., None] * (3, 2, 1)
    half = lambda value: value.astype(np.float16).astype(np.float32)
    source = half(source)
    color = sample(source, 24, 32)
    brightness = color.max(axis=-1, keepdims=True)
    soft = np.clip(brightness - 0.5 + 0.25, 0, 0.5) ** 2 / (1 + 1e-5)
    mips = [half(color * np.maximum(soft, brightness - 0.5) / np.maximum(brightness, 1e-5))]
    taps = [(0, 0, .125)] + [(x, y, .03125) for x in (-1, 1) for y in (-1, 1)]
    taps += [(x, y, .0625) for x, y in ((0, -1), (-1, 0), (1, 0), (0, 1))]
    taps += [(x, y, .125) for x in (-.5, .5) for y in (-.5, .5)]
    for height, width in ((12, 16), (6, 8)):
        mips.append(half(sum(sample(mips[-1], height, width, (x, y)) * weight for x, y, weight in taps)))
    low = mips[-1]
    for high in reversed(mips[:-1]):
        height, width = high.shape[:2]
        tent = sum(sample(low, height, width, (x, y)) * (2 if x == 0 else 1) * (2 if y == 0 else 1)
                   for x in (-1, 0, 1) for y in (-1, 0, 1)) / 16
        low = half(high * (1 - scatter) + tent * scatter)
    linear = np.clip(half(source + sample(low, 48, 64) * np.array(tint) * intensity), 0, 1)
    return np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)


class BloomPipeline(RenderPipeline):
    name = "Built-in Bloom audit"
    builds = 0

    def define_topology(self, graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        color = graph.create_texture("color", camera_target=True)
        graph.add_pass("Source").write_color(color).fullscreen_quad(str(self.source))
        result = graph.publish_pass_result("scene", {"color": color})
        with graph.pass_result(result):
            graph.effects("post", scope="composite", inputs={"color"}, outputs={"color"})
        graph.set_output(color)


class Host(RenderPipelineCallback):
    def __init__(self):
        super().__init__()
        self.stack = RenderStack()

    def render(self, context, camera):
        self.stack.render(context, camera)


def run(project):
    for directory in ("Assets", "Packages", "ProjectSettings"):
        (project / directory).mkdir(parents=True, exist_ok=True)
    source = project / "Assets/Source.frag.py"
    source.write_text('''import infernux.shader as shader
class Source(shader.fullscreen):
    def main(self, uv: shader.vec2) -> shader.vec4:
        color = shader.vec3(0.02 + 0.04 * uv.x, 0.03, 0.04)
        if shader.abs(uv.x - 0.5) < 0.1 and shader.abs(uv.y - 0.5) < 0.25:
            color += shader.vec3(3.0, 2.0, 1.0)
        return shader.vec4(color, 1.0)
''', encoding="utf-8")
    effect_path = project / "Assets/Bloom.effect"
    effect_path.write_text(json.dumps({"$schema": "infernux.render_effect", "feature_type": "infernux.post.bloom",
                                       "parameters": {"threshold": .5, "intensity": 0., "max_iterations": 3},
                                       "dependencies": []}), encoding="utf-8")
    frontend, host, console = inx.Engine(), Host(), ConsolePanel()
    native = frontend.get_native_engine()
    failures, completed = [], False
    try:
        frontend.init_renderer(64, 48, str(project))
        frontend.resize_game_render_target(64, 48)
        native.set_scene_view_visible(False)
        native.set_editor_fps_cap(240)
        native.set_editor_idle_fps(0)
        database = frontend.get_asset_database()
        for path in (source, effect_path):
            imported = AssetManager.import_asset(str(path), database=database)
            assert imported, imported.error
        effect_guid = imported.guid
        scene = SceneManager.instance().get_active_scene()
        for obj in scene.get_root_objects():
            scene.destroy_game_object(obj)
        scene.create_game_object("Bloom camera").add_component(inx.Camera)
        pipeline = BloomPipeline()
        pipeline.source = source
        host.stack._pipeline = pipeline
        host.stack.effect_slots = [EffectSlot(stage_id="post", slot_id="bloom", effect=RenderEffectRef(guid=effect_guid))]
        frontend.set_render_pipeline(host)
        native.set_game_camera_enabled(True)
        effect = AssetManager.load_by_guid(effect_guid)
        plans = ((0., (1., 1., 1.), .7), (.5, (1., 1., 1.), .7),
                 (1.5, (.2, .6, 1.), .3), (.8, (1., .2, .1), 1.))
        phase, frame, changed, ticket, initial_builds = 0, 0, 0, None, None

        def after_draw():
            nonlocal phase, frame, changed, ticket, completed, initial_builds
            try:
                frame += 1
                if ticket is None and frame >= changed + 8:
                    ticket = frontend.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy().copy().astype(np.float32)
                    expected = reference(*plans[phase])
                    np.testing.assert_allclose(pixels[..., :3], expected, atol=.006)
                    np.testing.assert_allclose(pixels[..., 3], 1., atol=.001)
                    assert not host.stack.effect_compile_errors, host.stack.effect_compile_errors
                    if initial_builds is None:
                        initial_builds = pipeline.builds
                    assert pipeline.builds == initial_builds, (initial_builds, pipeline.builds)
                    assert not [row for row in console._get_visible_log_snapshot(2000) if row["level"] in ("ERROR", "FATAL")]
                    print("BUILTIN_BLOOM", phase, "max_error", float(np.max(abs(pixels[..., :3] - expected))), flush=True)
                    phase += 1
                    if phase == len(plans):
                        completed = True
                        native.exit()
                        return
                    intensity, tint, scatter = plans[phase]
                    effect.set_float("intensity", intensity)
                    effect.set_color("tint", *tint)
                    effect.set_float("scatter", scatter)
                    changed, ticket = frame, None
                if frame > 150:
                    raise AssertionError("Built-in Bloom readback timed out")
            except BaseException as error:
                failures.append(error)
                native.exit()

        native.set_post_draw_callback(after_draw)
        native.run()
        if failures:
            raise failures[0]
        assert completed
        print("BUILTIN_BLOOM_RENDERGRAPH_OK", flush=True)
    finally:
        frontend.set_render_pipeline(None)
        host.stack.on_destroy()
        native.cleanup()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="infernux-builtin-bloom-") as folder:
        run(Path(folder))
