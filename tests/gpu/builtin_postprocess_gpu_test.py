"""Built-in typed effects share one graph; live bool/int/color edits retain it."""
import colorsys
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager
from infernux.rendergraph import Format
from infernux.renderstack import EffectSlot
from builtin_bloom_gpu_test import BloomPipeline, Host


def half(value):
    return value.astype(np.float16).astype(np.float64)


Y, X = np.meshgrid((np.arange(24) + .5) / 24, (np.arange(32) + .5) / 32, indexing="ij")
UV = np.stack((X, Y), axis=-1)
NAMES = ("white_balance", "color_adjustments", "vignette", "sharpen", "pixelation",
         "chromatic_aberration", "tonemapping", "film_grain")
PLANS = (
    ({"temperature": 0., "tint": 0.},
     {"post_exposure": 0., "contrast": 0., "saturation": 0., "hue_shift": 0.},
     {"intensity": 0., "smoothness": .3, "roundness": 1., "rounded": False, "color": (0., 0., 0., 1.)},
     {"intensity": 0.},
     {"pixel_size": 4, "pixel_aspect": 1., "grid_offset_x": 0., "grid_offset_y": 0.,
      "intensity": 0., "sampling": 0, "preserve_alpha_coverage": False},
     {"intensity": 0.},
     {"mode": 0, "exposure": 1.},
     {"intensity": 0., "response": .8, "preset": 3, "size": 2.4, "colored": False}),
    ({"temperature": 25., "tint": -10.},
     {"post_exposure": .5, "contrast": 20., "saturation": -15., "hue_shift": 45.},
     {"intensity": .4, "smoothness": .5, "roundness": .4, "rounded": True, "color": (.15, .05, .2, 1.)},
     {"intensity": .4},
     {"pixel_size": 4, "pixel_aspect": 1.4, "grid_offset_x": .2, "grid_offset_y": -.1,
      "intensity": .8, "sampling": 1, "preserve_alpha_coverage": True},
     {"intensity": .8},
     {"mode": 1, "exposure": 1.2},
     {"intensity": .5, "response": .8, "preset": 0, "size": 2.4, "colored": False}),
    ({"temperature": -25., "tint": 15.},
     {"post_exposure": -.5, "contrast": -10., "saturation": 25., "hue_shift": -30.},
     {"intensity": .6, "smoothness": .6, "roundness": .8, "rounded": False, "color": (.02, .08, .12, 1.)},
     {"intensity": .8},
     {"pixel_size": 5, "pixel_aspect": .8, "grid_offset_x": -.3, "grid_offset_y": .25,
      "intensity": 1., "sampling": 2, "preserve_alpha_coverage": False},
     {"intensity": .2},
     {"mode": 2, "exposure": 1.5},
     {"intensity": .8, "response": .3, "preset": 2, "size": 2.4, "colored": True}),
)


class PostprocessPipeline(BloomPipeline):
    def define_topology(self, graph):
        super().define_topology(graph)
        output = graph.create_texture("capture_clock", format=Format.RGBA16_SFLOAT)
        with graph.add_pass("CaptureClock") as p:
            p.set_texture("source_tex", graph.get_texture("color"))
            p.write_color(output)
            p.fullscreen_quad(str(self.clock))
        with graph.add_pass("CommitClock") as p:
            p.set_texture("source_tex", output)
            p.write_color(graph.get_texture("color"))
            p.fullscreen_quad("infernux/internal/fullscreen_copy")


def grain_reference(image, parameters, frame):
    size = parameters["size"] if parameters["preset"] == 3 else (1.4, 2.4, 3.8)[parameters["preset"]]
    coordinates = np.stack((X * 32, Y * 24), axis=-1) / size

    def plate(position, seed):
        cell = np.floor(position).astype(np.int64).astype(np.uint32)
        fraction = (position % 1) ** 2 * (3 - 2 * (position % 1))

        def density(x, y):
            bits = x * np.uint32(1973) ^ y * np.uint32(9277) ^ np.uint32((seed * 26699) & 0xFFFFFFFF)
            bits ^= bits >> 16
            bits *= np.uint32(0x7FEB352D)
            bits ^= bits >> 15
            bits *= np.uint32(0x846CA68B)
            bits ^= bits >> 16
            return ((bits & 65535).astype(np.float64) + (bits >> 16)) / 65535 - 1

        x, y = cell[..., 0], cell[..., 1]
        a = density(x, y) * (1 - fraction[..., 0]) + density(x + np.uint32(1), y) * fraction[..., 0]
        b = density(x, y + np.uint32(1)) * (1 - fraction[..., 0]) + density(x + np.uint32(1), y + np.uint32(1)) * fraction[..., 0]
        return a * (1 - fraction[..., 1]) + b * fraction[..., 1]

    if parameters["intensity"]:
        seed = frame * 5
        rotated = coordinates @ np.array(((.8, .6), (-.6, .8)))
        neutral = .74 * plate(rotated, seed) + .26 * plate(coordinates * 1.8 + (71, 13), seed + 1)
        grain = np.repeat(neutral[..., None], 3, axis=-1)
        if parameters["colored"]:
            grain = .7 * grain + .3 * np.stack([plate(coordinates, seed + channel) for channel in (2, 3, 4)], axis=-1)
        luma = np.clip(image[..., :3] @ (.2126, .7152, .0722), 0, 1)
        amplitude = parameters["intensity"] * (1 - np.sqrt(luma) * parameters["response"])
        image[..., :3] = np.maximum(image[..., :3] * (1 + grain * amplitude[..., None]), 0)
    return half(image)


def sample(image, uv):
    pixel = np.clip(uv * (32, 24) - .5, (0, 0), (31, 23))
    low = np.floor(pixel).astype(int)
    high = np.minimum(low + 1, (31, 23))
    tx, ty = (pixel - low)[..., 0, None], (pixel - low)[..., 1, None]
    return ((image[low[..., 1], low[..., 0]] * (1 - tx) + image[low[..., 1], high[..., 0]] * tx) * (1 - ty)
            + (image[high[..., 1], low[..., 0]] * (1 - tx) + image[high[..., 1], high[..., 0]] * tx) * ty)


def reference(plan, frame):
    wb, ca, vig, sharp, pix, chroma, tone, grain = plan
    image = half(np.stack((.1 + .7 * X, .05 + .5 * Y, .2 + .2 * X * Y, .1 + .8 * X), axis=-1))
    # Adapt XYZ independently of the production shader's fused LMS matrices.
    primaries = np.array(((.4124564, .3575761, .1804375), (.2126729, .7151522, .0721750), (.0193339, .1191920, .9503041)))
    cat = np.array(((.7328, .4296, -.1624), (-.7036, 1.6975, .0061), (.003, .0136, .9834)))
    t, x0, y0 = wb["temperature"] / 65., .31271, .32902
    x = x0 - t * (.1 if t < 0 else .05)
    y = y0 + (x - x0) * (2.87 - 3 * (x + x0)) + wb["tint"] / 65. * .05
    ratio = (cat @ (x0 / y0, 1, (1 - x0 - y0) / y0)) / (cat @ (x / y, 1, (1 - x - y) / y))
    transform = np.linalg.inv(primaries) @ np.linalg.inv(cat) @ np.diag(ratio) @ cat @ primaries
    image[..., :3] = np.maximum(image[..., :3] @ transform.T, 0)
    image = half(image)
    color = image[..., :3] * 2 ** ca["post_exposure"]
    logc = .244161 * np.log10(np.maximum(5.555556 * color + .047996, 1e-6)) + .386036
    logc = (logc - .4135884) * (1 + .01 * ca["contrast"]) + .4135884
    color = np.maximum((10 ** ((logc - .386036) / .244161) - .047996) / 5.555556, 0)
    luma = (color @ (.2126, .7152, .0722))[..., None]
    color = np.maximum(luma + (color - luma) * (1 + .01 * ca["saturation"]), 0)
    if abs(ca["hue_shift"]) > .5:
        hsv = np.array([colorsys.rgb_to_hsv(*rgb) for rgb in color.reshape(-1, 3)])
        hsv[:, 0] = (hsv[:, 0] + ca["hue_shift"] / 360) % 1
        color = np.array([colorsys.hsv_to_rgb(*value) for value in hsv]).reshape(color.shape)
    image[..., :3] = color
    image = half(image)
    distance = abs(UV - .5) * (vig["intensity"] * 3)
    if vig["rounded"]:
        distance[..., 0] *= 32 / 24
    distance = np.clip(distance, 0, 1) ** ((1 - vig["roundness"]) * 6 + vig["roundness"])
    factor = np.clip(1 - (distance ** 2).sum(axis=-1, keepdims=True), 0, 1) ** (vig["smoothness"] * 5)
    tint = np.array(vig["color"][:3])
    tint = np.where(tint <= .04045, tint / 12.92, ((tint + .055) / 1.055) ** 2.4)
    image[..., :3] *= tint + (1 - tint) * factor
    image = half(image)
    neighbors = [sample(image, UV + delta)[..., :3] for delta in ((0, -1 / 24), (0, 1 / 24), (-1 / 32, 0), (1 / 32, 0))]
    minimum, maximum = np.min([image[..., :3], *neighbors], axis=0), np.max([image[..., :3], *neighbors], axis=0)
    weight = np.clip(np.minimum(minimum, 1 - maximum) / (maximum - minimum + .001), 0, 1) ** 2
    image[..., :3] = np.clip(image[..., :3] + (image[..., :3] - sum(neighbors) / 4) * weight * sharp["intensity"] * 2, 0, 1)
    image = half(image)
    cell = np.array((max(1, np.round(pix["pixel_size"] * pix["pixel_aspect"])), pix["pixel_size"]))
    offset = np.array((pix["grid_offset_x"], pix["grid_offset_y"])) * cell
    pixel = UV * (32, 24)
    origin = np.floor((pixel - offset) / cell) * cell + offset
    center = pixel + (origin + cell * .5 - pixel) * pix["intensity"]
    if pix["sampling"] == 0:
        image = sample(image, center / (32, 24))
    else:
        width = 2 if pix["sampling"] == 1 else 3
        taps = np.stack([sample(image, (center + ((np.array((x, y)) + .5) / width - .5) * cell * pix["intensity"]) / (32, 24))
                         for y in range(width) for x in range(width)])
        alpha = taps[..., 3:].sum(axis=0)
        color = (taps[..., :3] * taps[..., 3:]).sum(axis=0) / alpha
        alpha = taps[..., 3:].max(axis=0) if pix["preserve_alpha_coverage"] else alpha / (width * width)
        image = np.concatenate((color, alpha), axis=-1)
    image = half(image)
    center = UV - .5
    offset = center * np.linalg.norm(center, axis=-1, keepdims=True) * chroma["intensity"] * .02
    red = sample(image, UV - offset)[..., 0]
    blue = sample(image, UV + offset)[..., 2]
    image[..., 0], image[..., 2] = red, blue
    image = half(image)
    color = image[..., :3] * tone["exposure"]
    if tone["mode"] == 1:
        color = color / (color + 1)
    elif tone["mode"] == 2:
        # Explicit rows catch transposed matrix construction during lowering.
        input_matrix = np.array(((.59719, .35458, .04823), (.076, .90834, .01566), (.0284, .13383, .83777)))
        output_matrix = np.array(((1.60475, -.53108, -.07367), (-.10208, 1.10813, -.00605), (-.00327, -.07276, 1.07602)))
        value = color @ input_matrix.T
        value = (value * (value + .0245786) - .000090537) / np.maximum(value * (.983729 * value + .432951) + .238081, 1e-6)
        color = value @ output_matrix.T
    image[..., :3] = np.clip(color, 0, 1)
    image = half(image)
    image = grain_reference(image, grain, frame)
    image[0, 0, 3] = half(np.array((frame & 255) / 255))
    image[0, 1, 3] = half(np.array((frame >> 8) / 255))
    image[..., :3] = np.where(image[..., :3] <= .0031308, image[..., :3] * 12.92,
                              1.055 * image[..., :3] ** (1 / 2.4) - .055)
    return image


def run(project):
    for directory in ("Assets", "Packages", "ProjectSettings"):
        (project / directory).mkdir()
    source = project / "Assets/source.frag.py"
    source.write_text('''import infernux.shader as shader
class Source(shader.fullscreen):
    def main(self, uv: shader.vec2) -> shader.vec4:
        return shader.vec4(0.1 + 0.7 * uv.x, 0.05 + 0.5 * uv.y, 0.2 + 0.2 * uv.x * uv.y, 0.1 + 0.8 * uv.x)
''', encoding="utf-8")
    clock = project / "Assets/clock.frag.py"
    clock.write_text('''import infernux.shader as shader
class Clock(shader.fullscreen):
    source_tex = shader.input("color", shader.Color2D, domain="display_linear_ldr")
    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        color = shader.load(self.source_tex, shader.ivec2(ctx.position.xy), 0)
        cadence = shader.u32(shader.floor(shader.frame_time() * 24.0)) & shader.u32(65535)
        if ctx.position.y < 1.0:
            if ctx.position.x < 1.0:
                color.a = float(cadence & shader.u32(255)) / 255.0
            elif ctx.position.x < 2.0:
                color.a = float(cadence >> shader.u32(8)) / 255.0
        return color
''', encoding="utf-8")
    frontend, host, console = inx.Engine(), Host(), ConsolePanel()
    native = frontend.get_native_engine()
    failures, completed = [], False
    try:
        frontend.init_renderer(32, 24, str(project))
        frontend.resize_game_render_target(32, 24)
        native.set_scene_view_visible(False)
        native.set_editor_fps_cap(240)
        native.set_editor_idle_fps(0)
        database = frontend.get_asset_database()
        assert AssetManager.import_asset(str(source), database=database)
        assert AssetManager.import_asset(str(clock), database=database)
        slots, effects = [], []
        for name, parameters in zip(NAMES, PLANS[0]):
            path = project / f"Assets/{name}.effect"
            prefix = "infernux.route." if name == "pixelation" else "infernux.post."
            path.write_text(json.dumps({"$schema": "infernux.render_effect", "feature_type": prefix + name,
                                        "parameters": parameters, "dependencies": []}), encoding="utf-8")
            imported = AssetManager.import_asset(str(path), database=database)
            assert imported, imported.error
            effects.append(AssetManager.load_by_guid(imported.guid))
            slots.append(EffectSlot(stage_id="post", slot_id=name, effect=RenderEffectRef(guid=imported.guid)))
        scene = SceneManager.instance().get_active_scene()
        for obj in scene.get_root_objects():
            scene.destroy_game_object(obj)
        scene.create_game_object("滤镜相机").add_component(inx.Camera)
        pipeline = PostprocessPipeline()
        pipeline.source, pipeline.clock = source, clock
        host.stack._pipeline = pipeline
        host.stack.effect_slots = slots
        frontend.set_render_pipeline(host)
        native.set_game_camera_enabled(True)
        frame = changed = phase = 0
        ticket = builds = None
        sampled_frames = []

        def after_draw():
            nonlocal frame, changed, phase, ticket, builds, completed
            try:
                frame += 1
                if ticket is None and frame >= changed + 8:
                    ticket = frontend.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy().copy().astype(np.float64)
                    cadence = round(pixels[0, 0, 3] * 255) + (round(pixels[0, 1, 3] * 255) << 8)
                    sampled_frames.append(cadence)
                    expected = reference(PLANS[phase], cadence)
                    np.testing.assert_allclose(pixels, expected, atol=.004)
                    assert not host.stack.effect_compile_errors, host.stack.effect_compile_errors
                    if builds is None:
                        builds = pipeline.builds
                    assert pipeline.builds == builds, (builds, pipeline.builds)
                    assert not [row for row in console._get_visible_log_snapshot(2000) if row["level"] in ("ERROR", "FATAL")]
                    print("BUILTIN_POSTPROCESS", phase, "cadence", cadence,
                          "max_error", float(abs(pixels - expected).max()), flush=True)
                    phase += 1
                    if phase == len(PLANS):
                        assert len(set(sampled_frames)) > 1, sampled_frames
                        completed = True
                        native.exit()
                        return
                    for effect, parameters in zip(effects, PLANS[phase]):
                        for name, value in parameters.items():
                            effect.set_param(name, value)
                    changed, ticket = frame, None
                if frame > 100:
                    raise AssertionError("Built-in effect readback timed out")
            except BaseException as error:
                failures.append(error)
                native.exit()

        native.set_post_draw_callback(after_draw)
        native.run()
        if failures:
            raise failures[0]
        assert completed
        print("BUILTIN_POSTPROCESS_OK", flush=True)
    finally:
        frontend.set_render_pipeline(None)
        host.stack.on_destroy()
        native.cleanup()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="infernux-builtin-postprocess-") as root:
        run(Path(root))
