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
from infernux.renderstack import EffectSlot
from builtin_bloom_gpu_test import BloomPipeline, Host


def half(value):
    return value.astype(np.float16).astype(np.float64)


Y, X = np.meshgrid((np.arange(24) + .5) / 24, (np.arange(32) + .5) / 32, indexing="ij")
UV = np.stack((X, Y), axis=-1)
NAMES = ("white_balance", "color_adjustments", "vignette", "sharpen", "pixelation",
         "chromatic_aberration", "tonemapping")
PLANS = (
    ({"temperature": 0., "tint": 0.},
     {"post_exposure": 0., "contrast": 0., "saturation": 0., "hue_shift": 0.},
     {"intensity": 0., "smoothness": .3, "roundness": 1., "rounded": False, "color": (0., 0., 0., 1.)},
     {"intensity": 0.},
     {"pixel_size": 4, "pixel_aspect": 1., "grid_offset_x": 0., "grid_offset_y": 0.,
      "intensity": 0., "sampling": 0, "preserve_alpha_coverage": False},
     {"intensity": 0.},
     {"mode": 0, "exposure": 1.}),
    ({"temperature": 25., "tint": -10.},
     {"post_exposure": .5, "contrast": 20., "saturation": -15., "hue_shift": 45.},
     {"intensity": .4, "smoothness": .5, "roundness": .4, "rounded": True, "color": (.15, .05, .2, 1.)},
     {"intensity": .4},
     {"pixel_size": 4, "pixel_aspect": 1.4, "grid_offset_x": .2, "grid_offset_y": -.1,
      "intensity": .8, "sampling": 1, "preserve_alpha_coverage": True},
     {"intensity": .8},
     {"mode": 1, "exposure": 1.2}),
    ({"temperature": -25., "tint": 15.},
     {"post_exposure": -.5, "contrast": -10., "saturation": 25., "hue_shift": -30.},
     {"intensity": .6, "smoothness": .6, "roundness": .8, "rounded": False, "color": (.02, .08, .12, 1.)},
     {"intensity": .8},
     {"pixel_size": 5, "pixel_aspect": .8, "grid_offset_x": -.3, "grid_offset_y": .25,
      "intensity": 1., "sampling": 2, "preserve_alpha_coverage": False},
     {"intensity": .2},
     {"mode": 2, "exposure": 1.5}),
)


def sample(image, uv):
    pixel = np.clip(uv * (32, 24) - .5, (0, 0), (31, 23))
    low = np.floor(pixel).astype(int)
    high = np.minimum(low + 1, (31, 23))
    tx, ty = (pixel - low)[..., 0, None], (pixel - low)[..., 1, None]
    return ((image[low[..., 1], low[..., 0]] * (1 - tx) + image[low[..., 1], high[..., 0]] * tx) * (1 - ty)
            + (image[high[..., 1], low[..., 0]] * (1 - tx) + image[high[..., 1], high[..., 0]] * tx) * ty)


def reference(plan):
    wb, ca, vig, sharp, pix, chroma, tone = plan
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
        pipeline = BloomPipeline()
        pipeline.source = source
        host.stack._pipeline = pipeline
        host.stack.effect_slots = slots
        frontend.set_render_pipeline(host)
        native.set_game_camera_enabled(True)
        frame = changed = phase = 0
        ticket = builds = None

        def after_draw():
            nonlocal frame, changed, phase, ticket, builds, completed
            try:
                frame += 1
                if ticket is None and frame >= changed + 8:
                    ticket = frontend.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy().copy().astype(np.float64)
                    expected = reference(PLANS[phase])
                    np.testing.assert_allclose(pixels, expected, atol=.004)
                    assert not host.stack.effect_compile_errors, host.stack.effect_compile_errors
                    if builds is None:
                        builds = pipeline.builds
                    assert pipeline.builds == builds, (builds, pipeline.builds)
                    assert not [row for row in console._get_visible_log_snapshot(2000) if row["level"] in ("ERROR", "FATAL")]
                    print("BUILTIN_POSTPROCESS", phase, "max_error", float(abs(pixels - expected).max()), flush=True)
                    phase += 1
                    if phase == len(PLANS):
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
