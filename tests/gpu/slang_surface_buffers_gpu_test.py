"""One authored cutout must cover every geometry output at the same pixels."""
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3


SURFACE = '''import infernux.shader as shader
class Cutout(shader.surface):
    clip_edge = shader.material.float(default=0.5)
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=shader.vec3(0.2, 0.4, 0.8), normal_ws=ctx.normal_ws,
                                  metallic=0.2, smoothness=0.4, occlusion=0.8,
                                  alpha=shader.step(self.clip_edge, ctx.uv0.x))
'''

CONSUMER = '''import infernux.shader as shader
class Compare(shader.fullscreen):
    color_tex = shader.input("color", shader.Color2D)
    normal_tex = shader.input("normal", shader.Color2D)
    base_tex = shader.input("base", shader.Color2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    motion_tex = shader.input("motion", shader.Color2D)
    picking_tex = shader.input("picking", shader.UInt2D)
    albedo_tex = shader.input("albedo", shader.Color2D)
    gnormal_tex = shader.input("gnormal", shader.Color2D)
    material_tex = shader.input("material", shader.Color2D)
    metadata_tex = shader.input("metadata", shader.UInt2D)
    object_id = shader.parameter.uint()

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        p = shader.ivec2(ctx.position.xy)
        color = shader.load(self.color_tex, p, 0)
        normal = shader.load(self.normal_tex, p, 0)
        base = shader.load(self.base_tex, p, 0)
        depth = shader.load(self.depth_tex, p, 0)
        motion = shader.load(self.motion_tex, p, 0)
        picking = shader.load(self.picking_tex, p, 0)
        albedo = shader.load(self.albedo_tex, p, 0)
        gnormal = shader.load(self.gnormal_tex, p, 0)
        material = shader.load(self.material_tex, p, 0)
        metadata = shader.load(self.metadata_tex, p, 0)
        covered = color.a > 0.5
        identified = picking.x == self.object_id and picking.y == 0
        mismatch = covered != (normal.a > 0.5) or covered != (base.a > 0.5)
        mismatch = mismatch or covered != (depth > 0.0) or covered != identified or covered != (albedo.a > 0.5)
        if covered:
            if mismatch:
                return shader.vec4(1.0, 0.0625, 1.0, 1.0)
            if shader.any(shader.abs(base - albedo) > 0.002):
                return shader.vec4(1.0, 0.125, 1.0, 1.0)
            if shader.any(shader.abs(normal.rgb - gnormal.rgb) > 0.002):
                return shader.vec4(1.0, 0.1875, 1.0, 1.0)
            if shader.abs(gnormal.a - 0.4) > 0.002:
                return shader.vec4(1.0, 0.25, 1.0, 1.0)
            if shader.any(shader.abs(material.rgb - shader.vec3(0.2, 0.8, 1.0)) > 0.002):
                return shader.vec4(1.0, 0.3125, 1.0, 1.0)
            if metadata.x != 1 or metadata.y == 0 or metadata.z != 1 or metadata.w != 0:
                return shader.vec4(1.0, 0.375, 1.0, 1.0)
        else:
            mismatch = mismatch or shader.any(picking.xy != shader.uvec2(0)) or shader.any(metadata != shader.uvec4(0))
        if mismatch:
            return shader.vec4(1.0, 0.0, 1.0, 1.0)
        if not covered:
            return shader.vec4(0.0, 0.0, 0.0, 1.0)
        return shader.vec4(0.0, 1.0, 0.0, 1.0) if shader.length(motion.xy) > 0.00001 else shader.vec4(1.0, 0.0, 0.0, 1.0)
'''


class BuffersPipeline(inx.renderstack.RenderPipeline):
    name = "Slang Surface Geometry Buffers"

    def define_topology(self, graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        fmt = inx.rendergraph.Format
        depth = graph.create_texture("depth", format=fmt.D32_SFLOAT, samples=1)
        graph.add_pass("Depth").write_depth(depth).set_clear(depth=0.0).draw_renderers(material_pass="depth")
        outputs = {"depth_tex": depth}
        for name, material_pass, format in (("color", "forward", fmt.RGBA16_SFLOAT),
                                             ("normal", "normal", fmt.RGBA16_SFLOAT),
                                             ("base", "base_color", fmt.RGBA16_SFLOAT),
                                             ("motion", "motion", fmt.RG16_SFLOAT)):
            texture = graph.create_texture(name, format=format, samples=1)
            render = graph.add_pass(name).write_color(texture).set_clear(color=(0, 0, 0, 0))
            if name == "color":
                color_depth = graph.create_texture("color_depth", format=fmt.D32_SFLOAT, samples=1)
                render.write_depth(color_depth).set_clear(color=(0, 0, 0, 0), depth=0.0)
            else:
                render.read(depth)
            render.draw_renderers(material_pass=material_pass)
            outputs[name + "_tex"] = texture
        picking = graph.create_texture("picking", format=fmt.RG32_UINT, samples=1)
        picking_depth = graph.create_texture("picking_depth", format=fmt.D32_SFLOAT, samples=1)
        graph.add_pass("Picking").write_color(picking).write_depth(picking_depth).set_clear(color=(0, 0, 0, 0), depth=0.0).draw_renderers(material_pass="picking")
        outputs["picking_tex"] = picking
        gbuffer_depth = graph.create_texture("gbuffer_depth", format=fmt.D32_SFLOAT, samples=1)
        gbuffer = graph.add_pass("GBuffer").write_depth(gbuffer_depth).set_clear(color=(0, 0, 0, 0), depth=0.0)
        for slot, name in enumerate(("albedo", "gnormal", "material", "emission", "metadata")):
            texture = graph.create_texture(name, format=fmt.RGBA32_UINT if name == "metadata" else fmt.RGBA16_SFLOAT, samples=1)
            gbuffer.write_color(texture, slot)
            if name != "emission":
                outputs[name + "_tex"] = texture
        gbuffer.draw_renderers(material_pass="gbuffer")
        target = graph.create_texture("result", camera_target=True)
        compare = graph.add_pass("Compare").write_color(target)
        for name, texture in outputs.items():
            compare.set_texture(name, texture)
        compare.set_param("object_id", self.object_id).fullscreen_quad(self.consumer)
        graph.set_output(target)


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-slang-surface-buffers-") as directory:
        root = Path(directory)
        for name in ("Assets", "Packages", "ProjectSettings"):
            (root / name).mkdir()
        source = root / "Assets/cutout.frag.py"
        consumer = root / "Assets/compare.frag.py"
        source.write_text(SURFACE, encoding="utf8")
        consumer.write_text(CONSUMER, encoding="utf8")
        frontend, console, pipeline = inx.Engine(), ConsolePanel(), BuffersPipeline()
        native = frontend.get_native_engine()
        pipeline.builds, pipeline.consumer = 0, str(consumer)
        failures, complete = [], False
        try:
            frontend.init_renderer(160, 120, directory)
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            database = frontend.get_asset_database()
            for path in (source, consumer):
                imported = AssetManager.import_asset(str(path), database=database)
                assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object("相机")
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component(inx.Camera)
            obj = scene.create_game_object("裁剪平面")
            renderer = obj.add_component(inx.MeshRenderer)
            renderer.set_inline_mesh_data(
                np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
                np.array([[0, 0, -1]] * 4, dtype=np.float32),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32), "Quad")
            material = inx.Material.from_shader(str(source), "裁剪材质")
            material.alpha_clip_enabled, material.alpha_clip_threshold = True, .5
            renderer.set_material(material)
            pipeline.object_id = obj.id
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            frame = changed = phase = 0
            ticket = initial = None

            def after_draw():
                nonlocal frame, changed, phase, ticket, initial, complete
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        red = (pixels[..., 0] > .9) & (pixels[..., 1] < .05) & (pixels[..., 2] < .05)
                        green = (pixels[..., 1] > .9) & (pixels[..., 0] < .05) & (pixels[..., 2] < .05)
                        bad = (pixels[..., 0] > .9) & (pixels[..., 2] > .9)
                        covered = red | green
                        if phase == 0 and not covered.any() and not bad.any() and frame < 80:
                            changed, ticket = frame, None
                            return
                        record = {"phase": phase, "covered": int(covered.sum()), "moving": int(green.sum()), "mismatch": int(bad.sum()), "builds": pipeline.builds}
                        print(record, flush=True)
                        if bad.any():
                            print("MISMATCH_CODES", np.unique(pixels[bad, 1]), flush=True)
                        assert not bad.any() and covered.sum() > 100, record
                        assert pipeline.builds == 1, record
                        assert not [row for row in console._get_visible_log_snapshot(2000) if row["level"] in ("ERROR", "FATAL")]
                        if phase == 0:
                            initial = pixels
                        elif phase == 1:
                            assert covered.sum() > np.count_nonzero(initial[..., 0] > .9) * 1.8
                        elif phase in (2, 4, 6):
                            np.testing.assert_array_equal(pixels, initial)
                        elif phase == 5:
                            assert covered.sum() < np.count_nonzero(initial[..., 0] > .9) * .75
                        if phase == 3:
                            assert green.sum() > covered.sum() * .9, record
                        else:
                            assert not green.any(), record
                        phase += 1
                        if phase == 7:
                            complete = True
                            native.exit()
                            return
                        if phase in (1, 2):
                            material.set_float("clip_edge", 0.0 if phase == 1 else .5)
                        elif phase == 4:
                            obj.transform.position = Vector3(0, 0, 0)
                        elif phase in (5, 6):
                            source.write_text(SURFACE.replace("self.clip_edge,", "self.clip_edge + 0.25,") if phase == 5 else SURFACE, encoding="utf8")
                            assert inx.Shader.reload(str(source))
                        changed, ticket = frame, None
                    if phase == 3:
                        obj.transform.position = Vector3(obj.transform.position.x + .02, 0, 0)
                    if frame > 250:
                        raise AssertionError("Surface geometry outputs did not complete")
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert complete
            print("SLANG_SURFACE_BUFFERS_OK", flush=True)
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()


if __name__ == "__main__":
    main()
