"""Real surface shading consumes per-view lights and per-object controls."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import CameraClearFlags, ConsolePanel, LightShadows, LightType, SceneManager, Vector3, vec4f


SOURCE = '''import infernux.shader as shader
class Painted(shader.surface):
    tint = shader.material.color(default=(0.4, 0.6, 0.8, 1.0), label="颜色")
    metallic = shader.material.float(default=0.2, range=(0.0, 1.0))
    smoothness = shader.material.float(default=0.4, range=(0.0, 1.0))
    coverage = shader.material.float(default=1.0)
    mask = shader.material.texture_2d(default="white")
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.tint.rgb, normal_ws=ctx.normal_ws,
                                  metallic=self.metallic, smoothness=self.smoothness,
                                  alpha=self.tint.a * ctx.uv0.x * self.coverage * shader.sample(self.mask, ctx.uv0).a)
'''

REFERENCE = '''#version 450
ShaderInfo {
    Name "Surface Baseline"
    ShadingModel PBR
    Properties { Color tint=[0.4,0.6,0.8,1.0] Float metallic=0.2 Float smoothness=0.4 Float coverage=1.0 Texture2D mask="white" }
}
void surface(out SurfaceData s) {
    s=InitSurfaceData();
    s.albedo=material.tint.rgb;
    s.normalWS=normalize(v_Normal);
    s.metallic=material.metallic;
    s.smoothness=material.smoothness;
    s.alpha=material.tint.a * v_TexCoord.x * material.coverage * texture(mask, v_TexCoord).a;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("forward", "forward_plus", "deferred"), required=True)
    parser.add_argument("--skinned", action="store_true")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="infernux-slang-surface-") as directory:
        root = Path(directory)
        for name in ("Assets", "Packages", "ProjectSettings"):
            (root / name).mkdir()
        source, reference = root / "Assets/表面.frag.py", root / "Assets/baseline.frag"
        source.write_text(SOURCE, encoding="utf-8")
        reference.write_text(REFERENCE, encoding="utf-8")
        mask_path = root / "Assets/mask.png"
        texels = np.full((8, 8, 4), 255, dtype=np.uint8)
        texels[4:, :4, 3] = 0
        Image.fromarray(texels).save(mask_path)
        frontend, console = inx.Engine(), ConsolePanel()
        native = frontend.get_native_engine()
        failures, complete = [], False
        pipeline = {"forward": inx.renderstack.DefaultForwardPipeline,
                    "forward_plus": inx.renderstack.DefaultForwardPlusPipeline,
                    "deferred": inx.renderstack.DefaultDeferredPipeline}[options.pipeline]()
        pipeline.msaa_samples = 1
        pipeline.shadow_resolution = 512
        try:
            frontend.init_renderer(96, 72, directory)
            frontend.resize_game_render_target(96, 72)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            database = frontend.get_asset_database()
            for path in (source, reference, mask_path):
                imported = AssetManager.import_asset(str(path), database=database)
                assert imported, imported.error
            mask_guid = imported.guid
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera_owner = scene.create_game_object("表面验收相机")
            camera_owner.transform.position = Vector3(0, 0, -5)
            camera = camera_owner.add_component(inx.Camera)
            camera.clear_flags = CameraClearFlags.SolidColor
            camera.background_color = vec4f(0, 0, 0, 1)
            obj = scene.create_game_object("表面")
            renderer = obj.add_component(inx.MeshRenderer)
            renderer.casts_shadows = False
            renderer.set_inline_mesh_data(
                np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
                np.array([[0, 0, -1]] * 4, dtype=np.float32),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32), "Quad")
            candidate = inx.Material.from_shader(str(source), "Slang 表面")
            baseline = inx.Material.create_lit("基准表面")
            baseline.frag_shader_name = "Surface Baseline"
            materials = (baseline, candidate)
            for material in materials:
                material.set_color("tint", .4, .6, .8, 1.)
                material.set_float("metallic", .2)
                material.set_float("smoothness", .4)
                material.set_texture("mask", mask_guid)
                material.cull_mode = 0 if options.skinned else 2
            renderer.set_material(0, baseline)
            lights = []
            for name, kind, position, color, intensity in (
                    ("太阳", LightType.Directional, (0, 0, 0), (1., .9, .8), .7),
                    ("点光", LightType.Point, (-2, 1, -3), (.2, .5, 1.), 0.),
                    ("聚光", LightType.Spot, (1, 1, -3), (1., .3, .1), 0.)):
                owner = scene.create_game_object(name)
                owner.transform.position = Vector3(*position)
                light = owner.add_component(inx.Light)
                light.light_type = kind
                light.color = color
                light.intensity = intensity
                light.range = 12
                lights.append(light)
            occluder = scene.create_game_object("遮挡物")
            occluder.transform.position = Vector3(-.8, .5, -1.5)
            occluder.transform.local_scale = Vector3(.5, .5, .5)
            occluder_renderer = occluder.add_component(inx.SkinnedMeshRenderer if options.skinned else inx.MeshRenderer)
            if options.skinned:
                from skinned_mesh_gpu_fixture import write_skinned_quad
                model_path = root / "Assets/caster.gltf"
                write_skinned_quad(model_path)
                imported = AssetManager.import_asset(str(model_path), database=database)
                assert imported, imported.error
                occluder_renderer.set_source_model_guid(imported.guid)
                occluder_renderer.submit_animation_pose("Scale", 0.0, 0.0, loop=False)
            else:
                # Keep caster coverage independent of primitive UV generation.
                occluder_renderer.set_inline_mesh_data(
                    np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32),
                    np.array([[0, 0, -1]] * 4, dtype=np.float32),
                    np.array([[.875, .125]] * 4, dtype=np.float32),
                    np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32), "Caster")
            occluder_renderer.set_material(baseline)
            occluder.active = False
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            frame = changed = phase = side = 0
            ticket = expected = override_reference = None

            def after_draw():
                nonlocal frame, changed, phase, side, ticket, expected, complete, override_reference
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        if phase == side == 0 and np.max(pixels[..., :3]) < .05 and frame < 80:
                            changed, ticket = frame, None
                            return
                        errors = [row for row in console._get_visible_log_snapshot(2000) if row["level"] in ("ERROR", "FATAL")]
                        assert not errors, errors
                        if side == 0:
                            expected = pixels
                            side = 1
                            renderer.set_material(0, candidate)
                            occluder_renderer.set_material(candidate)
                            if phase == 8:
                                occluder_renderer.casts_shadows = False
                            elif phase == 9:
                                occluder_renderer.set_parameter("coverage", 0.0)
                        else:
                            if phase in (6, 8):
                                assert abs(pixels - expected).max() > .02, "A changed surface or caster control did not affect the draw"
                            else:
                                np.testing.assert_allclose(pixels, expected, atol=.006)
                            print("SURFACE", options.pipeline, phase, "max_error", float(abs(pixels - expected).max()), flush=True)
                            side = 0
                            phase += 1
                            if phase == 10:
                                complete = True
                                native.exit()
                                return
                            if phase == 1:
                                for material in materials:
                                    material.set_float("metallic", .85)
                                    material.set_float("smoothness", .8)
                            elif phase == 2:
                                lights[1].intensity = 15
                                lights[2].intensity = 8
                            elif phase == 3:
                                lights[0].culling_mask = 0
                                lights[1].culling_mask = 0
                                obj.layer = 1
                            elif phase == 4:
                                obj.layer = 0
                                lights[1].culling_mask = 0xffffffff
                                lights[1].shadows = LightShadows.Hard
                                occluder.active = True
                            elif phase == 5:
                                renderer.receives_shadows = False
                                for material in materials:
                                    material.alpha_clip_enabled = True
                                    material.alpha_clip_threshold = .5
                            elif phase in (6, 7):
                                source.write_text(SOURCE.replace("albedo=self.tint.rgb", "albedo=self.tint.rgb * 0.5")
                                                  if phase == 6 else SOURCE, encoding="utf-8")
                                assert inx.Shader.reload(str(source))
                            elif phase == 8:
                                renderer.receives_shadows = True
                                for material in materials:
                                    material.alpha_clip_enabled = False
                                occluder_renderer.casts_shadows = True
                            elif phase == 9:
                                for material in materials:
                                    material.alpha_clip_enabled = True
                                override_reference = baseline.clone()
                                override_reference.set_float("coverage", 0.0)
                                occluder_renderer.casts_shadows = True
                            renderer.set_material(0, baseline)
                            occluder_renderer.set_material(override_reference if phase == 9 else baseline)
                            if options.skinned and phase in (1, 4, 7):
                                seconds = {1: .5, 4: 1., 7: 2.}[phase]
                                occluder_renderer.submit_animation_pose("Scale", seconds, seconds / 3, loop=False)
                        changed, ticket = frame, None
                    if frame > 400:
                        raise AssertionError("Surface audit did not complete")
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if failures:
                raise failures[0]
            assert complete
            print("SLANG_SURFACE_OK", options.pipeline, flush=True)
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()


if __name__ == "__main__":
    main()
