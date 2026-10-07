"""Route Bloom must preserve nearer geometry and keep its additive overflow."""
from __future__ import annotations

import tempfile
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from infernux import Engine
from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.lib import InxMaterial, RenderPipelineCallback, SceneManager, Vector3
from infernux.renderstack import Path as RenderPath, Queue, RenderPipeline, RenderStack
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import RenderEffectAsset
from infernux.components.fields import serialized_field
from infernux.renderstack import FullScreenEffect, RoutePolicy, render_effect_feature


@render_effect_feature("test.post.edge_detection", route_policy=RoutePolicy.MASK_AND_MODIFY)
class EdgeDetectionProbe(FullScreenEffect):
    name = "Probe Edge Detection"
    injection_point = "before_post_process"
    intensity: float = serialized_field(default=0.0)

    def get_shader_list(self):
        return ["Fullscreen Triangle", "Probe Edge Detection"]

    def setup_passes(self, graph, bus):
        from infernux.rendergraph import Format
        self.apply_single_source_effect(
            graph, bus, output_name="_edge_detection", pass_name="EdgeDetection",
            shader_name="Probe Edge Detection", format=Format.RGBA16_SFLOAT,
            params={"intensity": self.intensity},
        )


EDGE_SHADER = '''#version 450
ShaderInfo {
    Name "Probe Edge Detection"
    Resources { Texture2D _SourceTex }
    Capabilities [Fullscreen]
    PushConstants pc { Float intensity }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    vec2 stepUV = 1.0 / vec2(textureSize(_SourceTex, 0));
    vec4 source = texture(_SourceTex, inUV);
    vec3 dx = texture(_SourceTex, inUV + vec2(stepUV.x, 0.0)).rgb
            - texture(_SourceTex, inUV - vec2(stepUV.x, 0.0)).rgb;
    vec3 dy = texture(_SourceTex, inUV + vec2(0.0, stepUV.y)).rgb
            - texture(_SourceTex, inUV - vec2(0.0, stepUV.y)).rgb;
    vec3 edges = vec3(step(0.2, length(dx) + length(dy)));
    outColor = vec4(mix(source.rgb, edges, pc.intensity), source.a);
}
'''


class RouteProbe(RenderPipeline):
    name = "Bloom Occlusion Probe"

    def __init__(self, path="forward", scope="layer"):
        super().__init__()
        self.samples = 1
        self.path = path
        self.scope = scope

    def define(self, pipeline):
        pipeline.frame(hdr=True, msaa=self.samples)
        pipeline.lighting(clustered=True)
        with pipeline.opaque() as opaque:
            with (opaque.layer("Bright Objects") if self.scope == "layer" else nullcontext(opaque)) as selected:
                if self.path == "deferred":
                    selected.deferred(Queue(1000, 1099), fallback=RenderPath.FORWARD_PLUS).effects("bright_route")
                else:
                    getattr(selected, self.path)(Queue(1000, 1099)).effects("bright_route")
            opaque.otherwise().forward()
        pipeline.screen_ui()


class StackProbe(RenderPipelineCallback):
    def __init__(self, path="forward", scope="layer"):
        super().__init__()
        self.stack = RenderStack()
        self.stack._pipeline = RouteProbe(path, scope)

    def render(self, context, camera):
        self.stack.render(context, camera)

    def dispose(self):
        self.stack.on_destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("forward", "forward_plus", "deferred"), default="forward")
    parser.add_argument("--scope", choices=("layer", "route"), default="layer")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="infernux-bloom-occlusion-") as root:
        project = Path(root)
        assets = project / "Assets"
        assets.mkdir()
        (project / "ProjectSettings").mkdir()
        mesh = assets / "Probe.obj"
        mesh.write_text(
            "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\n"
            "vn 0 0 -1\nvn 0 0 1\n"
            "f 1//1 2//1 3//1\nf 1//1 3//1 4//1\n"
            "f 3//2 2//2 1//2\nf 4//2 3//2 1//2\n", encoding="ascii",
        )
        shader = assets / "EdgeDetection.frag"
        shader.write_text(EDGE_SHADER, encoding="ascii")
        frontend = Engine()
        engine = frontend.get_native_engine()
        failures = []
        completed = False
        try:
            frontend.init_renderer(128, 96, str(project))
            frontend.resize_game_render_target(128, 96)
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            imported = AssetManager.import_asset(str(mesh), database=frontend.get_asset_database())
            assert imported, imported.error
            shader_import = AssetManager.import_asset(str(shader), database=frontend.get_asset_database())
            assert shader_import, shader_import.error
            scene = SceneManager.instance().get_active_scene()
            camera = scene.create_game_object("Camera")
            camera.transform.position = Vector3(0, 0, -4)
            camera.add_component("Camera")
            for name, x, z, scale, queue, color in [
                ("Bright Red", -.3, 0, 1.0, 1000, (8., 0., 0., 1.)),
                ("Near Blue", .4, -1, .45, 1200, (0., 0., 1., 1.)),
            ]:
                obj = scene.create_game_object(name)
                obj.transform.position = Vector3(x, 0, z)
                obj.transform.local_scale = Vector3(scale, scale, scale)
                renderer = obj.add_component("MeshRenderer")
                renderer.set_mesh_asset_guid(imported.guid)
                material = InxMaterial.create_default_unlit()
                material.set_color("baseColor", *color)
                material.set_render_queue(queue)
                renderer.set_material(0, material)
            pipeline = StackProbe(args.path, args.scope)
            frontend.set_render_pipeline(pipeline)
            engine.set_game_camera_enabled(True)
            bloom = RenderEffect(RenderEffectAsset(feature_type="infernux.post.bloom", parameters={"intensity": 0.0}))
            edge = RenderEffect(RenderEffectAsset(feature_type="test.post.edge_detection", parameters={"intensity": 0.0}))
            frame, changed_frame, sample_index = 0, 0, 0
            sample_roundtrip = (4, 1, 4, 1)
            stage, ticket, baseline, blue, first_order = "baseline", None, None, None, None

            def after_draw():
                nonlocal frame, changed_frame, stage, ticket, baseline, blue, completed, sample_index, first_order
                try:
                    frame += 1
                    if stage in ("baseline", "zero_bloom", "zero_chain", "active_bloom", "bloom_then_edge", "edge_then_bloom", "msaa_roundtrip") and frame >= changed_frame + 8:
                        ticket = frontend.request_render_target_readback(True)
                        stage += "_readback"
                    elif stage.endswith("_readback") and ticket.done:
                        pixels = ticket.result_numpy().copy()
                        rgb = pixels[..., :3].astype(np.float32)
                        if stage == "baseline_readback":
                            baseline = pixels
                            blue = (rgb[..., 2] > .9) & (rgb[..., 0] < .05)
                            assert blue.sum() > 100, int(blue.sum())
                            pipeline.stack.add_effect_slot("bright_route", RenderEffectRef(effect=bloom))
                            stage = "zero_bloom"
                        elif stage == "zero_bloom_readback":
                            preserved = int((rgb[..., 2][blue] > .9).sum())
                            print("Bloom intensity 0: nearer blue pixels", preserved, "of", int(blue.sum()), flush=True)
                            np.testing.assert_array_equal(pixels, baseline, err_msg="zero-intensity Bloom repainted occluded route geometry")
                            pipeline.stack.add_effect_slot("bright_route", RenderEffectRef(effect=edge))
                            stage = "zero_chain"
                        elif stage == "zero_chain_readback":
                            assert not pipeline.stack.effect_compile_errors, pipeline.stack.effect_compile_errors
                            np.testing.assert_array_equal(pixels, baseline, err_msg="zero-intensity mixed chain changed the scene")
                            bloom.set_float("intensity", .8)
                            stage = "active_bloom"
                        elif stage == "active_bloom_readback":
                            assert np.all(rgb[..., 2][blue] > .9), "Bloom replaced nearer blue geometry"
                            background = np.all(baseline[..., :3] == 0, axis=-1)
                            halo = background & (rgb[..., 0] > .02)
                            assert halo.sum() > 20, ("Bloom overflow disappeared", int(halo.sum()))
                            print("Active Bloom preserves blue geometry; halo pixels", int(halo.sum()), flush=True)
                            edge.set_float("intensity", .8)
                            stage = "bloom_then_edge"
                        elif stage == "bloom_then_edge_readback":
                            assert np.all(rgb[..., 2][blue] > .9), "mixed chain replaced nearer geometry"
                            first_order = pixels
                            assert not np.array_equal(pixels, baseline), "combined effects did not execute"
                            pipeline.stack.set_effect_stage_slots("bright_route", tuple(reversed(pipeline.stack.get_effect_stage_slots("bright_route"))))
                            stage = "edge_then_bloom"
                        elif stage == "edge_then_bloom_readback":
                            assert not pipeline.stack.effect_compile_errors, pipeline.stack.effect_compile_errors
                            assert np.all(rgb[..., 2][blue] > .9), "reverse chain replaced nearer geometry"
                            assert np.any(np.abs(pixels.astype(np.float32) - first_order.astype(np.float32)) > .01), "slot order did not change the executed chain"
                            print("Bloom + Edge Detection execute in both slot orders and preserve foreground geometry", flush=True)
                            pipeline.stack.pipeline.samples = sample_roundtrip[sample_index]
                            pipeline.stack.invalidate_graph()
                            stage = "msaa_roundtrip"
                        else:
                            assert (rgb[..., 2][blue] > .9).sum() > blue.sum() * .8, "MSAA switch lost foreground geometry"
                            print("Bloom MSAA roundtrip", sample_roundtrip[sample_index], flush=True)
                            sample_index += 1
                            if sample_index == len(sample_roundtrip):
                                completed = True
                                engine.exit()
                            else:
                                pipeline.stack.pipeline.samples = sample_roundtrip[sample_index]
                                pipeline.stack.invalidate_graph()
                                stage = "msaa_roundtrip"
                        changed_frame = frame
                    if frame > 160:
                        raise AssertionError("Bloom route readback timed out")
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
