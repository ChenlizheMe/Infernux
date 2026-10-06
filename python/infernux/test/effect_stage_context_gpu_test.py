"""Managed effect parameter updates use each actual stage's resource context."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.assets import AssetManager
from infernux.core.asset_ref import RenderEffectRef
from infernux.lib import ConsolePanel, RenderPipelineCallback, SceneManager


@inx.renderstack.render_effect_feature("audit.stage_context.gpu")
class StageEffect(inx.renderstack.FullScreenEffect):
    name = "Stage Context"
    injection_point = "custom"
    requires = {"color", "light_list"}
    modifies = {"color"}
    strength: float = inx.serialized_field(default=1.0)

    def setup_passes(self, graph, bus):
        color = bus.require_texture("color")
        lights = bus.require_buffer("light_list")
        assert graph.current_pass_result.sample("color") is color
        target = graph.create_texture("stage_out", format=inx.rendergraph.Format.RGBA16_SFLOAT, samples=1)
        with graph.add_pass("Stage Context") as render_pass:
            render_pass.set_texture("_SourceTex", color).write_color(target).fullscreen_quad("Stage Context GPU")
            for name, value in {"strength": self.strength, "bytes": lights.byte_size / 1024.0,
                                "depth": .02 if bus.has("depth") else .01, "samples": color.samples * .01}.items():
                render_pass.set_param(name, value)
        bus.set("color", target)


class StagePipeline(inx.renderstack.RenderPipeline):
    name = "Stage Context GPU"
    builds = 0

    def define_topology(self, graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        color = graph.create_texture("color", camera_target=True)
        graph.add_pass("Baseline").write_color(color).fullscreen_quad("Stage Baseline GPU")
        lights_a = graph.create_buffer("lights_a", byte_size=64)
        lights_b = graph.create_buffer("lights_b", byte_size=128)
        depth = graph.create_texture("depth", format=inx.rendergraph.Format.D32_SFLOAT)
        first = graph.publish_pass_result("first", {"color": color, "light_list": lights_a})
        with graph.pass_result(first):
            graph.effects("first", scope="composite", inputs={"color", "light_list"}, outputs={"color"})
            second = graph.derive_pass_result("second", graph.current_pass_result,
                                              {"depth": depth, "light_list": lights_b})
            graph.replace_current_pass_result(second)
            graph.effects("second", scope="composite", inputs={"color", "depth", "light_list"}, outputs={"color"})
        graph.set_output(color)


class Host(RenderPipelineCallback):
    def __init__(self):
        super().__init__()
        self.stack = inx.renderstack.RenderStack()

    def render(self, context, camera):
        self.stack.render(context, camera)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    proof = {"passed": False, "package": inx.__file__, "phases": [],
             "scope": "Independent Vulkan, GUID-imported managed .effect mounted twice, typed buffer and depth absence/presence, numeric GPU parameter edits without rebuilding. No Editor pixels."}
    with tempfile.TemporaryDirectory(prefix="infernux-stage-context-") as folder:
        project = Path(folder)
        for name in ("Assets", "Packages", "ProjectSettings"):
            (project / name).mkdir()
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        host = Host()
        failures = []
        completed = False
        try:
            frontend.init_renderer(96, 64, folder)
            frontend.resize_game_render_target(96, 64)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240.)
            native.set_editor_idle_fps(0.)
            database = frontend.get_asset_database()
            sources = {
                "Baseline": '#version 450\nShaderInfo { Name "Stage Baseline GPU" Hidden On Capabilities [Fullscreen] Inputs { Float2 inUV } Outputs { Float4 outColor } }\nvoid main() { outColor=vec4(.03,.04,.05,1); }\n',
                "Effect": '#version 450\nShaderInfo { Name "Stage Context GPU" Hidden On Capabilities [Fullscreen] Resources { Texture2D _SourceTex } Inputs { Float2 inUV } Outputs { Float4 outColor } PushConstants pc { Float strength Float bytes Float depth Float samples } }\nvoid main() { outColor=vec4(texture(_SourceTex,inUV).rgb*pc.strength+vec3(pc.bytes,pc.depth,pc.samples),1); }\n',
            }
            shader_guid = None
            for name, text in sources.items():
                path = project / "Assets" / (name + ".frag")
                path.write_text(text, encoding="utf-8")
                result = AssetManager.import_asset(str(path), database=database)
                assert result, result.error
                if name == "Effect":
                    shader_guid = result.guid
            path = project / "Assets/Stage.effect"
            path.write_text(json.dumps({"$schema": "infernux.render_effect", "feature_type": "audit.stage_context.gpu",
                                        "parameters": {"strength": 1.0}, "dependencies": [{"guid": shader_guid}]}), encoding="utf-8")
            imported = AssetManager.import_asset(str(path), database=database)
            assert imported, imported.error
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            scene.create_game_object("Stage Camera").add_component(inx.Camera)
            pipeline = StagePipeline()
            host.stack._pipeline = pipeline
            for stage in ("first", "second"):
                host.stack.add_effect_slot(stage, RenderEffectRef(guid=imported.guid))
            frontend.set_render_pipeline(host)
            native.set_game_camera_enabled(True)
            console = ConsolePanel()
            phase = frames = changed = 0
            ticket = None

            def after_draw():
                nonlocal phase, frames, changed, ticket, completed
                try:
                    frames += 1
                    if ticket is None and frames >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        strength = (1., 2., 1.)[phase]
                        # Both composite stages commit back to the camera
                        # target, whose samples=0 inherits frame MSAA.
                        expected = np.array([.03, .04, .05]) * strength ** 2 + np.array([.0625, .01, 0.]) * strength + np.array([.125, .02, 0.])
                        expected = np.where(expected <= .0031308, expected * 12.92, 1.055 * expected ** (1 / 2.4) - .055)
                        np.testing.assert_allclose(pixels[..., :3], np.broadcast_to(expected, pixels[..., :3].shape), atol=.005)
                        assert not host.stack.effect_compile_errors, host.stack.effect_compile_errors
                        bindings = host.stack._graph_state.bindings
                        assert len(bindings) == 2 and bindings[0].source is bindings[1].source
                        for binding, byte_value, depth_value in zip(bindings, (.0625, .125), (.01, .02)):
                            rebuild, updates = binding.collect_updates()
                            assert not rebuild and len(updates) == 1
                            values = dict(updates[0].values)
                            assert set(values) == {"strength", "bytes", "depth", "samples"}
                            np.testing.assert_allclose([values[name] for name in ("strength", "bytes", "depth", "samples")],
                                                       [strength, byte_value, depth_value, 0.], atol=1e-7)
                        revision = host.stack._graph_state.description.source_revision
                        if phase:
                            assert pipeline.builds == proof["phases"][0]["builds"]
                            assert revision == proof["phases"][0]["revision"]
                        errors = [e for e in console._get_visible_log_snapshot(2000) if e["level"] in ("ERROR", "FATAL", "WARN", "WARNING")]
                        assert not errors, errors[:5]
                        proof["phases"].append({"strength": strength, "rgb": pixels[32, 48, :3].tolist(), "builds": pipeline.builds, "revision": revision})
                        print("PASS managed stage context strength", strength, flush=True)
                        if phase == 2:
                            completed = True
                            native.exit()
                            return
                        phase += 1
                        bindings[0].source.set_float("strength", (1., 2., 1.)[phase])
                        changed = frames
                        ticket = None
                    if frames > 100:
                        raise AssertionError("Stage context GPU proof timed out")
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert completed
        finally:
            frontend.set_render_pipeline(None)
            host.stack.on_destroy()
            native.cleanup()
    proof["passed"] = True
    if args.proof:
        args.proof.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print("PASS managed effect stage contexts", flush=True)


if __name__ == "__main__":
    main()
