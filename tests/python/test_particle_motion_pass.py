"""Particle Motion passes record valid bindings and publish real GPU motion."""
import json

import numpy as np
import pytest

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, Vector3
from infernux.particle import ParticleScriptCompiler
from infernux.rendergraph import Format
from infernux.renderstack import RenderPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service
from tests.python.test_play_enter_recovery import authoring_project
from tests.python.particle_gpu_fixture import playing_particle_world, particle_callback, run_particle_frames
from tests.gpu.renderer_fixture import GameFrames
from contextlib import ExitStack


MOTION_DISPLAY = '''import infernux.shader as shader

class ParticleMotionReadback(shader.fullscreen):
    shader_id = "tests/particle/motion_readback"
    hidden = True
    motion_tex = shader.input("motion", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        motion = shader.load(self.motion_tex, shader.ivec2(ctx.position.xy), 0).xy
        return shader.vec4(1.0 if shader.length(motion) > 0.00001 else 0.0, 0, 0, 1)
'''



class ParticleMotionPipeline(RenderPipeline):
    name = "Particle Motion Binding Probe"

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        source = graph.create_texture("source", format=Format.RGBA16_SFLOAT, samples=1)
        depth = graph.create_texture("depth", format=Format.D32_SFLOAT, samples=1)
        motion = graph.create_texture("motion", format=Format.RG16_SFLOAT, samples=1)
        graph.add_pass("Particles").write_color(source).write_depth(depth).set_clear(
            color=(0, 0, 0, 0), depth=0.0,
        ).draw_renderers()
        graph.add_pass("ParticleMotion").read(depth).write_color(motion).set_clear(
            color=(0, 0, 0, 0),
        ).draw_renderers(material_pass="motion")
        color = graph.create_texture("color", camera_target=True)
        graph.add_pass("MotionReadback").set_texture("motion_tex", motion).write_color(color).fullscreen_quad(
            "tests/particle/motion_readback",
        )
        graph.screen_ui_overlay_section(resources={"color"})
        graph.set_output(color)


def motion_source(kind):
    ribbon_init = "            particles.set_strip_id(0)\n            particles.set_ribbon_order(particles.id)" if kind == "ribbon" else ""
    return f'''
from infernux.particle import EmitterSettings, Parameter, ParticleBurst, ParticleEmitter, ParticleScript
class Motion(ParticleScript):
    parameters = (Parameter("drift", "Drift", "f32", 0.0),)
    class Emitter(ParticleEmitter):
        stable_id = "emitter"
        settings = EmitterSettings(capacity=8, spawn_rate=0.0, duration=1000.0, loop=False,
                                   bursts=(ParticleBurst(time=0.0, count=2),))
        def init(self, ctx, particles):
            particles.set_position((-1.0, 0.0, 0.0))
            particles.add_position((2.0, 0.0, 0.0) * particles.id)
            particles.set_size(0.5)
            particles.set_lifetime(1000.0)
{ribbon_init}
        def update(self, ctx, particles):
            particles.add_position((1.0, 0.0, 0.0) * ctx.parameter("drift") * ctx.delta_time)
        def rendering(self, ctx, particles):
            particles.{kind}()
'''


@pytest.mark.parametrize("kind", ["sprite", "ribbon"])
def test_particle_motion_pass_records_valid_bindings_and_camera_motion(engine, scene, tmp_path, monkeypatch, kind, authoring_project):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    shader = tmp_path / "MotionReadback.frag.py"
    shader.write_text(MOTION_DISPLAY, encoding="utf-8")
    result = AssetManager.import_asset(str(shader), database=database)
    assert result.succeeded, result.error
    graph = tmp_path / "Motion.particlegraph"
    ParticleScriptCompiler().parse(motion_source(kind)).save(str(graph))
    result = AssetManager.import_asset(str(graph), database=database)
    assert result.succeeded, result.error
    camera = scene.create_game_object("Motion Camera")
    camera.transform.position = Vector3(0, 0, -5)
    camera.add_component("Camera")
    component = scene.create_game_object("Motion Particles").add_py_component(ParticleSystem())
    console = ConsolePanel()
    pipeline = ParticleMotionPipeline()
    install_runtime_service("gpu-particles", engine)
    phases = ("static", "camera_motion", "particle_motion", "stopped")
    failures, results = [], []
    frame, phase_frame, phase_execution, ticket = 0, 0, 0, None
    runtime = ExitStack()
    component.play_on_awake = False
    component.graph = ParticleGraphRef(guid=result.guid)
    game_frames = GameFrames(engine)
    try:
        engine.resize_game_render_target(96, 96)
        engine.set_game_camera_enabled(True)
        runtime_scene = runtime.enter_context(playing_particle_world(authoring_project, 96, 96, pipeline))
        component = runtime_scene.find("Motion Particles").get_component(ParticleSystem)
        camera = runtime_scene.find("Motion Camera")
        assert component.game_object.scene.world_id == runtime_scene.world_id
        assert component.play(), component.last_compile_error

        @particle_callback(authoring_project)
        def after_draw():
            nonlocal frame, phase_frame, phase_execution, ticket
            try:
                if not game_frames.advance():
                    return
                frame += 1
                assert frame < 150, results
                execution = engine.renderer_frame_snapshot["game_render_graph_execution_count"]
                if ticket is None and engine.game_presentation_ready and execution >= phase_execution + 16:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy()
                    moving_pixels = int(np.count_nonzero(pixels[..., 0] > .9))
                    phase = phases[len(results)]
                    results.append(dict(phase=phase, moving_pixels=moving_pixels))
                    (tmp_path / "particle-motion.json").write_text(json.dumps(results), encoding="utf-8")
                    if phase in ("camera_motion", "particle_motion"):
                        assert moving_pixels > 20, results
                    else:
                        assert moving_pixels == 0, results
                    assert component.runtime_diagnostics()["resident"]
                    issues = [item for item in console._get_visible_log_snapshot(1000)
                              if item["level"] in ("ERROR", "FATAL", "WARN", "WARNING")]
                    assert not issues, issues
                    if len(results) == len(phases):
                        return
                    component.set_parameter("drift", .5 if phases[len(results)] == "particle_motion" else 0.0)
                    phase_frame, phase_execution, ticket = frame, execution, None
                if phases[len(results)] == "camera_motion":
                    camera.transform.position = Vector3((frame - phase_frame) * .01, 0, -5)
            except BaseException as error:
                failures.append(error)
                return

        engine.set_post_draw_callback(after_draw)
        run_particle_frames(authoring_project, lambda: failures or len(results) == len(phases))
        if failures:
            raise failures[0]
        assert len(results) == len(phases)
    finally:
        engine.set_post_draw_callback(None)
        component._remove_native_batch()
        runtime.close()
        assert remove_runtime_service("gpu-particles", engine)
        for path in (graph, shader):
            assert database.delete_asset(str(path))
