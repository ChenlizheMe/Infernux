"""Particle Motion passes record valid bindings and publish real GPU motion."""
import json

import numpy as np
import pytest

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3
from infernux.particle import ParticleScriptCompiler
from infernux.rendergraph import Format
from infernux.renderstack import RenderPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service


MOTION_DISPLAY = '''#version 450
ShaderInfo {
    Name "Particle Motion Readback"
    Hidden On
    Capabilities [Fullscreen]
    Resources { Texture2D motionTex }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    vec2 motion = texelFetch(motionTex, ivec2(gl_FragCoord.xy), 0).xy;
    outColor = vec4(length(motion) > 0.00001 ? 1.0 : 0.0, 0, 0, 1);
}
'''


class ParticleMotionPipeline(RenderPipeline):
    name = "Particle Motion Binding Probe"

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        source = graph.create_texture("source", format=Format.RGBA16_SFLOAT, samples=1)
        depth = graph.create_texture("depth", format=Format.D32_SFLOAT, samples=1)
        motion = graph.create_texture("motion", format=Format.RG16_SFLOAT, samples=1)
        graph.add_pass("Particles").write_color(source).write_depth(depth).set_clear(
            color=(0, 0, 0, 0), depth=1,
        ).draw_renderers()
        graph.add_pass("ParticleMotion").read(depth).write_color(motion).set_clear(
            color=(0, 0, 0, 0),
        ).draw_renderers(material_pass="motion")
        color = graph.create_texture("color", camera_target=True)
        graph.add_pass("MotionReadback").set_texture("motionTex", motion).write_color(color).fullscreen_quad(
            "Particle Motion Readback",
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
def test_particle_motion_pass_records_valid_bindings_and_camera_motion(engine, scene, tmp_path, monkeypatch, kind):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    shader = tmp_path / "MotionReadback.frag"
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
    manager, console = SceneManager.instance(), ConsolePanel()
    pipeline = ParticleMotionPipeline()
    install_runtime_service("gpu-particles", engine)
    phases = ("static", "camera_motion", "particle_motion", "stopped")
    failures, results = [], []
    frame, changed, ticket = 0, 0, None
    try:
        engine.set_render_pipeline(pipeline)
        engine.resize_game_render_target(96, 96)
        engine.set_game_camera_enabled(True)
        manager.play()
        manager.pause()
        component.graph = ParticleGraphRef(guid=result.guid)
        component.awake()
        assert component.play(), component.last_compile_error

        def update(_delta):
            component.update(1. / 120.)

        def after_draw():
            nonlocal frame, changed, ticket
            try:
                frame += 1
                assert frame < 150, results
                if ticket is None and frame >= changed + 16:
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
                        engine.exit()
                        return
                    component.set_parameter("drift", .5 if phases[len(results)] == "particle_motion" else 0.0)
                    changed, ticket = frame, None
                if phases[len(results)] == "camera_motion":
                    camera.transform.position = Vector3((frame - changed) * .01, 0, -5)
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(results) == len(phases)
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_post_draw_callback(None)
        component._remove_native_batch()
        engine.set_render_pipeline(None)
        manager.stop()
        assert remove_runtime_service("gpu-particles", engine)
        for path in (graph, shader):
            assert database.delete_asset(str(path))
