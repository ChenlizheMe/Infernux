"""Sample built-in surface textures through the real particle GPU renderer."""
import json

import numpy as np
import pytest

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, Vector3
from infernux.particle import AssetReference, ParticleScriptCompiler
from infernux.renderstack import RenderStackPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service
from tests.python.test_play_enter_recovery import authoring_project
from tests.python.particle_gpu_fixture import playing_particle_world, particle_callback, run_particle_frames
from tests.gpu.renderer_fixture import GameFrames
from contextlib import ExitStack


def shader_source(name, property_name, default="white"):
    return f'''import infernux.shader as shader

class BuiltinTexture(shader.particle):
    shader_id = {name!r}
    surface_type = "transparent"
    cull_mode = "none"
    depth_write = False
    {property_name} = shader.material.texture_2d(default={default!r})

    def main(self, ctx: shader.ParticleContext) -> shader.vec4:
        return shader.sample(self.{property_name}, ctx.uv0)
'''



def particle_source(shader_name, property_name, initial):
    return f'''
from infernux.particle import AssetReference, EmitterSettings, Parameter, ParticleBurst, ParticleEmitter, ParticleScript
class BuiltinTextures(ParticleScript):
    parameters = (Parameter("surface", "Surface", "texture2d", AssetReference(guid={initial!r})),)
    class Emitter(ParticleEmitter):
        stable_id = "emitter"
        settings = EmitterSettings(capacity=8, spawn_rate=0.0, duration=1000.0, loop=False,
                                   bursts=(ParticleBurst(time=0.0, count=1),))
        def init(self, ctx, particles):
            particles.set_position((0.0, 0.0, 0.0))
            particles.set_size(2.0)
            particles.set_lifetime(1000.0)
        def update(self, ctx, particles):
            pass
        def rendering(self, ctx, particles):
            particles.sprite(shader={shader_name!r}, properties={{{property_name!r}: ctx.parameter("surface")}})
'''


def test_connected_numeric_output_parameter_uses_graph_default():
    source = particle_source("infernux/particle/unlit", "glow", "white")
    source = source.replace('"texture2d", AssetReference(guid=\'white\')', '"f32", 2.5')
    compiled = ParticleScriptCompiler().compile(source)
    output = compiled.emitters[0].render_plan.outputs[0]
    component = ParticleSystem()
    component._ensure_runtime_state()
    initial = component._gpu_material_binding(output, "emitter", compiled.parameters)["native"]
    assert initial.get_float("glow") == 2.5
    component._parameter_overrides["surface"] = 4.0
    overridden = component._gpu_material_binding(output, "emitter", compiled.parameters)["native"]
    assert overridden is initial and overridden.get_float("glow") == 4.0
    component._parameter_overrides.clear()
    restored = component._gpu_material_binding(output, "emitter", compiled.parameters)["native"]
    assert restored is initial and restored.get_float("glow") == 2.5


@pytest.mark.parametrize("source_type, default, override, property_name, expected_default, expected_override", [
    ("f32", .25, .75, "base_color", [.25] * 4, [.75] * 4),
    ("vec2", [.25, .5], [.5, .75], "base_color", [.25, .5, 0, 0], [.5, .75, 0, 0]),
    ("vec3", [.25, .5, .75], [.5, .75, 1], "base_color", [.25, .5, .75, 0], [.5, .75, 1, 0]),
    ("vec4", [.25, .5, .75, 1], [.5, .75, 1, .25], "glow", .25, .5),
    ("color", [.25, .5, .75, 1], [.5, .75, 1, .25], "glow", .25, .5),
    ("i32", 2, 3, "glow", 2., 3.),
    ("u32", 2, 3, "base_color", [2.] * 4, [3.] * 4),
])
def test_connected_numeric_output_matches_compiled_port_shape(
    source_type, default, override, property_name, expected_default, expected_override,
):
    source = particle_source("infernux/particle/unlit", property_name, "white")
    source = source.replace('"texture2d", AssetReference(guid=\'white\')', f'{source_type!r}, {default!r}')
    compiled = ParticleScriptCompiler().compile(source)
    output = compiled.emitters[0].render_plan.outputs[0]
    component = ParticleSystem()
    component._ensure_runtime_state()
    material = None
    for overrides, expected in (({}, expected_default), ({"surface": override}, expected_override), ({}, expected_default)):
        component._parameter_overrides = overrides
        published = component._gpu_material_binding(output, "emitter", compiled.parameters)["native"]
        if material is not None:
            assert published is material
        material = published
        assert material.serialize_document()["properties"][property_name]["value"] == expected


def test_six_way_smoke_material_keeps_independent_axis_defaults():
    source = particle_source("infernux/particle/six_way_smoke", "negative_axes_map", "black")
    compiled = ParticleScriptCompiler().compile(source)
    output = compiled.emitters[0].render_plan.outputs[0]
    component = ParticleSystem()
    component._ensure_runtime_state()
    material = component._gpu_material_binding(output, "emitter", compiled.parameters)["native"]
    assert material.serialize_document()["properties"]["positive_axes_map"]["guid"] == "white"
    assert material.serialize_document()["properties"]["negative_axes_map"]["guid"] == "black"
    component._parameter_overrides["surface"] = AssetReference(guid="normal").to_dict()
    component._gpu_material_binding(output, "emitter", compiled.parameters)
    assert material.serialize_document()["properties"]["positive_axes_map"]["guid"] == "white"
    assert material.serialize_document()["properties"]["negative_axes_map"]["guid"] == "normal"
    component._parameter_overrides.clear()
    component._gpu_material_binding(output, "emitter", compiled.parameters)
    assert material.serialize_document()["properties"]["positive_axes_map"]["guid"] == "white"
    assert material.serialize_document()["properties"]["negative_axes_map"]["guid"] == "black"


@pytest.mark.parametrize("property_name", ["albedo_map", "normal_named_surface"])
@pytest.mark.parametrize("declared_default, initial", [
    ("white", ""), ("black", ""), ("normal", ""),
    ("white", "black"), ("black", "normal"), ("normal", "white"),
])
def test_particle_builtin_texture_values_and_live_changes(
    engine, scene, tmp_path, monkeypatch, property_name, initial, declared_default, authoring_project,
):
    from infernux.engine import project_context

    monkeypatch.setattr(project_context._context(), "root", str(tmp_path))
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    assets = tmp_path / "Assets"
    assets.mkdir()
    shader_name = f"tests/particle/builtin_{property_name}_{initial or 'empty'}_{declared_default}"
    shader = assets / "BuiltinSurface.frag.py"
    shader.write_text(shader_source(shader_name, property_name, declared_default), encoding="utf-8")
    result = AssetManager.import_asset(str(shader), database=database)
    assert result.succeeded, result.error
    graph = assets / "BuiltinTextures.particlegraph"
    ParticleScriptCompiler().parse(particle_source(shader_name, property_name, initial)).save(str(graph))
    result = AssetManager.import_asset(str(graph), database=database)
    assert result.succeeded, result.error
    camera = scene.create_game_object("Camera")
    camera.transform.position = Vector3(0, 0, -5)
    camera.add_component("Camera")
    component = scene.create_game_object("Particle").add_py_component(ParticleSystem())
    console = ConsolePanel()
    pipeline = RenderStackPipeline()
    install_runtime_service("gpu-particles", engine)
    # Every case already visits all built-ins, the shader default and reset.
    # Cover each declaration and explicit initial value without repeating their
    # full Cartesian product through the real renderer.
    phases = [initial, "black", "normal", "white", "", "reset"]
    expected = {"white": (1, 1, 1), "black": (0, 0, 0), "normal": (.5, .5, 1)}
    expected[""] = expected[declared_default]
    failures, colors = [], []
    frame, changed, ticket = 0, 0, None
    runtime = ExitStack()
    component.play_on_awake = False
    component.graph = ParticleGraphRef(guid=result.guid)
    game_frames = GameFrames(engine)
    try:
        engine.resize_game_render_target(64, 64)
        engine.set_game_camera_enabled(True)
        runtime_scene = runtime.enter_context(playing_particle_world(authoring_project, 64, 64, pipeline))
        component = runtime_scene.find("Particle").get_component(ParticleSystem)
        assert component.game_object.scene.world_id == runtime_scene.world_id
        assert component.play(), component.last_compile_error
        batch_id = component._batch_id

        @particle_callback(authoring_project)
        def after_draw():
            nonlocal frame, changed, ticket
            try:
                if not game_frames.advance():
                    return
                frame += 1
                assert frame < 200, colors
                execution = engine.renderer_frame_snapshot["game_render_graph_execution_count"]
                if ticket is None and engine.game_presentation_ready and execution >= changed + 12:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy()
                    center = pixels[pixels.shape[0] // 2, pixels.shape[1] // 2].copy()
                    operation = phases[len(colors)]
                    token = initial if operation == "reset" else operation
                    colors.append(dict(operation=operation, token=token, rgba=center.tolist()))
                    (tmp_path / "builtin-textures.json").write_text(json.dumps(colors), encoding="utf-8")
                    linear = np.array(expected[token])
                    display = np.where(linear <= .0031308, 12.92 * linear,
                                       1.055 * linear ** (1 / 2.4) - .055)
                    assert center[3] > .99, center
                    np.testing.assert_allclose(center[:3], display, atol=.005, err_msg=repr(colors))
                    assert component._batch_id == batch_id
                    assert component.runtime_diagnostics()["resident"]
                    issues = [item for item in console._get_visible_log_snapshot(1000)
                              if item["level"] in ("ERROR", "FATAL", "WARN", "WARNING")]
                    assert not issues, issues
                    if len(colors) == len(phases):
                        np.testing.assert_array_equal(colors[0]["rgba"], colors[-1]["rgba"])
                        return
                    else:
                        next_token = phases[len(colors)]
                        if next_token == "reset":
                            assert component.reset_parameter("surface")
                        else:
                            component.set_parameter("surface", AssetReference(guid=next_token))
                        changed, ticket = execution, None
            except BaseException as error:
                failures.append(error)
                return

        engine.set_post_draw_callback(after_draw)
        run_particle_frames(authoring_project, lambda: failures or len(colors) == len(phases))
        if failures:
            raise failures[0]
        assert len(colors) == len(phases)
    finally:
        engine.set_post_draw_callback(None)
        component._remove_native_batch()
        runtime.close()
        assert remove_runtime_service("gpu-particles", engine)
        for path in (graph, shader):
            assert database.delete_asset(str(path))
