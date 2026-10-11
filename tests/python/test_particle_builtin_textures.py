"""Sample built-in surface textures through the real particle GPU renderer."""
import json

import numpy as np
import pytest

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3
from infernux.particle import AssetReference, ParticleScriptCompiler
from infernux.renderstack import RenderStackPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service


def shader_source(name, property_name, default="white"):
    return f'''#version 450
ShaderInfo {{
    Name "{name}"
    ShadingModel Unlit
    Queue 3000
    Cull None
    DepthWrite Off
    Blend Alpha
    Properties {{ Texture2D {property_name} = {default} }}
}}
void surface(out SurfaceData s) {{
    s = InitSurfaceData();
    vec4 texel = sampleAlbedoAlpha({property_name});
    s.albedo = texel.rgb;
    s.alpha = texel.a;
}}
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
    source = particle_source("Particle Unlit", "glow", "white")
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
    ("f32", .25, .75, "baseColor", [.25] * 4, [.75] * 4),
    ("vec2", [.25, .5], [.5, .75], "baseColor", [.25, .5, 0, 0], [.5, .75, 0, 0]),
    ("vec3", [.25, .5, .75], [.5, .75, 1], "baseColor", [.25, .5, .75, 0], [.5, .75, 1, 0]),
    ("vec4", [.25, .5, .75, 1], [.5, .75, 1, .25], "glow", .25, .5),
    ("color", [.25, .5, .75, 1], [.5, .75, 1, .25], "glow", .25, .5),
    ("i32", 2, 3, "glow", 2., 3.),
    ("u32", 2, 3, "baseColor", [2.] * 4, [3.] * 4),
])
def test_connected_numeric_output_matches_compiled_port_shape(
    source_type, default, override, property_name, expected_default, expected_override,
):
    source = particle_source("Particle Unlit", property_name, "white")
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
    source = particle_source("Particle Six-Way Smoke", "negativeAxesMap", "black")
    compiled = ParticleScriptCompiler().compile(source)
    output = compiled.emitters[0].render_plan.outputs[0]
    component = ParticleSystem()
    component._ensure_runtime_state()
    material = component._gpu_material_binding(output, "emitter", compiled.parameters)["native"]
    assert material.serialize_document()["properties"]["positiveAxesMap"]["guid"] == "white"
    assert material.serialize_document()["properties"]["negativeAxesMap"]["guid"] == "black"
    component._parameter_overrides["surface"] = AssetReference(guid="normal").to_dict()
    component._gpu_material_binding(output, "emitter", compiled.parameters)
    assert material.serialize_document()["properties"]["positiveAxesMap"]["guid"] == "white"
    assert material.serialize_document()["properties"]["negativeAxesMap"]["guid"] == "normal"
    component._parameter_overrides.clear()
    component._gpu_material_binding(output, "emitter", compiled.parameters)
    assert material.serialize_document()["properties"]["positiveAxesMap"]["guid"] == "white"
    assert material.serialize_document()["properties"]["negativeAxesMap"]["guid"] == "black"


@pytest.mark.parametrize("property_name", ["texSampler", "normalNamedSurface"])
@pytest.mark.parametrize("initial", ["white", "black", "normal", ""])
@pytest.mark.parametrize("declared_default", ["white", "black", "normal"])
def test_particle_builtin_texture_values_and_live_changes(
    engine, scene, tmp_path, monkeypatch, property_name, initial, declared_default,
):
    from infernux.engine import project_context

    monkeypatch.setattr(project_context, "_project_root", str(tmp_path))
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    assets = tmp_path / "Assets"
    assets.mkdir()
    shader_name = f"Builtin Particle {property_name} {initial or 'empty'} {declared_default}"
    shader = assets / "BuiltinSurface.frag"
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
    manager, console = SceneManager.instance(), ConsolePanel()
    pipeline = RenderStackPipeline()
    install_runtime_service("gpu-particles", engine)
    phases = [initial, "black", "normal", "white", "", "reset"]
    expected = {"white": (1, 1, 1), "black": (0, 0, 0), "normal": (.5, .5, 1)}
    expected[""] = expected[declared_default]
    failures, colors = [], []
    frame, changed, ticket = 0, 0, None
    try:
        engine.set_render_pipeline(pipeline)
        engine.resize_game_render_target(64, 64)
        engine.set_game_camera_enabled(True)
        manager.play()
        manager.pause()
        component.graph = ParticleGraphRef(guid=result.guid)
        component.awake()
        assert component.play(), component.last_compile_error
        batch_id = component._batch_id

        def update(_delta):
            component.update(1. / 120.)

        def after_draw():
            nonlocal frame, changed, ticket
            try:
                frame += 1
                assert frame < 200, colors
                if ticket is None and frame >= changed + 12:
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
                        engine.exit()
                    else:
                        next_token = phases[len(colors)]
                        if next_token == "reset":
                            assert component.reset_parameter("surface")
                        else:
                            component.set_parameter("surface", AssetReference(guid=next_token))
                        changed, ticket = frame, None
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(colors) == len(phases)
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_post_draw_callback(None)
        component._remove_native_batch()
        engine.set_render_pipeline(None)
        manager.stop()
        assert remove_runtime_service("gpu-particles", engine)
        for path in (graph, shader):
            assert database.delete_asset(str(path))
