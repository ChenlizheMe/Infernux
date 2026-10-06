"""Effect import is stage independent; value replay retains its mount context."""
import json
from pathlib import Path

import pytest

import infernux as inx
from infernux.rendergraph.graph import Format, RenderGraph
from infernux.core.asset_ref import RenderEffectRef
from infernux.renderstack import render_effect_compiler as compiler
from infernux.renderstack.effect_slot import EffectSlot
from infernux.renderstack.resource_bus import ResourceBus
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import RenderEffectAsset, dump_render_effect_document


@pytest.fixture
def stage_features(monkeypatch):
    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, "_FEATURES", dict(compiler._FEATURES))
    compiler.RenderEffectArtifactRegistry.clear()
    yield
    compiler.RenderEffectArtifactRegistry.clear()


def test_import_does_not_execute_setup_without_a_mount_stage(tmp_path, monkeypatch, stage_features):
    calls = []

    @compiler.render_effect_feature("tests.stage.required")
    class Required(inx.renderstack.FullScreenEffect):
        name = "Stage Required"
        injection_point = "custom"
        requires = {"color", "missing_probe"}

        def setup_passes(self, graph, bus):
            calls.append(graph.name)
            if bus.get("missing_probe") is None:
                raise ValueError("missing effect-stage resource: missing_probe")

    monkeypatch.setattr(inx.engine.project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Required.effect"
    path.write_text(dump_render_effect_document(RenderEffectAsset("tests.stage.required")), encoding="utf-8")
    artifact, _ = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="stage-required")
    assert calls == []
    assert artifact.features[0]["requires"] == ["color", "missing_probe"]
    # The declaration must also survive loading the on-disk artifact.
    compiler.RenderEffectArtifactRegistry.clear()
    restored, _ = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="stage-required")
    assert restored.features == tuple(json.loads(Path(artifact.artifact_path).read_text(encoding="utf-8"))["features"])
    assert calls == []
    graph = RenderGraph()
    bus = ResourceBus({"color": graph.create_texture("stage_color")}, graph=graph)
    source = RenderEffect(RenderEffectAsset("tests.stage.required"))
    stage = type("Stage", (), {"stable_id": "custom"})()
    slot = EffectSlot(stage_id="custom", effect=RenderEffectRef(source))
    bindings, errors = compiler.compile_effect_slots(stage, [slot], graph, bus)
    assert bindings == []
    assert len(errors) == 1 and errors[0].startswith("custom/" + slot.slot_id + ":")
    assert "missing effect-stage resource: missing_probe" in errors[0]
    assert graph.pass_count == 0


def _register_context_effect():
    @compiler.render_effect_feature("tests.stage.context")
    class ContextEffect(inx.renderstack.FullScreenEffect):
        name = "Context Effect"
        injection_point = "custom"
        requires = {"color", "light_list"}
        strength: float = inx.serialized_field(default=1.0)

        def setup_passes(self, graph, bus):
            color = bus.require_texture("color")
            lights = bus.require_buffer("light_list")
            assert graph.get_texture(color.name) is color
            assert graph.current_effect_resources["color"] is color
            assert graph.current_pass_result.sample("color") is color
            before = graph.pass_count
            with graph.add_pass("Context") as render_pass:
                render_pass.read(color).write_color(color)
                render_pass.fullscreen_quad("fullscreen_blit")
                for name, value in {
                    "strength": self.strength,
                    "samples": float(color.samples),
                    "bytes": float(lights.byte_size),
                    "depth": float(bus.has("depth")),
                    "prefix": float(before),
                }.items():
                    render_pass.set_param(name, value)
            # Mutating the recorded inputs must not mutate the live graph or
            # poison the frozen context used for the following update.
            color.size_divisor += 1
            lights.byte_size += 16

    return ContextEffect


def _stage(samples, size, depth=False):
    graph = RenderGraph("Mount", output_samples=samples)
    color = graph.create_texture("view_color", format=Format.RGBA16_SFLOAT, samples=samples)
    lights = graph.create_buffer("view_lights", byte_size=size)
    resources = {"color": color, "light_list": lights}
    if depth:
        resources["depth"] = graph.create_texture("view_depth", format=Format.D32_SFLOAT)
    graph.add_pass("Prefix").write_color(color)
    result = graph.publish_pass_result("stage", resources)
    return graph, ResourceBus(resources, graph=graph), result


def test_parameter_updates_retain_exact_mount_resources_and_do_not_mutate_live_graph(stage_features):
    _register_context_effect()
    source = RenderEffect(RenderEffectAsset("tests.stage.context", parameters={"strength": 1.0}))
    graph, bus, result = _stage(4, 96)
    with graph.pass_result(result), graph.effect_resources({"color": bus.get("color")}):
        binding = compiler._compile_effect(source, graph, bus, binding_id="route/slot/0")
    live_state = (graph.pass_count, len(graph._textures), len(graph._buffers), bus.get("color").size_divisor, bus.get("light_list").byte_size)
    for strength in (2.0, 3.0):
        source.set_float("strength", strength)
        rebuild, updates = binding.collect_updates()
        assert not rebuild and len(updates) == 1
        assert dict(updates[0].values) == {"strength": strength, "samples": 4.0, "bytes": 96.0, "depth": 0.0, "prefix": 1.0}
        assert (graph.pass_count, len(graph._textures), len(graph._buffers), bus.get("color").size_divisor, bus.get("light_list").byte_size) == live_state


def test_same_asset_mounted_in_two_stages_keeps_independent_contexts(stage_features):
    _register_context_effect()
    source = RenderEffect(RenderEffectAsset("tests.stage.context", parameters={"strength": 1.0}))
    bindings = []
    for samples, size, depth in ((1, 64, False), (8, 128, True)):
        graph, bus, result = _stage(samples, size, depth)
        with graph.pass_result(result), graph.effect_resources({"color": bus.get("color")}):
            bindings.append(compiler._compile_effect(source, graph, bus, binding_id=str(samples)))
    source.set_float("strength", 2.0)
    for binding, expected in zip(bindings, ((1.0, 64.0, 0.0), (8.0, 128.0, 1.0))):
        rebuild, updates = binding.collect_updates()
        assert not rebuild
        values = dict(updates[0].values)
        assert (values["samples"], values["bytes"], values["depth"]) == expected


@pytest.mark.parametrize("change", ["shader", "count", "resource"])
def test_parameter_that_changes_undeclared_topology_requests_rebuild(stage_features, change):
    @compiler.render_effect_feature("tests.stage.topology")
    class Topology(inx.renderstack.FullScreenEffect):
        name = "Topology"
        injection_point = "custom"
        strength: float = inx.serialized_field(default=1.0)

        def setup_passes(self, graph, bus):
            color = bus.require_texture("color")
            alternate = self.strength > 1.0
            if change == "resource" and alternate:
                color = graph.create_texture("alternative", format=color.format)
            with graph.add_pass("Probe") as render_pass:
                render_pass.write_color(color)
                render_pass.fullscreen_quad("other_shader" if change == "shader" and alternate else "fullscreen_blit")
                render_pass.set_param("strength", self.strength)
            if change == "count" and alternate:
                graph.add_pass("Extra").write_color(color)

    source = RenderEffect(RenderEffectAsset("tests.stage.topology", parameters={"strength": 1.0}))
    graph, bus, _ = _stage(1, 64)
    binding = compiler._compile_effect(source, graph, bus, binding_id="topology")
    source.set_float("strength", 2.0)
    assert binding.collect_updates() == (True, [])
