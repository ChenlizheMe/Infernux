"""Effect imports, typed resource mounts and rejected temporal declarations."""
from types import SimpleNamespace

import pytest

from infernux.components.fields import serialized_field
from infernux.core.asset_ref import RenderEffectRef
from infernux.rendergraph.graph import Format, RenderGraph
from infernux.renderstack import FullScreenEffect, RenderEffect, RenderEffectAsset, RenderStack, TemporalAAEffect
from infernux.renderstack import render_effect_compiler as compiler
from infernux.renderstack.effect_slot import EffectSlot
from infernux.renderstack.render_effect_asset import dump_render_effect_document
from infernux.renderstack.resource_bus import ResourceBus


@pytest.fixture
def features(monkeypatch):
    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, "_FEATURES", dict(compiler._FEATURES))
    compiler.RenderEffectArtifactRegistry.clear()
    yield
    compiler.RenderEffectArtifactRegistry.clear()


@pytest.mark.parametrize("semantic,shader_binding", [("shadow_map", "shadowMap"), ("light_list", "lightList")])
@pytest.mark.parametrize("extra", [False, True])
def test_import_mount_and_dynamic_update_keep_typed_stage_resources(features, tmp_path, monkeypatch, semantic, shader_binding, extra):
    import infernux.engine.project_context as project_context

    calls = []
    type_id = "tests.resource_contract." + semantic

    @compiler.render_effect_feature(type_id)
    class ResourceEffect(FullScreenEffect):
        requires = {"color", semantic}
        name = "Resource Contract"
        injection_point = "inspect"
        intensity: float = serialized_field(default=1.0)

        def setup_passes(self, graph, bus):
            calls.append(graph.name)
            with graph.add_pass("InspectResource") as render_pass:
                self.bind_buffers(render_pass, bus, extra_bindings={"extra": bus.get(semantic)} if extra else None)
                render_pass.write_color(bus.require_texture("color"))
                render_pass.fullscreen_quad("Resource Contract")
                render_pass.set_param("intensity", self.intensity)

    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Inspect.effect"
    path.write_text(dump_render_effect_document(RenderEffectAsset(type_id)), encoding="utf-8")
    artifact, document = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="a" * 32)
    assert calls == []
    assert artifact.features[0]["requires"] == sorted({"color", semantic})
    compiler.RenderEffectArtifactRegistry.clear()
    restored, _ = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="a" * 32)
    assert restored.features == artifact.features and calls == []

    graph = RenderGraph("Real Stage")
    color = graph.create_texture("color")
    resource = (graph.create_buffer("lights", byte_size=64) if semantic == "light_list"
                else graph.create_texture("shadow", format=Format.D32_SFLOAT))
    bus = ResourceBus({"color": color, semantic: resource}, graph=graph)
    source = RenderEffect(document)
    slot = EffectSlot(stage_id="inspect", effect=RenderEffectRef(source))
    bindings, errors = compiler.compile_effect_slots(SimpleNamespace(stable_id="inspect"), [slot], graph, bus)
    assert not errors and len(bindings) == 1
    graph.set_output(color)
    description = graph.build()
    render_pass = next(item for item in description.passes if item.name.endswith("InspectResource"))
    inputs = dict(render_pass.commands[0].input_bindings)
    assert inputs[shader_binding] == resource.name
    if extra:
        assert inputs["extra"] == resource.name
    source.set_float("intensity", 2.5)
    rebuild, updates = bindings[0].collect_updates()
    assert not rebuild and len(updates) == 1
    assert dict(updates[0].values)["intensity"] == 2.5
    assert len(calls) == 2  # Mount plus replay, never import.


def test_rejected_temporal_effect_does_not_jitter_real_stack(features):
    @compiler.render_effect_feature("tests.rejected_temporal_contract")
    class RejectedTemporal(TemporalAAEffect):
        def setup_passes(self, graph, bus):
            super().setup_passes(graph, bus)
            raise ValueError("rejected temporal candidate")

    stack = RenderStack()
    stack.add_effect_slot("final", RenderEffectRef(RenderEffect(RenderEffectAsset("tests.rejected_temporal_contract"))))
    description = stack.build_graph()
    assert any("rejected temporal candidate" in error for error in stack.effect_compile_errors)
    assert not any("TAA_" in item.name for item in description.passes)
    assert not description.temporal_jitter


@pytest.mark.parametrize("prior", [False, True])
def test_rejected_effect_preserves_previous_temporal_declaration(features, prior):
    @compiler.render_effect_feature("tests.rejected_jitter_contract")
    class RejectedJitter(FullScreenEffect):
        name = "Rejected Jitter"
        injection_point = "final"
        def setup_passes(self, graph, bus):
            graph.set_temporal_jitter(not prior)
            raise ValueError("reject jitter declaration")

    graph = RenderGraph("Temporal state")
    color = graph.create_texture("color")
    graph.add_pass("Base").write_color(color).fullscreen_quad("fullscreen_blit")
    graph.set_output(color)
    graph.set_temporal_jitter(prior)
    bus = ResourceBus({"color": color}, graph=graph)
    source = RenderEffect(RenderEffectAsset("tests.rejected_jitter_contract"))
    slot = EffectSlot(stage_id="final", effect=RenderEffectRef(source))
    bindings, errors = compiler.compile_effect_slots(SimpleNamespace(stable_id="final"), [slot], graph, bus)
    assert not bindings and len(errors) == 1
    assert "reject jitter declaration" in errors[0]
    assert graph.build().temporal_jitter is prior
