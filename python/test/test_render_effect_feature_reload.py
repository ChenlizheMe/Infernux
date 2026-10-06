import json
from pathlib import Path

import pytest

from infernux.core.asset_ref import RenderEffectRef
from infernux.renderstack import render_effect_compiler as compiler
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import RenderEffectAsset, dump_render_effect_document
from infernux.renderstack.render_stack import RenderStack


@pytest.fixture
def feature_reload_catalog(monkeypatch):
    compiler._register_builtin_features()
    monkeypatch.setattr(compiler, "_FEATURES", dict(compiler._FEATURES))
    compiler.RenderEffectArtifactRegistry.clear()
    yield
    compiler.RenderEffectArtifactRegistry.clear()


def declare_feature(path, field="intensity", pass_name="ReloadBefore"):
    namespace = {"__name__": "test_effect_reload", "__file__": str(path)}
    source = f'''
import infernux as inx
@inx.renderstack.render_effect_feature("tests.post.reload")
class ReloadEffect(inx.renderstack.FullScreenEffect):
    name = "Reload Effect"
    injection_point = "before_post_process"
    default_order = 200
    modifies = {{"color"}}
    {field}: float = inx.serialized_field(default=.35)
    def setup_passes(self, graph, bus):
        self.apply_single_source_effect(graph, bus,
            output_name="reload_out", pass_name={pass_name!r},
            shader_name="fullscreen_blit", format=inx.rendergraph.Format.RGBA16_SFLOAT,
            params={{"intensity": self.{field}}})
'''
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["ReloadEffect"]


class RenderContext:
    output_samples = 0
    graph_instance_id = 101

    def __init__(self):
        self.applied = []
        self.updates = []

    def setup_camera_properties(self, camera):
        pass

    def cull(self, camera):
        return []

    def submit_culling(self, culling):
        pass

    def apply_graph(self, description):
        self.applied.append(description)

    def render_compiled(self, camera, revision):
        return True

    def update_parameter_blocks(self, updates):
        self.updates.extend(updates)


def test_feature_reload_rebuilds_mounted_graph_without_parameter_edit(tmp_path, feature_reload_catalog):
    feature_path = tmp_path / "reload_effect.py"
    declare_feature(feature_path)
    effect = RenderEffect(RenderEffectAsset("tests.post.reload", parameters={"intensity": .35}))
    stack = RenderStack()
    stack.add_effect_slot("final", RenderEffectRef(effect=effect))
    context = RenderContext()
    stack.render(context, object())
    stack.render(context, object())
    assert len(context.applied) == 1
    revision = effect.revision

    declare_feature(feature_path, pass_name="ReloadAfter")
    stack.render(context, object())

    assert effect.revision == revision
    assert len(context.applied) == 2
    assert any(p.name.endswith("ReloadAfter") for p in context.applied[-1].passes)
    assert not any(p.name.endswith("ReloadBefore") for p in context.applied[-1].passes)
    stack.render(context, object())
    assert len(context.applied) == 2


def test_schema_reload_diagnoses_unchanged_mounted_source_and_preserves_slot(tmp_path, feature_reload_catalog):
    feature_path = tmp_path / "reload_effect.py"
    declare_feature(feature_path)
    effect = RenderEffect(RenderEffectAsset("tests.post.reload", parameters={"intensity": .35}), guid="reload-guid")
    stack = RenderStack()
    slot = stack.add_effect_slot("final", RenderEffectRef(effect=effect))
    context = RenderContext()
    stack.render(context, object())
    assert not stack.effect_compile_errors

    declare_feature(feature_path, field="strength")
    stack.render(context, object())

    assert any("unknown parameters: ['intensity']" in error for error in stack.effect_compile_errors)
    assert stack.get_effect_stage_slots("final") == (slot,)
    assert slot.effect_ref.guid == "reload-guid"
    assert effect.to_asset().parameters == {"intensity": .35}


@pytest.mark.parametrize("clear_memory_cache", [False, True])
def test_schema_reload_rejects_unchanged_cached_artifact_until_json_repaired(
    tmp_path, monkeypatch, feature_reload_catalog, clear_memory_cache
):
    from infernux.engine import project_context
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    feature_path = tmp_path / "reload_effect.py"
    declare_feature(feature_path)
    path = tmp_path / "Reload.effect"
    path.write_text(dump_render_effect_document(RenderEffectAsset(
        "tests.post.reload", parameters={"intensity": .35})), encoding="utf-8")
    accepted, _ = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="reload-guid")
    artifact_bytes = Path(accepted.artifact_path).read_bytes()
    source_bytes = path.read_bytes()
    declare_feature(feature_path, field="strength")
    if clear_memory_cache:
        compiler.RenderEffectArtifactRegistry.clear()

    with pytest.raises(compiler.RenderEffectCompileError, match="unknown parameters.*intensity"):
        compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="reload-guid")
    assert path.read_bytes() == source_bytes
    assert Path(accepted.artifact_path).read_bytes() == artifact_bytes
    if not clear_memory_cache:
        assert compiler.RenderEffectArtifactRegistry.get(str(path), "reload-guid") is accepted

    document = json.loads(path.read_text(encoding="utf-8"))
    document["parameters"] = {"strength": .35}
    path.write_text(json.dumps(document), encoding="utf-8")
    repaired, _ = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="reload-guid")
    assert compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="reload-guid")[0] is repaired


def test_feature_reload_recompiles_unchanged_asset_with_new_passes(tmp_path, monkeypatch, feature_reload_catalog):
    from infernux.engine import project_context
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    feature_path = tmp_path / "reload_effect.py"
    declare_feature(feature_path)
    path = tmp_path / "Reload.effect"
    path.write_text(dump_render_effect_document(RenderEffectAsset(
        "tests.post.reload", parameters={"intensity": .35})), encoding="utf-8")
    accepted, _ = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="reload-guid")
    source_bytes = path.read_bytes()

    declare_feature(feature_path, pass_name="ReloadAfter")
    updated, _ = compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="reload-guid")

    assert path.read_bytes() == source_bytes
    assert updated is not accepted
    assert updated.features[0]["passes"][0]["name"].endswith("ReloadAfter")
    assert compiler.RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="reload-guid")[0] is updated


def test_identical_registration_and_rejected_owner_do_not_invalidate_graphs(tmp_path, feature_reload_catalog):
    effect_class = declare_feature(tmp_path / "reload_effect.py")
    generation = compiler.RenderEffectArtifactRegistry.topology_generation()
    compiler.register_render_effect_feature("tests.post.reload", effect_class)
    assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation
    with pytest.raises(ValueError, match="already registered"):
        declare_feature(tmp_path / "another_effect.py")
    assert compiler.RenderEffectArtifactRegistry.topology_generation() == generation
    assert compiler.get_render_effect_feature("tests.post.reload").effect_class is effect_class
