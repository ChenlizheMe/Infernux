import json
from pathlib import Path

import pytest

from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.renderstack.effect_slot import EffectSlot
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import (
    EffectAssetReference,
    RenderEffectAsset,
    RenderEffectGroupAsset,
    RenderEffectGroupEntry,
    dump_render_effect_document,
)
from infernux.renderstack.render_effect_compiler import (
    RenderEffectArtifactRegistry,
    RenderEffectCompileError,
    expand_render_effect_reference,
    publish_live_effect_group_document,
)
from infernux.renderstack.render_stack import RenderStack
from infernux.renderstack.render_pipeline import RenderPipeline


@pytest.fixture
def effect_catalog(monkeypatch):
    """Small GUID catalog for managed RenderEffect references in unit tests."""
    paths_by_guid = {}
    guids_by_path = {}

    class Catalog:
        @staticmethod
        def register(path):
            resolved = str(Path(path).resolve())
            existing = guids_by_path.get(resolved)
            if existing:
                return existing
            guid = f"effect-{len(paths_by_guid) + 1:024x}"
            paths_by_guid[guid] = resolved
            guids_by_path[resolved] = guid
            return guid

        @staticmethod
        def get_path_from_guid(guid):
            return paths_by_guid.get(str(guid), "")

        @staticmethod
        def get_guid_from_path(path):
            return guids_by_path.get(str(Path(path).resolve()), "")

        @classmethod
        def ref(cls, path):
            return RenderEffectRef(guid=cls.register(path), path_hint=str(path))

        @classmethod
        def asset(cls, path):
            return EffectAssetReference(guid=cls.register(path))

    monkeypatch.setattr(AssetManager, "_asset_database", Catalog())
    return Catalog


class _SingleCameraPipeline(RenderPipeline):
    name = "Single Camera Contract"

    def define_topology(self, graph):
        graph.create_texture("color", camera_target=True)
        with graph.add_pass("Opaque") as render_pass:
            render_pass.write_color("color")
            render_pass.draw_renderers()
        graph.set_output("color")


class _BufferedSingleCameraPipeline(RenderPipeline):
    name = "Buffered Single Camera Contract"

    def define_topology(self, graph):
        from infernux.rendergraph.graph import Format

        color = graph.create_texture("color", camera_target=True)
        depth = graph.create_texture("depth", format=Format.D32_SFLOAT)
        normal = graph.create_texture("normal", format=Format.RGBA16_SFLOAT)
        motion = graph.create_texture("motion", format=Format.RG16_SFLOAT)
        with graph.add_pass("Opaque") as render_pass:
            render_pass.write_color(color)
            render_pass.write_depth(depth)
            render_pass.draw_renderers()
        result = graph.publish_pass_result(
            "opaque",
            {
                "color": color,
                "depth": depth,
                "normal": normal,
                "motion": motion,
            },
        )
        with graph.pass_result(result):
            graph.injection_point(
                "after_opaque",
                resources={"color", "depth", "normal", "motion"},
            )
        graph.set_output(color)


class _SingleCameraContext:
    output_samples = 0

    def __init__(self):
        self.calls = []
        self.current_revision = 0

    def setup_camera_properties(self, camera):
        self.calls.append(("setup", camera))

    def cull(self, camera):
        self.calls.append(("cull", camera))
        return "culling"

    def is_graph_revision_current(self, revision):
        self.calls.append(("is_current", revision))
        return self.current_revision == revision

    def apply_graph(self, description):
        self.calls.append(("apply", description.source_revision))
        self.current_revision = description.source_revision

    def submit_culling(self, culling):
        self.calls.append(("submit", culling))


def test_render_effect_clone_is_runtime_only_and_parameter_isolated():
    shared = RenderEffect(
        RenderEffectAsset(
            feature_type="infernux.post.bloom",
            parameters={"intensity": 0.5},
        ),
        file_path="Assets/Effects/Bloom.effect",
        guid="bloom-guid",
    )

    instance = shared.clone()
    instance.set_float("intensity", 2.0)

    assert shared.get_float("intensity") == pytest.approx(0.5)
    assert instance.get_float("intensity") == pytest.approx(2.0)
    assert instance.guid == ""
    assert instance.file_path == ""


def test_render_effect_save_and_load_round_trip(tmp_path):
    path = tmp_path / "Bloom.effect"
    effect = RenderEffect(
        RenderEffectAsset(
            feature_type="infernux.post.bloom",
            parameters={"threshold": 1.0},
        )
    )
    assert effect.save(str(path))

    loaded = RenderEffect.load(str(path))

    assert loaded is not None
    assert loaded.feature_type == "infernux.post.bloom"
    assert loaded.get_float("threshold") == pytest.approx(1.0)


def _write_effect(path: Path, *, intensity=0.5, max_iterations=3, extra=None):
    parameters = {
        "intensity": intensity,
        "max_iterations": max_iterations,
    }
    parameters.update(extra or {})
    path.write_text(
        dump_render_effect_document(
            RenderEffectAsset(
                feature_type="infernux.post.bloom",
                parameters=parameters,
            )
        ),
        encoding="utf-8",
    )


def test_render_effect_aot_artifact_distinguishes_dynamic_and_structural_edits(
    tmp_path,
    monkeypatch,
):
    from infernux.engine import project_context

    RenderEffectArtifactRegistry.clear()
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Assets" / "Bloom.effect"
    path.parent.mkdir()
    _write_effect(path, intensity=0.5, max_iterations=3)

    first, _ = RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="bloom-guid")
    first_generation = RenderEffectArtifactRegistry.topology_generation()
    assert Path(first.artifact_path).is_file()

    _write_effect(path, intensity=1.5, max_iterations=3)
    dynamic, _ = RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="bloom-guid")
    assert dynamic.revision > first.revision
    assert dynamic.structural_hash == first.structural_hash
    assert RenderEffectArtifactRegistry.topology_generation() == first_generation

    _write_effect(path, intensity=1.5, max_iterations=5)
    structural, _ = RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="bloom-guid")
    assert structural.structural_hash != dynamic.structural_hash
    assert RenderEffectArtifactRegistry.topology_generation() == first_generation + 1


def test_render_effect_failed_compile_preserves_last_known_good_artifact(
    tmp_path,
    monkeypatch,
):
    from infernux.engine import project_context

    RenderEffectArtifactRegistry.clear()
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Assets" / "Bloom.effect"
    path.parent.mkdir()
    _write_effect(path)
    published, _ = RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="bloom-guid")
    artifact_text = Path(published.artifact_path).read_text(encoding="utf-8")

    _write_effect(path, extra={"not_a_bloom_parameter": 1.0})
    with pytest.raises(RenderEffectCompileError, match="unknown parameters"):
        RenderEffectArtifactRegistry.compile_and_publish(str(path), guid="bloom-guid")

    assert RenderEffectArtifactRegistry.get(str(path), "bloom-guid") == published
    assert Path(published.artifact_path).read_text(encoding="utf-8") == artifact_text


def test_asset_publish_updates_loaded_render_effect_in_place(tmp_path, monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.engine import project_context

    RenderEffectArtifactRegistry.clear()
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    path = tmp_path / "Assets" / "Bloom.effect"
    path.parent.mkdir()
    _write_effect(path, intensity=0.5)
    loaded = RenderEffect.load(str(path))
    assert loaded is not None
    monkeypatch.setattr(
        AssetManager,
        "_get_cached",
        classmethod(lambda _cls, guid: loaded if guid == "bloom-guid" else None),
    )

    assert AssetManager._compile_render_effect_runtime(str(path), "bloom-guid") == ""
    first_revision = loaded.artifact_revision
    _write_effect(path, intensity=2.0)
    assert AssetManager._compile_render_effect_runtime(str(path), "bloom-guid") == ""

    assert loaded.get_float("intensity") == pytest.approx(2.0)
    assert loaded.artifact_revision > first_revision


def test_effect_group_expands_in_order_with_non_destructive_overrides(tmp_path, effect_catalog):
    bloom_path = tmp_path / "Bloom.effect"
    tone_path = tmp_path / "Tone.effect"
    bloom_path.write_text(
        dump_render_effect_document(
            RenderEffectAsset(
                feature_type="infernux.post.bloom",
                parameters={"intensity": 0.5, "max_iterations": 2},
            )
        ),
        encoding="utf-8",
    )
    tone_path.write_text(
        dump_render_effect_document(
            RenderEffectAsset(
                feature_type="infernux.post.tonemapping",
                parameters={"exposure": 1.0},
            )
        ),
        encoding="utf-8",
    )
    group_path = tmp_path / "Post.effectgroup"
    group_path.write_text(
        dump_render_effect_document(
            RenderEffectGroupAsset(
                entries=(
                    RenderEffectGroupEntry(
                        "bloom",
                        effect_catalog.asset(bloom_path),
                        overrides={"intensity": 0.9},
                    ),
                    RenderEffectGroupEntry(
                        "tone",
                        effect_catalog.asset(tone_path),
                    ),
                )
            )
        ),
        encoding="utf-8",
    )

    effects = expand_render_effect_reference(effect_catalog.ref(group_path))

    assert [effect.feature_type for effect in effects] == [
        "infernux.post.bloom",
        "infernux.post.tonemapping",
    ]
    assert effects[0].get_float("intensity") == pytest.approx(0.9)


def test_effect_group_parameter_publication_reaches_compiled_render_stack(tmp_path, effect_catalog):
    RenderEffectArtifactRegistry.clear()
    bloom_path = tmp_path / "Bloom.effect"
    bloom_path.write_text(
        dump_render_effect_document(
            RenderEffectAsset(
                feature_type="infernux.post.bloom",
                parameters={"intensity": 0.5, "max_iterations": 2},
            )
        ),
        encoding="utf-8",
    )
    group_path = tmp_path / "World.effectgroup"
    original = RenderEffectGroupAsset(
        entries=(
            RenderEffectGroupEntry(
                "bloom",
                effect_catalog.asset(bloom_path),
                overrides={"intensity": 0.75},
            ),
        )
    )
    group_path.write_text(
        dump_render_effect_document(original),
        encoding="utf-8",
    )

    reference = effect_catalog.ref(group_path)
    projection = expand_render_effect_reference(reference)[0]
    previous_revision = projection.revision
    stack = RenderStack()
    stack.add_effect_slot(
        "final",
        effect_catalog.ref(group_path),
    )
    stack._graph_state.description = stack.build_graph()

    class Context:
        graph_instance_id = 41

    requires_rebuild, initial = stack._collect_effect_parameter_updates(Context())
    assert requires_rebuild is False
    assert any(dict(update.values).get("intensity") == pytest.approx(0.75) for update in initial)
    assert stack._collect_effect_parameter_updates(Context()) == (False, [])

    edited = RenderEffectGroupAsset(
        entries=(
            RenderEffectGroupEntry(
                "bloom",
                effect_catalog.asset(bloom_path),
                overrides={"intensity": 1.5},
            ),
        )
    )
    publish_live_effect_group_document(str(group_path), edited)
    assert expand_render_effect_reference(reference)[0] is projection
    assert projection.revision > previous_revision
    requires_rebuild, updates = stack._collect_effect_parameter_updates(Context())

    assert requires_rebuild is False
    assert any(dict(update.values).get("intensity") == pytest.approx(1.5) for update in updates)


def test_failed_effect_compile_rolls_back_partial_graph_mutation():
    from infernux.renderstack.fullscreen_effect import FullScreenEffect
    from infernux.renderstack.render_effect_compiler import register_render_effect_feature

    class BrokenEffect(FullScreenEffect):
        name = "Broken Test Effect"
        injection_point = "after_post_process"
        default_order = 1

        def setup_passes(self, graph, bus):
            graph.create_texture("partial")
            with graph.add_pass("Partial") as render_pass:
                render_pass.write_color("partial")
                render_pass.fullscreen_quad("fullscreen_blit")
            raise ValueError("intentional compile failure")

    register_render_effect_feature("tests.post.broken", BrokenEffect)
    stack = RenderStack()
    stack.add_effect_slot(
        "final",
        RenderEffectRef(effect=RenderEffect(RenderEffectAsset("tests.post.broken"))),
    )

    description = stack.build_graph()

    assert not any("Partial" in render_pass.name for render_pass in description.passes)
    assert any("intentional compile failure" in error for error in stack.effect_compile_errors)
