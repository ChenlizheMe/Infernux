"""Tests for infernux.renderstack — InjectionPoint, RenderPass, ResourceBus, FullScreenEffect (real C++ backend)."""

from __future__ import annotations


import sys


import types
from typing import Annotated, ClassVar


import pytest
from infernux.components.fields import NonSerialized


from infernux.renderstack.injection_point import InjectionPoint


from infernux.renderstack.resource_bus import ResourceBus


from infernux.renderstack.render_pass import RenderPass


from infernux.renderstack.fullscreen_effect import FullScreenEffect
from infernux.renderstack.render_pipeline import RenderPipeline


def test_pipeline_file_callback_adopts_transactionally_published_module(monkeypatch):
    from infernux.renderstack._render_pipeline_reload import PipelineReloadMixin
    import infernux.renderstack.discovery as discovery

    module_name = "_infernux_test_pipeline_reload"
    previous_module = types.ModuleType(module_name)
    published_module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, published_module)
    invalidations = []
    monkeypatch.setattr(
        discovery,
        "invalidate_discovery_cache",
        lambda: invalidations.append("discovery"),
    )

    class Subject(PipelineReloadMixin):
        def __init__(self):
            self._pipeline_module = previous_module
            self._pipeline = object()
            self.saved = 0
            self.invalidated = 0

        def _save_current_pipeline_params(self):
            self.saved += 1

        def invalidate_graph(self):
            self.invalidated += 1

    subject = Subject()
    subject._on_pipeline_file_changed("pipeline.py")

    assert subject._pipeline_module is published_module
    assert subject._pipeline is None
    assert subject.saved == 1
    assert subject.invalidated == 1
    assert invalidations == ["discovery"]


def test_player_missing_custom_pipeline_is_not_silently_replaced(monkeypatch):
    from infernux.renderstack.render_stack import RenderStack

    stack = RenderStack()
    stack.pipeline_class_name = "Missing Packaged Pipeline"
    monkeypatch.setenv("_INFERNUX_PLAYER_MODE", "1")
    monkeypatch.setattr(stack, "discover_pipelines", lambda: {})

    with pytest.raises(RuntimeError, match="Missing Packaged Pipeline"):
        stack._create_pipeline()

    assert stack.pipeline_class_name == "Missing Packaged Pipeline"


def test_removed_pipeline_keeps_selection_and_invalidates_for_transactional_rebuild():
    from infernux.renderstack._render_pipeline_reload import PipelineReloadMixin

    class Subject(PipelineReloadMixin):
        DEFAULT_PIPELINE_NAME = "Default Forward"

        def __init__(self):
            self.pipeline_class_name = "Authored Pipeline"
            self._pipeline_catalog_signature = ("Authored Pipeline",)
            self._pipeline = object()
            self._cached_ips = object()
            self.invalidations = 0
            self.saved = 0

        def discover_pipelines(self):
            return {}

        def _save_current_pipeline_params(self):
            assert self._pipeline is not None
            self.saved += 1

        def invalidate_graph(self):
            self.invalidations += 1

    subject = Subject()
    with pytest.warns(RuntimeWarning, match="Authored Pipeline"):
        subject._sync_pipeline_catalog()

    assert subject.pipeline_class_name == "Authored Pipeline"
    assert subject._pipeline is None
    assert subject._cached_ips is None
    assert subject.invalidations == 1
    assert subject.saved == 1


def test_set_pipeline_replaces_runtime_pipeline_and_invalidates_graph_once(monkeypatch):
    from infernux.renderstack.render_stack import RenderStack

    stack = RenderStack()
    stack.pipeline_class_name = "Pipeline A"
    stack._pipeline = object()
    stack._cached_ips = object()
    invalidations = []
    monkeypatch.setattr(stack, "_save_current_pipeline_params", lambda: None)
    monkeypatch.setattr(stack, "invalidate_graph", lambda: invalidations.append(True))

    stack.set_pipeline("Pipeline B")

    assert stack.pipeline_class_name == "Pipeline B"
    assert stack._pipeline is None
    assert stack._cached_ips is None
    assert invalidations == [True]

    # Reapplying the active pipeline is a no-op. Runtime showcases can update
    # their material/effect state without rebuilding the same graph twice.
    stack.set_pipeline("Pipeline B")
    assert invalidations == [True]


@pytest.mark.parametrize("base", [RenderPipeline, FullScreenEffect])
def test_render_parameters_share_typed_fields_and_round_trip(base):
    from infernux.components.fields import get_field_schema, get_serialized_fields, serialized_field
    from infernux.renderstack.render_stack import RenderStack

    class Parameters(base):
        name = "Shared parameter contract"
        injection_point: ClassVar[str] = "final"
        _reserved_attrs_ = frozenset({"route_metadata"})
        route_metadata: list[str] = ["color"]
        runtime_mode: ClassVar[int] = 7
        runtime_tag: ClassVar = "runtime"
        CONSTANT = 32
        _private = serialized_field(default=8)
        count: int = serialized_field(default=3, field_id="stable_count")
        labels: list[str] = serialized_field(default=[])
        computed: int
        masked: int = 2
        transient: int = 6

    class Derived(Parameters):
        _reserved_attrs_ = frozenset({"masked"})
        masked = 9
        count: int = serialized_field(default=4, field_id="stable_count")
        transient: Annotated[int, NonSerialized] = 5

    fields = get_serialized_fields(Derived)
    assert set(fields) == {"count", "labels", "computed"}
    assert get_serialized_fields(Parameters)["count"].default == 3
    assert fields["count"].default == 4
    assert get_field_schema(Derived, "count").attributes["field_id"] == "stable_count"
    assert get_field_schema(Derived, "labels").attributes["element_type"] == "FieldType.STRING"
    first, untouched, restored = Derived(), Derived(), Derived()
    assert first.computed == 0 and first.runtime_mode == 7 and first.runtime_tag == "runtime"
    assert first.masked == 9 and first.transient == 5
    assert first.route_metadata == ["color"] and first._private == 8
    first.count = 9
    first.labels.append("authored")
    assert untouched.labels == [] and fields["labels"].default == []

    if base is FullScreenEffect:
        document = first.get_params_dict()
        restored.set_params_dict(document)
    else:
        stack = RenderStack()
        stack.pipeline_class_name = first.name
        stack._pipeline = first
        stack._save_current_pipeline_params()
        document = stack._encode_pipeline_params(first)
        stack._restore_pipeline_params(restored)
    assert document == {"count": 9, "labels": ["authored"], "computed": 0}
    assert restored.count == 9 and restored.labels == ["authored"]


@pytest.mark.parametrize("base", [RenderPipeline, FullScreenEffect])
def test_render_parameter_ids_must_be_unique_across_inheritance(base):
    from infernux.components.fields import serialized_field

    class Parameters(base):
        count = serialized_field(default=3, field_id="stable_count")

    with pytest.raises(ValueError, match="duplicate field_id 'stable_count'"):
        class Invalid(Parameters):
            other = serialized_field(default=4, field_id="stable_count")
