"""Nested serialized editors retain their component-owned MCP identities."""
from types import SimpleNamespace

import pytest

from Infernux.components.fields import serialized_field
from Infernux.engine.ui import inspector_declarative as ui
from Infernux.renderstack.render_pipeline import RenderPipeline


class ExamplePipeline(RenderPipeline):
    name = "Nested semantic contract"
    radius: float = serialized_field(default=1.0, range=(0.0, 3.0))
    hidden_value: float = serialized_field(default=2.0, hidden=True)
    measured: int = serialized_field(default=3, readonly=True)


@pytest.mark.parametrize("capture", [False, True])
def test_nested_fields_are_scoped_to_owner_and_control(monkeypatch, capture):
    owner = SimpleNamespace(game_object=SimpleNamespace(id=7), component_id=11)
    records = []
    ctx = SimpleNamespace(semantic_capture_enabled=capture,
                          record_semantic_item=lambda *args: records.append(args))
    monkeypatch.setattr(ui, "max_label_w", lambda *_: 100.0)
    monkeypatch.setattr(ui, "render_serialized_field", lambda _ctx, _wid, _name, _meta, value, _lw: value)
    for key in ("first_pipeline", "second_pipeline"):
        # A replacement instance must not change the field's semantic identity.
        for _ in range(2):
            target = ExamplePipeline()
            ui._render_serialized_target(ctx, ui.InspectorSerializedTarget(
                key=key, target=lambda: target, owner=owner))
    expected = []
    if capture:
        for key in ("first_pipeline", "second_pipeline"):
            for _ in range(2):
                expected.extend([
                    ("inspector_field", "Radius", True, f"inspector.object.7.component.11.{key}.radius"),
                    ("inspector_field", "Measured", False, f"inspector.object.7.component.11.{key}.measured"),
                ])
    assert records == expected


def test_semantic_capture_does_not_bypass_field_transaction(monkeypatch):
    owner = SimpleNamespace(game_object=SimpleNamespace(id=17), component_id=21)
    target = ExamplePipeline()
    changes, records = [], []
    ctx = SimpleNamespace(semantic_capture_enabled=True,
                          record_semantic_item=lambda *args: records.append(args))
    monkeypatch.setattr(ui, "max_label_w", lambda *_: 100.0)
    monkeypatch.setattr(ui, "render_serialized_field", lambda _ctx, _wid, _name, _meta, value, _lw: value + 1)
    ui._render_serialized_target(ctx, ui.InspectorSerializedTarget(
        key="pipeline", target=lambda: target, owner=owner,
        on_change=lambda *args: changes.append(args)))
    assert changes == [(target, "radius", 1.0, 2.0)]
    assert target.radius == 1.0 and target.measured == 3
    assert len(records) == 2
