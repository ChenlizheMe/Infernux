"""Settings navigation and exception unwinding at the shared UI boundary."""

from types import SimpleNamespace

import pytest

from infernux.components.fields import FieldMetadata, FieldType
from infernux.engine.interaction import EditorInteractionCore, PanelInteractionDescriptor
from infernux.engine.ui.editor_panel import EditorPanel
from infernux.engine.ui.settings_workspace import SettingsWorkspaceMixin
from infernux.engine.ui.window_manager import WindowManager


def test_settings_entry_points_reuse_one_view_and_select_the_requested_page():
    class Settings(SettingsWorkspaceMixin, EditorPanel):
        SETTINGS_PAGES = tuple((key, key, key, "") for key in ("player", "physics", "tags_layers"))

        def __init__(self):
            super().__init__("Project settings", "project_settings")
            self._settings_page = "player"

    core = EditorInteractionCore()
    core.panels.register_type("project_settings", PanelInteractionDescriptor())
    registered = []
    engine = SimpleNamespace(select_docked_window=lambda window: None)
    manager = WindowManager(engine, core.panels, lambda window, panel: registered.append((window, panel)))
    manager.register_window_type("project_settings", Settings, "Project settings")
    for entry, page in (("build_settings", "player"), ("physics_settings", "physics"),
                        ("tag_layer_settings", "tags_layers")):
        manager.register_window_alias(entry, "project_settings", page)

    panel = manager.open_window_from_user("build_settings")
    # Switch pages while the first view registration is still pending.
    assert manager.open_window_from_user("physics_settings") is panel
    manager.process_pending_actions()
    assert panel.current_child_context_id() == "physics"
    assert registered == [("project_settings", panel)]
    for entry, page in (("tag_layer_settings", "tags_layers"), ("build_settings", "player")):
        assert manager.open_window(entry) is panel
        manager.process_pending_actions()
        assert manager.get_window_instance(entry) is panel
        assert manager.is_window_open(entry)
        assert panel.current_child_context_id() == page
    assert set(manager.get_open_windows()) == {"project_settings"}
    assert len(registered) == 1


def test_effect_mount_points_keep_pipeline_order_even_when_empty(monkeypatch):
    from infernux.engine.ui import inspector_renderstack_view as view
    from infernux.engine.ui.inspector_declarative import InspectorList

    stage_keys = ("opaque", "sky", "transparent", "camera_ui", "final")
    controls = [InspectorList(key=key, label=key, target=None, field_name=key,
                              metadata=None, value=lambda key=key: ["effect"] if key == "sky" else [])
                for key in stage_keys]
    events = []
    ctx = SimpleNamespace(
        inspector_label_width=lambda: 100, combo=lambda *args: 0,
        push_id_str=lambda key: events.append(("push", key)),
        pop_id=lambda: events.append(("pop",)),
    )
    choice = SimpleNamespace(label="pipeline", current_index=lambda: 0, options=lambda: ("default",))
    model = SimpleNamespace(key="stack", sections=(
        SimpleNamespace(controls=(choice, None)), SimpleNamespace(controls=controls),
    ))
    monkeypatch.setattr(view, "field_label", lambda *args: None)
    monkeypatch.setattr(view, "render_section_kicker", lambda *args: None)
    monkeypatch.setattr(view, "render_compact_section_header", lambda *args, **kwargs: False)
    monkeypatch.setattr(view, "_stage_title", lambda ctx, control, items, index: events.append(("stage", control.key, index)))
    monkeypatch.setattr(view, "_stage_items", lambda ctx, control, items: events.append(("items", control.key)))
    view.render_renderstack_inspector(ctx, None, model)
    assert [event[1] for event in events if event[0] == "stage"] == list(stage_keys)
    assert [event for event in events if event[0] == "items"] == [("items", "sky")]
    assert sum(event[0] == "push" for event in events) == sum(event[0] == "pop" for event in events)


def test_serialized_widget_failure_always_ends_its_group(monkeypatch):
    from infernux.engine.ui import field_draw

    depth = []
    ctx = SimpleNamespace(begin_group=lambda: depth.append("group"), end_group=lambda: depth.pop())
    metadata = FieldMetadata(name="speed", field_type=FieldType.FLOAT, default=0.0)

    def fail(*args):
        raise RuntimeError("widget failed")

    monkeypatch.setattr(field_draw, "_draw_value_widget", fail)
    with pytest.raises(RuntimeError, match="widget failed"):
        field_draw.draw_field(ctx, metadata, 1.0, label_width=100, on_change=lambda value: None)
    assert depth == []


def test_list_renderer_failure_always_pops_item_id_and_body(monkeypatch):
    from infernux.engine.ui import _inspector_list_field as lists
    from infernux.engine.ui.igui import IGUI

    scopes = []
    ctx = SimpleNamespace(push_id_str=lambda key: scopes.append("id"), pop_id=lambda: scopes.pop())
    monkeypatch.setattr(IGUI, "list_body_begin", lambda ctx: scopes.append("body"))
    monkeypatch.setattr(IGUI, "list_body_end", lambda ctx: scopes.pop())
    monkeypatch.setattr(IGUI, "reorder_separator", lambda *args: None)
    monkeypatch.setattr(lists, "_make_list_default_element", lambda *args: None)

    def fail(*args, **kwargs):
        assert scopes == ["body", "id"]
        raise RuntimeError("item failed")

    monkeypatch.setattr(lists, "_render_serializable_list_item", fail)
    metadata = FieldMetadata(name="items", field_type=FieldType.LIST, default=[])
    with pytest.raises(RuntimeError, match="item failed"):
        lists._render_list_items_body(ctx, object(), "items", metadata, [object()],
                                      FieldType.SERIALIZABLE_OBJECT, set(), 4, [])
    assert scopes == []
