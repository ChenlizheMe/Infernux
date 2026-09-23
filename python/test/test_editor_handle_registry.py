"""Public custom Scene View handles and plugin-owned retirement."""

import pytest

from Infernux.editor import (
    EditorHandleKind,
    EditorHandleProvider,
    EditorHandleRegistry,
    register_handle_provider,
)
from Infernux.engine.ui.panel_registry import PanelRegistry
from Infernux.gizmos import Gizmos
from Infernux.plugins.preload import _remove_editor_contribution_owner


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(EditorHandleRegistry, "_instance", None)
    Gizmos._begin_frame()
    value = EditorHandleRegistry.instance()
    yield value
    value.clear()
    Gizmos._begin_frame()


def test_public_provider_submits_position_direction_radius_and_limit(registry):
    changed = []

    def draw(handles):
        handles.position("origin", (0, 0, 0), changed.append)
        handles.direction("forward", (0, 0, 0), (1, 0, 0), changed.append, length=2.0)
        handles.radius("range", (0, 0, 0), 3.0, changed.append)
        handles.limit("travel", (0, 0, 0), (0, 1, 0), (-1.0, 4.0), changed.append)

    lease = register_handle_provider("sample", draw)
    registry.collect(object())

    assert [item.handle_id for item in registry.frame_handles] == [
        "sample:origin", "sample:forward", "sample:range", "sample:travel",
    ]
    assert [item.kind for item in registry.frame_handles] == [
        EditorHandleKind.POSITION,
        EditorHandleKind.DIRECTION,
        EditorHandleKind.RADIUS,
        EditorHandleKind.LIMIT,
    ]
    assert registry.hit_test((0, 0, 5), (0, 0, -1)) == "sample:origin"
    assert lease.close()
    assert not lease.close()
    assert registry.frame_handles == ()


def test_capture_cancel_restores_value_before_resource_retirement(registry):
    events = []

    def draw(handles):
        handles.position(
            "pivot",
            (0, 0, 0),
            lambda value: events.append(("change", value)),
            on_begin=lambda value: events.append(("begin", value)),
            on_commit=lambda value: events.append(("commit", value)),
            on_cancel=lambda value: events.append(("cancel", value)),
        )

    with PanelRegistry.contribution_scope("preload:handle-owner"):
        register_handle_provider(
            "owned", draw, on_retire=lambda: events.append(("retire", None))
        )

    registry.collect(object())
    assert registry.begin_capture("owned:pivot", (0, 0, 5), (0, 0, -1))
    assert registry.update_capture((1, 0, 5), (0, 0, -1))
    assert events[:2] == [
        ("begin", (0.0, 0.0, 0.0)),
        ("change", (1.0, 0.0, 0.0)),
    ]

    assert _remove_editor_contribution_owner("preload:handle-owner", runtime=False)
    assert events[-3:] == [
        ("change", (0.0, 0.0, 0.0)),
        ("cancel", (0.0, 0.0, 0.0)),
        ("retire", None),
    ]
    assert registry.active_capture_id == ""
    assert registry.frame_handles == ()
    assert registry.providers == ()


def test_owner_retirement_also_releases_core_transient_capture(registry, monkeypatch):
    from Infernux.engine.interaction.contexts import FocusService
    from Infernux.engine.interaction.transient_interactions import TransientInteractionService

    monkeypatch.setattr(FocusService, "_instance", None)
    monkeypatch.setattr(TransientInteractionService, "_instance", None)
    focus = FocusService.instance()
    focus.activate_panel("scene_view", reason="test", record_history=False)
    transient = TransientInteractionService.instance()

    with PanelRegistry.contribution_scope("preload:transient-owner"):
        register_handle_provider(
            "transient",
            lambda handles: handles.position(
                "pivot", (0, 0, 0), lambda _value: None,
            ),
        )
    registry.collect(object())
    assert registry.begin_capture(
        "transient:pivot",
        (0, 0, 5),
        (0, 0, -1),
        interaction_owner="scene_view",
    )
    assert transient.active is not None
    assert transient.active.kind == "custom_scene_handle_drag"

    assert registry.unregister_owner("preload:transient-owner") == 1
    assert transient.active is None
    assert registry.active_capture_id == ""


def test_disabled_provider_cancels_capture_and_drops_frame_callbacks(registry):
    enabled = [True]
    changes = []

    def draw(handles):
        handles.radius("radius", (0, 0, 0), 2.0, changes.append)

    registry.register(EditorHandleProvider("toggle", draw, lambda: enabled[0]))
    registry.collect(object())
    assert registry.begin_capture("toggle:radius", (2, 0, 5), (0, 0, -1))

    enabled[0] = False
    registry.collect(object())

    assert registry.active_capture_id == ""
    assert registry.frame_handles == ()
    assert changes == [2.0]


def test_cross_owner_replace_is_rejected_and_same_owner_retires_old_provider(registry):
    events = []
    first = EditorHandleProvider("tool", lambda _ctx: None, on_retire=lambda: events.append("first"))
    second = EditorHandleProvider("tool", lambda _ctx: None, on_retire=lambda: events.append("second"))

    with PanelRegistry.contribution_scope("first-owner"):
        registry.register(first)
        registry.register(second, replace=True)
    assert events == ["first"]

    with PanelRegistry.contribution_scope("second-owner"):
        with pytest.raises(ValueError, match="another contributor"):
            registry.register(EditorHandleProvider("tool", lambda _ctx: None), replace=True)

    assert registry.unregister_owner("first-owner") == 1
    assert events == ["first", "second"]
    assert registry.unregister_owner("first-owner") == 0
    with pytest.raises(ValueError, match="must not be empty"):
        registry.unregister_owner("")


def test_duplicate_local_handle_id_fails_the_frame_instead_of_overwriting(registry):
    def draw(handles):
        handles.position("same", (0, 0, 0), lambda _value: None)
        handles.radius("same", (0, 0, 0), 1.0, lambda _value: None)

    registry.register(EditorHandleProvider("broken", draw))
    with pytest.raises(ValueError, match="duplicate handle_id"):
        registry.collect(object())


def test_context_cannot_be_reused_after_its_collection_frame(registry):
    captured = []
    registry.register(
        EditorHandleProvider("one-frame", lambda handles: captured.append(handles))
    )
    registry.collect(object())
    with pytest.raises(RuntimeError, match="valid only during"):
        captured[0].position("late", (0, 0, 0), lambda _value: None)


def test_failed_provider_retires_previous_frame_and_live_capture(registry):
    fail = [False]

    def draw(handles):
        if fail[0]:
            raise RuntimeError("provider failed")
        handles.position("pivot", (0, 0, 0), lambda _value: None)

    registry.register(EditorHandleProvider("unstable", draw))
    registry.collect(object())
    assert registry.begin_capture("unstable:pivot", (0, 0, 5), (0, 0, -1))

    fail[0] = True
    with pytest.raises(RuntimeError, match="provider failed"):
        registry.collect(object())

    assert registry.active_capture_id == ""
    assert registry.hovered_handle_id == ""
    assert registry.frame_handles == ()


def test_retirement_clears_every_callback_even_when_cancel_and_retire_fail(registry):
    events = []

    def fail(label):
        events.append(label)
        raise RuntimeError(label)

    def draw(handles):
        handles.position(
            "pivot", (0, 0, 0),
            lambda value: fail("restore") if tuple(value) == (0.0, 0.0, 0.0) else None,
            on_cancel=lambda _value: fail("cancel"),
        )

    with PanelRegistry.contribution_scope("faulty-owner"):
        register_handle_provider("faulty", draw, on_retire=lambda: fail("retire"))
        register_handle_provider("other", lambda _handles: None)
    registry.collect(object())
    registry.begin_capture("faulty:pivot", (0, 0, 5), (0, 0, -1))
    registry.update_capture((1, 0, 5), (0, 0, -1))

    with pytest.raises(RuntimeError, match="owner retirement failed"):
        registry.unregister_owner("faulty-owner")

    assert events == ["restore", "cancel", "retire"]
    assert registry.active_capture_id == ""
    assert registry.providers == ()
    assert registry.frame_handles == ()
