from types import SimpleNamespace

from infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch
from infernux.engine.runtime_mouse_events import MouseEventDispatcher


def test_mouse_dispatcher_matches_enter_over_down_drag_up_button_exit(monkeypatch):
    events = []
    names = ("enter", "over", "exit", "down", "drag", "up", "up_as_button")
    component_type = type("MouseSequenceProbe", (), {
        f"on_mouse_{name}": lambda self, _name=name: events.append(f"on_mouse_{_name}")
        for name in names
    })
    component = component_type()
    publication = publish_runtime_dispatch_epoch((component_type,))
    publication.commit()
    target = SimpleNamespace(id=1, get_py_components=lambda: (component,))
    hits = iter([SimpleNamespace(game_object=target)] * 4 + [None])
    states = iter([(False, False, False), (True, True, False),
                   (True, False, False), (False, False, True),
                   (False, False, False)])
    monkeypatch.setattr("infernux.engine.runtime_mouse_events.Physics.raycast_screen",
                        lambda *args, **kwargs: next(hits))
    monkeypatch.setattr("infernux.engine.runtime_mouse_events.Input.get_game_mouse_frame_state",
                        lambda _button: (0, 0, 0, 0, *next(states)))
    try:
        dispatcher = MouseEventDispatcher()
        for _ in range(5):
            dispatcher.process(SimpleNamespace(culling_mask=0xffffffff), (1, 1), (10, 10))
        assert events == ["on_mouse_enter", "on_mouse_over", "on_mouse_over", "on_mouse_down",
                          "on_mouse_drag", "on_mouse_over", "on_mouse_drag", "on_mouse_over",
                          "on_mouse_up", "on_mouse_up_as_button", "on_mouse_exit"]
    finally:
        publication.rollback()


def test_dispatcher_accepts_a_precomputed_hit_without_raycast(monkeypatch):
    dispatcher = MouseEventDispatcher()
    hit = SimpleNamespace(game_object=SimpleNamespace(id=17, get_py_components=lambda: ()))
    calls = []
    monkeypatch.setattr("infernux.engine.runtime_mouse_events.Physics.raycast_screen",
                        lambda *args, **kwargs: calls.append(True))
    dispatcher.process(object(), (1, 2), (100, 100), hit=hit)
    assert calls == []


def test_direct_mouse_query_respects_camera_mask_and_ignore_raycast(monkeypatch):
    calls = []
    monkeypatch.setattr("infernux.engine.runtime_mouse_events.Physics.raycast_screen",
                        lambda *args, **kwargs: calls.append(kwargs))
    MouseEventDispatcher().process(SimpleNamespace(culling_mask=5), (1, 2), (100, 100),
                                   button_state=(False, False, False))
    assert calls == [dict(layer_mask=1, query_triggers=True)]


def test_stationary_hover_and_drag_capture_use_one_query_per_frame(monkeypatch):
    calls = []
    hit = SimpleNamespace(game_object=SimpleNamespace(id=23, get_py_components=lambda: ()))
    monkeypatch.setattr(
        "infernux.engine.runtime_mouse_events.Physics.raycast_screen",
        lambda *args, **kwargs: calls.append((args, kwargs)) or hit,
    )
    dispatcher = MouseEventDispatcher()
    camera = SimpleNamespace(culling_mask=0xffffffff)
    frames = (
        (False, False, False),  # stationary hover
        (False, False, False),
        (True, True, False),    # capture
        (True, False, False),   # captured drag
        (True, False, False),
        (False, False, True),   # release still resolves up-as-button
    )
    for button_state in frames:
        before = len(calls)
        dispatcher.process(camera, (1, 2), (100, 100), button_state=button_state)
        assert len(calls) == before + 1
    assert len(calls) == len(frames)


def test_canceled_touch_releases_capture_without_button_click():
    events = []
    component_type = type("CanceledTouchProbe", (), {
        "on_mouse_down": lambda self: events.append("down"),
        "on_mouse_up": lambda self: events.append("up"),
        "on_mouse_up_as_button": lambda self: events.append("click"),
    })
    component = component_type()
    publication = publish_runtime_dispatch_epoch((component_type,))
    publication.commit()
    target = SimpleNamespace(id=91, get_py_components=lambda: (component,))
    hit = SimpleNamespace(game_object=target)
    try:
        dispatcher = MouseEventDispatcher()
        dispatcher.process(object(), (1, 2), (100, 100), hit=hit,
                           button_state=(True, True, False))
        dispatcher.process(object(), (1, 2), (100, 100), hit=hit,
                           button_state=(False, False, True), canceled=True)
        assert events == ["down", "up"]
    finally:
        publication.rollback()
