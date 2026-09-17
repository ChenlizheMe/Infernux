from types import SimpleNamespace

from Infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch
from Infernux.engine.runtime_mouse_events import MouseEventDispatcher


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
    monkeypatch.setattr("Infernux.engine.runtime_mouse_events.Physics.raycast_screen",
                        lambda *args, **kwargs: next(hits))
    monkeypatch.setattr("Infernux.engine.runtime_mouse_events.Input.get_game_mouse_frame_state",
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
    monkeypatch.setattr("Infernux.engine.runtime_mouse_events.Physics.raycast_screen",
                        lambda *args, **kwargs: calls.append(True))
    dispatcher.process(object(), (1, 2), (100, 100), hit=hit)
    assert calls == []


def test_direct_mouse_query_respects_camera_mask_and_ignore_raycast(monkeypatch):
    calls = []
    monkeypatch.setattr("Infernux.engine.runtime_mouse_events.Physics.raycast_screen",
                        lambda *args, **kwargs: calls.append(kwargs))
    MouseEventDispatcher().process(SimpleNamespace(culling_mask=5), (1, 2), (100, 100),
                                   button_state=(False, False, False))
    assert calls == [dict(layer_mask=1, query_triggers=True)]
