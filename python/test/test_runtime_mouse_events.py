from types import SimpleNamespace

from Infernux.engine.runtime_mouse_events import MouseEventDispatcher


def test_mouse_dispatcher_matches_unity_enter_over_down_drag_up_button_exit(monkeypatch):
    events = []

    class Component:
        def __getattribute__(self, name):
            if name.startswith("on_mouse"):
                return lambda _name=name: events.append(_name)
            return object.__getattribute__(self, name)

    first = SimpleNamespace(id=1, get_py_components=lambda: (Component(),))
    second = SimpleNamespace(id=2, get_py_components=lambda: (Component(),))
    hits = iter([
        SimpleNamespace(game_object=first),
        SimpleNamespace(game_object=first),
        SimpleNamespace(game_object=first),
        SimpleNamespace(game_object=first),
        None,
    ])
    states = iter([(False, False, False), (True, True, False),
                   (False, True, True), (False, False, True),
                   (False, False, False)])
    monkeypatch.setattr(
        "Infernux.engine.runtime_mouse_events.Physics.raycast_screen",
        lambda *args, **kwargs: next(hits),
    )
    monkeypatch.setattr(
        "Infernux.engine.runtime_mouse_events.Input.get_mouse_button_down",
        lambda _button: next(states)[0],
    )
    # Drive the three button-state queries from one stable frame tuple.
    frame = {"value": (False, False, False)}
    monkeypatch.setattr(
        "Infernux.engine.runtime_mouse_events.Input.get_mouse_button",
        lambda _button: frame["value"][1],
    )
    monkeypatch.setattr(
        "Infernux.engine.runtime_mouse_events.Input.get_mouse_button_up",
        lambda _button: frame["value"][2],
    )
    dispatcher = MouseEventDispatcher()
    dispatcher.process(object(), (1, 1), (10, 10))
    frame["value"] = (True, True, False)
    dispatcher.process(object(), (1, 1), (10, 10))
    frame["value"] = (False, True, False)
    dispatcher.process(object(), (1, 1), (10, 10))
    frame["value"] = (False, False, True)
    dispatcher.process(object(), (1, 1), (10, 10))
    frame["value"] = (False, False, False)
    dispatcher.process(object(), (1, 1), (10, 10))
    names = list(events)
    assert names == ["on_mouse_enter", "on_mouse_over", "on_mouse_over", "on_mouse_down",
                     "on_mouse_drag", "on_mouse_over", "on_mouse_drag", "on_mouse_over",
                     "on_mouse_up", "on_mouse_up_as_button", "on_mouse_exit"]
