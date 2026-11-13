"""Per-World pointer state around the native logical Input phase.

Only the native WorldFrame invokes this phase. Editor presentation continues to
use the view's Host state, and retired worlds discard captures without callbacks.
"""
from contextlib import contextmanager
from infernux.lib import InputManager

_MISSING = object()


def discard_retired_pointer_states(view):
    """Host-safe cleanup: retired targets get no authored callback."""
    states = getattr(view, "_world_pointer_states", {})
    for identity in tuple(states):
        owner, state = states[identity]
        if owner.retired:
            state["_ui_event_processor"].discard()
            state["_mouse_event_dispatcher"].discard()
            del states[identity]


@contextmanager
def world_pointer_state(view, factories):
    native = InputManager.instance()
    world = int(native._read_world_id)
    if not world:
        raise RuntimeError("Gameplay pointer dispatch requires a World input scope")
    states = view.__dict__.setdefault("_world_pointer_states", {})
    discard_retired_pointer_states(view)
    if getattr(view, "_active_pointer_world", None) == world:
        yield
        return
    entry = states.get(world)
    if entry is None:
        entry = (native._read_owner, {name: factory() for name, factory in factories.items()})
        states[world] = entry
    _owner, state = entry
    previous = {name: getattr(view, name, _MISSING) for name in factories}
    previous_world = getattr(view, "_active_pointer_world", None)
    try:
        view._active_pointer_world = world
        for name, value in state.items():
            setattr(view, name, value)
        from infernux.ui.ui_event_system import logical_ui_callback_batch
        with logical_ui_callback_batch():
            yield
    finally:
        view._active_pointer_world = previous_world
        for name in factories:
            state[name] = getattr(view, name)
        for name, value in previous.items():
            if value is _MISSING:
                view.__dict__.pop(name, None)
            else:
                setattr(view, name, value)
