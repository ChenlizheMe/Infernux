"""Per-World pointer state around the native logical Input phase.

Only the native WorldFrame invokes this phase. Editor presentation continues to
use the view's Host state, and retired worlds discard captures without callbacks.
"""
from contextlib import contextmanager
from infernux.lib import InputManager
from infernux.ui.ui_event_system import logical_ui_callback_batch

_MISSING = object()


def discard_world_pointer_state(view, identity):
    """Release one retired World's captures without calling authored targets."""
    states = vars(view).get("_world_pointer_states")
    state = states.pop(identity, None) if states else None
    if state is not None:
        state["_ui_event_processor"].discard()
        state["_mouse_event_dispatcher"].discard()


@contextmanager
def world_pointer_state(view, factories):
    native = InputManager.instance()
    world = int(native._read_world_id)
    if not world:
        raise RuntimeError("Gameplay pointer dispatch requires a World input scope")
    attributes = vars(view)
    states = attributes.setdefault("_world_pointer_states", {})
    previous_world = attributes.get("_active_pointer_world")
    if previous_world == world:
        yield
        return
    state = states.get(world)
    if state is None:
        state = {name: factory() for name, factory in factories.items()}
        states[world] = state
    previous = {name: attributes.get(name, _MISSING) for name in factories}
    try:
        attributes["_active_pointer_world"] = world
        attributes.update(state)
        with logical_ui_callback_batch():
            yield
    finally:
        attributes["_active_pointer_world"] = previous_world
        for name in factories:
            state[name] = attributes[name]
        for name, value in previous.items():
            if value is _MISSING:
                attributes.pop(name, None)
            else:
                attributes[name] = value
