"""Per-World pointer state around the native logical Input phase.

Only the native WorldFrame invokes this phase. Editor presentation continues to
use the view's Host state, and retired worlds discard captures without callbacks.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

from infernux.engine.runtime_mouse_events import MouseEventDispatcher
from infernux.lib import InputManager
from infernux.ui.ui_event_system import UIEventProcessor, logical_ui_callback_batch


@dataclass(slots=True)
class RuntimePointerState:
    ui: UIEventProcessor = field(default_factory=UIEventProcessor)
    mouse: MouseEventDispatcher = field(default_factory=MouseEventDispatcher)
    scene_token: tuple[int, int] | None = None
    frame: int = -1

    def discard(self) -> None:
        self.ui.discard()
        self.mouse.discard()

    def synchronize_scene(self, scene) -> bool:
        token = (int(scene.world_id), int(scene.temporal_discontinuity_revision)) if scene is not None else None
        if token == self.scene_token:
            return False
        self.discard()
        self.scene_token = token
        return True


def discard_world_pointer_state(view, identity):
    """Release one retired World's captures without calling authored targets."""
    state = view._world_pointer_states.pop(identity, None)
    if state is not None:
        state.discard()


@contextmanager
def world_pointer_state(view):
    native = InputManager.instance()
    world = int(native._read_world_id)
    if not world:
        raise RuntimeError("Gameplay pointer dispatch requires a World input scope")
    previous = view._pointer_state
    previous_world = view._active_pointer_world
    if previous_world == world:
        yield previous
        return
    states = view._world_pointer_states
    state = states.get(world)
    if state is None:
        state = states[world] = type(previous)()
    try:
        view._active_pointer_world = world
        view._pointer_state = state
        with logical_ui_callback_batch(native, world):
            yield state
    finally:
        view._pointer_state = previous
        view._active_pointer_world = previous_world
