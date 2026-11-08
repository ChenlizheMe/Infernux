"""Unity-style 3D mouse event dispatch shared by Game View and Player."""

from __future__ import annotations

from weakref import ref

from infernux.physics import Physics
from infernux.input import Input
from infernux.engine.runtime_dispatch import current_runtime_epoch


_UNSET_HIT = object()
_MOUSE_CALLBACKS = (
    "on_mouse_enter", "on_mouse_over", "on_mouse_exit", "on_mouse_down",
    "on_mouse_drag", "on_mouse_up", "on_mouse_up_as_button",
)


class MouseEventDispatcher:
    """Translate one camera ray into MonoBehaviour-style component callbacks."""

    def __init__(self) -> None:
        self._hover_object = None
        self._pressed_object = None
        self._reset_revision = 0
        self._receiver_epoch = None
        self._has_receivers = False

    def needs_scene_hit(self) -> bool:
        """Skip the physics pick when the published scripts have no mouse hooks."""
        epoch = current_runtime_epoch()
        if self._receiver_epoch is None or self._receiver_epoch() is not epoch:
            self._receiver_epoch = ref(epoch)
            self._has_receivers = any(
                (method := descriptor.methods.get(name)) is not None
                and not getattr(method.raw, "__infernux_default_lifecycle__", False)
                for descriptor in epoch.descriptors.values()
                for name in _MOUSE_CALLBACKS
            )
        return self._has_receivers

    @staticmethod
    def _same_object(a, b) -> bool:
        if a is b:
            return True
        if a is None or b is None:
            return False
        identity = getattr(a, "id", None)
        return identity is not None and identity == getattr(b, "id", None)

    @staticmethod
    def _components(game_object):
        if game_object is None or not hasattr(game_object, "get_py_components"):
            return ()
        return tuple(game_object.get_py_components() or ())

    def _call(self, components, name: str, epoch, revision) -> None:
        for component in components:
            if revision != self._reset_revision:
                break
            if getattr(component, "_is_destroyed", False):
                continue
            method = epoch.require_descriptor(type(component)).methods.get(name)
            if method is not None:
                method.invoke(component)

    def process(self, camera, screen_position, viewport_size, *, button: int = 0,
                enabled: bool = True, hit=_UNSET_HIT, button_state=None,
                canceled: bool = False) -> None:
        """Dispatch a top-left viewport sample through one published epoch.

        Callers sharing UI input can pass its hit and button snapshot. Object
        membership is sampled once per owner this frame; author edits become
        callable only through the normal runtime publication boundary.
        """
        if not enabled or camera is None or viewport_size[0] <= 0 or viewport_size[1] <= 0:
            self.reset()
            return
        if not (0 <= screen_position[0] < viewport_size[0] and 0 <= screen_position[1] < viewport_size[1]):
            hit = None
        elif hit is _UNSET_HIT:
            hit = Physics.raycast_screen(camera, screen_position, viewport_size,
                                         layer_mask=int(camera.culling_mask) & ~(1 << 2),
                                         query_triggers=True)
        target = getattr(hit, "game_object", None) if hit is not None else None
        epoch = current_runtime_epoch()
        revision = self._reset_revision
        held, down, up = Input.get_game_mouse_frame_state(button)[4:] if button_state is None else button_state
        previous_hover = self._hover_object
        changed = not self._same_object(target, previous_hover)
        # Settle capture before author callbacks. A failed release or exit
        # must not leave an old transaction to replay on the next frame.
        self._hover_object = target
        if down:
            self._pressed_object = target
        pressed = self._pressed_object
        if up:
            self._pressed_object = None
        snapshots = {}

        def call(owner, name):
            if owner is None or revision != self._reset_revision:
                return
            key = id(owner)
            if key not in snapshots:
                snapshots[key] = self._components(owner)
            self._call(snapshots[key], name, epoch, revision)

        if changed:
            call(previous_hover, "on_mouse_exit")
            if target is not None:
                call(target, "on_mouse_enter")
        if target is not None:
            call(target, "on_mouse_over")
        if down and target is not None:
            call(target, "on_mouse_down")
        if pressed is not None and held:
            call(pressed, "on_mouse_drag")
        if up:
            if pressed is not None:
                call(pressed, "on_mouse_up")
                if not canceled and self._same_object(pressed, target):
                    call(pressed, "on_mouse_up_as_button")

    def reset(self) -> None:
        previous = self._hover_object
        self._hover_object = None
        self._pressed_object = None
        self._reset_revision += 1
        if previous is not None:
            self._call(self._components(previous), "on_mouse_exit", current_runtime_epoch(), self._reset_revision)

    def discard(self) -> None:
        """Forget captures after their scene graph has been retired."""

        self._hover_object = None
        self._pressed_object = None
        self._reset_revision += 1
