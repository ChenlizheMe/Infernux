"""Unity-style 3D mouse event dispatch shared by Game View and Player."""

from __future__ import annotations

from Infernux.physics import Physics


_UNSET_HIT = object()


class MouseEventDispatcher:
    """Translate one camera ray into MonoBehaviour-style component callbacks."""

    def __init__(self) -> None:
        self._hover_object = None
        self._pressed_object = None

    @staticmethod
    def _same_object(a, b) -> bool:
        if a is b:
            return True
        return a is not None and b is not None and getattr(a, "id", None) == getattr(b, "id", None)

    @staticmethod
    def _components(game_object):
        if game_object is None or not hasattr(game_object, "get_py_components"):
            return ()
        return tuple(game_object.get_py_components() or ())

    @classmethod
    def _call(cls, game_object, name: str, hit=None) -> None:
        for component in cls._components(game_object):
            callback = getattr(component, name, None)
            if callable(callback):
                callback()

    def process(self, camera, screen_position, viewport_size, *, button: int = 0,
                enabled: bool = True, hit=_UNSET_HIT) -> None:
        """Process one mouse frame using top-left viewport pixel coordinates."""
        if not enabled or camera is None or viewport_size[0] <= 0 or viewport_size[1] <= 0:
            self.reset()
            return
        if hit is _UNSET_HIT:
            hit = Physics.raycast_screen(camera, screen_position, viewport_size,
                                         query_triggers=True)
        target = getattr(hit, "game_object", None) if hit is not None else None
        if not self._same_object(target, self._hover_object):
            if self._hover_object is not None:
                self._call(self._hover_object, "on_mouse_exit")
            self._hover_object = target
            if target is not None:
                self._call(target, "on_mouse_enter")
        if target is not None:
            self._call(target, "on_mouse_over")
        if Input.get_mouse_button_down(button):
            if target is not None:
                self._pressed_object = target
                self._call(target, "on_mouse_down")
        if self._pressed_object is not None and Input.get_mouse_button(button):
            self._call(self._pressed_object, "on_mouse_drag")
        if Input.get_mouse_button_up(button):
            pressed = self._pressed_object
            if pressed is not None:
                self._call(pressed, "on_mouse_up")
                if self._same_object(pressed, target):
                    self._call(pressed, "on_mouse_up_as_button")
            self._pressed_object = None

    def reset(self) -> None:
        if self._hover_object is not None:
            self._call(self._hover_object, "on_mouse_exit")
        self._hover_object = None
        self._pressed_object = None


from Infernux.input import Input
