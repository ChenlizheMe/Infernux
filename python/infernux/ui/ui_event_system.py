"""Per-frame mouse and multi-touch dispatch for runtime screen UI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

from infernux.engine.runtime_dispatch import (
    current_runtime_epoch,
)

from .ui_event_data import PointerButton, PointerEventData, PointerType

if TYPE_CHECKING:
    from .inx_ui_screen_component import InxUIScreenComponent
    from .ui_canvas import UICanvas


_DRAG_THRESHOLD = 5.0
_DOUBLE_CLICK_TIME = 0.3
_MOUSE_POINTER_ID = -1


def _canvas_raycast(canvas, canvas_x: float, canvas_y: float):
    """Raycast through the current authoritative Canvas contract."""
    raycast = getattr(canvas, "raycast", None)
    if not callable(raycast):
        raise RuntimeError(
            "The active UICanvas does not provide raycast(); reload the scene "
            "with the current Infernux runtime"
        )
    return raycast(canvas_x, canvas_y)


@dataclass(frozen=True, slots=True)
class UIPointerFrame:
    """One physical pointer snapshot expressed in every canvas's coordinates."""

    pointer_id: int
    pointer_type: PointerType
    canvas_positions: Tuple[Tuple[float, float], ...]
    down: bool = False
    up: bool = False
    held: bool = False
    canceled: bool = False
    scroll_delta: Tuple[float, float] = (0.0, 0.0)
    press_canvas_positions: Tuple[Tuple[float, float], ...] = ()


@dataclass(slots=True)
class _PointerState:
    pointer_type: PointerType
    hover_target: Optional[InxUIScreenComponent] = None
    hover_canvas: Optional[UICanvas] = None
    press_target: Optional[InxUIScreenComponent] = None
    press_canvas: Optional[UICanvas] = None
    press_position: Tuple[float, float] = (0.0, 0.0)
    drag_target: Optional[InxUIScreenComponent] = None
    is_dragging: bool = False
    last_canvas_positions: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    last_click_time: float = 0.0
    click_count: int = 0


class UIEventProcessor:
    """Dispatch independent mouse and touch transactions to screen UI elements."""

    def __init__(self):
        self._pointers: dict[tuple[PointerType, int], _PointerState] = {}
        self._time = 0.0
        self._last_pointer_debug: dict = {}
        self._queued_callback_count = 0
        self._executed_callback_count = 0
        self._dispatch_generation = 0

    def process(
        self,
        canvases: List[UICanvas],
        canvas_positions: List[Tuple[float, float]],
        mouse_down: bool,
        mouse_up: bool,
        mouse_held: bool,
        scroll_delta: Tuple[float, float],
        dt: float,
    ) -> None:
        """Dispatch one mouse frame from the Editor Game View."""

        self.process_pointers(
            canvases,
            (
                UIPointerFrame(
                    pointer_id=_MOUSE_POINTER_ID,
                    pointer_type=PointerType.Mouse,
                    canvas_positions=tuple(canvas_positions),
                    down=bool(mouse_down),
                    up=bool(mouse_up),
                    held=bool(mouse_held),
                    scroll_delta=scroll_delta,
                ),
            ),
            dt,
        )

    def process_pointers(
        self,
        canvases: Sequence[UICanvas],
        pointers: Sequence[UIPointerFrame],
        dt: float,
    ) -> None:
        """Dispatch one complete physical pointer snapshot."""

        self._time += float(dt)
        seen: set[tuple[PointerType, int]] = set()
        epoch = current_runtime_epoch()
        for pointer in pointers:
            pointer_id = int(pointer.pointer_id)
            pointer_key = (pointer.pointer_type, pointer_id)
            if pointer_key in seen:
                raise ValueError(
                    f"Duplicate UI {pointer.pointer_type.name} pointer id {pointer_id}"
                )
            if len(pointer.canvas_positions) != len(canvases):
                raise ValueError(
                    "UI pointer canvas position count does not match canvas count"
                )
            if pointer.press_canvas_positions and len(pointer.press_canvas_positions) != len(canvases):
                raise ValueError(
                    "UI pointer press position count does not match canvas count"
                )
            seen.add(pointer_key)
            self._process_pointer(canvases, pointer, epoch)

        missing_touches = tuple(
            pointer_key
            for pointer_key, state in self._pointers.items()
            if state.pointer_type is PointerType.Touch and pointer_key not in seen
        )
        for pointer_key in missing_touches:
            self._cancel_pointer(pointer_key, epoch)

    def _process_pointer(self, canvases, pointer: UIPointerFrame, epoch) -> None:
        if pointer.down and pointer.up:
            # A short touch can begin and end between two game frames. Reuse
            # the ordinary press/release path with its original press location.
            self._process_pointer(
                canvases,
                replace(
                    pointer,
                    canvas_positions=pointer.press_canvas_positions or pointer.canvas_positions,
                    up=False,
                    held=False,
                    canceled=False,
                    press_canvas_positions=(),
                ),
                epoch,
            )
            pointer = replace(pointer, down=False, press_canvas_positions=())
        pointer_id = int(pointer.pointer_id)
        pointer_key = (pointer.pointer_type, pointer_id)
        state = self._pointers.get(pointer_key)
        if state is None:
            state = _PointerState(pointer_type=pointer.pointer_type)
            self._pointers[pointer_key] = state

        captured = state.press_target
        if captured is not None:
            accepts = getattr(captured, "is_effectively_interactable", None)
            if callable(accepts) and not accepts():
                current = self._xy(pointer.canvas_positions[0]) if pointer.canvas_positions else (0.0, 0.0)
                canceled_pointer = UIPointerFrame(
                    pointer_id=pointer_id,
                    pointer_type=pointer.pointer_type,
                    canvas_positions=pointer.canvas_positions,
                    up=True,
                    canceled=True,
                )
                self._release_pointer(
                    canceled_pointer, state, current, (0.0, 0.0),
                    None, None, current, epoch,
                )

        current = self._xy(pointer.canvas_positions[0]) if pointer.canvas_positions else (0.0, 0.0)
        previous = self._xy(state.last_canvas_positions[0]) if state.last_canvas_positions else current
        delta = (current[0] - previous[0], current[1] - previous[1])
        # A stationary pointer still observes moving/reparented/disabled UI.
        # Cache resolved hit geometry, not the result of a previous pointer
        # frame: native Transform changes and layout updates can change hover.

        hit_element = None
        hit_canvas = None
        hit_position = current
        def surface_priority(index):
            getter = getattr(canvases[index], "input_priority", None)
            return getter(pointer.canvas_positions[index], index) if callable(getter) else (1, index, 0.0)

        # Priority depends only on the surface and projected depth, not on the
        # hit. Visit front-to-back so overlapping world quads don't all walk
        # their component/group hierarchy after the winning target is known.
        # Python's stable sort preserves the original winner for equal depths.
        order = (range(len(canvases)) if len(canvases) < 2 else
                 sorted(range(len(canvases)), key=surface_priority, reverse=True))
        for index in order:
            canvas = canvases[index]
            position = pointer.canvas_positions[index]
            candidate = _canvas_raycast(canvas, position[0], position[1])
            if candidate is not None:
                # Broad-phase misses do not need another component/owner walk.
                # Surface eligibility still gates every actual event target.
                canvas_object = canvas.game_object
                if canvas_object is not None and not canvas_object.active_in_hierarchy:
                    continue
                if not getattr(canvas, "enabled", True):
                    continue
                hit_element = candidate
                hit_canvas = canvas
                hit_position = self._xy(position)
                break

        if hit_element is not None:
            accepts = getattr(hit_element, "is_effectively_interactable", None)
            if callable(accepts) and not accepts():
                hit_element = None

        if pointer.down or pointer.up or pointer.canceled:
            hit_object = getattr(hit_element, "game_object", None) if hit_element is not None else None
            click_event = getattr(hit_element, "on_click", None) if hit_element is not None else None
            self._last_pointer_debug = {
                "pointer_id": pointer_id,
                "pointer_type": pointer.pointer_type.name,
                "canvas_count": len(canvases),
                "canvas_position": [float(hit_position[0]), float(hit_position[1])],
                "down": bool(pointer.down),
                "up": bool(pointer.up),
                "held": bool(pointer.held),
                "canceled": bool(pointer.canceled),
                "hit_type": type(hit_element).__name__ if hit_element is not None else "",
                "hit_object": str(getattr(hit_object, "name", "") or ""),
                "hit_component_id": int(getattr(hit_element, "_component_id", 0) or 0),
                "hit_is_valid": bool(getattr(hit_element, "is_valid", False)),
                "hit_listener_count": int(getattr(click_event, "listener_count", 0) or 0),
                "press_type": type(state.press_target).__name__ if state.press_target is not None else "",
                "press_object": str(
                    getattr(getattr(state.press_target, "game_object", None), "name", "") or ""
                ),
            }

        previous_hover = state.hover_target
        if hit_element is not previous_hover:
            if previous_hover is not None:
                event = self._make_event(
                    pointer, hit_position, delta, state.hover_canvas, previous_hover
                )
                self._dispatch_pointer_callback(previous_hover, "on_pointer_exit", event, epoch)
            state.hover_target = hit_element
            state.hover_canvas = hit_canvas
            if hit_element is not None:
                event = self._make_event(pointer, hit_position, delta, hit_canvas, hit_element)
                self._dispatch_pointer_callback(hit_element, "on_pointer_enter", event, epoch)

        if pointer.down and hit_element is not None:
            state.press_target = hit_element
            state.press_canvas = hit_canvas
            state.press_position = hit_position
            state.drag_target = hit_element
            state.is_dragging = False
            event = self._make_event(pointer, hit_position, delta, hit_canvas, hit_element)
            event.press_position = state.press_position
            self._dispatch_pointer_callback(hit_element, "on_pointer_down", event, epoch)

        if pointer.held and state.drag_target is not None:
            press_current, press_previous = self._surface_positions(
                canvases,
                pointer.canvas_positions,
                state.last_canvas_positions,
                state.press_canvas,
                current,
            )
            press_delta = (
                press_current[0] - press_previous[0],
                press_current[1] - press_previous[1],
            )
            dx = press_current[0] - state.press_position[0]
            dy = press_current[1] - state.press_position[1]
            distance_squared = dx * dx + dy * dy
            event = self._make_event(
                pointer, press_current, press_delta, state.press_canvas, state.drag_target
            )
            event.press_position = state.press_position
            if not state.is_dragging and distance_squared > _DRAG_THRESHOLD * _DRAG_THRESHOLD:
                state.is_dragging = True
                self._dispatch_pointer_callback(state.drag_target, "on_begin_drag", event, epoch)
            elif state.is_dragging:
                self._dispatch_pointer_callback(state.drag_target, "on_drag", event, epoch)

        if pointer.scroll_delta != (0.0, 0.0) and hit_element is not None:
            event = self._make_event(
                pointer, hit_position, delta, hit_canvas, hit_element
            )
            self._dispatch_pointer_callback(hit_element, "on_scroll", event, epoch)

        if pointer.up or pointer.canceled:
            release_current, release_previous = self._surface_positions(
                canvases,
                pointer.canvas_positions,
                state.last_canvas_positions,
                state.press_canvas,
                current,
            )
            release_delta = (
                release_current[0] - release_previous[0],
                release_current[1] - release_previous[1],
            )
            self._release_pointer(
                pointer,
                state,
                release_current,
                release_delta,
                hit_element,
                hit_canvas,
                hit_position,
                epoch,
            )

        state.last_canvas_positions = pointer.canvas_positions
        if pointer.pointer_type is PointerType.Touch and (pointer.up or pointer.canceled):
            if state.hover_target is not None:
                event = self._make_event(
                    pointer, hit_position, delta, state.hover_canvas, state.hover_target
                )
                self._dispatch_pointer_callback(state.hover_target, "on_pointer_exit", event, epoch)
            self._pointers.pop(pointer_key, None)

    @staticmethod
    def _surface_positions(canvases, current_positions, previous_positions, surface, default):
        for index, candidate in enumerate(canvases):
            if candidate is surface:
                current = UIEventProcessor._xy(current_positions[index])
                previous = (
                    UIEventProcessor._xy(previous_positions[index])
                    if index < len(previous_positions)
                    else current
                )
                return current, previous
        return default, default

    @staticmethod
    def _xy(position):
        return float(position[0]), float(position[1])

    def _release_pointer(
        self,
        pointer,
        state,
        current,
        delta,
        hit_element,
        hit_canvas,
        hit_position,
        epoch,
    ) -> None:
        press_target = state.press_target
        press_canvas = state.press_canvas
        drag_target = state.drag_target if state.is_dragging else None
        state.press_target = None
        state.press_canvas = None
        state.drag_target = None
        state.is_dragging = False
        if press_target is not None:
            event = self._make_event(
                pointer,
                hit_position if hit_element is not None else current,
                delta,
                press_canvas,
                press_target,
            )
            event.press_position = state.press_position
            self._dispatch_pointer_callback(press_target, "on_pointer_up", event, epoch)

            if not pointer.canceled and hit_element is press_target:
                if (self._time - state.last_click_time) < _DOUBLE_CLICK_TIME:
                    state.click_count += 1
                else:
                    state.click_count = 1
                state.last_click_time = self._time
                click = self._make_event(pointer, hit_position, delta, hit_canvas, hit_element)
                click.press_position = state.press_position
                click.click_count = state.click_count
                self._dispatch_pointer_callback(hit_element, "on_pointer_click", click, epoch)
                debug_dispatch = getattr(hit_element, "debug_dispatch_state", None)
                if callable(debug_dispatch):
                    self._last_pointer_debug["persistent_dispatch"] = debug_dispatch()

        if drag_target is not None:
            event = self._make_event(
                pointer, current, delta, press_canvas, drag_target
            )
            event.press_position = state.press_position
            self._dispatch_pointer_callback(drag_target, "on_end_drag", event, epoch)

    def _cancel_pointer(
        self, pointer_key: tuple[PointerType, int], epoch
    ) -> None:
        state = self._pointers.pop(pointer_key)
        self._cancel_pointer_state(pointer_key, state, epoch)

    def _cancel_pointer_state(self, pointer_key, state, epoch) -> None:
        pointer_type, pointer_id = pointer_key
        positions = state.last_canvas_positions
        current = self._xy(positions[0]) if positions else (0.0, 0.0)
        pointer = UIPointerFrame(
            pointer_id=pointer_id,
            pointer_type=pointer_type,
            canvas_positions=positions,
            up=True,
            canceled=True,
        )
        self._release_pointer(
            pointer, state, current, (0.0, 0.0), None, None, current, epoch
        )
        if state.hover_target is not None:
            event = self._make_event(
                pointer, current, (0.0, 0.0), state.hover_canvas, state.hover_target
            )
            self._dispatch_pointer_callback(state.hover_target, "on_pointer_exit", event, epoch)

    def reset(self) -> None:
        """Cancel every active pointer transaction."""

        if not self._pointers:
            return
        epoch = current_runtime_epoch()
        pointers, self._pointers = self._pointers, {}
        for pointer_key, state in pointers.items():
            self._cancel_pointer_state(pointer_key, state, epoch)

    def discard(self) -> None:
        """Forget pointer state whose owning scene has already retired.

        Scene replacement destroys the old component graph before the next GUI
        frame.  Cancellation callbacks are meaningful while that graph is
        alive; after publication they would dereference retired targets.
        """

        self._pointers.clear()
        self._dispatch_generation += 1
        self._last_pointer_debug = {}

    def debug_state(self) -> dict:
        """Return the latest transition without polling input each frame."""

        return dict(self._last_pointer_debug)

    def _dispatch_pointer_callback(self, target, method_name, event, epoch) -> None:
        """Queue one callback outside the native GUI draw traversal.

        Pointer callbacks are allowed to replace scenes and destroy UI objects.
        Running them synchronously from ``PyGUIRenderable::OnRender`` lets that
        mutation invalidate the C++ renderable traversal on Android and other
        backends.  Resolve the runtime method again at the owner safe point so
        a scene retired before the drain cannot call stale Python objects.
        """
        if getattr(target, "_is_destroyed", False):
            return

        descriptor = epoch.require_descriptor(type(target))
        method = descriptor.methods.get(method_name)
        if method is None:
            return
        callback = method.bind(target)

        from infernux.engine.runtime_event_queue import enqueue

        self._queued_callback_count += 1
        dispatch_generation = self._dispatch_generation
        self._last_pointer_debug.update({
            "last_callback": method_name,
            "last_callback_status": "queued",
            "queued_callback_count": self._queued_callback_count,
            "executed_callback_count": self._executed_callback_count,
        })

        def invoke_later(target=target, callback=callback, event=event, method_name=method_name):
            if dispatch_generation != self._dispatch_generation:
                return
            if getattr(target, "_is_destroyed", False):
                self._last_pointer_debug.update({
                    "last_callback": method_name,
                    "last_callback_status": "target_destroyed",
                })
                return
            if not bool(getattr(target, "is_valid", False)):
                self._last_pointer_debug.update({
                    "last_callback": method_name,
                    "last_callback_status": "target_invalid",
                })
                return
            try:
                callback(event)
            except BaseException as exc:
                self._last_pointer_debug.update({
                    "last_callback": method_name,
                    "last_callback_status": "exception",
                    "last_callback_error": f"{type(exc).__name__}: {exc}",
                    "queued_callback_count": self._queued_callback_count,
                    "executed_callback_count": self._executed_callback_count,
                })
                raise
            debug_dispatch = getattr(target, "debug_dispatch_state", None)
            if callable(debug_dispatch):
                self._last_pointer_debug["persistent_dispatch"] = debug_dispatch()
            self._executed_callback_count += 1
            self._last_pointer_debug.update({
                "last_callback": method_name,
                "last_callback_status": "executed",
                "queued_callback_count": self._queued_callback_count,
                "executed_callback_count": self._executed_callback_count,
            })

        enqueue(invoke_later)

    @staticmethod
    def _make_event(
        pointer: UIPointerFrame,
        position: Tuple[float, float],
        delta: Tuple[float, float],
        canvas: Optional[UICanvas],
        target: Optional[InxUIScreenComponent],
    ) -> PointerEventData:
        event = PointerEventData()
        event.position = position
        event.delta = delta
        event.canvas_size = (
            canvas.input_logical_size if canvas is not None else (0.0, 0.0)
        )
        event.pointer_id = int(pointer.pointer_id)
        event.pointer_type = pointer.pointer_type
        event.canceled = bool(pointer.canceled)
        event.button = PointerButton.Left
        event.scroll_delta = pointer.scroll_delta
        event.canvas = canvas
        event.target = target
        return event
