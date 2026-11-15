"""Synthetic editor input operations backed by the Host capability API."""

from __future__ import annotations

import math
import time

from infernux.host import EditorAutomationHost, Operation, OperationError, OperationKind

from infernux.host.operation_support import on_editor, operation


def build_input_operations() -> tuple[Operation, ...]:
    common = {
        "wait_for_delivery": {"type": "boolean", "default": True},
        "timeout_seconds": {"type": "number", "default": 3.0},
    }
    return (
        operation(
            "infernux.input.status",
            OperationKind.QUERY,
            "Read synthetic editor-input queue delivery status.",
            _status,
            capability="input.read",
            tags=("input", "editor", "status", "validation"),
        ),
        operation(
            "infernux.input.key",
            OperationKind.COMMAND,
            "Queue one keyboard transition through the engine event path.",
            _key,
            capability="input.write",
            input_properties={
                "key": {"type": ["string", "integer"]},
                "pressed": {"type": "boolean"},
                "repeat": {"type": "boolean", "default": False},
                **common,
            },
            required=("key", "pressed"),
            side_effects=("Injects one trusted SDL keyboard event into the Editor.",),
            tags=("input", "keyboard", "editor", "validation"),
        ),
        operation(
            "infernux.input.key.chord",
            OperationKind.WORKFLOW,
            "Press a key chord in order and release it in reverse order.",
            _key_chord,
            capability="input.write",
            input_properties={
                "keys": {"type": "array", "items": {"type": ["string", "integer"]}},
                "timeout_seconds": {"type": "number", "default": 3.0},
            },
            required=("keys",),
            side_effects=("Injects ordered SDL keyboard transitions into the Editor.",),
            tags=("input", "keyboard", "chord", "editor", "validation"),
        ),
        operation(
            "infernux.input.key.hold",
            OperationKind.WORKFLOW,
            "Hold one key through the engine input queue for an explicit duration.",
            _key_hold,
            capability="input.write",
            input_properties={
                "key": {"type": ["string", "integer"]},
                "duration_seconds": {"type": "number", "default": 0.25},
                "repeat": {"type": "boolean", "default": False},
                "timeout_seconds": {"type": "number", "default": 3.0},
            },
            required=("key",),
            side_effects=(
                "Queues a key-down transition, preserves it for the requested engine time window, then queues key-up.",
            ),
            tags=("input", "keyboard", "hold", "gameplay", "editor", "validation"),
        ),
        operation(
            "infernux.input.pointer.move",
            OperationKind.COMMAND,
            "Move the editor pointer in window coordinates.",
            _pointer_move,
            capability="input.write",
            input_properties={
                "x": {"type": "number"},
                "y": {"type": "number"},
                "delta_x": {"type": "number", "default": 0.0},
                "delta_y": {"type": "number", "default": 0.0},
                **common,
            },
            required=("x", "y"),
            side_effects=("Injects one SDL pointer-motion event into the Editor.",),
            tags=("input", "pointer", "mouse", "editor"),
        ),
        operation(
            "infernux.input.pointer.button",
            OperationKind.COMMAND,
            "Queue one editor pointer-button transition.",
            _pointer_button,
            capability="input.write",
            input_properties={
                "button": {"type": "integer"},
                "pressed": {"type": "boolean"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                **common,
            },
            required=("button", "pressed", "x", "y"),
            side_effects=("Injects one SDL pointer-button event into the Editor.",),
            tags=("input", "pointer", "button", "editor"),
        ),
        operation(
            "infernux.input.pointer.click",
            OperationKind.WORKFLOW,
            "Move, press, and release the editor pointer at one window coordinate.",
            _pointer_click,
            capability="input.write",
            input_properties={
                "x": {"type": "number"},
                "y": {"type": "number"},
                "button": {"type": "integer", "default": 0},
                "timeout_seconds": {"type": "number", "default": 3.0},
            },
            required=("x", "y"),
            side_effects=("Injects pointer move, press, and release transitions into the Editor.",),
            tags=("input", "pointer", "click", "editor", "validation"),
        ),
        operation(
            "infernux.input.pointer.wheel",
            OperationKind.COMMAND,
            "Queue one editor pointer-wheel event.",
            _wheel,
            capability="input.write",
            input_properties={
                "horizontal": {"type": "number", "default": 0.0},
                "vertical": {"type": "number", "default": 0.0},
                **common,
            },
            side_effects=("Injects one SDL wheel event into the Editor.",),
            tags=("input", "pointer", "wheel", "editor"),
        ),
        operation(
            "infernux.input.text",
            OperationKind.COMMAND,
            "Send UTF-8 text to the Editor's logical input target.",
            _text,
            capability="input.write",
            input_properties={"text": {"type": "string"}, **common},
            required=("text",),
            side_effects=("Injects UTF-8 text into the Editor's logical focus target.",),
            tags=("input", "text", "editor", "validation"),
        ),
        operation(
            "infernux.input.window.close",
            OperationKind.COMMAND,
            "Request Editor window close through the normal input path.",
            _close,
            capability="input.write",
            input_properties=common,
            side_effects=("Requests the Editor's normal close lifecycle.",),
            tags=("input", "window", "close", "editor"),
        ),
        operation(
            "infernux.input.wait",
            OperationKind.QUERY,
            "Wait until one queued synthetic-input sequence reaches the Editor.",
            _wait,
            capability="input.read",
            input_properties={
                "sequence": {"type": "integer"},
                "timeout_seconds": {"type": "number", "default": 3.0},
            },
            required=("sequence",),
            tags=("input", "delivery", "wait", "editor", "validation"),
        ),
    )


def _status() -> dict[str, object]:
    return on_editor(
        "infernux.input.status", EditorAutomationHost.instance().input_status
    )


def _submit(
    kind: str,
    *,
    wait_for_delivery: bool = True,
    timeout_seconds: float = 3.0,
    **arguments: object,
) -> dict[str, object]:
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise OperationError(
            "operation.invalid_arguments", "timeout_seconds must be positive and finite"
        )
    result = on_editor(
        f"infernux.input.{kind}",
        lambda: EditorAutomationHost.instance().queue_input(kind, **arguments),
    )
    if not wait_for_delivery:
        return result
    sequence = int(result["sequence"])
    deadline = time.monotonic() + min(timeout, 30.0)
    while time.monotonic() < deadline:
        status = on_editor(
            "infernux.input.delivery",
            EditorAutomationHost.instance().input_status,
        )
        if int(status["last_processed_sequence"]) >= sequence:
            return {**result, **status, "delivered": True}
        time.sleep(0.01)
    raise OperationError(
        "input.timeout", f"Synthetic input sequence {sequence} was not delivered."
    )


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise OperationError("operation.invalid_arguments", f"{name} must be finite")
    return result


def _wait(sequence: int, timeout_seconds: float = 3.0):
    target = int(sequence)
    if target <= 0:
        raise OperationError("operation.invalid_arguments", "sequence must be positive")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise OperationError(
            "operation.invalid_arguments", "timeout_seconds must be positive and finite"
        )
    deadline = time.monotonic() + min(timeout, 30.0)
    while time.monotonic() < deadline:
        status = on_editor(
            "infernux.input.wait", EditorAutomationHost.instance().input_status
        )
        if int(status["last_processed_sequence"]) >= target:
            return {**status, "sequence": target, "delivered": True}
        time.sleep(0.01)
    raise OperationError(
        "input.timeout", f"Synthetic input sequence {target} was not delivered."
    )


def _key(key, pressed: bool, repeat: bool = False, wait_for_delivery: bool = True, timeout_seconds: float = 3.0):
    return _submit("key", key=key, pressed=pressed, repeat=repeat, wait_for_delivery=wait_for_delivery, timeout_seconds=timeout_seconds)


def _key_chord(keys: list, timeout_seconds: float = 3.0):
    values = list(keys or [])
    if not values or len(values) > 8:
        raise OperationError(
            "operation.invalid_arguments", "keys must contain between 1 and 8 entries"
        )
    pressed = []
    press_sequences = []
    release_sequences = []
    try:
        for key in values:
            result = _submit(
                "key", key=key, pressed=True, repeat=False,
                wait_for_delivery=True, timeout_seconds=timeout_seconds,
            )
            pressed.append(key)
            press_sequences.append(int(result["sequence"]))
    finally:
        for key in reversed(pressed):
            result = _submit(
                "key", key=key, pressed=False, repeat=False,
                wait_for_delivery=True, timeout_seconds=timeout_seconds,
            )
            release_sequences.append(int(result["sequence"]))
    return {
        "keys": values,
        "press_sequences": press_sequences,
        "release_sequences": release_sequences,
        "delivered": True,
    }


def _key_hold(
    key,
    duration_seconds: float = 0.25,
    repeat: bool = False,
    timeout_seconds: float = 3.0,
):
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise OperationError(
            "operation.invalid_arguments",
            "duration_seconds must be positive and finite",
        )
    pressed = _submit(
        "key",
        key=key,
        pressed=True,
        repeat=repeat,
        wait_for_delivery=True,
        timeout_seconds=timeout_seconds,
    )
    released = None
    try:
        # The Editor and simulation continue on their own threads while this
        # workflow retains the logical engine key state.
        time.sleep(duration)
    finally:
        released = _submit(
            "key",
            key=key,
            pressed=False,
            repeat=False,
            wait_for_delivery=True,
            timeout_seconds=timeout_seconds,
        )
    return {
        "key": key,
        "duration_seconds": duration,
        "press_sequence": int(pressed["sequence"]),
        "release_sequence": int(released["sequence"]),
        "delivered": True,
    }


def _pointer_move(x: float, y: float, delta_x: float = 0.0, delta_y: float = 0.0, wait_for_delivery: bool = True, timeout_seconds: float = 3.0):
    return _submit("pointer_move", x=_finite("x", x), y=_finite("y", y), delta_x=_finite("delta_x", delta_x), delta_y=_finite("delta_y", delta_y), wait_for_delivery=wait_for_delivery, timeout_seconds=timeout_seconds)


def _pointer_button(button: int, pressed: bool, x: float, y: float, wait_for_delivery: bool = True, timeout_seconds: float = 3.0):
    if isinstance(button, bool) or int(button) not in range(5):
        raise OperationError("operation.invalid_arguments", "button must be within [0, 4]")
    return _submit("pointer_button", button=button, pressed=pressed, x=_finite("x", x), y=_finite("y", y), wait_for_delivery=wait_for_delivery, timeout_seconds=timeout_seconds)


def _pointer_click(x: float, y: float, button: int = 0, timeout_seconds: float = 3.0):
    px = _finite("x", x)
    py = _finite("y", y)
    if isinstance(button, bool) or int(button) not in range(5):
        raise OperationError("operation.invalid_arguments", "button must be within [0, 4]")
    timeout = _finite("timeout_seconds", timeout_seconds)
    if timeout <= 0:
        raise OperationError("operation.invalid_arguments", "timeout_seconds must be positive and finite")

    def queue_click():
        host = EditorAutomationHost.instance()
        moved = host.queue_input("pointer_move", x=px, y=py, delta_x=0.0, delta_y=0.0)
        pressed = host.queue_input("pointer_button", button=button, pressed=True, x=px, y=py)
        released = host.queue_input("pointer_button", button=button, pressed=False, x=px, y=py)
        return moved, pressed, released

    # One owner-thread callback publishes the entire gesture. The native
    # drain can then enforce its hover -> press -> release GUI boundaries.
    # Waiting for individual event delivery split the gesture across batches
    # and allowed a press before ImGui had established the target hover.
    moved, pressed, released = on_editor("infernux.input.pointer.click.queue", queue_click)
    _wait(int(released["sequence"]), timeout_seconds=timeout)
    return {
        "x": px,
        "y": py,
        "button": int(button),
        "move_sequence": int(moved["sequence"]),
        "press_sequence": int(pressed["sequence"]),
        "release_sequence": int(released["sequence"]),
        "delivered": True,
    }


def _wheel(horizontal: float = 0.0, vertical: float = 0.0, wait_for_delivery: bool = True, timeout_seconds: float = 3.0):
    return _submit("wheel", horizontal=_finite("horizontal", horizontal), vertical=_finite("vertical", vertical), wait_for_delivery=wait_for_delivery, timeout_seconds=timeout_seconds)


def _text(text: str, wait_for_delivery: bool = True, timeout_seconds: float = 3.0):
    return _submit("text", text=text, wait_for_delivery=wait_for_delivery, timeout_seconds=timeout_seconds)


def _close(wait_for_delivery: bool = True, timeout_seconds: float = 3.0):
    return _submit("close", wait_for_delivery=wait_for_delivery, timeout_seconds=timeout_seconds)


__all__ = ["build_input_operations"]
