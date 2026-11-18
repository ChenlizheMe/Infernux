"""Engine render-target capture and GPU-pick operations."""

from __future__ import annotations

import math
import os
import re
import time

from infernux.host import EditorAutomationHost, Operation, OperationError, OperationKind

from infernux.host.context import automation_context
from infernux.host.operation_support import on_editor, operation


_TERMINAL = frozenset({"completed", "failed", "cancelled", "source_expired"})


def build_capture_operations() -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.capture.request",
            OperationKind.COMMAND,
            "Request an asynchronous Scene, Game, complete Editor, or Camera target_texture PNG.",
            _request_capture,
            capability="capture.write",
            input_properties={
                "source": {"type": "string", "enum": ["scene", "game", "editor", "camera"], "default": "game"},
                "file_name": {"type": "string", "default": ""},
                "camera_component_id": {"type": "integer", "minimum": 0, "default": 0},
            },
            side_effects=("Queues GPU readback and writes a session review artifact.",),
            tags=("capture", "render-target", "scene", "game", "editor", "camera", "png"),
        ),
        operation(
            "infernux.capture.status",
            OperationKind.QUERY,
            "Poll a render-target capture and return the PNG file path when available.",
            _capture_status,
            capability="capture.read",
            input_properties={"capture_id": {"type": "integer"}},
            required=("capture_id",),
            tags=("capture", "render-target", "status"),
        ),
        operation(
            "infernux.capture.cancel",
            OperationKind.COMMAND,
            "Cancel an unfinished engine render-target capture.",
            _capture_cancel,
            capability="capture.write",
            input_properties={"capture_id": {"type": "integer"}},
            required=("capture_id",),
            side_effects=("Cancels a pending GPU readback request.",),
            tags=("capture", "cancel", "render-target"),
        ),
        operation(
            "infernux.capture.scene-pick.request",
            OperationKind.COMMAND,
            "Request an asynchronous GPU object-ID pick from the Scene target.",
            _scene_pick_request,
            capability="capture.write",
            input_properties={
                "normalized_x": {"type": "number"},
                "normalized_y": {"type": "number"},
                "viewport_width": {"type": "integer"},
                "viewport_height": {"type": "integer"},
            },
            required=("normalized_x", "normalized_y", "viewport_width", "viewport_height"),
            side_effects=("Queues one GPU object-ID readback without changing selection.",),
            tags=("capture", "scene", "gpu-pick", "object-id"),
        ),
        operation(
            "infernux.capture.scene-pick.status",
            OperationKind.QUERY,
            "Poll one Scene GPU object-ID pick.",
            _scene_pick_status,
            capability="capture.read",
            input_properties={"request_id": {"type": "integer"}},
            required=("request_id",),
            tags=("capture", "scene", "gpu-pick", "status"),
        ),
    )


def _request_capture(source: str = "game", file_name: str = "", camera_component_id: int = 0):
    active = automation_context()
    source_name = str(source).strip().casefold()
    if source_name not in {"scene", "game", "editor", "camera"}:
        raise OperationError("operation.invalid_arguments", "source must be scene, game, editor, or camera")
    if (source_name == "camera") != (camera_component_id > 0):
        raise OperationError("operation.invalid_arguments", "camera capture requires camera_component_id; other sources do not use it")
    requested = os.path.basename(str(file_name).strip())
    if requested:
        stem, extension = os.path.splitext(requested)
        if extension.casefold() != ".png":
            raise OperationError("operation.invalid_arguments", "file_name must end in .png")
        requested = f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', stem).strip('.-') or source_name}.png"
    else:
        requested = f"{source_name}-{time.time_ns()}.png"
    review = os.path.join(active.artifact_root, "review")
    os.makedirs(review, exist_ok=True)
    output = os.path.abspath(os.path.join(review, requested))
    # One artifact URI represents exactly one request. Remove an older file at
    # that URI before queueing so a failed or timed-out capture cannot be
    # mistaken for a newly completed frame.
    if os.path.isfile(output):
        os.remove(output)
    capture_id = on_editor(
        "infernux.capture.request",
        lambda: EditorAutomationHost.instance().request_capture(source_name, output, camera_component_id),
    )
    return {
        "capture_id": int(capture_id),
        "source": source_name,
        "status": "pending_gpu",
        "artifact_uri": os.path.relpath(output, active.artifact_root).replace("\\", "/"),
        "pixel_origin": "engine_render_target",
        "output_path": output,
    }


def _capture_status(capture_id: int):
    active = automation_context()
    value = on_editor(
        "infernux.capture.status",
        lambda: EditorAutomationHost.instance().capture_status(capture_id),
    )
    output = str(value.pop("output_path", "") or "")
    value.update(
        {
            "capture_id": int(capture_id),
            "artifact_uri": os.path.relpath(output, active.artifact_root).replace("\\", "/") if output else "",
            "pixel_origin": "engine_render_target",
            "output_path": output,
            "terminal": str(value.get("status", "")) in _TERMINAL,
        }
    )
    if str(value.get("status", "")) == "completed" and output and os.path.isfile(output):
        value["byte_size"] = os.path.getsize(output)
    return value


def _capture_cancel(capture_id: int):
    automation_context()
    cancelled = on_editor(
        "infernux.capture.cancel",
        lambda: EditorAutomationHost.instance().cancel_capture(capture_id),
    )
    return {"capture_id": int(capture_id), "cancelled": bool(cancelled)}


def _coordinate(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OperationError("operation.invalid_arguments", f"{name} must be within [0, 1]")
    return result


def _extent(name: str, value: int) -> int:
    result = int(value)
    if result <= 0 or result > 32768:
        raise OperationError("operation.invalid_arguments", f"{name} must be within [1, 32768]")
    return result


def _scene_pick_request(normalized_x: float, normalized_y: float, viewport_width: int, viewport_height: int):
    x = _coordinate("normalized_x", normalized_x)
    y = _coordinate("normalized_y", normalized_y)
    width = _extent("viewport_width", viewport_width)
    height = _extent("viewport_height", viewport_height)
    pixel_x = x * max(width - 1, 0)
    pixel_y = y * max(height - 1, 0)
    request_id = on_editor(
        "infernux.capture.scene-pick.request",
        lambda: EditorAutomationHost.instance().request_scene_pick(pixel_x, pixel_y, float(width), float(height)),
    )
    if int(request_id) <= 0:
        raise OperationError("capture.pick_rejected", "Scene GPU pick was rejected.")
    return {
        "request_id": int(request_id),
        "status": "pending",
        "normalized_x": x,
        "normalized_y": y,
        "viewport_width": width,
        "viewport_height": height,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "selection_changed": False,
    }


def _scene_pick_status(request_id: int):
    value = on_editor(
        "infernux.capture.scene-pick.status",
        lambda: EditorAutomationHost.instance().scene_pick_status(request_id),
    )
    return {
        **value,
        "request_id": int(request_id),
        "selection_changed": False,
        "terminal": str(value.get("status", "")) != "pending",
    }


__all__ = ["build_capture_operations"]
