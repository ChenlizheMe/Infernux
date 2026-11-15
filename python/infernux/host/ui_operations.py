"""Semantic Editor UI observation operations."""

from __future__ import annotations

import math
import time

from infernux.host import EditorAutomationHost, Operation, OperationError, OperationKind

from infernux.host.operation_support import on_editor, operation


def build_ui_operations() -> tuple[Operation, ...]:
    filters = {
        "label": {"type": "string", "default": ""},
        "kind": {"type": "string", "default": ""},
        "window": {"type": "string", "default": ""},
        "semantic_id": {"type": "string", "default": ""},
        "visible_only": {"type": "boolean", "default": True},
        "limit": {"type": "integer", "default": 500},
    }
    return (
        operation(
            "infernux.ui.semantic.snapshot",
            OperationKind.QUERY,
            "Request and read one semantic snapshot of rendered Editor controls.",
            _snapshot,
            capability="ui.read",
            input_properties=filters,
            tags=("ui", "semantic", "snapshot", "editor", "validation"),
        ),
        operation(
            "infernux.ui.semantic.status",
            OperationKind.QUERY,
            "Read semantic-capture sequencing without scheduling a frame.",
            _status,
            capability="ui.read",
            tags=("ui", "semantic", "status", "editor"),
        ),
        operation(
            "infernux.ui.semantic.wait",
            OperationKind.QUERY,
            "Wait for a matching control to appear in a freshly rendered semantic frame.",
            _wait_for_target,
            capability="ui.read",
            input_properties={
                **filters,
                "timeout_seconds": {"type": "number", "default": 5.0},
                "poll_interval": {"type": "number", "default": 0.1},
            },
            tags=("ui", "semantic", "wait", "target", "editor", "validation"),
        ),
        operation(
            "infernux.ui.semantic.capture.set",
            OperationKind.COMMAND,
            "Enable or disable semantic Editor UI capture.",
            _set_capture,
            capability="ui.write",
            input_properties={"enabled": {"type": "boolean"}},
            required=("enabled",),
            side_effects=("Changes semantic capture state for subsequent Editor frames.",),
            tags=("ui", "semantic", "capture", "editor"),
        ),
    )


def _snapshot(label: str = "", kind: str = "", window: str = "", semantic_id: str = "", visible_only: bool = True, limit: int = 500):
    host = EditorAutomationHost.instance()
    # RequestSnapshot owns one frame; continuous capture is an explicit setting.
    request = on_editor("infernux.ui.semantic.request", host.request_semantic_snapshot)
    deadline = time.monotonic() + 0.5
    while True:
        raw = on_editor("infernux.ui.semantic.snapshot", host.semantic_snapshot)
        if int(request or 0) <= 0 or int(raw.get("request_sequence", 0) or 0) >= int(request):
            break
        if time.monotonic() >= deadline:
            raise OperationError(
                "ui.capture_timeout",
                "The requested semantic UI frame was not published before the timeout.",
            )
        time.sleep(0.01)
    needle = str(label).casefold()
    expected_kind = str(kind).casefold()
    expected_window = str(window).casefold()
    expected_semantic = str(semantic_id)
    targets = []
    for raw_target in raw.get("targets", []) or []:
        target = dict(raw_target)
        haystack = " ".join(str(target.get(key, "")) for key in ("label", "id", "semantic_id")).casefold()
        if needle and needle not in haystack:
            continue
        if expected_kind and str(target.get("kind", "")).casefold() != expected_kind:
            continue
        if expected_window and expected_window not in str(target.get("window", "")).casefold():
            continue
        if expected_semantic and str(target.get("semantic_id", "")) != expected_semantic:
            continue
        if visible_only and not bool(target.get("visible", False)):
            continue
        targets.append(target)
        if len(targets) >= max(1, min(int(limit), 2000)):
            break
    result = {
        "request_sequence": int(request or 0),
        "frame": int(raw.get("frame", 0) or 0),
        "capture_enabled": bool(raw.get("capture_enabled")),
        "mouse": raw["mouse"],
        "targets": targets,
        "returned": len(targets),
    }
    # The MCP plug-in can update independently of an older engine. Preserve
    # its native window-coordinate targets, but never invent a pixel mapping
    # when that native version does not publish coordinate provenance.
    if "coordinates" in raw:
        result["coordinates"] = dict(raw["coordinates"])
    return result


def _status():
    raw = on_editor(
        "infernux.ui.semantic.status", EditorAutomationHost.instance().semantic_snapshot
    )
    return {
        "capture_enabled": bool(raw.get("capture_enabled")),
        "published_frame": int(raw.get("frame", 0) or 0),
        "published_request_sequence": int(raw.get("request_sequence", 0) or 0),
        "capture_state": dict(raw.get("capture_state") or {}),
    }


def _set_capture(enabled: bool):
    value = on_editor(
        "infernux.ui.semantic.capture.set",
        lambda: EditorAutomationHost.instance().semantic_capture_enabled(enabled),
    )
    return {"capture_enabled": bool(value)}


def _wait_for_target(
    label: str = "",
    kind: str = "",
    window: str = "",
    semantic_id: str = "",
    visible_only: bool = True,
    limit: int = 500,
    timeout_seconds: float = 5.0,
    poll_interval: float = 0.1,
):
    timeout = float(timeout_seconds)
    interval = float(poll_interval)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 30:
        raise OperationError(
            "operation.invalid_arguments", "timeout_seconds must be within (0, 30]"
        )
    if not math.isfinite(interval) or interval <= 0 or interval > 2:
        raise OperationError(
            "operation.invalid_arguments", "poll_interval must be within (0, 2]"
        )
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        latest = _snapshot(
            label=label,
            kind=kind,
            window=window,
            semantic_id=semantic_id,
            visible_only=visible_only,
            limit=limit,
        )
        if latest["targets"]:
            return {**latest, "matched": True}
        time.sleep(min(interval, max(deadline - time.monotonic(), 0.0)))
    raise OperationError(
        "ui.target_timeout",
        "No matching semantic UI target appeared before the timeout.",
        details=latest,
    )


__all__ = ["build_ui_operations"]
