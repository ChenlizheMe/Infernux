"""Play Mode operations backed by the editor runtime state machine."""

from __future__ import annotations

import time

from infernux.host import EditorAutomationHost, Operation, OperationError, OperationKind

from infernux.host.operation_support import on_editor, operation


def build_runtime_operations() -> tuple[Operation, ...]:
    transition_input = {"timeout_seconds": {"type": "number", "default": 10.0}}
    return (
        operation(
            "infernux.runtime.performance.begin",
            OperationKind.COMMAND,
            "Reset and begin the bounded native frame timing window; does not change Play state.",
            lambda sample_count=240: on_editor("infernux.runtime.performance.begin", lambda: {
                "first_frame": EditorAutomationHost.instance().begin_renderer_performance_window(sample_count)}),
            capability="runtime.write",
            input_properties={"sample_count": {
                "type": "integer", "minimum": 1, "maximum": 65536, "default": 240,
            }},
            side_effects=("Resets previously collected frame timing samples.",),
            tags=("runtime", "performance"),
        ),
        operation(
            "infernux.runtime.performance.get",
            OperationKind.QUERY,
            "Read native frame timing percentiles without GPU readback or resetting the window.",
            lambda: on_editor("infernux.runtime.performance.get",
                              EditorAutomationHost.instance().renderer_performance_window),
            capability="runtime.read",
            tags=("runtime", "performance"),
        ),
        operation(
            "infernux.runtime.gizmos.statistics",
            OperationKind.QUERY,
            "Read the last Gizmo callback, geometry, packing, upload, and batch observation.",
            lambda: on_editor(
                "infernux.runtime.gizmos.statistics",
                EditorAutomationHost.instance().gizmo_collection_observation,
            ),
            capability="runtime.read",
            tags=("runtime", "performance", "gizmos"),
        ),
        operation(
            "infernux.runtime.ui.input_state",
            OperationKind.QUERY,
            "Read current raw, owned and Game mouse frames, viewport mapping and the last UI pointer transition; does not dispatch input.",
            lambda: on_editor("infernux.runtime.ui.input_state", EditorAutomationHost.instance().ui_input_state),
            capability="runtime.read",
            tags=("runtime", "ui", "input", "diagnostics"),
        ),
        operation(
            "infernux.runtime.status",
            OperationKind.QUERY,
            "Read Play Mode state, timing, pause state, and time scale.",
            _runtime_status,
            capability="runtime.read",
            tags=("runtime", "play-mode", "status", "time"),
        ),
        operation(
            "infernux.runtime.compute.statistics",
            OperationKind.QUERY,
            "Read cumulative compute transfers, submissions and CPU waits. GPU time is the latest completed submission, not a frame total.",
            lambda: on_editor("infernux.runtime.compute.statistics",
                              EditorAutomationHost.instance().compute_statistics),
            capability="runtime.read",
            tags=("runtime", "performance", "compute"),
        ),
        operation(
            "infernux.runtime.compute.reset_statistics",
            OperationKind.COMMAND,
            "Return the current compute counters and reset their accumulation; does not submit pending work or change Play state.",
            lambda: on_editor("infernux.runtime.compute.reset_statistics", lambda:
                              EditorAutomationHost.instance().compute_statistics(reset=True)),
            capability="runtime.write",
            side_effects=("Resets cumulative compute counters, not GPU timestamp history.",),
            tags=("runtime", "performance", "compute"),
        ),
        operation(
            "infernux.runtime.compute.profiling",
            OperationKind.COMMAND,
            "Explicitly enable or disable native GPU submission timestamps; changing query resources waits for in-flight compute.",
            lambda enabled: on_editor("infernux.runtime.compute.profiling", lambda:
                                     EditorAutomationHost.instance().set_compute_profiling(enabled)),
            capability="runtime.write",
            input_properties={"enabled": {"type": "boolean"}},
            required=("enabled",),
            side_effects=("May wait for in-flight compute while changing native timestamp resources.",),
            tags=("runtime", "performance", "compute"),
        ),
        operation(
            "infernux.runtime.play",
            OperationKind.COMMAND,
            "Enter Play Mode through the editor state machine.",
            lambda timeout_seconds=10.0: _transition(
                "infernux.runtime.play", "enter_play_mode", "playing", timeout_seconds
            ),
            capability="runtime.write",
            input_properties=transition_input,
            side_effects=("Replaces the edit world with a Play Mode world.",),
            tags=("runtime", "play-mode", "play"),
        ),
        operation(
            "infernux.runtime.stop",
            OperationKind.COMMAND,
            "Exit Play Mode through the editor state machine.",
            lambda timeout_seconds=10.0: _transition(
                "infernux.runtime.stop", "exit_play_mode", "edit", timeout_seconds
            ),
            capability="runtime.write",
            input_properties=transition_input,
            side_effects=("Stops Play Mode and restores the edit world.",),
            tags=("runtime", "play-mode", "stop"),
        ),
        operation(
            "infernux.runtime.pause",
            OperationKind.COMMAND,
            "Pause the active Play Mode world.",
            lambda timeout_seconds=10.0: _transition(
                "infernux.runtime.pause", "pause", "paused", timeout_seconds
            ),
            capability="runtime.write",
            input_properties=transition_input,
            side_effects=("Pauses Play Mode simulation.",),
            tags=("runtime", "play-mode", "pause"),
        ),
        operation(
            "infernux.runtime.resume",
            OperationKind.COMMAND,
            "Resume a paused Play Mode world.",
            lambda timeout_seconds=10.0: _transition(
                "infernux.runtime.resume", "resume", "playing", timeout_seconds
            ),
            capability="runtime.write",
            input_properties=transition_input,
            side_effects=("Resumes Play Mode simulation.",),
            tags=("runtime", "play-mode", "resume"),
        ),
        operation(
            "infernux.runtime.step",
            OperationKind.COMMAND,
            "Advance one frame while Play Mode is paused.",
            lambda timeout_seconds=10.0: _transition(
                "infernux.runtime.step", "step_frame", "paused", timeout_seconds,
            ),
            capability="runtime.write",
            input_properties=transition_input,
            side_effects=("Advances the paused simulation by one frame.",),
            tags=("runtime", "play-mode", "step", "frame"),
        ),
        operation(
            "infernux.runtime.time-scale.set",
            OperationKind.COMMAND,
            "Set Play Mode time scale.",
            _set_time_scale,
            capability="runtime.write",
            input_properties={"value": {"type": "number"}},
            required=("value",),
            side_effects=("Changes the simulation time scale.",),
            tags=("runtime", "play-mode", "time", "scale"),
        ),
    )


def _runtime_status() -> dict[str, object]:
    return on_editor(
        "infernux.runtime.status",
        lambda: {"runtime": EditorAutomationHost.instance().runtime_status()},
    )


def _transition(
    operation_id: str,
    method: str,
    expected_state: str,
    timeout_seconds: float,
    *,
    require_truthy: bool = True,
) -> dict[str, object]:
    def execute():
        result = EditorAutomationHost.instance().runtime_transition(method)
        if require_truthy and not result["accepted"]:
            raise OperationError("runtime.transition_rejected", f"Runtime transition was rejected: {method}")
        return result

    result = on_editor(operation_id, execute)
    timeout = max(0.0, min(float(timeout_seconds), 60.0))
    deadline = time.monotonic() + timeout
    runtime = dict(result.get("runtime") or {})
    target_step = result.get("target_step_sequence")
    def completed(value: dict[str, object]) -> bool:
        return (
            str(value.get("state", "")) == expected_state
            and not bool(value.get("transition_pending", False))
            and (target_step is None or int(value.get("step_sequence", -1)) >= target_step)
        )

    while not completed(runtime) and time.monotonic() < deadline:
        if method == "step_frame" and str(runtime.get("state", "")) != "paused":
            raise OperationError("runtime.step_interrupted", "The requested step was interrupted by a runtime state change")
        time.sleep(0.02)
        runtime = on_editor(
            operation_id,
            lambda: EditorAutomationHost.instance().runtime_status(),
        )
    complete = completed(runtime)
    if require_truthy and not complete:
        raise OperationError(
            "runtime.transition_timeout",
            f"Runtime did not reach {expected_state!r} within {timeout:.1f} seconds.",
            details={"runtime": runtime, "method": method},
        )
    receipt = {
        "accepted": bool(result.get("accepted", True)),
        "transition_complete": complete,
        "target_state": expected_state,
        "runtime": runtime,
    }
    if target_step is not None:
        receipt["target_step_sequence"] = target_step
    return receipt


def _set_time_scale(value: float) -> dict[str, object]:
    return on_editor(
        "infernux.runtime.time-scale.set",
        lambda: EditorAutomationHost.instance().set_time_scale(value),
    )


__all__ = ["build_runtime_operations"]
