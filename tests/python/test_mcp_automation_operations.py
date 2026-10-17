from __future__ import annotations

import ast
import threading
from types import SimpleNamespace
import pytest
from pathlib import Path

from infernux.host import EditorAutomationHost, MainThreadCommandQueue
from infernux_mcp import session
from infernux_mcp.operations import build_operations


def test_authoring_queue_drains_before_native_frame_and_reload(monkeypatch):
    from contextlib import nullcontext
    import infernux.lib as native
    from infernux.engine import engine as engine_module

    calls = []
    callbacks = {}
    active_frame = False

    def begin():
        nonlocal active_frame
        active_frame = True
        calls.append("begin")

    def drain():
        assert not active_frame, "Authoring cannot retire scripts during an execution frame"
        calls.append("authoring")

    scene_manager = SimpleNamespace(
        get_global_transform_serial=lambda: 0,
        set_runtime_lifecycle_callbacks=lambda *args: callbacks.update(begin=args[0]),
        set_runtime_frame_barrier_callback=lambda _callback: None,
    )
    monkeypatch.setattr(native, "SceneManager", SimpleNamespace(instance=lambda: scene_manager))
    monkeypatch.setattr(MainThreadCommandQueue, "instance", lambda: SimpleNamespace(drain=drain))
    monkeypatch.setattr(engine_module, "_PLAYER_MODE", False)
    engine = engine_module.Engine.__new__(engine_module.Engine)
    engine._mode = engine_module.RuntimeMode.Graphical
    engine._engine = SimpleNamespace(
        set_pre_scene_update_callback=lambda cb: callbacks.update(pre_scene=cb),
    )
    engine._render_submission_frame = 0
    engine._runtime_scheduler = SimpleNamespace(
        change_journal=SimpleNamespace(transaction=nullcontext),
        begin_native_frame=begin,
        end_native_frame=lambda: None,
        execute_native_editor_update=lambda: None,
        bind_native_bridge=lambda _manager: None,
    )
    engine.tick_play_mode = lambda dt: calls.append(("reload", dt))
    engine._install_pre_scene_time_callback()
    callbacks["pre_scene"](0.125)
    callbacks["begin"]()
    assert calls == ["authoring", ("reload", 0.125), "begin"]
    assert engine._render_submission_frame == 1
    # This fake does not own a native engine to destroy.
    engine._engine = None


def test_runtime_compute_operations_use_host_and_editor_boundary(monkeypatch):
    from infernux_mcp import runtime_operations
    from infernux.host import OperationKind

    calls = []
    snapshot = {"dispatch_count": 9, "wait_ms": 1.5, "gpu_time_ms": None}
    host = SimpleNamespace(
        compute_statistics=lambda **kw: calls.append(("stats", kw)) or snapshot,
        set_compute_profiling=lambda enabled: calls.append(("profile", enabled)) or {"enabled": enabled},
    )
    monkeypatch.setattr(EditorAutomationHost, "instance", staticmethod(lambda: host))
    monkeypatch.setattr(runtime_operations, "on_editor", lambda name, callback: calls.append(name) or callback())
    operations = {item.schema.id: item for item in runtime_operations.build_runtime_operations()}
    prefix = "infernux.runtime.compute."
    query = operations[prefix + "statistics"]
    assert query.schema.kind == OperationKind.QUERY
    assert query.schema.capabilities == ("runtime.read",)
    assert not query.schema.side_effects
    assert query.handler() == snapshot
    assert operations[prefix + "reset_statistics"].handler() == snapshot
    profile = operations[prefix + "profiling"]
    assert profile.schema.capabilities == ("runtime.write",)
    assert profile.handler(True) == {"enabled": True}
    assert profile.handler(False) == {"enabled": False}
    assert calls == [
        prefix + "statistics",
        ("stats", {}),
        prefix + "reset_statistics",
        ("stats", {"reset": True}),
        prefix + "profiling",
        ("profile", True),
        prefix + "profiling",
        ("profile", False),
    ]


def test_runtime_compute_host_reuses_public_statistics_and_propagates_errors(monkeypatch):
    from dataclasses import dataclass
    from infernux import compute

    @dataclass
    class Snapshot:
        dispatch_count: int = 7
        gpu_time_ms: float | None = None

    calls = []
    monkeypatch.setattr(compute, "statistics", lambda **kw: calls.append(kw) or Snapshot())
    monkeypatch.setattr(compute, "set_profiling_enabled", lambda value: calls.append(value))
    host = EditorAutomationHost()
    assert host.compute_statistics() == {"dispatch_count": 7, "gpu_time_ms": None}
    host.compute_statistics(reset=True)
    assert host.set_compute_profiling(True) == {"enabled": True}
    assert calls == [{"reset": False}, {"reset": True}, True]

    def unavailable(_enabled):
        raise compute.ComputeCapabilityError("no timestamp support")

    monkeypatch.setattr(compute, "set_profiling_enabled", unavailable)
    with pytest.raises(compute.ComputeCapabilityError, match="no timestamp support"):
        host.set_compute_profiling(True)


class _AutomationHost(EditorAutomationHost):
    def __init__(self, capture_path: Path):
        self.capture_path = capture_path
        self.events = []
        self.sequence = 0
        self.capture_enabled = False
        self.runtime_state = "edit"
        self.runtime_target = "edit"
        self.runtime_pending_polls = 0
        self.runtime_status_calls = 0

    def queue_input(self, kind: str, **arguments):
        self.sequence += 1
        self.events.append((kind, arguments))
        return {
            "sequence": self.sequence,
            "last_processed_sequence": self.sequence,
            "pending_event_count": 0,
        }

    def input_status(self):
        return {
            "last_processed_sequence": self.sequence,
            "pending_event_count": 0,
        }

    def semantic_capture_enabled(self, enabled: bool):
        self.capture_enabled = bool(enabled)
        return self.capture_enabled

    def request_semantic_snapshot(self):
        return 9

    def semantic_snapshot(self):
        return {
            "capture_enabled": self.capture_enabled,
            "frame": 27,
            "request_sequence": 9,
            "mouse": (18.0, 24.0),
            "coordinates": {
                "space": "sdl_window",
                "display_origin": (0.0, 0.0),
                "display_size": (1536.0, 990.0),
                "framebuffer_scale": (2.0, 2.0),
                "ui_scale": 1.0,
            },
            "targets": [
                {
                    "id": "button:main:run:1",
                    "semantic_id": "run",
                    "kind": "button",
                    "label": "Run",
                    "window": "Main",
                    "visible": True,
                },
                {
                    "id": "hidden",
                    "semantic_id": "hidden",
                    "kind": "button",
                    "label": "Hidden",
                    "window": "Main",
                    "visible": False,
                },
            ],
        }

    def request_capture(self, source: str, output_path: str, camera_component_id: int = 0):
        self.capture_path = Path(output_path)
        self.capture_source = (source, camera_component_id)
        return 17

    def capture_status(self, capture_id: int):
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        self.capture_path.write_bytes(b"engine-render-target")
        return {
            "capture_id": capture_id,
            "status": "completed",
            "output_path": str(self.capture_path),
        }

    def cancel_capture(self, capture_id: int):
        return capture_id == 17

    def console_read(self, limit=100, levels=()):
        return {
            "entries": [{"level": "WARN", "message": "native warning"}],
            "source": "native_console",
            "surface": "console",
            "status_bar": {"surface": "status_bar", "message": "profile"},
        }

    def runtime_transition(self, method: str):
        self.runtime_target = {
            "enter_play_mode": "playing",
            "exit_play_mode": "edit",
            "pause": "paused",
            "resume": "playing",
            "step_frame": "paused",
        }[method]
        return {"accepted": True, "runtime": {"state": self.runtime_state}}

    def runtime_status(self):
        self.runtime_status_calls += 1
        self.runtime_state = self.runtime_target
        pending = self.runtime_pending_polls > 0
        if pending:
            self.runtime_pending_polls -= 1
        return {
            "state": self.runtime_state,
            "transition_pending": pending,
        }


def _operations(tmp_path):
    session.configure(
        str(tmp_path),
        {
            "profile": "global_validation",
            "session": {"build_profile": "debug_feedback"},
        },
    )
    queue = MainThreadCommandQueue()
    queue._main_thread_id = threading.get_ident()
    MainThreadCommandQueue._instance = queue
    host = _AutomationHost(tmp_path / "capture.png")
    EditorAutomationHost.set_provider(host)
    values = {item.schema.id: item.handler for item in build_operations(str(tmp_path))}
    return values, host


def test_pointer_click_publishes_one_native_gesture_batch(tmp_path, monkeypatch):
    from infernux_mcp import input_operations

    operations, host = _operations(tmp_path)
    original = input_operations.on_editor
    published = []

    def record_batch(name, callback):
        before = len(host.events)
        result = original(name, callback)
        if len(host.events) > before:
            published.append(host.events[before:])
        return result

    monkeypatch.setattr(input_operations, "on_editor", record_batch)
    try:
        result = operations["infernux.input.pointer.click"](40.0, 80.0)
        assert result["delivered"] is True
        assert len(published) == 1
        assert [kind for kind, _ in published[0]] == ["pointer_move", "pointer_button", "pointer_button"]
        assert [arguments["pressed"] for _, arguments in published[0][1:]] == [True, False]
    finally:
        EditorAutomationHost.set_provider(None)


def test_runtime_transition_waits_for_the_deferred_target_state(tmp_path):
    operations, host = _operations(tmp_path)
    try:
        result = operations["infernux.runtime.play"](timeout_seconds=1.0)
        assert result["accepted"] is True
        assert result["transition_complete"] is True
        assert result["runtime"]["state"] == "playing"
        assert host.runtime_state == "playing"
    finally:
        EditorAutomationHost.set_provider(None)


def test_runtime_stop_waits_for_scene_restore_after_state_becomes_edit(tmp_path):
    operations, host = _operations(tmp_path)
    try:
        host.runtime_state = "playing"
        host.runtime_target = "playing"
        host.runtime_pending_polls = 2
        result = operations["infernux.runtime.stop"](timeout_seconds=1.0)

        assert result["transition_complete"] is True
        assert result["runtime"]["state"] == "edit"
        assert result["runtime"]["transition_pending"] is False
        assert host.runtime_status_calls >= 3
    finally:
        EditorAutomationHost.set_provider(None)


def test_capture_surface_cannot_fall_back_to_operating_system_pixels():
    plugin = (
        Path(__file__).parents[2]
        / "external"
        / "plugins"
        / "infernux_mcp"
        / "package"
        / "editor"
        / "infernux_mcp"
        / "capture_operations.py"
    )
    tree = ast.parse(plugin.read_text(encoding="utf-8"), filename=str(plugin))
    forbidden_roots = {"PIL", "pyautogui", "mss", "ImageGrab", "win32gui", "win32ui"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(str(node.module or "").split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
    assert imported.isdisjoint(forbidden_roots)
