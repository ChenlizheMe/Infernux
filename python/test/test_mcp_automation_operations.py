from __future__ import annotations

import ast
import threading
from types import SimpleNamespace
import pytest
from pathlib import Path

from Infernux.host import EditorAutomationHost, MainThreadCommandQueue, OperationError
from infernux_mcp import session
from infernux_mcp.operations import build_operations


def test_authoring_queue_drains_before_native_frame_and_reload(monkeypatch):
    from contextlib import nullcontext
    import Infernux.lib as native
    from Infernux.engine import engine as engine_module

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


def test_renderer_parameter_operations_are_thin_shared_material_safe_projection(monkeypatch):
    from infernux_mcp import material_operations

    class Renderer:
        def __init__(self):
            self.runtime = {}
            self.persistent = {}

        def set_parameter(self, name, value, *, material_slot=0, persistent=False, owner="script"):
            target = self.persistent if persistent else self.runtime
            target[(material_slot, name)] = value

        def get_parameter(self, name, *, material_slot=0, persistent_only=False):
            key = (material_slot, name)
            if not persistent_only and key in self.runtime:
                return self.runtime[key]
            return self.persistent.get(key)

        def remove_parameter(self, name, *, material_slot=0, persistent=False, owner="script"):
            target = self.persistent if persistent else self.runtime
            return target.pop((material_slot, name), None) is not None

        def clear_parameters(self, *, material_slot=0, persistent=False, owner="script"):
            target = self.persistent if persistent else self.runtime
            for key in [key for key in target if key[0] == material_slot]:
                del target[key]

    renderer = Renderer()
    monkeypatch.setattr(material_operations, "component", lambda *_args: (object(), renderer))
    monkeypatch.setattr(material_operations, "on_editor", lambda _name, callback: callback())
    operations = {
        item.schema.id: item for item in material_operations.build_material_operations()
    }

    set_parameter = operations["infernux.renderer.parameter.set"]
    assert set_parameter.schema.capabilities == ("material.write",)
    assert "shared material" in set_parameter.schema.summary
    set_parameter.handler(1, 2, "baseColor", [0.8, 0.1, 0.2, 1.0], 0, True)
    set_parameter.handler(1, 2, "baseColor", [0.1, 0.8, 0.2, 1.0])

    get_parameter = operations["infernux.renderer.parameter.get"].handler
    assert get_parameter(1, 2, "baseColor")["value"] == [0.1, 0.8, 0.2, 1.0]
    assert get_parameter(1, 2, "baseColor", persistent_only=True)["value"] == [
        0.8,
        0.1,
        0.2,
        1.0,
    ]
    assert operations["infernux.renderer.parameter.remove"].handler(
        1, 2, "baseColor"
    )["removed"] is True
    assert get_parameter(1, 2, "baseColor")["value"] == [0.8, 0.1, 0.2, 1.0]
    assert operations["infernux.renderer.parameter.clear"].handler(
        1, 2, persistent=True
    )["cleared"] is True
    assert get_parameter(1, 2, "baseColor")["inherited"] is True


def test_runtime_compute_operations_use_host_and_editor_boundary(monkeypatch):
    from infernux_mcp import runtime_operations
    from Infernux.host import OperationKind

    calls = []
    snapshot = {"dispatch_count": 9, "wait_ms": 1.5, "gpu_time_ms": None}
    host = SimpleNamespace(
        compute_statistics=lambda **kw: calls.append(("stats", kw)) or snapshot,
        set_compute_profiling=lambda enabled: calls.append(("profile", enabled)) or {"enabled": enabled},
    )
    monkeypatch.setattr(EditorAutomationHost, "instance", staticmethod(lambda: host))
    monkeypatch.setattr(runtime_operations, "on_editor",
                        lambda name, callback: calls.append(name) or callback())
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
        prefix + "statistics", ("stats", {}),
        prefix + "reset_statistics", ("stats", {"reset": True}),
        prefix + "profiling", ("profile", True),
        prefix + "profiling", ("profile", False),
    ]


def test_runtime_compute_host_reuses_public_statistics_and_propagates_errors(monkeypatch):
    from dataclasses import dataclass
    from Infernux import compute

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


def test_runtime_performance_operations_reuse_native_window(monkeypatch):
    from infernux_mcp import runtime_operations
    events = []
    snapshot = {"sample_count": 17, "timings": {"frame": {"p95_ms": 4.5}}}
    native = SimpleNamespace(
        begin_renderer_performance_window=lambda count: events.append(("begin", count)) or 42,
        get_renderer_performance_window=lambda: events.append("get") or snapshot,
        resident_mesh_vertex_buffer_count=3,
        pending_mesh_gpu_upload_count=1,
        submitted_mesh_gpu_upload_count=7,
        completed_mesh_gpu_upload_count=6,
        pending_texture_cpu_load_count=2,
        pending_texture_gpu_upload_count=3,
        submitted_texture_gpu_upload_count=9,
        completed_texture_gpu_upload_count=8,
        gpu_residency_snapshot={"resident_textures": 5},
    )
    host = EditorAutomationHost()
    monkeypatch.setattr(host, "_native_engine", lambda: native)
    monkeypatch.setattr(EditorAutomationHost, "instance", staticmethod(lambda: host))
    monkeypatch.setattr(runtime_operations, "on_editor", lambda name, callback: callback())
    operations = {item.schema.id: item for item in runtime_operations.build_runtime_operations()}
    assert operations["infernux.runtime.performance.begin"].handler() == {"first_frame": 42}
    query = operations["infernux.runtime.performance.get"]
    expected = {
        **snapshot,
        "resources": {
            "resident_mesh_vertex_buffers": 3,
            "pending_mesh_uploads": 1,
            "submitted_mesh_uploads": 7,
            "completed_mesh_uploads": 6,
            "pending_texture_cpu_loads": 2,
            "pending_texture_uploads": 3,
            "submitted_texture_uploads": 9,
            "completed_texture_uploads": 8,
        },
        "gpu_residency": {"resident_textures": 5},
    }
    assert query.handler() == expected
    assert query.handler() == expected
    assert events == [("begin", 240), "get", "get"]
    assert operations["infernux.runtime.performance.begin"].handler(12000) == {"first_frame": 42}
    assert events[-1] == ("begin", 12000)
    assert not query.schema.side_effects


def test_runtime_gizmo_statistics_reuses_editor_host(monkeypatch):
    from infernux_mcp import runtime_operations

    snapshot = {
        "timing_enabled": True,
        "callback_ms": 0.12,
        "geometry_build_ms": 0.08,
        "pack_ms": 0.03,
        "upload_ms": 0.01,
    }
    host = SimpleNamespace(gizmo_collection_observation=lambda: snapshot)
    monkeypatch.setattr(EditorAutomationHost, "instance", staticmethod(lambda: host))
    monkeypatch.setattr(runtime_operations, "on_editor", lambda _name, callback: callback())

    operation = {
        item.schema.id: item for item in runtime_operations.build_runtime_operations()
    }["infernux.runtime.gizmos.statistics"]

    assert operation.handler() == snapshot
    assert operation.schema.capabilities == ("runtime.read",)
    assert not operation.schema.side_effects


def test_loaded_scene_activation_routes_through_the_editor_host(monkeypatch):
    from Infernux.host import scene_operations

    calls = []
    host = SimpleNamespace(
        activate_loaded_scene=lambda world_id: calls.append(world_id)
        or {"active_world_id": world_id, "scenes": []}
    )
    monkeypatch.setattr(EditorAutomationHost, "instance", staticmethod(lambda: host))
    monkeypatch.setattr(scene_operations, "on_editor", lambda _name, callback: callback())
    operations = {
        item.schema.id: item for item in scene_operations.build_scene_operations()
    }

    result = operations["infernux.scene.active.set"].handler(73)

    assert result == {"active_world_id": 73, "scenes": []}
    assert calls == [73]


def test_asset_identity_uses_native_metadata_method(monkeypatch, tmp_path):
    from Infernux.lib import ResourceType
    from Infernux.host import operation_support

    database = SimpleNamespace(
        get_meta_by_path=lambda path: SimpleNamespace(get_resource_type=lambda: ResourceType.Mesh),
        get_guid_from_path=lambda path: "mesh-guid",
    )
    monkeypatch.setattr(operation_support, "asset_database", lambda: database)
    identity = operation_support.asset_identity(str(tmp_path / "saved.inxmesh"))
    assert identity["resource_type"] == "mesh"
    assert identity["guid"] == "mesh-guid"
    database.get_meta_by_path = lambda path: None
    assert operation_support.asset_identity(str(tmp_path))["resource_type"] == "folder"
    with pytest.raises(OperationError, match="metadata is unavailable"):
        operation_support.asset_identity(str(tmp_path / "missing.inxmesh"))


def test_component_projection_summarizes_inline_mesh_data_by_default():
    from infernux_mcp.operation_support import serializable_component

    component = SimpleNamespace(
        component_id=37,
        type_name="MeshRenderer",
        enabled=True,
        serialize_document=lambda: {
            "inlineMeshName": "Generated Grid",
            "inlineVertices": [[0.0] * 23 for _ in range(4)],
            "inlineIndices": [0, 1, 2, 2, 1, 3],
        },
    )

    compact = serializable_component(component)
    assert compact["document"] == {
        "inlineMeshName": "Generated Grid",
        "inlineMeshSummary": {"vertexCount": 4, "indexCount": 6},
    }
    complete = serializable_component(component, include_mesh_data=True)
    assert len(complete["document"]["inlineVertices"]) == 4
    assert complete["document"]["inlineIndices"] == [0, 1, 2, 2, 1, 3]


def test_component_projection_does_not_hide_serializer_failures():
    from infernux_mcp.operation_support import serializable_component

    def fail():
        raise RuntimeError("broken component document")

    component = SimpleNamespace(
        component_id=38,
        type_name="BrokenComponent",
        enabled=True,
        serialize_document=fail,
    )
    with pytest.raises(RuntimeError, match="broken component document"):
        serializable_component(component)


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


def test_input_semantic_ui_console_and_docs_are_schema_operations(tmp_path):
    operations, host = _operations(tmp_path)
    try:
        sent = operations["infernux.input.key"]("space", True)
        assert sent["delivered"] is True
        assert host.events == [("key", {"key": "space", "pressed": True, "repeat": False})]

        chord = operations["infernux.input.key.chord"](["ctrl", "s"])
        assert chord["delivered"] is True
        assert [event[1]["pressed"] for event in host.events[-4:]] == [True, True, False, False]
        assert [event[1]["key"] for event in host.events[-4:]] == ["ctrl", "s", "s", "ctrl"]

        held = operations["infernux.input.key.hold"]("w", duration_seconds=0.001)
        assert held["delivered"] is True
        assert held["release_sequence"] > held["press_sequence"]
        assert [event[1]["pressed"] for event in host.events[-2:]] == [True, False]
        assert [event[1]["key"] for event in host.events[-2:]] == ["w", "w"]

        clicked = operations["infernux.input.pointer.click"](40.0, 80.0)
        assert clicked["release_sequence"] > clicked["press_sequence"] > clicked["move_sequence"]
        assert [event[0] for event in host.events[-3:]] == [
            "pointer_move",
            "pointer_button",
            "pointer_button",
        ]
        assert operations["infernux.input.wait"](clicked["release_sequence"])["delivered"] is True

        snapshot = operations["infernux.ui.semantic.snapshot"](label="run")
        assert snapshot["frame"] == 27
        assert [item["semantic_id"] for item in snapshot["targets"]] == ["run"]
        assert snapshot["coordinates"]["framebuffer_scale"] == (2.0, 2.0)
        assert snapshot["coordinates"]["ui_scale"] == 1.0
        assert snapshot["mouse"] == (18.0, 24.0)
        waited = operations["infernux.ui.semantic.wait"](semantic_id="run")
        assert waited["matched"] is True

        console = operations["infernux.console.read"](limit=4, levels=["WARN"])
        assert console["entries"][0]["message"] == "native warning"
        assert console["status_bar"]["surface"] == "status_bar"

        docs = operations["infernux.docs.search"]("asset guid")
        assert docs["guides"][0]["id"] == "assets"
    finally:
        EditorAutomationHost.set_provider(None)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("timeout", [False, True])
def test_semantic_snapshot_does_not_change_continuous_capture(tmp_path, monkeypatch, continuous, timeout):
    from infernux_mcp import ui_operations

    operations, host = _operations(tmp_path)
    host.capture_enabled = continuous
    monkeypatch.setattr(host, "semantic_capture_enabled",
                        lambda enabled: pytest.fail("A one-shot snapshot changed continuous capture"))
    if timeout:
        monkeypatch.setattr(host, "request_semantic_snapshot", lambda: 10)
        clock = iter((0.0, 1.0))
        monkeypatch.setattr(ui_operations, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    try:
        if timeout:
            with pytest.raises(OperationError, match="not published"):
                operations["infernux.ui.semantic.snapshot"](label="run")
        else:
            snapshot = operations["infernux.ui.semantic.snapshot"](label="run")
            assert snapshot["request_sequence"] == 9
            assert snapshot["returned"] == 1
        assert host.capture_enabled is continuous
    finally:
        EditorAutomationHost.set_provider(None)


def test_semantic_snapshot_preserves_older_native_without_inventing_pixel_mapping(tmp_path):
    operations, host = _operations(tmp_path)
    native_snapshot = host.semantic_snapshot

    def legacy_snapshot():
        value = native_snapshot()
        value.pop("coordinates")
        return value

    host.semantic_snapshot = legacy_snapshot
    try:
        snapshot = operations["infernux.ui.semantic.snapshot"](label="run")
        assert [item["semantic_id"] for item in snapshot["targets"]] == ["run"]
        assert "coordinates" not in snapshot
    finally:
        EditorAutomationHost.set_provider(None)


def test_capture_returns_review_artifact_metadata_without_pixels(tmp_path):
    operations, host = _operations(tmp_path)
    try:
        requested = operations["infernux.capture.request"]("game", "review.png")
        assert requested["capture_id"] == 17
        assert requested["pixel_origin"] == "engine_render_target"
        assert requested["pixel_access"] is False
        status = operations["infernux.capture.status"](17)
        assert status["terminal"] is True
        assert status["byte_size"] == len(b"engine-render-target")
        assert "sha256" not in status
        assert "output_path" not in status
        assert "pixels" not in status
        assert host.capture_path.name == "review.png"

        # Reusing an artifact name starts a fresh lifecycle. A failed request
        # must not expose the previous PNG or report its byte count.
        host.capture_path.write_bytes(b"stale-frame")
        operations["infernux.capture.request"]("game", "review.png")
        assert not host.capture_path.exists()
        host.capture_status = lambda capture_id: {
            "capture_id": capture_id,
            "status": "failed",
            "output_path": str(host.capture_path),
            "error": "Capture source frame was not submitted before timeout",
        }
        failed = operations["infernux.capture.status"](17)
        assert failed["terminal"] is True
        assert "byte_size" not in failed

        editor_requested = operations["infernux.capture.request"]("editor", "editor.png")
        assert editor_requested["source"] == "editor"
        assert editor_requested["pixel_origin"] == "engine_render_target"
        assert host.capture_path.name == "editor.png"

        camera_requested = operations["infernux.capture.request"]("camera", "camera.png", 42)
        assert camera_requested["source"] == "camera"
        assert host.capture_source == ("camera", 42)
        for source, camera_id in (("camera", 0), ("game", 42), ("editor", 42)):
            with pytest.raises(OperationError, match="camera_component_id"):
                operations["infernux.capture.request"](source, "invalid.png", camera_id)
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


def test_operation_handlers_depend_on_host_api_not_editor_implementation():
    plugin = (
        Path(__file__).parents[2]
        / "external"
        / "plugins"
        / "infernux_mcp"
        / "package"
        / "editor"
        / "infernux_mcp"
    )
    forbidden = (
        "Infernux.engine",
        "Infernux.lib",
        "Infernux.core",
        "Infernux.components",
        "Infernux.particle",
    )
    violations = []
    for path in sorted(plugin.glob("*operations.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and str(node.module or "").startswith(forbidden):
                violations.append((path.name, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden):
                        violations.append((path.name, node.lineno, alias.name))
    assert violations == []


def test_editor_host_deduplicates_components_and_prefers_api_wrappers(monkeypatch):
    from Infernux.components.builtin_component import BuiltinComponent

    class RawLight:
        component_id = 7
        type_name = "HostTestLight"

    class RawPythonProxy:
        component_id = 8
        type_name = "PyComponentProxy"

    class PythonComponent:
        component_id = 8
        type_name = "Controller"

    wrapped = type(
        "WrappedLight",
        (),
        {
            "component_id": 7,
            "type_name": "HostTestLight",
            "_is_builtin_component_wrapper": True,
        },
    )()

    class WrapperType:
        @classmethod
        def _get_or_create_wrapper(cls, value, owner):
            assert isinstance(value, RawLight)
            assert owner is game_object
            return wrapped

    game_object = type(
        "GameObject",
        (),
        {
            "get_py_components": lambda self: [PythonComponent()],
            "get_components": lambda self: [RawLight(), RawPythonProxy()],
        },
    )()
    scene = type("Scene", (), {"find_by_id": lambda self, _id: game_object})()
    host = EditorAutomationHost()
    monkeypatch.setattr(host, "active_scene", lambda: scene)
    monkeypatch.setitem(
        BuiltinComponent._builtin_registry, "HostTestLight", WrapperType
    )

    values = host.scene_components(42)

    assert len(values) == 2
    assert isinstance(values[0], PythonComponent)
    assert values[1] is wrapped
    assert [value.component_id for value in values] == [8, 7]


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

    native_sources = list((Path(__file__).parents[2] / "cpp").rglob("*Capture*.cpp"))
    native_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in native_sources)
    forbidden_native = ("BitBlt(", "PrintWindow(", "GetDC(", "XGetImage(", "CGWindowListCreateImage")
    assert all(symbol not in native_text for symbol in forbidden_native)


def test_mcp_automation_cannot_control_operating_system_window_activation():
    plugin_root = (
        Path(__file__).parents[2]
        / "external"
        / "plugins"
        / "infernux_mcp"
        / "package"
        / "editor"
        / "infernux_mcp"
    )
    python_sources = list(plugin_root.rglob("*.py"))
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in python_sources
    )
    forbidden_symbols = (
        "SetForegroundWindow",
        "BringWindowToTop",
        "SetActiveWindow",
        "AttachThreadInput",
        "GetForegroundWindow",
        "win32gui",
        "pyautogui",
    )
    assert all(symbol not in source_text for symbol in forbidden_symbols)


def test_player_schema_exposes_managed_input_and_motion_capture(tmp_path, monkeypatch):
    operations, _host = _operations(tmp_path)

    class _Supervisor:
        def player_send_mouse_button(self, button, pressed, x, y, *, timeout_seconds):
            return {"button": button, "pressed": pressed, "x": x, "y": y}

        def player_press_key(self, key, duration_seconds, **kwargs):
            return {"key": key, "duration_seconds": duration_seconds, **kwargs}

        def player_motion_capture_arm(self, object_names, **kwargs):
            return {"capture_id": "motion-1", "object_names": object_names, **kwargs}

        def player_motion_capture_status(self, capture_id, *, timeout_seconds):
            return {"capture_id": capture_id, "status": "sampling"}

        def player_motion_capture_cancel(self, capture_id, *, timeout_seconds):
            return {"capture_id": capture_id, "cancelled": True}

    monkeypatch.setattr(
        "infernux_mcp.player_operations.SupervisorSession.attach_current_host",
        lambda *args, **kwargs: _Supervisor(),
    )
    from infernux_mcp import player_operations

    player_operations._LOCAL_SUPERVISORS.clear()
    session.configure(
        str(tmp_path),
        {
            "profile": "developer_assist",
            "session": {"build_profile": "debug_feedback"},
        },
    )
    try:
        pointer = operations["infernux.player.validation.pointer.button"](0, True, 10, 20)
        assert pointer == {"button": 0, "pressed": True, "x": 10, "y": 20}
        press = operations["infernux.player.validation.key.press"]("space", 0.2)
        assert press["duration_seconds"] == 0.2
        armed = operations["infernux.player.validation.motion.arm"](["Ball"])
        assert armed["capture_id"] == "motion-1"
        assert operations["infernux.player.validation.motion.status"]("motion-1")["status"] == "sampling"
        assert operations["infernux.player.validation.motion.cancel"]("motion-1")["cancelled"] is True
    finally:
        player_operations._LOCAL_SUPERVISORS.clear()
        EditorAutomationHost.set_provider(None)


def test_player_build_is_available_without_global_validation(tmp_path):
    session.configure(
        str(tmp_path),
        {
            "profile": "developer_assist",
            "session": {"build_profile": "debug_feedback"},
        },
    )
    queue = MainThreadCommandQueue()
    queue._main_thread_id = threading.get_ident()
    MainThreadCommandQueue._instance = queue
    host = _AutomationHost(tmp_path / "capture.png")
    received = {}

    def build_player(project_root, **arguments):
        received.update({"project_root": project_root, **arguments})
        return {"output_dir": str(tmp_path / "Build"), "executable_exists": True}

    host.build_player = build_player
    host.player_build_targets = lambda: {
        "current_host_target": "windows-x64",
        "targets": [{"id": "windows-x64"}, {"id": "android-arm64"}],
    }
    EditorAutomationHost.set_provider(host)
    try:
        operations = {item.schema.id: item.handler for item in build_operations(str(tmp_path))}
        targets = operations["infernux.player.targets"]()
        result = operations["infernux.player.build"](
            target="android-arm64",
            game_name="BalanceBall",
            debug_mode=True,
            android_artifact="aab",
            compress_resources=True,
        )
        assert [item["id"] for item in targets["targets"]] == [
            "windows-x64",
            "android-arm64",
        ]
        assert result["executable_exists"] is True
        assert received["project_root"] == str(tmp_path)
        assert received["game_name"] == "BalanceBall"
        assert received["debug_mode"] is True
        assert received["target"] == "android-arm64"
        assert received["android_artifact"] == "aab"
        assert received["compress_resources"] is True
    finally:
        EditorAutomationHost.set_provider(None)
