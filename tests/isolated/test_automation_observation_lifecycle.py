"""Real HTTP/SDK/MCP lifecycle checks without creating a renderer.

Use the existing engine native module for DocumentStore and package containers;
do not replace engine handlers/transport with mocks.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import pytest

from infernux.host.bridge import AutomationBridge
from infernux.host.commands import MainThreadCommandQueue
from infernux.host.editor import EditorAutomationHost
from infernux.host.editor_operations import build_editor_operations, install_editor_operations
from infernux.host.operation_support import operation
from infernux.host.operations import OperationKind, OperationRegistry
from infernux.host.queries import emit_signal
from infernux_automation import AutomationClient, AutomationError


@pytest.fixture
def mcp_server_factory(monkeypatch):
    engine_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(engine_root / "external/plugins/infernux_mcp/package/editor"))
    from infernux_mcp.server import create_server

    return create_server


@contextmanager
def host(project, *, lease=90):
    registry = OperationRegistry()
    state = {"value": 1}
    registry.register(operation("test.audit.read", OperationKind.QUERY, "Read mutable state",
                                lambda: state, capability="test.read"))
    bridge = AutomationBridge(project, registry=registry, session_lease_seconds=lease)
    try:
        yield bridge, AutomationClient(project), state
    finally:
        bridge.close()


def ready(engine):
    watch = engine.observe("test.audit.read", signals=["audit.changed"])
    snapshot = engine.observation(watch["id"], after=0)
    assert snapshot["value"] == {"value": 1}
    return watch["id"]


def test_turn_history_two_consumers_quota_and_stop_are_one_lifecycle(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        watch = ready(engine)
        report = engine.report(watch, "chat-A", reports=3, mode="turns")
        other = engine.report(watch, "chat-B", reports=1, mode="turns")
        assert engine.reports("chat-A", cycle_id="A")[0]["remaining"] == 2
        assert engine.reports("chat-A", cycle_id="B")[0]["remaining"] == 1
        assert engine.reports("chat-A", cycle_id="A") == []
        assert engine.reports("chat-B", cycle_id="A")[0]["remaining"] == 0
        assert watch in bridge.queries._queries  # chat A still owns a report.
        assert engine.reports("chat-A", cycle_id="C")[0]["remaining"] == 0
        assert watch not in bridge.queries._queries
        assert not bridge.reports._reports
        assert engine.reports("chat-A", cycle_id="D") == []
        watch = ready(engine)
        engine.report(watch, "chat-A", reports=2)
        assert engine.stop_observing(watch)
        assert engine.reports("chat-A") == []


def test_signals_snapshot_mutable_results_and_same_turn_do_not_double_count(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        watch = ready(engine)
        engine.report(watch, "chat", reports=3, mode="changes")
        first = engine.reports("chat", cycle_id="actual-turn")
        state["value"] = 7
        emit_signal("audit.changed")
        changed = engine.observation(watch, after=1, timeout=2)
        assert changed["value"] == {"value": 7} and changed["revision"] == 2
        assert engine.reports("chat", cycle_id="actual-turn") == []
        assert engine.reports("chat", cycle_id="next-turn")[0]["remaining"] == 1


def test_turn_reports_include_pending_real_state_without_extending_the_window(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        release = Event()
        bridge.registry.register(operation("test.audit.pending", OperationKind.QUERY, "Pending read",
                                            lambda: release.wait(3) and {"ready": True}, capability="test.read"))
        try:
            watch = engine.observe("test.audit.pending")
            engine.report(watch["id"], "chat", reports=2, mode="turns")
            report = engine.reports("chat", cycle_id="actual-turn")
            assert report[0]["observation"]["status"] == "pending"
            assert report[0]["remaining"] == 1
            assert engine.reports("chat", cycle_id="actual-turn") == []
        finally:
            release.set()


def test_provider_replacement_retires_query_instead_of_rebinding_old_observation(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        watch = ready(engine)
        engine.report(watch, "chat", reports=4, mode="changes")
        replacement = operation("test.audit.read", OperationKind.QUERY, "A different current provider",
                                lambda: {"value": 99}, capability="test.read")
        bridge.registry.register(replacement)
        snapshot = engine.observation(watch, after=1, timeout=2)
        assert snapshot["status"] == "closed" and snapshot["value"] is None
        assert "retired" in snapshot["error"]
        reports = engine.reports("chat")
        assert reports[0]["observation"]["status"] == "closed"
        assert watch not in bridge.queries._queries


def test_terminal_capture_releases_signal_and_report_lifetimes(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        reads = []
        def capture_status():
            reads.append(1)
            return {"status": "completed", "terminal": True, "output_path": "published.png"}
        bridge.registry.register(operation("infernux.capture.status", OperationKind.QUERY, "Completed capture",
                                            capture_status, capability="capture.read"))
        watch = engine.observe("infernux.capture.status")
        snapshot = engine.observation(watch["id"], after=0)
        assert snapshot["status"] == "ready" and snapshot["terminal"]
        assert not bridge.queries.has_subscribers("capture.completed")
        emit_signal("capture.completed")
        assert reads == [1]
        engine.report(watch["id"], "chat", reports=9)
        reports = engine.reports("chat")
        assert reports[0]["observation"]["value"]["status"] == "completed"
        assert watch["id"] not in bridge.queries._queries and not bridge.reports._reports


def test_expired_dead_client_releases_observations_without_frame_polling(tmp_path):
    with host(tmp_path, lease=0.1) as (bridge, engine, state):
        stopped = Event()
        bridge.queries.add_stop_listener(lambda _: stopped.set())
        watch = ready(engine)
        engine.report(watch, "disconnected", reports=9)
        assert stopped.wait(3), "An unreachable client must not leave a permanent world observer"
        assert watch not in bridge.queries._queries
        assert not bridge.reports._reports


def test_cli_and_external_harness_use_same_engine_operations(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        result = subprocess.run([sys.executable, "-m", "infernux_automation", "--project", str(tmp_path),
                                 "call", "test.audit.read"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"value": 1}


def test_real_prompt_hook_requires_an_event_id_and_deduplicates_retries(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        watch = ready(engine)
        engine.report(watch, "claude:session", reports=3, mode="turns")
        def hook(identity=None):
            event = {"session_id": "session", "cwd": str(tmp_path), "prompt": "same actual prompt"}
            if identity is not None:
                event["request_id"] = identity
            result = subprocess.run([sys.executable, "-m", "infernux_automation.claude_hook", "--project", str(tmp_path)],
                                    input=json.dumps(event), capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            return json.loads(json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"])
        unknown = hook()
        assert unknown["cycle_id"] is None and unknown["reports"] == []
        assert "Unavailable" in unknown["turn_delivery"]
        assert hook("real-A")["reports"][0]["remaining"] == 2
        assert hook("real-A")["reports"] == []
        assert hook("real-B")["reports"][0]["remaining"] == 1
        assert hook("real-A")["reports"] == []


def test_mcp_two_sessions_disconnect_and_bad_observe_do_not_leak(tmp_path, mcp_server_factory):
    from fastmcp import Client

    async def scenario(bridge, engine):
        service = mcp_server_factory(tmp_path)
        async with Client(service) as second:
            async with Client(service) as first:
                a = await first.call_tool("observe", {"operation": "test.audit.read", "arguments": {}})
                b = await second.call_tool("observe", {"operation": "test.audit.read", "arguments": {}})
                assert len(bridge.queries._queries) == 2
                with pytest.raises(Exception):
                    await first.call_tool("observe", {"operation": "test.audit.read", "arguments": {}, "mode": "invalid"})
                assert len(bridge.queries._queries) == 2
            assert a.data["observation"]["id"] not in bridge.queries._queries
            assert b.data["observation"]["id"] in bridge.queries._queries
            value = await second.call_tool("operation_call", {"operation": "test.audit.read", "arguments": {}})
            assert value.data.get("result", value.data) == {"value": 1}
        assert not bridge.queries._queries and not bridge.reports._reports

    with host(tmp_path) as (bridge, engine, state):
        asyncio.run(scenario(bridge, engine))


def test_streamable_http_session_delete_releases_only_that_clients_queries(tmp_path, mcp_server_factory):
    import socket
    from time import monotonic
    import uvicorn
    from fastmcp import Client

    with host(tmp_path) as (bridge, engine, state):
        channel = socket.socket()
        channel.bind(("127.0.0.1", 0))
        channel.listen()
        endpoint = f"http://127.0.0.1:{channel.getsockname()[1]}/mcp"
        service = mcp_server_factory(tmp_path)
        server = uvicorn.Server(uvicorn.Config(service.http_app(path="/mcp"), log_level="warning"))
        thread = Thread(target=lambda: server.run(sockets=[channel]), daemon=True)
        thread.start()
        deadline = monotonic() + 5
        while not server.started and monotonic() < deadline:
            Event().wait(0.01)
        assert server.started
        closed = {}
        bridge.queries.add_stop_listener(lambda identity: closed[identity].set() if identity in closed else None)
        async def scenario():
            async with Client(endpoint) as second:
                async with Client(endpoint) as first:
                    a = await first.call_tool("observe", {"operation": "test.audit.read", "arguments": {}})
                    b = await second.call_tool("observe", {"operation": "test.audit.read", "arguments": {}})
                    closed[a.data["observation"]["id"]] = Event()
                    closed[b.data["observation"]["id"]] = Event()
                    assert len(bridge.queries._queries) == 2
                assert await asyncio.to_thread(closed[a.data["observation"]["id"]].wait, 2)
                assert a.data["observation"]["id"] not in bridge.queries._queries
                assert b.data["observation"]["id"] in bridge.queries._queries
            assert await asyncio.to_thread(closed[b.data["observation"]["id"]].wait, 2)
            assert not bridge.queries._queries and not bridge.reports._reports
        try:
            asyncio.run(scenario())
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            channel.close()
        assert not thread.is_alive()


def test_disconnect_keeps_running_job_handle_and_applied_effect(tmp_path, mcp_server_factory):
    from fastmcp import Client

    with host(tmp_path) as (bridge, engine, state):
        started, finish = Event(), Event()
        def work():
            started.set()
            assert finish.wait(5)
            state["value"] = 17
            return dict(state)
        bridge.registry.register(operation("test.audit.work", OperationKind.WORKFLOW, "Owned job",
                                            work, capability="test.write"))
        async def scenario():
            async with Client(mcp_server_factory(tmp_path)) as client:
                result = await client.call_tool("operation_submit", {"operation": "test.audit.work", "arguments": {}})
                return result.data["job_id"]
        try:
            job = asyncio.run(scenario())
            assert started.wait(2)
            assert engine.job(job)["done"] is False
            assert engine.cancel(job) is False
            finish.set()
            bridge.jobs._jobs[job].future.result(timeout=3)
            assert engine.job(job)["result"] == {"value": 17}
        finally:
            finish.set()


def test_timed_out_world_write_is_applied_once_and_never_replayed(tmp_path):
    with host(tmp_path) as (bridge, engine, state):
        release, started = Event(), Event()
        applications = []
        def write():
            applications.append(1)
            state["value"] = 23
            started.set()
            release.wait(3)
            return dict(state)
        bridge.registry.register(operation("test.audit.write", OperationKind.COMMAND, "One world write",
                                            write, capability="test.write"))
        impatient = AutomationClient(tmp_path, timeout=0.1)
        try:
            with pytest.raises(AutomationError) as error:
                impatient.call("test.audit.write")
            assert error.value.code == "transport.unknown_outcome"
            assert started.is_set()
            assert engine.call("test.audit.read") == {"value": 23}
            assert applications == [1]
        finally:
            release.set()


def test_bridge_shutdown_finishes_all_cleanup_and_keeps_newer_instance_descriptor(tmp_path):
    first = AutomationBridge(tmp_path, registry=OperationRegistry())
    second = AutomationBridge(tmp_path, registry=OperationRegistry())
    completed = []
    first._cleanups.extend([lambda: completed.append("framework"),
                             lambda: (_ for _ in ()).throw(RuntimeError("user cleanup failed"))])
    try:
        with pytest.raises(RuntimeError, match="cleanup errors"):
            first.close()
        assert completed == ["framework"]
        assert AutomationClient(tmp_path).info()["instance_id"] == second.context.instance_id
    finally:
        first.close()
        second.close()


def test_public_plugin_operations_publish_update_and_uninstall_the_real_package(tmp_path, mcp_server_factory):
    from infernux.plugins import InxPackage, PluginManager
    from infernux.plugins.cache import SharedPackageCache
    project = tmp_path / "project"
    project.mkdir()
    source = tmp_path / "author/package"
    (source / "runtime").mkdir(parents=True)
    manifest = source / "inx_package.json"
    metadata = {"reference": "audit/harness", "name": "Audit harness", "version": "1.0.0", "engine": ">=0.4.8"}
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    (source / "runtime/value.txt").write_text("publisher 1", encoding="utf-8")
    (source / "runtime/removed.txt").write_text("old", encoding="utf-8")
    first = InxPackage.export_source(str(source), str(tmp_path / "first.inxpkg"))
    manager = PluginManager(str(project), runtime=True)
    manager._shared_package_cache = SharedPackageCache(tmp_path / "library")
    class Provider(EditorAutomationHost):
        def plugin_manager(self):
            return manager
    previous = EditorAutomationHost._provider
    EditorAutomationHost.set_provider(Provider())
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    registry = OperationRegistry()
    install_editor_operations(str(project), registry)
    bridge = AutomationBridge(project, registry=registry)
    def invoke(identifier, arguments):
        from fastmcp import Client
        async def scenario():
            async def owner_ticks():
                while True:
                    queue.drain()
                    await asyncio.sleep(0.001)
            tick = asyncio.create_task(owner_ticks())
            try:
                async with Client(mcp_server_factory(project)) as client:
                    result = await client.call_tool("operation_call", {"operation": identifier, "arguments": arguments})
                    return result.data
            finally:
                tick.cancel()
                await asyncio.gather(tick, return_exceptions=True)
        return asyncio.run(scenario())
    try:
        invoke("infernux.plugin.install", {"source": first.package_path, "install_dependencies": False})
        installed = project / "Packages/audit/harness/runtime"
        (installed / "value.txt").write_text("local publisher edit", encoding="utf-8")
        (installed / "user.txt").write_text("user keeps this", encoding="utf-8")
        metadata["version"] = "2.0.0"
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        (source / "runtime/value.txt").write_text("publisher 2", encoding="utf-8")
        (source / "runtime/removed.txt").unlink()
        second = InxPackage.export_source(str(source), str(tmp_path / "second.inxpkg"))
        invoke("infernux.plugin.update", {"reference": "audit/harness", "source": second.package_path})
        assert (installed / "value.txt").read_text() == "publisher 2"
        assert not (installed / "removed.txt").exists()
        assert (installed / "user.txt").read_text() == "user keeps this"
        invoke("infernux.plugin.enabled.set", {"reference": "audit/harness", "enabled": False})
        assert manager.registry.installed_record("audit/harness")["enabled"] is False
        invoke("infernux.plugin.uninstall", {"reference": "audit/harness"})
        assert not (installed / "value.txt").exists()
        assert (installed / "user.txt").is_file()
    finally:
        bridge.close()
        manager.shutdown()
        queue.release_owner()
        EditorAutomationHost.set_provider(previous)


def test_optional_harness_is_an_ordinary_plugin_with_owned_public_operations(tmp_path, monkeypatch):
    from infernux.plugins import InxPackage, PluginManager
    from infernux.plugins.cache import SharedPackageCache
    source = tmp_path / "source/package"
    (source / "runtime").mkdir(parents=True)
    (source / "inx_package.json").write_text(json.dumps({"reference": "audit/optional-harness",
        "name": "Optional harness", "version": "1.0.0", "engine": ">=0.4.8"}), encoding="utf-8")
    (source / "runtime/provider.py").write_text('''from infernux.lifecycle import InxPreload
from infernux.host import OperationRegistry, OperationKind
from infernux.host.operation_support import operation
from infernux_automation import AutomationClient

class OptionalHarness(InxPreload):
    def preload(self, context):
        OperationRegistry.instance().register(operation(
            "audit.harness.read", OperationKind.QUERY, "Read ordinary harness state",
            lambda: {"connected": True, "sdk": AutomationClient.__module__},
            owner=context.package_reference, capability="harness.read", thread="any"))
''', encoding="utf-8")
    package = InxPackage.export_source(str(source), str(tmp_path / "harness.inxpkg"))
    project = tmp_path / "project"
    project.mkdir()
    registry = OperationRegistry()
    monkeypatch.setattr(OperationRegistry, "_instance", registry)
    manager = PluginManager(str(project), runtime=True)
    manager._shared_package_cache = SharedPackageCache(tmp_path / "library")
    bridge = AutomationBridge(project, registry=registry)
    try:
        manager.install_package(package.package_path, install_dependencies=False)
        engine = AutomationClient(project)
        assert engine.call("audit.harness.read") == {"connected": True, "sdk": "infernux_automation.client"}
        watch = engine.observe("audit.harness.read")
        engine.observation(watch["id"], after=0)
        manager.uninstall("audit/optional-harness")
        with pytest.raises(AutomationError) as error:
            engine.call("audit.harness.read")
        assert error.value.code == "operation.not_found"
        assert engine.observation(watch["id"], after=1, timeout=2)["status"] == "closed"
    finally:
        bridge.close()
        manager.shutdown()
