"""The public Host works without an MCP plugin and uses real signal delivery."""

from concurrent.futures import ThreadPoolExecutor
from threading import Condition, Event

from infernux.host.bridge import AutomationBridge
from infernux.host.operation_support import operation
from infernux.host.operations import OperationError, OperationJobRegistry, OperationKind, OperationRegistry
from infernux.host.queries import emit_signal
from infernux_automation import AutomationClient, AutomationError
import pytest


def test_batches_signal_queries_and_a_self_retiring_command_returns_its_result(engine, tmp_path):
    from dataclasses import replace
    registry = OperationRegistry()
    state = {"value": 1}
    registry.register(operation("test.batch.read", OperationKind.QUERY, "Read batch state",
                                lambda: dict(state), capability="test.read"))
    def retire():
        state["value"] = 9
        registry.unregister_owner("self-retiring")
        return {"finished": True}
    registry.register(replace(operation("test.batch.retire", OperationKind.COMMAND, "Retire this provider",
                                        retire, capability="test.write"), owner="self-retiring"))
    bridge = AutomationBridge(tmp_path, registry=registry)
    try:
        client = AutomationClient(tmp_path)
        watch = client.observe("test.batch.read")
        initial = client.observation(watch["id"], after=0)
        result = client.request("batch", calls=[{"operation": "test.batch.retire", "arguments": {}}])
        assert result[0]["result"] == {"finished": True}
        changed = client.observation(watch["id"], after=initial["revision"])
        assert changed["value"] == {"value": 9}
    finally:
        bridge.close()


def test_a_source_provider_is_unavailable_until_activation(engine):
    from infernux.engine.reload.owners import ReloadOwner
    from infernux.host.operations import OperationError
    registry = OperationRegistry()
    owner = ReloadOwner(())
    with owner.enter():
        registry.register(operation("test.initializing.read", OperationKind.QUERY, "Read current provider",
                                    lambda: {"ready": True}, capability="test.read"))
    with pytest.raises(OperationError, match="unavailable"):
        registry.execute("test.initializing.read", capabilities=("*",))
    owner.active = True
    assert registry.execute("test.initializing.read", capabilities=("*",)) == {"ready": True}
    owner.close()
    with pytest.raises(OperationError, match="Unknown operation"):
        registry.get("test.initializing.read")


def test_public_bridge_queries_jobs_and_conversation_cycles(engine, tmp_path):
    registry = OperationRegistry()
    state = {"value": 1, "reads": 0}

    def read():
        state["reads"] += 1
        return {"value": state["value"]}

    registry.register(operation("test.value.read", OperationKind.QUERY, "Read a value", read, capability="test.read"))
    registry.register(operation("test.value.set", OperationKind.COMMAND, "Set a value",
                                lambda value: state.update(value=value) or {"value": value}, capability="test.write",
                                input_properties={"value": {"type": "integer"}}, required=("value",)))
    bridge = AutomationBridge(tmp_path, registry=registry)
    try:
        client = AutomationClient(tmp_path)
        assert client.info()["instance_id"] == bridge.context.instance_id
        assert client.call("test.value.read") == {"value": 1}
        watch = client.observe("test.value.read", signals=("value.changed",))
        initial = client.observation(watch["id"], after=0)
        assert initial["value"] == {"value": 1}
        reads = state["reads"]
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(client.observation, watch["id"], after=initial["revision"], timeout=2.0)
            assert client.call("test.value.set", {"value": 7}) == {"value": 7}
            emit_signal("value.changed")
            changed = waiting.result(timeout=5)
        assert changed["value"] == {"value": 7}
        assert state["reads"] == reads + 1
        report = client.request("report_start", query_id=watch["id"], consumer="chat", reports=2, mode="turns")
        assert client.request("reports", consumer="chat") == []
        first = client.request("reports", consumer="chat", cycle_id="actual-user-turn-1")
        assert first[0]["remaining"] == 1
        assert client.request("reports", consumer="chat", cycle_id="actual-user-turn-1") == []
        assert client.request("reports", consumer="chat", cycle_id="actual-user-turn-2")[0]["remaining"] == 0
        # Exhausting a managed report retires its subscription immediately.
        assert not client.request("report_stop", report_id=report["id"])
        assert not client.stop_observing(watch["id"])
        stale = AutomationClient(endpoint=client.endpoint, instance_id="different-instance")
        with pytest.raises(AutomationError, match="no longer current"):
            stale.call("test.value.set", {"value": 100})
        assert state["value"] == 7
    finally:
        bridge.close()
    assert not (tmp_path / "Cache/Automation/engine.json").exists()


def test_queued_job_cancellation_reports_the_actual_job_state(engine, tmp_path):
    registry = OperationRegistry()
    release = Event()
    started = Event()

    def work():
        started.set()
        release.wait(5)
        return {"finished": True}

    registry.register(operation("test.long.work", OperationKind.WORKFLOW, "Do work", work, capability="test.write"))
    bridge = AutomationBridge(tmp_path, registry=registry)
    try:
        client = AutomationClient(tmp_path)
        job_id = client.submit("test.long.work")["job_id"]
        assert started.wait(2)
        assert client.cancel(job_id) is False
        assert client.job(job_id)["done"] is False
        release.set()
        # Future completion is a real event; do not measure it with sleeps.
        bridge.jobs._jobs[job_id].future.result(timeout=2)
        assert client.job(job_id)["result"] == {"finished": True}
    finally:
        release.set()
        bridge.close()


@pytest.mark.parametrize("wrapped_execution", [False, True])
def test_queued_job_never_runs_a_replacement_provider(engine, wrapped_execution):
    from dataclasses import replace
    registry = OperationRegistry()
    release = Event()
    started = Event()
    invoked = []

    def block():
        started.set()
        assert release.wait(5)
        return {"finished": True}

    registry.register(operation("test.blocker.work", OperationKind.WORKFLOW, "Hold the worker", block,
                                capability="test.write"))
    provider = replace(operation("test.plugin.work", OperationKind.COMMAND, "Invoke the plugin",
                                 lambda: invoked.append("old") or {}, capability="test.write"), owner="test/plugin")
    registry.register(provider)
    jobs = OperationJobRegistry(registry, max_workers=1)
    wrappers = []

    def execute(*args, **kwargs):
        wrappers.append(kwargs["expected_provider"])
        return registry.execute(*args, **kwargs)

    try:
        jobs.submit("test.blocker.work", capabilities=("*",))
        assert started.wait(2)
        identity = jobs.submit("test.plugin.work", capabilities=("*",),
                               execute=execute if wrapped_execution else None)
        assert not jobs.status(identity)["done"]
        registry.unregister_owner("test/plugin")
        registry.register(replace(provider, handler=lambda: invoked.append("replacement") or {}))
        release.set()
        with pytest.raises(OperationError, match="provider was retired"):
            jobs._jobs[identity].future.result(timeout=2)
        assert jobs.status(identity)["error"]["code"] == "operation.provider_changed"
        assert invoked == []
        if wrapped_execution:
            assert wrappers == [provider]
    finally:
        release.set()
        jobs.shutdown()


def test_sdk_queued_job_fails_when_plugin_is_unloaded_and_reenabled(engine, tmp_path):
    from dataclasses import replace
    registry = OperationRegistry()
    release = Event()
    started = Condition()
    workers = []
    invoked = []

    def block():
        with started:
            workers.append(True)
            started.notify_all()
        assert release.wait(5)
        return {"finished": True}

    registry.register(operation("test.blocker.work", OperationKind.WORKFLOW, "Hold a job worker", block,
                                capability="test.write"))
    provider = replace(operation("test.plugin.work", OperationKind.COMMAND, "Invoke a plugin",
                                 lambda: invoked.append("old") or {}, capability="test.write"), owner="test/plugin")
    registry.register(provider)
    bridge = AutomationBridge(tmp_path, registry=registry)
    try:
        with AutomationClient(tmp_path) as client:
            for _ in range(4):
                client.submit("test.blocker.work")
            with started:
                assert started.wait_for(lambda: len(workers) == 4, timeout=2)
            identity = client.submit("test.plugin.work")["job_id"]
            assert not client.job(identity)["done"]
            registry.unregister_owner("test/plugin")
            registry.register(replace(provider, handler=lambda: invoked.append("replacement") or {}))
            release.set()
            with pytest.raises(OperationError, match="provider was retired"):
                bridge.jobs._jobs[identity].future.result(timeout=2)
            assert client.job(identity)["error"]["code"] == "operation.provider_changed"
            assert invoked == []
    finally:
        release.set()
        bridge.close()


def test_independent_service_source_stop_releases_real_host_queries_and_keeps_running_job(engine, tmp_path):
    import asyncio
    import os
    from pathlib import Path
    import shutil
    import sys
    import infernux_automation
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    registry = OperationRegistry()
    release = Event()
    started = Event()
    writes = []

    def work():
        writes.append("submitted-effect")
        started.set()
        assert release.wait(15)
        return {"finished": True}

    registry.register(operation("test.service.work", OperationKind.WORKFLOW, "A real engine job", work,
                                capability="test.write"))
    registry.register(operation("test.service.read", OperationKind.QUERY, "Read engine state",
                                lambda: {"writes": list(writes)}, capability="test.read"))
    registry.register(operation("test.service.write", OperationKind.COMMAND, "A later service call",
                                lambda: writes.append("stale-call") or {}, capability="test.write"))
    modules = tmp_path / "independent-modules"
    source = Path(__file__).resolve().parents[2] / "external/plugins/infernux_mcp/package/editor/infernux_mcp"
    for name, origin in (("infernux_mcp", source),
                         ("infernux_automation", Path(infernux_automation.__file__).parent)):
        shutil.copytree(origin, modules / name, ignore=shutil.ignore_patterns("__pycache__"))
    bridge = AutomationBridge(tmp_path / "project", registry=registry)
    environment = dict(os.environ, PYTHONPATH=str(modules), INFERNUX_NATIVE_MODULE_DIR=str(tmp_path / "no-native"))
    log_path = tmp_path / "independent-service.log"

    async def exercise():
        with log_path.open("w", encoding="utf-8") as log:
            transport = StdioTransport(command=sys.executable,
                args=["-m", "infernux_mcp", "--project", str(tmp_path / "project")], env=environment, log_file=log)
            async with Client(transport, timeout=3) as client:
                job_id = (await client.call_tool("operation_submit", {"operation": "test.service.work", "arguments": {}})).data["job_id"]
                assert await asyncio.to_thread(started.wait, 2)
                watch = (await client.call_tool("observe", {"operation": "test.service.read", "arguments": {}})).data
                observed = await client.call_tool("observation", {"query_id": watch["observation"]["id"], "after": 0})
                assert observed.data["value"] == {"writes": ["submitted-effect"]}
                (modules / "infernux_mcp/server.py").write_text("def broken(:\n", encoding="utf-8")
                def queries_closed():
                    with bridge.queries._condition:
                        return bridge.queries._condition.wait_for(lambda: not bridge.queries._queries, timeout=4)
                assert await asyncio.to_thread(queries_closed)
                with pytest.raises(Exception):
                    await asyncio.wait_for(client.call_tool("operation_call", {
                        "operation": "test.service.write", "arguments": {}}), 3)
                with AutomationClient(tmp_path / "project") as sdk:
                    assert sdk.job(job_id)["done"] is False
                    release.set()
                    await asyncio.to_thread(bridge.jobs._jobs[job_id].future.result, 2)
                    assert sdk.job(job_id)["result"] == {"finished": True}
                assert writes == ["submitted-effect"]
    try:
        asyncio.run(exercise())
        assert "restart to load current code" in log_path.read_text(encoding="utf-8")
    finally:
        release.set()
        bridge.close()
