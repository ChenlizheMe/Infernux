"""Every executed gateway operation must survive in its owning saved attempt."""
import json
from threading import Event
import time

import pytest

from infernux.host import OperationError, OperationJobRegistry, OperationKind
from infernux_mcp import adapter, session, trace
from infernux_mcp.operation_support import operation


class GatewayRecorder:
    def __init__(self):
        self.tools = {}

    def tool(self, *, name):
        def register(fn):
            self.tools[name] = fn
            return fn
        return register


@pytest.fixture
def gateway(tmp_path):
    (tmp_path / "Assets").mkdir()
    transport = GatewayRecorder()
    adapter.register_gateways(transport, str(tmp_path), {
        "profile": "global_validation", "granted_capabilities": ["session.*"],
        "session": {"managed_checkpoints_required": False},
        "features": {"trace_recorder": True, "session_call_log": True},
    })
    adapter._jobs.shutdown()
    adapter._jobs = OperationJobRegistry(adapter._registry, max_workers=1)
    try:
        yield transport.tools, tmp_path
    finally:
        adapter.shutdown_adapter()
        trace.stop_trace(str(tmp_path), save=False)


def start(tools, task="first"):
    result = tools["operation_command_execute"]("infernux.mcp.attempt.start", {"task": task, "checkpoint": "baseline"})
    assert result["ok"], result
    return result["data"]["result"]


def stop(tools, project):
    result = tools["operation_command_execute"]("infernux.mcp.attempt.stop", {})
    assert result["ok"], result
    saved = json.loads((project / result["data"]["result"]["trace_path"]).read_text(encoding="utf-8"))
    assert saved == trace.last_trace()["trace"]
    assert [step["index"] for step in saved["steps"]] == list(range(len(saved["steps"])))
    return saved


def await_job(tools, job):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = tools["operation_job_status"](job)
        assert result["ok"], result
        if result["data"]["done"]:
            return result["data"]
        time.sleep(.005)
    pytest.fail("operation job did not finish")


def invoke(tools, route, name, arguments):
    if route == "direct":
        return tools["operation_execute"](name, arguments)
    if route == "batch":
        return tools["operation_batch_execute"]([{"operation": name, "arguments": arguments}])["data"]["results"][0]
    result = tools["operation_job_submit"](name, arguments)
    assert result["ok"], result
    return await_job(tools, result["data"]["job_id"])


@pytest.mark.parametrize("route", ["direct", "batch", "job"])
@pytest.mark.parametrize("failure", [False, True])
def test_gateway_records_real_checkpoint_operation_once(gateway, route, failure):
    tools, project = gateway
    start(tools)
    name = "infernux.mcp.checkpoint.status" if failure else "infernux.mcp.checkpoint.list"
    invoke(tools, route, name, {})  # Missing required checkpoint exercises real argument validation.
    saved = stop(tools, project)
    steps = [step for step in saved["steps"] if step["operation"] == name]
    assert len(steps) == 1
    assert steps[0]["ok"] is not failure


@pytest.mark.parametrize("stop_on_error,expected", [(True, [True, False]), (False, [True, False, True])])
def test_batch_preserves_step_order_and_stop_policy(gateway, stop_on_error, expected):
    tools, project = gateway
    start(tools)
    calls = [{"operation": "infernux.mcp.checkpoint.list"}, {"operation": "test.missing"},
             {"operation": "infernux.mcp.checkpoint.list"}]
    result = tools["operation_batch_execute"](calls, stop_on_error)
    assert [value["ok"] for value in result["data"]["results"]] == expected
    steps = [step for step in stop(tools, project)["steps"] if step["operation"] != "infernux.mcp.attempt.start"]
    assert [step["ok"] for step in steps] == expected
    assert [step["operation"] for step in steps] == [call["operation"] for call in calls[:len(expected)]]


def test_pending_and_cancelled_jobs_cannot_disappear_or_cross_attempts(gateway):
    tools, project = gateway
    entered, release = Event(), Event()
    def work():
        entered.set()
        assert release.wait(5)
        return {"completed": True}
    name = "test.trace.blocking"
    adapter._registry.register(operation(name, OperationKind.QUERY, "Controlled work", work, capability="session.read"))
    first = start(tools)
    try:
        running = tools["operation_job_submit"](name, {})["data"]["job_id"]
        assert entered.wait(3)
        queued = tools["operation_job_submit"]("infernux.mcp.checkpoint.list", {})["data"]["job_id"]
        refused = tools["operation_command_execute"]("infernux.mcp.attempt.stop", {})
        assert not refused["ok"] and refused["error"]["code"] == "trace.pending_operations"
        assert session.current().attempt_active
        assert not list((project / ".infernux/mcp_traces").glob("*.json"))
        assert tools["operation_job_cancel"](queued)["data"]["cancelled"]
    finally:
        release.set()
    assert await_job(tools, running)["result"] == {"completed": True}
    saved = stop(tools, project)
    cancelled = [step for step in saved["steps"] if step.get("status") == "cancelled"]
    assert len(cancelled) == 1 and cancelled[0]["operation"] == "infernux.mcp.checkpoint.list"
    assert cancelled[0]["ok"] is False
    assert saved["context"]["attempt_id"] == first["attempt_id"]
    start(tools, "second")
    second = stop(tools, project)
    assert not any(step["operation"] in {name, "infernux.mcp.checkpoint.list"} for step in second["steps"])


@pytest.mark.parametrize("route", ["direct", "batch", "job"])
@pytest.mark.parametrize("fail", [False, True])
def test_trace_and_session_log_redact_secrets_on_every_route(gateway, route, fail):
    tools, project = gateway
    secret = "private-test-credential-739"
    name = "test.trace.secret"
    def work(password):
        if fail:
            raise OperationError("test.failed", "Cannot use password " + password)
        return {"access_token": password, "nested": {"lease": password}}
    adapter._registry.register(operation(name, OperationKind.QUERY, "Secret result", work,
        capability="session.read", input_properties={"password": {"type": "string"}}, required=("password",)))
    start(tools)
    invoke(tools, route, name, {"password": secret})
    saved = stop(tools, project)
    assert secret not in json.dumps(saved)
    log = trace.read_session_log(str(project))
    assert secret not in json.dumps(log)
    assert len([step for step in saved["steps"] if step["operation"] == name]) == 1
    assert len([step for step in log["entries"] if step.get("operation") == name]) == 1


@pytest.mark.parametrize("route", ["direct", "batch", "job"])
def test_permission_failures_are_recorded_without_running_handler(gateway, route):
    tools, project = gateway
    def forbidden():
        pytest.fail("permission-denied handler ran")
    name = "test.trace.forbidden"
    adapter._registry.register(operation(name, OperationKind.QUERY, "Denied operation", forbidden, capability="denied.read"))
    start(tools)
    result = invoke(tools, route, name, {})
    assert result["ok"] is False and result["error"]["code"] == "operation.permission_denied"
    matching = [step for step in stop(tools, project)["steps"] if step["operation"] == name]
    assert len(matching) == 1 and matching[0]["ok"] is False


@pytest.mark.parametrize("origin", ["outside_attempt", "capacity", "shutdown"])
def test_job_origin_and_retirement_are_fixed_at_submission(gateway, origin):
    tools, project = gateway
    entered, release = Event(), Event()
    def work():
        entered.set()
        assert release.wait(5)
        return {"completed": True}
    name = "test.trace.origin"
    adapter._registry.register(operation(name, OperationKind.QUERY, "Held work", work, capability="session.read"))
    if origin != "outside_attempt":
        start(tools)
    if origin == "capacity":
        adapter._jobs._max_jobs = 1
    try:
        running = tools["operation_job_submit"](name, {})["data"]["job_id"]
        assert entered.wait(3)
        if origin == "outside_attempt":
            start(tools)
        else:
            submitted = tools["operation_job_submit"]("infernux.mcp.checkpoint.list", {})
            if origin == "capacity":
                assert not submitted["ok"] and submitted["error"]["code"] == "job.capacity"
            else:
                adapter._jobs.shutdown(wait=False)
                assert await_job(tools, submitted["data"]["job_id"])["cancelled"]
    finally:
        release.set()
    await_job(tools, running)
    saved = stop(tools, project)
    if origin == "outside_attempt":
        assert not any(step["operation"] == name for step in saved["steps"])
    else:
        matching = [step for step in saved["steps"] if step["operation"] == "infernux.mcp.checkpoint.list"]
        assert len(matching) == 1 and matching[0]["ok"] is False
        if origin == "shutdown":
            assert matching[0]["status"] == "cancelled"


def test_batch_can_start_operate_and_save_an_attempt(gateway):
    tools, project = gateway
    result = tools["operation_batch_execute"]([
        {"operation": "infernux.mcp.attempt.start", "arguments": {"task": "batch lifecycle", "checkpoint": "baseline"}},
        {"operation": "infernux.mcp.checkpoint.list"},
        {"operation": "infernux.mcp.attempt.stop"},
    ])
    assert all(step["ok"] for step in result["data"]["results"]), result
    saved = stop(tools, project)  # The existing idempotent stop contract remains.
    assert [step["operation"] for step in saved["steps"]] == ["infernux.mcp.attempt.start", "infernux.mcp.checkpoint.list"]


def test_concurrent_jobs_have_unique_reserved_indices_and_complete_records(gateway):
    tools, project = gateway
    adapter._jobs.shutdown()
    adapter._jobs = OperationJobRegistry(adapter._registry, max_workers=4)
    start(tools)
    name = "infernux.mcp.checkpoint.list"
    jobs = [tools["operation_job_submit"](name, {})["data"]["job_id"] for _ in range(32)]
    for job in jobs:
        assert await_job(tools, job)["result"] == {"checkpoints": []}
    observed = trace.current_trace()
    observed["trace"]["steps"].clear()
    saved = stop(tools, project)
    steps = [step for step in saved["steps"] if step["operation"] == name]
    assert len(steps) == 32 and all(step["ok"] and "status" not in step for step in steps)
    assert len([step for step in trace.read_session_log(str(project))["entries"] if step.get("operation") == name]) == 32
