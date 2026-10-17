from __future__ import annotations

import threading

import pytest
from infernux.host.commands import CommandOwner, command_owner_scope

from infernux.host import (
    MainThreadCommandQueue,
    Operation,
    OperationError,
    OperationKind,
    OperationRegistry,
    OperationSchema,
)


def test_retiring_command_owner_cancels_its_pending_work_only():
    queue = MainThreadCommandQueue()
    owner = CommandOwner("MCP")
    calls = []
    with command_owner_scope(owner):
        cancelled = queue.submit("owned", lambda: calls.append("owned"))
    unrelated = queue.submit("other service", lambda: calls.append("other"))
    assert cancelled.can_execute()
    owner.close()
    with pytest.raises(TimeoutError, match="Host service stopped: MCP"):
        cancelled.result(0)
    with command_owner_scope(owner):
        rejected = queue.submit("after shutdown", lambda: calls.append("late"))
    with pytest.raises(TimeoutError, match="Host service stopped: MCP"):
        rejected.result(0)
    queue.drain()
    assert unrelated.result(0) is None
    assert calls == ["other"]


def test_retiring_command_owner_allows_started_authoring_to_finish():
    queue = MainThreadCommandQueue()
    queue.drain(0)
    owner = CommandOwner("MCP")
    calls = []

    def transaction():
        owner.close()
        calls.append("committed")
        return 42

    with command_owner_scope(owner):
        future = queue.submit("started transaction", transaction)
    assert future.result(0) == 42
    assert calls == ["committed"]


def _schema(
    operation_id: str = "test.math.add",
    *,
    kind: OperationKind = OperationKind.QUERY,
) -> OperationSchema:
    return OperationSchema(
        id=operation_id,
        kind=kind,
        summary="Add two values.",
        input_schema={
            "type": "object",
            "properties": {
                "left": {"type": "number"},
                "right": {"type": "number"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        output_schema={"type": "number"},
        errors=({"code": "test.failed"},),
        thread="any",
        side_effects=(),
        reversible=False,
        capabilities=("math.read",),
        cost={"class": "constant"},
        tags=("math", "addition"),
    )


def test_operation_registry_revision_events_conflicts_and_cache_resync():
    registry = OperationRegistry()
    events = []
    token = registry.subscribe(events.append)
    start_revision = registry.revision
    registry.register(Operation(_schema(), lambda left, right: left + right, "owner-a"))

    assert registry.revision == start_revision + 1
    assert events == [
        {
            "revision": start_revision + 1,
            "action": "registered",
            "operations": ["test.math.add"],
            "owner": "owner-a",
        }
    ]
    assert registry.search("math addition")[0]["id"] == "test.math.add"
    with pytest.raises(OperationError, match="already owned") as conflict:
        registry.register(Operation(_schema(), lambda left, right: 0, "owner-b"))
    assert conflict.value.code == "operation.conflict"

    assert registry.unregister_owner("owner-a") == 1
    assert events[-1]["revision"] == start_revision + 2
    assert events[-1]["action"] == "unregistered"
    assert registry.unsubscribe(token) is True
    assert registry.unsubscribe(token) is False


def test_operation_execution_validates_capability_kind_and_json_arguments():
    registry = OperationRegistry()
    registry.register(Operation(_schema(), lambda left, right: left + right, "owner"))

    assert registry.execute("test.math.add", {"left": 2, "right": 3.5}, capabilities=("math.read",)) == 5.5
    with pytest.raises(OperationError) as denied:
        registry.execute("test.math.add", {"left": 2, "right": 3})
    assert denied.value.code == "operation.permission_denied"
    with pytest.raises(OperationError) as wrong_kind:
        registry.execute(
            "test.math.add",
            {"left": 2, "right": 3},
            capabilities=("*",),
            expected_kind=OperationKind.COMMAND,
        )
    assert wrong_kind.value.code == "operation.kind_mismatch"
    for arguments in (
        {"left": 1},
        {"left": 1, "right": 2, "extra": 3},
        {"left": True, "right": 2},
    ):
        with pytest.raises(OperationError) as invalid:
            registry.execute("test.math.add", arguments, capabilities=("*",))
        assert invalid.value.code == "operation.invalid_arguments"


def test_operation_batch_has_structured_stop_and_continue_semantics():
    registry = OperationRegistry()
    registry.register(Operation(_schema(), lambda left, right: left + right, "owner"))
    calls = (
        {"operation": "test.math.add", "arguments": {"left": 1, "right": 2}},
        {"operation": "missing.operation", "arguments": {}},
        {"operation": "test.math.add", "arguments": {"left": 4, "right": 5}},
    )

    stopped = registry.execute_batch(calls, capabilities=("*",), stop_on_error=True)
    continued = registry.execute_batch(calls, capabilities=("*",), stop_on_error=False)
    assert len(stopped) == 2
    assert stopped[1]["error"]["code"] == "operation.not_found"
    assert len(continued) == 3
    assert continued[2]["result"] == 9


def test_main_thread_queue_wakes_graphical_owner_for_background_submission():
    queue = MainThreadCommandQueue()
    queue.drain(0)
    wakes = []
    queued = []
    queue.set_wake_callback(lambda: wakes.append(threading.get_ident()))

    def worker():
        queued.append(queue.submit("wake-owner", lambda: "done"))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(1)

    assert not thread.is_alive()
    assert len(wakes) == 1
    assert wakes[0] != threading.get_ident()
    assert queue.drain() == 1
    assert queued[0].result(0) == "done"

    queue.release_owner()
    assert queue._wake_callback is None
