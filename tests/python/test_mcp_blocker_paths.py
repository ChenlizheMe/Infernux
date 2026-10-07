"""Report persistence through the real MCP registry, session and trace boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from infernux_mcp import adapter, session
from infernux_mcp.operations import build_operations


class _GatewayRecorder:
    """Collect transport callbacks; no operation or persistence implementation is replaced."""

    def __init__(self):
        self.tools = {}

    def tool(self, *, name):
        def register(callback):
            self.tools[name] = callback
            return callback
        return register


@pytest.fixture
def report_session(tmp_path):
    project = tmp_path / "中文 空格 project"
    (project / "Assets").mkdir(parents=True)
    gateway = _GatewayRecorder()
    adapter.register_gateways(gateway, str(project), {
        "profile": "global_validation",
        "granted_capabilities": ["session.write"],
        "session": {"session_id": "report-path-test", "managed_checkpoints_required": False},
        "features": {"session_call_log": False},
    })
    execute = gateway.tools["operation_command_execute"]
    try:
        started = execute("infernux.mcp.attempt.start", {
            "task": "report persistence boundary", "checkpoint": "before-report",
        })
        assert started["ok"], started
        stopped = execute("infernux.mcp.attempt.stop", {})
        assert stopped["ok"], stopped
        yield project, execute
    finally:
        adapter.shutdown_adapter()


def _report(**overrides):
    return {
        "category": "project_bug", "title": "Local report fixture",
        "expected": "Write only a new session report", "actual": "Observed fixture",
        "normal_workflow": ["Observe the fixture"], "logic_evidence": {"fixture": True},
        "persistence_proof": "Local temporary files; not an Editor workflow claim",
        **overrides,
    }


@pytest.mark.parametrize("route", ["direct", "gateway"])
@pytest.mark.parametrize("kind", ["relative", "absolute", "backslash"])
def test_report_path_escape_preserves_existing_asset(report_session, tmp_path, route, kind):
    project, execute = report_session
    victim = (tmp_path / "outside.json") if kind == "absolute" else project / "Assets/victim.json"
    victim.write_text('{"author_document":"must survive"}', encoding="utf-8")
    original = victim.read_bytes()
    report_id = {
        "relative": "../../../../Assets/victim",
        "backslash": "..\\..\\..\\..\\Assets\\victim",
        "absolute": str(victim.with_suffix("")),
    }[kind]
    if route == "direct":
        with pytest.raises(session.McpPolicyError, match="report_id"):
            session.write_blocker(_report(report_id=report_id))
    else:
        result = execute("infernux.mcp.blocker.report", {"report": _report(report_id=report_id)})
        assert not result["ok"], result
        assert "report_id" in str(result)
    assert victim.read_bytes() == original
    assert not (Path(session.current().artifact_root) / "reports").exists()


@pytest.mark.parametrize("report_id", [None, "", 123, False, [], {}, ".", "..", "with space",
    "nested/report", "nested\\report", "C:drive", "a:stream", "trailing.", "line\n",
    "CON", "nUl", "COM1", "lpt9", "a" * 97, "C:\\outside", "/outside",
    "\\\\server\\share\\report", "\\\\?\\C:\\outside", "nul\0byte"])
def test_report_identifier_is_rejected_before_persistence(report_session, report_id):
    _, execute = report_session
    result = execute("infernux.mcp.blocker.report", {"report": _report(report_id=report_id)})
    assert not result["ok"], result
    assert "report_id" in str(result)
    assert not (Path(session.current().artifact_root) / "reports").exists()


@pytest.mark.parametrize("report_id", [None, "report-One_2", "A" * 96, "CONSOLE", "LPT10"])
def test_report_normal_write_keeps_trace_and_identity(report_session, report_id):
    _, execute = report_session
    payload = _report() if report_id is None else _report(report_id=report_id)
    result = execute("infernux.mcp.blocker.report", {"report": payload})
    assert result["ok"], result
    data = result["data"]["result"]
    report_root = Path(session.current().artifact_root) / "reports"
    path = Path(data["path"])
    assert path.parent.resolve() == report_root.resolve()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == data["report"]
    assert stored["report_id"] == data["report_id"] == path.stem
    if report_id is not None:
        assert stored["report_id"] == report_id
    assert stored["trace_id"] == session.current().last_trace_id
    assert stored["attempt_id"] == session.current().attempt_id


def test_report_duplicate_never_overwrites_first_record(report_session):
    _, execute = report_session
    first = execute("infernux.mcp.blocker.report", {"report": _report(report_id="first")})
    assert first["ok"], first
    path = Path(first["data"]["result"]["path"])
    original = path.read_bytes()
    second = execute("infernux.mcp.blocker.report", {
        "report": _report(report_id="first", title="Replacement must not be written"),
    })
    assert not second["ok"], second
    assert "already exists" in str(second)
    assert path.read_bytes() == original


def test_report_cannot_overwrite_hardlinked_asset(report_session):
    project, execute = report_session
    victim = project / "Assets/victim.json"
    victim.write_text("author data", encoding="utf-8")
    report_root = Path(session.current().artifact_root) / "reports"
    report_root.mkdir()
    os.link(victim, report_root / "linked.json")
    result = execute("infernux.mcp.blocker.report", {"report": _report(report_id="linked")})
    assert not result["ok"], result
    assert victim.read_text(encoding="utf-8") == "author data"


def test_concurrent_report_identity_has_one_writer(report_session):
    _, execute = report_session
    start = Barrier(2)

    def write(title):
        start.wait(timeout=10)
        return execute("infernux.mcp.blocker.report", {"report": _report(report_id="shared", title=title)})

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(write, ("First writer", "Second writer")))
    successful = [result["data"]["result"] for result in results if result["ok"]]
    assert len(successful) == 1, results
    winner = successful[0]
    assert json.loads(Path(winner["path"]).read_text(encoding="utf-8")) == winner["report"]


def test_report_serialization_error_does_not_publish_partial_file(report_session):
    with pytest.raises(TypeError):
        session.write_blocker(_report(report_id="invalid-json", notes=object()))
    assert not (Path(session.current().artifact_root) / "reports").exists()


def test_report_directory_link_cannot_redirect_writes(report_session, tmp_path):
    _, execute = report_session
    outside = tmp_path / "outside-reports"
    outside.mkdir()
    report_root = Path(session.current().artifact_root) / "reports"
    if os.name == "nt":
        # A junction is unprivileged on Windows and exercises real reparse resolution.
        import _winapi
        _winapi.CreateJunction(str(outside), str(report_root))
    else:
        report_root.symlink_to(outside, target_is_directory=True)
    result = execute("infernux.mcp.blocker.report", {"report": _report(report_id="escaped")})
    assert not result["ok"], result
    assert not list(outside.iterdir())


def test_report_permission_denial_does_not_create_directory(report_session, monkeypatch):
    _, execute = report_session
    monkeypatch.setitem(adapter._config, "granted_capabilities", ["session.read"])
    result = execute("infernux.mcp.blocker.report", {"report": _report(report_id="denied")})
    assert not result["ok"], result
    assert "operation.permission_denied" in str(result)
    assert not (Path(session.current().artifact_root) / "reports").exists()


def test_report_contract_exposes_the_same_identifier_rule(report_session):
    project, _ = report_session
    operation = next(op for op in build_operations(str(project))
                     if op.schema.id == "infernux.mcp.blocker.report")
    schema = operation.schema.input_schema["properties"]["report"]["properties"]["report_id"]
    assert schema["type"] == "string"
    for valid in ("report-One_2", "A" * 96, "CONSOLE", "LPT10"):
        assert re.search(schema["pattern"], valid)
    for invalid in ("../outside", "C:\\absolute", "with space", "CON", "nUl", "lpt9", "line\n", "A" * 97):
        assert not re.search(schema["pattern"], invalid)
    assert "report_id" in session.blocker_report_contract()["optional_arguments"]
