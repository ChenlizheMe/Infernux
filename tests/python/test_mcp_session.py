from __future__ import annotations

import json

import pytest

from infernux_mcp import checkpoints as checkpoint_store
from infernux_mcp import session
from infernux_mcp import trace


def _config(mode: str = "developer_assist", **session_overrides):
    policy = {
        "build_profile": "debug_feedback",
        "recording_enabled": False,
        "allowed_project_roots": [],
        "whl_readonly_source": [],
        "workaround_allowlist": [],
    }
    policy.update(session_overrides)
    return {"profile": mode, "session": policy}


def test_supervisor_lease_is_verified_but_never_exposed_in_status_or_trace(tmp_path, monkeypatch):
    lease = "private-supervisor-lease"
    monkeypatch.setenv("INFERNUX_MCP_EDITOR_INSTANCE_ID", "editor-instance-for-test")
    monkeypatch.setenv("INFERNUX_MCP_SUPERVISOR_LEASE", lease)
    configured = session.configure(str(tmp_path), _config("global_validation"))

    status = session.status()
    assert status["editor_instance_id"] == "editor-instance-for-test"
    assert status["supervisor_lease_configured"] is True
    assert status["supervisor_lease_fingerprint"]
    assert lease not in json.dumps(status)
    assert session.require_supervisor_lease(lease) is configured
    with pytest.raises(session.McpPolicyError, match="invalid"):
        session.require_supervisor_lease("wrong-lease")

    session.start_attempt("lease trace redaction", "before-normal-shutdown")

    trace.record_operation(
        "infernux.mcp.supervisor.shutdown",
        ok=True,
        arguments={"lease_token": lease},
        result={"close_requested": True},
    )
    stopped = session.stop_attempt()

    with open(tmp_path / stopped["trace_path"], "r", encoding="utf-8") as f:
        trace_payload = json.load(f)
    assert lease not in json.dumps(trace_payload)
    assert trace_payload["steps"][0]["arguments"] == {"lease_token": "<redacted>"}


def test_developer_assist_attempt_persists_mode_and_checkpoint(tmp_path):
    configured = session.configure(str(tmp_path), _config("developer_assist"))

    attempt = session.start_attempt("author public HUD script", "before-hud-script")
    stopped = session.stop_attempt()

    with open(tmp_path / stopped["trace_path"], "r", encoding="utf-8") as f:
        trace = json.load(f)
    assert trace["context"]["mode"] == "developer_assist"
    assert trace["context"]["session_id"] == configured.session_id
    assert attempt["checkpoint"] == "before-hud-script"


def test_managed_attempt_requires_exact_checkpoint_and_writes_persistence_delta(tmp_path):
    assets = tmp_path / "Assets"
    settings = tmp_path / "ProjectSettings"
    assets.mkdir()
    settings.mkdir()
    scene = assets / "Race.scene"
    scene.write_text("clean\n", encoding="utf-8")
    configured = session.configure(
        str(tmp_path),
        _config(
            "global_validation",
            session_id="managed-attempt-session",
            managed_checkpoints_required=True,
        ),
    )
    checkpoint_store.create_checkpoint(
        str(tmp_path),
        configured.artifact_root,
        "clean-race-001",
        session_id=configured.session_id,
    )

    status = session.checkpoint_status("clean-race-001")
    attempt = session.start_attempt("managed persistence proof", "clean-race-001")
    (assets / "Created.prefab").write_text("created\n", encoding="utf-8")
    scene.write_text("modified\n", encoding="utf-8")
    stopped = session.stop_attempt()

    assert status["current_match"] is True
    assert attempt["checkpoint_proof"]["managed"] is True
    with open(tmp_path / stopped["persistence_proof_path"], "r", encoding="utf-8") as stream:
        proof = json.load(stream)
    assert proof["changed"] is True
    assert proof["delta"]["added"] == ["Assets/Created.prefab"]
    assert proof["delta"]["modified"] == ["Assets/Race.scene"]
    assert proof["delta"]["deleted"] == []

    with pytest.raises(session.McpPolicyError, match="does not match"):
        session.start_attempt("must restore first", "clean-race-001")


def test_checkpoint_list_returns_payload_verified_choices_without_project_scan(tmp_path):
    assets = tmp_path / "Assets"
    settings = tmp_path / "ProjectSettings"
    assets.mkdir()
    settings.mkdir()
    (assets / "Race.scene").write_text("clean\n", encoding="utf-8")
    (settings / "BuildSettings.json").write_text("{}\n", encoding="utf-8")
    configured = session.configure(
        str(tmp_path),
        _config("global_validation", session_id="checkpoint-list-session", managed_checkpoints_required=True),
    )
    checkpoint_store.create_checkpoint(
        str(tmp_path),
        configured.artifact_root,
        "race-clean-001",
        session_id=configured.session_id,
        metadata={"reason": "weak-agent entry"},
    )

    listed = session.list_checkpoints()

    assert len(listed) == 1
    assert listed[0]["checkpoint_id"] == "race-clean-001"
    assert listed[0]["payload_valid"] is True
    assert listed[0]["file_count"] == 2
    assert listed[0]["metadata"] == {"reason": "weak-agent entry"}


def test_global_validation_writes_logic_backed_blocker(tmp_path):
    session.configure(str(tmp_path), _config("global_validation"))
    attempt = session.start_attempt("physics contact validation", "clean-physics-001")
    stopped = session.stop_attempt()

    result = session.write_blocker(
        {
            "category": "engine_bug",
            "title": "Physics contact was never emitted",
            "expected": "contact event count is positive",
            "actual": "contact event count is zero",
            "normal_workflow": ["create colliders", "enter play mode", "advance fixed steps"],
            "logic_evidence": {"fixed_steps": 120, "contact_event_count": 0},
            "persistence_proof": "passed",
        }
    )

    report_path = result["path"]
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["mode"] == "global_validation"
    assert report["build_profile"] == "debug_feedback"
    assert report["category"] == "engine_bug"
    assert report["attempt_id"] == attempt["attempt_id"]
    assert report["trace_id"] == stopped["trace_id"]
    assert report["attempt_manifest_path"] == stopped["attempt_manifest_path"]


def test_global_validation_trace_persists_attempt_and_session_context(tmp_path):
    configured = session.configure(str(tmp_path), _config("global_validation", recording_enabled=True))

    attempt = session.start_attempt("save persistence validation", "before-scene-save")
    stopped = session.stop_attempt()

    trace_path = tmp_path / stopped["trace_path"]
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    assert trace["task"] == "save persistence validation"
    context = dict(trace["context"])
    assert context.pop("build_identity") == configured.build_identity
    assert context == {
        "attempt_id": attempt["attempt_id"],
        "checkpoint": "before-scene-save",
        "session_id": configured.session_id,
        "mode": "global_validation",
        "build_profile": "debug_feedback",
        "recording_enabled": True,
    }


def test_global_validation_trace_persists_compact_operation_results(tmp_path, monkeypatch):
    session.configure(str(tmp_path), _config("global_validation"))
    monkeypatch.setattr(
        "infernux_mcp.capabilities.limit",
        lambda name, default=None: 12 if name == "trace_result_max_string" else default,
    )

    session.start_attempt("runtime assertion evidence", "before-runtime-assertion")
    trace.record_operation(
        "infernux.runtime.assert",
        ok=True,
        arguments={"assertions": [{"kind": "scene_name", "equals": "Results"}]},
        result={"ok": True, "data": {"passed": True, "detail": "0123456789abcdef"}},
    )
    stopped = session.stop_attempt()

    with open(tmp_path / stopped["trace_path"], "r", encoding="utf-8") as f:
        saved = json.load(f)
    step = saved["steps"][0]
    assert step["arguments"]["assertions"][0]["kind"] == "scene_name"
    assert step["result"] == {
        "ok": True,
        "data": {"passed": True, "detail": "0123456789ab...<truncated>"},
    }


def test_global_validation_attempt_stop_is_idempotent_and_tracks_activity(tmp_path):
    session.configure(str(tmp_path), _config("global_validation"))

    attempt = session.start_attempt("menu state validation", "before-menu-open")
    assert attempt["attempt_active"] is True
    assert session.status()["attempt_active"] is True
    with pytest.raises(session.McpPolicyError, match="already active"):
        session.start_attempt("another validation", "another-checkpoint")

    stopped = session.stop_attempt()
    repeated = session.stop_attempt()

    assert stopped["already_stopped"] is False
    assert repeated == {
        "attempt_id": attempt["attempt_id"],
        "checkpoint": "before-menu-open",
        "trace_id": stopped["trace_id"],
        "trace_path": stopped["trace_path"],
        "attempt_manifest_path": stopped["attempt_manifest_path"],
        "elapsed_seconds": 0.0,
        "already_stopped": True,
    }
    assert session.status()["attempt_active"] is False
