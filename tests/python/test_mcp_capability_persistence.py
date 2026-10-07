"""Capability policy is local state, never a default-on recovery artifact."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infernux_mcp import capabilities, checkpoints, server, supervisor as supervisor_module
from infernux_mcp.supervisor import SupervisorSession


@pytest.fixture(autouse=True)
def isolate_active_policy(monkeypatch):
    monkeypatch.setattr(capabilities, "_CURRENT_CONFIG", copy.deepcopy(capabilities.DEFAULT_CAPABILITY_CONFIG))
    monkeypatch.setattr(capabilities, "_PROJECT_PATH", "")


def write_policy(root: Path, value: bytes) -> Path:
    path = root / "ProjectSettings" / "mcp_capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


INVALID_POLICIES = [
    b'{"enabled": false, "granted_capabilities": [], BROKEN}',
    b'<<<<<<< HEAD\n{"enabled": false}\n=======\n{}\n>>>>>>> peer\n',
    b'null', b'[]', b'"policy"',
    b'{"enabled": "false"}', b'{"enabled": 0}',
    b'{"write_default_config_on_bootstrap": "false"}',
    b'{"granted_capabilities": "scene.read"}',
    b'{"granted_capabilities": ["scene.read", null]}',
    b'{"profile": "unknown"}',
    b'{"features": []}', b'{"features": {"discovery_files": "false"}}',
    b'{"session": null}', b'{"session": {"recording_enabled": 1}}',
    b'{"session": {"managed_checkpoints_required": "false"}}',
    b'{"session": {"allowed_project_roots": "C:/Project"}}',
    b'{"session": {"whl_readonly_source": [null]}}',
    b'{"session": {"session_id": 1}}',
    b'{"limits": []}', b'{"limits": {"batch_max_steps": true}}',
    b'{"limits": {"batch_max_steps": 0}}',
    b'{"limits": {"main_thread_timeout_ms": -1}}',
    b'{"limits": {"trace_argument_max_string": 1.5}}',
    b'{"limits": {"trace_result_max_string": NaN}}',
    b'{"enabled": false, "enabled": true}',
]


@pytest.mark.parametrize("raw", INVALID_POLICIES)
def test_invalid_existing_policy_rejects_without_rewriting_or_activating(tmp_path, raw):
    active = tmp_path / "active"
    capabilities.apply_config(str(active), {"enabled": False, "granted_capabilities": []})
    before = capabilities.current_config()
    previous_root = capabilities.project_path()
    policy = write_policy(tmp_path / "invalid", raw)
    with pytest.raises(ValueError, match="mcp_capabilities.json"):
        capabilities.configure(str(policy.parent.parent))
    assert policy.read_bytes() == raw
    assert capabilities.current_config() == before
    assert capabilities.project_path() == previous_root


def test_valid_existing_policy_is_read_without_reformatting_or_replacing(tmp_path):
    raw = b'{ "enabled": false, "granted_capabilities": [], "unknown": 7 }\r\n'
    policy = write_policy(tmp_path, raw)
    before = policy.stat()
    config = capabilities.configure(str(tmp_path))
    assert config["enabled"] is False
    assert config["granted_capabilities"] == []
    assert "unknown" not in config
    assert policy.read_bytes() == raw
    assert policy.stat().st_mtime_ns == before.st_mtime_ns
    assert policy.stat().st_ino == before.st_ino


def test_config_directory_is_not_treated_as_missing(tmp_path):
    policy = tmp_path / "ProjectSettings" / "mcp_capabilities.json"
    policy.mkdir(parents=True)
    with pytest.raises(OSError):
        capabilities.load_capability_config(str(tmp_path))
    assert policy.is_dir()


@pytest.mark.parametrize("write_default", [True, False])
def test_missing_config_has_explicit_bootstrap_behavior(tmp_path, write_default):
    result = capabilities.configure(str(tmp_path), write_default=write_default)
    policy = tmp_path / "ProjectSettings" / "mcp_capabilities.json"
    assert result == capabilities.DEFAULT_CAPABILITY_CONFIG
    assert policy.exists() is write_default
    if write_default:
        assert json.loads(policy.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("operation", ["configure", "apply", "save"])
def test_configuration_failure_does_not_change_active_policy(tmp_path, monkeypatch, operation):
    active = tmp_path / "active"
    capabilities.apply_config(str(active), {"enabled": False, "granted_capabilities": []})
    before, previous_root = capabilities.current_config(), capabilities.project_path()

    def fail_write(*args, **kwargs):
        raise PermissionError("owned test injection")

    monkeypatch.setattr(capabilities, "_write_json_atomically", fail_write)
    with pytest.raises((ValueError, PermissionError)):
        if operation == "configure":
            capabilities.configure(str(tmp_path / "new"))
        elif operation == "apply":
            capabilities.apply_config(str(tmp_path / "new"), {"enabled": "false"})
        else:
            capabilities.save_config({"enabled": True}, str(tmp_path / "new"))
    assert capabilities.current_config() == before
    assert capabilities.project_path() == previous_root


def test_bootstrap_does_not_replace_a_concurrently_created_policy(tmp_path, monkeypatch):
    capabilities.apply_config(str(tmp_path / "active"), {"enabled": False})
    before, previous_root = capabilities.current_config(), capabilities.project_path()
    original = capabilities._write_json_atomically
    restricted = b'{"enabled": false, "granted_capabilities": []}\n'

    def appeared_before_publish(path, value, **kwargs):
        write_policy(tmp_path / "new", restricted)
        return original(path, value, **kwargs)

    monkeypatch.setattr(capabilities, "_write_json_atomically", appeared_before_publish)
    with pytest.raises(FileExistsError):
        capabilities.configure(str(tmp_path / "new"))
    assert (tmp_path / "new" / "ProjectSettings" / "mcp_capabilities.json").read_bytes() == restricted
    assert capabilities.current_config() == before
    assert capabilities.project_path() == previous_root
    assert not list((tmp_path / "new" / "ProjectSettings").glob(".mcp-capabilities-*.tmp"))


def test_malformed_policy_prevents_transport_creation(tmp_path, monkeypatch):
    policy = write_policy(tmp_path, INVALID_POLICIES[0])
    monkeypatch.setattr(server, "_active_state", lambda: None)

    def transport_must_not_start(*args, **kwargs):
        pytest.fail("invalid capability policy reached transport construction")

    monkeypatch.setattr(server, "_import_fastmcp", lambda: transport_must_not_start)
    with pytest.raises(ValueError, match="mcp_capabilities.json"):
        server.start_server(str(tmp_path), host="127.0.0.1", port=9728)
    assert policy.read_bytes() == INVALID_POLICIES[0]


@pytest.mark.parametrize("current", [
    b'{"enabled": false, "granted_capabilities": []}\n',
    b'{"enabled": true, "granted_capabilities": ["scene.read"], "limits": {"batch_max_steps": 2}}\n',
    INVALID_POLICIES[0], None,
])
def test_checkpoint_restores_content_and_preserves_current_policy_bytes(tmp_path, current):
    project = tmp_path / "project"
    policy = write_policy(project, b'{"enabled": true}\n')
    scene = project / "Assets" / "level.scene"
    scene.parent.mkdir()
    scene.write_text("checkpoint scene", encoding="utf-8")
    artifacts = project / ".infernux" / "test-checkpoints"
    created = checkpoints.create_checkpoint(str(project), str(artifacts), "before", session_id="owned-test")
    assert not (Path(created["payload_path"]) / "ProjectSettings" / "mcp_capabilities.json").exists()
    if current is None:
        policy.unlink()
    else:
        policy.write_bytes(current)
    scene.write_text("draft scene", encoding="utf-8")
    proof = checkpoints.restore_checkpoint(str(project), str(artifacts), "before", session_id="owned-test")
    assert proof["verified"] is True
    assert scene.read_text(encoding="utf-8") == "checkpoint scene"
    if current is None:
        assert not policy.exists()
    else:
        assert policy.read_bytes() == current


@pytest.mark.parametrize("enabled", [True, False])
def test_supervisor_restore_preserves_current_permission_policy(tmp_path, monkeypatch, enabled):
    project = tmp_path / "project"
    instance = SupervisorSession(str(project), session_id="owned-test", mcp_port=9728)
    monkeypatch.setattr(supervisor_module, "_mcp_health_is_alive", lambda _endpoint: False)
    instance.prepare_project()
    scene = project / "Assets" / "level.scene"
    scene.write_text("saved", encoding="utf-8")
    instance.create_checkpoint("before", restart_editor=False)
    config = capabilities.load_capability_config(str(project))
    config.update(enabled=enabled, granted_capabilities=["scene.read"])
    config["features"]["discovery_files"] = False
    config["limits"]["batch_max_steps"] = 2
    write_policy(project, json.dumps(config).encode("utf-8"))
    scene.write_text("draft", encoding="utf-8")
    result = instance.restore_checkpoint("before", restart_editor=False)
    assert result["checkpoint_restore"]["state"] == "completed"
    restored = capabilities.load_capability_config(str(project))
    assert restored == config
    assert scene.read_text(encoding="utf-8") == "saved"


def test_supervisor_does_not_replace_malformed_policy(tmp_path):
    policy = write_policy(tmp_path, INVALID_POLICIES[0])
    instance = SupervisorSession(str(tmp_path), session_id="owned-test", mcp_port=9728)
    with pytest.raises(ValueError, match="mcp_capabilities.json"):
        instance.prepare_project()
    assert policy.read_bytes() == INVALID_POLICIES[0]


def test_config_read_permission_error_is_not_a_missing_file(tmp_path, monkeypatch):
    raw = b'{"enabled": false, "granted_capabilities": []}'
    policy = write_policy(tmp_path, raw)

    def denied(*args, **kwargs):
        raise PermissionError("owned policy read injection")

    monkeypatch.setattr(capabilities, "open", denied, raising=False)
    with pytest.raises(PermissionError, match="owned policy read"):
        capabilities.configure(str(tmp_path))
    assert policy.read_bytes() == raw


def test_explicit_save_failure_preserves_previous_disk_and_active_policy(tmp_path, monkeypatch):
    raw = b'{"enabled": false, "granted_capabilities": []}'
    policy = write_policy(tmp_path, raw)
    before = capabilities.configure(str(tmp_path))
    root = capabilities.project_path()

    def denied(source, destination):
        raise PermissionError("owned policy replace injection")

    monkeypatch.setattr(capabilities.os, "replace", denied)
    with pytest.raises(PermissionError, match="owned policy replace"):
        capabilities.save_config({"enabled": True}, str(tmp_path))
    assert policy.read_bytes() == raw
    assert capabilities.current_config() == before
    assert capabilities.project_path() == root
    assert not list(policy.parent.glob(".mcp-capabilities-*.tmp"))


def test_checkpoint_manifest_cannot_claim_machine_local_permissions(tmp_path):
    project = tmp_path / "project"
    raw = b'{"enabled": false, "granted_capabilities": []}\n'
    policy = write_policy(project, raw)
    scene = project / "Assets" / "level.scene"
    scene.parent.mkdir()
    scene.write_text("saved", encoding="utf-8")
    artifacts = project / ".infernux" / "test-checkpoints"
    created = checkpoints.create_checkpoint(str(project), str(artifacts), "before", session_id="owned-test")
    manifest = json.loads(Path(created["manifest_path"]).read_text(encoding="utf-8"))
    # Policy files are excluded from the content digest. An inconsistent
    # manifest must not use that exclusion to claim ownership of local policy.
    payload_policy = Path(created["payload_path"]) / "ProjectSettings" / "mcp_capabilities.json"
    payload_policy.write_text('{"enabled": true}', encoding="utf-8")
    manifest["ledger"]["entries"].append({
        "path": "ProjectSettings/mcp_capabilities.json", "size": 17,
        "sha256": "not-used-for-local-policy", "kind": "project_setting",
    })
    Path(created["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
    scene.write_text("draft", encoding="utf-8")
    with pytest.raises(checkpoints.CheckpointError, match="local project setting"):
        checkpoints.restore_checkpoint(str(project), str(artifacts), "before", session_id="owned-test")
    assert policy.read_bytes() == raw
    assert scene.read_text(encoding="utf-8") == "draft"
