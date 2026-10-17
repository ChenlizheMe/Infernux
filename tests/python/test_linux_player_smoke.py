from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "acceptance" / "linux_player_smoke.py"
WINDOWS_SCRIPT = ROOT / "tests" / "acceptance" / "windows_player_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("linux_player_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", (SCRIPT, WINDOWS_SCRIPT))
def test_player_smoke_cli_bootstraps_repository_python_from_any_working_directory(script: Path, tmp_path: Path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_gameplay_control_ready_requires_started_runtime_and_unpaused_input():
    module = _module()

    assert not module._gameplay_control_ready({"gameplay_ready": False})
    assert module._gameplay_control_ready({"gameplay_ready": True, "scene_manager_paused": False})
    assert not module._gameplay_control_ready({"gameplay_ready": True, "scene_manager_paused": True})
    assert module._gameplay_control_ready(
        {"gameplay_ready": True, "scene_manager_paused": True},
        allow_paused=True,
    )


@pytest.mark.parametrize("state_location", ("LOCALAPPDATA", "XDG_STATE_HOME", "home"))
def test_player_smoke_reads_errors_from_packaged_player_state_directory(state_location, tmp_path, monkeypatch):
    module = _module()
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    state_home = tmp_path / ".local" / "state"
    if state_location != "home":
        state_home = tmp_path / state_location
        monkeypatch.setenv(state_location, str(state_home))
    actual_log = state_home / "Infernux" / "Players" / "Balance" / "Logs" / "player.log"
    actual_log.parent.mkdir(parents=True)
    actual_log.write_text("[Error] scene failed\n", encoding="utf-8")

    log_path = module._state_log("Balance")
    assert str(log_path) == str(actual_log)
    assert module._fatal_lines(module._new_log_text(log_path, 0)) == ["[Error] scene failed"]


def test_linux_smoke_position_and_axis_delta():
    module = _module()
    data = {"objects": {"PlayerBall": {"position": [1, 2.5, -3]}}}

    initial = module._position(data, "PlayerBall")

    assert initial == (1.0, 2.5, -3.0)
    assert module._axis_delta(initial, (1.0, 2.5, 4.0), "z") == 7.0


def test_player_smoke_rejects_a_body_that_falls_through_the_floor():
    module = _module()

    module._require_minimum_final_y((0.0, 0.5, 2.0), 0.0)
    with pytest.raises(RuntimeError, match="fell to y=-0.250000"):
        module._require_minimum_final_y((0.0, -0.25, 2.0), 0.0)


def test_linux_smoke_report_is_atomic(tmp_path: Path):
    module = _module()
    report = tmp_path / "evidence" / "linux.json"

    module._write_json_atomic(report, {"schema": "infernux.linux_player_smoke", "status": "passed"})

    assert json.loads(report.read_text(encoding="utf-8")) == {
        "schema": "infernux.linux_player_smoke",
        "status": "passed",
    }
    assert not list(report.parent.glob("*.tmp"))


def test_linux_smoke_component_probe_asserts_nested_public_state():
    module = _module()
    probe = module._parse_component_probe(
        json.dumps(
            {
                "object_name": "GoalParticles",
                "component_type": "ParticleSystem",
                "fields": ["is_playing", "time", "last_compile_error"],
                "assertions": [
                    {"field": "is_playing", "operator": "truthy"},
                    {"field": "time", "operator": "greater", "value": 0.0},
                    {
                        "field": "last_compile_error",
                        "operator": "equal",
                        "value": "",
                    },
                ],
            }
        )
    )
    observation = {
        "objects": {
            "GoalParticles": {
                "component_fields": {
                    "ParticleSystem[0]": {
                        "is_playing": True,
                        "time": 0.75,
                        "last_compile_error": "",
                    }
                }
            }
        }
    }

    results, fields = module._assert_component_probes(observation, [probe])

    assert len(results) == 3
    assert fields["GoalParticles/ParticleSystem[0]"]["time"] == 0.75


def test_linux_smoke_fatal_scan_includes_vulkan_validation():
    module = _module()

    lines = module._fatal_lines("normal line\nVulkan Validation Error: VUID-RuntimeSpirv-test\n")

    assert lines == ["Vulkan Validation Error: VUID-RuntimeSpirv-test"]
