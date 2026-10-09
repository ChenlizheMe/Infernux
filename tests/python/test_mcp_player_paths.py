"""Player launch resolves authored paths against the project, never the host cwd."""
import json
import os
from pathlib import Path

import pytest

from infernux.host import OperationError
from infernux_mcp import player_operations


@pytest.fixture
def launch_project(tmp_path, monkeypatch):
    project = tmp_path / "项目 with spaces"
    (project / "ProjectSettings").mkdir(parents=True)
    elsewhere = tmp_path / "unrelated host directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return project


@pytest.mark.parametrize("relative", ["Builds/development", r"Builds\development"])
def test_configured_player_output_is_relative_to_project(launch_project, relative):
    settings = launch_project / "ProjectSettings/BuildSettings.json"
    settings.write_text(json.dumps(dict(output_dir=relative, game_name="Puzzle")), encoding="utf-8")
    executable = player_operations._configured_executable(str(launch_project))
    suffix = ".exe" if os.name == "nt" else ""
    assert Path(executable) == launch_project / "Builds/development" / ("Puzzle" + suffix)


def test_configured_player_keeps_explicit_external_output(launch_project, tmp_path):
    external = tmp_path / "delivered game"
    settings = launch_project / "ProjectSettings/BuildSettings.json"
    settings.write_text(json.dumps(dict(output_dir=str(external), game_name="Puzzle")), encoding="utf-8")
    assert Path(player_operations._configured_executable(str(launch_project))).parent == external


@pytest.mark.parametrize("output", ["", "   "])
def test_missing_build_destination_does_not_become_host_directory(launch_project, output):
    settings = launch_project / "ProjectSettings/BuildSettings.json"
    settings.write_text(json.dumps(dict(output_dir=output, game_name="Puzzle")), encoding="utf-8")
    with pytest.raises(OperationError, match="output_dir"):
        player_operations._configured_executable(str(launch_project))


@pytest.mark.parametrize("explicit", ["Builds/Puzzle.exe", r"Builds\Puzzle.exe", "absolute"])
def test_explicit_launch_uses_same_project_path_boundary(launch_project, monkeypatch, explicit):
    expected = launch_project / "Builds/Puzzle.exe"
    requested = str(expected) if explicit == "absolute" else explicit
    received = []

    class Supervisor:
        def launch_player(self, executable, **options):
            received.append((executable, options))
            return {"player_ready": True}

    monkeypatch.setattr(player_operations, "_supervisor", lambda: Supervisor())
    assert player_operations._launch(str(launch_project), requested, "", 3.0)["player_ready"]
    assert Path(received[0][0]) == expected
