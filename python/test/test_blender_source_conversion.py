"""Source conversion owns one process and one derived-file publication."""
import subprocess
from pathlib import Path

import pytest

from Infernux.engine.model_import.blender import BlenderImportError, convert_blend


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / "模型 文件.blend"
    executable = tmp_path / "Blender Tool.exe"
    destination = tmp_path / "Library" / "Artifacts" / "Models" / "model.glb"
    source.write_bytes(b"author source")
    executable.touch()
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"previous import")
    return source, executable, destination


def test_conversion_uses_one_explicit_tool_and_only_replaces_derived_file(paths, monkeypatch):
    source, executable, destination = paths
    calls = []

    def run(command, **options):
        calls.append(command)
        assert command[0] == str(executable.resolve())
        assert "--disable-autoexec" in command
        assert command[-2] == str(source.resolve())
        assert options["timeout"] == 120
        assert "shell" not in options
        assert destination.read_bytes() == b"previous import"
        staging = Path(command[-1])
        assert staging.parent.parent == destination.parent
        staging.write_bytes(b"converted model")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert convert_blend(source, destination, blender_executable=executable) == destination
    assert len(calls) == 1
    assert source.read_bytes() == b"author source"
    assert destination.read_bytes() == b"converted model"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing"])
def test_failed_conversion_never_publishes_partial_output_or_retries(paths, monkeypatch, failure):
    source, executable, destination = paths
    calls = []

    def run(command, **options):
        calls.append(command)
        if failure != "missing":
            Path(command[-1]).write_bytes(b"incomplete")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0,
                                           "diagnostic from exporter", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(BlenderImportError):
        convert_blend(source, destination, blender_executable=executable)
    assert len(calls) == 1
    assert destination.read_bytes() == b"previous import"
    assert source.read_bytes() == b"author source"
    assert list(destination.parent.iterdir()) == [destination]


def test_conversion_rejects_wrong_output_before_launch(paths, monkeypatch):
    source, executable, _ = paths
    def run(*args, **kwargs):
        raise AssertionError("invalid conversion must not launch Blender")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match=".glb"):
        convert_blend(source, source, blender_executable=executable)
