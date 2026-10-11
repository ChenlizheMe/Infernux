"""The release target repairs fresh wheels without modifying the host install."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[2] / "cmake/package_cpu_jit_dependency.py"
    spec = importlib.util.spec_from_file_location("cpu_dependency_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform,tool", [("win32", "delvewheel"), ("linux", "auditwheel")])
def test_only_fresh_wheel_is_repaired(builder, monkeypatch, tmp_path, platform, tool):
    monkeypatch.setattr(builder.sys, "platform", platform)
    monkeypatch.setattr(builder.importlib.util, "find_spec", lambda name: object())
    output = tmp_path / "wheels"
    output.mkdir()
    old = output / "llvmlite-old.whl"
    old.write_bytes(b"old")
    calls = []

    def run(command, **options):
        calls.append(command)
        assert options["check"]
        if len(calls) == 1:
            assert command[1:4] == ["setup.py", "bdist_wheel", "--dist-dir"]
            assert options["env"]["LLVMLITE_PACKAGE_FORMAT"] == "wheel"
            (Path(command[-1]) / "llvmlite-new.whl").write_bytes(b"new")
        else:
            assert command[1:6] == ["-m", tool, "repair", "-w", str(output)]
            assert ("--analyze-existing" in command) == (platform == "win32")
            assert Path(command[-1]).read_bytes() == b"new"

    monkeypatch.setattr(builder.subprocess, "run", run)
    builder.build_wheel(tmp_path, output)
    assert len(calls) == 2
    assert not Path(calls[0][-1]).exists()
    assert old.read_bytes() == b"old"


def test_missing_repair_tool_rejects_before_build(builder, monkeypatch, tmp_path):
    monkeypatch.setattr(builder.sys, "platform", "win32")
    monkeypatch.setattr(builder.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="delvewheel"):
        builder.build_wheel(tmp_path, tmp_path / "wheels")
    assert not (tmp_path / "wheels").exists()


def test_repair_failure_is_not_published_as_success(builder, monkeypatch, tmp_path):
    monkeypatch.setattr(builder.sys, "platform", "win32")
    monkeypatch.setattr(builder.importlib.util, "find_spec", lambda name: object())

    def run(command, **options):
        if "bdist_wheel" in command:
            (Path(command[-1]) / "llvmlite-new.whl").write_bytes(b"new")
        else:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(builder.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        builder.build_wheel(tmp_path, tmp_path / "wheels")
    assert list((tmp_path / "wheels").iterdir()) == []
