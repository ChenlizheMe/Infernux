from __future__ import annotations

import json
from pathlib import Path

import pytest


from project_python_runtime import (
    project_runtime_directory,
    read_project_python_version,
    write_project_python_version,
)


def test_project_python_binding_round_trips(tmp_path: Path) -> None:
    write_project_python_version(tmp_path, "3.13")

    settings = json.loads(
        (tmp_path / "ProjectSettings" / "PythonRuntime.json").read_text(
            encoding="utf-8"
        )
    )
    assert settings == {"pythonVersion": "3.13"}
    assert read_project_python_version(tmp_path) == "3.13"


def test_private_runtime_cannot_supply_the_shared_python_binding(
    tmp_path: Path,
) -> None:
    (tmp_path / ".runtime" / "python312").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="exact Python ABI"):
        read_project_python_version(tmp_path)
    assert not (tmp_path / "ProjectSettings" / "PythonRuntime.json").exists()


def test_unbound_project_with_multiple_runtimes_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".runtime" / "python312").mkdir(parents=True)
    (tmp_path / ".runtime" / "python313").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="exact Python ABI"):
        read_project_python_version(tmp_path)


def test_explicit_binding_selects_one_of_multiple_runtime_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / ".runtime" / "python312").mkdir(parents=True)
    (tmp_path / ".runtime" / "python313").mkdir(parents=True)
    write_project_python_version(tmp_path, "3.13")

    assert read_project_python_version(tmp_path) == "3.13"
    assert project_runtime_directory(tmp_path, "3.13").endswith(
        str(Path(".runtime") / "python313")
    )


def test_missing_project_python_binding_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Restore ProjectSettings/PythonRuntime.json"):
        read_project_python_version(tmp_path)


def test_venv_config_cannot_supply_the_shared_python_binding(tmp_path):
    config = tmp_path / ".venv" / "pyvenv.cfg"
    config.parent.mkdir()
    config.write_text("version = 3.13.5\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact Python ABI"):
        read_project_python_version(tmp_path)
