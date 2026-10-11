"""Project-level Python ABI binding used by Infernux Hub."""

from __future__ import annotations

import json
import os
from pathlib import Path

from python_runtime_catalog import PythonRuntimeId


PROJECT_RUNTIME_SETTINGS = os.path.join(
    "ProjectSettings", "PythonRuntime.json"
)


def project_runtime_settings_path(project_dir: str | os.PathLike[str]) -> Path:
    return Path(project_dir) / PROJECT_RUNTIME_SETTINGS


def write_project_python_version(
    project_dir: str | os.PathLike[str], version: str | PythonRuntimeId
) -> str:
    runtime_id = PythonRuntimeId.parse(version)
    path = project_runtime_settings_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pythonVersion": runtime_id.series,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return runtime_id.series


def read_project_python_version(project_dir: str | os.PathLike[str]) -> str:
    """Read the shared ABI pin; private runtime folders cannot choose it."""
    path = project_runtime_settings_path(project_dir)
    if not path.is_file():
        raise RuntimeError(
            f"The project must declare its exact Python ABI in {path}. "
            "Restore ProjectSettings/PythonRuntime.json from the project checkout."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if type(payload) is not dict or set(payload) != {"pythonVersion"}:
            raise RuntimeError(f"Invalid project Python runtime settings: {path}")
        return PythonRuntimeId.parse(payload["pythonVersion"]).series
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Invalid project Python runtime settings: {path}") from exc


def project_runtime_directory(
    project_dir: str | os.PathLike[str], version: str | PythonRuntimeId
) -> str:
    runtime_id = PythonRuntimeId.parse(version)
    return os.path.join(str(project_dir), ".runtime", runtime_id.directory_name)


__all__ = [
    "PROJECT_RUNTIME_SETTINGS",
    "project_runtime_directory",
    "project_runtime_settings_path",
    "read_project_python_version",
    "write_project_python_version",
]
