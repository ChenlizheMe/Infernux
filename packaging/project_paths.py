"""Path and metadata helpers shared by Infernux Hub project workflows."""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_INVALID_NAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class ProjectPathError(ValueError):
    """Raised when a selected path is not a safe Infernux project path."""


@dataclass(frozen=True)
class ExistingProjectInfo:
    name: str
    path: str
    engine_version: str


def canonical_project_path(path: str) -> str:
    """Return an absolute display path with aliases resolved."""
    if not path or not str(path).strip():
        raise ProjectPathError("Project path cannot be empty.")
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path).strip())))


def normalize_project_path(path: str) -> str:
    """Return a stable, case-normalized key for database identity."""
    return os.path.normcase(canonical_project_path(path))


def validate_project_name(name: str) -> str:
    """Validate that *name* is one safe directory component."""
    value = str(name or "").strip()
    if not value:
        raise ProjectPathError("Project name cannot be empty.")
    if value in {".", ".."} or os.path.basename(value) != value or _INVALID_NAME_CHARACTERS.search(value):
        raise ProjectPathError("Project name must be a single directory name without path characters.")
    if value.endswith((" ", ".")):
        raise ProjectPathError("Project name cannot end with a space or period.")
    stem = value.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise ProjectPathError(f"'{value}' is a reserved Windows name.")
    return value


def new_project_target(parent: str, name: str) -> tuple[str, str]:
    """Validate a new project location and return ``(parent, target)``."""
    safe_name = validate_project_name(name)
    parent_path = canonical_project_path(parent)
    if not os.path.isdir(parent_path):
        raise ProjectPathError(f"Project location does not exist:\n{parent_path}")
    target = canonical_project_path(os.path.join(parent_path, safe_name))
    try:
        contained = os.path.normcase(os.path.commonpath((parent_path, target))) == os.path.normcase(parent_path)
    except ValueError:
        contained = False
    if not contained or target == parent_path:
        raise ProjectPathError("Project directory must stay inside the selected location.")
    return parent_path, target


def _read_project_name(project_path: str) -> str:
    candidates = sorted(Path(project_path).glob("*.ini"))
    for candidate in candidates:
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(candidate, encoding="utf-8")
            value = parser.get("Project", "name", fallback="").strip()
            if value:
                return value
        except (OSError, configparser.Error):
            continue
    return os.path.basename(project_path)


def _read_engine_version(project_path: str) -> str:
    version_path = os.path.join(project_path, ".infernux-version")
    try:
        with open(version_path, "r", encoding="utf-8") as stream:
            versions = [line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")]
            return versions[0] if len(versions) == 1 else ""
    except OSError:
        pass
    return ""


def inspect_existing_project(path: str) -> ExistingProjectInfo:
    """Validate and describe an existing Infernux project directory."""
    project_path = canonical_project_path(path)
    if not os.path.isdir(project_path):
        raise ProjectPathError(f"Project directory does not exist:\n{project_path}")

    missing = [
        name
        for name in ("Assets", "ProjectSettings")
        if not os.path.isdir(os.path.join(project_path, name))
    ]
    if missing:
        raise ProjectPathError(
            "The selected directory is not an Infernux project. "
            f"Missing: {', '.join(missing)}."
        )

    name = _read_project_name(project_path)
    if not name:
        raise ProjectPathError("The selected project has no usable name.")
    return ExistingProjectInfo(name=name, path=project_path, engine_version=_read_engine_version(project_path))


__all__ = [
    "ExistingProjectInfo",
    "ProjectPathError",
    "canonical_project_path",
    "inspect_existing_project",
    "new_project_target",
    "normalize_project_path",
    "validate_project_name",
]
