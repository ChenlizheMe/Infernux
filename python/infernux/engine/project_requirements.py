"""Generic project-level dependency checker.

On every engine startup (editor or player) this module reads
``<project>/ProjectSettings/requirements.txt`` and ensures that every
listed package satisfies its version/source constraint and is importable. In editor mode missing packages are
installed automatically; in player mode the check is informational only.

Users can customise their project environment by editing the file.
"""

from __future__ import annotations

import importlib.util
import importlib.metadata
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from infernux.version import ENGINE_VERSION
from .path_utils import process_executable_path

_log = logging.getLogger("infernux.project_requirements")

# Packages whose importable name differs from the pip name.
_IMPORT_NAME_MAP: dict[str, str] = {
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "ordered-set": "ordered_set",
}


# ── Helpers ──────────────────────────────────────────────────────────

def _run_python(args: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    return subprocess.run([sys.executable, *args], executable=process_executable_path(sys.executable), **kwargs)


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ensure_pip() -> bool:
    completed = _run_python(["-m", "pip", "--version"], timeout=60)
    if completed.returncode == 0:
        return True
    completed = _run_python(["-m", "ensurepip", "--upgrade"], timeout=600)
    if completed.returncode != 0:
        return False
    completed = _run_python(["-m", "pip", "--version"], timeout=60)
    return completed.returncode == 0


def _pip_name_to_import(pip_name: str) -> str:
    """Best-effort conversion of a pip package name to an importable module."""
    key = canonicalize_name(pip_name)
    if key in _IMPORT_NAME_MAP:
        return _IMPORT_NAME_MAP[key]
    # Common convention: dashes → underscores
    return key.replace("-", "_")


def _parse_requirements(path: str) -> list[tuple[str, str]]:
    """Return ``[(pip_spec, import_name), ...]`` from a requirements file."""
    entries: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                raise ValueError("Project requirements must contain PEP 508 requirements, not pip options")
            spec = line.split(" #", 1)[0].strip()
            requirement = Requirement(spec)
            if canonicalize_name(requirement.name) == "infernux":
                raise ValueError(
                    "Infernux is supplied by the exact engine version pinned in "
                    ".infernux-version; project requirements cannot install the engine"
                )
            if requirement.marker is None or requirement.marker.evaluate():
                entries.append((spec, _pip_name_to_import(requirement.name)))
    return entries


def _has_requirement(spec: str, import_name: str) -> bool:
    requirement = Requirement(spec)
    try:
        distribution = importlib.metadata.distribution(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        return False
    if not requirement.specifier.contains(distribution.version, prereleases=True):
        return False
    if requirement.url:
        import json
        direct_url = distribution.read_text("direct_url.json")
        if direct_url is None or json.loads(direct_url).get("url") != requirement.url:
            return False
    return _has_module(import_name)


def _install_packages(specs: list[str], *, project_path: str) -> bool:
    """pip-install a list of requirement specifiers."""
    cache = Path(project_path) / "Cache" / "Python"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="install-", dir=cache) as workspace:
        constraint = Path(workspace) / "engine-constraint.txt"
        constraint.write_text(f"Infernux=={ENGINE_VERSION}\n", encoding="utf-8", newline="\n")
        completed = _run_python(
            [
                "-m", "pip", "install",
                "--constraint", str(constraint),
                "--disable-pip-version-check",
                "--no-input",
                "--prefer-binary",
                "--upgrade",
                *specs,
            ],
            timeout=1800,
        )
    return completed.returncode == 0


# ── Public API ───────────────────────────────────────────────────────

def requirements_path(project_path: str) -> str:
    """Return the canonical path to the project requirements file."""
    return os.path.join(project_path, "ProjectSettings", "requirements.txt")


def ensure_project_requirements(
    project_path: str,
    *,
    auto_install: bool = True,
) -> bool:
    """Check (and optionally install) packages listed in ProjectSettings/requirements.txt.

    Returns ``True`` when all active requirements are satisfied after the check.
    Authored requirements must already exist; startup never seeds shared files.
    """
    req_file = requirements_path(project_path)
    if not os.path.isfile(req_file):
        _log.error("Project requirements file is missing; restore it from the project checkout: %s", req_file)
        return False

    entries = _parse_requirements(req_file)
    if not entries:
        return True

    # Find missing packages
    missing: list[tuple[str, str]] = []  # (pip_spec, import_name)
    for pip_spec, import_name in entries:
        if not _has_requirement(pip_spec, import_name):
            missing.append((pip_spec, import_name))

    if not missing:
        return True

    names = ", ".join(m for _, m in missing)

    if not auto_install:
        _log.warning("Project requirements check: missing packages: %s", names)
        return False

    _log.info("Auto-installing missing project requirements: %s", names)

    if not _ensure_pip():
        _log.warning("Project requirements check failed: pip is unavailable.")
        return False

    specs = [s for s, _ in missing]
    if not _install_packages(specs, project_path=project_path):
        _log.warning("Project requirements check failed: pip install returned an error.")
        return False

    # Verify that all packages are now importable
    importlib.invalidate_caches()
    still_missing = [m for spec, m in entries if not _has_requirement(spec, m)]
    if still_missing:
        _log.warning(
            "Project requirements check: still missing after install: %s",
            ", ".join(still_missing),
        )
        return False

    return True
