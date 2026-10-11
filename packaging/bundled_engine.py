"""Seed engine wheels shipped by the installer into the shared Engines cache.

The installer embeds the engine wheel of its own release under
``<app dir>/InfernuxHubData/engines`` so a fresh installation can create its
first project without network access. The wheel is not part of the Hub update
manifest; this module copies it into the Engines cache through
``VersionManager.install_local_wheel`` (which validates the wheel identity).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from hub_utils import get_hub_data_dir


BUNDLED_ENGINES_DIRNAME = "engines"
# Lives in the Engines cache; names starting with "_" are never engine entries.
SEEDED_STATE_FILENAME = "_bundled_engines_seeded.json"

_LOGGER = logging.getLogger(__name__)


def bundled_engines_dir(hub_data_dir: str | os.PathLike[str] | None = None) -> Path:
    return Path(hub_data_dir or get_hub_data_dir()) / BUNDLED_ENGINES_DIRNAME


def _default_state_path() -> Path:
    import version_manager

    return Path(version_manager._VERSIONS_DIR) / SEEDED_STATE_FILENAME


def _read_seeded(path: Path) -> set[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as exc:
        _LOGGER.warning("Ignoring unreadable bundled engine state %s: %s", path, exc)
        return set()
    wheels = document.get("seeded") if isinstance(document, dict) else None
    if not isinstance(wheels, list):
        return set()
    return {name for name in wheels if isinstance(name, str)}


def _write_seeded(path: Path, seeded: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps({"seeded": sorted(seeded)}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_bundled_engines(
    version_manager,
    *,
    hub_data_dir: str | os.PathLike[str] | None = None,
    state_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Install engine wheels bundled with the Hub into the Engines cache.

    Call once at Hub startup, after the managed Python runtime is available::

        from bundled_engine import seed_bundled_engines
        seed_bundled_engines(self.version_manager)

    Every ``infernux-*.whl`` in ``<hub data dir>/engines`` (default: the
    running Hub's ``InfernuxHubData/engines``) is installed with
    ``version_manager.install_local_wheel`` unless it is already installed.
    Each wheel is seeded at most once: successful (or already-present) wheels
    are recorded in ``state_path`` (default:
    ``Engines/_bundled_engines_seeded.json``), so an engine the user later
    removes is not silently reinstalled on the next start.

    Wheels whose Python runtime is not installed yet are skipped and retried on
    a later start. Errors are logged and never raised.

    Returns the engine releases installed by this call.
    """
    try:
        from version_manager import (
            wheel_platform_compatible,
            wheel_python_version,
            wheel_release,
        )

        source = bundled_engines_dir(hub_data_dir)
        wheels = sorted(source.glob("infernux-*.whl")) if source.is_dir() else []
        if not wheels:
            return []
        state = Path(state_path) if state_path else _default_state_path()
        seeded = _read_seeded(state)
    except Exception:
        _LOGGER.exception("Could not inspect bundled engine wheels")
        return []

    installed: list[str] = []
    changed = False
    for wheel in wheels:
        if wheel.name in seeded or not wheel.is_file():
            continue
        try:
            release = wheel_release(wheel.name)
            python_version = wheel_python_version(wheel.name)
            if not release or not python_version or not wheel_platform_compatible(wheel.name):
                _LOGGER.warning("Ignoring incompatible bundled engine wheel: %s", wheel.name)
                continue
            if not version_manager.is_python_runtime_installed(python_version):
                _LOGGER.info(
                    "Deferring bundled engine %s until Python %s is installed",
                    wheel.name,
                    python_version,
                )
                continue
            if not version_manager.is_installed(release, python_version):
                version_manager.install_local_wheel(str(wheel))
                installed.append(release)
                _LOGGER.info("Installed bundled engine %s", wheel.name)
            seeded.add(wheel.name)
            changed = True
        except Exception:
            _LOGGER.exception("Could not install bundled engine wheel %s", wheel)

    if changed:
        try:
            _write_seeded(state, seeded)
        except OSError:
            _LOGGER.exception("Could not record bundled engine state in %s", state)
    return installed
