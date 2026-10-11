"""Physics settings — load / save / apply project-level physics configuration.

Manages gravity, fixed timestep, and max fixed delta time.  Settings are
persisted as ``ProjectSettings/PhysicsSettings.json`` inside the project.

Example::

    from infernux.physics import settings
    cfg = settings.load("/path/to/project")
    cfg["gravity"] = [0, -20, 0]
    settings.save("/path/to/project", cfg)
    settings.apply(cfg)
"""

from __future__ import annotations

from typing import Dict

DEFAULT_PHYSICS_SETTINGS: Dict[str, object]
"""Default physics values: gravity ``[0,-9.81,0]``, fixed_delta_time ``0.02``, max ``0.1``."""

def settings_path(project_path: str) -> str:
    """Return the absolute path to ``PhysicsSettings.json`` for *project_path*."""
    ...

def load(project_path: str) -> dict:
    """Load physics settings from disk, falling back to :data:`DEFAULT_PHYSICS_SETTINGS`.

    Args:
        project_path: Root path of the Infernux project.
    """
    ...

def save(project_path: str, settings: dict) -> None:
    """Persist physics settings to ``ProjectSettings/PhysicsSettings.json``.

    Args:
        project_path: Root path of the Infernux project.
        settings: Dict with keys ``gravity``, ``fixed_delta_time``, ``max_fixed_delta_time``.
    """
    ...

def apply(settings: dict) -> None:
    """Push *settings* into the live engine (gravity, fixed timestep, etc.).

    Updates :class:`Physics.gravity`, SceneManager fixed-step, and
    :data:`Time.fixed_delta_time` in one call.

    Args:
        settings: Dict with keys ``gravity``, ``fixed_delta_time``, ``max_fixed_delta_time``.
    """
    ...
