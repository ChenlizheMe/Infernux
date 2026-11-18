"""Editor theme packages.

A theme package is an ordinary InxPackage whose control root carries
``themes/*.json``. Each file registers one editor theme:

    {"name": "...", "base": "infernux",
     "colors": {"TOKEN": [r, g, b, a]}, "vec2s": {"TOKEN": [x, y]},
     "floats": {"TOKEN": 1.0}}

Theme packages are listed in the Themes panel rather than the plugin list.
"""

from __future__ import annotations

import glob
import json
import os

from .package import package_control_root

THEMES_DIRECTORY = "themes"

_THEME_PACKAGES: dict[str, tuple[str, ...]] = {}
_THEME_ERRORS: dict[str, str] = {}


def _load_theme_file(path: str, reference: str = "") -> str:
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("theme document must be an object")
    name = str(document.get("name") or os.path.splitext(os.path.basename(path))[0]).strip()
    if not name:
        raise ValueError("theme name is empty")
    from infernux.engine.ui.theme import register_editor_theme

    register_editor_theme(
        name,
        {key: tuple(value) for key, value in dict(document.get("colors") or {}).items()},
        {key: tuple(value) for key, value in dict(document.get("vec2s") or {}).items()},
        {key: float(value) for key, value in dict(document.get("floats") or {}).items()},
        owner=reference or None,
    )
    return name


def scan_installed_themes(project_root: str, registry) -> dict[str, tuple[str, ...]]:
    """Register every theme shipped by installed packages."""

    _THEME_PACKAGES.clear()
    _THEME_ERRORS.clear()
    for record in registry.installed():
        reference = str(record.get("reference") or "")
        if not reference:
            continue
        directory = os.path.join(package_control_root(project_root, reference), THEMES_DIRECTORY)
        files = sorted(glob.glob(os.path.join(directory, "*.json")))
        if not files:
            continue
        names: list[str] = []
        for path in files:
            # Theme files come from third-party packages; one bad file must not
            # break the plugin manager rebuild.
            try:
                names.append(_load_theme_file(path, reference))
            except (OSError, ValueError, TypeError) as exc:
                _THEME_ERRORS[reference] = f"{os.path.basename(path)}: {exc}"
        _THEME_PACKAGES[reference] = tuple(names)
    return dict(_THEME_PACKAGES)


def is_theme_package(reference: str) -> bool:
    key = str(reference).casefold()
    return any(name.casefold() == key for name in _THEME_PACKAGES)


def theme_packages() -> dict[str, tuple[str, ...]]:
    return dict(_THEME_PACKAGES)


def theme_package_errors() -> dict[str, str]:
    return dict(_THEME_ERRORS)


__all__ = [
    "THEMES_DIRECTORY",
    "is_theme_package",
    "scan_installed_themes",
    "theme_package_errors",
    "theme_packages",
]
