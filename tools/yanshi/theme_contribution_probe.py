"""Verify editor theme contributions on 050 without a GUI.

The native registry gains package-owned themes (048's contribution model) while
050 keeps its built-in table and ImGui palette. This probe checks the native
catalog lifecycle and the Python theme service/validator, all without an ImGui
context (palette application is skipped when no context exists).
"""

from __future__ import annotations

import json
import os
import tempfile

import infernux  # noqa: F401


def run() -> dict:
    from infernux.lib import (
        editor_theme_generation,
        get_editor_theme,
        list_editor_themes,
        remove_editor_themes,
        replace_editor_themes,
        set_editor_theme,
    )

    builtin = list(list_editor_themes())
    definition = {
        "name": "probe/ember",
        "colors": {"ROLE_ACCENT": (1.0, 0.35, 0.1, 1.0)},
        "vec2s": {},
        "floats": {"EDITOR_TITLE_FONT_SIZE": 15.0},
    }
    before = editor_theme_generation()
    replace_editor_themes("probe/owner", [definition], "")
    registered = list(list_editor_themes())
    generation_bumped = editor_theme_generation() > before
    selected = bool(set_editor_theme("probe/ember"))
    active = str(get_editor_theme())
    remove_editor_themes("probe/owner")
    removed = list(list_editor_themes())
    fallback = str(get_editor_theme())

    root = tempfile.mkdtemp(prefix="theme_service_")
    os.makedirs(os.path.join(root, "ProjectSettings"), exist_ok=True)
    os.makedirs(os.path.join(root, "Packages"), exist_ok=True)
    from infernux.plugins.registry import PluginRegistry
    from infernux.engine.editor_themes import EditorThemes
    from infernux.theme_data import validate_theme_document

    service = EditorThemes(root, PluginRegistry(root))
    catalog = service.catalog()
    document = {
        "$schema": "infernux.editor_theme",
        "id": "ember",
        "name": "Ember",
        "author": "probe",
        "description": "warm accent",
        "colors": {"ROLE_ACCENT": [1.0, 0.35, 0.1, 1.0]},
        "vec2s": {},
        "floats": {"EDITOR_TITLE_FONT_SIZE": 15.0},
    }
    normalized = validate_theme_document(document, service._tokens)
    rejected = False
    try:
        validate_theme_document(dict(document, unknown_field=True), service._tokens)
    except ValueError:
        rejected = True
    service.close()

    failures = []
    if "probe/ember" not in registered:
        failures.append("contributed theme was not registered")
    if not generation_bumped:
        failures.append("generation did not bump")
    if not selected or active != "probe/ember":
        failures.append(f"selection failed: selected={selected} active={active}")
    if "probe/ember" in removed:
        failures.append("contributed theme survived removal")
    if fallback != "infernux":
        failures.append(f"removal did not fall back to the default: {fallback}")
    if not catalog:
        failures.append("theme service catalog is empty")
    if normalized.get("id") != "ember":
        failures.append("valid theme document was not normalized")
    if not rejected:
        failures.append("invalid theme document was accepted")

    return {
        "ok": not failures,
        "failures": failures,
        "builtin": builtin,
        "registered": registered,
        "removed": removed,
        "catalog_size": len(catalog),
        "validated": normalized,
        "invalid_rejected": rejected,
    }


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
