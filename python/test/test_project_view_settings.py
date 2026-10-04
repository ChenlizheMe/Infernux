from __future__ import annotations

import importlib

import pytest

from infernux.engine.project_view_settings import (
    load_project_view_settings,
    project_view_settings_path,
    write_project_view_settings_section,
)


def test_local_project_view_settings_preserve_other_sections(tmp_path):
    path = project_view_settings_path(str(tmp_path))

    write_project_view_settings_section(path, "UIEditor", {"zoom": "1.25"})
    write_project_view_settings_section(path, "GameView", {"preset_index": "2"})
    parser = load_project_view_settings(path)
    assert parser["UIEditor"]["zoom"] == "1.25"
    assert parser["GameView"]["preset_index"] == "2"

    write_project_view_settings_section(path, "UIEditor", {"zoom": "0.75"})
    parser = load_project_view_settings(path)
    assert parser["UIEditor"]["zoom"] == "0.75"
    assert parser["GameView"]["preset_index"] == "2"


@pytest.mark.parametrize("panel_module,class_name,loader", [
    ("game_view_panel", "GameViewPanel", "_load_resolution_settings"),
    ("ui_editor_panel", "UIEditorPanel", "_load_view_settings"),
])
def test_opening_editor_views_does_not_create_shared_settings(tmp_path, monkeypatch, panel_module, class_name, loader):
    module = importlib.import_module("infernux.engine.ui." + panel_module)
    monkeypatch.setattr(module, "get_project_root", lambda: str(tmp_path))
    panel = getattr(module, class_name)()
    getattr(panel, loader)()
    assert panel._settings_ini_path() == project_view_settings_path(str(tmp_path))
    assert not (tmp_path / "ProjectSettings").exists()
    assert not (tmp_path / "Library" / "GameView.ini").exists()


def test_old_shared_view_settings_are_ignored(tmp_path):
    legacy = tmp_path / "ProjectSettings" / "GameView.ini"
    legacy.parent.mkdir()
    original = b"[GameView]\npreset_index = 2\n\n[UIEditor]\nzoom = 1.25\n"
    legacy.write_bytes(original)
    local = project_view_settings_path(str(tmp_path))
    assert load_project_view_settings(local).sections() == []
    write_project_view_settings_section(local, "UIEditor", {"zoom": "0.75"})
    assert legacy.read_bytes() == original
    parser = load_project_view_settings(local)
    assert parser["UIEditor"]["zoom"] == "0.75"
    assert "GameView" not in parser

    legacy.write_text("[GameView]\npreset_index = 4\n", encoding="utf-8")
    assert "GameView" not in load_project_view_settings(local)
