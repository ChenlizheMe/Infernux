"""Theme data participates in normal plugin installation and retirement."""

import json
import pytest
from pathlib import Path
from infernux.plugins import PluginManager
from infernux.engine.preferences_store import PreferencesStore
from infernux.engine.ui.theme import Theme, active_editor_theme, list_editor_themes
from test_inxpackage_plugins import _source, _project, _export, _fake_inxpack as _fake_inxpack


def test_theme_install_edit_failure_disable_and_user_preference(tmp_path, monkeypatch):
    preferences = {}
    monkeypatch.setattr(PreferencesStore, "get", lambda self, key, default=None: preferences.get(key, default))
    monkeypatch.setattr(PreferencesStore, "set", lambda self, key, value: preferences.update({key: value}))
    project = _project(tmp_path / "project")
    source = _source(tmp_path / "theme", "vendor/sky")
    themes = source / "editor/themes"
    themes.mkdir(parents=True)
    document = {"id": "sky", "name": "青空", "colors": {"ROLE_ACCENT": [0.2, 0.5, 0.8, 1.0]},
                "vec2s": {"CONTENT_SPACING": [9, 7]}}
    (themes / "sky.theme.json").write_text(json.dumps(document), encoding="utf-8")
    manager = PluginManager(str(project))
    try:
        state = manager.install_package(str(_export(source, tmp_path / "theme.inxpkg")), install_dependencies=False)
        assert state.loaded
        assert "vendor/sky/sky" in list_editor_themes()
        game_default = Theme.UI_DEFAULT_LABEL_COLOR
        manager.themes.select("vendor/sky/sky")
        assert active_editor_theme() == "vendor/sky/sky"
        assert preferences["editor_theme"] == "vendor/sky/sky"
        assert tuple(Theme.ROLE_ACCENT[:3]) == pytest.approx((0.2, 0.5, 0.8), abs=1e-6)
        assert Theme.CONTENT_SPACING == (9.0, 7.0)
        path = project / "Packages/vendor/sky/editor/themes/sky.theme.json"
        path.write_text("{ broken", encoding="utf-8")
        manager.themes.resource_changed(str(path))
        assert active_editor_theme() == "infernux"
        assert "vendor/sky/sky" not in list_editor_themes()
        assert manager.themes.errors
        path.write_text(json.dumps(document), encoding="utf-8")
        manager.themes.resource_changed(str(path))
        assert active_editor_theme() == "vendor/sky/sky"
        manager.set_enabled("vendor/sky", False)
        assert active_editor_theme() == "infernux"
        assert "vendor/sky/sky" not in list_editor_themes()
        assert Theme.UI_DEFAULT_LABEL_COLOR == game_default
    finally:
        manager.shutdown()
