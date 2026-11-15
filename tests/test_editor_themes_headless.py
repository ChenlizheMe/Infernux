"""Theme contracts using the native registry without an engine/window fixture."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from infernux import lib as native
from infernux.engine.editor_themes import EditorThemes
from infernux.engine.preferences_store import PreferencesStore
from infernux.engine.ui.theme import Theme, on_theme_changed
from infernux.theme_data import theme_token_names, validate_theme_document

ROOT = Path(__file__).resolve().parents[1]


class Registry:
    def __init__(self, *references):
        self.records = [{"reference": ref, "version": "1.0.0", "enabled": True} for ref in references]

    def installed(self):
        return tuple(self.records)

    def installed_record(self, reference):
        return next((item for item in self.records if item["reference"].casefold() == reference.casefold()), None)


@pytest.fixture
def isolated_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERNUX_DATA_ROOT", str(tmp_path / "user"))
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(PreferencesStore, "_instance", None)
    monkeypatch.setattr(PreferencesStore, "_initialized", False)
    yield
    native.set_editor_theme("infernux")


@pytest.fixture
def service(tmp_path, isolated_preferences):
    value = EditorThemes(tmp_path, Registry("tests/theme"))
    yield value
    value.close()


def write(service, name="sample.theme.json", document=None, reference="tests/theme"):
    path = service.root / "Packages" / reference / "editor/themes" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document or {"id": "sample"}), encoding="utf-8")
    return path


def test_surface_styles_follow_theme_publication_and_preserve_player_defaults(service):
    document = {"id": "sample", "colors": {
        "ROLE_TEXT": [0.2, 0.3, 0.4, 1],
        "ROLE_BG_SURFACE": [0.9, 0.9, 0.85, 1],
        "PANEL_GRADIENT_TOP": [0.03, 0.12, 0.25, 0.8],
        "PANEL_GRADIENT_BOTTOM": [0.01, 0.04, 0.08, 0.2],
        "BUTTON_ACTIVE_GRADIENT_TOP": [0.1, 0.4, 0.8, 1],
        "SURFACE_HEX_TINT": [0.1, 0.5, 0.9, 1],
    }, "floats": {"PANEL_CORNER_CUT": 10, "PANEL_FRAME_WIDTH": 1.2,
                  "SURFACE_HEX_OPACITY": 0.16, "SURFACE_GRAIN_SCALE": 0.75,
                  "SURFACE_PATTERN_KIND": 2, "SURFACE_PATTERN_OPACITY": 0.3,
                  "PANEL_BEVEL_DEPTH": 3, "PANEL_FASTENER_SIZE": 1.5,
                  "EDITOR_MONO_UI": 1}}
    before = Theme.UI_DEFAULT_BUTTON_BG
    path = write(service, document=document)
    service.refresh_all()
    service.select("tests/theme/sample")
    assert not service.errors
    assert Theme.PANEL_CORNER_CUT == 10
    assert Theme.SCENE_TOOL_ICON == pytest.approx(document["colors"]["ROLE_TEXT"])
    assert Theme.PANEL_GRADIENT_TOP == pytest.approx(document["colors"]["PANEL_GRADIENT_TOP"])
    assert Theme.OBJECT_FIELD_BG == pytest.approx(document["colors"]["ROLE_BG_SURFACE"])
    assert Theme.SURFACE_HEX_OPACITY == pytest.approx(0.16)
    assert Theme.SURFACE_PATTERN_KIND == 2
    assert Theme.PANEL_BEVEL_DEPTH == 3
    assert Theme.EDITOR_MONO_UI == 1
    document["floats"]["PANEL_CORNER_CUT"] = 6
    path.write_text(json.dumps(document), encoding="utf-8")
    service.resource_changed(path)
    assert Theme.PANEL_CORNER_CUT == 6
    service.remove_package("tests/theme")
    assert Theme.PANEL_CORNER_CUT == Theme.SURFACE_HEX_OPACITY == 0
    assert Theme.SURFACE_PATTERN_KIND == Theme.PANEL_BEVEL_DEPTH == Theme.EDITOR_MONO_UI == 0
    assert Theme.UI_DEFAULT_BUTTON_BG == before


@pytest.mark.parametrize("token,value", [
    ("SURFACE_GRAIN_OPACITY", -0.1), ("SURFACE_HEX_OPACITY", 1.1),
    ("SURFACE_GRAIN_SCALE", 0), ("SURFACE_HEX_SCALE", 9),
    ("PANEL_CORNER_CUT", -1), ("BUTTON_GLOW_RADIUS", -1),
    ("SURFACE_PATTERN_OPACITY", 1.1), ("SURFACE_PATTERN_SCALE", 0),
    ("SURFACE_PATTERN_KIND", 1.5), ("SURFACE_PATTERN_KIND", 4), ("EDITOR_MONO_UI", 2),
])
def test_packager_and_native_registry_reject_invalid_surface_dimensions(token, value):
    tokens = theme_token_names((ROOT / "cpp/infernux/function/editor/EditorThemeTable.inl").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        validate_theme_document({"id": "surface", "floats": {token: value}}, tokens)
    with pytest.raises(ValueError):
        native.register_editor_theme("tests/surface", "theme:tests/surface", {}, {}, {token: value})


@pytest.mark.parametrize("document", [
    {"id": None}, {"id": "x", "name": {}}, {"id": "x", "author": []},
    {"id": "x", "description": False}, {"id": "x", "colours": {}},
    {"id": "x", "colors": {"ROLE_ACCENT": [True, 0, 0, 1]}},
    {"id": "x", "colors": {"UI_DEFAULT_BUTTON_BG": [0, 0, 0, 1]}},
    {"id": "x", "vec2s": {"GLOBAL_FRAME_PAD": [-1, 2]}},
])
def test_invalid_declarations_are_unavailable(service, document):
    write(service, document=document)
    service.refresh_all()
    assert service.errors
    assert not service.entries
    assert all(isinstance(row["name"], str) for row in service.catalog())
    assert any(not row["available"] and row["error"] for row in service.catalog())


def test_duplicate_identity_retires_the_contribution(service):
    write(service)
    service.refresh_all()
    service.select("tests/theme/sample")
    write(service, "other.theme.json")
    service.refresh_all()
    assert native.get_editor_theme() == "infernux"
    assert "tests/theme/sample" not in native.list_editor_themes()
    assert len(service.errors) == 2
    assert next(row for row in service.catalog() if row["id"] == service.selection)["available"] is False


def test_remove_cleans_error_and_refresh_all_prunes_owner(service):
    write(service, document={"id": "x", "name": []})
    service.refresh_all()
    service.remove_package("tests/theme")
    assert not service.errors
    assert not any(row["reference"] == "tests/theme" for row in service.catalog())
    write(service)
    service.refresh_all()
    service.select("tests/theme/sample")
    service.registry.records.clear()
    service.refresh_all()
    assert "tests/theme/sample" not in native.list_editor_themes()
    assert native.get_editor_theme() == "infernux"


def test_unchanged_and_metadata_refresh_do_not_republish_active_tokens(service):
    path = write(service)
    service.refresh_all()
    service.select("tests/theme/sample")
    generation, revision = native.editor_theme_generation(), service.revision
    calls = []
    remove = on_theme_changed(lambda: calls.append(True))
    try:
        service.refresh_all()
        assert (native.editor_theme_generation(), service.revision) == (generation, revision)
        assert calls == []
        path.write_text(json.dumps({"id": "sample", "name": "Renamed"}), encoding="utf-8")
        service.resource_changed(path)
        assert native.editor_theme_generation() == generation
        assert service.revision > revision
        assert calls == []
        assert service.catalog() is service.catalog()
    finally:
        remove()


def test_native_replace_and_remove_update_python_mirror_once(service):
    path = write(service, document={"id": "sample", "colors": {"ROLE_ACCENT": [0.1, 0.6, 0.9, 1]}})
    service.refresh_all()
    service.select("tests/theme/sample")
    calls = []
    remove = on_theme_changed(lambda: calls.append(True))
    try:
        path.write_text(json.dumps({"id": "sample", "colors": {"ROLE_ACCENT": [0.2, 0.7, 0.3, 1]}}), encoding="utf-8")
        generation = native.editor_theme_generation()
        service.resource_changed(path)
        assert native.editor_theme_generation() == generation + 1
        assert len(calls) == 1
        assert Theme.ROLE_ACCENT[:3] == pytest.approx((0.2, 0.7, 0.3))
        native.remove_editor_themes("theme:tests/theme")
        assert Theme.ROLE_ACCENT == native.get_editor_theme_colors()["ROLE_ACCENT"]
        assert len(calls) == 2
    finally:
        remove()


def test_nested_reference_routes_to_its_owner(tmp_path, isolated_preferences):
    service = EditorThemes(tmp_path, Registry("tests/theme", "tests/theme/child"))
    try:
        write(service)
        path = write(service, reference="tests/theme/child")
        service.refresh_all()
        path.write_text(json.dumps({"id": "sample", "name": "Child renamed"}), encoding="utf-8")
        service.resource_changed(path)
        assert service.entries["tests/theme/child/sample"]["name"] == "Child renamed"
    finally:
        service.close()


def test_preference_restore_and_invalid_selection_row(service):
    write(service)
    service.refresh_all()
    service.select("tests/theme/sample")
    service.close()
    restarted = EditorThemes(service.root, service.registry)
    try:
        restarted.refresh_all()
        assert native.get_editor_theme() == "tests/theme/sample"
        restarted.registry.records[0]["enabled"] = False
        restarted.refresh_all()
        assert native.get_editor_theme() == "infernux"
        selected = next(row for row in restarted.catalog() if row["id"] == restarted.selection)
        assert not selected["available"] and selected["preferred"]
    finally:
        restarted.close()


def test_public_theme_selection_uses_editor_preferences_and_preserves_failure_fallback(service):
    from infernux.engine.ui.theme import set_editor_theme

    path = write(service)
    service.refresh_all()
    service.select("tests/theme/sample")
    assert Theme.set_theme("amber")
    assert service.selection == PreferencesStore().get("editor_theme") == "amber"
    service.refresh_all()
    assert native.get_editor_theme() == "amber"
    assert set_editor_theme("amber"), "Selecting the current theme still succeeds"
    assert not set_editor_theme("tests/theme/missing")
    assert service.selection == "amber"

    assert set_editor_theme("tests/theme/sample")
    path.write_text("{ broken", encoding="utf-8")
    service.resource_changed(path)
    assert native.get_editor_theme() == "infernux"
    assert service.selection == PreferencesStore().get("editor_theme") == "tests/theme/sample"
    path.write_text('{"id":"sample"}', encoding="utf-8")
    service.resource_changed(path)
    assert native.get_editor_theme() == "tests/theme/sample"
    assert Theme.set_theme("amber")
    service.close()
    restarted = EditorThemes(service.root, service.registry)
    try:
        restarted.refresh_all()
        assert native.get_editor_theme() == "amber"
    finally:
        restarted.close()


def test_public_theme_selection_without_editor_only_changes_native_tokens(isolated_preferences):
    assert EditorThemes.instance() is None
    assert Theme.set_theme("graphite")
    assert native.get_editor_theme() == "graphite"
    assert Theme.ROLE_ACCENT == native.get_editor_theme_colors()["ROLE_ACCENT"]
    assert PreferencesStore().get("editor_theme") is None
    assert not Theme.set_theme("missing")


def export_template(destination):
    path = ROOT / "scripts/plugins/export_theme_template.py"
    spec = importlib.util.spec_from_file_location("theme_template_export_test", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    return exporter.export(destination)


def test_template_packages_offline_and_rejects_bad_data(tmp_path):
    template = export_template(tmp_path / "template")
    archive = tmp_path / "theme.inxpkg"
    command = [sys.executable, "-I", str(template / "package.py"), "build", str(archive)]
    result = subprocess.run(command, cwd=template, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert archive.is_file()
    from infernux.plugins import InxPackage
    assert InxPackage.inspect(str(archive)).metadata["category"] == "theme"
    exported = InxPackage.export_source(str(template), str(tmp_path / "source-export.inxpkg"))
    assert exported.metadata["category"] == "theme"
    assert (template / "theme_schema.py").read_bytes() == (ROOT / "python/infernux/theme_data.py").read_bytes()
    assert (template / "README.zh-CN.md").is_file()
    tokens = json.loads((template / "theme-tokens.json").read_text(encoding="utf-8"))
    examples = sorted((template / "examples").glob("*.theme.json"))
    assert {path.stem.removesuffix(".theme") for path in examples} == {"manuscript", "metal", "cassette", "notion"}
    for example in examples:
        document = validate_theme_document(json.loads(example.read_text(encoding="utf-8")), tokens)
        assert document["id"] == example.stem.removesuffix(".theme")
    theme = template / "package/editor/themes/studio.theme.json"
    theme.write_text('{"id":"x", "name":{}}', encoding="utf-8")
    result = subprocess.run(command, cwd=template, capture_output=True, text=True)
    assert result.returncode != 0
    assert "studio.theme.json" in result.stderr


def test_background_install_finalizes_themes_and_reports_import_errors(tmp_path, isolated_preferences):
    from infernux.plugins import PluginManager
    template = export_template(tmp_path / "template")
    archive = tmp_path / "theme.inxpkg"
    subprocess.run([sys.executable, "-I", str(template / "package.py"), "build", str(archive)], check=True, capture_output=True)
    project = tmp_path / "project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    manager = PluginManager(str(project))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(manager.install_package, str(archive), install_dependencies=False).result()
        assert "example/studio-theme/studio" not in native.list_editor_themes()
        state = manager.finalize_background_install("example/studio-theme")
        assert state.loaded, state.error
        assert manager.registry.installed_metadata()[0]["category"] == "theme"
        assert "example/studio-theme/studio" in native.list_editor_themes()
        path = project / "Packages/example/studio-theme/editor/themes/studio.theme.json"
        path.write_text("{ broken", encoding="utf-8")
        manager._on_script_catalog_changed(str(path), "modified")
        assert manager.states["example/studio-theme"].loaded
        assert manager.states["example/studio-theme"].theme_errors
        assert str(path.resolve()).replace("\\", "/") in manager.states["example/studio-theme"].error.replace("\\", "/")
    finally:
        manager.shutdown()


def test_theme_move_keeps_guid_location_then_retires_when_it_leaves_theme_domain(service, monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.engine.resources_manager import ResourceChangeHandler, ResourcesManager
    path = write(service)
    service.refresh_all()
    service.select("tests/theme/sample")
    moved = service.root / "Assets/relocated.theme.json"
    moved.parent.mkdir()
    path.rename(moved)
    guid = "a" * 32
    service.registry.records[0]["files"] = [{"guid": guid, "logical_path": "editor/themes/sample.theme.json"}]
    service._guid_paths = lambda: {guid: str(moved)}
    handler = ResourceChangeHandler.__new__(ResourceChangeHandler)
    handler._asset_database = SimpleNamespace(get_guid_from_path=lambda value: guid if value == str(path) else "")
    handler._dependency_graph = SimpleNamespace(project_root=str(service.root))
    handler._rejected_compiled_assets = set()
    events = []

    def notify(value, event):
        events.append((value, event))
        if value.lower().endswith(".theme.json"):
            service.resource_changed(value)

    monkeypatch.setattr(AssetManager, "move_asset", lambda *args, **kwargs: SimpleNamespace(database_committed=True))
    monkeypatch.setattr(AssetManager, "reimport_asset", lambda *args, **kwargs: True)
    monkeypatch.setattr(ResourcesManager, "instance", classmethod(lambda cls: SimpleNamespace(notify_script_catalog_changed=notify)))
    handler._commit_moved(str(path), str(moved))
    assert events == [(str(path), "deleted"), (str(moved), "moved")]
    assert service.entries["tests/theme/sample"]["path"] == str(moved.resolve())
    assert native.get_editor_theme() == "tests/theme/sample"
    outside = moved.with_suffix(".txt")
    moved.rename(outside)
    service._guid_paths = lambda: {guid: str(outside)}
    handler._commit_moved(str(moved), str(outside))
    assert "tests/theme/sample" not in native.list_editor_themes()
    assert native.get_editor_theme() == "infernux"


def test_catalog_change_signals_do_not_poll_or_publish_unchanged_data(service, monkeypatch):
    from infernux.host import queries
    signals = []
    monkeypatch.setattr(queries, "emit_signal", lambda name, payload=None: signals.append((name, payload)))
    write(service)
    service.refresh_all()
    assert len(signals) == 1 and signals[0][0] == "themes.changed"
    cached = service.catalog()
    revision = service.revision
    service.refresh_all()
    assert service.catalog() is cached and service.revision == revision
    assert len(signals) == 1
    service.select("tests/theme/sample")
    assert len(signals) == 2
    assert signals[-1][1]["selection"] == "tests/theme/sample"
    native.set_editor_theme("amber")
    assert len(signals) == 3
    assert next(row for row in service.catalog() if row["id"] == "amber")["active"]


def test_package_install_finishes_after_originating_panel_is_retired(tmp_path, isolated_preferences):
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.reload.owners import ReloadOwner
    from infernux.engine.ui.plugin_install_progress import PluginInstallProgressService
    from infernux.engine.ui.plugin_panel import PluginPanel
    from infernux.plugins import PluginManager

    template = tmp_path / "theme-source"
    subprocess.run([sys.executable, str(ROOT / "scripts/plugins/export_theme_template.py"), str(template)],
                   check=True, capture_output=True)
    archive = tmp_path / "theme.inxpkg"
    subprocess.run([sys.executable, "-I", str(template / "package.py"), "build", str(archive)],
                   check=True, capture_output=True)
    project = tmp_path / "project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    manager = PluginManager(str(project))
    core = EditorInteractionCore()
    service = PluginInstallProgressService()
    source_owner = ReloadOwner(("plugin-panel-source",))
    source_owner.active = True
    try:
        with source_owner.enter():
            panel = PluginPanel()
            assert service.begin_package(manager=manager, label=str(archive), action="source",
                                         observer=panel._on_install_finished)
        transaction = service._transaction
        transaction.presented_phase = "opening"
        service.post_present_tick()
        assert transaction.worker_done.wait(10)
        source_owner.close()
        assert not panel._reload_owner.active
        transaction.presented_phase = "running"
        service.post_present_tick()
        transaction.presented_phase = "complete"
        transaction.completed_at -= 1
        service.post_present_tick()
        assert not service.is_active
        state = manager.states["example/studio-theme"]
        assert state.loaded and not state.error
        assert "example/studio-theme/studio" in native.list_editor_themes()
        assert panel._message == "", "An old panel callback was executed after retirement"
        current_panel = PluginPanel()
        current_panel.select_category("theme")
        visible = current_panel._visible_rows(manager)
        assert [row["reference"] for row in visible] == ["example/studio-theme"]
        assert visible[0]["_category_key"] == "theme"
    finally:
        source_owner.close()
        service.shutdown()
        manager.shutdown()
        core.shutdown()


def test_replacing_editor_theme_authority_retires_previous_listener_and_contributions(tmp_path, isolated_preferences):
    first = EditorThemes(tmp_path / "first", Registry("first/theme"))
    write(first, reference="first/theme")
    first.refresh_all()
    first.select("first/theme/sample")
    second = EditorThemes(tmp_path / "second", Registry("second/theme"))
    try:
        assert not first.entries and not first.owners
        assert "first/theme/sample" not in native.list_editor_themes()
        write(second, reference="second/theme")
        second.refresh_all()
        revision = first.revision
        second.select("second/theme/sample")
        assert first.revision == revision
        first.close()
        assert native.get_editor_theme() == "second/theme/sample"
    finally:
        second.close()


def test_theme_menu_reveals_the_same_open_window(isolated_preferences):
    from infernux.engine._bootstrap_wiring import BootstrapWiringMixin
    from infernux.engine.interaction import CommandSource, EditorInteractionCore, PanelInteractionDescriptor
    from infernux.engine.ui.theme_panel import ThemePanel
    from infernux.engine.ui.window_manager import WindowManager

    core = EditorInteractionCore()
    registered = {}
    engine = SimpleNamespace(
        select_docked_window=lambda *args, **kwargs: None,
        unregister_gui=lambda view_id: registered.pop(view_id, None),
    )
    windows = WindowManager(engine, core.panels, lambda view_id, view: registered.update({view_id: view}))
    core.panels.register_type("themes", PanelInteractionDescriptor())
    windows.register_window_type("themes", ThemePanel, "Themes", factory=ThemePanel, menu_path="")
    harness = SimpleNamespace(interaction_core=core, engine=None)
    BootstrapWiringMixin._register_core_editor_commands(harness, windows, None)
    menu = native.MenuBarPanel()
    menu.execute_command = lambda command, source, argument: core.commands.execute(
        command, source=CommandSource.MENU,
    ).accepted
    try:
        assert menu.execute_command("window.open.themes", "menu", "")
        windows.process_pending_actions()
        first = windows.get_window_instance("themes")
        assert first.window_id == "themes" and windows.is_window_open("themes")
        assert menu.execute_command("window.open.themes", "menu", "")
        windows.process_pending_actions()
        assert windows.is_window_open("themes")
        assert windows.get_window_instance("themes") is first
        assert registered == {"themes": first}
        assert core.commands.get("window.toggle.themes") is None
    finally:
        windows.close_window("themes", force=True)
        windows.process_pending_actions()
        core.shutdown()
