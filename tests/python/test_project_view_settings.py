from __future__ import annotations

import importlib
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    "panel_module,class_name,loader",
    [
        ("game_view_panel", "GameViewPanel", "_load_resolution_settings"),
        ("ui_editor_panel", "UIEditorPanel", "_load_view_settings"),
    ],
)
def test_opening_editor_views_does_not_create_shared_settings(tmp_path, monkeypatch, panel_module, class_name, loader):
    module = importlib.import_module("infernux.engine.ui." + panel_module)
    monkeypatch.setattr(module, "get_project_root", lambda: str(tmp_path))
    panel = getattr(module, class_name)()
    getattr(panel, loader)()
    assert panel._settings_ini_path() == project_view_settings_path(str(tmp_path))
    assert not (tmp_path / "ProjectSettings").exists()
    assert not (tmp_path / "Library" / "GameView.ini").exists()


def test_fixed_project_output_does_not_lock_user_game_view_resolution(tmp_path, monkeypatch):
    from infernux.engine.interaction import ViewCommandService
    from infernux.engine.ui import game_view_panel as module
    from infernux.engine.undo import UndoManager

    monkeypatch.setattr(module, "get_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(UndoManager, "_instance", None)
    monkeypatch.setattr(ViewCommandService, "_instance", None)
    resized = []
    engine = SimpleNamespace(
        get_play_mode_manager=lambda: None,
        resize_game_render_target=lambda width, height, **options: resized.append((width, height, options)),
        get_game_texture_id=lambda: 1,
    )
    panel = module.GameViewPanel(engine=engine)
    panel._display_scale, panel._fit_mode = 0.375, False
    panel._apply_project_output({"build": {"render_output": {"width": 480, "height": 270, "filter": "nearest"}}})

    class Toolbar:
        selected = 0
        combos = []
        drags = []

        def combo(self, name, current, choices, height):
            self.combos.append((name, current, tuple(choices)))
            return self.selected

        def set_next_item_width(self, width):
            pass

        def same_line(self, *args):
            pass

        def label(self, text):
            pass

        def drag_int(self, name, current, *args):
            self.drags.append(name)
            return 1600 if name == "##CW" else 1000

    toolbar = Toolbar()
    for selected, expected in ((0, (1920, 1080)), (2, (2560, 1440)), (5, (1600, 1000))):
        toolbar.selected = selected
        panel._render_resolution_toolbar(toolbar, 1.0)
        assert toolbar.combos and toolbar.combos[-1][0] == "##Resolution"
        assert panel._current_target_resolution() == expected
        panel.prepare_render_target()
        assert resized[-1] == (*expected, {"point_filter": True})
        assert panel.render_target_size == expected

    selected_state = panel._capture_view_state()
    panel._apply_project_output({"build": {"render_output": {"width": 960, "height": 540, "filter": "linear"}}})
    assert panel._capture_view_state() == selected_state
    panel.prepare_render_target()
    assert resized[-1] == (1600, 1000, {"point_filter": False})
    assert panel._display_scale == 0.375 and panel._fit_mode is False
    assert len(toolbar.combos) == 3

    manager = UndoManager()
    toolbar.selected = panel._AUTO_RESOLUTION_INDEX
    panel._render_resolution_toolbar(toolbar, 1.0)
    assert toolbar.combos[-1][2][5:] == ("Custom", "Auto")
    assert toolbar.drags == ["##CW", "##CH"]
    assert panel._fit_mode is True
    assert panel._current_target_resolution() == (0, 0)
    before_measurement = len(resized)
    panel.prepare_render_target()
    assert len(resized) == before_measurement
    assert panel._render_scale_toolbar(toolbar, 1.0) == (0, 0, 1.0)

    screen = SimpleNamespace(logical_width=1200, logical_height=800,
                             framebuffer_width=2400, framebuffer_height=1600, pixel_ratio=3.0)
    published_rects = []
    native = SimpleNamespace(screen_state=screen,
                             set_editor_game_viewport=lambda *rects: published_rects.append(rects))
    monkeypatch.setattr(module, "_NativeInputManager", SimpleNamespace(instance=lambda: native))
    monkeypatch.setattr(module.Input, "_game_viewport_origin", (0.0, 0.0))
    monkeypatch.setattr(module.Input, "_game_viewport_size", (0.0, 0.0))
    monkeypatch.setattr(module, "capture_viewport_info", lambda ctx: SimpleNamespace(image_min_x=20, image_min_y=30))
    routed_sizes = []
    monkeypatch.setattr(panel, "_route_game_input", lambda ctx, w, h, clicked: routed_sizes.append((w, h)))
    monkeypatch.setattr(panel, "_render_screen_ui", lambda *args: None)

    class Viewport:
        available = (600.0, 300.0)
        child = False
        images = []

        def begin_child(self, *args):
            self.child = True
            return True

        def end_child(self):
            self.child = False

        def get_content_region_avail_width(self):
            return self.available[0] if self.child else 1100.0

        def get_content_region_avail_height(self):
            return self.available[1] if self.child else 750.0

        def get_cursor_pos_x(self):
            return 0.0

        get_cursor_pos_y = get_cursor_pos_x

        def set_cursor_pos_x(self, value):
            assert value == 0.0

        set_cursor_pos_y = set_cursor_pos_x

        def image(self, texture, width, height, *uv):
            self.images.append((width, height))

        def invisible_button(self, name, width, height):
            assert (width, height) == self.available

        def record_semantic_item(self, *args):
            pass

        def is_item_clicked(self, button):
            return False

        def get_item_visible_rect(self):
            return (20, 30, *self.available)

    viewport = Viewport()
    for available, expected in (((600.0, 300.0), (1200, 600)), ((400.0, 500.0), (800, 1000))):
        viewport.available = available
        panel._render_game_viewport(viewport, *panel._current_target_resolution(), 1.0)
        assert resized[-1] == (*expected, {"point_filter": False})
        assert panel.render_target_size == expected == routed_sizes[-1]
        assert viewport.images[-1] == available
        assert published_rects[-1][0] == (20, 30, *available)
        assert panel._display_scale == 0.5

    assert panel._render_scale_toolbar(toolbar, 1.0) == (800, 1000, 1.0)
    auto_state = panel._capture_view_state()
    panel._apply_project_output({"build": {"render_output": {"width": 480, "height": 270, "filter": "nearest"}}})
    assert panel._capture_view_state() == auto_state
    panel.prepare_render_target()
    assert resized[-1] == (800, 1000, {"point_filter": True})
    panel._save_resolution_settings()
    assert load_project_view_settings(panel._settings_ini_path())["GameView"].getint("preset_index") == panel._AUTO_RESOLUTION_INDEX
    manager.undo()
    assert panel._current_target_resolution() == (1600, 1000)
    assert panel._fit_mode is False
    manager.redo()
    assert panel._current_target_resolution() == (0, 0)
    assert panel._fit_mode is True
    panel._set_resolution_preset(2)
    assert panel._current_target_resolution() == (2560, 1440)
    assert panel._fit_mode is True
