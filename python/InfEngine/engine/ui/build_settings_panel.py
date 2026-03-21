"""
Build Settings — Unity-style floating window for managing game builds.

NOT a dockable panel.  Rendered by MenuBarPanel each frame when visible;
never registered through WindowManager / engine.register_gui().

Features:
  * Scene list (drag-drop from Project panel or "Add Open Scene")
  * Output directory picker
  * Display mode: Fullscreen Borderless / Windowed (custom size)
  * Splash items list: images (fade in/out + duration) and videos
  * Build / Build & Run with background progress
"""

import os
import json
import threading
from typing import Dict, List, Optional

from InfEngine.debug import Debug
from InfEngine.engine.project_context import get_project_root
from InfEngine.engine.i18n import t
from .theme import Theme, ImGuiCol, ImGuiStyleVar


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

BUILD_SETTINGS_FILE = "BuildSettings.json"

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}

_DISPLAY_MODES_KEYS = ["build.fullscreen_borderless", "build.windowed"]
_DISPLAY_MODE_KEYS = ["fullscreen_borderless", "windowed"]


def _settings_path() -> Optional[str]:
    root = get_project_root()
    if not root:
        return None
    return os.path.join(root, "ProjectSettings", BUILD_SETTINGS_FILE)


def load_build_settings() -> dict:
    """Load BuildSettings.json."""
    path = _settings_path()
    if not path or not os.path.isfile(path):
        return {"scenes": []}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        data = {"scenes": []}
    if "scenes" not in data:
        data["scenes"] = []
    return data


def save_build_settings(settings: dict):
    path = _settings_path()
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Drag-drop type & style constants
# ---------------------------------------------------------------------------

DRAG_DROP_SCENE = "SCENE_FILE"
DRAG_DROP_REORDER = "BUILD_REORDER"
_DRAG_TARGET_COLOR = Theme.DRAG_DROP_TARGET
_WIN_FLAGS = Theme.WINDOW_FLAGS_DIALOG


class BuildSettingsPanel:
    """Standalone floating Build Settings window."""

    def __init__(self):
        self._visible: bool = False
        self._first_open: bool = True
        self._scenes: List[str] = []
        self._output_dir: str = ""
        self._display_mode_idx: int = 0  # 0=fullscreen, 1=windowed
        self._window_width: int = 1280
        self._window_height: int = 720
        self._splash_items: List[Dict] = []
        self._load()

        # Build state
        self._building: bool = False
        self._build_progress: float = 0.0
        self._build_message: str = ""
        self._build_error: Optional[str] = None
        self._build_output_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self):
        self._visible = True
        self._first_open = True
        self._load()

    def close(self):
        self._visible = False

    @property
    def is_open(self) -> bool:
        return self._visible

    def get_scene_list(self) -> List[str]:
        return list(self._scenes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        data = load_build_settings()
        self._scenes = list(data.get("scenes", []))
        self._output_dir = data.get("output_dir", "")
        mode_key = data.get("display_mode", "fullscreen_borderless")
        self._display_mode_idx = (
            _DISPLAY_MODE_KEYS.index(mode_key)
            if mode_key in _DISPLAY_MODE_KEYS else 0
        )
        self._window_width = data.get("window_width", 1280)
        self._window_height = data.get("window_height", 720)
        self._splash_items = list(data.get("splash_items", []))

    def _save(self):
        save_build_settings({
            "scenes": self._scenes,
            "output_dir": self._output_dir,
            "display_mode": _DISPLAY_MODE_KEYS[self._display_mode_idx],
            "window_width": self._window_width,
            "window_height": self._window_height,
            "splash_items": self._splash_items,
        })

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, ctx):
        if not self._visible:
            return

        x0, y0, dw, dh = ctx.get_main_viewport_bounds()
        cx = x0 + (dw - 980) * 0.5
        cy = y0 + (dh - 720) * 0.5
        ctx.set_next_window_pos(cx, cy, Theme.COND_ALWAYS, 0.0, 0.0)
        ctx.set_next_window_size(980, 720, Theme.COND_ALWAYS)

        visible, still_open = ctx.begin_window_closable(
            t("menu.build_settings") + "###build_settings", self._visible, _WIN_FLAGS
        )

        if not still_open:
            self._visible = False
            from .closable_panel import ClosablePanel
            active = ClosablePanel.get_active_panel_id()
            if active:
                ClosablePanel.focus_panel_by_id(active)
            ctx.end_window()
            return

        if visible:
            self._render_body(ctx)

        ctx.end_window()

    # ------------------------------------------------------------------

    def _render_body(self, ctx):
        # Reserve ~80px at the bottom for build controls
        child_h = max(0, ctx.get_content_region_avail_height() - 80)
        if ctx.begin_child("##build_body", 0, child_h, False):
            self._render_output_section(ctx)
            ctx.separator()
            self._render_display_section(ctx)
            ctx.separator()
            self._render_splash_section(ctx)
            ctx.separator()
            self._render_scene_section(ctx)
        ctx.end_child()

        ctx.separator()
        self._render_build_controls(ctx)

    # ------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------------

    def _render_output_section(self, ctx):
        ctx.label(t("build.output_directory"))
        new_val = ctx.text_input("##output_dir", self._output_dir, 512)
        if new_val != self._output_dir:
            self._output_dir = new_val
            self._save()
        ctx.same_line()
        ctx.button(t("build.browse") + "##browse_out", self._browse_output_dir, width=70)

    def _browse_output_dir(self):
        def _do():
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                folder = filedialog.askdirectory(
                    parent=root, title="Choose Output Directory"
                )
                root.destroy()
                if folder:
                    self._output_dir = folder
                    self._save()
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    # ------------------------------------------------------------------
    # DISPLAY MODE
    # ------------------------------------------------------------------

    def _render_display_section(self, ctx):
        ctx.label(t("build.display_mode"))
        display_modes = [t(k) for k in _DISPLAY_MODES_KEYS]
        new_idx = ctx.combo("##display_mode", self._display_mode_idx, display_modes)
        if new_idx != self._display_mode_idx:
            self._display_mode_idx = new_idx
            self._save()

        if self._display_mode_idx == 1:  # Windowed
            ctx.label(t("build.window_size"))
            new_w = ctx.input_int(t("build.width") + "##win_w", self._window_width, 16, 160)
            if new_w != self._window_width:
                self._window_width = max(320, min(7680, new_w))
                self._save()
            ctx.same_line()
            new_h = ctx.input_int(t("build.height") + "##win_h", self._window_height, 16, 160)
            if new_h != self._window_height:
                self._window_height = max(240, min(4320, new_h))
                self._save()



    # ------------------------------------------------------------------
    # SPLASH ITEMS
    # ------------------------------------------------------------------

    def _render_splash_section(self, ctx):
        ctx.label(t("build.splash_sequence"))
        ctx.button(t("build.add_splash") + "##add_splash", self._browse_splash_file, width=200)

        remove_idx: Optional[int] = None

        for i, item in enumerate(self._splash_items):
            ctx.push_id(i + 10000)

            fname = os.path.basename(item.get("path", "<none>"))
            item_type = item.get("type", "image")
            badge = "[IMG]" if item_type == "image" else "[VID]"

            ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.BUILD_SETTINGS_ROW_SPC)

            # Row label
            ctx.label(f"  {i}  {badge}  {fname}")

            # Inline controls
            if item_type == "image":
                # Duration
                ctx.same_line(280)
                new_dur = ctx.input_float(t("build.duration") + f"##dur{i}", item.get("duration", 3.0), 0.1, 1.0)
                if new_dur != item.get("duration", 3.0):
                    item["duration"] = max(0.1, new_dur)
                    self._save()

            # Fade in/out
            ctx.same_line(420)
            new_fi = ctx.input_float(t("build.fade_in") + f"##fi{i}", item.get("fade_in", 0.5), 0.1, 0.5)
            if new_fi != item.get("fade_in", 0.5):
                item["fade_in"] = max(0.0, new_fi)
                self._save()

            ctx.same_line(490)
            new_fo = ctx.input_float(t("build.fade_out") + f"##fo{i}", item.get("fade_out", 0.5), 0.1, 0.5)
            if new_fo != item.get("fade_out", 0.5):
                item["fade_out"] = max(0.0, new_fo)
                self._save()

            # Up / Down / Remove buttons
            ctx.same_line(560)
            if i > 0:
                def _up(idx=i):
                    self._splash_items[idx - 1], self._splash_items[idx] = (
                        self._splash_items[idx], self._splash_items[idx - 1]
                    )
                    self._save()
                ctx.button(f"Up##{i}", _up)
                ctx.same_line()

            if i < len(self._splash_items) - 1:
                def _down(idx=i):
                    self._splash_items[idx], self._splash_items[idx + 1] = (
                        self._splash_items[idx + 1], self._splash_items[idx]
                    )
                    self._save()
                ctx.button(f"Dn##{i}", _down)
                ctx.same_line()

            def _rm(idx=i):
                nonlocal remove_idx
                remove_idx = idx
            ctx.button(f"X##{i}", _rm)

            ctx.pop_style_var(1)
            ctx.pop_id()

        if remove_idx is not None:
            del self._splash_items[remove_idx]
            self._save()

        if not self._splash_items:
            ctx.label("  " + t("build.no_splash_items"))

    def _browse_splash_file(self):
        def _do():
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                path = filedialog.askopenfilename(
                    parent=root,
                    title="Add Splash Item",
                    filetypes=[
                        ("Images", "*.png *.jpg *.jpeg *.bmp"),
                        ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                        ("All Files", "*.*"),
                    ],
                )
                root.destroy()
                if path:
                    ext = os.path.splitext(path)[1].lower()
                    itype = "video" if ext in _VIDEO_EXTS else "image"
                    self._splash_items.append({
                        "type": itype,
                        "path": os.path.abspath(path),
                        "duration": 3.0 if itype == "image" else 0.0,
                        "fade_in": 0.5,
                        "fade_out": 0.5,
                    })
                    self._save()
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    # ------------------------------------------------------------------
    # SCENE LIST
    # ------------------------------------------------------------------

    def _render_scene_section(self, ctx):
        ctx.label(t("build.scenes_in_build"))

        def _add_current():
            from InfEngine.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm and sfm.current_scene_path:
                self._add_scene(sfm.current_scene_path)

        ctx.button("  " + t("build.add_open_scene") + "  ", _add_current)

        remove_idx: Optional[int] = None
        swap_pair: Optional[tuple] = None

        for i, scene_path in enumerate(self._scenes):
            ctx.push_id(i)

            name = os.path.splitext(os.path.basename(scene_path))[0]
            root = get_project_root() or ""
            rel = os.path.relpath(scene_path, root)

            ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.BUILD_SETTINGS_ROW_SPC)
            ctx.selectable(f"  {i}    {name}    ({rel})##row", False, 16, 0, 0)

            # Drag source — reorder
            if ctx.begin_drag_drop_source(0):
                ctx.set_drag_drop_payload(DRAG_DROP_REORDER, i)
                ctx.label(f"{i}: {name}")
                ctx.end_drag_drop_source()

            # Drop target
            from .igui import IGUI
            def _on_drop(dtype, payload, _i=i):
                nonlocal swap_pair
                if dtype == DRAG_DROP_REORDER:
                    swap_pair = (int(payload), _i)
                elif dtype == DRAG_DROP_SCENE:
                    self._add_scene(str(payload))
            IGUI.multi_drop_target(ctx, (DRAG_DROP_REORDER, DRAG_DROP_SCENE), _on_drop)

            btn_area = 160 if i > 0 and i < len(self._scenes) - 1 else 110
            ctx.same_line(max(ctx.get_window_width() - btn_area, 200))
            if i > 0:
                def _up(idx=i):
                    self._scenes[idx - 1], self._scenes[idx] = self._scenes[idx], self._scenes[idx - 1]
                    self._save()
                ctx.button(f"Up##{i}", _up)
                ctx.same_line()

            if i < len(self._scenes) - 1:
                def _down(idx=i):
                    self._scenes[idx], self._scenes[idx + 1] = self._scenes[idx + 1], self._scenes[idx]
                    self._save()
                ctx.button(f"Down##{i}", _down)
                ctx.same_line()

            def _rm(idx=i):
                nonlocal remove_idx
                remove_idx = idx
            ctx.button(f"Remove##{i}", _rm)

            ctx.pop_style_var(1)
            ctx.pop_id()

        if remove_idx is not None:
            del self._scenes[remove_idx]
            self._save()
        if swap_pair is not None:
            src, dst = swap_pair
            if 0 <= src < len(self._scenes) and 0 <= dst < len(self._scenes) and src != dst:
                item = self._scenes.pop(src)
                self._scenes.insert(dst, item)
                self._save()

        # Drop target for the entire scene section
        from .igui import IGUI
        IGUI.drop_target(ctx, DRAG_DROP_SCENE, lambda p: self._add_scene(str(p)))

        if not self._scenes:
            ctx.label("")
            ctx.label("  " + t("build.list_empty"))
            ctx.label("  " + t("build.drag_scenes_hint"))

    # ------------------------------------------------------------------
    # Build controls
    # ------------------------------------------------------------------

    def _render_build_controls(self, ctx):
        if self._building:
            ctx.label(self._build_message or t("build.building"))
            ctx.progress_bar(self._build_progress, -1.0, 20.0, "")
        elif self._build_error:
            ctx.push_style_color(ImGuiCol.Text, 1.0, 0.3, 0.3, 1.0)
            ctx.label(t("build.failed").format(err=self._build_error))
            ctx.pop_style_color(1)
            ctx.same_line()
            ctx.button("OK##dismiss_err", self._dismiss_build_error)
        elif self._build_output_dir:
            ctx.push_style_color(ImGuiCol.Text, 0.3, 1.0, 0.3, 1.0)
            ctx.label(t("build.succeeded").format(path=os.path.basename(self._build_output_dir) + "/"))
            ctx.pop_style_color(1)
            ctx.same_line()

            def _open_folder():
                import subprocess as _sp
                import sys as _sys
                if _sys.platform == "win32":
                    os.startfile(self._build_output_dir)
                elif _sys.platform == "darwin":
                    _sp.Popen(["open", self._build_output_dir])
                else:
                    _sp.Popen(["xdg-open", self._build_output_dir])

            ctx.button(t("build.open_folder"), _open_folder)
            ctx.same_line()
            ctx.button("OK##dismiss_ok", self._dismiss_build_result)
        else:
            can_build = len(self._scenes) > 0 and bool(self._output_dir)

            if not can_build:
                ctx.push_style_color(ImGuiCol.Button, 0.3, 0.3, 0.3, 1.0)
                ctx.push_style_color(ImGuiCol.ButtonHovered, 0.3, 0.3, 0.3, 1.0)
                ctx.push_style_color(ImGuiCol.ButtonActive, 0.3, 0.3, 0.3, 1.0)

            ctx.button("  " + t("build.build") + "  ",
                        self._start_build if can_build else lambda: None,
                        width=120, height=30)
            ctx.same_line(0, 12)
            ctx.button("  " + t("build.build_and_run") + "  ",
                        self._start_build_and_run if can_build else lambda: None,
                        width=200, height=30)

            if not can_build:
                ctx.pop_style_color(3)

    def _dismiss_build_error(self):
        self._build_error = None

    def _dismiss_build_result(self):
        self._build_output_dir = None

    # ------------------------------------------------------------------
    # Build execution
    # ------------------------------------------------------------------

    def _make_builder(self):
        from InfEngine.engine.game_builder import GameBuilder
        project_root = get_project_root()
        return GameBuilder(
            project_root,
            self._output_dir,
            display_mode=_DISPLAY_MODE_KEYS[self._display_mode_idx],
            window_width=self._window_width,
            window_height=self._window_height,
            splash_items=self._splash_items,
        )

    def _start_build(self):
        if self._building:
            return
        self._building = True
        self._build_progress = 0.0
        self._build_message = "Starting build..."
        self._build_error = None
        self._build_output_dir = None

        if not get_project_root():
            self._building = False
            self._build_error = "No project root found"
            return

        def _run():
            try:
                builder = self._make_builder()
                result = builder.build(on_progress=self._on_build_progress)
                self._build_output_dir = result
            except Exception as exc:
                self._build_error = str(exc)
            finally:
                self._building = False

        threading.Thread(target=_run, daemon=True).start()

    def _start_build_and_run(self):
        if self._building:
            return
        self._building = True
        self._build_progress = 0.0
        self._build_message = "Starting build..."
        self._build_error = None
        self._build_output_dir = None

        if not get_project_root():
            self._building = False
            self._build_error = "No project root found"
            return

        def _run():
            try:
                import subprocess
                builder = self._make_builder()
                result = builder.build(on_progress=self._on_build_progress)
                self._build_output_dir = result

                launcher = os.path.join(result, f"{builder.project_name}.exe")
                if os.path.isfile(launcher):
                    subprocess.Popen([launcher], cwd=result)
            except Exception as exc:
                self._build_error = str(exc)
            finally:
                self._building = False

        threading.Thread(target=_run, daemon=True).start()

    def _on_build_progress(self, message: str, fraction: float):
        self._build_message = message
        self._build_progress = fraction

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_scene(self, path: str):
        abs_path = os.path.abspath(path)
        if not abs_path.lower().endswith(".scene"):
            return
        for existing in self._scenes:
            if os.path.normcase(os.path.abspath(existing)) == os.path.normcase(abs_path):
                return
        self._scenes.append(abs_path)
        self._save()
        Debug.log_internal(f"Added scene to build list: {os.path.basename(path)}")
