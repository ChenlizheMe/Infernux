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
import sys
import threading
import ctypes
from typing import Dict, List, Optional


class _BuildCancelled(Exception):
    """Raised when the user cancels the build."""
    pass

from Infernux.debug import Debug
from Infernux.engine.project_context import get_project_root
from Infernux.engine.game_builder import (
    BuildOutputDirectoryError,
    GameBuilder,
    _BuildCancelled as _GameBuilderCancelled,
)
from Infernux.engine.nuitka_builder import _BuildCancelled as _NuitkaCancelled
from Infernux.engine.i18n import t
from .theme import Theme, ImGuiCol, ImGuiStyleVar


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

BUILD_SETTINGS_FILE = "BuildSettings.json"

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}
_ICON_EXTS = {".png", ".jpg", ".jpeg", ".ico"}

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


def _win32_pick_folder(title: str) -> Optional[str]:
    import ctypes
    import ctypes.wintypes as wt

    COINIT_APARTMENTTHREADED = 0x2
    BIF_RETURNONLYFSDIRS = 0x00000001
    BIF_NEWDIALOGSTYLE = 0x00000040
    MAX_PATH = 260

    class BROWSEINFOW(ctypes.Structure):
        _fields_ = [
            ("hwndOwner", wt.HWND),
            ("pidlRoot", ctypes.c_void_p),
            ("pszDisplayName", wt.LPWSTR),
            ("lpszTitle", wt.LPCWSTR),
            ("ulFlags", wt.UINT),
            ("lpfn", ctypes.c_void_p),
            ("lParam", ctypes.c_void_p),
            ("iImage", ctypes.c_int),
        ]

    display_name = ctypes.create_unicode_buffer(MAX_PATH)
    browse = BROWSEINFOW()
    browse.pszDisplayName = ctypes.cast(display_name, wt.LPWSTR)
    browse.lpszTitle = title
    browse.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE

    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32
    user32 = ctypes.windll.user32
    shell32.SHBrowseForFolderW.argtypes = [ctypes.POINTER(BROWSEINFOW)]
    shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
    shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, wt.LPWSTR]
    shell32.SHGetPathFromIDListW.restype = wt.BOOL
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wt.DWORD]
    ole32.CoInitializeEx.restype = ctypes.HRESULT
    ole32.CoUninitialize.argtypes = []
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None

    browse.hwndOwner = user32.GetActiveWindow()

    coinit_hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    should_uninitialize = coinit_hr in (0, 1)
    try:
        pidl = shell32.SHBrowseForFolderW(ctypes.byref(browse))
        if not pidl:
            return None
        path_buf = ctypes.create_unicode_buffer(MAX_PATH)
        if shell32.SHGetPathFromIDListW(pidl, path_buf):
            return path_buf.value
        return None
    finally:
        if 'pidl' in locals() and pidl:
            ole32.CoTaskMemFree(pidl)
        if should_uninitialize:
            ole32.CoUninitialize()


def _win32_pick_file(title: str, filter_text: str) -> Optional[str]:
    import ctypes
    import ctypes.wintypes as wt

    COINIT_APARTMENTTHREADED = 0x2
    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_NOCHANGEDIR = 0x00000008
    OFN_EXPLORER = 0x00080000
    MAX_PATH = 4096

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wt.DWORD),
            ("hwndOwner", wt.HWND),
            ("hInstance", wt.HINSTANCE),
            ("lpstrFilter", wt.LPCWSTR),
            ("lpstrCustomFilter", wt.LPWSTR),
            ("nMaxCustFilter", wt.DWORD),
            ("nFilterIndex", wt.DWORD),
            ("lpstrFile", wt.LPWSTR),
            ("nMaxFile", wt.DWORD),
            ("lpstrFileTitle", wt.LPWSTR),
            ("nMaxFileTitle", wt.DWORD),
            ("lpstrInitialDir", wt.LPCWSTR),
            ("lpstrTitle", wt.LPCWSTR),
            ("Flags", wt.DWORD),
            ("nFileOffset", wt.WORD),
            ("nFileExtension", wt.WORD),
            ("lpstrDefExt", wt.LPCWSTR),
            ("lCustData", ctypes.c_void_p),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wt.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", wt.DWORD),
            ("FlagsEx", wt.DWORD),
        ]

    buf = ctypes.create_unicode_buffer(MAX_PATH)
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter = filter_text
    ofn.lpstrFile = ctypes.cast(buf, wt.LPWSTR)
    ofn.nMaxFile = MAX_PATH
    ofn.lpstrTitle = title
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER
    ofn.hwndOwner = ctypes.windll.user32.GetActiveWindow()

    comdlg32 = ctypes.windll.comdlg32
    ole32 = ctypes.windll.ole32
    comdlg32.GetOpenFileNameW.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
    comdlg32.GetOpenFileNameW.restype = wt.BOOL
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wt.DWORD]
    ole32.CoInitializeEx.restype = ctypes.HRESULT
    ole32.CoUninitialize.argtypes = []

    coinit_hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    should_uninitialize = coinit_hr in (0, 1)
    try:
        if comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return buf.value
        return None
    finally:
        if should_uninitialize:
            ole32.CoUninitialize()


def _pick_folder_dialog(title: str) -> Optional[str]:
    if sys.platform == "win32":
        try:
            return _win32_pick_folder(title)
        except Exception as exc:
            Debug.log_warning(f"Win32 folder dialog failed: {exc}")
            return None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(parent=root, title=title)
    finally:
        root.destroy()


def _pick_file_dialog(title: str) -> Optional[str]:
    if sys.platform == "win32":
        try:
            return _win32_pick_file(
                title,
                "Images (*.png;*.jpg;*.jpeg;*.bmp)\0*.png;*.jpg;*.jpeg;*.bmp\0"
                "Videos (*.mp4;*.avi;*.mov;*.mkv;*.webm)\0*.mp4;*.avi;*.mov;*.mkv;*.webm\0"
                "All Files (*.*)\0*.*\0\0",
            )
        except Exception as exc:
            Debug.log_warning(f"Win32 open-file dialog failed: {exc}")
            return None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(
            parent=root,
            title=title,
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All Files", "*.*"),
            ],
        )
    finally:
        root.destroy()


def _pick_icon_file_dialog(title: str) -> Optional[str]:
    if sys.platform == "win32":
        try:
            return _win32_pick_file(
                title,
                "Images (*.png;*.jpg;*.jpeg;*.ico)\0*.png;*.jpg;*.jpeg;*.ico\0"
                "All Files (*.*)\0*.*\0\0",
            )
        except Exception as exc:
            Debug.log_warning(f"Win32 icon picker failed: {exc}")
            return None

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(
            parent=root,
            title=title,
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.ico"),
                ("All Files", "*.*"),
            ],
        )
    finally:
        root.destroy()


def _show_system_error_dialog(title: str, message: str) -> None:
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10 | 0x0)
        return

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        messagebox.showerror(title, message, parent=root)
    finally:
        root.destroy()


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
        self._game_name: str = ""
        self._scenes: List[str] = []
        self._output_dir: str = ""
        self._icon_path: str = ""
        self._display_mode_idx: int = 0  # 0=fullscreen, 1=windowed
        self._window_width: int = 1280
        self._window_height: int = 720
        self._window_resizable: bool = True
        self._splash_items: List[Dict] = []
        self._load()

        # Build state
        self._building: bool = False
        self._build_progress: float = 0.0
        self._build_message: str = ""
        self._build_error: Optional[str] = None
        self._build_output_dir: Optional[str] = None
        self._cancel_event: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self):
        self._visible = True
        self._first_open = True
        self._load()
        self._prune_missing_splash()

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
        self._game_name = data.get("game_name", "")
        self._scenes = list(data.get("scenes", []))
        self._output_dir = data.get("output_dir", "")
        self._icon_path = data.get("icon_path", "")
        mode_key = data.get("display_mode", "fullscreen_borderless")
        self._display_mode_idx = (
            _DISPLAY_MODE_KEYS.index(mode_key)
            if mode_key in _DISPLAY_MODE_KEYS else 0
        )
        self._window_width = data.get("window_width", 1280)
        self._window_height = data.get("window_height", 720)
        self._window_resizable = data.get("window_resizable", True)
        self._debug_mode = data.get("debug_mode", False)
        self._lto = data.get("lto", True)
        self._splash_items = list(data.get("splash_items", []))

    def _prune_missing_splash(self):
        """Remove splash items whose source files no longer exist."""
        before = len(self._splash_items)
        self._splash_items = [
            it for it in self._splash_items
            if os.path.isfile(it.get("path", ""))
        ]
        if len(self._splash_items) < before:
            self._save()

    def _save(self):
        save_build_settings({
            "game_name": self._game_name,
            "scenes": self._scenes,
            "output_dir": self._output_dir,
            "icon_path": self._icon_path,
            "display_mode": _DISPLAY_MODE_KEYS[self._display_mode_idx],
            "window_width": self._window_width,
            "window_height": self._window_height,
            "window_resizable": self._window_resizable,
            "debug_mode": self._debug_mode,
            "lto": self._lto,
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
            if self._building:
                ctx.begin_disabled(True)
            self._render_body(ctx)
            if self._building:
                ctx.end_disabled()

        ctx.end_window()

    # ------------------------------------------------------------------

    def _render_body(self, ctx):
        ctx.dummy(0.0, 4.0)
        # Reserve ~80px at the bottom for build controls
        child_h = max(0, ctx.get_content_region_avail_height() - 80)
        
        ctx.push_style_color(ImGuiCol.ChildBg, 0.0, 0.0, 0.0, 0.0)
        ctx.push_style_var_float(ImGuiStyleVar.ChildBorderSize, 0.0)
        if ctx.begin_child("##build_body", 0, child_h, False):
            self._render_output_section(ctx)
            ctx.separator()
            self._render_display_section(ctx)
            ctx.separator()
            self._render_splash_section(ctx)
            ctx.separator()
            self._render_scene_section(ctx)
        ctx.end_child()
        ctx.pop_style_var(1)
        ctx.pop_style_color(1)

        ctx.separator()
        self._render_build_controls(ctx)

    # ------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------------

    def _render_output_section(self, ctx):
        ctx.label(t("build.game_name"))
        root = get_project_root()
        placeholder = os.path.basename(root) if root else "MyGame"
        ctx.set_next_item_width(300)
        new_name = ctx.text_input("##game_name", self._game_name, 256)
        if new_name != self._game_name:
            self._game_name = new_name
            self._save()
        ctx.same_line(0, 20)
        new_debug = ctx.checkbox(t("build.debug_mode") + "##debug_mode", self._debug_mode)
        if new_debug != self._debug_mode:
            self._debug_mode = new_debug
            self._save()
        ctx.same_line(0, 20)
        new_lto = ctx.checkbox(t("build.lto") + "##lto", self._lto)
        if new_lto != self._lto:
            self._lto = new_lto
            self._save()
        if not self._game_name:
            ctx.same_line()
            ctx.push_style_color(ImGuiCol.Text, 0.5, 0.5, 0.5, 1.0)
            ctx.label(t("build.game_name_hint").format(name=placeholder))
            ctx.pop_style_color(1)

        ctx.label(t("build.output_directory"))
        ctx.set_next_item_width(ctx.get_content_region_avail_width() - 84)
        new_val = ctx.text_input("##output_dir", self._output_dir, 512)
        if new_val != self._output_dir:
            self._output_dir = new_val
            self._save()
        ctx.same_line()
        ctx.button(t("build.browse") + "##browse_out", self._browse_output_dir, width=80)
        ctx.push_style_color(ImGuiCol.Text, 0.5, 0.5, 0.5, 1.0)
        ctx.label(t("build.output_directory_hint").format(marker=GameBuilder.OUTPUT_MARKER_FILENAME))
        ctx.pop_style_color(1)

        ctx.label(t("build.icon"))
        clear_btn_w = 80 if self._icon_path else 0
        icon_input_w = ctx.get_content_region_avail_width() - 84 - (clear_btn_w + (4 if clear_btn_w else 0))
        ctx.set_next_item_width(max(120, icon_input_w))
        new_icon = ctx.text_input("##build_icon", self._icon_path, 512)
        if new_icon != self._icon_path:
            self._icon_path = new_icon
            self._save()
        ctx.same_line()
        ctx.button(t("build.browse") + "##browse_icon", self._browse_icon_path, width=80)
        if self._icon_path:
            ctx.same_line(0, 4)
            ctx.button(t("build.clear_icon") + "##clear_icon", self._clear_icon_path, width=80)
        else:
            ctx.push_style_color(ImGuiCol.Text, 0.5, 0.5, 0.5, 1.0)
            ctx.label("  " + t("build.icon_hint"))
            ctx.pop_style_color(1)

    def _browse_output_dir(self):
        def _do():
            try:
                folder = _pick_folder_dialog("Choose Output Directory")
                if folder:
                    self._output_dir = folder
                    self._save()
            except Exception as exc:
                Debug.log_warning(f"Build Settings output directory browse failed: {exc}")
        threading.Thread(target=_do, daemon=True).start()

    def _browse_icon_path(self):
        def _do():
            try:
                path = _pick_icon_file_dialog("Choose Build Icon")
                if path:
                    ext = os.path.splitext(path)[1].lower()
                    if ext not in _ICON_EXTS:
                        raise ValueError("Unsupported icon format")
                    self._icon_path = os.path.abspath(path)
                    self._save()
            except Exception as exc:
                Debug.log_warning(f"Build Settings icon picker failed: {exc}")
        threading.Thread(target=_do, daemon=True).start()

    def _clear_icon_path(self):
        if not self._icon_path:
            return
        self._icon_path = ""
        self._save()

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

            new_resizable = ctx.checkbox(t("build.window_resizable") + "##resizable", self._window_resizable)
            if new_resizable != self._window_resizable:
                self._window_resizable = new_resizable
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
            ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.BUILD_SETTINGS_ROW_SPC)

            fname = os.path.basename(item.get("path", "<none>"))
            item_type = item.get("type", "image")
            badge = "[IMG]" if item_type == "image" else "[VID]"

            # ── Row 1: name ──
            ctx.label(f"  {i + 1}. {badge}  {fname}")
            if ctx.is_item_hovered():
                ctx.set_tooltip(item.get("path", ""))

            # ── Row 2: numeric fields ──
            if item_type == "image":
                ctx.label(f"      {t('build.duration')} ({t('build.seconds_short')})")
                ctx.same_line(0, 8)
                ctx.set_next_item_width(120)
                new_dur = ctx.input_float(f"##dur{i}", item.get("duration", 3.0), 0.1, 1.0)
                if new_dur != item.get("duration", 3.0):
                    item["duration"] = max(0.1, new_dur)
                    self._save()
                ctx.same_line(0, 24)
            else:
                ctx.label("      ")
                ctx.same_line(0, 0)

            ctx.label(f"{t('build.fade_in')} ({t('build.seconds_short')})")
            ctx.same_line(0, 8)
            ctx.set_next_item_width(120)
            new_fi = ctx.input_float(f"##fi{i}", item.get("fade_in", 0.5), 0.1, 0.5)
            if new_fi != item.get("fade_in", 0.5):
                item["fade_in"] = max(0.0, new_fi)
                self._save()

            ctx.same_line(0, 24)
            ctx.label(f"{t('build.fade_out')} ({t('build.seconds_short')})")
            ctx.same_line(0, 8)
            ctx.set_next_item_width(120)
            new_fo = ctx.input_float(f"##fo{i}", item.get("fade_out", 0.5), 0.1, 0.5)
            if new_fo != item.get("fade_out", 0.5):
                item["fade_out"] = max(0.0, new_fo)
                self._save()

            # ── Row 3: action buttons ──
            ctx.label(" ")
            
            btn_w = 64
            btn_spc = 4
            num_btns = 1 + int(i > 0) + int(i < len(self._splash_items) - 1)
            btn_area = num_btns * btn_w + (num_btns - 1) * btn_spc + 24
            
            ctx.same_line(max(ctx.get_window_width() - btn_area, 200))
            
            if i > 0:
                def _up(idx=i):
                    self._splash_items[idx - 1], self._splash_items[idx] = (
                        self._splash_items[idx], self._splash_items[idx - 1]
                    )
                    self._save()
                ctx.button(t("build.move_up") + f"##sp_{i}", _up, width=btn_w)
                ctx.same_line(0, btn_spc)

            if i < len(self._splash_items) - 1:
                def _down(idx=i):
                    self._splash_items[idx], self._splash_items[idx + 1] = (
                        self._splash_items[idx + 1], self._splash_items[idx]
                    )
                    self._save()
                ctx.button(t("build.move_down") + f"##sp_{i}", _down, width=btn_w)
                ctx.same_line(0, btn_spc)

            def _rm(idx=i):
                nonlocal remove_idx
                remove_idx = idx
            ctx.button(t("build.remove") + f"##sp_{i}", _rm, width=btn_w)

            ctx.separator()

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
                path = _pick_file_dialog("Add Splash Item")
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
            except Exception as exc:
                Debug.log_warning(f"Build Settings splash picker failed: {exc}")
        threading.Thread(target=_do, daemon=True).start()

    # ------------------------------------------------------------------
    # SCENE LIST
    # ------------------------------------------------------------------

    def _render_scene_section(self, ctx):
        ctx.label(t("build.scenes_in_build"))

        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        can_add_open_scene = bool(sfm and sfm.current_scene_path)

        def _add_current():
            if sfm and sfm.current_scene_path:
                self._add_scene(sfm.current_scene_path)

        if not can_add_open_scene:
            ctx.begin_disabled(True)
        ctx.button("  " + t("build.add_open_scene") + "  ", _add_current)
        if not can_add_open_scene:
            ctx.end_disabled()
            if ctx.is_item_hovered():
                ctx.set_tooltip("Please save the current scene before adding it to Build Settings.")

        remove_idx: Optional[int] = None
        swap_pair: Optional[tuple] = None

        for i, scene_path in enumerate(self._scenes):
            ctx.push_id(i)

            name = os.path.splitext(os.path.basename(scene_path))[0]
            root = get_project_root() or ""
            rel = os.path.relpath(scene_path, root)

            ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.BUILD_SETTINGS_ROW_SPC)
            
            # Use a fixed row height so selectable and buttons align
            row_h = 24
            ctx.selectable(f"  {i}    {name}    ({rel})##row", False, 16, 0, row_h)

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

            btn_w = 64
            btn_spc = 4
            num_btns = 1 + int(i > 0) + int(i < len(self._scenes) - 1)
            btn_area = num_btns * btn_w + (num_btns - 1) * btn_spc + 24
            
            ctx.same_line(max(ctx.get_window_width() - btn_area, 200))
            if i > 0:
                def _up(idx=i):
                    self._scenes[idx - 1], self._scenes[idx] = self._scenes[idx], self._scenes[idx - 1]
                    self._save()
                ctx.button(t("build.move_up") + f"##{i}", _up, width=btn_w, height=row_h)
                ctx.same_line(0, btn_spc)

            if i < len(self._scenes) - 1:
                def _down(idx=i):
                    self._scenes[idx], self._scenes[idx + 1] = self._scenes[idx + 1], self._scenes[idx]
                    self._save()
                ctx.button(t("build.move_down") + f"##{i}", _down, width=btn_w, height=row_h)
                ctx.same_line(0, btn_spc)

            def _rm(idx=i):
                nonlocal remove_idx
                remove_idx = idx
            ctx.button(t("build.remove") + f"##{i}", _rm, width=btn_w, height=row_h)

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
        # Build controls zone is always interactive (not affected by
        # the disabled wrapper around the settings body).
        if self._building:
            ctx.end_disabled()
            ctx.label(self._build_message or t("build.building"))
            ctx.progress_bar(self._build_progress, -1.0, 20.0, "")
            ctx.button("  " + t("build.cancel") + "  ##cancel_build",
                       self._cancel_build, width=120, height=30)
            ctx.begin_disabled(True)
        elif self._build_error:
            ctx.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
            ctx.label(t("build.failed").format(err=self._build_error))
            ctx.pop_style_color(1)
            ctx.same_line()
            ctx.button("OK##dismiss_err", self._dismiss_build_error)
        elif self._build_output_dir:
            ctx.push_style_color(ImGuiCol.Text, *Theme.SUCCESS_TEXT)
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
                ctx.push_style_color(ImGuiCol.Button, *Theme.BTN_DISABLED)
                ctx.push_style_color(ImGuiCol.ButtonHovered, *Theme.BTN_DISABLED)
                ctx.push_style_color(ImGuiCol.ButtonActive, *Theme.BTN_DISABLED)

            # Align build buttons to the right
            ctx.same_line(max(ctx.get_window_width() - 360, 200))

            ctx.button("  " + t("build.build") + "  ",
                        self._start_build if can_build else lambda: None,
                        width=140, height=36)
            ctx.same_line(0, 16)
            ctx.button("  " + t("build.build_and_run") + "  ",
                        self._start_build_and_run if can_build else lambda: None,
                        width=160, height=36)

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
        from Infernux.engine.game_builder import GameBuilder
        project_root = get_project_root()
        game_name = self._game_name.strip() or os.path.basename(project_root)
        return GameBuilder(
            project_root,
            self._output_dir,
            game_name=game_name,
            icon_path=self._icon_path.strip() or None,
            display_mode=_DISPLAY_MODE_KEYS[self._display_mode_idx],
            window_width=self._window_width,
            window_height=self._window_height,
            window_resizable=self._window_resizable,
            splash_items=self._splash_items,
            debug_mode=self._debug_mode,
            lto=self._lto,
        )

    def _cancel_build(self):
        self._cancel_event.set()

    def _format_output_directory_error(self, exc: BuildOutputDirectoryError) -> str:
        if exc.reason == "required":
            return t("build.output_directory_error_required")
        if exc.reason == "path-is-file":
            return t("build.output_directory_error_path_is_file").format(path=exc.path)
        if exc.reason == "path-not-directory":
            return t("build.output_directory_error_not_directory").format(path=exc.path)

        found_line = ""
        if exc.entries:
            found_line = "\n\n" + t("build.output_directory_error_found").format(
                entries=", ".join(exc.entries[:5]) + (", ..." if len(exc.entries) > 5 else "")
            )

        return t("build.output_directory_error_not_empty").format(
            path=exc.path,
            marker=exc.marker_filename,
        ) + found_line

    def _show_output_directory_error(self, exc: BuildOutputDirectoryError) -> None:
        message = self._format_output_directory_error(exc)
        self._build_error = message
        _show_system_error_dialog(t("build.output_directory_error_title"), message)

    def _on_build_progress(self, message: str, fraction: float):
        self._build_message = message
        self._build_progress = fraction
        if self._cancel_event.is_set():
            raise _BuildCancelled()

    def _start_build(self):
        self._do_build(run_after=False)

    def _start_build_and_run(self):
        self._do_build(run_after=True)

    def _do_build(self, *, run_after: bool):
        if self._building:
            return
        self._building = True
        self._build_progress = 0.0
        self._build_message = "Starting build..."
        self._build_error = None
        self._build_output_dir = None
        self._cancel_event.clear()

        if not get_project_root():
            self._building = False
            self._build_error = "No project root found"
            return

        try:
            builder = self._make_builder()
            builder._validate_output_directory()
        except BuildOutputDirectoryError as exc:
            self._building = False
            self._show_output_directory_error(exc)
            return

        def _run():
            try:
                result = builder.build(
                    on_progress=self._on_build_progress,
                    cancel_event=self._cancel_event,
                )
                self._build_output_dir = result

                if run_after:
                    import subprocess
                    exe_name = f"{builder.project_name}.exe"
                    launcher = os.path.join(result, exe_name)
                    if os.path.isfile(launcher):
                        subprocess.Popen([launcher], cwd=result)
            except (_BuildCancelled, _GameBuilderCancelled, _NuitkaCancelled):
                self._build_error = t("build.cancelled")
            except BuildOutputDirectoryError as exc:
                self._show_output_directory_error(exc)
            except Exception as exc:
                log_path = os.path.join(builder.project_path, "Logs", "build.log")
                if os.path.isfile(log_path):
                    self._build_error = f"{exc}\n\nSee: {log_path}"
                else:
                    self._build_error = str(exc)
            finally:
                self._building = False

        threading.Thread(target=_run, daemon=True).start()

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
