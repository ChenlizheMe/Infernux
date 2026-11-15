"""
Infernux Editor Theme
==============================================

Python helpers mirror the native EditorThemeRegistry. Theme plugins supply
color and layout overrides; EditorThemeTable.inl owns the builtin defaults.

All colors are **sRGB-space RGBA tuples** (float, 0-1).

Usage::

    from infernux.engine.ui.theme import Theme, ImGuiCol

    Theme.push_ghost_button_style(ctx)
    ctx.button("Click me", on_click)
    ctx.pop_style_color(3)
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple

from infernux.lib import ImGuiStyleVar

# Color type alias (R, G, B, A) all float [0..1]
RGBA = Tuple[float, float, float, float]


def srgb_to_linear(s: float) -> float:
    """
    Convert a single sRGB [0,1] component to linear space."""
    if s <= 0.04045:
        return s / 12.92
    return ((s + 0.055) / 1.055) ** 2.4

def srgb3(r: float, g: float, b: float, a: float = 1.0) -> RGBA:
    """
    Convert sRGB 0-1 components to linear RGBA tuple."""
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a)

def hex_to_linear(hex_r: int, hex_g: int, hex_b: int, a: float = 1.0) -> RGBA:
    """
    Convert 0-255 sRGB hex components to linear RGBA tuple."""
    return srgb3(hex_r / 255.0, hex_g / 255.0, hex_b / 255.0, a)


class ImGuiCol:
    Text                       = 0
    TextDisabled               = 1
    WindowBg                   = 2
    ChildBg                    = 3
    PopupBg                    = 4
    Border                     = 5
    BorderShadow               = 6
    FrameBg                    = 7
    FrameBgHovered             = 8
    FrameBgActive              = 9
    TitleBg                    = 10
    TitleBgActive              = 11
    TitleBgCollapsed           = 12
    MenuBarBg                  = 13
    ScrollbarBg                = 14
    ScrollbarGrab              = 15
    ScrollbarGrabHovered       = 16
    ScrollbarGrabActive        = 17
    CheckMark                  = 18
    SliderGrab                 = 19
    SliderGrabActive           = 20
    Button                     = 21
    ButtonHovered              = 22
    ButtonActive               = 23
    Header                     = 24
    HeaderHovered              = 25
    HeaderActive               = 26
    Separator                  = 27
    SeparatorHovered           = 28
    SeparatorActive            = 29
    ResizeGrip                 = 30
    ResizeGripHovered          = 31
    ResizeGripActive           = 32
    InputTextCursor            = 33
    TabHovered                 = 34
    Tab                        = 35
    TabSelected                = 36
    TabSelectedOverline        = 37
    TabDimmed                  = 38
    TabDimmedSelected          = 39
    TabDimmedSelectedOverline  = 40
    DockingPreview             = 41
    DockingEmptyBg             = 42
    PlotLines                  = 43
    PlotLinesHovered           = 44
    PlotHistogram              = 45
    PlotHistogramHovered       = 46
    TableHeaderBg              = 47
    TableBorderStrong          = 48
    TableBorderLight           = 49
    TableRowBg                 = 50
    TableRowBgAlt              = 51
    TextLink                   = 52
    TextSelectedBg             = 53
    TreeLines                  = 54
    DragDropTarget             = 55
    DragDropTargetBg           = 56
    UnsavedMarker              = 57
    NavCursor                  = 58
    NavWindowingHighlight      = 59
    NavWindowingDimBg          = 60
    ModalWindowDimBg           = 61

class ImGuiWindowFlags:
    NoTitleBar                  = 1 << 0
    NoResize                    = 1 << 1
    NoMove                      = 1 << 2
    NoScrollbar                 = 1 << 3
    NoScrollWithMouse           = 1 << 4
    NoCollapse                  = 1 << 5
    AlwaysAutoResize            = 1 << 6
    NoBackground                = 1 << 7
    NoSavedSettings             = 1 << 8
    NoMouseInputs               = 1 << 9
    NoFocusOnAppearing          = 1 << 12
    NoBringToFrontOnFocus       = 1 << 13
    NoNavInputs                 = 1 << 16
    NoNavFocus                  = 1 << 17
    UnsavedDocument             = 1 << 18
    NoDocking                   = 1 << 19
    NoNav                       = (1 << 16) | (1 << 17)
    NoDecoration                = NoTitleBar | NoResize | NoScrollbar | NoCollapse
    NoInputs                    = NoMouseInputs | NoNavInputs | NoNavFocus

class ImGuiTreeNodeFlags:
    Selected                    = 1 << 0
    Framed                      = 1 << 1
    AllowOverlap                = 1 << 2
    NoTreePushOnOpen            = 1 << 3
    NoAutoOpenOnLog             = 1 << 4
    DefaultOpen                 = 1 << 5
    OpenOnDoubleClick           = 1 << 6
    OpenOnArrow                 = 1 << 7
    Leaf                        = 1 << 8
    Bullet                      = 1 << 9
    FramePadding                = 1 << 10
    SpanAvailWidth              = 1 << 11
    SpanFullWidth               = 1 << 12
    SpanAllColumns              = 1 << 13
    CollapsingHeader            = Framed | NoTreePushOnOpen | NoAutoOpenOnLog

class ImGuiMouseCursor:
    Arrow      = 0
    TextInput  = 1
    ResizeAll  = 2
    ResizeNS   = 3
    ResizeEW   = 4
    ResizeNESW = 5
    ResizeNWSE = 6
    Hand       = 7


class Theme:
    """
    Python view of native theme tokens plus editor drawing helpers.

    Native tokens are refreshed when the active theme changes.
    """

    PREFAB_ICON       : str  = "\u25C6"  # Prefab icon (diamond)

    UI_EDITOR_NEW_TEXT_POS    = (-80.0, -20.0)
    UI_EDITOR_NEW_IMAGE_SIZE  = (100.0, 100.0)
    UI_EDITOR_NEW_IMAGE_POS   = (-50.0, -50.0)
    UI_EDITOR_NEW_BUTTON_SIZE = (160.0, 40.0)
    UI_EDITOR_NEW_BUTTON_POS  = (-80.0, -20.0)

    UI_EDITOR_SNAP_TABLE = (
        (1.0,  1),
        (0.75, 2),
        (0.5,  5),
        (0.35, 10),
        (0.2,  20),
        (0.1,  50),
    )

    WINDOW_FLAGS_VIEWPORT  = (ImGuiWindowFlags.NoFocusOnAppearing
                              | ImGuiWindowFlags.NoBringToFrontOnFocus)
    WINDOW_FLAGS_NO_SCROLL = (ImGuiWindowFlags.NoScrollbar
                              | ImGuiWindowFlags.NoScrollWithMouse)
    WINDOW_FLAGS_NO_DECOR  = (ImGuiWindowFlags.NoTitleBar
                              | ImGuiWindowFlags.NoResize
                              | ImGuiWindowFlags.NoMove
                              | ImGuiWindowFlags.NoScrollbar
                              | ImGuiWindowFlags.NoScrollWithMouse
                              | ImGuiWindowFlags.NoSavedSettings
                              | ImGuiWindowFlags.NoFocusOnAppearing
                              | ImGuiWindowFlags.NoDocking
                              | ImGuiWindowFlags.NoInputs)
    WINDOW_FLAGS_FLOATING  = (ImGuiWindowFlags.NoCollapse
                              | ImGuiWindowFlags.NoSavedSettings)
    WINDOW_FLAGS_DIALOG    = (ImGuiWindowFlags.NoCollapse
                              | ImGuiWindowFlags.NoSavedSettings
                              | ImGuiWindowFlags.NoDocking
                              | ImGuiWindowFlags.NoResize
                              | ImGuiWindowFlags.NoMove)

    COND_FIRST_USE_EVER = 4  # Set only on first use
    COND_ALWAYS         = 1  # Set every frame

    BORDER_SIZE_NONE    = 0.0  # No border

    ICON_PLUS          : str = "+"
    ICON_MINUS         : str = "-"
    ICON_REMOVE        : str = "\u00d7"  # Multiplication sign
    ICON_PICKER        : str = "\u2299"  # Circled dot
    ICON_WARNING       : str = "\u25b2"  # Triangle
    ICON_ERROR         : str = "\u25cf"  # Filled circle
    ICON_DOT           : str = "\u00b7"  # Middle dot
    ICON_CHECK         : str = "v"  # Check mark

    ICON_IMG_PLUS      : str = "plus"
    ICON_IMG_MINUS     : str = "minus"
    ICON_IMG_REMOVE    : str = "remove"
    ICON_IMG_PICKER    : str = "picker"
    ICON_IMG_WARNING   : str = "warning"
    ICON_IMG_ERROR     : str = "error"
    ICON_IMG_UI_CANVAS : str = "ui_canvas"
    ICON_IMG_UI_TEXT   : str = "ui_text"
    ICON_IMG_UI_IMAGE  : str = "ui_image"
    ICON_IMG_UI_BUTTON : str = "ui_button"

    @staticmethod
    def push_ghost_button_style(ctx) -> int:
        """
        Push transparent button colors. Returns color count pushed (3)."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_GHOST_HOVERED)
        ctx.push_style_color(ImGuiCol.ButtonActive,   *Theme.BTN_GHOST_ACTIVE)
        return 3

    @staticmethod
    def push_flat_button_style(ctx, r: float, g: float, b: float, a: float = 1.0) -> int:
        """
        Push flat solid-color button style. Returns 3."""
        ctx.push_style_color(ImGuiCol.Button,        r, g, b, a)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  min(r + 0.06, 1), min(g + 0.06, 1), min(b + 0.06, 1), a)
        ctx.push_style_color(ImGuiCol.ButtonActive,   min(r + 0.12, 1), min(g + 0.12, 1), min(b + 0.12, 1), a)
        return 3

    @staticmethod
    def push_toolbar_vars(ctx) -> int:
        """
        Push compact toolbar spacing preset. Returns var count (5)."""
        from .dpi import scaled_editor_metric
        for key, value in ((ImGuiStyleVar.WindowPadding, Theme.TOOLBAR_WIN_PAD),
                           (ImGuiStyleVar.FramePadding, Theme.TOOLBAR_FRAME_PAD),
                           (ImGuiStyleVar.ItemSpacing, Theme.TOOLBAR_ITEM_SPC)):
            ctx.push_style_var_vec2(key, *(scaled_editor_metric(ctx, item) for item in value))
        ctx.push_style_var_float(ImGuiStyleVar.FrameRounding, scaled_editor_metric(ctx, Theme.TOOLBAR_FRAME_RND))
        ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, scaled_editor_metric(ctx, Theme.TOOLBAR_FRAME_BRD))
        return 5

    @staticmethod
    def push_popup_vars(ctx) -> int:
        """
        Push popup spacing preset. Returns 3."""
        from .dpi import scaled_editor_metric
        for key, value in ((ImGuiStyleVar.WindowPadding, Theme.POPUP_WIN_PAD),
                           (ImGuiStyleVar.ItemSpacing, Theme.POPUP_ITEM_SPC),
                           (ImGuiStyleVar.FramePadding, Theme.POPUP_FRAME_PAD)):
            ctx.push_style_var_vec2(key, *(scaled_editor_metric(ctx, item) for item in value))
        return 3

    @staticmethod
    def push_status_bar_button_style(ctx) -> int:
        """
        Push status bar button style. Returns 3."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_SB_HOVERED)
        ctx.push_style_color(ImGuiCol.ButtonActive,   *Theme.BTN_SB_ACTIVE)
        return 3

    @staticmethod
    def push_transparent_border(ctx) -> int:
        """
        Push transparent border color. Returns 1."""
        ctx.push_style_color(ImGuiCol.Border, *Theme.BORDER_TRANSPARENT)
        return 1

    @staticmethod
    def push_drag_drop_target_style(ctx) -> int:
        """
        Push drag-drop target highlight color. Returns 1."""
        ctx.push_style_color(ImGuiCol.DragDropTarget, *Theme.DRAG_DROP_TARGET)
        return 1

    @staticmethod
    def push_console_toolbar_vars(ctx) -> int:
        """
        Push console toolbar compact spacing. Returns 3."""
        from .dpi import scaled_editor_metric
        for key, value in ((ImGuiStyleVar.FramePadding, Theme.CONSOLE_FRAME_PAD),
                           (ImGuiStyleVar.ItemSpacing, Theme.CONSOLE_ITEM_SPC)):
            ctx.push_style_var_vec2(key, *(scaled_editor_metric(ctx, item) for item in value))
        ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, scaled_editor_metric(ctx, Theme.TOOLBAR_FRAME_BRD))
        return 3

    @staticmethod
    def push_splitter_style(ctx) -> int:
        """
        Push splitter button style. Returns 3."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.SPLITTER_HOVER)
        ctx.push_style_color(ImGuiCol.ButtonActive,   *Theme.SPLITTER_ACTIVE)
        return 3

    @staticmethod
    def push_selected_icon_style(ctx) -> int:
        """
        Push selected icon button highlight. Returns 2."""
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_SELECTED)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_SELECTED)
        return 2

    @staticmethod
    def push_unselected_icon_style(ctx) -> int:
        """
        Push unselected icon button style. Returns 2 colors (caller also pops 1 var)."""
        ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, 0.0)
        ctx.push_style_color(ImGuiCol.Button,        *Theme.BTN_GHOST)
        ctx.push_style_color(ImGuiCol.ButtonHovered,  *Theme.BTN_SUBTLE_HOVER)
        return 2

    @staticmethod
    def get_play_border_color(is_paused: bool) -> RGBA:
        """
        Return the play-mode border color."""
        return Theme.BORDER_PAUSE if is_paused else Theme.BORDER_PLAY

    @staticmethod
    def push_inline_button_style(ctx, active: bool = False) -> int:
        """
        Push inline button style. Returns color count (3)."""
        if active:
            ctx.push_style_color(ImGuiCol.Button, *Theme.INSPECTOR_INLINE_BTN_ON)
            ctx.push_style_color(ImGuiCol.ButtonHovered, *Theme.INSPECTOR_INLINE_BTN_ON)
            ctx.push_style_color(ImGuiCol.ButtonActive, *Theme.INSPECTOR_INLINE_BTN_ACTIVE)
        else:
            ctx.push_style_color(ImGuiCol.Button, *Theme.INSPECTOR_INLINE_BTN_IDLE)
            ctx.push_style_color(ImGuiCol.ButtonHovered, *Theme.INSPECTOR_INLINE_BTN_HOVER)
            ctx.push_style_color(ImGuiCol.ButtonActive, *Theme.INSPECTOR_INLINE_BTN_ACTIVE)
        return 3

    @staticmethod
    def render_inline_button_row(
        ctx,
        row_id: str,
        items: Iterable[tuple[str, str]],
        *,
        active_items: Optional[Iterable[str]] = None,
        height: float = 0.0,
        semantic_base: str = "",
    ):
        """
        Render a row of evenly sized buttons. Returns clicked item id."""
        entries = list(items)
        if not entries:
            return None

        active_set = set(active_items or [])
        spacing = Theme.INSPECTOR_INLINE_BTN_GAP
        button_h = height if height > 0.0 else Theme.INSPECTOR_INLINE_BTN_H
        avail_w = max(0.0, ctx.get_content_region_avail_width())
        total_gap = spacing * max(0, len(entries) - 1)
        button_w = max(1.0, (avail_w - total_gap) / max(1, len(entries)))

        clicked = [None]
        for idx, (item_id, label) in enumerate(entries):
            color_count = Theme.push_inline_button_style(ctx, item_id in active_set)

            def _on_click(iid=item_id):
                clicked[0] = iid

            ctx.button(f"{label}##{row_id}_{item_id}", _on_click, width=button_w, height=button_h)
            recorder = getattr(ctx, "record_semantic_item", None)
            if semantic_base and callable(recorder):
                recorder("inline_button", label, True, f"{semantic_base}.{item_id}")
            ctx.pop_style_color(color_count)
            if idx + 1 < len(entries):
                ctx.same_line(0, spacing)
        return clicked[0]


_theme_listeners = {}

def on_theme_changed(callback):
    from infernux.engine.runtime_dispatch import ReloadableCallbackRef

    reference = ReloadableCallbackRef.normalize(callback)
    _theme_listeners[callback] = reference

    def unsubscribe():
        if _theme_listeners.get(callback) is reference:
            del _theme_listeners[callback]

    from infernux.engine.reload.owners import own_cleanup
    own_cleanup(callback, unsubscribe)
    return unsubscribe

def _apply_native_theme_overrides() -> None:
    from infernux.lib import (
        get_editor_theme_colors,
        get_editor_theme_floats,
        get_editor_theme_vec2s,
    )

    generation = _native_theme_generation()
    if Theme._NATIVE_THEME_GENERATION == generation:
        return
    applied = 0
    for name, value in get_editor_theme_colors().items():
        setattr(Theme, name, tuple(value))
        applied += 1
    for name, value in get_editor_theme_vec2s().items():
        setattr(Theme, name, tuple(value))
        applied += 1
    for name, value in get_editor_theme_floats().items():
        setattr(Theme, name, float(value))
        applied += 1
    Theme._NATIVE_OVERRIDES_APPLIED = applied
    Theme._NATIVE_THEME_GENERATION = generation
    for callback, reference in tuple(_theme_listeners.items()):
        if _theme_listeners.get(callback) is not reference:
            continue
        try:
            reference.invoke(propagate_exceptions=True)
        except Exception as exc:
            from infernux.debug import Debug
            Debug.log_exception(exc)

def _native_theme_generation() -> int:
    from infernux.lib import editor_theme_generation

    return int(editor_theme_generation())

def _set_editor_theme_native(name: str) -> bool:
    """Apply a theme without changing the user's preferred selection."""
    from infernux.lib import set_editor_theme as _native_set

    ok = bool(_native_set(name))
    if ok and Theme._NATIVE_THEME_GENERATION != _native_theme_generation():
        _apply_native_theme_overrides()
    return ok


def set_editor_theme(name: str) -> bool:
    """Switch the active editor theme and retain the editor selection.

    The C++ registry re-skins every built-in ImGui widget (C++ and Python
    panels) in one call. An active editor theme service also saves the user's
    preference. Without an editor, only the native and mirrored tokens change.
    Returns ``True`` on success, including an already selected theme.
    """
    import sys

    module = sys.modules.get("infernux.engine.editor_themes")
    service = module.EditorThemes.instance() if module is not None else None
    if service is not None:
        if name not in list_editor_themes():
            return False
        service.select(name)
        return True
    return _set_editor_theme_native(name)

def list_editor_themes() -> list:
    """Return the names of all registered editor themes."""
    from infernux.lib import list_editor_themes as _native_list

    return list(_native_list())

def active_editor_theme() -> str:
    """Return the active editor theme name."""
    from infernux.lib import get_editor_theme as _native_get

    return str(_native_get())

# Convenience API on the Theme class: Theme.set_theme("amber") / Theme.list_themes()
Theme.set_theme = staticmethod(set_editor_theme)
Theme.list_themes = staticmethod(list_editor_themes)
Theme.active_theme = staticmethod(active_editor_theme)
Theme.refresh = staticmethod(_apply_native_theme_overrides)

Theme._NATIVE_OVERRIDES_APPLIED = 0
Theme._NATIVE_THEME_GENERATION = -1
_apply_native_theme_overrides()
Theme._NATIVE_THEME_GENERATION = _native_theme_generation()
