"""
Preferences — floating editor preferences window.

Rendered by MenuBarPanel each frame when visible.
Currently exposes only language selection; more settings can be added later.
"""

from __future__ import annotations

from Infernux.engine.i18n import t, get_locale, set_locale
from Infernux.engine.ide_preference import get_ide, set_ide
from .theme import Theme


_LOCALES = ["en", "zh"]
_LOCALE_LABELS = ["English", "简体中文"]
_IDES = ["vscode", "pycharm"]


class PreferencesPanel:
    """Standalone floating Preferences window."""

    def __init__(self) -> None:
        self._visible: bool = False
        self._first_open: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._visible = True
        self._first_open = True

    def close(self) -> None:
        self._visible = False

    @property
    def is_open(self) -> bool:
        return self._visible

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, ctx) -> None:
        if not self._visible:
            return

        x0, y0, dw, dh = ctx.get_main_viewport_bounds()
        cx = x0 + (dw - 980) * 0.5
        cy = y0 + (dh - 720) * 0.5
        ctx.set_next_window_pos(cx, cy, Theme.COND_ALWAYS, 0.0, 0.0)
        ctx.set_next_window_size(980, 720, Theme.COND_ALWAYS)

        visible, still_open = ctx.begin_window_closable(
            t("prefs.title") + "###prefs", self._visible, Theme.WINDOW_FLAGS_DIALOG
        )

        if not still_open:
            self._visible = False
            ctx.end_window()
            return

        if visible:
            self._render_body(ctx)

        ctx.end_window()

    def _render_body(self, ctx) -> None:
        ctx.label(t("prefs.language"))
        ctx.same_line(150)
        avail = ctx.get_content_region_avail_width()
        ctx.set_next_item_width(avail)

        current_idx = _LOCALES.index(get_locale()) if get_locale() in _LOCALES else 0
        new_idx = ctx.combo("##language", current_idx, _LOCALE_LABELS)
        if new_idx != current_idx:
            set_locale(_LOCALES[new_idx])

        ctx.label(t("prefs.ide"))
        ctx.same_line(150)
        avail = ctx.get_content_region_avail_width()
        ctx.set_next_item_width(avail)

        ide_labels = [t("prefs.ide.vscode"), t("prefs.ide.pycharm")]
        current_ide = get_ide()
        current_ide_idx = _IDES.index(current_ide) if current_ide in _IDES else 0
        new_ide_idx = ctx.combo("##preferred_ide", current_ide_idx, ide_labels)
        if new_ide_idx != current_ide_idx:
            set_ide(_IDES[new_ide_idx])

