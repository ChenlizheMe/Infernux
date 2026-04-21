"""
Base class for closable editor panels.
"""

from Infernux.lib import InxGUIRenderable, InxGUIContext
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .window_manager import WindowManager


_HOVERED_CHILD_WINDOWS = 1  # ImGuiHoveredFlags_ChildWindows
_HOVERED_NO_POPUP_HIERARCHY = 8  # ImGuiHoveredFlags_NoPopupHierarchy
_PANEL_ACTIVATION_HOVER_FLAGS = _HOVERED_CHILD_WINDOWS | _HOVERED_NO_POPUP_HIERARCHY


class ClosablePanel(InxGUIRenderable):
    """
    Base class for panels that can be closed via the window close button.
    """
    
    # Class-level registration info
    WINDOW_TYPE_ID: Optional[str] = None
    WINDOW_DISPLAY_NAME: Optional[str] = None
    WINDOW_TITLE_KEY: Optional[str] = None

    # ── Class-level focus tracking ──
    _active_panel_id: Optional[str] = None
    _on_panel_focus_changed: Optional[Callable[[str, str], None]] = None
    
    def __init__(self, title: str, window_id: Optional[str] = None):
        super().__init__()
        self._title = title
        self._title_key: Optional[str] = getattr(self.__class__, 'WINDOW_TITLE_KEY', None)
        self._window_id = window_id or self.__class__.__name__
        self._is_open = True
        self._window_manager: Optional['WindowManager'] = None
        self._panel_was_focused: bool = False
    
    @property
    def window_id(self) -> str:
        return self._window_id
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    def set_window_manager(self, window_manager: 'WindowManager'):
        """Set the window manager reference."""
        self._window_manager = window_manager

    def open(self):
        """Ensure this panel is visible."""
        self._is_open = True

    def close(self):
        """Close this panel and notify the window manager."""
        self._is_open = False
        if self._window_manager:
            self._window_manager.set_window_open(self._window_id, False)

    def can_close(self, ctx: InxGUIContext) -> bool:
        """Return whether the panel can close when the titlebar X is clicked."""
        return True

    def _resolve_panel_display_title(self) -> str:
        if self._title_key:
            try:
                from Infernux.engine.i18n import t

                return t(self._title_key)
            except Exception:
                return self._title
        return self._title

    def _resolve_panel_save_handler(self):
        save_fn = getattr(self, "_do_save", None)
        if callable(save_fn):
            def _wrapped_save():
                try:
                    save_fn()
                except Exception:
                    return False
                # If panels expose _dirty, require it to be cleared after save.
                if hasattr(self, "_dirty"):
                    try:
                        return not bool(getattr(self, "_dirty"))
                    except Exception:
                        return False
                return True

            return _wrapped_save

        # Fallback for clip-like editors exposing _save_clip(active_clip).
        save_clip_fn = getattr(self, "_save_clip", None)
        if callable(save_clip_fn):
            def _wrapped_clip_save():
                try:
                    clip = getattr(self, "_active_clip", None)
                    if clip is None:
                        return False
                    save_clip_fn(clip)
                except Exception:
                    return False
                if hasattr(self, "_dirty"):
                    try:
                        return not bool(getattr(self, "_dirty"))
                    except Exception:
                        return False
                return True

            return _wrapped_clip_save

        return None

    def _sync_dirty_registry(self) -> None:
        try:
            from Infernux.engine.project_context import set_panel_dirty

            dirty = bool(getattr(self, "_dirty", False))
            set_panel_dirty(
                self._window_id,
                dirty,
                title=self._resolve_panel_display_title(),
                save_handler=self._resolve_panel_save_handler(),
            )
        except Exception:
            pass

    def _confirm_close_with_dirty_registry(self) -> bool:
        try:
            from Infernux.engine.project_context import is_panel_dirty, set_panel_dirty
            from ._dialogs import ask_save_discard_cancel

            if not is_panel_dirty(self._window_id):
                return True

            title = self._resolve_panel_display_title()
            choice = ask_save_discard_cancel(
                title=f"Unsaved {title}",
                message=f"{title} has unsaved changes. Save before closing?",
            )
            if choice == "cancel":
                return False
            if choice == "discard":
                if hasattr(self, "_dirty"):
                    try:
                        setattr(self, "_dirty", False)
                    except Exception:
                        pass
                set_panel_dirty(
                    self._window_id,
                    False,
                    title=title,
                    save_handler=self._resolve_panel_save_handler(),
                )
                return True

            # save
            save_handler = self._resolve_panel_save_handler()
            if not callable(save_handler):
                return False
            ok = bool(save_handler())
            if not ok:
                return False
            self._sync_dirty_registry()
            return not bool(getattr(self, "_dirty", False))
        except Exception:
            return True

    def request_focus(self, ctx: InxGUIContext):
        """Programmatically focus this panel on the next frame."""
        ctx.set_next_window_focus()

    def _activate_panel(self, ctx: InxGUIContext, *, focus_window: bool = False):
        if focus_window:
            ctx.set_window_focus()

        old_id = ClosablePanel._active_panel_id or ""
        if old_id == self._window_id:
            return

        ClosablePanel._active_panel_id = self._window_id

        # Canonical channel: EditorEventBus.PANEL_FOCUSED.
        # The legacy class-level callback is kept for now so existing
        # bootstrap wiring keeps working, but new subscribers should prefer
        # EditorEventBus to avoid the dual-channel split that previously
        # left PANEL_FOCUSED defined-but-never-emitted.
        try:
            from .event_bus import EditorEvent, EditorEventBus
            EditorEventBus.instance().emit(EditorEvent.PANEL_FOCUSED, self._window_id)
        except Exception:
            # Event bus unavailable during bootstrap import — fall through to the
            # legacy callback so the focus signal is never silently lost.
            pass

        cb = ClosablePanel._on_panel_focus_changed
        if cb is not None:
            cb(old_id, self._window_id)

    @classmethod
    def set_on_panel_focus_changed(cls, callback: Optional[Callable[[str, str], None]]):
        """Set a class-level callback ``(old_panel_id, new_panel_id)`` fired on focus changes."""
        cls._on_panel_focus_changed = callback

    @classmethod
    def get_active_panel_id(cls) -> Optional[str]:
        return cls._active_panel_id

    def _window_title_suffix(self) -> str:
        """Return a suffix appended to the window title (e.g. ' *' for dirty)."""
        return ""

    @classmethod
    def focus_panel_by_id(cls, panel_id: str):
        """Mark *panel_id* as active (used by undo replay to set focus target)."""
        cls._pending_focus_panel_id = panel_id

    # Request that the NEXT on_render cycle focuses this panel
    _pending_focus_panel_id: Optional[str] = None
    
    def _begin_closable_window(self, ctx: InxGUIContext, flags: int = 0) -> bool:
        """
        Begin a closable window. Returns True if window content should be rendered.
        Handles close button automatically.
        """
        # If this panel was requested to be focused, do it before begin
        if ClosablePanel._pending_focus_panel_id == self._window_id:
            ctx.set_next_window_focus()
            ClosablePanel._pending_focus_panel_id = None

        # Resolve title via i18n if a title_key is set
        if self._title_key:
            from Infernux.engine.i18n import t
            display = t(self._title_key)
        else:
            display = self._title

        self._sync_dirty_registry()

        display += self._window_title_suffix()
        safe_title = str(display).replace('\x00', '�').encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        # Use ### to keep a stable ImGui window ID independent of the
        # displayed title so docking layout survives locale changes.
        safe_title = f"{safe_title}###{self._window_id}"
        visible, self._is_open = ctx.begin_window_closable(safe_title, self._is_open, flags)
        
        # If the titlebar close button was pressed, let the panel veto close
        # (for example, unsaved-change confirmation popups).
        if not self._is_open:
            if self._confirm_close_with_dirty_registry() and self.can_close(ctx):
                if self._window_manager:
                    self._window_manager.set_window_open(self._window_id, False)
            else:
                self._is_open = True

        # ── Focus tracking ──
        if visible and self._is_open:
            pointer_activated = ctx.is_window_hovered(_PANEL_ACTIVATION_HOVER_FLAGS) and any(
                ctx.is_mouse_button_clicked(button) for button in (0, 1, 2)
            )
            if pointer_activated:
                self._activate_panel(ctx, focus_window=True)

            focused = ctx.is_window_focused(0)
            if focused and not self._panel_was_focused:
                self._activate_panel(ctx)
            self._panel_was_focused = focused
        
        return visible and self._is_open
