"""
Base class for closable editor panels.
"""

from InfEngine.lib import InfGUIRenderable, InfGUIContext
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .window_manager import WindowManager


class ClosablePanel(InfGUIRenderable):
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

    def request_focus(self, ctx: InfGUIContext):
        """Programmatically focus this panel on the next frame."""
        ctx.set_next_window_focus()

    @classmethod
    def set_on_panel_focus_changed(cls, callback: Optional[Callable[[str, str], None]]):
        """Set a class-level callback ``(old_panel_id, new_panel_id)`` fired on focus changes."""
        cls._on_panel_focus_changed = callback

    @classmethod
    def get_active_panel_id(cls) -> Optional[str]:
        return cls._active_panel_id

    @classmethod
    def focus_panel_by_id(cls, panel_id: str):
        """Mark *panel_id* as active (used by undo replay to set focus target)."""
        cls._pending_focus_panel_id = panel_id

    # Request that the NEXT on_render cycle focuses this panel
    _pending_focus_panel_id: Optional[str] = None
    
    def _begin_closable_window(self, ctx: InfGUIContext, flags: int = 0) -> bool:
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
            from InfEngine.engine.i18n import t
            display = t(self._title_key)
        else:
            display = self._title
        safe_title = str(display).replace('\x00', '�').encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        # Use ### to keep a stable ImGui window ID independent of the
        # displayed title so docking layout survives locale changes.
        safe_title = f"{safe_title}###{self._window_id}"
        visible, self._is_open = ctx.begin_window_closable(safe_title, self._is_open, flags)
        
        # If window was closed, notify window manager
        if not self._is_open and self._window_manager:
            self._window_manager.set_window_open(self._window_id, False)

        # ── Focus tracking ──
        if visible and self._is_open:
            focused = ctx.is_window_focused(0)
            if focused and not self._panel_was_focused:
                old_id = ClosablePanel._active_panel_id or ""
                if old_id != self._window_id:
                    ClosablePanel._active_panel_id = self._window_id
                    cb = ClosablePanel._on_panel_focus_changed
                    if cb is not None:
                        cb(old_id, self._window_id)
            self._panel_was_focused = focused
        
        return visible and self._is_open
