"""
EditorPanel — Unified base class for editor panels.

All panels should inherit from this class and override
``on_render_content(ctx)``. The base class handles window frame management,
style push/pop, lifecycle hooks, and service access.

Creating a custom panel::

    from Infernux.engine.ui import EditorPanel, editor_panel, EditorEvent

    @editor_panel("My Debug Panel")
    class MyDebugPanel(EditorPanel):
        def on_enable(self):
            self.events.subscribe(EditorEvent.SELECTION_CHANGED, self._on_sel)

        def on_disable(self):
            self.events.unsubscribe(EditorEvent.SELECTION_CHANGED, self._on_sel)

        def on_render_content(self, ctx):
            ctx.text("Hello from my custom panel!")

        def _on_sel(self, obj):
            pass
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .closable_panel import ClosablePanel

if TYPE_CHECKING:
    from Infernux.lib import InxGUIContext
    from .editor_services import EditorServices
    from .event_bus import EditorEventBus


class EditorPanel(ClosablePanel):
    """Unified base class for editor panels.

    Provides:
    - ``self.services`` to access :class:`EditorServices`
    - ``self.events`` to access :class:`EditorEventBus`
    - ``on_enable()`` when the panel is created or reopened
    - ``on_disable()`` when the panel closes
    - ``on_render_content(ctx)`` for the panel body

    Overridable hooks:
    - ``_window_flags()`` returns ImGui window flags
    - ``_initial_size()`` returns the initial window size or ``None``
    - ``_push_window_style(ctx)`` pushes styles before ``begin_window``
    - ``_pop_window_style(ctx)`` pops styles after ``end_window``
    - ``_on_visible_pre(ctx)`` runs before ``on_render_content``
    - ``save_state() / load_state(data)`` persist panel state
    """

    def __init__(self, title: str, window_id: Optional[str] = None):
        super().__init__(title, window_id)
        self._enable_called = False

    # ------------------------------------------------------------------
    # Service and Event Access
    # ------------------------------------------------------------------

    @property
    def services(self) -> EditorServices:
        """Access editor subsystems."""
        from .editor_services import EditorServices
        return EditorServices.instance()

    @property
    def events(self) -> EditorEventBus:
        """Access the editor event bus."""
        from .event_bus import EditorEventBus
        return EditorEventBus.instance()

    # ------------------------------------------------------------------
    # Lifecycle Hooks
    # ------------------------------------------------------------------

    def on_enable(self) -> None:
        """Called once when the panel is first rendered.

        Subscribe to events here.
        """
        pass

    def on_disable(self) -> None:
        """Called when the panel is closed.

        Unsubscribe here.
        """
        pass

    # ------------------------------------------------------------------
    # Window Configuration Hooks
    # ------------------------------------------------------------------

    def _window_flags(self) -> int:
        """Return ImGui window flags for this panel.

        The default is 0.
        """
        return 0

    def _initial_size(self) -> Optional[tuple[float, float]]:
        """Return the initial window size ``(w, h)``.

        Return ``None`` to use the ImGui default.
        """
        return None

    def _push_window_style(self, ctx) -> None:
        """Push style vars and colors before ``begin_window``.

        Subclasses must pop the same number in ``_pop_window_style``.
        """
        pass

    def _pop_window_style(self, ctx) -> None:
        """Pop style vars and colors after ``end_window``.

        The pop count must match ``_push_window_style``.
        """
        pass

    def _on_visible_pre(self, ctx) -> None:
        """Run after ``begin_window`` succeeds and before content rendering.

        Useful for one-shot per-frame setup such as focus tracking.
        """
        pass

    def _on_not_visible(self, ctx) -> None:
        """Run when ``begin_window`` returns ``False``.

        Useful for resource management such as pausing render targets.
        """
        pass

    def _pre_render(self, ctx) -> None:
        """Run in ``on_render`` before the window begins.

        Use this for per-frame work that must happen outside the window frame.
        """
        pass

    # ------------------------------------------------------------------
    # Content Rendering
    # ------------------------------------------------------------------

    def on_render_content(self, ctx: InxGUIContext) -> None:
        """Render panel content.

        Override this instead of ``on_render``.
        """
        pass

    # ------------------------------------------------------------------
    # Unified Empty State
    # ------------------------------------------------------------------

    def _render_empty_state(
        self,
        ctx: InxGUIContext,
        hint: Optional[str] = None,
        *,
        drop_types: Optional[List[str]] = None,
        on_drop=None,
        min_height: float = 220.0,
    ) -> None:
        """Draw a centered bordered hint box — the standard "nothing loaded" UI.

        This is the canonical empty-state rendering that all panels should
        use so every editor window has a consistent look.

        Args:
            ctx: The ImGui context.
            hint: Text shown inside the drop zone.  Falls back to
                ``_empty_state_hint()``.
            drop_types: Accepted drag-and-drop payload types.
                Falls back to ``_empty_state_drop_types()``.
            on_drop: Callback ``(payload_type, payload)`` when a drop is
                accepted.  Falls back to ``_on_empty_state_drop``.
            min_height: Minimum height of the empty-state region.
        """
        from .igui import IGUI

        hint = hint or self._empty_state_hint()
        drop_types = drop_types if drop_types is not None else self._empty_state_drop_types()
        on_drop = on_drop or getattr(self, '_on_empty_state_drop', None)

        avail_w = ctx.get_content_region_avail_width()
        empty_h = max(ctx.get_content_region_avail_height(), min_height)

        ctx.begin_child(f"##{self._window_id}_empty_state", avail_w, empty_h, True)
        try:
            region_w = ctx.get_content_region_avail_width()
            region_h = ctx.get_content_region_avail_height()
            zone_w = min(max(region_w - 28.0, 220.0), 460.0)
            zone_h = min(max(region_h - 36.0, 140.0), 250.0)
            start_x = ctx.get_cursor_pos_x() + (region_w - zone_w) * 0.5
            start_y = ctx.get_cursor_pos_y() + (region_h - zone_h) * 0.5

            ctx.set_cursor_pos_x(start_x)
            ctx.set_cursor_pos_y(start_y)
            ctx.invisible_button(f"##{self._window_id}_drop_zone", zone_w, zone_h)

            bx0 = ctx.get_item_rect_min_x()
            by0 = ctx.get_item_rect_min_y()
            bx1 = ctx.get_item_rect_max_x()
            by1 = ctx.get_item_rect_max_y()
            ctx.draw_rect(bx0, by0, bx1, by1, 0.55, 0.55, 0.55, 0.55, 2.0, 8.0)
            ctx.draw_text_aligned(
                bx0, by0, bx1, by1,
                hint,
                0.72, 0.72, 0.72, 0.95,
                0.5, 0.5,
            )

            if drop_types and on_drop:
                IGUI.multi_drop_target(ctx, drop_types, on_drop, outline=True)
        finally:
            ctx.end_child()

    def _empty_state_hint(self) -> str:
        """Return the hint text for the default empty state.

        Override to customize the empty-state message.
        """
        from Infernux.engine.i18n import t
        return t("panel.empty_hint")

    def _empty_state_drop_types(self) -> List[str]:
        """Return accepted drop-target payload types for the empty state.

        Override to accept specific drop types.  Return ``[]`` to disable
        the drop zone.
        """
        return []

    # ------------------------------------------------------------------
    # State Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        """Return a panel state dict for persistence."""
        return {}

    def load_state(self, data: dict) -> None:
        """Restore panel state from persisted data."""
        pass

    # ------------------------------------------------------------------
    # Unified Render Frame
    # ------------------------------------------------------------------

    def on_render(self, ctx) -> None:
        """Unified render frame for the panel.

        Subclasses should not override this method. Override the hook methods
        above instead.
        """
        if not self._is_open:
            return

        # Trigger on_enable once.
        if not self._enable_called:
            self._enable_called = True
            self.on_enable()

        # Apply the initial size on first use if provided.
        init_size = self._initial_size()
        if init_size is not None:
            from .theme import Theme
            ctx.set_next_window_size(init_size[0], init_size[1], Theme.COND_FIRST_USE_EVER)

        # Run pre-frame logic before the window begins.
        self._pre_render(ctx)

        # Push window styles.
        self._push_window_style(ctx)

        visible = self._begin_closable_window(ctx, self._window_flags())
        if visible:
            self._on_visible_pre(ctx)
            self.on_render_content(ctx)
        else:
            self._on_not_visible(ctx)
        ctx.end_window()

        # Pop window styles.
        self._pop_window_style(ctx)

        # Fire the close hook when the panel is closed.
        if not self._is_open:
            self.on_disable()
