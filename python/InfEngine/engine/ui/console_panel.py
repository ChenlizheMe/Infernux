"""
Unity-style Console panel — Notion dark theme.

Features:
  - Toolbar: Clear, Collapse, Clear on Play, Error Pause, filter toggles w/ counts
  - Selectable log entries with level coloring and unique IDs (no ImGui ID conflicts)
  - Collapse identical messages with count badge on the right
  - Detail pane at bottom showing stack trace of selected log
  - Smart auto-scroll: scrolls to bottom unless user scrolls up manually
  - Filters out internal engine noise — only user-relevant logs displayed
"""

from InfEngine.lib import InfGUIContext
from InfEngine.engine.i18n import t
from .editor_panel import EditorPanel
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol, ImGuiStyleVar
from .inspector_utils import render_inspector_checkbox


@editor_panel("Console", type_id="console", title_key="panel.console")
class ConsolePanel(EditorPanel):
    WINDOW_TYPE_ID = "console"
    WINDOW_DISPLAY_NAME = "Console"

    # Aliases pointing to the central Theme palette
    _CLR_INFO    = Theme.LOG_INFO
    _CLR_WARN    = Theme.LOG_WARNING
    _CLR_ERROR   = Theme.LOG_ERROR
    _CLR_TRACE   = Theme.LOG_TRACE
    _CLR_BADGE   = Theme.LOG_BADGE
    _CLR_SEL     = Theme.SELECTION_BG

    # ImGui key codes
    _KEY_C = 548            # ImGuiKey_C
    _MOD_CTRL = 4096        # ImGuiMod_Ctrl (1 << 12)
    # ImGuiInputTextFlags_ReadOnly
    _INPUT_TEXT_READONLY = 1 << 14

    def __init__(self, title: str = "Console"):
        super().__init__(title, window_id="console")
        self._logs: list = []
        self._max_logs = 2000
        self._show_info = True
        self._show_warnings = True
        self._show_errors = True
        self._auto_scroll = True
        self._collapse = False
        self._clear_on_play = True
        self._error_pause = False
        self._selected_vis_index = -1
        self._user_scrolled_up = False   # track manual scroll
        self._log_counter = 0            # monotonic counter for unique IDs
        self._play_mode_manager = None
        self._detail_h = 90.0            # draggable detail pane height
        self._request_focus = False       # set True to focus console next frame
        # ---- Cached filter / count state (avoid rebuilding every frame) ----
        self._cache_dirty = True          # set True when _logs changes
        self._filter_dirty = True         # set True when filter toggles change
        self._cached_visible: list = []
        self._cached_counts = (0, 0, 0)   # (info, warn, err)
        self._prev_show_info = True
        self._prev_show_warnings = True
        self._prev_show_errors = True
        self._prev_collapse = False
        # ---- Virtual scrolling ----
        self._row_height = 22.0           # estimated; self-calibrated on first render
        self._row_height_measured = False
        self._register_debug_listener()

    # ---- State persistence ----
    def save_state(self) -> dict:
        return {
            "show_info": self._show_info,
            "show_warnings": self._show_warnings,
            "show_errors": self._show_errors,
            "collapse": self._collapse,
            "clear_on_play": self._clear_on_play,
            "error_pause": self._error_pause,
            "auto_scroll": self._auto_scroll,
        }

    def load_state(self, data: dict) -> None:
        self._show_info = data.get("show_info", True)
        self._show_warnings = data.get("show_warnings", True)
        self._show_errors = data.get("show_errors", True)
        self._collapse = data.get("collapse", False)
        self._clear_on_play = data.get("clear_on_play", True)
        self._error_pause = data.get("error_pause", False)
        self._auto_scroll = data.get("auto_scroll", True)
        self._filter_dirty = True

    def set_play_mode_manager(self, manager):
        """Wire the console to a PlayModeManager so Clear on Play and
        Error Pause actually work."""
        from InfEngine.engine.play_mode import PlayModeState
        self._play_mode_manager = manager
        def _on_state_change(event):
            if event.new_state == PlayModeState.PLAYING and self._clear_on_play:
                self.clear()
        manager.add_state_change_listener(_on_state_change)

    def set_status_bar(self, status_bar) -> None:
        """Wire the status bar so its counters are reset when the console is cleared."""
        self._status_bar = status_bar

    def select_latest_entry(self) -> None:
        """Select the last visible log entry and request focus.
        Called by the status bar when the user clicks the bottom bar."""
        self._is_open = True
        self._request_focus = True
        # Selection will be applied in _render_body when visible list is built
        self._selected_vis_index = -2   # sentinel: "select last"

    # ------------------------------------------------------------------
    # Debug listener
    # ------------------------------------------------------------------
    def _register_debug_listener(self):
        from InfEngine.debug import DebugConsole
        console = DebugConsole.instance()
        console.add_listener(self._on_debug_log)
        for entry in console.get_entries():
            self._add_log_entry(entry)

    @staticmethod
    def _sanitize_text(value) -> str:
        """Normalize arbitrary text into a UI-safe UTF-8 string."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            text = value.decode('utf-8', errors='replace')
        else:
            text = str(value)
        text = text.replace('\x00', '�')
        return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

    def _on_debug_log(self, entry):
        self._add_log_entry(entry)

    @staticmethod
    def _is_internal(entry) -> bool:
        """Return True if this entry is internal engine noise.
        
        Checks the ``internal`` flag set at the source via
        ``Debug.log_internal()``.  Also filters Dear ImGui programmer
        errors that leak through the C++ layer.
        """
        if getattr(entry, 'internal', False):
            return True
        msg = entry.message if hasattr(entry, 'message') else str(entry)
        if "DEAR IMGUI" in msg or "PushID" in msg or "conflicting ID" in msg:
            return True
        return False

    def _add_log_entry(self, entry):
        from InfEngine.debug import LogType
        level_map = {
            LogType.LOG: "info", LogType.WARNING: "warning",
            LogType.ERROR: "error", LogType.ASSERT: "error",
            LogType.EXCEPTION: "error",
        }
        msg = self._sanitize_text(getattr(entry, 'message', ''))
        level = level_map.get(entry.log_type, "info")
        # Skip internal engine noise
        if self._is_internal(entry):
            return
        self._log_counter += 1
        self._logs.append({
            "message": msg,
            "level": level,
            "timestamp": self._sanitize_text(entry.get_formatted_time()),
            "stack_trace": self._sanitize_text(getattr(entry, "stack_trace", "")),
            "source_file": self._sanitize_text(getattr(entry, "source_file", "")),
            "source_line": int(getattr(entry, "source_line", 0) or 0),
            "uid": self._log_counter,
        })
        if len(self._logs) > self._max_logs:
            self._logs.pop(0)
        self._cache_dirty = True
        # Error Pause: pause play mode on first error
        if level == "error" and self._error_pause:
            pm = self._play_mode_manager
            if pm is None:
                from InfEngine.engine.play_mode import PlayModeManager
                pm = PlayModeManager.instance()
            if pm is not None and pm.is_playing and not pm.is_paused:
                pm.pause()

    def log(self, message: str, level: str = "info"):
        """Manually add a log entry (used for startup messages)."""
        from datetime import datetime
        self._log_counter += 1
        self._logs.append({
            "message": self._sanitize_text(message), "level": level,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "stack_trace": "",
            "uid": self._log_counter,
        })
        if len(self._logs) > self._max_logs:
            self._logs.pop(0)
        self._cache_dirty = True

    def clear(self):
        self._logs.clear()
        self._selected_vis_index = -1
        self._log_counter = 0
        self._cache_dirty = True
        from InfEngine.debug import DebugConsole
        DebugConsole.instance().clear()
        # Keep status bar in sync
        if hasattr(self, '_status_bar') and self._status_bar is not None:
            self._status_bar.clear_counts()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_filter_change(self) -> bool:
        """Return True and mark dirty if any filter toggle changed."""
        changed = (
            self._show_info != self._prev_show_info
            or self._show_warnings != self._prev_show_warnings
            or self._show_errors != self._prev_show_errors
            or self._collapse != self._prev_collapse
        )
        if changed:
            self._prev_show_info = self._show_info
            self._prev_show_warnings = self._show_warnings
            self._prev_show_errors = self._show_errors
            self._prev_collapse = self._collapse
            self._filter_dirty = True
        return changed

    def _ensure_cache(self):
        """Rebuild counts & visible list only when logs or filters changed."""
        self._detect_filter_change()
        if not self._cache_dirty and not self._filter_dirty:
            return
        # Rebuild counts (always from full _logs)
        i = w = e = 0
        for lg in self._logs:
            lv = lg["level"]
            if lv == "info":     i += 1
            elif lv == "warning": w += 1
            else:                 e += 1
        self._cached_counts = (i, w, e)

        # Rebuild visible list
        visible = []
        collapse_map = {}
        for lg in self._logs:
            lv = lg["level"]
            if lv == "info" and not self._show_info:
                continue
            if lv == "warning" and not self._show_warnings:
                continue
            if lv == "error" and not self._show_errors:
                continue
            if self._collapse:
                key = (lv, lg["message"])
                if key in collapse_map:
                    ci = collapse_map[key]
                    visible[ci]["count"] += 1
                    visible[ci]["timestamp"] = lg["timestamp"]
                    visible[ci]["uid"] = lg["uid"]
                    continue
                collapse_map[key] = len(visible)
            visible.append({
                "message": self._sanitize_text(lg["message"]), "level": lv,
                "timestamp": self._sanitize_text(lg["timestamp"]),
                "stack_trace": self._sanitize_text(lg.get("stack_trace", "")),
                "source_file": self._sanitize_text(lg.get("source_file", "")),
                "source_line": lg.get("source_line", 0),
                "count": 1,
                "uid": lg["uid"],
            })
        self._cached_visible = visible
        self._cache_dirty = False
        self._filter_dirty = False

    # ------------------------------------------------------------------
    # EditorPanel hooks
    # ------------------------------------------------------------------

    def _initial_size(self):
        return (700, 250)

    def _pre_render(self, ctx: InfGUIContext):
        self._ensure_cache()
        if self._request_focus:
            ctx.set_next_window_focus()
            self._request_focus = False

    def on_render_content(self, ctx: InfGUIContext):
        log_cnt, warn_cnt, err_cnt = self._cached_counts
        self._render_toolbar(ctx, log_cnt, warn_cnt, err_cnt)
        ctx.separator()
        self._render_body(ctx)

    # ------ toolbar ------
    def _render_toolbar(self, ctx, log_cnt, warn_cnt, err_cnt):
        Theme.push_console_toolbar_vars(ctx)  # FramePadding + ItemSpacing + FrameBorderSize (3 vars)

        ctx.button(t("console.clear"), self.clear)
        ctx.same_line()
        self._collapse = render_inspector_checkbox(ctx, t("console.collapse"), self._collapse)
        ctx.same_line()
        self._clear_on_play = render_inspector_checkbox(ctx, t("console.clear_on_play"), self._clear_on_play)
        ctx.same_line()
        self._error_pause = render_inspector_checkbox(ctx, t("console.error_pause"), self._error_pause)

        # Right-align Log / Warn / Error filter toggles
        filter_width = 240  # approximate total width of 3 filter checkboxes
        left_items_w = 350  # approximate total width of left-side controls
        win_w = ctx.get_window_width()
        filter_x = max(win_w - filter_width, left_items_w)
        if win_w >= left_items_w + filter_width:
            ctx.same_line(filter_x)
        # else: new line — let items wrap naturally
        self._show_info = render_inspector_checkbox(
            ctx, f"Log {log_cnt}" if log_cnt else "Log", self._show_info)

        ctx.same_line()
        ctx.push_style_color(ImGuiCol.Text, *Theme.LOG_WARNING)
        self._show_warnings = render_inspector_checkbox(
            ctx, f"Warn {warn_cnt}" if warn_cnt else "Warn", self._show_warnings)
        ctx.pop_style_color(1)

        ctx.same_line()
        ctx.push_style_color(ImGuiCol.Text, *Theme.LOG_ERROR)
        self._show_errors = render_inspector_checkbox(
            ctx, f"Error {err_cnt}" if err_cnt else "Error", self._show_errors)
        ctx.pop_style_color(1)

        ctx.pop_style_var(3)

    # ------ body (list + detail) ------
    def _render_body(self, ctx):
        avail_h = ctx.get_content_region_avail_height()
        has_detail = self._selected_vis_index >= 0 or self._selected_vis_index == -2

        visible = self._cached_visible

        # Resolve sentinel -2 → last visible entry
        if self._selected_vis_index == -2:
            if visible:
                self._selected_vis_index = len(visible) - 1
            else:
                self._selected_vis_index = -1
            has_detail = self._selected_vis_index >= 0

        # Clamp selection
        if self._selected_vis_index >= len(visible):
            self._selected_vis_index = -1
            has_detail = False

        # Compute heights
        splitter_h = 3.0
        if has_detail:
            self._detail_h = max(40.0, min(self._detail_h, avail_h - 60.0))
            list_h = max(avail_h - self._detail_h - splitter_h, 40.0)
        else:
            list_h = -1

        total = len(visible)
        row_h = self._row_height

        # --- Log list (virtual-scrolled) ---
        Theme.push_transparent_border(ctx)  # 1 colour
        if ctx.begin_child("##ConsoleLogList", 0, list_h, True):
            # Determine visible row range from scroll position
            scroll_y = ctx.get_scroll_y()
            viewport_h = ctx.get_content_region_avail_height()
            first_vis = max(int(scroll_y / row_h), 0) if row_h > 0 else 0
            last_vis = min(first_vis + int(viewport_h / row_h) + 2, total - 1) if total > 0 else -1

            # Top spacer for rows above the viewport
            if first_vis > 0:
                avail_w = ctx.get_content_region_avail_width()
                ctx.dummy(avail_w, first_vis * row_h)

            # Render only visible rows
            for idx in range(max(first_vis, 0), last_vis + 1):
                entry = visible[idx]
                # Self-calibrate row height from the first rendered row
                if not self._row_height_measured:
                    y0 = ctx.get_cursor_pos_y()
                    self._render_row(ctx, idx, entry)
                    y1 = ctx.get_cursor_pos_y()
                    measured = y1 - y0
                    if measured > 1.0:
                        self._row_height = measured
                        self._row_height_measured = True
                else:
                    self._render_row(ctx, idx, entry)

            # Bottom spacer for rows below the viewport
            remaining = total - (last_vis + 1)
            if remaining > 0:
                avail_w = ctx.get_content_region_avail_width()
                ctx.dummy(avail_w, remaining * row_h)

            # Ctrl+C: copy selected entry text to clipboard
            if (0 <= self._selected_vis_index < total
                    and ctx.is_key_down(527)   # ImGuiKey_LeftCtrl
                    and ctx.is_key_pressed(self._KEY_C)):
                sel = visible[self._selected_vis_index]
                copy_text = sel["message"]
                st = sel.get("stack_trace", "")
                if st:
                    copy_text += "\n" + st
                ctx.set_clipboard_text(copy_text)

            # Smart auto-scroll: only if user hasn't scrolled up
            scroll_y = ctx.get_scroll_y()
            scroll_max = ctx.get_scroll_max_y()
            if scroll_max > 0:
                at_bottom = (scroll_max - scroll_y) < 20.0
                if at_bottom:
                    self._user_scrolled_up = False
                else:
                    self._user_scrolled_up = True

            if not self._user_scrolled_up and visible:
                ctx.set_scroll_here_y(1.0)
        ctx.end_child()
        ctx.pop_style_color(1)

        # --- Draggable splitter ---
        if has_detail:
            avail_w = ctx.get_content_region_avail_width()
            Theme.push_splitter_style(ctx)  # 3 colours
            ctx.invisible_button("##ConsoleSplitter", avail_w, splitter_h)
            if ctx.is_item_active():
                dy = ctx.get_mouse_drag_delta_y(0)
                if abs(dy) > 0.5:
                    self._detail_h = max(40.0, self._detail_h - dy)
                    ctx.reset_mouse_drag_delta(0)
                ctx.set_mouse_cursor(3)  # ResizeNS
            elif ctx.is_item_hovered():
                ctx.set_mouse_cursor(3)  # ResizeNS
            ctx.pop_style_color(3)

        # --- Detail pane (selectable / copyable text) ---
        if has_detail and 0 <= self._selected_vis_index < len(visible):
            sel = visible[self._selected_vis_index]
            lv = sel["level"]
            clr = self._level_color(lv)

            # Build full text for the multiline widget
            detail_text = self._sanitize_text(f"[{sel['timestamp']}]  {sel['message']}")
            st = sel.get("stack_trace", "")
            if st:
                detail_text += "\n\n" + self._sanitize_text(st)

            # Read-only multiline input — supports mouse text selection & Ctrl+C
            ctx.push_style_color(ImGuiCol.Text, *clr)
            ctx.push_style_color(ImGuiCol.WindowBg, *Theme.ROW_NONE)    # transparent
            ctx.push_style_color(ImGuiCol.FrameBg, *Theme.ROW_NONE)    # transparent
            ctx.push_style_var_float(ImGuiStyleVar.WindowBorderSize, Theme.BORDER_SIZE_NONE)
            ctx.input_text_multiline(
                "##ConsoleDetail", detail_text,
                buffer_size=len(detail_text) + 1,
                width=-1, height=-1,
                flags=self._INPUT_TEXT_READONLY)
            ctx.pop_style_var(1)
            ctx.pop_style_color(3)

    # ------ single row ------
    def _render_row(self, ctx, idx, entry):
        lv = entry["level"]
        clr = self._level_color(lv)
        uid = entry.get("uid", idx)
        is_sel = (idx == self._selected_vis_index)

        # Row background: selected = steel-blue highlight, otherwise alternating
        if is_sel:
            ctx.push_style_color(ImGuiCol.Header, *Theme.SELECTION_BG)
        elif idx % 2 == 1:
            ctx.push_style_color(ImGuiCol.Header, *Theme.ROW_ALT)
        else:
            ctx.push_style_color(ImGuiCol.Header, *Theme.ROW_NONE)
        ctx.push_style_color(ImGuiCol.HeaderHovered, *Theme.SELECTION_BG)
        ctx.push_style_color(ImGuiCol.HeaderActive,  *Theme.SELECTION_BG)
        ctx.push_style_color(ImGuiCol.Text, *clr)

        # Cache first_line per entry to avoid split() every frame
        first_line = entry.get("_first_line")
        if first_line is None:
            msg = self._sanitize_text(entry["message"])
            first_line = msg.split('\n', 1)[0]
            entry["_first_line"] = first_line
        count = entry.get("count", 1)

        # Use unique ID suffix to avoid ImGui ID conflicts
        label = f"{first_line}##clog_{uid}_{idx}"
        if ctx.selectable(label, is_sel, 0, 0, 0):
            self._selected_vis_index = idx
        # Detect double-click on this row → open source file
        if ctx.is_item_hovered() and ctx.is_mouse_double_clicked(0):
            filepath, line = self._resolve_entry_source(entry)
            self._open_in_editor(filepath, line)

        # Collapse count badge on the right side
        if count > 1:
            ctx.same_line(ctx.get_content_region_avail_width() - 20)
            ctx.push_style_color(ImGuiCol.Text, *Theme.LOG_BADGE)
            ctx.label(f"{count}")
            ctx.pop_style_color(1)

        ctx.pop_style_color(4)

    @classmethod
    def _resolve_entry_source(cls, entry) -> tuple[str, int]:
        """Resolve the best file/line target for a console entry."""
        source_file = cls._sanitize_text(entry.get("source_file", ""))
        source_line = int(entry.get("source_line", 0) or 0)
        if source_file:
            return source_file, source_line

        stack_trace = cls._sanitize_text(entry.get("stack_trace", ""))
        if stack_trace:
            import re
            matches = re.findall(r'File\s+"([^"]+)",\s+line\s+(\d+)', stack_trace)
            if matches:
                file_path, line_text = matches[-1]
                try:
                    return file_path, int(line_text)
                except (ValueError, TypeError):
                    return file_path, 0

        return "", 0
    # ------ open in editor ------
    @staticmethod
    def _open_in_editor(filepath: str, line: int) -> None:
        """Open *filepath* at *line* in VS Code (or the default editor)."""
        import os
        if not filepath or not os.path.isfile(filepath):
            return
        from InfEngine.engine.project_context import get_project_root
        from InfEngine.engine.ui.project_utils import open_in_vscode, open_file_with_system
        project_root = get_project_root() or ""
        safe_path = os.path.abspath(filepath)
        if open_in_vscode(safe_path, line=line, project_root=project_root):
            return
        open_file_with_system(safe_path, project_root=project_root)

    # ------ color helper ------
    def _level_color(self, lv):
        if lv == "error":   return self._CLR_ERROR
        if lv == "warning": return self._CLR_WARN
        return self._CLR_INFO
