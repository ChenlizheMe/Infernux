"""
2D Animation Clip Editor — visual editor for creating and editing .animclip2d files.

Drag a sprite-sheet texture onto the panel to load it, then click frames to
build animation sequences.  Supports multiple clips per texture, live
preview playback, and direct save to .animclip2d files.

Opened from Window menu → 2D Animation Clip Editor.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Infernux.debug import Debug
from Infernux.engine.i18n import t
from Infernux.lib import InxGUIContext

from .editor_panel import EditorPanel
from .igui import IGUI
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol, ImGuiStyleVar


# ═══════════════════════════════════════════════════════════════════════════
# Internal state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _ClipState:
    """Editable state for one animation clip."""
    name: str = "NewClip"
    frame_indices: List[int] = field(default_factory=list)
    fps: float = 12.0
    saved_path: str = ""          # .animclip2d file path (empty = unsaved)


@dataclass
class _TextureState:
    """Cached sprite-sheet info for the currently loaded texture."""
    file_path: str = ""
    texture_id: int = 0
    tex_w: int = 0
    tex_h: int = 0
    frames: list = field(default_factory=list)   # List[SpriteFrame]
    guid: str = ""
    # Sampling state tracking — re-upload when these change
    filter_tag: str = ""
    srgb_tag: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Panel
# ═══════════════════════════════════════════════════════════════════════════

_PALETTE_THUMB_SIZE = 40.0   # frame slices in palette (bottom)
_SEQ_THUMB_SIZE = 36.0       # thumbnails in sequence strip
_THUMB_PAD = 3.0
_PREVIEW_MAX_SIZE = 320.0
_TABS_CARD_H = 44.0
_INFO_CARD_H = 44.0
_DETAILS_CARD_H = 124.0
_PREVIEW_CARD_H = 300.0
_SEQ_VISIBLE_ROWS = 2
_PALETTE_H = 180.0
_WIDE_LAYOUT_MIN_W = 920.0

_PLAYBACK_STOPPED = 0
_PLAYBACK_PLAYING = 1


@editor_panel(
    "2D Animation Clip Editor",
    type_id="animclip2d_editor",
    title_key="panel.animclip2d_editor",
    menu_path="Animation",
)
class AnimClip2DEditorPanel(EditorPanel):
    """Visual editor for building .animclip2d files from sprite sheets."""

    def __init__(self):
        super().__init__(title="2D Animation Clip Editor", window_id="animclip2d_editor")
        self._tex: Optional[_TextureState] = None
        self._clips: List[_ClipState] = [_ClipState()]
        self._active_clip_idx: int = 0
        # Preview playback
        self._playback: int = _PLAYBACK_STOPPED
        self._preview_frame_idx: int = 0  # index into active clip's frame_indices
        self._last_frame_time: float = 0.0
        # Cache name for texture cleanup
        self._cache_name: str = ""
        # Raw pixel data kept for re-upload on sampling changes
        self._raw_pixels: Optional[list] = None
        self._raw_w: int = 0
        self._raw_h: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initial_size(self):
        return (1080, 760)

    def on_enable(self):
        pass

    def on_disable(self):
        self._cleanup_texture()

    def save_state(self) -> dict:
        clips_data = []
        for c in self._clips:
            clips_data.append({
                "name": c.name, "frame_indices": c.frame_indices,
                "fps": c.fps, "saved_path": c.saved_path,
            })
        d: dict = {"active_clip": self._active_clip_idx, "clips": clips_data}
        if self._tex:
            d["texture_path"] = self._tex.file_path
        return d

    def load_state(self, data: dict):
        tex_path = data.get("texture_path", "")
        if tex_path and os.path.isfile(tex_path):
            self._load_texture(tex_path)
        clips_data = data.get("clips", [])
        if clips_data:
            self._clips = []
            for cd in clips_data:
                self._clips.append(_ClipState(
                    name=cd.get("name", "NewClip"),
                    frame_indices=list(cd.get("frame_indices", [])),
                    fps=float(cd.get("fps", 12.0)),
                    saved_path=cd.get("saved_path", ""),
                ))
        if not self._clips:
            self._clips = [_ClipState()]
        self._active_clip_idx = max(0, min(
            int(data.get("active_clip", 0)), len(self._clips) - 1))

    # ------------------------------------------------------------------
    # Render — layout: header -> tabs -> preview/details -> sequence -> palette
    # ------------------------------------------------------------------

    # ImGuiKey / ImGuiMod constants
    _IMGUI_MOD_CTRL = 1 << 12  # 4096
    _IMGUI_KEY_S = 564

    def on_render_content(self, ctx: InxGUIContext):
        # Ctrl+S save shortcut
        if ctx.is_key_down(self._IMGUI_MOD_CTRL) and ctx.is_key_pressed(self._IMGUI_KEY_S):
            clip = self._active_clip
            if clip is not None and self._tex is not None and len(clip.frame_indices) > 0:
                self._save_clip(clip)

        avail_w = ctx.get_content_region_avail_width()
        self._render_texture_slot(ctx, avail_w)
        ctx.dummy(0, 8)

        if self._tex is None or self._tex.texture_id == 0:
            self._render_empty_state(ctx)
            return

        # Guard against stale texture (e.g. import settings changed externally)
        if not self._validate_texture():
            self._render_empty_state(ctx)
            return

        try:
            clip = self._active_clip
            if clip is None:
                return

            self._render_main_workspace(ctx, clip, ctx.get_content_region_avail_width())

        except Exception as exc:
            Debug.log_warning(f"[AnimClipEditor] Render error: {exc}")

    # ------------------------------------------------------------------
    # Header / empty state helpers
    # ------------------------------------------------------------------

    def _render_texture_slot(self, ctx: InxGUIContext, avail_w: float):
        tex = self._tex
        display = os.path.basename(tex.file_path) if tex else t("animclip_editor.drop_texture_hint")

        IGUI.object_field(
            ctx, "animclip_tex_slot", display, "Texture",
            clickable=False,
            accept="TEXTURE_FILE",
            on_drop=self._on_texture_drop,
            on_clear=self._on_texture_clear if tex else None,
        )

        IGUI.drop_target(ctx, "ANIMCLIP_FILE", self._on_animclip_drop, outline=True)

    def _on_texture_clear(self):
        self._cleanup_texture()
        self._tex = None

    def _empty_state_hint(self) -> str:
        return t("animclip_editor.drop_texture_hint")

    def _empty_state_drop_types(self):
        return ["TEXTURE_FILE", "ANIMCLIP_FILE"]

    def _on_empty_state_drop(self, drop_type: str, payload):
        if drop_type == "TEXTURE_FILE":
            self._on_texture_drop(payload)
        elif drop_type == "ANIMCLIP_FILE":
            self._on_animclip_drop(payload)

    def _render_main_workspace(self, ctx: InxGUIContext, clip: _ClipState, avail_w: float):
        wide = avail_w >= _WIDE_LAYOUT_MIN_W

        # Name / save / basic info card — always on top, full width
        ctx.begin_child("##animclip_info_card", avail_w, _INFO_CARD_H, True)
        try:
            self._render_clip_info(ctx, clip, max(avail_w - 20.0, 120.0))
        finally:
            ctx.end_child()

        ctx.dummy(0, 8)

        if wide:
            seq_w = min(max(avail_w * 0.48, 340.0), 460.0)
            preview_w = max(avail_w - seq_w - 10.0, 260.0)

            ctx.begin_child("##animclip_sequence_card", seq_w, _PREVIEW_CARD_H, True)
            try:
                self._render_sequence_content(ctx, clip)
            finally:
                ctx.end_child()

            ctx.same_line(0, 10)

            ctx.begin_child("##animclip_preview_card", preview_w, _PREVIEW_CARD_H, True)
            try:
                self._render_preview(ctx, clip, preview_w)
            finally:
                ctx.end_child()
        else:
            ctx.begin_child("##animclip_preview_card", avail_w, _DETAILS_CARD_H, True)
            try:
                self._render_preview(ctx, clip, avail_w)
            finally:
                ctx.end_child()

            ctx.dummy(0, 8)

            ctx.begin_child("##animclip_sequence_card", avail_w, _PREVIEW_CARD_H, True)
            try:
                self._render_sequence_content(ctx, clip)
            finally:
                ctx.end_child()

        ctx.dummy(0, 8)
        self._render_frame_palette(ctx, avail_w)

    @staticmethod
    def _calc_grid_cols(width: float, thumb_size: float) -> int:
        cell_w = thumb_size + 16.0
        return max(1, int(max(width, thumb_size) / max(cell_w, 1.0)))

    # ------------------------------------------------------------------
    # Clip tabs
    # ------------------------------------------------------------------

    def _render_clip_tabs(self, ctx: InxGUIContext):
        ctx.begin_group()
        for i, clip in enumerate(self._clips):
            if i > 0:
                ctx.same_line(0, 2)
            is_active = (i == self._active_clip_idx)

            if is_active:
                ctx.push_style_color(ImGuiCol.Button, 0.25, 0.45, 0.7, 1.0)

            label = clip.name if clip.name else f"Clip {i}"
            if ctx.button(f"{label}##clip_tab_{i}"):
                self._active_clip_idx = i
                self._stop_playback()

            if is_active:
                ctx.pop_style_color(1)

        ctx.same_line(0, 8)
        if ctx.button(f"+##add_clip"):
            self._clips.append(_ClipState(name=f"Clip_{len(self._clips)}"))
            self._active_clip_idx = len(self._clips) - 1
            self._stop_playback()

        # Delete clip (only if more than 1)
        if len(self._clips) > 1:
            ctx.same_line(0, 8)
            ctx.push_style_color(ImGuiCol.Button, 0.6, 0.15, 0.15, 0.8)
            if ctx.button(t("animclip_editor.delete_clip")):
                idx = self._active_clip_idx
                self._clips.pop(idx)
                self._active_clip_idx = max(0, min(idx, len(self._clips) - 1))
                self._stop_playback()
            ctx.pop_style_color(1)

        ctx.end_group()

    # ------------------------------------------------------------------
    # Clip info — name, fps, save (compact top bar)
    # ------------------------------------------------------------------

    def _render_clip_info(self, ctx: InxGUIContext, clip: _ClipState, avail_w: float):
        # Row: Name + Save
        name_label = t("animclip_editor.clip_name")

        ctx.label(name_label)
        ctx.same_line(0, 4)
        ctx.set_next_item_width(min(140, avail_w * 0.2))
        new_name = ctx.text_input("##clip_name", clip.name, 256)
        if new_name != clip.name:
            clip.name = new_name
            # Notify FSM editor of clip name change
            if clip.saved_path:
                from .event_bus import EditorEventBus
                EditorEventBus.instance().emit("clip_name_changed", clip.saved_path, new_name)

        # Save button on same line as name
        ctx.same_line(0, 16)
        fc = len(clip.frame_indices)
        save_label = t("animclip_editor.save_clip")
        can_save = self._tex is not None and fc > 0
        if not can_save:
            ctx.begin_disabled(True)
        if ctx.button(save_label + "##info_save"):
            self._save_clip(clip)
        if not can_save:
            ctx.end_disabled()
        if clip.saved_path:
            ctx.same_line(0, 8)
            ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
            ctx.label(os.path.basename(clip.saved_path))
            ctx.pop_style_color(1)

    # ------------------------------------------------------------------
    # Preview — centered animated playback with transport controls
    # ------------------------------------------------------------------

    def _render_preview(self, ctx: InxGUIContext, clip: _ClipState, avail_w: float):
        tex = self._tex
        if tex is None:
            return

        ctx.label(t("animclip_editor.preview"))
        preview_frame = None
        preview_source_index = -1
        if clip.frame_indices:
            safe_idx = max(0, min(self._preview_frame_idx, len(clip.frame_indices) - 1))
            preview_source_index = clip.frame_indices[safe_idx]
            if 0 <= preview_source_index < len(tex.frames):
                preview_frame = tex.frames[preview_source_index]

        if preview_frame is not None:
            ctx.same_line(0, 8)
            ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
            ctx.label(f"#{preview_source_index}  {preview_frame.w}x{preview_frame.h}")
            ctx.pop_style_color(1)

        ctx.separator()

        # Transport controls row
        fc = len(clip.frame_indices)
        if not fc:
            ctx.begin_disabled(True)

        is_playing = self._playback == _PLAYBACK_PLAYING
        if is_playing:
            if ctx.button(t("animclip_editor.pause") + "##transport"):
                self._playback = _PLAYBACK_STOPPED
        else:
            if ctx.button(t("animclip_editor.play") + "##transport"):
                self._playback = _PLAYBACK_PLAYING
                self._last_frame_time = time.perf_counter()
                if self._preview_frame_idx >= fc:
                    self._preview_frame_idx = 0

        ctx.same_line(0, 4)
        if ctx.button(t("animclip_editor.stop") + "##transport"):
            self._stop_playback()

        ctx.same_line(0, 8)
        if ctx.button("|<##step_back"):
            self._preview_frame_idx = max(0, self._preview_frame_idx - 1)
        ctx.same_line(0, 2)
        if ctx.button(">|##step_fwd"):
            self._preview_frame_idx = min(
                max(0, fc - 1), self._preview_frame_idx + 1)

        if fc:
            ctx.same_line(0, 12)
            ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
            ctx.label(f"{self._preview_frame_idx + 1}/{fc}")
            ctx.pop_style_color(1)

        # FPS / duration — always editable
        fps_label = t("animclip_editor.clip_fps")
        ctx.same_line(0, 16)
        ctx.label(fps_label)
        ctx.same_line(0, 4)
        ctx.set_next_item_width(60)
        new_fps = ctx.drag_float("##clip_fps", clip.fps, 0.1, 0.1, 120.0)
        if new_fps != clip.fps:
            clip.fps = max(0.1, new_fps)

        dur = fc / max(clip.fps, 0.1)
        ctx.same_line(0, 8)
        ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
        ctx.label(f"{fc}f {dur:.2f}s")
        ctx.pop_style_color(1)

        if not fc:
            ctx.end_disabled()

        # Preview area
        ctx.begin_child("##preview_area", 0, 0, False)
        try:
            child_w = ctx.get_content_region_avail_width()
            child_h = ctx.get_content_region_avail_height()

            if not clip.frame_indices:
                ctx.dummy(0, child_h * 0.4)
                hint = t("animclip_editor.sequence_empty_hint")
                tw = ctx.calc_text_width(hint)
                ctx.set_cursor_pos_x(ctx.get_cursor_pos_x() + (child_w - tw) * 0.5)
                ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
                ctx.label(hint)
                ctx.pop_style_color(1)
            else:
                is_playing = self._playback == _PLAYBACK_PLAYING

                # Advance frame if playing
                if is_playing and clip.fps > 0:
                    now = time.perf_counter()
                    elapsed = now - self._last_frame_time
                    interval = 1.0 / clip.fps
                    if elapsed >= interval:
                        steps = int(elapsed / interval)
                        self._preview_frame_idx += steps
                        self._last_frame_time = now
                        if self._preview_frame_idx >= fc:
                            self._preview_frame_idx = self._preview_frame_idx % fc

                self._preview_frame_idx = max(0, min(self._preview_frame_idx, fc - 1))
                fidx = clip.frame_indices[self._preview_frame_idx]

                if 0 <= fidx < len(tex.frames):
                    frame = tex.frames[fidx]
                    uv0_x = frame.x / max(tex.tex_w, 1)
                    uv0_y = frame.y / max(tex.tex_h, 1)
                    uv1_x = (frame.x + frame.w) / max(tex.tex_w, 1)
                    uv1_y = (frame.y + frame.h) / max(tex.tex_h, 1)

                    # Fit preview into available space, centered
                    max_dim = max(8.0, min(_PREVIEW_MAX_SIZE, child_w - 24.0, child_h - 16.0))
                    aspect = frame.w / max(frame.h, 1)
                    if aspect >= 1.0:
                        pw = max_dim
                        ph = max_dim / aspect
                    else:
                        ph = max_dim
                        pw = max_dim * aspect

                    # Center horizontally and vertically
                    pad_x = (child_w - pw) * 0.5
                    pad_y = (child_h - ph) * 0.5
                    if pad_x > 0:
                        ctx.set_cursor_pos_x(ctx.get_cursor_pos_x() + pad_x)
                    if pad_y > 0:
                        ctx.set_cursor_pos_y(ctx.get_cursor_pos_y() + pad_y)

                    if tex.texture_id:
                        ctx.image(tex.texture_id, pw, ph, uv0_x, uv0_y, uv1_x, uv1_y)
        except Exception as exc:
            Debug.log_warning(f"[AnimClipEditor] Preview error: {exc}")
        finally:
            ctx.end_child()

    # ------------------------------------------------------------------
    # Sequence strip (ordered frame thumbnails)
    # ------------------------------------------------------------------

    def _render_sequence_content(self, ctx: InxGUIContext, clip: _ClipState):
        """Render frame sequence content (called inside a wrapping begin_child)."""
        tex = self._tex
        if tex is None or tex.texture_id == 0:
            return

        ctx.label(t("animclip_editor.sequence"))
        if clip.frame_indices:
            clear_label = t("animclip_editor.clear_sequence")
            clear_w = ctx.calc_text_width(clear_label) + 20.0
            ctx.same_line(max(ctx.get_window_width() - clear_w - 18.0, 180.0))
            if ctx.button(clear_label + "##seq_clear"):
                clip.frame_indices = []
                self._stop_playback()

        ctx.separator()
        ctx.begin_child("##seq_strip", 0, 0, False)
        try:
            if not clip.frame_indices:
                ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
                ctx.label(t("animclip_editor.sequence_empty_hint"))
                ctx.pop_style_color(1)
            else:
                thumb = _SEQ_THUMB_SIZE
                child_w = ctx.get_content_region_avail_width()
                cols = self._calc_grid_cols(child_w, thumb)
                to_remove = None

                if ctx.begin_table("##seq_grid", cols, 0, 0.0):
                    for seq_i, frame_idx in enumerate(clip.frame_indices):
                        ctx.table_next_column()

                        if 0 <= frame_idx < len(tex.frames):
                            frame = tex.frames[frame_idx]

                            uv0_x = frame.x / max(tex.tex_w, 1)
                            uv0_y = frame.y / max(tex.tex_h, 1)
                            uv1_x = (frame.x + frame.w) / max(tex.tex_w, 1)
                            uv1_y = (frame.y + frame.h) / max(tex.tex_h, 1)

                            # Aspect-preserving image inside square button
                            aspect = frame.w / max(frame.h, 1)
                            if aspect >= 1.0:
                                iw, ih = thumb, thumb / aspect
                            else:
                                iw, ih = thumb * aspect, thumb

                            is_preview = (
                                self._playback == _PLAYBACK_PLAYING
                                and seq_i == self._preview_frame_idx
                            )
                            if is_preview:
                                ctx.push_style_color(ImGuiCol.Button, 0.2, 0.6, 0.2, 0.8)
                                ctx.push_style_color(ImGuiCol.ButtonHovered, 0.3, 0.7, 0.3, 0.8)

                            save_x = ctx.get_cursor_pos_x()
                            save_y = ctx.get_cursor_pos_y()
                            ctx.button(f"##seq_{seq_i}", width=thumb, height=thumb)
                            hovered = ctx.is_item_hovered()

                            if is_preview:
                                ctx.pop_style_color(2)

                            bx = ctx.get_item_rect_min_x()
                            by = ctx.get_item_rect_min_y()

                            # Overlay centered image
                            ctx.set_cursor_pos_x(save_x + (thumb - iw) * 0.5)
                            ctx.set_cursor_pos_y(save_y + (thumb - ih) * 0.5)
                            ctx.image(tex.texture_id, iw, ih, uv0_x, uv0_y, uv1_x, uv1_y)

                            ctx.draw_text(bx + 2, by + 1, str(frame_idx), 1.0, 1.0, 1.0, 0.9)

                            if hovered and ctx.is_mouse_button_clicked(1):
                                to_remove = seq_i
                        else:
                            ctx.button(f"?##seq_{seq_i}", width=thumb, height=thumb)

                    ctx.end_table()

                if to_remove is not None:
                    clip.frame_indices.pop(to_remove)
        finally:
            ctx.end_child()

    # ------------------------------------------------------------------
    # Frame palette — grid of sprite slices at the bottom
    # ------------------------------------------------------------------

    def _render_frame_palette(self, ctx: InxGUIContext, avail_w: float):
        tex = self._tex
        if tex is None or tex.texture_id == 0:
            return

        palette_h = max(ctx.get_content_region_avail_height(), _PALETTE_H)
        ctx.begin_child("##animclip_palette_card", avail_w, palette_h, True)
        try:
            ctx.label(t("animclip_editor.frame_palette"))
            ctx.separator()

            ctx.begin_child("##frame_palette", 0, 0, False)
            try:
                active_clip = self._active_clip
                thumb = _PALETTE_THUMB_SIZE
                child_w = ctx.get_content_region_avail_width()
                cols = self._calc_grid_cols(child_w, thumb)

                if ctx.begin_table("##palette_grid", cols, 0, 0.0):
                    for i, frame in enumerate(tex.frames):
                        ctx.table_next_column()

                        uv0_x = frame.x / max(tex.tex_w, 1)
                        uv0_y = frame.y / max(tex.tex_h, 1)
                        uv1_x = (frame.x + frame.w) / max(tex.tex_w, 1)
                        uv1_y = (frame.y + frame.h) / max(tex.tex_h, 1)

                        # Aspect-preserving image inside square button
                        aspect = frame.w / max(frame.h, 1)
                        if aspect >= 1.0:
                            iw, ih = thumb, thumb / aspect
                        else:
                            iw, ih = thumb * aspect, thumb

                        in_clip = (
                            active_clip is not None
                            and i in active_clip.frame_indices
                        )
                        if in_clip:
                            ctx.push_style_color(ImGuiCol.Button, 0.2, 0.5, 0.8, 0.6)
                            ctx.push_style_color(ImGuiCol.ButtonHovered, 0.3, 0.6, 0.9, 0.7)

                        save_x = ctx.get_cursor_pos_x()
                        save_y = ctx.get_cursor_pos_y()
                        clicked = ctx.button(f"##palette_{i}", width=thumb, height=thumb)
                        hovered = ctx.is_item_hovered()

                        if in_clip:
                            ctx.pop_style_color(2)

                        bx = ctx.get_item_rect_min_x()
                        by = ctx.get_item_rect_min_y()

                        # Overlay centered image
                        ctx.set_cursor_pos_x(save_x + (thumb - iw) * 0.5)
                        ctx.set_cursor_pos_y(save_y + (thumb - ih) * 0.5)
                        ctx.image(tex.texture_id, iw, ih, uv0_x, uv0_y, uv1_x, uv1_y)

                        if clicked and active_clip is not None:
                            active_clip.frame_indices.append(i)

                        if hovered:
                            ctx.set_tooltip(f"#{i}  {frame.name}  ({frame.w}x{frame.h})")

                        ctx.draw_text(bx + 2, by + 1, str(i), 1.0, 1.0, 1.0, 0.9)

                    ctx.end_table()
            finally:
                ctx.end_child()

        except Exception as exc:
            Debug.log_warning(f"[AnimClipEditor] Frame palette error: {exc}")
        finally:
            ctx.end_child()

    # ------------------------------------------------------------------
    # Texture loading
    # ------------------------------------------------------------------

    def _on_texture_drop(self, payload):
        """Handle TEXTURE_FILE drop — payload is a file path."""
        if isinstance(payload, str) and payload:
            self._load_texture(payload)

    def _on_animclip_drop(self, payload):
        """Handle ANIMCLIP_FILE drop — open an existing .animclip2d."""
        if isinstance(payload, str) and payload:
            self._open_animclip(payload)

    def _load_texture(self, file_path: str):
        """Load a sprite-sheet texture and extract frame data from its .meta."""
        # Invalidate current texture state immediately so no stale
        # descriptor-set can be used if the re-upload below fails.
        old_cache_name = self._cache_name
        self._tex = None

        try:
            from Infernux.lib import TextureLoader
            td = TextureLoader.load_from_file(file_path)
            if not td or td.width <= 0 or not td.is_valid():
                Debug.log_warning(f"[AnimClipEditor] TextureLoader failed: {file_path}")
                self._cleanup_texture()
                return
        except Exception as exc:
            Debug.log_warning(f"[AnimClipEditor] TextureLoader exception: {exc}")
            self._cleanup_texture()
            return

        # Read import settings (filter_mode, srgb) for sampling-aware upload
        filter_tag = "default"
        srgb_tag = "linear"
        try:
            from Infernux.core.asset_types import read_texture_import_settings, FilterMode
            settings = read_texture_import_settings(file_path)
            cur_filter = getattr(settings, 'filter_mode', None)
            cur_srgb = getattr(settings, 'srgb', False)
            filter_tag = cur_filter.name if cur_filter else "default"
            srgb_tag = "srgb" if cur_srgb else "linear"
        except Exception:
            cur_filter = None
            cur_srgb = False

        # Store raw pixels for potential re-upload on settings change
        raw_pixels = td.get_pixels_list()
        raw_w, raw_h = td.width, td.height
        self._raw_pixels = raw_pixels
        self._raw_w = raw_w
        self._raw_h = raw_h

        # Upload to ImGui with sampling-aware processing
        try:
            from .editor_services import EditorServices
            svc = EditorServices.instance()
            native = svc.native_engine if svc else None
            if not native:
                Debug.log_warning("[AnimClipEditor] No native engine")
                self._cleanup_texture()
                return

            cache_name = (f"__animclip_editor__{srgb_tag}_{filter_tag}__"
                          f"{os.path.normpath(file_path)}")

            if native.has_imgui_texture(cache_name):
                texture_id = native.get_imgui_texture_id(cache_name)
            else:
                pixels = list(raw_pixels)
                w, h = raw_w, raw_h

                # Apply sRGB gamma curve for preview
                if cur_srgb:
                    import array
                    _LUT = [int(((i / 255.0) ** (1.0 / 2.2)) * 255.0 + 0.5)
                            for i in range(256)]
                    pixels = list(array.array('B', (
                        _LUT[pixels[j]] if (j % 4) != 3 else pixels[j]
                        for j in range(len(pixels))
                    )))

                # Determine if point (nearest) filtering is requested
                use_nearest = False
                try:
                    from Infernux.core.asset_types import FilterMode as FM
                    use_nearest = (cur_filter == FM.POINT)
                except Exception:
                    pass

                texture_id = native.upload_texture_for_imgui(
                    cache_name, pixels, w, h, nearest=use_nearest)

            if not texture_id:
                Debug.log_warning(f"[AnimClipEditor] upload_texture_for_imgui returned 0")
                self._cleanup_texture()
                return

            # New upload succeeded — now safe to remove the old texture.
            if old_cache_name and old_cache_name != cache_name:
                try:
                    if native.has_imgui_texture(old_cache_name):
                        native.remove_imgui_texture(old_cache_name)
                except Exception:
                    pass

            self._cache_name = cache_name
        except Exception as exc:
            Debug.log_warning(f"[AnimClipEditor] Upload exception: {exc}")
            self._cleanup_texture()
            return

        # Read sprite frames from .meta
        frames = []
        try:
            from Infernux.core.asset_types import read_texture_import_settings, SpriteFrame
            settings = read_texture_import_settings(file_path)
            if settings.sprite_frames:
                frames = list(settings.sprite_frames)
        except Exception:
            pass

        if not frames:
            from Infernux.core.asset_types import SpriteFrame
            frames = [SpriteFrame(name="frame_0", x=0, y=0, w=td.width, h=td.height)]

        # Resolve GUID
        guid = ""
        try:
            from Infernux.engine.bootstrap import EditorBootstrap
            adb = EditorBootstrap.instance().engine.get_asset_database()
            if adb:
                guid = adb.get_guid_from_path(file_path) or ""
        except Exception:
            pass

        self._tex = _TextureState(
            file_path=file_path,
            texture_id=texture_id,
            tex_w=td.width,
            tex_h=td.height,
            frames=frames,
            guid=guid,
            filter_tag=filter_tag,
            srgb_tag=srgb_tag,
        )

    def _cleanup_texture(self):
        """Release ImGui texture cache."""
        if self._cache_name:
            try:
                from .editor_services import EditorServices
                svc = EditorServices.instance()
                native = svc.native_engine if svc else None
                if native and native.has_imgui_texture(self._cache_name):
                    native.remove_imgui_texture(self._cache_name)
            except Exception:
                pass
            self._cache_name = ""
        self._raw_pixels = None
        self._raw_w = 0
        self._raw_h = 0

    # ------------------------------------------------------------------
    # Texture validation — detect stale handles & sampling changes
    # ------------------------------------------------------------------

    def _validate_texture(self) -> bool:
        """Check if the loaded texture is still valid; re-upload if needed.

        Returns False if the texture is irrecoverable (file gone, etc.).
        """
        tex = self._tex
        if tex is None:
            return False

        try:
            from .editor_services import EditorServices
            svc = EditorServices.instance()
            native = svc.native_engine if svc else None
            if native is None:
                return False
        except Exception:
            return False

        # Re-read current import settings to detect changes
        cur_filter_tag = tex.filter_tag
        cur_srgb_tag = tex.srgb_tag
        try:
            from Infernux.core.asset_types import read_texture_import_settings
            settings = read_texture_import_settings(tex.file_path)
            cur_filter = getattr(settings, 'filter_mode', None)
            cur_srgb = getattr(settings, 'srgb', False)
            cur_filter_tag = cur_filter.name if cur_filter else "default"
            cur_srgb_tag = "srgb" if cur_srgb else "linear"

            # Also refresh sprite frames
            if settings.sprite_frames:
                tex.frames = list(settings.sprite_frames)
            elif not tex.frames:
                from Infernux.core.asset_types import SpriteFrame
                tex.frames = [SpriteFrame(
                    name="frame_0", x=0, y=0, w=tex.tex_w, h=tex.tex_h)]
        except Exception:
            pass

        settings_changed = (
            cur_filter_tag != tex.filter_tag or cur_srgb_tag != tex.srgb_tag
        )
        texture_missing = (
            not self._cache_name
            or not native.has_imgui_texture(self._cache_name)
        )

        if settings_changed or texture_missing:
            # Save file_path before _load_texture (which sets self._tex = None)
            file_path = tex.file_path
            if self._raw_pixels and file_path:
                self._load_texture(file_path)
            elif file_path and os.path.isfile(file_path):
                self._load_texture(file_path)
            # After _load_texture, self._tex is either a new valid state or None
            return self._tex is not None and self._tex.texture_id != 0

        return tex.texture_id != 0

    # ------------------------------------------------------------------
    # Open existing .animclip2d
    # ------------------------------------------------------------------

    def _open_animclip(self, animclip_path: str):
        """Load an existing .animclip2d file into the editor."""
        from Infernux.core.animation_clip import AnimationClip
        clip_data = AnimationClip.load(animclip_path)
        if clip_data is None:
            Debug.log_warning(f"[AnimClipEditor] Failed to load: {animclip_path}")
            return

        # Resolve texture from clip's GUID or path
        tex_resolved = False
        if clip_data.authoring_texture_guid:
            try:
                from Infernux.engine.bootstrap import EditorBootstrap
                adb = EditorBootstrap.instance().engine.get_asset_database()
                if adb:
                    tex_path = adb.get_path_from_guid(clip_data.authoring_texture_guid)
                    if tex_path and os.path.isfile(tex_path):
                        self._load_texture(tex_path)
                        tex_resolved = True
            except Exception:
                pass

        if not tex_resolved and clip_data.authoring_texture_path:
            tp = clip_data.authoring_texture_path
            if os.path.isfile(tp):
                self._load_texture(tp)
            else:
                # Try relative to project root
                try:
                    from Infernux.engine.project_context import get_project_root
                    pr = get_project_root()
                    if pr:
                        abs_tp = os.path.join(pr, tp)
                        if os.path.isfile(abs_tp):
                            self._load_texture(abs_tp)
                except Exception:
                    pass

        # Import clip state
        cs = _ClipState(
            name=clip_data.name,
            frame_indices=list(clip_data.frame_indices),
            fps=clip_data.fps,
            saved_path=animclip_path,
        )
        self._clips = [cs]
        self._active_clip_idx = 0
        self._stop_playback()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_clip(self, clip: _ClipState):
        """Save the active clip as a .animclip2d file."""
        if clip.saved_path:
            # Already has a path — save directly
            self._do_save_clip(clip, clip.saved_path)
            return
        # New clip — always open Save As dialog
        self._show_save_as_dialog(clip)

    def _do_save_clip(self, clip: _ClipState, save_path: str):
        """Write the .animclip2d file to *save_path*."""
        from Infernux.core.animation_clip import AnimationClip
        tex = self._tex

        ac = AnimationClip(
            name=os.path.splitext(os.path.basename(save_path))[0],
            authoring_texture_guid=tex.guid if tex else "",
            authoring_texture_path=tex.file_path if tex else "",
            frame_indices=list(clip.frame_indices),
            fps=clip.fps,
            loop=True,
        )
        ac.file_path = save_path
        ok = ac.save()
        if ok:
            clip.saved_path = save_path
            Debug.log(f"[AnimClipEditor] Saved: {save_path}")
            try:
                from Infernux.core.assets import AssetManager
                AssetManager.reimport_asset(save_path)
            except Exception:
                pass
        else:
            Debug.log_warning(f"[AnimClipEditor] Failed to save: {save_path}")

    def _show_save_as_dialog(self, clip: _ClipState):
        """Open a native Save As file dialog on a background thread."""
        try:
            from Infernux.engine.project_context import get_project_root
            initial_dir = os.path.join(get_project_root() or ".", "Assets")
        except Exception:
            initial_dir = "."

        safe_name = clip.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        default_filename = f"{safe_name}.animclip2d"

        def _on_result(result_path):
            if result_path:
                self._do_save_clip(clip, result_path)

        def _run():
            result = None
            try:
                from ._dialogs import save_file_dialog
                result = save_file_dialog(
                    title="Save 2D Animation Clip",
                    win32_filter="2D AnimClip files (*.animclip2d)\0*.animclip2d\0All files (*.*)\0*.*\0\0",
                    initial_dir=initial_dir,
                    default_filename=default_filename,
                    default_ext="animclip2d",
                    tk_filetypes=[("2D AnimClip", "*.animclip2d"), ("All Files", "*.*")],
                )
            except Exception as exc:
                Debug.log_warning(f"[AnimClipEditor] Save dialog error: {exc}")
            _on_result(result)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------------

    def _stop_playback(self):
        self._playback = _PLAYBACK_STOPPED
        self._preview_frame_idx = 0
        self._last_frame_time = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _active_clip(self) -> Optional[_ClipState]:
        if 0 <= self._active_clip_idx < len(self._clips):
            return self._clips[self._active_clip_idx]
        return None
