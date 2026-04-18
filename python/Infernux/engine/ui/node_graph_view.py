"""
NodeGraphView — Reusable ImGui canvas for rendering and interacting
with a :class:`~Infernux.core.node_graph.NodeGraph`.

Handles:
- Background grid (scales with zoom)
- Node rendering (header + body + pins) with shadows
- Curved connection lines (cubic bezier) with arrow heads
- Camera as graph-space view centre + zoom; pan is derived from the real canvas item rect
- Canvas panning (middle-mouse drag)
- Scroll-wheel zoom (centred on cursor)
- Node dragging
- Connection creation by dragging from pin to pin
- Node / link selection with hover highlight
- Right-click context menu
- Minimap in bottom-right corner
- Drop targets on the canvas
- Keyboard delete for selected nodes / links
- Callbacks for graph mutations
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from Infernux.core.node_graph import (
    GraphLink,
    GraphNode,
    NodeGraph,
    NodeTypeDef,
    PinDef,
    PinKind,
)
from Infernux.engine.ui.imgui_keys import KEY_C, KEY_DELETE, KEY_V, MOD_CTRL
from Infernux.engine.ui.theme import Theme

if TYPE_CHECKING:
    from Infernux.lib import InxGUIContext


# ═══════════════════════════════════════════════════════════════════════════
# Bezier helper
# ═══════════════════════════════════════════════════════════════════════════

def _bezier_points(
    x1: float, y1: float, x2: float, y2: float,
    segments: int = None,
) -> List[Tuple[float, float]]:
    if segments is None:
        segments = Theme.NODE_GRAPH_LINK_SEGMENTS
    dx = abs(x2 - x1) * 0.5
    dx = max(dx, 30.0)
    cx1, cy1 = x1 + dx, y1
    cx2, cy2 = x2 - dx, y2
    pts: List[Tuple[float, float]] = []
    for i in range(segments + 1):
        t = i / segments
        it = 1.0 - t
        px = it**3 * x1 + 3 * it**2 * t * cx1 + 3 * it * t**2 * cx2 + t**3 * x2
        py = it**3 * y1 + 3 * it**2 * t * cy1 + 3 * it * t**2 * cy2 + t**3 * y2
        pts.append((px, py))
    return pts


def _resolve_node_header_rgba(node: GraphNode, typedef: NodeTypeDef) -> Tuple[float, float, float, float]:
    """Return header tint: ``node.data[''header_color'']`` if valid, else ``typedef.header_color``."""
    raw = node.data.get("header_color")
    fb = typedef.header_color
    if isinstance(fb, (list, tuple)) and len(fb) >= 3:
        fr = float(fb[0])
        fg = float(fb[1])
        fb_ = float(fb[2])
        fa = float(fb[3]) if len(fb) > 3 else 1.0
    else:
        fr, fg, fb_, fa = 0.3, 0.3, 0.3, 1.0
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return fr, fg, fb_, max(0.0, min(1.0, fa))
    r = max(0.0, min(1.0, float(raw[0])))
    g = max(0.0, min(1.0, float(raw[1])))
    b = max(0.0, min(1.0, float(raw[2])))
    a = max(0.0, min(1.0, float(raw[3]))) if len(raw) > 3 else 1.0
    return r, g, b, a


# ═══════════════════════════════════════════════════════════════════════════
# Cached layout per node
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _PinLayout:
    pin_def: PinDef
    cx: float = 0.0
    cy: float = 0.0


@dataclass
class _NodeLayout:
    node: GraphNode
    typedef: NodeTypeDef
    sx: float = 0.0
    sy: float = 0.0
    w: float = 140.0
    h: float = 60.0
    input_pins: List[_PinLayout] = field(default_factory=list)
    output_pins: List[_PinLayout] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# View
# ═══════════════════════════════════════════════════════════════════════════

class NodeGraphView:
    """Stateful widget that renders a :class:`NodeGraph` onto an ImGui canvas."""

    def __init__(self) -> None:
        self.graph: Optional[NodeGraph] = None

        # Camera: (cam_center_gx, cam_center_gy) is the graph point kept at the
        # canvas item centre; pan_x/y are derived each frame from origin + item rect.
        self.cam_center_gx: float = 0.0
        self.cam_center_gy: float = 0.0
        self.zoom: float = 1.0
        self.pan_x: float = 50.0
        self.pan_y: float = 50.0
        self._cam_center_initialized: bool = False
        self._pending_camera: Optional[Tuple[float, float, float]] = None
        self._viewport_icx: float = 0.0
        self._viewport_icy: float = 0.0

        # Selection
        self.selected_nodes: List[str] = []
        self.selected_link: str = ""

        # In-progress connection drag
        self._dragging_pin: bool = False
        self._drag_src_node: str = ""
        self._drag_src_pin: str = ""
        self._drag_src_kind: PinKind = PinKind.OUTPUT
        self._drag_end_x: float = 0.0
        self._drag_end_y: float = 0.0

        # Node drag
        self._dragging_node: bool = False
        self._drag_node_id: str = ""

        # Canvas panning
        self._panning: bool = False

        # Canvas origin (screen coords)
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._canvas_w: float = 0.0
        self._canvas_h: float = 0.0
        # Last non-zero graph child size (for persistence when save_state runs outside render)
        self._last_graph_canvas_w: float = 0.0
        self._last_graph_canvas_h: float = 0.0
        # Screen rect of ``##canvas_bg`` (updated each frame after ``invisible_button``)
        self._canvas_item_min_x: float = 0.0
        self._canvas_item_max_x: float = 0.0
        self._canvas_item_min_y: float = 0.0
        self._canvas_item_max_y: float = 0.0
        # Cached layouts
        self._layouts: Dict[str, _NodeLayout] = {}

        # Hovered link uid (for highlight)
        self._hovered_link: str = ""

        # Hovered pin: (node_uid, pin_id, pin_kind) or empty
        self._hovered_pin: Tuple[str, str, PinKind] = ("", "", PinKind.OUTPUT)

        # ── Callbacks ─────────────────────────────────────────────────
        self.on_link_created: Optional[Callable[[str, str, str, str], None]] = None
        self.on_link_deleted: Optional[Callable[[str], None]] = None
        self.on_nodes_deleted: Optional[Callable[[List[str]], None]] = None
        self.on_node_add_request: Optional[Callable[[str, float, float], None]] = None
        self.on_node_selected: Optional[Callable[[str], None]] = None
        # Called immediately before user-driven selection changes (click), so editors can snapshot undo.
        self.on_before_selection_change: Optional[Callable[[], None]] = None
        # Drop handler: (payload_type, payload_str, graph_x, graph_y)
        self.on_canvas_drop: Optional[Callable[[str, str, float, float], None]] = None
        # If set, called on left-click on a node (after pins); return True to skip drag.
        self.on_node_primary_click: Optional[Callable[[str, float, float], bool]] = None
        self.on_node_drag_start: Optional[Callable[[str], None]] = None
        self.on_node_drag_end: Optional[Callable[[str], None]] = None
        self.on_copy: Optional[Callable[[], None]] = None
        self.on_paste: Optional[Callable[[], None]] = None
        # Header color swatch popup (see ``NodeTypeDef.header_color_swatch``).
        self.on_node_header_color_begin: Optional[Callable[[str], None]] = None
        self.on_node_header_color_end: Optional[Callable[[str], None]] = None
        self.on_node_header_color_changed: Optional[Callable[[str], None]] = None

        self._body_renderers: Dict[str, Callable] = {}
        self._hdr_color_popup_uid: str = ""
        self._hdr_popup_session_open: bool = False

    def _clamp_zoom(self, z: float) -> float:
        return max(Theme.NODE_GRAPH_ZOOM_MIN, min(Theme.NODE_GRAPH_ZOOM_MAX, float(z)))

    def reset_camera_defaults(self) -> None:
        """Reset camera for a blank graph (e.g. editor New)."""
        self.cam_center_gx = 0.0
        self.cam_center_gy = 0.0
        self.zoom = 1.0
        self.pan_x = 50.0
        self.pan_y = 50.0
        self._cam_center_initialized = False
        self._pending_camera = None

    def schedule_camera(self, gx: float, gy: float, zoom: float) -> None:
        """Queue graph-space centre + zoom; applied on next render when layout anchors exist."""
        self._pending_camera = (float(gx), float(gy), self._clamp_zoom(zoom))

    def set_legacy_pan_zoom(self, pan_x: float, pan_y: float, zoom: float) -> None:
        """Restore only legacy pan/zoom; graph centre is inferred on first layout."""
        self.pan_x = float(pan_x)
        self.pan_y = float(pan_y)
        self.zoom = self._clamp_zoom(zoom)
        self._cam_center_initialized = False
        self._pending_camera = None

    def _apply_pending_camera(self) -> None:
        if self._pending_camera is None:
            return
        gx, gy, z = self._pending_camera
        self._pending_camera = None
        self.cam_center_gx = gx
        self.cam_center_gy = gy
        self.zoom = z
        self._cam_center_initialized = True

    def _sync_pan_from_camera_anchor(self) -> None:
        if self._canvas_item_max_x <= self._canvas_item_min_x:
            return
        icx = (self._canvas_item_min_x + self._canvas_item_max_x) * 0.5
        icy = (self._canvas_item_min_y + self._canvas_item_max_y) * 0.5
        self._viewport_icx = icx
        self._viewport_icy = icy
        z = self._clamp_zoom(self.zoom)
        self.zoom = z
        if not self._cam_center_initialized:
            self.cam_center_gx = (icx - self._origin_x - self.pan_x) / z
            self.cam_center_gy = (icy - self._origin_y - self.pan_y) / z
            self._cam_center_initialized = True
        self.pan_x = icx - self._origin_x - self.cam_center_gx * z
        self.pan_y = icy - self._origin_y - self.cam_center_gy * z

    def _notify_before_selection_change(self) -> None:
        if self.on_before_selection_change:
            self.on_before_selection_change()

    # ── Public API ────────────────────────────────────────────────────

    def register_body_renderer(self, type_id: str, renderer: Callable) -> None:
        self._body_renderers[type_id] = renderer

    def get_layout(self, uid: str) -> Optional[_NodeLayout]:
        """Return cached screen-space layout for *uid*, or None."""
        return self._layouts.get(uid)

    def screen_to_graph(self, sx: float, sy: float) -> Tuple[float, float]:
        gx = (sx - self._origin_x - self.pan_x) / self.zoom
        gy = (sy - self._origin_y - self.pan_y) / self.zoom
        return gx, gy

    def graph_to_screen(self, gx: float, gy: float) -> Tuple[float, float]:
        sx = self._origin_x + gx * self.zoom + self.pan_x
        sy = self._origin_y + gy * self.zoom + self.pan_y
        return sx, sy

    def center_on_nodes(self) -> None:
        if not self.graph or not self.graph.nodes:
            return
        min_x = min(n.pos_x for n in self.graph.nodes)
        max_x = max(n.pos_x for n in self.graph.nodes)
        min_y = min(n.pos_y for n in self.graph.nodes)
        max_y = max(n.pos_y for n in self.graph.nodes)
        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5
        self.cam_center_gx = cx
        self.cam_center_gy = cy
        self._cam_center_initialized = True
        if self._canvas_item_max_x > self._canvas_item_min_x:
            self._sync_pan_from_camera_anchor()
        else:
            z = self._clamp_zoom(self.zoom)
            self.zoom = z
            self._cam_center_initialized = True
            self.pan_x = self._canvas_w * 0.5 - cx * z
            self.pan_y = self._canvas_h * 0.5 - cy * z

    def queue_camera_restore_graph_center(
        self,
        *,
        center_gx: float,
        center_gy: float,
        zoom: float,
    ) -> None:
        """Restore camera from graph-space viewport centre (stable across dock / resize)."""
        self.schedule_camera(center_gx, center_gy, zoom)

    def queue_camera_restore(
        self,
        *,
        pan_x: float,
        pan_y: float,
        zoom: float,
        ref_w: float,
        ref_h: float,
    ) -> None:
        """Legacy: convert saved pan + reference child size to graph-space centre."""
        rw = float(ref_w)
        rh = float(ref_h)
        px = float(pan_x)
        py = float(pan_y)
        z = self._clamp_zoom(zoom)
        if rw >= 1.0 and rh >= 1.0:
            gx = (rw * 0.5 - px) / z
            gy = (rh * 0.5 - py) / z
            self.schedule_camera(gx, gy, z)
        else:
            self.set_legacy_pan_zoom(px, py, z)

    # ── Main render ───────────────────────────────────────────────────

    def render(self, ctx: InxGUIContext) -> None:
        if self.graph is None:
            return

        canvas_w = ctx.get_content_region_avail_width()
        canvas_h = ctx.get_content_region_avail_height()
        if canvas_w < 1 or canvas_h < 1:
            return

        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        if canvas_w >= 1.0 and canvas_h >= 1.0:
            self._last_graph_canvas_w = canvas_w
            self._last_graph_canvas_h = canvas_h

        if not ctx.begin_child("##node_graph_canvas", canvas_w, canvas_h, False):
            ctx.end_child()
            return

        self._origin_x = ctx.get_window_pos_x()
        self._origin_y = ctx.get_window_pos_y()

        # Invisible button for mouse events
        ctx.set_cursor_pos_x(0)
        ctx.set_cursor_pos_y(0)
        ctx.invisible_button("##canvas_bg", canvas_w, canvas_h)
        self._canvas_item_min_x = ctx.get_item_rect_min_x()
        self._canvas_item_max_x = ctx.get_item_rect_max_x()
        self._canvas_item_min_y = ctx.get_item_rect_min_y()
        self._canvas_item_max_y = ctx.get_item_rect_max_y()
        self._apply_pending_camera()
        self._sync_pan_from_camera_anchor()
        canvas_hovered = ctx.is_item_hovered()

        # Clipping
        clip_x0 = self._origin_x
        clip_y0 = self._origin_y
        clip_x1 = self._origin_x + canvas_w
        clip_y1 = self._origin_y + canvas_h
        ctx.push_draw_list_clip_rect(clip_x0, clip_y0, clip_x1, clip_y1)

        # Background
        ctx.draw_filled_rect(clip_x0, clip_y0, clip_x1, clip_y1, *Theme.NODE_GRAPH_BG)

        # Grid
        self._draw_grid(ctx, clip_x0, clip_y0, clip_x1, clip_y1)

        # Compute node layouts
        self._compute_layouts()

        # Detect hovered pin (for highlight ring)
        mx_h = ctx.get_mouse_pos_x()
        my_h = ctx.get_mouse_pos_y()
        if self._dragging_pin:
            # During drag, highlight the nearest valid target pin
            self._hovered_pin = self._find_drag_target_pin(mx_h, my_h)
        else:
            h_node, h_pin, h_kind = self._hit_test_pin(mx_h, my_h)
            self._hovered_pin = (h_node, h_pin or "", h_kind)

        # Links (behind nodes)
        self._draw_links(ctx)

        # Nodes
        self._draw_nodes(ctx)

        # Pending connection line
        if self._dragging_pin:
            self._draw_pending_link(ctx)

        # Minimap
        self._draw_minimap(ctx, clip_x0, clip_y0, clip_x1, clip_y1)

        # Zoom indicator
        if abs(self.zoom - 1.0) > 0.01:
            zt = Theme.TEXT_DIM
            ctx.draw_text(
                clip_x0 + 8, clip_y1 - 22,
                f"{self.zoom * 100:.0f}%", zt[0], zt[1], zt[2], 0.8, 0.0,
            )

        ctx.pop_draw_list_clip_rect()

        # Handle interaction
        self._handle_interaction(ctx, canvas_hovered, canvas_w, canvas_h)
        self._draw_header_color_popup(ctx)

        # Drop targets on the canvas
        if ctx.begin_drag_drop_target():
            for dtype in ("ANIMCLIP_FILE", "ANIMFSM_FILE", "ASSET_FILE"):
                payload = ctx.accept_drag_drop_payload(dtype)
                if payload and self.on_canvas_drop:
                    mx = ctx.get_mouse_pos_x()
                    my = ctx.get_mouse_pos_y()
                    gx, gy = self.screen_to_graph(mx, my)
                    self.on_canvas_drop(dtype, payload, gx, gy)
                    break
            ctx.end_drag_drop_target()

        # Context menu
        if ctx.begin_popup_context_window("##node_graph_ctx", 1):
            self._draw_context_menu(ctx)
            ctx.end_popup()

        ctx.end_child()

    # ── Grid ──────────────────────────────────────────────────────────

    def _draw_grid(self, ctx, x0, y0, x1, y1):
        step = Theme.NODE_GRAPH_GRID_SIZE * self.zoom
        if step < 4.0:
            step = Theme.NODE_GRAPH_GRID_SIZE * 5 * self.zoom
        ox = self.pan_x % step
        oy = self.pan_y % step

        alpha = min(1.0, self.zoom)
        gc = Theme.NODE_GRAPH_GRID_COLOR
        col = (gc[0], gc[1], gc[2], gc[3] * alpha)
        x = x0 + ox
        while x < x1:
            ctx.draw_line(x, y0, x, y1, *col, 0.5)
            x += step
        y = y0 + oy
        while y < y1:
            ctx.draw_line(x0, y, x1, y, *col, 0.5)
            y += step

        big_step = step * 5
        big_ox = self.pan_x % big_step
        big_oy = self.pan_y % big_step
        x = x0 + big_ox
        while x < x1:
            ctx.draw_line(x, y0, x, y1, *Theme.NODE_GRAPH_GRID_COLOR_ALT, 1.0)
            x += big_step
        y = y0 + big_oy
        while y < y1:
            ctx.draw_line(x0, y, x1, y, *Theme.NODE_GRAPH_GRID_COLOR_ALT, 1.0)
            y += big_step

    # ── Layout ────────────────────────────────────────────────────────

    def _compute_layouts(self) -> None:
        self._layouts.clear()
        graph = self.graph
        if graph is None:
            return

        z = self.zoom
        for node in graph.nodes:
            typedef = graph.get_type(node.type_id)
            if typedef is None:
                continue

            in_pins = typedef.input_pins()
            out_pins = typedef.output_pins()
            max_pins = max(len(in_pins), len(out_pins), 1)

            w = typedef.min_width * z
            extra_pad = getattr(typedef, "body_bottom_pad", 0.0) or 0.0
            h = (
                Theme.NODE_GRAPH_NODE_HEADER_H
                + max_pins * Theme.NODE_GRAPH_NODE_PIN_ROW_H
                + Theme.NODE_GRAPH_NODE_BODY_MIN_H
                + extra_pad
            ) * z

            sx = self._origin_x + node.pos_x * z + self.pan_x
            sy = self._origin_y + node.pos_y * z + self.pan_y

            layout = _NodeLayout(node=node, typedef=typedef, sx=sx, sy=sy, w=w, h=h)

            hdr_h = Theme.NODE_GRAPH_NODE_HEADER_H * z
            row_h = Theme.NODE_GRAPH_NODE_PIN_ROW_H * z

            for i, pdef in enumerate(in_pins):
                cy = sy + hdr_h + i * row_h + row_h * 0.5
                layout.input_pins.append(_PinLayout(pin_def=pdef, cx=sx, cy=cy))

            for i, pdef in enumerate(out_pins):
                cy = sy + hdr_h + i * row_h + row_h * 0.5
                layout.output_pins.append(_PinLayout(pin_def=pdef, cx=sx + w, cy=cy))

            self._layouts[node.uid] = layout

    # ── Node drawing ──────────────────────────────────────────────────

    def _draw_nodes(self, ctx) -> None:
        for uid, layout in self._layouts.items():
            self._draw_one_node(ctx, layout)

    def _draw_one_node(self, ctx, layout: _NodeLayout) -> None:
        sx, sy, w, h = layout.sx, layout.sy, layout.w, layout.h
        z = self.zoom
        is_selected = layout.node.uid in self.selected_nodes
        rounding = Theme.NODE_GRAPH_NODE_ROUNDING * z
        hdr_h = Theme.NODE_GRAPH_NODE_HEADER_H * z
        pad_x = Theme.NODE_GRAPH_NODE_PAD_X * z

        # Shadow
        sh = 3.0 * z
        ctx.draw_filled_rect(
            sx + sh, sy + sh, sx + w + sh, sy + h + sh,
            *Theme.NODE_GRAPH_NODE_SHADOW, rounding,
        )

        # Body
        ctx.draw_filled_rect(sx, sy, sx + w, sy + h, *Theme.NODE_GRAPH_NODE_BODY, rounding)

        # Header
        hdr = _resolve_node_header_rgba(layout.node, layout.typedef)
        ctx.draw_filled_rect(sx, sy, sx + w, sy + hdr_h, *hdr, rounding)
        flat_h = min(rounding, hdr_h * 0.5)
        ctx.draw_filled_rect(sx, sy + hdr_h - flat_h, sx + w, sy + hdr_h, *hdr, 0)

        swatch_reserve = 0.0
        if layout.typedef.header_color_swatch:
            swatch_reserve = (
                Theme.NODE_GRAPH_HEADER_SWATCH_GAP * z
                + Theme.NODE_GRAPH_HEADER_SWATCH * z
            )

        # Header label
        label = layout.node.data.get("label", layout.typedef.label)
        font_sz = max(
            Theme.NODE_GRAPH_NODE_TITLE_FONT_MIN,
            Theme.NODE_GRAPH_NODE_TITLE_FONT_ZOOM_SCALE * z,
        )
        ctx.draw_text_aligned(
            sx + pad_x, sy, sx + w - pad_x - swatch_reserve, sy + hdr_h,
            label, *Theme.NODE_GRAPH_TEXT, 0.0, 0.5, font_sz,
        )

        if layout.typedef.header_color_swatch:
            gap = Theme.NODE_GRAPH_HEADER_SWATCH_GAP * z
            sw = Theme.NODE_GRAPH_HEADER_SWATCH * z
            ox2 = sx + w - pad_x - sw
            oy2 = sy + (hdr_h - sw) * 0.5
            hc = _resolve_node_header_rgba(layout.node, layout.typedef)
            rnd = max(1.0, 2.0 * z * 0.35)
            ctx.draw_filled_rect(ox2, oy2, ox2 + sw, oy2 + sw, hc[0], hc[1], hc[2], hc[3], rnd)
            bd = Theme.NODE_GRAPH_NODE_BORDER
            ctx.draw_rect(
                ox2, oy2, ox2 + sw, oy2 + sw,
                bd[0], bd[1], bd[2], bd[3],
                max(1.0, Theme.NODE_GRAPH_NODE_BORDER_THICKNESS * z), rnd,
            )

        # Subtitle (e.g. clip path)
        subtitle = layout.node.data.get("subtitle", "")
        if subtitle:
            body_top = sy + hdr_h + 2 * z
            sub_font = max(
                Theme.NODE_GRAPH_NODE_SUBTITLE_FONT_MIN,
                Theme.NODE_GRAPH_NODE_SUBTITLE_FONT_ZOOM_SCALE * z,
            )
            ctx.draw_text_aligned(
                sx + pad_x, body_top, sx + w - pad_x, body_top + 16 * z,
                subtitle, *Theme.NODE_GRAPH_TEXT_BODY, 0.0, 0.0, sub_font,
            )

        # Border
        if is_selected:
            ctx.draw_rect(sx, sy, sx + w, sy + h,
                          *Theme.APPLY_BUTTON, 2.5 * z, rounding)
        else:
            ctx.draw_rect(sx, sy, sx + w, sy + h,
                          *Theme.NODE_GRAPH_NODE_BORDER,
                          Theme.NODE_GRAPH_NODE_BORDER_THICKNESS * z, rounding)

        # Pins
        pin_r = Theme.NODE_GRAPH_PIN_RADIUS * z
        node_uid = layout.node.uid
        for pl in layout.input_pins:
            self._draw_pin(ctx, pl, PinKind.INPUT, pin_r, node_uid)
        for pl in layout.output_pins:
            self._draw_pin(ctx, pl, PinKind.OUTPUT, pin_r, node_uid)

        # Pin labels
        dim_font = max(
            Theme.NODE_GRAPH_NODE_PIN_FONT_MIN,
            Theme.NODE_GRAPH_NODE_PIN_FONT_ZOOM_SCALE * z,
        )
        plc = layout.typedef.pin_label_color
        if plc is not None and isinstance(plc, (list, tuple)) and len(plc) >= 3:
            pin_lbl = (
                float(plc[0]), float(plc[1]), float(plc[2]),
                float(plc[3]) if len(plc) > 3 else 1.0,
            )
        else:
            pin_lbl = Theme.NODE_GRAPH_TEXT_DIM
        row_h = Theme.NODE_GRAPH_NODE_PIN_ROW_H * z
        for pl in layout.input_pins:
            ctx.draw_text_aligned(
                pl.cx + pin_r + 4 * z, pl.cy - row_h * 0.5,
                pl.cx + w * 0.45, pl.cy + row_h * 0.5,
                pl.pin_def.label, pin_lbl[0], pin_lbl[1], pin_lbl[2], pin_lbl[3], 0.0, 0.5, dim_font,
            )
        for pl in layout.output_pins:
            ctx.draw_text_aligned(
                sx + w * 0.55, pl.cy - row_h * 0.5,
                pl.cx - pin_r - 4 * z, pl.cy + row_h * 0.5,
                pl.pin_def.label, pin_lbl[0], pin_lbl[1], pin_lbl[2], pin_lbl[3], 1.0, 0.5, dim_font,
            )

        # Custom body renderer
        renderer = self._body_renderers.get(layout.typedef.type_id)
        if renderer:
            body_y = (sy + hdr_h
                      + max(len(layout.input_pins), len(layout.output_pins)) * row_h)
            renderer(ctx, layout.node, sx + pad_x, body_y, w - pad_x * 2)

    def _draw_pin(self, ctx, pl: _PinLayout, kind: PinKind, radius: float,
                   node_uid: str = "") -> None:
        color = pl.pin_def.color
        connected = self._is_pin_connected(
            node_uid, pl.pin_def.id, kind == PinKind.OUTPUT
        )
        if connected:
            ctx.draw_filled_circle(pl.cx, pl.cy, radius, *color)
        else:
            ctx.draw_circle(pl.cx, pl.cy, radius, *color, 1.5 * self.zoom)
            ctx.draw_filled_circle(pl.cx, pl.cy, radius * 0.35,
                                   color[0], color[1], color[2], 0.3)
        # Hover / drag-target highlight ring
        hp_node, hp_pin, hp_kind = self._hovered_pin
        if (hp_pin and hp_pin == pl.pin_def.id and hp_kind == kind
                and hp_node == node_uid):
            ctx.draw_circle(pl.cx, pl.cy, radius + 3.0 * self.zoom,
                            *Theme.NODE_GRAPH_PIN_HOVER_RING, 1.8 * self.zoom)

    def _is_pin_connected(self, node_uid: str, pin_id: str, is_output: bool) -> bool:
        if self.graph is None:
            return False
        for lk in self.graph.links:
            if is_output and lk.source_node == node_uid and lk.source_pin == pin_id:
                return True
            if not is_output and lk.target_node == node_uid and lk.target_pin == pin_id:
                return True
        return False

    # ── Link drawing ──────────────────────────────────────────────────

    def _draw_links(self, ctx) -> None:
        if self.graph is None:
            return

        # Pre-compute hovered link
        mx = ctx.get_mouse_pos_x()
        my = ctx.get_mouse_pos_y()
        self._hovered_link = self._hit_test_link(mx, my)

        for lk in self.graph.links:
            src_l = self._layouts.get(lk.source_node)
            dst_l = self._layouts.get(lk.target_node)
            if src_l is None or dst_l is None:
                continue

            sx2, sy2 = self._find_pin_pos(src_l, lk.source_pin, PinKind.OUTPUT)
            ex2, ey2 = self._find_pin_pos(dst_l, lk.target_pin, PinKind.INPUT)
            if sx2 is None or ex2 is None:
                continue

            is_sel = lk.uid == self.selected_link
            is_hov = lk.uid == self._hovered_link

            if is_sel:
                color, thick = Theme.APPLY_BUTTON, 3.5 * self.zoom
            elif is_hov:
                color, thick = Theme.NODE_GRAPH_LINK_HOVER, 3.0 * self.zoom
            else:
                color, thick = (
                    Theme.NODE_GRAPH_LINK_DEFAULT,
                    Theme.NODE_GRAPH_LINK_THICKNESS * self.zoom,
                )

            self._draw_bezier(ctx, sx2, sy2, ex2, ey2, color, thick)

    def _draw_pending_link(self, ctx) -> None:
        src_l = self._layouts.get(self._drag_src_node)
        if src_l is None:
            return
        sx2, sy2 = self._find_pin_pos(src_l, self._drag_src_pin, self._drag_src_kind)
        if sx2 is None:
            return
        self._draw_bezier(
            ctx, sx2, sy2, self._drag_end_x, self._drag_end_y,
            Theme.NODE_GRAPH_LINK_PENDING, 2.0 * self.zoom,
        )

    def _draw_bezier(self, ctx, x1, y1, x2, y2, color, thickness):
        pts = _bezier_points(x1, y1, x2, y2)
        for i in range(len(pts) - 1):
            ctx.draw_line(
                pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                *color, thickness,
            )
        # Arrow head
        if len(pts) >= 2:
            ex, ey = pts[-1]
            px, py = pts[-2]
            angle = math.atan2(ey - py, ex - px)
            a_len = 8.0 * self.zoom
            for off in (-0.35, 0.35):
                ax = ex - a_len * math.cos(angle + off)
                ay = ey - a_len * math.sin(angle + off)
                ctx.draw_line(ex, ey, ax, ay, *color, thickness)

    def _header_swatch_screen_rect(self, layout: _NodeLayout) -> Optional[Tuple[float, float, float, float]]:
        if not layout.typedef.header_color_swatch:
            return None
        z = self.zoom
        sx, sy, w = layout.sx, layout.sy, layout.w
        hdr_h = Theme.NODE_GRAPH_NODE_HEADER_H * z
        pad_x = Theme.NODE_GRAPH_NODE_PAD_X * z
        sw = Theme.NODE_GRAPH_HEADER_SWATCH * z
        ox2 = sx + w - pad_x - sw
        oy2 = sy + (hdr_h - sw) * 0.5
        return ox2, oy2, ox2 + sw, oy2 + sw

    def _hit_header_swatch(self, mx: float, my: float) -> str:
        for uid in reversed(list(self._layouts)):
            layout = self._layouts[uid]
            r = self._header_swatch_screen_rect(layout)
            if r is None:
                continue
            x0, y0, x1, y1 = r
            if x0 <= mx <= x1 and y0 <= my <= y1:
                return uid
        return ""

    def _draw_header_color_popup(self, ctx) -> None:
        uid = self._hdr_color_popup_uid
        if not uid or self.graph is None:
            return
        nid = "##ng_hdr_color_" + uid
        opened = ctx.begin_popup(nid)
        if opened:
            node = self.graph.find_node(uid)
            td = self.graph.get_type(node.type_id) if node else None
            if node is None or td is None:
                ctx.end_popup()
                return
            if not self._hdr_popup_session_open:
                self._hdr_popup_session_open = True
                if self.on_node_header_color_begin:
                    self.on_node_header_color_begin(uid)
            hdr = _resolve_node_header_rgba(node, td)
            hb = td.header_color
            br = float(hb[0]) if isinstance(hb, (list, tuple)) and len(hb) > 0 else 0.3
            bg = float(hb[1]) if isinstance(hb, (list, tuple)) and len(hb) > 1 else 0.3
            bb = float(hb[2]) if isinstance(hb, (list, tuple)) and len(hb) > 2 else 0.3
            ba = float(hb[3]) if isinstance(hb, (list, tuple)) and len(hb) > 3 else 1.0
            nr, ng, nb, na = ctx.color_edit("##hdr_col_pick", hdr[0], hdr[1], hdr[2], hdr[3])
            if (nr, ng, nb, na) != hdr:
                if (abs(nr - br) < 1e-3 and abs(ng - bg) < 1e-3 and abs(nb - bb) < 1e-3
                        and abs(na - ba) < 1e-3):
                    node.data.pop("header_color", None)
                else:
                    node.data["header_color"] = [nr, ng, nb, na]
                if self.on_node_header_color_changed:
                    self.on_node_header_color_changed(uid)
            if ctx.button("Default##ng_hdr_def_btn"):
                node.data.pop("header_color", None)
                if self.on_node_header_color_changed:
                    self.on_node_header_color_changed(uid)
            ctx.end_popup()
        else:
            if self._hdr_popup_session_open:
                self._hdr_popup_session_open = False
                self._hdr_color_popup_uid = ""
                if self.on_node_header_color_end:
                    self.on_node_header_color_end(uid)
            # else: popup not yet visible this frame — keep uid

    def _find_pin_pos(self, layout, pin_id, kind):
        pins = layout.output_pins if kind == PinKind.OUTPUT else layout.input_pins
        for pl in pins:
            if pl.pin_def.id == pin_id:
                return pl.cx, pl.cy
        return None, None

    # ── Minimap ───────────────────────────────────────────────────────

    def _draw_minimap(self, ctx, cx0, cy0, cx1, cy1):
        if not self.graph or not self.graph.nodes:
            return

        # Compute graph-space bounding box of all nodes
        nodes = self.graph.nodes
        n_x0 = min(n.pos_x for n in nodes)
        n_x1 = max(n.pos_x + 160 for n in nodes)
        n_y0 = min(n.pos_y for n in nodes)
        n_y1 = max(n.pos_y + 80 for n in nodes)

        # Visible graph-space rect from actual canvas item corners (padding-safe).
        if self._canvas_item_max_x > self._canvas_item_min_x:
            v_gx0, v_gy0 = self.screen_to_graph(self._canvas_item_min_x, self._canvas_item_min_y)
            v_gx1, v_gy1 = self.screen_to_graph(self._canvas_item_max_x, self._canvas_item_max_y)
            if v_gx0 > v_gx1:
                v_gx0, v_gx1 = v_gx1, v_gx0
            if v_gy0 > v_gy1:
                v_gy0, v_gy1 = v_gy1, v_gy0
        else:
            z = max(self.zoom, 1e-6)
            v_gx0 = -self.pan_x / z
            v_gy0 = -self.pan_y / z
            v_gx1 = v_gx0 + self._canvas_w / z
            v_gy1 = v_gy0 + self._canvas_h / z

        # Union of nodes and viewport — this is the total extent we must show
        total_x0 = min(n_x0, v_gx0) - 20
        total_y0 = min(n_y0, v_gy0) - 20
        total_x1 = max(n_x1, v_gx1) + 20
        total_y1 = max(n_y1, v_gy1) + 20
        total_w = max(total_x1 - total_x0, 1.0)
        total_h = max(total_y1 - total_y0, 1.0)

        # Auto-size minimap to aspect ratio (capped)
        aspect = total_w / total_h
        mm_w = Theme.NODE_GRAPH_MINIMAP_SIZE
        mm_h = mm_w / max(aspect, 0.3)
        mm_h = min(mm_h, Theme.NODE_GRAPH_MINIMAP_SIZE)
        mm_w = min(mm_w, mm_h * aspect) if aspect < 0.3 else mm_w

        mm_x = cx1 - mm_w - Theme.NODE_GRAPH_MINIMAP_PAD
        mm_y = cy1 - mm_h - Theme.NODE_GRAPH_MINIMAP_PAD

        # Background
        ctx.draw_filled_rect(mm_x, mm_y, mm_x + mm_w, mm_y + mm_h,
                             *Theme.NODE_GRAPH_MINIMAP_BG, 4.0)

        # Clip minimap contents to its bounds
        ctx.push_draw_list_clip_rect(mm_x, mm_y, mm_x + mm_w, mm_y + mm_h)

        # Compute mapping: graph coords → minimap screen coords
        pad = 6.0
        inner_w = mm_w - pad * 2
        inner_h = mm_h - pad * 2
        sx = inner_w / total_w
        sy = inner_h / total_h
        s = min(sx, sy)
        off_x = mm_x + pad + (inner_w - total_w * s) * 0.5
        off_y = mm_y + pad + (inner_h - total_h * s) * 0.5

        # Draw node rectangles
        for n in nodes:
            nx = off_x + (n.pos_x - total_x0) * s
            ny = off_y + (n.pos_y - total_y0) * s
            nw = max(3, 120 * s)
            nh = max(2, 50 * s)
            ctx.draw_filled_rect(nx, ny, nx + nw, ny + nh, *Theme.NODE_GRAPH_MINIMAP_NODE, 1.0)

        # Draw viewport rectangle
        vx0 = off_x + (v_gx0 - total_x0) * s
        vy0 = off_y + (v_gy0 - total_y0) * s
        vx1 = off_x + (v_gx1 - total_x0) * s
        vy1 = off_y + (v_gy1 - total_y0) * s
        ab = Theme.APPLY_BUTTON
        ctx.draw_rect(vx0, vy0, vx1, vy1, ab[0], ab[1], ab[2], 0.45, 1.0, 2.0)

        ctx.pop_draw_list_clip_rect()

    # ── Interaction ───────────────────────────────────────────────────

    def _handle_interaction(self, ctx, canvas_hovered, canvas_w, canvas_h):
        mx = ctx.get_mouse_pos_x()
        my = ctx.get_mouse_pos_y()

        # Pin dragging
        if self._dragging_pin:
            self._drag_end_x = mx
            self._drag_end_y = my
            if not ctx.is_mouse_button_down(0):
                self._try_complete_link(mx, my)
                self._dragging_pin = False
            return

        # Node dragging (divide delta by zoom)
        if self._dragging_node:
            if ctx.is_mouse_button_down(0):
                dx = ctx.get_mouse_drag_delta_x(0)
                dy = ctx.get_mouse_drag_delta_y(0)
                node = self.graph.find_node(self._drag_node_id) if self.graph else None
                if node:
                    node.pos_x += dx / self.zoom
                    node.pos_y += dy / self.zoom
                ctx.reset_mouse_drag_delta(0)
            else:
                ended_id = self._drag_node_id
                self._dragging_node = False
                if self.on_node_drag_end:
                    self.on_node_drag_end(ended_id)
            return

        # Panning
        if self._panning:
            if ctx.is_mouse_button_down(2):
                dx = ctx.get_mouse_drag_delta_x(2)
                dy = ctx.get_mouse_drag_delta_y(2)
                z = self._clamp_zoom(self.zoom)
                self.zoom = z
                self.cam_center_gx -= dx / z
                self.cam_center_gy -= dy / z
                self._cam_center_initialized = True
                self._sync_pan_from_camera_anchor()
                ctx.reset_mouse_drag_delta(2)
            else:
                self._panning = False
            return

        # Delete key — remove selected nodes or link even when the cursor
        # is no longer hovering the canvas after selection.
        if ctx.is_key_pressed(KEY_DELETE):
            if self.selected_nodes:
                if self.on_nodes_deleted:
                    self.on_nodes_deleted(list(self.selected_nodes))
                elif self.graph:
                    for uid in list(self.selected_nodes):
                        self.graph.remove_node(uid)
                self.selected_nodes.clear()
                return
            if self.selected_link:
                link_uid = self.selected_link
                if self.on_link_deleted:
                    self.on_link_deleted(link_uid)
                elif self.graph:
                    self.graph.remove_link(link_uid)
                self.selected_link = ""
                return

        if canvas_hovered and ctx.is_key_down(MOD_CTRL):
            if ctx.is_key_pressed(KEY_C) and self.on_copy:
                self.on_copy()
            if ctx.is_key_pressed(KEY_V) and self.on_paste:
                self.on_paste()

        if not canvas_hovered:
            return

        # Zoom (scroll wheel): keep graph point under cursor fixed.
        wheel = ctx.get_mouse_wheel_delta()
        if abs(wheel) > 0.01:
            icx = self._viewport_icx
            icy = self._viewport_icy
            gx_m = (mx - self._origin_x - self.pan_x) / self.zoom
            gy_m = (my - self._origin_y - self.pan_y) / self.zoom
            old_z = self.zoom
            new_z = self._clamp_zoom(self.zoom + wheel * Theme.NODE_GRAPH_ZOOM_SPEED)
            if abs(new_z - old_z) >= 1e-9:
                self.zoom = new_z
                self.cam_center_gx = gx_m - (mx - icx) / new_z
                self.cam_center_gy = gy_m - (my - icy) / new_z
                self._cam_center_initialized = True
                self._sync_pan_from_camera_anchor()

        # Middle-mouse → panning
        if ctx.is_mouse_button_clicked(2):
            self._panning = True
            return

        # Right-click — select link under cursor for context menu
        if ctx.is_mouse_button_clicked(1):
            hit_lk = self._hit_test_link(mx, my)
            if hit_lk:
                self._notify_before_selection_change()
                self.selected_link = hit_lk
                self.selected_nodes.clear()
                if self.on_node_selected:
                    self.on_node_selected("")

        # Left click
        if ctx.is_mouse_button_clicked(0):
            sw_uid = self._hit_header_swatch(mx, my)
            if sw_uid:
                self._notify_before_selection_change()
                self.selected_nodes = [sw_uid]
                self.selected_link = ""
                if self.on_node_selected:
                    self.on_node_selected(sw_uid)
                self._hdr_color_popup_uid = sw_uid
                ctx.open_popup("##ng_hdr_color_" + sw_uid)
                return
            # Pins first
            hit_node, hit_pin, hit_kind = self._hit_test_pin(mx, my)
            if hit_pin is not None:
                # Drag-to-disconnect: if dragging from an input pin that
                # already has a link, detach and re-drag from the source end.
                if hit_kind == PinKind.INPUT and self.graph:
                    existing = self._find_link_to_input(hit_node, hit_pin)
                    if existing:
                        src_n, src_p = existing.source_node, existing.source_pin
                        if self.on_link_deleted:
                            self.on_link_deleted(existing.uid)
                        else:
                            self.graph.remove_link(existing.uid)
                        # Start a new drag from the output end
                        self._dragging_pin = True
                        self._drag_src_node = src_n
                        self._drag_src_pin = src_p
                        self._drag_src_kind = PinKind.OUTPUT
                        self._drag_end_x = mx
                        self._drag_end_y = my
                        return
                self._dragging_pin = True
                self._drag_src_node = hit_node
                self._drag_src_pin = hit_pin
                self._drag_src_kind = hit_kind
                self._drag_end_x = mx
                self._drag_end_y = my
                return

            # Nodes
            hit_uid = self._hit_test_node(mx, my)
            if hit_uid:
                self._notify_before_selection_change()
                self.selected_nodes = [hit_uid]
                self.selected_link = ""
                if self.on_node_selected:
                    self.on_node_selected(hit_uid)
                consumed = False
                if self.on_node_primary_click:
                    consumed = bool(self.on_node_primary_click(hit_uid, mx, my))
                if not consumed:
                    self._dragging_node = True
                    self._drag_node_id = hit_uid
                    if self.on_node_drag_start:
                        self.on_node_drag_start(hit_uid)
                return

            # Links
            hit_lk = self._hit_test_link(mx, my)
            if hit_lk:
                self._notify_before_selection_change()
                self.selected_link = hit_lk
                self.selected_nodes.clear()
                if self.on_node_selected:
                    self.on_node_selected("")
                return

            # Empty space — deselect
            self._notify_before_selection_change()
            self.selected_nodes.clear()
            self.selected_link = ""
            if self.on_node_selected:
                self.on_node_selected("")

    # ── Hit testing ───────────────────────────────────────────────────

    def _hit_test_pin(self, mx, my):
        hit_r = Theme.NODE_GRAPH_PIN_HIT_RADIUS * self.zoom
        for uid, layout in self._layouts.items():
            for pl in layout.output_pins:
                if _dist(mx, my, pl.cx, pl.cy) <= hit_r:
                    return uid, pl.pin_def.id, PinKind.OUTPUT
            for pl in layout.input_pins:
                if _dist(mx, my, pl.cx, pl.cy) <= hit_r:
                    return uid, pl.pin_def.id, PinKind.INPUT
        return "", None, PinKind.OUTPUT

    def _hit_test_node(self, mx, my):
        for uid in reversed(list(self._layouts)):
            layout = self._layouts[uid]
            if (layout.sx <= mx <= layout.sx + layout.w
                    and layout.sy <= my <= layout.sy + layout.h):
                return uid
        return ""

    def _find_drag_target_pin(self, mx, my):
        """Find the nearest valid target pin during a link drag."""
        hit_r = Theme.NODE_GRAPH_PIN_HIT_RADIUS * self.zoom
        want_kind = (PinKind.INPUT if self._drag_src_kind == PinKind.OUTPUT
                     else PinKind.OUTPUT)
        for uid, layout in self._layouts.items():
            if uid == self._drag_src_node:
                continue
            pins = (layout.input_pins if want_kind == PinKind.INPUT
                    else layout.output_pins)
            for pl in pins:
                if _dist(mx, my, pl.cx, pl.cy) <= hit_r:
                    return uid, pl.pin_def.id, want_kind
        return "", "", PinKind.OUTPUT

    def _hit_test_link(self, mx, my, threshold=6.0):
        if self.graph is None:
            return ""
        t_scaled = threshold * max(1.0, self.zoom)
        for lk in self.graph.links:
            src_l = self._layouts.get(lk.source_node)
            dst_l = self._layouts.get(lk.target_node)
            if not src_l or not dst_l:
                continue
            sx2, sy2 = self._find_pin_pos(src_l, lk.source_pin, PinKind.OUTPUT)
            ex2, ey2 = self._find_pin_pos(dst_l, lk.target_pin, PinKind.INPUT)
            if sx2 is None or ex2 is None:
                continue
            pts = _bezier_points(sx2, sy2, ex2, ey2, segments=12)
            for i in range(len(pts) - 1):
                if _point_segment_dist(mx, my, *pts[i], *pts[i + 1]) < t_scaled:
                    return lk.uid
        return ""

    def _find_link_to_input(self, node_uid: str, pin_id: str):
        """Find an existing link targeting the given input pin, or None."""
        if self.graph is None:
            return None
        for lk in self.graph.links:
            if lk.target_node == node_uid and lk.target_pin == pin_id:
                return lk
        return None

    def _try_complete_link(self, mx, my):
        target_node, target_pin, target_kind = self._hit_test_pin(mx, my)
        if target_pin is None:
            return
        if target_kind == self._drag_src_kind:
            return
        if target_node == self._drag_src_node:
            return
        if self._drag_src_kind == PinKind.OUTPUT:
            src_n, src_p = self._drag_src_node, self._drag_src_pin
            dst_n, dst_p = target_node, target_pin
        else:
            src_n, src_p = target_node, target_pin
            dst_n, dst_p = self._drag_src_node, self._drag_src_pin
        if self.on_link_created:
            self.on_link_created(src_n, src_p, dst_n, dst_p)
        elif self.graph:
            self.graph.add_link(src_n, src_p, dst_n, dst_p)

    # ── Context menu ──────────────────────────────────────────────────

    def _draw_context_menu(self, ctx) -> None:
        mx = ctx.get_mouse_pos_x()
        my = ctx.get_mouse_pos_y()
        gx, gy = self.screen_to_graph(mx, my)

        if self.graph is None:
            return

        # Add node sub-menu
        if ctx.begin_menu("Add Node"):
            for typedef in self.graph.registered_types():
                if ctx.menu_item(typedef.label, "", False, True):
                    if self.on_node_add_request:
                        self.on_node_add_request(typedef.type_id, gx, gy)
                    else:
                        self.graph.add_node(typedef.type_id, gx, gy)
            ctx.end_menu()

        # Delete selected nodes
        if self.selected_nodes:
            label = f"Delete Node ({len(self.selected_nodes)})"
            if ctx.menu_item(label, "", False, True):
                if self.on_nodes_deleted:
                    self.on_nodes_deleted(list(self.selected_nodes))
                else:
                    for uid in self.selected_nodes:
                        self.graph.remove_node(uid)
                self.selected_nodes.clear()

        # Delete selected link
        if self.selected_link:
            if ctx.menu_item("Delete Link", "", False, True):
                if self.on_link_deleted:
                    self.on_link_deleted(self.selected_link)
                else:
                    self.graph.remove_link(self.selected_link)
                self.selected_link = ""

        ctx.separator()
        if ctx.menu_item("Center View", "", False, True):
            self.center_on_nodes()
        if ctx.menu_item("Reset Zoom", "", False, True):
            self.zoom = 1.0
            self._cam_center_initialized = True
            self._sync_pan_from_camera_anchor()


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def _point_segment_dist(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-8:
        return _dist(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    return _dist(px, py, ax + t * dx, ay + t * dy)
