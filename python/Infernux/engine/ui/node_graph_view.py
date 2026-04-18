"""
NodeGraphView — Reusable ImGui canvas for rendering and interacting
with a :class:`~Infernux.core.node_graph.NodeGraph`.

Handles:
- Background grid (scales with zoom)
- Node rendering (header + body + pins) with shadows
- Curved connection lines (cubic bezier) with arrow heads
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

if TYPE_CHECKING:
    from Infernux.lib import InxGUIContext


# ═══════════════════════════════════════════════════════════════════════════
# Visual constants (at zoom = 1.0)
# ═══════════════════════════════════════════════════════════════════════════

_GRID_SIZE = 20.0
_GRID_COLOR = (0.14, 0.14, 0.16, 1.0)
_GRID_COLOR2 = (0.19, 0.19, 0.22, 1.0)

_NODE_ROUNDING = 8.0
_NODE_BORDER_THICKNESS = 1.2
_NODE_HEADER_H = 28.0
_NODE_PIN_ROW_H = 24.0
_NODE_PAD_X = 10.0
_NODE_BODY_MIN_H = 8.0
_PIN_RADIUS = 5.5
_PIN_HIT_RADIUS = 12.0

_LINK_THICKNESS = 2.5
_LINK_SEGMENTS = 28

_BG_COLOR = (0.10, 0.10, 0.11, 1.0)
_NODE_BODY_COLOR = (0.18, 0.18, 0.20, 1.0)
_NODE_SHADOW_COLOR = (0.0, 0.0, 0.0, 0.35)
_NODE_SELECTED_BORDER = (0.95, 0.70, 0.15, 1.0)
_NODE_BORDER_COLOR = (0.30, 0.30, 0.34, 1.0)
_PIN_HOVER_COLOR = (1.0, 1.0, 1.0, 0.7)

_LINK_DEFAULT_COLOR = (0.60, 0.60, 0.62, 0.90)
_LINK_SELECTED_COLOR = (0.95, 0.70, 0.15, 1.0)
_LINK_HOVER_COLOR = (0.80, 0.80, 0.85, 1.0)
_PENDING_LINK_COLOR = (1.0, 1.0, 1.0, 0.45)

_TEXT_COLOR = (0.92, 0.92, 0.92, 1.0)
_TEXT_DIM_COLOR = (0.55, 0.55, 0.58, 1.0)
_TEXT_BODY_COLOR = (0.42, 0.42, 0.45, 1.0)

_ZOOM_MIN = 0.3
_ZOOM_MAX = 2.5
_ZOOM_SPEED = 0.08

_MINIMAP_SIZE = 120.0
_MINIMAP_PAD = 8.0
_MINIMAP_BG = (0.08, 0.08, 0.10, 0.7)
_MINIMAP_NODE = (0.45, 0.45, 0.50, 0.6)
_MINIMAP_VIEW = (0.95, 0.70, 0.15, 0.5)

# ImGuiKey constant for Delete
_KEY_DELETE = 522


# ═══════════════════════════════════════════════════════════════════════════
# Bezier helper
# ═══════════════════════════════════════════════════════════════════════════

def _bezier_points(
    x1: float, y1: float, x2: float, y2: float, segments: int = _LINK_SEGMENTS
) -> List[Tuple[float, float]]:
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

        # Camera
        self.pan_x: float = 50.0
        self.pan_y: float = 50.0
        self.zoom: float = 1.0

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
        # Drop handler: (payload_type, payload_str, graph_x, graph_y)
        self.on_canvas_drop: Optional[Callable[[str, str, float, float], None]] = None

        self._body_renderers: Dict[str, Callable] = {}

    # ── Public API ────────────────────────────────────────────────────

    def register_body_renderer(self, type_id: str, renderer: Callable) -> None:
        self._body_renderers[type_id] = renderer

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
        self.pan_x = self._canvas_w * 0.5 - cx * self.zoom
        self.pan_y = self._canvas_h * 0.5 - cy * self.zoom

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

        if not ctx.begin_child("##node_graph_canvas", canvas_w, canvas_h, False):
            ctx.end_child()
            return

        self._origin_x = ctx.get_window_pos_x()
        self._origin_y = ctx.get_window_pos_y()

        # Invisible button for mouse events
        ctx.set_cursor_pos_x(0)
        ctx.set_cursor_pos_y(0)
        ctx.invisible_button("##canvas_bg", canvas_w, canvas_h)
        canvas_hovered = ctx.is_item_hovered()

        # Clipping
        clip_x0 = self._origin_x
        clip_y0 = self._origin_y
        clip_x1 = self._origin_x + canvas_w
        clip_y1 = self._origin_y + canvas_h
        ctx.push_draw_list_clip_rect(clip_x0, clip_y0, clip_x1, clip_y1)

        # Background
        ctx.draw_filled_rect(clip_x0, clip_y0, clip_x1, clip_y1, *_BG_COLOR)

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
            ctx.draw_text(
                clip_x0 + 8, clip_y1 - 22,
                f"{self.zoom * 100:.0f}%", 0.55, 0.55, 0.58, 0.8, 0.0,
            )

        ctx.pop_draw_list_clip_rect()

        # Handle interaction
        self._handle_interaction(ctx, canvas_hovered, canvas_w, canvas_h)

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
        step = _GRID_SIZE * self.zoom
        if step < 4.0:
            step = _GRID_SIZE * 5 * self.zoom
        ox = self.pan_x % step
        oy = self.pan_y % step

        alpha = min(1.0, self.zoom)
        col = (_GRID_COLOR[0], _GRID_COLOR[1], _GRID_COLOR[2], _GRID_COLOR[3] * alpha)
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
            ctx.draw_line(x, y0, x, y1, *_GRID_COLOR2, 1.0)
            x += big_step
        y = y0 + big_oy
        while y < y1:
            ctx.draw_line(x0, y, x1, y, *_GRID_COLOR2, 1.0)
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
            h = (_NODE_HEADER_H + max_pins * _NODE_PIN_ROW_H + _NODE_BODY_MIN_H) * z

            sx = self._origin_x + node.pos_x * z + self.pan_x
            sy = self._origin_y + node.pos_y * z + self.pan_y

            layout = _NodeLayout(node=node, typedef=typedef, sx=sx, sy=sy, w=w, h=h)

            hdr_h = _NODE_HEADER_H * z
            row_h = _NODE_PIN_ROW_H * z

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
        rounding = _NODE_ROUNDING * z
        hdr_h = _NODE_HEADER_H * z
        pad_x = _NODE_PAD_X * z

        # Shadow
        sh = 3.0 * z
        ctx.draw_filled_rect(
            sx + sh, sy + sh, sx + w + sh, sy + h + sh,
            *_NODE_SHADOW_COLOR, rounding,
        )

        # Body
        ctx.draw_filled_rect(sx, sy, sx + w, sy + h, *_NODE_BODY_COLOR, rounding)

        # Header
        hdr = layout.typedef.header_color
        ctx.draw_filled_rect(sx, sy, sx + w, sy + hdr_h, *hdr, rounding)
        flat_h = min(rounding, hdr_h * 0.5)
        ctx.draw_filled_rect(sx, sy + hdr_h - flat_h, sx + w, sy + hdr_h, *hdr, 0)

        # Header label
        label = layout.node.data.get("label", layout.typedef.label)
        font_sz = max(11.0, 13.0 * z)
        ctx.draw_text_aligned(
            sx + pad_x, sy, sx + w - pad_x, sy + hdr_h,
            label, *_TEXT_COLOR, 0.0, 0.5, font_sz,
        )

        # Subtitle (e.g. clip path)
        subtitle = layout.node.data.get("subtitle", "")
        if subtitle:
            body_top = sy + hdr_h + 2 * z
            sub_font = max(9.0, 10.0 * z)
            ctx.draw_text_aligned(
                sx + pad_x, body_top, sx + w - pad_x, body_top + 16 * z,
                subtitle, *_TEXT_BODY_COLOR, 0.0, 0.0, sub_font,
            )

        # Border
        if is_selected:
            ctx.draw_rect(sx, sy, sx + w, sy + h,
                          *_NODE_SELECTED_BORDER, 2.5 * z, rounding)
        else:
            ctx.draw_rect(sx, sy, sx + w, sy + h,
                          *_NODE_BORDER_COLOR, _NODE_BORDER_THICKNESS * z, rounding)

        # Pins
        pin_r = _PIN_RADIUS * z
        node_uid = layout.node.uid
        for pl in layout.input_pins:
            self._draw_pin(ctx, pl, PinKind.INPUT, pin_r, node_uid)
        for pl in layout.output_pins:
            self._draw_pin(ctx, pl, PinKind.OUTPUT, pin_r, node_uid)

        # Pin labels
        dim_font = max(9.0, 10.5 * z)
        row_h = _NODE_PIN_ROW_H * z
        for pl in layout.input_pins:
            ctx.draw_text_aligned(
                pl.cx + pin_r + 4 * z, pl.cy - row_h * 0.5,
                pl.cx + w * 0.45, pl.cy + row_h * 0.5,
                pl.pin_def.label, *_TEXT_DIM_COLOR, 0.0, 0.5, dim_font,
            )
        for pl in layout.output_pins:
            ctx.draw_text_aligned(
                sx + w * 0.55, pl.cy - row_h * 0.5,
                pl.cx - pin_r - 4 * z, pl.cy + row_h * 0.5,
                pl.pin_def.label, *_TEXT_DIM_COLOR, 1.0, 0.5, dim_font,
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
        connected = self._is_pin_connected(pl.pin_def.id, kind == PinKind.OUTPUT)
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
                            *_PIN_HOVER_COLOR, 1.8 * self.zoom)

    def _is_pin_connected(self, pin_id: str, is_output: bool) -> bool:
        if self.graph is None:
            return False
        for lk in self.graph.links:
            if is_output and lk.source_pin == pin_id:
                return True
            if not is_output and lk.target_pin == pin_id:
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
                color, thick = _LINK_SELECTED_COLOR, 3.5 * self.zoom
            elif is_hov:
                color, thick = _LINK_HOVER_COLOR, 3.0 * self.zoom
            else:
                color, thick = _LINK_DEFAULT_COLOR, _LINK_THICKNESS * self.zoom

            cond = lk.data.get("condition", "")
            self._draw_bezier(ctx, sx2, sy2, ex2, ey2, color, thick, cond)

    def _draw_pending_link(self, ctx) -> None:
        src_l = self._layouts.get(self._drag_src_node)
        if src_l is None:
            return
        sx2, sy2 = self._find_pin_pos(src_l, self._drag_src_pin, self._drag_src_kind)
        if sx2 is None:
            return
        self._draw_bezier(
            ctx, sx2, sy2, self._drag_end_x, self._drag_end_y,
            _PENDING_LINK_COLOR, 2.0 * self.zoom,
        )

    def _draw_bezier(self, ctx, x1, y1, x2, y2, color, thickness, label=""):
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
        # Condition label at midpoint
        if label:
            mid = pts[len(pts) // 2]
            fsz = max(9.0, 10.0 * self.zoom)
            ctx.draw_text(
                mid[0] + 4, mid[1] - 10 * self.zoom,
                label, 0.75, 0.75, 0.55, 0.9, fsz,
            )

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

        # Compute visible graph-space viewport
        v_gx0 = -self.pan_x / self.zoom
        v_gy0 = -self.pan_y / self.zoom
        v_gx1 = v_gx0 + self._canvas_w / self.zoom
        v_gy1 = v_gy0 + self._canvas_h / self.zoom

        # Union of nodes and viewport — this is the total extent we must show
        total_x0 = min(n_x0, v_gx0) - 20
        total_y0 = min(n_y0, v_gy0) - 20
        total_x1 = max(n_x1, v_gx1) + 20
        total_y1 = max(n_y1, v_gy1) + 20
        total_w = max(total_x1 - total_x0, 1.0)
        total_h = max(total_y1 - total_y0, 1.0)

        # Auto-size minimap to aspect ratio (capped)
        aspect = total_w / total_h
        mm_w = _MINIMAP_SIZE
        mm_h = mm_w / max(aspect, 0.3)
        mm_h = min(mm_h, _MINIMAP_SIZE)
        mm_w = min(mm_w, mm_h * aspect) if aspect < 0.3 else mm_w

        mm_x = cx1 - mm_w - _MINIMAP_PAD
        mm_y = cy1 - mm_h - _MINIMAP_PAD

        # Background
        ctx.draw_filled_rect(mm_x, mm_y, mm_x + mm_w, mm_y + mm_h,
                             *_MINIMAP_BG, 4.0)

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
            ctx.draw_filled_rect(nx, ny, nx + nw, ny + nh, *_MINIMAP_NODE, 1.0)

        # Draw viewport rectangle
        vx0 = off_x + (v_gx0 - total_x0) * s
        vy0 = off_y + (v_gy0 - total_y0) * s
        vx1 = off_x + (v_gx1 - total_x0) * s
        vy1 = off_y + (v_gy1 - total_y0) * s
        ctx.draw_rect(vx0, vy0, vx1, vy1, *_MINIMAP_VIEW, 1.0, 2.0)

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
                self._dragging_node = False
            return

        # Panning
        if self._panning:
            if ctx.is_mouse_button_down(2):
                dx = ctx.get_mouse_drag_delta_x(2)
                dy = ctx.get_mouse_drag_delta_y(2)
                self.pan_x += dx
                self.pan_y += dy
                ctx.reset_mouse_drag_delta(2)
            else:
                self._panning = False
            return

        # Delete key — remove selected nodes or link even when the cursor
        # is no longer hovering the canvas after selection.
        if ctx.is_key_pressed(_KEY_DELETE):
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

        if not canvas_hovered:
            return

        # Zoom (scroll wheel, centred on cursor)
        wheel = ctx.get_mouse_wheel_delta()
        if abs(wheel) > 0.01:
            old_zoom = self.zoom
            self.zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self.zoom + wheel * _ZOOM_SPEED))
            ratio = self.zoom / old_zoom
            self.pan_x = mx - self._origin_x - (mx - self._origin_x - self.pan_x) * ratio
            self.pan_y = my - self._origin_y - (my - self._origin_y - self.pan_y) * ratio

        # Middle-mouse → panning
        if ctx.is_mouse_button_clicked(2):
            self._panning = True
            return

        # Right-click — select link under cursor for context menu
        if ctx.is_mouse_button_clicked(1):
            hit_lk = self._hit_test_link(mx, my)
            if hit_lk:
                self.selected_link = hit_lk
                self.selected_nodes.clear()
                if self.on_node_selected:
                    self.on_node_selected("")

        # Left click
        if ctx.is_mouse_button_clicked(0):
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
                self.selected_nodes = [hit_uid]
                self.selected_link = ""
                self._dragging_node = True
                self._drag_node_id = hit_uid
                if self.on_node_selected:
                    self.on_node_selected(hit_uid)
                return

            # Links
            hit_lk = self._hit_test_link(mx, my)
            if hit_lk:
                self.selected_link = hit_lk
                self.selected_nodes.clear()
                if self.on_node_selected:
                    self.on_node_selected("")
                return

            # Empty space — deselect
            self.selected_nodes.clear()
            self.selected_link = ""
            if self.on_node_selected:
                self.on_node_selected("")

    # ── Hit testing ───────────────────────────────────────────────────

    def _hit_test_pin(self, mx, my):
        hit_r = _PIN_HIT_RADIUS * self.zoom
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
        hit_r = _PIN_HIT_RADIUS * self.zoom
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
