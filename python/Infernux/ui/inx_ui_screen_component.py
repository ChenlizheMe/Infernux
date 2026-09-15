"""InxUIScreenComponent — base for 2D screen-space UI elements.

Provides anchor-aware position, size, and appearance data for screen UI.

Hierarchy:
    InxComponent → InxUIComponent → InxUIScreenComponent
"""

import math

from Infernux.components import serialized_field
from Infernux.components.fields import FieldType
from .inx_ui_component import InxUIComponent
from .enums import ScreenAlignH, ScreenAlignV, UILayoutPosition, UILayoutSizing
from .ui_render_revision import is_unchanged_ui_scalar, mark_runtime_ui_dirty

# Rect cache — avoids repeated hierarchy walks and serialized-field reads.
# Runtime callers retain it until hierarchy or geometry changes; editor tools
# may still provide a unique token when they need frame-local invalidation.
_rect_cache: dict = {}
_rect_cache_frame = None
_rect_cache_revision = 0

# Canvas-free UI is authored in logical pixels and converted by one engine
# convention. This is not an authored surface or a per-object setting.
WORLD_UI_PIXELS_PER_UNIT = 100.0


def clear_rect_cache(frame_id=0) -> None:
    """Clear cached rectangles when the caller's invalidation token changes."""
    global _rect_cache_frame
    if frame_id != _rect_cache_frame:
        _rect_cache.clear()
        _rect_cache_frame = frame_id


def _invalidate_rect_cache() -> None:
    """Discard cached hierarchy rectangles after a geometry mutation."""
    global _rect_cache_revision
    _rect_cache.clear()
    _rect_cache_revision += 1


def _get_layout_revision() -> int:
    """Include measured text sizes in consumers of the resolved UI geometry."""
    return _rect_cache_revision


class InxUIScreenComponent(InxUIComponent):
    """2D screen-space UI element with a canvas-pixel rectangle.

    Attributes:
        x: Horizontal position in canvas pixels (from canvas left edge).
        y: Vertical position in canvas pixels (from canvas top edge).
        width: Width in canvas pixels (unrotated content size).
        height: Height in canvas pixels (unrotated content size).
        rotation: Visual rotation in degrees (any angle).

    Position and rotation are owned by the GameObject Transform.  ``x``,
    ``y`` and ``rotation`` remain hidden only so 0.4.0 scenes can migrate
    without a second runtime authority.
    """

    _hide_transform_: bool = False
    _GEOMETRY_FIELDS = frozenset({
        "align_h", "align_v", "x", "y", "width", "height", "rotation",
        "layout_position", "width_sizing", "height_sizing", "min_width",
        "min_height", "max_width", "max_height", "layout_weight",
    })
    _HIT_POLICY_FIELDS = frozenset({"raycast_target", "clip_content"})

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        unchanged = is_unchanged_ui_scalar(self, name, value)
        super().__setattr__(name, value)
        if name in {"x", "y", "rotation"} and not getattr(
            self, "_syncing_legacy_layout", False
        ):
            self._publish_legacy_layout_value_to_transform(name)
        if unchanged:
            return
        mark_runtime_ui_dirty(self, binding=name in ("material", "text_material", "texture"))
        if name in self._GEOMETRY_FIELDS:
            _invalidate_rect_cache()
        if name in self._HIT_POLICY_FIELDS:
            from .ui_render_revision import _mark_hit_policy_dirty

            _mark_hit_policy_dirty()

    align_h: ScreenAlignH = serialized_field(default=ScreenAlignH.Left, tooltip="Horizontal anchor", group="Position")
    align_v: ScreenAlignV = serialized_field(default=ScreenAlignV.Top, tooltip="Vertical anchor", group="Position")
    # Hidden compatibility storage for 0.4.0 documents. Geometry reads the
    # GameObject Transform as its sole position/rotation authority.
    x: float = serialized_field(default=0.0, hidden=True)
    y: float = serialized_field(default=0.0, hidden=True)
    rotation: float = serialized_field(default=0.0, hidden=True)
    mirror_x: bool = serialized_field(default=False, tooltip="Mirror horizontally", group="Position")
    mirror_y: bool = serialized_field(default=False, tooltip="Mirror vertically", group="Position")

    width: float = serialized_field(default=160.0, tooltip="Width in canvas pixels", group="Layout")
    height: float = serialized_field(default=40.0, tooltip="Height in canvas pixels", group="Layout")
    lock_aspect_ratio: bool = serialized_field(default=False, tooltip="Preserve width/height ratio while resizing", group="Layout")
    layout_position: UILayoutPosition = serialized_field(
        default=UILayoutPosition.Flow, tooltip="Participate in parent auto layout or use x/y", group="Layout"
    )
    width_sizing: UILayoutSizing = serialized_field(
        default=UILayoutSizing.Fixed, tooltip="Fixed, content-hugging, or fill width", group="Layout"
    )
    height_sizing: UILayoutSizing = serialized_field(
        default=UILayoutSizing.Fixed, tooltip="Fixed, content-hugging, or fill height", group="Layout"
    )
    min_width: float = serialized_field(default=0.0, range=(0.0, 100000.0), group="Layout")
    min_height: float = serialized_field(default=0.0, range=(0.0, 100000.0), group="Layout")
    max_width: float = serialized_field(default=0.0, range=(0.0, 100000.0), tooltip="0 means unlimited", group="Layout")
    max_height: float = serialized_field(default=0.0, range=(0.0, 100000.0), tooltip="0 means unlimited", group="Layout")
    layout_weight: float = serialized_field(default=1.0, range=(0.001, 1000.0), group="Layout")
    opacity: float = serialized_field(default=1.0, range=(0.0, 1.0), tooltip="Element opacity", group="Appearance", slider=True)
    corner_radius: float = serialized_field(default=0.0, range=(0.0, 1000.0), tooltip="Corner radius in canvas pixels", group="Appearance")
    material = serialized_field(
        default=None,
        field_type=FieldType.MATERIAL,
        tooltip="Material used by this UI element; empty uses the engine UI material",
        group="Appearance",
    )

    # ── Interaction ──
    raycast_target: bool = serialized_field(default=True, tooltip="Receive pointer events", group="Interaction")

    def _set_game_object(self, game_object):
        previous = self.__dict__.get("_game_object")
        super()._set_game_object(game_object)
        if game_object is not None and previous is not game_object:
            self._publish_legacy_layout_value_to_transform("layout")

    def _layout_reference(self):
        """Return the authored reference and parent rect for this UI element."""
        canvas = self.get_canvas()
        if canvas is not None:
            width = max(1.0, float(canvas.reference_width))
            height = max(1.0, float(canvas.reference_height))
            return width, height, self._get_parent_world_rect(width, height), 1.0
        parent = self._get_parent_ui_component()
        if parent is None:
            width = max(1.0, float(self.width))
            height = max(1.0, float(self.height))
        else:
            width = max(1.0, float(parent.width))
            height = max(1.0, float(parent.height))
        return width, height, (0.0, 0.0, width, height), WORLD_UI_PIXELS_PER_UNIT

    def _publish_legacy_layout_value_to_transform(self, changed_name: str) -> None:
        game_object = self._try_get_game_object()
        if game_object is None or self.is_world_space():
            return
        from Infernux.lib import Vector3

        transform = game_object.transform
        if changed_name in {"x", "y", "layout"}:
            _cw, _ch, (px, py, pw, ph), units = self._layout_reference()
            anchor_x, anchor_y = self._anchor_origin(pw, ph)
            center_x = px + anchor_x + float(self.x) + float(self.width) * 0.5
            center_y = py + anchor_y + float(self.y) + float(self.height) * 0.5
            position = transform.local_position
            transform.local_position = Vector3(
                (center_x - (px + pw * 0.5)) / units,
                -((center_y - (py + ph * 0.5)) / units),
                0.0 if changed_name == "layout" else float(position.z),
            )
        if changed_name in {"rotation", "layout"}:
            angles = transform.local_euler_angles
            transform.local_euler_angles = Vector3(
                float(angles.x), float(angles.y), float(self.rotation)
            )

    def _layout_offset(self, reference_width=None, reference_height=None) -> tuple[float, float]:
        game_object = self._try_get_game_object()
        if game_object is None:
            return float(self.x), float(self.y)
        if self.is_world_space() and self._get_parent_ui_component() is None:
            return 0.0, 0.0
        position = game_object.transform.local_position
        units = 1.0
        if self.is_world_space():
            units = WORLD_UI_PIXELS_PER_UNIT
        ref_width = float(reference_width if reference_width is not None else self.width)
        ref_height = float(reference_height if reference_height is not None else self.height)
        anchor_x, anchor_y = self._anchor_origin(ref_width, ref_height)
        return (
            ref_width * 0.5 + float(position.x) * units - float(self.width) * 0.5 - anchor_x,
            ref_height * 0.5 - float(position.y) * units - float(self.height) * 0.5 - anchor_y,
        )

    def _set_layout_rect_origin(self, rect_x: float, rect_y: float,
                                canvas_width: float, canvas_height: float) -> None:
        """Move the authoritative Transform so the UI rect starts at x/y."""
        game_object = self._try_get_game_object()
        if game_object is None or (
            self.is_world_space() and self._get_parent_ui_component() is None
        ):
            self._sync_legacy_layout(rect_x, rect_y, 0.0, 0.0)
            return
        px, py, pw, ph = self._get_parent_world_rect(canvas_width, canvas_height)
        game_object.transform.local_position = self._local_position_for_rect_origin(
            rect_x, rect_y, canvas_width, canvas_height
        )
        anchor_x, anchor_y = self._anchor_origin(pw, ph)
        self._sync_legacy_layout(
            float(rect_x) - px - anchor_x,
            float(rect_y) - py - anchor_y,
            px,
            py,
        )
        _invalidate_rect_cache()
        mark_runtime_ui_dirty()

    def _local_position_for_rect_origin(self, rect_x: float, rect_y: float,
                                        canvas_width: float, canvas_height: float):
        """Resolve a layout rect origin into the one native Transform position."""
        game_object = self._try_get_game_object()
        if game_object is None or (
            self.is_world_space() and self._get_parent_ui_component() is None
        ):
            return None
        from Infernux.lib import Vector3

        px, py, pw, ph = self._get_parent_world_rect(canvas_width, canvas_height)
        units = WORLD_UI_PIXELS_PER_UNIT if self.is_world_space() else 1.0
        center_x = float(rect_x) + float(self.width) * 0.5
        center_y = float(rect_y) + float(self.height) * 0.5
        position = game_object.transform.local_position
        return Vector3(
            (center_x - (px + pw * 0.5)) / units,
            -((center_y - (py + ph * 0.5)) / units),
            float(position.z),
        )

    def _local_position_for_visual_origin(self, visual_x: float, visual_y: float,
                                          canvas_width: float, canvas_height: float):
        """Resolve a rotated visual origin into the native Transform position."""
        width, height = float(self.width), float(self.height)
        visual_width, visual_height = self.calc_visual_size(width, height)
        rect_x = float(visual_x) + (visual_width - width) * 0.5
        rect_y = float(visual_y) + (visual_height - height) * 0.5
        return self._local_position_for_rect_origin(
            rect_x, rect_y, canvas_width, canvas_height
        )

    def _sync_legacy_layout(self, x: float, y: float, _px=0.0, _py=0.0) -> None:
        object.__setattr__(self, "_syncing_legacy_layout", True)
        try:
            super().__setattr__("x", float(x))
            super().__setattr__("y", float(y))
        finally:
            object.__setattr__(self, "_syncing_legacy_layout", False)

    def get_layout_rotation(self) -> float:
        game_object = self._try_get_game_object()
        if game_object is None:
            return float(self.rotation)
        return float(game_object.transform.local_euler_angles.z)

    def set_layout_rotation(self, degrees: float) -> None:
        game_object = self._try_get_game_object()
        if game_object is None:
            self.rotation = float(degrees)
            return
        from Infernux.lib import Vector3

        angles = game_object.transform.local_euler_angles
        game_object.transform.local_euler_angles = Vector3(
            float(angles.x), float(angles.y), float(degrees)
        )
        object.__setattr__(self, "_syncing_legacy_layout", True)
        try:
            super().__setattr__("rotation", float(degrees))
        finally:
            object.__setattr__(self, "_syncing_legacy_layout", False)
        _invalidate_rect_cache()
        mark_runtime_ui_dirty()

    def _serialize_fields_document(self):
        game_object = self._try_get_game_object()
        if game_object is not None:
            if not self.is_world_space():
                cw, ch, (px, py, pw, ph), _units = self._layout_reference()
                rect_x, rect_y, _width, _height = self.get_rect(cw, ch)
                anchor_x, anchor_y = self._anchor_origin(pw, ph)
                self._sync_legacy_layout(
                    rect_x - px - anchor_x,
                    rect_y - py - anchor_y,
                )
            object.__setattr__(self, "_syncing_legacy_layout", True)
            try:
                super().__setattr__("rotation", self.get_layout_rotation())
            finally:
                object.__setattr__(self, "_syncing_legacy_layout", False)
        return super()._serialize_fields_document()

    def get_canvas(self):
        """Return the nearest Canvas, or None when this is ordinary world UI."""
        return self._canvas_for_owner(self._try_get_game_object())

    def _canvas_for_owner(self, current):
        """Resolve hierarchy using the owner already validated by this query."""
        from .ui_canvas import UICanvas

        scene = current.scene if current is not None else None
        key = (current, scene, scene.world_id, scene.structure_version) if scene is not None else None
        cached = self.__dict__.get("_canvas_ancestor")
        if key is not None and cached is not None and cached[0] == key:
            return cached[1]
        canvas = None
        while current is not None:
            for component in current.get_py_components():
                if isinstance(component, UICanvas):
                    canvas = component
                    break
            if canvas is not None:
                break
            current = current.get_parent()
        object.__setattr__(self, "_canvas_ancestor", (key, canvas))
        return canvas

    def is_world_space(self) -> bool:
        """World UI is inferred from hierarchy; no special Canvas is required."""
        return self.get_canvas() is None

    def _get_parent_ui_component(self):
        """Return the nearest authored UI parent, without inventing a surface."""
        current = self._try_get_game_object()
        current = current.get_parent() if current is not None else None
        while current is not None:
            for component in current.get_py_components():
                if isinstance(component, InxUIScreenComponent):
                    return component
            current = current.get_parent()
        return None

    def world_ui_matrix(self):
        """Return this element's ordinary world pose without UI scale."""
        game_object = self._try_get_game_object()
        if game_object is None:
            return [1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0]
        transform = game_object.transform
        position = transform.position
        rotation = transform.rotation
        x, y, z, w = (
            float(rotation.x), float(rotation.y),
            float(rotation.z), float(rotation.w),
        )
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return [
            1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0,
            2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0,
            2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0,
            float(position.x), float(position.y), float(position.z), 1.0,
        ]

    def get_effective_group_state(self) -> tuple[float, bool, bool]:
        """Return subtree alpha, interactable and raycast-blocking state."""
        from .ui_group import UIGroup

        alpha = 1.0
        interactable = True
        blocks_raycast = True
        current = self._try_get_game_object()
        while current is not None:
            for component in current.get_py_components():
                if not isinstance(component, UIGroup) or not getattr(component, "enabled", True):
                    continue
                alpha *= max(0.0, min(1.0, float(component.alpha)))
                interactable = interactable and bool(component.interactable)
                blocks_raycast = blocks_raycast and bool(component.blocks_raycast)
            current = current.get_parent()
        return alpha, interactable, blocks_raycast

    def is_effectively_interactable(self) -> bool:
        """Return whether this element may receive a pointer transaction."""
        _alpha, group_interactable, _blocks = self.get_effective_group_state()
        return group_interactable and bool(getattr(self, "interactable", True))

    def effectively_blocks_raycast(self) -> bool:
        """Return whether this element participates in hit blocking."""
        _alpha, _interactable, blocks = self.get_effective_group_state()
        return bool(self.raycast_target) and blocks

    def get_effective_clip_rect(self, canvas_width: float, canvas_height: float):
        """Return the intersection of ancestor UIFrame clip rectangles."""
        if self.is_world_space():
            return None
        from .ui_frame import UIFrame

        clip = None
        current = self._try_get_game_object()
        current = current.get_parent() if current is not None else None
        while current is not None:
            for component in current.get_py_components():
                if not isinstance(component, UIFrame) or not component.clip_content:
                    continue
                x, y, width, height = component.get_rect(canvas_width, canvas_height)
                candidate = (x, y, x + width, y + height)
                if clip is None:
                    clip = candidate
                else:
                    clip = (
                        max(clip[0], candidate[0]), max(clip[1], candidate[1]),
                        min(clip[2], candidate[2]), min(clip[3], candidate[3]),
                    )
            current = current.get_parent()
        return clip

    def _anchor_origin(self, ref_width: float, ref_height: float):
        """Anchor offset within a reference rectangle (parent or canvas)."""
        if self.align_h == ScreenAlignH.Center:
            anchor_x = ref_width * 0.5
        elif self.align_h == ScreenAlignH.Right:
            anchor_x = ref_width
        else:
            anchor_x = 0.0

        if self.align_v == ScreenAlignV.Center:
            anchor_y = ref_height * 0.5
        elif self.align_v == ScreenAlignV.Bottom:
            anchor_y = ref_height
        else:
            anchor_y = 0.0
        return anchor_x, anchor_y

    def _get_parent_world_rect(self, canvas_width: float, canvas_height: float):
        """Return the world rect (x, y, w, h) of the nearest parent UI element.

        Walks up the GameObject hierarchy looking for a parent with an
        InxUIScreenComponent.  If none is found, returns the canvas rect.
        """
        go = self._try_get_game_object()
        if go is None:
            return (0.0, 0.0, canvas_width, canvas_height)
        parent_go = go.get_parent()
        while parent_go is not None:
            for py_comp in parent_go.get_py_components():
                if isinstance(py_comp, InxUIScreenComponent):
                    return py_comp.get_rect(canvas_width, canvas_height)
            parent_go = parent_go.get_parent()
        return (0.0, 0.0, canvas_width, canvas_height)

    def get_resolved_size(self) -> tuple[float, float]:
        """Effective local geometry size; text may derive it without editing fields."""
        return float(self.width), float(self.height)

    def get_rect(self, canvas_width=None, canvas_height=None):
        """Return (x, y, w, h) of the *unrotated* content rect in canvas-space.

        Position is parent-relative: ``x`` / ``y`` are offsets from the
        parent UI element's top-left (or from the canvas origin if this
        element has no parent UI component).  Anchor is computed within
        the parent's width/height.
        """
        game_object = self._try_get_game_object()
        if self._canvas_for_owner(game_object) is None:
            # World UI has no shared reference rectangle. Every element owns
            # only its local geometry; hierarchy affects its ordinary world
            # Transform and stable order, never a hidden Canvas range.
            return (0.0, 0.0, *self.get_resolved_size())
        transform = game_object.transform if game_object is not None else None
        position = transform.local_position if transform is not None else None
        if canvas_width is None or canvas_height is None:
            if position is None:
                return (float(self.x), float(self.y), self.width, self.height)
            return (
                float(position.x) - float(self.width) * 0.5,
                -float(position.y) - float(self.height) * 0.5,
                self.width,
                self.height,
            )
        # Reuse this query's owner and position instead of entering lifecycle
        # resolution again for a signature and then once more for the parent.
        signature = (
            (float(position.x), float(position.y), float(transform.local_euler_angles.z))
            if transform is not None else None
        )
        cache_key = (
            id(self), canvas_width, canvas_height, signature
        )
        cached = _rect_cache.get(cache_key)
        if cached is not None:
            return cached
        cw = float(canvas_width)
        ch = float(canvas_height)
        parent = game_object.get_parent() if game_object is not None else None
        layout_parent = None
        if parent is not None:
            layout_parent = next((
                component for component in parent.get_py_components()
                if callable(getattr(component, "_layout_rect_for_child", None))
            ), None)
        if layout_parent is not None:
            result = layout_parent._layout_rect_for_child(self, cw, ch)
            _rect_cache[cache_key] = result
            return result
        result = self._absolute_rect(cw, ch)
        _rect_cache[cache_key] = result
        return result

    @staticmethod
    def _clamp_layout_extent(value: float, minimum: float, maximum: float) -> float:
        value = max(float(minimum), float(value))
        if float(maximum) > 0.0:
            value = min(value, float(maximum))
        return value

    def _layout_desired_size(self) -> tuple[float, float]:
        return (
            self._clamp_layout_extent(self.width, self.min_width, self.max_width),
            self._clamp_layout_extent(self.height, self.min_height, self.max_height),
        )

    def _absolute_rect(self, canvas_width: float, canvas_height: float,
                       reference_rect=None, size=None):
        cw = float(canvas_width)
        ch = float(canvas_height)
        if reference_rect is None:
            px, py, pw, ph = self._get_parent_world_rect(cw, ch)
        else:
            px, py, pw, ph = reference_rect
        width, height = size if size is not None else self._layout_desired_size()
        anchor_x, anchor_y = self._anchor_origin(pw, ph)
        offset_x, offset_y = self._layout_offset(pw, ph)
        return (px + anchor_x + offset_x, py + anchor_y + offset_y, width, height)

    def _rot_sincos(self):
        """Return (sin, cos) for the current rotation, cached per value."""
        rot = self.get_layout_rotation() % 360.0
        rad = math.radians(rot)
        return math.sin(rad), math.cos(rad)

    def get_visual_rect(self, canvas_width=None, canvas_height=None):
        """Return the axis-aligned bounding box of the rotated content rect.

        Rotation is applied around the center of the unrotated rect.
        Returns (vx, vy, vw, vh) in canvas-space.
        """
        rx, ry, rw, rh = self.get_rect(canvas_width, canvas_height)
        rot = self.get_layout_rotation() % 360.0
        if abs(rot) < 0.001:
            return (rx, ry, rw, rh)
        sin_a, cos_a = self._rot_sincos()
        acos = abs(cos_a)
        asin = abs(sin_a)
        vw = rw * acos + rh * asin
        vh = rw * asin + rh * acos
        return (rx + rw * 0.5 - vw * 0.5, ry + rh * 0.5 - vh * 0.5, vw, vh)

    def calc_visual_size(self, width: float, height: float):
        """Return rotated AABB size for the given unrotated width/height."""
        rot = self.get_layout_rotation() % 360.0
        width = float(width)
        height = float(height)
        if abs(rot) < 0.001:
            return (width, height)
        sin_a, cos_a = self._rot_sincos()
        acos = abs(cos_a)
        asin = abs(sin_a)
        return (
            width * acos + height * asin,
            width * asin + height * acos,
        )

    def _rotated_corner_offset(self, width: float, height: float, corner_index: int,
                                _sincos=None):
        """Return offset from rect origin to the specified rotated corner."""
        width = float(width)
        height = float(height)
        hw, hh = width * 0.5, height * 0.5
        # Local corner offsets: TL, TR, BR, BL
        if corner_index == 0:
            lx, ly = -hw, -hh
        elif corner_index == 1:
            lx, ly = hw, -hh
        elif corner_index == 2:
            lx, ly = hw, hh
        else:
            lx, ly = -hw, hh
        if _sincos is not None:
            sin_a, cos_a = _sincos
        else:
            sin_a, cos_a = self._rot_sincos()
        rx = lx * cos_a - ly * sin_a
        ry = lx * sin_a + ly * cos_a
        return (hw + rx, hh + ry)

    def get_rotated_corners(self, canvas_width=None, canvas_height=None):
        """Return rotated rect corners in TL, TR, BR, BL order."""
        rect_x, rect_y, rect_w, rect_h = self.get_rect(canvas_width, canvas_height)
        hw, hh = rect_w * 0.5, rect_h * 0.5
        sin_a, cos_a = self._rot_sincos()
        # Inline all 4 corners with shared sin/cos
        corners = []
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            corners.append((rect_x + hw + lx * cos_a - ly * sin_a,
                            rect_y + hh + lx * sin_a + ly * cos_a))
        return corners

    def set_rect(self, rect_x: float, rect_y: float, rect_w: float, rect_h: float,
                 canvas_width: float, canvas_height: float):
        """Store a canvas-space rect back into parent-relative serialized fields."""
        cw, ch = float(canvas_width), float(canvas_height)
        px, py, pw, ph = self._get_parent_world_rect(cw, ch)
        anchor_x, anchor_y = self._anchor_origin(pw, ph)
        self.width = float(rect_w)
        self.height = float(rect_h)
        self._set_layout_rect_origin(float(rect_x), float(rect_y), cw, ch)
        _invalidate_rect_cache()

    def set_visual_position(self, vis_x: float, vis_y: float,
                            canvas_width: float, canvas_height: float):
        """Move the element so the visual AABB top-left is at (vis_x, vis_y).

        Keeps width/height/rotation unchanged; only adjusts x/y.
        """
        cw, ch = float(canvas_width), float(canvas_height)
        position = self._local_position_for_visual_origin(vis_x, vis_y, cw, ch)
        game_object = self._try_get_game_object()
        if game_object is not None and position is not None:
            game_object.transform.local_position = position
            rect_x = float(vis_x) + (self.calc_visual_size(self.width, self.height)[0] - self.width) * 0.5
            rect_y = float(vis_y) + (self.calc_visual_size(self.width, self.height)[1] - self.height) * 0.5
            px, py, pw, ph = self._get_parent_world_rect(cw, ch)
            anchor_x, anchor_y = self._anchor_origin(pw, ph)
            self._sync_legacy_layout(rect_x - px - anchor_x, rect_y - py - anchor_y)
        else:
            self._sync_legacy_layout(float(vis_x), float(vis_y))
        _invalidate_rect_cache()
        mark_runtime_ui_dirty()

    def set_size_preserve_visual_position(self, width: float, height: float,
                                          canvas_width: float, canvas_height: float):
        """Set width/height while keeping current visual AABB top-left unchanged."""
        vis_x, vis_y, _, _ = self.get_visual_rect(canvas_width, canvas_height)
        new_width = float(width)
        new_height = float(height)
        new_vis_w, new_vis_h = self.calc_visual_size(new_width, new_height)
        vis_cx = vis_x + new_vis_w * 0.5
        vis_cy = vis_y + new_vis_h * 0.5
        new_rx = vis_cx - new_width * 0.5
        new_ry = vis_cy - new_height * 0.5
        cw, ch = float(canvas_width), float(canvas_height)
        px, py, pw, ph = self._get_parent_world_rect(cw, ch)
        anchor_x, anchor_y = self._anchor_origin(pw, ph)
        self.width = new_width
        self.height = new_height
        self._set_layout_rect_origin(new_rx, new_ry, cw, ch)
        _invalidate_rect_cache()

    def set_size_preserve_center(self, width: float, height: float,
                                 canvas_width: float, canvas_height: float):
        """Set width/height while keeping the element's visual center fixed."""
        rect_x, rect_y, rect_w, rect_h = self.get_rect(canvas_width, canvas_height)
        center_x = rect_x + rect_w * 0.5
        center_y = rect_y + rect_h * 0.5
        new_width = float(width)
        new_height = float(height)
        new_rect_x = center_x - new_width * 0.5
        new_rect_y = center_y - new_height * 0.5
        cw, ch = float(canvas_width), float(canvas_height)
        px, py, pw, ph = self._get_parent_world_rect(cw, ch)
        anchor_x, anchor_y = self._anchor_origin(pw, ph)
        self.width = new_width
        self.height = new_height
        self._set_layout_rect_origin(new_rect_x, new_rect_y, cw, ch)
        _invalidate_rect_cache()

    def set_size_preserve_corner(self, width: float, height: float,
                                 canvas_width: float, canvas_height: float,
                                 corner: str = "top_left"):
        """Set width/height while keeping a rotated corner fixed."""
        corner_map = {
            "top_left": 0,
            "top_right": 1,
            "bottom_right": 2,
            "bottom_left": 3,
        }
        corner_index = corner_map.get(corner, 0)
        fixed_corner_x, fixed_corner_y = self.get_rotated_corners(canvas_width, canvas_height)[corner_index]
        new_width = float(width)
        new_height = float(height)
        sc = self._rot_sincos()
        off_x, off_y = self._rotated_corner_offset(new_width, new_height, corner_index, _sincos=sc)
        new_rect_x = fixed_corner_x - off_x
        new_rect_y = fixed_corner_y - off_y
        cw, ch = float(canvas_width), float(canvas_height)
        px, py, pw, ph = self._get_parent_world_rect(cw, ch)
        anchor_x, anchor_y = self._anchor_origin(pw, ph)
        self.width = new_width
        self.height = new_height
        self._set_layout_rect_origin(new_rect_x, new_rect_y, cw, ch)
        _invalidate_rect_cache()

    # ------------------------------------------------------------------
    # Scene editing
    # ------------------------------------------------------------------

    def on_draw_gizmos_selected(self):
        """Draw this world UI element from its own authoritative Transform."""
        if not self.is_world_space():
            return
        game_object = self._try_get_game_object()
        if game_object is None:
            return

        from Infernux.gizmos import Gizmos

        logical_width, logical_height = (max(1.0, value) for value in self.get_resolved_size())
        origin_x = logical_width * 0.5
        origin_y = logical_height * 0.5
        pixels_per_unit = WORLD_UI_PIXELS_PER_UNIT
        corners = [
            (
                (x - origin_x) / pixels_per_unit,
                -(y - origin_y) / pixels_per_unit,
                0.0,
            )
            for x, y in (
                (0.0, 0.0),
                (logical_width, 0.0),
                (logical_width, logical_height),
                (0.0, logical_height),
            )
        ]

        previous_color = Gizmos.color
        previous_matrix = Gizmos.matrix
        try:
            Gizmos.color = (1.0, 0.62, 0.22)
            Gizmos.matrix = self.world_ui_matrix()
            for index in range(4):
                Gizmos.draw_line(corners[index], corners[(index + 1) % 4])
        finally:
            Gizmos.color = previous_color
            Gizmos.matrix = previous_matrix

    # ------------------------------------------------------------------
    # Pointer event hooks (override in subclasses)
    # ------------------------------------------------------------------

    def on_pointer_enter(self, event_data):
        """Called when the pointer enters this element's rect."""
        pass

    def on_pointer_exit(self, event_data):
        """Called when the pointer leaves this element's rect."""
        pass

    def on_pointer_down(self, event_data):
        """Called when a mouse button is pressed over this element."""
        pass

    def on_pointer_up(self, event_data):
        """Called when a mouse button is released over this element."""
        pass

    def on_pointer_click(self, event_data):
        """Called on a complete click (down + up on the same element)."""
        pass

    def on_begin_drag(self, event_data):
        """Called when a drag gesture starts on this element."""
        pass

    def on_drag(self, event_data):
        """Called each frame during a drag gesture."""
        pass

    def on_end_drag(self, event_data):
        """Called when a drag gesture ends."""
        pass

    def on_scroll(self, event_data):
        """Called when the scroll wheel is used over this element."""
        pass

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------

    def contains_point(self, px: float, py: float,
                       canvas_width: float, canvas_height: float,
                       tolerance: float = 0.0) -> bool:
        """Test whether canvas-space point (px, py) lies inside this element.

        Uses the oriented (rotated) bounding box for accurate hit-testing.
        *tolerance* expands the hit area by that many canvas-space pixels on each side.
        """
        if not self.raycast_target:
            return False

        rx, ry, rw, rh = self.get_rect(canvas_width, canvas_height)
        if tolerance > 0.0:
            rx -= tolerance
            ry -= tolerance
            rw += tolerance * 2.0
            rh += tolerance * 2.0
        rot = self.get_layout_rotation() % 360.0

        if abs(rot) < 0.001:
            # Fast axis-aligned path
            return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)

        # Rotate point into element's local frame (negate angle)
        sin_a, cos_a = self._rot_sincos()
        dx = px - (rx + rw * 0.5)
        dy = py - (ry + rh * 0.5)
        lx = dx * cos_a + dy * sin_a + rw * 0.5
        ly = -dx * sin_a + dy * cos_a + rh * 0.5
        return (0.0 <= lx <= rw) and (0.0 <= ly <= rh)


def is_ui_screen_component(component) -> bool:
    """Match screen UI elements across duplicated project module overlays."""
    component_type = type(component)
    for base in getattr(component_type, "__mro__", ()):
        if (
            base.__module__ == InxUIScreenComponent.__module__
            and base.__qualname__ == InxUIScreenComponent.__qualname__
        ):
            return True
    # A preloaded Web module can retain the concrete UI class without its
    # original base class in the current interpreter.  Built-in screen
    # controls still have stable module identities, so recognize that finite
    # component surface explicitly while leaving Canvas itself out.
    if (
        component_type.__module__.startswith("Infernux.ui.ui_")
        and component_type.__module__.rsplit(".", 1)[-1]
        in {
            "ui_button", "ui_frame", "ui_image", "ui_progress_bar", "ui_slider", "ui_text",
        }
    ):
        return True
    return (
        isinstance(component, InxUIScreenComponent)
        or (
            component_type.__module__ == InxUIScreenComponent.__module__
            and component_type.__qualname__.startswith(
                InxUIScreenComponent.__qualname__
            )
        )
    )
