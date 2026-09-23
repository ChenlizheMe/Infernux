"""
Camera — Python InxComponent wrapper for the C++ Camera component.

Exposes projection settings, clear flags, culling mask, and coordinate
conversion as CppProperty descriptors and delegate methods.

Provides built-in Gizmos drawing (frustum wireframe) via
``on_draw_gizmos_selected()``, rendered automatically by the GizmosCollector
when the camera is selected.

Example::

    from Infernux.components.builtin import Camera
    from Infernux.lib import CameraProjection

    class CinematicCamera(InxComponent):
        def start(self):
            cam = self.game_object.get_component(Camera)
            cam.field_of_view = 45.0
            cam.near_clip = 0.1
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from Infernux.components.builtin_component import BuiltinComponent, CppProperty
from Infernux.components._gizmo_ids import ICON_KIND_CAMERA


def _vec4_to_list(v):
    """Convert C++ vec4 to [r, g, b, a] list for COLOR field."""
    return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]


def _list_to_vec4(v):
    """Convert RGBA list/tuple back to vec4f for C++ setter."""
    if isinstance(v, (list, tuple)):
        from Infernux.lib import vec4f
        return vec4f(float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return v


def _wrap_target_texture(state):
    from Infernux.core.render_texture import RenderTexture
    from Infernux.core.asset_ref import RenderTextureRef
    guid, native = state
    if native is None:
        return RenderTextureRef(guid) if guid else None
    result = RenderTexture.__new__(RenderTexture)
    result._native = native
    return result


def _set_target_texture(cpp, value):
    from Infernux.core.render_texture import RenderTexture
    from Infernux.core.asset_ref import RenderTextureRef
    if isinstance(value, RenderTextureRef):
        resolved = value.resolve()
        if resolved is None:
            cpp.target_texture_guid = value.guid
            return
        value = resolved
    if value is not None and not isinstance(value, RenderTexture):
        raise TypeError("Camera.target_texture requires RenderTexture, RenderTextureRef or None")
    cpp.target_texture = None if value is None else value._native


def _is_perspective(comp) -> bool:
    return int(comp.projection_mode) == 0


def _uses_physical_properties(comp) -> bool:
    return _is_perspective(comp) and bool(comp.use_physical_properties)


def _uses_field_of_view(comp) -> bool:
    return _is_perspective(comp) and not bool(comp.use_physical_properties)


def _set_near_clip(cpp, value) -> None:
    # Inspector/Python wrapper edits are clamped before crossing the native
    # boundary, so a drag cannot publish an invalid transient clip plane.
    near = min(max(float(value), 0.001), float(cpp.far_clip) - 0.001)
    cpp.near_clip = max(near, 0.001)


def _set_far_clip(cpp, value) -> None:
    far = max(float(value), float(cpp.near_clip) + 0.001)
    cpp.far_clip = min(far, 1_000_000_000.0)


# Maximum far-plane distance for gizmo visualization (Unity caps ~1000)
_FAR_CLIP_VISUAL_CAP = 1000.0

# Gizmo color — Unity uses white for camera gizmos
_CAMERA_GIZMO_COLOR = (1.0, 1.0, 1.0)

# Unity CameraEditor sensor presets.  Sensor Type is a derived Inspector
# convenience rather than Camera document state; choosing a preset commits the
# authoritative sensor_size so Undo and serialization remain exact.
_SENSOR_PRESET_SIZES = (
    (4.8, 3.5), (5.79, 4.01), (10.26, 7.49), (12.522, 7.417),
    (21.95, 9.35), (21.946, 16.002), (24.89, 18.66),
    (20.726, 15.545), (24.892, 18.669), (20.955, 11.328),
    (21.946, 18.593), (54.12, 25.59), (52.476, 23.012),
    (70.41, 52.63),
)


def _render_sensor_type(ctx, comp, label_width) -> None:
    from Infernux.engine.ui.inspector_components import (
        _record_builtin_property,
        _serialized_field_label,
    )
    from Infernux.engine.ui.inspector_utils import (
        has_field_changed,
        render_serialized_field,
    )

    prop = type(comp).sensor_type
    current_type = comp.sensor_type
    sensor_label = _serialized_field_label("sensor_type", prop.metadata)
    selected_type = render_serialized_field(
        ctx, "##camera_sensor_type", sensor_label, prop.metadata,
        current_type, label_width,
    )
    if not has_field_changed(prop.metadata.field_type, current_type, selected_type):
        return
    preset_index = int(selected_type)
    if 0 <= preset_index < len(_SENSOR_PRESET_SIZES):
        from Infernux.lib import Vector2
        width, height = _SENSOR_PRESET_SIZES[preset_index]
        _record_builtin_property(
            comp, "sensor_size", comp.sensor_size,
            Vector2(width, height), "Set Camera Sensor Type",
        )


def _render_culling_mask(ctx, comp, label_width) -> None:
    from Infernux.engine.ui.inspector_components import _record_builtin_property
    from Infernux.engine.ui.inspector_utils import field_label
    from Infernux.lib import TagLayerManager

    names = list(TagLayerManager.instance().get_all_layers() or [])
    names = [str(name).strip() or f"Layer {index}" for index, name in enumerate(names)]
    names = (names + [f"Layer {i}" for i in range(len(names), 32)])[:32]
    old_mask = int(comp.culling_mask) & 0xFFFFFFFF
    selected = sum(1 for i in range(32) if old_mask & (1 << i))
    label = "Everything" if selected == 32 else ("Nothing" if selected == 0 else f"{selected} Layers")
    field_label(ctx, "Culling Mask", label_width)
    if ctx.button(f"{label}##camera_culling_mask"):
        ctx.open_popup("##camera_culling_mask_popup")
    if ctx.begin_popup("##camera_culling_mask_popup"):
        new_mask = old_mask
        if ctx.button("Everything##camera_culling_everything"):
            new_mask = 0xFFFFFFFF
        ctx.same_line()
        if ctx.button("Nothing##camera_culling_nothing"):
            new_mask = 0
        for index, name in enumerate(names):
            checked = bool(new_mask & (1 << index))
            updated = ctx.checkbox(f"{name}##camera_layer_{index}", checked)
            if updated != checked:
                new_mask ^= 1 << index
        if new_mask != old_mask:
            _record_builtin_property(
                comp, "culling_mask", old_mask, new_mask,
                "Set Camera Culling Mask",
            )
        ctx.end_popup()


class Camera(BuiltinComponent):
    """Python wrapper for the C++ Camera component.

    Properties delegate to the C++ ``Camera`` via CppProperty.
    Draws a Unity-style frustum wireframe gizmo when selected.
    ``target_texture`` accepts an imported or runtime RenderTexture with depth.
    Imported targets persist by GUID; missing assets retain their reference.
    """

    _cpp_type_name = "Camera"
    _component_category_ = "Rendering"
    _display_name_key = "component.camera"

    # Gizmo visibility: only show frustum wireframe when camera is selected
    _always_show = False

    # Scene icon: white diamond shown at camera position (Unity-style)
    _gizmo_icon_color = (1.0, 1.0, 1.0)
    _gizmo_icon_kind = ICON_KIND_CAMERA

    # Native declarations own types, defaults and Inspector metadata.
    # Visibility callbacks and Python/native value adapters stay outside the catalog.
    projection_mode = CppProperty.from_native("Camera", "projection_mode")
    field_of_view = CppProperty.from_native(
        "Camera", "field_of_view", visible_when=_uses_field_of_view,
    )
    use_physical_properties = CppProperty.from_native(
        "Camera", "use_physical_properties", visible_when=_is_perspective,
    )
    iso = CppProperty.from_native(
        "Camera", "iso", visible_when=_uses_physical_properties,
    )
    shutter_speed = CppProperty.from_native(
        "Camera", "shutter_speed", visible_when=_uses_physical_properties,
    )
    aperture = CppProperty.from_native(
        "Camera", "aperture", visible_when=_uses_physical_properties,
    )
    focus_distance = CppProperty.from_native(
        "Camera", "focus_distance", visible_when=_uses_physical_properties,
    )
    blade_count = CppProperty.from_native(
        "Camera", "blade_count", visible_when=_uses_physical_properties,
    )
    curvature = CppProperty.from_native(
        "Camera", "curvature", visible_when=_uses_physical_properties,
    )
    barrel_clipping = CppProperty.from_native(
        "Camera", "barrel_clipping", visible_when=_uses_physical_properties,
    )
    anamorphism = CppProperty.from_native(
        "Camera", "anamorphism", visible_when=_uses_physical_properties,
    )
    focal_length = CppProperty.from_native(
        "Camera", "focal_length", visible_when=_uses_physical_properties,
    )
    sensor_type = CppProperty.from_native(
        "Camera", "sensor_type", visible_when=_uses_physical_properties,
    )
    sensor_size = CppProperty.from_native(
        "Camera", "sensor_size", visible_when=_uses_physical_properties,
    )
    lens_shift = CppProperty.from_native(
        "Camera", "lens_shift", visible_when=_uses_physical_properties,
    )
    gate_fit = CppProperty.from_native(
        "Camera", "gate_fit", visible_when=_uses_physical_properties,
    )
    orthographic_size = CppProperty.from_native(
        "Camera", "orthographic_size", visible_when=lambda comp: int(comp.projection_mode) == 1,
    )
    near_clip = CppProperty.from_native("Camera", "near_clip", native_setter=_set_near_clip)
    far_clip = CppProperty.from_native("Camera", "far_clip", native_setter=_set_far_clip)
    depth = CppProperty.from_native("Camera", "depth")
    culling_mask = CppProperty.from_native("Camera", "culling_mask")
    clear_flags = CppProperty.from_native("Camera", "clear_flags")
    background_color = CppProperty.from_native(
        "Camera", "background_color", visible_when=lambda comp: int(comp.clear_flags) == 1,
        get_converter=_vec4_to_list, set_converter=_list_to_vec4,
    )
    stop_nans = CppProperty.from_native("Camera", "stop_nans")
    dithering = CppProperty.from_native("Camera", "dithering")
    target_texture = CppProperty.from_native(
        "Camera", "target_texture",
        native_getter=lambda cpp: (cpp.target_texture_guid, cpp.target_texture),
        get_converter=_wrap_target_texture, native_setter=_set_target_texture,
    )

    def render_inspector(self, ctx) -> None:
        """Render the camera with a named, multi-select layer mask.

        The serialized value remains Unity-compatible 32-bit bits, but the
        authoring surface never asks users to type a mask integer. Physical
        properties follow Unity's Perspective + Physical Camera checkbox and
        remain hidden for orthographic or ordinary perspective cameras.
        """
        from Infernux.engine.ui.inspector_components import render_builtin_via_setters

        render_builtin_via_setters(
            ctx, self, type(self),
            custom_fields={
                "sensor_type": _render_sensor_type,
                "culling_mask": _render_culling_mask,
            },
        )

    # ------------------------------------------------------------------
    # Read-only properties (delegates)
    # ------------------------------------------------------------------

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height) — read-only, computed from viewport."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.aspect_ratio
        return 1.778

    @property
    def pixel_width(self) -> int:
        """Render target width in pixels (read-only)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.pixel_width
        return 0

    @property
    def pixel_height(self) -> int:
        """Render target height in pixels (read-only)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.pixel_height
        return 0

    # ------------------------------------------------------------------
    # Coordinate conversion (delegate methods)
    # ------------------------------------------------------------------

    def set_clip_planes(self, near_clip: float, far_clip: float) -> None:
        """Set both clipping planes atomically; requires finite 0 < near < far."""
        self._require_cpp_component().set_clip_planes(near_clip, far_clip)

    @property
    def projection_matrix(self):
        """NumPy (4,4) matrix copy, indexed [row, column].

        Engine clip space is left-handed (+Z forward), depth [0,1], with
        projection Y inverted for top-left pixels. Assignment installs a
        runtime override; FOV/aspect/clip edits resume after reset. No scene
        fields are overwritten. Unlike Unity's CPU projection matrix, this
        is already the engine GPU clip-space convention; do not flip it again.
        """
        return self._require_cpp_component().projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, matrix):
        self._require_cpp_component().projection_matrix = matrix

    @property
    def view_matrix(self):
        """Affine world-to-camera NumPy (4,4) copy (+Z forward).

        Setting it overrides the rendered pose without changing Transform.
        Call reset_view_matrix() to follow Transform again. Reflection callers
        also toggle invert_culling relative to their source camera.
        """
        return self._require_cpp_component().view_matrix

    @view_matrix.setter
    def view_matrix(self, matrix):
        self._require_cpp_component().view_matrix = matrix

    @property
    def camera_to_world_matrix(self):
        """Inverse of the effective view, including a runtime override."""
        return self._require_cpp_component().camera_to_world_matrix

    @property
    def has_custom_view_matrix(self) -> bool:
        return self._require_cpp_component().has_custom_view_matrix

    def reset_view_matrix(self) -> None:
        """Resume Transform-driven viewing; does not change invert_culling."""
        self._require_cpp_component().reset_view_matrix()

    def reset_history(self) -> None:
        """Discard accumulated history before this camera's next render.

        Use after a small teleport or a discontinuous change not covered by
        automatic camera-cut detection. Each view of this camera resets once;
        other cameras, Transform, projection and saved assets are unchanged.
        """
        self._require_cpp_component().reset_history()

    @property
    def invert_culling(self) -> bool:
        """Runtime winding inversion for this camera, not light-space shadows."""
        return self._require_cpp_component().invert_culling

    @invert_culling.setter
    def invert_culling(self, value: bool) -> None:
        self._require_cpp_component().invert_culling = value

    @property
    def has_custom_projection_matrix(self) -> bool:
        return self._require_cpp_component().has_custom_projection_matrix

    def reset_projection_matrix(self) -> None:
        """Resume the current authored FOV/orthographic/aspect/clip settings."""
        self._require_cpp_component().reset_projection_matrix()

    def calculate_oblique_matrix(self, clip_plane):
        """Return a projection clipped by camera-space (nx,ny,nz,d), without applying it.

        Points on the positive side are retained. Use ``projection_matrix =
        calculate_oblique_matrix(...)`` to apply; use reset before deriving a
        fresh plane from authored settings on a later frame.
        """
        return self._require_cpp_component().calculate_oblique_matrix(clip_plane)

    def screen_to_world_point(
        self, x: float, y: float, depth: float = 0.0
    ) -> Optional[Tuple[float, float, float]]:
        """Convert top-left screen pixels and normalized depth [0..1] to world position."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.screen_to_world_point(x, y, depth)
        return None

    def world_to_screen_point(
        self, x: float, y: float, z: float
    ) -> Optional[Tuple[float, float]]:
        """Convert world position to top-left screen pixel coordinates (x, y)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.world_to_screen_point(x, y, z)
        return None

    def screen_point_to_ray(
        self, x: float, y: float,
        viewport_width: Optional[float] = None,
        viewport_height: Optional[float] = None,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Build a ray from top-left, viewport-relative screen coordinates.

        Returns ``((ox, oy, oz), (dx, dy, dz))`` — origin at the near
        plane and a normalised direction vector.
        """
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.screen_point_to_ray(
                x, y, viewport_width, viewport_height
            )
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialize Camera to JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.serialize()
        return "{}"

    def deserialize(self, json_str: str) -> bool:
        """Deserialize Camera from JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.deserialize(json_str)
        return False

    # ------------------------------------------------------------------
    # Gizmos — Unity-style camera frustum + body icon
    # ------------------------------------------------------------------

    def on_draw_gizmos_selected(self):
        """Draw camera frustum wireframe and body icon when selected.

        Called automatically by the GizmosCollector when the owning
        GameObject (or an ancestor) is selected in the editor.
        Replicates Unity's camera gizmo appearance: wireframe frustum
        (perspective or orthographic), a small film-gate rectangle, a
        camera body, and a film reel triangle on top.
        """
        from Infernux.gizmos import Gizmos

        if self._get_bound_native_component() is None:
            return
        positions, indices = self._frustum_wire_geometry(
            self.projection_matrix, self.view_matrix, _FAR_CLIP_VISUAL_CAP)
        Gizmos.color = _CAMERA_GIZMO_COLOR
        Gizmos.draw_lines(positions, indices)

    @staticmethod
    def _frustum_wire_geometry(projection, view, distance_cap):
        """Use the rendered projection, including asymmetric/oblique near planes."""
        import numpy as np

        clip = np.array([[-1, -1, 0, 1], [1, -1, 0, 1],
                         [1, 1, 0, 1], [-1, 1, 0, 1]], dtype=np.float64)
        inverse = np.linalg.inv(np.asarray(projection, dtype=np.float64))
        near_h = clip @ inverse.T
        near = near_h[:, :3] / near_h[:, 3:4]
        clip[:, 2] = 1
        far_h = clip @ inverse.T
        direction = far_h[:, :3] - near * far_h[:, 3:4]
        lengths = np.linalg.norm(direction, axis=1, keepdims=True)
        # Infinite far is legitimate. Limit display length along each ray,
        # not by scaling the entire orthographic rectangle towards the eye.
        extent = np.full((4, 1), np.inf)
        np.divide(lengths, far_h[:, 3:4], out=extent, where=far_h[:, 3:4] > 0)
        far = near + direction / lengths * np.minimum(extent, distance_cap)
        points = np.column_stack((np.concatenate((near, far)), np.ones(8)))
        world = points @ np.linalg.inv(np.asarray(view, dtype=np.float64)).T
        indices = np.array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4,
                            0, 4, 1, 5, 2, 6, 3, 7], dtype=np.uint32)
        return np.ascontiguousarray(world[:, :3] / world[:, 3:4], dtype=np.float32), indices.reshape(-1, 2)


