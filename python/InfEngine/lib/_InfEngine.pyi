"""
Type stubs for InfEngine native bindings (_InfEngine.pyd).

Generated from pybind11 binding definitions in cpp/infengine/tools/pybinding/.
Provides IDE autocompletion for all C++ types exposed to Python.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

# =============================================================================
# Module-level constants
# =============================================================================

GIZMO_X_AXIS_ID: int
GIZMO_Y_AXIS_ID: int
GIZMO_Z_AXIS_ID: int

# =============================================================================
# Enums
# =============================================================================


class LogLevel(IntEnum):
    """Engine log verbosity level."""

    Debug: int
    Info: int
    Warn: int
    Error: int
    Fatal: int


class PrimitiveType(IntEnum):
    """Primitive mesh types that can be created in a scene."""

    Cube: int
    Sphere: int
    Capsule: int
    Cylinder: int
    Plane: int


class Space(IntEnum):
    """Coordinate space for rotation operations."""

    Self: int
    """Rotate relative to the object's local axes (default)."""
    World: int
    """Rotate relative to world axes."""


class LightType(IntEnum):
    """Types of light sources."""

    Directional: int
    Point: int
    Spot: int
    Area: int


class LightShadows(IntEnum):
    """Shadow mode for lights."""

    NoShadows: int
    Hard: int
    Soft: int


class ScenePassType(IntEnum):
    """Built-in scene render pass types."""

    DepthPrePass: int
    ShadowPass: int
    MainColor: int
    Transparent: int
    UI: int
    Custom: int


class GraphPassActionType(IntEnum):
    """Render action for a graph pass."""

    NONE: int
    DRAW_RENDERERS: int
    DRAW_SKYBOX: int
    COMPUTE: int
    CUSTOM: int
    DRAW_SHADOW_CASTERS: int


class ResourceType(IntEnum):
    """File resource types in the asset system."""

    Meta: int
    Shader: int
    Texture: int
    Mesh: int
    Material: int
    Script: int
    Audio: int
    DefaultText: int
    DefaultBinary: int


class SortingCriteria(IntEnum):
    """Sorting criteria for draw calls."""

    NONE: int
    COMMON_OPAQUE: int
    COMMON_TRANSPARENT: int
    OPTIMIZE_STATE_CHANGES: int


class VkFormat(IntEnum):
    """Vulkan image format (subset for render target creation)."""

    R8G8B8A8_UNORM: int
    R8G8B8A8_SRGB: int
    B8G8R8A8_UNORM: int
    R16G16B16A16_SFLOAT: int
    R32G32B32A32_SFLOAT: int
    R32_SFLOAT: int
    R8_UNORM: int
    R8G8_UNORM: int
    R16G16_SFLOAT: int
    A2R10G10B10_UNORM_PACK32: int
    R16_SFLOAT: int
    D32_SFLOAT: int
    D24_UNORM_S8_UINT: int


class VkSampleCount(IntEnum):
    """Vulkan MSAA sample count."""

    COUNT_1: int
    COUNT_2: int
    COUNT_4: int
    COUNT_8: int


# =============================================================================
# Math types
# =============================================================================


class Vector2:
    """2D float vector with swizzle-style properties and arithmetic operators."""

    x: float
    y: float
    r: float
    g: float

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, x: float, y: float) -> None: ...
    def __getitem__(self, i: int) -> float: ...
    def __setitem__(self, i: int, value: float) -> None: ...
    def __add__(self, other: Union[Vector2, int, float]) -> Vector2: ...
    def __radd__(self, other: Union[int, float]) -> Vector2: ...
    def __sub__(self, other: Union[Vector2, int, float]) -> Vector2: ...
    def __rsub__(self, other: Union[int, float]) -> Vector2: ...
    def __mul__(self, other: Union[Vector2, int, float]) -> Vector2: ...
    def __rmul__(self, other: Union[int, float]) -> Vector2: ...
    def __truediv__(self, other: Union[Vector2, int, float]) -> Vector2: ...
    def __rtruediv__(self, other: Union[int, float]) -> Vector2: ...
    def __iadd__(self, other: Union[Vector2, int, float]) -> Vector2: ...
    def __isub__(self, other: Union[Vector2, int, float]) -> Vector2: ...
    def __imul__(self, other: Union[int, float]) -> Vector2: ...
    def __itruediv__(self, other: Union[int, float]) -> Vector2: ...
    def __eq__(self, other: Vector2) -> bool: ...
    def __ne__(self, other: Vector2) -> bool: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def up() -> Vector2: ...
    @staticmethod
    def down() -> Vector2: ...
    @staticmethod
    def left() -> Vector2: ...
    @staticmethod
    def right() -> Vector2: ...
    @staticmethod
    def zero() -> Vector2: ...
    @staticmethod
    def one() -> Vector2: ...
    @staticmethod
    def negative_infinity() -> Vector2: ...
    @staticmethod
    def positive_infinity() -> Vector2: ...
    @staticmethod
    def cross(a: Vector2, b: Vector2) -> float: ...
    @staticmethod
    def magnitude(v: Vector2) -> float: ...
    @staticmethod
    def normalize(v: Vector2) -> Vector2: ...
    @staticmethod
    def sqr_magnitude(v: Vector2) -> float: ...
    @staticmethod
    def angle(a: Vector2, b: Vector2) -> float: ...
    @staticmethod
    def clamp_magnitude(v: Vector2, max_length: float) -> Vector2: ...
    @staticmethod
    def distance(a: Vector2, b: Vector2) -> float: ...
    @staticmethod
    def dot(a: Vector2, b: Vector2) -> float: ...
    @staticmethod
    def lerp(a: Vector2, b: Vector2, t: float) -> Vector2: ...
    @staticmethod
    def lerp_unclamped(a: Vector2, b: Vector2, t: float) -> Vector2: ...
    @staticmethod
    def max(a: Vector2, b: Vector2) -> Vector2: ...
    @staticmethod
    def min(a: Vector2, b: Vector2) -> Vector2: ...
    @staticmethod
    def move_towards(current: Vector2, target: Vector2, max_delta: float) -> Vector2: ...
    @staticmethod
    def reflect(direction: Vector2, normal: Vector2) -> Vector2: ...
    @staticmethod
    def signed_angle(a: Vector2, b: Vector2) -> float: ...
    @staticmethod
    def smooth_damp(
        current: Vector2,
        target: Vector2,
        current_velocity: Vector2,
        smooth_time: float,
        max_speed: float,
        delta_time: float,
    ) -> tuple[Vector2, Vector2]: ...


class Vector3:
    """3D float vector with swizzle-style properties and arithmetic operators."""

    x: float
    y: float
    z: float
    r: float
    g: float
    b: float

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float) -> None: ...
    def __getitem__(self, i: int) -> float: ...
    def __setitem__(self, i: int, value: float) -> None: ...
    def __add__(self, other: Union[Vector3, int, float]) -> Vector3: ...
    def __radd__(self, other: Union[int, float]) -> Vector3: ...
    def __sub__(self, other: Union[Vector3, int, float]) -> Vector3: ...
    def __rsub__(self, other: Union[int, float]) -> Vector3: ...
    def __mul__(self, other: Union[Vector3, int, float]) -> Vector3: ...
    def __rmul__(self, other: Union[int, float]) -> Vector3: ...
    def __truediv__(self, other: Union[Vector3, int, float]) -> Vector3: ...
    def __rtruediv__(self, other: Union[int, float]) -> Vector3: ...
    def __iadd__(self, other: Union[Vector3, int, float]) -> Vector3: ...
    def __isub__(self, other: Union[Vector3, int, float]) -> Vector3: ...
    def __imul__(self, other: Union[int, float]) -> Vector3: ...
    def __itruediv__(self, other: Union[int, float]) -> Vector3: ...
    def __eq__(self, other: Vector3) -> bool: ...
    def __ne__(self, other: Vector3) -> bool: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def up() -> Vector3: ...
    @staticmethod
    def down() -> Vector3: ...
    @staticmethod
    def left() -> Vector3: ...
    @staticmethod
    def right() -> Vector3: ...
    @staticmethod
    def forward() -> Vector3: ...
    @staticmethod
    def back() -> Vector3: ...
    @staticmethod
    def zero() -> Vector3: ...
    @staticmethod
    def one() -> Vector3: ...
    @staticmethod
    def negative_infinity() -> Vector3: ...
    @staticmethod
    def positive_infinity() -> Vector3: ...
    @staticmethod
    def cross(a: Vector3, b: Vector3) -> Vector3: ...
    @staticmethod
    def magnitude(v: Vector3) -> float: ...
    @staticmethod
    def normalize(v: Vector3) -> Vector3: ...
    @staticmethod
    def sqr_magnitude(v: Vector3) -> float: ...
    @staticmethod
    def angle(a: Vector3, b: Vector3) -> float: ...
    @staticmethod
    def clamp_magnitude(v: Vector3, max_length: float) -> Vector3: ...
    @staticmethod
    def distance(a: Vector3, b: Vector3) -> float: ...
    @staticmethod
    def dot(a: Vector3, b: Vector3) -> float: ...
    @staticmethod
    def lerp(a: Vector3, b: Vector3, t: float) -> Vector3: ...
    @staticmethod
    def lerp_unclamped(a: Vector3, b: Vector3, t: float) -> Vector3: ...
    @staticmethod
    def max(a: Vector3, b: Vector3) -> Vector3: ...
    @staticmethod
    def min(a: Vector3, b: Vector3) -> Vector3: ...
    @staticmethod
    def ortho_normalize(v1: Vector3, v2: Vector3, v3: Vector3) -> tuple[Vector3, Vector3, Vector3]: ...
    @staticmethod
    def project(vector: Vector3, on_normal: Vector3) -> Vector3: ...
    @staticmethod
    def project_on_plane(vector: Vector3, plane_normal: Vector3) -> Vector3: ...
    @staticmethod
    def move_towards(current: Vector3, target: Vector3, max_delta: float) -> Vector3: ...
    @staticmethod
    def reflect(direction: Vector3, normal: Vector3) -> Vector3: ...
    @staticmethod
    def rotate_towards(
        current: Vector3, target: Vector3, max_radians_delta: float, max_magnitude_delta: float
    ) -> Vector3: ...  # returns the rotated vector
    @staticmethod
    def slerp(a: Vector3, b: Vector3, t: float) -> Vector3: ...
    @staticmethod
    def slerp_unclamped(a: Vector3, b: Vector3, t: float) -> Vector3: ...
    @staticmethod
    def signed_angle(from_: Vector3, to: Vector3, axis: Vector3) -> float: ...
    @staticmethod
    def smooth_damp(
        current: Vector3,
        target: Vector3,
        current_velocity: Vector3,
        smooth_time: float,
        max_speed: float,
        delta_time: float,
    ) -> tuple[Vector3, Vector3]: ...


class vec4f:
    """4D float vector with swizzle-style properties and arithmetic operators."""

    x: float
    y: float
    z: float
    w: float
    r: float
    g: float
    b: float
    a: float

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...
    def __getitem__(self, i: int) -> float: ...
    def __setitem__(self, i: int, value: float) -> None: ...
    def __add__(self, other: Union[vec4f, int, float]) -> vec4f: ...
    def __radd__(self, other: Union[int, float]) -> vec4f: ...
    def __sub__(self, other: Union[vec4f, int, float]) -> vec4f: ...
    def __rsub__(self, other: Union[int, float]) -> vec4f: ...
    def __mul__(self, other: Union[vec4f, int, float]) -> vec4f: ...
    def __rmul__(self, other: Union[int, float]) -> vec4f: ...
    def __truediv__(self, other: Union[vec4f, int, float]) -> vec4f: ...
    def __rtruediv__(self, other: Union[int, float]) -> vec4f: ...
    def __iadd__(self, other: Union[vec4f, int, float]) -> vec4f: ...
    def __isub__(self, other: Union[vec4f, int, float]) -> vec4f: ...
    def __imul__(self, other: Union[int, float]) -> vec4f: ...
    def __itruediv__(self, other: Union[int, float]) -> vec4f: ...
    def __eq__(self, other: vec4f) -> bool: ...
    def __ne__(self, other: vec4f) -> bool: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def magnitude(v: vec4f) -> float: ...
    @staticmethod
    def normalize(v: vec4f) -> vec4f: ...
    @staticmethod
    def sqr_magnitude(v: vec4f) -> float: ...
    @staticmethod
    def distance(a: vec4f, b: vec4f) -> float: ...
    @staticmethod
    def project(a: vec4f, b: vec4f) -> vec4f: ...
    @staticmethod
    def dot(a: vec4f, b: vec4f) -> float: ...
    @staticmethod
    def lerp(a: vec4f, b: vec4f, t: float) -> vec4f: ...
    @staticmethod
    def lerp_unclamped(a: vec4f, b: vec4f, t: float) -> vec4f: ...
    @staticmethod
    def max(a: vec4f, b: vec4f) -> vec4f: ...
    @staticmethod
    def min(a: vec4f, b: vec4f) -> vec4f: ...
    @staticmethod
    def move_towards(current: vec4f, target: vec4f, max_delta: float) -> vec4f: ...
    @staticmethod
    def smooth_damp(
        current: vec4f,
        target: vec4f,
        current_velocity: vec4f,
        smooth_time: float,
        max_speed: float,
        delta_time: float,
    ) -> tuple[vec4f, vec4f]: ...


class quatf:
    """Quaternion (x, y, z, w) backed by glm::quat."""

    x: float
    y: float
    z: float
    w: float

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...
    def __getitem__(self, i: int) -> float: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[float]: ...
    def __mul__(self, other: quatf) -> quatf: ...
    def __eq__(self, other: quatf) -> bool: ...
    def __ne__(self, other: quatf) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __copy__(self) -> quatf: ...
    def __deepcopy__(self, memo: Any) -> quatf: ...
    def to_tuple(self) -> Tuple[float, float, float, float]: ...

    @property
    def euler_angles(self) -> Vector3: ...
    @property
    def normalized(self) -> quatf: ...

    @staticmethod
    def identity() -> quatf: ...
    @staticmethod
    def euler(x: float, y: float, z: float) -> quatf: ...
    @staticmethod
    def angle_axis(angle: float, axis: Vector3) -> quatf: ...
    @staticmethod
    def look_rotation(forward: Vector3, up: Vector3 = ...) -> quatf: ...
    @staticmethod
    def dot(a: quatf, b: quatf) -> float: ...
    @staticmethod
    def angle(a: quatf, b: quatf) -> float: ...
    @staticmethod
    def slerp(a: quatf, b: quatf, t: float) -> quatf: ...
    @staticmethod
    def lerp(a: quatf, b: quatf, t: float) -> quatf: ...
    @staticmethod
    def inverse(q: quatf) -> quatf: ...
    @staticmethod
    def rotate_towards(from_: quatf, to: quatf, max_degrees_delta: float) -> quatf: ...


# =============================================================================
# Scene — Components
# =============================================================================


class Component:
    """Base class for all components attached to a GameObject."""

    @property
    def type_name(self) -> str:
        """Component type name (e.g. 'Transform', 'MeshRenderer', 'Light')."""
        ...
    @property
    def component_id(self) -> int:
        """Unique component ID."""
        ...
    @property
    def enabled(self) -> bool:
        """Whether this component is enabled."""
        ...
    @enabled.setter
    def enabled(self, value: bool) -> None: ...
    def serialize(self) -> str:
        """Serialize component to JSON string."""
        ...
    def deserialize(self, json_str: str) -> None:
        """Deserialize component from JSON string."""
        ...


class Transform(Component):
    """Transform component — position, rotation, scale.
    
    Follows Unity convention:
      - position / euler_angles → world space
      - local_position / local_euler_angles / local_scale → local (parent) space
    """

    # ---- World-space properties ----
    @property
    def position(self) -> Vector3:
        """Position in world space (considering parent hierarchy)."""
        ...
    @position.setter
    def position(self, value: Vector3) -> None: ...
    @property
    def euler_angles(self) -> Vector3:
        """Rotation as Euler angles (degrees) in world space."""
        ...
    @euler_angles.setter
    def euler_angles(self, value: Vector3) -> None: ...

    # ---- Local-space properties ----
    @property
    def local_position(self) -> Vector3:
        """Position in local (parent) space."""
        ...
    @local_position.setter
    def local_position(self, value: Vector3) -> None: ...
    @property
    def local_euler_angles(self) -> Vector3:
        """Rotation as Euler angles (degrees) in local space."""
        ...
    @local_euler_angles.setter
    def local_euler_angles(self, value: Vector3) -> None: ...
    @property
    def local_scale(self) -> Vector3:
        """Scale in local space."""
        ...
    @local_scale.setter
    def local_scale(self, value: Vector3) -> None: ...
    @property
    def lossy_scale(self) -> Vector3:
        """Approximate world-space scale (read-only, like Unity lossyScale)."""
        ...

    # ---- Direction vectors (world) ----
    @property
    def forward(self) -> Vector3:
        """Forward direction in world space (negative Z)."""
        ...
    @property
    def right(self) -> Vector3:
        """Right direction in world space (positive X)."""
        ...
    @property
    def up(self) -> Vector3:
        """Up direction in world space (positive Y)."""
        ...

    # ---- Direction vectors (local) ----
    @property
    def local_forward(self) -> Vector3:
        """Forward direction in local space (negative Z)."""
        ...
    @property
    def local_right(self) -> Vector3:
        """Right direction in local space (positive X)."""
        ...
    @property
    def local_up(self) -> Vector3:
        """Up direction in local space (positive Y)."""
        ...

    # ---- Quaternion rotation (Unity: transform.rotation / transform.localRotation) ----
    @property
    def rotation(self) -> Tuple[float, float, float, float]:
        """World-space rotation as quaternion (x, y, z, w)."""
        ...
    @rotation.setter
    def rotation(self, value: Tuple[float, float, float, float]) -> None: ...
    @property
    def local_rotation(self) -> Tuple[float, float, float, float]:
        """Local-space rotation as quaternion (x, y, z, w)."""
        ...
    @local_rotation.setter
    def local_rotation(self, value: Tuple[float, float, float, float]) -> None: ...

    # ---- Hierarchy (Unity: transform.parent, root, childCount, etc.) ----
    @property
    def parent(self) -> Optional[Transform]:
        """Parent Transform (None if root). Unity: transform.parent"""
        ...
    @parent.setter
    def parent(self, value: Optional[Transform]) -> None: ...
    @property
    def root(self) -> Transform:
        """Topmost Transform in the hierarchy. Unity: transform.root"""
        ...
    @property
    def child_count(self) -> int:
        """Number of children. Unity: transform.childCount"""
        ...
    def set_parent(self, parent: Optional[Transform], world_position_stays: bool = True) -> None:
        """Set parent Transform. Unity: transform.SetParent(parent, worldPositionStays)"""
        ...
    def get_child(self, index: int) -> Optional[Transform]:
        """Get child Transform by index. Unity: transform.GetChild(index)"""
        ...
    def find(self, name: str) -> Optional[Transform]:
        """Find child Transform by name (non-recursive). Unity: transform.Find(name)"""
        ...
    def detach_children(self) -> None:
        """Unparent all children. Unity: transform.DetachChildren()"""
        ...
    def is_child_of(self, parent: Transform) -> bool:
        """Is this transform a child of parent? Unity: transform.IsChildOf(parent)"""
        ...
    def get_sibling_index(self) -> int:
        """Get sibling index. Unity: transform.GetSiblingIndex()"""
        ...
    def set_sibling_index(self, index: int) -> None:
        """Set sibling index. Unity: transform.SetSiblingIndex(index)"""
        ...
    def set_as_first_sibling(self) -> None:
        """Move to first sibling. Unity: transform.SetAsFirstSibling()"""
        ...
    def set_as_last_sibling(self) -> None:
        """Move to last sibling. Unity: transform.SetAsLastSibling()"""
        ...

    # ---- Space conversion methods ----
    def transform_point(self, point: Vector3) -> Vector3:
        """Transform point from local to world space. Unity: transform.TransformPoint(point)"""
        ...
    def inverse_transform_point(self, point: Vector3) -> Vector3:
        """Transform point from world to local space. Unity: transform.InverseTransformPoint(point)"""
        ...
    def transform_direction(self, direction: Vector3) -> Vector3:
        """Transform direction from local to world (rotation only). Unity: transform.TransformDirection(dir)"""
        ...
    def inverse_transform_direction(self, direction: Vector3) -> Vector3:
        """Transform direction from world to local (rotation only). Unity: transform.InverseTransformDirection(dir)"""
        ...
    def transform_vector(self, vector: Vector3) -> Vector3:
        """Transform vector from local to world (with scale). Unity: transform.TransformVector(vec)"""
        ...
    def inverse_transform_vector(self, vector: Vector3) -> Vector3:
        """Transform vector from world to local (with scale). Unity: transform.InverseTransformVector(vec)"""
        ...

    # ---- Matrices ----
    def local_to_world_matrix(self) -> List[float]:
        """Get local-to-world transformation matrix (16 floats, column-major). Unity: transform.localToWorldMatrix"""
        ...
    def world_to_local_matrix(self) -> List[float]:
        """Get world-to-local transformation matrix (16 floats, column-major). Unity: transform.worldToLocalMatrix"""
        ...

    # ---- Methods ----
    def look_at(self, target: Vector3) -> None:
        """Rotate to face a world-space target position."""
        ...
    def translate(self, delta: Vector3) -> None:
        """Translate in world space."""
        ...
    def translate_local(self, delta: Vector3) -> None:
        """Translate in local space (relative to own axes)."""
        ...
    def rotate(self, euler: Vector3, space: int = Space.Self) -> None:
        """Rotate by Euler angles (degrees). space: Space.Self (default) or Space.World."""
        ...
    def rotate_around(self, point: Vector3, axis: Vector3, angle: float) -> None:
        """Rotate around a world-space point. Unity: transform.RotateAround(point, axis, angle)"""
        ...

    # ---- hasChanged ----
    @property
    def has_changed(self) -> bool:
        """Has the transform changed since last reset? Unity: transform.hasChanged"""
        ...
    @has_changed.setter
    def has_changed(self, value: bool) -> None: ...



class InfMesh:
    """Runtime mesh asset — the loaded representation of a 3D model (.fbx, .obj, .gltf, …)."""

    @property
    def name(self) -> str:
        """Mesh asset name."""
        ...
    @property
    def guid(self) -> str:
        """Mesh asset GUID."""
        ...
    @property
    def file_path(self) -> str:
        """Source file path."""
        ...
    @property
    def vertex_count(self) -> int:
        """Total vertex count."""
        ...
    @property
    def index_count(self) -> int:
        """Total index count."""
        ...
    @property
    def submesh_count(self) -> int:
        """Number of submeshes."""
        ...
    @property
    def material_slot_count(self) -> int:
        """Number of material slots."""
        ...
    @property
    def material_slot_names(self) -> List[str]:
        """Material slot names from model file."""
        ...
    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get AABB as (minX, minY, minZ, maxX, maxY, maxZ)."""
        ...
    def get_submesh_info(self, index: int) -> dict:
        """Get submesh info as dict (name, index_start, index_count, vertex_start, vertex_count, material_slot, bounds_min, bounds_max)."""
        ...


class MeshRenderer(Component):
    """Renders a mesh with a material."""

    def has_inline_mesh(self) -> bool:
        """Check if this renderer uses an inline (non-resource) mesh."""
        ...
    @property
    def inline_mesh_name(self) -> str:
        """Display name for inline (primitive) meshes, e.g. 'Cube', 'Sphere'."""
        ...
    @inline_mesh_name.setter
    def inline_mesh_name(self, value: str) -> None: ...
    def has_render_material(self) -> bool:
        """Check if a custom material is assigned."""
        ...
    @property
    def render_material(self) -> Optional[InfMaterial]:
        """The material used for rendering."""
        ...
    @render_material.setter
    def render_material(self, value: Optional[InfMaterial]) -> None: ...
    def get_effective_material(self) -> Optional[InfMaterial]:
        """Get the effective material (custom or default)."""
        ...
    @property
    def vertex_count(self) -> int:
        """Number of vertices in inline mesh (0 if using resource mesh)."""
        ...
    @property
    def index_count(self) -> int:
        """Number of indices in inline mesh (0 if using resource mesh)."""
        ...
    def get_positions(self) -> List[Tuple[float, float, float]]:
        """Get all vertex positions as (x, y, z) tuples."""
        ...
    def get_normals(self) -> List[Tuple[float, float, float]]:
        """Get all vertex normals as (x, y, z) tuples."""
        ...
    def get_uvs(self) -> List[Tuple[float, float]]:
        """Get all vertex UVs as (u, v) tuples."""
        ...
    def get_indices(self) -> List[int]:
        """Get all indices as a flat list."""
        ...
    @property
    def casts_shadows(self) -> bool:
        """Whether this renderer casts shadows."""
        ...
    @casts_shadows.setter
    def casts_shadows(self, value: bool) -> None: ...
    @property
    def receives_shadows(self) -> bool:
        """Whether this renderer receives shadows."""
        ...
    @receives_shadows.setter
    def receives_shadows(self, value: bool) -> None: ...
    @property
    def submesh_index(self) -> int:
        """Submesh index to render (-1 = all, >= 0 = specific submesh)."""
        ...
    @submesh_index.setter
    def submesh_index(self, value: int) -> None: ...
    @property
    def mesh_pivot_offset(self) -> Vector3:
        """Pivot offset to re-center submesh geometry around the transform."""
        ...
    @mesh_pivot_offset.setter
    def mesh_pivot_offset(self, value: Vector3) -> None: ...
    def serialize(self) -> str:
        """Serialize the mesh renderer to JSON."""
        ...


class Light(Component):
    """Light component (Directional, Point, Spot, Area)."""

    @property
    def light_type(self) -> LightType:
        """Type of light."""
        ...
    @light_type.setter
    def light_type(self, value: LightType) -> None: ...
    @property
    def color(self) -> Vector3:
        """Light color (linear RGB)."""
        ...
    @color.setter
    def color(self, value: Vector3) -> None: ...
    @property
    def intensity(self) -> float:
        """Light intensity multiplier."""
        ...
    @intensity.setter
    def intensity(self, value: float) -> None: ...
    @property
    def range(self) -> float:
        """Light range (Point/Spot lights)."""
        ...
    @range.setter
    def range(self, value: float) -> None: ...
    @property
    def spot_angle(self) -> float:
        """Inner spot angle in degrees."""
        ...
    @spot_angle.setter
    def spot_angle(self, value: float) -> None: ...
    @property
    def outer_spot_angle(self) -> float:
        """Outer spot angle in degrees."""
        ...
    @outer_spot_angle.setter
    def outer_spot_angle(self, value: float) -> None: ...
    @property
    def shadows(self) -> LightShadows:
        """Shadow type (None, Hard, Soft)."""
        ...
    @shadows.setter
    def shadows(self, value: LightShadows) -> None: ...
    @property
    def shadow_strength(self) -> float:
        """Shadow strength (0-1)."""
        ...
    @shadow_strength.setter
    def shadow_strength(self, value: float) -> None: ...
    @property
    def shadow_bias(self) -> float:
        """Shadow depth bias."""
        ...
    @shadow_bias.setter
    def shadow_bias(self, value: float) -> None: ...
    def get_light_view_matrix(self) -> List[float]:
        """Get the light's view matrix for shadow mapping (16 floats, column-major)."""
        ...
    def get_light_projection_matrix(
        self,
        shadow_extent: float = 20.0,
        near_plane: float = 0.1,
        far_plane: float = 100.0,
    ) -> List[float]:
        """Get the light's projection matrix for shadow mapping (16 floats, column-major)."""
        ...
    def serialize(self) -> str:
        """Serialize the light component to JSON."""
        ...


class PyComponentProxy(Component):
    """C++ proxy wrapping a Python InfComponent instance."""

    def get_py_component(self) -> Any:
        """Get the underlying Python component."""
        ...
    def get_py_type_name(self) -> str:
        """Get the Python type name."""
        ...
    def is_valid(self) -> bool:
        """Check if this proxy holds a valid Python component."""
        ...


# =============================================================================
# Audio — AudioClip, AudioSource, AudioListener, AudioEngine
# =============================================================================


class AudioClip:
    """Loaded audio clip data (WAV). Use load_from_file() to load."""

    def __init__(self) -> None: ...
    def load_from_file(self, file_path: str) -> bool:
        """Load audio data from a WAV file. Returns True on success."""
        ...
    def unload(self) -> None:
        """Unload audio data and free memory."""
        ...
    @property
    def is_loaded(self) -> bool:
        """Whether the clip has loaded data."""
        ...
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        ...
    @property
    def sample_count(self) -> int:
        """Total sample frames."""
        ...
    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        ...
    @property
    def channels(self) -> int:
        """Number of channels (1=mono, 2=stereo)."""
        ...
    @property
    def file_path(self) -> str:
        """Source file path."""
        ...
    @property
    def name(self) -> str:
        """Clip name (filename without extension)."""
        ...


class AudioSource(Component):
    """Audio playback component. Attach to a GameObject to play AudioClips."""

    def play(self) -> None:
        """Start playing the assigned clip."""
        ...
    def stop(self) -> None:
        """Stop playback and reset position."""
        ...
    def pause(self) -> None:
        """Pause playback."""
        ...
    def un_pause(self) -> None:
        """Resume paused playback."""
        ...
    @property
    def is_playing(self) -> bool:
        """Whether currently playing."""
        ...
    @property
    def is_paused(self) -> bool:
        """Whether paused."""
        ...
    @property
    def clip(self) -> Optional[AudioClip]:
        """The audio clip to play."""
        ...
    @clip.setter
    def clip(self, value: Optional[AudioClip]) -> None: ...
    @property
    def volume(self) -> float:
        """Playback volume (0.0 to 1.0)."""
        ...
    @volume.setter
    def volume(self, value: float) -> None: ...
    @property
    def pitch(self) -> float:
        """Playback pitch multiplier."""
        ...
    @pitch.setter
    def pitch(self, value: float) -> None: ...
    @property
    def mute(self) -> bool:
        """Whether the audio source is muted."""
        ...
    @mute.setter
    def mute(self, value: bool) -> None: ...
    @property
    def loop(self) -> bool:
        """Whether the clip loops on completion."""
        ...
    @loop.setter
    def loop(self, value: bool) -> None: ...
    @property
    def play_on_awake(self) -> bool:
        """Whether the audio starts playing on Awake."""
        ...
    @play_on_awake.setter
    def play_on_awake(self, value: bool) -> None: ...
    @property
    def min_distance(self) -> float:
        """Within this distance the sound is at full volume."""
        ...
    @min_distance.setter
    def min_distance(self, value: float) -> None: ...
    @property
    def max_distance(self) -> float:
        """Beyond this distance the sound is inaudible."""
        ...
    @max_distance.setter
    def max_distance(self, value: float) -> None: ...
    @property
    def output_bus(self) -> str:
        """The audio mixer bus name for output routing."""
        ...
    @output_bus.setter
    def output_bus(self, value: str) -> None: ...
    @property
    def game_object_id(self) -> int:
        """ID of the owning GameObject."""
        ...
    def serialize(self) -> str:
        """Serialize the audio source to JSON."""
        ...
    def deserialize(self, json_str: str) -> None:
        """Deserialize the audio source from JSON."""
        ...


class AudioListener(Component):
    """Audio listener component — the 'ears' in the scene."""

    @property
    def game_object_id(self) -> int:
        """ID of the owning GameObject."""
        ...
    def serialize(self) -> str:
        """Serialize the listener to JSON."""
        ...
    def deserialize(self, json_str: str) -> None:
        """Deserialize the listener from JSON."""
        ...


class AudioEngine:
    """Core audio engine (singleton). Access via AudioEngine.instance()."""

    @staticmethod
    def instance() -> "AudioEngine":
        """Get the singleton AudioEngine instance."""
        ...
    def initialize(self) -> bool:
        """Initialize the audio subsystem."""
        ...
    def shutdown(self) -> None:
        """Shutdown the audio subsystem."""
        ...
    @property
    def is_initialized(self) -> bool:
        """Whether the audio engine is initialized."""
        ...
    @property
    def master_volume(self) -> float:
        """The master volume (0.0 to 1.0)."""
        ...
    @master_volume.setter
    def master_volume(self, value: float) -> None: ...
    def pause_all(self) -> None:
        """Pause all audio playback."""
        ...
    def resume_all(self) -> None:
        """Resume all audio playback."""
        ...
    @property
    def is_paused(self) -> bool:
        """Whether all audio is paused."""
        ...
    @property
    def sample_rate(self) -> int:
        """Audio output sample rate in Hz."""
        ...
    @property
    def channel_count(self) -> int:
        """Number of audio output channels."""
        ...


# =============================================================================
# Scene — GameObject, Scene, SceneManager
# =============================================================================


class GameObject:
    """A game object in the scene hierarchy with components."""

    @property
    def name(self) -> str:
        """The name of this GameObject."""
        ...
    @name.setter
    def name(self, value: str) -> None: ...
    @property
    def active(self) -> bool:
        """Whether this GameObject is active."""
        ...
    @active.setter
    def active(self, value: bool) -> None: ...
    @property
    def id(self) -> int:
        """Unique object ID."""
        ...
    @property
    def transform(self) -> Transform:
        """Get the Transform component."""
        ...
    def get_transform(self) -> Transform:
        """Get the Transform component."""
        ...
    def add_component(self, component_type: Union[str, type]) -> Optional[Component]:
        """Add a C++ component by type or type name."""
        ...
    def remove_component(self, component: Component) -> bool:
        """Remove a component instance (cannot remove Transform)."""
        ...
    def get_components(self) -> List[Component]:
        """Get all components (including Transform)."""
        ...
    def get_component(self, type_name: str) -> Optional[Component]:
        """Get a component by type name (e.g., 'Transform', 'MeshRenderer', 'Light')."""
        ...
    def get_cpp_component(self, type_name: str) -> Optional[Component]:
        """Get a C++ component by type name (e.g., 'Transform', 'MeshRenderer', 'Light')."""
        ...
    def get_cpp_components(self, type_name: str) -> List[Component]:
        """Get all C++ components of a given type name."""
        ...
    def add_py_component(self, component_instance: Any) -> Optional[Any]:
        """Add a Python InfComponent instance to this GameObject."""
        ...
    def get_py_component(self, component_type: type) -> Optional[Any]:
        """Get a Python component of the specified type."""
        ...
    def get_py_components(self) -> List[Any]:
        """Get all Python components attached to this GameObject."""
        ...
    def remove_py_component(self, component: Any) -> bool:
        """Remove a Python component instance."""
        ...
    def get_parent(self) -> Optional[GameObject]:
        """Get the parent GameObject."""
        ...
    def set_parent(self, parent: Optional[GameObject], world_position_stays: bool = True) -> None:
        """Set the parent GameObject (None for root). world_position_stays preserves world transform."""
        ...
    def get_children(self) -> List[GameObject]:
        """Get list of child GameObjects."""
        ...
    def get_child_count(self) -> int:
        """Get the number of children."""
        ...
    def is_active_in_hierarchy(self) -> bool:
        """Check if this object and all parents are active."""
        ...
    @property
    def active_self(self) -> bool:
        """Is this object itself active? Alias for active. Unity: gameObject.activeSelf"""
        ...
    @property
    def active_in_hierarchy(self) -> bool:
        """Is this object active in the hierarchy? Unity: gameObject.activeInHierarchy"""
        ...
    @property
    def is_static(self) -> bool:
        """Static flag. Unity: gameObject.isStatic"""
        ...
    @is_static.setter
    def is_static(self, value: bool) -> None: ...
    @property
    def prefab_guid(self) -> str:
        """GUID of the source .prefab asset (empty = not a prefab instance)."""
        ...
    @prefab_guid.setter
    def prefab_guid(self, value: str) -> None: ...
    @property
    def prefab_root(self) -> bool:
        """True if this object is the root of a prefab instance hierarchy."""
        ...
    @prefab_root.setter
    def prefab_root(self, value: bool) -> None: ...
    @property
    def is_prefab_instance(self) -> bool:
        """True if this object belongs to a prefab instance."""
        ...
    @property
    def scene(self) -> Optional[Scene]:
        """The Scene this GameObject belongs to. Unity: gameObject.scene"""
        ...
    def get_child(self, index: int) -> Optional[GameObject]:
        """Get child by index. Unity: transform.GetChild(index).gameObject"""
        ...
    def find_child(self, name: str) -> Optional[GameObject]:
        """Find direct child by name (non-recursive)."""
        ...
    def find_descendant(self, name: str) -> Optional[GameObject]:
        """Find descendant by name (recursive depth-first search)."""
        ...
    @property
    def tag(self) -> str:
        """Tag string for this GameObject."""
        ...
    @tag.setter
    def tag(self, value: str) -> None: ...
    @property
    def layer(self) -> int:
        """Layer index (0-31) for this GameObject."""
        ...
    @layer.setter
    def layer(self, value: int) -> None: ...
    def compare_tag(self, tag: str) -> bool:
        """Returns True if this GameObject's tag matches the given tag."""
        ...
    def serialize(self) -> str:
        """Serialize GameObject to JSON string."""
        ...
    def deserialize(self, json_str: str) -> None:
        """Deserialize GameObject from JSON string."""
        ...


class PendingPyComponent:
    """Data for a Python component awaiting restoration after deserialization."""

    @property
    def game_object_id(self) -> int:
        """ID of the GameObject this component belongs to."""
        ...
    @property
    def type_name(self) -> str:
        """Python type name of the component."""
        ...
    @property
    def script_guid(self) -> str:
        """GUID of the script asset."""
        ...
    @property
    def fields_json(self) -> str:
        """Serialized field data as a JSON string."""
        ...
    @property
    def enabled(self) -> bool:
        """Whether the component was enabled."""
        ...


class Scene:
    """A scene containing GameObjects."""

    @property
    def name(self) -> str:
        """The name of this scene."""
        ...
    @name.setter
    def name(self, value: str) -> None: ...
    def create_game_object(self, name: str = "GameObject") -> GameObject:
        """Create a new empty GameObject in this scene."""
        ...
    def create_primitive(self, type: PrimitiveType, name: str = "") -> GameObject:
        """Create a primitive GameObject (Cube, Sphere, Capsule, Cylinder, Plane)."""
        ...
    def get_root_objects(self) -> List[GameObject]:
        """Get all root-level GameObjects."""
        ...
    def get_all_objects(self) -> List[GameObject]:
        """Get all GameObjects in the scene."""
        ...
    def find(self, name: str) -> Optional[GameObject]:
        """Find a GameObject by name."""
        ...
    def find_by_id(self, id: int) -> Optional[GameObject]:
        """Find a GameObject by ID."""
        ...
    def find_with_tag(self, tag: str) -> Optional[GameObject]:
        """Find the first GameObject with a given tag."""
        ...
    def find_game_objects_with_tag(self, tag: str) -> List[GameObject]:
        """Find all GameObjects with a given tag."""
        ...
    def find_game_objects_in_layer(self, layer: int) -> List[GameObject]:
        """Find all GameObjects in a given layer."""
        ...
    def destroy_game_object(self, game_object: GameObject) -> None:
        """Destroy a GameObject (removed at end of frame)."""
        ...
    def instantiate_game_object(self, source: GameObject, parent: Optional[GameObject] = None) -> Optional[GameObject]:
        """Clone a GameObject (deep copy). Unity: Object.Instantiate()."""
        ...
    def instantiate_from_json(self, json_str: str, parent: Optional[GameObject] = None) -> Optional[GameObject]:
        """Instantiate a GameObject hierarchy from a JSON string (e.g. prefab). Fresh IDs are assigned."""
        ...
    def create_from_model(self, guid: str, name: str = "") -> Optional[GameObject]:
        """Create a GameObject from a mesh asset GUID."""
        ...
    def process_pending_destroys(self) -> None:
        """Process pending GameObject destroys."""
        ...
    def is_playing(self) -> bool:
        """Check if the scene is in play mode."""
        ...
    def start(self) -> None:
        """Trigger Awake+Start on all components (idempotent — skipped if already started)."""
        ...
    def awake_object(self, game_object: GameObject) -> None:
        """Re-run Awake+OnEnable on a GameObject and its descendants (used after undo deserialization)."""
        ...
    def serialize(self) -> str:
        """Serialize scene to JSON string."""
        ...
    def deserialize(self, json_str: str) -> None:
        """Deserialize scene from JSON string."""
        ...
    def save_to_file(self, path: str) -> None:
        """Save scene to a JSON file."""
        ...
    def load_from_file(self, path: str) -> None:
        """Load scene from a JSON file."""
        ...
    def has_pending_py_components(self) -> bool:
        """Check if there are pending Python components to restore."""
        ...
    def take_pending_py_components(self) -> List[PendingPyComponent]:
        """Get and clear pending Python components for restoration."""
        ...


class SceneManager:
    """Singleton scene manager controlling scene lifecycle and play mode."""

    @staticmethod
    def instance() -> SceneManager:
        """Get the singleton SceneManager instance."""
        ...
    def create_scene(self, name: str) -> Scene:
        """Create a new empty scene."""
        ...
    def get_active_scene(self) -> Optional[Scene]:
        """Get the currently active scene."""
        ...
    def set_active_scene(self, scene: Scene) -> None:
        """Set the active scene."""
        ...
    def get_scene(self, name: str) -> Optional[Scene]:
        """Get a scene by name."""
        ...
    def is_playing(self) -> bool:
        """Check if in play mode."""
        ...
    def play(self) -> None:
        """Enter play mode."""
        ...
    def stop(self) -> None:
        """Stop play mode."""
        ...
    def pause(self) -> None:
        """Pause play mode."""
        ...
    def is_paused(self) -> bool:
        """Check if paused."""
        ...
    def step(self, delta_time: float = 0.016) -> None:
        """Execute one frame while paused (FixedUpdate + Update + LateUpdate + EndFrame). No-op if not paused."""
        ...
    def dont_destroy_on_load(self, game_object: GameObject) -> None:
        """Mark a root GameObject so it survives scene switches. Unity: DontDestroyOnLoad()"""
        ...


# =============================================================================
# Resources
# =============================================================================


class ResourceMeta:
    """Resource metadata (GUID, name, type, custom key-value data)."""

    def get_resource_name(self) -> str:
        """Get resource filename without extension."""
        ...
    def get_hash_code(self) -> int:
        """Get internal hash code."""
        ...
    def get_guid(self) -> str:
        """Get stable GUID for this resource."""
        ...
    def get_resource_type(self) -> ResourceType:
        """Get the resource type."""
        ...
    def has_key(self, key: str) -> bool:
        """Check if metadata has a specific key."""
        ...
    def get_string(self, key: str) -> str:
        """Get a string metadata value."""
        ...
    def get_int(self, key: str) -> int:
        """Get an integer metadata value."""
        ...
    def get_float(self, key: str) -> float:
        """Get a float metadata value."""
        ...


class TextureData:
    """Raw texture pixel data loaded from file or created programmatically."""

    @property
    def width(self) -> int:
        """Texture width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Texture height in pixels."""
        ...
    @property
    def channels(self) -> int:
        """Number of color channels (always 4 for RGBA)."""
        ...
    @property
    def name(self) -> str:
        """Texture name/identifier."""
        ...
    @property
    def source_path(self) -> str:
        """Original file path."""
        ...
    def is_valid(self) -> bool:
        """Check if texture data is valid."""
        ...
    def get_size_bytes(self) -> int:
        """Get total size in bytes."""
        ...
    def get_pixels(self) -> bytes:
        """Get raw pixel data as bytes (RGBA format)."""
        ...
    def get_pixels_list(self) -> List[int]:
        """Get raw pixel data as list of unsigned char (for upload_texture_for_imgui)."""
        ...


class TextureLoader:
    """Static methods for loading textures."""

    @staticmethod
    def load_from_file(file_path: str, name: str = "") -> TextureData:
        """Load texture from file."""
        ...
    @staticmethod
    def load_from_memory(data: bytes, name: str = "") -> TextureData:
        """Load texture from memory buffer."""
        ...
    @staticmethod
    def create_solid_color(
        width: int, height: int, r: int, g: int, b: int, a: int, name: str = "solid_color"
    ) -> TextureData:
        """Create a solid color texture."""
        ...
    @staticmethod
    def create_checkerboard(
        width: int, height: int, checker_size: int = 8, name: str = "checkerboard"
    ) -> TextureData:
        """Create a checkerboard texture."""
        ...


class InfMaterial:
    """Material definition with shader and property bindings."""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, name: str) -> None: ...
    @overload
    def __init__(self, name: str, vert_shader_path: str, frag_shader_path: str) -> None: ...
    @property
    def name(self) -> str:
        """Material name."""
        ...
    @name.setter
    def name(self, value: str) -> None: ...
    @property
    def guid(self) -> str:
        """Material GUID."""
        ...
    @guid.setter
    def guid(self, value: str) -> None: ...
    @property
    def file_path(self) -> str:
        """File path for saving."""
        ...
    @file_path.setter
    def file_path(self, value: str) -> None: ...
    @property
    def is_builtin(self) -> bool:
        """Whether this is a built-in material."""
        ...
    @is_builtin.setter
    def is_builtin(self, value: bool) -> None: ...
    @property
    def shader_name(self) -> str:
        """Shader name (e.g. 'lit', 'unlit') — sets both vert and frag."""
        ...
    @shader_name.setter
    def shader_name(self, value: str) -> None: ...
    @property
    def vert_shader_name(self) -> str:
        """Vertex shader name."""
        ...
    @vert_shader_name.setter
    def vert_shader_name(self, value: str) -> None: ...
    @property
    def frag_shader_name(self) -> str:
        """Fragment shader name."""
        ...
    @frag_shader_name.setter
    def frag_shader_name(self, value: str) -> None: ...
    def set_shader(self, shader_name: str) -> None:
        """Set the material's shader by name (sets both vert and frag)."""
        ...
    def get_render_queue(self) -> int:
        """Get the render queue value."""
        ...
    def set_render_queue(self, queue: int) -> None:
        """Set the render queue value."""
        ...
    def save(self) -> None:
        """Save material to its file path."""
        ...
    def save_to(self, path: str) -> None:
        """Save material to specified path."""
        ...
    def set_float(self, name: str, value: float) -> None:
        """Set a float property."""
        ...
    def set_vector2(self, name: str, value: Union[Tuple[float, float], Sequence[float]]) -> None:
        """Set a vec2 property."""
        ...
    def set_vector3(
        self, name: str, value: Union[Tuple[float, float, float], Sequence[float]]
    ) -> None:
        """Set a vec3 property."""
        ...
    def set_vector4(
        self, name: str, value: Union[Tuple[float, float, float, float], Sequence[float]]
    ) -> None:
        """Set a vec4 property."""
        ...
    def set_color(
        self, name: str, color: Union[Tuple[float, float, float, float], Sequence[float]]
    ) -> None:
        """Set a color property (vec4)."""
        ...
    def set_int(self, name: str, value: int) -> None:
        """Set an int property."""
        ...
    def set_matrix(self, name: str, value: List[float]) -> None:
        """Set a mat4 property."""
        ...
    def set_texture(self, name: str, texture_path: str) -> None:
        """Set a texture property."""
        ...
    def has_property(self, name: str) -> bool:
        """Check if material has a property."""
        ...
    def get_property(self, name: str) -> Optional[Any]:
        """Get a property value by name."""
        ...
    def get_all_properties(self) -> dict:
        """Get all properties as a dictionary."""
        ...
    def is_pipeline_dirty(self) -> bool:
        """Check if pipeline needs recreation."""
        ...
    def clear_pipeline_dirty(self) -> None:
        """Clear the pipeline dirty flag."""
        ...
    def get_pipeline_hash(self) -> int:
        """Get a hash of the pipeline configuration."""
        ...
    def serialize(self) -> str:
        """Serialize material to JSON string."""
        ...
    def deserialize(self, json_str: str) -> None:
        """Deserialize material from JSON string."""
        ...
    @staticmethod
    def create_default_lit() -> InfMaterial:
        """Create the default lit opaque material (built-in)."""
        ...
    @staticmethod
    def create_default_unlit() -> InfMaterial:
        """Create a default unlit opaque material."""
        ...
    def get_render_state(self) -> RenderState:
        """Get a copy of the material's render state."""
        ...
    def set_render_state(self, state: RenderState) -> None:
        """Set the material's render state."""
        ...
    @property
    def render_state_overrides(self) -> int:
        """Bitmask of user-overridden RenderState fields."""
        ...
    @render_state_overrides.setter
    def render_state_overrides(self, value: int) -> None: ...
    def mark_override(self, flag: RenderStateOverride) -> None:
        """Mark a RenderState field as user-overridden."""
        ...
    def clear_override(self, flag: RenderStateOverride) -> None:
        """Clear a RenderState field override (revert to shader default)."""
        ...
    def has_override(self, flag: RenderStateOverride) -> bool:
        """Check if a RenderState field is user-overridden."""
        ...


class RenderStateOverride:
    """Bitmask flags for per-material render state overrides."""

    NONE: int
    """No overrides."""
    CULL_MODE: int
    """Override cull mode."""
    DEPTH_WRITE: int
    """Override depth write."""
    DEPTH_TEST: int
    """Override depth test."""
    DEPTH_COMPARE_OP: int
    """Override depth compare operation."""
    BLEND_ENABLE: int
    """Override blend enable."""
    BLEND_MODE: int
    """Override blend mode."""
    RENDER_QUEUE: int
    """Override render queue."""
    SURFACE_TYPE: int
    """Override surface type."""
    ALPHA_CLIP: int
    """Override alpha clipping."""


class RenderState:
    """GPU pipeline state for materials."""

    cull_mode: int
    """Face culling mode."""
    front_face: int
    """Front face winding order."""
    polygon_mode: int
    """Polygon rasterization mode."""
    line_width: float
    """Line width for wireframe rendering."""
    depth_test_enable: bool
    """Whether depth testing is enabled."""
    depth_write_enable: bool
    """Whether depth writing is enabled."""
    depth_compare_op: int
    """Depth comparison operation."""
    blend_enable: bool
    """Whether alpha blending is enabled."""
    src_color_blend_factor: int
    """Source color blend factor."""
    dst_color_blend_factor: int
    """Destination color blend factor."""
    color_blend_op: int
    """Color blend operation."""
    src_alpha_blend_factor: int
    """Source alpha blend factor."""
    dst_alpha_blend_factor: int
    """Destination alpha blend factor."""
    alpha_blend_op: int
    """Alpha blend operation."""
    render_queue: int
    """Render queue value for sorting."""


class MaterialManager:
    """Thin cache of engine built-in material pointers.  Real ownership is in AssetRegistry."""

    @staticmethod
    def instance() -> MaterialManager:
        """Get the MaterialManager singleton."""
        ...
    def load_default_material_from_file(self, mat_file_path: str) -> bool:
        """Load default material from a .mat file."""
        ...


class AssetRegistry:
    """Unified asset cache — owns all loaded C++ asset instances."""

    @staticmethod
    def instance() -> AssetRegistry:
        """Get the AssetRegistry singleton."""
        ...
    def is_initialized(self) -> bool:
        """Whether the asset registry is initialized."""
        ...
    def get_asset_database(self) -> Optional[AssetDatabase]:
        """Get the underlying asset database."""
        ...
    # Material
    def load_material(self, path: str) -> Optional[InfMaterial]:
        """Load a material by file path."""
        ...
    def load_material_by_guid(self, guid: str) -> Optional[InfMaterial]:
        """Load a material by its GUID."""
        ...
    def get_material(self, guid: str) -> Optional[InfMaterial]:
        """Get a cached material by GUID."""
        ...
    def get_builtin_material(self, key: str) -> Optional[InfMaterial]:
        """Get a built-in material by key (e.g. 'DefaultLit')."""
        ...
    # Mesh
    def load_mesh(self, path: str) -> Optional[InfMesh]:
        """Load a mesh by file path (.fbx, .obj, .gltf, …)."""
        ...
    def load_mesh_by_guid(self, guid: str) -> Optional[InfMesh]:
        """Load a mesh by its GUID."""
        ...
    def get_mesh(self, guid: str) -> Optional[InfMesh]:
        """Get a cached mesh by GUID."""
        ...
    # Hot-reload
    def reload_asset(self, guid: str) -> bool:
        """Reload an asset by GUID. Returns True on success."""
        ...
    def invalidate_asset(self, guid: str) -> None:
        """Mark an asset as needing reload."""
        ...
    def remove_asset(self, guid: str) -> None:
        """Remove an asset from the cache."""
        ...
    # File events
    def on_asset_modified(self, path: str) -> None:
        """Notify that an asset file was modified."""
        ...
    def on_asset_moved(self, old_path: str, new_path: str) -> None:
        """Notify that an asset file was moved."""
        ...
    def on_asset_deleted(self, path: str) -> None:
        """Notify that an asset file was deleted."""
        ...
    # Queries
    def is_loaded(self, guid: str) -> bool:
        """Check if an asset is loaded in the cache."""
        ...


class AssetDatabase:
    """GUID-based asset tracking database."""

    def __init__(self) -> None: ...
    def initialize(self, project_root: str) -> None:
        """Initialize asset database with project root."""
        ...
    def refresh(self) -> None:
        """Refresh assets by scanning Assets folder."""
        ...
    def import_asset(self, path: str) -> None:
        """Import a single asset."""
        ...
    def delete_asset(self, path: str) -> None:
        """Delete asset and its meta."""
        ...
    def move_asset(self, old_path: str, new_path: str) -> None:
        """Move/rename asset preserving GUID."""
        ...
    def on_asset_created(self, path: str) -> None:
        """Notify that a new asset was created."""
        ...
    def on_asset_modified(self, path: str) -> None:
        """Notify that an asset was modified."""
        ...
    def on_asset_deleted(self, path: str) -> None:
        """Notify that an asset was deleted."""
        ...
    def on_asset_moved(self, old_path: str, new_path: str) -> None:
        """Notify that an asset was moved or renamed."""
        ...
    def contains_guid(self, guid: str) -> bool:
        """Check if a GUID is tracked."""
        ...
    def contains_path(self, path: str) -> bool:
        """Check if a path is tracked."""
        ...
    def get_guid_from_path(self, path: str) -> str:
        """Get the GUID for an asset path."""
        ...
    def get_path_from_guid(self, guid: str) -> str:
        """Get the file path for a GUID."""
        ...
    def get_meta_by_guid(self, guid: str) -> Optional[ResourceMeta]:
        """Get resource metadata by GUID."""
        ...
    def get_meta_by_path(self, path: str) -> Optional[ResourceMeta]:
        """Get resource metadata by file path."""
        ...
    def get_all_guids(self) -> List[str]:
        """Get all tracked asset GUIDs."""
        ...
    def is_asset_path(self, path: str) -> bool:
        """Check if a path is within the Assets folder."""
        ...
    @property
    def project_root(self) -> str:
        """The project root directory."""
        ...
    @property
    def assets_root(self) -> str:
        """The Assets folder path."""
        ...


# =============================================================================
# Render Pipeline
# =============================================================================


class DrawingSettings:
    """Settings controlling how renderers are drawn."""

    sorting_criteria: SortingCriteria
    """The sorting criteria for draw calls."""
    override_material_index: int
    """Index of the override material (-1 for none)."""

    def __init__(self) -> None: ...


class FilteringSettings:
    """Settings controlling which renderers pass the filter."""

    render_queue_min: int
    """Minimum render queue value."""
    render_queue_max: int
    """Maximum render queue value."""
    layer_mask: int
    """Layer mask for filtering renderers."""

    def __init__(self) -> None: ...
    @staticmethod
    def opaque() -> FilteringSettings:
        """Create settings for opaque objects (queue 0-2500)."""
        ...
    @staticmethod
    def transparent() -> FilteringSettings:
        """Create settings for transparent objects (queue 2501-5000)."""
        ...


class CullingResults:
    """Results of camera frustum culling."""

    @property
    def visible_object_count(self) -> int:
        """Number of objects visible after culling."""
        ...
    @property
    def visible_light_count(self) -> int:
        """Number of lights visible after culling."""
        ...


class CameraProjection:
    """Camera projection mode enum."""

    Perspective: CameraProjection
    Orthographic: CameraProjection

    @property
    def value(self) -> int: ...


class CameraClearFlags:
    """Camera clear flags enum (Unity URP-style)."""

    Skybox: CameraClearFlags
    SolidColor: CameraClearFlags
    DepthOnly: CameraClearFlags
    DontClear: CameraClearFlags

    @property
    def value(self) -> int: ...


class Camera:
    """Camera component with Unity-like API."""

    # Projection
    @property
    def projection_mode(self) -> CameraProjection:
        """The camera projection mode (Perspective or Orthographic)."""
        ...
    @projection_mode.setter
    def projection_mode(self, value: CameraProjection) -> None: ...

    @property
    def field_of_view(self) -> float:
        """The vertical field of view in degrees."""
        ...
    @field_of_view.setter
    def field_of_view(self, value: float) -> None: ...

    @property
    def aspect_ratio(self) -> float:
        """The aspect ratio (width / height)."""
        ...
    @aspect_ratio.setter
    def aspect_ratio(self, value: float) -> None: ...

    @property
    def orthographic_size(self) -> float:
        """Half-size of the camera in orthographic mode."""
        ...
    @orthographic_size.setter
    def orthographic_size(self, value: float) -> None: ...

    @property
    def near_clip(self) -> float:
        """The near clipping plane distance."""
        ...
    @near_clip.setter
    def near_clip(self, value: float) -> None: ...

    @property
    def far_clip(self) -> float:
        """The far clipping plane distance."""
        ...
    @far_clip.setter
    def far_clip(self, value: float) -> None: ...

    # Multi-camera
    @property
    def depth(self) -> float:
        """Camera rendering order (lower renders first)."""
        ...
    @depth.setter
    def depth(self, value: float) -> None: ...

    @property
    def culling_mask(self) -> int:
        """Layer mask for culling objects from this camera."""
        ...
    @culling_mask.setter
    def culling_mask(self, value: int) -> None: ...

    # Phase 1: Clear flags & background color
    @property
    def clear_flags(self) -> CameraClearFlags:
        """How the camera clears the background."""
        ...
    @clear_flags.setter
    def clear_flags(self, value: CameraClearFlags) -> None: ...

    @property
    def background_color(self) -> tuple[float, float, float, float]:
        """Background clear color (r, g, b, a)."""
        ...
    @background_color.setter
    def background_color(self, value: tuple[float, float, float, float]) -> None: ...

    # Phase 0: Screen dimensions (read-only, set by renderer)
    @property
    def pixel_width(self) -> int:
        """The width of the camera's render target in pixels."""
        ...
    @property
    def pixel_height(self) -> int:
        """The height of the camera's render target in pixels."""
        ...

    # Phase 0: Coordinate conversion
    def screen_to_world_point(self, x: float, y: float, depth: float = 0.0) -> tuple[float, float, float]:
        """Convert screen coordinates (x, y) + depth [0..1] to world position."""
        ...

    def world_to_screen_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """Convert world position to screen coordinates (x, y)."""
        ...

    def serialize(self) -> str:
        """Serialize camera settings to JSON."""
        ...
    def deserialize(self, json_str: str) -> bool:
        """Deserialize camera settings from JSON."""
        ...


class ScriptableRenderContext:
    """Context for executing render commands in a custom pipeline."""

    def setup_camera_properties(self, camera: Camera) -> None:
        """Set camera VP matrices for rendering."""
        ...
    def cull(self, camera: Camera) -> CullingResults:
        """Cull scene objects, return CullingResults."""
        ...
    def apply_graph(self, description: RenderGraphDescription) -> None:
        """Apply a Python-defined RenderGraph topology."""
        ...
    def submit_culling(self, culling: CullingResults) -> None:
        """Submit all culling results as full draw calls."""
        ...
    def get_camera_target(self, camera: Camera) -> None:
        """Get a handle representing the final camera render target."""
        ...
    def set_global_texture(self, name: str, handle: RenderTargetHandle) -> None:
        """Set a global texture shader parameter."""
        ...
    def set_global_float(self, name: str, value: float) -> None:
        """Set a global float shader parameter."""
        ...
    def set_global_vector(
        self, name: str, x: float, y: float, z: float, w: float
    ) -> None:
        """Set a global vec4 shader parameter."""
        ...


class RenderPipelineCallback:
    """Abstract base for custom Python render pipelines (override render())."""

    def __init__(self) -> None: ...
    def render(self, context: ScriptableRenderContext, cameras: List[Camera]) -> None:
        """Called once per frame to define rendering pass sequence."""
        ...
    def dispose(self) -> None:
        """Called when the pipeline is being replaced."""
        ...


# =============================================================================
# Render Graph
# =============================================================================


class ScenePassConfig:
    """Configuration for a single render pass."""

    name: str
    """Pass name."""
    type: ScenePassType
    """Pass type."""
    enabled: bool
    """Whether this pass is enabled."""

    clear_color: bool
    """Whether to clear the color buffer."""
    clear_depth: bool
    """Whether to clear the depth buffer."""
    clear_depth_value: float
    """Depth clear value."""
    has_own_render_target: bool
    """Whether this pass uses its own render target."""
    enable_readback: bool
    """Whether CPU readback is enabled for this pass."""
    input_passes: List[str]
    """Names of passes whose outputs are inputs to this pass."""

    def __init__(self) -> None: ...
    def set_clear_color_value(
        self, r: float, g: float, b: float, a: float = 1.0
    ) -> None:
        """Set the clear color value."""
        ...
    def get_clear_color_value(self) -> Tuple[float, float, float, float]: ...


class GraphTextureDesc:
    """Description of a texture resource in the Python-defined graph."""

    name: str
    """Texture resource name."""
    format: int
    """Vulkan format enum value."""
    is_backbuffer: bool
    """Whether this texture represents the backbuffer."""
    is_depth: bool
    """Whether this is a depth texture."""

    def __init__(self) -> None: ...


class GraphPassDesc:
    """Description of a single render pass in the Python-defined graph."""

    name: str
    """Pass name."""
    read_textures: List[str]
    """Textures read by this pass."""
    write_colors: List[Tuple[int, str]]
    """Color attachment outputs (slot, texture_name)."""
    write_depth: str
    """Depth attachment output texture name."""
    clear_color: bool
    """Whether to clear color."""
    clear_depth: bool
    """Whether to clear depth."""
    clear_color_r: float
    clear_color_g: float
    clear_color_b: float
    clear_color_a: float
    clear_depth_value: float
    action: GraphPassActionType
    """The draw action for this pass."""
    queue_min: int
    """Minimum render queue."""
    queue_max: int
    """Maximum render queue."""
    sort_mode: str
    """Draw call sorting mode."""
    input_bindings: List[Tuple[str, str]]
    """Sampler-to-texture bindings."""

    def __init__(self) -> None: ...


class RenderGraphDescription:
    """Complete render graph topology defined by Python."""

    name: str
    """Graph name."""
    textures: List[GraphTextureDesc]
    """All texture resources."""
    passes: List[GraphPassDesc]
    """All render passes in execution order."""
    output_texture: str
    """Name of the final output texture."""

    def __init__(self) -> None: ...


class SceneRenderGraph:
    """The active scene render graph — configure passes, readback pixels."""

    def remove_pass(self, name: str) -> None:
        """Remove a pass by name."""
        ...
    def set_pass_enabled(self, name: str, enabled: bool) -> None:
        """Enable/disable a pass."""
        ...
    def clear_passes(self) -> None:
        """Clear all passes."""
        ...
    def mark_dirty(self) -> None:
        """Force rebuild of the render graph on next frame."""
        ...
    def apply_python_graph(self, description: RenderGraphDescription) -> None:
        """Apply a Python-defined render graph topology."""
        ...
    def has_python_graph(self) -> bool:
        """Check if a Python graph topology has been applied."""
        ...
    def get_pass_count(self) -> int:
        """Get number of configured passes."""
        ...
    def get_debug_string(self) -> str:
        """Get debug visualization of the render graph."""
        ...


# =============================================================================
# GUI (ImGui)
# =============================================================================


class InfGUIContext:
    """ImGui context for building editor panels and tool windows."""

    # Text & labels
    def label(self, text: str) -> None: ...

    # Buttons
    def button(self, label: str, on_click: object = None, width: float = 0.0, height: float = 0.0) -> None: ...
    def radio_button(self, label: str, active: bool) -> bool: ...
    def selectable(
        self,
        label: str,
        selected: bool = False,
        flags: int = 0,
        width: float = 0.0,
        height: float = 0.0,
    ) -> bool: ...

    # Input widgets
    def checkbox(self, label: str, value: bool) -> bool: ...
    def int_slider(self, label: str, value: int, min: int, max: int) -> int: ...
    def float_slider(
        self, label: str, value: float, min: float, max: float
    ) -> float: ...
    def drag_int(
        self, label: str, value: int, speed: float, min: int, max: int
    ) -> int: ...
    def drag_float(
        self, label: str, value: float, speed: float, min: float, max: float
    ) -> float: ...
    def text_input(
        self, label: str, value: str, buffer_size: int
    ) -> str: ...
    def text_area(self, label: str, text: str) -> str: ...
    def input_text_with_hint(
        self, label: str, hint: str, text: str
    ) -> str: ...
    def input_int(
        self,
        label: str,
        value: int,
        step: int = 1,
        step_fast: int = 100,
        flags: int = 0,
    ) -> int: ...
    def input_float(
        self,
        label: str,
        value: float,
        step: float = 0.0,
        step_fast: float = 0.0,
        flags: int = 0,
    ) -> float: ...
    def color_edit(self, label: str, color: List[float]) -> Optional[List[float]]: ...
    def color_picker(
        self, label: str, r: float, g: float, b: float, a: float = 1.0, flags: int = 0
    ) -> Tuple[bool, float, float, float, float]: ...
    def vector2(
        self, label: str, x: float, y: float
    ) -> Tuple[float, float]: ...
    def vector3(
        self, label: str, x: float, y: float, z: float
    ) -> Tuple[float, float, float]: ...
    def vector4(
        self, label: str, x: float, y: float, z: float, w: float
    ) -> Tuple[float, float, float, float]: ...
    def combo(
        self,
        label: str,
        current_item: int,
        items: List[str],
        popup_max_height_in_items: int = -1,
    ) -> int: ...
    def list_box(
        self,
        label: str,
        current_item: int,
        items: List[str],
        height_in_items: int = -1,
    ) -> int: ...
    def progress_bar(self, fraction: float) -> None: ...

    # Layout
    def begin_group(self, name: str = "") -> None: ...
    def end_group(self) -> None: ...
    def same_line(
        self, offset_from_start_x: float = 0.0, spacing: float = -1.0
    ) -> None: ...
    def separator(self) -> None: ...
    def spacing(self) -> None: ...
    def dummy(self, width: float, height: float) -> None: ...
    def new_line(self) -> None: ...

    # Tree
    def tree_node(self, label: str) -> bool: ...
    def tree_node_ex(self, label: str, flags: int) -> bool: ...
    def tree_pop(self) -> None: ...
    def set_next_item_open(self, is_open: bool, cond: int = 0) -> None: ...
    def set_next_item_allow_overlap(self) -> None: ...
    def collapsing_header(self, label: str) -> bool: ...
    def is_item_clicked(self, mouse_button: int = 0) -> bool: ...

    # Tabs
    def begin_tab_bar(self, id: str) -> bool: ...
    def end_tab_bar(self) -> None: ...
    def begin_tab_item(self, label: str) -> bool: ...
    def end_tab_item(self) -> None: ...

    # Menu
    def begin_main_menu_bar(self) -> bool: ...
    def end_main_menu_bar(self) -> None: ...
    def begin_menu(self, label: str, enabled: bool = True) -> bool: ...
    def end_menu(self) -> None: ...
    def menu_item(self, label: str) -> bool: ...

    # Child windows
    def begin_child(self, id: str, width: float = 0, height: float = 0) -> bool: ...
    def end_child(self) -> None: ...

    # Popups
    def open_popup(self, id: str) -> None: ...
    def begin_popup(self, id: str) -> bool: ...
    def begin_popup_modal(self, title: str, flags: int = 0) -> bool:
        """Open a modal popup. Returns true while open. Must call end_popup() when true."""
        ...
    def begin_popup_context_item(
        self, id: str = "", mouse_button: int = 1
    ) -> bool: ...
    def begin_popup_context_window(
        self, id: str = "", mouse_button: int = 1
    ) -> bool: ...
    def end_popup(self) -> None: ...

    # Tooltips
    def begin_tooltip(self) -> None: ...
    def end_tooltip(self) -> None: ...
    def set_tooltip(self, text: str) -> None: ...

    # Images
    def image(
        self,
        texture_id: int,
        width: float,
        height: float,
        uv0_x: float = 0.0,
        uv0_y: float = 0.0,
        uv1_x: float = 1.0,
        uv1_y: float = 1.0,
    ) -> None: ...
    def image_button(
        self,
        id: str,
        texture_id: int,
        width: float,
        height: float,
        uv0_x: float = 0.0,
        uv0_y: float = 0.0,
        uv1_x: float = 1.0,
        uv1_y: float = 1.0,
    ) -> bool: ...

    # Tables
    def begin_table(self, id: str, columns: int) -> bool: ...
    def end_table(self) -> None: ...
    def table_setup_column(self, label: str) -> None: ...
    def table_headers_row(self) -> None: ...
    def table_next_row(self) -> None: ...
    def table_set_column_index(self, column: int) -> None: ...
    def table_next_column(self) -> None: ...
    def checkbox_flags(self, label: str, flags: int, flag_value: int) -> int: ...

    # Size & position
    def set_next_item_width(self, width: float) -> None: ...
    def set_next_window_size(self, width: float, height: float, cond: int = ...) -> None: ...
    def set_next_window_pos(self, x: float, y: float, cond: int = ..., pivot_x: float = ..., pivot_y: float = ...) -> None: ...
    def set_next_window_focus(self) -> None: ...

    # Windows
    def begin_window(self, name: str, *args: Any) -> bool: ...
    def begin_window_closable(
        self, name: str, is_open: bool = True, flags: int = 0
    ) -> Tuple[bool, bool]:
        """Returns (is_visible, is_open)."""
        ...
    def end_window(self) -> None: ...

    # Layout queries
    def get_content_region_avail_width(self) -> float: ...
    def get_content_region_avail_height(self) -> float: ...
    def get_cursor_pos_x(self) -> float: ...
    def get_cursor_pos_y(self) -> float: ...
    def set_cursor_pos_x(self, x: float) -> None: ...
    def set_cursor_pos_y(self, y: float) -> None: ...
    def get_window_pos_x(self) -> float: ...
    def get_window_pos_y(self) -> float: ...
    def get_item_rect_min_x(self) -> float: ...
    def get_item_rect_min_y(self) -> float: ...
    def get_item_rect_max_x(self) -> float: ...
    def get_item_rect_max_y(self) -> float: ...

    # Interaction
    def invisible_button(self, id: str, width: float, height: float) -> bool: ...
    def is_item_active(self) -> bool: ...
    def is_item_hovered(self) -> bool: ...
    def set_keyboard_focus_here(self, offset: int = 0) -> None: ...
    def is_item_deactivated(self) -> bool: ...
    def is_item_deactivated_after_edit(self) -> bool: ...
    def get_mouse_drag_delta_y(self, button: int = 0) -> float: ...
    def reset_mouse_drag_delta(self, button: int = 0) -> None: ...

    # ID stack
    def push_id(self, id: int) -> None: ...
    def push_id_str(self, id: str) -> None: ...
    def pop_id(self) -> None: ...

    # Style
    def push_style_color(
        self, idx: int, r: float, g: float, b: float, a: float
    ) -> None: ...
    def pop_style_color(self, count: int = 1) -> None: ...
    def begin_disabled(self, disabled: bool = True) -> None: ...
    def end_disabled(self) -> None: ...

    # Drag and Drop
    def begin_drag_drop_source(self, flags: int = 0) -> bool: ...
    def set_drag_drop_payload(self, type: str, data: int) -> None: ...
    def set_drag_drop_payload_str(self, type: str, data: str) -> None: ...
    def end_drag_drop_source(self) -> None: ...
    def begin_drag_drop_target(self) -> bool: ...
    def accept_drag_drop_payload(self, type: str) -> Optional[Union[int, str]]: ...
    def end_drag_drop_target(self) -> None: ...

    # Mouse cursor
    def set_mouse_cursor(self, cursor_type: int) -> None: ...

    # Mouse input
    def is_mouse_button_down(self, button: int) -> bool: ...
    def is_mouse_button_clicked(self, button: int) -> bool: ...
    def is_mouse_double_clicked(self, button: int = 0) -> bool: ...
    def is_mouse_dragging(
        self, button: int, lock_threshold: float = -1.0
    ) -> bool: ...
    def get_mouse_drag_delta_x(self, button: int = 0) -> float: ...
    def get_mouse_pos_x(self) -> float: ...
    def get_mouse_pos_y(self) -> float: ...
    def get_mouse_wheel_delta(self) -> float: ...

    # Display / viewport bounds
    def get_main_viewport_bounds(self) -> Tuple[float, float, float, float]: ...
    def get_display_bounds(self) -> Tuple[float, float, float, float]: ...
    def get_global_mouse_pos_x(self) -> float: ...
    def get_global_mouse_pos_y(self) -> float: ...

    # Clipboard
    def set_clipboard_text(self, text: str) -> None: ...
    def get_clipboard_text(self) -> str: ...

    # Multiline text input
    def input_text_multiline(self, label: str, text: str, buffer_size: int = 4096, width: float = ..., height: float = ..., flags: int = ...) -> str: ...
    def draw_rect(self, min_x: float, min_y: float, max_x: float, max_y: float, r: float, g: float, b: float, a: float, thickness: float = 1.0, rounding: float = 0.0) -> None: ...
    def draw_filled_rect(self, min_x: float, min_y: float, max_x: float, max_y: float, r: float, g: float, b: float, a: float, rounding: float = 0.0) -> None: ...
    def draw_image_rect(self, texture_id: int, min_x: float, min_y: float, max_x: float, max_y: float, uv0_x: float = 0.0, uv0_y: float = 0.0, uv1_x: float = 1.0, uv1_y: float = 1.0, tint_r: float = 1.0, tint_g: float = 1.0, tint_b: float = 1.0, tint_a: float = 1.0) -> None: ...
    def set_window_font_scale(self, scale: float) -> None: ...
    def draw_text(self, x: float, y: float, text: str, r: float, g: float, b: float, a: float, font_size: float = 0.0) -> None: ...
    def draw_text_aligned(self, min_x: float, min_y: float, max_x: float, max_y: float, text: str, r: float, g: float, b: float, a: float, align_x: float = 0.0, align_y: float = 0.0, font_size: float = 0.0, clip: bool = False) -> None: ...
    def draw_text_ex_aligned(self, min_x: float, min_y: float, max_x: float, max_y: float, text: str, r: float, g: float, b: float, a: float, align_x: float = 0.0, align_y: float = 0.0, font_size: float = 0.0, wrap_width: float = 0.0, rotation: float = 0.0, mirror_h: bool = False, mirror_v: bool = False, clip: bool = False, font_path: str = "", line_height: float = 1.0, letter_spacing: float = 0.0) -> None: ...
    def calc_text_size(self, text: str, font_size: float = 0.0, font_path: str = "", line_height: float = 1.0, letter_spacing: float = 0.0) -> tuple[float, float]: ...
    def calc_text_size_wrapped(self, text: str, font_size: float = 0.0, wrap_width: float = 0.0, font_path: str = "", line_height: float = 1.0, letter_spacing: float = 0.0) -> tuple[float, float]: ...
    def push_draw_list_clip_rect(self, min_x: float, min_y: float, max_x: float, max_y: float, intersect_with_current: bool = True) -> None: ...
    def pop_draw_list_clip_rect(self) -> None: ...

    # Keyboard input
    def is_key_down(self, key_code: int) -> bool: ...
    def is_key_pressed(self, key_code: int) -> bool: ...
    def is_key_released(self, key_code: int) -> bool: ...

    # Focus
    def is_window_focused(self, flags: int = 0) -> bool: ...
    def is_window_hovered(self, flags: int = 0) -> bool: ...
    def capture_mouse_from_app(self, capture: bool) -> None: ...
    def capture_keyboard_from_app(self, capture: bool) -> None: ...


class InfGUIRenderable:
    """Base class for ImGui renderables — override on_render(ctx)."""

    def __init__(self) -> None: ...
    def on_render(self, ctx: InfGUIContext) -> None:
        """Called every frame to render ImGui widgets."""
        ...


class ResourcePreviewManager:
    """Manages resource previewers for Inspector."""

    def has_previewer(self, extension: str) -> bool: ...
    def get_previewer_type_name(self, extension: str) -> str: ...
    def get_all_supported_extensions(self) -> List[str]: ...
    def load_preview(self, file_path: str) -> None: ...
    def render_preview(
        self, ctx: InfGUIContext, avail_width: float, avail_height: float
    ) -> None: ...
    def render_metadata(self, ctx: InfGUIContext) -> None: ...
    def unload_preview(self) -> None: ...
    def is_preview_loaded(self) -> bool: ...
    def get_loaded_path(self) -> str: ...
    def get_current_type_name(self) -> str: ...


# =============================================================================
# EditorCamera (property-based camera controller)
# =============================================================================


class EditorCamera:
    """Editor camera controller with property-based access."""

    @property
    def fov(self) -> float: ...
    @fov.setter
    def fov(self, value: float) -> None: ...
    @property
    def near_clip(self) -> float: ...
    @near_clip.setter
    def near_clip(self, value: float) -> None: ...
    @property
    def far_clip(self) -> float: ...
    @far_clip.setter
    def far_clip(self, value: float) -> None: ...
    @property
    def position(self) -> Vector3: ...
    @property
    def rotation(self) -> Tuple[float, float]: ...
    @property
    def focus_point(self) -> Vector3: ...
    @property
    def focus_distance(self) -> float: ...
    rotation_speed: float
    pan_speed: float
    zoom_speed: float
    move_speed: float
    move_speed_boost: float
    def reset(self) -> None: ...
    def focus_on(self, x: float, y: float, z: float, distance: float = 10.0) -> None: ...
    def restore_state(
        self,
        pos_x: float, pos_y: float, pos_z: float,
        focus_x: float, focus_y: float, focus_z: float,
        focus_dist: float, yaw: float, pitch: float,
    ) -> None: ...
    def world_to_screen_point(self, x: float, y: float, z: float) -> Vector2: ...


# =============================================================================
# InfEngine (main engine facade)
# =============================================================================


class InfEngine:
    """Main engine facade — Vulkan renderer, scene management, editor integration."""

    def __init__(self, dll_path: str) -> None: ...
    def init_renderer(self, width: int, height: int, project_path: str) -> None: ...
    def set_gui_font(self, font_path: str, font_size: float = 18.0) -> None: ...
    def run(self) -> None: ...
    def set_log_level(self, level: LogLevel) -> None: ...
    def register_gui_renderable(
        self, name: str, renderable: InfGUIRenderable
    ) -> None: ...
    def unregister_gui_renderable(self, name: str) -> None: ...
    def exit(self) -> None:
        """Exit the InfEngine application."""
        ...
    def is_close_requested(self) -> bool:
        """True when the user clicked the window close button but Python has not yet confirmed."""
        ...
    def confirm_close(self) -> None:
        """Actually close the engine (call after save dialogs are handled)."""
        ...
    def cancel_close(self) -> None:
        """Cancel a pending close request (user chose Cancel in save dialog)."""
        ...
    def show(self) -> None:
        """Show the InfEngine window."""
        ...
    def hide(self) -> None:
        """Hide the InfEngine window."""
        ...
    def modify_resources(self, file_path: str) -> None: ...
    def delete_resources(self, file_path: str) -> None: ...
    def move_resources(self, old_file_path: str, new_file_path: str) -> None: ...
    def reload_shader(self, shader_path: str) -> None:
        """Reload a shader file and refresh materials using it."""
        ...
    def get_asset_database(self) -> AssetDatabase:
        """Get the asset database instance."""
        ...
    def upload_texture_for_imgui(
        self, name: str, pixels: List[int], width: int, height: int
    ) -> int:
        """Upload texture data for ImGui display, returns texture ID."""
        ...
    def remove_imgui_texture(self, name: str) -> None:
        """Remove a previously uploaded ImGui texture."""
        ...
    def has_imgui_texture(self, name: str) -> bool:
        """Check if an ImGui texture exists."""
        ...
    def get_imgui_texture_id(self, name: str) -> int:
        """Get texture ID for an uploaded texture."""
        ...
    def get_resource_preview_manager(self) -> ResourcePreviewManager:
        """Get the resource preview manager."""
        ...

    # Editor Camera (property-based object access)
    @property
    def editor_camera(self) -> EditorCamera: ...

    # Scene camera input
    def process_scene_view_input(
        self,
        delta_time: float,
        right_mouse_down: bool,
        middle_mouse_down: bool,
        mouse_delta_x: float,
        mouse_delta_y: float,
        scroll_delta: float,
        key_w: bool,
        key_a: bool,
        key_s: bool,
        key_d: bool,
        key_q: bool,
        key_e: bool,
        key_shift: bool,
    ) -> None:
        """Process scene view input for editor camera control."""
        ...

    # Scene render target
    def get_scene_texture_id(self) -> int:
        """Get scene render target texture ID for ImGui display."""
        ...
    def resize_scene_render_target(self, width: int, height: int) -> None:
        """Resize the scene render target."""
        ...

    # Scene picking
    def pick_scene_object_id(
        self,
        screen_x: float,
        screen_y: float,
        viewport_width: float,
        viewport_height: float,
    ) -> int:
        """Pick a scene object by screen-space coordinates (0 if none)."""
        ...
    def pick_scene_object_ids(
        self,
        screen_x: float,
        screen_y: float,
        viewport_width: float,
        viewport_height: float,
    ) -> list[int]:
        """Pick ordered scene object candidate IDs by screen-space coordinates."""
        ...

    # Editor tools (gizmo highlight + ray)
    def set_editor_tool_highlight(self, axis: int) -> None:
        """Set highlighted gizmo axis. 0=None, 1=X, 2=Y, 3=Z."""
        ...
    def screen_to_world_ray(
        self,
        screen_x: float,
        screen_y: float,
        viewport_width: float,
        viewport_height: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Build a world-space ray. Returns (ox,oy,oz, dx,dy,dz)."""
        ...

    # Editor gizmos
    def set_show_grid(self, show: bool) -> None:
        """Show or hide the editor grid."""
        ...
    def is_show_grid(self) -> bool:
        """Check if the editor grid is visible."""
        ...
    def set_selection_outline(self, object_id: int) -> None:
        """Set selection outline for a game object. Pass 0 to clear."""
        ...
    def set_selection_outlines(self, object_ids: list[int]) -> None:
        """Set combined selection outline for multiple game objects."""
        ...
    def get_selected_object_id(self) -> int:
        """Get the currently selected object ID (0 if none)."""
        ...
    def clear_selection_outline(self) -> None: ...

    # Material pipeline
    def refresh_material_pipeline(self, material: InfMaterial) -> None:
        """Refresh a material's rendering pipeline."""
        ...

    # Render pipeline (SRP)
    def set_render_pipeline(
        self, pipeline: Optional[RenderPipelineCallback]
    ) -> None:
        """Set a custom RenderPipelineCallback. Pass None for default."""
        ...

    # Render graph access
    def get_scene_render_graph(self) -> SceneRenderGraph:
        """Get the scene render graph."""
        ...

    # MSAA configuration
    def set_msaa_samples(self, samples: int) -> None:
        """Set MSAA sample count (1=off, 2, 4, 8) for both scene and game render targets."""
        ...
    def get_msaa_samples(self) -> int:
        """Get current MSAA sample count (1=off)."""
        ...

    # Window management
    def get_display_scale(self) -> float:
        """Get the OS display scale factor (e.g. 2.0 for 200% scaling)."""
        ...
    def set_window_icon(self, icon_path: str) -> None:
        """Set the window icon from a PNG file."""
        ...
    def set_fullscreen(self, fullscreen: bool) -> None:
        """Set the window to fullscreen or windowed mode."""
        ...
    def set_window_title(self, title: str) -> None:
        """Set the window title bar text."""
        ...
    def reset_imgui_layout(self) -> None:
        """Clear ImGui docking layout and delete saved ini."""
        ...
    def cleanup(self) -> None:
        """Destroy renderer and release all GPU resources."""
        ...
    def set_pre_gui_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a Python callback invoked each frame before GUI rendering."""
        ...

    # Asset hot-reload
    def reload_texture(self, texture_path: str) -> None:
        """Invalidate cached texture and force materials to reload it."""
        ...
    def reload_mesh(self, mesh_path: str) -> None:
        """Reload a mesh asset and notify dependent MeshRenderers."""
        ...
    def reload_audio(self, audio_path: str) -> None:
        """Reload an audio clip asset and notify dependents."""
        ...

    # GPU sync
    def wait_for_gpu_idle(self) -> None:
        """Drain pending GPU work before destructive scene replacement."""
        ...

    # Game render target
    def get_game_texture_id(self) -> int:
        """Get game render target texture ID for ImGui display."""
        ...
    def resize_game_render_target(self, width: int, height: int) -> None:
        """Resize the game render target (lazy-initializes on first call)."""
        ...
    def set_game_camera_enabled(self, enabled: bool) -> None:
        """Enable/disable game camera rendering."""
        ...
    def is_game_camera_enabled(self) -> bool:
        """Check if game camera rendering is enabled."""
        ...
    def set_scene_view_visible(self, visible: bool) -> None:
        """Enable/disable scene view rendering."""
        ...

    # Screen UI
    def get_screen_ui_renderer(self) -> Optional[InfScreenUIRenderer]:
        """Get the screen UI renderer for GPU-based 2D screen-space UI."""
        ...

    # Present mode
    def set_present_mode(self, mode: int) -> None:
        """Set present mode: 0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED."""
        ...
    def get_present_mode(self) -> int:
        """Get current present mode."""
        ...

    # Extended editor tools
    def pick_gizmo_axis(self, screen_x: float, screen_y: float, viewport_width: float, viewport_height: float) -> int:
        """Lightweight gizmo axis proximity test for hover highlighting."""
        ...
    def set_editor_tool_mode(self, mode: int) -> None:
        """Set the active tool mode. 0=None, 1=Translate, 2=Rotate, 3=Scale."""
        ...
    def get_editor_tool_mode(self) -> int:
        """Get the active tool mode."""
        ...
    def set_editor_tool_local_mode(self, local: bool) -> None:
        """Enable/disable local coordinate mode for editor tools."""
        ...

    # Component gizmos
    def upload_component_gizmos(self, vertices: Any, vertex_count: int, indices: Any, descriptors: Any, descriptor_count: int) -> None:
        """Upload per-component gizmo geometry via buffer protocol."""
        ...
    def clear_component_gizmos(self) -> None:
        """Clear all component gizmo geometry."""
        ...
    def upload_component_gizmo_icons(self, positions: Any, object_ids: Any, icon_count: int) -> None:
        """Upload component gizmo icon entries via buffer protocol."""
        ...
    def clear_component_gizmo_icons(self) -> None:
        """Clear all component gizmo icon data."""
        ...

    # Material pipeline cleanup
    def remove_material_pipeline(self, material_name: str) -> None:
        """Remove pipeline render data for a deleted material."""
        ...


# =============================================================================
# Physics enums
# =============================================================================


class ForceMode(IntEnum):
    """Force application mode for Rigidbody."""

    Force: int
    Acceleration: int
    Impulse: int
    VelocityChange: int


class RigidbodyConstraints(IntEnum):
    """Bitmask for freezing Rigidbody axes."""

    NoneFlag: int
    FreezePositionX: int
    FreezePositionY: int
    FreezePositionZ: int
    FreezeRotationX: int
    FreezeRotationY: int
    FreezeRotationZ: int
    FreezePosition: int
    FreezeRotation: int
    FreezeAll: int


class CollisionDetectionMode(IntEnum):
    """Collision detection algorithm for Rigidbody."""

    Discrete: int
    Continuous: int
    ContinuousDynamic: int
    ContinuousSpeculative: int


class RigidbodyInterpolation(IntEnum):
    """Rigidbody interpolation mode."""

    NoneFlag: int
    Interpolate: int


class ScreenUIList(IntEnum):
    """Screen UI draw list selection."""

    Camera: int
    Overlay: int


# =============================================================================
# Render target handle
# =============================================================================


class RenderTargetHandle:
    """Opaque handle to a temporary or persistent render target."""

    @property
    def id(self) -> int: ...
    def is_valid(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...


# =============================================================================
# CommandBuffer
# =============================================================================


class CommandBuffer:
    """Deferred rendering command buffer."""

    def __init__(self, name: str = "") -> None: ...

    # Render target management
    def get_temporary_rt(
        self,
        width: int,
        height: int,
        format: VkFormat = ...,
        samples: VkSampleCount = ...,
    ) -> RenderTargetHandle:
        """Allocate a temporary render target."""
        ...
    def release_temporary_rt(self, handle: RenderTargetHandle) -> None:
        """Release a temporary render target back to the pool."""
        ...
    def set_render_target(self, color: RenderTargetHandle) -> None:
        """Set the active color render target."""
        ...
    def set_render_target_with_depth(self, color: RenderTargetHandle, depth: RenderTargetHandle) -> None:
        """Set the active color and depth render targets."""
        ...
    def clear_render_target(
        self,
        clear_color: bool,
        clear_depth: bool,
        r: float,
        g: float,
        b: float,
        a: float,
        depth: float = 1.0,
    ) -> None:
        """Clear the active render target."""
        ...

    # Global shader parameters
    def set_global_texture(self, name: str, handle: RenderTargetHandle) -> None: ...
    def set_global_float(self, name: str, value: float) -> None: ...
    def set_global_vector(self, name: str, x: float, y: float, z: float, w: float) -> None: ...
    def set_global_matrix(self, name: str, data: List[float]) -> None:
        """Set a global 4x4 matrix (16 floats, column-major)."""
        ...

    # Async readback
    def request_async_readback(self, handle: RenderTargetHandle, callback_id: int) -> None: ...

    # Misc
    def clear(self) -> None:
        """Clear all recorded commands."""
        ...
    @property
    def name(self) -> str: ...
    @property
    def command_count(self) -> int: ...


# =============================================================================
# Physics types
# =============================================================================


class CollisionInfo:
    """Collision event data passed to on_collision callbacks."""

    @property
    def collider(self) -> Collider: ...
    @property
    def game_object(self) -> GameObject: ...
    @property
    def contact_point(self) -> Vector3: ...
    @property
    def contact_normal(self) -> Vector3: ...
    @property
    def relative_velocity(self) -> Vector3: ...
    @property
    def impulse(self) -> float: ...


class RaycastHit:
    """Result of a physics raycast query."""

    @property
    def point(self) -> Vector3: ...
    @property
    def normal(self) -> Vector3: ...
    @property
    def distance(self) -> float: ...
    @property
    def game_object(self) -> GameObject: ...
    @property
    def collider(self) -> Collider: ...


class Collider(Component):
    """Base class for all physics collider components."""

    is_trigger: bool
    center: Vector3
    friction: float
    bounciness: float
    def serialize(self) -> str: ...
    def deserialize(self, json_str: str) -> None: ...


class BoxCollider(Collider):
    """Axis-aligned box collider."""

    size: Vector3
    def serialize(self) -> str: ...
    def deserialize(self, json_str: str) -> None: ...


class SphereCollider(Collider):
    """Sphere collider."""

    radius: float
    def serialize(self) -> str: ...
    def deserialize(self, json_str: str) -> None: ...


class CapsuleCollider(Collider):
    """Capsule collider (cylinder + hemisphere caps)."""

    radius: float
    height: float
    direction: int
    def serialize(self) -> str: ...
    def deserialize(self, json_str: str) -> None: ...


class MeshCollider(Collider):
    """Mesh-based collider."""

    convex: bool
    def serialize(self) -> str: ...
    def deserialize(self, json_str: str) -> None: ...


class Rigidbody(Component):
    """Physics rigid body component."""

    mass: float
    drag: float
    angular_drag: float
    use_gravity: bool
    is_kinematic: bool
    constraints: int
    freeze_rotation: bool
    collision_detection_mode: CollisionDetectionMode
    interpolation: RigidbodyInterpolation
    max_angular_velocity: float
    max_linear_velocity: float
    velocity: Vector3
    angular_velocity: Vector3

    @property
    def world_center_of_mass(self) -> Vector3: ...
    @property
    def position(self) -> Vector3: ...
    @property
    def rotation(self) -> quatf: ...

    def add_force(self, force: Vector3, mode: ForceMode = ...) -> None: ...
    def add_torque(self, torque: Vector3, mode: ForceMode = ...) -> None: ...
    def add_force_at_position(self, force: Vector3, position: Vector3, mode: ForceMode = ...) -> None: ...
    def move_position(self, position: Vector3) -> None: ...
    def move_rotation(self, rotation: quatf) -> None: ...
    def is_sleeping(self) -> bool: ...
    def wake_up(self) -> None: ...
    def sleep(self) -> None: ...
    def serialize(self) -> str: ...
    def deserialize(self, json_str: str) -> None: ...


class Physics:
    """Static physics query interface."""

    @staticmethod
    def raycast(
        origin: Vector3,
        direction: Vector3,
        max_distance: float = 1000.0,
        layer_mask: int = 0xFFFFFFFF,
        query_triggers: bool = True,
    ) -> Optional[RaycastHit]:
        """Cast a ray and return the closest hit, or None."""
        ...
    @staticmethod
    def raycast_all(
        origin: Vector3,
        direction: Vector3,
        max_distance: float = 1000.0,
        layer_mask: int = 0xFFFFFFFF,
        query_triggers: bool = True,
    ) -> List[RaycastHit]:
        """Cast a ray and return all hits."""
        ...
    @staticmethod
    def overlap_sphere(
        center: Vector3,
        radius: float,
        layer_mask: int = 0xFFFFFFFF,
        query_triggers: bool = True,
    ) -> List[Collider]:
        """Find all colliders within a sphere."""
        ...
    @staticmethod
    def overlap_box(
        center: Vector3,
        half_extents: Vector3,
        layer_mask: int = 0xFFFFFFFF,
        query_triggers: bool = True,
    ) -> List[Collider]:
        """Find all colliders within an axis-aligned box."""
        ...
    @staticmethod
    def sphere_cast(
        origin: Vector3,
        radius: float,
        direction: Vector3,
        max_distance: float = 1000.0,
        layer_mask: int = 0xFFFFFFFF,
        query_triggers: bool = True,
    ) -> Optional[RaycastHit]:
        """Sweep a sphere and return the closest hit."""
        ...
    @staticmethod
    def box_cast(
        center: Vector3,
        half_extents: Vector3,
        direction: Vector3,
        max_distance: float = 1000.0,
        layer_mask: int = 0xFFFFFFFF,
        query_triggers: bool = True,
    ) -> Optional[RaycastHit]:
        """Sweep a box and return the closest hit."""
        ...
    @staticmethod
    def get_gravity() -> Vector3: ...
    @staticmethod
    def set_gravity(gravity: Vector3) -> None: ...
    @staticmethod
    def ignore_layer_collision(layer1: int, layer2: int, ignore: bool = True) -> None: ...
    @staticmethod
    def get_ignore_layer_collision(layer1: int, layer2: int) -> bool: ...


# =============================================================================
# TagLayerManager
# =============================================================================


class TagLayerManager:
    """Singleton manager for tags and physics layers."""

    @staticmethod
    def instance() -> TagLayerManager: ...

    # Tags
    def get_tag(self, index: int) -> str: ...
    def get_tag_index(self, tag: str) -> int:
        """Return tag index, or -1 if not found."""
        ...
    def add_tag(self, tag: str) -> int:
        """Add a tag and return its index."""
        ...
    def remove_tag(self, tag: str) -> None: ...
    def get_all_tags(self) -> List[str]: ...
    def is_builtin_tag(self, tag: str) -> bool: ...

    # Layers
    def get_layer_name(self, layer: int) -> str: ...
    def get_layer_by_name(self, name: str) -> int:
        """Return layer index, or -1 if not found."""
        ...
    def set_layer_name(self, layer: int, name: str) -> None: ...
    def get_all_layers(self) -> List[str]: ...
    def is_builtin_layer(self, layer: int) -> bool: ...
    def get_layer_collision_mask(self, layer: int) -> int: ...
    def set_layer_collision_mask(self, layer: int, mask: int) -> None: ...
    def get_layers_collide(self, layer_a: int, layer_b: int) -> bool: ...
    def set_layers_collide(self, layer_a: int, layer_b: int, should_collide: bool) -> None: ...

    @staticmethod
    def layer_to_mask(layer: int) -> int: ...
    def get_mask(self, layer_names: List[str]) -> int: ...

    # Serialization
    def serialize(self) -> str: ...
    def deserialize(self, json_str: str) -> None: ...
    def save_to_file(self, path: str) -> None: ...
    def load_from_file(self, path: str) -> None: ...


# =============================================================================
# EngineConfig
# =============================================================================


class EngineConfig:
    """Singleton engine configuration."""

    @staticmethod
    def get() -> EngineConfig: ...

    # Rendering
    max_materials_per_pool: int
    ubo_descriptors_per_material: int
    sampler_descriptors_per_material: int
    enable_mipmap: bool
    anisotropy_scale: float
    preferred_swapchain_image_count: int
    max_frames_in_flight: int

    # Physics
    physics_temp_allocator_size: int
    physics_max_jobs: int
    physics_max_barriers: int
    physics_max_bodies: int
    physics_max_body_pairs: int
    physics_max_contact_constraints: int
    physics_collision_steps: int
    physics_gravity: Vector3
    physics_max_worker_threads: int

    # Default collider properties
    default_collider_friction: float
    default_collider_bounciness: float

    # Default rigidbody properties
    default_rigidbody_mass: float
    default_rigidbody_drag: float
    default_rigidbody_angular_drag: float
    default_max_angular_velocity: float
    default_max_linear_velocity: float
    default_max_depenetration_velocity: float

    # Physics layers
    physics_layer_count: int
    default_query_layer_mask: int

    # Render queue ranges
    component_gizmo_queue_min: int
    component_gizmo_queue_max: int
    editor_gizmo_queue_min: int
    editor_gizmo_queue_max: int
    editor_tools_queue_min: int
    editor_tools_queue_max: int
    skybox_queue: int


# =============================================================================
# InfScreenUIRenderer
# =============================================================================


class InfScreenUIRenderer:
    """GPU-based 2D screen-space UI renderer."""

    def begin_frame(self, width: int, height: int) -> None: ...

    def add_filled_rect(
        self,
        list: ScreenUIList,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        r: float = 1.0,
        g: float = 1.0,
        b: float = 1.0,
        a: float = 1.0,
        rounding: float = 0.0,
    ) -> None: ...

    def add_image(
        self,
        list: ScreenUIList,
        texture_id: int,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        uv0_x: float = 0.0,
        uv0_y: float = 0.0,
        uv1_x: float = 1.0,
        uv1_y: float = 1.0,
        r: float = 1.0,
        g: float = 1.0,
        b: float = 1.0,
        a: float = 1.0,
        rotation: float = 0.0,
        mirror_h: bool = False,
        mirror_v: bool = False,
        rounding: float = 0.0,
    ) -> None: ...

    def add_text(
        self,
        list: ScreenUIList,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        text: str,
        r: float = 1.0,
        g: float = 1.0,
        b: float = 1.0,
        a: float = 1.0,
        align_x: float = 0.5,
        align_y: float = 0.5,
        font_size: float = 0.0,
        wrap_width: float = 0.0,
        rotation: float = 0.0,
        mirror_h: bool = False,
        mirror_v: bool = False,
        font_path: str = "",
        line_height: float = 1.0,
        letter_spacing: float = 0.0,
    ) -> None: ...

    def measure_text(
        self,
        text: str,
        font_size: float = 0.0,
        wrap_width: float = 0.0,
        font_path: str = "",
        line_height: float = 1.0,
        letter_spacing: float = 0.0,
    ) -> Tuple[float, float]: ...

    def has_commands(self, list: ScreenUIList) -> bool: ...
    def set_enabled(self, enabled: bool) -> None: ...
    def is_enabled(self) -> bool: ...


# =============================================================================
# InputManager — Unity-style input state manager
# =============================================================================


class InputManager:
    """
    Low-level input state manager (singleton).

    Use the Python ``Input`` class from ``InfEngine.input`` for the public API.
    This class exposes the raw C++ InputManager for direct access.
    """

    @staticmethod
    def instance() -> InputManager:
        """Get the singleton InputManager instance."""
        ...

    # ---- Keyboard ----
    def get_key(self, scancode: int) -> bool:
        """True while the key (by SDL scancode) is held down."""
        ...
    def get_key_down(self, scancode: int) -> bool:
        """True during the frame the key was pressed."""
        ...
    def get_key_up(self, scancode: int) -> bool:
        """True during the frame the key was released."""
        ...
    def any_key(self) -> bool:
        """True if any key is currently held down."""
        ...
    def any_key_down(self) -> bool:
        """True during the frame any key was first pressed."""
        ...

    # ---- Mouse buttons ----
    def get_mouse_button(self, button: int) -> bool:
        """True while mouse button is held (0=left, 1=right, 2=middle)."""
        ...
    def get_mouse_button_down(self, button: int) -> bool:
        """True during the frame the mouse button was pressed."""
        ...
    def get_mouse_button_up(self, button: int) -> bool:
        """True during the frame the mouse button was released."""
        ...

    # ---- Mouse position & delta (properties) ----
    @property
    def mouse_position_x(self) -> float:
        """Current mouse X (window-space pixels)."""
        ...
    @property
    def mouse_position_y(self) -> float:
        """Current mouse Y (window-space pixels)."""
        ...
    @property
    def mouse_delta_x(self) -> float:
        """Mouse X movement this frame."""
        ...
    @property
    def mouse_delta_y(self) -> float:
        """Mouse Y movement this frame."""
        ...

    # ---- Scroll wheel (properties) ----
    @property
    def mouse_scroll_delta_y(self) -> float:
        """Vertical scroll delta (positive = up)."""
        ...
    @property
    def mouse_scroll_delta_x(self) -> float:
        """Horizontal scroll delta."""
        ...

    # ---- Text input ----
    @property
    def input_string(self) -> str:
        """Characters typed this frame (UTF-8)."""
        ...

    # ---- Touch ----
    @property
    def touch_count(self) -> int:
        """Number of active touch contacts."""
        ...

    # ---- File drop (OS drag-drop) ----
    def has_dropped_files(self) -> bool:
        """True if files were dropped onto the window this frame."""
        ...
    def get_dropped_files(self) -> List[str]:
        """List of file paths dropped this frame."""
        ...

    # ---- Utility ----
    def reset_all(self) -> None:
        """Clear all input state."""
        ...
    @staticmethod
    def name_to_scancode(name: str) -> int:
        """Map key name to SDL scancode. Returns -1 if unknown."""
        ...
    @staticmethod
    def scancode_to_name(scancode: int) -> str:
        """Get human-readable name for a scancode."""
        ...


# =============================================================================
# Module-level functions
# =============================================================================


def generate_wire_sphere(
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    segments: int,
    cr: float,
    cg: float,
    cb: float,
) -> Tuple[npt.NDArray[np.float32], int, npt.NDArray[np.int32]]:
    """Generate 3 axis-aligned wire circles.

    Returns ``(vertices_flat, vertex_count, indices_flat)``.
    Vertex format: 6 floats per vertex (x, y, z, r, g, b).
    """
    ...


def generate_wire_arc(
    cx: float,
    cy: float,
    cz: float,
    nx: float,
    ny: float,
    nz: float,
    radius: float,
    start_deg: float,
    arc_deg: float,
    segments: int,
    cr: float,
    cg: float,
    cb: float,
) -> Tuple[npt.NDArray[np.float32], int, npt.NDArray[np.int32]]:
    """Generate a wire arc in the plane perpendicular to the given normal.

    Returns ``(vertices_flat, vertex_count, indices_flat)``.
    Vertex format: 6 floats per vertex (x, y, z, r, g, b).
    """
    ...
