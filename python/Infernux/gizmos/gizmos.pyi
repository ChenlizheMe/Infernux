from __future__ import annotations

from typing import List, Optional, Tuple
from numpy import ndarray
from Infernux.compute import Buffer

Vec3 = Tuple[float, float, float]


class Gizmos:
    """Draw visual debugging helpers in the Scene view."""

    color: Tuple[float, float, float]
    """The color used for drawing gizmos."""
    matrix: Optional[List[float]]
    """The transformation matrix for gizmo drawing."""

    @classmethod
    def _begin_frame(cls) -> None: ...
    @classmethod
    def draw_line(cls, start: Vec3, end: Vec3) -> None:
        """Draw a line from start to end in the Scene view."""
        ...
    @classmethod
    def draw_lines(cls, positions: ndarray | Buffer, indices: ndarray) -> None:
        """Draw indexed lines from NumPy or GPU-resident (N,3) positions."""
        ...
    @classmethod
    def draw_ray(cls, origin: Vec3, direction: Vec3) -> None:
        """Draw a ray starting at origin in the given direction."""
        ...
    @classmethod
    def draw_icon(cls, position: Vec3, object_id: int,
                  color: Optional[Tuple[float, float, float]] = ...) -> None:
        """Draw an icon at the given world position."""
        ...
    @classmethod
    def draw_wire_cube(cls, center: Vec3, size: Vec3) -> None:
        """Draw a wireframe cube in the Scene view."""
        ...
    @classmethod
    def draw_wire_sphere(cls, center: Vec3, radius: float, segments: int = ...) -> None:
        """Draw a wireframe sphere in the Scene view."""
        ...
    @classmethod
    def draw_wire_spheres(cls, centers: Buffer, radius: float, segments: int = ...,
                          center_indices: ndarray | None = ...) -> None:
        """Draw many wire spheres from GPU-resident center positions."""
        ...
    @classmethod
    def draw_frustum(cls, position: Vec3, fov_deg: float, aspect: float,
                     near: float, far: float,
                     forward: Vec3 = ..., up: Vec3 = ...,
                     right: Vec3 = ...) -> None:
        """Draw a camera frustum wireframe in the Scene view."""
        ...
    @classmethod
    def draw_wire_arc(cls, center: Vec3, normal: Vec3, radius: float,
                      start_angle_deg: float = ..., arc_deg: float = ...,
                      segments: int = ...) -> None:
        """Draw a wireframe arc in the Scene view."""
        ...
