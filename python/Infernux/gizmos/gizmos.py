"""
Gizmos — Unity-style immediate-mode gizmo drawing API.

Usage in InxComponent subclasses::

    from Infernux.gizmos import Gizmos

    class MyComponent(InxComponent):
        always_show: bool = True   # show gizmo even when not selected

        def on_draw_gizmos(self):
            # Called for ALL components every frame (if always_show=True)
            Gizmos.color = (0, 1, 0)
            Gizmos.draw_wire_cube(self.transform.position, (1, 1, 1))

        def on_draw_gizmos_selected(self):
            # Called only when this object (or an ancestor) is selected
            Gizmos.color = (1, 1, 0)
            Gizmos.draw_wire_sphere(self.transform.position, 2.0)

The Gizmos class accumulates line segments during callback invocations,
then the GizmosCollector packs everything and uploads to C++ in one batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import count
from typing import Tuple, Optional, List
import numpy as np

from Infernux.compute import Buffer, buffer, index, kernel, launch

from Infernux.components._gizmo_ids import (
    ICON_KIND_CAMERA,
    ICON_KIND_DEFAULT,
    ICON_KIND_LIGHT,
    ICON_KIND_PARTICLE,
)

# Type alias for 3-component tuples
Vec3 = Tuple[float, float, float]


@kernel
def _resident_line_vertex_kernel(domain, positions, vertices, red, green, blue):
    """Expand resident positions into the renderer's canonical Vertex stream."""
    i = index(domain)
    vertices[i, 0] = positions[i, 0]
    vertices[i, 1] = positions[i, 1]
    vertices[i, 2] = positions[i, 2]
    vertices[i, 3] = 0.0
    vertices[i, 4] = 1.0
    vertices[i, 5] = 0.0
    vertices[i, 6] = 1.0
    vertices[i, 7] = 0.0
    vertices[i, 8] = 0.0
    vertices[i, 9] = 1.0
    vertices[i, 10] = red
    vertices[i, 11] = green
    vertices[i, 12] = blue
    for lane in range(13, 23):
        vertices[i, lane] = 0.0


@dataclass(slots=True)
class _ResidentLineState:
    identity: int
    indices: np.ndarray
    domain: Buffer
    vertices: Buffer

    def close(self) -> None:
        self.domain.close()
        self.vertices.close()


@kernel
def _resident_wire_sphere_kernel(domain, centers, center_indices, unit_positions,
                                 positions, unit_count, radius):
    i = index(domain)
    center_slot = i // unit_count
    unit_slot = i - center_slot * unit_count
    center_index = center_indices[center_slot]
    positions[i, 0] = centers[center_index, 0] + unit_positions[unit_slot, 0] * radius
    positions[i, 1] = centers[center_index, 1] + unit_positions[unit_slot, 1] * radius
    positions[i, 2] = centers[center_index, 2] + unit_positions[unit_slot, 2] * radius


@dataclass(slots=True)
class _ResidentWireSphereState:
    source_indices: np.ndarray
    domain: Buffer
    center_indices: Buffer
    unit_positions: Buffer
    positions: Buffer
    line_indices: np.ndarray
    unit_count: int

    def close(self) -> None:
        self.domain.close()
        self.center_indices.close()
        self.unit_positions.close()
        self.positions.close()


_resident_identity = count(1)

# Try to import C++ gizmo geometry helpers (available after engine build)
try:
    from Infernux.lib import generate_wire_sphere as _cpp_wire_sphere
    from Infernux.lib import generate_wire_arc as _cpp_wire_arc
    _HAS_CPP_GIZMOS = True
except ImportError:
    _HAS_CPP_GIZMOS = False


class Gizmos:
    """Unity-style immediate-mode gizmo drawing.

    All methods are class-level (static-ish).  State resets each frame
    via ``_begin_frame()``, called by the collector.

    Drawing primitives accumulate line-segment vertices into a shared
    per-frame buffer.  The collector packs and uploads them to the C++
    ``GizmosDrawCallBuffer`` before ``SubmitCulling()``.
    """

    # ---- Per-frame state (reset each frame) ----
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    matrix: Optional[List[float]] = None  # 16-float column-major; None = identity

    # ---- Per-frame accumulation buffers ----
    # Each entry: (vertex_list, index_list, world_matrix_16_floats)
    _draw_batches: List[Tuple[List[List[float]], List[int], List[float]]] = []
    # (state, current world matrix); state owns immutable topology and a
    # canonical GPU Vertex stream derived directly from the source buffer.
    _resident_draw_batches: list[tuple[_ResidentLineState, List[float]]] = []

    # Icon entries: (position_vec3, object_id_int, color_vec3, icon_kind_int)
    _icon_entries: List[Tuple[Vec3, int, Tuple[float, float, float], int]] = []

    # ---- Internal ----
    _identity_matrix: List[float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]

    @classmethod
    def _begin_frame(cls):
        """Reset per-frame state.  Called by GizmosCollector at frame start."""
        cls.color = (1.0, 1.0, 1.0)
        cls.matrix = None
        cls._draw_batches.clear()
        cls._resident_draw_batches.clear()
        cls._icon_entries.clear()

    @classmethod
    def _current_matrix(cls) -> List[float]:
        return cls.matrix if cls.matrix is not None else cls._identity_matrix

    # ====================================================================
    # Primitive: line
    # ====================================================================

    @classmethod
    def draw_line(cls, start: Vec3, end: Vec3):
        """Draw a single line segment from *start* to *end*."""
        c = cls.color
        verts = [
            [start[0], start[1], start[2], c[0], c[1], c[2]],
            [end[0], end[1], end[2], c[0], c[1], c[2]],
        ]
        indices = [0, 1]
        cls._draw_batches.append((verts, indices, list(cls._current_matrix())))

    @classmethod
    def draw_lines(cls, positions, indices):
        """Draw indexed line pairs in one batch from NumPy arrays.

        Positions have shape (N,3); integer indices have shape (L,2). Inputs
        are captured at the call, with the current color and world matrix.
        """
        if isinstance(positions, Buffer):
            cls._draw_resident_lines(positions, indices)
            return
        if not isinstance(positions, np.ndarray) or positions.ndim != 2 or positions.shape[1] != 3:
            raise TypeError("Gizmo positions must be a NumPy array or GPU inx.buffer with shape (N,3)")
        if not isinstance(indices, np.ndarray) or indices.ndim != 2 or indices.shape[1] != 2 or indices.dtype.kind not in 'iu':
            raise TypeError("Gizmo indices must be an integer NumPy array with shape (L,2)")
        if not len(indices):
            return
        if indices.min() < 0 or indices.max() >= len(positions):
            raise ValueError("Gizmo line index is outside the position array")
        vertices = np.empty((len(positions),6),dtype=np.float32)
        vertices[:,:3] = positions
        vertices[:,3:] = cls.color
        cls._draw_batches.append((vertices, indices.astype(np.uint32,copy=True).ravel(),
                                  list(cls._current_matrix())))

    @classmethod
    def _draw_resident_lines(cls, positions: Buffer, indices) -> None:
        if positions.device != "gpu":
            raise TypeError("Resident Gizmo positions require a GPU inx.buffer")
        if positions.dtype == "vector3":
            position_count = positions.shape[0]
        elif positions.dtype in {"float", "float32"} and len(positions.shape) == 2 and positions.shape[1] == 3:
            position_count = positions.shape[0]
        else:
            raise TypeError("Resident Gizmo positions must use vector3 or float32 shape (N,3)")
        if not isinstance(indices, np.ndarray) or indices.ndim != 2 or indices.shape[1] != 2 or indices.dtype.kind not in "iu":
            raise TypeError("Gizmo indices must be an integer NumPy array with shape (L,2)")
        if not len(indices):
            return
        if indices.min() < 0 or indices.max() >= position_count:
            raise ValueError("Gizmo line index is outside the position buffer")

        states = getattr(positions, "_gizmo_line_states", None)
        if states is None:
            states = {}
            positions._gizmo_line_states = states
        key = id(indices)
        state = states.get(key)
        if state is None or state.indices is not indices:
            state = _ResidentLineState(
                identity=next(_resident_identity),
                indices=indices,
                domain=buffer(shape=position_count, dtype=np.int32, device="gpu"),
                vertices=buffer(shape=(position_count, 23), dtype=np.float32, device="gpu"),
            )
            states[key] = state
            positions._retain_dependent(state)
        red, green, blue = (float(value) for value in cls.color)
        launch(
            _resident_line_vertex_kernel,
            params=(state.domain, positions, state.vertices, red, green, blue),
        )
        cls._resident_draw_batches.append((state, list(cls._current_matrix())))

    # ====================================================================
    # Primitive: ray
    # ====================================================================

    @classmethod
    def draw_ray(cls, origin: Vec3, direction: Vec3):
        """Draw a ray from *origin* in *direction* (magnitude = length)."""
        end = (
            origin[0] + direction[0],
            origin[1] + direction[1],
            origin[2] + direction[2],
        )
        cls.draw_line(origin, end)

    # ====================================================================
    # Primitive: icon (billboard diamond at a world position)
    # ====================================================================

    @classmethod
    def draw_icon(cls, position: Vec3, object_id: int,
                  color: Optional[Tuple[float, float, float]] = None,
                  icon_kind: int = ICON_KIND_DEFAULT):
        """Register a clickable icon at *position* for the given GameObject.

        Icons are rendered as camera-facing diamond quads in the scene view.
        Clicking an icon selects the owning GameObject (Unity-style).

        Args:
            position: World-space position for the icon center.
            object_id: The owning GameObject's ID (used for picking).
            color: Icon tint color ``(r, g, b)``.  Defaults to ``Gizmos.color``.
            icon_kind: Built-in icon kind used by the native billboard material.
        """
        c = color if color is not None else cls.color
        cls._icon_entries.append((position, object_id, c, int(icon_kind)))

    # ====================================================================
    # Primitive: wire cube
    # ====================================================================

    @classmethod
    def draw_wire_cube(cls, center: Vec3, size: Vec3):
        """Draw a wireframe axis-aligned box centered at *center* with *size*."""
        hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
        cx, cy, cz = center

        # 8 corners
        corners = [
            (cx - hx, cy - hy, cz - hz),  # 0
            (cx + hx, cy - hy, cz - hz),  # 1
            (cx + hx, cy + hy, cz - hz),  # 2
            (cx - hx, cy + hy, cz - hz),  # 3
            (cx - hx, cy - hy, cz + hz),  # 4
            (cx + hx, cy - hy, cz + hz),  # 5
            (cx + hx, cy + hy, cz + hz),  # 6
            (cx - hx, cy + hy, cz + hz),  # 7
        ]

        # 12 edges as line pairs
        edges = [
            0, 1, 1, 2, 2, 3, 3, 0,  # front face
            4, 5, 5, 6, 6, 7, 7, 4,  # back face
            0, 4, 1, 5, 2, 6, 3, 7,  # connecting edges
        ]

        c = cls.color
        verts = [[p[0], p[1], p[2], c[0], c[1], c[2]] for p in corners]
        cls._draw_batches.append((verts, edges, list(cls._current_matrix())))

    # ====================================================================
    # Primitive: wire sphere
    # ====================================================================

    @classmethod
    def draw_wire_sphere(cls, center: Vec3, radius: float, segments: int = 24):
        """Draw a wireframe sphere as three axis-aligned circles."""
        c = cls.color
        cx, cy, cz = center
        mat = list(cls._current_matrix())

        if _HAS_CPP_GIZMOS:
            vert_flat, vert_count, idx_flat = _cpp_wire_sphere(
                cx, cy, cz, radius, segments, c[0], c[1], c[2])
            cls._draw_batches.append((vert_flat.reshape(vert_count, 6), idx_flat, mat))
            return

        verts = []
        indices = []

        # Pre-compute trig table
        import math as _math
        _two_pi = 2.0 * _math.pi
        cos_tab = [_math.cos(_two_pi * i / segments) for i in range(segments)]
        sin_tab = [_math.sin(_two_pi * i / segments) for i in range(segments)]

        for axis in range(3):
            base = len(verts)
            for i in range(segments):
                ca = cos_tab[i] * radius
                sa = sin_tab[i] * radius
                if axis == 0:  # YZ circle
                    p = (cx, cy + ca, cz + sa)
                elif axis == 1:  # XZ circle
                    p = (cx + ca, cy, cz + sa)
                else:  # XY circle
                    p = (cx + ca, cy + sa, cz)
                verts.append([p[0], p[1], p[2], c[0], c[1], c[2]])

            for i in range(segments):
                indices.append(base + i)
                indices.append(base + (i + 1) % segments)

        cls._draw_batches.append((verts, indices, mat))

    @classmethod
    def draw_wire_spheres(cls, centers: Buffer, radius: float, segments: int = 24,
                          center_indices: np.ndarray | None = None) -> None:
        """Draw many wire spheres directly from resident GPU centers.

        Unit-circle topology is authored once for the center set. Each frame a
        compute pass expands positions on the GPU, then the ordinary resident
        line path renders that buffer without a synchronization readback.
        """
        if not isinstance(centers, Buffer) or centers.device != "gpu":
            raise TypeError("Resident wire spheres require a GPU inx.buffer")
        if centers.dtype == "vector3":
            center_count = centers.shape[0]
        elif centers.dtype in {"float", "float32"} and len(centers.shape) == 2 and centers.shape[1] == 3:
            center_count = centers.shape[0]
        else:
            raise TypeError("Resident wire sphere centers must use vector3 or float32 shape (N,3)")
        if not isinstance(segments, int) or segments < 3:
            raise ValueError("Resident wire sphere segments must be at least 3")
        states = getattr(centers, "_gizmo_wire_sphere_states", None)
        if states is None:
            states = {}
            centers._gizmo_wire_sphere_states = states
        use_all_centers = center_indices is None
        if center_indices is not None:
            if not isinstance(center_indices, np.ndarray) or center_indices.ndim != 1 or center_indices.dtype.kind not in "iu":
                raise TypeError("Resident wire sphere center_indices must be a one-dimensional integer NumPy array")
            if not len(center_indices):
                return
            if center_indices.min() < 0 or center_indices.max() >= center_count:
                raise ValueError("Resident wire sphere center index is outside the center buffer")
        key = (("all", center_count) if use_all_centers else id(center_indices), segments)
        state = states.get(key)
        if state is None or (not use_all_centers and state.source_indices is not center_indices):
            if use_all_centers:
                center_indices = np.arange(center_count, dtype=np.int32)
            angle = np.arange(segments, dtype=np.float32) * (2.0 * np.pi / segments)
            cosine, sine = np.cos(angle), np.sin(angle)
            unit = np.zeros((segments * 3, 3), dtype=np.float32)
            unit[0:segments, 1] = cosine
            unit[0:segments, 2] = sine
            unit[segments:segments * 2, 0] = cosine
            unit[segments:segments * 2, 2] = sine
            unit[segments * 2:, 0] = cosine
            unit[segments * 2:, 1] = sine
            unit_edges = np.empty((segments * 3, 2), dtype=np.uint32)
            for axis in range(3):
                begin = axis * segments
                unit_edges[begin:begin + segments, 0] = np.arange(begin, begin + segments, dtype=np.uint32)
                unit_edges[begin:begin + segments, 1] = begin + np.roll(np.arange(segments, dtype=np.uint32), -1)
            expanded_edges = (
                unit_edges[None, :, :] +
                np.arange(len(center_indices), dtype=np.uint32)[:, None, None] * len(unit)
            ).reshape(-1, 2)
            expanded_count = len(center_indices) * len(unit)
            state = _ResidentWireSphereState(
                source_indices=center_indices,
                domain=buffer(shape=expanded_count, dtype=np.int32, device="gpu"),
                center_indices=buffer(shape=len(center_indices), dtype=np.int32, device="gpu", data=center_indices),
                unit_positions=buffer(shape=unit.shape, dtype=np.float32, device="gpu", data=unit),
                positions=buffer(shape=(expanded_count, 3), dtype=np.float32, device="gpu"),
                line_indices=np.ascontiguousarray(expanded_edges),
                unit_count=len(unit),
            )
            states[key] = state
            centers._retain_dependent(state)
        launch(
            _resident_wire_sphere_kernel,
            params=(state.domain, centers, state.center_indices, state.unit_positions,
                    state.positions, state.unit_count, float(radius)),
        )
        cls.draw_lines(state.positions, state.line_indices)

    # ====================================================================
    # Primitive: wire frustum
    # ====================================================================

    @classmethod
    def draw_frustum(cls, position: Vec3, fov_deg: float, aspect: float,
                     near: float, far: float,
                     forward: Vec3 = (0, 0, -1),
                     up: Vec3 = (0, 1, 0),
                     right: Vec3 = (1, 0, 0)):
        """Draw a camera frustum wireframe.

        Args:
            position: Camera position.
            fov_deg: Vertical field of view in degrees.
            aspect: Width / height aspect ratio.
            near: Near clip distance.
            far: Far clip distance.
            forward, up, right: Camera basis vectors (world-space).
        """
        half_fov = math.radians(fov_deg * 0.5)
        tan_fov = math.tan(half_fov)

        near_h = near * tan_fov
        near_w = near_h * aspect
        far_h = far * tan_fov
        far_w = far_h * aspect

        px, py, pz = position
        fx, fy, fz = forward
        ux, uy, uz = up
        rx, ry, rz = right

        def _add(a, b):
            return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

        def _scale(v, s):
            return (v[0]*s, v[1]*s, v[2]*s)

        nc = _add(position, _scale(forward, near))
        fc = _add(position, _scale(forward, far))

        # Near plane corners
        ntl = _add(_add(nc, _scale(up, near_h)), _scale(right, -near_w))
        ntr = _add(_add(nc, _scale(up, near_h)), _scale(right, near_w))
        nbl = _add(_add(nc, _scale(up, -near_h)), _scale(right, -near_w))
        nbr = _add(_add(nc, _scale(up, -near_h)), _scale(right, near_w))

        # Far plane corners
        ftl = _add(_add(fc, _scale(up, far_h)), _scale(right, -far_w))
        ftr = _add(_add(fc, _scale(up, far_h)), _scale(right, far_w))
        fbl = _add(_add(fc, _scale(up, -far_h)), _scale(right, -far_w))
        fbr = _add(_add(fc, _scale(up, -far_h)), _scale(right, far_w))

        corners = [ntl, ntr, nbr, nbl, ftl, ftr, fbr, fbl]
        # same edge topology as a cube
        edges = [
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        ]

        c = cls.color
        verts = [[p[0], p[1], p[2], c[0], c[1], c[2]] for p in corners]
        cls._draw_batches.append((verts, edges, list(cls._current_matrix())))

    # ====================================================================
    # Primitive: wire arc / circle
    # ====================================================================

    @classmethod
    def draw_wire_arc(cls, center: Vec3, normal: Vec3, radius: float,
                      start_angle_deg: float = 0.0, arc_deg: float = 360.0,
                      segments: int = 32):
        """Draw a wireframe arc (or full circle) in a plane defined by *normal*."""
        c = cls.color
        mat = list(cls._current_matrix())

        if _HAS_CPP_GIZMOS:
            vert_flat, vert_count, idx_flat = _cpp_wire_arc(
                center[0], center[1], center[2],
                normal[0], normal[1], normal[2],
                radius, start_angle_deg, arc_deg, segments,
                c[0], c[1], c[2])
            if vert_count == 0:
                return
            cls._draw_batches.append((vert_flat.reshape(vert_count, 6), idx_flat, mat))
            return

        # Build local basis from normal
        nx, ny, nz = normal
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length < 1e-8:
            return
        nx, ny, nz = nx/length, ny/length, nz/length

        # Choose a non-parallel axis for cross product
        if abs(ny) < 0.99:
            ax, ay, az = 0, 1, 0
        else:
            ax, ay, az = 1, 0, 0

        # u = normalize(cross(normal, arbitrary))
        ux = ny * az - nz * ay
        uy = nz * ax - nx * az
        uz = nx * ay - ny * ax
        ul = math.sqrt(ux*ux + uy*uy + uz*uz)
        ux, uy, uz = ux/ul, uy/ul, uz/ul

        # v = cross(normal, u)
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux

        cx, cy, cz = center
        c = cls.color
        verts = []
        indices = []

        start_rad = math.radians(start_angle_deg)
        arc_rad = math.radians(arc_deg)

        for i in range(segments + 1):
            angle = start_rad + arc_rad * i / segments
            ca, sa = math.cos(angle), math.sin(angle)
            px = cx + radius * (ca * ux + sa * vx)
            py = cy + radius * (ca * uy + sa * vy)
            pz = cz + radius * (ca * uz + sa * vz)
            verts.append([px, py, pz, c[0], c[1], c[2]])
            if i > 0:
                indices.append(i - 1)
                indices.append(i)

        cls._draw_batches.append((verts, indices, list(cls._current_matrix())))

    # ====================================================================
    # Utility: get packed data for upload
    # ====================================================================

    @classmethod
    def _get_packed_data(cls):
        """Pack all draw batches into contiguous typed buffers for C++ upload.

        Returns:
            ``(vert_buf, vert_count, idx_buf, desc_buf, desc_count)``
            using float32/uint32 NumPy arrays, or ``None`` if empty.
        """
        if not cls._draw_batches:
            return None

        vertex_count = sum(len(vertices) for vertices, _, _ in cls._draw_batches)
        index_count = sum(len(indices) for _, indices, _ in cls._draw_batches)
        vertices_out = np.empty((vertex_count, 6), dtype=np.float32)
        indices_out = np.empty(index_count, dtype=np.uint32)
        # Vertex colour is already carried by each vertex.  The world matrix is
        # therefore the only draw state that requires a separate descriptor.
        # Immediate-mode helpers commonly emit dozens of adjacent lines under
        # one matrix; keep them in one native draw instead of splitting the
        # packed upload back into one GPU draw per helper call.
        descriptor_rows = []
        vert_offset = 0
        idx_offset = 0

        for verts, indices, matrix in cls._draw_batches:
            n_verts = len(verts)
            n_indices = len(indices)

            vertices_out[vert_offset:vert_offset + n_verts] = verts
            target_indices = indices_out[idx_offset:idx_offset + n_indices]
            target_indices[:] = indices
            target_indices += vert_offset
            matrix_key = tuple(matrix)
            if descriptor_rows and descriptor_rows[-1][2] == matrix_key:
                descriptor_rows[-1][1] += n_indices
            else:
                descriptor_rows.append([idx_offset, n_indices, matrix_key])

            vert_offset += n_verts
            idx_offset += n_indices

        descriptors = np.empty((len(descriptor_rows), 18), dtype=np.float32)
        for row_index, (index_start, index_count, matrix) in enumerate(descriptor_rows):
            descriptors[row_index, :2] = index_start, index_count
            descriptors[row_index, 2:] = matrix

        return vertices_out.ravel(), vert_offset, indices_out, descriptors.ravel(), len(descriptor_rows)

    @classmethod
    def _get_resident_data(cls):
        """Return native GPU line descriptors without reading their vertices."""
        return [
            (state.identity, state.vertices._native, state.vertices.shape[0],
             np.ascontiguousarray(state.indices, dtype=np.uint32).reshape(-1), matrix)
            for state, matrix in cls._resident_draw_batches
        ]

    # ====================================================================
    # Utility: get packed icon data for upload
    # ====================================================================

    @classmethod
    def _get_packed_icon_data(cls):
        """Pack all icon entries into flat ``array.array`` buffers for C++ upload.

        Returns:
                        ``(pos_color_buf, id_buf, kind_buf, icon_count)``
            using stdlib ``array.array`` (no numpy), or ``None`` if empty.

            - ``pos_color_buf``: float32 array ``[x, y, z, r, g, b]`` per icon
            - ``id_buf``: uint32 array ``[lo, hi]`` per icon (64-bit object ID
              split into two 32-bit halves)
                        - ``kind_buf``: uint32 array ``[icon_kind]`` per icon
        """
        if not cls._icon_entries:
            return None

        import array as _array

        pos_color_buf = _array.array('f')   # float32: x,y,z,r,g,b per icon
        id_buf = _array.array('I')          # uint32: lo,hi per icon
        kind_buf = _array.array('I')        # uint32: icon kind per icon

        for position, object_id, color, icon_kind in cls._icon_entries:
            pos_color_buf.extend([
                position[0], position[1], position[2],
                color[0], color[1], color[2],
            ])
            lo = object_id & 0xFFFFFFFF
            hi = (object_id >> 32) & 0xFFFFFFFF
            id_buf.append(lo)
            id_buf.append(hi)
            kind_buf.append(int(icon_kind))

        return pos_color_buf, id_buf, kind_buf, len(cls._icon_entries)
