"""CylinderCollider wrapper for the native Jolt cylinder shape."""

from __future__ import annotations

import math

from Infernux.components.builtin.collider import Collider, _native_collider_properties
from Infernux.components.builtin_component import CppProperty


class CylinderCollider(Collider):
    """A finite cylinder collider with a selectable local axis."""

    _cpp_type_name = "CylinderCollider"

    center, is_trigger, physic_material = _native_collider_properties(_cpp_type_name)
    radius = CppProperty.from_native(_cpp_type_name, "radius")
    height = CppProperty.from_native(_cpp_type_name, "height")
    direction = CppProperty.from_native(_cpp_type_name, "direction")

    def on_draw_gizmos_selected(self):
        from Infernux.gizmos import Gizmos

        transform = self.transform
        cpp = self._get_bound_native_component()
        if transform is None or cpp is None:
            return

        radius = float(cpp.radius)
        half_height = float(cpp.height) * 0.5
        center = cpp.center
        direction = int(cpp.direction)
        if direction == 0:
            axis, tangent, bitangent = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        elif direction == 2:
            axis, tangent, bitangent = (0, 0, 1), (1, 0, 0), (0, 1, 0)
        else:
            axis, tangent, bitangent = (0, 1, 0), (1, 0, 0), (0, 0, 1)

        old_matrix = Gizmos.matrix
        old_color = Gizmos.color
        Gizmos.matrix = transform.local_to_world_matrix()
        Gizmos.color = (0.53, 1.0, 0.29)
        top = tuple(center[i] + axis[i] * half_height for i in range(3))
        bottom = tuple(center[i] - axis[i] * half_height for i in range(3))
        Gizmos.draw_wire_arc(top, axis, radius, 0, 360, 24)
        Gizmos.draw_wire_arc(bottom, axis, radius, 0, 360, 24)
        for angle_degrees in (0, 90, 180, 270):
            angle = math.radians(angle_degrees)
            offset = tuple(
                radius * (math.cos(angle) * tangent[i] + math.sin(angle) * bitangent[i])
                for i in range(3)
            )
            Gizmos.draw_line(
                tuple(top[i] + offset[i] for i in range(3)),
                tuple(bottom[i] + offset[i] for i in range(3)),
            )
        Gizmos.color = old_color
        Gizmos.matrix = old_matrix
