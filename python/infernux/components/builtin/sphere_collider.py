"""
SphereCollider — Python BuiltinComponent wrapper for C++ SphereCollider.

Draws a green wireframe sphere Gizmo in the Scene View.

Example::

    from infernux.components.builtin import SphereCollider

    class MyScript(InxComponent):
        def start(self):
            col = self.game_object.get_component(SphereCollider)
            col.radius = 2.0
"""

from __future__ import annotations

from infernux.components.builtin_component import CppProperty
from infernux.components.builtin.collider import Collider, _native_collider_properties


class SphereCollider(Collider):
    """Python wrapper for the C++ SphereCollider component."""

    _cpp_type_name = "SphereCollider"

    center, is_trigger, physic_material = _native_collider_properties(_cpp_type_name)
    radius = CppProperty.from_native(_cpp_type_name, "radius")

    # ------------------------------------------------------------------
    # Gizmos — green wireframe sphere
    # ------------------------------------------------------------------

    def on_draw_gizmos_selected(self):
        """Draw green wireframe sphere when selected (Unity-style)."""
        from infernux.gizmos import Gizmos

        transform = self.transform
        if transform is None:
            return

        scale = transform.lossy_scale

        cpp = self._get_bound_native_component()
        if cpp is None:
            return

        r = cpp.radius

        world_center = tuple(transform.transform_point(cpp.center))

        # Sphere uses max scale axis for uniform radius (matches C++ physics shape)
        max_scale = max(abs(scale.x), abs(scale.y), abs(scale.z))
        final_radius = r * max_scale + 0.02

        old_color = Gizmos.color
        Gizmos.color = (0.53, 1.0, 0.29)
        Gizmos.draw_wire_sphere(world_center, final_radius)
        Gizmos.color = old_color
