"""
MeshCollider — Python BuiltinComponent wrapper for C++ MeshCollider.

Requires sibling ``MeshRenderer`` geometry. Static and kinematic bodies may
use triangle-mesh collision; switching to a dynamic Rigidbody sets the public
``convex`` property before rebuilding the shape.
"""

from __future__ import annotations

from Infernux.components.builtin.collider import Collider, _native_collider_properties
from Infernux.components.builtin_component import CppProperty
from Infernux.components.fields import FieldType


class MeshCollider(Collider):
    """Python wrapper for the C++ MeshCollider component."""

    _cpp_type_name = "MeshCollider"

    center, is_trigger, physic_material = _native_collider_properties(_cpp_type_name)
    convex = CppProperty.from_native(_cpp_type_name, "convex")
    shape_error = CppProperty(
        "shape_error",
        FieldType.STRING,
        default="",
        readonly=True,
    )
    is_cooking = CppProperty(
        "is_cooking",
        FieldType.BOOL,
        default=False,
        readonly=True,
    )

    def recook(self) -> None:
        """Request collision rebuilding from the current MeshRenderer geometry.

        Visual NumPy mesh updates do not recook collision. This request uses
        a snapshot of the mesh at the time of this call, so subsequent visual
        edits do not change the request. Collider scale/center still apply.
        It uses the existing cooking worker and physics publication boundary; observe
        ``is_cooking`` and ``shape_error`` for completion and errors.
        """
        self._require_cpp_component().recook()

    # ------------------------------------------------------------------
    # Custom inspector: force-check convex when dynamic Rigidbody exists
    # ------------------------------------------------------------------

    def render_inspector(self, ctx) -> None:
        from Infernux.engine.ui.inspector_components import render_builtin_via_setters
        from Infernux.engine.ui.inspector_utils import render_inspector_checkbox

        go = self.game_object
        rb = go.get_component('Rigidbody')
        forced_convex = rb is not None and not rb.is_kinematic

        if forced_convex:
            render_builtin_via_setters(ctx, self, type(self), skip_fields={'convex', 'shape_error', 'is_cooking'})
            ctx.begin_disabled(True)
            render_inspector_checkbox(ctx, "Convex", self.convex)
            ctx.end_disabled()
        else:
            render_builtin_via_setters(ctx, self, type(self), skip_fields={'shape_error', 'is_cooking'})

        shape_error = self.shape_error
        if shape_error:
            from Infernux.engine.ui.theme import ImGuiCol, Theme
            ctx.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
            ctx.text_wrapped(shape_error)
            ctx.pop_style_color(1)
