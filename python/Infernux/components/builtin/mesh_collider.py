"""
MeshCollider — Python BuiltinComponent wrapper for C++ MeshCollider.

Uses sibling ``MeshRenderer`` geometry when available. Static and kinematic
bodies use triangle-mesh collision; dynamic rigidbodies automatically use a
convex hull, matching common engine constraints.
"""

from __future__ import annotations

from Infernux.components.builtin.collider import Collider
from Infernux.components.builtin_component import CppProperty
from Infernux.debug import Debug
from Infernux.components.serialized_field import FieldType


class MeshCollider(Collider):
    """Python wrapper for the C++ MeshCollider component."""

    _cpp_type_name = "MeshCollider"

    convex = CppProperty(
        "convex",
        FieldType.BOOL,
        default=False,
        tooltip="Use convex hull collision. Dynamic rigidbodies force convex mode.",
    )

    # ------------------------------------------------------------------
    # Custom inspector: force-check convex when dynamic Rigidbody exists
    # ------------------------------------------------------------------

    def render_inspector(self, ctx) -> None:
        from Infernux.engine.ui.inspector_components import render_builtin_via_setters
        from Infernux.engine.ui.inspector_utils import render_inspector_checkbox

        go = getattr(self, 'game_object', None)
        forced_convex = False
        if go is not None:
            rb = go.get_component('Rigidbody')
            if rb is not None:
                try:
                    forced_convex = not rb.is_kinematic
                except (RuntimeError, AttributeError) as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                    pass

        if forced_convex:
            render_builtin_via_setters(ctx, self, type(self), skip_fields={'convex'})
            ctx.begin_disabled(True)
            render_inspector_checkbox(ctx, "Convex", True)
            ctx.end_disabled()
        else:
            render_builtin_via_setters(ctx, self, type(self))
