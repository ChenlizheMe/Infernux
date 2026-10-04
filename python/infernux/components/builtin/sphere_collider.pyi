from __future__ import annotations

from infernux.components.builtin.collider import Collider

class SphereCollider(Collider):
    """A sphere-shaped collider primitive."""

    _cpp_type_name: str

    # ---- CppProperty fields as properties ----

    @property
    def radius(self) -> float:
        """The radius of the sphere collider."""
        ...
    @radius.setter
    def radius(self, value: float) -> None: ...

    # ---- Gizmos ----

    def on_draw_gizmos_selected(self) -> None:
        """Draw the collider wireframe when selected in the editor."""
        ...
