from __future__ import annotations

from typing import Any

from infernux.components.builtin.collider import Collider

class BoxCollider(Collider):
    """A box-shaped collider primitive."""

    _cpp_type_name: str

    # ---- CppProperty fields as properties ----

    @property
    def size(self) -> Any:
        """The size of the box collider in local space."""
        ...
    @size.setter
    def size(self, value: Any) -> None: ...

    # ---- Gizmos ----

    def on_draw_gizmos_selected(self) -> None:
        """Draw the collider wireframe when selected in the editor."""
        ...
