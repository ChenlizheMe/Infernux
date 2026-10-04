from __future__ import annotations

from infernux.components.builtin.collider import Collider

class CapsuleCollider(Collider):
    """A capsule-shaped collider primitive."""

    _cpp_type_name: str

    # ---- CppProperty fields as properties ----

    @property
    def radius(self) -> float:
        """The radius of the capsule collider."""
        ...
    @radius.setter
    def radius(self, value: float) -> None: ...

    @property
    def height(self) -> float:
        """The height of the capsule collider."""
        ...
    @height.setter
    def height(self, value: float) -> None: ...

    @property
    def direction(self) -> int:
        """The axis direction of the capsule (0=X, 1=Y, 2=Z)."""
        ...
    @direction.setter
    def direction(self, value: int) -> None: ...

    # ---- Gizmos ----

    def on_draw_gizmos_selected(self) -> None:
        """Draw the collider wireframe when selected in the editor."""
        ...
