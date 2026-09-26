from __future__ import annotations

from Infernux.components.builtin.collider import Collider

class MeshCollider(Collider):
    """A collider that uses a mesh shape."""

    _cpp_type_name: str

    def recook(self) -> None:
        """Request collision rebuilding after visual mesh updates."""
        ...

    # ---- CppProperty fields as properties ----

    @property
    def convex(self) -> bool:
        """Whether the mesh collider uses a convex hull."""
        ...
    @convex.setter
    def convex(self, value: bool) -> None: ...

    @property
    def shape_error(self) -> str:
        """Last native mesh cooking error, or an empty string after success."""
        ...

    @property
    def is_cooking(self) -> bool:
        """Whether collision geometry is currently cooking on a worker."""
        ...
