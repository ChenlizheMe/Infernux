"""Public, versioned Mesh resource.

``Mesh`` is the scripting surface over the engine's canonical ``InxMesh``
resource. Imported and runtime-created meshes use the same AssetRegistry
identity, publication, sharing, and renderer invalidation path.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from Infernux.lib import AssetRegistry, InxMesh


class Mesh:
    """A shared Mesh resource with explicit copy and mutation semantics."""

    def __init__(self, native: InxMesh):
        if native is None:
            raise ValueError("Cannot wrap a None InxMesh")
        self._native = native

    @staticmethod
    def create(name: str = "Mesh") -> "Mesh":
        """Create an empty transient Mesh in the authoritative registry."""
        return Mesh(AssetRegistry.instance().create_runtime_mesh(name))

    @staticmethod
    def from_data(
        positions: Any,
        indices: Any,
        *,
        normals: Any = None,
        uvs: Any = None,
        tangents: Any = None,
        colors: Any = None,
        submeshes: Optional[Iterable[Mapping[str, Any]]] = None,
        material_slots: Sequence[str] = (),
        name: str = "Mesh",
    ) -> "Mesh":
        """Create and populate a transient Mesh from NumPy-compatible arrays."""
        mesh = Mesh.create(name)
        mesh.set_data(
            positions,
            indices,
            normals=normals,
            uvs=uvs,
            tangents=tangents,
            colors=colors,
            submeshes=submeshes,
            material_slots=material_slots,
        )
        return mesh

    @staticmethod
    def load(path: str) -> Optional["Mesh"]:
        """Load the registered Mesh at *path*, sharing its live asset identity."""
        native = AssetRegistry.instance().load_mesh(path)
        return Mesh(native) if native is not None else None

    @staticmethod
    def load_guid(guid: str) -> Optional["Mesh"]:
        """Load a Mesh by GUID, sharing its live asset identity."""
        native = AssetRegistry.instance().load_mesh_by_guid(guid)
        return Mesh(native) if native is not None else None

    @staticmethod
    def from_native(native: InxMesh) -> "Mesh":
        return Mesh(native)

    @property
    def native(self) -> InxMesh:
        return self._native

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def guid(self) -> str:
        return self._native.guid

    @property
    def file_path(self) -> str:
        return self._native.file_path

    @property
    def generation(self) -> int:
        return int(self._native.generation)

    @property
    def vertex_count(self) -> int:
        return int(self._native.vertex_count)

    @property
    def index_count(self) -> int:
        return int(self._native.index_count)

    @property
    def submesh_count(self) -> int:
        return int(self._native.submesh_count)

    @property
    def material_slots(self) -> tuple[str, ...]:
        return tuple(self._native.material_slot_names)

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return tuple(self._native.get_bounds())

    @property
    def vertex_buffer(self) -> dict[str, Any]:
        """Return independent NumPy copies of all CPU-readable vertex streams."""
        return dict(self._native.get_vertex_data())

    @property
    def index_buffer(self) -> Any:
        """Return an independent uint32 NumPy copy of the triangle indices."""
        return self._native.get_index_data()

    def get_submesh(self, index: int) -> dict[str, Any]:
        return dict(self._native.get_submesh_info(index))

    def copy(self, name: str = "") -> "Mesh":
        """Create an independently versioned copy; reading never clones implicitly."""
        return Mesh(AssetRegistry.instance().copy_mesh(self.guid, name))

    def destroy(self) -> None:
        """Destroy a transient Mesh and invalidate every renderer using it."""
        AssetRegistry.instance().destroy_runtime_mesh(self.guid)

    def set_data(
        self,
        positions: Any,
        indices: Any,
        *,
        normals: Any = None,
        uvs: Any = None,
        tangents: Any = None,
        colors: Any = None,
        submeshes: Optional[Iterable[Mapping[str, Any]]] = None,
        material_slots: Sequence[str] = (),
    ) -> None:
        """Atomically replace geometry, topology, submeshes, and material slots.

        Positions/normals/colors use ``(N, 3)``, UVs use ``(N, 2)``, tangents
        use ``(N, 4)``, and indices are a flat uint32-compatible array. Omitted
        normals and tangents are derived once during publication. The registry
        copies all inputs; later caller mutations do not affect the Mesh.
        """
        AssetRegistry.instance().set_mesh_data(
            self.guid,
            positions,
            indices,
            normals,
            uvs,
            tangents,
            colors,
            None if submeshes is None else list(submeshes),
            list(material_slots),
        )

    def update_vertices(
        self,
        first: int,
        *,
        positions: Any = None,
        normals: Any = None,
        uvs: Any = None,
        tangents: Any = None,
        colors: Any = None,
    ) -> None:
        """Atomically update equal-length vertex-stream ranges in place.

        Unspecified streams are preserved. Position edits do not silently
        rebuild normals/tangents or collider shapes; request those operations
        explicitly when needed.
        """
        AssetRegistry.instance().update_mesh_vertices(
            self.guid, first, positions, normals, uvs, tangents, colors
        )

    def recalculate_normals(self) -> None:
        AssetRegistry.instance().recalculate_mesh_normals(self.guid)

    def recalculate_tangents(self) -> None:
        AssetRegistry.instance().recalculate_mesh_tangents(self.guid)

    def serialize_source(self) -> bytes:
        """Encode the Mesh in Infernux's authoring format without writing a file."""
        return self._native.serialize_source()

    def __repr__(self) -> str:
        return (
            f"Mesh(name={self.name!r}, vertices={self.vertex_count}, "
            f"indices={self.index_count}, guid={self.guid!r})"
        )
