from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, overload

from Infernux.components.builtin_component import BuiltinComponent
from Infernux.compute import Buffer
from Infernux.core.mesh import Mesh

class MeshRenderer(BuiltinComponent):
    """Renders a mesh with assigned materials."""

    _cpp_type_name: str
    _component_category_: str

    # ---- CppProperty fields as properties ----

    @property
    def casts_shadows(self) -> bool:
        """Whether this renderer casts shadows."""
        ...
    @casts_shadows.setter
    def casts_shadows(self, value: bool) -> None: ...

    @property
    def receives_shadows(self) -> bool:
        """Whether this renderer receives shadows."""
        ...
    @receives_shadows.setter
    def receives_shadows(self, value: bool) -> None: ...

    # ---- Material properties ----

    @property
    def material_guid(self) -> str:
        """The asset GUID of the material at slot 0."""
        ...
    @material_guid.setter
    def material_guid(self, value: str) -> None: ...

    def has_render_material(self) -> bool:
        """Return whether a material is assigned at slot 0."""
        ...
    def get_effective_material(self, slot: int = ...) -> Any:
        """Return the effective material for the given slot, including fallbacks."""
        ...

    # ---- Multi-material API ----

    @property
    def material_count(self) -> int:
        """The number of material slots on this renderer."""
        ...

    def get_material(self, slot: int) -> Any:
        """Return the material at the specified slot index."""
        ...
    @overload
    def set_material(self, material: Any) -> None:
        """Assign a material to slot zero."""
        ...
    @overload
    def set_material(self, slot: int, material: Any) -> None:
        """Assign a material to the specified slot."""
        ...
    def get_material_guids(self) -> List[str]:
        """Return the list of material GUIDs for all slots."""
        ...
    def set_materials(self, guids: List[str]) -> None:
        """Set all material slots from a list of asset GUIDs."""
        ...
    def set_material_slot_count(self, count: int) -> None:
        """Set the number of material slots on this renderer."""
        ...
    def set_parameter(
        self,
        name: str,
        value: Any,
        *,
        material_slot: int = ...,
        persistent: bool = ...,
        owner: str = ...,
    ) -> None: ...
    def get_parameter(
        self,
        name: str,
        *,
        material_slot: int = ...,
        persistent_only: bool = ...,
        owner: str = ...,
    ) -> Any: ...
    def remove_parameter(
        self,
        name: str,
        *,
        material_slot: int = ...,
        persistent: bool = ...,
        owner: str = ...,
    ) -> bool: ...
    def clear_parameters(self, *, material_slot: int = ..., persistent: bool = ..., owner: str = ...) -> None: ...

    # ---- Mesh data access ----

    def has_inline_mesh(self) -> bool:
        """Return whether the renderer has an inline (non-asset) mesh."""
        ...

    @property
    def has_mesh_asset(self) -> bool:
        """Whether a mesh asset is assigned to this renderer."""
        ...
    @property
    def mesh_asset_guid(self) -> str:
        """The asset GUID of the assigned mesh."""
        ...
    @property
    def mesh_name(self) -> str:
        """The name of the assigned mesh."""
        ...

    def get_mesh_asset(self) -> Any:
        """Return the InxMesh asset object, or None."""
        ...
    @property
    def mesh(self) -> Optional[Mesh]: ...
    @mesh.setter
    def mesh(self, value: Optional[Mesh]) -> None: ...
    @property
    def shared_mesh(self) -> Optional[Mesh]: ...
    @shared_mesh.setter
    def shared_mesh(self, value: Optional[Mesh]) -> None: ...
    def get_material_slot_names(self) -> List[str]:
        """Return material slot names from the model file."""
        ...
    def get_submesh_infos(self) -> List[Dict[str, Any]]:
        """Return info dicts for each submesh."""
        ...

    @property
    def vertex_count(self) -> int:
        """The number of vertices in the mesh."""
        ...
    @property
    def index_count(self) -> int:
        """The number of indices in the mesh."""
        ...
    @property
    def inline_mesh_version(self) -> int: ...

    def get_positions(self) -> List[Tuple[float, float, float]]:
        """Return the list of vertex positions."""
        ...
    def get_normals(self) -> List[Tuple[float, float, float]]:
        """Return the list of vertex normals."""
        ...
    def get_tangents(self) -> List[Tuple[float, float, float, float]]:
        """Return tangent direction and handedness for every vertex."""
        ...
    def get_uvs(self) -> List[Tuple[float, float]]:
        """Return the list of UV coordinates."""
        ...
    def get_indices(self) -> List[int]:
        """Return the list of triangle indices."""
        ...
    def set_inline_mesh_data(self, positions: Any, normals: Any, uvs: Any, indices: Any, name: str = "Inline Mesh", tangents: Any = None) -> None:
        """Copy NumPy geometry; omitted normals/tangents are derived. No collision recook."""
        ...
    def recalculate_normals(self) -> None: ...
    def recalculate_tangents(self) -> None: ...
    def recalculate_bounds(self) -> None: ...
    def create_vertex_buffer(self, *, device: str = "gpu", auto_normals: bool = True, auto_tangents: bool = True, capacity: int | None = None) -> Buffer:
        """Create resident vertex storage with automatic GPU normal/tangent rebuilding."""
        ...
    def set_vertex_buffer(self, value: Buffer, bounds_min: Any, bounds_max: Any, *, space: str = ...) -> None:
        """Render directly from resident vertex storage in explicit local or world space."""
        ...
    def clear_vertex_buffer(self) -> None:
        """Return rendering to the authored CPU vertex stream."""
        ...
    @property
    def vertex_buffer_capacity(self) -> int: ...
    def set_primitive_mesh(self, primitive_type: Any) -> None:
        """Assign one of the built-in primitive meshes."""
        ...
    def set_mesh_asset_guid(self, guid: str) -> None:
        """Assign a model/mesh asset by GUID."""
        ...
    def clear_mesh_asset(self) -> None:
        """Clear the assigned asset mesh."""
        ...

    # ---- Serialization ----

    def serialize(self) -> str:
        """Serialize the component to a JSON string."""
        ...
