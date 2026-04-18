from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from Infernux.components.builtin_component import BuiltinComponent

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
    def set_material(self, slot: int, guid: str) -> None:
        """Assign a material to the specified slot by asset GUID."""
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

    def get_positions(self) -> List[Tuple[float, float, float]]:
        """Return the list of vertex positions."""
        ...
    def get_normals(self) -> List[Tuple[float, float, float]]:
        """Return the list of vertex normals."""
        ...
    def get_uvs(self) -> List[Tuple[float, float]]:
        """Return the list of UV coordinates."""
        ...
    def get_indices(self) -> List[int]:
        """Return the list of triangle indices."""
        ...

    # ---- Serialization ----

    def serialize(self) -> str:
        """Serialize the component to a JSON string."""
        ...
