"""
MeshRenderer — Python InfComponent wrapper for the C++ MeshRenderer component.

Exposes shadow settings and material access as CppProperty descriptors.
Mesh data access (vertices, normals, UVs, indices) is provided via
delegate methods.

Example::

    from InfEngine.components.builtin import MeshRenderer

    class MyShadowToggle(InfComponent):
        def start(self):
            mr = self.get_component(MeshRenderer)
            mr.casts_shadows = False
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from InfEngine.components.builtin_component import BuiltinComponent, CppProperty
from InfEngine.components.serialized_field import FieldType


class MeshRenderer(BuiltinComponent):
    """Python wrapper for the C++ MeshRenderer component.

    Properties delegate to the C++ ``MeshRenderer`` via CppProperty.
    """

    _cpp_type_name = "MeshRenderer"
    _component_category_ = "Rendering"

    # ---- Shadow settings ----
    casts_shadows = CppProperty(
        "casts_shadows",
        FieldType.BOOL,
        default=True,
        tooltip="Whether this renderer casts shadows",
    )
    receives_shadows = CppProperty(
        "receives_shadows",
        FieldType.BOOL,
        default=True,
        tooltip="Whether this renderer receives shadows",
    )

    # ------------------------------------------------------------------
    # Material (non-CppProperty — complex reference type)
    # ------------------------------------------------------------------

    @property
    def render_material(self):
        """The material used for rendering (InfMaterial or None)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.render_material
        return None

    @render_material.setter
    def render_material(self, value) -> None:
        cpp = self._cpp_component
        if cpp is not None:
            cpp.render_material = value

    @property
    def material_guid(self) -> str:
        """The material GUID (empty string if no material assigned)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.material_guid
        return ""

    @material_guid.setter
    def material_guid(self, value: str) -> None:
        cpp = self._cpp_component
        if cpp is not None:
            cpp.material_guid = value

    def has_render_material(self) -> bool:
        """Check if a custom material is assigned."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.has_render_material()
        return False

    def get_effective_material(self, slot: int = 0):
        """Get the effective material for a given slot (custom or default)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_effective_material(slot)
        return None

    # ------------------------------------------------------------------
    # Multi-material API
    # ------------------------------------------------------------------

    @property
    def material_count(self) -> int:
        """Number of material slots."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.material_count
        return 0

    def get_material(self, slot: int):
        """Get the material at a given slot index."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_material(slot)
        return None

    def set_material(self, slot: int, guid: str) -> None:
        """Set the material at a given slot index by GUID."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_material(slot, guid)

    def get_material_guids(self) -> List[str]:
        """Get all material slot GUIDs as a list."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_material_guids()
        return []

    def set_materials(self, guids: List[str]) -> None:
        """Set all material slots from a list of GUIDs."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_materials(guids)

    def set_material_slot_count(self, count: int) -> None:
        """Set the number of material slots."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_material_slot_count(count)

    # ------------------------------------------------------------------
    # Mesh data access (read-only, for AI / CV / inspection)
    # ------------------------------------------------------------------

    def has_inline_mesh(self) -> bool:
        """Check if the renderer has inline mesh data."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.has_inline_mesh()
        return False

    @property
    def has_mesh_asset(self) -> bool:
        """Check if the renderer references an asset-managed mesh."""
        cpp = self._cpp_component
        if cpp is not None:
            return bool(cpp.has_mesh_asset)
        return False

    @property
    def mesh_asset_guid(self) -> str:
        """GUID of the referenced mesh asset, if any."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.mesh_asset_guid
        return ""

    @property
    def mesh_name(self) -> str:
        """Display name of the referenced mesh asset, if any."""
        cpp = self._cpp_component
        if cpp is not None:
            return getattr(cpp, "mesh_name", "")
        return ""

    def get_mesh_asset(self):
        """Get the InfMesh asset object (None if no asset mesh)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_mesh_asset()
        return None

    def get_material_slot_names(self) -> List[str]:
        """Material slot names from the model file (e.g. 'Body', 'Glass')."""
        mesh = self.get_mesh_asset()
        if mesh is not None:
            names = list(mesh.material_slot_names)
            cpp = self._cpp_component
            submesh_index = getattr(cpp, "submesh_index", -1) if cpp is not None else -1
            if submesh_index >= 0 and submesh_index < mesh.submesh_count:
                info = mesh.get_submesh_info(submesh_index)
                slot = int(info.get("material_slot", 0))
                if 0 <= slot < len(names):
                    return [names[slot]]
                sub_name = info.get("name", "")
                return [sub_name] if sub_name else []
            return names
        return []

    def get_submesh_infos(self) -> List[dict]:
        """Get info dicts for each submesh: name, vertex/index counts, material slot."""
        mesh = self.get_mesh_asset()
        if mesh is None:
            return []
        cpp = self._cpp_component
        submesh_index = getattr(cpp, "submesh_index", -1) if cpp is not None else -1
        if submesh_index >= 0 and submesh_index < mesh.submesh_count:
            return [mesh.get_submesh_info(submesh_index)]
        result = []
        for i in range(mesh.submesh_count):
            result.append(mesh.get_submesh_info(i))
        return result

    @property
    def vertex_count(self) -> int:
        """Number of vertices in inline mesh (0 if using resource mesh)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.vertex_count
        return 0

    @property
    def index_count(self) -> int:
        """Number of indices in inline mesh (0 if using resource mesh)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.index_count
        return 0

    def get_positions(self) -> List[Tuple[float, float, float]]:
        """Get all vertex positions as (x, y, z) tuples."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_positions()
        return []

    def get_normals(self) -> List[Tuple[float, float, float]]:
        """Get all vertex normals as (x, y, z) tuples."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_normals()
        return []

    def get_uvs(self) -> List[Tuple[float, float]]:
        """Get all vertex UVs as (u, v) tuples."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_uvs()
        return []

    def get_indices(self) -> List[int]:
        """Get all indices as a flat list."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_indices()
        return []

    def serialize(self) -> str:
        """Serialize MeshRenderer to JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.serialize()
        return "{}"
