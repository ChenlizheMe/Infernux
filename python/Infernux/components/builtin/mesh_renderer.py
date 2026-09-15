"""
MeshRenderer — Python InxComponent wrapper for the C++ MeshRenderer component.

Exposes shadow settings and material access as CppProperty descriptors.
Mesh data access (vertices, normals, UVs, indices) is provided via
delegate methods.

Example::

    from Infernux.components.builtin import MeshRenderer

    class MyShadowToggle(InxComponent):
        def start(self):
            mr = self.game_object.get_component(MeshRenderer)
            mr.casts_shadows = False
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from Infernux.components.builtin_component import BuiltinComponent, CppProperty
from Infernux.components.fields import FieldType


def _to_native_material(value):
    """Unwrap a Python Material wrapper to native InxMaterial.

    Passes through strings (GUIDs) and None unchanged.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    native = getattr(value, "_native", None) or getattr(value, "native", None)
    if native is not None:
        return native
    return value


class MeshRenderer(BuiltinComponent):
    """Python wrapper for the C++ MeshRenderer component.

    Properties delegate to the C++ ``MeshRenderer`` via CppProperty.

    Material properties follow Unity naming conventions:

    * ``material`` / ``sharedMaterial`` \u2014 slot 0
    * ``materials`` / ``sharedMaterials`` \u2014 all slots

    Accepts any of: Python ``Material`` wrapper, native ``InxMaterial``,
    GUID string, or ``None``.
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
    # Material — Unity-style API
    # ------------------------------------------------------------------

    @property
    def material(self):
        """The material for slot 0 (Unity-style)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_material(0)
        return None

    @material.setter
    def material(self, value) -> None:
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_material(0, _to_native_material(value))

    @property
    def sharedMaterial(self):
        """Alias for ``material`` (Unity compatibility)."""
        return self.material

    @sharedMaterial.setter
    def sharedMaterial(self, value) -> None:
        self.material = value

    @property
    def materials(self) -> list:
        """All materials across every slot (Unity-style)."""
        cpp = self._cpp_component
        if cpp is None:
            return []
        return [cpp.get_material(i) for i in range(cpp.material_count)]

    @materials.setter
    def materials(self, value: list) -> None:
        cpp = self._cpp_component
        if cpp is None:
            return
        for i, mat in enumerate(value):
            cpp.set_material(i, _to_native_material(mat))

    @property
    def sharedMaterials(self) -> list:
        """Alias for ``materials`` (Unity compatibility)."""
        return self.materials

    @sharedMaterials.setter
    def sharedMaterials(self, value: list) -> None:
        self.materials = value

    @property
    def material_guid(self) -> str:
        """The material GUID for slot 0 (empty string if none)."""
        cpp = self._cpp_component
        if cpp is not None:
            guids = cpp.get_material_guids()
            return guids[0] if guids else ""
        return ""

    @material_guid.setter
    def material_guid(self, value: str) -> None:
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_material(0, value)

    def has_render_material(self) -> bool:
        """Check if a custom material is assigned at slot 0."""
        return self.material is not None

    def get_effective_material(self, slot: int = 0):
        """Get the effective material at slot (custom or default)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_effective_material(slot)
        return None

    # ------------------------------------------------------------------
    # Multi-material API  (Unity Renderer alignment)
    # ------------------------------------------------------------------

    @property
    def material_count(self) -> int:
        """Number of material slots."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.material_count
        return 0

    materialCount = material_count  # Unity PascalCase alias

    def get_material(self, slot: int):
        """Get the material at a given slot index."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_material(slot)
        return None

    def set_material(self, slot_or_material, material=None) -> None:
        """Set a material.  Two calling conventions::

            mr.set_material(0, some_material)   # slot + material
            mr.set_material(some_material)       # slot 0 shorthand

        Accepts Material wrapper, InxMaterial, GUID string, or None.
        """
        cpp = self._cpp_component
        if cpp is None:
            return
        if material is None and not isinstance(slot_or_material, int):
            cpp.set_material(0, _to_native_material(slot_or_material))
        else:
            cpp.set_material(slot_or_material, _to_native_material(material))

    def get_material_guids(self) -> List[str]:
        """Get all material slot GUIDs as a list."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_material_guids()
        return []

    # ---- Unity Renderer.GetMaterials / GetSharedMaterials ----

    def get_materials(self, result: Optional[list] = None) -> list:
        """Return all materials as a list.

        If *result* is provided it is cleared and filled in-place
        (Unity non-alloc pattern); otherwise a new list is returned.

        Matches Unity ``Renderer.GetMaterials``.
        """
        mats = self.materials
        if result is not None:
            result.clear()
            result.extend(mats)
            return result
        return mats

    GetMaterials = get_materials  # Unity PascalCase alias

    def get_shared_materials(self, result: Optional[list] = None) -> list:
        """Return all shared materials (equivalent to ``get_materials``).

        Matches Unity ``Renderer.GetSharedMaterials``.
        """
        return self.get_materials(result)

    GetSharedMaterials = get_shared_materials  # Unity PascalCase alias

    # ---- Unity Renderer.SetMaterials / SetSharedMaterials ----

    def set_materials(self, materials_list: list) -> None:
        """Set all material slots from a list.

        Accepts Material wrappers, InxMaterial objects, GUID strings, or None.
        Matches Unity ``Renderer.SetMaterials``.
        """
        cpp = self._cpp_component
        if cpp is None:
            return
        for i, mat in enumerate(materials_list):
            native = _to_native_material(mat)
            if isinstance(native, str):
                cpp.set_material(i, native)
            else:
                cpp.set_material(i, native)

    SetMaterials = set_materials  # Unity PascalCase alias

    def set_shared_materials(self, materials_list: list) -> None:
        """Alias for ``set_materials`` (Unity compatibility).

        Matches Unity ``Renderer.SetSharedMaterials``.
        """
        self.set_materials(materials_list)

    SetSharedMaterials = set_shared_materials  # Unity PascalCase alias

    def set_material_slot_count(self, count: int) -> None:
        """Set the number of material slots."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_material_slot_count(count)

    def set_parameter(
        self,
        name: str,
        value,
        *,
        material_slot: int = 0,
        persistent: bool = False,
        owner: str = "script",
    ) -> None:
        """Override one reflected material parameter for this renderer.

        Runtime overrides are the default and disappear with this component's
        Play/runtime lifetime. Set ``persistent=True`` for authored scene data.
        Neither mode modifies or clones the shared material.
        Matrix parameters accept NumPy (4,4) arrays indexed [row, column];
        legacy flat 16-number sequences remain column-major.
        """
        cpp = self._require_cpp_component()
        cpp.set_parameter(name, value, material_slot, persistent, owner)

    def get_parameter(
        self,
        name: str,
        *,
        material_slot: int = 0,
        persistent_only: bool = False,
        owner: str = "",
    ):
        """Return this renderer's override, or ``None`` when it inherits the material.

        With the default empty ``owner``, this reads the effective value after
        runtime writers and persistent scene data are resolved.  A non-empty
        ``owner`` inspects only that runtime writer; it does not fall through
        to another writer or to persistent data.
        """
        cpp = self._require_cpp_component()
        return cpp.get_parameter(name, material_slot, persistent_only, owner)

    def remove_parameter(
        self,
        name: str,
        *,
        material_slot: int = 0,
        persistent: bool = False,
        owner: str = "script",
    ) -> bool:
        """Remove one override without disturbing other parameter writers."""
        cpp = self._require_cpp_component()
        return bool(cpp.remove_parameter(name, material_slot, persistent, owner))

    def clear_parameters(
        self, *, material_slot: int = 0, persistent: bool = False, owner: str = "script"
    ) -> None:
        """Clear one slot's persistent values or one runtime writer.

        Runtime clearing only removes values written by ``owner``.  Other
        systems using the same renderer keep their values.
        """
        cpp = self._require_cpp_component()
        cpp.clear_parameters(material_slot, persistent, owner)

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
        """Get the InxMesh asset object (None if no asset mesh)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_mesh_asset()
        return None

    @property
    def mesh(self):
        """The shared public Mesh resource; reading never creates a copy."""
        native = self.get_mesh_asset()
        if native is None:
            return None
        from Infernux.core.mesh import Mesh

        return Mesh.from_native(native)

    @mesh.setter
    def mesh(self, value) -> None:
        cpp = self._require_cpp_component()
        if value is None:
            cpp.clear_mesh_asset()
            return
        native = getattr(value, "native", value)
        guid = str(getattr(native, "guid", "") or "")
        if not guid:
            raise ValueError("MeshRenderer.mesh requires a registered Mesh resource")
        cpp.set_mesh_asset_guid(guid)

    @property
    def shared_mesh(self):
        """Alias for :attr:`mesh`; use ``mesh.copy()`` for independent data."""
        return self.mesh

    @shared_mesh.setter
    def shared_mesh(self, value) -> None:
        self.mesh = value

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

    @property
    def inline_mesh_version(self) -> int:
        """Monotonic generation used to publish inline geometry."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.inline_mesh_version
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

    def get_tangents(self) -> List[Tuple[float, float, float, float]]:
        """Get all vertex tangent frames as (x, y, z, handedness)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.get_tangents()
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

    def set_inline_mesh_data(
        self, positions, normals, uvs, indices, name: str = "Inline Mesh", tangents=None,
    ) -> None:
        """Replace the mesh from NumPy arrays, copying data into native storage.

        Positions/normals are (N, 3), UVs are (N, 2), and indices are flat.
        Pass normals=None to generate area-weighted vertex normals in native
        code. Pass tangents=None to derive tangent frames from positions,
        normals and UVs. Shared indices are smoothed; duplicated seam vertices
        are not welded. Degenerate/unreferenced vertices receive deterministic
        zero-normal/default-tangent attributes. Explicit attributes are
        preserved.
        Native bindings validate shapes and convert to float32/uint32.
        This replaces the whole mesh; it is not a GPU-resident update API.
        Rendering uploads the new geometry on its next update. Existing
        collision stays unchanged until MeshCollider.recook() is requested.
        """
        self._require_cpp_component().set_inline_mesh_data(positions, normals, uvs, indices, name, tangents)

    def recalculate_normals(self) -> None:
        """Rebuild normals on a CPU inline mesh; resident meshes derive them on GPU."""
        self._require_cpp_component().recalculate_normals()

    def recalculate_tangents(self) -> None:
        """Rebuild tangent frames on a CPU inline mesh."""
        self._require_cpp_component().recalculate_tangents()

    def recalculate_bounds(self) -> None:
        """Rebuild local bounds on a CPU inline mesh."""
        self._require_cpp_component().recalculate_bounds()

    def create_vertex_buffer(
        self, *, device: str = "gpu", auto_normals: bool = True,
        auto_tangents: bool = True, capacity: int | None = None,
    ):
        """Create ``inx.buffer`` storage from the current inline mesh.

        The buffer uses the engine's canonical interleaved vertex layout. Its
        first six float columns are position and normal; the remaining columns
        preserve tangent, colour, UV, skin-index bits, and skin weights. Bind
        it with :meth:`set_vertex_buffer` after compute kernels are ready to
        update it in place. By default, kernels that touch this stream trigger
        one GPU normal rebuild at the end of their engine phase. Pass
        ``auto_normals=False`` only when a kernel writes deliberate normals.
        Tangents are rebuilt from the current positions and authored UVs by
        default; disable ``auto_tangents`` when a kernel supplies its own
        tangent frame or the material never consumes one. Derived attributes
        follow the authored index topology, so UV/hard-edge splits stay split.
        ``capacity`` reserves canonical vertex slots without increasing the
        effective draw range.  It must be at least the current vertex count;
        the initial authored stream occupies the prefix and spare slots remain
        zeroed.  No mesh data is read back during rendering.
        """
        if device != "gpu":
            raise ValueError("Mesh vertex buffers must use device='gpu'")
        import numpy as np
        from Infernux.compute import buffer, _enable_automatic_mesh_attributes

        cpp = self._require_cpp_component()
        data = np.asarray(cpp.get_vertex_buffer_data(), dtype=np.float32)
        vertex_count = int(data.shape[0])
        if capacity is None:
            capacity = vertex_count
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("Mesh vertex buffer capacity must be an integer")
        if capacity < vertex_count:
            raise ValueError("Mesh vertex buffer capacity cannot be smaller than vertex_count")
        storage = np.zeros((capacity, data.shape[1]), dtype=np.float32)
        storage[:vertex_count] = data
        result = buffer(shape=storage.shape, dtype=np.float32, device="gpu", data=storage)
        if auto_normals or auto_tangents:
            _enable_automatic_mesh_attributes(
                result,
                np.asarray(cpp.get_positions(), dtype=np.float32),
                np.asarray(cpp.get_indices(), dtype=np.int32),
                normals=bool(auto_normals),
                tangents=bool(auto_tangents),
            )
        return result

    def set_vertex_buffer(self, value, bounds_min, bounds_max, *, space="local") -> None:
        """Render directly from a resident GPU vertex buffer.

        ``value`` must come from :meth:`create_vertex_buffer`. ``space`` is
        explicit: local vertices use the GameObject Transform, while world
        vertices already contain their final position. Bounds use the same
        coordinate space and compute writes never trigger a hidden readback.
        """
        from Infernux.compute import Buffer

        if not isinstance(value, Buffer) or value.device != "gpu":
            raise TypeError("MeshRenderer.set_vertex_buffer expects a GPU inx.buffer")
        value._require_open()
        if value.dtype != "float32" or len(value.shape) != 2 or value.shape[1] != 23:
            raise ValueError(
                "MeshRenderer vertex storage must use float32 shape (capacity, 23)"
            )
        if value.shape[0] < self.vertex_count:
            raise ValueError(
                "MeshRenderer vertex storage capacity cannot be smaller than vertex_count"
            )
        if space not in {"local", "world"}:
            raise ValueError("MeshRenderer vertex buffer space must be 'local' or 'world'")
        from Infernux.math.coerce import coerce_vec3

        self._require_cpp_component().set_vertex_buffer(
            value._native, coerce_vec3(bounds_min), coerce_vec3(bounds_max), space == "world"
        )

    def clear_vertex_buffer(self) -> None:
        """Return rendering to the authored CPU vertex stream."""
        self._require_cpp_component().clear_vertex_buffer()

    @property
    def vertex_buffer_capacity(self) -> int:
        """Allocated resident vertex slots; zero when no GPU stream is bound."""
        return int(self._require_cpp_component().vertex_buffer_capacity)

    def set_primitive_mesh(self, primitive_type) -> None:
        """Assign one of the built-in primitive meshes."""
        cpp = self._cpp_component
        if cpp is not None:
            cpp.set_primitive_mesh(primitive_type)

    def set_mesh_asset_guid(self, guid: str) -> None:
        """Assign a model/mesh asset by GUID."""
        cpp = self._cpp_component
        if cpp is not None and hasattr(cpp, "set_mesh_asset_guid"):
            cpp.set_mesh_asset_guid(guid or "")

    def clear_mesh_asset(self) -> None:
        """Clear the assigned asset mesh."""
        cpp = self._cpp_component
        if cpp is not None and hasattr(cpp, "clear_mesh_asset"):
            cpp.clear_mesh_asset()

    def serialize(self) -> str:
        """Serialize MeshRenderer to JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.serialize()
        return "{}"
