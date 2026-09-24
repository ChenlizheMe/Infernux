# MeshRenderer

<div class="class-info">
class in <b>Infernux.components.builtin</b>
</div>

**Inherits from:** [BuiltinComponent](Component.md)

## Description

Renders a mesh with assigned materials.

<!-- USER CONTENT START --> description
**Status:** Preview · **Verified with:** 0.4.0

A MeshRenderer can use an inline primitive or an imported mesh asset and supports multiple material slots. Prove mesh and material assignment before debugging lighting effects.
<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| casts_shadows | `bool` | Whether this renderer casts shadows. |
| receives_shadows | `bool` | Whether this renderer receives shadows. |
| material_guid | `str` | The asset GUID of the material at slot 0. |
| material_count | `int` | The number of material slots on this renderer. *(read-only)* |
| has_mesh_asset | `bool` | Whether a mesh asset is assigned to this renderer. *(read-only)* |
| mesh_asset_guid | `str` | The asset GUID of the assigned mesh. *(read-only)* |
| mesh_name | `str` | The name of the assigned mesh. *(read-only)* |
| mesh | `Optional[Mesh]` |  |
| shared_mesh | `Optional[Mesh]` |  |
| vertex_count | `int` | The number of vertices in the mesh. *(read-only)* |
| index_count | `int` | The number of indices in the mesh. *(read-only)* |
| inline_mesh_version | `int` |  *(read-only)* |
| vertex_buffer_capacity | `int` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `has_render_material() → bool` | Return whether a material is assigned at slot 0. |
| `get_effective_material(slot: int = ...) → Any` | Return the effective material for the given slot, including fallbacks. |
| `get_material(slot: int) → Any` | Return the material at the specified slot index. |
| `get_material_guids() → List[str]` | Return the list of material GUIDs for all slots. |
| `set_materials(guids: List[str]) → None` | Set all material slots from a list of asset GUIDs. |
| `set_material_slot_count(count: int) → None` | Set the number of material slots on this renderer. |
| `set_parameter(name: str, value: Any, material_slot: int = ..., persistent: bool = ..., owner: str = ...) → None` |  |
| `get_parameter(name: str, material_slot: int = ..., persistent_only: bool = ..., owner: str = ...) → Any` |  |
| `remove_parameter(name: str, material_slot: int = ..., persistent: bool = ..., owner: str = ...) → bool` |  |
| `clear_parameters(material_slot: int = ..., persistent: bool = ..., owner: str = ...) → None` |  |
| `has_inline_mesh() → bool` | Return whether the renderer has an inline (non-asset) mesh. |
| `get_mesh_asset() → Any` | Return the InxMesh asset object, or None. |
| `get_material_slot_names() → List[str]` | Return material slot names from the model file. |
| `get_submesh_infos() → List[Dict[str, Any]]` | Return info dicts for each submesh. |
| `get_positions() → List[Tuple[float, float, float]]` | Return the list of vertex positions. |
| `get_normals() → List[Tuple[float, float, float]]` | Return the list of vertex normals. |
| `get_tangents() → List[Tuple[float, float, float, float]]` | Return tangent direction and handedness for every vertex. |
| `get_uvs() → List[Tuple[float, float]]` | Return the list of UV coordinates. |
| `get_indices() → List[int]` | Return the list of triangle indices. |
| `set_inline_mesh_data(positions: Any, normals: Any, uvs: Any, indices: Any, name: str = 'Inline Mesh', tangents: Any = None) → None` | Copy NumPy geometry; omitted normals/tangents are derived. |
| `recalculate_normals() → None` |  |
| `recalculate_tangents() → None` |  |
| `recalculate_bounds() → None` |  |
| `create_vertex_buffer(device: str = 'gpu', auto_normals: bool = True, auto_tangents: bool = True, capacity: int | None = None) → Buffer` | Create resident vertex storage with automatic GPU normal/tangent rebuilding. |
| `set_vertex_buffer(value: Buffer, bounds_min: Any, bounds_max: Any, space: str = ...) → None` | Render directly from resident vertex storage in explicit local or world space. |
| `clear_vertex_buffer() → None` | Return rendering to the authored CPU vertex stream. |
| `set_primitive_mesh(primitive_type: Any) → None` | Assign one of the built-in primitive meshes. |
| `set_mesh_asset_guid(guid: str) → None` | Assign a model/mesh asset by GUID. |
| `clear_mesh_asset() → None` | Clear the assigned asset mesh. |
| `serialize() → str` | Serialize the component to a JSON string. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example
```python
import infernux as inx

display = inx.GameObject.find("DisplayObject")
if display is not None:
    renderer = display.get_component(inx.MeshRenderer)
    if renderer is not None:
        renderer.set_primitive_mesh(inx.PrimitiveType.Cube)
        renderer.casts_shadows = True
```
<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also
- [Material](Material.md)
- [Light](Light.md)
- [RenderStack](RenderStack.md)
<!-- USER CONTENT END -->
