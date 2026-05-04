# MeshRenderer

<div class="class-info">
类位于 <b>Infernux.components.builtin</b>
</div>

**继承自:** [BuiltinComponent](Component.md)

## 描述

使用网格和材质渲染 3D 几何体的组件。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| casts_shadows | `bool` | Whether this renderer casts shadows. |
| receives_shadows | `bool` | Whether this renderer receives shadows. |
| material_guid | `str` | The asset GUID of the material at slot 0. |
| material_count | `int` | The number of material slots on this renderer. *(只读)* |
| has_mesh_asset | `bool` | Whether a mesh asset is assigned to this renderer. *(只读)* |
| mesh_asset_guid | `str` | The asset GUID of the assigned mesh. *(只读)* |
| mesh_name | `str` | The name of the assigned mesh. *(只读)* |
| vertex_count | `int` | The number of vertices in the mesh. *(只读)* |
| index_count | `int` | The number of indices in the mesh. *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `has_render_material() → bool` | Return whether a material is assigned at slot 0. |
| `get_effective_material(slot: int = ...) → Any` | Return the effective material for the given slot, including fallbacks. |
| `get_material(slot: int) → Any` | Return the material at the specified slot index. |
| `set_material(slot: int, guid: str) → None` | Assign a material to the specified slot by asset GUID. |
| `get_material_guids() → List[str]` | Return the list of material GUIDs for all slots. |
| `set_materials(guids: List[str]) → None` | Set all material slots from a list of asset GUIDs. |
| `set_material_slot_count(count: int) → None` | Set the number of material slots on this renderer. |
| `has_inline_mesh() → bool` | Return whether the renderer has an inline (non-asset) mesh. |
| `get_mesh_asset() → Any` | Return the InxMesh asset object, or None. |
| `get_material_slot_names() → List[str]` | Return material slot names from the model file. |
| `get_submesh_infos() → List[Dict[str, Any]]` | Return info dicts for each submesh. |
| `get_positions() → List[Tuple[float, float, float]]` | Return the list of vertex positions. |
| `get_normals() → List[Tuple[float, float, float]]` | Return the list of vertex normals. |
| `get_uvs() → List[Tuple[float, float]]` | Return the list of UV coordinates. |
| `get_indices() → List[int]` | Return the list of triangle indices. |
| `set_primitive_mesh(primitive_type: Any) → None` | Assign one of the built-in primitive meshes. |
| `set_mesh_asset_guid(guid: str) → None` | Assign a model/mesh asset by GUID. |
| `clear_mesh_asset() → None` | Clear the assigned asset mesh. |
| `serialize() → str` | Serialize the component to a JSON string. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example
```python
# TODO: Add example for MeshRenderer
```
<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
