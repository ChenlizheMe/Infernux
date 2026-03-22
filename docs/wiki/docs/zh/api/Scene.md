# Scene

<div class="class-info">
类位于 <b>InfEngine</b>
</div>

## 描述

运行时场景，包含 GameObject 层级。

<!-- USER CONTENT START --> description

Scene 代表引擎中的一个运行时场景，包含 [GameObject](GameObject.md) 的层级结构。每个对象都存在于某个场景中，场景提供创建、查找和枚举对象的方法。

使用 `create_game_object()` 在场景中实例化新对象，使用 `find()` 根据名称定位对象，使用 `find_with_tag()` 根据标签搜索。`get_root_game_objects()` 方法仅返回顶层对象（即没有父级的对象），适合遍历整个层级结构。

场景的加载和卸载通过 [SceneManager](SceneManager.md) 完成。当前活动场景决定了新创建对象的默认放置位置。

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| name | `str` | 场景名称。 *(只读)* |
| structure_version | `int` |  *(只读)* |
| main_camera | `Optional[Camera]` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `set_playing(playing: bool) → None` |  |
| `create_game_object(name: str = 'GameObject') → GameObject` |  |
| `create_primitive(type: PrimitiveType, name: str = '') → GameObject` |  |
| `create_from_model(guid: str, name: str = '') → Optional[GameObject]` |  |
| `get_root_objects() → List[GameObject]` |  |
| `get_all_objects() → List[GameObject]` |  |
| `find(name: str) → Optional[GameObject]` |  |
| `find_by_id(id: int) → Optional[GameObject]` |  |
| `find_object_by_id(id: int) → Optional[GameObject]` |  |
| `find_with_tag(tag: str) → Optional[GameObject]` |  |
| `find_game_objects_with_tag(tag: str) → List[GameObject]` |  |
| `find_game_objects_in_layer(layer: int) → List[GameObject]` |  |
| `destroy_game_object(game_object: GameObject) → None` |  |
| `instantiate_game_object(source: GameObject, parent: Optional[GameObject] = None) → Optional[GameObject]` |  |
| `instantiate_from_json(json_str: str, parent: Optional[GameObject] = None) → Optional[GameObject]` |  |
| `process_pending_destroys() → None` |  |
| `is_playing() → bool` |  |
| `awake_object(game_object: GameObject) → None` |  |
| `serialize() → str` |  |
| `deserialize(json_str: str) → None` |  |
| `save_to_file(path: str) → None` |  |
| `load_from_file(path: str) → None` |  |
| `has_pending_py_components() → bool` |  |
| `take_pending_py_components() → List[PendingPyComponent]` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 生命周期方法

| 方法 | 描述 |
|------|------|
| `start() → None` |  |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent
from InfEngine.math import vector3

class SceneSetup(InfComponent):
    def start(self):
        scene = self.game_object.scene

        # 在场景中创建新对象
        ground = scene.create_game_object("Ground")
        ground.transform.position = vector3(0, -0.5, 0)

        # 根据名称查找已有对象
        player = scene.find("Player")
        if player:
            print(f"已找到：{player.name}")

        # 根据标签查找对象
        enemy = scene.find_with_tag("Enemy")

        # 遍历所有根级对象
        roots = scene.get_root_game_objects()
        for obj in roots:
            print(f"根对象：{obj.name}")
```

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

- [SceneManager 场景管理器](SceneManager.md)
- [GameObject](GameObject.md)
- [Transform](Transform.md)

<!-- USER CONTENT END -->
