# GameObject

<div class="class-info">
类位于 <b>InfEngine</b>
</div>

## 描述

场景层级中具有组件的游戏对象。

<!-- USER CONTENT START --> description

GameObject 是 InfEngine 场景的基础构建单元。游戏中的每一个实体——角色、灯光、摄像机、道具——都是一个 GameObject。单独的 GameObject 只是一个有名称的容器，它完全通过挂载的[组件](Component.md)来获得具体功能。

每个 GameObject 自动包含一个 [Transform](Transform.md) 组件，用于定义它在场景层级中的位置、旋转和缩放。通过 `add_component()` 添加更多组件来赋予对象视觉表现、物理特性、音频或自定义游戏逻辑。运行时使用 `get_component()` 和 `get_py_component()` 获取已挂载的组件。

GameObject 形成父子层级结构。通过 `set_parent()` 设置父对象后，子对象的 [Transform](Transform.md) 将相对于父对象进行计算，因此移动父对象会带动所有子对象一起移动。使用 `find_child()` 和 `find_descendant()` 根据名称查找对象，使用 `compare_tag()` 通过标签识别对象，实现碰撞过滤、目标选择等游戏逻辑。

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| name | `str` | 此 GameObject 的名称。 *(只读)* |
| active | `bool` | 此 GameObject 是否处于活动状态。 *(只读)* |
| tag | `str` | 此 GameObject 的标签字符串。 *(只读)* |
| layer | `int` | 此 GameObject 的层级索引 (0-31)。 *(只读)* |
| is_static | `bool` | 静态标志。 *(只读)* |
| prefab_guid | `str` |  *(只读)* |
| prefab_root | `bool` |  *(只读)* |
| active_self | `bool` | 此对象自身是否处于活动状态。active 的别名。 *(只读)* |
| active_in_hierarchy | `bool` | 此对象在层级中是否处于活动状态。 *(只读)* |
| id | `int` | 唯一对象标识符。 *(只读)* |
| is_prefab_instance | `bool` |  *(只读)* |
| transform | `Transform` | 获取 Transform 组件。 *(只读)* |
| scene | `Scene` | 此 GameObject 所属的场景。 *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `compare_tag(tag: str) → bool` | 此 GameObject 的标签是否与给定标签匹配。 |
| `get_transform() → Transform` | 获取 Transform 组件。 |
| `add_component(component_type: Any) → Optional[Component]` | 通过类型或类型名称添加 C++ 组件。 |
| `remove_component(component: Component) → bool` | 移除一个组件实例（无法移除 Transform）。 |
| `can_remove_component(component: Component) → bool` |  |
| `get_remove_component_blockers(component: Component) → List[str]` |  |
| `get_components() → List[Component]` | 获取所有组件（包括 Transform）。 |
| `get_component(type_name: str) → Optional[Component]` |  |
| `get_cpp_component(type_name: str) → Optional[Component]` | 根据类型名称获取 C++ 组件。 |
| `get_cpp_components(type_name: str) → List[Component]` | 获取指定类型名称的所有 C++ 组件。 |
| `add_py_component(component_instance: Any) → Any` | 向此 GameObject 添加 Python InfComponent 实例。 |
| `get_py_component(component_type: Any) → Any` | 获取指定类型的 Python 组件。 |
| `get_py_components() → List[Any]` | 获取附加到此 GameObject 的所有 Python 组件。 |
| `remove_py_component(component: Any) → bool` | 移除一个 Python 组件实例。 |
| `get_parent() → Optional[GameObject]` | 获取父级 GameObject。 |
| `set_parent(parent: Optional[GameObject], world_position_stays: bool = True) → None` | 设置父级 GameObject（None 表示根级）。 |
| `get_children() → List[GameObject]` | 获取子 GameObject 列表。 |
| `get_child_count() → int` | 获取子对象数量。 |
| `get_child(index: int) → GameObject` | 根据索引获取子对象。 |
| `find_child(name: str) → Optional[GameObject]` | 根据名称查找直接子对象（非递归）。 |
| `find_descendant(name: str) → Optional[GameObject]` | 根据名称查找后代对象（递归深度优先搜索）。 |
| `is_active_in_hierarchy() → bool` | 检查此对象及所有父对象是否处于活动状态。 |
| `get_component_in_children(component_type: Any, include_inactive: bool = False) → Any` |  |
| `get_component_in_parent(component_type: Any, include_inactive: bool = False) → Any` |  |
| `serialize() → str` | 将 GameObject 序列化为 JSON 字符串。 |
| `deserialize(json_str: str) → None` | 从 JSON 字符串反序列化 GameObject。 |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 静态方法

| 方法 | 描述 |
|------|------|
| `static GameObject.find(name: str) → Optional[GameObject]` |  |
| `static GameObject.find_with_tag(tag: str) → Optional[GameObject]` |  |
| `static GameObject.find_game_objects_with_tag(tag: str) → List[GameObject]` |  |
| `static GameObject.instantiate(original: GameObject, parent: Optional[GameObject] = None) → Optional[GameObject]` |  |
| `static GameObject.destroy(game_object: GameObject) → None` |  |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent, serialized_field
from InfEngine.math import vector3

class PlayerSetup(InfComponent):
    speed: float = serialized_field(default=5.0)

    def start(self):
        # 重命名游戏对象
        self.game_object.name = "Player"
        self.game_object.tag = "Player"

        # 添加网格渲染器以显示外观
        renderer = self.game_object.add_component("MeshRenderer")

        # 创建子对象作为武器
        scene = self.game_object.scene
        weapon = scene.create_game_object("Sword")
        weapon.set_parent(self.game_object)
        weapon.transform.local_position = vector3(0.5, 0.0, 1.0)

    def update(self, delta_time: float):
        # 每帧向前移动
        direction = self.transform.forward
        self.transform.translate(direction * self.speed * delta_time)

        # 遍历所有子对象
        for child in self.game_object.get_children():
            pass  # 处理每个子对象
```

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

- [Transform](Transform.md)
- [Component 组件](Component.md)
- [InfComponent 脚本组件](InfComponent.md)
- [Scene 场景](Scene.md)
- [SceneManager 场景管理器](SceneManager.md)

<!-- USER CONTENT END -->
