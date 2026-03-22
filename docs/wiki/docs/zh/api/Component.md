# Component

<div class="class-info">
类位于 <b>InfEngine</b>
</div>

## 描述

附加到 GameObject 的所有组件的基类。

<!-- USER CONTENT START --> description

Component 是 InfEngine 中所有内置 C++ 组件的基类，包括 [Transform](Transform.md)、[Camera](Camera.md)、[Light](Light.md) 和 [MeshRenderer](MeshRenderer.md) 等。组件挂载在 [GameObject](GameObject.md) 上，提供特定的功能。

不应直接实例化 Component。应使用 `GameObject.add_component()` 通过类型名称添加内置组件。每个 Component 都持有对所属 `game_object` 和 `transform` 的引用，方便访问对象层级。

如需编写自定义的 Python 游戏逻辑，应继承 [InfComponent](InfComponent.md) 而非 Component。InfComponent 扩展了 Component，提供 `start()`、`update()` 等生命周期回调。

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| type_name | `str` |  *(只读)* |
| component_id | `int` |  *(只读)* |
| enabled | `bool` | 此组件是否已启用。 *(只读)* |
| execution_order | `int` |  |
| game_object | `GameObject` | 此组件附加到的 GameObject。 *(只读)* |
| required_component_types | `List[str]` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `is_component_type(type_name: str) → bool` |  |
| `serialize() → str` |  |
| `deserialize(json_str: str) → None` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent

class ComponentQuery(InfComponent):
    def start(self):
        # 每个组件都可以访问其所属的 GameObject
        owner = self.game_object
        print(f"挂载于：{owner.name}")

        # 通过组件快捷方式访问 Transform
        pos = self.transform.position
        print(f"位置：{pos}")

        # 获取同级的 C++ 组件
        renderer = owner.get_cpp_component("MeshRenderer")
        if renderer:
            print("已找到 MeshRenderer")

        # 列出此对象上的所有组件
        for comp in owner.get_components():
            print(f"组件：{type(comp).__name__}")
```

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

- [GameObject](GameObject.md)
- [Transform](Transform.md)
- [InfComponent 脚本组件](InfComponent.md)

<!-- USER CONTENT END -->
