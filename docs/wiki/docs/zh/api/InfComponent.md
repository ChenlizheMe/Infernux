# InfComponent

<div class="class-info">
类位于 <b>InfEngine.components</b>
</div>

## 描述

用户脚本组件的基类，类似于 Unity 的 MonoBehaviour。

<!-- USER CONTENT START --> description

InfComponent 是 InfEngine 中所有用户编写的游戏脚本的基类，相当于 Unity 的 MonoBehaviour。继承 InfComponent 即可创建可挂载到 [GameObject](GameObject.md) 上的自定义行为。

引擎会自动调用脚本的生命周期方法：`awake()` 在组件创建时执行一次，`start()` 在第一帧前执行，`update()` 每帧执行并传入时间增量，`late_update()` 在所有更新之后执行，`on_destroy()` 在组件移除时执行。只需重写所需的方法即可。

使用 `serialized_field()` 标注字段可将其暴露到编辑器的检查器面板中，让设计人员无需修改代码即可调整速度、生命值、引用等数值。使用 `get_component()` 和 `get_py_component()` 在同一 GameObject 上的组件之间进行通信。

<!-- USER CONTENT END -->

## 构造函数

| 签名 | 描述 |
|------|------|
| `InfComponent.__init__() → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| game_object | `Optional[GameObject]` | 此组件附加到的 GameObject。 *(只读)* |
| transform | `Optional[Transform]` | 附加到此 GameObject 的 Transform。 *(只读)* |
| is_valid | `bool` | Whether the underlying GameObject reference is still alive. *(只读)* |
| enabled | `bool` | 此组件是否已启用。 |
| type_name | `str` | Class name of this component. *(只读)* |
| execution_order | `int` | Execution order (lower value runs earlier). |
| component_id | `int` | Unique auto-incremented ID for this component instance. *(只读)* |
| tag | `str` | Tag of the attached GameObject. |
| game_object_layer | `int` | Layer index (0–31) of the attached GameObject. |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `destroy() → None` | 销毁此组件或指定的 GameObject。 |
| `on_collision_enter(collision: Any) → None` | Called when this collider starts touching another collider. |
| `on_collision_stay(collision: Any) → None` | Called every fixed-update while two colliders remain in contact. |
| `on_collision_exit(collision: Any) → None` | Called when two colliders stop touching. |
| `on_trigger_enter(other: Any) → None` | Called when another collider enters this trigger volume. |
| `on_trigger_stay(other: Any) → None` | Called every fixed-update while another collider is inside this trigger. |
| `on_trigger_exit(other: Any) → None` | Called when another collider exits this trigger volume. |
| `start_coroutine(generator: Any) → Coroutine` | 启动一个协程。 |
| `stop_coroutine(coroutine: Coroutine) → None` | 停止一个协程。 |
| `stop_all_coroutines() → None` | 停止所有协程。 |
| `get_component(component_type: Union[type, str]) → Optional[Any]` | Get another component of the specified type on the same GameObject. |
| `add_component(component_type: Union[type, str]) → Optional[Any]` | Add a component to this component's GameObject. |
| `get_components(component_type: Union[type, str]) → List[Any]` | Get all components of the specified type on the same GameObject. |
| `try_get_component(component_type: Union[type, str]) → Tuple[bool, Optional[Any]]` | Try to get a component; returns (found, component_or_None). |
| `get_mesh_renderer() → Optional[MeshRenderer]` | Shortcut to get the MeshRenderer on the same GameObject. |
| `compare_tag(tag: str) → bool` | Returns True if the attached GameObject's tag matches. |
| `get_component_in_children(component_type: Union[type, str], include_inactive: bool = ...) → Optional[Any]` | Get a component of the specified type on this or any child GameObject. |
| `get_component_in_parent(component_type: Union[type, str], include_inactive: bool = ...) → Optional[Any]` | Get a component of the specified type on this or any parent GameObject. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 生命周期方法

| 方法 | 描述 |
|------|------|
| `awake() → None` | 组件创建时调用一次。 |
| `start() → None` | 首次 Update 之前调用一次。 |
| `update(delta_time: float) → None` | 每帧调用一次。 |
| `fixed_update(fixed_delta_time: float) → None` | 以固定时间间隔调用。 |
| `late_update(delta_time: float) → None` | 在所有 Update 调用之后每帧调用。 |
| `on_destroy() → None` | 组件即将被销毁时调用。 |
| `on_enable() → None` | 组件启用时调用。 |
| `on_disable() → None` | 组件禁用时调用。 |
| `on_validate() → None` | 编辑器中属性变更时调用。 |
| `reset() → None` | Called when the component is reset to defaults (editor only). |
| `on_after_deserialize() → None` | Called after deserialization (scene load / undo). |
| `on_before_serialize() → None` | Called before serialization (scene save). |
| `on_draw_gizmos() → None` | 每帧绘制 Gizmos 时调用。 |
| `on_draw_gizmos_selected() → None` | 选中时绘制 Gizmos。 |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## 运算符

| 方法 | 返回值 |
|------|------|
| `__repr__() → str` | `str` |

<!-- USER CONTENT START --> operators

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent, serialized_field
from InfEngine.math import vector3

class PlayerController(InfComponent):
    move_speed: float = serialized_field(default=5.0)
    jump_height: float = serialized_field(default=2.0)

    def awake(self):
        self._velocity = vector3(0, 0, 0)

    def start(self):
        print(f"{self.game_object.name} 已就绪")

    def update(self, delta_time: float):
        # 读取输入并移动角色
        h = Input.get_axis("Horizontal")
        v = Input.get_axis("Vertical")
        move = vector3(h, 0, v) * self.move_speed * delta_time
        self.transform.translate(move)

    def late_update(self, delta_time: float):
        # 在所有更新完成后限制位置
        pos = self.transform.position
        pos.y = max(pos.y, 0.0)
        self.transform.position = pos

    def on_destroy(self):
        print(f"{self.game_object.name} 已销毁")
```

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

- [GameObject](GameObject.md)
- [Component 组件](Component.md)
- [serialized_field 序列化字段](serialized_field.md)
- [Input 输入](Input.md)

<!-- USER CONTENT END -->
