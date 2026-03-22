# Transform

<div class="class-info">
类位于 <b>InfEngine</b>
</div>

**继承自:** [Component](Component.md)

## 描述

对象在场景中的位置、旋转和缩放。

<!-- USER CONTENT START --> description

Transform 决定 [GameObject](GameObject.md) 在场景中的位置、旋转和缩放。每个 GameObject 必定拥有且仅有一个 Transform，且不可移除。Transform 构成层级关系：当某个 Transform 存在父级时，其 `local_position`、`local_rotation` 和 `local_scale` 相对于父级计算。使用 `position` 和 `rotation` 属性可获取世界空间下的值。

方向辅助属性——`forward`、`right` 和 `up`——返回对象在世界空间中的当前朝向轴，便于实现移动和瞄准逻辑。使用 `translate()` 和 `rotate()` 进行增量运动，或直接设置 `position` 和 `rotation` 来实现瞬移和定位。

父子关系通过 [GameObject](GameObject.md) 的 `set_parent()` 建立。当父级移动时，所有子级随之移动。访问 `local_position` 和 `local_rotation` 可以相对于父级进行偏移，而无需关心世界坐标。

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| position | `Vector3` | 世界空间中的位置。 *(只读)* |
| euler_angles | `Vector3` | 世界空间中以欧拉角表示的旋转（度）。 *(只读)* |
| rotation | `quatf` | 世界空间中的旋转（四元数）。 *(只读)* |
| local_position | `Vector3` | 相对于父变换的位置。 *(只读)* |
| local_euler_angles | `Vector3` | 相对于父变换的欧拉角旋转（度）。 *(只读)* |
| local_scale | `Vector3` | 相对于父变换的缩放。 *(只读)* |
| local_rotation | `quatf` | 相对于父变换的旋转。 *(只读)* |
| lossy_scale | `Vector3` | 对象的全局缩放（只读）。 *(只读)* |
| forward | `Vector3` | 世界空间中的前方向向量（蓝轴）。 *(只读)* |
| right | `Vector3` | 世界空间中的右方向向量（红轴）。 *(只读)* |
| up | `Vector3` | 世界空间中的上方向向量（绿轴）。 *(只读)* |
| local_forward | `Vector3` |  *(只读)* |
| local_right | `Vector3` |  *(只读)* |
| local_up | `Vector3` |  *(只读)* |
| parent | `Optional[Transform]` | 父变换。 *(只读)* |
| root | `Transform` | 层级视图中最顶层的变换。 *(只读)* |
| child_count | `int` | 子变换数量。 *(只读)* |
| has_changed | `bool` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `set_parent(parent: Optional[Transform], world_position_stays: bool = True) → None` | 设置变换的父级。 |
| `get_child(index: int) → Transform` | 按索引获取子变换。 |
| `find(name: str) → Optional[Transform]` | 按名称查找子变换。 |
| `detach_children() → None` | 清除所有子项的父级。 |
| `is_child_of(parent: Transform) → bool` |  |
| `get_sibling_index() → int` | 获取同级索引。 |
| `set_sibling_index(index: int) → None` | 设置同级索引。 |
| `set_as_first_sibling() → None` | 将变换移动到兄弟列表的开头。 |
| `set_as_last_sibling() → None` | 将变换移动到兄弟列表的末尾。 |
| `look_at(target: Vector3) → None` | 旋转变换使前方向指向目标位置。 |
| `translate(delta: Vector3, space: int = ...) → None` | 按指定方向和距离移动变换。 |
| `translate_local(delta: Vector3) → None` |  |
| `rotate(euler: Vector3, space: int = ...) → None` | 按欧拉角旋转变换。 |
| `rotate_around(point: Vector3, axis: Vector3, angle: float) → None` | 围绕指定轴和点旋转。 |
| `transform_point(point: Vector3) → Vector3` | 将点从本地空间变换到世界空间。 |
| `inverse_transform_point(point: Vector3) → Vector3` | 将点从世界空间变换到本地空间。 |
| `transform_direction(direction: Vector3) → Vector3` | 将方向从本地空间变换到世界空间。 |
| `inverse_transform_direction(direction: Vector3) → Vector3` | 将方向从世界空间变换到本地空间。 |
| `transform_vector(vector: Vector3) → Vector3` |  |
| `inverse_transform_vector(vector: Vector3) → Vector3` |  |
| `local_to_world_matrix() → List[float]` |  |
| `world_to_local_matrix() → List[float]` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent, serialized_field
from InfEngine.math import vector3

class Mover(InfComponent):
    speed: float = serialized_field(default=3.0)
    rotation_speed: float = serialized_field(default=90.0)

    def update(self, delta_time: float):
        # 沿对象朝向方向前进
        self.transform.translate(self.transform.forward * self.speed * delta_time)

        # 绕 Y 轴旋转
        self.transform.rotate(vector3(0, self.rotation_speed * delta_time, 0))

        # 读取世界空间位置
        pos = self.transform.position
        if pos.y < -10:
            # 如果掉出地图则重置位置
            self.transform.position = vector3(0, 5, 0)

        # 访问相对于父级的局部空间值
        local_pos = self.transform.local_position
        local_pos.y = 1.0  # 保持与父级的固定高度偏移
        self.transform.local_position = local_pos
```

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

- [GameObject](GameObject.md)
- [vector3 三维向量](vector3.md)
- [Component 组件](Component.md)

<!-- USER CONTENT END -->
