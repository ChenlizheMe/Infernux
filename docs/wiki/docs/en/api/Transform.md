# Transform

<div class="class-info">
class in <b>InfEngine</b>
</div>

**Inherits from:** [Component](Component.md)

## Description

Transform component — position, rotation, scale, hierarchy.

<!-- USER CONTENT START --> description

Transform determines the position, rotation, and scale of a [GameObject](GameObject.md) in the scene. Every GameObject has exactly one Transform, and it cannot be removed. Transforms form a hierarchy: when a Transform has a parent, its `local_position`, `local_rotation`, and `local_scale` are relative to the parent. The `position` and `rotation` properties give world-space values.

Direction helpers — `forward`, `right`, and `up` — return the object's current orientation axes in world space, making it straightforward to implement movement and aiming. Use `translate()` and `rotate()` for incremental motion, or set `position` and `rotation` directly for teleportation and snapping.

The parent-child relationship is established through `set_parent()` on the [GameObject](GameObject.md). When the parent moves, all children move with it. Accessing `local_position` and `local_rotation` lets you offset children relative to their parent without worrying about world coordinates.

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| position | `Vector3` |  *(read-only)* |
| euler_angles | `Vector3` |  *(read-only)* |
| rotation | `quatf` |  *(read-only)* |
| local_position | `Vector3` |  *(read-only)* |
| local_euler_angles | `Vector3` |  *(read-only)* |
| local_scale | `Vector3` |  *(read-only)* |
| local_rotation | `quatf` |  *(read-only)* |
| lossy_scale | `Vector3` |  *(read-only)* |
| forward | `Vector3` |  *(read-only)* |
| right | `Vector3` |  *(read-only)* |
| up | `Vector3` |  *(read-only)* |
| local_forward | `Vector3` |  *(read-only)* |
| local_right | `Vector3` |  *(read-only)* |
| local_up | `Vector3` |  *(read-only)* |
| parent | `Optional[Transform]` |  *(read-only)* |
| root | `Transform` |  *(read-only)* |
| child_count | `int` |  *(read-only)* |
| has_changed | `bool` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_parent(parent: Optional[Transform], world_position_stays: bool = True) → None` |  |
| `get_child(index: int) → Transform` |  |
| `find(name: str) → Optional[Transform]` |  |
| `detach_children() → None` |  |
| `is_child_of(parent: Transform) → bool` |  |
| `get_sibling_index() → int` |  |
| `set_sibling_index(index: int) → None` |  |
| `set_as_first_sibling() → None` |  |
| `set_as_last_sibling() → None` |  |
| `look_at(target: Vector3) → None` |  |
| `translate(delta: Vector3, space: int = ...) → None` |  |
| `translate_local(delta: Vector3) → None` |  |
| `rotate(euler: Vector3, space: int = ...) → None` |  |
| `rotate_around(point: Vector3, axis: Vector3, angle: float) → None` |  |
| `transform_point(point: Vector3) → Vector3` |  |
| `inverse_transform_point(point: Vector3) → Vector3` |  |
| `transform_direction(direction: Vector3) → Vector3` |  |
| `inverse_transform_direction(direction: Vector3) → Vector3` |  |
| `transform_vector(vector: Vector3) → Vector3` |  |
| `inverse_transform_vector(vector: Vector3) → Vector3` |  |
| `local_to_world_matrix() → List[float]` |  |
| `world_to_local_matrix() → List[float]` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent, serialized_field
from InfEngine.math import vector3

class Mover(InfComponent):
    speed: float = serialized_field(default=3.0)
    rotation_speed: float = serialized_field(default=90.0)

    def update(self, delta_time: float):
        # Move forward in the object's facing direction
        self.transform.translate(self.transform.forward * self.speed * delta_time)

        # Rotate around the Y axis
        self.transform.rotate(vector3(0, self.rotation_speed * delta_time, 0))

        # Read world-space position
        pos = self.transform.position
        if pos.y < -10:
            # Reset position if fallen off the map
            self.transform.position = vector3(0, 5, 0)

        # Access local-space values relative to parent
        local_pos = self.transform.local_position
        local_pos.y = 1.0  # maintain a fixed height offset from parent
        self.transform.local_position = local_pos
```

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

- [GameObject](GameObject.md)
- [vector3](vector3.md)
- [Component](Component.md)

<!-- USER CONTENT END -->
