# Physics

<div class="class-info">
class in <b>InfEngine.physics</b>
</div>

## Description

Global physics system for raycasting and spatial queries.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| gravity | `Any` | The global gravity vector applied to all rigidbodies. |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Static Methods

| Method | Description |
|------|------|
| `static Physics.get_gravity() → Any` | Get the global gravity vector. |
| `static Physics.set_gravity(value: Any) → None` | Set the global gravity vector. |
| `static Physics.raycast(origin: Any, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` | Cast a ray and return the first hit, or None. |
| `static Physics.raycast_all(origin: Any, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` | Cast a ray and return all hits. |
| `static Physics.overlap_sphere(center: Any, radius: float, layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` | Find all colliders within a sphere. |
| `static Physics.overlap_box(center: Any, half_extents: Any, layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` | Find all colliders within an axis-aligned box. |
| `static Physics.sphere_cast(origin: Any, radius: float, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` | Cast a sphere along a direction and return the first hit, or None. |
| `static Physics.box_cast(center: Any, half_extents: Any, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` | Cast a box along a direction and return the first hit, or None. |
| `static Physics.ignore_layer_collision(layer1: int, layer2: int, ignore: bool = ...) → None` | Set whether collisions between two layers are ignored. |
| `static Physics.get_ignore_layer_collision(layer1: int, layer2: int) → bool` | Check if collisions between two layers are ignored. |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for Physics
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
