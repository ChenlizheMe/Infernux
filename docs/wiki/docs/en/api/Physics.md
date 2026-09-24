# Physics

<div class="class-info">
class in <b>Infernux.physics</b>
</div>

## Description

Global physics system for raycasting and spatial queries.

<!-- USER CONTENT START --> description
**Status:** Preview · **Verified with:** 0.4.0

Spatial queries accept layer masks and explicit trigger handling. Keep query volumes and distances bounded, and use the simplest query that answers the gameplay question.
<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| body_count | `int` | Number of native physics bodies currently owned by the world. *(read-only)* |
| query_generation | `int` | Monotonic token for the currently published physics query world. *(read-only)* |
| gravity | `Any` | The global gravity vector applied to all rigidbodies. |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Static Methods

| Method | Description |
|------|------|
| `static Physics.compute_penetration(collider_a: Any, position_a: Any, rotation_a: Any, collider_b: Any, position_b: Any, rotation_b: Any) → Optional[PenetrationResult]` |  |
| `static Physics.get_rigidbody_states(rigidbodies: List[Any], out: Optional[dict[str, Any]] = ...) → dict[str, Any]` |  |
| `static Physics.get_rigidbody_states_and_box_states(rigidbodies: List[Any], state_out: dict[str, Any], box_out: dict[str, Any], query_triggers: bool = ...) → tuple[dict[str, Any], dict[str, Any]]` |  |
| `static Physics.get_rigidbody_box_states(rigidbodies: List[Any], out: Optional[dict[str, Any]] = ..., query_triggers: bool = ...) → dict[str, Any]` |  |
| `static Physics.query_rigidbody_box_states_in_bounds(minimum: Any, maximum: Any, out: dict[str, Any], layer_mask: int = ..., query_triggers: bool = ...) → dict[str, Any]` |  |
| `static Physics.query_rigidbody_states_and_box_states_in_bounds(minimum: Any, maximum: Any, state_out: dict[str, Any], box_out: dict[str, Any], layer_mask: int = ..., query_triggers: bool = ...) → tuple[dict[str, Any], dict[str, Any]]` |  |
| `static Physics.apply_rigidbody_impulses(rigidbodies: List[Any], linear_impulses: Any, angular_impulses: Any) → None` |  |
| `static Physics.set_contact_event_stream_enabled(enabled: bool = ..., include_triggers: bool = ...) → None` |  |
| `static Physics.get_contact_events() → dict[str, Any]` |  |
| `static Physics.set_contact_impulse_stream_enabled(enabled: bool = ...) → None` |  |
| `static Physics.get_contact_impulses() → dict[str, Any]` |  |
| `static Physics.query_rigidbodies_in_bounds(minimum: Any, maximum: Any, layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` |  |
| `static Physics.get_gravity() → Any` | Get the global gravity vector. |
| `static Physics.set_gravity(value: Any) → None` | Set the global gravity vector. |
| `static Physics.raycast(origin: Any, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` | Cast a ray and return the first hit, or None. |
| `static Physics.raycast_screen(camera: Any, screen_position: Tuple[float, float], viewport_size: Tuple[float, float], max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` | Cast from a top-left-origin camera viewport pixel position. |
| `static Physics.raycast_batch(origins: npt.NDArray[np.float32], directions: npt.NDArray[np.float32], out: RaycastBatchOutput, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ..., profile: bool = ...) → RaycastBatchOutput` |  |
| `static Physics.raycast_all(origin: Any, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` | Cast a ray and return all hits. |
| `static Physics.overlap_sphere(center: Any, radius: float, layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` | Find all colliders within a sphere. |
| `static Physics.overlap_box(center: Any, half_extents: Any, orientation: Any = ..., layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` | Find all colliders within an oriented box. |
| `static Physics.overlap_capsule(point0: Any, point1: Any, radius: float, layer_mask: int = ..., query_triggers: bool = ...) → List[Any]` |  |
| `static Physics.sphere_cast(origin: Any, radius: float, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` | Cast a sphere along a direction and return the first hit, or None. |
| `static Physics.box_cast(center: Any, half_extents: Any, direction: Any, orientation: Any = ..., max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` | Cast a box along a direction and return the first hit, or None. |
| `static Physics.capsule_cast(point0: Any, point1: Any, radius: float, direction: Any, max_distance: float = ..., layer_mask: int = ..., query_triggers: bool = ...) → Optional[Any]` |  |
| `static Physics.ignore_layer_collision(layer1: int, layer2: int, ignore: bool = ...) → None` | Set whether collisions between two layers are ignored. |
| `static Physics.get_ignore_layer_collision(layer1: int, layer2: int) → bool` | Check if collisions between two layers are ignored. |
| `static Physics.ignore_collision(collider1: Any, collider2: Any, ignore: bool = ...) → None` |  |
| `static Physics.get_ignore_collision(collider1: Any, collider2: Any) → bool` |  |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example
```python
import infernux as inx


class GroundProbe(inx.InxComponent):
    def is_grounded(self) -> bool:
        hit = inx.physics.Physics.raycast(
            self.transform.position,
            inx.Vector3(0.0, -1.0, 0.0),
            max_distance=1.1,
        )
        return hit is not None
```
<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also
- [Rigidbody](Rigidbody.md)
- [Collider](Collider.md)
- [BoxCollider](BoxCollider.md)
<!-- USER CONTENT END -->
