# Rigidbody

<div class="class-info">
class in <b>InfEngine.components.builtin</b>
</div>

**Inherits from:** [BuiltinComponent](Component.md)

## Description

Controls physics simulation for the GameObject.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| mass | `float` | The mass of the rigidbody in kilograms. |
| drag | `float` | The linear drag coefficient. |
| angular_drag | `float` | The angular drag coefficient. |
| use_gravity | `bool` | Whether gravity affects this rigidbody. |
| is_kinematic | `bool` | Whether the rigidbody is kinematic (not driven by physics). |
| constraints | `int` | The raw constraint flags as an integer bitmask. |
| collision_detection_mode | `CollisionDetectionMode` | The collision detection mode used by this rigidbody. |
| interpolation | `RigidbodyInterpolation` | The interpolation mode for smoothing rigidbody movement. |
| max_angular_velocity | `float` | The maximum angular velocity in radians per second. |
| max_linear_velocity | `float` | The maximum linear velocity of the rigidbody. |
| freeze_rotation | `bool` | Shortcut to freeze or unfreeze all rotation axes. |
| constraints_flags | `RigidbodyConstraints` | The constraint flags as a RigidbodyConstraints enum. |
| velocity | `Any` | The linear velocity of the rigidbody in world space. |
| angular_velocity | `Any` | The angular velocity of the rigidbody in radians per second. |
| world_center_of_mass | `Any` | The center of mass in world space. *(read-only)* |
| position | `Any` | The position of the rigidbody in world space. *(read-only)* |
| rotation | `Tuple[float, float, float, float]` | The rotation of the rigidbody as a quaternion (x, y, z, w). *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `has_constraint(constraint: RigidbodyConstraints) → bool` | Return whether the specified constraint flag is set. |
| `add_constraint(constraint: RigidbodyConstraints) → None` | Add a constraint flag to the rigidbody. |
| `remove_constraint(constraint: RigidbodyConstraints) → None` | Remove a constraint flag from the rigidbody. |
| `add_force(force: Any, mode: Any = ...) → None` | Apply a force to the rigidbody. |
| `add_torque(torque: Any, mode: Any = ...) → None` | Apply a torque to the rigidbody. |
| `add_force_at_position(force: Any, position: Any, mode: Any = ...) → None` | Apply a force at a specific world-space position. |
| `move_position(position: Any) → None` | Move the kinematic rigidbody to the specified position. |
| `move_rotation(rotation: Any) → None` | Rotate the kinematic rigidbody to the specified rotation. |
| `is_sleeping() → bool` | Return whether the rigidbody is currently sleeping. |
| `wake_up() → None` | Force the rigidbody to wake up. |
| `sleep() → None` | Force the rigidbody to sleep. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for Rigidbody
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
