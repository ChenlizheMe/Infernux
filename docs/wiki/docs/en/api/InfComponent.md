# InfComponent

<div class="class-info">
class in <b>InfEngine.components</b>
</div>

## Description

Base class for Python-scripted game components (Unity-style lifecycle).

Subclass this to create game logic scripts.  Use ``serialized_field()``
class variables for Inspector-editable properties.

Example::

    class PlayerController(InfComponent):
        speed: float = serialized_field(default=5.0, range=(0, 100))

        def start(self):
            Debug.log("PlayerController started")

        def update(self, delta_time: float):
            pos = self.transform.position
            self.transform.position = Vector3(
                pos.x + self.speed * delta_time, pos.y, pos.z
            )

<!-- USER CONTENT START --> description

InfComponent is the base class for all user-written gameplay scripts in InfEngine, equivalent to Unity's MonoBehaviour. Derive from InfComponent to create custom behaviors that attach to [GameObjects](GameObject.md).

The engine calls lifecycle methods on your script automatically: `awake()` runs once when the component is created, `start()` runs before the first frame, `update()` runs every frame with the delta time, `late_update()` runs after all updates, and `on_destroy()` runs when the component is removed. Override only the methods you need.

Expose fields to the editor Inspector by annotating them with `serialized_field()`. This lets designers tune values like speed, health, and references without modifying code. Use `get_component()` and `get_py_component()` to communicate between components on the same GameObject.

<!-- USER CONTENT END -->

## Constructors

| Signature | Description |
|------|------|
| `InfComponent.__init__() → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| game_object | `Optional[GameObject]` | The GameObject this component is attached to. *(read-only)* |
| transform | `Optional[Transform]` | Shortcut to ``self.game_object.transform``. *(read-only)* |
| is_valid | `bool` | Whether the underlying GameObject reference is still alive. *(read-only)* |
| enabled | `bool` | Whether the component is enabled. |
| type_name | `str` | Class name of this component. *(read-only)* |
| execution_order | `int` | Execution order (lower value runs earlier). |
| component_id | `int` | Unique auto-incremented ID for this component instance. *(read-only)* |
| tag | `str` | Tag of the attached GameObject. |
| game_object_layer | `int` | Layer index (0–31) of the attached GameObject. |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `destroy() → None` | Remove this component from its owning GameObject (Unity-style). |
| `on_collision_enter(collision: Any) → None` | Called when this collider starts touching another collider. |
| `on_collision_stay(collision: Any) → None` | Called every fixed-update while two colliders remain in contact. |
| `on_collision_exit(collision: Any) → None` | Called when two colliders stop touching. |
| `on_trigger_enter(other: Any) → None` | Called when another collider enters this trigger volume. |
| `on_trigger_stay(other: Any) → None` | Called every fixed-update while another collider is inside this trigger. |
| `on_trigger_exit(other: Any) → None` | Called when another collider exits this trigger volume. |
| `start_coroutine(generator: Any) → Coroutine` | Start a coroutine on this component. |
| `stop_coroutine(coroutine: Coroutine) → None` | Stop a specific coroutine previously started with ``start_coroutine()``. |
| `stop_all_coroutines() → None` | Stop all coroutines running on this component. |
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

## Lifecycle Methods

| Method | Description |
|------|------|
| `awake() → None` | Called once when the component is first created. |
| `start() → None` | Called before the first Update after the component is enabled. |
| `update(delta_time: float) → None` | Called every frame. |
| `fixed_update(fixed_delta_time: float) → None` | Called at a fixed time step (default 50 Hz). |
| `late_update(delta_time: float) → None` | Called every frame after all Update calls. |
| `on_destroy() → None` | Called when the component or its GameObject is destroyed. |
| `on_enable() → None` | Called when the component is enabled. |
| `on_disable() → None` | Called when the component is disabled. |
| `on_validate() → None` | Called when a serialized field is changed in the Inspector (editor only). |
| `reset() → None` | Called when the component is reset to defaults (editor only). |
| `on_after_deserialize() → None` | Called after deserialization (scene load / undo). |
| `on_before_serialize() → None` | Called before serialization (scene save). |
| `on_draw_gizmos() → None` | Called every frame in the editor to draw gizmos for this component. |
| `on_draw_gizmos_selected() → None` | Called every frame in the editor ONLY when this object is selected. |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## Operators

| Method | Returns |
|------|------|
| `__repr__() → str` | `str` |

<!-- USER CONTENT START --> operators

<!-- USER CONTENT END -->

## Example

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
        print(f"{self.game_object.name} is ready")

    def update(self, delta_time: float):
        # Read input and move the character
        h = Input.get_axis("Horizontal")
        v = Input.get_axis("Vertical")
        move = vector3(h, 0, v) * self.move_speed * delta_time
        self.transform.translate(move)

    def late_update(self, delta_time: float):
        # Clamp position after all updates
        pos = self.transform.position
        pos.y = max(pos.y, 0.0)
        self.transform.position = pos

    def on_destroy(self):
        print(f"{self.game_object.name} destroyed")
```

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

- [GameObject](GameObject.md)
- [Component](Component.md)
- [serialized_field](serialized_field.md)
- [Input](Input.md)

<!-- USER CONTENT END -->
