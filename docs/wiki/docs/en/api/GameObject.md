# GameObject

<div class="class-info">
class in <b>InfEngine</b>
</div>

## Description

Game object in the scene hierarchy.

<!-- USER CONTENT START --> description

GameObject is the fundamental building block of InfEngine scenes. Every entity in the game — characters, lights, cameras, props — is a GameObject. On its own, a GameObject is just a named container; it gains behavior entirely through the [Components](Component.md) attached to it.

Every GameObject automatically includes a [Transform](Transform.md) component that defines its position, rotation, and scale within the scene hierarchy. Attach additional components with `add_component()` to give the object visual appearance, physics, audio, or custom gameplay logic. Use `get_component()` and `get_py_component()` to retrieve components at runtime.

GameObjects form a parent-child hierarchy. Setting a parent with `set_parent()` makes the child's [Transform](Transform.md) relative to the parent, so moving the parent moves all its children. Use `find_child()` and `find_descendant()` to locate objects by name, and `compare_tag()` to identify objects by tag for gameplay logic such as collision filtering or target selection.

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| name | `str` |  *(read-only)* |
| active | `bool` |  *(read-only)* |
| tag | `str` |  *(read-only)* |
| layer | `int` |  *(read-only)* |
| is_static | `bool` |  *(read-only)* |
| prefab_guid | `str` |  *(read-only)* |
| prefab_root | `bool` |  *(read-only)* |
| active_self | `bool` |  *(read-only)* |
| active_in_hierarchy | `bool` |  *(read-only)* |
| id | `int` |  *(read-only)* |
| is_prefab_instance | `bool` |  *(read-only)* |
| transform | `Transform` |  *(read-only)* |
| scene | `Scene` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `compare_tag(tag: str) → bool` |  |
| `get_transform() → Transform` |  |
| `add_component(component_type: Any) → Optional[Component]` |  |
| `remove_component(component: Component) → bool` |  |
| `can_remove_component(component: Component) → bool` |  |
| `get_remove_component_blockers(component: Component) → List[str]` |  |
| `get_components() → List[Component]` |  |
| `get_component(type_name: str) → Optional[Component]` |  |
| `get_cpp_component(type_name: str) → Optional[Component]` |  |
| `get_cpp_components(type_name: str) → List[Component]` |  |
| `add_py_component(component_instance: Any) → Any` |  |
| `get_py_component(component_type: Any) → Any` |  |
| `get_py_components() → List[Any]` |  |
| `remove_py_component(component: Any) → bool` |  |
| `get_parent() → Optional[GameObject]` |  |
| `set_parent(parent: Optional[GameObject], world_position_stays: bool = True) → None` |  |
| `get_children() → List[GameObject]` |  |
| `get_child_count() → int` |  |
| `get_child(index: int) → GameObject` |  |
| `find_child(name: str) → Optional[GameObject]` |  |
| `find_descendant(name: str) → Optional[GameObject]` |  |
| `is_active_in_hierarchy() → bool` |  |
| `get_component_in_children(component_type: Any, include_inactive: bool = False) → Any` |  |
| `get_component_in_parent(component_type: Any, include_inactive: bool = False) → Any` |  |
| `serialize() → str` |  |
| `deserialize(json_str: str) → None` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Static Methods

| Method | Description |
|------|------|
| `static GameObject.find(name: str) → Optional[GameObject]` |  |
| `static GameObject.find_with_tag(tag: str) → Optional[GameObject]` |  |
| `static GameObject.find_game_objects_with_tag(tag: str) → List[GameObject]` |  |
| `static GameObject.instantiate(original: GameObject, parent: Optional[GameObject] = None) → Optional[GameObject]` |  |
| `static GameObject.destroy(game_object: GameObject) → None` |  |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent, serialized_field
from InfEngine.math import vector3

class PlayerSetup(InfComponent):
    speed: float = serialized_field(default=5.0)

    def start(self):
        # Rename the game object
        self.game_object.name = "Player"
        self.game_object.tag = "Player"

        # Add a mesh renderer for visual appearance
        renderer = self.game_object.add_component("MeshRenderer")

        # Create a child object for the weapon
        scene = self.game_object.scene
        weapon = scene.create_game_object("Sword")
        weapon.set_parent(self.game_object)
        weapon.transform.local_position = vector3(0.5, 0.0, 1.0)

    def update(self, delta_time: float):
        # Move the object forward each frame
        direction = self.transform.forward
        self.transform.translate(direction * self.speed * delta_time)

        # List all children
        for child in self.game_object.get_children():
            pass  # process each child
```

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

- [Transform](Transform.md)
- [Component](Component.md)
- [InfComponent](InfComponent.md)
- [Scene](Scene.md)
- [SceneManager](SceneManager.md)

<!-- USER CONTENT END -->
