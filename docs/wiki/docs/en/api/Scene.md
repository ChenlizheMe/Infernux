# Scene

<div class="class-info">
class in <b>InfEngine</b>
</div>

## Description

A single scene containing GameObjects.

<!-- USER CONTENT START --> description

Scene represents a single runtime scene in the engine, containing a hierarchy of [GameObjects](GameObject.md). Every object exists within a Scene, and a Scene provides methods to create, find, and enumerate those objects.

Use `create_game_object()` to instantiate new objects in the scene, `find()` to locate an object by name, and `find_with_tag()` to search by tag. The `get_root_game_objects()` method returns only the top-level objects (those without a parent), which is useful for iterating the full hierarchy.

Scenes are loaded and unloaded through the [SceneManager](SceneManager.md). The currently active Scene determines where newly created objects are placed by default.

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| name | `str` |  *(read-only)* |
| structure_version | `int` |  *(read-only)* |
| main_camera | `Optional[Camera]` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_playing(playing: bool) → None` |  |
| `create_game_object(name: str = 'GameObject') → GameObject` |  |
| `create_primitive(type: PrimitiveType, name: str = '') → GameObject` |  |
| `create_from_model(guid: str, name: str = '') → Optional[GameObject]` |  |
| `get_root_objects() → List[GameObject]` |  |
| `get_all_objects() → List[GameObject]` |  |
| `find(name: str) → Optional[GameObject]` |  |
| `find_by_id(id: int) → Optional[GameObject]` |  |
| `find_object_by_id(id: int) → Optional[GameObject]` |  |
| `find_with_tag(tag: str) → Optional[GameObject]` |  |
| `find_game_objects_with_tag(tag: str) → List[GameObject]` |  |
| `find_game_objects_in_layer(layer: int) → List[GameObject]` |  |
| `destroy_game_object(game_object: GameObject) → None` |  |
| `instantiate_game_object(source: GameObject, parent: Optional[GameObject] = None) → Optional[GameObject]` |  |
| `instantiate_from_json(json_str: str, parent: Optional[GameObject] = None) → Optional[GameObject]` |  |
| `process_pending_destroys() → None` |  |
| `is_playing() → bool` |  |
| `awake_object(game_object: GameObject) → None` |  |
| `serialize() → str` |  |
| `deserialize(json_str: str) → None` |  |
| `save_to_file(path: str) → None` |  |
| `load_from_file(path: str) → None` |  |
| `has_pending_py_components() → bool` |  |
| `take_pending_py_components() → List[PendingPyComponent]` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Lifecycle Methods

| Method | Description |
|------|------|
| `start() → None` |  |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example

```python
from InfEngine import InfComponent
from InfEngine.math import vector3

class SceneSetup(InfComponent):
    def start(self):
        scene = self.game_object.scene

        # Create new objects in the scene
        ground = scene.create_game_object("Ground")
        ground.transform.position = vector3(0, -0.5, 0)

        # Find an existing object by name
        player = scene.find("Player")
        if player:
            print(f"Found: {player.name}")

        # Find objects by tag
        enemy = scene.find_with_tag("Enemy")

        # Iterate all root-level objects
        roots = scene.get_root_game_objects()
        for obj in roots:
            print(f"Root: {obj.name}")
```

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

- [SceneManager](SceneManager.md)
- [GameObject](GameObject.md)
- [Transform](Transform.md)

<!-- USER CONTENT END -->
