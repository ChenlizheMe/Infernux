# AudioListener

<div class="class-info">
class in <b>Infernux.components.builtin</b>
</div>

**Inherits from:** [BuiltinComponent](Component.md)

## Description

Represents the listener/ears for 3D audio in the scene.

Attach one AudioListener to the main camera in most games. The engine keeps
one active listener; additional enabled listeners remain registered but can
be standby instead of immediately replacing the active listener.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| game_object_id | `int` | The ID of the GameObject this component is attached to. *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `serialize() → str` | Serialize the component to a JSON string. |
| `deserialize(json_str: str) → bool` | Deserialize the component from a JSON string. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example
```python
# TODO: Add example for AudioListener
```
<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
