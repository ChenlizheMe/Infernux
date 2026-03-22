# PointerEventData

<div class="class-info">
class in <b>InfEngine.ui</b>
</div>

## Description

Data container for a single pointer event.

Attributes:
    position: Current pointer position in *canvas design* pixels.
    delta: Frame-to-frame delta in canvas design pixels.
    button: Which mouse button triggered this event.
    press_position: Canvas-space position where the button was pressed.
    click_count: Number of rapid clicks (1 = single, 2 = double, …).
    canvas: The ``UICanvas`` owning the target element.
    target: The ``InfUIScreenComponent`` this event is addressed to.
    used: Set to ``True`` in a handler to stop further propagation.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Constructors

| Signature | Description |
|------|------|
| `PointerEventData.__init__()` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `Use()` | Mark event as consumed (stops propagation to parent elements). |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for PointerEventData
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
