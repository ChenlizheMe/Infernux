# warmup

<div class="class-info">
function in <b>Infernux.jit</b>
</div>

```python
warmup(fn: Callable[..., Any]) → None
```

## Description

Prepare a compiled CPU function on isolated inputs; errors propagate.

<!-- USER CONTENT START --> description
Compile and validate one runtime signature using isolated argument copies.
Preparation failures propagate and never cause a second execution through a
different backend. A target without CPU JIT removes this call while cooking
ordinary Python bytecode; the runtime API itself never becomes a no-op.
<!-- USER CONTENT END -->

## Parameters

| Name | Type | Description |
|------|------|------|
| fn | `Callable[..., Any]` |  |

## Example

<!-- USER CONTENT START --> example
```python
jit.warmup(integrate, positions, velocities, 1.0 / 60.0)
```
<!-- USER CONTENT END -->
