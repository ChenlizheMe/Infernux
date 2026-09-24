# compile

<div class="class-info">
function in <b>Infernux.jit</b>
</div>

```python
compile(fn: Callable[..., Any] = ...) → Any
```

## Description

Compile a CPU function; automatic serial/parallel selection is enabled by default.

<!-- USER CONTENT START --> description
Use this decorator for CPU numeric work. Infernux analyzes the function and
selects a serial or proven-parallel native implementation. Desktop targets
without the bundled CPU compiler reject the declaration. Web explicitly
cooks the decorated function into ordinary Python bytecode and removes the
decorator before the Player runs; that target contract is not an error-time
backend fallback. Calling an uncooked JIT decorator without the bundled
compiler remains an error.
<!-- USER CONTENT END -->

## Parameters

| Name | Type | Description |
|------|------|------|
| fn | `Callable[..., Any]` |  (default: `...`) |

## Example

<!-- USER CONTENT START --> example
```python
from infernux import jit

@jit.compile
def integrate(positions, velocities, dt):
    for i in range(len(positions)):
        positions[i] += velocities[i] * dt
```
<!-- USER CONTENT END -->
