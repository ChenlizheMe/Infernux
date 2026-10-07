# statistics

<div class="class-info">
function in <b>Infernux.jit</b>
</div>

```python
statistics(fn: Callable[..., Any]) → Statistics
```

## Description

Detached compilation/decision/mapped-memory snapshot, without executing fn.

<!-- USER CONTENT START --> description
Returns a detached snapshot for a function produced by `jit.compile`, including
the selected mode, preparation timings, specializations, cache decisions, and
owned native-code memory when the bundled backend can report it.
<!-- USER CONTENT END -->

## Parameters

| Name | Type | Description |
|------|------|------|
| fn | `Callable[..., Any]` |  |

## Example

<!-- USER CONTENT START --> example
```python
from infernux import jit

report = jit.statistics(integrate)
print(report.selected_mode, report.specializations)
```
<!-- USER CONTENT END -->
