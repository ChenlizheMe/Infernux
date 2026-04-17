# njit

<div class="class-info">
函数位于 <b>Infernux.jit</b>
</div>

```python
njit() → Any
```

## 描述

Numba ``njit`` decorator, or a no-op fallback when Numba is unavailable.

The decorated function gains a ``.py`` attribute pointing to the
original pure-Python source.

Supports ``auto_parallel=True`` as an Infernux extension. In that mode
the wrapper prepares both serial and ``parallel=True`` variants,
conservatively upgrades simple ``for ... in range(...)`` reduction loops
to ``prange`` when possible, defaults to the parallel one, and lets
:func:`warmup` first trigger compilation and then benchmark steady-state
runtime before pinning the faster choice.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example
```python
# TODO: Add example for njit
```
<!-- USER CONTENT END -->
