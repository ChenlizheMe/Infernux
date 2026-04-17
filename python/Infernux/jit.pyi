from __future__ import annotations

from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

JIT_AVAILABLE: bool

def ensure_jit_runtime(*, auto_install: bool = True) -> bool:
    """Ensure the current runtime can import Numba-backed JIT helpers."""
    ...

def njit(*args: Any, **kwargs: Any) -> Any:
    """Numba ``njit`` decorator, or a no-op fallback when Numba is unavailable.

    The decorated function gains a ``.py`` attribute pointing to the
    original pure-Python source.

    Supports ``auto_parallel=True`` as an Infernux extension. In that mode
    the wrapper prepares both serial and ``parallel=True`` variants,
    conservatively upgrades simple ``for ... in range(...)`` reduction loops
    to ``prange`` when possible, defaults to the parallel one, and lets
    :func:`warmup` first trigger compilation and then benchmark steady-state
    runtime before pinning the faster choice.
    """
    ...

def warmup(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Pre-compile a ``@njit`` function by calling it with representative args."""
    ...

def precompile_jit() -> None:
    """No-op kept for backward compatibility."""
    ...

prange: Any

__all__: list[str]