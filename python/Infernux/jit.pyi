from __future__ import annotations

from typing import Any, Callable

JIT_AVAILABLE: bool

def compile(fn: Callable[..., Any] = ..., **options: Any) -> Any:
    """Compile a CPU function; automatic serial/parallel selection is enabled by default."""
    ...

def warmup(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Prepare a compiled CPU function on isolated inputs; errors propagate."""
    ...

__all__: list[str]
