from __future__ import annotations

from typing import Any, Callable
from Infernux.jit_runtime import (
    CpuCompilationStatistics as Statistics,
    CpuOptimizationReport as OptimizationReport,
    CpuPassTiming as PassTiming,
    CpuSpecializationStatistics as SpecializationStatistics,
)

JIT_AVAILABLE: bool

def compile(fn: Callable[..., Any] = ..., **options: Any) -> Any:
    """Compile a CPU function; automatic serial/parallel selection is enabled by default."""
    ...

def warmup(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Prepare a compiled CPU function on isolated inputs; errors propagate."""
    ...

def statistics(fn: Callable[..., Any]) -> Statistics:
    """Detached compilation/decision/mapped-memory snapshot, without executing fn."""
    ...

__all__: list[str]
