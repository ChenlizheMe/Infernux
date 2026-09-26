"""Renderer-external protocol for optional array-oriented parallel backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, Tuple, runtime_checkable


class ParallelTaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ParallelCapabilities:
    backend_name: str
    asynchronous: bool = False
    zero_copy: bool = False
    dlpack: bool = False

    def __post_init__(self) -> None:
        if not self.backend_name.strip():
            raise ValueError("parallel backend name must not be empty")


@dataclass(frozen=True)
class ParallelBufferView:
    """Opaque identity and NumPy-compatible array metadata."""

    handle: str
    buffer: Any
    schema: Tuple[Tuple[str, str], ...]
    shape: Tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.handle.strip():
            raise ValueError("parallel buffer handle must not be empty")
        if not self.shape or any(int(dimension) < 0 for dimension in self.shape):
            raise ValueError("parallel buffer shape must contain non-negative dimensions")
        if not self.dtype.strip():
            raise ValueError("parallel buffer dtype must not be empty")
        if tuple(getattr(self.buffer, "shape", ())) != tuple(self.shape):
            raise ValueError("parallel buffer shape does not match its array")
        if str(getattr(self.buffer, "dtype", "")) != self.dtype:
            raise ValueError("parallel buffer dtype does not match its array")
        flags = getattr(self.buffer, "flags", None)
        if flags is None or not bool(getattr(flags, "c_contiguous", False)):
            raise ValueError("parallel buffers must be C-contiguous")


@runtime_checkable
class ParallelBackend(Protocol):
    """Contract implementable by Taichi, Numba, or another array backend."""

    def capabilities(self) -> ParallelCapabilities: ...

    def create_buffer_view(
        self,
        schema: Sequence[Tuple[str, str]],
        shape: Sequence[int],
        dtype: str,
    ) -> ParallelBufferView: ...

    def submit(
        self,
        kernel: Any,
        inputs: Sequence[ParallelBufferView],
        outputs: Sequence[ParallelBufferView],
    ) -> str: ...

    def poll(self, task_handle: str) -> ParallelTaskState: ...

    def wait(self, task_handle: str, timeout: float | None = None) -> ParallelTaskState: ...

    def cancel(self, task_handle: str) -> bool: ...

    def commit(self, output: ParallelBufferView, destination_handle: str) -> None: ...


__all__ = [
    "ParallelBackend",
    "ParallelBufferView",
    "ParallelCapabilities",
    "ParallelTaskState",
]
