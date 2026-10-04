from __future__ import annotations
from typing import Any, Protocol, Sequence, Tuple

class ParallelTaskState(str):
    PENDING: ParallelTaskState
    RUNNING: ParallelTaskState
    COMPLETED: ParallelTaskState
    FAILED: ParallelTaskState
    CANCELLED: ParallelTaskState

class ParallelCapabilities:
    backend_name: str
    asynchronous: bool
    zero_copy: bool
    dlpack: bool
    def __init__(
        self,
        backend_name: str,
        asynchronous: bool = ...,
        zero_copy: bool = ...,
        dlpack: bool = ...,
    ) -> None: ...

class ParallelBufferView:
    handle: str
    buffer: Any
    schema: Tuple[Tuple[str, str], ...]
    shape: Tuple[int, ...]
    dtype: str
    def __init__(
        self,
        handle: str,
        buffer: Any,
        schema: Tuple[Tuple[str, str], ...],
        shape: Tuple[int, ...],
        dtype: str,
    ) -> None: ...

class ParallelBackend(Protocol):
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
    def wait(self, task_handle: str, timeout: float | None = ...) -> ParallelTaskState: ...
    def cancel(self, task_handle: str) -> bool: ...
    def commit(self, output: ParallelBufferView, destination_handle: str) -> None: ...
