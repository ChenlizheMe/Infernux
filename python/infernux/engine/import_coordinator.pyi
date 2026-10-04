from enum import Enum
from typing import Callable, Optional

class AssetFsEventKind(str, Enum):
    CREATED: AssetFsEventKind
    MODIFIED: AssetFsEventKind
    DELETED: AssetFsEventKind
    MOVED: AssetFsEventKind
    META_DELETED: AssetFsEventKind

class AssetFsEvent:
    kind: AssetFsEventKind
    path: str
    destination: str
    guid_hint: str
    observed_at: float
    attempt: int

class ImportCoordinator:
    def __init__(self, *, debounce_seconds: float = ..., delete_grace_seconds: float = ...,
                 meta_delete_grace_seconds: float = ..., retry_delay_seconds: float = ...,
                 max_attempts: int = ..., clock: Callable[[], float] = ...) -> None: ...
    @property
    def pending_count(self) -> int: ...
    def submit(self, kind: AssetFsEventKind, path: str, *, destination: str = ...,
               guid_hint: str = ..., observed_at: Optional[float] = ...,
               debounce_seconds: Optional[float] = ..., source_registered: bool = ...) -> None: ...
    def drain(self, *, force: bool = ..., now: Optional[float] = ...) -> list[AssetFsEvent]: ...
    def retry(self, event: AssetFsEvent, *, now: Optional[float] = ...) -> bool: ...
    def clear(self) -> None: ...

def is_document_store_temporary_path(path: str) -> bool: ...
