from typing import Optional

from infernux.lib import (
    DocumentFileState,
    DocumentWriteCancelled,
    DocumentWriteOptions,
    DocumentWriteSuperseded,
    DocumentWriteTicket,
    FileWriteMode,
    NativeDocumentStore,
    PathWriteBusy,
    PathWriteScope,
)


class DocumentStore:
    @classmethod
    def instance(cls) -> NativeDocumentStore: ...
    @classmethod
    def shutdown(cls) -> None: ...
    @classmethod
    def flush(cls, path: Optional[str] = ...) -> None: ...
    @classmethod
    def is_idle(cls) -> bool: ...


def capture_document_file_state(path: str) -> DocumentFileState: ...
def read_document_text_snapshot(path: str) -> tuple[str | None, DocumentFileState]: ...
def write_document_text(path: str, content: str | bytes | bytearray, *, mode: FileWriteMode = ..., expected_file_state: DocumentFileState | None = ..., commit_chain_token: str = ..., chain_id: str = ..., write_scope: PathWriteScope | None = ...) -> int: ...
def submit_document_text(path: str, content: str, *, mode: FileWriteMode = ..., expected_file_state: DocumentFileState | None = ..., commit_chain_token: str = ..., chain_id: str = ..., write_scope: PathWriteScope | None = ...) -> DocumentWriteTicket: ...


__all__: list[str]
