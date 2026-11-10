"""Thin Python access to the native cross-language DocumentStore."""
from __future__ import annotations

from typing import Optional
from pathlib import Path

from infernux.lib import (
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
    """Access the C++-owned generation/coalescing write service."""

    @classmethod
    def instance(cls) -> NativeDocumentStore:
        return NativeDocumentStore.instance()

    @classmethod
    def shutdown(cls) -> None:
        NativeDocumentStore.instance().shutdown()

    @classmethod
    def flush(cls, path: Optional[str] = None) -> None:
        store = NativeDocumentStore.instance()
        if path is None:
            store.flush_all()
        else:
            store.flush_path(path)

    @classmethod
    def is_idle(cls) -> bool:
        """Return whether asynchronous document IO has fully drained."""
        return bool(NativeDocumentStore.instance().is_idle)


def capture_document_file_state(path: str):
    """Capture the durable target state used by conditional atomic writes."""
    return NativeDocumentStore.instance().capture_file_state(path)


def read_document_text_snapshot(path: str):
    """Read content with the exact baseline for its next conditional write."""
    return NativeDocumentStore.read_text_snapshot(path)


def write_document_text(
    path: str,
    content: str | bytes | bytearray,
    *,
    mode: FileWriteMode = FileWriteMode.Replace,
    expected_file_state=None,
    commit_chain_token: str = "",
    chain_id: str = "",
    write_scope: PathWriteScope | None = None,
) -> int:
    """Write one UTF-8 document and return its path generation."""
    if Path(path).suffix.casefold() == ".inxdoc":
        raise ValueError("Cooked asset documents are read-only; save an authoring copy instead")
    options = DocumentWriteOptions()
    options.mode = mode
    options.expected_file_state = expected_file_state
    options.commit_chain_token = str(commit_chain_token or chain_id or "")
    if write_scope is not None:
        options.write_scope = write_scope
    return NativeDocumentStore.instance().write_and_wait(path, content, options)


def submit_document_text(
    path: str,
    content: str,
    *,
    mode: FileWriteMode = FileWriteMode.Replace,
    expected_file_state=None,
    commit_chain_token: str = "",
    chain_id: str = "",
    write_scope: PathWriteScope | None = None,
) -> DocumentWriteTicket:
    """Queue one UTF-8 document write without blocking the caller.

    The native store coalesces newer generations for the same path and writes
    them atomically on its IO workers.  Callers that need durability before
    shutdown should use :meth:`DocumentStore.flush`.
    """
    if Path(path).suffix.casefold() == ".inxdoc":
        raise ValueError("Cooked asset documents are read-only; save an authoring copy instead")
    options = DocumentWriteOptions()
    options.mode = mode
    options.expected_file_state = expected_file_state
    options.commit_chain_token = str(commit_chain_token or chain_id or "")
    if write_scope is not None:
        options.write_scope = write_scope
    return NativeDocumentStore.instance().submit(path, content, options)


__all__ = [
    "DocumentStore",
    "DocumentWriteCancelled",
    "DocumentWriteOptions",
    "DocumentWriteSuperseded",
    "DocumentWriteTicket",
    "PathWriteBusy",
    "PathWriteScope",
    "capture_document_file_state",
    "read_document_text_snapshot",
    "submit_document_text",
    "write_document_text",
]
