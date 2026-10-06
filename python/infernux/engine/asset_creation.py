"""Explicit ownership returned by one Project asset creator."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class AssetCreationResult:
    """Only set created_path after this operation actually publishes its item.

    A rejected conditional write owns no path, even if another writer has now
    populated the destination. Later import failures retain the published path
    so the command can clean up only its own item.
    """

    success: bool
    detail: str = ""
    created_path: str = ""

    def __bool__(self) -> bool:
        return self.success

    def __iter__(self) -> Iterator[bool | str]:
        """Expose the UI's success/message pair separately from file ownership."""
        yield self.success
        yield self.detail
