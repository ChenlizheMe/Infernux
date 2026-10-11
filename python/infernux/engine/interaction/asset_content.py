"""Reversible content projections used by asset relocation transactions."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from typing import Optional

AssetRenameTransform = Callable[[str, str, str], str]


def _rename_top_level_json_name(content: str, _source: str, destination: str) -> str:
    payload = json.loads(content)
    if not isinstance(payload, dict) or "name" not in payload:
        return content
    payload["name"] = os.path.splitext(os.path.basename(destination))[0]
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


class AssetRenameContentRegistry:
    """Own extension-specific, pure rename projections.

    Adapters only transform text. Workspace writes, rollback, catalog mutation,
    document projection, and Undo identity remain owned by the relocation
    transaction.
    """

    _instance: Optional["AssetRenameContentRegistry"] = None

    def __init__(self) -> None:
        self._adapters: dict[str, AssetRenameTransform] = {}
        self.register((".mat", ".particlegraph"), _rename_top_level_json_name)
        # Script filenames and component class names are independent. Moving a
        # Python asset preserves its source bytes; it is not a symbol refactor.
        AssetRenameContentRegistry._instance = self

    @classmethod
    def instance(cls) -> "AssetRenameContentRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, extensions, transform: AssetRenameTransform) -> None:
        if not callable(transform):
            raise TypeError("asset rename content adapter must be callable")
        for extension in extensions:
            normalized = str(extension or "").strip().lower()
            if not normalized.startswith("."):
                raise ValueError("asset rename content extension must start with '.'")
            self._adapters[normalized] = transform

    def unregister(self, extension: str) -> None:
        self._adapters.pop(str(extension or "").strip().lower(), None)

    def build_patch(self, source: str, destination: str) -> tuple[str, str] | None:
        if not os.path.isfile(source):
            return None
        transform = self._adapters.get(os.path.splitext(source)[1].lower())
        if transform is None:
            return None
        with open(source, "r", encoding="utf-8") as stream:
            original = stream.read()
        updated = transform(original, source, destination)
        if not isinstance(updated, str):
            raise TypeError("asset rename content adapter must return text")
        return (original, updated) if updated != original else None

    def shutdown(self) -> None:
        self._adapters.clear()
        if AssetRenameContentRegistry._instance is self:
            AssetRenameContentRegistry._instance = None


__all__ = [
    "AssetRenameContentRegistry",
    "AssetRenameTransform",
]
