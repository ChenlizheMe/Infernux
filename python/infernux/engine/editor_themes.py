"""Editor theme service used by the Automation Host.

0.5 owns the theme table in ``infernux.engine.ui.theme``; 0.4.8's richer
``EditorThemes`` contribution cache is not part of this tree yet. This module
keeps the host operations (``infernux.theme.list/select/refresh``) working
against the current engine surface without pretending per-package theme
contributions exist.
"""

from __future__ import annotations

from typing import Any


class EditorThemes:
    """Process-wide theme facade for automation; no package contributions."""

    _instance: "EditorThemes | None" = None

    @classmethod
    def instance(cls) -> "EditorThemes | None":
        if cls._instance is None:
            try:
                from infernux.engine.ui.theme import list_editor_themes  # noqa: F401
            except Exception:
                return None
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _catalog() -> list[Any]:
        from infernux.engine.ui.theme import list_editor_themes

        return list(list_editor_themes())

    def snapshot(self) -> dict[str, Any]:
        from infernux.engine.ui.theme import active_editor_theme

        return {"active": str(active_editor_theme()), "themes": self._catalog()}

    def select(self, name: str) -> None:
        from infernux.engine.ui.theme import set_editor_theme

        if not set_editor_theme(str(name)):
            raise ValueError(f"Unknown editor theme: {name}")

    def refresh_all(self) -> None:
        """Theme tables are read live in 0.5; nothing is cached to refresh."""

    def refresh_package(self, reference: str) -> None:
        """Packages cannot contribute themes in 0.5; keep the call site valid."""

    def remove_package(self, reference: str) -> None:
        """Packages cannot contribute themes in 0.5; keep the call site valid."""
