"""Bounded Editor Console observation operation."""

from __future__ import annotations

from infernux.host import EditorAutomationHost, Operation, OperationKind

from infernux.host.operation_support import on_editor, operation


def build_console_operations() -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.console.read",
            OperationKind.QUERY,
            "Read a bounded Editor Console snapshot with status-bar data separated.",
            _read,
            capability="console.read",
            input_properties={
                "limit": {"type": "integer", "default": 100},
                "levels": {"type": "array", "items": {"type": "string"}},
            },
            tags=("console", "logs", "debug", "editor"),
        ),
    )


def _read(limit: int = 100, levels: list[str] | None = None):
    return on_editor(
        "infernux.console.read",
        lambda: EditorAutomationHost.instance().console_read(
            max(1, min(int(limit), 2000)), levels or ()
        ),
    )


__all__ = ["build_console_operations"]
