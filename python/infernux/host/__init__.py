"""Transport-neutral automation host contracts."""

from .commands import CommandFuture, MainThreadCommandQueue
from .editor import EditorAutomationHost
from .editor_operations import build_editor_operations, install_editor_operations
from .operations import (
    Operation,
    OperationError,
    OperationJobRegistry,
    OperationKind,
    OperationRegistry,
    OperationSchema,
    capability_granted,
)

__all__ = [
    "CommandFuture",
    "EditorAutomationHost",
    "MainThreadCommandQueue",
    "Operation",
    "OperationError",
    "OperationJobRegistry",
    "OperationKind",
    "OperationRegistry",
    "OperationSchema",
    "capability_granted",
    "build_editor_operations",
    "install_editor_operations",
]
