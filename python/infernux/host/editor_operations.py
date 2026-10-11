"""Engine-owned editor authoring operations.

The operation surface exists independently of any transport plugin. MCP and
future local/agentic frontends project this registry instead of owning scene or
asset capabilities themselves.
"""

from __future__ import annotations

from .asset_operations import build_asset_operations
from .data_asset_operations import build_data_asset_operations
from .operation_support import OWNER
from .operations import Operation, OperationRegistry
from .scene_operations import build_scene_operations


def build_editor_operations(project_path: str) -> tuple[Operation, ...]:
    return (
        build_scene_operations()
        + build_asset_operations(str(project_path))
        + build_data_asset_operations()
    )


def install_editor_operations(
    project_path: str,
    registry: OperationRegistry | None = None,
) -> tuple[str, ...]:
    target = registry or OperationRegistry.instance()
    target.unregister_owner(OWNER)
    operations = build_editor_operations(str(project_path))
    for operation in operations:
        target.register(operation)
    return tuple(operation.schema.id for operation in operations)


__all__ = ["build_editor_operations", "install_editor_operations"]
