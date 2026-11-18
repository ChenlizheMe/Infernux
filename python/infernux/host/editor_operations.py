"""Engine-owned editor authoring operations.

The operation surface exists independently of any transport plugin. MCP and
future local/agentic frontends project this registry instead of owning scene or
asset capabilities themselves.
"""

from __future__ import annotations

from .asset_operations import build_asset_operations
from .data_asset_operations import build_data_asset_operations
from .operation_support import OWNER
from .operations import Operation, OperationRegistry, OperationKind
from .operation_support import operation, on_editor
from .editor import EditorAutomationHost
from .scene_operations import build_scene_operations


def build_editor_operations(project_path: str) -> tuple[Operation, ...]:
    from .shader_operations import build_shader_operations

    return (
        build_scene_operations()
        + build_asset_operations(str(project_path))
        + build_data_asset_operations()
        + build_shader_operations(str(project_path))
        + _additional_operations(project_path)
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


def _additional_operations(project_path):
    from .camera_operations import build_camera_operations
    from .capture_operations import build_capture_operations
    from .console_operations import build_console_operations
    from .docs_operations import build_docs_operations
    from .input_operations import build_input_operations
    from .material_operations import build_material_operations
    from .particle_operations import build_particle_operations
    from .runtime_operations import build_runtime_operations
    from .ui_operations import build_ui_operations
    from .player_operations import build_player_operations
    from .plugin_operations import build_plugin_operations
    from .sim_operations import build_sim_operations
    return (
        (operation("infernux.project.info", OperationKind.QUERY,
                   "Read the current project and scene loading state.",
                   lambda: on_editor("infernux.project.info", lambda: EditorAutomationHost.instance().project_info(project_path)),
                   capability="project.read", tags=("project", "scene", "status")),)
        + build_camera_operations()
        + build_capture_operations()
        + build_console_operations()
        + build_docs_operations()
        + build_input_operations()
        + build_material_operations()
        + build_particle_operations()
        + build_runtime_operations()
        + build_sim_operations()
        + build_ui_operations()
        + build_player_operations(project_path)
        + build_plugin_operations()
    )
