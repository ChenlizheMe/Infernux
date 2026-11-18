"""Editor and game camera operations."""

from __future__ import annotations

from infernux.host import EditorAutomationHost, Operation, OperationKind

from infernux.host.operation_support import on_editor, operation


_VECTOR3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def build_camera_operations() -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.camera.editor.inspect",
            OperationKind.QUERY,
            "Read the editor Scene-view camera state.",
            _inspect_editor_camera,
            capability="camera.read",
            tags=("camera", "editor", "scene-view", "inspect"),
        ),
        operation(
            "infernux.camera.editor.state.set",
            OperationKind.COMMAND,
            "Restore the complete editor Scene-view camera state.",
            _set_editor_camera,
            capability="camera.write",
            input_properties={
                "position": _VECTOR3,
                "focus": _VECTOR3,
                "distance": {"type": "number"},
                "yaw": {"type": "number"},
                "pitch": {"type": "number"},
            },
            required=("position", "focus", "distance", "yaw", "pitch"),
            side_effects=("Moves the editor Scene-view camera.",),
            tags=("camera", "editor", "scene-view", "move"),
        ),
        operation(
            "infernux.camera.editor.focus",
            OperationKind.COMMAND,
            "Focus the editor camera on a world-space point.",
            _focus_editor_camera,
            capability="camera.write",
            input_properties={
                "point": _VECTOR3,
                "distance": {"type": "number", "default": 10.0},
            },
            required=("point",),
            side_effects=("Moves the editor Scene-view camera.",),
            tags=("camera", "editor", "focus", "scene-view"),
        ),
        operation(
            "infernux.camera.game.inspect",
            OperationKind.QUERY,
            "Read the effective game Camera component document.",
            _inspect_game_camera,
            capability="camera.read",
            tags=("camera", "game", "runtime", "inspect"),
        ),
    )


def _inspect_editor_camera() -> dict[str, object]:
    return on_editor(
        "infernux.camera.editor.inspect",
        lambda: {"camera": EditorAutomationHost.instance().editor_camera_state()},
    )


def _set_editor_camera(
    position: list[float],
    focus: list[float],
    distance: float,
    yaw: float,
    pitch: float,
) -> dict[str, object]:
    def edit():
        return {
            "camera": EditorAutomationHost.instance().restore_editor_camera(
                position, focus, distance, yaw, pitch
            )
        }

    return on_editor("infernux.camera.editor.state.set", edit)


def _focus_editor_camera(point: list[float], distance: float = 10.0) -> dict[str, object]:
    def edit():
        return {
            "camera": EditorAutomationHost.instance().focus_editor_camera(
                point, distance
            )
        }

    return on_editor("infernux.camera.editor.focus", edit)


def _inspect_game_camera() -> dict[str, object]:
    def read():
        return {"camera": EditorAutomationHost.instance().game_camera_state()}

    return on_editor("infernux.camera.game.inspect", read)


__all__ = ["build_camera_operations"]
