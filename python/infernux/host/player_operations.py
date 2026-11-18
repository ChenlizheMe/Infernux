"""Player builds use the common engine build service."""
from infernux.host import EditorAutomationHost, Operation, OperationKind
from .operation_support import operation, on_editor
from .players import managed_players


def build_player_operations(project_path: str) -> tuple[Operation, ...]:
    return (
        operation("infernux.player.launch", OperationKind.WORKFLOW, "Launch an owned desktop Debug Player and await readiness.",
                  lambda executable, arguments=None, start_scene="", timeout=30.0: managed_players().launch(executable, arguments or (), start_scene, timeout),
                  capability="player.write", input_properties={"executable": {"type": "string"}, "arguments": {"type": "array", "items": {"type": "string"}},
                  "start_scene": {"type": "string"}, "timeout": {"type": "number"}}, required=("executable",), thread="mixed"),
        operation("infernux.player.status", OperationKind.QUERY, "Read the owned Player process and readiness state.",
                  lambda: managed_players().status(), capability="player.read", thread="mixed"),
        operation("infernux.player.control", OperationKind.COMMAND, "Control the Player channel: observe, key, press, mouse_button, mouse_motion, capture, motion_capture_arm/status/cancel or shutdown.",
                  lambda action, arguments=None, timeout=30.0: managed_players().control(action, arguments, timeout), capability="player.write",
                  input_properties={"action": {"type": "string"}, "arguments": {"type": "object"}, "timeout": {"type": "number"}}, required=("action",), thread="mixed"),
        operation("infernux.player.logs", OperationKind.QUERY, "Read a bounded stdout log tail from the owned Player.",
                  lambda limit=200: managed_players().logs(limit), capability="player.read", input_properties={"limit": {"type": "integer", "minimum": 1}}, thread="mixed"),
        operation("infernux.player.close", OperationKind.COMMAND, "Request normal shutdown of the owned Player.",
                  lambda timeout=15.0: managed_players().stop(timeout), capability="player.write", input_properties={"timeout": {"type": "number"}}, thread="mixed"),
        operation(
            "infernux.player.targets",
            OperationKind.QUERY,
            "List Player targets contributed by core and enabled platform plugins.",
            lambda: on_editor("infernux.player.targets", EditorAutomationHost.instance().player_build_targets),
            capability="player.read",
            tags=("player", "build", "target", "capability"),
        ),
        operation(
            "infernux.player.build",
            OperationKind.WORKFLOW,
            "Build one registered Player target through the shared build service.",
            lambda target="", output_dir="", game_name="", debug_mode=None,
            lto=None, android_artifact="",
            compress_resources=None, persist_settings=True: _build(
                project_path,
                target=target,
                output_dir=output_dir,
                game_name=game_name,
                debug_mode=debug_mode,
                lto=lto,
                android_artifact=android_artifact,
                compress_resources=compress_resources,
                persist_settings=persist_settings,
            ),
            capability="player.write",
            input_properties={
                "target": {
                    "type": "string",
                    "default": "",
                    "description": "Registered target ID; empty uses Build Settings or the current desktop target.",
                },
                "output_dir": {"type": "string", "default": ""},
                "game_name": {"type": "string", "default": ""},
                "debug_mode": {"type": ["boolean", "null"], "default": None},
                "lto": {"type": ["boolean", "null"], "default": None},
                "android_artifact": {
                    "type": "string",
                    "enum": ["", "apk", "aab"],
                    "default": "",
                },
                "compress_resources": {
                    "type": ["boolean", "null"],
                    "default": None,
                },
                "persist_settings": {"type": "boolean", "default": True},
            },
            side_effects=("Writes a standalone Player package and optionally Build Settings.",),
            tags=("player", "build", "package", "workflow"),
        ),
    )


def _build(
    project_path: str,
    *,
    target: str = "",
    output_dir: str = "",
    game_name: str = "",
    debug_mode: bool | None = None,
    lto: bool | None = None,
    android_artifact: str = "",
    compress_resources: bool | None = None,
    persist_settings: bool = True,
) -> dict[str, object]:
    return EditorAutomationHost.instance().build_player(
        project_path,
        target=target,
        output_dir=output_dir,
        game_name=game_name,
        debug_mode=debug_mode,
        lto=lto,
        android_artifact=android_artifact,
        compress_resources=compress_resources,
        persist_settings=bool(persist_settings),
    )
