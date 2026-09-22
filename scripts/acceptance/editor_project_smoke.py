"""Exercise project load, save, Play Mode, and shutdown in the GUI Editor."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from Infernux import release_engine
from Infernux.engine.path_utils import resolved_path, same_path
from Infernux.host.commands import MainThreadCommandQueue
from Infernux.host.editor import EditorAutomationHost


_FATAL_PATTERNS = (
    "Validation Error",
    "VUID-",
    "CRASH:",
    "Traceback (most recent call last)",
    "[ERROR]",
    "X Error of failed request",
    "VK_ERROR_DEVICE_LOST",
    "device lost",
    "Aborted",
    "SIGABRT",
    "Segmentation fault",
    "SIGSEGV",
    "segfault",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="Project root containing Assets and ProjectSettings")
    parser.add_argument("--scene", required=True, help="Project-relative .scene path")
    parser.add_argument("--play-seconds", type=float, default=5.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--transition-timeout", type=float, default=30.0)
    parser.add_argument(
        "--capture",
        action="append",
        choices=("scene", "game", "editor"),
        default=[],
        help=(
            "Capture an engine-owned render target while Play Mode is active. "
            "May be repeated; files are written under the project's persistent data root."
        ),
    )
    parser.add_argument(
        "--dialog-timeout",
        type=float,
        default=120.0,
        help="Maximum time for a person or desktop driver to complete each native dialog",
    )
    parser.add_argument(
        "--native-open-dialog",
        default="",
        help="Open this existing file through the Editor's native file dialog",
    )
    parser.add_argument(
        "--native-save-dialog",
        default="",
        help="Choose this destination through the Editor's native save dialog",
    )
    parser.add_argument(
        "--process-log",
        default="",
        help=(
            "Editor stdout/stderr log to scan after shutdown. On Linux a regular "
            "file attached to stdout is discovered automatically."
        ),
    )
    return parser


def _emit(event: str, **payload: object) -> None:
    print(
        json.dumps(
            {"schema": "infernux.editor_project_smoke", "event": event, **payload},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _log_start(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _new_log_text(path: Path, start_size: int) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(min(start_size, path.stat().st_size))
        return stream.read().decode("utf-8", errors="replace")


def _fatal_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if any(pattern.casefold() in line.casefold() for pattern in _FATAL_PATTERNS)
    ]


def _stdout_log_path() -> Path | None:
    """Return a redirected stdout file on Linux without inventing a log path."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        target = os.readlink("/proc/self/fd/1")
    except OSError:
        return None
    if not os.path.isabs(target):
        return None
    path = Path(target).resolve()
    return path if path.is_file() else None


def _acceptance_logs(project: str, process_log: str) -> list[Path]:
    paths = [Path(project) / "Logs" / "engine.log"]
    explicit = str(process_log or os.environ.get("INFERNUX_EDITOR_SMOKE_LOG", "")).strip()
    process_path = Path(resolved_path(explicit)) if explicit else _stdout_log_path()
    if process_path is not None and process_path not in paths:
        paths.append(process_path)
    return paths


def _wait_until(
    predicate: Callable[[], Any],
    *,
    timeout: float,
    label: str,
    interval: float = 0.05,
) -> Any:
    deadline = time.monotonic() + max(float(timeout), 0.01)
    last_value: Any = None
    while time.monotonic() < deadline:
        last_value = predicate()
        if last_value:
            return last_value
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for {label}; last value: {last_value!r}")


def _require_dialog_path(result: dict[str, object], expected_path: str) -> str:
    error = str(result.get("error", "") or "")
    if error:
        raise RuntimeError(f"Native file dialog failed: {error}")
    if not bool(result.get("accepted")):
        raise RuntimeError(f"Native file dialog did not accept a path: {result!r}")
    selected_path = resolved_path(str(result.get("path", "") or ""))
    if not same_path(selected_path, expected_path):
        raise RuntimeError(
            f"Native file dialog selected {selected_path!r}, expected {expected_path!r}"
        )
    return selected_path


def _scene_manager_ready(manager: object) -> bool:
    """Return whether the editor's deferred initial scene load has settled.

    ``SceneFileManager.open_scene`` rejects requests while its deferred load is
    active.  The automation host can observe the project document before that
    load reaches the owner safe point, so checking only ``project-info`` races
    with the manager.  A non-loading manager with a current scene path is the
    stable state needed before deciding whether another open is necessary.
    """

    if isinstance(manager, dict):
        is_loading = manager.get("is_loading", True)
        current_scene_path = manager.get("current_scene_path", "")
    else:
        is_loading = getattr(manager, "is_loading", True)
        current_scene_path = getattr(manager, "current_scene_path", "")
    return not bool(is_loading) and bool(str(current_scene_path or "").strip())


def _run_smoke(
    project: str,
    scene_path: str,
    *,
    play_seconds: float,
    startup_timeout: float,
    transition_timeout: float,
    dialog_timeout: float,
    native_open_dialog: str,
    native_save_dialog: str,
    capture_sources: tuple[str, ...],
    outcome: dict[str, object],
) -> None:
    queue = MainThreadCommandQueue.instance()
    try:
        if not queue.wait_until_ready(startup_timeout):
            raise TimeoutError("Editor main-thread command queue did not become ready")
        host = EditorAutomationHost.instance()

        def run(name: str, fn: Callable[[], Any]) -> Any:
            return queue.run_sync(
                f"editor-smoke.{name}",
                fn,
                timeout_ms=max(int(transition_timeout * 1000), 1),
            )

        def run_dialog(name: str, fn: Callable[[], Any]) -> Any:
            return queue.run_sync(
                f"editor-smoke.{name}",
                fn,
                timeout_ms=max(int(dialog_timeout * 1000), 1),
            )

        def project_info() -> dict[str, object]:
            return run("project-info", lambda: host.project_info(project))

        _wait_until(
            lambda: project_info().get("active_scene", {}).get("path"),
            timeout=startup_timeout,
            label="initial scene",
        )

        from Infernux.engine.scene_manager import SceneFileManager

        manager = run("scene-manager", SceneFileManager.instance)
        if manager is None:
            raise RuntimeError("SceneFileManager is unavailable")

        def scene_manager_state() -> dict[str, object]:
            # SceneFileManager is owner-thread state.  Never read its mutable
            # properties directly from this worker thread while deferred loads
            # are being published; queue every snapshot through the host.
            return dict(
                run(
                    "scene-manager-state",
                    lambda: {
                        "is_loading": bool(manager.is_loading),
                        "current_scene_path": str(manager.current_scene_path or ""),
                    },
                )
            )

        _wait_until(
            lambda: _scene_manager_ready(scene_manager_state()),
            timeout=startup_timeout,
            label="initial scene deferred load",
        )
        current_scene_path = str(scene_manager_state().get("current_scene_path", ""))
        if not same_path(current_scene_path, scene_path):
            accepted = run("open-scene", lambda: manager.open_scene(scene_path))
            if not accepted:
                raise RuntimeError(f"Editor rejected scene open: {scene_path}")

            def requested_scene_ready() -> bool:
                state = scene_manager_state()
                return _scene_manager_ready(state) and same_path(
                    str(state.get("current_scene_path", "")), scene_path
                )

            _wait_until(
                requested_scene_ready,
                timeout=startup_timeout,
                label="requested scene",
            )

        saved = run("save-scene", manager.save_current_scene)
        if not saved:
            raise RuntimeError(f"Editor failed to save scene: {scene_path}")
        _emit("editor-ready", project=project, scene=scene_path, saved=True)

        if native_open_dialog or native_save_dialog:
            from Infernux.lib import _Infernux as native

            if native_open_dialog:
                result = run_dialog(
                    "native-open-dialog",
                    lambda: native._show_native_file_dialog(
                        "open_file",
                        "Infernux Open Dialog Acceptance",
                        os.path.dirname(native_open_dialog),
                        [("Text", "txt")],
                    ),
                )
                selected = _require_dialog_path(result, native_open_dialog)
                _emit("native-open-dialog", path=selected)

            if native_save_dialog:
                result = run_dialog(
                    "native-save-dialog",
                    lambda: native._show_native_file_dialog(
                        "save_file",
                        "Infernux Save Dialog Acceptance",
                        native_save_dialog,
                        [("Text", "txt")],
                    ),
                )
                selected = _require_dialog_path(result, native_save_dialog)
                _emit("native-save-dialog", path=selected)

        entered = run(
            "enter-play",
            lambda: host.runtime_transition("enter_play_mode"),
        )
        if not bool(entered.get("accepted")):
            raise RuntimeError(f"Editor rejected Play Mode: {entered!r}")
        playing = _wait_until(
            lambda: (
                status
                if (status := run("runtime-status", host.runtime_status)).get("state")
                == "playing"
                else None
            ),
            timeout=transition_timeout,
            label="Play Mode",
        )
        _emit("play-entered", runtime=playing)

        time.sleep(max(float(play_seconds), 0.1))
        played = run("runtime-after-play", host.runtime_status)
        if float(played.get("total_play_time", 0.0)) <= 0.0:
            raise RuntimeError(f"Play Mode clock did not advance: {played!r}")

        if capture_sources:
            from Infernux.application import Application

            persistent_root = run("persistent-data-path", Application.persistent_data_path)
            scene_name = os.path.splitext(os.path.basename(scene_path))[0]
            capture_root = os.path.join(persistent_root, "AcceptanceCaptures")
            for source in capture_sources:
                output_path = os.path.join(capture_root, f"{scene_name}-{source}.png")
                capture_id = run(
                    f"capture-{source}",
                    lambda source=source, output_path=output_path: (
                        Application.request_render_target_capture(source, output_path)
                    ),
                )
                snapshot = _wait_until(
                    lambda capture_id=capture_id: (
                        status
                        if str(
                            (
                                status := run(
                                    "capture-status",
                                    lambda capture_id=capture_id: (
                                        Application.query_render_target_capture(capture_id)
                                    ),
                                )
                            ).get("status", "")
                        )
                        in {"completed", "failed", "cancelled", "source_expired"}
                        else None
                    ),
                    timeout=transition_timeout,
                    label=f"{source} render-target capture",
                )
                if str(snapshot.get("status")) != "completed":
                    raise RuntimeError(f"{source} capture failed: {snapshot!r}")
                if not os.path.isfile(output_path):
                    raise RuntimeError(f"{source} capture did not write {output_path!r}")
                _emit(
                    "capture",
                    source=source,
                    path=resolved_path(output_path),
                    width=int(snapshot.get("width", 0)),
                    height=int(snapshot.get("height", 0)),
                    pixel_origin=str(snapshot.get("pixel_origin", "")),
                )

        exited = run(
            "exit-play",
            lambda: host.runtime_transition("exit_play_mode"),
        )
        if not bool(exited.get("accepted")):
            raise RuntimeError(f"Editor rejected Play Mode exit: {exited!r}")
        editing = _wait_until(
            lambda: (
                status
                if (status := run("runtime-status", host.runtime_status)).get("state")
                == "edit"
                else None
            ),
            timeout=transition_timeout,
            label="Edit Mode restore",
        )
        outcome["passed"] = {
            "project": project,
            "scene": scene_path,
            "play_seconds": float(played.get("total_play_time", 0.0)),
            "enter_timings_ms": playing.get("transition_timings_ms", {}),
            "exit_timings_ms": editing.get("transition_timings_ms", {}),
        }
        run("close", host.request_editor_close)
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        try:
            queue.run_sync(
                "editor-smoke.close-after-failure",
                EditorAutomationHost.instance().request_editor_close,
                timeout_ms=5000,
            )
        except BaseException:
            pass


def main() -> int:
    args = _parser().parse_args()
    project = resolved_path(args.project)
    scene_argument = os.path.expandvars(os.path.expanduser(str(args.scene)))
    if os.path.isabs(scene_argument):
        scene_path = resolved_path(scene_argument)
    else:
        scene_relative = scene_argument.replace("\\", "/")
        scene_path = resolved_path(os.path.join(project, *scene_relative.split("/")))
    if not os.path.isdir(project):
        raise FileNotFoundError(project)
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(scene_path)
    if args.play_seconds <= 0:
        raise ValueError("--play-seconds must be positive")

    log_paths = _acceptance_logs(project, args.process_log)
    log_starts = {path: _log_start(path) for path in log_paths}
    outcome: dict[str, object] = {}

    worker = threading.Thread(
        target=_run_smoke,
        args=(project, scene_path),
        kwargs={
            "play_seconds": args.play_seconds,
            "startup_timeout": args.startup_timeout,
            "transition_timeout": args.transition_timeout,
            "dialog_timeout": args.dialog_timeout,
            "native_open_dialog": resolved_path(args.native_open_dialog)
            if args.native_open_dialog
            else "",
            "native_save_dialog": resolved_path(args.native_save_dialog)
            if args.native_save_dialog
            else "",
            "capture_sources": tuple(dict.fromkeys(args.capture)),
            "outcome": outcome,
        },
        name="InfernuxEditorProjectSmoke",
        daemon=True,
    )
    worker.start()
    try:
        release_engine(project)
    except BaseException as exc:
        outcome.setdefault("error", f"{type(exc).__name__}: {exc}")
    worker.join(timeout=max(args.transition_timeout, 5.0))
    if worker.is_alive():
        outcome.setdefault("error", "Editor smoke worker did not stop after engine shutdown")

    fatal_records = [
        {"path": str(path), "lines": _fatal_lines(_new_log_text(path, log_starts[path]))}
        for path in log_paths
    ]
    fatal_records = [record for record in fatal_records if record["lines"]]
    if fatal_records:
        outcome.setdefault("error", "Editor emitted fatal diagnostics")
    if "error" in outcome:
        _emit("failed", error=outcome["error"], fatal_logs=fatal_records)
        return 1

    passed = outcome.get("passed")
    if not isinstance(passed, dict):
        _emit("failed", error="Editor smoke completed without a success result")
        return 1
    _emit("passed", **passed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
