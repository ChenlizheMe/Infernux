"""Measure runtime pointer dispatch and its physical-query budget in a real project.

The benchmark loads one authored scene, enters Play, and then drives the same
MouseEventDispatcher / UIEventProcessor paths used by Game View and Player.
Temporary cameras, UI controls and a Collider exist only in the runtime scene;
the project document is never saved. Physics-call time, Python scheduling,
callback dispatch and the explicit transform synchronization are reported
separately. This is a measurement artifact, not a self-selected pass threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if (ROOT / "python").is_dir():
    sys.path.insert(0, str(ROOT / "python"))

from Infernux import run_headless
from Infernux.components import InxComponent
from Infernux.debug import DebugConsole, LogType
from Infernux.engine.path_utils import resolved_path, same_path
from Infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch
from Infernux.engine.runtime_mouse_events import MouseEventDispatcher
from Infernux.engine.runtime_screen_ui import (
    collect_runtime_ui_input_surfaces,
    map_runtime_ui_pointer,
    map_runtime_ui_pointers,
)
from Infernux.engine.scene_manager import SceneFileManager
from Infernux.lib import Physics as NativePhysics
from Infernux.lib import SceneManager as NativeSceneManager
from Infernux.lib import Vector3
from Infernux.physics import Physics
from Infernux.scene import SceneManager
from Infernux.ui import UIButton, UICanvas
from Infernux.ui.ui_event_data import PointerType
from Infernux.ui.ui_event_system import UIEventProcessor, UIPointerFrame


class _MouseProbe(InxComponent):
    def _record(self) -> None:
        self.callback_count = int(getattr(self, "callback_count", 0)) + 1

    on_mouse_enter = _record
    on_mouse_over = _record
    on_mouse_exit = _record
    on_mouse_down = _record
    on_mouse_drag = _record
    on_mouse_up = _record
    on_mouse_up_as_button = _record


class _PointerProbe(UIButton):
    def _record(self, _event) -> None:
        self.callback_count = int(getattr(self, "callback_count", 0)) + 1

    on_pointer_enter = _record
    on_pointer_exit = _record
    on_pointer_down = _record
    on_pointer_up = _record
    on_pointer_click = _record
    on_begin_drag = _record
    on_drag = _record
    on_end_drag = _record


class _TimedUIEventProcessor(UIEventProcessor):
    def __init__(self, timing: dict[str, int]) -> None:
        super().__init__()
        self._timing = timing

    def _dispatch_pointer_callback(self, target, method_name, event, epoch) -> None:
        start = time.perf_counter_ns()
        try:
            UIEventProcessor._dispatch_pointer_callback(target, method_name, event, epoch)
        finally:
            self._timing["callback_ns"] += time.perf_counter_ns() - start


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pointer-count", type=int, default=8)
    parser.add_argument("--viewport", default="1920x1080")
    parser.add_argument("--output")
    return parser


def _percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(max(values)),
    }


def _set_position_along_ray(obj, origin, direction, distance: float) -> None:
    obj.transform.position = Vector3(
        float(origin.x) + float(direction.x) * distance,
        float(origin.y) + float(direction.y) * distance,
        float(origin.z) + float(direction.z) * distance,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.pointer_count <= 0:
        raise ValueError("samples/pointer-count must be positive and warmup cannot be negative")
    try:
        viewport_width, viewport_height = (
            int(value) for value in str(args.viewport).lower().split("x", 1)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("--viewport must use WIDTHxHEIGHT") from exc
    if viewport_width <= 0 or viewport_height <= 0:
        raise ValueError("viewport dimensions must be positive")

    project = resolved_path(args.project)
    scene_argument = os.path.expandvars(os.path.expanduser(str(args.scene)))
    if os.path.isabs(scene_argument):
        scene_path = resolved_path(scene_argument)
        scene_relative = os.path.relpath(scene_path, project).replace("\\", "/")
    else:
        scene_relative = scene_argument.replace("\\", "/")
        scene_path = resolved_path(os.path.join(project, *scene_relative.split("/")))
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(scene_path)

    state = {"requested": False, "active": False, "error": "", "result": None}
    errors: list[dict[str, str]] = []

    def on_error(entry):
        if entry.log_type in (LogType.ERROR, LogType.EXCEPTION, LogType.ASSERT):
            errors.append({"type": entry.log_type.name, "message": entry.message})

    def update(_engine, _frame: int) -> bool:
        if errors:
            return False
        if not state["requested"]:
            state["requested"] = True
            manager = SceneFileManager.instance()
            if manager is None or not manager.open_scene(scene_path):
                state["error"] = f"failed to request scene: {scene_relative}"
                return False
            return True

        manager = SceneFileManager.instance()
        if not state["active"]:
            if (
                manager is not None
                and not SceneManager.is_scene_load_pending()
                and not manager.is_loading
                and same_path(manager.current_scene_path, scene_path)
            ):
                native_manager = NativeSceneManager.instance()
                if native_manager is None:
                    state["error"] = "native SceneManager unavailable"
                    return False
                native_manager.play()
                state["active"] = True
            return True

        native_manager = NativeSceneManager.instance()
        scene = native_manager.get_active_scene() if native_manager is not None else None
        camera = scene.effective_game_camera if scene is not None else None
        if scene is None or camera is None:
            state["error"] = "active runtime scene/camera unavailable"
            return False

        center = (viewport_width * 0.5, viewport_height * 0.5)
        origin, direction = camera.screen_point_to_ray(
            center[0], center[1], float(viewport_width), float(viewport_height)
        )
        created = []
        publication = publish_runtime_dispatch_epoch((_MouseProbe, _PointerProbe))
        publication.commit()
        originals = {
            "raycast": Physics.raycast,
            "raycast_all": Physics.raycast_all,
            "raycast_batch": Physics.raycast_batch,
        }
        query = {"count": 0, "elapsed_ns": 0, "kinds": {}}
        callback_timing = {"callback_ns": 0}

        def wrap_query(name, function):
            def measured(*positional, **keywords):
                start = time.perf_counter_ns()
                try:
                    return function(*positional, **keywords)
                finally:
                    elapsed = time.perf_counter_ns() - start
                    query["count"] += 1
                    query["elapsed_ns"] += elapsed
                    query["kinds"][name] = query["kinds"].get(name, 0) + 1
            return measured

        for name, function in originals.items():
            setattr(Physics, name, staticmethod(wrap_query(name, function)))

        measurements = []

        def timed_dispatcher() -> MouseEventDispatcher:
            dispatcher = MouseEventDispatcher()
            original_call = dispatcher._call

            def measured_call(*positional, **keywords):
                start = time.perf_counter_ns()
                try:
                    return original_call(*positional, **keywords)
                finally:
                    callback_timing["callback_ns"] += time.perf_counter_ns() - start

            dispatcher._call = measured_call
            return dispatcher

        def measure(name: str, dispatch, **metadata) -> None:
            for index in range(args.warmup):
                dispatch(index)
            totals = []
            query_times = []
            callback_times = []
            python_times = []
            query_counts = []
            query_kinds: dict[str, int] = {}
            for index in range(args.samples):
                query["count"] = 0
                query["elapsed_ns"] = 0
                query["kinds"] = {}
                callback_timing["callback_ns"] = 0
                start = time.perf_counter_ns()
                dispatch(index)
                total_ns = time.perf_counter_ns() - start
                query_ns = query["elapsed_ns"]
                callback_ns = callback_timing["callback_ns"]
                totals.append(total_ns / 1_000_000.0)
                query_times.append(query_ns / 1_000_000.0)
                callback_times.append(callback_ns / 1_000_000.0)
                python_times.append(max(0, total_ns - query_ns - callback_ns) / 1_000_000.0)
                query_counts.append(int(query["count"]))
                for kind, count in query["kinds"].items():
                    query_kinds[kind] = query_kinds.get(kind, 0) + count
            measurements.append({
                "scenario": name,
                "samples": args.samples,
                "physics_queries_per_frame_min": min(query_counts),
                "physics_queries_per_frame_max": max(query_counts),
                "physics_query_kinds_total": query_kinds,
                "total": _percentiles(totals),
                "physics_query": _percentiles(query_times),
                "python_dispatch": _percentiles(python_times),
                "callback_dispatch": _percentiles(callback_times),
                "async_wait_ms": 0.0,
                **metadata,
            })

        try:
            target = scene.create_game_object("__PointerBenchmarkCollider")
            created.append(target)
            _set_position_along_ray(target, origin, direction, 3.0)
            collider = target.add_component("BoxCollider")
            collider.size = Vector3(1.0, 1.0, 1.0)
            target.add_py_component(_MouseProbe())

            extra_cameras = []
            for index in range(3):
                owner = scene.create_game_object(f"__PointerBenchmarkCamera_{index}")
                created.append(owner)
                owner.transform.position = camera.game_object.transform.position
                owner.transform.euler_angles = camera.game_object.transform.euler_angles
                extra = owner.add_component("Camera")
                extra.depth = 1000.0 + index
                extra.culling_mask = camera.culling_mask
                extra_cameras.append(extra)

            sync_start = time.perf_counter_ns()
            NativePhysics.sync_transforms()
            sync_ms = (time.perf_counter_ns() - sync_start) / 1_000_000.0

            stationary = timed_dispatcher()
            measure(
                "single_viewport_stationary_hover",
                lambda _index: stationary.process(
                    camera, center, (viewport_width, viewport_height),
                    button_state=(False, False, False),
                ),
                viewport_count=1,
            )

            drag = timed_dispatcher()
            drag_states = ((True, True, False), (True, False, False), (False, False, True))
            measure(
                "drag_capture",
                lambda index: drag.process(
                    camera, center, (viewport_width, viewport_height),
                    button_state=drag_states[index % len(drag_states)],
                ),
            )

            multiple_camera_dispatcher = timed_dispatcher()
            measure(
                "multiple_active_cameras_single_output_view",
                lambda _index: multiple_camera_dispatcher.process(
                    scene.effective_game_camera,
                    center,
                    (viewport_width, viewport_height),
                    button_state=(False, False, False),
                ),
                active_camera_count=len(tuple(scene.active_game_cameras)),
            )

            world_owner = scene.create_game_object("__PointerBenchmarkWorldUI")
            created.append(world_owner)
            _set_position_along_ray(world_owner, origin, direction, 6.0)
            world_owner.transform.euler_angles = camera.game_object.transform.euler_angles
            world_button = _PointerProbe()
            world_button.width = 400.0
            world_button.height = 240.0
            world_owner.add_py_component(world_button)
            world_surfaces = collect_runtime_ui_input_surfaces(scene)
            world_dispatcher = timed_dispatcher()
            world_ui = _TimedUIEventProcessor(callback_timing)

            def world_overlap(_index):
                positions, hit = map_runtime_ui_pointer(
                    world_surfaces,
                    camera,
                    center[0], center[1],
                    viewport_width, viewport_height,
                    include_scene_hit=True,
                )
                world_ui.process(world_surfaces, positions, False, False, False, (0.0, 0.0), 1.0 / 60.0)
                world_dispatcher.process(
                    camera, center, (viewport_width, viewport_height), hit=hit,
                    button_state=(False, False, False),
                )

            measure("world_ui_and_3d_overlap_shared_query", world_overlap)

            screen_canvas_owner = scene.create_game_object("__PointerBenchmarkScreenCanvas")
            created.append(screen_canvas_owner)
            screen_canvas = UICanvas()
            screen_canvas_owner.add_py_component(screen_canvas)
            screen_owner = scene.create_game_object("__PointerBenchmarkScreenButton")
            created.append(screen_owner)
            screen_owner.set_parent(screen_canvas_owner)
            screen_button = _PointerProbe()
            screen_button.x = 0.0
            screen_button.y = 0.0
            screen_button.width = float(viewport_width)
            screen_button.height = float(viewport_height)
            screen_owner.add_py_component(screen_button)
            screen_surfaces = collect_runtime_ui_input_surfaces(scene)
            screen_dispatcher = timed_dispatcher()
            screen_ui = _TimedUIEventProcessor(callback_timing)

            def screen_overlap(_index):
                positions, hit = map_runtime_ui_pointer(
                    screen_surfaces,
                    camera,
                    center[0], center[1],
                    viewport_width, viewport_height,
                    include_scene_hit=True,
                )
                screen_ui.process(screen_surfaces, positions, False, False, False, (0.0, 0.0), 1.0 / 60.0)
                screen_dispatcher.process(
                    camera, center, (viewport_width, viewport_height), hit=hit,
                    button_state=(False, False, False),
                )

            measure("blocking_screen_ui_and_3d_overlap", screen_overlap)

            screen_canvas.enabled = False
            touch_surfaces = collect_runtime_ui_input_surfaces(scene)
            touch_ui = _TimedUIEventProcessor(callback_timing)
            touch_points = tuple(
                (
                    center[0] + ((index % 4) - 1.5) * 4.0,
                    center[1] + ((index // 4) - 0.5) * 4.0,
                )
                for index in range(args.pointer_count)
            )

            def pointer_batch(_index):
                mapped = map_runtime_ui_pointers(
                    touch_surfaces, camera, touch_points,
                    viewport_width, viewport_height,
                )
                frames = [
                    UIPointerFrame(
                        pointer_id=pointer_id,
                        pointer_type=PointerType.Touch,
                        canvas_positions=positions,
                        held=True,
                    )
                    for pointer_id, positions in enumerate(mapped)
                ]
                touch_ui.process_pointers(touch_surfaces, frames, 1.0 / 60.0)

            measure(
                "world_ui_pointer_batch",
                pointer_batch,
                pointer_count=args.pointer_count,
            )

            state["result"] = {
                "sync_transforms_ms": sync_ms,
                "active_camera_count": len(tuple(scene.active_game_cameras)),
                "measurements": measurements,
            }
        finally:
            for name, function in originals.items():
                setattr(Physics, name, staticmethod(function))
            publication.rollback()
            for obj in reversed(created):
                scene.destroy_game_object(obj)
            if created:
                scene.process_pending_destroys()
                NativePhysics.sync_transforms()
        return False

    console = DebugConsole.instance()
    console.add_listener(on_error)
    try:
        run_headless(project, update, fixed_delta=1.0 / 60.0, max_frames=1200)
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        console.remove_listener(on_error)

    document = {
        "schema": "infernux.pointer_query_benchmark",
        "status": "passed" if state["result"] is not None and not state["error"] and not errors else "failed",
        "project": project,
        "scene": scene_relative,
        "viewport": [viewport_width, viewport_height],
        "samples": args.samples,
        "warmup": args.warmup,
        "pointer_count": args.pointer_count,
        "error": state["error"],
        "runtime_errors": errors,
        **(state["result"] or {"measurements": []}),
    }
    encoded = json.dumps(document, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        output = resolved_path(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        Path(output).write_text(encoded + "\n", encoding="utf-8")
    return 0 if document["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
