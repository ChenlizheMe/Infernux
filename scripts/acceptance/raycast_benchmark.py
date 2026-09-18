"""Measure the public RaycastBatch path in a real headless physics world.

This is deliberately a measurement tool, not a pass/fail FPS claim.  It uses
the same published snapshot and reusable output arrays as gameplay code and
reports the Python/native call boundary separately from the scene load.
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

import infernux as inx
from Infernux import run_headless
from Infernux.debug import DebugConsole, LogType
from Infernux.engine.path_utils import resolved_path, same_path
from Infernux.engine.scene_manager import SceneFileManager
from Infernux.lib import Physics as NativePhysics
from Infernux.lib import SceneManager as NativeSceneManager
from Infernux.lib import Vector3
from Infernux.scene import SceneManager


def _output(count: int) -> dict[str, np.ndarray]:
    return {
        "hit": np.zeros(count, dtype=np.uint8),
        "point": np.zeros((count, 3), dtype=np.float32),
        "normal": np.zeros((count, 3), dtype=np.float32),
        "distance": np.zeros(count, dtype=np.float32),
        "body_id": np.zeros(count, dtype=np.uint32),
        "sub_shape_id": np.zeros(count, dtype=np.uint32),
        "triangle_index": np.zeros(count, dtype=np.uint32),
        "collider_id": np.zeros(count, dtype=np.uint64),
        "game_object_id": np.zeros(count, dtype=np.uint64),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--counts", default="1,100,1000,10000")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--collider-counts",
        default="0",
        help="optional comma-separated counts of temporary static BoxColliders; each count is measured as a separate world-size row",
    )
    parser.add_argument(
        "--pattern",
        choices=("all_hit", "sparse", "all_miss"),
        default="all_hit",
        help="ray distribution: all_hit reproduces the dense baseline, sparse mixes one hit with seven misses, all_miss avoids scene geometry",
    )
    parser.add_argument("--output")
    return parser


def _create_static_box_matrix(scene, count: int) -> list:
    """Create a temporary broadphase population without touching the scene document."""
    if count <= 0:
        return []
    objects = []
    side = max(1, int(np.ceil(np.sqrt(count))))
    for index in range(count):
        obj = scene.create_game_object(f"__RaycastBenchmarkCollider_{index}")
        obj.transform.position = Vector3(
            (index % side) * 0.75 - (side - 1) * 0.375,
            0.0,
            (index // side) * 0.75 - (side - 1) * 0.375,
        )
        collider = obj.add_component("BoxCollider")
        collider.size = Vector3(0.5, 0.5, 0.5)
        objects.append(obj)
    NativePhysics.sync_transforms()
    return objects


def _destroy_static_box_matrix(scene, objects: list) -> None:
    for obj in objects:
        scene.destroy_game_object(obj)
    if objects:
        scene.process_pending_destroys()
        NativePhysics.sync_transforms()


def main() -> int:
    args = _parser().parse_args()
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
    counts = tuple(sorted({int(item) for item in str(args.counts).split(",") if item}))
    collider_counts = tuple(sorted({int(item) for item in str(args.collider_counts).split(",") if item}))
    if not counts or any(item <= 0 for item in counts):
        raise ValueError("--counts must contain positive integers")
    if not collider_counts or any(item < 0 for item in collider_counts):
        raise ValueError("--collider-counts must contain non-negative integers")
    if args.samples <= 0 or args.warmup < 0:
        raise ValueError("--samples must be positive and --warmup cannot be negative")

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

        NativePhysics.sync_transforms()
        measurements = []
        active_scene = NativeSceneManager.instance().get_active_scene()
        if active_scene is None:
            state["error"] = "active scene unavailable for collider-growth benchmark"
            return False
        for collider_count in collider_counts:
            temporary_colliders = _create_static_box_matrix(active_scene, collider_count)
            try:
                for count in counts:
                    origins = np.zeros((count, 3), dtype=np.float32)
                    origins[:, 1] = 5.0
                    # Spread the probes over the scene so dense and sparse worlds can
                    # use the same benchmark without creating one Python ray object per
                    # query.
                    side = max(1, int(np.ceil(np.sqrt(count))))
                    axis = (np.arange(count, dtype=np.float32) % side) / max(1, side - 1)
                    origins[:, 0] = axis * 8.0 - 4.0
                    origins[:, 2] = (np.arange(count, dtype=np.float32) // side) * 8.0 / side - 4.0
                    if args.pattern == "all_miss":
                        origins[:, 0] = 1000.0
                        origins[:, 2] = 1000.0
                    elif args.pattern == "sparse":
                        miss = (np.arange(count, dtype=np.int64) % 8) != 0
                        origins[miss, 0] = 1000.0
                        origins[miss, 2] = 1000.0
                    directions = np.zeros_like(origins)
                    directions[:, 1] = -1.0
                    output = _output(count)
                    for _ in range(args.warmup):
                        inx.physics.Physics.raycast_batch(origins, directions, output, max_distance=20.0)
                    timings = []
                    for _ in range(args.samples):
                        start = time.perf_counter_ns()
                        inx.physics.Physics.raycast_batch(origins, directions, output, max_distance=20.0)
                        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
                    measurements.append({
                        "collider_count": collider_count,
                        "count": count,
                        "p50_ms": float(np.percentile(timings, 50)),
                        "p95_ms": float(np.percentile(timings, 95)),
                        "max_ms": float(max(timings)),
                        "rays_per_second_p50": float(count / (np.percentile(timings, 50) / 1000.0)),
                        "hits_last_sample": int(output["hit"].sum()),
                    })
            finally:
                _destroy_static_box_matrix(active_scene, temporary_colliders)
        state["result"] = measurements
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
        "schema": "infernux.raycast_batch_benchmark",
        "status": "passed" if state["result"] is not None and not state["error"] and not errors else "failed",
        "project": project,
        "scene": scene_relative,
        "pattern": args.pattern,
        "samples": args.samples,
        "warmup": args.warmup,
        "collider_counts": list(collider_counts),
        "error": state["error"],
        "runtime_errors": errors,
        "measurements": state["result"] or [],
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
