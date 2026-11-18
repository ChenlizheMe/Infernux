"""Manual physics-step operations for deterministic simulation drivers.

These operations bind the deterministic step/clock/pose contract when the host
is a graphical Infernux Editor. The manual path for headless work is
:mod:`infernux.sim`; the host operations expose the same clock and pose
behavior, with a paused Play Mode step requested with an explicit fixed delta
instead of the last wall-clock frame.
"""

from __future__ import annotations

from typing import Any

from .operations import OperationError, OperationKind
from .operation_support import game_object, on_editor, operation


def _required(name: str) -> dict[str, object]:
    return {"type": "integer", "minimum": 0, "description": name}


def build_sim_operations() -> tuple[Any, ...]:
    return (
        operation(
            "infernux.sim.clock",
            OperationKind.QUERY,
            "Read the admitted simulation clock: published physics tick, rendered frame, scene time and fixed step.",
            _clock,
            capability="sim.read",
            tags=("sim", "clock", "physics", "step", "read"),
        ),
        operation(
            "infernux.sim.pose",
            OperationKind.QUERY,
            "Read one object's world position and unit quaternion together with the physics tick that produced it.",
            _pose,
            capability="sim.read",
            input_properties={"object_id": _required("Scene object identifier.")},
            required=("object_id",),
            tags=("sim", "pose", "rigid", "transform", "read"),
        ),
        operation(
            "infernux.sim.step",
            OperationKind.COMMAND,
            "Request exactly one physics step in paused Play Mode using an explicit fixed delta; poll sim.clock for completion.",
            _step,
            capability="sim.write",
            input_properties={
                "delta_time": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Logical frame delta; must equal the world's live fixed step.",
                },
                "expected_tick": _required("Current published physics tick; stale values are rejected."),
            },
            side_effects=(
                "Advances the paused simulation by exactly one fixed physics step.",
                "Rejects requests while the world is not in exclusive paused Play Mode.",
            ),
            tags=("sim", "step", "physics", "deterministic", "write"),
        ),
    )


def _clock() -> dict[str, object]:
    return on_editor("infernux.sim.clock", _read_clock)


def _read_clock() -> dict[str, object]:
    from infernux.lib import SceneManager

    native = SceneManager.instance()
    with _world_scope(native):
        frame = native._frame_time
        return {
            "tick": int(native.published_fixed_step_index),
            "frame": int(frame.frame_count),
            "sim_s": float(frame.time),
            "fixed_dt": float(native.get_fixed_time_step()),
            "playing": bool(native.is_playing()),
            "paused": bool(native.is_paused()),
            "stepping": bool(native.is_stepping()),
        }


def _pose(object_id: int) -> dict[str, object]:
    return on_editor(
        "infernux.sim.pose",
        lambda: _read_pose(int(object_id)),
    )


def _read_pose(object_id: int) -> dict[str, object]:
    from infernux.lib import SceneManager

    native = SceneManager.instance()
    with _world_scope(native):
        try:
            finder = getattr(native, "find_runtime_object_by_id", None)
            obj = finder(int(object_id)) if callable(finder) else None
            if obj is None:
                obj = game_object(int(object_id))
        except Exception as exc:
            raise OperationError(
                "sim.object_not_found",
                f"No live object with id {object_id} exists in the bound world.",
                details={"object_id": int(object_id)},
            ) from exc
        transform = obj.get_transform()
        position = transform.position
        rotation = transform.rotation
        observed = _read_clock()
    return {
        "object_id": int(getattr(obj, "id", object_id)),
        "position_m": [float(position.x), float(position.y), float(position.z)],
        "quat_wxyz": [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)],
        "tick": observed["tick"],
        "sim_s": observed["sim_s"],
        "frame": "world",
    }


def _step(delta_time: float | None = None, expected_tick: int | None = None) -> dict[str, object]:
    return on_editor(
        "infernux.sim.step",
        lambda: _request_step(
            None if delta_time is None else float(delta_time),
            None if expected_tick is None else int(expected_tick),
        ),
    )


def _request_step(delta_time: float | None, expected_tick: int | None) -> dict[str, object]:
    from infernux.lib import SceneManager

    native = SceneManager.instance()
    play = _play_manager()
    with _world_scope(native):
        if play is None or not native.is_playing():
            raise OperationError(
                "sim.not_playing",
                "sim.step requires an active Play Mode world.",
            )
        if not native.is_paused():
            raise OperationError(
                "sim.not_paused",
                "sim.step requires Play Mode to be paused; an unpaused world advances on its own.",
            )
        fixed = float(native.get_fixed_time_step())
        delta = fixed if delta_time is None else float(delta_time)
        if abs(delta - fixed) > 1e-9:
            raise OperationError(
                "sim.delta_mismatch",
                f"Requested delta {delta!r}s does not match the world fixed step {fixed!r}s.",
                details={"requested": delta, "fixed_dt": fixed},
            )
        tick = int(native.published_fixed_step_index)
        if expected_tick is not None and int(expected_tick) != tick:
            raise OperationError(
                "sim.stale_tick",
                f"Expected tick {int(expected_tick)} but the world is at {tick}.",
                details={"expected_tick": int(expected_tick), "tick": tick},
            )
        # Native paused-step admission: the editor frame loop completes exactly
        # this one step, and the caller polls sim.clock until target_tick is
        # published.
        native.step(delta)
        frame = int(native._frame_time.frame_count)
    return {
        "accepted": True,
        "target_tick": tick + 1,
        "delta_time": delta,
        "fixed_dt": fixed,
        "tick": tick,
        "frame": frame,
    }


def _play_manager():
    from infernux.engine.play_mode import PlayModeManager

    return PlayModeManager.instance()


def _world_scope(native):
    """Run native clock/step reads inside the active Play runtime world.

    Graphical Play runs a runtime world beside the authoring world; native
    ``is_playing``/``step``/``published_fixed_step_index`` answer for the
    *current* world, so a host call must select the runtime world first.
    """
    from contextlib import nullcontext

    ids = list(getattr(native, "_runtime_world_ids", ()) or ())
    if not ids:
        return nullcontext()
    identity = int(ids[0])
    if int(getattr(native, "_current_runtime_world_id", 0)) == identity:
        return nullcontext()
    from infernux.lib import _Infernux

    return _Infernux._RuntimeWorldScope(identity)
