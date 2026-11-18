"""Deterministic physics-step clock for manual simulation drivers.

Yanshi binds ``sim.step@1`` to :func:`step` and ``rigid.pose@1`` to
:func:`pose`. A step advances exactly one fixed **physics** step in a paused
Play Mode world - never a rendered frame - and a request that cannot produce
exactly one step is reported as a contract error instead of silently
advancing the world.

Canonical headless driver::

    engine = inx.Engine(inx.LogLevel.Warn, inx.RuntimeMode.Headless)
    engine.init_headless(project)
    # ... build the fixture scene, then enter Play Mode through the host ...
    inx.sim.settle(engine)             # drain the Play startup frame
    inx.sim.pause()
    body = inx.sim.body("Slider")      # runtime object, not the authored one
    body.velocity = inx.Vector3(0.5, 0, 0)
    base = inx.sim.clock()
    for k in range(11):
        sample = inx.sim.pose(body, tick=base["tick"] + k) if k == 0 else None
        inx.sim.step(engine, expected_tick=base["tick"] + k)
        sample = inx.sim.pose(body)

Reads are pure: :func:`clock` and :func:`pose` only observe published state,
so repeating a read never advances the simulation.
"""

from __future__ import annotations

from typing import Any

from infernux.lib import SceneManager

__all__ = [
    "SimStepError",
    "body",
    "clock",
    "fixed_delta_time",
    "pause",
    "pose",
    "settle",
    "step",
]


class SimStepError(RuntimeError):
    """A physics-step request violated the manual-stepping contract."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(str(message))
        self.code = str(code)
        self.details = dict(details or {})

    def envelope(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {"code": self.code, "message": str(self), "details": dict(self.details)},
        }


def _native() -> Any:
    manager = SceneManager.instance()
    if manager is None:
        raise SimStepError("sim.unavailable", "The engine has no SceneManager yet.")
    return manager


def fixed_delta_time() -> float:
    """The live fixed physics step of the bound world, in seconds."""
    return float(_native().get_fixed_time_step())


def clock() -> dict[str, Any]:
    """Read the admitted simulation clock without advancing it.

    ``tick`` is the published fixed-step index, ``sim_s`` is the scene time
    admitted with that step, and ``frame`` is the rendered frame counter.
    """
    native = _native()
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


def pause() -> dict[str, Any]:
    """Pause the bound world so manual stepping owns the clock."""
    native = _native()
    if not native.is_playing():
        raise SimStepError("sim.not_playing", "sim.pause requires an active Play Mode world.")
    if not native.is_paused():
        native.pause()
    if not native.is_paused():
        raise SimStepError("sim.pause_rejected", "The world rejected the pause request.")
    return clock()


def settle(engine: Any, *, delta_time: float | None = None, max_ticks: int = 4) -> dict[str, Any]:
    """Advance the Play startup frame so manual stepping starts from a drained accumulator.

    Play admission can admit one frame before the first fixed step; that frame
    must be consumed while the fixture still has no velocity, otherwise the
    first manual step would inherit a partial accumulator. Call this once after
    Play Mode starts and before applying the fixture's initial conditions.
    """
    if engine is None or not hasattr(engine, "tick"):
        raise SimStepError("sim.invalid_engine", "sim.settle requires an initialized Engine facade.")
    native = _native()
    if not native.is_playing():
        raise SimStepError("sim.not_playing", "sim.settle requires an active Play Mode world.")
    delta = fixed_delta_time() if delta_time is None else float(delta_time)
    before = clock()
    for _ in range(max(1, int(max_ticks))):
        engine.tick(delta)
        after = clock()
        if after["tick"] > before["tick"] and not after["stepping"]:
            return after
    raise SimStepError(
        "sim.settle_timeout",
        "Play admission did not publish a fixed step within the settle budget.",
        details={"tick": before["tick"]},
    )


def step(
    engine: Any,
    *,
    expected_tick: int | None = None,
    delta_time: float | None = None,
    max_ticks: int = 4,
) -> dict[str, Any]:
    """Advance exactly one fixed physics step in a paused Play Mode world.

    The step runs through the engine's native paused-step admission and the
    caller's frame ticks only complete that admission; no second step can slip
    into the same request. A stale ``expected_tick`` or any advance other than
    one step raises :class:`SimStepError`.
    """
    if engine is None or not hasattr(engine, "tick"):
        raise SimStepError("sim.invalid_engine", "sim.step requires an initialized Engine facade.")
    native = _native()
    fixed = float(native.get_fixed_time_step())
    delta = fixed if delta_time is None else float(delta_time)
    if abs(delta - fixed) > 1e-9:
        raise SimStepError(
            "sim.delta_mismatch",
            f"Requested delta {delta!r}s does not match the world fixed step {fixed!r}s.",
            details={"requested": delta, "fixed_dt": fixed},
        )
    if not native.is_playing():
        raise SimStepError("sim.not_playing", "sim.step requires an active Play Mode world.")
    if not native.is_paused():
        raise SimStepError(
            "sim.not_paused",
            "sim.step requires a paused world; pause before manual stepping.",
        )
    before = clock()
    if expected_tick is not None and int(expected_tick) != before["tick"]:
        raise SimStepError(
            "sim.stale_tick",
            f"Expected tick {int(expected_tick)} but the world is at {before['tick']}.",
            details={"expected_tick": int(expected_tick), "tick": before["tick"]},
        )
    native.step(delta)
    after = before
    for _ in range(max(1, int(max_ticks))):
        engine.tick(delta)
        after = clock()
        if after["tick"] > before["tick"] and not after["stepping"]:
            break
    advanced = after["tick"] - before["tick"]
    if advanced != 1:
        raise SimStepError(
            "sim.step_not_exact",
            f"One request advanced {advanced} physics steps; the world is not in exclusive manual mode.",
            details={"before_tick": before["tick"], "after_tick": after["tick"], "advanced": advanced},
        )
    return {
        "tick": after["tick"],
        "sim_s": after["sim_s"],
        "frame": after["frame"],
        "dt": delta,
        "advanced": advanced,
    }


def body(name: str) -> Any:
    """Resolve a live object by name in the bound world.

    Play Mode runs a runtime clone, so fixtures must resolve objects after
    entering Play; the authored object and the runtime object are distinct.
    """
    native = _native()
    found = native.find_runtime_object(str(name))
    if found is None:
        raise SimStepError("sim.object_not_found", f"No live runtime object named {name!r}.")
    return found


def _resolve_object(subject: Any, object_id: int | None) -> Any:
    if subject is not None and hasattr(subject, "get_transform"):
        return subject
    native = _native()
    if object_id is not None:
        finder = getattr(native, "find_runtime_object_by_id", None)
        if callable(finder):
            found = finder(int(object_id))
            if found is not None:
                return found
    raise SimStepError(
        "sim.object_not_found",
        "sim.pose needs a live GameObject or an object id present in the current world.",
        details={"object_id": object_id},
    )


def pose(subject: Any = None, *, object_id: int | None = None) -> dict[str, Any]:
    """Read world pose plus the clock that produced it; never advances.

    Returns position in metres and a unit quaternion in ``w, x, y, z`` order,
    matching the ``rigid.pose@1`` capability contract.
    """
    game_object = _resolve_object(subject, object_id)
    transform = game_object.get_transform()
    position = transform.position
    rotation = transform.rotation
    observed = clock()
    return {
        "object_id": int(game_object.id),
        "position_m": [float(position.x), float(position.y), float(position.z)],
        "quat_wxyz": [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)],
        "tick": observed["tick"],
        "sim_s": observed["sim_s"],
        "frame": "world",
        "playing": observed["playing"],
    }
