"""Deterministic physics-step clock for manual simulation drivers.

Yanshi binds ``sim.step@1`` to :func:`step` and ``rigid.pose@1`` to
:func:`pose`. A step advances exactly one fixed physics step - never a
rendered frame - and a delta that does not match the world's fixed step is
reported as a contract error instead of silently advancing the world.

The manual driver owns the loop::

    engine = inx.Engine(mode=inx.RuntimeMode.Headless)
    engine.init_headless(project)
    for k in range(11):
        inx.sim.step(engine, expected_tick=k)
        sample = inx.sim.pose(body)

Reads are pure: :func:`clock` and :func:`pose` only observe the published
state, so repeating a read never advances the simulation.
"""

from __future__ import annotations

from typing import Any

from infernux.lib import SceneManager

__all__ = [
    "SimStepError",
    "clock",
    "fixed_delta_time",
    "pose",
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
    }


def step(
    engine: Any,
    *,
    expected_tick: int | None = None,
    delta_time: float | None = None,
) -> dict[str, Any]:
    """Advance exactly one fixed physics step and return the new clock.

    ``engine`` is the ``infernux.Engine`` facade in manual (headless) mode.
    The caller must supply ``delta_time`` equal to the world's live fixed
    step; passing ``None`` uses that fixed step. A stale ``expected_tick``
    or any advance other than one step raises :class:`SimStepError`, so a
    driver can never mistake a partial or duplicated advance for success.
    """
    if engine is None or not hasattr(engine, "tick"):
        raise SimStepError("sim.invalid_engine", "sim.step requires an initialized Engine facade.")
    fixed = fixed_delta_time()
    delta = fixed if delta_time is None else float(delta_time)
    if abs(delta - fixed) > 1e-9:
        raise SimStepError(
            "sim.delta_mismatch",
            f"Requested delta {delta!r}s does not match the world fixed step {fixed!r}s.",
            details={"requested": delta, "fixed_dt": fixed},
        )
    before = clock()
    if expected_tick is not None and int(expected_tick) != before["tick"]:
        raise SimStepError(
            "sim.stale_tick",
            f"Expected tick {int(expected_tick)} but the world is at {before['tick']}.",
            details={"expected_tick": int(expected_tick), "tick": before["tick"]},
        )
    # A Play start may admit one startup frame before the first fixed step.
    # Tick only until the published step advances, then stop: never run a
    # second step inside one request.
    advanced = 0
    after = before
    for _ in range(2):
        engine.tick(delta)
        after = clock()
        advanced = after["tick"] - before["tick"]
        if advanced:
            break
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


def _resolve_object(subject: Any, object_id: int | None) -> Any:
    if subject is not None and hasattr(subject, "get_transform"):
        return subject
    from infernux.lib import SceneManager as _SceneManager

    manager = _SceneManager.instance()
    if object_id is not None and manager is not None:
        finder = getattr(manager, "find_runtime_object_by_id", None)
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
