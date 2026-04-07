"""
Static ``Time`` class — Unity-style frame timing information.

Access timing data from anywhere in gameplay scripts without instantiation::

    from Infernux import Time

    class PlayerController(InxComponent):
        speed = 10.0

        def update(self, dt):
            velocity = self.speed * Time.delta_time
            if Time.frame_count % 60 == 0:
                debug.log(f"Play time: {Time.time:.1f}s")

Note:
    The ``dt`` parameter passed to ``update()`` / ``fixed_update()`` is the
    **raw** (unscaled) delta coming from C++.  ``Time.delta_time`` is the
    *scaled* delta (raw × ``Time.time_scale``), matching Unity's behaviour.
"""

from __future__ import annotations

import time as _time_mod
from Infernux.debug import Debug


# ---------------------------------------------------------------------------
# Metaclass — enables property access directly on the *class* object
# (``Time.delta_time`` instead of ``Time().delta_time``).
# ---------------------------------------------------------------------------

class _TimeMeta(type):
    """Metaclass enabling ``Time.xxx`` as class-level properties."""

    # -- Scaled timing ------------------------------------------------------

    @property
    def time(cls) -> float:
        """Scaled elapsed time since play mode started (seconds)."""
        return cls._time

    @property
    def delta_time(cls) -> float:
        """Scaled duration of the last frame (seconds)."""
        return cls._delta_time

    # -- Unscaled timing ----------------------------------------------------

    @property
    def unscaled_time(cls) -> float:
        """Unscaled elapsed time since play mode started (seconds)."""
        return cls._unscaled_time

    @property
    def unscaled_delta_time(cls) -> float:
        """Unscaled (wall-clock) duration of the last frame (seconds)."""
        return cls._unscaled_delta_time

    # -- Game-only timing (excludes editor panel overhead) ------------------

    @property
    def game_delta_time(cls) -> float:
        """Game-only frame cost (seconds), excluding editor panel overhead.

        Sum of SceneManager Update/LateUpdate + PrepareFrame + Game camera render.
        Use ``1.0 / Time.game_delta_time`` for game-only FPS estimation.
        Returns 0 when no data is available (e.g. before the first rendered frame).
        """
        return cls._game_delta_time

    # -- Fixed (physics) timing ---------------------------------------------

    @property
    def fixed_delta_time(cls) -> float:
        """Physics / ``fixed_update`` interval (seconds, default 0.02 = 50 Hz)."""
        return cls._fixed_delta_time

    @fixed_delta_time.setter
    def fixed_delta_time(cls, value: float) -> None:
        cls._fixed_delta_time = max(0.001, float(value))

    @property
    def fixed_time(cls) -> float:
        """Scaled time accumulated across physics steps."""
        return cls._fixed_time

    @property
    def fixed_unscaled_time(cls) -> float:
        """Unscaled time accumulated across physics steps."""
        return cls._fixed_unscaled_time

    # -- Time scale ---------------------------------------------------------

    @property
    def time_scale(cls) -> float:
        """Global time multiplier (0 = frozen, 1 = normal)."""
        return cls._time_scale

    @time_scale.setter
    def time_scale(cls, value: float) -> None:
        cls._time_scale = max(0.0, float(value))
        # Keep PlayModeManager in sync (lazy import to avoid cycles)
        try:
            from Infernux.engine.play_mode import PlayModeManager
            pm = PlayModeManager.instance()
            if pm is not None:
                pm._time_scale = cls._time_scale
        except ImportError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass  # PlayModeManager not yet loaded during early init
        except Exception as exc:
            import sys
            print(f"[Time] Failed to sync time_scale to PlayModeManager: {exc}", file=sys.stderr)

    # -- Frame counting -----------------------------------------------------

    @property
    def frame_count(cls) -> int:
        """Number of frames elapsed since play mode started."""
        return cls._frame_count

    # -- Real (wall-clock) timing -------------------------------------------

    @property
    def realtime_since_startup(cls) -> float:
        """Wall-clock seconds since the engine process launched."""
        return _time_mod.time() - cls._startup_time

    # -- Safety clamp -------------------------------------------------------

    @property
    def maximum_delta_time(cls) -> float:
        """Upper clamp for ``delta_time`` to prevent spiral-of-death (default 0.1 s)."""
        return cls._maximum_delta_time

    @maximum_delta_time.setter
    def maximum_delta_time(cls, value: float) -> None:
        cls._maximum_delta_time = max(0.01, float(value))


# ---------------------------------------------------------------------------
# Time class
# ---------------------------------------------------------------------------

class Time(metaclass=_TimeMeta):
    """Unity-style static Time class — access frame timing without instantiation.

    **All members are class-level** — never instantiate ``Time()``.

    Commonly used properties::

        Time.time                   # scaled elapsed play time
        Time.delta_time             # scaled frame duration
        Time.unscaled_delta_time    # raw frame duration
        Time.fixed_delta_time       # physics step interval (default 0.02)
        Time.time_scale             # get / set  (0 = frozen, 1 = normal)
        Time.frame_count            # frame number since play started
        Time.realtime_since_startup # wall-clock since engine launch
    """

    # -- Internal state (written exclusively by the engine) -----------------
    _time: float = 0.0
    _delta_time: float = 0.0
    _unscaled_delta_time: float = 0.0
    _game_delta_time: float = 0.0          # game-only cost (seconds)
    _fixed_delta_time: float = 0.02        # 1/50 Hz — matches C++ default
    _time_scale: float = 1.0
    _frame_count: int = 0
    _unscaled_time: float = 0.0
    _fixed_time: float = 0.0
    _fixed_unscaled_time: float = 0.0
    _startup_time: float = _time_mod.time()
    _maximum_delta_time: float = 0.1

    # -- Engine hooks (called by PlayModeManager) ---------------------------

    @classmethod
    def _reset(cls) -> None:
        """Reset all timing counters.  Called when entering play mode."""
        cls._time = 0.0
        cls._delta_time = 0.0
        cls._unscaled_delta_time = 0.0
        cls._game_delta_time = 0.0
        cls._time_scale = 1.0
        cls._frame_count = 0
        cls._unscaled_time = 0.0
        cls._fixed_time = 0.0
        cls._fixed_unscaled_time = 0.0

    @classmethod
    def _tick(cls, raw_delta_time: float) -> None:
        """Advance one frame.  Called by ``PlayModeManager.tick()``."""
        clamped = min(max(raw_delta_time, 0.0), cls._maximum_delta_time)
        cls._unscaled_delta_time = clamped
        cls._delta_time = clamped * cls._time_scale
        cls._time += cls._delta_time
        cls._unscaled_time += clamped
        cls._frame_count += 1

    @classmethod
    def _tick_fixed(cls, fixed_dt: float) -> None:
        """Advance fixed time by one step.  Called each physics iteration."""
        cls._fixed_time += fixed_dt * cls._time_scale
        cls._fixed_unscaled_time += fixed_dt
