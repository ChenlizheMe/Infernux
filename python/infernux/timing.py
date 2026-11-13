"""Read the engine-owned simulation and rendering clocks.

``Time`` is a class-level view of native state. The native frame driver
advances the game clock before lifecycle callbacks; Python never samples a
second clock, accumulates time, or uploads per-frame Shader parameters.
Game time pauses and scales; real time and the render clock have separate
engine-defined meanings. ``fixed_update`` uses ``fixed_delta_time``.
"""
from __future__ import annotations

from infernux.lib import SceneManager as _SceneManager


def _current_native_engine():
    from infernux.application import Application

    engine = Application._current_engine()
    return engine.get_native_engine() if engine is not None else None


class _TimeMeta(type):
    @property
    def time(cls) -> float:
        """Scaled game time since this Play session began."""
        return cls._frame_time.time

    @property
    def delta_time(cls) -> float:
        """Scaled duration admitted to the current simulation frame."""
        return cls._frame_time.delta_time

    @property
    def smooth_delta_time(cls) -> float:
        """Smoothed scaled game-frame duration."""
        return cls._frame_time.smooth_delta_time

    @property
    def unscaled_time(cls) -> float:
        """Game-frame duration accumulated before scaling or clamping."""
        return cls._frame_time.unscaled_time

    @property
    def unscaled_delta_time(cls) -> float:
        """Duration of this simulation frame before scaling or clamping."""
        return cls._frame_time.unscaled_delta_time

    @property
    def game_delta_time(cls) -> float:
        """Previous frame's game CPU cost, excluding editor panel overhead."""
        native = _current_native_engine()
        return native.get_game_only_frame_ms() / 1000.0 if native is not None else 0.0

    @property
    def fixed_delta_time(cls) -> float:
        """Fixed simulation interval, in scaled seconds."""
        return cls._scene_manager.get_fixed_time_step()

    @fixed_delta_time.setter
    def fixed_delta_time(cls, value: float) -> None:
        cls._scene_manager.set_fixed_time_step(value)

    @property
    def fixed_time(cls) -> float:
        """Scaled time at the currently admitted native fixed step."""
        return cls._scene_manager.fixed_time

    @property
    def fixed_unscaled_time(cls) -> float:
        """Unscaled time represented by admitted native fixed steps."""
        return cls._scene_manager.fixed_unscaled_time

    @property
    def time_scale(cls) -> float:
        """Scale applied by the engine to simulation frames."""
        return cls._scene_manager.time_scale

    @time_scale.setter
    def time_scale(cls, value: float) -> None:
        cls._scene_manager.time_scale = value

    @property
    def frame_count(cls) -> int:
        """Simulation frames admitted in this Play session."""
        return cls._frame_time.frame_count

    @property
    def realtime_since_startup(cls) -> float:
        """Native monotonic time since the engine started, including pauses."""
        return cls._scene_manager.get_realtime_since_startup()

    @property
    def shader_time(cls) -> float:
        """Exact render time published to Shader frame globals, or zero before rendering."""
        native = _current_native_engine()
        return native.get_shader_time_seconds() if native is not None else 0.0

    @property
    def maximum_delta_time(cls) -> float:
        """Maximum scaled simulation duration admitted per ordinary frame."""
        return cls._scene_manager.get_max_fixed_delta_time()

    @maximum_delta_time.setter
    def maximum_delta_time(cls, value: float) -> None:
        cls._scene_manager.set_max_fixed_delta_time(value)


class Time(metaclass=_TimeMeta):
    """Class-level access to native time, such as ``Time.delta_time``.

    Game scripts and tools read the same native state. Explicit ``Engine.tick``
    advances it without installing a Python timing callback. Changing scenes
    discards loading delay without restarting the Play-session clock.
    """
    _scene_manager = _SceneManager.instance()
    _frame_time = _scene_manager._frame_time
