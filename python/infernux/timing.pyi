from __future__ import annotations


class Time:
    """Provides access to time information for the current frame."""

    # Class-level properties (via _TimeMeta metaclass)
    time: float
    """The time in seconds since the start of the game."""
    delta_time: float
    """The time in seconds since the last frame."""
    unscaled_time: float
    """The unscaled time in seconds since the start of the game."""
    unscaled_delta_time: float
    """The unscaled time in seconds since the last frame."""
    fixed_delta_time: float
    """The interval in seconds at which physics and fixed updates are performed."""
    fixed_time: float
    """The time since the last fixed update."""
    fixed_unscaled_time: float
    """The unscaled time since the last fixed update."""
    time_scale: float
    """The scale at which time passes (1.0 = normal speed)."""
    frame_count: int
    """The total number of frames rendered since the start of the game."""
    realtime_since_startup: float
    """The real time in seconds since the application started."""
    maximum_delta_time: float
    """The maximum time a frame can take before delta_time is clamped."""

    @classmethod
    def _reset(cls) -> None: ...
    @classmethod
    def _tick(cls, raw_delta_time: float) -> None: ...
    @classmethod
    def _tick_fixed(cls, fixed_dt: float) -> None: ...
