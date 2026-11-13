from __future__ import annotations


class Time:
    """Provides access to time information for the current frame."""

    # Class-level properties (via _TimeMeta metaclass)
    time: float
    """The time in seconds since the start of the game."""
    delta_time: float
    """The time in seconds since the last frame."""
    smooth_delta_time: float
    """Smoothed scaled game-frame duration; unaffected by fixed-step sampling."""
    unscaled_time: float
    """The unscaled time in seconds since the start of the game."""
    unscaled_delta_time: float
    """The unscaled time in seconds since the last frame."""
    fixed_delta_time: float
    """The interval in seconds at which physics and fixed updates are performed."""
    fixed_time: float
    """Scaled time at the currently admitted native fixed step."""
    fixed_unscaled_time: float
    """Unscaled time represented by admitted native fixed steps."""
    time_scale: float
    """The scale at which time passes (1.0 = normal speed)."""
    frame_count: int
    """Simulation frames admitted in the current Play session."""
    realtime_since_startup: float
    """The real time in seconds since the application started."""
    maximum_delta_time: float
    """The maximum time a frame can take before delta_time is clamped."""

    game_delta_time: float
    """Previous game-frame CPU cost, excluding editor panel overhead."""
    shader_time: float
    """Exact engine-owned time published to Shader globals."""
