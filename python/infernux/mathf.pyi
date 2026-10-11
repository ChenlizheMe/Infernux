from __future__ import annotations

from typing import Tuple


class Mathf:
    """A collection of common math functions and constants."""

    # Constants
    PI: float
    """The value of pi (3.14159...)."""
    TAU: float
    """The value of tau (2 * pi)."""
    Infinity: float
    """Positive infinity."""
    NegativeInfinity: float
    """Negative infinity."""
    Epsilon: float
    """The smallest float value greater than zero."""
    Deg2Rad: float
    """Degrees-to-radians conversion constant."""
    Rad2Deg: float
    """Radians-to-degrees conversion constant."""

    # Clamping / Interpolation
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between a minimum and a maximum."""
        ...
    @staticmethod
    def clamp01(value: float) -> float:
        """Clamp a value between 0 and 1."""
        ...
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linearly interpolate between two values."""
        ...
    @staticmethod
    def lerp_unclamped(a: float, b: float, t: float) -> float:
        """Linearly interpolate between two values without clamping t."""
        ...
    @staticmethod
    def inverse_lerp(a: float, b: float, value: float) -> float:
        """Calculate the t parameter that produces value between a and b."""
        ...
    @staticmethod
    def move_towards(current: float, target: float, max_delta: float) -> float:
        """Move current towards target by at most max_delta."""
        ...
    @staticmethod
    def smooth_step(from_val: float, to_val: float, t: float) -> float:
        """Hermite interpolation between two values."""
        ...
    @staticmethod
    def smooth_damp(
        current: float,
        target: float,
        current_velocity: float,
        smooth_time: float,
        max_speed: float = ...,
        delta_time: float = ...,
    ) -> Tuple[float, float]:
        """Smoothly damp a value towards a target. Returns (value, velocity)."""
        ...

    # Angle helpers
    @staticmethod
    def delta_angle(current: float, target: float) -> float:
        """Calculate the shortest difference between two angles in degrees."""
        ...
    @staticmethod
    def lerp_angle(a: float, b: float, t: float) -> float:
        """Linearly interpolate between two angles in degrees."""
        ...
    @staticmethod
    def move_towards_angle(current: float, target: float, max_delta: float) -> float:
        """Move current angle towards target angle by at most max_delta degrees."""
        ...

    # Repeating patterns
    @staticmethod
    def repeat(t: float, length: float) -> float:
        """Loop a value t so that it is within 0 and length."""
        ...
    @staticmethod
    def ping_pong(t: float, length: float) -> float:
        """Ping-pong a value t so that it bounces between 0 and length."""
        ...

    # Comparison
    @staticmethod
    def approximately(a: float, b: float) -> bool:
        """Returns True if two floats are approximately equal."""
        ...
    @staticmethod
    def sign(f: float) -> float:
        """Return the sign of f: -1, 0, or 1."""
        ...

    # Transcendental
    @staticmethod
    def sin(f: float) -> float:
        """Return the sine of angle f in radians."""
        ...
    @staticmethod
    def cos(f: float) -> float:
        """Return the cosine of angle f in radians."""
        ...
    @staticmethod
    def tan(f: float) -> float:
        """Return the tangent of angle f in radians."""
        ...
    @staticmethod
    def asin(f: float) -> float:
        """Return the arc-sine of f in radians."""
        ...
    @staticmethod
    def acos(f: float) -> float:
        """Return the arc-cosine of f in radians."""
        ...
    @staticmethod
    def atan(f: float) -> float:
        """Return the arc-tangent of f in radians."""
        ...
    @staticmethod
    def atan2(y: float, x: float) -> float:
        """Return the angle in radians whose tangent is y/x."""
        ...

    @staticmethod
    def sqrt(f: float) -> float:
        """Return the square root of f."""
        ...
    @staticmethod
    def pow(f: float, p: float) -> float:
        """Return f raised to the power p."""
        ...
    @staticmethod
    def exp(power: float) -> float:
        """Return e raised to the given power."""
        ...
    @staticmethod
    def log(f: float, base: float = ...) -> float:
        """Return the logarithm of f in the given base (default: natural log)."""
        ...
    @staticmethod
    def log10(f: float) -> float:
        """Return the base-10 logarithm of f."""
        ...

    # Rounding / Abs / Min / Max
    @staticmethod
    def abs(f: float) -> float:
        """Return the absolute value of f."""
        ...
    @staticmethod
    def min(*values: float) -> float:
        """Return the smallest of the given values."""
        ...
    @staticmethod
    def max(*values: float) -> float:
        """Return the largest of the given values."""
        ...
    @staticmethod
    def floor(f: float) -> float:
        """Return the largest integer less than or equal to f."""
        ...
    @staticmethod
    def ceil(f: float) -> float:
        """Return the smallest integer greater than or equal to f."""
        ...
    @staticmethod
    def round(f: float) -> float:
        """Round f to the nearest integer."""
        ...
    @staticmethod
    def floor_to_int(f: float) -> int: ...
    @staticmethod
    def ceil_to_int(f: float) -> int: ...
    @staticmethod
    def round_to_int(f: float) -> int: ...

    # Power-of-two helpers
    @staticmethod
    def is_power_of_two(value: int) -> bool: ...
    @staticmethod
    def next_power_of_two(value: int) -> int: ...
    @staticmethod
    def closest_power_of_two(value: int) -> int: ...
