"""
``Mathf`` — Unity-style math utility class.

All methods are ``@staticmethod``; access them on the class directly::

    from Infernux import Mathf

    t = Mathf.clamp01(ratio)
    angle = Mathf.lerp_angle(from_deg, to_deg, 0.5)
    value = Mathf.smooth_damp(current, target, vel, 0.3, delta_time=Time.delta_time)
"""

from __future__ import annotations

import builtins as _builtins
import math as _math
import sys as _sys
from typing import Tuple


class Mathf:
    """Collection of common math functions and constants (Unity-style API)."""

    # ====================================================================
    # Constants
    # ====================================================================
    PI: float = _math.pi
    """The famous 3.14159… constant."""

    TAU: float = _math.tau
    """2π — a full turn in radians."""

    Infinity: float = float("inf")
    NegativeInfinity: float = float("-inf")

    Epsilon: float = _sys.float_info.epsilon
    """Smallest float such that ``1.0 + Epsilon != 1.0``."""

    Deg2Rad: float = _math.pi / 180.0
    """Multiply degrees by this to get radians."""

    Rad2Deg: float = 180.0 / _math.pi
    """Multiply radians by this to get degrees."""

    # ====================================================================
    # Clamping / Interpolation
    # ====================================================================

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp *value* between *min_val* and *max_val*."""
        if value < min_val:
            return min_val
        if value > max_val:
            return max_val
        return value

    @staticmethod
    def clamp01(value: float) -> float:
        """Clamp *value* between 0 and 1."""
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linearly interpolate between *a* and *b* by clamped *t* ∈ [0, 1]."""
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return a + (b - a) * t

    @staticmethod
    def lerp_unclamped(a: float, b: float, t: float) -> float:
        """Linearly interpolate between *a* and *b* by *t* (unclamped)."""
        return a + (b - a) * t

    @staticmethod
    def inverse_lerp(a: float, b: float, value: float) -> float:
        """Inverse of :meth:`lerp` — returns *t* for *value* in [a, b], clamped to [0, 1]."""
        denom = b - a
        if _math.fabs(denom) < 1e-12:
            return 0.0
        t = (value - a) / denom
        if t < 0.0:
            return 0.0
        if t > 1.0:
            return 1.0
        return t

    @staticmethod
    def move_towards(current: float, target: float, max_delta: float) -> float:
        """Move *current* towards *target* by at most *max_delta*."""
        diff = target - current
        if _math.fabs(diff) <= max_delta:
            return target
        return current + _math.copysign(max_delta, diff)

    @staticmethod
    def smooth_step(from_val: float, to_val: float, t: float) -> float:
        """Hermite-interpolated smooth step between *from_val* and *to_val*."""
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        t = t * t * (3.0 - 2.0 * t)
        return from_val + (to_val - from_val) * t

    @staticmethod
    def smooth_damp(
        current: float,
        target: float,
        current_velocity: float,
        smooth_time: float,
        max_speed: float = float("inf"),
        delta_time: float = 0.0,
    ) -> Tuple[float, float]:
        """Critically-damped spring towards *target* (Game Programming Gems 4).

        Returns:
            ``(new_value, new_velocity)`` — store *new_velocity* for next frame.

        Example::

            self._vel = 0.0

            def update(self, dt):
                self.x, self._vel = Mathf.smooth_damp(
                    self.x, self.target_x, self._vel,
                    smooth_time=0.3, delta_time=Time.delta_time,
                )
        """
        if smooth_time < 0.0001:
            smooth_time = 0.0001
        omega = 2.0 / smooth_time
        x = omega * delta_time
        exp_factor = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        change = current - target
        original_to = target

        max_change = max_speed * smooth_time
        if change < -max_change:
            change = -max_change
        elif change > max_change:
            change = max_change
        target = current - change

        temp = (current_velocity + omega * change) * delta_time
        new_velocity = (current_velocity - omega * temp) * exp_factor
        output = target + (change + temp) * exp_factor

        if (original_to - current > 0.0) == (output > original_to):
            output = original_to
            new_velocity = (output - original_to) / delta_time if delta_time > 0.0 else 0.0

        return output, new_velocity

    # ====================================================================
    # Angle helpers (degrees)
    # ====================================================================

    @staticmethod
    def delta_angle(current: float, target: float) -> float:
        """Shortest signed difference between two angles in degrees."""
        delta = (target - current) % 360.0
        if delta > 180.0:
            delta -= 360.0
        return delta

    @staticmethod
    def lerp_angle(a: float, b: float, t: float) -> float:
        """Linearly interpolate between angles *a* and *b* (degrees), taking the shortest path."""
        delta = Mathf.delta_angle(a, b)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return a + delta * t

    @staticmethod
    def move_towards_angle(current: float, target: float, max_delta: float) -> float:
        """Move angle *current* towards *target* (degrees) by at most *max_delta*."""
        delta = Mathf.delta_angle(current, target)
        if -max_delta < delta < max_delta:
            return target
        return Mathf.move_towards(current, current + delta, max_delta)

    # ====================================================================
    # Repeating patterns
    # ====================================================================

    @staticmethod
    def repeat(t: float, length: float) -> float:
        """Wrap *t* into ``[0, length)``."""
        if length == 0.0:
            return 0.0
        return t - _math.floor(t / length) * length

    @staticmethod
    def ping_pong(t: float, length: float) -> float:
        """Ping-pong *t* within ``[0, length]``."""
        t = Mathf.repeat(t, length * 2.0)
        return length - _math.fabs(t - length)

    # ====================================================================
    # Comparison
    # ====================================================================

    @staticmethod
    def approximately(a: float, b: float) -> bool:
        """``True`` if *a* and *b* are almost equal (Unity precision)."""
        return _math.fabs(b - a) < _builtins.max(
            1e-6 * _builtins.max(_math.fabs(a), _math.fabs(b)),
            Mathf.Epsilon * 8,
        )

    @staticmethod
    def sign(f: float) -> float:
        """Return ``1.0`` if *f* >= 0, else ``-1.0``."""
        return 1.0 if f >= 0.0 else -1.0

    # ====================================================================
    # Transcendental wrappers (thin, for API discoverability)
    # ====================================================================

    @staticmethod
    def sin(f: float) -> float:
        return _math.sin(f)

    @staticmethod
    def cos(f: float) -> float:
        return _math.cos(f)

    @staticmethod
    def tan(f: float) -> float:
        return _math.tan(f)

    @staticmethod
    def asin(f: float) -> float:
        return _math.asin(f)

    @staticmethod
    def acos(f: float) -> float:
        return _math.acos(f)

    @staticmethod
    def atan(f: float) -> float:
        return _math.atan(f)

    @staticmethod
    def atan2(y: float, x: float) -> float:
        return _math.atan2(y, x)

    @staticmethod
    def sqrt(f: float) -> float:
        """Square root (clamped to >= 0)."""
        return _math.sqrt(_builtins.max(0.0, f))

    @staticmethod
    def pow(f: float, p: float) -> float:
        return _math.pow(f, p)

    @staticmethod
    def exp(power: float) -> float:
        return _math.exp(power)

    @staticmethod
    def log(f: float, base: float = _math.e) -> float:
        if f <= 0.0:
            return float("-inf")
        return _math.log(f, base)

    @staticmethod
    def log10(f: float) -> float:
        if f <= 0.0:
            return float("-inf")
        return _math.log10(f)

    # ====================================================================
    # Rounding / Abs / Min / Max
    # ====================================================================

    @staticmethod
    def abs(f: float) -> float:
        return _math.fabs(f)

    @staticmethod
    def min(*values: float) -> float:
        """Return the smallest of the given values."""
        return _builtins.min(values)

    @staticmethod
    def max(*values: float) -> float:
        """Return the largest of the given values."""
        return _builtins.max(values)

    @staticmethod
    def floor(f: float) -> float:
        return float(_math.floor(f))

    @staticmethod
    def ceil(f: float) -> float:
        return float(_math.ceil(f))

    @staticmethod
    def round(f: float) -> float:
        return float(_builtins.round(f))

    @staticmethod
    def floor_to_int(f: float) -> int:
        return int(_math.floor(f))

    @staticmethod
    def ceil_to_int(f: float) -> int:
        return int(_math.ceil(f))

    @staticmethod
    def round_to_int(f: float) -> int:
        return int(_builtins.round(f))

    # ====================================================================
    # Power-of-two helpers
    # ====================================================================

    @staticmethod
    def is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    @staticmethod
    def next_power_of_two(value: int) -> int:
        """Smallest power of two >= *value*."""
        if value <= 0:
            return 1
        value -= 1
        value |= value >> 1
        value |= value >> 2
        value |= value >> 4
        value |= value >> 8
        value |= value >> 16
        return value + 1

    @staticmethod
    def closest_power_of_two(value: int) -> int:
        """Power of two closest to *value*."""
        next_p = Mathf.next_power_of_two(value)
        prev_p = next_p >> 1
        if prev_p == 0:
            return next_p
        return prev_p if (value - prev_p) < (next_p - value) else next_p
