"""Authored curves and gradients, including gameplay-safe CPU curve sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from infernux._animation_curve import AnimationCurve, Keyframe, WrapMode, WeightedMode


MAX_RAMP_KEYS = 16
CURVE_WRAP_MODES = ("clamp", "repeat", "ping_pong", "default", "once")
GRADIENT_MODES = ("linear", "fixed", "perceptual_blend")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class GradientKey:
    time: float
    color: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        time = _number(self.time, "gradient key time")
        if not 0.0 <= time <= 1.0:
            raise ValueError("gradient key time must be between 0 and 1")
        if not isinstance(self.color, (list, tuple)) or len(self.color) != 4:
            raise ValueError("gradient key color must contain exactly four numbers")
        color = tuple(
            _number(component, "gradient color component") for component in self.color
        )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "color", color)

    def to_dict(self) -> dict[str, Any]:
        return {"time": self.time, "color": list(self.color)}

    @classmethod
    def from_dict(cls, value: Any) -> "GradientKey":
        if type(value) is not dict or set(value) != {"time", "color"}:
            raise ValueError("gradient keys require time and color")
        return cls(value["time"], value["color"])


@dataclass(frozen=True)
class Gradient:
    keys: tuple[GradientKey, ...] = (
        GradientKey(0.0, (1.0, 1.0, 1.0, 1.0)),
        GradientKey(1.0, (0.0, 0.0, 0.0, 0.0)),
    )
    mode: str = "linear"

    def __post_init__(self) -> None:
        keys = tuple(
            key if isinstance(key, GradientKey) else GradientKey.from_dict(key)
            for key in self.keys
        )
        if not 1 <= len(keys) <= MAX_RAMP_KEYS:
            raise ValueError(f"gradient requires between 1 and {MAX_RAMP_KEYS} keys")
        if any(left.time >= right.time for left, right in zip(keys, keys[1:])):
            raise ValueError("gradient key times must be strictly increasing")
        if self.mode not in GRADIENT_MODES:
            raise ValueError(
                "gradient mode must be linear, fixed or perceptual_blend"
            )
        object.__setattr__(self, "keys", keys)

    def to_dict(self) -> dict[str, Any]:
        return {"keys": [key.to_dict() for key in self.keys], "mode": self.mode}

    @classmethod
    def from_dict(cls, value: Any) -> "Gradient":
        if type(value) is not dict or set(value) != {"keys", "mode"}:
            raise ValueError("gradient requires keys and mode")
        if type(value["keys"]) is not list:
            raise ValueError("gradient keys must be an array")
        return cls(
            tuple(GradientKey.from_dict(key) for key in value["keys"]), value["mode"]
        )


__all__ = [
    "CURVE_WRAP_MODES",
    "GRADIENT_MODES",
    "MAX_RAMP_KEYS",
    "AnimationCurve",
    "Keyframe",
    "WrapMode",
    "WeightedMode",
    "Gradient",
    "GradientKey",
]
