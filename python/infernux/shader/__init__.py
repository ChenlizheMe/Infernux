"""GPU author declarations for statically compiled ``*.frag.py`` assets.

Shader source is parsed, never imported as a game component. Mathematical
operations use GPU types and Slang semantics; they do not run in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._language import INTRINSICS, TYPES


class fullscreen:
    """Fragment program paired with the engine's fullscreen triangle."""


class _Value:
    def __init__(self, *components):
        self.components = components


@dataclass(frozen=True)
class _Parameter:
    kind: str
    default: object
    range: tuple[float, float] | None = None
    label: str | None = None
    hdr: bool = False


class _Parameters:
    @staticmethod
    def float(default=0.0, *, range=None, label=None):
        return _Parameter("float", default, range, label)

    @staticmethod
    def int(default=0, *, range=None, label=None):
        return _Parameter("int", default, range, label)

    @staticmethod
    def bool(default=False, *, label=None):
        return _Parameter("bool", default, label=label)

    @staticmethod
    def color(default=(1.0, 1.0, 1.0, 1.0), *, label=None, hdr=False):
        return _Parameter("color", default, label=label, hdr=hdr)


parameter = _Parameters()


def _gpu_intrinsic(name):
    def intrinsic(*args):
        raise RuntimeError(f"shader.{name} executes in a compiled GPU program")
    intrinsic.__name__ = name
    return intrinsic


for _name in TYPES:
    globals()[_name] = type(_name, (_Value,), {"__module__": __name__})
for _name in INTRINSICS:
    globals()[_name] = _gpu_intrinsic(_name)

__all__ = ["fullscreen", "parameter", *TYPES, *INTRINSICS]
