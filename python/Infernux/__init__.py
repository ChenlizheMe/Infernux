"""Convenient top-level runtime API for game scripts.

This module intentionally re-exports the most common scripting symbols so
user scripts can simply do ``from Infernux import *``.
"""

import importlib

# ── Runtime API (used by game scripts) ─────────────────────────────
from Infernux.engine import release_engine, Engine, LogLevel
from Infernux.math import Vector2, Vector3, vec4f, quatf, vector2, vector3, vector4, quaternion
from Infernux import components as _components_module
from Infernux.components import *
from Infernux import core
from Infernux.core import *
from Infernux.lib import GameObject, Transform, Component, Space, PrimitiveType
from Infernux.debug import Debug, debug, log, log_warning, log_error, log_exception
from Infernux import rendergraph
from Infernux import renderstack
from Infernux import scene
from Infernux.scene import GameObjectQuery, LayerMask, SceneManager
from Infernux import input
from Infernux import ui
from Infernux.timing import Time
from Infernux.mathf import Mathf
from Infernux.coroutine import (
    Coroutine,
    WaitForSeconds,
    WaitForSecondsRealtime,
    WaitForEndOfFrame,
    WaitForFixedUpdate,
    WaitUntil,
    WaitWhile,
)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def __getattr__(name: str):
    """Lazily expose optional JIT helpers without bloating ``from Infernux import *``.

    This keeps Numba out of ordinary star-import paths while still supporting:

        from Infernux import njit
        from Infernux import precompile_jit
        from Infernux import jit
    """
    if name == "jit":
        return importlib.import_module("Infernux.jit")
    if name in {
        "njit", "warmup", "precompile_jit", "ensure_jit_runtime",
        "JIT_AVAILABLE",
    }:
        jit_module = importlib.import_module("Infernux.jit")
        return getattr(jit_module, name)
    raise AttributeError(f"module 'Infernux' has no attribute {name!r}")


__all__ = _dedupe(
    list(_components_module.__all__)
    + list(core.__all__)
    + [
        "release_engine",
        "Engine",
        "LogLevel",
        "Vector2",
        "Vector3",
        "vec4f",
        "quatf",
        "vector2",
        "vector3",
        "vector4",
        "quaternion",
        "GameObject",
        "Transform",
        "Component",
        "Space",
        "PrimitiveType",
        "Debug",
        "debug",
        "log",
        "log_warning",
        "log_error",
        "log_exception",
        "core",
        "rendergraph",
        "renderstack",
        "scene",
        "input",
        "ui",
        "GameObjectQuery",
        "LayerMask",
        "SceneManager",
        "Time",
        "Mathf",
        "Coroutine",
        "WaitForSeconds",
        "WaitForSecondsRealtime",
        "WaitForEndOfFrame",
        "WaitForFixedUpdate",
        "WaitUntil",
        "WaitWhile",
    ]
)