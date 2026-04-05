"""
Internal JIT bootstrap — DO NOT import directly.

Use the public API instead::

    from Infernux.jit import njit, warmup, JIT_AVAILABLE
"""

from __future__ import annotations

_HAS_NUMBA = False
_real_njit = None
try:
    from numba import njit as _numba_njit  # type: ignore[import-untyped]
    _real_njit = _numba_njit
    _HAS_NUMBA = True
except Exception as _exc:
    import sys as _sys
    if hasattr(_sys, '_INFERNUX_DEBUG'):
        print(f"[_jit_kernels] numba unavailable: {type(_exc).__name__}: {_exc}",
              flush=True)
    del _sys


JIT_AVAILABLE = _HAS_NUMBA

# In Nuitka standalone builds user scripts are compiled to .pyc and the
# originals removed.  Numba's cache locator requires the source .py to
# exist, so ``cache=True`` would raise RuntimeError.
_NUITKA_COMPILED = "__compiled__" in globals()


# ── njit wrapper ──────────────────────────────────────────────────────

def njit(*args, **kwargs):
    """``numba.njit`` wrapper — safe for both editor and standalone builds.

    The returned callable always has a ``.py`` attribute pointing to the
    original pure-Python function, so callers can force the fallback::

        @njit(cache=True, fastmath=True)
        def burn(n: int) -> float: ...

        burn(100)       # JIT-accelerated (or fallback if no Numba)
        burn.py(100)    # always pure Python
    """
    if not _HAS_NUMBA:
        # No-op fallback — attach .py for uniform API
        def _wrap(fn):
            fn.py = fn
            return fn
        if args and callable(args[0]):
            args[0].py = args[0]
            return args[0]
        return _wrap

    if _NUITKA_COMPILED:
        kwargs.pop("cache", None)

    # @njit  (bare decorator, no parentheses)
    if args and callable(args[0]):
        fn = args[0]
        compiled = _real_njit(fn)
        compiled.py = fn
        return compiled

    # @njit(cache=True, ...)  (decorator factory)
    inner = _real_njit(**kwargs)

    def _decorator(fn):
        compiled = inner(fn)
        compiled.py = fn
        return compiled
    return _decorator


# ── warmup helper ─────────────────────────────────────────────────────

def warmup(fn, *args, **kwargs):
    """Pre-compile a ``@njit`` function by calling it once.

    No-op when Numba is unavailable or inside a Nuitka standalone build.
    Exceptions during warmup are silently swallowed.

    Usage::

        @njit(cache=True, fastmath=True)
        def burn(n: int) -> float: ...

        warmup(burn, 1)
    """
    if not _HAS_NUMBA or _NUITKA_COMPILED:
        return
    try:
        fn(*args, **kwargs)
    except Exception:
        pass


__all__ = [
    "njit",
    "warmup",
    "JIT_AVAILABLE",
]
