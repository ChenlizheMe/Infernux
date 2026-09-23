"""Private CPU JIT adapter used by :mod:`Infernux.jit`.

Targets without the bundled compiler reject compilation; this module does not
provide a Python execution fallback.
"""

from __future__ import annotations

from typing import Any

JIT_AVAILABLE: bool

def njit(*args: Any, **kwargs: Any) -> Any: ...
def warmup(fn: Any, *args: Any, **kwargs: Any) -> None: ...

__all__: list[str]
