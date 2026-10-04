"""Type stubs for infernux.lib."""

from __future__ import annotations

lib_dir: str

# Re-exports everything from _Infernux (the compiled pybind11 module)
from ._Infernux import *
