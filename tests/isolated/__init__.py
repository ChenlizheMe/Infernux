"""Native lifetime regressions run in child processes, without a parent engine."""

from importlib.util import find_spec
from pathlib import Path


def engine_python_path():
    return str(Path(find_spec("infernux").origin).resolve().parent.parent)
