"""Interpreter-owned module classification shared by script loading and analysis."""

import sys


def is_stdlib_module(name: str) -> bool:
    """Recognize standard-library roots, including builtins and submodules."""
    return name.split(".", 1)[0] in sys.stdlib_module_names
