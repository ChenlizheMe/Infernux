"""Load runtime-neutral contracts without bootstrapping Vulkan or the Editor."""

import importlib
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture(scope="module")
def service_contract():
    # A private namespace executes the actual source and its relative imports.
    # Never stub Infernux in sys.modules: native tests can share this process.
    name = "_infernux_service_contract_tests"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "python/infernux/engine")]
    sys.modules[name] = package
    try:
        yield importlib.import_module(f"{name}.player_service_graph")
    finally:
        for module in tuple(sys.modules):
            if module == name or module.startswith(name + "."):
                del sys.modules[module]
