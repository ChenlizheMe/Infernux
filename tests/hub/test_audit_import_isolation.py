"""Portable packaging tests must not claim the public SDK module namespace."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("sdk_already_present", [False, True])
def test_portable_audit_collection_preserves_sdk_modules(sdk_already_present):
    module = Path(__file__).with_name("test_player_package_audit.py")
    result = subprocess.run(
        [sys.executable, "-I", "-c", r'''
import importlib.abc
import runpy
import sys
import types

class DenySDKBootstrap(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "infernux" or fullname.startswith("infernux."):
            raise AssertionError("Portable audit bootstrapped the SDK: " + fullname)

sys.meta_path.insert(0, DenySDKBootstrap())
if sys.argv[2] == "True":
    sdk = types.ModuleType("infernux")
    sdk.__path__ = []
    sdk.marker = object()
    sys.modules["infernux"] = sdk
    attributes = vars(sdk).copy()
before = {name: value for name, value in sys.modules.items()
          if name == "infernux" or name.startswith("infernux.")}
scope = runpy.run_path(sys.argv[1])
after = {name: value for name, value in sys.modules.items()
         if name == "infernux" or name.startswith("infernux.")}
assert before == after, "Collection changed public SDK module ownership"
if sys.argv[2] == "True":
    assert vars(sdk) == attributes, "Collection mutated the existing SDK package"
assert callable(scope["audit_player_package"])
''', str(module), str(sdk_already_present)],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
