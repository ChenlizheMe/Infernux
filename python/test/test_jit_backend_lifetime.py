"""Real native optimization resources must be retired, not just detached."""

import gc
from unittest.mock import Mock

import pytest

from Infernux._jit_compat import prepare_cpu_backend
from llvmlite import binding as llvm
from llvmlite.binding import ffi, newpassmanagers


@pytest.mark.parametrize(("factory", "dispose_name"), [
    (llvm.create_new_module_pass_manager, "LLVMPY_DisposeNewModulePassManger"),
    (llvm.create_new_function_pass_manager, "LLVMPY_DisposeNewFunctionPassManger"),
])
@pytest.mark.parametrize("retirement", ["close", "exception", "gc"])
def test_pass_manager_releases_native_allocation_once(monkeypatch, factory, dispose_name, retirement):
    prepare_cpu_backend()
    dispose = Mock(wraps=getattr(ffi.lib, dispose_name))
    monkeypatch.setitem(ffi.lib._fntab, dispose_name, dispose)
    manager = factory()
    if retirement == "close":
        manager.close()
        manager.close()
    elif retirement == "exception":
        with pytest.raises(ValueError, match="authored"):
            with manager:
                raise ValueError("authored")
    del manager
    gc.collect()
    dispose.assert_called_once()


@pytest.mark.parametrize("manager_type", [newpassmanagers.ModulePassManager, newpassmanagers.FunctionPassManager])
def test_compatibility_repairs_only_empty_base_disposer(monkeypatch, manager_type):
    monkeypatch.setattr(manager_type, "_dispose", ffi.ObjectRef._dispose)
    prepare_cpu_backend()
    assert manager_type._dispose is newpassmanagers.NewPassManager._dispose

    def upstream_dispose(self):
        raise AssertionError("test only; no native manager is instantiated")

    monkeypatch.setattr(manager_type, "_dispose", upstream_dispose)
    prepare_cpu_backend()
    assert manager_type._dispose is upstream_dispose
