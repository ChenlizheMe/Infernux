"""Loader for the private, engine-owned Python-to-Vulkan compiler."""

from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys


_NATIVE_NAME = "Infernux._compiler.taichi._native._infernux_gpu_compiler"


class CompilerInstallationError(RuntimeError):
    """The engine's private Python-to-Vulkan compiler payload is unusable."""


def _vendor_dir() -> Path:
    configured = os.environ.get("INFERNUX_GPU_JIT_VENDOR_DIR")
    if configured:
        # This private loader runs before the project path service. The
        # explicit wheel/env location is authoritative; do not canonicalise
        # it with a second filesystem policy here.
        return Path(configured)
    packaged = Path(__file__).parent / "_vendor" / "taichi"
    if packaged.is_dir():
        return packaged

    # A source checkout uses PYTHONPATH for the public package while CMake
    # stages the private compiler payload in gpu-jit-wheel. Derive that one
    # canonical sibling from the native module directory so a Release build
    # can be launched without a second hand-written environment variable.
    native_dir = os.environ.get("INFERNUX_NATIVE_MODULE_DIR", "").strip()
    if native_dir:
        native_path = Path(native_dir)
        candidates = (
            native_path / "gpu-jit-wheel" / "Infernux" / "_compiler" / "taichi" / "_vendor" / "taichi",
            native_path.parent / "gpu-jit-wheel" / "Infernux" / "_compiler" / "taichi" / "_vendor" / "taichi",
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    return packaged


def load_native():
    """Load only the private native compiler binding.

    The author-facing Taichi package is deliberately not imported or installed
    into the process namespace. Infernux owns the kernel frontend separately.
    """
    existing = sys.modules.get(_NATIVE_NAME)
    if existing is not None:
        return existing
    core = _vendor_dir() / "_lib" / "core"
    native_path = next(
        (core / f"_infernux_gpu_compiler{suffix}" for suffix in EXTENSION_SUFFIXES
         if (core / f"_infernux_gpu_compiler{suffix}").is_file()),
        None,
    )
    if native_path is None:
        raise CompilerInstallationError(
            "The Infernux installation has no Python-to-Vulkan compiler binding"
        )
    spec = spec_from_file_location(_NATIVE_NAME, native_path)
    if spec is None or spec.loader is None:
        raise CompilerInstallationError(
            f"The Infernux Python-to-Vulkan compiler binding cannot be loaded: {native_path}"
        )
    module = module_from_spec(spec)
    sys.modules[_NATIVE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exception:
        sys.modules.pop(_NATIVE_NAME, None)
        if isinstance(exception, (KeyboardInterrupt, SystemExit)):
            raise
        raise CompilerInstallationError(
            f"The Infernux Python-to-Vulkan compiler binding failed to initialize: {native_path}"
        ) from exception
    return module


__all__ = ["CompilerInstallationError", "load_native"]
