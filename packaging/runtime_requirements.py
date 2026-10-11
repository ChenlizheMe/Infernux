from __future__ import annotations

from packaging.requirements import Requirement

# Keep the managed runtime limited to the packages needed to launch Infernux
# projects and build standalone players from a project's private Python copy.
# Hub UI dependencies like PySide6 belong in the frozen launcher, not here.
_RUNTIME_PACKAGE_SPECS: tuple[tuple[str, str], ...] = (
    ("pip", "pip"),
    ("setuptools", "setuptools"),
    ("wheel", "wheel"),
    ("ordered-set", "ordered_set"),
    ("nuitka", "nuitka"),
    ("numpy", "numpy"),
    ("numba", "numba"),
    ("watchdog", "watchdog"),
    ("Pillow", "PIL"),
    ("imageio", "imageio"),
    ("av", "av"),
    # The default MCP plugin uses the SDK v1 server and FastMCP v3 client APIs.
    # Keep these ranges aligned with the plugin's own requirements.txt.
    ("mcp>=1.24,<2", "mcp"),
    ("fastmcp>=3,<4", "fastmcp"),
)


def runtime_package_specs() -> tuple[tuple[str, str], ...]:
    return _RUNTIME_PACKAGE_SPECS


def runtime_packages() -> tuple[str, ...]:
    return tuple(package for package, _module in _RUNTIME_PACKAGE_SPECS)


def runtime_modules() -> tuple[str, ...]:
    return tuple(module for _package, module in _RUNTIME_PACKAGE_SPECS)


def runtime_probe_code(module_names: tuple[str, ...]) -> str:
    """Check imports and declared API versions in the target Python process."""
    requirements = []
    for package, module in _RUNTIME_PACKAGE_SPECS:
        requirement = Requirement(package)
        if module in module_names and requirement.specifier:
            requirements.append((requirement.name, str(requirement.specifier)))
    return f'''import importlib.util
from importlib.metadata import PackageNotFoundError, version

def ready():
    if not all(importlib.util.find_spec(name) is not None for name in {module_names!r}):
        return False
    requirements = {requirements!r}
    if requirements:
        from packaging.specifiers import SpecifierSet
        for name, specifier in requirements:
            try:
                installed = version(name)
            except PackageNotFoundError:
                return False
            if installed not in SpecifierSet(specifier):
                return False
    return True

print(int(ready()))
'''


__all__ = [
    "runtime_modules",
    "runtime_package_specs",
    "runtime_packages",
    "runtime_probe_code",
]
