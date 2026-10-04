"""
Minimal setup.py to force platform-specific wheel tags.

Infernux ships pre-built native extensions (.pyd / .dll) as package data,
so the wheel must NOT be tagged 'py3-none-any'.  Overriding has_ext_modules()
makes setuptools produce a platform wheel (e.g. cp313-win_amd64).
"""

from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.build_py import build_py as _build_py

import os
import json
import shutil
from pathlib import Path


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def _case_insensitive_module_paths(canonical: Path) -> bool:
    lowercase = canonical.parent / "infernux"
    return lowercase.is_dir() and lowercase.samefile(canonical)


def generate_public_namespace_stubs(build_lib: str) -> None:
    """Publish the canonical type surface as a PEP 561 lowercase stub package.

    A second runtime package differing only in case cannot coexist on Windows.
    Re-export stubs also resolve back to themselves on that filesystem, so the
    build publishes the existing declarations directly instead. Case-sensitive
    filesystems can re-export the canonical types without duplicating them.
    """
    output_root = Path(build_lib).resolve()
    canonical = output_root / "Infernux"
    if not (canonical / "__init__.pyi").is_file():
        raise RuntimeError("The built Infernux package has no public type declarations.")
    mirror_declarations = _case_insensitive_module_paths(canonical)
    destination = output_root / "infernux-stubs"
    if destination.is_symlink():
        raise RuntimeError("The generated public stub directory must not be a symlink.")
    if destination.exists():
        destination.resolve().relative_to(output_root)
        shutil.rmtree(destination)
    destination.mkdir()
    for source in sorted(canonical.rglob("*")):
        if source.suffix not in (".py", ".pyi"):
            continue
        if source.suffix == ".py" and source.with_suffix(".pyi").is_file():
            continue
        relative = source.relative_to(canonical)
        target = (destination / relative).with_suffix(".pyi")
        target.parent.mkdir(parents=True, exist_ok=True)
        if mirror_declarations:
            shutil.copyfile(source, target)
        else:
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join(["Infernux", *parts])
            content = f"from {module} import *\n"
            if not parts:
                content += "from Infernux import __version__ as __version__\n"
            target.write_text(content, encoding="utf-8")
    (destination / "py.typed").write_text("partial\n", encoding="utf-8")


class CleanPackageDataBuild(_build_py):
    """Do not let removed package data survive setuptools' reusable tree."""

    def run(self):
        if os.environ.get("INFERNUX_STAGED_WHEEL_BUILD") != "1":
            raise RuntimeError(
                "Infernux wheels contain compiled native runtime files and cannot be "
                "built directly from the source checkout. Build the platform wheel "
                "through the CMake package_python target instead."
            )

        native_source = Path.cwd() / "python" / "Infernux" / "lib"
        native_extensions = tuple(native_source.glob("_Infernux*.pyd")) + tuple(
            native_source.glob("_Infernux*.so")
        ) + tuple(native_source.glob("_Infernux*.dylib"))
        if not native_extensions:
            raise RuntimeError(
                "The staged Infernux wheel source is missing the compiled _Infernux "
                "native extension. Rebuild the CMake package_python target."
            )
        contract_path = native_source / "PlayerNativeContract.json"
        expected_contract = {
            "contract": "infernux.player-native",
            "runtime_linkage": "static",
        }
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "The staged wheel source has no readable Player native contract."
            ) from exc
        if contract != expected_contract:
            raise RuntimeError("The staged wheel source has an invalid Player native contract.")

        font_output = Path(self.build_lib) / "Infernux" / "resources" / "fonts"
        if font_output.is_dir():
            shutil.rmtree(font_output)
        super().run()
        public_stub = Path.cwd() / "python" / "infernux.pyi"
        if not public_stub.is_file():
            raise RuntimeError(
                "The staged Infernux wheel source is missing python/infernux.pyi. "
                "Rebuild the CMake package_python target."
            )
        shutil.copy2(public_stub, Path(self.build_lib) / "infernux.pyi")
        generate_public_namespace_stubs(self.build_lib)


setup(distclass=BinaryDistribution, cmdclass={"build_py": CleanPackageDataBuild})
