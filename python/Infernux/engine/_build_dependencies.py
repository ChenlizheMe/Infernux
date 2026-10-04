"""BuildDependencyMixin — extracted from GameBuilder."""
from __future__ import annotations

import os
import re
from typing import List

import Infernux._jit_kernels as _jit_kernels
from Infernux.engine.project_requirements import (
    _has_requirement,
    _parse_requirements,
    requirements_path,
)


class BuildDependencyMixin:
    """BuildDependencyMixin method group for GameBuilder."""

    @staticmethod
    def _normalized_requirement_name(line: str) -> str:
        text = line.strip()
        if not text or text.startswith("#") or text.startswith("-"):
            return ""
        name = re.split(r"[><=!;\[\s]", text, maxsplit=1)[0].strip()
        return name.lower().replace("_", "-")

    def _game_build_excluded_packages(self) -> set[str]:
        packages = getattr(self, "_GAME_BUILD_EXCLUDED_PACKAGES", frozenset())
        return {str(pkg).lower().replace("_", "-") for pkg in packages}

    def _is_game_build_excluded_requirement(self, line: str) -> bool:
        name = self._normalized_requirement_name(line)
        if not name:
            return False
        if name in self._game_build_excluded_packages():
            return True
        return not bool(getattr(self, "include_jit_runtime", False)) and name in {
            "numba",
            "llvmlite",
        }

    def _collect_user_dependencies(self) -> List[str]:
        """Scan user scripts for third-party imports and return package names.

        Detection sources (in order of priority):
        1. ``ProjectSettings/requirements.txt`` — the same active PEP 508
           requirements checked by the editor, including versions and markers.
        2. AST-based import scanning of all ``.py`` files under ``Assets/``.
           Only top-level package names are collected (``import a.b`` → ``a``).

        The results are de-duplicated and stdlib/engine names are filtered out.
        Every remaining package must be installed in the build environment.
        """
        import ast
        import importlib.util

        found: set[str] = set()
        uses_infernux_jit = False
        direct_parallel_runtime_imports: set[str] = set()

        # --- Source 1: shared project requirements ----------------------
        req_path = requirements_path(self.project_path)
        if os.path.isfile(req_path):
            entries = [
                (spec, module) for spec, module in _parse_requirements(req_path)
                if not self._is_game_build_excluded_requirement(spec)
            ]
            unresolved = [spec for spec, module in entries if not _has_requirement(spec, module)]
            if unresolved:
                raise RuntimeError(
                    "Player build dependencies do not satisfy project requirements: "
                    + ", ".join(unresolved)
                )
            found.update(module for _spec, module in entries)

        # --- Source 2: AST import scanning ------------------------------
        assets_dir = os.path.join(self.project_path, "Assets")
        if os.path.isdir(assets_dir):
            for root, _, files in os.walk(assets_dir):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        source = f.read()
                    tree = ast.parse(source, filename=fpath)
                    if _jit_kernels.cpu_jit_declarations(source):
                        uses_infernux_jit = True
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                root_name = alias.name.split(".")[0]
                                found.add(root_name)
                                if root_name in {"numba", "llvmlite"}:
                                    direct_parallel_runtime_imports.add(root_name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.level == 0:
                                root_name = node.module.split(".")[0]
                                found.add(root_name)
                                if root_name in {"numba", "llvmlite"}:
                                    direct_parallel_runtime_imports.add(root_name)
        # --- Filter: remove stdlib / engine / excluded ------------------
        found -= self._BUILTIN_MODULES
        found -= self._collect_internal_asset_module_names()
        excluded_imports = self._game_build_excluded_packages()
        skipped = {
            pkg for pkg in found
            if pkg.lower().replace("_", "-") in excluded_imports
        }
        found -= skipped

        include_jit_runtime = bool(getattr(self, "include_jit_runtime", False))
        if direct_parallel_runtime_imports and not include_jit_runtime:
            names = ", ".join(sorted(direct_parallel_runtime_imports))
            raise RuntimeError(
                "Auto Parallel is disabled, but project scripts directly import "
                f"{names}. Enable the CPU JIT build capability or remove the "
                "direct compiler-runtime dependency."
            )

        # A public CPU declaration requires the complete bundled runtime. Merely
        # importing ``Infernux.jit`` for capability inspection does not.
        if include_jit_runtime and (uses_infernux_jit or "numba" in found or "llvmlite" in found):
            found.add("numba")
            found.add("llvmlite")
            found.add("numpy")
        elif not include_jit_runtime:
            found.discard("numba")
            found.discard("llvmlite")

        dependencies = sorted(found)
        missing = [
            package
            for package in dependencies
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise RuntimeError(
                "Player build dependencies are not installed: "
                + ", ".join(missing)
            )
        return dependencies

    def _collect_internal_asset_module_names(self) -> set[str]:
        """Return top-level module names that belong to the project's Assets tree."""
        names: set[str] = {"Assets"}
        assets_dir = os.path.join(self.project_path, "Assets")
        if not os.path.isdir(assets_dir):
            return names

        for entry in os.scandir(assets_dir):
            name = entry.name
            if name.startswith(".") or name in {"__pycache__"}:
                continue
            if entry.is_dir():
                names.add(name)
                continue
            stem, ext = os.path.splitext(name)
            if ext in {".py", ".pyc"} and stem and not stem.startswith("_"):
                names.add(stem)
        return names

