"""BuildDependencyMixin — extracted from GameBuilder."""
from __future__ import annotations

import os
import re
from typing import List

import infernux._jit_kernels as _jit_kernels
from infernux.engine.project_requirements import (
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
        """Collect requirements and imports from the compiled Player closure."""
        import ast
        import importlib.util

        source_paths = self.cooked_python_source_paths()
        sources = getattr(self, "_player_python_source_texts", {})
        if set(source_paths) != set(sources):
            raise RuntimeError("Player dependency sources do not match the compiled source closure")
        found: set[str] = set()
        uses_infernux_jit = False
        direct_parallel_runtime_imports: set[str] = set()
        self._player_dependency_requirements = ()

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
            self._player_dependency_requirements = tuple(spec for spec, _module in entries)

        # --- Source 2: AST import scanning ------------------------------
        for fpath in source_paths:
            source = sources[fpath]
            tree = ast.parse(source, filename=fpath)
            if _jit_kernels.cpu_jit_declarations(source):
                uses_infernux_jit = True
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    found.add(node.module.split(".")[0])
        direct_parallel_runtime_imports = found & {"numba", "llvmlite"}
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
        # importing ``infernux.jit`` for capability inspection does not.
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
        """Return import roots actually shipped by the project script loaders."""
        from pathlib import Path

        names: set[str] = {"Assets", "_infernux_packages"}
        assets_dir = os.path.join(self.project_path, "Assets")
        for source in self.cooked_python_source_paths():
            relative = Path(source).relative_to(self.project_path)
            if relative.parts[0].casefold() != "assets":
                continue
            relative = Path(source).relative_to(assets_dir)
            names.add(relative.parts[0] if len(relative.parts) > 1 else relative.stem)
        return names

