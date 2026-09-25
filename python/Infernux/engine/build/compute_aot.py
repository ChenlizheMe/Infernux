"""Seal editor-compiled GPU kernels into source-less Player content."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import struct

from Infernux.engine.path_utils import resolved_path
from Infernux.engine.project_context import get_script_module_name
from Infernux._compiler.kernel_contract import (
    attribute_name,
    implicit_receiver_attribute,
    implicit_receiver_name,
    kernel_diagnostic,
)


_ARTIFACT_MAGIC = b"INXGPU\x01"


class ComputeAotBuildError(RuntimeError):
    """The selected Player closure has no complete GPU kernel artifact set."""

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


@dataclass(frozen=True, slots=True)
class ComputeAotResult:
    artifact_count: int
    kernel_count: int


def _attribute_name(node: ast.expr) -> str:
    return attribute_name(node)


def _compute_decorator_names(tree: ast.Module) -> set[str]:
    names = {
        "compute.kernel",
        "inx.compute.kernel",
        "Infernux.compute.kernel",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"Infernux", "infernux"}:
                    root = alias.asname or alias.name
                    names.add(f"{root}.compute.kernel")
                elif alias.name in {"Infernux.compute", "infernux.compute"}:
                    names.add(f"{alias.asname or 'compute'}.kernel")
        elif isinstance(node, ast.ImportFrom):
            if node.module in {"Infernux", "infernux"}:
                for alias in node.names:
                    if alias.name == "compute":
                        names.add(f"{alias.asname or alias.name}.kernel")
            elif node.module in {"Infernux.compute", "infernux.compute"}:
                for alias in node.names:
                    if alias.name == "kernel":
                        names.add(alias.asname or alias.name)
    return names


def _kernel_source_diagnostic(
    source_path: Path,
    qualified: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    reason: str,
    advice: str,
    *,
    target: str,
    line: int | None = None,
    column: int | None = None,
) -> str:
    return kernel_diagnostic(
        source_path,
        qualified,
        node,
        target=target,
        reason=reason,
        advice=advice,
        line=line,
        column=column,
    )


def _validate_class_kernel_source(
    source_path: Path,
    qualified: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    target: str,
) -> None:
    """Reject implicit Python receivers before AOT artifact lookup.

    A class method using the conventional ``self``/``cls`` receiver cannot be
    represented by the fixed GPU ABI: ``Kernel`` is intentionally not a
    descriptor and no implicit receiver value can be serialized.  A
    ``@staticmethod`` declaration is explicit and remains legal.  Performing
    this validation from source gives Android/Web build diagnostics the same
    identity and location as Editor compilation.
    """
    first = implicit_receiver_name(node, in_class=True)
    if first is not None:
        raise ComputeAotBuildError(_kernel_source_diagnostic(
            source_path,
            qualified,
            node,
            f"class-contained kernels cannot use an implicit instance receiver '{first}'",
            "put @staticmethod above @inx.compute.kernel (decorator order: @staticmethod, then @inx.compute.kernel) or move the kernel to module scope",
            target=target,
        ), missing=(qualified,))
    access = implicit_receiver_attribute(node)
    if access is not None:
        receiver, attribute = access
        raise ComputeAotBuildError(_kernel_source_diagnostic(
            source_path,
            qualified,
            node,
            f"class-contained kernels cannot access unbound receiver field '{receiver}.{attribute.attr}'",
            "pass the required scalar or inx.buffer explicitly, or put @staticmethod above @inx.compute.kernel",
            target=target,
            line=attribute.lineno,
            column=attribute.col_offset + 1,
        ), missing=(qualified,))


def declared_kernel_names(
    source_paths: tuple[str | Path, ...],
    project_root: str | Path,
    *,
    target: str = "Player/AOT",
) -> tuple[str, ...]:
    """Return runtime-qualified kernels in the selected Python closure."""

    root = Path(resolved_path(Path(project_root).expanduser()))
    names: set[str] = set()
    for source_path in sorted(
        (Path(resolved_path(Path(path).expanduser())) for path in source_paths),
        key=lambda value: value.as_posix().casefold(),
    ):
        if not source_path.is_file() or source_path.suffix.casefold() != ".py":
            continue
        module_name = get_script_module_name(str(source_path), project_root=str(root))
        if not module_name:
            continue
        try:
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(source_path))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ComputeAotBuildError(
                f"GPU AOT cannot inspect selected script: {source_path}: {error}"
            ) from error
        decorators = _compute_decorator_names(tree)

        class Collector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[tuple[str, bool]] = []

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                if any(
                    _attribute_name(item.func if isinstance(item, ast.Call) else item)
                    in decorators
                    for item in node.decorator_list
                ):
                    qualified: list[str] = []
                    for scope_name, is_function in self.scope:
                        qualified.append(scope_name)
                        if is_function:
                            qualified.append("<locals>")
                    qualified.append(node.name)
                    runtime_name = f"{module_name}.{'.'.join(qualified)}"
                    if any(not is_function for _, is_function in self.scope):
                        _validate_class_kernel_source(
                            source_path, runtime_name, node, target=target
                        )
                    names.add(runtime_name)
                self.scope.append((node.name, True))
                self.generic_visit(node)
                self.scope.pop()

            visit_FunctionDef = _visit_function
            visit_AsyncFunctionDef = _visit_function

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.scope.append((node.name, False))
                self.generic_visit(node)
                self.scope.pop()

        Collector().visit(tree)
    return tuple(sorted(names))


def _artifact_functions(path: Path) -> tuple[str, ...]:
    try:
        payload = path.read_bytes()
        if not payload.startswith(_ARTIFACT_MAGIC):
            raise ValueError("invalid magic")
        header_size = struct.unpack_from("<I", payload, len(_ARTIFACT_MAGIC))[0]
        header_begin = len(_ARTIFACT_MAGIC) + 4
        header_end = header_begin + header_size
        if header_end > len(payload):
            raise ValueError("truncated header")
        header = json.loads(payload[header_begin:header_end].decode("utf-8"))
        cursor = header_end
        for size in header["task_sizes"]:
            cursor += int(size)
        if cursor != len(payload):
            raise ValueError("invalid task payload size")
        locations = header["diagnostic_locations"]
        functions = tuple(
            str(item["function"])
            for item in locations
            if isinstance(item, dict) and str(item.get("function", ""))
        )
        if not functions:
            raise ValueError("missing kernel identity")
        return functions
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ComputeAotBuildError(
            f"GPU AOT artifact is unreadable: {path}: {error}"
        ) from error


def stage_compute_artifacts(
    project_root: str | Path,
    source_paths: tuple[str | Path, ...],
    data_directory: str | Path,
    *,
    target: str = "Player/AOT",
) -> ComputeAotResult:
    """Stage the exact selected kernel set plus engine compute primitives.

    The editor compiler cache is a build input, never a Player runtime cache.
    Artifact filenames already encode source and specialization identity; the
    Player recomputes that identity and opens the matching immutable file.
    """

    root = Path(resolved_path(Path(project_root).expanduser()))
    expected = declared_kernel_names(source_paths, root, target=target)
    destination = (
        Path(resolved_path(Path(data_directory).expanduser()))
        / "Library"
        / "Artifacts"
        / "Compute"
    )
    if not expected:
        shutil.rmtree(destination, ignore_errors=True)
        return ComputeAotResult(0, 0)

    source = root / "Library" / "Artifacts" / "Compute"
    if not source.is_dir():
        raise ComputeAotBuildError(
            "GPU AOT artifacts are missing. Run the selected scenes once in the "
            "Editor so every @inx.compute.kernel specialization is prepared.",
            missing=expected,
        )

    selected: list[Path] = []
    covered: set[str] = set()
    expected_set = set(expected)
    for artifact in sorted(source.glob("*.inxgpu"), key=lambda value: value.name):
        functions = _artifact_functions(artifact)
        matched = expected_set.intersection(functions)
        engine_primitive = any(name.startswith("Infernux.compute.") for name in functions)
        if not matched and not engine_primitive:
            continue
        selected.append(artifact)
        covered.update(matched)

    missing = tuple(sorted(expected_set - covered))
    if missing:
        raise ComputeAotBuildError(
            "GPU AOT is incomplete for the selected Player closure: "
            + ", ".join(missing),
            missing=missing,
        )

    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in selected:
        shutil.copy2(artifact, destination / artifact.name)
    (destination / "AotOnly").write_text(
        "GPU kernels in this Player are immutable build artifacts.\n",
        encoding="utf-8",
        newline="\n",
    )
    return ComputeAotResult(len(selected), len(expected))


__all__ = [
    "ComputeAotBuildError",
    "ComputeAotResult",
    "declared_kernel_names",
    "stage_compute_artifacts",
]
