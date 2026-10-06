"""Seal editor-compiled GPU kernels into source-less Player content."""

from __future__ import annotations

import ast
import __future__
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import struct
import time

import numpy as np

from infernux.engine.path_utils import resolved_path
from infernux.engine.project_context import get_script_module_name
from infernux._compiler.source_metadata import compute_decorator_names, embed_compute_sources
from infernux._compiler.kernel_contract import (
    attribute_name,
    implicit_receiver_attribute,
    implicit_receiver_name,
    kernel_diagnostic,
    receiver_field_names,
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


def _gpu_buffer_descriptor(shape: tuple[int, ...], dtype):
    """Create a compiler-only GPU buffer descriptor without acquiring an RHI."""
    from infernux import compute

    value = object.__new__(compute.Buffer)
    value._dtype = compute._buffer_dtype(dtype)
    value._shape = shape
    value._device = "gpu"
    return value


def _engine_specializations():
    from infernux import compute
    from infernux.math import vector3

    domain = _gpu_buffer_descriptor((1,), np.int32)
    vector_values = _gpu_buffer_descriptor((1,), vector3)
    vertices = _gpu_buffer_descriptor((1, 25), np.float32)
    triangles = _gpu_buffer_descriptor((1, 3), np.int32)
    adjacency = _gpu_buffer_descriptor((1, 1), np.int32)
    counts = _gpu_buffer_descriptor((1,), np.int32)
    return (
        (compute._transform_anchor_points, (domain, vector_values, *([0.0] * 20))),
        (compute._transform_anchor_vectors, (domain, vector_values, *([0.0] * 14))),
        (
            compute._mesh_attribute_kernel,
            (domain, vertices, triangles, adjacency, counts, 1, 1, 1),
        ),
    )


def ensure_engine_compute_artifacts(project_root: str | Path) -> float:
    """Compile every engine-owned Player compute specialization into Library."""
    from infernux._compiler.taichi.frontend import compile_kernel
    from infernux.engine.project_context import using_project_root

    started = time.perf_counter()
    try:
        with using_project_root(project_root):
            for declaration, params in _engine_specializations():
                compile_kernel(declaration.function, params)
    except Exception as error:
        raise ComputeAotBuildError(
            f"Engine GPU AOT prewarm failed: {error}"
        ) from error
    return (time.perf_counter() - started) * 1000.0


def _attribute_name(node: ast.expr) -> str:
    return attribute_name(node)


def _compute_decorator_names(tree: ast.Module) -> set[str]:
    return compute_decorator_names(tree, kinds=("kernel",))


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
    declared_fields: frozenset[str] = frozenset(),
) -> None:
    """Reject implicit Python receivers before AOT artifact lookup.

    A class method using the conventional ``self``/``cls`` receiver is legal
    only when every receiver field is explicitly declared as an engine-owned
    scalar or ``inx.buffer``.  The runtime descriptor lowers those fields into
    the fixed GPU ABI; the AOT pass mirrors that contract from source so
    Android/Web diagnostics have the same identity and location as Editor
    compilation.  A ``@staticmethod`` declaration remains explicit and legal.
    """
    first = implicit_receiver_name(node, in_class=True)
    if first is not None:
        fields = receiver_field_names(node)
        if not fields:
            raise ComputeAotBuildError(_kernel_source_diagnostic(
                source_path,
                qualified,
                node,
                f"class-contained kernels cannot use an implicit instance receiver '{first}'",
                "put @staticmethod above @inx.compute.kernel (decorator order: @staticmethod, then @inx.compute.kernel) or move the kernel to module scope",
                target=target,
            ), missing=(qualified,))
        missing = tuple(name for name in fields if name not in declared_fields)
        if not missing:
            return
        missing_name = missing[0]
        if not declared_fields:
            access = implicit_receiver_attribute(node)
            receiver, attribute = access if access is not None else (first, node)
            raise ComputeAotBuildError(_kernel_source_diagnostic(
                source_path,
                qualified,
                node,
                f"class-contained kernels cannot access unbound receiver field '{receiver}.{getattr(attribute, 'attr', missing_name)}'",
                "pass the required scalar or inx.buffer explicitly, or declare an engine-owned receiver field",
                target=target,
                line=getattr(attribute, "lineno", None),
                column=(getattr(attribute, "col_offset", 0) + 1),
            ), missing=(qualified,))
        raise ComputeAotBuildError(_kernel_source_diagnostic(
            source_path,
            qualified,
            node,
            f"class-contained kernel receiver field '{first}.{missing_name}' is not declared as an engine-owned scalar or inx.buffer",
            "declare the field with a numeric annotation/class value, pass it explicitly, or put @staticmethod above @inx.compute.kernel",
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
    _source_snapshots: dict[Path, str] | None = None,
) -> tuple[str, ...]:
    """Return runtime-qualified kernels in the selected Python closure."""

    root = Path(resolved_path(Path(project_root).expanduser()))
    names: set[str] = set()
    for source_path in sorted(
        (Path(resolved_path(Path(path).expanduser())) for path in source_paths),
        key=lambda value: value.as_posix().casefold(),
    ):
        if source_path.suffix.casefold() != ".py":
            continue
        module_name = get_script_module_name(str(source_path), project_root=str(root))
        if not module_name:
            continue
        try:
            source = (_source_snapshots[source_path] if _source_snapshots is not None
                      else source_path.read_text(encoding="utf-8"))
            tree = ast.parse(source, filename=str(source_path))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ComputeAotBuildError(
                f"GPU AOT cannot inspect selected script: {source_path}: {error}"
            ) from error
        decorators = _compute_decorator_names(tree)

        class Collector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[tuple[str, bool, frozenset[str]]] = []

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                if any(
                    _attribute_name(item.func if isinstance(item, ast.Call) else item)
                    in decorators
                    for item in node.decorator_list
                ):
                    qualified: list[str] = []
                    for scope_name, is_function, _ in self.scope:
                        qualified.append(scope_name)
                        if is_function:
                            qualified.append("<locals>")
                    qualified.append(node.name)
                    runtime_name = f"{module_name}.{'.'.join(qualified)}"
                    classes = [fields for _, is_function, fields in self.scope if not is_function]
                    if classes:
                        _validate_class_kernel_source(
                            source_path,
                            runtime_name,
                            node,
                            target=target,
                            declared_fields=classes[-1],
                        )
                    names.add(runtime_name)
                self.scope.append((node.name, True, frozenset()))
                self.generic_visit(node)
                self.scope.pop()

            visit_FunctionDef = _visit_function
            visit_AsyncFunctionDef = _visit_function

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                fields: set[str] = set()
                for statement in node.body:
                    if isinstance(statement, (ast.AnnAssign, ast.Assign)):
                        targets = [statement.target] if isinstance(statement, ast.AnnAssign) else statement.targets
                        for target_node in targets:
                            if isinstance(target_node, ast.Name):
                                fields.add(target_node.id)
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for item in ast.walk(statement):
                            if (
                                isinstance(item, ast.Attribute)
                                and isinstance(item.value, ast.Name)
                                and item.value.id in {"self", "cls"}
                                and isinstance(item.ctx, ast.Store)
                            ):
                                fields.add(item.attr)
                self.scope.append((node.name, False, frozenset(fields)))
                self.generic_visit(node)
                self.scope.pop()

        Collector().visit(tree)
    return tuple(sorted(names))


def _artifact_record(path: Path) -> tuple[tuple[str, ...], str]:
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
        specialization = header.get("specialization")
        if not isinstance(specialization, str) or not specialization:
            raise ValueError("missing direct specialization identity; warm the kernel again")
        return functions, specialization
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ComputeAotBuildError(
            f"GPU AOT artifact is unreadable: {path}: {error}"
        ) from error


def _specialization_parameters(specialization: str) -> tuple:
    """Recreate ABI descriptors, never runtime buffers or guessed scalar values."""
    result = []
    from infernux.math import vector2, vector3, vector4

    dtypes = {"int": int, "float": float, "vector2": vector2, "vector3": vector3, "vector4": vector4}
    records = json.loads(specialization)
    if not isinstance(records, list):
        raise ValueError("invalid GPU specialization")
    for record in records:
        if record == ["int32"]:
            result.append(0)
        elif record == ["float32"]:
            result.append(0.0)
        elif (isinstance(record, list) and len(record) == 3 and record[0] == "buffer"
              and type(record[2]) is int and 1 <= record[2] <= 8):
            result.append(_gpu_buffer_descriptor((1,) * record[2], dtypes.get(record[1], record[1])))
        else:
            raise ValueError(f"invalid GPU specialization parameter: {record!r}")
    return tuple(result)


def _select_current_artifacts(root: Path, snapshots: dict[Path, str], expected: tuple[str, ...]):
    """Select only artifacts prepared from this build's current source closure.

    Candidate imports are private and never borrow the editor's published
    project modules. A same-named LKG function is not a valid build input.
    """
    from infernux._compiler.taichi import frontend
    from infernux.engine.candidate_import import CandidateImportTransaction
    from infernux.engine.project_context import using_project_root

    class BuildSourceImports(CandidateImportTransaction):
        def _reuse_project_lkg(self, name):
            return None

        def _has_lkg_descendant(self, name):
            return False

    cache = root / "Library/Artifacts/Compute"
    specializations: dict[str, set[str]] = {name: set() for name in expected}
    for path in sorted(cache.glob("*.inxgpu")):
        functions, specialization = _artifact_record(path)
        for name in set(functions).intersection(specializations):
            specializations[name].add(specialization)

    selected: dict[str, Path] = {}
    manifest = []
    missing: set[str] = set()

    def select(function, params, *, bindings=None):
        name = f"{function.__module__}.{function.__qualname__}"
        plan = frontend.prepare_kernel(function, params, receiver_fields=bindings)
        path = cache / f"{plan.artifact_key}.inxgpu"
        if not path.is_file():
            return False
        functions, specialization = _artifact_record(path)
        if name not in functions or specialization != frontend._specialization_identity(params):
            raise ComputeAotBuildError(f"GPU AOT artifact identity disagrees with its source: {path}")
        selected[path.name] = path
        manifest.append({"function": name, "specialization": specialization, "artifact": path.name})
        return True

    with using_project_root(root):
        broker = BuildSourceImports()
        try:
            modules = {}
            for path, source in snapshots.items():
                module_name = get_script_module_name(str(path), project_root=str(root))
                if not module_name:
                    continue
                broker.register(module_name, str(path), source=embed_compute_sources(source))
                modules[module_name] = path
            owners: dict[str, list[str]] = {}
            for name in expected:
                module_name = max((module for module in modules if name.startswith(module + ".")), key=len)
                owners.setdefault(module_name, []).append(name)
            for module_name, names in owners.items():
                path = modules[module_name]
                module = broker.load(module_name)
                sources = module.__dict__.get("__infernux_compute_sources__", {})
                for name in names:
                    qualified = name[len(module_name) + 1:]
                    if qualified not in sources:
                        raise ComputeAotBuildError(f"GPU AOT source is missing: {name}", missing=(name,))
                    definition = ast.parse(sources[qualified]).body[0]
                    if definition.args.defaults or any(value is not None for value in definition.args.kw_defaults):
                        raise ComputeAotBuildError(f"GPU kernel requires parameters without defaults: {name}")
                    fields = receiver_field_names(definition) if implicit_receiver_name(definition, in_class=True) else None
                    definition.decorator_list = []
                    namespace = dict(module.__dict__)
                    # Rebuild the declaration without instantiating components or
                    # calling factories. Closure captures cannot have warm artifacts.
                    tree = ast.Module(body=[definition], type_ignores=[])
                    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag,
                                 dont_inherit=True), namespace)
                    function = namespace[definition.name]
                    function.__qualname__ = qualified
                    matched = False
                    for specialization in sorted(specializations[name]):
                        params = _specialization_parameters(specialization)
                        bindings = tuple(zip(fields, params)) if fields is not None else None
                        expected_count = len(definition.args.posonlyargs) + len(definition.args.args)
                        if fields is not None:
                            expected_count += len(fields) - 1
                        # Historical arities do not describe the current declaration.
                        if len(params) != expected_count:
                            continue
                        matched = select(function, params, bindings=bindings) or matched
                    if not matched:
                        missing.add(name)
            for declaration, params in _engine_specializations():
                if not select(declaration.function, params):
                    missing.add(f"{declaration.function.__module__}.{declaration.function.__qualname__}")
        finally:
            broker.rollback()
    if missing:
        ordered = tuple(sorted(missing))
        raise ComputeAotBuildError(
            "GPU AOT is incomplete for the current source of the selected Player closure: "
            + ", ".join(ordered) + ". Prepare these kernels in the Editor before building.",
            missing=ordered,
        )
    return list(selected.values()), manifest


def stage_compute_artifacts(
    project_root: str | Path,
    source_paths: tuple[str | Path, ...],
    data_directory: str | Path,
    *,
    target: str = "Player/AOT",
    source_snapshots: dict[str, str] | None = None,
) -> ComputeAotResult:
    """Stage the exact selected kernel set plus engine compute primitives.

    The editor compiler cache is a build input, never a Player runtime cache.
    Artifact filenames already encode source and specialization identity; the
    Player recomputes that identity and opens the matching immutable file.
    """

    root = Path(resolved_path(Path(project_root).expanduser()))
    paths = {Path(resolved_path(Path(path).expanduser())) for path in source_paths
             if Path(path).suffix.casefold() == ".py"}
    if source_snapshots is None:
        snapshots = {path: path.read_text(encoding="utf-8") for path in sorted(paths)}
    else:
        snapshots = {Path(resolved_path(path)): source for path, source in source_snapshots.items()}
        if set(snapshots) != paths:
            raise ComputeAotBuildError("GPU AOT source snapshots disagree with the frozen Player closure")
    expected = declared_kernel_names(source_paths, root, target=target, _source_snapshots=snapshots)
    destination = (
        Path(resolved_path(Path(data_directory).expanduser()))
        / "Library"
        / "Artifacts"
        / "Compute"
    )
    if not expected:
        _clear_staged_gpu_artifacts(destination)
        return ComputeAotResult(0, 0)

    engine_elapsed_ms = ensure_engine_compute_artifacts(root)
    source = root / "Library" / "Artifacts" / "Compute"
    if not source.is_dir():
        raise ComputeAotBuildError(
            "GPU AOT artifacts are missing. Run the selected scenes once in the "
            "Editor so every @inx.compute.kernel specialization is prepared.",
            missing=expected,
        )

    selected, manifest_records = _select_current_artifacts(root, snapshots, expected)

    _clear_staged_gpu_artifacts(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in selected:
        shutil.copy2(artifact, destination / artifact.name)
    manifest_records.sort(key=lambda item: (item["function"], item["specialization"]))
    (destination / "AotManifest.json").write_text(
        json.dumps({"artifacts": manifest_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "AotOnly").write_text(
        "GPU kernels in this Player are immutable build artifacts.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "INFERNUX_BUILD_GPU_AOT_READY "
        f"target={target} project_kernels={len(expected)} artifacts={len(selected)} "
        f"engine_prewarm_ms={engine_elapsed_ms:.1f}"
    )
    return ComputeAotResult(len(selected), len(expected))


def _clear_staged_gpu_artifacts(destination: Path) -> None:
    """Replace GPU-owned files without deleting the sibling CPU JIT cache."""
    if not destination.is_dir():
        return
    for artifact in destination.glob("*.inxgpu"):
        artifact.unlink()
    for name in ("AotManifest.json", "AotOnly"):
        marker = destination / name
        if marker.exists():
            marker.unlink()
    if not any(destination.iterdir()):
        destination.rmdir()


__all__ = [
    "ComputeAotBuildError",
    "ComputeAotResult",
    "declared_kernel_names",
    "ensure_engine_compute_artifacts",
    "stage_compute_artifacts",
]
