"""Seal editor-compiled GPU kernels into source-less Player content."""

from __future__ import annotations

import ast
import __future__
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import time

import numpy as np
from infernux._source_kinds import is_host_python_source

from infernux.engine.path_utils import resolved_path
from infernux.engine.project_context import get_script_module_name
from infernux._compiler.source_metadata import compute_decorator_names, embed_compute_sources
from infernux._compiler.kernel_contract import (
    WORK_ITEM_FAST_MATH,
    attribute_name,
    implicit_receiver_attribute,
    implicit_receiver_name,
    kernel_diagnostic,
    receiver_field_names,
)


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


def ensure_engine_compute_artifacts(project_root: str | Path, *, target: str | None = None) -> float:
    """Compile every engine-owned Player compute specialization into Library."""
    from infernux._compiler.slang import compile_kernel
    from infernux.engine.project_context import using_project_root

    started = time.perf_counter()
    try:
        with using_project_root(project_root):
            for declaration, params in _engine_specializations():
                compile_kernel(declaration.function, params, **({"target": target} if target else {}))
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
    """Return runtime-qualified Python work items in the selected closure."""
    return _inspect_kernels(source_paths, project_root, target, _source_snapshots)[0]


def _inspect_kernels(source_paths, project_root, target, source_snapshots):
    root = Path(resolved_path(Path(project_root).expanduser()))
    names: set[str] = set()
    native_names: set[str] = set()
    for source_path in sorted(
        (Path(resolved_path(Path(path).expanduser())) for path in source_paths),
        key=lambda value: value.as_posix().casefold(),
    ):
        if not is_host_python_source(source_path):
            continue
        module_name = get_script_module_name(str(source_path), project_root=str(root))
        if not module_name:
            continue
        try:
            source = (source_snapshots[source_path] if source_snapshots is not None
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

            def _visit_assignment(self, node):
                call = node.value
                if (not isinstance(call, ast.Call) or _attribute_name(call.func) not in decorators
                        or any(is_function for _, is_function, _ in self.scope)
                        or not {"entry_point", "domain"}.issubset(keyword.arg for keyword in call.keywords)):
                    return
                targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
                scope = [module_name, *(name for name, _, _ in self.scope)]
                native_names.update(".".join((*scope, target.id)) for target in targets if isinstance(target, ast.Name))

            visit_Assign = _visit_assignment
            visit_AnnAssign = _visit_assignment

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
    return tuple(sorted(names)), tuple(sorted(native_names))


def _artifact_record(path: Path):
    from infernux._compiler.gpu_artifact import _read_artifact

    try:
        result = _read_artifact(path)
        if result is None:
            raise FileNotFoundError(path)
        header, artifact = result
        functions = tuple(
            str(item["function"])
            for item in artifact.diagnostic_locations
            if item.get("role", "entry") == "entry" and item.get("function")
        )
        if not functions:
            raise ValueError("missing kernel identity")
        return functions, header["specialization"], artifact
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, RuntimeError) as error:
        raise ComputeAotBuildError(
            f"GPU AOT artifact is unreadable: {path}: {error}"
        ) from error


def _specialization_parameters(specialization: str) -> tuple:
    """Recreate ABI descriptors, never runtime buffers or guessed scalar values."""
    result = []
    from infernux.math import vector2, vector3, vector4
    from infernux._compiler.gpu_artifact import _compilation_request

    dtypes = {"int": int, "float": float, "vector2": vector2, "vector3": vector3, "vector4": vector4}
    identity = json.loads(specialization)
    if not isinstance(identity, dict) or set(identity) != {"parameters", "autodiff_mode", "gradient_parameters", "fast_math", "target"}:
        raise ValueError("invalid GPU specialization identity")
    if type(identity["fast_math"]) is not bool or identity["fast_math"] is not WORK_ITEM_FAST_MATH:
        raise ValueError("GPU specialization changed the fixed WorkItem floating-point policy")
    records = identity["parameters"]
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
    params = tuple(result)
    gradients = _compilation_request(params, identity["autodiff_mode"], identity["gradient_parameters"])
    return params, {"target": identity["target"], "autodiff_mode": identity["autodiff_mode"], "gradient_parameters": gradients}


def _select_current_artifacts(root, data, snapshots, expected, native_names, *, shader_target):
    """Select only artifacts prepared from this build's current source closure.

    Candidate imports are private and never borrow the editor's published
    project modules. A same-named LKG function is not a valid build input.
    """
    from infernux import compute
    from infernux._compiler import slang as frontend
    from infernux._compiler.gpu_artifact import (
        ArtifactAbiMismatchError, _encode_artifact, _specialization_identity, parameter_key,
    )
    from infernux._compiler.work_item import _parse_function_source, capture_function_source
    from infernux.engine.candidate_import import CandidateImportTransaction
    from infernux.components._component_registration import candidate_component_registration_scope
    from infernux.engine.project_context import using_project_root

    class BuildSourceImports(CandidateImportTransaction):
        def _reuse_project_lkg(self, name):
            return None

        def _has_lkg_descendant(self, name):
            return False

    cache = root / "Library/Artifacts/Compute"
    specializations: dict[str, set[str]] = {name: set() for name in expected}
    native_specializations: dict[tuple[str, str], set[str]] = {}
    for path in sorted(cache.glob("*.inxgpu")):
        try:
            functions, specialization, artifact = _artifact_record(path)
        except ComputeAotBuildError as error:
            # Historical ABIs cannot describe this build's current closure.
            # Missing contracts/corrupt current artifacts still fail strictly.
            if isinstance(error.__cause__, ArtifactAbiMismatchError):
                continue
            raise
        if artifact.native_dispatch:
            entry = next(item for item in artifact.diagnostic_locations if item["role"] == "entry")
            native_specializations.setdefault((entry["path"], entry["entry_point"]), set()).add(specialization)
        elif shader_target is None or artifact.target == shader_target:
            for name in set(functions).intersection(specializations):
                specializations[name].add(specialization)

    selected: dict[str, Path | bytes] = {}
    manifest = []
    missing: set[str] = set()
    declarations = set(expected)

    def publish(name, specialization, filename, payload):
        selected[filename] = payload
        manifest.append({"function": name, "specialization": specialization, "artifact": filename})

    def select(function, params, *, target="spirv", bindings=None, autodiff_mode="none", gradient_parameters=()):
        name = f"{function.__module__}.{function.__qualname__}"
        request = {"target": target, "autodiff_mode": autodiff_mode, "gradient_parameters": gradient_parameters}
        plan = frontend.prepare_kernel(function, params, receiver_fields=bindings, **request)
        path = cache / f"{plan.artifact_key}.inxgpu"
        if not path.is_file():
            return False
        functions, specialization, _ = _artifact_record(path)
        if name not in functions or specialization != _specialization_identity(params, **request):
            raise ComputeAotBuildError(f"GPU AOT artifact identity disagrees with its source: {path}")
        publish(name, specialization, path.name, path)
        return True

    def select_native(name, declaration):
        source_args = declaration._slang_source
        candidates = native_specializations.get(source_args[:2])
        if not candidates:
            return False
        target = shader_target or "spirv"

        def read_source(asset):
            return (data / asset).read_text(encoding="utf-8-sig")

        try:
            source = frontend._read_slang_kernel_source(*source_args, read_source=read_source)
            for specialization in sorted(candidates, key=lambda value: (json.loads(value)["target"] != target, value)):
                key = source.artifact_key(specialization)
                warmed = cache / f"{key}.inxgpu"
                if not warmed.is_file():
                    continue
                params, request = _specialization_parameters(specialization)
                _, recorded, artifact = _artifact_record(warmed)
                if recorded != specialization or artifact.identity != key or not artifact.native_dispatch:
                    raise ValueError(f"cached Slang artifact identity disagrees with its source: {warmed}")
                layout = tuple(("buffer", entry["dtype"], 1) for entry in artifact.argument_layout["parameters"])
                if parameter_key(params) != layout:
                    continue  # A failed editor binding may have compiled a cache entry before validation.

                target_specialization = _specialization_identity(params, target=target)
                target_key = source.artifact_key(target_specialization)
                filename = f"{target_key}.inxgpu"
                payload = selected.get(filename)
                if payload is None:
                    if request["target"] == target:
                        payload = warmed
                    else:
                        target_artifact = source.compile(target, target_key)
                        if target_artifact.argument_layout != artifact.argument_layout:
                            raise ValueError(f"{target} bindings disagree with the prepared Slang buffer signature")
                        payload = _encode_artifact(target_artifact, target_specialization)
                publish(name, target_specialization, filename, payload)
                return True
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            raise ComputeAotBuildError(f"GPU AOT cannot prepare Slang declaration {name}: {error}") from error
        return False

    with using_project_root(root), candidate_component_registration_scope():
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
            for name in sorted(declarations | set(native_names)):
                module_name = max((module for module in modules if name.startswith(module + ".")), key=len)
                owners.setdefault(module_name, []).append(name)
            for module_name, names in owners.items():
                path = modules[module_name]
                module = broker.load(module_name)
                sources = {name: record for (name, _), record in
                           module.__dict__.get("__infernux_compute_sources__", {}).items()}
                for name in names:
                    qualified = name[len(module_name) + 1:]
                    if name in native_names:
                        declaration = module
                        for attribute in qualified.split("."):
                            declaration = getattr(declaration, attribute, None)
                        if isinstance(declaration, compute.Kernel) and declaration._slang_source is not None:
                            declarations.add(name)
                            if not select_native(name, declaration):
                                missing.add(name)
                            continue
                        if name not in declarations:
                            continue
                    record = sources.get(qualified)
                    if record is None:
                        raise ComputeAotBuildError(f"GPU AOT source is missing: {name}", missing=(name,))
                    definition = _parse_function_source(record[0])
                    if definition.args.defaults or any(value is not None for value in definition.args.kw_defaults):
                        raise ComputeAotBuildError(f"GPU kernel requires parameters without defaults: {name}")
                    fields = receiver_field_names(definition) if implicit_receiver_name(definition, in_class=True) else None
                    definition.decorator_list = []
                    namespace = dict(module.__dict__)
                    namespace.pop("__infernux_compute_sources__", None)
                    # Rebuild the declaration without instantiating components or
                    # calling factories. Closure captures cannot have warm artifacts.
                    tree = ast.Module(body=[definition], type_ignores=[])
                    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag,
                                 dont_inherit=True), namespace)
                    function = namespace[definition.name]
                    function.__qualname__ = qualified
                    capture_function_source(function, record=record)
                    matched = False
                    for specialization in sorted(specializations[name]):
                        params, request = _specialization_parameters(specialization)
                        bindings = tuple(zip(fields, params)) if fields is not None else None
                        expected_count = len(definition.args.posonlyargs) + len(definition.args.args)
                        if fields is not None:
                            expected_count += len(fields) - 1
                        # Historical arities do not describe the current declaration.
                        if len(params) != expected_count:
                            continue
                        selected_request = select(function, params, bindings=bindings, **request)
                        # Public @compute.kernel launches the default primal.
                        # Additional AD requests cannot satisfy that closure.
                        if request["autodiff_mode"] == "none" and not request["gradient_parameters"]:
                            matched = selected_request or matched
                    if not matched:
                        missing.add(name)
            for declaration, params in _engine_specializations():
                if not select(declaration.function, params, target=shader_target or "spirv"):
                    missing.add(f"{declaration.function.__module__}.{declaration.function.__qualname__}")
        finally:
            broker.rollback()
    return selected, manifest, tuple(sorted(missing)), len(declarations)


def stage_compute_artifacts(
    project_root: str | Path,
    source_paths: tuple[str | Path, ...],
    data_directory: str | Path,
    *,
    target: str = "Player/AOT",
    source_snapshots: dict[str, str] | None = None,
    resource_root: str | Path | None = None,
) -> ComputeAotResult:
    """Stage matching prewarmed kernels plus engine compute primitives.

    A module or class-level Slang declaration takes its signature from an editor
    prepare and its source closure from the staged assets. The collector can
    compile that exact specialization for the Player's target. The Player also
    retains the source and resident compiler for new legal configurations.
    """

    root = Path(resolved_path(Path(project_root).expanduser()))
    paths = {Path(resolved_path(Path(path).expanduser())) for path in source_paths
             if is_host_python_source(path)}
    if source_snapshots is None:
        snapshots = {path: path.read_text(encoding="utf-8") for path in sorted(paths)}
    else:
        snapshots = {Path(resolved_path(path)): source for path, source in source_snapshots.items()}
        if set(snapshots) != paths:
            raise ComputeAotBuildError("GPU AOT source snapshots disagree with the frozen Player closure")
    expected, native_names = _inspect_kernels(source_paths, root, target, snapshots)
    data = Path(resolved_path(Path(data_directory).expanduser()))
    destination = data / "Library" / "Artifacts" / "Compute"
    if not expected and not native_names:
        _clear_staged_gpu_artifacts(destination)
        return ComputeAotResult(0, 0)

    shader_target = "wgsl" if target == "Web/Player" else None
    engine_elapsed_ms = ensure_engine_compute_artifacts(root, target=shader_target)
    from infernux._compiler.shader_library import using_cooked_shader_sources
    from infernux.lib import _Infernux
    from infernux.resources import resources_path

    resources = Path(resources_path) if resource_root is None else Path(resource_root)
    catalog = data / "Library" / "RuntimeAssetRecords.json"
    shader_paths = {}
    if resource_root is not None:
        from infernux.engine.build.shader_programs import _cooked_shader_sources

        _, _, shader_paths, _ = _cooked_shader_sources(data, catalog, resources)
    with using_cooked_shader_sources(resources, shader_paths, ray_surfaces=lambda: _Infernux._shader_project_surface_sources(
            str(data), str(resources), str(catalog))):
        selected, manifest_records, runtime_kernels, kernel_count = _select_current_artifacts(
            root, data, snapshots, expected, native_names, shader_target=shader_target,
        )

    _clear_staged_gpu_artifacts(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, payload in selected.items():
        if isinstance(payload, Path):
            shutil.copy2(payload, destination / filename)
        else:
            (destination / filename).write_bytes(payload)
    manifest_records.sort(key=lambda item: (item["function"], item["specialization"]))
    (destination / "AotManifest.json").write_text(
        json.dumps({"artifacts": manifest_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "INFERNUX_BUILD_GPU_AOT_READY "
        f"target={target} project_kernels={kernel_count} artifacts={len(selected)} "
        f"runtime_kernels={len(runtime_kernels)} engine_prewarm_ms={engine_elapsed_ms:.1f}"
    )
    return ComputeAotResult(len(selected), kernel_count)


def _clear_staged_gpu_artifacts(destination: Path) -> None:
    """Replace GPU-owned files without deleting the sibling CPU JIT cache."""
    if not destination.is_dir():
        return
    for artifact in destination.glob("*.inxgpu"):
        artifact.unlink()
    (destination / "AotManifest.json").unlink(missing_ok=True)
    if not any(destination.iterdir()):
        destination.rmdir()


__all__ = [
    "ComputeAotBuildError",
    "ComputeAotResult",
    "declared_kernel_names",
    "ensure_engine_compute_artifacts",
    "stage_compute_artifacts",
]
