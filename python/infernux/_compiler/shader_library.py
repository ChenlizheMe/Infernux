"""Static ShaderLib imports and typed functions shared by GPU author domains."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path

from infernux.shader._language import INTRINSICS, TYPES
from .slang_ast import SlangEmitter, qualified
from .shader_types import (
    BUFFER_ELEMENTS, BUFFER_TYPES, MULTISAMPLED_TYPES, RESOURCE_DIMENSIONS,
    RESOURCE_ELEMENTS, RESOURCE_MODULE, RESOURCE_TYPES, STRUCT_FIELDS, STRUCT_TYPES, SURFACE_FIELDS,
)
from .diagnostics import parse_gpu_module


_RAY_ARGUMENTS = ("float3", "float3", "float", "float", "uint", "float")
_RAY_FUNCTIONS = {"ray.trace_closest": ("inx_world_ray_closest", "InxRaySurfaceHit", _RAY_ARGUMENTS),
                  "ray.trace_visibility": ("inx_world_ray_visibility", "bool", _RAY_ARGUMENTS),
                  "ray._thin_reflection": ("inx_ray_thin_reflection", "float4", (
                      "float3", "float3", "InxBuffer_uint4", "float2", "InxBuffer_uint4")),
                  "ray._radiance_parts": ("inx_ray_radiance_parts", "float4", (
                      "float3", "float3", "InxBuffer_uint4", "uint", "int", "float")),
                  "ray._tail_pixel": ("inx_ray_tail_pixel", "uint2", ("uint", "uint2")),
                  "ray._tail_request": ("inx_ray_tail_primary_request", "InxRayTailRequest", (
                      "InxSurfaceData", "InxShadingContext", "float3", "uint", "bool", "bool")),
                  "ray._tail_thin_request": ("inx_ray_tail_thin_request", "InxRayTailRequest", ("float3", "float3")),
                  "ray._tail_estimate": ("inx_ray_tail_estimate", "float4", (
                      "InxBuffer_uint4", "InxBuffer_uint4", "InxBuffer_uint4", "uint")),
                  "ray._indirect_signal": ("inx_ray_indirect_signal", "InxRaySignal", (
                      "InxSurfaceData", "InxShadingContext", "float3", "InxBuffer_uint4", "uint", "bool", "bool", "InxBuffer_uint4"))}
_ENGINE_FUNCTIONS = {"frame_time": ("inx_frame_time", "float", ()),
                     "frame_delta_time": ("inx_frame_delta_time", "float", ()),
                     "_previous_ray_geometry_serial": ("inx_previous_ray_geometry_serial", "uint", ()),
                     "view": ("inx_view_context", "InxViewContext", ()),
                     "camera": ("inx_camera_context", "InxCameraContext", ()), **_RAY_FUNCTIONS}
_ENGINE_FUNCTIONS.update({
    "lighting._shade_ray": ("inx_evaluate_material_ray", "float3", (
        "InxSurfaceData", "InxShadingContext", "uint", "float4", "float4")),
    "lighting.particle_pbr": ("inx_particle_pbr", "float3", ("InxSurfaceData", "InxParticleInput")),
    "lighting.six_way_smoke": ("inx_particle_six_way_smoke", "float3", (
        "InxParticleInput", "float3", "float3", "float3", "float", "float", "float", "float", "float")),
})
_ENGINE_MODULES = {"inx_ui_context": ("UIContext", "WorldUIContext"),
                   "inx_particle": ("ParticleContext",),
                   "inx_particle_lighting": ("lighting.particle_pbr", "lighting.six_way_smoke"),
                   "inx_frame": ("frame_time", "frame_delta_time", "_previous_ray_geometry_serial", "view", "ViewContext"),
                   "inx_camera": ("camera", "CameraContext")}
_CONSTRUCTORS = {"ShadingContext": ("InxShadingContext", "inx_shading_context"),
                 "FragmentOutput": ("InxFragmentOutput", "inx_fragment_output"),
                 "ThinSheetData": ("InxThinSheetData", "inx_thin_sheet_data")}


def engine_imports(body, alias, *, graph, owner):
    names = {
        qualified(node): node
        for declaration in body for node in ast.walk(declaration)
        if isinstance(node, ast.Attribute)
    }
    modules = [module for module, exports in _ENGINE_MODULES.items()
               if any(alias + "." + name in names for name in exports)]
    ray_use = next((names[alias + "." + name] for name in _RAY_FUNCTIONS
                    if alias + "." + name in names), None)
    if ray_use is not None:
        from .shader import _world_ray_module

        modules.append(_world_ray_module(graph, owner, ray_use).name)
    if any(alias + "." + name in names for name in ("lighting.shade", "lighting._shade_ray")):
        from .shading_model import surface_lighting_module

        modules.append(surface_lighting_module(graph, owner, body[0]).name)
    return tuple("import " + module + ";" for module in modules)


def uses_surface_lighting(body, alias, functions):
    return any(function.lighting_specialization for function in functions.values()) or any(
        isinstance(node, ast.Call) and qualified(node.func) in {
            alias + ".lighting.shade", alias + ".lighting._shade_ray", alias + ".lighting.particle_pbr",
            alias + ".lighting.six_way_smoke"}
        for declaration in body for node in ast.walk(declaration)
    )


@dataclass(frozen=True)
class Function:
    symbol: str
    result: str
    arguments: tuple[str, ...]
    lighting_specialization: bool = False
    effects: tuple[str, ...] = ()
    source_signature: str = ""
    documentation: str = ""


_FRAGMENT_EFFECTS = {"ddx": "derivatives", "ddy": "derivatives", "fwidth": "derivatives",
                     "sample": "implicit_lod", "sample_bias": "implicit_lod", "clip": "discard"}


def typed_function_call(owner, node, function, *, lighting_mode="false"):
    if node.keywords or len(node.args) != len(function.arguments):
        owner.error(node, "Shader helper arguments must match its typed positional signature")
    specialization = "<" + lighting_mode + ">" if function.lighting_specialization else ""
    return function.symbol + specialization + "(" + ", ".join(owner.visit(arg) for arg in node.args) + ")"


def _function_calls(definition):
    return tuple(qualified(node.func) for node in ast.walk(definition) if isinstance(node, ast.Call))


def function_effects(definitions, functions, owner, *, method=False):
    """Propagate effects through local and already-resolved imported calls."""
    definitions = {("self." if method else "") + definition.name: definition for definition in definitions}
    calls = {name: _function_calls(definition) for name, definition in definitions.items()}
    direct = {name: {_FRAGMENT_EFFECTS[owner.symbol(call)] for call in targets
                     if owner.symbol(call) != call and owner.symbol(call) in _FRAGMENT_EFFECTS}
              for name, targets in calls.items()}
    effects = direct.copy()
    while True:
        propagated = {name: direct[name].union(*(effects[call] if call in effects else functions[call].effects
                                                for call in targets if call in functions))
                      for name, targets in calls.items()}
        if propagated == effects:
            break
        effects = propagated
    return {name: replace(function, effects=tuple(sorted(effects[name]))) if name in effects else function
            for name, function in functions.items()}


@dataclass
class Module:
    name: str
    path: str
    source: str
    functions: dict[str, Function]
    source_ranges: tuple = ()


@dataclass(slots=True)
class _LibrarySource:
    module: Module
    text: str
    alias: str
    definitions: list[ast.FunctionDef]
    functions: dict[str, Function]
    header: list[str]

    def emit(self, emitter_type, *, differentiable=()):
        code = list(self.header)
        ranges = []
        for definition in self.definitions:
            function = self.module.functions[definition.name]
            emitter = emitter_type(self.text, self.module.path, aliases=(self.alias,), functions=self.functions)
            code.extend(emitter.emit_function(definition, function, public=True,
                                              differentiable=function.symbol in differentiable))
            ranges.extend(emitter.source_ranges)
        code.append("}")
        self.module.source = "\n".join(code)
        self.module.source_ranges = tuple(ranges)


class ShaderEmitter(SlangEmitter):
    lighting_mode = "false"
    stage = None

    def __init__(self, source, path, *, aliases=("shader",), functions=None, types=None):
        super().__init__(source, path, aliases=aliases, types=types)
        self.functions = dict(functions or {})

    def visit_Attribute(self, node):
        if self.infer(node.value) in STRUCT_FIELDS:
            self.infer(node)
        return super().visit_Attribute(node)

    def infer(self, node):
        if isinstance(node, ast.Subscript):
            element = BUFFER_ELEMENTS.get(self.infer(node.value))
            if element is not None:
                return element
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"
                and len(node.args) == 1 and self.infer(node.args[0]) in BUFFER_ELEMENTS):
            return "uint"
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) != qualified(node.func)
                and self.symbol(qualified(node.func)) == "lighting.shade"):
            return "float3"
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) != qualified(node.func)
                and self.symbol(qualified(node.func)) in _CONSTRUCTORS):
            return _CONSTRUCTORS[self.symbol(qualified(node.func))][0]
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) != qualified(node.func)
                and self.symbol(qualified(node.func)) in _ENGINE_FUNCTIONS):
            return _ENGINE_FUNCTIONS[self.symbol(qualified(node.func))][1]
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) == "SurfaceData"
                and self.symbol(qualified(node.func)) != qualified(node.func)):
            return "InxSurfaceData"
        if isinstance(node, ast.Attribute):
            fields = STRUCT_FIELDS.get(self.infer(node.value))
            if fields is not None:
                if node.attr not in fields:
                    self.error(node, "Unknown shader stage input: " + node.attr, code="author.type")
                return fields[node.attr]
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) != qualified(node.func)
                and self.symbol(qualified(node.func)) in {"sample", "sample_level", "sample_bias", "sample_grad", "load"}):
            if not node.args:
                self.error(node, "A texture operation needs a declared GPU image", code="author.type")
            return RESOURCE_ELEMENTS.get(self.infer(node.args[0]))
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) != qualified(node.func)
                and self.symbol(qualified(node.func)) == "dimensions"):
            if not node.args or (kind := self.infer(node.args[0])) not in RESOURCE_DIMENSIONS:
                self.error(node, "dimensions receives a declared GPU image", code="author.type")
            return "uint" + str(RESOURCE_DIMENSIONS[kind])
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) == "sample_count"
                and self.symbol(qualified(node.func)) != qualified(node.func)):
            return "uint"
        if isinstance(node, ast.Call) and (function := self.functions.get(qualified(node.func))):
            return function.result
        return super().infer(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1:
            if node.keywords or self.infer(node.args[0]) not in BUFFER_ELEMENTS:
                self.error(node, "len receives a typed GPU Buffer", code="author.type")
            return "inx_buffer_length(" + self.visit(node.args[0]) + ")"
        operation = self.symbol(qualified(node.func))
        intrinsic = operation != qualified(node.func)
        effects = ((_FRAGMENT_EFFECTS[operation],) if intrinsic and operation in _FRAGMENT_EFFECTS else
                   self.functions[qualified(node.func)].effects if qualified(node.func) in self.functions else ())
        if self.stage not in {None, "fragment"} and effects:
            self.error(node, qualified(node.func) + " requires the fragment stage (" + ", ".join(effects) + ")",
                       code="author.stage")
        if intrinsic and operation == "vec4" and self.stage != "compute":
            # Each argument is evaluated once. Scalar lanes remain separable
            # when an auxiliary pass needs alpha but not color.
            value = super().visit_Call(node)
            return "inx_make_float4" + value[len("float4"):]
        if intrinsic and operation == "lighting.shade":
            expected = ("InxSurfaceData", "InxShadingContext", "uint")
            if node.keywords or len(node.args) != len(expected):
                self.error(node, "lighting.shade takes a SurfaceData, ShadingContext and uint model ID", code="author.type")
            for value, kind in zip(node.args, expected):
                if self.infer(value) != kind:
                    self.error(value, "Lighting argument requires " + kind, code="author.type")
            return "inx_evaluate_material<" + self.lighting_mode + ">(" + ", ".join(self.visit(value) for value in node.args) + ")"
        if intrinsic and operation in _CONSTRUCTORS:
            kind, constructor = _CONSTRUCTORS[operation]
            fields = STRUCT_FIELDS[kind]
            values = {keyword.arg: keyword.value for keyword in node.keywords}
            if node.args or values.keys() != fields.keys() or len(values) != len(node.keywords):
                self.error(node, operation + " declares named fields: " + ", ".join(fields), code="author.type")
            for name, kind in fields.items():
                if self.infer(values[name]) != kind:
                    self.error(values[name], operation + "." + name + " requires " + kind, code="author.type")
            return constructor + "(" + ", ".join(self.visit(values[name]) for name in fields) + ")"
        if intrinsic and operation == "clip":
            if node.keywords or len(node.args) != 1:
                self.error(node, "shader.clip takes one signed clipping distance", code="author.type")
            return "clip(" + self.visit(node.args[0]) + ")"
        if operation in _ENGINE_FUNCTIONS and operation != qualified(node.func):
            symbol, _, arguments = _ENGINE_FUNCTIONS[operation]
            if node.keywords or len(node.args) != len(arguments):
                count = str(len(arguments)) if arguments else "no"
                self.error(node, "shader." + operation + " takes " + count + " arguments", code="author.type")
            for value, kind in zip(node.args, arguments):
                if self.infer(value) != kind:
                    self.error(value, "shader." + operation + " requires " + kind, code="author.type")
            specialization = "<" + self.lighting_mode + ">" if operation in {
                "lighting._shade_ray", "lighting.particle_pbr", "lighting.six_way_smoke"} else ""
            return symbol + specialization + "(" + ", ".join(self.visit(value) for value in node.args) + ")"
        if operation == "SurfaceData" and operation != qualified(node.func):
            values = {keyword.arg: keyword.value for keyword in node.keywords}
            if node.args or values.keys() - SURFACE_FIELDS.keys() or len(values) != len(node.keywords):
                self.error(node, "SurfaceData uses named surface fields", code="author.type")
            fields = (self.visit(values[name]) if name in values else default
                      for name, (_, default) in SURFACE_FIELDS.items())
            return "inx_surface_data(" + ", ".join(fields) + ")"
        if intrinsic and operation == "sample_count":
            if node.keywords or len(node.args) != 1 or self.infer(node.args[0]) not in MULTISAMPLED_TYPES:
                self.error(node, "sample_count receives a multisampled image", code="author.type")
            return "inx_image_samples(" + self.visit(node.args[0]) + ")"
        image_operations = {"sample": (2, "Sample"), "sample_level": (3, "SampleLevel"), "sample_bias": (3, "SampleBias"),
                            "sample_grad": (4, "SampleGrad"), "load": (3, "Load"), "dimensions": (2, "GetDimensions")}
        if operation in image_operations and operation != qualified(node.func):
            arity, intrinsic = image_operations[operation]
            kind = self.infer(node.args[0]) if node.args else None
            if operation == "dimensions" and kind in MULTISAMPLED_TYPES:
                arity = 1
            if node.keywords or len(node.args) != arity:
                self.error(node, f"shader.{operation} takes {arity} typed positional arguments")
            if kind not in RESOURCE_ELEMENTS:
                self.error(node, "Texture operations receive a declared GPU image")
            if (kind == "InxUInt2D" or kind in MULTISAMPLED_TYPES) and operation not in {"load", "dimensions"}:
                self.error(node, "Integer and multisampled images use shader.load with pixel coordinates", code="author.type")
            image, *arguments = (self.visit(argument) for argument in node.args)
            if operation == "dimensions":
                if kind in MULTISAMPLED_TYPES:
                    return f"inx_image_dimensions({image})"
                return f"inx_image_dimensions({image}, uint({arguments[0]}))"
            if operation == "load":
                if kind in MULTISAMPLED_TYPES:
                    return f"{image}.texture.Load(int2({arguments[0]}), int({arguments[1]}))"
                if RESOURCE_DIMENSIONS[kind] == 3:
                    return f"{image}.texture.Load(int4({arguments[0]}, {arguments[1]}))"
                return f"{image}.texture.Load(int3({arguments[0]}, {arguments[1]}))"
            return image + ".texture." + intrinsic + "(" + ", ".join([image + ".sampler", *arguments]) + ")"
        function = self.functions.get(qualified(node.func))
        if function is None:
            return super().visit_Call(node)
        return typed_function_call(self, node, function, lighting_mode=self.lighting_mode)

    def annotation(self, node):
        if isinstance(node, ast.Subscript):
            if qualified(node.value) not in {alias + ".Buffer" for alias in self.aliases}:
                self.error(node, "GPU buffer types use shader.Buffer[element]", code="author.type")
            element = self.annotation(node.slice)
            if element not in BUFFER_TYPES:
                self.error(node, "Buffer elements use 32-bit float, int, uint or their vectors", code="author.type")
            return BUFFER_TYPES[element]
        spelling = qualified(node)
        name = spelling.rsplit(".", 1)[-1]
        if spelling not in {"float", "int", "bool"} and not any(spelling.startswith(alias + ".") for alias in self.aliases):
            self.error(node, "GPU value types come from infernux.shader", code="author.type")
        if name in STRUCT_TYPES:
            return STRUCT_TYPES[name]
        return RESOURCE_TYPES[name] if name in RESOURCE_TYPES else super().annotation(node)

    def visit_Subscript(self, node):
        if self.infer(node.value) in BUFFER_ELEMENTS:
            if isinstance(node.ctx, ast.Store):
                self.error(node, "Shader Buffer inputs are read-only; update them in Compute", code="author.resource")
            if self.infer(node.slice) not in {"int", "uint"}:
                self.error(node.slice, "Buffer indices use int or uint", code="author.type")
            return "inx_buffer_load(" + self.visit(node.value) + ", uint(" + self.visit(node.slice) + "))"
        return super().visit_Subscript(node)

    def signature(self, definition, symbol, *, method=False):
        arguments = definition.args
        if (definition.decorator_list or arguments.vararg or arguments.kwarg
                or arguments.kwonlyargs or arguments.defaults or arguments.posonlyargs):
            self.error(definition, "GPU helpers use a fixed typed positional signature")
        fields = arguments.args
        if method:
            if not fields or fields[0].arg != "self":
                self.error(definition, "Shader helpers are instance methods")
            fields = fields[1:]
        for field in fields:
            if field.annotation is None:
                self.error(field, "GPU helper parameters need type annotations")
        if definition.returns is None:
            self.error(definition, "GPU helpers need a return type annotation")
        signature = definition.name + "(" + ", ".join(field.arg + ": " + ast.unparse(field.annotation) for field in fields)
        signature += ") -> " + ast.unparse(definition.returns)
        return Function(symbol, self.annotation(definition.returns),
                        tuple(self.annotation(field.annotation) for field in fields),
                        source_signature=signature, documentation=ast.get_docstring(definition) or "")

    def emit_function(self, definition, function, *, method=False, public=False, differentiable=False):
        if function.lighting_specialization:
            self.lighting_mode = "tiled_lighting"
        fields = definition.args.args[1:] if method else definition.args.args
        self.types.update((field.arg, kind) for field, kind in zip(fields, function.arguments))
        self.declare_locals(definition.body, parameters=(field.arg for field in fields))
        self.local_declarations()
        self.block(definition.body)
        parameters = ", ".join(kind + " " + field.arg for field, kind in zip(fields, function.arguments))
        specialization = "<let tiled_lighting : bool>" if function.lighting_specialization else ""
        derivative = ["[Differentiable]"] if differentiable else []
        inline = ["[ForceInline]"] if function.result in {"float4", "InxFragmentOutput"} else []
        return [*derivative, self.source_directive(definition), *inline,
                f'{"public " if public else ""}{function.result} {definition.name}{specialization}({parameters})',
                "{", *self.lines, "}"]


class _ComputeLibraryEmitter(ShaderEmitter):
    stage = "compute"

    def annotation(self, node):
        kind = super().annotation(node)
        if kind not in TYPES.values():
            self.error(node, "Compute ShaderLibs use scalar, vector or matrix value types", code="author.type")
        return kind

    def visit_Call(self, node):
        name = qualified(node.func)
        operation = self.symbol(name)
        if (operation != name and operation not in TYPES and operation not in INTRINSICS
                and operation not in _FRAGMENT_EFFECTS):
            self.error(node, "Compute ShaderLibs receive their inputs as values; cannot use " + name,
                       code="author.stage")
        return super().visit_Call(node)


def declarations(source, path):
    """Separate GPU declarations without importing or executing their Python module."""
    emitter = ShaderEmitter(source, path)
    imports, body, aliases = [], [], []
    for statement in parse_gpu_module(source, path).body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name != "infernux.shader":
                    emitter.error(statement, "GPU files import infernux.shader and static ShaderLib declarations; use GPU value types instead of NumPy or host modules",
                                  code="author.import")
                aliases.append(alias.asname or alias.name)
        elif isinstance(statement, ast.ImportFrom):
            imports.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        elif isinstance(statement, (ast.ClassDef, ast.FunctionDef)):
            body.append(statement)
        else:
            emitter.error(statement, "GPU files contain static declarations; host statements are not executed")
    if len(aliases) != 1:
        emitter.error(ast.Constant(lineno=1, col_offset=0), "Import infernux.shader once in each GPU source")
    return aliases[0], imports, body


class LibraryGraph:
    def __init__(self, reader=None, *, compute_autodiff=None, allow_world_rays=False):
        self.modules: dict[Path, Module] = {}
        self.ordered: list[Module] = []
        self.loading: set[Path] = set()
        self.lighting_registry: Module | None = None
        self.ray_lit: Module | None = None
        self.allow_world_rays = allow_world_rays
        self.reader = reader or (lambda path: Path(path).read_text(encoding="utf-8-sig"))
        self.compute_autodiff = compute_autodiff
        self._derivative_sources: list[_LibrarySource] = []

    def reference(self, call, owner, *, stage, suffix):
        if (not isinstance(call, ast.Call) or owner.qualified(call.func) != owner.alias + ".ref"
                or call.keywords or len(call.args) != 1):
            owner.error(call, "GPU stages refer to one asset with shader.ref", code="author.reference")
        reference = owner.literal(call.args[0])
        if not isinstance(reference, str):
            owner.error(call, "A GPU asset reference uses a registered ID or " + suffix + " path", code="author.reference")
        if reference.endswith(suffix):
            return (Path(owner.path).parent / reference).resolve()
        from infernux.core._shader_catalog import get_shader_file_path

        try:
            resolved = get_shader_file_path(reference, suffix[:-3])
        except (RuntimeError, ValueError) as error:
            owner.error(call, str(error), code="author.reference")
        if not resolved:
            owner.error(call, "GPU asset ID is not registered: " + reference, code="author.reference")
        return Path(resolved)

    def _import_base(self, module, level, owner, location, *, declaring_path=None):
        if level:
            base = Path(declaring_path or owner.path).parent
            for _ in range(level - 1):
                base = base.parent
            return base.joinpath(*(module or "").split("."))
        if module and module.startswith("infernux.shader.lib."):
            from infernux.resources import resources_path

            return Path(resources_path, "shaders", "lib", *module.split(".")[3:])
        owner.error(location, "ShaderLib imports use a relative path or infernux.shader.lib")

    def load_reference(self, module, level, owner, location, *, declaring_path=None):
        base = self._import_base(module, level, owner, location, declaring_path=declaring_path)
        return self.load(base.with_name(base.name + ".shaderlib.py"), owner, location)

    def imports(self, imports, owner: ShaderEmitter):
        functions, dependencies = {}, []
        for statement in imports:
            if any(item.name == "*" for item in statement.names):
                owner.error(statement, "ShaderLib imports name explicit exports")
            if statement.module:
                module = self.load_reference(statement.module, statement.level, owner, statement)
                dependencies.append(module.name)
                for alias in statement.names:
                    if alias.name not in module.functions:
                        name = statement.module.rsplit(".", 1)[-1]
                        owner.error(statement, f"ShaderLib {name!r} has no export {alias.name!r}")
                    name = alias.asname or alias.name
                    if name in functions:
                        owner.error(statement, "Duplicate imported ShaderLib name: " + name)
                    functions[name] = module.functions[alias.name]
            else:
                base = self._import_base(None, statement.level, owner, statement)
                for alias in statement.names:
                    module = self.load(base / (alias.name + ".shaderlib.py"), owner, statement)
                    dependencies.append(module.name)
                    for name, function in module.functions.items():
                        functions[(alias.asname or alias.name) + "." + name] = function
        return functions, tuple(dict.fromkeys(dependencies))

    def load(self, path, owner, location, *, source=None):
        path = path.resolve()
        if path in self.loading:
            owner.error(location, "ShaderLib import cycle reaches " + str(path))
        if path in self.modules:
            return self.modules[path]
        if source is None:
            try:
                source = self.reader(str(path))
            except FileNotFoundError:
                owner.error(location, "ShaderLib source does not exist: " + str(path))
        if source is None:
            owner.error(location, "ShaderLib source does not exist: " + str(path))
        source = source.removeprefix("\ufeff")
        alias, imports, body = declarations(source, str(path))
        numeric_only = self.compute_autodiff is not None
        emitter_type = _ComputeLibraryEmitter if numeric_only else ShaderEmitter
        emitter = emitter_type(source, str(path), aliases=(alias,))
        module = Module("inx_library_" + str(len(self.modules)), str(path), "", {})
        self.modules[path] = module
        self.loading.add(path)
        functions, dependencies = self.imports(imports, emitter)
        lighting = uses_surface_lighting(body, alias, functions)
        for definition in body:
            if not isinstance(definition, ast.FunctionDef):
                emitter.error(definition, "ShaderLib exports typed functions")
            if definition.name in functions:
                emitter.error(definition, "Duplicate ShaderLib function: " + definition.name)
            function = replace(emitter.signature(definition, module.name + "::" + definition.name),
                               lighting_specialization=lighting)
            module.functions[definition.name] = functions[definition.name] = function
        functions = function_effects(body, functions, emitter)
        module.functions = {definition.name: functions[definition.name] for definition in body}
        code = ["import inx_geometry;"] if numeric_only else [
            "import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", *engine_imports(body, alias, graph=self, owner=emitter)]
        code.extend("import " + name + ";" for name in dependencies)
        code.append("namespace " + module.name + " {")
        library = _LibrarySource(module, source, alias, body, functions, code)
        if self.compute_autodiff not in {None, "none"}:
            self._derivative_sources.append(library)
        else:
            library.emit(emitter_type)
        self.loading.remove(path)
        self.ordered.append(module)
        return module

    def prepare_derivatives(self, roots):
        """Emit Slang derivatives for the resolved helper call closure."""
        calls = {}
        for library in self._derivative_sources:
            for definition in library.definitions:
                symbol = library.module.functions[definition.name].symbol
                calls[symbol] = tuple(library.functions[name].symbol for name in _function_calls(definition)
                                      if name in library.functions)
        pending = [function.symbol for function in roots]
        differentiable = set()
        while pending:
            symbol = pending.pop()
            if symbol not in differentiable:
                differentiable.add(symbol)
                pending.extend(calls[symbol])
        for library in self._derivative_sources:
            library.emit(_ComputeLibraryEmitter, differentiable=differentiable)
        self._derivative_sources.clear()

    def sources(self):
        builtins = () if self.compute_autodiff is not None else (RESOURCE_MODULE,)
        return (*builtins, *((module.name, module.path, module.source) for module in self.ordered))

    def ranges(self):
        return tuple(location for module in self.ordered for location in module.source_ranges)
