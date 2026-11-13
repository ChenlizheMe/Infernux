"""Static ShaderLib imports and typed functions shared by GPU author domains."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .slang_ast import SlangEmitter, qualified
from .shader_types import RESOURCE_ELEMENTS, RESOURCE_MODULE, RESOURCE_TYPES, STRUCT_FIELDS, STRUCT_TYPES, SURFACE_FIELDS
from .diagnostics import parse_gpu_module


_ENGINE_FUNCTIONS = {"frame_time": ("inx_frame_time", "float"),
                     "frame_delta_time": ("inx_frame_delta_time", "float"),
                     "view": ("inx_view_context", "InxViewContext"),
                     "camera": ("inx_camera_context", "InxCameraContext")}
_ENGINE_MODULES = {"inx_frame": ("frame_time", "frame_delta_time", "view", "ViewContext"),
                   "inx_camera": ("camera", "CameraContext")}


def engine_imports(body, alias):
    names = {
        qualified(node)
        for declaration in body for node in ast.walk(declaration)
        if isinstance(node, ast.Attribute)
    }
    return tuple("import " + module + ";" for module, exports in _ENGINE_MODULES.items()
                 if any(alias + "." + name in names for name in exports))


@dataclass(frozen=True)
class Function:
    symbol: str
    result: str
    arguments: tuple[str, ...]
    lighting_specialization: bool = False


@dataclass
class Module:
    name: str
    path: str
    source: str
    functions: dict[str, Function]
    source_ranges: tuple = ()


class ShaderEmitter(SlangEmitter):
    def __init__(self, source, path, *, aliases=("shader",), functions=None, types=None):
        super().__init__(source, path, aliases=aliases, types=types)
        self.functions = dict(functions or {})

    def visit_Attribute(self, node):
        if self.infer(node.value) in STRUCT_FIELDS:
            self.infer(node)
        return super().visit_Attribute(node)

    def infer(self, node):
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
                and self.symbol(qualified(node.func)) in {"sample", "sample_level", "sample_grad", "load"}):
            if not node.args:
                self.error(node, "A texture operation needs a declared GPU image", code="author.type")
            return RESOURCE_ELEMENTS.get(self.infer(node.args[0]))
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) != qualified(node.func)
                and self.symbol(qualified(node.func)) == "dimensions"):
            return "uint2"
        if isinstance(node, ast.Call) and (function := self.functions.get(qualified(node.func))):
            return function.result
        return super().infer(node)

    def visit_Call(self, node):
        operation = self.symbol(qualified(node.func))
        if operation in _ENGINE_FUNCTIONS and operation != qualified(node.func):
            if node.args or node.keywords:
                self.error(node, "shader." + operation + " takes no arguments", code="author.type")
            return _ENGINE_FUNCTIONS[operation][0] + "()"
        if operation == "SurfaceData" and operation != qualified(node.func):
            values = {keyword.arg: keyword.value for keyword in node.keywords}
            if node.args or values.keys() - SURFACE_FIELDS.keys() or len(values) != len(node.keywords):
                self.error(node, "SurfaceData uses named surface fields", code="author.type")
            fields = (self.visit(values[name]) if name in values else default
                      for name, (_, default) in SURFACE_FIELDS.items())
            return "inx_surface_data(" + ", ".join(fields) + ")"
        image_operations = {"sample": (2, "Sample"), "sample_level": (3, "SampleLevel"),
                            "sample_grad": (4, "SampleGrad"), "load": (3, "Load"), "dimensions": (2, "GetDimensions")}
        if operation in image_operations and operation != qualified(node.func):
            arity, intrinsic = image_operations[operation]
            if node.keywords or len(node.args) != arity:
                self.error(node, f"shader.{operation} takes {arity} typed positional arguments")
            kind = self.infer(node.args[0])
            if kind not in RESOURCE_ELEMENTS:
                self.error(node, "Texture operations receive a declared GPU image")
            if kind == "InxUInt2D" and operation not in {"load", "dimensions"}:
                self.error(node, "Integer images use shader.load with pixel coordinates", code="author.type")
            image, *arguments = (self.visit(argument) for argument in node.args)
            if operation == "dimensions":
                return f"inx_image_dimensions({image}, uint({arguments[0]}))"
            if operation == "load":
                return f"{image}.texture.Load(int3({arguments[0]}, {arguments[1]}))"
            return image + ".texture." + intrinsic + "(" + ", ".join([image + ".sampler", *arguments]) + ")"
        function = self.functions.get(qualified(node.func))
        if function is None:
            return super().visit_Call(node)
        if node.keywords or len(node.args) != len(function.arguments):
            self.error(node, "Shader helper arguments must match its typed positional signature")
        specialization = "<tiled_lighting>" if function.lighting_specialization else ""
        return function.symbol + specialization + "(" + ", ".join(self.visit(arg) for arg in node.args) + ")"

    def annotation(self, node):
        spelling = qualified(node)
        name = spelling.rsplit(".", 1)[-1]
        if spelling not in {"float", "int", "bool"} and not any(spelling.startswith(alias + ".") for alias in self.aliases):
            self.error(node, "GPU value types come from infernux.shader", code="author.type")
        if name in STRUCT_TYPES:
            return STRUCT_TYPES[name]
        return RESOURCE_TYPES[name] if name in RESOURCE_TYPES else super().annotation(node)

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
        return Function(symbol, self.annotation(definition.returns),
                        tuple(self.annotation(field.annotation) for field in fields))

    def emit_function(self, definition, function, *, method=False, public=False):
        fields = definition.args.args[1:] if method else definition.args.args
        self.types.update((field.arg, kind) for field, kind in zip(fields, function.arguments))
        self.declare_locals(definition.body)
        self.local_declarations()
        self.block(definition.body)
        parameters = ", ".join(kind + " " + field.arg for field, kind in zip(fields, function.arguments))
        specialization = "<let tiled_lighting : bool>" if function.lighting_specialization else ""
        return [self.source_directive(definition),
                f'{"public " if public else ""}{function.result} {definition.name}{specialization}({parameters})',
                "{", *self.lines, "}"]


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
    def __init__(self, reader=None):
        self.modules: dict[Path, Module] = {}
        self.ordered: list[Module] = []
        self.loading: set[Path] = set()
        self.reader = reader or (lambda path: Path(path).read_text(encoding="utf-8-sig"))

    def reference(self, call, owner, *, stage, suffix):
        if (not isinstance(call, ast.Call) or owner.qualified(call.func) != owner.alias + ".ref"
                or call.keywords or len(call.args) != 1):
            owner.error(call, "GPU stages refer to one asset with shader.ref", code="author.reference")
        reference = owner.literal(call.args[0])
        if not isinstance(reference, str):
            owner.error(call, "A GPU asset reference uses a registered ID or " + suffix + " path", code="author.reference")
        if reference.endswith(suffix):
            return (Path(owner.path).parent / reference).resolve()
        from infernux.lib import _Infernux

        try:
            resolved = _Infernux._shader_source_path(reference, stage)
        except (RuntimeError, ValueError) as error:
            owner.error(call, str(error), code="author.reference")
        if not resolved:
            owner.error(call, "GPU asset ID is not registered: " + reference, code="author.reference")
        return Path(resolved)

    def imports(self, imports, owner: ShaderEmitter):
        functions, dependencies = {}, []
        for statement in imports:
            if any(item.name == "*" for item in statement.names):
                owner.error(statement, "ShaderLib imports name explicit exports")
            if statement.level:
                base = Path(owner.path).parent
                for _ in range(statement.level - 1):
                    base = base.parent
                base = base.joinpath(*(statement.module or "").split("."))
            elif statement.module and statement.module.startswith("infernux.shader.lib."):
                from infernux.resources import resources_path

                base = Path(resources_path) / "shaders" / "lib"
                base = base.joinpath(*statement.module.split(".")[3:])
            else:
                owner.error(statement, "ShaderLib imports use a relative path or infernux.shader.lib")
            if statement.module:
                module = self.load(base.with_name(base.name + ".shaderlib.py"), owner, statement)
                dependencies.append(module.name)
                for alias in statement.names:
                    if alias.name not in module.functions:
                        owner.error(statement, f"ShaderLib {base.name!r} has no export {alias.name!r}")
                    name = alias.asname or alias.name
                    if name in functions:
                        owner.error(statement, "Duplicate imported ShaderLib name: " + name)
                    functions[name] = module.functions[alias.name]
            else:
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
        emitter = ShaderEmitter(source, str(path), aliases=(alias,))
        module = Module("inx_library_" + str(len(self.modules)), str(path), "", {})
        self.modules[path] = module
        self.loading.add(path)
        functions, dependencies = self.imports(imports, emitter)
        for definition in body:
            if not isinstance(definition, ast.FunctionDef):
                emitter.error(definition, "ShaderLib exports typed functions")
            if definition.name in functions:
                emitter.error(definition, "Duplicate ShaderLib function: " + definition.name)
            function = emitter.signature(definition, module.name + "::" + definition.name)
            module.functions[definition.name] = functions[definition.name] = function
        code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", *engine_imports(body, alias),
                *("import " + name + ";" for name in dependencies),
                "namespace " + module.name + " {"]
        ranges = []
        for definition in body:
            function_emitter = ShaderEmitter(source, str(path), aliases=(alias,), functions=functions)
            code.extend(function_emitter.emit_function(definition, module.functions[definition.name], public=True))
            ranges.extend(function_emitter.source_ranges)
        code.append("}")
        module.source = "\n".join(code)
        module.source_ranges = tuple(ranges)
        self.loading.remove(path)
        self.ordered.append(module)
        return module

    def sources(self):
        return (RESOURCE_MODULE, *((module.name, module.path, module.source) for module in self.ordered))

    def ranges(self):
        return tuple(location for module in self.ordered for location in module.source_ranges)
