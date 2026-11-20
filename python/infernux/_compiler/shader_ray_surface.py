"""Frozen authored Surface adapters; addresses come from target reflection."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass

from infernux._compiler.shader import _Author, _read_fragment, _RAY_TEXTURE_MODULE
from infernux._compiler.shader_info import read_shader_info
from infernux._compiler.shader_library import RESOURCE_MODULE, LibraryGraph, ShaderEmitter, declarations, qualified
from infernux._compiler.shading_model import read_shading_model
from infernux._compiler.slang_ast import add_module_imports


_CONTEXT_FIELDS = {"position_ws", "normal_ws", "tangent_ws", "color", "uv0", "uv1", "surface_id", "line_color"}
_SLOT_TYPES = {"float": "float", "vec2": "float2", "vec3": "float3", "vec4": "float4", "color": "float4",
               "int": "int", "uint": "uint", "bool": "uint", "texture_2d": "uint"}
_DYNAMIC_OPERATIONS = {"camera", "frame_time", "frame_delta_time", "_previous_ray_geometry_serial", "_ray_lighting_changed",
                       "_ray_proposals_changed",
                       "_previous_ray_light_serial", "_previous_ray_other_change_serial"}


class _RayRules:
    stage = "ray_surface"

    def visit_Attribute(self, node):
        own_field = isinstance(node.value, ast.Name) and node.value.id == "self"
        if not own_field and self.infer(node.value) == "InxMeshInput" and node.attr not in _CONTEXT_FIELDS:
            self.error(node, "Secondary surfaces cannot read raster context '" + node.attr + "'",
                       code="author.ray_context")
        return super().visit_Attribute(node)

    def visit_Call(self, node):
        operation = self.symbol(qualified(node.func))
        if operation in _DYNAMIC_OPERATIONS or operation.startswith("lighting."):
            self.error(node, "Static secondary surfaces cannot read " + operation,
                       code="author.ray_context")
        return super().visit_Call(node)


class _RayLibraryEmitter(_RayRules, ShaderEmitter):
    pass


class _RayAuthor(_RayRules, _Author):
    ray_sampling_type = None
    ray_entry_visibility = "public "

    def visit_Call(self, node):
        operation = self.symbol(qualified(node.func))
        if operation == "sample_level" and node.args and self.infer(node.args[0]) == "InxColor2D":
            image = node.args[0]
            if (not isinstance(image, ast.Attribute) or not isinstance(image.value, ast.Name)
                    or image.value.id != "self" or image.attr not in self.fields
                    or self.fields[image.attr].kind != "texture_2d"):
                self.error(node, "Secondary texture sampling uses a declared material image", code="author.ray_texture")
            if len(node.args) != 3 or node.keywords or self.infer(node.args[1]) != "float2" or self.infer(node.args[2]) != "float":
                self.error(node, "Explicit secondary sampling uses (image, vec2 UV, float LOD)", code="author.type")
            return "inx_ray_sample_texture(" + ", ".join(self.visit(arg) for arg in node.args) + ")"
        if qualified(node.func) in self.functions and any(self.infer(arg) == "InxColor2D" for arg in node.args):
            self.error(node, "Sample material images explicitly in the Surface; pass sampled values to ShaderLib",
                       code="author.ray_texture")
        if self._surface_sampling_operation(node) is not None:
            self.error(node, "Custom secondary surfaces use explicit shader.sample_level", code="author.ray_texture")
        return super().visit_Call(node)


def _validate_library_calls(graph, body, functions):
    modules = {module.name: module for module in graph.ordered}
    pending = [functions[qualified(node.func)] for node in ast.walk(body) if isinstance(node, ast.Call)
               and qualified(node.func) in functions]
    visited = set()
    while pending:
        function = pending.pop()
        if function.symbol in visited:
            continue
        visited.add(function.symbol)
        namespace, _, name = function.symbol.rpartition("::")
        if namespace not in modules:
            continue
        module = modules[namespace]
        text = graph.reader(module.path)
        alias, imports, definitions = declarations(text, module.path)
        emitter = _RayLibraryEmitter(text, module.path, aliases=(alias,))
        resolved, _ = graph.imports(imports, emitter)
        resolved.update(module.functions)
        emitter.functions = resolved
        definition = next(node for node in definitions if node.name == name)
        emitter.emit_function(definition, function, public=True)
        pending.extend(resolved[qualified(node.func)] for node in ast.walk(definition)
                       if isinstance(node, ast.Call) and qualified(node.func) in resolved)


@dataclass(frozen=True)
class RaySurfaceAdapter:
    source_path: str
    module_name: str
    generated_source: str
    libraries: tuple
    source_ranges: tuple
    manifest: dict


def parameter_slot_declarations(parameters, prefix):
    """Explicit 16-byte slots have the same member addresses in raw and structured storage."""
    code = []
    for kind in dict.fromkeys(field.kind for field in parameters):
        scalar = "int" if kind == "int" else "uint" if kind in {"uint", "bool", "texture_2d"} else "float"
        value_type = _SLOT_TYPES[kind]
        lanes = int(value_type[-1]) if value_type[-1:].isdigit() else 1
        code += ["public struct " + prefix + "_Slot_" + kind + " {",
                 "    public " + _SLOT_TYPES[kind] + " value;",
                 *("    " + scalar + " _padding" + str(i) + ";" for i in range(4-lanes)), "};"]
    return code


def prepare_ray_surface(source: str, source_path: str, module_name: str, reader=None, *, schema_binding=0) -> RaySurfaceAdapter:
    """Prepare the actual author, with the same logical fields and import closure.

    This first adapter slice covers static PBR expressions and numeric helpers.
    Explicit texture LOD uses the same published source and normalized page as
    builtin secondary surfaces; raster derivatives have no secondary meaning.
    """
    graph = LibraryGraph(reader, module_prefix=module_name + "_library_")
    fragment = _read_fragment(source.removeprefix("\ufeff"), source_path, graph=graph, author_type=_RayAuthor)
    author = fragment.author
    if not author.is_surface:
        author.error(fragment.main, "Secondary adapters require a shader.surface author", code="author.ray_surface")
    if fragment.helpers:
        author.error(fragment.helpers[0], "Secondary Surface helpers belong to a typed ShaderLib in this adapter",
                     code="author.ray_method")
    _validate_library_calls(graph, fragment.main, author.functions)
    path, location = fragment.shading_model
    model = read_shading_model(path, graph, author, location)
    if model.deferred_model != "PBR":
        author.error(location, "This secondary adapter requires the built-in PBR response",
                     code="author.ray_shading_model")
    parameters = tuple(author.fields.values())
    values = tuple(field for field in parameters if field.kind != "texture_2d")
    textures = tuple(field for field in parameters if field.kind == "texture_2d")
    declaration = ast.parse(source.removeprefix("\ufeff")).body
    fields = {node.targets[0].id: node for root in declaration if isinstance(root, ast.ClassDef)
              for node in root.body if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)}
    for field in parameters:
        if field.kind not in _SLOT_TYPES:
            author.error(fields.get(field.name, fragment.main),
                         "Secondary parameter '" + field.name + "' requires an implemented numeric slot",
                         code="author.ray_parameter")
    record = module_name + "_Material"
    payload = module_name + "_Payload"
    entry = module_name + "_evaluate"
    author.ray_material_type = record
    author.ray_entry = entry
    emitted = author.function(fragment.main)
    code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", "import inx_ray_texture_pages;",
            *("import " + name + ";" for name in fragment.dependencies)]
    add_module_imports(code, author.required_modules)
    code.extend(parameter_slot_declarations(parameters, module_name))
    code += ["public struct " + payload + " {",
             *("    public " + module_name + "_Slot_" + field.kind + " " + field.name + ";" for field in parameters), "};",
             "public struct " + module_name + "_Values {",
             *(line if line.lstrip().startswith("#") else "    public " + line.strip()
               for line in author.parameter_field_declarations(values)), "};",
             "public struct " + record + " { public " + module_name + "_Values values;",
             *("    public uint " + field.name + ";" for field in textures), "};",
             "[[vk::binding(" + str(schema_binding) + ", 3)]] StructuredBuffer<" + payload + "> " + module_name + "_schema;",
             "public " + record + " " + module_name + "_load(ByteAddressBuffer source, uint address)", "{",
             "    " + payload + " payload = source.Load<" + payload + ">(address);",
             "    " + record + " result;",
             *("    result.values." + field.name + " = payload." + field.name + ".value;" for field in values),
             *("    result." + field.name + " = payload." + field.name + ".value;" for field in textures),
             "    return result;", "}", *emitted]
    # The manifest binds the generated entry to this source and logical fields;
    # the native owner supplies the imported asset/program identity, not a name heuristic.
    manifest = {"entry": entry, "family": "standard_pbr",
                "record_type": record, "payload_type": payload, "schema_resource": module_name + "_schema",
                "parameters": [asdict(field) for field in parameters],
                "may_emit": _may_emit(fragment.main)}
    return RaySurfaceAdapter(source_path, module_name, "\n".join(code), (*graph.sources(), _RAY_TEXTURE_MODULE),
                             (*graph.ranges(), *author.source_ranges), manifest)


def _may_emit(main):
    """Prove the common absent/literal-zero case; all other expressions keep support."""
    for node in ast.walk(main):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not qualified(value.func).endswith(".SurfaceData"):
            return True
        if value.args:
            return True
        emission = next((keyword.value for keyword in value.keywords if keyword.arg == "emission"), None)
        if emission is None:
            continue
        zero = isinstance(emission, ast.Constant) and emission.value == 0
        if isinstance(emission, ast.Call) and qualified(emission.func).endswith(".vec3"):
            zero = bool(emission.args) and all(isinstance(arg, ast.Constant) and arg.value == 0 for arg in emission.args)
        if not zero:
            return True
    return False


def optional_ray_surface(source, source_path, module_name, reader=None, *, schema_binding=1):
    """Retain a located authoring diagnostic when raster expressions have no secondary meaning."""
    from infernux._compiler.diagnostics import ShaderSourceError
    try:
        info = read_shader_info(source, source_path)
        tree = ast.parse(source.removeprefix("\ufeff"))
        if not any(isinstance(node, ast.ClassDef) and any(qualified(base).endswith(".surface") for base in node.bases)
                   for node in tree.body):
            return None, None
        if info["surface_type"] == "transparent":
            return None, {"code": "author.ray_surface", "path": source_path, "line": 1,
                          "message": "Secondary Surface adapters currently require opaque or cutout geometry"}
        return prepare_ray_surface(source, source_path, module_name, reader, schema_binding=schema_binding), None
    except ShaderSourceError as error:
        # The owning asset retains its author diagnostic. An invalid or unsupported
        # project Surface cannot make an unrelated ray program fail to compile.
        return None, dict(error.diagnostics[0], severity="warning")
