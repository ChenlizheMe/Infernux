"""Compile the actual static library graph into numerical acceptance programs."""

import ast
from pathlib import Path

from infernux._compiler.shader_library import LibraryGraph, ShaderEmitter
from infernux.lib import _Infernux as native
from infernux.resources import resources_path


def compile_library_compute(library, source, label):
    graph = LibraryGraph()
    path = Path(resources_path) / "shaders/lib" / (library + ".shaderlib.py")
    owner = ShaderEmitter(source, "<" + label + ">")
    module = graph.load(path, owner, ast.Constant(value=None, lineno=1, col_offset=0))
    program = native._compile_shader_program(
        "import " + module.name + ";\n" + source.replace("LIBRARY::", module.name + "::"),
        owner.path, label.replace("-", "_"), graph.sources(), ("spirv", "wgsl"),
        source_ranges=graph.ranges(), root_constants=True,
    )
    return program["targets"][0]["entries"][0]
