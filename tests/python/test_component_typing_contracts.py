"""Published authoring stubs expose the actual component entry points."""
import ast
import importlib
import inspect
from pathlib import Path

import pytest


def stub_tree(module):
    return ast.parse(Path(module.__file__).with_suffix(".pyi").read_text(encoding="utf-8"))


def declaration(module, name):
    for node in stub_tree(module).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    imported = importlib.import_module("." * node.level + node.module, module.__package__)
                    assert getattr(module, name) is getattr(imported, alias.name)
                    return declaration(imported, alias.name)
    raise AssertionError(f"Missing declaration: {module.__name__}.{name}")


def assert_parameters_match(module, name):
    args = declaration(module, name).args
    declared = {arg.arg:inspect.Parameter.POSITIONAL_ONLY for arg in args.posonlyargs}
    declared.update({arg.arg:inspect.Parameter.POSITIONAL_OR_KEYWORD for arg in args.args})
    declared.update({arg.arg:inspect.Parameter.KEYWORD_ONLY for arg in args.kwonlyargs})
    if args.vararg:
        declared[args.vararg.arg] = inspect.Parameter.VAR_POSITIONAL
    if args.kwarg:
        declared[args.kwarg.arg] = inspect.Parameter.VAR_KEYWORD
    actual = {name:parameter.kind for name,parameter in inspect.signature(getattr(module,name)).parameters.items()}
    assert declared == actual


def test_fields_stub_does_not_advertise_removed_functions():
    from infernux.components import fields
    declared = {node.name for node in stub_tree(fields).body if isinstance(node, ast.FunctionDef)}
    assert not {name for name in declared if not callable(getattr(fields,name,None))}


@pytest.mark.parametrize("module_name", ["infernux.components", "infernux.components.fields"])
def test_serialized_field_keywords_match_both_public_import_paths(module_name):
    assert_parameters_match(importlib.import_module(module_name), "serialized_field")


def test_loader_stub_preserves_positional_type_name_and_keyword_guid():
    from infernux.components import script_loader
    assert_parameters_match(script_loader, "load_and_create_component")


def test_declared_author_keywords_work_with_native_components(scene, tmp_path):
    from infernux.components import InxComponent, SerializableObject, serialized_field, load_and_create_component
    from infernux.components.registry import unregister_component_script

    class Payload(SerializableObject):
        health: int = 12

    class TypedAuthor(InxComponent):
        payload: Payload = serialized_field(default_factory=Payload, field_id="payload-v1")

    component = scene.create_game_object("Typed author").add_component(TypedAuthor)
    assert component.payload.health == 12
    path = tmp_path / "typed_keyword_author.py"
    path.write_text("from infernux.components import InxComponent\nclass Author(InxComponent):\n    value: int = 31\n", encoding="utf-8")
    try:
        loaded = load_and_create_component(str(path), type_name="Author", script_guid="8" * 32)
        assert loaded is not None
        scene.create_game_object("Loaded author").add_py_component(loaded)
        assert loaded.value == 31
    finally:
        unregister_component_script(str(path))
