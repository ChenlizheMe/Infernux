"""The shipped pointer declarations must describe the author-facing API."""
import ast
import dataclasses
import inspect
from pathlib import Path

from infernux.ui import ui_event_data, ui_event_system


def declaration(module, name):
    tree = ast.parse(Path(module.__file__).with_suffix(".pyi").read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_pointer_source_enum_is_available_to_type_checkers():
    node = declaration(ui_event_data, "PointerType")
    values = {target.id: ast.literal_eval(item.value) for item in node.body if isinstance(item, ast.Assign)
              for target in item.targets}
    assert values == {name: int(member) for name, member in ui_event_data.PointerType.__members__.items()}


def test_pointer_event_identity_and_cancel_fields_are_typed():
    node = declaration(ui_event_data, "PointerEventData")
    fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign)}
    assert fields == set(ui_event_data.PointerEventData.__slots__)


def test_pointer_snapshot_constructor_defaults_and_frozen_contract_match():
    node = declaration(ui_event_system, "UIPointerFrame")
    fields = {item.target.id: item for item in node.body if isinstance(item, ast.AnnAssign)}
    actual = dataclasses.fields(ui_event_system.UIPointerFrame)
    assert list(fields) == [field.name for field in actual]
    for field in actual:
        value = fields[field.name].value
        if field.default is dataclasses.MISSING:
            assert value is None
        else:
            assert ast.literal_eval(value) == field.default
    decorator = next(item for item in node.decorator_list
                     if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "dataclass")
    assert {item.arg: ast.literal_eval(item.value) for item in decorator.keywords} == {"frozen": True, "slots": True}


def test_pointer_processor_public_entrypoints_are_declared():
    node = declaration(ui_event_system, "UIEventProcessor")
    methods = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}
    actual = {name: value for name, value in vars(ui_event_system.UIEventProcessor).items()
              if not name.startswith("_") and inspect.isfunction(value)}
    assert set(actual) <= set(methods)
    for name, value in actual.items():
        assert list(inspect.signature(value).parameters) == [arg.arg for arg in methods[name].args.args]
