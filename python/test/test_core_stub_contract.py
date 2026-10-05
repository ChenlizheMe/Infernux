"""Keep shipped core declarations usable by project authors and IDEs."""
import ast
from dataclasses import fields
from pathlib import Path

import pytest

import infernux.core as core
from infernux.core import asset_ref, asset_types


def stub_tree(module):
    return ast.parse(Path(module.__file__).with_suffix(".pyi").read_text(encoding="utf-8"))


def declarations(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def test_core_exports_are_declared_and_reexported():
    tree = stub_tree(core)
    exported = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets))
    assert set(exported) == set(core.__all__)
    imports = {alias.asname for node in tree.body if isinstance(node, ast.ImportFrom)
               for alias in node.names if alias.asname == alias.name}
    assert set(core.__all__) <= imports


def test_all_specialized_asset_references_have_declarations():
    expected = {name for name, value in vars(asset_ref).items()
                if isinstance(value, type) and issubclass(value, asset_ref.AssetRefBase)}
    assert expected <= declarations(stub_tree(asset_ref)).keys()


@pytest.mark.parametrize("name", ["SpriteFrame", "TextureImportSettings"])
def test_sprite_authoring_constructor_fields_are_typed(name):
    classes = declarations(stub_tree(asset_types))
    assert name in classes
    declared = {node.target.id for node in classes[name].body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    assert {field.name for field in fields(getattr(asset_types, name))} <= declared


def test_texture_type_members_are_typed():
    declaration = declarations(stub_tree(asset_types))["TextureType"]
    declared = {target.id for node in declaration.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    assert set(asset_types.TextureType.__members__) <= declared
