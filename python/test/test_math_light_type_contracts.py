"""Shipping declarations agree with the native math and Light interfaces."""
import ast
from pathlib import Path

import pytest

from infernux.components.builtin import Light
from infernux.math import vector2, vector3, vector4
from infernux.math.coerce import coerce_quat


ROOT = Path(__file__).resolve().parents[1] / "infernux"


def declared_class(relative, name):
    return next(node for node in ast.parse((ROOT / relative).read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef) and node.name == name)


@pytest.mark.parametrize("vector", [vector2, vector3, vector4])
def test_vector_properties_are_not_declared_as_static_callables(vector):
    node = declared_class("math/vector.pyi", vector.__name__)
    for declaration in node.body:
        if isinstance(declaration, ast.FunctionDef) and any(
                isinstance(item, ast.Name) and item.id == "staticmethod" for item in declaration.decorator_list):
            assert callable(getattr(vector, declaration.name)), declaration.name
    value = vector.one
    assert value.sqr_magnitude == pytest.approx(value.magnitude ** 2)


@pytest.mark.parametrize("vector", [vector2, vector3, vector4])
def test_smooth_damp_declares_both_actual_results(vector):
    result = vector.smooth_damp(vector.zero, vector.one, vector.zero, .5, 100., .1)
    method = next(node for node in declared_class("math/vector.pyi", vector.__name__).body
                  if isinstance(node, ast.FunctionDef) and node.name == "smooth_damp")
    assert type(result) is tuple and len(result) == 2
    assert isinstance(method.returns, ast.Subscript)
    assert ast.unparse(method.returns.value) in {"tuple", "Tuple"}
    declared = method.returns.slice.elts
    assert len(declared) == len(result)
    assert [ast.unparse(item) for item in declared] == [type(item).__name__ for item in result]


def test_orthonormalization_declares_three_actual_vectors():
    result = vector3.ortho_normalize(vector3.right, vector3.up, vector3.forward)
    method = next(node for node in declared_class("math/vector.pyi", "vector3").body
                  if isinstance(node, ast.FunctionDef) and node.name == "ortho_normalize")
    assert type(result) is tuple and len(result) == 3
    assert isinstance(method.returns, ast.Subscript)
    assert ast.unparse(method.returns.value) in {"tuple", "Tuple"}
    assert len(method.returns.slice.elts) == 3
    assert all(isinstance(item, vector3) for item in result)


@pytest.mark.parametrize("field", ["shadow_bias", "shadow_normal_bias"])
def test_light_declares_engine_managed_bias_as_readonly(scene, field):
    light = scene.create_game_object("Light typing").add_component(Light)
    node = declared_class("components/builtin/light.pyi", "Light")
    setters = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == field
               and any(isinstance(decorator, ast.Attribute) and decorator.attr == "setter" for decorator in item.decorator_list)]
    assert bool(setters) == (getattr(Light, field).fset is not None)
    before = getattr(light, field)
    with pytest.raises(AttributeError):
        setattr(light, field, .5)
    assert getattr(light, field) == before


def test_quaternion_coercion_is_declared_and_keeps_identity_default():
    functions = {node.name for node in ast.parse((ROOT / "math/coerce.pyi").read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef)}
    assert "coerce_quat" in functions
    value = coerce_quat(None)
    assert (value.x, value.y, value.z, value.w) == (0, 0, 0, 1)
