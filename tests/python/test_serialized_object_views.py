from __future__ import annotations

from infernux.components import InxComponent, SerializableObject, serialized_field, FieldType
from infernux.components.fields import get_raw_field_value
from infernux.components.ref_wrappers import MaterialRef
from infernux.core import Material, Texture
from infernux.core.asset_ref import TextureRef
from infernux.lib import InxMaterial


class RefViewData(SerializableObject):
    mat: Material = None
    tex: Texture = None


class RefViewComponent(InxComponent):
    mat: Material = None
    tex: Texture = None
    materials = serialized_field(default=[], field_type=FieldType.LIST, element_type=FieldType.MATERIAL)
    textures = serialized_field(default=[], field_type=FieldType.LIST, element_type=FieldType.TEXTURE)
    data: RefViewData = serialized_field(default=RefViewData())


def _make_material() -> Material:
    return Material.from_native(InxMaterial.create_default_lit())


def _make_texture() -> Texture:
    tex = Texture.solid_color(4, 4, 255, 0, 0, 255)
    assert tex is not None
    return tex


def test_component_resource_views_share_identity_through_nested_fields_and_lists():
    comp = RefViewComponent()
    mat, tex = _make_material(), _make_texture()
    try:
        comp.mat, comp.tex = mat, tex
        comp.materials, comp.textures = [mat], [tex]
        comp.data.mat, comp.data.tex = mat, tex
        assert comp.mat is comp.data.mat is mat
        assert comp.tex is comp.data.tex is tex
        assert comp.materials == [mat] and comp.textures == [tex]
        assert isinstance(get_raw_field_value(comp, "mat"), MaterialRef)
        assert isinstance(get_raw_field_value(comp, "tex"), TextureRef)
        assert isinstance(get_raw_field_value(comp, "materials")[0], MaterialRef)
        assert isinstance(get_raw_field_value(comp, "textures")[0], TextureRef)
        assert isinstance(get_raw_field_value(comp.data, "mat"), MaterialRef)
        assert isinstance(get_raw_field_value(comp.data, "tex"), TextureRef)
    finally:
        comp._call_on_destroy()
