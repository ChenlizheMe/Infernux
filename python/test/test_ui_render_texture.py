"""Live UI sources retain GPU owners and invalidate cached draws on resize."""
from types import SimpleNamespace

import pytest

from Infernux.core.render_texture import RenderTexture
from Infernux.ui import UIImage
from Infernux.ui.ui_render_dispatch import image_texture_source, material_visual_state, runtime_ui_revision


def target():
    result = RenderTexture.__new__(RenderTexture)
    result._native = SimpleNamespace(revision=1, asset_guid='')
    return result


def test_image_runtime_override_preserves_authored_texture():
    from Infernux.components.fields import get_raw_field_value
    from Infernux.core.asset_ref import TextureRef

    image = UIImage()
    authored = TextureRef(path_hint='Assets/Textures/monitor.png')
    image.texture = authored
    texture = target()
    image.texture = texture
    assert image_texture_source(image, material_visual_state(image)) is texture
    assert get_raw_field_value(image, 'texture') is authored
    image.texture = None
    assert get_raw_field_value(image, 'texture') is authored
    with pytest.raises(TypeError, match='Texture.Sampled requires'):
        image.texture = 'invented-resource-path'


def test_button_runtime_override_preserves_guid_backed_background_texture():
    from Infernux.components.fields import get_raw_field_value
    from Infernux.core.asset_ref import TextureRef
    from Infernux.ui import UIButton

    button = UIButton()
    authored = TextureRef(
        guid="button-texture-guid",
        path_hint="Assets/Textures/button.png",
    )
    button.background_texture = authored
    runtime = target()

    button.background_texture = runtime
    assert image_texture_source(button, material_visual_state(button)) is runtime
    assert get_raw_field_value(button, "background_texture") is authored
    button.background_texture = None
    assert get_raw_field_value(button, "background_texture") is authored
    assert image_texture_source(button, material_visual_state(button)) is authored


def test_screen_command_cache_tracks_resize_without_component_mutation(scene):
    from Infernux.ui import UICanvas
    image = UIImage()
    image.texture = target()
    owner = scene.create_game_object("Canvas")
    canvas = UICanvas()
    owner.add_py_component(canvas)
    obj = scene.create_game_object("Image")
    obj.set_parent(owner)
    obj.add_py_component(image)
    canvases = [canvas]
    before = runtime_ui_revision(scene, canvases, 800, 600)
    assert runtime_ui_revision(scene, canvases, 800, 600) == before
    image.texture._native.revision += 1
    assert runtime_ui_revision(scene, canvases, 800, 600) != before


def test_button_command_cache_tracks_live_background_resize(scene):
    from Infernux.ui import UIButton, UICanvas

    button = UIButton()
    button.background_texture = target()
    owner = scene.create_game_object("Button Canvas")
    canvas = UICanvas()
    owner.add_py_component(canvas)
    obj = scene.create_game_object("Button")
    obj.set_parent(owner)
    obj.add_py_component(button)
    canvases = [canvas]

    before = runtime_ui_revision(scene, canvases, 800, 600)
    assert runtime_ui_revision(scene, canvases, 800, 600) == before
    button.background_texture._native.revision += 1
    assert runtime_ui_revision(scene, canvases, 800, 600) != before


def test_material_live_source_revision_and_explicit_image_priority():
    runtime = SimpleNamespace(revision=1)
    material = SimpleNamespace(guid='authored-material', name='Authored Material', get_version=lambda: 2,
        _texture_assets_pending=False,
        _get_render_texture=lambda name: runtime,
        has_property=lambda name: name == 'texSampler',
        get_texture=lambda _name: 'material-texture-guid')
    image = UIImage()
    image.material = material
    before = material_visual_state(image)
    assert image_texture_source(image, before) is runtime
    runtime.revision += 1
    assert material_visual_state(image)['signature'] != before['signature']
    image.texture = target()
    assert image_texture_source(image, material_visual_state(image)) is image.texture


def test_material_sampled_texture_keeps_guid_identity_until_cache_resolution():
    from Infernux.core.asset_ref import TextureRef

    texture_guid = "8a51dcd72aa64cd5963731d9b4f2507f"
    material = SimpleNamespace(
        guid="authored-material",
        name="Authored Material",
        get_version=lambda: 2,
        _texture_assets_pending=False,
        _get_render_texture=lambda _name: None,
        has_property=lambda name: name in {"baseColor", "texSampler"},
        get_color=lambda _name: [1.0, 1.0, 1.0, 1.0],
        get_texture=lambda _name: texture_guid,
    )
    image = UIImage()
    image.material = material

    state = material_visual_state(image)
    source = image_texture_source(image, state)

    assert state["texture_guid"] == texture_guid
    assert isinstance(source, TextureRef)
    assert source.guid == texture_guid
    assert source.path_hint == ""


def test_ui_gpu_texture_publication_does_not_resolve_an_asset_path(monkeypatch):
    from Infernux.lib import _Infernux
    from Infernux.ui import ui_texture_cache as module

    class NativeTarget:
        pass

    monkeypatch.setattr(_Infernux, '_RenderTexture', NativeTarget)
    texture = target()
    texture._native = NativeTarget()
    published = []
    native = SimpleNamespace(_get_render_texture_ui_texture_id=lambda value: published.append(value) or 71)
    engine = SimpleNamespace(get_native_engine=lambda: native)
    cache = module.UITextureCache()
    assert cache.get(engine, texture) == 71
    assert cache.get(engine, texture._native) == 71
    assert published == [texture._native, texture._native]
