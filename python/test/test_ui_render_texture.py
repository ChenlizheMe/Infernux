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
    image = UIImage()
    image.texture_path = 'Assets/Textures/monitor.png'
    texture = target()
    image.texture = texture
    assert image_texture_source(image, material_visual_state(image)) is texture
    assert image.texture_path == 'Assets/Textures/monitor.png'
    image.texture = None
    assert image_texture_source(image, material_visual_state(image)).path_hint == image.texture_path
    with pytest.raises(TypeError, match='Texture.Sampled requires'):
        image.texture = 'invented-resource-path'


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


def test_material_live_source_revision_and_explicit_image_priority():
    runtime = SimpleNamespace(revision=1)
    material = SimpleNamespace(guid='authored-material', name='Authored Material', get_version=lambda: 2,
        _texture_assets_pending=False,
        _get_render_texture=lambda name: runtime,
        has_property=lambda name: name == 'texSampler')
    image = UIImage()
    image.material = material
    before = material_visual_state(image)
    assert image_texture_source(image, before) is runtime
    runtime.revision += 1
    assert material_visual_state(image)['signature'] != before['signature']
    image.texture = target()
    assert image_texture_source(image, material_visual_state(image)) is image.texture


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
