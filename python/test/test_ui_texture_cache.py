import pytest

from Infernux.core.assets import AssetManager
from Infernux.ui import ui_texture_cache as texture_cache_module


class _NativeTexturePreview:
    def __init__(self):
        self.live_texture_id = 0
        self.imported_texture_id = 0
        self.queries = []

    def get_texture_preview_texture_id(self, resource_key):
        self.queries.append(resource_key)
        return self.live_texture_id

    def _get_imported_texture_ui_texture_id(self, resource_key, guid):
        self.queries.append((resource_key, guid))
        return self.imported_texture_id


class _Engine:
    def __init__(self, native):
        self._native = native

    def get_native_engine(self):
        return self._native


def test_cached_ui_texture_refreshes_replaced_native_descriptor(monkeypatch, tmp_path):
    from Infernux.engine import project_context

    texture = tmp_path / "Assets" / "UI" / "button.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"png")
    monkeypatch.setattr(project_context, "_project_root", str(tmp_path))
    monkeypatch.setattr(texture_cache_module, "texture_stamp", lambda *_args: 17)
    monkeypatch.setattr(
        AssetManager,
        "_asset_database",
        type(
            "_AssetDatabase",
            (),
            {"get_guid_from_path": lambda _self, _path: "button-texture-guid"},
        )(),
    )

    scheduled = iter(((101, 4, 4), (303, 4, 4)))
    monkeypatch.setattr(
        texture_cache_module,
        "query_or_schedule_texture",
        lambda *_args, **_kwargs: next(scheduled),
    )

    native = _NativeTexturePreview()
    cache = texture_cache_module.UITextureCache()
    engine = _Engine(native)

    assert cache.get(engine, "Assets/UI/button.png") == 101
    first_generation = cache.generation

    native.live_texture_id = 202
    assert cache.get(engine, "Assets/UI/button.png") == 202
    assert cache.generation == first_generation + 1

    native.live_texture_id = 0
    assert cache.get(engine, "Assets/UI/button.png") == 303
    assert cache.generation == first_generation + 2


def test_ui_texture_requires_registered_asset_guid(monkeypatch):
    monkeypatch.setattr(
        AssetManager,
        "_asset_database",
        type(
            "_AssetDatabase",
            (),
            {"get_guid_from_path": lambda _self, _path: ""},
        )(),
    )

    cache = texture_cache_module.UITextureCache()
    with pytest.raises(KeyError, match="not registered in AssetDatabase"):
        cache.get(_Engine(_NativeTexturePreview()), "Assets/UI/missing.png")


def test_texture_ref_uses_native_guid_binding_without_path_round_trip(monkeypatch):
    from Infernux.core.asset_ref import TextureRef
    monkeypatch.setattr(AssetManager, "require_asset_database", lambda: pytest.fail("GUID binding must not need AssetDatabase"))
    monkeypatch.setattr(
        texture_cache_module,
        "query_or_schedule_texture",
        lambda *_args, **_kwargs: pytest.fail("GUID binding must not decode a source image"),
    )

    cache = texture_cache_module.UITextureCache()
    reference = TextureRef(guid="texture-guid", path_hint="Assets/UI/stale.png")
    native = _NativeTexturePreview()
    native.imported_texture_id = 404
    assert cache.get(_Engine(native), reference) == 404
    assert native.queries == [("ui_img|texture-guid", "texture-guid")]
    assert tuple(cache._cache) == ("texture-guid",)
    generation = cache.generation
    native.imported_texture_id = 606
    assert cache.get(_Engine(native), reference) == 606
    assert cache.generation == generation + 1


def test_player_texture_ref_uses_native_cooked_guid_without_editor_database(monkeypatch):
    from Infernux.application import Application
    from Infernux.core.asset_ref import TextureRef
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(AssetManager, "require_asset_database", lambda: pytest.fail("Player must not need AssetDatabase"))
    monkeypatch.setattr(texture_cache_module, "query_or_schedule_texture", lambda *_args, **_kwargs: pytest.fail("Cooked artifacts are not source images"))

    cache = texture_cache_module.UITextureCache()
    reference = TextureRef(guid="ui-guid", path_hint="C:/host-only/stale.png")
    native = _NativeTexturePreview()
    native.imported_texture_id = 505
    assert cache.get(_Engine(native), reference) == 505
    assert native.queries == [("ui_img|ui-guid", "ui-guid")]
    assert tuple(cache._cache) == ("ui-guid",)
