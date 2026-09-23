import pytest


def test_sprite_renderer_structured_path_hint_does_not_recover_guid():
    from Infernux.components.builtin.sprite_renderer import SpriteRenderer

    class Probe:
        def _resolve_texture_guid(self, path):
            raise AssertionError(f"structured reference must not load {path}")

        def _extract_guid(self, value):
            return ""

    with pytest.raises(ValueError, match="imported Texture"):
        SpriteRenderer._resolve_sprite_reference_candidate(
            Probe(), {"path_hint": "Assets/Sprites/legacy.png"}
        )


def test_asset_ref_from_dict_ignores_path_only_document():
    from Infernux.core.asset_ref import TextureRef

    reference = TextureRef.from_dict({"path_hint": "Assets/Textures/legacy.png"})

    assert reference.guid == ""
    assert reference.path_hint == ""
    assert not reference


def test_asset_ref_persistence_drops_display_path():
    from Infernux.core.asset_ref import TextureRef
    from Infernux.components.value_codec import VALUE_CODECS

    reference = TextureRef(
        guid="texture-guid", path_hint="Assets/Textures/renamed-later.png"
    )

    assert reference.to_dict() == {"guid": "texture-guid"}
    assert VALUE_CODECS.encode(reference) == {
        "$type": "asset_ref",
        "asset_type": "Texture",
        "guid": "texture-guid",
    }


def test_material_slot_structured_reference_never_recovers_guid_from_path(monkeypatch):
    from types import SimpleNamespace
    import Infernux.lib as lib
    from Infernux.engine.ui import _inspector_extra_renderers as inspector

    class Database:
        def get_guid_from_path(self, _path):
            raise AssertionError("structured material reference must not use its path")

    monkeypatch.setattr(
        lib,
        "AssetRegistry",
        SimpleNamespace(
            instance=lambda: SimpleNamespace(get_asset_database=lambda: Database())
        ),
    )
    edits = []
    monkeypatch.setattr(
        inspector,
        "_record_material_slot",
        lambda *args: edits.append(args),
    )
    component = SimpleNamespace(get_material_guids=lambda: [""])

    inspector._set_material_slot_from_path(
        component,
        0,
        {"path_hint": "Assets/Materials/obsolete.mat"},
    )

    assert edits == []


def test_material_slot_structured_reference_uses_guid(monkeypatch):
    from types import SimpleNamespace
    import Infernux.lib as lib
    from Infernux.engine.ui import _inspector_extra_renderers as inspector

    monkeypatch.setattr(
        lib,
        "AssetRegistry",
        SimpleNamespace(
            instance=lambda: SimpleNamespace(get_asset_database=lambda: object())
        ),
    )
    edits = []
    monkeypatch.setattr(
        inspector,
        "_record_material_slot",
        lambda *args: edits.append(args),
    )
    component = SimpleNamespace(get_material_guids=lambda: ["old-guid"])

    inspector._set_material_slot_from_path(
        component,
        0,
        {
            "guid": "new-guid",
            "path_hint": "Assets/Materials/not-authoritative.mat",
        },
    )

    assert edits[0][2:4] == ("old-guid", "new-guid")
