from Infernux.graph.types import AssetReference


def test_asset_reference_uses_only_guid_for_identity_and_truthiness():
    legacy = AssetReference.from_dict(
        {"path": "Assets/Models/Legacy.fbx", "unknown": "ignored"}
    )
    display_only = AssetReference.from_dict(
        {"guid": "", "path_hint": "Assets/Models/Legacy.fbx", "extra": True}
    )
    current = AssetReference.from_dict(
        {"guid": "mesh-guid", "path_hint": "Assets/Models/Old.fbx"}
    )

    assert legacy == AssetReference()
    assert not legacy
    assert not display_only
    assert display_only.path_hint == "Assets/Models/Legacy.fbx"
    assert current
    assert current == AssetReference(guid="mesh-guid", path_hint="Assets/Models/New.fbx")
