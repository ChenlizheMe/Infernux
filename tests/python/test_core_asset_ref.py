"""GUID reference resolution, cache invalidation and typed identity."""

from infernux.core.asset_ref import AssetRefBase


def test_reference_cache_follows_guid_changes_and_explicit_invalidation():
    calls = []

    class ResolvingReference(AssetRefBase):
        def _do_resolve(self):
            value = object()
            calls.append((self.guid, value))
            return value

    reference = ResolvingReference()
    assert not reference and reference.resolve() is None
    reference.guid = "first-guid"
    first = reference.resolve()
    assert reference.resolve() is first
    reference.guid = "first-guid"
    assert reference.resolve() is first and len(calls) == 1
    reference.guid = "second-guid"
    second = reference.resolve()
    assert second is not first and calls[-1][0] == "second-guid"
    reference.invalidate()
    assert reference.resolve() is not second and len(calls) == 3


def test_asset_reference_documents_persist_guid_and_ignore_display_paths():
    reference = AssetRefBase(guid="asset-guid", path_hint="Assets/Display.png")
    assert reference.to_dict() == {"guid": "asset-guid"}
    restored = AssetRefBase.from_dict(
        {
            "guid": "asset-guid",
            "path_hint": "Assets/Obsolete.png",
            "path": "old",
        }
    )
    assert restored.guid == reference.guid and restored.path_hint == ""
    assert reference.display_name == "Display.png"
    assert "GUID:" in AssetRefBase(guid="abcdefgh1234").display_name
