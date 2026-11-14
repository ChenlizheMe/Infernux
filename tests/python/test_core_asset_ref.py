"""GUID reference resolution, cache invalidation and typed identity."""

from infernux.core.asset_ref import AssetRefBase


def test_reference_cache_follows_guid_changes_and_explicit_invalidation(monkeypatch, tmp_path):
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

    import json
    from infernux.application import Application
    from infernux.core.assets import AssetManager
    from infernux.core.asset_ref import AnimStateMachineRef, TimelineFSMRef
    from infernux.core.anim_state_machine import AnimStateMachine

    path = tmp_path / "shared.animfsm"
    path.write_text(json.dumps(AnimStateMachine(name="shared").to_dict()), encoding="utf-8")
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: False))
    monkeypatch.setattr(AssetManager, "_cache", {})
    monkeypatch.setattr(AssetManager, "_texture_cache", {})
    monkeypatch.setattr(AssetManager, "_get_path_from_guid", classmethod(lambda cls, guid: str(path)))
    refs = [kind(guid="6" * 32) for kind in (AnimStateMachineRef, AnimStateMachineRef, TimelineFSMRef)]
    shared = refs[0].resolve()
    assert shared is not None
    assert all(ref.resolve() is shared for ref in refs)
    refs[1].invalidate()
    assert refs[1].resolve() is shared
    document = shared.to_dict()
    document["mode"] = "3d"
    path.write_text(json.dumps(document), encoding="utf-8")
    AssetManager.invalidate("6" * 32)
    for ref in refs:
        ref.invalidate()
    replacement = refs[0].resolve()
    assert replacement is not shared and replacement.mode == "3d"
    assert all(ref.resolve() is replacement for ref in refs)


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
