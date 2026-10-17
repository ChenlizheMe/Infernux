def test_material_slot_structured_reference_uses_guid(monkeypatch):
    from types import SimpleNamespace
    import infernux.lib as lib
    from infernux.engine.ui import _inspector_extra_renderers as inspector

    monkeypatch.setattr(
        lib,
        "AssetRegistry",
        SimpleNamespace(instance=lambda: SimpleNamespace(get_asset_database=lambda: object())),
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
