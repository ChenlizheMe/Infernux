import json

import pytest

from Infernux.core.asset_ref import RenderEffectRef
from Infernux.core.assets import AssetManager
from Infernux.renderstack.effect_stage import EffectResourceContract, EffectScope, EffectStage
from Infernux.renderstack.render_effect_asset import (
    EffectAssetReference,
    RenderEffectAsset,
    RenderEffectGroupAsset,
    RenderEffectGroupEntry,
    direct_effect_dependencies,
    dump_render_effect_document,
    parse_render_effect_document,
)
from Infernux.renderstack.render_effect import EditableRenderEffectGroup


def test_effect_stage_has_stable_identity_scope_and_contract():
    stage = EffectStage(
        "opaque.toon_finish",
        EffectScope.ROUTE,
        contract=EffectResourceContract(
            inputs={"color", "depth"},
            outputs={"color"},
            capabilities={"hdr", "isolated_target"},
        ),
    )

    assert stage.display_name == "Opaque.Toon Finish"
    assert stage.stable_id == "opaque.toon_finish"
    assert stage.contract.inputs == frozenset({"color", "depth"})


@pytest.mark.parametrize("stable_id", ["", "Final", "1final", "final stage", "final-"])
def test_effect_stage_rejects_unstable_identifiers(stable_id):
    with pytest.raises(ValueError):
        EffectStage(stable_id, EffectScope.COMPOSITE)


def test_render_effect_round_trips_as_deterministic_json():
    document = RenderEffectAsset(
        feature_type="infernux.post.bloom",
        parameters={"threshold": 1.0, "intensity": 0.6},
        dependencies=(
            EffectAssetReference(guid="bloom-material-guid"),
        ),
    )

    encoded = dump_render_effect_document(document)
    decoded = parse_render_effect_document(encoded)

    assert decoded == document
    assert encoded.endswith("\n")
    assert list(json.loads(encoded)) == [
        "$schema",
        "dependencies",
        "feature_type",
        "parameters",
    ]


def test_render_effect_group_preserves_order_and_direct_dependencies():
    bloom = EffectAssetReference(guid="bloom-guid")
    tone = EffectAssetReference(guid="tone-guid")
    group = RenderEffectGroupAsset(
        entries=(
            RenderEffectGroupEntry("bloom", bloom, overrides={"intensity": 0.8}),
            RenderEffectGroupEntry("tonemapping", tone),
        )
    )

    restored = parse_render_effect_document(dump_render_effect_document(group))

    assert restored == group
    assert direct_effect_dependencies(restored) == (bloom, tone)


def test_editable_render_effect_group_replaces_its_strict_document(tmp_path):
    initial = RenderEffectGroupAsset()
    resource = EditableRenderEffectGroup(
        initial,
        file_path=str(tmp_path / "Post.effectgroup"),
        guid="group-guid",
    )
    document = RenderEffectGroupAsset(
        entries=(
            RenderEffectGroupEntry(
                "bloom",
                EffectAssetReference(guid="bloom-guid"),
            ),
        )
    ).to_dict()

    assert resource.deserialize_document(document)
    assert resource.serialize_document() == document
    assert resource.entries[0].entry_id == "bloom"


def test_render_effect_reference_resolves_effect_group_as_a_live_resource(
    monkeypatch, tmp_path
):
    path = tmp_path / "Assets" / "Default Post Processing.effectgroup"
    path.parent.mkdir()
    path.write_text(
        dump_render_effect_document(RenderEffectGroupAsset()),
        encoding="utf-8",
    )

    class Database:
        @staticmethod
        def get_guid_from_path(candidate):
            return "group-guid" if str(candidate) == str(path) else ""

        @staticmethod
        def get_path_from_guid(guid):
            return str(path) if guid == "group-guid" else ""

    monkeypatch.setattr(AssetManager, "_asset_database", Database())
    assert AssetManager._type_from_extension(".effectgroup").__name__ == "RenderEffect"
    # Author paths cross the AssetDatabase boundary before the document is
    # loaded. A structured RenderEffectRef never treats its display hint as
    # identity.
    resource = AssetManager.load_by_guid("group-guid")
    reference = RenderEffectRef(path_hint=str(path))

    assert isinstance(resource, EditableRenderEffectGroup)
    assert resource.file_path == str(path)
    assert resource.entries == ()
    assert reference.resolve() is None
    assert not reference


def test_render_effect_documents_consume_only_current_guid_fields():
    effect = parse_render_effect_document(
        {
            "$schema": "infernux.render_effect",
            "feature_type": "infernux.post.bloom",
            "parameters": {},
            "dependencies": [
                {
                    "guid": "shader-guid",
                    "path_hint": "Assets/Shaders/Stale.frag",
                    "legacy_path": "Shaders/Stale.frag",
                }
            ],
            "scope": "final",
            "mount": "route",
        }
    )
    group = parse_render_effect_document(
        {
            "$schema": "infernux.render_effect_group",
            "entries": [
                {
                    "entry_id": "bloom",
                    "asset": {
                        "guid": "effect-guid",
                        "path_hint": "Assets/Effects/Stale.effect",
                        "path": "Effects/Stale.effect",
                    },
                    "enabled": True,
                    "overrides": {},
                    "scope": "final",
                }
            ],
            "legacy_version": 1,
        }
    )

    assert json.loads(dump_render_effect_document(effect))["dependencies"] == [
        {"guid": "shader-guid"}
    ]
    assert json.loads(dump_render_effect_document(group))["entries"][0]["asset"] == {
        "guid": "effect-guid"
    }


def test_render_effect_documents_require_all_current_fields():
    with pytest.raises(ValueError, match="missing required"):
        parse_render_effect_document(
            {
                "$schema": "infernux.render_effect",
                "feature_type": "infernux.post.bloom",
                "parameters": {},
            }
        )


def test_path_only_group_entries_are_not_current_entries():
    document = parse_render_effect_document(
        {
            "$schema": "infernux.render_effect_group",
            "entries": [
                {
                    "entry_id": "legacy",
                    "asset": {"path_hint": "Assets/Effects/Legacy.effect"},
                    "enabled": "obsolete value is not inspected",
                }
            ],
        }
    )

    assert document.entries == ()


def test_render_effect_group_rejects_duplicate_entry_ids():
    reference = EffectAssetReference(guid="same")
    with pytest.raises(ValueError, match="unique"):
        RenderEffectGroupAsset(
            entries=(
                RenderEffectGroupEntry("duplicate", reference),
                RenderEffectGroupEntry("duplicate", reference),
            )
        )


def test_render_effect_asset_does_not_encode_mount_scope():
    effect = RenderEffectAsset(feature_type="project.effects.grayscale")
    serialized = effect.to_dict()

    assert "scope" not in serialized
    assert "stage" not in serialized
    assert "queue" not in serialized


def test_author_path_resolves_guid_from_asset_database(monkeypatch, tmp_path):
    target = tmp_path / "Style Comic Fisheye.effect"
    target.write_text("{}", encoding="utf-8")

    class Database:
        @staticmethod
        def get_guid_from_path(path):
            return (
                "472285e708b5659d12e27eb9138c313a"
                if str(path) == str(target).replace("\\", "/")
                else ""
            )

    monkeypatch.setattr(AssetManager, "_asset_database", Database())

    stamped = EffectAssetReference.from_author_path(str(target))

    assert stamped.guid == "472285e708b5659d12e27eb9138c313a"
    assert stamped.to_dict() == {"guid": "472285e708b5659d12e27eb9138c313a"}


def test_path_only_document_references_do_not_query_catalog_or_create_edges(monkeypatch):
    class Database:
        @staticmethod
        def get_guid_from_path(_path):
            pytest.fail("document parsing must not resolve a path-only field")

    monkeypatch.setattr(AssetManager, "_asset_database", Database())
    effect = parse_render_effect_document(
        {
            "$schema": "infernux.render_effect",
            "feature_type": "infernux.post.bloom",
            "parameters": {},
            "dependencies": [
                {"guid": "", "path_hint": "Assets/Shaders/Legacy.frag"},
                {"path_hint": "Assets/Shaders/AlsoLegacy.frag"},
            ],
        }
    )
    group = parse_render_effect_document(
        {
            "$schema": "infernux.render_effect_group",
            "entries": [
                {
                    "entry_id": "legacy",
                    "asset": {"path_hint": "Assets/Rendering/Legacy.effect"},
                    "enabled": True,
                    "overrides": {},
                }
            ],
        }
    )

    assert direct_effect_dependencies(effect) == ()
    assert direct_effect_dependencies(group) == ()


def test_effect_reference_dump_is_stable_when_asset_path_moves(monkeypatch):
    current_path = ["Assets/Effects/Before.effect"]

    class Database:
        @staticmethod
        def get_guid_from_path(path):
            return "stable-guid" if str(path).replace("\\", "/") == current_path[0] else ""

    monkeypatch.setattr(AssetManager, "_asset_database", Database())
    before_reference = EffectAssetReference.from_author_path(current_path[0])
    before = dump_render_effect_document(
        RenderEffectAsset("infernux.post.bloom", dependencies=(before_reference,))
    )

    current_path[0] = "Assets/Effects/After.effect"
    after_reference = EffectAssetReference.from_author_path(current_path[0])
    after = dump_render_effect_document(
        RenderEffectAsset("infernux.post.bloom", dependencies=(after_reference,))
    )

    assert after == before


def test_direct_construction_rejects_untyped_dependency_values():
    with pytest.raises(TypeError, match="EffectAssetReference"):
        RenderEffectAsset(
            feature_type="infernux.post.bloom",
            dependencies=({"guid": "not-a-reference"},),
        )

    with pytest.raises(TypeError, match="RenderEffectGroupEntry"):
        RenderEffectGroupAsset(entries=({"entry_id": "not-an-entry"},))
