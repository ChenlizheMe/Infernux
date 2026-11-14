from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from infernux.components import (
    FieldType,
    SerializableObject,
    serialized_field,
)
from infernux.core import AssetManager, DataAsset, DataAssetRef
from infernux.core.data_asset import (
    decode_data_asset_artifact,
    encode_data_asset_artifact,
    get_registered_data_asset_types,
)
from tests.python.test_player_resident_scene_api import (
    _configure_player_contract,
    player_runtime,
)


class DataAssetStats(SerializableObject):
    __serialized_type_id__ = "tests.data_asset.stats"

    health: int = serialized_field(default=100)
    label: str = serialized_field(default="default")


class CharacterDataAsset(DataAsset):
    __serialized_type_id__ = "tests.data_asset.character"

    stats: DataAssetStats = serialized_field(default=DataAssetStats())
    tags: list[str] = serialized_field(default=[], field_type=FieldType.LIST, element_type=FieldType.STRING)
    speed: float = serialized_field(default=4.0)


class DataAssetLink(DataAsset):
    __serialized_type_id__ = "tests.data_asset.link"

    target: DataAssetRef = serialized_field(
        default=DataAssetRef(),
        field_type=FieldType.ASSET,
        asset_type="DataAsset",
    )


class FactoryDataAsset(DataAsset):
    __serialized_type_id__ = "tests.data_asset.factory"

    stats: DataAssetStats = serialized_field(
        default_factory=DataAssetStats,
        field_type=FieldType.SERIALIZABLE_OBJECT,
        serializable_class=DataAssetStats,
    )


class CurrentDataAsset(DataAsset):
    __serialized_type_id__ = "tests.data_asset.current"

    move_speed: float = 4.0
    enabled: bool = serialized_field(default=True)


class CurrentNested(SerializableObject):
    __serialized_type_id__ = "tests.data_asset.current_nested"

    current_value: int = 3


class CurrentContainer(DataAsset):
    __serialized_type_id__ = "tests.data_asset.current_container"

    nested: CurrentNested = serialized_field(default=CurrentNested())


class CurrentLink(DataAsset):
    __serialized_type_id__ = "tests.data_asset.current_link"

    target: DataAssetRef = serialized_field(
        default=DataAssetRef(),
        field_type=FieldType.ASSET,
        asset_type="DataAsset",
    )


def _install_database(engine, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    return database


def _activate_data_asset_player(player, scene, database, source):
    from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog

    # This cache-domain test uses the real imported source document as its
    # cataloged payload; binary cooking has its own contract test above.
    relative = Path(source.file_path).relative_to(database.project_root).as_posix()
    assert database.get_guid_from_path(source.file_path) == source.guid
    artifact_id = f"data-asset:{source.guid}"
    catalog = PlayerRuntimeAssetCatalog.from_documents(
        database.project_root,
        {"artifacts": [{
            "runtime_artifact_id": artifact_id,
            "runtime_path": relative,
            "package": "Game_Data/Content.inxpkg",
            "logical_type": "data_asset",
            "asset_guid": source.guid,
            "dependencies": [],
        }]},
        {"entries": [{
            "guid": source.guid,
            "runtime_path": relative,
            "primary_runtime_artifact_id": artifact_id,
            "runtime_artifact_ids": [artifact_id],
            "dependencies": [],
        }]},
    )
    scene.create_game_object("DataAssetIsolationWitness")
    _configure_player_contract(player.session, catalog)
    assert player.session.activate()
    identity = player.session._runtime_world_id
    return identity, player.host._world_runtimes[identity]


def _stop_data_asset_player(player):
    from infernux.application import Application
    from infernux.lib import SceneManager

    player.session.shutdown()
    native = SceneManager.instance()
    assert native._collect_retired_runtime_worlds()
    assert not player.host._world_runtimes
    assert native._current_runtime_world_id == native._authoring_runtime_world_id
    player.session.release_content()
    # End this borrowed Player companion's process role before checking the
    # surviving author catalog. Its shared fixture restores the previous role.
    Application._unbind_engine(player.host)


def test_data_asset_document_round_trip_is_strict_and_polymorphic():
    source = CharacterDataAsset(
        stats=DataAssetStats(health=72, label="runner"),
        tags=["campaign", "hard"],
        speed=8.5,
    )

    document = source.serialize_document()
    assert set(document) == {"$type", "type_id", "fields"}
    assert document["$type"] == "data_asset"
    assert document["type_id"] == CharacterDataAsset.__serialized_type_id__

    restored = DataAsset.from_document(document)
    assert type(restored) is CharacterDataAsset
    assert restored.stats.health == 72
    assert restored.stats.label == "runner"
    assert restored.tags == ["campaign", "hard"]
    assert restored.speed == 8.5

    with pytest.raises(ValueError, match="unknown SerializableObject"):
        DataAsset.from_document({**document, "type_id": "tests.missing.Type"})


def test_data_asset_current_typed_asset_reference_is_preserved():
    from infernux.components.fields import get_raw_field_value

    authored = {
        "$type": "data_asset",
        "type_id": CurrentLink.__serialized_type_id__,
        "fields": {
            "target": {
                "$type": "asset_ref",
                "asset_type": "DataAsset",
                "guid": "target-guid",
                "path_hint": "Assets/Config/Target.inxdata",
            }
        },
    }
    restored = DataAsset.from_document(authored)
    reference = get_raw_field_value(restored, "target")
    assert reference.guid == "target-guid"
    assert reference.path_hint == ""


def test_data_asset_document_application_preserves_persistent_identity():
    source = CharacterDataAsset(speed=3.0)
    source._bind_asset("Assets/Character.inxdata", "character-guid")
    edited = CharacterDataAsset(
        stats=DataAssetStats(health=44, label="edited"),
        tags=["updated"],
        speed=9.0,
    )

    assert source.deserialize_document(edited.serialize_document())
    assert source.guid == "character-guid"
    assert source.file_path.endswith("Assets\\Character.inxdata") or source.file_path.endswith(
        "Assets/Character.inxdata"
    )
    assert source.stats.health == 44
    assert source.tags == ["updated"]
    assert source.speed == 9.0


def test_data_asset_player_artifact_is_binary_and_deterministic():
    source = CharacterDataAsset(
        stats=DataAssetStats(health=72, label="binary"),
        tags=["player"],
        speed=7.5,
    )
    document = source.serialize_document()

    first = encode_data_asset_artifact(document)
    second = encode_data_asset_artifact(document)
    assert first == second
    assert first.startswith(b"INXDATA\0")
    assert b'"type_id"' not in first
    assert decode_data_asset_artifact(first) == document

    with pytest.raises(ValueError, match="unsupported DataAsset artifact"):
        decode_data_asset_artifact(b"NOTDATA!" + first[8:])


def test_data_asset_uses_shared_document_transaction_and_undo():
    from infernux.engine.interaction import (
        DocumentKey,
        DocumentKind,
        DocumentRegistry,
        ensure_editable_resource_document,
    )
    from infernux.engine.undo import UndoManager

    previous_registry = DocumentRegistry._instance
    previous_manager = UndoManager._instance
    registry = DocumentRegistry()
    manager = UndoManager()
    source = CharacterDataAsset(speed=3.0)
    source._bind_asset("Assets/Character.inxdata", "character-guid")
    state = SimpleNamespace(resource_controller=None, settings=source)
    controller = ensure_editable_resource_document(
        category="data_asset",
        document_kind=DocumentKind.DATA_ASSET,
        file_path=source.file_path,
        resource=source,
        guid=source.guid,
        view_id="inspector",
        state=state,
    )
    try:
        edited = source.instantiate()
        edited.speed = 11.0
        assert controller.apply_document(
            edited.serialize_document(),
            view_id="inspector",
            edit_key="data_asset.speed",
            description="Set speed",
        )
        document = registry.require(controller.document_id)
        assert document.key == DocumentKey.asset(DocumentKind.DATA_ASSET, "character-guid")
        assert state.settings is source
        assert controller.resource is source
        assert source.speed == 11.0
        assert document.is_dirty

        manager.undo()
        assert source.speed == 3.0
        assert source.guid == "character-guid"
        manager.redo()
        assert source.speed == 11.0
    finally:
        from infernux.core.assets import AssetManager

        AssetManager.cancel_scheduled_save(source.file_path)
        DocumentRegistry._instance = previous_registry
        UndoManager._instance = previous_manager


def test_data_asset_save_load_cache_reference_and_explicit_copy(engine, monkeypatch):
    database = _install_database(engine, monkeypatch)
    target = Path(database.assets_root) / "DataAssetContract.inxdata"
    source = CharacterDataAsset(
        stats=DataAssetStats(health=91, label="shared"),
        tags=["one"],
        speed=6.25,
    )

    try:
        assert source.save_to(str(target)) == str(target.resolve())
        assert source.is_persistent
        assert source.guid == database.get_guid_from_path(str(target))
        assert json.loads(target.read_text(encoding="utf-8"))["$type"] == "data_asset"

        shared = AssetManager.load_by_guid(source.guid, asset_type=DataAsset)
        assert shared is source
        assert DataAssetRef(guid=source.guid).resolve() is source

        copy = source.instantiate()
        assert type(copy) is CharacterDataAsset
        assert not copy.is_persistent
        copy.stats.health = 12
        copy.tags.append("two")
        assert source.stats.health == 91
        assert source.tags == ["one"]

        AssetManager.invalidate(source.guid)
        loaded = DataAsset.load(str(target))
        assert type(loaded) is CharacterDataAsset
        assert loaded.guid == source.guid
        assert loaded.stats.label == "shared"
        assert loaded.tags == ["one"]
    finally:
        AssetManager.flush()
        if database.contains_path(str(target)):
            database.delete_asset(str(target))


def test_data_asset_play_cache_isolated_from_authored_identity(engine, monkeypatch, scene, request):
    from infernux.lib import _Infernux

    database = _install_database(engine, monkeypatch)
    target = Path(database.assets_root) / "PlayIsolatedCharacter.inxdata"
    authored = CharacterDataAsset(speed=5.0)
    player = None
    try:
        authored.save_to(str(target))
        reference = DataAssetRef(guid=authored.guid)
        assert reference.resolve() is authored
        # Save/import belongs to authoring, before the normal Player fixture
        # binds its read-only process role. Do not create a second native host.
        player = request.getfixturevalue("player_runtime")
        identity, domain = _activate_data_asset_player(player, scene, database, authored)
        with _Infernux._RuntimeWorldScope(identity):
            runtime = AssetManager._get_cached(authored.guid)
            assert runtime is not authored
            assert type(runtime) is domain.serializable_types[CharacterDataAsset.__serialized_type_id__]
            assert type(runtime) is not CharacterDataAsset
            assert type(runtime.stats) is domain.serializable_types[DataAssetStats.__serialized_type_id__]
            assert runtime.guid == authored.guid
            assert runtime.file_path == authored.file_path
            assert reference.resolve() is runtime

            runtime.speed = 19.0
            runtime.stats.health = 1
            assert runtime.speed == 19.0
            assert runtime.stats.health == 1
            assert AssetManager._get_cached(authored.guid) is runtime
            assert authored.speed == 5.0
            assert authored.stats.health == 100
            assert runtime._play_isolated
            with pytest.raises(RuntimeError, match="read-only in Player"):
                runtime.save()

        _stop_data_asset_player(player)
        assert not domain.data_assets
        assert AssetManager._get_cached(authored.guid) is authored
        assert reference.resolve() is authored
        # The original Play-isolated write restriction survives Stop too;
        # the held runtime clone must never overwrite its authored identity.
        with pytest.raises(RuntimeError, match="Play-isolated"):
            runtime.save()
    finally:
        if player is not None:
            _stop_data_asset_player(player)
        AssetManager.flush()
        if database.contains_path(str(target)):
            database.delete_asset(str(target))


def test_data_asset_first_loaded_during_play_is_discarded_on_stop(engine, monkeypatch, scene, request):
    from infernux.lib import _Infernux

    database = _install_database(engine, monkeypatch)
    target = Path(database.assets_root) / "PlayOnlyData.inxdata"
    source = CharacterDataAsset(speed=7.0)
    player = None
    try:
        source.save_to(str(target))
        guid = source.guid
        AssetManager.invalidate(guid)
        assert AssetManager._get_cached(guid) is None
        player = request.getfixturevalue("player_runtime")
        identity, domain = _activate_data_asset_player(player, scene, database, source)
        with _Infernux._RuntimeWorldScope(identity):
            assert guid not in domain.data_assets
            runtime = AssetManager.load_by_guid(guid, asset_type=DataAsset)
            assert runtime is not None
            assert type(runtime) is domain.serializable_types[CharacterDataAsset.__serialized_type_id__]
            assert type(runtime) is not CharacterDataAsset
            assert type(runtime.stats) is domain.serializable_types[DataAssetStats.__serialized_type_id__]
            assert runtime.speed == 7.0
            runtime.speed = 33.0
            assert runtime.speed == 33.0
            assert domain.data_assets[guid] is runtime
        _stop_data_asset_player(player)
        assert not domain.data_assets
        assert AssetManager._get_cached(guid) is None
        authored = AssetManager.load_by_guid(guid, asset_type=DataAsset)
        assert authored is not runtime
        assert type(authored) is CharacterDataAsset
        assert type(authored.stats) is DataAssetStats
        assert authored.speed == 7.0
    finally:
        if player is not None:
            _stop_data_asset_player(player)
        AssetManager.flush()
        if database.contains_path(str(target)):
            database.delete_asset(str(target))


def test_data_asset_import_publishes_nested_asset_dependencies(engine, monkeypatch):
    from infernux.lib import AssetDependencyGraph

    database = _install_database(engine, monkeypatch)
    target_path = Path(database.assets_root) / "DataAssetDependency.inxdata"
    owner_path = Path(database.assets_root) / "DataAssetOwner.inxdata"
    target = CharacterDataAsset()
    try:
        target.save_to(str(target_path))
        owner = DataAssetLink(
            target=DataAssetRef(
                guid=target.guid,
                path_hint="Assets/DataAssetDependency.inxdata",
            )
        )
        owner.save_to(str(owner_path))

        assert AssetDependencyGraph.instance().get_dependencies(owner.guid) == {target.guid}
    finally:
        AssetManager.flush()
        for path in (owner_path, target_path):
            if database.contains_path(str(path)):
                database.delete_asset(str(path))


def test_data_asset_catalog_and_project_creation_use_published_type(engine, monkeypatch):
    database = _install_database(engine, monkeypatch)
    target = Path(database.assets_root) / "NewCharacterData.inxdata"
    from infernux.engine.ui.project_file_ops import create_data_asset

    assert (CharacterDataAsset.__serialized_type_id__, CharacterDataAsset) in (get_registered_data_asset_types())
    try:
        created, error = create_data_asset(
            database.assets_root,
            "NewCharacterData",
            CharacterDataAsset.__serialized_type_id__,
            database,
        )
        assert created, error
        loaded = DataAsset.load(str(target))
        assert type(loaded) is CharacterDataAsset
        assert loaded.stats.health == 100

        created, error = create_data_asset(
            database.assets_root,
            "UnknownData",
            "tests.data_asset.missing",
            database,
        )
        assert not created
        assert "Unknown DataAsset type" in error
    finally:
        AssetManager.flush()
        if database.contains_path(str(target)):
            database.delete_asset(str(target))


def test_data_asset_authoring_rejects_wrong_location_extension_and_player(engine, monkeypatch):
    database = _install_database(engine, monkeypatch)
    source = CharacterDataAsset()

    with pytest.raises(ValueError, match="Assets or Packages"):
        source.save_to(str(Path(database.project_root) / "ProjectSettings" / "outside.inxdata"))
    with pytest.raises(ValueError, match=r"\.inxdata extension"):
        source.save_to(str(Path(database.assets_root) / "wrong.json"))

    from infernux.application import Application

    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    with pytest.raises(RuntimeError, match="read-only in Player"):
        source.save_to(str(Path(database.assets_root) / "player.inxdata"))
