from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from Infernux.components import (
    FieldType,
    SerializableObject,
    serialized_field,
)
from Infernux.core import AssetManager, DataAsset, DataAssetRef
from Infernux.core.data_asset import (
    decode_data_asset_artifact,
    encode_data_asset_artifact,
    get_registered_data_asset_types,
)


class DataAssetStats(SerializableObject):
    __serialized_type_id__ = "tests.data_asset.stats"

    health: int = serialized_field(default=100)
    label: str = serialized_field(default="default")


class CharacterDataAsset(DataAsset):
    __serialized_type_id__ = "tests.data_asset.character"

    stats: DataAssetStats = serialized_field(default=DataAssetStats())
    tags: list[str] = serialized_field(
        default=[], field_type=FieldType.LIST, element_type=FieldType.STRING
    )
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


def test_data_asset_subclasses_require_explicit_stable_type_id():
    with pytest.raises(TypeError, match="explicit stable"):
        class MissingTypeId(DataAsset):
            value: int = serialized_field(default=0)


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


def test_data_asset_load_consumes_only_current_fields_and_ignores_extra_data():
    authored = {
        "$type": "data_asset",
        "type_id": CurrentDataAsset.__serialized_type_id__,
        "schema_version": 999,
        "removed_envelope_value": "ignored",
        "fields": {"speed": 7, "removed_value": "old"},
    }

    restored = DataAsset.from_document(authored)
    assert type(restored) is CurrentDataAsset
    assert restored.move_speed == pytest.approx(4.0)
    assert restored.enabled is True
    current = restored.serialize_document()
    assert current == {
        "$type": "data_asset",
        "type_id": CurrentDataAsset.__serialized_type_id__,
        "fields": {"move_speed": 4.0, "enabled": True},
    }


def test_data_asset_cook_normalizes_to_current_fields_only():
    authored = {
        "$type": "data_asset",
        "type_id": CurrentDataAsset.__serialized_type_id__,
        "schema_version": 2,
        "fields": {"move_speed": 9.5, "removed": "ignored"},
    }
    cooked = decode_data_asset_artifact(encode_data_asset_artifact(authored))
    assert "schema_version" not in cooked
    assert cooked["fields"] == {"move_speed": 9.5, "enabled": True}


def test_data_asset_current_field_normalization_recurses_into_nested_data():
    authored = {
        "$type": "data_asset",
        "type_id": CurrentContainer.__serialized_type_id__,
        "fields": {
            "nested": {
                "$type": "serializable_object",
                "type_id": CurrentNested.__serialized_type_id__,
                "schema_version": 7,
                "fields": {"old_value": 17, "current_value": 8},
            }
        },
    }
    restored = DataAsset.from_document(authored)
    assert restored.nested.current_value == 8
    current = restored.serialize_document()
    assert "schema_version" not in current
    assert "schema_version" not in current["fields"]["nested"]
    assert current["fields"]["nested"]["fields"] == {"current_value": 8}


def test_data_asset_current_typed_asset_reference_is_preserved():
    from Infernux.components.fields import get_raw_field_value

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


def test_serializable_object_default_factory_is_independent_and_declarative():
    first = FactoryDataAsset()
    second = FactoryDataAsset()
    first.stats.health = 12
    assert second.stats.health == 100

    class CustomInit(SerializableObject):
        __serialized_type_id__ = "tests.data_asset.custom_init"

        value: int = serialized_field(default=1)

        def __init__(self):
            super().__init__()

    with pytest.raises(TypeError, match="custom __init__"):
        serialized_field(
            default_factory=CustomInit,
            field_type=FieldType.SERIALIZABLE_OBJECT,
            serializable_class=CustomInit,
        )


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
    from Infernux.engine.interaction import (
        DocumentKey,
        DocumentKind,
        DocumentRegistry,
        ensure_editable_resource_document,
    )
    from Infernux.engine.undo import UndoManager

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
        assert document.key == DocumentKey.asset(
            DocumentKind.DATA_ASSET, "character-guid"
        )
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
        from Infernux.core.assets import AssetManager

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


def test_data_asset_play_cache_isolated_from_authored_identity(engine, monkeypatch):
    _install_database(engine, monkeypatch)
    authored = CharacterDataAsset(speed=5.0)
    authored._bind_asset("Assets/Character.inxdata", "character-guid")
    AssetManager._put_cache(authored.guid, authored)
    reference = DataAssetRef(guid=authored.guid)
    assert reference.resolve() is authored

    AssetManager._begin_play_data_asset_isolation()
    try:
        runtime = AssetManager._get_cached(authored.guid)
        assert runtime is not authored
        assert type(runtime) is CharacterDataAsset
        assert runtime.guid == authored.guid
        assert runtime.file_path == authored.file_path
        assert reference.resolve() is runtime

        runtime.speed = 19.0
        runtime.stats.health = 1
        assert AssetManager._get_cached(authored.guid) is runtime
        assert authored.speed == 5.0
        assert authored.stats.health == 100
        with pytest.raises(RuntimeError, match="Play-isolated"):
            runtime.save()
    finally:
        AssetManager._end_play_data_asset_isolation()

    assert AssetManager._get_cached(authored.guid) is authored
    assert reference.resolve() is authored
    AssetManager.flush()


def test_data_asset_first_loaded_during_play_is_discarded_on_stop(engine, monkeypatch):
    database = _install_database(engine, monkeypatch)
    target = Path(database.assets_root) / "PlayOnlyData.inxdata"
    source = CharacterDataAsset(speed=7.0)
    try:
        source.save_to(str(target))
        guid = source.guid
        AssetManager.invalidate(guid)

        AssetManager._begin_play_data_asset_isolation()
        runtime = AssetManager.load_by_guid(guid, asset_type=DataAsset)
        assert runtime is not None
        assert runtime.speed == 7.0
        runtime.speed = 33.0
        AssetManager._end_play_data_asset_isolation()

        authored = AssetManager.load_by_guid(guid, asset_type=DataAsset)
        assert authored is not runtime
        assert authored.speed == 7.0
    finally:
        AssetManager._end_play_data_asset_isolation()
        AssetManager.flush()
        if database.contains_path(str(target)):
            database.delete_asset(str(target))


def test_data_asset_import_publishes_nested_asset_dependencies(engine, monkeypatch):
    from Infernux.lib import AssetDependencyGraph

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

        assert AssetDependencyGraph.instance().get_dependencies(owner.guid) == {
            target.guid
        }
    finally:
        AssetManager.flush()
        for path in (owner_path, target_path):
            if database.contains_path(str(path)):
                database.delete_asset(str(path))


def test_data_asset_catalog_and_project_creation_use_published_type(engine, monkeypatch):
    database = _install_database(engine, monkeypatch)
    target = Path(database.assets_root) / "NewCharacterData.inxdata"
    from Infernux.engine.ui.project_file_ops import create_data_asset

    assert (CharacterDataAsset.__serialized_type_id__, CharacterDataAsset) in (
        get_registered_data_asset_types()
    )
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

    from Infernux.application import Application

    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    with pytest.raises(RuntimeError, match="read-only in Player"):
        source.save_to(str(Path(database.assets_root) / "player.inxdata"))
