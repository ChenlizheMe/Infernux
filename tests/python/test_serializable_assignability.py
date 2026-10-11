"""Nested documents obey declared types before any owner state is published."""
import copy
import json

import pytest

from infernux.components import InxComponent, SerializableObject, serialized_field
from infernux.components.fields import FieldType, get_serialized_fields
from infernux.components.serializable_object import get_serializable_type_id
from infernux.components.value_codec import VALUE_CODECS
from infernux.components.value_document import make_serializable_object
from infernux.engine.component_restore import (
    PythonComponentRestoreError, deserialize_game_object_document_transactionally,
    deserialize_scene_document_transactionally,
)
from infernux.engine.scene_authoring import decode_scene_document


class ContractStats(SerializableObject):
    health: int = 10


class ContractExtendedStats(ContractStats):
    armor: int = 3


class ContractUnrelated(SerializableObject):
    health: int = 999


class ContractGroup(SerializableObject):
    child: ContractStats
    members: list[ContractStats]


class AssignableComponent(InxComponent):
    marker: int = 7
    child: ContractStats
    members: list[ContractStats]
    group: ContractGroup


class AssignableData(SerializableObject):
    marker: int = 7
    child: ContractStats
    members: list[ContractStats]
    group: ContractGroup


@pytest.mark.parametrize("boundary", ["component", "data", "object", "scene"])
@pytest.mark.parametrize("location", ["child", "members", "group.child", "group.members"])
def test_unrelated_document_is_rejected_before_owner_publication(scene, boundary, location):
    component = scene.create_game_object("Declared type").add_component(AssignableComponent)
    if boundary == "data":
        owner = AssignableData()
        before = owner._serialize()
        document = copy.deepcopy(before)
        fields = document["fields"]
        restore = lambda: SerializableObject._deserialize(document)
        snapshot = owner._serialize
    elif boundary == "scene":
        before = scene.serialize_document()
        document = copy.deepcopy(before)
        fields = document["objects"][0]["components"][0]["data"]
        restore = lambda: deserialize_scene_document_transactionally(scene, document)
        snapshot = scene.serialize_document
    elif boundary == "object":
        owner = component.game_object
        before = owner.serialize_document()
        document = copy.deepcopy(before)
        fields = document["components"][0]["data"]
        restore = lambda: deserialize_game_object_document_transactionally(owner, document)
        snapshot = owner.serialize_document
    else:
        owner = component
        before = owner._serialize_fields_document()
        document = copy.deepcopy(before)
        fields = document
        restore = lambda: owner._deserialize_fields_document(document)
        snapshot = owner._serialize_fields_document
    fields["marker"] = 99
    if location.startswith("group."):
        fields = fields["group"]["fields"]
    wrong = ContractUnrelated()._serialize()
    field = location.rsplit(".", 1)[-1]
    fields[field] = [ContractStats(health=31)._serialize(), wrong] if field == "members" else wrong
    with pytest.raises((TypeError, PythonComponentRestoreError), match="expected ContractStats"):
        restore()
    assert snapshot() == before
    assert component.marker == 7


def test_wrong_type_is_rejected_before_its_defaults_are_materialized(monkeypatch):
    from infernux.components import fields
    calls = []

    class WrongDefaults(SerializableObject):
        value = serialized_field(default_factory=ContractStats)

    copy_default = fields.copy_serialized_field_default

    def record_default(metadata):
        calls.append(metadata.name)
        return copy_default(metadata)

    monkeypatch.setattr(fields, "copy_serialized_field_default", record_default)
    document = make_serializable_object(get_serializable_type_id(WrongDefaults), {})
    with pytest.raises(TypeError, match="expected ContractStats"):
        VALUE_CODECS.decode(document, get_serialized_fields(AssignableComponent)["child"])
    assert calls == []


@pytest.mark.parametrize("preserve_ids", [True, False])
def test_script_rename_keeps_normalized_fields_during_reference_remap(scene, tmp_path, preserve_ids):
    from infernux.components.component_identity import bind_asset_script_guid
    from infernux.components.script_loader import load_component_class_from_file

    script = tmp_path / "rename_remap.py"
    script.write_text(
        "from infernux.components import InxComponent\n"
        "class BeforeRename(InxComponent):\n    value: int = 5\n",
        encoding="utf-8",
    )
    component_type = load_component_class_from_file(str(script))
    guid = "d" * 32
    bind_asset_script_guid(component_type, guid)
    owner = scene.create_game_object("Renamed script")
    component = owner.add_component(component_type)
    component._script_guid = guid
    component._script_path = str(script)
    component.value = 83
    document = owner.serialize_document()
    script.write_text(script.read_text(encoding="utf-8").replace("BeforeRename", "AfterRename"), encoding="utf-8")
    # Scene restore consumes the latest published script revision, just as
    # the asset watcher publishes it before a document is opened or cloned.
    renamed_type = load_component_class_from_file(str(script))
    bind_asset_script_guid(renamed_type, guid)
    from infernux.components.registry import publish_component_script_types
    publish_component_script_types(str(script), (renamed_type,))

    class ScriptCatalog:
        def get_path_from_guid(self, requested):
            assert requested == guid
            return str(script)

        def get_guid_from_path(self, requested):
            assert requested == str(script)
            return guid

    assert deserialize_game_object_document_transactionally(
        owner, document, ScriptCatalog(), preserve_document_ids=preserve_ids,
    )
    restored, = owner.get_py_components()
    assert type(restored).__name__ == "AfterRename"
    assert restored.value == 83
    assert not scene.has_pending_py_components()


@pytest.mark.parametrize("kind", [ContractStats, ContractExtendedStats, None])
@pytest.mark.parametrize("boundary", ["data", "scene"])
def test_polymorphic_and_explicit_null_values_survive_real_document_roundtrip(scene, tmp_path, kind, boundary):
    owner = (AssignableData() if boundary == "data" else
             scene.create_game_object("Typed scene").add_component(AssignableComponent))
    owner.child = None if kind is None else kind(health=31)
    owner.members = [None, None if kind is None else kind(health=41)]
    if boundary == "data":
        document = json.loads(json.dumps(owner._serialize()))
        restored = SerializableObject._deserialize(document)
        assert restored._serialize() == document
    else:
        path = tmp_path / "Typed.scene"
        assert scene.save_to_file(str(path))
        document = decode_scene_document(json.loads(path.read_text(encoding="utf-8")))
        assert deserialize_scene_document_transactionally(scene, document)
        restored = scene.find("Typed scene").get_component(AssignableComponent)
    assert restored.members[0] is None
    if kind is None:
        assert restored.child is None and restored.members[1] is None
    else:
        assert type(restored.child) is kind and restored.child.health == 31
        assert type(restored.members[1]) is kind and restored.members[1].health == 41


def test_unrestricted_serializable_field_still_allows_any_registered_type():
    value = ContractUnrelated(health=27)
    restored = VALUE_CODECS.decode(value._serialize(), FieldType.SERIALIZABLE_OBJECT)
    assert type(restored) is ContractUnrelated and restored.health == 27
