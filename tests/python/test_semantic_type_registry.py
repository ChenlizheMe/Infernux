"""Real native catalog publication of compiled Python field declarations."""

from copy import deepcopy

import pytest

from infernux.components import InxComponent, serialized_field
from infernux.components.fields import get_field_schema
from infernux.lib import _Infernux as native


@pytest.fixture
def declaration(request):
    class CatalogProbe(InxComponent):
        speed: float = serialized_field(default=8.0, range=(0.0, 5.0), tooltip="Speed")

    owner = f"test:semantic-catalog:{request.node.name}"
    document = {
        "type_guid": owner + ":probe",
        "readable_id": owner + ".probe",
        "owner": owner,
        "origin": "python",
        "display_name": "Catalog probe",
        "base_type_guid": "",
        "constructible": True,
        "serializable": True,
        "runtime_available": True,
        "runtime_profiles": ["editor", "player"],
        "lifecycle": [],
        "fields": [get_field_schema(CatalogProbe, "speed").to_document()],
    }
    yield document
    snapshot = native._semantic_catalog_snapshot()
    if document["type_guid"] in snapshot.type_guids:
        native._semantic_catalog_prepare([{"owner": owner, "types": []}]).publish()


def prepare(document):
    return native._semantic_catalog_prepare([{"owner": document["owner"], "types": [document]}])


def test_compiled_fields_publish_without_live_component_or_editor(declaration):
    before = native._semantic_catalog_snapshot()
    expected = deepcopy(declaration)
    publication = prepare(declaration)
    candidate = publication.candidate
    guid = declaration["type_guid"]
    assert guid not in before.type_guids
    assert guid not in native._semantic_catalog_snapshot().type_guids
    assert candidate.revision == before.revision + 1
    expected["revision"] = candidate.revision
    assert candidate.type_document(guid) == expected
    assert expected["fields"][0]["attributes"]["default"] == 5.0

    declaration["fields"][0]["attributes"]["default"] = 99.0
    assert candidate.type_document(guid) == expected
    assert publication.publish() == candidate.revision
    active = native._semantic_catalog_snapshot()
    assert active.type_document(guid) == expected
    exported = active.type_document(guid)
    exported["fields"].clear()
    assert active.type_document(guid) == expected

    retirement = native._semantic_catalog_prepare([{"owner": declaration["owner"], "types": []}])
    assert guid in native._semantic_catalog_snapshot().type_guids
    assert guid not in retirement.candidate.type_guids
    assert retirement.publish() == active.revision + 1
    with pytest.raises(KeyError, match=guid):
        native._semantic_catalog_snapshot().type_document(guid)
    assert active.type_document(guid) == expected
    assert before.revision == candidate.revision - 1


def test_stale_publication_does_not_change_active_catalog(declaration):
    first = prepare(declaration)
    stale = prepare(declaration)
    first.publish()
    active = native._semantic_catalog_snapshot()
    with pytest.raises(RuntimeError, match="stale"):
        stale.publish()
    with pytest.raises(RuntimeError, match="stale"):
        first.publish()
    assert native._semantic_catalog_snapshot().revision == active.revision
    assert native._semantic_catalog_snapshot().type_document(declaration["type_guid"]) == active.type_document(
        declaration["type_guid"]
    )


def test_audio_source_declaration_owns_public_fields_and_coupled_track_count(scene):
    from infernux.components.builtin.audio_source import AudioSource
    from infernux.components.builtin_component import CppProperty
    from infernux.engine.interaction.components import ComponentCommandService
    from infernux.engine.undo import UndoManager

    owner = scene.create_game_object("DeclaredAudioSource")
    native_source = owner.add_component("AudioSource")
    source = AudioSource._get_or_create_wrapper(native_source, owner)
    descriptor = native._semantic_catalog_snapshot().type_document("native:infernux.AudioSource")
    fields = {field["attributes"]["field_id"]: field for field in descriptor["fields"]}
    assert set(fields) == {
        "track_count",
        "volume",
        "pitch",
        "mute",
        "loop",
        "play_on_awake",
        "spatial_blend",
        "min_distance",
        "max_distance",
        "output_bus",
        "one_shot_pool_size",
        "priority",
    }
    document = native_source.serialize_document()
    for name, field in fields.items():
        prop = getattr(AudioSource, name)
        assert isinstance(prop, CppProperty)
        assert prop.schema.to_document() == field
        assert document[field["attributes"]["serialized_name"]] == field["attributes"]["default"]
    assert fields["track_count"]["attributes"]["range"] == [1, 16]
    assert fields["track_count"]["attributes"]["setter_owns_document_shape"] is True
    assert AudioSource.one_shot_pool_size.metadata.hidden is True
    assert fields["priority"]["attributes"]["range"] == [0, 255]

    previous = UndoManager._instance
    manager = UndoManager()
    try:
        assert ComponentCommandService().set_field(source, "track_count", 4)
        changed = native_source.serialize_document()
        assert changed["track_count"] == 4
        assert len(changed["tracks"]) == 4
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        restored = native_source.serialize_document()
        assert restored["track_count"] == 1
        assert len(restored["tracks"]) == 1
    finally:
        manager.clear()
        UndoManager._instance = previous


def test_skinned_renderer_declares_inherited_fields_for_host_schema_and_edits(scene):
    from infernux.components.builtin.skinned_mesh_renderer import SkinnedMeshRenderer
    from infernux.engine.interaction.components import ComponentCommandService
    from infernux.engine.undo import UndoManager
    from infernux.host.editor import EditorAutomationHost

    catalog = native._semantic_catalog_snapshot()
    base = catalog.type_document("native:infernux.MeshRenderer")
    descriptor = catalog.type_document("native:infernux.SkinnedMeshRenderer")
    assert descriptor["base_type_guid"] == base["type_guid"]
    assert descriptor["fields"][:-1] == base["fields"]
    assert descriptor["fields"][-1] == SkinnedMeshRenderer.active_take_name.schema.to_document()

    owner = scene.create_game_object("SkinnedSchemaEdit")
    renderer = owner.add_component("SkinnedMeshRenderer")
    wrapper = SkinnedMeshRenderer._get_or_create_wrapper(renderer, owner)
    schema = EditorAutomationHost.instance().scene_component_schema(owner.id, renderer.component_id)
    assert {field["name"] for field in schema["fields"]} == {
        "enabled", "castShadows", "receivesShadows", "submeshIndex", "meshPivotOffset", "activeTakeName"
    }
    before = renderer.serialize_document()
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        assert ComponentCommandService().set_field(wrapper, "casts_shadows", False)
        assert renderer.serialize_document()["castShadows"] is False
        manager.undo()
        assert renderer.serialize_document() == before
    finally:
        manager.clear()
        UndoManager._instance = previous


def test_host_schema_preserves_python_enum_value_types(scene):
    from enum import Enum, IntEnum
    from infernux.host.editor import EditorAutomationHost

    class Ease(str, Enum):
        Linear = "Linear"
        OutQuad = "OutQuad"

    class Phase(IntEnum):
        Normal = 0
        Late = 1

    class EnumControls(InxComponent):
        ease: Ease = serialized_field(default=Ease.Linear)
        phase: Phase = serialized_field(default=Phase.Normal)

    owner = scene.create_game_object("EnumControls")
    component = owner.add_component(EnumControls)
    schema = EditorAutomationHost.instance().scene_component_schema(owner.id, component.component_id)
    fields = {field["name"]: field for field in schema["fields"]}
    assert fields["ease"]["enum"] == [{"name": "Linear", "value": "Linear"}, {"name": "OutQuad", "value": "OutQuad"}]
    assert fields["phase"]["enum"] == [{"name": "Normal", "value": 0}, {"name": "Late", "value": 1}]
    assert component.ease is Ease.Linear and component.phase is Phase.Normal


def test_box_collider_editor_command_uses_native_schema_material_boundary_and_undo(scene):
    from infernux.components.builtin.box_collider import BoxCollider
    from infernux.core.asset_ref import PhysicMaterialRef
    from infernux.engine.interaction.components import ComponentCommandService
    from infernux.engine.undo import UndoManager
    from infernux.lib import Vector3

    owner = scene.create_game_object("ColliderSchemaEdit")
    native_collider = owner.add_component("BoxCollider")
    collider = BoxCollider._get_or_create_wrapper(native_collider, owner)
    before = native_collider.serialize_document()
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        service = ComponentCommandService()
        assert service.set_field(collider, "center", Vector3(1.0, 2.0, 3.0))
        assert native_collider.serialize_document()["center"] == pytest.approx([1.0, 2.0, 3.0])
        manager.undo()
        assert native_collider.serialize_document() == before

        reference = PhysicMaterialRef(guid="0123456789abcdef0123456789abcdef")
        assert service.set_field(collider, "physic_material", reference)
        assert native_collider.serialize_document()["physic_material_guid"] == reference.guid
        manager.undo()
        assert native_collider.serialize_document() == before
    finally:
        manager.clear()
        UndoManager._instance = previous


@pytest.mark.parametrize("field,candidate", [("collision_detection_mode", native.CollisionDetectionMode.Continuous)])
def test_rigidbody_editor_command_uses_declared_schema_and_undo(scene, monkeypatch, field, candidate):
    from infernux.components.builtin.rigidbody import Rigidbody
    from infernux.engine.interaction.components import ComponentCommandService
    from infernux.engine.interaction import serialized_properties
    from infernux.engine.undo import UndoManager

    obj = scene.create_game_object("RigidbodySchemaEdit")
    rigidbody = obj.add_component("Rigidbody")
    wrapper = Rigidbody._get_or_create_wrapper(rigidbody, obj)
    before = rigidbody.serialize_document()
    descriptor = getattr(Rigidbody, field)
    captured = []
    original = serialized_properties.make_attribute_property_transaction

    def capture(*args, **kwargs):
        captured.append(kwargs["schema"])
        return original(*args, **kwargs)

    monkeypatch.setattr(serialized_properties, "make_attribute_property_transaction", capture)
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        assert ComponentCommandService().set_field(wrapper, field, candidate)
        assert captured == [descriptor.schema]
        assert captured[0] is descriptor.schema
        after = rigidbody.serialize_document()
        assert after != before
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert rigidbody.serialize_document() == before
        manager.redo()
        assert rigidbody.serialize_document() == after
    finally:
        manager.clear()
        UndoManager._instance = previous


@pytest.mark.parametrize(
    ("type_name", "wrapper_module", "wrapper_name"),
    [
        ("HingeJoint", "hinge_joint", "HingeJoint"),
        ("SliderJoint", "slider_joint", "SliderJoint"),
    ],
)
def test_native_joint_component_reference_uses_document_id_and_undo(
    scene,
    type_name,
    wrapper_module,
    wrapper_name,
):
    import importlib

    from infernux.engine.interaction.components import ComponentCommandService
    from infernux.engine.undo import UndoManager

    wrapper_type = getattr(
        importlib.import_module(f"infernux.components.builtin.{wrapper_module}"),
        wrapper_name,
    )
    support = scene.create_game_object(f"{type_name}Support")
    support_native = support.add_component("Rigidbody")
    support.add_component("BoxCollider")
    owner = scene.create_game_object(type_name)
    owner.add_component("Rigidbody")
    owner.add_component("BoxCollider")
    joint_native = owner.add_component(type_name)
    joint = wrapper_type._get_or_create_wrapper(joint_native, owner)

    descriptor = wrapper_type.connected_body
    assert descriptor.schema.attributes["serialized_name"] == "connected_body_component_id"
    assert descriptor.validate_value(joint, support_native) == ""

    previous = UndoManager._instance
    manager = UndoManager()
    try:
        assert ComponentCommandService().set_field(
            joint,
            "connected_body",
            support_native,
        )
        assert joint_native.serialize_document()["connected_body_component_id"] == support_native.component_id
        manager.undo()
        assert joint_native.serialize_document()["connected_body_component_id"] == 0
        manager.redo()
        assert joint_native.serialize_document()["connected_body_component_id"] == support_native.component_id
    finally:
        manager.clear()
        UndoManager._instance = previous


@pytest.mark.parametrize("field,candidate", [("field_of_view", 200.0)])
def test_camera_wrapper_edits_use_native_schema_and_common_codec(scene, monkeypatch, field, candidate):
    from infernux.components.builtin.camera import Camera
    from infernux.components.value_codec import VALUE_CODECS
    from infernux.engine.interaction.components import ComponentCommandService
    from infernux.engine.interaction import serialized_properties
    from infernux.engine.undo import UndoManager

    obj = scene.create_game_object("CameraSchemaEdit")
    camera = obj.add_component("Camera")
    wrapper = Camera._get_or_create_wrapper(camera, obj)
    before = camera.serialize_document()
    descriptor = getattr(Camera, field)
    original = serialized_properties.make_attribute_property_transaction
    captured = []

    def capture(*args, **kwargs):
        captured.append(kwargs["schema"])
        return original(*args, **kwargs)

    monkeypatch.setattr(serialized_properties, "make_attribute_property_transaction", capture)
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        assert ComponentCommandService().set_field(wrapper, field, candidate)
        assert captured == [descriptor.schema]
        assert captured[0] is descriptor.schema
        after = camera.serialize_document()
        assert after != before
        assert len(manager.action_journal.applied_entries()) == 1
        expected = VALUE_CODECS.encode(179.0 if field == "field_of_view" else candidate)
        actual = VALUE_CODECS.encode(getattr(wrapper, field))
        assert actual == (expected if isinstance(expected, dict) else pytest.approx(expected))
        manager.undo()
        assert camera.serialize_document() == before
        manager.redo()
        assert camera.serialize_document() == after
    finally:
        manager.clear()
        UndoManager._instance = previous
