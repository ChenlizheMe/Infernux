"""Real native catalog publication of compiled Python field declarations."""

from copy import deepcopy

import pytest

from Infernux.components import InxComponent, serialized_field
from Infernux.components.fields import get_field_schema
from Infernux.lib import _Infernux as native


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
    assert native._semantic_catalog_snapshot().type_document(declaration["type_guid"]) == active.type_document(declaration["type_guid"])


@pytest.mark.parametrize("key", ["fields", "types"])
def test_descriptor_collections_must_be_arrays(declaration, key):
    edit = {"owner": declaration["owner"], "types": [declaration]}
    if key == "fields":
        declaration[key] = {}
    else:
        edit[key] = {}
    before = native._semantic_catalog_snapshot().revision
    with pytest.raises(ValueError, match=key):
        native._semantic_catalog_prepare([edit])
    assert native._semantic_catalog_snapshot().revision == before


def test_engine_python_fields_survive_native_snapshot_roundtrip(declaration):
    from Infernux.components.registry import ensure_engine_component_catalog_loaded, get_all_types
    from Infernux.components.fields import get_serialized_fields
    from Infernux.field_schema import FieldSchema

    ensure_engine_component_catalog_loaded()
    component_types = {
        cls for cls in get_all_types().values()
        if cls.__module__.startswith("Infernux.") and getattr(cls, "_uses_component_data_store", True)
    }
    assert any(cls.__name__ == "UIText" for cls in component_types)
    documents = []
    schemas = {}
    for cls in sorted(component_types, key=lambda item: (item.__module__, item.__qualname__)):
        document = deepcopy(declaration)
        # Test-owned copies, not production registration or replacement of engine identities.
        document["type_guid"] = declaration["owner"] + ":" + cls.__module__ + ":" + cls.__qualname__
        document["readable_id"] = document["type_guid"]
        fields = [get_field_schema(cls, name) for name in get_serialized_fields(cls)]
        schemas[document["type_guid"]] = fields
        document["fields"] = [field.to_document() for field in fields]
        documents.append(document)
    before = native._semantic_catalog_snapshot()
    publication = native._semantic_catalog_prepare([{"owner": declaration["owner"], "types": documents}])
    try:
        publication.publish()
        active = native._semantic_catalog_snapshot()
        assert active.revision == before.revision + 1
        for guid, fields in schemas.items():
            assert [FieldSchema.from_document(item) for item in active.type_document(guid)["fields"]] == fields
        assert all(guid not in before.type_guids for guid in schemas)
    finally:
        native._semantic_catalog_prepare([{"owner": declaration["owner"], "types": []}]).publish()


def test_native_transform_declaration_matches_existing_scene_data(scene):
    from Infernux.components.fields import FieldType
    from Infernux.components.value_codec import VALUE_CODECS
    from Infernux.field_schema import get_native_field_schema

    obj = scene.create_game_object("DeclaredTransform")
    transform = obj.get_transform()
    document = transform.serialize_document()
    snapshot = native._semantic_catalog_snapshot()
    descriptor = snapshot.type_document("native:infernux.Transform")
    assert descriptor["origin"] == "native"
    assert descriptor["owner"] == "engine:native"
    assert obj.serialize_document()["transform"] == document
    assert document["type"] == "Transform"
    for item in descriptor["fields"]:
        schema = get_native_field_schema(descriptor["type_guid"], item["attributes"]["field_id"])
        default = list(schema.attributes["default"])
        VALUE_CODECS.validate(default, FieldType.VEC3, schema.property_path)
        assert document[schema.attributes["serialized_name"]] == default
        assert VALUE_CODECS.encode(getattr(transform, schema.attributes["field_id"])) == default
    transform.local_position = native.Vector3(3, 4, 5)
    assert get_native_field_schema(descriptor["type_guid"], "local_position").attributes["default"] == (0.0, 0.0, 0.0)
    with pytest.raises(KeyError):
        get_native_field_schema(descriptor["type_guid"], "missing_field")


@pytest.mark.parametrize("field", ["local_position", "local_euler_angles", "local_scale"])
def test_native_transform_editor_command_uses_declared_schema_and_undo(scene, monkeypatch, field):
    from Infernux.engine.interaction.components import ComponentCommandService
    from Infernux.engine.interaction import serialized_properties
    from Infernux.engine.undo import UndoManager

    transform = scene.create_game_object("SchemaCommand").get_transform()
    before = [getattr(transform, field)[i] for i in range(3)]
    captured = []
    original = serialized_properties.make_attribute_property_transaction

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(kwargs["schema"])
        return result

    monkeypatch.setattr(serialized_properties, "make_attribute_property_transaction", capture)
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        service = ComponentCommandService()
        assert service.set_field(transform, field, [2.0, 3.0, 4.0])
        assert captured[0].value_type == "FieldType.VEC3"
        assert captured[0].attributes["field_id"] == field
        assert [getattr(transform, field)[i] for i in range(3)] == pytest.approx([2, 3, 4])
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert [getattr(transform, field)[i] for i in range(3)] == pytest.approx(before)
        manager.redo()
        assert [getattr(transform, field)[i] for i in range(3)] == pytest.approx([2, 3, 4])
        for invalid in ([1, 2], [float("nan"), 0, 0], None, "invalid"):
            with pytest.raises(RuntimeError, match=f"Transform.{field}"):
                service.set_field(transform, field, invalid)
            assert [getattr(transform, field)[i] for i in range(3)] == pytest.approx([2, 3, 4])
        assert len(manager.action_journal.applied_entries()) == 1
    finally:
        manager.clear()
        UndoManager._instance = previous


def test_camera_declarations_match_native_defaults_and_wrapper_metadata(scene):
    from Infernux.components.builtin.camera import Camera
    from Infernux.components.builtin_component import CppProperty
    from Infernux.components.value_codec import VALUE_CODECS

    obj = scene.create_game_object("DeclaredCamera")
    camera = obj.add_component("Camera")
    document = camera.serialize_document()
    descriptor = native._semantic_catalog_snapshot().type_document("native:infernux.Camera")
    # The semantic catalog includes the four physical-camera fields added to
    # the Camera contract; the old 13-field count predates that API.
    assert len(descriptor["fields"]) == 17
    assert any(record["type_id"] == descriptor["type_guid"] for record in obj.serialize_document()["components"])
    for field in descriptor["fields"]:
        attributes = field["attributes"]
        raw = getattr(camera, attributes["field_id"])
        encoded = VALUE_CODECS.encode(raw)
        expected = attributes["default"]
        if field["value_type"] == "FieldType.ENUM":
            assert attributes["enum"]["type_id"] == f"native:infernux.{type(raw).__name__}"
            assert encoded == expected
            assert int(raw) == document[attributes["serialized_name"]]
        elif field["value_type"] == "FieldType.ASSET":
            assert attributes["asset_type"] == expected["asset_type"] == "RenderTexture"
            assert raw is None and document[attributes["serialized_name"]] == expected["guid"] == ""
        else:
            assert encoded == pytest.approx(expected)
            assert document[attributes["serialized_name"]] == pytest.approx(expected)
        prop = getattr(Camera, attributes["field_id"])
        if isinstance(prop, CppProperty):
            assert prop.schema.to_document() == field
            assert VALUE_CODECS.encode(prop.metadata.default) == expected
    assert Camera.far_clip.metadata.default == camera.far_clip == 5000.0
    assert Camera.projection_mode.metadata.enum_type is native.CameraProjection
    assert Camera.clear_flags.metadata.enum_labels[1] == "camera.clear.solid_color"
    assert Camera.field_of_view.metadata.visible_when is not None


def test_light_declaration_owns_serialized_shape_and_wrapper_metadata(scene):
    from Infernux.components.builtin.light import Light
    from Infernux.components.builtin_component import CppProperty

    obj = scene.create_game_object("DeclaredLight")
    light = obj.add_component("Light")
    document = light.serialize_document()
    descriptor = native._semantic_catalog_snapshot().type_document("native:infernux.Light")
    fields = {field["attributes"]["field_id"]: field for field in descriptor["fields"]}

    assert len(fields) == 15
    assert {field["attributes"]["serialized_name"] for field in descriptor["fields"]} == (
        set(document) - {
            "type", "typeId", "componentId", "enabled", "executionOrder",
            "component_id", "execution_order",
        }
    )
    assert fields["color"]["value_type"] == "FieldType.VEC3"
    assert fields["light_type"]["attributes"]["enum"]["type_id"] == "native:infernux.LightType"
    assert fields["shadows"]["attributes"]["default"]["name"] == "Hard"
    assert fields["render_mode"]["attributes"]["enum"]["type_id"] == "native:infernux.LightRenderMode"
    assert fields["influence_domains"]["attributes"]["default"] == 3

    projected = {
        "light_type", "intensity", "range", "spot_angle", "outer_spot_angle", "area_size",
        "area_two_sided", "shadows", "shadow_strength", "shadow_softness",
    }
    for name in projected:
        prop = getattr(Light, name)
        assert isinstance(prop, CppProperty)
        assert prop.schema.to_document() == fields[name]
    assert Light.light_type.metadata.enum_type is native.LightType
    assert Light.shadows.metadata.enum_type is native.LightShadows
    assert Light.range.metadata.visible_when is not None


def test_audio_source_declaration_owns_public_fields_and_coupled_track_count(scene):
    from Infernux.components.builtin.audio_source import AudioSource
    from Infernux.components.builtin_component import CppProperty
    from Infernux.engine.interaction.components import ComponentCommandService
    from Infernux.engine.undo import UndoManager

    owner = scene.create_game_object("DeclaredAudioSource")
    native_source = owner.add_component("AudioSource")
    source = AudioSource._get_or_create_wrapper(native_source, owner)
    descriptor = native._semantic_catalog_snapshot().type_document(
        "native:infernux.AudioSource"
    )
    fields = {field["attributes"]["field_id"]: field for field in descriptor["fields"]}
    assert set(fields) == {
        "track_count", "volume", "pitch", "mute", "loop", "play_on_awake",
        "spatial_blend", "min_distance", "max_distance", "output_bus",
        "one_shot_pool_size", "priority",
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


@pytest.mark.parametrize(("type_name", "wrapper_module", "wrapper_name", "specific"), [
    ("BoxCollider", "box_collider", "BoxCollider", {"size"}),
    ("SphereCollider", "sphere_collider", "SphereCollider", {"radius"}),
    ("CapsuleCollider", "capsule_collider", "CapsuleCollider", {"radius", "height", "direction"}),
    ("CylinderCollider", "cylinder_collider", "CylinderCollider", {"radius", "height", "direction"}),
    ("MeshCollider", "mesh_collider", "MeshCollider", {"convex"}),
])
def test_collider_declarations_own_serialized_fields_and_wrapper_schema(
    scene, type_name, wrapper_module, wrapper_name, specific,
):
    import importlib

    from Infernux.components.builtin_component import CppProperty
    from Infernux.components.value_codec import VALUE_CODECS

    wrapper_type = getattr(
        importlib.import_module(f"Infernux.components.builtin.{wrapper_module}"),
        wrapper_name,
    )
    owner = scene.create_game_object(f"Declared{type_name}")
    native_collider = owner.add_component(type_name)
    document = native_collider.serialize_document()
    descriptor = native._semantic_catalog_snapshot().type_document(
        f"native:infernux.{type_name}"
    )
    fields = {field["attributes"]["field_id"]: field for field in descriptor["fields"]}
    assert set(fields) == {"center", "is_trigger", "physic_material"} | specific
    assert fields["physic_material"]["attributes"]["serialized_name"] == "physic_material_guid"
    assert fields["physic_material"]["attributes"]["asset_type"] == "PhysicMaterial"
    assert fields["physic_material"]["attributes"]["setter_owns_document_shape"] is True
    for name, field in fields.items():
        prop = getattr(wrapper_type, name)
        assert isinstance(prop, CppProperty)
        assert prop.schema.to_document() == field
        serialized_name = field["attributes"]["serialized_name"]
        if name == "physic_material":
            assert document[serialized_name] == ""
            assert VALUE_CODECS.encode(prop.metadata.default) == field["attributes"]["default"]
        else:
            assert document[serialized_name] == pytest.approx(field["attributes"]["default"])


def test_box_collider_editor_command_uses_native_schema_material_boundary_and_undo(scene):
    from Infernux.components.builtin.box_collider import BoxCollider
    from Infernux.core.asset_ref import PhysicMaterialRef
    from Infernux.engine.interaction.components import ComponentCommandService
    from Infernux.engine.undo import UndoManager
    from Infernux.lib import Vector3

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


def test_rigidbody_declaration_owns_serialized_defaults_and_wrapper_schema(scene):
    from Infernux.components.builtin.rigidbody import (
        CollisionDetectionMode,
        Rigidbody,
        RigidbodyInterpolation,
    )
    from Infernux.components.builtin_component import CppProperty

    obj = scene.create_game_object("DeclaredRigidbody")
    rigidbody = obj.add_component("Rigidbody")
    document = rigidbody.serialize_document()
    descriptor = native._semantic_catalog_snapshot().type_document("native:infernux.Rigidbody")
    assert len(descriptor["fields"]) == 10
    fields = {field["attributes"]["field_id"]: field for field in descriptor["fields"]}
    assert set(fields) == {
        "mass", "drag", "angular_drag", "use_gravity", "is_kinematic", "constraints",
        "collision_detection_mode", "interpolation", "max_angular_velocity", "max_linear_velocity",
    }
    for name, field in fields.items():
        attributes = field["attributes"]
        if field["value_type"] == "FieldType.ENUM":
            default_name = attributes["default"]["name"]
            default_value = next(member["value"] for member in attributes["enum"]["members"]
                                 if member["name"] == default_name)
            assert document[attributes["serialized_name"]] == default_value
        else:
            assert document[attributes["serialized_name"]] == pytest.approx(attributes["default"])
        wrapper_property = getattr(Rigidbody, name, None)
        if isinstance(wrapper_property, CppProperty):
            assert wrapper_property.schema.to_document() == field
    assert fields["constraints"]["attributes"]["flags_type"] == "native:infernux.RigidbodyConstraints"
    assert fields["collision_detection_mode"]["attributes"]["enum"]["type_id"] == (
        "native:infernux.CollisionDetectionMode"
    )
    assert Rigidbody.collision_detection_mode.metadata.enum_type is CollisionDetectionMode
    assert Rigidbody.interpolation.metadata.enum_type is RigidbodyInterpolation


@pytest.mark.parametrize("field,candidate", [
    ("mass", 2.5),
    ("drag", 1.25),
    ("collision_detection_mode", native.CollisionDetectionMode.Continuous),
    ("interpolation", native.RigidbodyInterpolation.Interpolate),
])
def test_rigidbody_editor_command_uses_declared_schema_and_undo(scene, monkeypatch, field, candidate):
    from Infernux.components.builtin.rigidbody import Rigidbody
    from Infernux.engine.interaction.components import ComponentCommandService
    from Infernux.engine.interaction import serialized_properties
    from Infernux.engine.undo import UndoManager

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


@pytest.mark.parametrize(("type_name", "wrapper_module", "wrapper_name"), [
    ("HingeJoint", "hinge_joint", "HingeJoint"),
    ("SliderJoint", "slider_joint", "SliderJoint"),
])
def test_native_joint_component_reference_uses_document_id_and_undo(
    scene, type_name, wrapper_module, wrapper_name,
):
    import importlib

    from Infernux.engine.interaction.components import ComponentCommandService
    from Infernux.engine.undo import UndoManager

    wrapper_type = getattr(
        importlib.import_module(f"Infernux.components.builtin.{wrapper_module}"),
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
            joint, "connected_body", support_native,
        )
        assert joint_native.serialize_document()["connected_body_component_id"] == support_native.component_id
        manager.undo()
        assert joint_native.serialize_document()["connected_body_component_id"] == 0
        manager.redo()
        assert joint_native.serialize_document()["connected_body_component_id"] == support_native.component_id
    finally:
        manager.clear()
        UndoManager._instance = previous


@pytest.mark.parametrize("field,candidate", [
    ("field_of_view", 200.0), ("far_clip", 4000.0),
    ("projection_mode", native.CameraProjection.Orthographic),
    ("clear_flags", native.CameraClearFlags.SolidColor),
    ("background_color", [0.2, 0.3, 0.4, 1.0]), ("dithering", True),
])
def test_camera_wrapper_edits_use_native_schema_and_common_codec(scene, monkeypatch, field, candidate):
    from Infernux.components.builtin.camera import Camera
    from Infernux.components.value_codec import VALUE_CODECS
    from Infernux.engine.interaction.components import ComponentCommandService
    from Infernux.engine.interaction import serialized_properties
    from Infernux.engine.undo import UndoManager

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
