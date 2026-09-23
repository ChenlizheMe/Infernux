from __future__ import annotations

import pytest


def test_component_add_resolves_engine_python_and_native_targets(monkeypatch):
    import Infernux.components.registry as component_registry
    import Infernux.engine.undo as undo_module
    from Infernux.components.spirit_animator import SpiritAnimator
    from Infernux.engine.interaction import ComponentCommandService

    captured = []

    class FakeGameObject:
        id = 17

        @staticmethod
        def get_py_components():
            return []

        @staticmethod
        def get_add_component_blockers(_type_name):
            return []

    class FakeAddCommand:
        def __init__(self, _object_id, type_name, **kwargs):
            python_instance = kwargs.get("python_instance")
            self.result_component = (
                python_instance if python_instance is not None else object()
            )
            captured.append((type_name, python_instance))

    monkeypatch.setattr(
        undo_module,
        "AddComponentTransactionCommand",
        FakeAddCommand,
    )
    monkeypatch.setattr(
        component_registry,
        "ensure_engine_component_catalog_loaded",
        lambda: None,
    )
    monkeypatch.setattr(
        component_registry,
        "get_type",
        lambda type_name: SpiritAnimator if type_name == "SpiritAnimator" else None,
    )
    service = ComponentCommandService()
    monkeypatch.setattr(service, "_execute", lambda *_args, **_kwargs: None)
    try:
        python_component = service.add(FakeGameObject(), "SpiritAnimator")
        native_component = service.add(FakeGameObject(), "MeshRenderer")

        assert isinstance(python_component, SpiritAnimator)
        assert isinstance(captured[0][1], SpiritAnimator)
        assert captured[1] == ("MeshRenderer", None)
        assert native_component is not None
    finally:
        service.shutdown()


def test_project_component_add_binds_asset_database_guid(monkeypatch, tmp_path):
    import types

    import Infernux.components.registry as component_registry
    import Infernux.components.script_loader as script_loader
    import Infernux.engine.interaction as interaction_module
    from Infernux.engine.interaction import ComponentCommandService

    script_path = str(tmp_path / "Assets" / "Scripts" / "Controller.py")
    script_guid = "a" * 32
    loaded = object()
    calls = {}

    class ProjectController:
        _cpp_type_name = ""

    class Database:
        @staticmethod
        def get_guid_from_path(path):
            assert path == script_path
            return script_guid

        @staticmethod
        def get_path_from_guid(guid):
            assert guid == script_guid
            return script_path

    core = types.SimpleNamespace(
        project_assets=types.SimpleNamespace(asset_database=Database())
    )
    monkeypatch.setattr(component_registry, "ensure_engine_component_catalog_loaded", lambda: None)
    monkeypatch.setattr(component_registry, "get_type", lambda _name: ProjectController)
    monkeypatch.setattr(
        component_registry,
        "get_type_registration",
        lambda _name: types.SimpleNamespace(
            project_script=True,
            script_path=script_path,
        ),
    )
    monkeypatch.setattr(component_registry, "get_python_attachment_blockers", lambda *_args: ())
    monkeypatch.setattr(interaction_module.EditorInteractionCore, "instance", lambda: core)

    def load(path, **arguments):
        calls.update({"path": path, **arguments})
        return loaded

    monkeypatch.setattr(script_loader, "load_and_create_component", load)
    resolved = ComponentCommandService._resolve_add_target(
        object(), "ProjectController", None
    )

    assert resolved is loaded
    assert calls["path"] == script_path
    assert calls["script_guid"] == script_guid
    assert calls["asset_database"] is core.project_assets.asset_database


def test_component_document_edit_is_atomic_and_replayable():
    from Infernux.engine.interaction import ComponentCommandService
    from Infernux.engine.undo import UndoManager

    class Probe:
        def __init__(self):
            self.value = 1

        def serialize_document(self):
            return {"value": self.value}

        def deserialize_document(self, document):
            self.value = int(document["value"])
            return True

    previous_manager = UndoManager._instance
    manager = UndoManager()
    service = ComponentCommandService()
    probe = Probe()
    try:
        result = service.edit_document(
            probe,
            lambda: setattr(probe, "value", 7),
            description="Edit Probe",
            edit_key="value",
        )

        assert result.changed is True
        assert probe.value == 7
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert probe.value == 1
        manager.redo()
        assert probe.value == 7
    finally:
        service.shutdown()
        UndoManager._instance = previous_manager


def test_component_document_noop_does_not_enter_history():
    from Infernux.engine.interaction import ComponentCommandService
    from Infernux.engine.undo import UndoManager

    class Probe:
        value = 1

        def serialize_document(self):
            return {"value": self.value}

        def deserialize_document(self, document):
            self.value = int(document["value"])
            return True

    previous_manager = UndoManager._instance
    manager = UndoManager()
    service = ComponentCommandService()
    probe = Probe()
    try:
        result = service.edit_document(
            probe,
            lambda: None,
            description="No-op Probe",
        )
        assert result.changed is False
        assert not manager.action_journal.applied_entries()
    finally:
        service.shutdown()
        UndoManager._instance = previous_manager


def test_component_service_inherits_automation_origin():
    from Infernux.engine.interaction import (
        ActionOrigin,
        ComponentCommandService,
        action_origin_scope,
    )
    from Infernux.engine.undo import UndoManager

    class Probe:
        value = 1

    previous_manager = UndoManager._instance
    manager = UndoManager()
    service = ComponentCommandService()
    probe = Probe()
    try:
        with action_origin_scope(ActionOrigin.AUTOMATION):
            assert service.set_field(probe, "value", 2)
        entry = manager.action_journal.applied_entries()[0]
        assert entry.origin is ActionOrigin.AUTOMATION
    finally:
        service.shutdown()
        UndoManager._instance = previous_manager


def test_component_property_batch_is_one_atomic_history_entry():
    from Infernux.engine.interaction import ComponentCommandService
    from Infernux.engine.undo import UndoManager

    class Probe:
        x = 1
        y = 2

    previous_manager = UndoManager._instance
    manager = UndoManager()
    service = ComponentCommandService()
    probe = Probe()
    try:
        assert service.execute_property_changes(
            [
                (probe, "x", 1, 10, "Set x"),
                (probe, "y", 2, 20, "Set y"),
            ],
            description="Move Probe",
        )
        assert (probe.x, probe.y) == (10, 20)
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert (probe.x, probe.y) == (1, 2)
        manager.redo()
        assert (probe.x, probe.y) == (10, 20)
    finally:
        service.shutdown()
        UndoManager._instance = previous_manager


def test_rejected_live_component_batch_rolls_back_model():
    from Infernux.engine.interaction import ComponentCommandService
    from Infernux.engine.undo import UndoManager

    class Probe:
        value = 2

    previous_manager = UndoManager._instance
    manager = UndoManager()
    service = ComponentCommandService()
    probe = Probe()
    try:
        manager.record = lambda *_args, **_kwargs: False
        assert not service.record_applied_property_changes(
            [(probe, "value", 1, 2, "Set value")],
            description="Edit Probe",
        )
        assert probe.value == 1
        assert manager.action_journal.entries == ()
    finally:
        service.shutdown()
        UndoManager._instance = previous_manager


def test_light_cpp_properties_are_not_python_document_fields():
    from Infernux.components.builtin.light import Light
    from Infernux.engine.interaction.components import ComponentCommandService

    light = Light()
    assert ComponentCommandService._is_python_component(light) is False
    assert ComponentCommandService._is_cpp_property_field(light, "intensity") is True
    assert ComponentCommandService._is_python_serialized_field(light, "intensity") is False
    assert ComponentCommandService._is_python_serialized_field(light, "range") is False
    assert ComponentCommandService._is_python_serialized_field(light, "enabled") is False


def test_transform_serialized_fields_edit_local_pose_and_undo(scene):
    from Infernux.components.value_codec import VALUE_CODECS
    from Infernux.engine.interaction import ComponentCommandService
    from Infernux.engine.undo import UndoManager
    from Infernux.math import Vector3

    parent = scene.create_game_object("TransformSchemaParent")
    child = scene.create_game_object("TransformSchemaChild")
    parent.transform.local_position = Vector3(10.0, 0.0, 0.0)
    child.set_parent(parent, False)

    previous_manager = UndoManager._instance
    manager = UndoManager()
    service = ComponentCommandService()
    try:
        assert service.set_field(child.transform, "position", [2.0, 3.0, 4.0])
        assert VALUE_CODECS.encode(child.transform.local_position) == pytest.approx(
            [2.0, 3.0, 4.0]
        )
        assert VALUE_CODECS.encode(child.transform.position) == pytest.approx(
            [12.0, 3.0, 4.0]
        )

        assert service.set_field(child.transform, "scale", [2.0, 3.0, 4.0])
        assert VALUE_CODECS.encode(child.transform.local_scale) == pytest.approx(
            [2.0, 3.0, 4.0]
        )
        assert len(manager.action_journal.applied_entries()) == 2

        manager.undo()
        assert VALUE_CODECS.encode(child.transform.local_scale) == pytest.approx(
            [1.0, 1.0, 1.0]
        )
        manager.undo()
        assert VALUE_CODECS.encode(child.transform.local_position) == pytest.approx(
            [0.0, 0.0, 0.0]
        )
        manager.redo()
        assert VALUE_CODECS.encode(child.transform.local_position) == pytest.approx(
            [2.0, 3.0, 4.0]
        )

        with pytest.raises(ValueError, match="not declared writable"):
            service.set_field(child.transform, "component_id", 12)
    finally:
        service.shutdown()
        manager.clear()
        UndoManager._instance = previous_manager


def test_transform_automation_schema_exposes_only_authoritative_fields(scene, monkeypatch):
    from Infernux.host import EditorAutomationHost

    owner = scene.create_game_object("TransformAutomationSchema")
    host = EditorAutomationHost()
    monkeypatch.setattr(host, "scene_component", lambda *_args: owner.transform)

    schema = host.scene_component_schema(owner.id, owner.transform.component_id)

    assert schema["component_type"] == "Transform"
    assert schema["fields"] == [
        {"name": "position", "type": "vec3", "readonly": False, "hidden": False},
        {"name": "rotation", "type": "vec3", "readonly": False, "hidden": False},
        {"name": "scale", "type": "vec3", "readonly": False, "hidden": False},
    ]


def test_native_automation_schema_projects_catalog_metadata(scene, monkeypatch):
    from Infernux.components.builtin.camera import Camera
    from Infernux.host import EditorAutomationHost

    owner = scene.create_game_object("CameraAutomationSchema")
    native_camera = owner.add_component("Camera")
    camera = Camera._get_or_create_wrapper(native_camera, owner)
    host = EditorAutomationHost()
    monkeypatch.setattr(host, "scene_component", lambda *_args: camera)

    schema = host.scene_component_schema(owner.id, camera.component_id)
    fields = {field["name"]: field for field in schema["fields"]}

    assert len(fields) == 27
    assert fields["targetTextureGuid"]["type"] == "asset"
    assert fields["targetTextureGuid"]["asset_type"] == "RenderTexture"
    assert fields["targetTextureGuid"]["nullable"] is True
    assert fields["fov"]["range"] == [1.0, 179.0]
    assert fields["projectionMode"]["enum"] == [
        {"name": "Perspective", "value": 0},
        {"name": "Orthographic", "value": 1},
    ]
    assert fields["gateFit"]["enum"] == [
        {"name": "None", "value": 0},
        {"name": "Vertical", "value": 1},
        {"name": "Horizontal", "value": 2},
        {"name": "Fill", "value": 3},
        {"name": "Overscan", "value": 4},
    ]


def test_light_automation_schema_uses_native_serialized_names(scene, monkeypatch):
    from Infernux.components.builtin.light import Light
    from Infernux.host import EditorAutomationHost

    owner = scene.create_game_object("LightAutomationSchema")
    native_light = owner.add_component("Light")
    light = Light._get_or_create_wrapper(native_light, owner)
    host = EditorAutomationHost()
    monkeypatch.setattr(host, "scene_component", lambda *_args: light)

    schema = host.scene_component_schema(owner.id, light.component_id)
    fields = {field["name"]: field for field in schema["fields"]}

    assert len(fields) == 15
    assert fields["lightType"]["enum"][0] == {"name": "Directional", "value": 0}
    assert fields["intensity"]["range"] == [0.0, 10.0]
    assert fields["color"]["type"] == "vec3"
    assert fields["influenceDomains"]["type"] == "int"


def test_automation_schema_does_not_infer_undeclared_json_fields(monkeypatch):
    from Infernux.host import EditorAutomationHost

    class SerializeOnlyProbe:
        component_id = 41

        @staticmethod
        def serialize_document():
            return {"guessed": 12, "also_guessed": [1.0, 2.0, 3.0]}

    host = EditorAutomationHost()
    monkeypatch.setattr(host, "scene_component", lambda *_args: SerializeOnlyProbe())

    schema = host.scene_component_schema(7, 41)

    assert schema["fields"] == []


@pytest.mark.parametrize('type_name,field,value', [
    ('Camera', 'clearFlags', 1),
    ('Camera', 'backgroundColor', [1.0, 0.25, 0.5, 1.0]),
    ('Camera', 'fov', 73.0),
    ('Light', 'lightType', 1),
    ('Light', 'color', [0.2, 0.4, 0.8]),
    ('AudioSource', 'mute', True),
    ('AudioSource', 'track_count', 3),
])
def test_native_automation_edits_its_advertised_document_field(scene, monkeypatch, type_name, field, value):
    from types import SimpleNamespace
    from Infernux.components.builtin_component import BuiltinComponent
    from Infernux.engine.interaction import ComponentCommandService
    from Infernux.engine.undo import UndoManager
    from Infernux.host import EditorAutomationHost

    owner = scene.create_game_object('NativeSchemaEdit')
    native = owner.add_component(type_name)
    component = BuiltinComponent._get_or_create_wrapper(native, owner)
    host = EditorAutomationHost()
    service = ComponentCommandService()
    previous_manager = UndoManager._instance
    manager = UndoManager()
    monkeypatch.setattr(host, 'scene_component', lambda *_: component)
    monkeypatch.setattr(host, 'interaction_core', lambda: SimpleNamespace(components=service))
    # Bound field edits must use setters, not reload the whole component (which
    # would restart an AudioSource's tracks merely to change mute or volume).
    if field in ('mute', 'track_count'):
        monkeypatch.setattr(service, 'restore_document', lambda *_a, **_k: pytest.fail('field reloaded component'))
    before = native.serialize_document()
    try:
        assert field in {f['name'] for f in host.scene_component_schema(owner.id, component.component_id)['fields']}
        host.set_scene_component_field(owner.id, component.component_id, field, value)
        assert native.serialize_document()[field] == pytest.approx(value)
        assert len(manager.action_journal.applied_entries()) == 1
        if type_name == 'Camera':
            stable = native.serialize_document()
            for bad_field, bad_value in [('clearFlags', 999), ('clearFlags', True)]:
                with pytest.raises((ValueError, TypeError, RuntimeError)):
                    host.set_scene_component_field(owner.id, component.component_id, bad_field, bad_value)
                assert native.serialize_document() == stable
                assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert native.serialize_document() == before
        manager.redo()
        assert native.serialize_document()[field] == pytest.approx(value)
        with pytest.raises(Exception, match='not declared writable'):
            host.set_scene_component_field(owner.id, component.component_id, 'made_up_field', 5)
        assert len(manager.action_journal.applied_entries()) == 1
    finally:
        service.shutdown()
        manager.clear()
        UndoManager._instance = previous_manager
