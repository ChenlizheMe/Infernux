from __future__ import annotations

import pytest


def test_required_component_removal_is_rejected_before_journal_mutation(monkeypatch):
    from types import SimpleNamespace
    from infernux.engine.interaction import ComponentCommandService

    component = SimpleNamespace(type_name="Rigidbody")
    owner = SimpleNamespace(
        id=17,
        get_remove_component_blockers=lambda target: ["TargetReporter"] if target is component else [],
    )
    service = ComponentCommandService()
    monkeypatch.setattr(service, "_manager", lambda: pytest.fail("must reject before journal access"))
    try:
        with pytest.raises(ValueError, match="required by TargetReporter"):
            service.remove(owner, component)
    finally:
        service.shutdown()


def test_component_add_resolves_engine_python_and_native_targets(monkeypatch):
    import infernux.components.registry as component_registry
    import infernux.engine.undo as undo_module
    from infernux.components.sprite_animator import SpriteAnimator
    from infernux.engine.interaction import ComponentCommandService

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
            self.result_component = python_instance if python_instance is not None else object()
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
        lambda type_name: SpriteAnimator if type_name == "SpriteAnimator" else None,
    )
    service = ComponentCommandService()
    monkeypatch.setattr(service, "_execute", lambda *_args, **_kwargs: None)
    try:
        python_component = service.add(FakeGameObject(), "SpriteAnimator")
        native_component = service.add(FakeGameObject(), "MeshRenderer")

        assert isinstance(python_component, SpriteAnimator)
        assert isinstance(captured[0][1], SpriteAnimator)
        assert captured[1] == ("MeshRenderer", None)
        assert native_component is not None
    finally:
        service.shutdown()


@pytest.mark.parametrize("script_error", [None, "candidate schema failed"])
def test_project_component_add_binds_asset_database_guid(monkeypatch, tmp_path, script_error):
    import types

    import infernux.components.registry as component_registry
    import infernux.components.script_loader as script_loader
    import infernux.engine.interaction as interaction_module
    from infernux.engine.interaction import ComponentCommandService

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

    core = types.SimpleNamespace(project_assets=types.SimpleNamespace(asset_database=Database()))
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
    monkeypatch.setattr(script_loader, "get_script_error_by_path", lambda _path: script_error)
    if script_error:
        with pytest.raises(ValueError, match="current revision failed to load"):
            ComponentCommandService._resolve_add_target(object(), "ProjectController", None)
        assert calls == {}
        return
    resolved = ComponentCommandService._resolve_add_target(object(), "ProjectController", None)

    assert resolved is loaded
    assert calls["path"] == script_path
    assert calls["script_guid"] == script_guid
    assert calls["asset_database"] is core.project_assets.asset_database


@pytest.mark.parametrize("name_collision", ["project", "native", "missing"])
def test_explicit_script_guid_owns_attachment_before_short_name_lookup(monkeypatch, tmp_path, name_collision):
    from types import SimpleNamespace
    import infernux.components.registry as registry
    import infernux.components.script_loader as loader
    import infernux.engine.interaction as interaction

    requested_path = str(tmp_path / "Requested.py")
    requested_guid = "b" * 32
    unrelated_path = str(tmp_path / "Other.py")

    class Requested:
        pass

    class Other:
        _cpp_type_name = "SharedController" if name_collision == "native" else ""

    requested = Requested()
    database = SimpleNamespace(get_path_from_guid=lambda guid: requested_path if guid == requested_guid else "")
    core = SimpleNamespace(project_assets=SimpleNamespace(asset_database=database))
    calls = []
    monkeypatch.setattr(registry, "ensure_engine_component_catalog_loaded", lambda: None)
    monkeypatch.setattr(registry, "get_type", lambda _name: None if name_collision == "missing" else Other)
    monkeypatch.setattr(
        registry,
        "get_type_registration",
        lambda _name: SimpleNamespace(project_script=True, script_path=unrelated_path),
    )
    monkeypatch.setattr(registry, "get_python_attachment_blockers", lambda *_args: ())
    monkeypatch.setattr(interaction.EditorInteractionCore, "instance", lambda: core)
    monkeypatch.setattr(loader, "get_script_error_by_path", lambda _path: None)

    def load(path, **arguments):
        calls.append((path, arguments))
        return requested

    monkeypatch.setattr(loader, "load_and_create_component", load)
    owner = SimpleNamespace(get_add_component_blockers=lambda _name: ())
    result = interaction.ComponentCommandService._resolve_add_target(
        owner,
        "SharedController",
        None,
        script_guid=requested_guid,
    )
    assert result is requested
    assert calls == [
        (requested_path, {"asset_database": database, "type_name": "SharedController", "script_guid": requested_guid})
    ]


@pytest.mark.parametrize("failure", ["missing-guid", "rejected-source", "missing-class"])
def test_explicit_script_guid_failure_does_not_attach_same_named_component(monkeypatch, failure):
    from types import SimpleNamespace
    import infernux.components.registry as registry
    import infernux.components.script_loader as loader
    import infernux.engine.interaction as interaction

    class Other:
        _cpp_type_name = ""

    owner = SimpleNamespace(
        get_add_component_blockers=lambda _name: pytest.fail("must reject before native attachment")
    )
    database = SimpleNamespace(get_path_from_guid=lambda _guid: "" if failure == "missing-guid" else "requested.py")
    core = SimpleNamespace(project_assets=SimpleNamespace(asset_database=database))
    monkeypatch.setattr(registry, "ensure_engine_component_catalog_loaded", lambda: None)
    monkeypatch.setattr(registry, "get_type", lambda _name: Other)
    monkeypatch.setattr(
        registry, "get_type_registration", lambda _name: SimpleNamespace(project_script=True, script_path="other.py")
    )
    monkeypatch.setattr(interaction.EditorInteractionCore, "instance", lambda: core)
    monkeypatch.setattr(
        loader, "get_script_error_by_path", lambda _path: "invalid candidate" if failure == "rejected-source" else None
    )
    monkeypatch.setattr(loader, "load_and_create_component", lambda *_args, **_kwargs: None)
    message = {
        "missing-guid": "GUID was not found",
        "rejected-source": "current revision failed",
        "missing-class": "was not found in",
    }[failure]
    with pytest.raises(ValueError, match=message):
        interaction.ComponentCommandService._resolve_add_target(owner, "SharedController", None, script_guid="b" * 32)


def test_component_document_edit_is_atomic_and_replayable():
    from infernux.engine.interaction import ComponentCommandService
    from infernux.engine.undo import UndoManager

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


def test_component_property_batch_is_one_atomic_history_entry():
    from infernux.engine.interaction import ComponentCommandService
    from infernux.engine.undo import UndoManager

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
    from infernux.engine.interaction import ComponentCommandService
    from infernux.engine.undo import UndoManager

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


def test_transform_serialized_fields_edit_local_pose_and_undo(scene):
    from infernux.components.value_codec import VALUE_CODECS
    from infernux.engine.interaction import ComponentCommandService
    from infernux.engine.undo import UndoManager
    from infernux.math import Vector3

    parent = scene.create_game_object("TransformSchemaParent")
    child = scene.create_game_object("TransformSchemaChild")
    parent.transform.local_position = Vector3(10.0, 0.0, 0.0)
    child.set_parent(parent, False)

    previous_manager = UndoManager._instance
    manager = UndoManager()
    service = ComponentCommandService()
    try:
        assert service.set_field(child.transform, "position", [2.0, 3.0, 4.0])
        assert VALUE_CODECS.encode(child.transform.local_position) == pytest.approx([2.0, 3.0, 4.0])
        assert VALUE_CODECS.encode(child.transform.position) == pytest.approx([12.0, 3.0, 4.0])

        assert service.set_field(child.transform, "scale", [2.0, 3.0, 4.0])
        assert VALUE_CODECS.encode(child.transform.local_scale) == pytest.approx([2.0, 3.0, 4.0])
        assert len(manager.action_journal.applied_entries()) == 2

        manager.undo()
        assert VALUE_CODECS.encode(child.transform.local_scale) == pytest.approx([1.0, 1.0, 1.0])
        manager.undo()
        assert VALUE_CODECS.encode(child.transform.local_position) == pytest.approx([0.0, 0.0, 0.0])
        manager.redo()
        assert VALUE_CODECS.encode(child.transform.local_position) == pytest.approx([2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="not declared writable"):
            service.set_field(child.transform, "component_id", 12)
    finally:
        service.shutdown()
        manager.clear()
        UndoManager._instance = previous_manager
