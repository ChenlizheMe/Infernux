"""Property history never selects a peer after its addressed component retires."""
import copy
from types import SimpleNamespace

import pytest

from infernux.components import InxComponent, serialized_field
from infernux import BoxCollider
from infernux.engine.undo import (
    GenericComponentCommand, PythonComponentDocumentCommand, SetPropertyCommand,
    SetMaterialSlotCommand, UndoManager,
)
from infernux.engine.undo._renderstack import (
    RenderStackEffectSlotsCommand, RenderStackFieldCommand, RenderStackSetPipelineCommand,
)
from infernux.engine.component_restore import deserialize_scene_document_transactionally
from infernux.lib import SceneManager


class HistoryPuzzleValue(InxComponent):
    value: int = serialized_field(default=0)


KINDS = ("native-property", "wrapper-property", "native-document", "python-property", "python-document")


def _prepare(scene, kind):
    owner = scene.create_game_object("HistoryOwner")
    python_kind = kind.startswith("python")
    if python_kind:
        survivor = owner.add_py_component(HistoryPuzzleValue())
        target = owner.add_py_component(HistoryPuzzleValue())
        survivor.value = 99
        target.value = 10
        if kind == "python-document":
            before = target._serialize_fields_document()
            target.value = 20
            after = target._serialize_fields_document()
            target.value = 10
            command = PythonComponentDocumentCommand(target, before, after)
        else:
            command = SetPropertyCommand(target, "value", 10, 20)
        unchanged = lambda: survivor.value == 99
        remove = lambda: owner.remove_py_component(target)
    else:
        survivor = owner.add_component("BoxCollider")
        native = owner.add_component("BoxCollider")
        survivor.enabled = True
        native.enabled = False
        target = BoxCollider._get_or_create_wrapper(native, owner) if kind == "wrapper-property" else native
        if kind == "native-document":
            before = native.serialize_document()
            native.enabled = True
            after = native.serialize_document()
            native.enabled = False
            command = GenericComponentCommand(target, before, after)
        else:
            command = SetPropertyCommand(target, "enabled", False, True)
        unchanged = lambda: survivor.enabled is True
        remove = lambda: owner.remove_component(native)
    return owner, target, command, remove, unchanged


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("direction", ["execute", "undo", "redo"])
def test_missing_property_target_rejects_without_writing_same_type_peer(scene, kind, direction):
    owner, target, command, remove, unchanged = _prepare(scene, kind)
    if direction in ("undo", "redo"):
        command.execute()
    if direction == "redo":
        command.undo()
    assert remove()
    before = copy.deepcopy(scene.serialize_document())
    with pytest.raises((RuntimeError, ReferenceError), match="target|not found|unavailable|destroyed|retired|invalid"):
        getattr(command, direction)()
    assert unchanged()
    assert scene.serialize_document() == before


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("direction", ["undo", "redo"])
def test_missing_property_target_keeps_journal_cursor_and_nonactive_owner(scene, kind, direction):
    old_manager = UndoManager._instance
    manager = UndoManager()
    try:
        owner, target, command, remove, unchanged = _prepare(scene, kind)
        assert manager.execute(command)
        if direction == "redo":
            manager.undo()
        native = SceneManager.instance()
        native.set_active_scene(native.create_scene("OtherActiveWorld"))
        assert remove()
        before = (manager.can_undo, manager.can_redo)
        getattr(manager, direction)()
        assert (manager.can_undo, manager.can_redo) == before
        assert unchanged()
    finally:
        manager.shutdown()
        UndoManager._instance = old_manager


@pytest.mark.parametrize("kind", KINDS)
def test_property_history_follows_exact_identity_after_whole_world_replacement(scene, kind):
    owner, target, command, remove, unchanged = _prepare(scene, kind)
    owner_id, component_id = owner.id, target.component_id
    command.execute()
    replacement = copy.deepcopy(scene.serialize_document())
    assert deserialize_scene_document_transactionally(scene, replacement)
    owner = scene.find_by_id(owner_id)
    components = owner.get_py_components() if kind.startswith("python") else owner.get_components("BoxCollider")
    restored = next(component for component in components if component.component_id == component_id)
    survivor = next(component for component in components if component.component_id != component_id)
    native = SceneManager.instance()
    native.set_active_scene(native.create_scene("OtherActiveWorld"))
    for _ in range(3):
        command.undo()
        assert restored.value == 10 if kind.startswith("python") else restored.enabled is False
        command.redo()
        assert restored.value == 20 if kind.startswith("python") else restored.enabled is True
        assert survivor.value == 99 if kind.startswith("python") else survivor.enabled is True


@pytest.mark.parametrize("direction", ["execute", "undo", "redo"])
@pytest.mark.parametrize("kind", ["material", "pipeline", "pipeline-field", "effect-slots"])
def test_renderer_and_renderstack_history_cannot_select_replacement(scene, kind, direction):
    from infernux.renderstack import RenderStack

    owner = scene.create_game_object("RenderOwner")
    if kind == "material":
        target = owner.add_component("MeshRenderer")
        command = SetMaterialSlotCommand(target, 0, "", "")
        assert owner.remove_component(target)
        replacement = owner.add_component("MeshRenderer")
        snapshot = replacement.serialize_document
    else:
        target = owner.add_py_component(RenderStack())
        if kind == "pipeline":
            command = RenderStackSetPipelineCommand(target, "", "")
        elif kind == "pipeline-field":
            command = RenderStackFieldCommand(target, SimpleNamespace(value=0), "value", 0, 1)
        else:
            command = RenderStackEffectSlotsCommand(target, "stage", [], [])
        assert owner.remove_py_component(target)
        replacement = owner.add_py_component(RenderStack())
        snapshot = replacement._serialize_fields_document
    before = copy.deepcopy(snapshot())
    with pytest.raises(RuntimeError, match="History target is unavailable"):
        getattr(command, direction)()
    assert snapshot() == before


def test_material_history_merge_distinguishes_recreated_renderer(scene):
    owner = scene.create_game_object("RenderOwner")
    first = owner.add_component("MeshRenderer")
    old = SetMaterialSlotCommand(first, 0, "", "")
    assert owner.remove_component(first)
    replacement = owner.add_component("MeshRenderer")
    new = SetMaterialSlotCommand(replacement, 0, "", "")
    new.timestamp = old.timestamp + 0.1
    assert not old.can_merge(new)


def test_pipeline_field_merge_distinguishes_stack_owners_and_command_semantics(scene):
    from infernux.renderstack import RenderStack

    first = scene.create_game_object("First").add_py_component(RenderStack())
    second = scene.create_game_object("Second").add_py_component(RenderStack())
    shared_projection = SimpleNamespace(value=0)
    left = RenderStackFieldCommand(first, shared_projection, "value", 0, 1)
    right = RenderStackFieldCommand(second, shared_projection, "value", 0, 1)
    plain = SetPropertyCommand(shared_projection, "value", 0, 1)
    for command in (right, plain):
        command.timestamp = left.timestamp + 0.1
        assert not left.can_merge(command)
    left.timestamp = plain.timestamp + 0.1
    assert not plain.can_merge(left)
