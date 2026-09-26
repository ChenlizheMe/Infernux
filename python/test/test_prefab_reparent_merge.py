"""A prefab node keeps its identity when either author reparents it."""
import copy

import pytest

from Infernux.engine.prefab_manager import (
    _read_prefab_document, instantiate_prefab, save_prefab,
)
from Infernux.engine.prefab_overrides import (
    apply_overrides_to_prefab, build_prefab_apply_command,
    resolve_scene_prefab_documents,
)


def _fixture(scene, tmp_path):
    root = scene.create_game_object("Topology")
    left = scene.create_game_object("Left")
    right = scene.create_game_object("Right")
    child = scene.create_game_object("Moving")
    left.set_parent(root)
    right.set_parent(root)
    child.set_parent(left)
    child.add_component("BoxCollider")
    path = str(tmp_path / "topology.prefab")
    assert save_prefab(root, path)
    instances = [instantiate_prefab(file_path=path, guid="topology", scene=scene)
                 for _ in range(2)]
    return path, instances


def _nodes(root):
    yield root
    for child in root.get_children():
        yield from _nodes(child)


def _named(root, name):
    return next(node for node in _nodes(root) if node.name == name)


@pytest.mark.parametrize("source_moves", [True, False])
def test_reparent_and_other_author_property_edit_merge_by_identity(scene, tmp_path, source_moves):
    path, (source, peer) = _fixture(scene, tmp_path)
    target = _named(peer, "Moving")
    target_id = target.id
    component_id = target.get_components()[1].component_id
    mover, editor = (source, peer) if source_moves else (peer, source)
    _named(mover, "Moving").set_parent(_named(mover, "Right"))
    _named(editor, "Moving").name = "Renamed"
    command = build_prefab_apply_command(source, path)
    command.execute()
    target = scene.find_by_id(target_id)
    assert target.name == "Renamed"
    assert target.get_parent().name == "Right"
    assert target.get_components()[1].component_id == component_id
    assert len(list(_nodes(peer))) == 4
    command.undo()
    command.redo()
    target = scene.find_by_id(target_id)
    assert target.name == "Renamed" and target.get_parent().name == "Right"


def test_offline_cook_reparent_keeps_other_author_component_edit(scene, tmp_path):
    path, (source, peer) = _fixture(scene, tmp_path)
    target = _named(peer, "Moving")
    target_id = target.id
    target.get_components()[1].is_trigger = True
    saved = {"objects": [peer.serialize_document()]}
    _named(source, "Moving").set_parent(_named(source, "Right"))
    assert apply_overrides_to_prefab(source, path)
    before = copy.deepcopy(saved)
    cooked = resolve_scene_prefab_documents(saved, lambda _: _read_prefab_document(path)["root_object"])
    assert saved == before
    right = next(node for node in cooked["objects"][0]["children"] if node["name"] == "Right")
    assert len(right["children"]) == 1
    moved = right["children"][0]
    assert moved["id"] == target_id
    assert moved["components"][0]["data"]["is_trigger"] is True


def test_conflicting_parent_cycle_rejected_before_asset_publication(scene, tmp_path):
    from pathlib import Path

    path, (source, peer) = _fixture(scene, tmp_path)
    _named(source, "Left").set_parent(_named(source, "Right"))
    _named(peer, "Right").set_parent(_named(peer, "Left"))
    before_file = Path(path).read_bytes()
    before_source = source.serialize_document()
    before_peer = peer.serialize_document()
    assert not apply_overrides_to_prefab(source, path)
    assert Path(path).read_bytes() == before_file
    assert source.serialize_document() == before_source
    assert peer.serialize_document() == before_peer


def test_source_deletes_parent_but_instance_moves_child_out(scene, tmp_path):
    path, (source, peer) = _fixture(scene, tmp_path)
    child = _named(peer, "Moving")
    identity = child.id
    child.set_parent(_named(peer, "Right"))
    scene.destroy_game_object(_named(source, "Left"))
    scene.process_pending_destroys()
    assert apply_overrides_to_prefab(source, path)
    assert [obj.name for obj in peer.get_children()] == ["Right"]
    assert scene.find_by_id(identity).get_parent().name == "Right"


def test_deleted_source_ancestry_retained_for_local_descendant_override(scene, tmp_path):
    path, (source, peer) = _fixture(scene, tmp_path)
    child = _named(peer, "Moving")
    identity = child.id
    child.name = "Local Override"
    scene.destroy_game_object(_named(source, "Left"))
    scene.process_pending_destroys()
    assert apply_overrides_to_prefab(source, path)
    assert scene.find_by_id(identity).name == "Local Override"
    assert scene.find_by_id(identity).get_parent().name == "Left"


def test_source_addition_under_locally_deleted_parent_does_not_escape(scene, tmp_path):
    path, (source, peer) = _fixture(scene, tmp_path)
    scene.destroy_game_object(_named(peer, "Left"))
    scene.process_pending_destroys()
    new = scene.create_game_object("New Under Deleted")
    new.set_parent(_named(source, "Left"))
    assert apply_overrides_to_prefab(source, path)
    assert [obj.name for obj in _nodes(peer)] == ["Topology (Clone)", "Right"]
