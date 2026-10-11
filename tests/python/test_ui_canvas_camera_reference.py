"""Canvas camera ownership is a serialized reference, including hierarchy clones."""
from infernux.lib import GameObject
from infernux.ui import UICanvas


def test_camera_id_compatibility_serializes_a_typed_reference(scene):
    camera = scene.create_game_object("Assigned Camera")
    camera.add_component("Camera")
    canvas = UICanvas()
    canvas.target_camera_id = camera.id
    document = canvas._serialize_fields_document()
    assert document["target_camera"] == {"$type": "game_object_ref", "object_id": camera.id}
    assert "target_camera_id" not in document
    assert canvas.target_camera.id == camera.id


def test_canvas_target_follows_cloned_camera_in_same_hierarchy(scene):
    root = scene.create_game_object("Camera and Canvas")
    camera = scene.create_game_object("Assigned Camera")
    camera.set_parent(root, False)
    camera.add_component("Camera")
    canvas_owner = scene.create_game_object("Assigned Canvas")
    canvas_owner.set_parent(root, False)
    canvas = UICanvas()
    canvas_owner.add_py_component(canvas)
    canvas.target_camera = camera
    clone = GameObject.instantiate(root)
    children = {owner.name: owner for owner in clone.get_children()}
    cloned_camera = children["Assigned Camera"]
    cloned_canvas = next(component for component in children["Assigned Canvas"].get_py_components()
                         if isinstance(component, UICanvas))
    assert cloned_camera.id != camera.id
    assert cloned_canvas.target_camera_id == cloned_camera.id


def test_legacy_runtime_canvas_snapshot_restores_camera_reference(scene):
    camera = scene.create_game_object("Legacy Camera")
    camera.add_component("Camera")
    canvas = UICanvas()
    canvas._deserialize_fields_document({"target_camera_id": camera.id})
    assert canvas.target_camera.id == camera.id
    assert canvas._serialize_fields_document()["target_camera"]["object_id"] == camera.id
