"""Cloning an authored collider must not fit it back to the renderer bounds."""
import pytest

from infernux import Instantiate
from infernux.lib import Physics, PrimitiveType, Vector3


@pytest.mark.parametrize("shape", ["BoxCollider", "SphereCollider", "CapsuleCollider", "CylinderCollider", "MeshCollider"])
@pytest.mark.parametrize("nested", [False, True])
def test_clone_preserves_authored_shape_on_mesh_owner(scene, shape, nested):
    owner = scene.create_primitive(PrimitiveType.Cube, "Authored collider")
    collider = owner.get_component("BoxCollider")
    if shape != "BoxCollider":
        assert owner.remove_component(collider)
        collider = owner.add_component(shape)
    collider.center = Vector3(.3, .7, -.4)
    if shape == "BoxCollider":
        collider.size = Vector3(2, 3, 4)
    elif shape in ("CapsuleCollider", "CylinderCollider"):
        collider.height = 3
        collider.radius = .7
    elif shape == "SphereCollider":
        collider.radius = .9
    Physics.sync_transforms()
    root = scene.create_game_object("Collider parent") if nested else owner
    if nested:
        owner.set_parent(root, True)
    cloned_root = Instantiate(root)
    cloned_root.transform.position = Vector3(100, 0, 0)
    clone = cloned_root.get_child(0) if nested else cloned_root
    Physics.sync_transforms()
    copied = clone.get_component(shape)
    assert copied.component_id != collider.component_id
    assert tuple(copied.center) == pytest.approx(tuple(collider.center))
    if shape == "BoxCollider":
        assert tuple(copied.size) == pytest.approx(tuple(collider.size))
    elif shape != "MeshCollider":
        assert copied.radius == pytest.approx(collider.radius)
        if shape != "SphereCollider":
            assert copied.height == pytest.approx(collider.height)
    source_hit = Physics.raycast(Vector3(.3, 10, -.4), Vector3(0, -1, 0), 20)
    copied_hit = Physics.raycast(Vector3(100.3, 10, -.4), Vector3(0, -1, 0), 20)
    assert source_hit is not None and copied_hit is not None
    assert source_hit.collider.component_id == collider.component_id
    assert copied_hit.collider.component_id == copied.component_id
    assert copied_hit.distance == pytest.approx(source_hit.distance, abs=1e-4)
