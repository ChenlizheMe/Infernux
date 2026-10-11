"""Automatically fitted capsule geometry satisfies setters, documents and queries."""
import json

import numpy as np
import pytest

from infernux.engine.component_restore import deserialize_scene_document_transactionally
from infernux.engine.scene_authoring import decode_scene_document
from infernux.lib import Physics, PrimitiveType, Vector3


CASES = [("bare", 1, 0), ("capsule", 1, 0), ("cube", 0, 0)] + [
    ("bounds", axis, delta) for axis in range(3) for delta in (0, .0001, .0005, .001, .002)
]


def make_capsule(scene, kind, axis, delta):
    if kind == "bare":
        owner = scene.create_game_object("AutoFit")
    else:
        primitive = PrimitiveType.Capsule if kind == "capsule" else PrimitiveType.Cube
        owner = scene.create_primitive(primitive, "AutoFit")
        if primitive == PrimitiveType.Cube:
            assert owner.remove_component(owner.get_component("BoxCollider"))
        if kind == "bounds":
            renderer = owner.get_component("MeshRenderer")
            scale = np.ones(3, dtype=np.float32)
            scale[axis] += delta
            positions = np.asarray(renderer.get_positions(), dtype=np.float32) * scale + [.2, -.3, .4]
            renderer.set_inline_mesh_data(positions, None,
                np.asarray(renderer.get_uvs(), dtype=np.float32),
                np.asarray(renderer.get_indices(), dtype=np.uint32))
    capsule = owner.get_component("CapsuleCollider") or owner.add_component("CapsuleCollider")
    return owner, capsule


def check_shape(capsule):
    capsule.radius = capsule.radius
    capsule.height = capsule.height
    capsule._cpp_component.validate_document(capsule.serialize_document())
    Physics.sync_transforms()
    center = np.asarray(tuple(capsule.center))
    for axis, extent in [(capsule.direction, capsule.height / 2), ((capsule.direction + 1) % 3, capsule.radius)]:
        direction = np.zeros(3)
        direction[axis] = -1
        origin = center - direction * (extent + 2)
        hit = Physics.raycast(Vector3(*origin), Vector3(*direction), 4)
        assert hit is not None and hit.collider.component_id == capsule.component_id
        assert hit.distance == pytest.approx(2, abs=1e-4)


@pytest.mark.parametrize("kind,axis,delta", CASES)
def test_autofit_parameters_match_authored_and_physics_contract(scene, kind, axis, delta):
    _, capsule = make_capsule(scene, kind, axis, delta)
    check_shape(capsule)


@pytest.mark.parametrize("kind", ["bare", "capsule", "cube"])
def test_autofitted_capsule_scene_disk_roundtrip(scene, tmp_path, kind):
    owner, capsule = make_capsule(scene, kind, 0, 0)
    expected = capsule.serialize_document()
    path = tmp_path / "Fitted.scene"
    assert scene.save_to_file(str(path))
    document = decode_scene_document(json.loads(path.read_text(encoding="utf-8")))
    assert deserialize_scene_document_transactionally(scene, document)
    restored = scene.find("AutoFit").get_component("CapsuleCollider")
    restored_document = restored.serialize_document()
    for field in ("center", "radius", "height", "direction", "is_trigger"):
        assert restored_document[field] == expected[field]
    check_shape(restored)
