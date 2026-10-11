"""Invalid public overloads fail before creating or destroying scene objects."""

import pytest

from infernux import Destroy, Instantiate
from infernux.components import GameObjectRef, PrefabRef
from infernux.core import Material
from infernux.lib import Vector3, quatf


def source_for(scene, kind):
    owner = scene.create_game_object("ArgumentSource")
    if kind == "object":
        return owner
    if kind == "object-ref":
        return GameObjectRef(owner)
    return PrefabRef()


@pytest.mark.parametrize("kind", ["object", "object-ref", "prefab-ref"])
@pytest.mark.parametrize("kwargs,message", [
    ({"position": "invalid"}, "position"),
    ({"rotation": "invalid"}, "rotation"),
    ({"parent": "invalid"}, "parent"),
    ({"instantiate_in_world_space": 1}, "bool"),
    ({"unexpected_option": True}, "unexpected"),
])
def test_invalid_scalar_keywords_raise_type_error_without_scene_mutation(scene, kind, kwargs, message):
    source = source_for(scene, kind)
    before = scene.serialize_document()
    with pytest.raises(TypeError, match=message):
        Instantiate(source, **kwargs)
    assert scene.serialize_document() == before


@pytest.mark.parametrize("kind", ["object", "object-ref", "prefab-ref"])
@pytest.mark.parametrize("duplicate", ["parent", "world-space", "position", "rotation", "placed-parent", "world-space-alias"])
def test_duplicate_scalar_parameters_are_rejected_without_selecting_one_value(scene, kind, duplicate):
    source = source_for(scene, kind)
    parent = scene.create_game_object("FirstParent")
    other = scene.create_game_object("SecondParent")
    position = Vector3(1, 2, 3)
    rotation = quatf(0, 0, 0, 1)
    args, kwargs = {
        "parent": ((parent,), {"parent": other}),
        "world-space": ((parent, True), {"instantiate_in_world_space": False}),
        "position": ((position, rotation), {"position": Vector3(4, 5, 6)}),
        "rotation": ((position, rotation), {"rotation": rotation}),
        "placed-parent": ((position, rotation, parent), {"parent": other}),
        "world-space-alias": ((), {"instantiate_in_world_space": False, "instantiateInWorldSpace": True}),
    }[duplicate]
    before = scene.serialize_document()
    with pytest.raises(TypeError, match="multiple values"):
        Instantiate(source, *args, **kwargs)
    assert scene.serialize_document() == before


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_material_clones_reject_transform_overloads_instead_of_ignoring_them(scene, native, positional):
    material = Material.create_unlit("ArgumentMaterial")
    source = material._native if native else material
    parent = scene.create_game_object("MaterialParent")
    before = scene.serialize_document()
    with pytest.raises(TypeError, match="[Mm]aterial"):
        if positional:
            Instantiate(source, parent)
        else:
            Instantiate(source, parent=parent)
    assert scene.serialize_document() == before


@pytest.mark.parametrize("delay", [3.0, -1.0, float("inf")])
def test_unsupported_delay_is_rejected_before_queuing_destruction(scene, delay):
    owner = scene.create_game_object("DelayedDestroyOwner")
    identity = owner.id
    with pytest.raises(NotImplementedError, match="delayed destruction"):
        Destroy(owner, delay)
    scene.process_pending_destroys()
    assert scene.find_by_id(identity) is owner
    assert not owner.is_destroying


def test_immediate_destroy_still_uses_pending_scene_lifetime(scene):
    owner = scene.create_game_object("ImmediateDestroyOwner")
    identity = owner.id
    Destroy(owner)
    scene.process_pending_destroys()
    assert scene.find_by_id(identity) is None


@pytest.mark.parametrize("reference", [GameObjectRef, PrefabRef])
def test_missing_reference_with_valid_arguments_returns_none_without_scene_mutation(scene, reference):
    before = scene.serialize_document()
    assert Instantiate(reference()) is None
    assert scene.serialize_document() == before


@pytest.mark.parametrize("kind", ["object", "object-ref", "prefab-ref"])
def test_batch_rejects_duplicate_world_space_aliases_before_instantiation(scene, kind):
    source = source_for(scene, kind)
    before = scene.serialize_document()
    with pytest.raises(TypeError, match="multiple values"):
        Instantiate(source, positions=[Vector3(1, 2, 3)],
                    instantiate_in_world_space=False, instantiateInWorldSpace=True)
    assert scene.serialize_document() == before


@pytest.mark.parametrize("kind", ["object", "object-ref", "prefab-ref"])
def test_excess_positional_arguments_propagate_through_references(scene, kind):
    source = source_for(scene, kind)
    before = scene.serialize_document()
    with pytest.raises(TypeError, match="at most"):
        Instantiate(source, None, None, None, None)
    assert scene.serialize_document() == before


@pytest.mark.parametrize("delay", ["3", None])
def test_invalid_delay_type_does_not_destroy_owner(scene, delay):
    owner = scene.create_game_object("InvalidDelayOwner")
    with pytest.raises(TypeError, match="real number"):
        Destroy(owner, delay)
    scene.process_pending_destroys()
    assert scene.find_by_id(owner.id) is owner
    assert not owner.is_destroying
