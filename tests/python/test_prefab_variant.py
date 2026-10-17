"""Variant source and local additions retain independent persistent identities."""

import copy
import json

import pytest

from infernux.components.value_document import TYPE_KEY, GAME_OBJECT_REF, COMPONENT_REF
from infernux.engine.prefab_manager import PrefabDocumentError
from infernux.engine.prefab_variant import (
    create_variant_definition,
    rebase_variant_definition,
    validate_variant_definition,
    rebase_variant_graph,
    variant_document,
    variant_definition,
    variant_property_modifications,
    revert_variant_property,
)


def node(identity, name, children=(), components=()):
    return dict(
        local_id=identity,
        name=name,
        active=True,
        is_static=False,
        tag="Untagged",
        layer=0,
        transform=dict(position=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1]),
        children=list(children),
        components=list(components),
    )


def prefab():
    return dict(
        root_object=node(1, "Root", [node(2, "Same", components=[dict(component_id=1, type_id="Test", value=1)])]),
        next_local_id=3,
        next_component_id=2,
    )


def test_variant_inherits_base_edits_preserving_explicit_overrides():
    base = prefab()
    own = copy.deepcopy(base)
    own["root_object"]["name"] = "Variant"
    own["root_object"]["children"][0]["components"][0]["value"] = 8
    definition = create_variant_definition("base-guid", base, own)
    changed = copy.deepcopy(base)
    changed["root_object"]["name"] = "New base"
    changed["root_object"]["layer"] = 5
    changed["root_object"]["children"][0]["components"][0]["value"] = 20
    before = copy.deepcopy(definition)
    updated = rebase_variant_definition(definition, changed)
    root = updated["document"]["root_object"]
    assert root["name"] == "Variant" and root["layer"] == 5
    assert root["children"][0]["components"][0]["value"] == 8
    assert definition == before
    assert rebase_variant_definition(updated, changed) == updated


def test_variant_revert_projects_references_into_local_identity_domain():
    base = prefab()
    base["root_object"]["components"] = [
        dict(
            component_id=2,
            type_id="Ref",
            target={TYPE_KEY: COMPONENT_REF, "game_object_id": 2, "component_id": 1, "component_type": "Test"},
        )
    ]
    base["next_component_id"] = 3
    own = copy.deepcopy(base)
    own["root_object"]["components"][0]["target"]["game_object_id"] = 0
    definition = create_variant_definition("base-guid", base, own)
    # Relocate the inherited child and component to a different local domain.
    definition["document"]["root_object"]["children"][0]["local_id"] = 12
    definition["document"]["root_object"]["children"][0]["components"][0]["component_id"] = 11
    definition["document"].update(next_local_id=13, next_component_id=12)
    definition["base"]["object_sources"] = [[1, 1], [12, 2]]
    definition["base"]["component_sources"] = [[11, 1], [2, 2]]
    reverted = revert_variant_property(variant_document(definition), 1, 2, ("target",))
    reference = reverted["root_object"]["components"][0]["target"]
    assert reference["game_object_id"] == 12 and reference["component_id"] == 11
    assert not variant_property_modifications(reverted)


def test_concurrent_same_name_additions_and_typed_references_do_not_alias():
    base = prefab()
    own = copy.deepcopy(base)
    own["root_object"]["children"].append(node(3, "Same", components=[dict(component_id=2, type_id="Test", value=9)]))
    own.update(next_local_id=4, next_component_id=3)
    definition = create_variant_definition("base-guid", base, own)
    changed = copy.deepcopy(base)
    changed["root_object"]["children"].append(
        node(3, "Same", components=[dict(component_id=2, type_id="Test", value=30)])
    )
    changed["root_object"]["components"] = [
        dict(
            component_id=3,
            type_id="Reference",
            target={TYPE_KEY: GAME_OBJECT_REF, "object_id": 3},
            component={TYPE_KEY: COMPONENT_REF, "game_object_id": 3, "component_id": 2, "component_type": "Test"},
        )
    ]
    changed.update(next_local_id=4, next_component_id=4)
    updated = rebase_variant_definition(definition, changed)
    root = updated["document"]["root_object"]
    private = next(child for child in root["children"] if child["local_id"] == 3)
    inherited = next(child for child in root["children"] if child["local_id"] == 4)
    assert private["components"][0]["component_id"] == 2
    assert private["components"][0]["value"] == 9
    assert inherited["components"][0]["value"] == 30
    ref = root["components"][0]
    assert ref["target"]["object_id"] == 4
    assert ref["component"]["game_object_id"] == 4
    assert ref["component"]["component_id"] == inherited["components"][0]["component_id"]
    assert json.loads(json.dumps(updated)) == updated


def test_removed_base_ids_are_reserved_across_save_reload_and_later_additions():
    base = prefab()
    definition = create_variant_definition("base-guid", base)
    removed = copy.deepcopy(base)
    removed["root_object"]["children"] = []
    updated = rebase_variant_definition(definition, removed)
    assert updated["document"]["root_object"]["children"] == []
    assert [2, 2] in updated["base"]["object_sources"]
    newer = copy.deepcopy(removed)
    newer["root_object"]["children"] = [node(3, "Same")]
    newer["next_local_id"] = 4
    reloaded = rebase_variant_definition(json.loads(json.dumps(updated)), newer)
    assert reloaded["document"]["root_object"]["children"][0]["local_id"] == 3


@pytest.mark.parametrize("edit", ["delete", "reparent"])
def test_inherited_objects_cannot_be_removed_or_reparented(edit):
    base = prefab()
    own = copy.deepcopy(base)
    inherited = own["root_object"]["children"].pop()
    if edit == "reparent":
        own["root_object"]["children"] = [node(3, "New parent", [inherited])]
        own["next_local_id"] = 4
    with pytest.raises(PrefabDocumentError, match="inherited|Inherited"):
        create_variant_definition("base-guid", base, own)


def test_component_removal_and_object_deactivation_are_variant_overrides():
    base = prefab()
    own = copy.deepcopy(base)
    own["root_object"]["children"][0].update(active=False, components=[])
    definition = create_variant_definition("base-guid", base, own)
    changed = copy.deepcopy(base)
    changed["root_object"]["children"][0]["components"][0]["value"] = 2
    updated = rebase_variant_definition(definition, changed)
    assert updated["document"]["root_object"]["children"][0]["components"] == []
    assert not updated["document"]["root_object"]["children"][0]["active"]


@pytest.mark.parametrize("mutation", ["duplicate"])
def test_invalid_projections_rejected_at_authoring_boundary(mutation):
    definition = create_variant_definition("base-guid", prefab())
    pairs = definition["base"]["object_sources"]
    if mutation == "duplicate":
        pairs.append([2, 3])
    elif mutation == "missing":
        pairs.pop()
    elif mutation == "watermark":
        pairs.append([3, 3])
    else:
        pairs[0][0] = True
    with pytest.raises(PrefabDocumentError):
        validate_variant_definition(definition)


def test_variant_chain_resolves_shared_base_once_without_modifying_inputs():
    base = prefab()
    first = create_variant_definition("base", base)
    authored = copy.deepcopy(first["document"])
    authored["root_object"]["layer"] = 4
    second = create_variant_definition("first", first["document"], authored)
    peer = create_variant_definition("base", base)
    definitions = {"second": second, "first": first, "peer": peer}
    original = copy.deepcopy(definitions)
    base["root_object"]["name"] = "Updated base"
    base["root_object"]["layer"] = 2
    loaded = []

    def load(guid):
        loaded.append(guid)
        return base

    updated = rebase_variant_graph(definitions, load)
    assert loaded == ["base"]
    assert updated["second"]["document"]["root_object"]["name"] == "Updated base"
    assert updated["second"]["document"]["root_object"]["layer"] == 4
    assert updated["peer"]["document"]["root_object"]["layer"] == 2
    assert definitions == original


@pytest.mark.parametrize("cycle", [False, True])
def test_graph_failure_never_publishes_partial_definition(cycle):
    base = prefab()
    definitions = {
        "first": create_variant_definition("second", base),
        "second": create_variant_definition("first" if cycle else "missing", base),
    }
    before = copy.deepcopy(definitions)

    def missing(guid):
        raise FileNotFoundError(guid)

    with pytest.raises(PrefabDocumentError if cycle else FileNotFoundError):
        rebase_variant_graph(definitions, missing)
    assert definitions == before


def test_nested_source_projection_does_not_rewrite_inner_baseline():
    base = prefab()
    definition = create_variant_definition("base", base)
    # Reserve the next local identity for a Variant-private object.
    definition["document"]["root_object"]["children"].append(node(3, "Private"))
    definition["document"]["next_local_id"] = 4
    nested = node(3, "Nested", components=[dict(component_id=2, type_id="Test")])
    inner = node(10, "Inner", components=[dict(component_id=20, type_id="Test")])
    nested["nested_prefab"] = dict(guid="inner", baseline=inner, object_sources=[[3, 10]], component_sources=[[2, 20]])
    base["root_object"]["children"].append(nested)
    base.update(next_local_id=4, next_component_id=3)
    updated = rebase_variant_definition(definition, base)
    nested = next(n for n in updated["document"]["root_object"]["children"] if "nested_prefab" in n)
    assert nested["local_id"] == 4
    assert nested["nested_prefab"]["object_sources"] == [[4, 10]]
    assert nested["nested_prefab"]["baseline"] == inner


def test_rebased_variant_instantiates_through_native_prefab_path(engine, scene, tmp_path):
    from infernux.engine.prefab_manager import serialize_prefab_document, save_prefab_document, instantiate_prefab

    source = scene.create_game_object("Native base")
    source.add_component("BoxCollider")
    base = serialize_prefab_document(source)
    own = copy.deepcopy(base)
    own["root_object"]["name"] = "Variant collider"
    own["root_object"]["transform"]["scale"] = [2, 3, 4]
    definition = create_variant_definition("base-guid", base, own)
    changed = copy.deepcopy(base)
    changed["root_object"]["layer"] = 5
    updated = rebase_variant_definition(definition, changed)
    path = str(tmp_path / "Resolved.prefab")
    asset_database = engine.get_asset_database()
    assert save_prefab_document(updated["document"], path, asset_database=asset_database)
    guid = str(asset_database.get_guid_from_path(path) or "")
    assert guid
    instance = instantiate_prefab(guid=guid, scene=scene, asset_database=asset_database)
    assert instance is not None
    assert instance.name == "Variant collider (Clone)"
    assert instance.layer == 5
    assert instance.transform.local_scale.y == 3
    assert instance.get_component("BoxCollider") is not None


def test_explicit_variant_overrides_survive_equal_base_edits_and_selective_revert():
    from infernux.engine.prefab_variant import edit_variant_document

    base = prefab()
    own = copy.deepcopy(base)
    own["root_object"]["name"] = "Private"
    own["root_object"]["layer"] = 3
    own["root_object"]["children"][0]["layer"] = 5
    definition = create_variant_definition("base-guid", base, own)

    # Equality with a newer base never implicitly deletes override intent.
    base["root_object"]["layer"] = 3
    base["root_object"]["children"][0]["layer"] = 5
    matching = rebase_variant_definition(definition, base)
    base["root_object"]["layer"] = 7
    base["root_object"]["children"][0]["layer"] = 9
    updated = rebase_variant_definition(json.loads(json.dumps(matching)), base)
    assert updated["document"]["root_object"]["layer"] == 3
    assert updated["document"]["root_object"]["children"][0]["layer"] == 5

    # An explicit user edit to a matching value remains an explicit override.
    current = variant_document(updated)
    authored = copy.deepcopy(current)
    authored["root_object"]["layer"] = 7
    saved = edit_variant_document(current, authored)
    base["root_object"]["layer"] = 11
    updated = rebase_variant_definition(variant_definition(saved), base)
    assert updated["document"]["root_object"]["layer"] == 7

    # Revert only the child's equal-valued override; preserve the root edits.
    base["root_object"]["children"][0]["layer"] = 5
    current = variant_document(rebase_variant_definition(updated, base))
    row = next(
        item for item in variant_property_modifications(current) if item["object"] == 2 and item["path"] == ["layer"]
    )
    assert row["value"] == row["source_value"] == 5
    before = copy.deepcopy(current)
    reverted = revert_variant_property(current, 2, 0, ("layer",))
    assert current == before
    assert reverted["root_object"] == current["root_object"]
    assert {(item["object"], tuple(item["path"])) for item in variant_property_modifications(reverted)} == {
        (1, ("name",)),
        (1, ("layer",)),
    }
    base["root_object"]["children"][0]["layer"] = 13
    later = rebase_variant_definition(variant_definition(reverted), base)["document"]
    assert later["root_object"]["children"][0]["layer"] == 13
    assert later["root_object"]["name"] == "Private"
    assert later["root_object"]["layer"] == 7
