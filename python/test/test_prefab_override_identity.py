"""Reference identity domains and repeated records in prefab override comparison."""

import copy

import pytest

from Infernux.components.value_document import make_component_ref, make_game_object_ref
from Infernux.engine.prefab_overrides import _diff_components, _same_value


@pytest.mark.parametrize("reference", [make_game_object_ref, lambda value: make_component_ref(value, "Target")])
def test_reference_comparison_keeps_scene_and_asset_ids_separate(reference):
    ids = {0: 0, 101: 1, 102: 2}
    assert _same_value(reference(102), reference(2), ids)
    assert _same_value(reference(0), reference(0), ids)
    assert not _same_value(reference(101), reference(2), ids)
    assert not _same_value(reference(0), reference(2), ids)
    assert not _same_value(reference(2), reference(2), ids)
    assert not _same_value(reference(103), reference(3), ids)


def test_reference_comparison_traverses_nested_fields_without_mutating_values():
    instance = {"nested": [{"ref": make_game_object_ref(102), "number": 102}]}
    source = {"nested": [{"ref": make_game_object_ref(2), "number": 102}]}
    before = copy.deepcopy(instance)
    assert _same_value(instance, source, {102: 2})
    assert instance == before
    source["nested"][0]["number"] = 2
    assert not _same_value(instance, source, {102: 2})
    assert not _same_value(make_component_ref(102, "First"), make_component_ref(2, "Second"), {102: 2})


def test_duplicate_component_types_compare_each_occurrence_once():
    sources = [
        {"type_id": "Collider", "component_id": index, "data": {"size": size}}
        for index, size in enumerate((2, 5), start=1)
    ]
    instances = copy.deepcopy(sources)
    for instance in instances:
        instance["component_id"] += 100
        instance["instance_guid"] = str(instance["component_id"])
    overrides = []
    _diff_components(instances, sources, "Root", "components", overrides, {})
    assert overrides == []
    instances[1]["data"]["size"] = 7
    _diff_components(instances, sources, "Root", "components", overrides, {})
    assert len(overrides) == 1
    assert overrides[0].prefab_value == {"size": 5}
    assert overrides[0].instance_value == {"size": 7}


@pytest.mark.parametrize("added", [False, True])
def test_duplicate_component_count_changes_are_reported(added):
    one = [{"type_id": "Collider", "data": {}}]
    two = one * 2
    overrides = []
    _diff_components(two if added else one, one if added else two, "Root", "components", overrides, {})
    assert len(overrides) == 1
    assert overrides[0].key == ("added_components:Collider" if added else "removed_components:Collider")
