"""Inspector selection and native authoring payload contracts."""

from infernux.lib import (
    InspectorAddComponentEntry,
    InspectorComponentInfo,
    InspectorObjectInfo,
    InspectorPanel,
    InspectorPrefabInfo,
    InspectorTransformData,
)


def test_inspector_switches_between_object_asset_and_detail_selection():
    panel = InspectorPanel()
    panel.set_selected_object_id(42)
    assert panel.get_selected_object_id() == 42

    panel.set_detail_file("Assets/Selected.mat", "material")
    assert panel.get_selected_object_id() == 42
    assert panel.get_selected_file() == "Assets/Selected.mat"

    panel.set_selected_file("Assets/Other.mat", "material")
    assert panel.get_selected_object_id() == 0
    assert panel.get_selected_file() == "Assets/Other.mat"
    panel.clear_selected_file()
    assert panel.get_selected_file() == ""
    panel.set_selected_object_id(99)
    panel.clear_selected_object()
    assert panel.get_selected_object_id() == 0


def test_native_authoring_payload_preserves_identity_transform_and_prefab_state():
    component = InspectorComponentInfo()
    component.type_name = "MeshRenderer"
    component.component_id = 123
    component.enabled = True
    component.is_native = True
    component.is_script = False
    component.is_broken = False

    owner = InspectorObjectInfo()
    owner.name = "Cube"
    owner.active = True
    owner.tag = "Player"
    owner.layer = 3
    owner.transform_component_id = 91
    owner.driven_transform_properties = 5

    transform = InspectorTransformData()
    transform.px, transform.py_, transform.pz = 1.0, 2.0, 3.0
    transform.ry = 90.0
    transform.sx = transform.sy = transform.sz = 1.0

    entry = InspectorAddComponentEntry()
    entry.display_name = "Light"
    entry.category = "Built-in"
    entry.is_native = True
    entry.script_path = ""
    prefab = InspectorPrefabInfo()
    prefab.override_count = 3
    prefab.is_readonly = True
    prefab.is_transform_readonly = False

    assert (component.type_name, component.component_id, component.enabled) == (
        "MeshRenderer", 123, True,
    )
    assert component.is_native and not component.is_script and not component.is_broken
    assert (owner.name, owner.tag, owner.layer) == ("Cube", "Player", 3)
    assert (owner.transform_component_id, owner.driven_transform_properties) == (91, 5)
    assert (transform.px, transform.py_, transform.pz, transform.ry) == (1, 2, 3, 90)
    assert (transform.sx, transform.sy, transform.sz) == (1, 1, 1)
    assert (entry.display_name, entry.category, entry.is_native) == ("Light", "Built-in", True)
    assert prefab.override_count == 3 and prefab.is_readonly
    assert not prefab.is_transform_readonly
