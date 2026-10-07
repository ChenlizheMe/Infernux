"""Component enumeration must agree with typed lookups and native callbacks."""

import pytest
from types import SimpleNamespace

import infernux as inx


@pytest.mark.parametrize('component_type', [
    inx.BoxCollider, inx.SphereCollider, inx.CapsuleCollider, inx.CylinderCollider,
    inx.Rigidbody, inx.Camera, inx.Light, inx.MeshRenderer, inx.AudioSource,
])
def test_untyped_enumeration_returns_existing_public_component(scene, component_type):
    owner = scene.create_game_object('PublicEnumeration')
    expected = owner.add_component(component_type)
    received, = [component for component in owner.get_components()
                 if component.component_id == expected.component_id]
    assert received is expected is owner.get_component(component_type)
    assert isinstance(received, component_type)
    assert received.game_object is owner
    assert owner.transform in owner.get_components()


def test_untyped_enumeration_preserves_distinct_same_type_components(scene):
    owner = scene.create_game_object('CompoundEnumeration')
    first = owner.add_component(inx.BoxCollider)
    second = owner.add_component(inx.BoxCollider)
    colliders = [component for component in owner.get_components()
                 if component.component_id in (first.component_id, second.component_id)]
    assert len(colliders) == 2
    assert colliders[0] is first and colliders[1] is second


def test_untyped_enumeration_reuses_script_instances_and_builtin_wrappers(scene):
    class EnumerationScript(inx.InxComponent):
        pass

    owner = scene.create_game_object('MixedEnumeration')
    script = owner.add_component(EnumerationScript)
    collider = owner.add_component(inx.BoxCollider)
    components = owner.get_components()
    assert any(component is script for component in components)
    assert any(component is collider for component in components)
    assert any(component is owner.transform for component in components)
    assert len(components) == 3


@pytest.mark.parametrize('direction', ['children', 'parent'])
def test_hierarchy_lookup_reuses_enumeration_component(scene, direction):
    parent = scene.create_game_object('QueryParent')
    child = scene.create_game_object('QueryChild')
    child.set_parent(parent)
    source = child if direction == 'children' else parent
    querying = parent if direction == 'children' else child
    expected = source.add_component(inx.BoxCollider)
    assert getattr(querying, 'get_component_in_' + direction)(inx.BoxCollider) is expected
    assert next(c for c in source.get_components() if c.component_id == expected.component_id) is expected


def test_inspector_classifies_public_builtin_proxy_as_native_component(scene):
    from infernux.engine.bootstrap_inspector._wire import _wire_cache_init, _wire_component_list
    from infernux.lib import InspectorComponentInfo, SceneManager

    class InspectorQueryScript(inx.InxComponent):
        pass

    owner = scene.create_game_object('InspectorPublicEnumeration')
    collider = owner.add_component(inx.BoxCollider)
    script = owner.add_component(InspectorQueryScript)
    ctx = SimpleNamespace(
        SceneManager=SceneManager, InspectorComponentInfo=InspectorComponentInfo,
        InxComponent=inx.InxComponent,
        _inspector_support=SimpleNamespace(get_component_structure_version=lambda: 1),
        get_component_icon_id=lambda *_args: 0,
        engine=SimpleNamespace(get_asset_database=lambda: None), ip=SimpleNamespace(),
    )
    _wire_cache_init(ctx)
    _wire_component_list(ctx)
    records = {record.component_id: record for record in ctx.ip.get_component_list(owner.id)}
    assert records[collider.component_id].is_native
    assert not records[collider.component_id].is_script
    assert records[script.component_id].is_script
    assert ctx.resolve_component(owner.id, collider.component_id, True) is collider
    assert ctx.resolve_component(owner.id, script.component_id, False) is script


def test_undo_snapshot_resolves_exact_builtin_proxy_ordinal(scene):
    from infernux.engine.undo._snapshots import _get_nth_live_native_component

    owner = scene.create_game_object('UndoPublicEnumeration')
    first = owner.add_component(inx.BoxCollider)
    second = owner.add_component(inx.BoxCollider)
    second.size = inx.Vector3(2, 3, 4)
    assert _get_nth_live_native_component(owner.id, 'BoxCollider', 0) is first
    received = _get_nth_live_native_component(owner.id, 'BoxCollider', 1)
    assert received is second
    assert received.serialize_document()['size'] == [2, 3, 4]
