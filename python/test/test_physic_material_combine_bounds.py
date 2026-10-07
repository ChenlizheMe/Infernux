"""Validate combine modes before numeric narrowing, preserving published values."""
import json

import pytest

from infernux.core.physic_material import PhysicMaterial
from infernux.lib import AssetRegistry, InxPhysicMaterial


INVALID = [-1, 4, 255, 256, 257, 259, 260, 512, -256, -255,
           2**31 - 1, -(2**31), 2**32, 2**32 + 3, 2**63 - 1, 2**64 - 1, -(2**63)]


@pytest.mark.parametrize('field', ['friction_combine', 'bounce_combine'])
@pytest.mark.parametrize('entry', ['native', 'public', 'document'])
@pytest.mark.parametrize('value', list(range(4)) + INVALID)
def test_combine_preserves_original_integer_and_atomicity(field, entry, value):
    material = PhysicMaterial() if entry == 'public' else InxPhysicMaterial()
    material.friction_combine = 2
    material.bounce_combine = 3
    before = material.serialize_document()

    def assign():
        if entry == 'document':
            material.deserialize_document(before | {'friction': 0.125, field: value})
        else:
            setattr(material, field, value)

    if value in range(4):
        assign()
        assert getattr(material, field) == value
        assert material.serialize_document()[field] == value
    else:
        with pytest.raises((ValueError, TypeError, OverflowError)):
            assign()
        assert material.serialize_document() == before


@pytest.mark.parametrize('field', ['friction_combine', 'bounce_combine'])
@pytest.mark.parametrize('value', [256, -256, 2**32, 2**64 - 1, 1, 3])
def test_asset_reload_rejects_wrapped_mode_without_partial_publication(engine, tmp_path, field, value):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    source = tmp_path / 'Combine.physmat'
    before = {'friction': 0.5, 'bounciness': 0.25, 'friction_combine': 2, 'bounce_combine': 2}
    source.write_text(json.dumps(before), encoding='utf-8')
    imported = database.import_asset(str(source))
    assert imported, imported.error
    guid = imported.guid
    try:
        material = registry.load_physic_material_by_guid(guid)
        assert material is not None
        source.write_text(json.dumps(before | {'friction': 0.125, field: value}), encoding='utf-8')
        if value in range(4):
            assert registry.reload_asset(guid)
            assert getattr(material, field) == value
            assert material.friction == 0.125
        else:
            version = registry.get_asset_residency(guid).runtime_version
            assert not registry.reload_asset(guid)
            assert material.serialize_document() == before
            assert registry.get_asset_residency(guid).runtime_version == version
            source.write_text(json.dumps(before | {field: 1}), encoding='utf-8')
            assert registry.reload_asset(guid)
            assert getattr(material, field) == 1
        assert material.guid == guid
        assert registry.load_physic_material_by_guid(guid) is material
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(source))
