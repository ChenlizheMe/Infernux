"""Authored material publication preserves live identity and refreshes bindings."""
import copy
import json

import pytest

from infernux.core import RenderTexture
from infernux.lib import AssetRegistry, InxMaterial
from test_render_texture_asset import imported_target
from test_shader_guid_rename import rename_assets


@pytest.mark.parametrize('width', [0.5, 1.0, 2.0, 4.0])
def test_line_width_edit_is_published_after_full_state_authorship(width):
    material = InxMaterial('AuthoredLines', 'Unlit')
    state = material.get_render_state()
    state.line_width = 1.25
    material.set_render_state(state)
    before = material.get_version()
    state.line_width = width
    material.set_render_state(state)
    assert material.get_render_state().line_width == width
    assert material.serialize_document()['renderState']['lineWidth'] == width
    assert material.get_version() == before + 1
    material.set_render_state(state)
    assert material.get_version() == before + 1


@pytest.mark.parametrize('api', ['document', 'string', 'registry'])
@pytest.mark.parametrize('override', [False, True])
@pytest.mark.parametrize('edit', ['replace', 'clear', 'remove', 'float', 'invalid'])
def test_document_reconciles_texture_publication(engine, imported_target, tmp_path, api, override, edit):
    _, guid, target_document, database = imported_target
    registry = AssetRegistry.instance()
    second = tmp_path / 'Second.rendertexture'
    target_document = copy.deepcopy(target_document)
    target_document['size'] = {'width': 17, 'height': 11}
    second.write_text(json.dumps(target_document), encoding='utf-8')
    imported = database.import_asset(str(second))
    assert imported, imported.error
    material_path = tmp_path / 'Publication.mat'
    material_guid = None
    try:
        material = InxMaterial.create_default_lit()
        material.set_texture_guid('texSampler', guid)
        if api == 'registry':
            material_path.write_text(material.serialize(), encoding='utf-8')
            result = database.import_asset(str(material_path))
            assert result, result.error
            material_guid = result.guid
            material = registry.load_material_by_guid(material_guid)
        transient = RenderTexture(7, 5) if override else None
        if transient:
            material._set_render_texture('texSampler', transient._native)
        engine._prepare_material_texture_assets(material)
        binding = material._get_render_texture('texSampler')
        assert binding is not None
        assert not material._texture_assets_pending
        before = material.serialize_document()
        version = material.get_version()
        document = copy.deepcopy(before)
        if edit == 'replace':
            document['properties']['texSampler']['guid'] = imported.guid
        elif edit == 'clear':
            document['properties']['texSampler']['guid'] = ''
        elif edit == 'remove':
            del document['properties']['texSampler']
        elif edit == 'float':
            document['properties']['texSampler'] = {'type': 0, 'value': 0.375}
        else:
            document['properties']['texSampler']['guid'] = imported.guid
            document['renderState']['lineWidth'] = 0
        if api == 'registry':
            material_path.write_text(json.dumps(document), encoding='utf-8')
            accepted = registry.reload_asset(material_guid)
            assert registry.load_material_by_guid(material_guid) is material
        else:
            accepted = (material.deserialize_document(document) if api == 'document'
                        else material.deserialize(json.dumps(document)))
        assert accepted == (edit != 'invalid')
        if edit == 'invalid':
            assert material.serialize_document() == before
            assert material.get_version() == version
            assert not material._texture_assets_pending
            assert material._get_render_texture('texSampler') is binding
            return
        assert material._texture_assets_pending
        keep_override = override and edit in ('replace', 'clear')
        assert material._get_render_texture('texSampler') is (binding if keep_override else None)
        engine._prepare_material_texture_assets(material)
        expected = (binding if keep_override else
                    RenderTexture.load_by_guid(imported.guid)._native if edit == 'replace' else None)
        assert material._get_render_texture('texSampler') is expected
        assert not material._texture_assets_pending
        published = material.get_version()
        engine._prepare_material_texture_assets(material)
        assert material.get_version() == published
    finally:
        if material_guid:
            registry.invalidate_asset(material_guid)
            assert database.delete_asset(str(material_path))
        registry.invalidate_asset(imported.guid)
        assert database.delete_asset(str(second))


@pytest.mark.parametrize('stage', ['vertex', 'fragment'])
@pytest.mark.parametrize('edit', ['wrong_type', 'wrong_stage', 'stale_label', 'missing_guid', 'invalid_document'])
def test_registry_shader_resolution_precedes_publication(rename_assets, stage, edit):
    assets = rename_assets
    registry = AssetRegistry.instance()
    guid = assets['material_guid']
    material = registry.load_material_by_guid(guid)
    before = material.serialize_document()
    version = material.get_version()
    residency = registry.get_asset_residency(guid).runtime_version
    document = copy.deepcopy(before)
    document['properties']['metallic']['value'] = 0.75
    reference = document['shaders'][stage]
    if edit == 'wrong_type':
        reference['guid'] = guid
    elif edit == 'wrong_stage':
        reference['guid'] = assets['fragment' if stage == 'vertex' else 'vertex'][1]
    elif edit == 'stale_label':
        reference['shader_id'] = 'Old cached display label'
    elif edit == 'missing_guid':
        reference['guid'] = 'deadbeef' * 4
    else:
        document['renderState']['lineWidth'] = 0
    assets['material_path'].write_text(json.dumps(document), encoding='utf-8')
    accepted = registry.reload_asset(guid)
    assert accepted == (edit in ('stale_label', 'missing_guid'))
    if not accepted:
        assert material.serialize_document() == before
        assert material.get_version() == version
        assert registry.get_asset_residency(guid).runtime_version == residency
        assets['material_path'].write_text(json.dumps(before), encoding='utf-8')
        assert registry.reload_asset(guid)
    else:
        assert material.get_float('metallic') == 0.75
        expected = before['shaders'][stage] if edit == 'stale_label' else reference
        assert material.serialize_document()['shaders'][stage] == expected
    assert material.guid == guid
    assert registry.load_material_by_guid(guid) is material


@pytest.mark.parametrize('stage', ['vertex', 'fragment'])
@pytest.mark.parametrize('edit', ['wrong_type', 'wrong_stage', 'stale_label'])
def test_first_load_validates_resolved_shader_identity(rename_assets, stage, edit):
    assets = rename_assets
    registry = AssetRegistry.instance()
    guid = assets['material_guid']
    original = assets['material_path'].read_text(encoding='utf-8')
    document = json.loads(original)
    reference = document['shaders'][stage]
    if edit == 'wrong_type':
        reference['guid'] = guid
    elif edit == 'wrong_stage':
        reference['guid'] = assets['fragment' if stage == 'vertex' else 'vertex'][1]
    else:
        reference['shader_id'] = 'Old cached display label'
    assets['material_path'].write_text(json.dumps(document), encoding='utf-8')
    material = registry.load_material_by_guid(guid)
    if edit == 'stale_label':
        assert material is not None
        assert material.serialize_document()['shaders'][stage]['shader_id'] == assets[stage][2]
    else:
        assert material is None
        assets['material_path'].write_text(original, encoding='utf-8')
        registry.invalidate_asset(guid)
        assert registry.load_material_by_guid(guid) is not None
