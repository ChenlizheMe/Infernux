"""Imported targets share native ownership and the ordinary field/asset chain."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import Infernux as inx
from Infernux.application import Application
from Infernux.components.fields import resolve_annotation, get_serialized_fields
from Infernux.core import AssetManager, RenderTexture, RenderTextureRef
from Infernux.lib import AssetRegistry, PixelFormat


def test_project_file_manager_creates_render_texture_description(tmp_path):
    from Infernux.engine.ui.project_file_ops import create_render_texture

    ok, error = create_render_texture(str(tmp_path), "GameplayTarget")
    assert ok, error
    asset = tmp_path / "GameplayTarget.rendertexture"
    assert asset.is_file()
    document = json.loads(asset.read_text(encoding="utf-8"))
    assert document["$type"] == "render_texture"
    assert "schema_version" not in document
    assert document["size"] == {"width": 256, "height": 256}
    assert document["depth_format"] == "d32_sfloat"
    ok, error = create_render_texture(str(tmp_path), "GameplayTarget")
    assert not ok
    assert "already exists" in error


class MonitorDescription(inx.SerializableObject):
    __serialized_type_id__ = 'tests.render_texture.monitor'
    target: inx.RenderTexture
    mirrors: list[inx.RenderTextureRef]


def test_raw_image_alias_reuses_the_managed_image_component():
    from Infernux.ui import UIImage, UIRawImage

    assert UIRawImage is UIImage


def test_image_persistent_target_uses_concrete_guid_and_shared_owner(imported_target):
    from Infernux.ui import UIImage
    from Infernux.components.fields import get_raw_field_value
    source, guid, _, _ = imported_target
    image = UIImage()
    image.texture = RenderTextureRef(guid, str(source))
    assert not AssetRegistry.instance().is_loaded(guid)
    document = image._serialize_fields_document()
    assert 'texture_path' not in document
    assert document['texture']['asset_type'] == 'RenderTexture'
    assert document['texture']['guid'] == guid
    restored = UIImage()
    restored._deserialize_fields_document(document)
    assert isinstance(get_raw_field_value(restored, 'texture'), RenderTextureRef)
    assert restored.texture._native is image.texture._native
    override = RenderTexture(17, 11)
    restored.texture = override
    assert restored.texture is override
    assert restored._serialize_fields_document() == document
    restored.texture = None
    assert restored.texture._native is image.texture._native
    restored.texture = None
    assert restored._serialize_fields_document()['texture'] is None


def test_image_target_reimport_delete_and_restore_keeps_saved_guid(imported_target):
    from Infernux.ui import UIImage
    source, guid, document, database = imported_target
    image = UIImage()
    image.texture = RenderTexture.load_by_guid(guid)
    saved = image._serialize_fields_document()
    owner = image.texture._native
    document['size'] = {'width': 71, 'height': 43}
    publish(source, document, database)
    assert image.texture._native is owner
    assert image.texture.width == 71
    meta = Path(str(source) + '.meta').read_bytes()
    AssetManager.delete_asset(str(source), database=database)
    assert image.texture is None
    assert image._serialize_fields_document()['texture']['guid'] == guid
    # The normal asset mutation publication restores the deleted identity.
    Path(str(source) + '.meta').write_bytes(meta)
    source.write_text(json.dumps(document), encoding='utf-8')
    restored = AssetManager.import_asset(str(source), database=database)
    assert restored.succeeded, restored.error
    assert image.texture.guid == guid
    assert image._serialize_fields_document()['texture']['asset_type'] == 'RenderTexture'
    image._deserialize_fields_document(saved)
    assert image.texture.width == 71


def test_image_legacy_texture_path_migrates_to_single_guid_slot(imported_target, tmp_path):
    from Infernux.ui import UIImage
    from Infernux.components.fields import get_raw_field_value
    from Infernux.core.asset_ref import TextureRef
    _, _, _, database = imported_target
    from PIL import Image
    path = tmp_path / 'Image.png'
    Image.new('RGBA', (13, 7), 'cyan').save(path)
    imported = database.import_asset(str(path))
    assert imported.succeeded
    image = UIImage()
    image._deserialize_fields_document({'texture_path': str(path)})
    ref = get_raw_field_value(image, 'texture')
    assert isinstance(ref, TextureRef) and ref.guid == imported.guid
    document = image._serialize_fields_document()
    assert document['texture']['asset_type'] == 'Texture'
    assert 'texture_path' not in document
    assert image.texture.width == 13
    assert image.texture.height == 7
    canonical = dict(document, texture=None, texture_path=str(path))
    image._deserialize_fields_document(canonical)
    assert image.texture is None  # New documents win; no legacy resurrection.


def test_sampled_field_rejects_union_identity_and_keeps_static_fields_narrow():
    from Infernux.ui import UIImage
    from Infernux.components.value_codec import VALUE_CODECS
    from Infernux.components.value_document import make_asset_ref
    metadata = get_serialized_fields(UIImage)['texture']
    assert metadata.asset_type == 'Texture.Sampled'
    for kind in ('Texture', 'RenderTexture'):
        VALUE_CODECS.validate(make_asset_ref(kind, 'deleted-guid'), metadata)
    for kind in ('Texture.Sampled', 'Material'):
        with pytest.raises(TypeError, match='requires'):
            VALUE_CODECS.validate(make_asset_ref(kind, 'deleted-guid'), metadata)
    with pytest.raises(TypeError, match='requires'):
        VALUE_CODECS.validate(make_asset_ref('RenderTexture', 'deleted-guid'), inx.FieldType.TEXTURE)


def test_image_inspector_assignment_uses_concrete_type(imported_target):
    from Infernux.ui import UIImage
    from Infernux.engine.ui._inspector_references import _create_asset_ref_from_payload
    source, guid, _, _ = imported_target
    metadata = get_serialized_fields(UIImage)['texture']
    ref = _create_asset_ref_from_payload(metadata, str(source))
    assert isinstance(ref, RenderTextureRef) and ref.guid == guid


def test_image_real_drawer_keeps_all_descriptor_drop_types(monkeypatch):
    from Infernux.ui import UIImage
    from Infernux.engine.ui import _inspector_references as drawer
    from Infernux.engine.ui.igui import IGUI
    monkeypatch.setattr(drawer, 'field_label', lambda *args: None)
    monkeypatch.setattr(drawer, 'semantic_capture_enabled', lambda ctx: False)
    models = []
    monkeypatch.setattr(IGUI, 'object_field_model', lambda ctx, model, **kwargs: models.append(model))
    image = UIImage()
    for name, kind in [('texture', inx.FieldType.ASSET), ('material', inx.FieldType.MATERIAL)]:
        drawer._render_asset_reference_field(object(), image, name,
            get_serialized_fields(UIImage)[name], None, kind, 80)
    assert 'RENDER_TEXTURE_FILE' in models[0].accept
    assert 'TEXTURE_FILE' in models[0].accept
    assert models[1].asset_type == 'Material'


def test_image_object_field_document_transaction_clear_override_and_undo(imported_target):
    from Infernux.ui import UIImage
    from Infernux.engine.ui._inspector_references import _create_asset_ref_from_payload
    from Infernux.engine.interaction import AssetReferenceFieldModel, make_python_component_property_transaction
    from Infernux.engine.undo import UndoManager
    source, guid, _, _ = imported_target
    image = UIImage()
    meta = get_serialized_fields(UIImage)['texture']
    previous, manager = UndoManager._instance, UndoManager()
    try:
        transaction = make_python_component_property_transaction(
            (image,), 'texture', decode_input=lambda value:
                _create_asset_ref_from_payload(meta, value) if value is not None else None)
        model = AssetReferenceFieldModel(field_id='image', display_text='None',
            type_hint='Texture.Sampled', transaction=transaction)
        model.dispatch_drop(str(source))
        assert image._serialize_fields_document()['texture']['guid'] == guid
        manager.clear()  # A separate authoring action, outside the drop gesture.
        image.texture = RenderTexture(11, 7)
        assert model.clear_reference()
        assert image.texture is None
        assert image._serialize_fields_document()['texture'] is None
        manager.undo()
        assert image.texture.guid == guid
        manager.redo()
        assert image.texture is None
        # The inherited material slot uses the same document/schema drawer.
        AssetReferenceFieldModel(field_id='material', display_text='None', type_hint='Material',
            transaction=make_python_component_property_transaction((image,), 'material'))
        with pytest.raises(ValueError, match='does not match'):
            AssetReferenceFieldModel(field_id='wrong', display_text='None', type_hint='Material',
                transaction=transaction)
    finally:
        manager.clear()
        UndoManager._instance = previous


@pytest.fixture
def imported_target(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    monkeypatch.setattr(AssetManager, '_registry', AssetRegistry.instance())
    monkeypatch.setattr(AssetManager, '_native_engine', classmethod(lambda cls: engine))
    monkeypatch.setattr(Application, '_current_engine', staticmethod(
        lambda: SimpleNamespace(get_native_engine=lambda: engine)))
    source = tmp_path / 'Monitor.rendertexture'
    document = {'$type': 'render_texture', 'schema_version': 1,
                'size': {'width': 53, 'height': 29}, 'format': 'rgba8_unorm',
                'depth_format': 'd32_sfloat', 'samples': 1, 'filter': 'linear',
                'storage': False, 'sampled_depth': False}
    source.write_text(json.dumps(document), encoding='utf-8')
    result = database.import_asset(str(source))
    assert result.succeeded, result.error
    guid = result.guid
    try:
        yield source, guid, document, database
    finally:
        AssetManager.invalidate(guid)
        if source.exists():
            AssetManager.delete_asset(str(source), database=database)


def publish(source, document, database):
    source.write_text(json.dumps(document), encoding='utf-8')
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result.succeeded, result.error


@pytest.mark.parametrize('annotation', [inx.RenderTexture, inx.RenderTextureRef,
                                      'inx.RenderTexture', 'inx.RenderTextureRef'])
def test_registered_annotations_use_one_asset_schema(annotation):
    meta = resolve_annotation(annotation)
    assert meta.field_type == inx.FieldType.ASSET
    assert meta.asset_type == 'RenderTexture'
    assert isinstance(meta.default, RenderTextureRef)


def test_material_imported_target_keeps_cpu_document_and_shared_gpu_owner(imported_target, engine):
    source, guid, _, _ = imported_target
    material = inx.Material.create_unlit('AssetMonitor')
    material.set_texture_guid('texSampler', guid)
    document = material.to_dict()
    assert document['properties']['texSampler']['guid'] == guid
    assert not AssetRegistry.instance().is_loaded(guid)
    assert material.native._get_render_texture('texSampler') is None
    engine._prepare_material_texture_assets(material.native)
    target = RenderTexture.load_by_guid(guid)
    assert material.native._get_render_texture('texSampler') is target._native
    assert material.to_dict() == document
    assert not material.native._texture_assets_pending
    material.set_float('some_numeric_parameter', .5)
    assert not material.native._texture_assets_pending
    restored = inx.Material.create_unlit('ReloadedMonitor')
    assert restored.deserialize_document(document)
    engine._prepare_material_texture_assets(restored.native)
    assert restored.native._get_render_texture('texSampler') is target._native


def test_material_target_reimport_delete_restore_includes_runtime_clones(imported_target, engine):
    source, guid, document, database = imported_target
    meta_path = Path(str(source) + '.meta')
    metadata = meta_path.read_text(encoding='utf-8')
    material = inx.Material.create_unlit('OriginalMonitor')
    target = RenderTexture.load_by_guid(guid)
    material.set_texture('texSampler', target)
    assert material.to_dict()['properties']['texSampler']['guid'] == guid
    clone = material.clone()
    for item in (material, clone):
        engine._prepare_material_texture_assets(item.native)
        assert item.native._get_render_texture('texSampler') is target._native
    document['size'] = {'width': 91, 'height': 47}
    publish(source, document, database)
    for item in (material, clone):
        # Reimport already replaced the resource generation atomically. Do not
        # drop a live owner and allocate another while rebuilding descriptors.
        assert item.native._get_render_texture('texSampler') is target._native
        engine._prepare_material_texture_assets(item.native)
        assert item.native._get_render_texture('texSampler').width == 91
    assert AssetManager.delete_asset(str(source), database=database).succeeded
    for item in (material, clone):
        assert item.native._get_render_texture('texSampler') is None
        assert item.to_dict()['properties']['texSampler']['guid'] == guid
        engine._prepare_material_texture_assets(item.native)
    source.write_text(json.dumps(document), encoding='utf-8')
    meta_path.write_text(metadata, encoding='utf-8')
    result = database.import_asset(str(source))
    assert result.succeeded, result.error
    for item in (material, clone):
        engine._prepare_material_texture_assets(item.native)
        assert item.native._get_render_texture('texSampler') is target._native


def test_material_runtime_override_does_not_replace_persistent_slot(imported_target, engine):
    source, guid, _, database = imported_target
    material = inx.Material.create_unlit('OverrideMonitor')
    material.set_texture_guid('texSampler', guid)
    transient = RenderTexture(7, 5)
    material.set_texture('texSampler', transient)
    engine._prepare_material_texture_assets(material.native)
    assert material.native._get_render_texture('texSampler') is transient._native
    assert material.to_dict()['properties']['texSampler']['guid'] == guid
    clone = material.clone()
    engine._prepare_material_texture_assets(clone.native)
    assert clone.native._get_render_texture('texSampler') is transient._native
    assert AssetManager.delete_asset(str(source), database=database).succeeded
    assert clone.native._get_render_texture('texSampler') is transient._native
    clone.clear_texture('texSampler')
    engine._prepare_material_texture_assets(clone.native)
    assert clone.native._get_render_texture('texSampler') is None
    assert clone.to_dict()['properties']['texSampler']['guid'] == ''


def test_ui_material_samples_imported_target_without_source_path_lookup(imported_target, engine):
    from Infernux.ui.ui_render_dispatch import material_visual_state, image_texture_source
    _, guid, _, _ = imported_target
    material = inx.Material.create_unlit('UIMonitor')
    material.set_texture_guid('texSampler', guid)
    from Infernux.ui import UIImage
    image = UIImage()
    image.material = material
    state = material_visual_state(image)
    assert state['texture_path'] == ''
    assert state['texture'] is RenderTexture.load_by_guid(guid)._native
    assert image_texture_source(image, state) is state['texture']


def test_sampled_texture_picker_accepts_targets_without_widening_static_texture_fields(imported_target):
    from Infernux.core.asset_reference_types import asset_type_registry
    source, guid, _, _ = imported_target
    payload = {'asset_type': 'RenderTexture', 'guid': guid, 'path_hint': str(source)}
    assert asset_type_registry.require('Texture.Sampled').incompatibility(payload) == ''
    assert asset_type_registry.require('Texture').incompatibility(payload)


def test_load_path_guid_reference_and_camera_share_owner(imported_target, scene):
    source, guid, _, _ = imported_target
    target = RenderTexture.load(str(source))
    assert target.guid == guid and Path(target.file_path) == source
    assert target.display_name == source.name
    from Infernux.engine.ui._inspector_references import _get_reference_display_name
    assert _get_reference_display_name(inx.FieldType.ASSET, target) == source.name
    assert (target.width, target.height) == (53, 29)
    assert RenderTexture.load_by_guid(guid)._native is target._native
    assert AssetManager.load(str(source))._native is target._native
    ref = RenderTextureRef(guid)
    assert ref.resolve()._native is target._native
    camera = scene.create_game_object('AssetCamera').add_component('Camera')
    try:
        camera.target_texture = target
        assert camera.target_texture._native is target._native
        assert camera.target_texture.guid == guid
        assert RenderTexture(2, 2).guid == ''
        assert RenderTexture(2, 2).display_name == 'RenderTexture'
    finally:
        camera.target_texture = None


def test_new_project_asset_assigns_to_camera_without_depth_setup(imported_target, scene, engine, tmp_path):
    from Infernux.engine.ui.project_file_ops import create_render_texture
    _, _, _, database = imported_target
    assert create_render_texture(str(tmp_path), 'NewCameraTarget') == (True, '')
    source = tmp_path / 'NewCameraTarget.rendertexture'
    imported = database.import_asset(str(source))
    assert imported.succeeded, imported.error
    camera = scene.create_game_object('NewTargetCamera').add_component('Camera')
    try:
        camera.target_texture = RenderTextureRef(imported.guid, str(source))
        engine.resize_game_render_target(320, 180)
        assert engine.get_game_texture_id() == 0
        target = camera.target_texture
        assert isinstance(target, RenderTexture)
        assert target.guid == imported.guid and target.depth_format == PixelFormat.D32_SFLOAT
        assert (target.width, target.height) == (256, 256)
        assert camera._require_cpp_component().serialize_document()['targetTextureGuid'] == imported.guid
    finally:
        camera.target_texture = None
        AssetManager.delete_asset(str(source), database=database)


def test_camera_native_schema_exposes_one_persistent_asset_slot():
    field = inx.Camera.target_texture
    assert field.metadata.asset_type == 'RenderTexture'
    assert field.schema.attributes['serialized_name'] == 'targetTextureGuid'
    assert field.schema.attributes['nullable']
    assert isinstance(field.metadata.default, RenderTextureRef)


def test_camera_scene_identity_decodes_without_allocating_then_renders(imported_target, scene, engine):
    _, guid, _, _ = imported_target
    camera = scene.create_game_object('AuthoredMonitorCamera').add_component('Camera')
    cpp = camera._require_cpp_component()
    document = cpp.serialize_document() | {'targetTextureGuid': guid}
    assert cpp.deserialize_document(document)
    assert cpp.target_texture is None
    assert not AssetRegistry.instance().is_loaded(guid)
    assert isinstance(camera.target_texture, RenderTextureRef)
    assert camera.target_texture.guid == guid
    from Infernux.engine.ui.inspector_components import _declared_native_fields
    for instance in (cpp, camera):
        projected = {name: value for name, _, _, value in _declared_native_fields(instance)}
        assert projected['targetTextureGuid'].guid == guid
    assert not AssetRegistry.instance().is_loaded(guid)
    # A normal renderer query prepares the authored camera, including Edit mode.
    engine.resize_game_render_target(320, 180)
    assert engine.get_game_texture_id() == 0  # No screen-output camera in this scene.
    target = camera.target_texture
    assert isinstance(target, RenderTexture) and target.guid == guid
    assert (target.width, target.height) == (53, 29)
    version = scene.structure_version
    for _ in range(8):
        assert engine.get_game_texture_id() == 0
    assert scene.structure_version == version  # No per-query rebinding/publication.
    assert cpp.serialize_document() == document


def test_camera_imported_assignment_clone_clear_and_legacy_scene(imported_target, scene):
    _, guid, _, _ = imported_target
    owner = scene.create_game_object('SavedTargetCamera')
    camera = owner.add_component('Camera')
    camera.target_texture = RenderTextureRef(guid)
    saved = camera.serialize_document()
    assert saved['targetTextureGuid'] == guid
    clone = scene._clone_game_object(owner)
    cloned_camera = clone.get_component(inx.Camera)
    assert cloned_camera.serialize_document()['targetTextureGuid'] == guid
    assert cloned_camera.target_texture._native is camera.target_texture._native
    camera.target_texture = None
    assert camera.serialize_document()['targetTextureGuid'] == ''
    assert camera.deserialize_document(saved)
    assert camera.serialize_document()['targetTextureGuid'] == guid
    legacy = dict(saved)
    legacy.pop('targetTextureGuid')
    assert camera.deserialize_document(legacy)
    assert camera.target_texture is None


def test_camera_missing_target_does_not_become_screen_camera(scene, engine):
    engine.resize_game_render_target(320, 180)
    camera = scene.create_game_object('MissingTargetCamera').add_component('Camera')
    missing_guid = '1234567890abcdef1234567890abcdef'
    camera._require_cpp_component().target_texture_guid = missing_guid
    assert camera.target_texture.guid == missing_guid
    assert engine.get_game_texture_id() == 0
    screen = scene.create_game_object('ScreenCamera').add_component('Camera')
    assert engine.get_game_texture_id() != 0
    screen.enabled = False
    assert engine.get_game_texture_id() == 0
    assert camera.serialize_document()['targetTextureGuid'] == missing_guid


def test_camera_delete_and_restore_preserve_guid_and_reconnect(imported_target, scene, engine):
    engine.resize_game_render_target(320, 180)
    source, guid, document, database = imported_target
    metadata = Path(str(source) + '.meta').read_bytes()
    camera = scene.create_game_object('RestorableTargetCamera').add_component('Camera')
    camera.target_texture = RenderTexture.load_by_guid(guid)
    lease = camera.target_texture
    assert AssetManager.delete_asset(str(source), database=database).succeeded
    assert camera._require_cpp_component().target_texture is None
    assert camera.serialize_document()['targetTextureGuid'] == guid
    assert engine.get_game_texture_id() == 0
    document['size'] = {'width': 83, 'height': 43}
    source.write_text(json.dumps(document), encoding='utf-8')
    Path(str(source) + '.meta').write_bytes(metadata)
    assert database.import_asset(str(source)).guid == guid
    assert engine.get_game_texture_id() == 0
    assert camera.target_texture._native is lease._native
    assert (camera.target_texture.width, camera.target_texture.height) == (83, 43)


@pytest.mark.parametrize('scope', ['scene', 'subtree'])
def test_camera_asset_events_follow_published_component_identity(imported_target, scene, engine, scope):
    from Infernux.lib import _Infernux as native
    source, guid, _, database = imported_target
    owner = scene.create_game_object('ReopenedTargetCamera')
    camera = owner.add_component('Camera')
    camera.target_texture = RenderTexture.load_by_guid(guid)
    published_id = camera.component_id
    document = scene.serialize_document()
    native._preflight_scene_resource_dependencies(document)
    if scope == 'scene':
        assert scene._commit_document(document)
    else:
        assert owner._commit_document(owner.serialize_document())
    camera = scene.find('ReopenedTargetCamera').get_component(inx.Camera)
    assert camera.component_id == published_id
    engine.resize_game_render_target(320, 180)
    engine.get_game_texture_id()
    assert isinstance(camera.target_texture, RenderTexture)
    assert AssetManager.delete_asset(str(source), database=database).succeeded
    assert camera._require_cpp_component().target_texture is None
    assert camera.target_texture.guid == guid
    assert engine.get_game_texture_id() == 0


def test_camera_target_is_in_native_scene_cook_dependency_closure(imported_target, scene):
    from Infernux.lib import _Infernux as native
    _, guid, _, _ = imported_target
    camera = scene.create_game_object('CookedTargetCamera').add_component('Camera')
    camera._require_cpp_component().target_texture_guid = guid
    dependencies = native._collect_scene_resource_dependencies(scene.serialize_document())
    assert (guid, 'RenderTexture') in [tuple(value) for value in dependencies]
    assert not AssetRegistry.instance().is_loaded(guid)
    from Infernux.engine.runtime_artifact_catalog import _asset_refs
    assert (guid, '') in list(_asset_refs(scene.serialize_document()))


def test_camera_asset_slot_edit_undo_redo_uses_native_identity(imported_target, scene):
    from Infernux.engine.ui.inspector_components import _apply_multi_builtin_change
    from Infernux.engine.undo import UndoManager
    _, guid, _, _ = imported_target
    camera = scene.create_game_object('UndoTargetCamera').add_component('Camera')
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        _apply_multi_builtin_change(
            (camera,), ('target_texture', 'target_texture'),
            inx.Camera.target_texture.metadata, RenderTextureRef(guid),
        )
        assert camera.serialize_document()['targetTextureGuid'] == guid
        manager.undo()
        assert camera.target_texture is None
        manager.redo()
        assert camera.serialize_document()['targetTextureGuid'] == guid
    finally:
        manager.clear()
        UndoManager._instance = previous


def test_automation_camera_asset_slot_uses_public_adapter_and_shared_history(imported_target, scene, monkeypatch):
    from Infernux.host import EditorAutomationHost
    from Infernux.engine.interaction.components import ComponentCommandService
    from Infernux.engine.undo import UndoManager

    _, guid, _, _ = imported_target
    owner = scene.create_game_object('AutomationTargetCamera')
    camera = owner.add_component('Camera')
    previous = UndoManager._instance, ComponentCommandService._instance
    manager, service = UndoManager(), ComponentCommandService()
    host = EditorAutomationHost()
    monkeypatch.setattr(host, 'scene_component', lambda *_: camera)
    monkeypatch.setattr(host, 'interaction_core', lambda: SimpleNamespace(components=service))
    try:
        host.set_scene_component_field(owner.id, camera.component_id, 'targetTextureGuid',
                                       {'$type': 'asset_ref', 'asset_type': 'RenderTexture',
                                        'guid': guid, 'path_hint': ''})
        assert camera.serialize_document()['targetTextureGuid'] == guid
        assert camera.target_texture._native is RenderTexture.load_by_guid(guid)._native
        manager.undo()
        assert camera.target_texture is None
        manager.redo()
        assert camera.serialize_document()['targetTextureGuid'] == guid
        host.set_scene_component_field(owner.id, camera.component_id, 'targetTextureGuid', None)
        assert camera.target_texture is None
    finally:
        manager.clear()
        UndoManager._instance, ComponentCommandService._instance = previous


def test_serialized_fields_keep_guids_not_native_handles_or_pixels(imported_target):
    source, guid, _, _ = imported_target
    target = RenderTexture.load(str(source))
    value = MonitorDescription(target=target, mirrors=[RenderTextureRef(guid)])
    fields = get_serialized_fields(MonitorDescription)
    assert fields['target'].asset_type == fields['mirrors'].asset_type == 'RenderTexture'
    document = value._serialize()
    assert document['fields']['target']['guid'] == guid
    assert document['fields']['mirrors'][0]['guid'] == guid
    restored = inx.SerializableObject._deserialize(json.loads(json.dumps(document)))
    assert restored.target._native is target._native
    assert restored.mirrors[0]._native is target._native


def test_reimport_updates_live_owner_across_fixed_and_relative_modes(imported_target, engine, scene):
    source, guid, document, database = imported_target
    target = RenderTexture.load_by_guid(guid)
    native = target._native
    camera = scene.create_game_object('ReimportCamera').add_component('Camera')
    before = engine.renderer_frame_snapshot
    previous = before['game_target_width'], before['game_target_height']
    try:
        engine.resize_game_render_target(641, 401)
        camera.target_texture = target
        document.update(size={'scale': [.5, .25]}, format='rgba16_sfloat', samples=4, sampled_depth=True)
        publish(source, document, database)
        assert target._native is native
        assert camera.target_texture._native is native
        assert (target.width, target.height) == (321, 101)
        assert target.format == PixelFormat.RGBA16_SFLOAT and target.samples == 4
        assert target.sampled_depth and target.scale == (.5, .25)
        engine.resize_game_render_target(803, 601)
        assert (target.width, target.height) == (402, 151)
        document['size'] = {'width': 71, 'height': 41}
        publish(source, document, database)
        assert target.scale is None
        engine.resize_game_render_target(640, 360)
        assert (target.width, target.height) == (71, 41)
        document['size'] = {'scale': [.5, .5]}
        publish(source, document, database)
        engine.resize_game_render_target(803, 601)
        assert (target.width, target.height) == (402, 301)
    finally:
        camera.target_texture = None
        if all(previous):
            engine.resize_game_render_target(*previous)


def test_device_rejection_does_not_publish_cpu_or_gpu_generation(imported_target):
    source, guid, document, database = imported_target
    target = RenderTexture.load_by_guid(guid)
    registry = AssetRegistry.instance()
    previous = registry.get_asset_version(guid), target.revision
    document['size'] = {'width': 2**32 - 1, 'height': 29}
    # This is a valid portable description, but cannot be allocated on this GPU.
    source.write_text(json.dumps(document), encoding='utf-8')
    assert database.reimport_asset(str(source)).succeeded
    with pytest.raises(ValueError, match='limits'):
        registry.reload_asset(guid)
    assert (registry.get_asset_version(guid), target.revision) == previous
    assert (target.width, target.height) == (53, 29)
    document['size'] = {'width': 57, 'height': 31}
    publish(source, document, database)
    assert (target.width, target.height) == (57, 31)


def test_camera_depth_requirement_survives_disable_clone_and_owner_removal(imported_target, scene):
    source, guid, document, database = imported_target
    target = RenderTexture.load_by_guid(guid)
    owner = scene.create_game_object('DepthConsumer')
    camera = owner.add_component('Camera')
    camera.target_texture = target
    clone = scene._clone_game_object(owner)
    clone_camera = clone.get_component('Camera')
    camera.enabled = clone_camera.enabled = False
    document['depth_format'] = 'undefined'
    source.write_text(json.dumps(document), encoding='utf-8')
    assert database.reimport_asset(str(source)).succeeded
    registry = AssetRegistry.instance()
    before = registry.get_asset_version(guid), target.revision
    with pytest.raises(ValueError, match='depth attachment'):
        registry.reload_asset(guid)
    assert (registry.get_asset_version(guid), target.revision) == before
    owner.remove_component(camera)
    with pytest.raises(ValueError, match='depth attachment'):
        registry.reload_asset(guid)
    scene.destroy_game_object(clone)
    scene.process_pending_destroys()
    assert registry.reload_asset(guid)
    assert target.depth_format == PixelFormat.UNDEFINED


def test_graphics_lease_pins_cpu_identity_until_last_owner_released(imported_target):
    _, guid, _, _ = imported_target
    registry = AssetRegistry.instance()
    original_budget = registry.cpu_budget_bytes
    target = RenderTexture.load_by_guid(guid)
    try:
        registry.cpu_budget_bytes = 1
        registry.trim_cpu_budget()
        assert registry.is_loaded(guid)
        del target
        gc.collect()
        registry.trim_cpu_budget()
        assert not registry.is_loaded(guid)
    finally:
        registry.cpu_budget_bytes = original_budget


def test_removed_asset_is_not_resurrected_by_native_or_reference_cache(imported_target):
    source, guid, document, database = imported_target
    metadata = Path(str(source) + '.meta').read_bytes()
    ref = RenderTextureRef(guid)
    lease = ref.resolve()
    assert lease.guid == guid
    assert AssetManager.delete_asset(str(source), database=database).succeeded
    assert ref.resolve() is None and ref.guid == guid
    with pytest.raises(ValueError, match='imported asset GUID'):
        RenderTexture.load_by_guid(guid)
    # Restore the same source/meta identity as the Project delete command's
    # Undo, while an existing graphics consumer still owns its allocation.
    document['size'] = {'width': 79, 'height': 47}
    source.write_text(json.dumps(document), encoding='utf-8')
    Path(str(source) + '.meta').write_bytes(metadata)
    assert database.import_asset(str(source)).guid == guid
    restored = ref.resolve()
    assert restored._native is lease._native
    assert (lease.width, lease.height) == (79, 47)


def test_inspector_document_is_cpu_only_and_rejects_invalid_edit(imported_target):
    from Infernux.engine.ui.render_texture_inspector import RenderTextureDocument
    from Infernux.core.asset_types import asset_category_from_extension
    from Infernux.engine.ui import asset_details_renderer

    source, guid, document, _ = imported_target
    registry = AssetRegistry.instance()
    assert not registry.is_loaded(guid)
    resource = RenderTextureDocument(str(source))
    assert resource.serialize_document() == document
    assert not registry.is_loaded(guid)  # Selecting an asset never creates its GPU target.
    with pytest.raises(ValueError):
        resource.deserialize_document(document | {'size': {'width': 0, 'height': 1}})
    assert resource.serialize_document() == document
    assert asset_category_from_extension('.RenderTexture') == 'render_texture'
    asset_details_renderer._ensure_categories()
    category = asset_details_renderer._categories['render_texture']
    assert category.access_mode is asset_details_renderer.AssetAccessMode.READ_WRITE_RESOURCE
    assert category.load_fn(str(source))[0].serialize_document() == document


def test_asset_document_save_undo_redo_reconfigures_same_graphics_owner(imported_target):
    from Infernux.engine.ui.render_texture_inspector import RenderTextureDocument
    from Infernux.engine.interaction import DocumentKind, DocumentRegistry, ensure_editable_resource_document
    from Infernux.engine.undo import UndoManager

    source, guid, original, _ = imported_target
    target = RenderTexture.load_by_guid(guid)
    resource = RenderTextureDocument(str(source))
    previous_registry, previous_manager = DocumentRegistry._instance, UndoManager._instance
    registry, manager = DocumentRegistry(), UndoManager()
    try:
        controller = ensure_editable_resource_document(
            category='render_texture', document_kind=DocumentKind.RENDER_TEXTURE,
            file_path=str(source), resource=resource, guid=guid, view_id='inspector')
        document = registry.require(controller.document_id)
        modified = original | {'size': {'width': 91, 'height': 55}, 'format': 'rgba16_sfloat'}
        assert controller.apply_document(modified, view_id='inspector',
                                         edit_key='size', description='Edit RenderTexture')
        assert document.is_dirty
        assert controller.flush_autosave(force=True)
        assert not document.is_dirty
        assert json.loads(source.read_text(encoding='utf-8')) == modified
        assert (target.width, target.height, target.format) == (91, 55, PixelFormat.RGBA16_SFLOAT)
        assert RenderTexture.load_by_guid(guid)._native is target._native
        assert RenderTextureDocument(str(source)).serialize_document() == modified
        manager.undo()
        assert controller.flush_autosave(force=True)
        assert resource.serialize_document() == original
        assert (target.width, target.height) == (53, 29)
        manager.redo()
        assert controller.flush_autosave(force=True)
        assert (target.width, target.height) == (91, 55)
        assert RenderTexture.load_by_guid(guid)._native is target._native
    finally:
        AssetManager.cancel_scheduled_save(str(source))
        DocumentRegistry._instance, UndoManager._instance = previous_registry, previous_manager


def test_bilingual_asset_example_uses_game_object_component_access(imported_target, scene):
    import ast
    import re

    guide = Path(__file__).parents[2] / 'docs/learn/rendergraph-advanced.md'
    blocks = re.findall(r'```python\n(.*?)\n```', guide.read_text(encoding='utf-8'), re.DOTALL)
    samples = [block for block in blocks if 'class Monitor(inx.InxComponent):' in block]
    assert len(samples) == 2 and samples[0] == samples[1]
    # Execute the documented lifecycle body without registering a duplicate
    # script class. All resource/component calls below use the real engine.
    tree = ast.parse(samples[0])
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef))
    scope = {'inx': inx}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(guide), 'exec'), scope)
    owner = scene.create_game_object('DocumentedMonitor')
    camera = owner.add_component('Camera')
    target = RenderTexture.load_by_guid(imported_target[1])
    try:
        scope['start'](SimpleNamespace(game_object=owner, output=target))
        assert camera.target_texture._native is target._native
    finally:
        camera.target_texture = None
