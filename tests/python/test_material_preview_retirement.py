"""Rejected/deleted material previews cannot block other preview consumers."""
import io
import json
import os
import time
import uuid

import pytest
from PIL import Image

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline


def pump_until(engine, check):
    pipeline = RenderStackPipeline()
    engine.set_render_pipeline(pipeline)
    errors = []
    frame = 0
    deadline = time.monotonic() + 15
    try:
        def tick():
            nonlocal frame
            try:
                frame += 1
                assert time.monotonic() < deadline, engine.preview_task_snapshots
                engine.request_full_speed_frame()
                engine.pump_preview_tasks()
                if check(frame):
                    engine.exit()
            except BaseException as error:
                errors.append(error)
                engine.exit()
        engine.set_post_draw_callback(tick)
        engine.run()
        if errors:
            raise errors[0]
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        pipeline.dispose()


def invalid_material():
    document = InxMaterial.create_default_lit().serialize_document()
    document['builtin'] = False
    document['shaders']['fragment'] = {'guid': uuid.uuid4().hex, 'shader_id': 'DeletedShader'}
    return json.dumps(document)


def test_rejected_material_does_not_block_texture_or_repeat_without_change(engine, scene, tmp_path):
    key = f'mat|{tmp_path / "Missing.mat"}'
    texture_key = f'preview-control|{tmp_path}'
    source = invalid_material()
    console = ConsolePanel()
    output = io.BytesIO()
    Image.new('RGB', (8, 16), 'blue').save(output, format='JPEG')
    engine.query_or_schedule_material_preview(key, '', source, 0, True)
    assert engine.schedule_texture_preview_from_memory(texture_key, output.getvalue(), 1, False)
    try:
        settled_frames = 0
        def rejected_and_uploaded(frame):
            nonlocal settled_frames
            engine.query_or_schedule_material_preview(key, '', source, 0, True)
            # Material preview has a real cooldown. Wait for its terminal
            # state, then query repeatedly to verify rejection stays terminal.
            material = next(r for r in engine.preview_task_snapshots
                            if r['kind'] == 'material' and r['resource_key'].endswith('/missing.mat' if os.name == 'nt' else '/Missing.mat'))
            settled_frames = settled_frames + 1 if not material['in_flight'] else 0
            return settled_frames >= 25 and bool(engine.get_texture_preview_texture_id(texture_key))
        pump_until(engine, rejected_and_uploaded)
        errors = [r for r in console._get_visible_log_snapshot(1000) if r['level'] == 'ERROR']
        assert len(errors) == 1, errors
        assert 'Material preview rejected' in errors[0]['message']
        assert engine.get_material_preview_texture_id(key) == 0
        assert tuple(engine.get_texture_preview_size(texture_key)) == (8, 16)
        # A corrected source is a new revision, not a retry of the failed one.
        good = InxMaterial.create_default_lit().serialize()
        engine.query_or_schedule_material_preview(key, '', good, 0, True)
        pump_until(engine, lambda _: bool(engine.get_material_preview_texture_id(key)))
        assert console.get_error_count() == 1
    finally:
        engine.release_material_preview_task(key)
        engine.release_texture_preview_task(texture_key)


@pytest.mark.parametrize('recreate', [False, True])
def test_asset_delete_retires_queued_preview_before_shader_resolution(engine, scene, tmp_path, monkeypatch, recreate):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    path = tmp_path / 'Deleted.mat'
    key = f'mat|{path}'
    console = ConsolePanel()
    good = InxMaterial.create_default_lit().serialize()
    path.write_text(good, encoding='utf-8')
    assert AssetManager.import_asset(str(path))
    engine.query_or_schedule_material_preview(key, str(path), invalid_material(), 0, True)
    path.unlink()
    assert AssetManager.delete_asset(str(path))
    try:
        if recreate:
            path.write_text(good, encoding='utf-8')
            assert AssetManager.import_asset(str(path))
        pump_until(engine, lambda frame: frame >= 25 and
                   (not recreate or bool(engine.get_material_preview_texture_id(key))))
        assert console.get_error_count() == console.get_warning_count() == 0
        if not recreate:
            assert engine.get_material_preview_texture_id(key) == 0
            normalized = key.replace('\\', '/')
            if os.name == 'nt':
                normalized = normalized.lower()
            snapshot = next(r for r in engine.preview_task_snapshots if r['resource_key'] == normalized)
            assert not snapshot['in_flight'] and not snapshot['has_render_ticket']
    finally:
        if path.exists():
            path.unlink()
            assert AssetManager.delete_asset(str(path))
        engine.release_material_preview_task(key)
