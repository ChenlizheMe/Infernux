import json
import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel
from tests.gpu.mesh_preview_dependency_case import MeshPreviewDependencyCase
from test_material_preview_retirement import pump_until


@pytest.mark.parametrize('consumer', ['python', 'native'])
@pytest.mark.parametrize('mode', MeshPreviewDependencyCase.modes)
def test_model_and_submesh_previews_follow_published_dependencies(engine, scene, tmp_path, monkeypatch, consumer, mode):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = MeshPreviewDependencyCase(engine, tmp_path / mode, mode, consumer)
    try:
        def tick(_):
            # The native fixture does not install Engine's Python post-draw
            # owner. Drain the same deferred texture publication at this point.
            AssetManager.flush_pending_gpu_texture_reloads(paths=[str(case.texture)])
            return case.tick()['done']
        pump_until(engine, tick)
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0, console._get_visible_log_snapshot(100)
    finally:
        (tmp_path / 'preview-dependencies.json').write_text(json.dumps(case.observations, indent=2), encoding='utf-8')
        case.close()


@pytest.mark.parametrize('stage', ['queued', 'rendering', 'uploading'])
def test_mesh_preview_finishes_latest_request_without_repeated_queries(engine, scene, tmp_path, monkeypatch, stage):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = MeshPreviewDependencyCase(engine, tmp_path / stage, 'edit', 'native')
    key, path = case.keys['model'], case.paths['model']
    phase = 'baseline'
    baseline = None
    settled = 0
    observations = []

    def publish(color):
        case.material.set_vector4('baseColor', *color, 1.)
        AssetManager.note_asset_edit(str(case.material_path), material_json=case.material.serialize())
        engine.query_or_schedule_mesh_preview(key, path, 0)

    try:
        engine.query_or_schedule_mesh_preview(key, path, 0)

        def tick(_):
            nonlocal phase, baseline, settled
            row = next(dict(r) for r in engine.preview_task_snapshots if r['resource_key'] == key)
            ready = row['texture_id'] and row['generation'] == row['ready_generation'] and not row['in_flight']
            if phase == 'baseline' and ready:
                baseline = row
                observations.append(row)
                publish((0., 0., 1.))
                phase = 'supersede'
            if phase == 'supersede':
                # The query above queues work but does not render it inline.
                row = next(dict(r) for r in engine.preview_task_snapshots if r['resource_key'] == key)
                reached = (stage == 'queued' or
                           stage == 'rendering' and row['has_render_ticket'] or
                           stage == 'uploading' and row['pending_upload_version'] != 0)
                if reached:
                    observations.append(row)
                    publish((1., 0., 0.))
                    phase = 'latest'
            elif phase == 'latest' and ready:
                assert row['generation'] == baseline['generation'] + 2, row
                assert row['pixel_generation'] == row['generation'], row
                assert row['pixel_hash'] == baseline['pixel_hash'], (baseline, row)
                settled += 1
                if settled >= 30:
                    observations.append(row)
                    return True
            return False

        pump_until(engine, tick)
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0, console._get_visible_log_snapshot(100)
    finally:
        (tmp_path / 'preview-supersession.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
        case.close()


@pytest.mark.parametrize('replacement', ['recreate', 'reimport'])
def test_material_content_deduplication_does_not_outlive_its_source(engine, scene, tmp_path, monkeypatch, replacement):
    from infernux.lib import InxMaterial

    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    directory = tmp_path / replacement
    case = MeshPreviewDependencyCase(engine, directory, 'edit', 'native')
    try:
        pump_until(engine, lambda _: case.tick()['done'])
        if replacement == 'recreate':
            case.close()
            case = None
            case = MeshPreviewDependencyCase(engine, directory, 'edit', 'native')
        else:
            authored = InxMaterial.create_default_unlit()
            authored.set_vector4('baseColor', 1., 0., 0., 1.)
            assert authored.save_to(str(case.material_path))
            assert AssetManager.reimport_asset(str(case.material_path))
            case.material = case.registry.load_material(str(case.material_path))
            case.step, case.frames, case.phase = 'baseline', 0, 0
        pump_until(engine, lambda _: case.tick()['done'])
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0, console._get_visible_log_snapshot(100)
    finally:
        if case is not None:
            (tmp_path / 'preview-source-replacement.json').write_text(json.dumps(case.observations, indent=2), encoding='utf-8')
            case.close()
