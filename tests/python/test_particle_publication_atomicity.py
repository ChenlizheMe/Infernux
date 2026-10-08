"""One published revision must own native, Python and serialized particle state."""
import json

import pytest

from infernux.components import ParticleSystem
from infernux.core.assets import AssetManager
from infernux.runtime_services import install_runtime_service, remove_runtime_service
from tests.gpu.particle_publication_case import ParticlePublicationCase


@pytest.mark.parametrize('deferred,failure', [
    (False, 'none'), (True, 'none'), (False, 'dependency'), (True, 'dependency'),
    (False, 'native'), (True, 'native'), (True, 'cancel'),
])
@pytest.mark.parametrize('empty_parameters', [False, True])
def test_particle_publication_keeps_accepted_instance_state(
        engine, scene, tmp_path, monkeypatch, deferred, failure, empty_parameters):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    install_runtime_service('gpu-particles', engine)
    case = None
    try:
        case = ParticlePublicationCase(engine, scene, tmp_path)
        component = case.component
        before = case.snapshot()
        assert json.loads(before['parameter_json']) == {'keep':5., 'removed':7.}
        assert before['color'] == [1.,0.,0.,1.]
        assert all(value > 0 for value in before['native_revisions'].values())
        revision = case.replacement(empty_parameters=empty_parameters)
        submit = component._submit_gpu_publication
        if failure == 'dependency':
            case.delete_dependency()
        elif failure == 'native':
            def reject_capacity(publication):
                # Fault injection at the boundary; the actual native validator
                # rejects this descriptor. Its return/result is not mocked.
                publication.request['programs'][0]['capacity'] = 0
                return submit(publication)
            monkeypatch.setattr(component, '_submit_gpu_publication', reject_capacity)
        if deferred:
            ParticleSystem._begin_native_publication_batch()
        accepted = component._load_saved_artifact(force=True)
        if deferred or failure != 'none':
            assert case.snapshot() == before
        if deferred:
            if failure in ('dependency', 'native'):
                with pytest.raises(RuntimeError):
                    ParticleSystem._end_native_publication_batch(commit=True)
            else:
                ParticleSystem._end_native_publication_batch(commit=failure != 'cancel')
        if failure != 'none':
            assert case.snapshot() == before
            saved_fields = component._serialize_fields_document()
            assert saved_fields['_parameter_overrides_json'] == before['parameter_json']
            assert saved_fields['_emitter_overrides_json'] == before['emitter_json']
            if failure == 'dependency':
                assert not accepted and 'cannot load' in component.last_compile_error
                case.restore_dependency()
            elif failure == 'native':
                assert component.last_compile_error
                monkeypatch.setattr(component, '_submit_gpu_publication', submit)
            assert component._load_saved_artifact(force=True), component.last_compile_error
        after = case.snapshot()
        assert after['artifact_revision'] >= revision
        assert len(set(after['native_revisions'].values())) == 1
        assert next(iter(after['native_revisions'].values())) == after['artifact_revision']
        assert after['parameters'] == ([] if empty_parameters else ['keep'])
        assert after['emitters'] == ['emitter-keep', 'emitter-new']
        assert json.loads(after['parameter_json']) == ({} if empty_parameters else {'keep':5.})
        assert after['emitter_json'] == '{}'
        assert after['color'] == [0.,1.,0.,1.]
        assert not component.last_compile_error
        (tmp_path / 'particle-publication.json').write_text(json.dumps(dict(
            before=before, after=after, deferred=deferred, failure=failure)), encoding='utf-8')
    finally:
        if ParticleSystem._native_publication_batch_depth:
            ParticleSystem._end_native_publication_batch(commit=False)
        if case is not None:
            case.close()
        assert remove_runtime_service('gpu-particles', engine)


def test_multi_component_publication_is_atomic_and_coalesces_candidates(engine, scene, tmp_path, monkeypatch):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    install_runtime_service('gpu-particles', engine)
    cases = []
    try:
        cases.extend(ParticlePublicationCase(engine, scene, tmp_path / str(i)) for i in range(2))
        before = [case.snapshot() for case in cases]
        for case in cases:
            case.replacement()
        first, second = [case.component for case in cases]
        submit = second._submit_gpu_publication
        def reject(publication):
            publication.request['programs'][0]['capacity'] = 0
            return submit(publication)
        monkeypatch.setattr(second, '_submit_gpu_publication', reject)
        ParticleSystem._begin_native_publication_batch()
        assert first._load_saved_artifact(force=True)
        assert first._load_saved_artifact(force=True)
        assert second._load_saved_artifact(force=True)
        assert len(ParticleSystem._native_publication_batch) == 2
        assert [case.snapshot() for case in cases] == before
        with pytest.raises(RuntimeError):
            ParticleSystem._end_native_publication_batch(commit=True)
        assert [case.snapshot() for case in cases] == before
        monkeypatch.setattr(second, '_submit_gpu_publication', submit)

        cases[1].delete_dependency()
        ParticleSystem._begin_native_publication_batch()
        assert first._load_saved_artifact(force=True)
        assert not second._load_saved_artifact(force=True)
        with pytest.raises(RuntimeError, match='preparation failed'):
            ParticleSystem._end_native_publication_batch(commit=True)
        assert [case.snapshot() for case in cases] == before
        cases[1].restore_dependency()

        ParticleSystem._begin_native_publication_batch()
        ParticleSystem._begin_native_publication_batch()
        assert first._load_saved_artifact(force=True)
        ParticleSystem._end_native_publication_batch(commit=False)
        assert second._load_saved_artifact(force=True)
        with pytest.raises(RuntimeError, match='preparation failed'):
            ParticleSystem._end_native_publication_batch(commit=True)
        assert [case.snapshot() for case in cases] == before

        ParticleSystem._begin_native_publication_batch()
        assert first._load_saved_artifact(force=True)
        assert second._load_saved_artifact(force=True)
        ParticleSystem._end_native_publication_batch(commit=True)
        for case in cases:
            snapshot = case.snapshot()
            assert json.loads(snapshot['parameter_json']) == {'keep':5.}
            assert snapshot['emitter_json'] == '{}'
            assert all(value == snapshot['artifact_revision'] for value in snapshot['native_revisions'].values())

        # Destruction cancels an unpublished candidate instead of resurrecting
        # its renderer at the end of the containing scene transaction.
        live_ids = tuple(first._gpu_emitter_ids)
        ParticleSystem._begin_native_publication_batch()
        assert first._load_saved_artifact(force=True)
        first._remove_native_batch()
        ParticleSystem._end_native_publication_batch(commit=True)
        assert not first._has_runtime()
        assert all(engine._gpu_particle_artifact_revision(i) == 0 for i in live_ids)
    finally:
        while ParticleSystem._native_publication_batch_depth:
            ParticleSystem._end_native_publication_batch(commit=False)
        for case in cases:
            case.close()
        assert remove_runtime_service('gpu-particles', engine)


def test_failed_particle_publication_preserves_scene_save_reopen_and_history(engine, scene, tmp_path, monkeypatch):
    from infernux.engine.scene_authoring import encode_scene_document
    from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction
    from infernux.engine.undo import PythonComponentDocumentCommand
    from infernux.particle.artifact import ParticleArtifactRegistry

    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    install_runtime_service('gpu-particles', engine)
    case = None
    try:
        case = ParticlePublicationCase(engine, scene, tmp_path)
        component = case.component
        old_fields = component._serialize_fields_document()
        component.set_parameter('Removed', 9.)
        new_fields = component._serialize_fields_document()
        command = PythonComponentDocumentCommand(component, old_fields, new_fields)
        case.replacement()
        case.delete_dependency()
        assert not component._load_saved_artifact(force=True)
        command.undo()
        assert component.get_parameter('Removed') == 7.
        command.redo()
        assert component.get_parameter('Removed') == 9.
        saved_fields = component._serialize_fields_document()
        path = tmp_path / 'Publication.scene'
        path.write_text(json.dumps(encode_scene_document(scene.serialize_document())), encoding='utf-8')

        # Restore the valid A asset schema before opening the Scene. A Scene
        # stores instance authoring, not a private copy of an old graph asset.
        ParticleArtifactRegistry.save_graph_asset(case.first, str(case.source), guid=case.guid)
        transaction = SceneDocumentTransaction(scene, path=path,
            asset_database=engine.get_asset_database(), native_engine=engine)
        assert transaction.run_to_completion(), transaction.error
        case.owner = scene.find('Particle Publication Owner')
        case.component = component = case.owner.get_py_components()[0]
        restored = component._serialize_fields_document()
        assert restored['_parameter_overrides_json'] == saved_fields['_parameter_overrides_json']
        assert restored['_emitter_overrides_json'] == saved_fields['_emitter_overrides_json']
        assert component.editor_preview_begin(), component.last_compile_error
        assert component.get_parameter('Removed') == 9.
        assert component.get_parameter('Keep') == 5.
        assert component._emitter_instance_options('emitter-removed')['enabled'] is False
    finally:
        if case is not None:
            case.close()
        assert remove_runtime_service('gpu-particles', engine)


@pytest.mark.parametrize('serialized', [False, True])
def test_first_batch_publication_accepts_lifecycle_parameter_edits(engine, scene, tmp_path, monkeypatch, serialized):
    from infernux.core.asset_ref import ParticleGraphRef
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    install_runtime_service('gpu-particles', engine)
    case = fresh = None
    try:
        case = ParticlePublicationCase(engine, scene, tmp_path)
        fresh = scene.create_game_object('Lifecycle initialization').add_py_component(ParticleSystem())
        fresh.graph = ParticleGraphRef(guid=case.guid)
        fresh.awake()
        ParticleSystem._begin_native_publication_batch()
        assert fresh._load_saved_artifact(force=True)
        assert fresh._particle_metadata is None and not fresh._has_runtime()
        # A script initializing another component during the same scene
        # finalize must still be able to author its declared instance fields.
        if serialized:
            fresh._parameter_overrides_json = '{"keep":8.0}'
            fresh._emitter_overrides_json = '{"emitter-removed":{"enabled":false,"play_on_start":true}}'
        else:
            fresh.set_parameter('Keep', 8.)
            assert fresh.set_emitter_options('emitter-removed', enabled=False)
            assert fresh.exposed_parameter_schema()[0]['value'] == 8.
            assert fresh.emitter_instance_schema()[1]['enabled'] is False
        assert fresh._particle_metadata is None and not fresh._has_runtime()
        ParticleSystem._end_native_publication_batch(commit=True)
        assert fresh.get_parameter('Keep') == 8.
        assert fresh._emitter_instance_options('emitter-removed')['enabled'] is False
        assert fresh._has_runtime()
        assert all(engine._gpu_particle_artifact_revision(i) == fresh._artifact_revision for i in fresh._gpu_emitter_ids)
    finally:
        while ParticleSystem._native_publication_batch_depth:
            ParticleSystem._end_native_publication_batch(commit=False)
        if fresh is not None:
            fresh._remove_native_batch()
        if case is not None:
            case.close()
        assert remove_runtime_service('gpu-particles', engine)
