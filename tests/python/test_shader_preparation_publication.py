import json

import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel
from tests.gpu.shader_preparation_case import ShaderPreparationCase
from test_material_preview_retirement import pump_until
from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction


@pytest.mark.parametrize('mode', ShaderPreparationCase.modes)
def test_old_shader_preparation_cannot_replace_current_pixels(engine, scene, tmp_path, monkeypatch, mode):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = ShaderPreparationCase(engine, tmp_path / mode, mode)
    observations = []
    previous = engine.is_play_mode_rendering()
    engine.set_play_mode_rendering(True)
    try:
        def tick(_):
            result = case.tick()
            if result['done']:
                observations.append(result)
            return result['done']
        pump_until(engine, tick)
        assert len(observations) == 1
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0
    finally:
        engine.set_play_mode_rendering(previous)
        (tmp_path/'preparation.json').write_text(json.dumps(case.proof, indent=2), encoding='utf-8')
        case.close()


def test_scene_transaction_prepares_current_revision_after_old_ticket_is_superseded(engine, scene, tmp_path, monkeypatch):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = ShaderPreparationCase(engine, tmp_path / 'transaction', 'new_ticket')
    owner = scene.create_game_object('PreparedMaterial')
    owner.add_component('MeshRenderer').material = case.material
    document = scene.serialize_document()
    document['objects'][0]['name'] = 'PublishedMaterial'
    tickets = []

    class ObservedEngine:
        def begin_prepare_linked_shader_programs(self, guids):
            assert case.material_guid in guids
            ticket = case.old if not tickets else engine.begin_prepare_linked_shader_programs(guids)
            tickets.append(ticket)
            return ticket

        def try_commit_linked_shader_programs(self, ticket):
            if ticket is case.old:
                case._change_source()
            return engine.try_commit_linked_shader_programs(ticket)

    transaction = SceneDocumentTransaction(scene, document=document,
        asset_database=engine.get_asset_database(), native_engine=ObservedEngine())
    try:
        transaction.start()
        pump_until(engine, lambda _: transaction.poll())
        assert transaction.status == 'completed', transaction.error
        assert len(tickets) == 2 and tickets[0].superseded and tickets[1].committed
        assert scene.find('PreparedMaterial') is None
        published = scene.find('PublishedMaterial')
        assert published.get_component('MeshRenderer').material.guid == case.material_guid
    finally:
        transaction.cancel()
        case.close()


def test_same_source_concurrent_preparations_can_both_publish(engine, scene, tmp_path):
    case = ShaderPreparationCase(engine, tmp_path / 'same-source', 'unchanged')
    second = engine.begin_prepare_linked_shader_programs([case.material_guid])
    try:
        pump_until(engine, lambda _: case.old.complete and second.complete)
        assert engine.try_commit_linked_shader_programs(second)
        assert engine.try_commit_linked_shader_programs(case.old)
        assert not second.superseded and not case.old.superseded
        assert second.committed and case.old.committed
    finally:
        second.cancel()
        case.close()
