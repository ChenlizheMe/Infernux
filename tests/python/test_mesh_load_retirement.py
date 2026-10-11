"""A completed preview worker cannot revive a retired resource revision."""
import json
import time

import pytest

from infernux.core.assets import AssetManager
from tests.acceptance.asset_load_retirement import MeshLoadRetirementCase


@pytest.mark.parametrize('mutation', ['unchanged','invalidate','delete','restore','reimport','native-reload'])
@pytest.mark.parametrize('allow_stale', [False,True])
def test_mesh_worker_respects_current_content_generation(engine, tmp_path, monkeypatch, mutation, allow_stale):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = MeshLoadRetirementCase(engine, tmp_path/'Source')
    try:
        deadline = time.monotonic() + 15
        while not case.ready():
            assert time.monotonic() < deadline
            time.sleep(.001)
        result = case.exercise(mutation, allow_stale)
        (tmp_path/'mesh-retirement.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        case.close()
