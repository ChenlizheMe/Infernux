from __future__ import annotations

from Infernux.core.asset_types import read_mesh_import_settings
from test_model_async_import import model, poll


def test_model_reimport_reports_disk_and_live_publication_timings(model):
    database, source, _guid = model
    settings = read_mesh_import_settings(str(source))

    database.begin_model_reimport(str(source), settings.to_dict())
    result = poll(database)

    assert result and result.database_committed, result.error
    timings = (
        database.last_model_reimport_worker_ms,
        database.last_model_reimport_prepare_ms,
        database.last_model_reimport_persistence_ms,
        database.last_model_reimport_live_publication_ms,
    )
    assert all(value >= 0.0 for value in timings)
    assert sum(timings) > 0.0
