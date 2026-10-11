"""Native import must let a concurrent Python metadata reader close its file."""
import os
from pathlib import Path
import threading
import subprocess
import sys

import pytest

from infernux.lib import NativeDocumentStore


@pytest.mark.skipif(os.name != 'nt', reason='Windows readers deny atomic replacement until closed')
@pytest.mark.parametrize('operation', ['import_asset', 'reimport_asset'])
def test_import_releases_gil_while_metadata_writer_waits(tmp_path, operation):
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()),
                             str(tmp_path), operation], capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(tmp_path, operation):
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode
    project = tmp_path / 'Project'
    assets = project / 'Assets'
    assets.mkdir(parents=True)
    (project / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        check_reader(engine.get_asset_database(), project, operation)
    finally:
        engine.exit()


def check_reader(database, project, operation):
    assets = project / 'Assets'
    source = assets / 'Triangle.obj'
    source.write_text('o Triangle\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n', encoding='utf-8')
    initial = database.import_asset(str(source))
    assert initial, initial.error
    meta = Path(str(source) + '.meta')
    content = meta.read_bytes()
    if operation == 'import_asset':
        assert database.delete_asset(str(source))
        source.write_text('o Triangle\nv 0 0 0\nv 3 0 0\nv 0 1 0\nf 1 2 3\n', encoding='utf-8')
        meta.write_bytes(content)
    else:
        source.write_text('o Triangle\nv 0 0 0\nv 3 0 0\nv 0 1 0\nf 1 2 3\n', encoding='utf-8')

    store = NativeDocumentStore.instance()
    opened = threading.Event()
    finish = threading.Event()
    saw_writer = threading.Event()
    errors = []

    def reader():
        try:
            with meta.open('rb') as stream:
                assert stream.read() == content
                opened.set()
                # Close precisely when the importer has submitted its write.
                # A GIL-holding native wait prevents this owner from retiring
                # the handle, exhausting even transient Windows write retries.
                while not finish.wait(.001):
                    metrics = store.get_metrics(str(meta))
                    if metrics.active_generation or metrics.pending_generation:
                        saw_writer.set()
                        break
        except BaseException as error:
            errors.append(error)
            opened.set()

    worker = threading.Thread(target=reader)
    worker.start()
    try:
        assert opened.wait(5) and not errors, errors
        result = getattr(database, operation)(str(source))
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive() and not errors, errors
    assert result, result.error
    assert saw_writer.is_set(), 'Python reader could not run during the native write'
    assert result.guid == initial.guid
    assert not (project / 'Library/AssetRefresh.transaction').exists()


if __name__ == '__main__':
    exercise(Path(sys.argv[1]), sys.argv[2])
