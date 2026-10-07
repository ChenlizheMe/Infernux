"""A premature relocation notification cannot partially publish asset mappings."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('kind', ['text', 'texture'])
@pytest.mark.parametrize('mode', ['single', 'batch', 'mixed'])
@pytest.mark.parametrize('premature', [False, True])
def test_asset_move_requires_existing_destination(tmp_path, kind, mode, premature):
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), str(tmp_path), kind, mode, str(int(premature))],
        env=dict(os.environ), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ASSET_MOVE_PRECONDITION_OK' in result.stdout


def exercise(project, kind, mode, premature):
    from PIL import Image
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode

    assets = project / 'Assets'
    assets.mkdir()
    (project / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        suffix = '.png' if kind == 'texture' else '.txt'
        sources = [assets / ('First' + suffix)]
        if mode == 'mixed':
            sources.append(assets / ('Second' + suffix))
        targets = [assets / ('Moved ' + source.name) for source in sources]
        guids = []
        for source in sources:
            if kind == 'texture':
                Image.new('RGBA', (8, 4), (255, 0, 0, 255)).save(source)
            else:
                source.write_text(source.name, encoding='utf-8')
            imported = database.import_asset(str(source))
            assert imported.succeeded, imported.error
            guids.append(imported.guid)
        metadata = [Path(str(source) + '.meta').read_bytes() for source in sources]
        metadata_times = [Path(str(source) + '.meta').stat().st_mtime_ns for source in sources]
        generation = database.query_generation
        mappings = list(zip(map(str, sources), map(str, targets)))

        def notify():
            return [database.move_asset(*mappings[0])] if mode == 'single' else database.move_assets_batch(mappings)

        if premature:
            if mode == 'mixed':
                sources[0].replace(targets[0])
            results = notify()
            assert len(results) == 1 and not results[0].succeeded
            assert not results[0].database_committed and not results[0].changed
            assert results[0].error
            assert database.query_generation == generation
            for source, target, guid, sidecar, modified in zip(sources, targets, guids, metadata, metadata_times):
                assert database.get_guid_from_path(str(source)) == guid
                assert not database.get_guid_from_path(str(target))
                assert Path(database.get_path_from_guid(guid)) == source
                assert Path(str(source) + '.meta').read_bytes() == sidecar
                assert Path(str(source) + '.meta').stat().st_mtime_ns == modified
                assert not Path(str(target) + '.meta').exists()

        for source, target in zip(sources, targets):
            if source.exists():
                source.replace(target)
        results = notify()
        assert len(results) == len(sources) and all(result.succeeded for result in results)
        for source, target, guid in zip(sources, targets, guids):
            assert database.get_guid_from_path(str(target)) == guid
            assert not database.get_guid_from_path(str(source))
            assert Path(database.get_path_from_guid(guid)) == target
            assert Path(str(target) + '.meta').is_file()
        database.refresh()
        assert all(database.get_guid_from_path(str(path)) == guid for path, guid in zip(targets, guids))
    finally:
        engine.exit()
    print('ASSET_MOVE_PRECONDITION_OK')


if __name__ == '__main__':
    exercise(Path(sys.argv[1]), sys.argv[2], sys.argv[3], bool(int(sys.argv[4])))
