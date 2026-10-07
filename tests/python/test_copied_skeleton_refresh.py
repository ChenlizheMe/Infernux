"""Copied rig definitions must work during refresh and clean project import."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", [
    "warm", "warm-both", "shared-warm", "clean", "shared-clean",
    "missing", "failed-definition", "cycle", "self",
])
def test_copied_skeleton_batch_import(tmp_path, case):
    import infernux
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), case],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def verify_catalog(database, expected):
    import json
    from infernux.lib import AssetRegistry
    index = json.loads(Path(database.asset_index_path).read_text(encoding="utf-8"))
    entries = {entry['guid']: entry for entry in index['entries']}
    for relative, guid in expected.items():
        assert database.get_guid_from_path(str(Path(database.project_root) / relative)) == guid
        assert entries[guid]['import_succeeded'], (relative, entries[guid]['import_error'])
        assert database.get_meta_by_guid(guid) is not None
        mesh = AssetRegistry.instance().load_mesh_by_guid(guid)
        assert mesh and mesh.has_skinned_data and mesh.skinned_bone_count > 0
        assert mesh.skeleton_definition['guid'] == expected['Assets/Z_Definition.fbx']
        assert mesh.skeleton_definition['subresource_id'] == 'skeleton'


def exercise(project, case):
    import json
    import shutil
    from infernux.core.assets import AssetManager
    from infernux.core.asset_types import read_mesh_import_settings, write_mesh_import_settings
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode

    assets = project / 'Assets'
    assets.mkdir()
    (project / 'ProjectSettings').mkdir()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    expected = {}
    try:
        engine.init_headless(str(project))
        database = AssetManager.require_asset_database()
        fixture = Path(__file__).resolve().parents[2] / 'external/assimp/test/models/FBX/animation_with_skeleton.fbx'
        # Lexical traversal sees dependents before their definitions.
        owner = assets / 'Z_Definition.fbx'
        copied = assets / 'M_Copy.fbx'
        shutil.copyfile(fixture, owner)
        definition = AssetManager.import_asset(str(owner))
        assert definition and definition.guid
        expected[owner.relative_to(project).as_posix()] = definition.guid
        source_guid = definition.guid
        paths = [copied]
        if case in {'shared-clean', 'shared-warm'}:
            paths.append(assets / 'A_SecondCopy.fbx')
        for path in paths:
            shutil.copyfile(fixture, path)
            imported = AssetManager.import_asset(str(path))
            assert imported and imported.guid
            settings = read_mesh_import_settings(str(path))
            settings.skeleton_definition_mode = 'copy'
            settings.skeleton_definition_guid = source_guid
            settings.skeleton_definition_id = 'skeleton'
            settings.rig_root_node = ''
            result = AssetManager.reimport_asset(str(path), import_settings=settings.to_dict())
            assert result, result.error
            expected[path.relative_to(project).as_posix()] = imported.guid
        database.flush_derived_index()
        verify_catalog(database, expected)
        if case in {'warm', 'warm-both', 'shared-warm'}:
            changed = paths + ([owner] if case == 'warm-both' else [])
            for path in changed:
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            database.refresh()
            verify_catalog(database, expected)
        elif case in {'missing', 'failed-definition', 'cycle', 'self'}:
            independent = assets / 'Independent.fbx'
            shutil.copyfile(fixture, independent)
            control = AssetManager.import_asset(str(independent))
            assert control and control.guid
            copied_guid = expected[copied.relative_to(project).as_posix()]
            old_artifact = (project / f'Library/Artifacts/Mesh/{copied_guid}.inxmesh').read_bytes()
            if case == 'missing':
                owner.unlink()
                Path(str(owner) + '.meta').unlink()
            elif case == 'failed-definition':
                owner.write_bytes(b'not a valid FBX definition')
            elif case == 'cycle':
                settings = read_mesh_import_settings(str(owner))
                settings.skeleton_definition_mode = 'copy'
                settings.skeleton_definition_guid = copied_guid
                settings.skeleton_definition_id = 'skeleton'
                settings.rig_root_node = ''
                assert write_mesh_import_settings(str(owner), settings)
            else:
                settings = read_mesh_import_settings(str(copied))
                settings.skeleton_definition_guid = copied_guid
                assert write_mesh_import_settings(str(copied), settings)
            for path in (copied, independent):
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            database.refresh()
            index = json.loads(Path(database.asset_index_path).read_text(encoding='utf-8'))
            entries = {entry['guid']: entry for entry in index['entries']}
            assert entries[control.guid]['import_succeeded']
            assert not entries[copied_guid]['import_succeeded']
            reason = {'missing': 'unavailable', 'failed-definition': 'definition import failed',
                      'cycle': 'dependency cycle', 'self': 'another mesh asset GUID'}[case]
            assert reason in entries[copied_guid]['import_error'], entries[copied_guid]['import_error']
            if case == 'cycle':
                assert 'dependency cycle' in entries[definition.guid]['import_error']
            assert (project / f'Library/Artifacts/Mesh/{copied_guid}.inxmesh').read_bytes() == old_artifact
            assert database.get_guid_from_path(str(copied)) == copied_guid
    finally:
        engine.exit()
    if case in {'clean', 'shared-clean'}:
        clone = project / 'clean'
        shutil.copytree(assets, clone / 'Assets')
        shutil.copytree(project / 'ProjectSettings', clone / 'ProjectSettings')
        (clone / 'expected.json').write_text(json.dumps(expected), encoding='utf-8')
        result = subprocess.run(
            [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), str(clone), 'verify-clean'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def verify_clean(project):
    import json
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode
    assert not (project / 'Library').exists()
    PreferencesStore()._path = str(project / 'preferences.json')
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        verify_catalog(engine.get_asset_database(), json.loads((project / 'expected.json').read_text(encoding='utf-8')))
    finally:
        engine.exit()


if __name__ == '__main__':
    if sys.argv[2] == 'verify-clean':
        verify_clean(Path(sys.argv[1]))
    else:
        exercise(Path(sys.argv[1]), sys.argv[2])
