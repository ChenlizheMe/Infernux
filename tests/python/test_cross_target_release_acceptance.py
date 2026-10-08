from pathlib import Path
import json
import zipfile

import pytest
import yaml

from infernux.engine.player_package_native import write_pack
from tests.acceptance.verify_release_player_package import verify


def sealed_fixture(tmp_path, target, *, flavor='PlayerRelease', debug=False, control='disabled'):
    manifest = tmp_path / 'BuildManifest.json'
    manifest.write_text(json.dumps(dict(debug_build=debug, runtime_contract=dict(
        product=dict(flavor=flavor), runtime_policy=dict(player_control=control)))), encoding='utf-8')
    catalog = tmp_path / 'AssetCatalog.inxcat'
    write_pack([('BuildManifest.json', manifest)], catalog)
    entry = 'InfernuxPlatformFixture_Data/AssetCatalog.inxcat'
    if target == 'web':
        artifact = tmp_path / 'player.inxpkg'
        write_pack([(entry, catalog)], artifact)
    else:
        artifact = tmp_path / 'player.apk'
        with zipfile.ZipFile(artifact, 'w') as apk:
            apk.write(catalog, 'assets/player/' + entry)
    return artifact


@pytest.mark.parametrize('target', ('web', 'android'))
def test_reads_release_contract_from_actual_nested_package(tmp_path, target):
    artifact = sealed_fixture(tmp_path, target)
    assert verify(artifact, target)['flavor'] == 'PlayerRelease'


@pytest.mark.parametrize('target', ('web', 'android'))
@pytest.mark.parametrize('mutation', (dict(flavor='PlayerDebug'), dict(debug=True), dict(control='enabled')))
def test_rejects_mislabeled_release_artifact(tmp_path, target, mutation):
    artifact = sealed_fixture(tmp_path, target, **mutation)
    with pytest.raises(RuntimeError, match='PlayerRelease package'):
        verify(artifact, target)


def test_browser_acceptance_runs_both_published_flavors():
    root = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((root / '.github/workflows/platform-player.yml').read_text())['jobs']
    entries = jobs['web-browser']['strategy']['matrix']['include']
    assert {row['configuration'] for row in entries} == {'development', 'release'}
    published = [step['with']['name'] for step in jobs['web-player']['steps']
                 if step.get('uses') == 'actions/upload-artifact@v4']
    assert all(any(name.startswith(row['artifact'] + '-') for name in published) for row in entries)
    commands = '\n'.join(step.get('run', '') for step in jobs['web-player']['steps'])
    assert '--configuration release' in commands
    assert 'verify_release_player_package.py' in commands
