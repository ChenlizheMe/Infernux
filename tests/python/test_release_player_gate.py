"""Publication must consume Player acceptance from the caller's exact commit."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import yaml

from tests.tool_discovery import find_executable


ROOT = Path(__file__).resolve().parents[2]


def _workflow(name):
    return yaml.safe_load((ROOT / '.github/workflows' / name).read_text(encoding='utf-8'))


def test_publication_requires_successful_same_revision_player_workflow():
    desktop = _workflow('ci.yml')
    publish = desktop['jobs']['publish-release']
    acceptance = desktop['jobs']['release-player-acceptance']
    assert 'release-player-acceptance' in publish['needs']
    assert acceptance['if'] == publish['if']
    assert acceptance['uses'] == './.github/workflows/platform-player.yml'
    assert not acceptance.get('continue-on-error', False)
    assert 'always()' not in publish['if']

    players = _workflow('platform-player.yml')
    # PyYAML's YAML 1.1 loader recognizes the workflow's `on` as True.
    assert 'workflow_call' in players[True]
    for platform in ('windows-player', 'linux-player', 'web-player', 'web-browser', 'android-player'):
        job = players['jobs'][platform]
        assert not job.get('continue-on-error', False)
        checkout = next(step for step in job['steps'] if step.get('uses', '').startswith('actions/checkout@'))
        assert not checkout.get('with', {}).get('ref')
    assert players['jobs']['web-browser']['needs'] == 'web-player'


def test_called_and_standalone_player_runs_cannot_cancel_each_other():
    desktop = _workflow('ci.yml')
    players = _workflow('platform-player.yml')
    player_group = players['concurrency']['group']
    assert '${{ github.workflow }}' in player_group
    assert player_group.startswith('platform-player-')
    assert desktop['concurrency']['group'].startswith('preset-ci-')


def test_physics_acceptance_evidence_is_uploaded_on_failure():
    desktop = _workflow('ci.yml')
    for platform in ('windows', 'linux'):
        uploads = [step for step in desktop['jobs'][platform + '-desktop']['steps']
                   if step.get('uses', '').startswith('actions/upload-artifact@')]
        evidence = next(step for step in uploads if f'physics-profile-{platform}.log' in step['with']['path'])
        assert evidence['if'] == 'always()'


def test_artifact_publisher_has_no_independent_dispatch_bypass():
    publisher = _workflow('publish-desktop-release.yml')
    assert set(publisher[True]) == {'workflow_call'}
    caller = _workflow('ci.yml')['jobs']['publish-release']
    assert caller['with']['run_id'] == '${{ github.run_id }}'
    assert 'release-player-acceptance' in caller['needs']


@pytest.mark.parametrize('scenario', [
    'gated_caller', 'other_successful_run', 'other_revision', 'repair_branch',
    'fork', 'untrusted_workflow', 'pull_request', 'invalid_run_id', 'api_failure',
])
def test_publisher_source_guard_executes_before_exposing_artifacts(tmp_path, scenario):
    powershell = find_executable('pwsh')
    if powershell is None:
        pytest.skip('The Windows publication step requires PowerShell 7')
    source = dict(head_repository=dict(full_name='ChenlizheMe/Infernux'),
                  head_branch='master', path='.github/workflows/ci.yml',
                  event='workflow_dispatch', conclusion=None,
                  head_sha='a' * 40, run_attempt=1)
    environment = dict(os.environ, RUN_ID='123', GITHUB_RUN_ID='123',
                       GITHUB_SHA='a' * 40, GITHUB_REPOSITORY='ChenlizheMe/Infernux',
                       GITHUB_OUTPUT=str(tmp_path / 'outputs'),
                       MOCK_GH_SOURCE=str(tmp_path / 'source.json'), MOCK_GH_EXIT='0')
    if scenario == 'other_successful_run':
        environment['RUN_ID'] = '122'
        source['conclusion'] = 'success'
    elif scenario == 'other_revision':
        source['head_sha'] = 'b' * 40
    elif scenario == 'repair_branch':
        source['head_branch'] = '041-post/repair'
    elif scenario == 'fork':
        source['head_repository']['full_name'] = 'someone/Infernux'
    elif scenario == 'untrusted_workflow':
        source['path'] = '.github/workflows/test-windows-signing.yml'
    elif scenario == 'pull_request':
        source['event'] = 'pull_request'
    elif scenario == 'invalid_run_id':
        environment['RUN_ID'] = '123/other'
    elif scenario == 'api_failure':
        environment['MOCK_GH_EXIT'] = '17'
    Path(environment['MOCK_GH_SOURCE']).write_text(json.dumps(source), encoding='utf-8')
    step = _workflow('publish-desktop-release.yml')['jobs']['verify']['steps'][0]
    assert step['shell'] == 'pwsh'
    python = "'" + sys.executable.replace("'", "''") + "'"
    # Mock only GitHub transport; execute the workflow's actual PowerShell
    # guard, including the real native command exit status and output writes.
    prefix = f'''$ErrorActionPreference = 'Stop'
function gh {{
    & {python} -c 'import os,sys;from pathlib import Path;print(Path(os.environ["MOCK_GH_SOURCE"]).read_text());sys.exit(int(os.environ["MOCK_GH_EXIT"]))'
    $global:LASTEXITCODE = $LASTEXITCODE
}}
'''
    script = tmp_path / 'verify.ps1'
    script.write_text(prefix + step['run'] + '\nexit $LASTEXITCODE\n', encoding='utf-8')
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(script)],
                            env=environment, cwd=tmp_path, capture_output=True,
                            text=True, encoding='utf-8', timeout=30)
    output = Path(environment['GITHUB_OUTPUT'])
    if scenario == 'gated_caller':
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.read_text(encoding='utf-8').splitlines() == [
            'sha=' + 'a' * 40, 'branch=master', 'attempt=1',
        ]
    else:
        assert result.returncode != 0, scenario
        assert not output.exists(), 'Rejected source must not expose release artifact identity'
        if scenario == 'api_failure':
            assert result.returncode == 17
