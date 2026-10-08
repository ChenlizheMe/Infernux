"""Publication must consume Player acceptance from the caller's exact commit."""
from pathlib import Path

import yaml


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
