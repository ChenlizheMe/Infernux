"""Approve recovery of a draft using its original fully tested distributions."""
from __future__ import annotations

import argparse
import json
import re
import subprocess

PUBLICATION_FILES = frozenset({
    '.github/workflows/publish-desktop-release.yml',
    '.github/workflows/recover-release.yml',
    'README.md',
    'README-zh.md',
    'scripts/release/deploy_hub_catalog.py',
    'scripts/release/publish_hub_objects.py',
    'scripts/release/r2-upload-worker.mjs',
    'scripts/release/verify_release_recovery.py',
    'scripts/release/sync_release_site.py',
    'scripts/release/README.md',
    'tests/hub/test_release_signing.py',
    'tests/hub/test_release_recovery.py',
    'tests/release/test-r2-upload-worker.mjs',
})
WORKFLOW_PREFIXES = {
    '.github/workflows/release.yml': 'build-test-publish / ',
    '.github/workflows/ci.yml': '',
}
BUILD_JOBS = frozenset((
    'Portable Hub regressions (ubuntu-22.04)',
    'Portable Hub regressions (windows-2022)',
    'windows-desktop', 'linux-desktop',
    'Validate Players before publication / windows-player',
    'Validate Players before publication / linux-player',
    'Validate Players before publication / android-player',
    'Validate Players before publication / web-player',
    'Validate Players before publication / web-browser (release, web-release-player-build)',
    'Validate Players before publication / web-browser (development, web-player-build)',
))
PUBLISH_JOB = 'Publish verified distributions / publish'


def validate(run: dict, jobs: dict, repository: str, expected_sha: str,
             publisher_sha: str, changed_files: list[str]) -> dict:
    if not all(re.fullmatch('[0-9a-f]{40}', value) for value in (expected_sha, publisher_sha)):
        raise ValueError('Recovery requires exact source and publisher revisions')
    if ((run['status'], run['conclusion'], run['event']) != ('completed', 'failure', 'workflow_dispatch')
            or run['path'] not in WORKFLOW_PREFIXES):
        raise ValueError('Recovery requires a failed official release workflow')
    if run['head_repository']['full_name'] != repository or run['head_branch'] not in ('master', 'main'):
        raise ValueError('Recovery source must be the official release branch')
    if run['head_sha'] != expected_sha:
        raise ValueError('Recovery source revision differs from the requested revision')
    if any(path not in PUBLICATION_FILES and not path.startswith('docs/') for path in changed_files):
        raise ValueError('Engine, build, or payload changes require a new complete release run')
    if jobs['total_count'] != len(jobs['jobs']):
        raise ValueError('Recovery must inspect every source job')
    prefix = WORKFLOW_PREFIXES[run['path']]
    required_jobs = {prefix + name for name in BUILD_JOBS}
    publish_job = prefix + PUBLISH_JOB
    names = [job['name'] for job in jobs['jobs']]
    if len(names) != len(set(names)) or not required_jobs <= set(names) or publish_job not in names:
        raise ValueError('Required platform acceptance jobs are missing or ambiguous')
    for job in jobs['jobs']:
        expected = 'failure' if job['name'] == publish_job else 'success'
        if job['status'] != 'completed' or job['conclusion'] != expected:
            raise ValueError('Recovery cannot accept incomplete or failed build/acceptance jobs')
    return {'sha': expected_sha, 'publisher_sha': publisher_sha,
            'branch': run['head_branch'], 'attempt': run['run_attempt']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--publisher-sha', required=True)
    args = parser.parse_args()
    if not args.run_id.isdecimal():
        raise ValueError('Invalid release run ID')

    def api(path):
        return json.loads(subprocess.check_output(['gh', 'api', path], text=True, encoding='utf-8', timeout=60))

    base = f'repos/{args.repository}/actions/runs/{args.run_id}'
    run = api(base)
    jobs = api(f"{base}/attempts/{run['run_attempt']}/jobs?per_page=100")
    changes = subprocess.check_output(['git', 'diff', '--name-only', run['head_sha'], args.publisher_sha],
                                      text=True, encoding='utf-8', timeout=60).splitlines()
    print(json.dumps(validate(run, jobs, args.repository, args.expected_sha, args.publisher_sha, changes)))


if __name__ == '__main__':
    main()
