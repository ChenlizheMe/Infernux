"""Version and publish engine-coupled plugin repositories from verified artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from build_platform_plugins import ROOT, PLATFORM_PLAYERS, assemble, read


def run(*args, cwd=None, check=True):
    result = subprocess.run([str(arg) for arg in args], cwd=cwd, encoding='utf-8',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr.strip()}')
    return result


def validate_source(checkout, revision, ref, expected):
    """Publication may change package identity, but never the tested plugin code."""
    metadata_path = 'package/inx_package.json'
    changed = run('git', 'diff', '--name-only', revision, ref, cwd=checkout).stdout.splitlines()
    if any(name != metadata_path for name in changed):
        raise ValueError(f'Plugin code changed after the engine build: {expected["reference"]}')
    before = json.loads(run('git', 'show', f'{revision}:{metadata_path}', cwd=checkout).stdout)
    after = json.loads(run('git', 'show', f'{ref}:{metadata_path}', cwd=checkout).stdout)
    for document in (before, after):
        document.pop('version', None)
        document.pop('engine', None)
    if before != after or after['reference'] != expected['reference']:
        raise ValueError(f'Plugin metadata changed after the engine build: {expected["reference"]}')


def prepare_repository(checkout, manifest):
    """Prepare a metadata-only commit. Remote mutation is a separate publication step."""
    revision = manifest['plugin_source_revision']
    tag = 'v' + manifest['version']
    tag_ref = f'refs/tags/{tag}'
    expected = {'reference': manifest['reference'], 'version': manifest['version'], 'engine': manifest['engine']}
    run('git', 'fetch', '--no-tags', 'origin', revision, cwd=checkout)
    if run('git', 'ls-remote', '--refs', 'origin', tag_ref, cwd=checkout).stdout.strip():
        run('git', 'fetch', '--no-tags', 'origin', f'{tag_ref}:{tag_ref}', cwd=checkout)
        validate_source(checkout, revision, tag, expected)
        metadata = json.loads(run('git', 'show', f'{tag}:package/inx_package.json', cwd=checkout).stdout)
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError(f'Existing immutable tag has another plugin identity: {tag}')
        return tag
    run('git', 'switch', '--detach', revision, cwd=checkout)
    metadata_path = checkout / 'package/inx_package.json'
    metadata = read(metadata_path)
    if any(metadata[key] != value for key, value in expected.items()):
        metadata.update(expected)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        run('git', 'add', '--', 'package/inx_package.json', cwd=checkout)
        run('git', '-c', 'user.name=Infernux Release', '-c', 'user.email=release@infernux-engine.com',
            'commit', '-m', f'Release {manifest["reference"]} {manifest["version"]}', cwd=checkout)
    run('git', 'tag', tag, cwd=checkout)
    validate_source(checkout, revision, tag, expected)
    return tag


def publish_repository(repository, manifest, artifact, manifest_path, *, clone_parent):
    """Publish the tested source tag and its immutable release assets."""
    checkout = clone_parent / manifest['reference'].replace('/', '.')
    run('gh', 'repo', 'clone', repository, checkout, '--', '--quiet')
    tag = prepare_repository(checkout, manifest)
    run('git', 'push', 'origin', f'refs/tags/{tag}', cwd=checkout)
    release = run('gh', 'release', 'view', tag, '--repo', repository,
                  '--json', 'isDraft,assets', check=False)
    if release.returncode == 0:
        info = json.loads(release.stdout)
        if not info['isDraft']:
            stored = run('gh', 'release', 'download', tag, '--repo', repository,
                         '--pattern', manifest_path.name, '--output', '-').stdout
            if json.loads(stored) != manifest:
                raise ValueError(f'Published plugin release is immutable: {repository}/{tag}')
            # The release manifest commits to artifact name, size and digest.
            assets = {asset['name']: asset for asset in info['assets']}
            if assets.get(artifact.name, {}).get('size') != artifact.stat().st_size:
                raise ValueError(f'Published plugin payload is incomplete: {repository}/{tag}')
            return
    else:
        run('gh', 'release', 'create', tag, '--repo', repository, '--verify-tag',
            '--title', f'{manifest["reference"]} {manifest["version"]}',
            '--notes', f'Player for Infernux {manifest["engine_release"]}.', '--draft')
    run('gh', 'release', 'upload', tag, artifact, manifest_path, '--repo', repository, '--clobber')
    run('gh', 'release', 'edit', tag, '--repo', repository, '--draft=false', '--latest=false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--check', action='store_true', help='Validate artifacts and source pins without remote writes')
    args = parser.parse_args()
    assemble(args.directory)
    repositories = read(ROOT / 'packaging/official_plugin_repositories.json')
    plans = []
    for descriptor in PLATFORM_PLAYERS.values():
        reference = descriptor['reference']
        stem = reference.replace('/', '.')
        path = args.directory / (stem + '.release.json')
        manifest = read(path)
        gitlink = run('git', 'ls-tree', manifest['source_revision'], '--',
                      descriptor['repository_path'], cwd=ROOT).stdout.split(maxsplit=3)
        if gitlink[:3] != ['160000', 'commit', manifest['plugin_source_revision']]:
            raise ValueError(f'Plugin revision differs from the tested engine gitlink: {reference}')
        plans.append((repositories[reference], manifest, args.directory / (stem + '.inxpkg'), path))
    if args.check:
        print(f'Validated {len(plans)} engine-coupled repository publications')
        return
    if not os.environ.get('GH_TOKEN'):
        raise ValueError('GH_TOKEN must have contents:write on every engine-coupled plugin repository')
    with tempfile.TemporaryDirectory(prefix='infernux-plugin-publication-') as folder:
        for repository, manifest, artifact, path in plans:
            publish_repository(repository, manifest, artifact.resolve(), path.resolve(), clone_parent=Path(folder))


if __name__ == '__main__':
    main()
