"""Verify that GitHub and PyPI publish the exact final local release bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_pypi(root: Path, version: str, *, allow_missing: bool = False) -> None:
    wheels = sorted(root.glob('*.whl'))
    if len(wheels) != 2:
        raise ValueError('Expected exactly two release wheels')
    try:
        with urllib.request.urlopen(f'https://pypi.org/pypi/infernux/{version}/json', timeout=60) as response:
            files = json.load(response)['urls']
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and allow_missing:
            return
        raise
    remote = {item['filename']: item for item in files}
    for wheel in wheels:
        item = remote.get(wheel.name)
        if item is None:
            if allow_missing:
                continue
            raise ValueError(f'PyPI wheel missing: {wheel.name}')
        if item['digests']['sha256'] != digest(wheel) or item['size'] != wheel.stat().st_size:
            raise ValueError(f'PyPI wheel differs: {wheel.name}; increment the build number')
        print(f'PyPI matches: {wheel.name}')


def verify_github(root: Path, document: dict) -> None:
    local = {p.name: p for p in root.iterdir() if p.is_file()}
    remote = {item['name']: item for item in document['assets']}
    if not local or set(local) != set(remote):
        raise ValueError('GitHub asset inventory does not match the final release')
    for name, path in local.items():
        item = remote[name]
        if item.get('digest') != 'sha256:' + digest(path) or item['size'] != path.stat().st_size:
            raise ValueError(f'GitHub asset differs or has no verified digest: {name}')
        print(f'GitHub matches: {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--pypi-version')
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--github-json', type=Path)
    args = parser.parse_args()
    if not args.pypi_version and not args.github_json:
        parser.error('Select at least one publication channel to verify')
    if args.pypi_version:
        verify_pypi(args.release_dir, args.pypi_version, allow_missing=args.allow_missing)
    if args.github_json:
        verify_github(args.release_dir, json.loads(args.github_json.read_text(encoding='utf-8')))
