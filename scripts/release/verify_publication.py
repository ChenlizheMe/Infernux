"""Verify that GitHub and PyPI publish the exact final local release bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_pypi(root: Path, version: str, *, allow_missing: bool = False, wait_seconds: float = 0) -> None:
    wheels = sorted(root.glob('*.whl'))
    if len(wheels) != 2:
        raise ValueError('Expected exactly two release wheels')
    if wait_seconds < 0:
        raise ValueError('PyPI visibility wait must be nonnegative')
    expected = {wheel.name: (digest(wheel), wheel.stat().st_size) for wheel in wheels}
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            with urllib.request.urlopen(f'https://pypi.org/pypi/infernux/{version}/json', timeout=60) as response:
                files = json.load(response)['urls']
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            if allow_missing:
                return
            if not wait_seconds:
                raise
            files = []
        remote = {item['filename']: item for item in files}
        missing = []
        for wheel in wheels:
            item = remote.get(wheel.name)
            if item is None:
                missing.append(wheel.name)
            elif (item['digests']['sha256'], item['size']) != expected[wheel.name]:
                raise ValueError(f'PyPI wheel differs: {wheel.name}; increment the build number')
        if not missing or allow_missing:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f'PyPI wheel missing: {", ".join(missing)}')
        # Successful uploads can precede visibility of PyPI's cached JSON.
        # Wait only for missing files; changed bytes and other errors fail now.
        print(f'Waiting for PyPI metadata visibility: {", ".join(missing)}', flush=True)
        time.sleep(min(5, remaining))
    for wheel in wheels:
        if wheel.name in remote:
            print(f'PyPI matches: {wheel.name}')


def verify_github(root: Path, document: dict, *, platform_dir: Path | None = None) -> None:
    directories = (root,) if platform_dir is None else (root, platform_dir)
    paths = [path for directory in directories for path in directory.iterdir() if path.is_file()]
    local = {path.name: path for path in paths}
    remote = {item['name']: item for item in document['assets']}
    if len(local) != len(paths):
        raise ValueError('Duplicate filenames in the final GitHub asset inventory')
    if not local or set(local) != set(remote):
        raise ValueError(f'GitHub asset inventory mismatch: missing={sorted(local.keys() - remote.keys())}, '
                         f'unexpected={sorted(remote.keys() - local.keys())}')
    for name, path in local.items():
        item = remote[name]
        if item.get('digest') != 'sha256:' + digest(path) or item['size'] != path.stat().st_size:
            raise ValueError(f'GitHub asset differs or has no verified digest: {name}')
        print(f'GitHub matches: {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--platform-dir', type=Path, help='Matching platform assets included in the GitHub release')
    parser.add_argument('--pypi-version')
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--wait-seconds', type=float, default=0, help='Bounded wait for post-upload PyPI metadata visibility')
    parser.add_argument('--github-json', type=Path)
    args = parser.parse_args()
    if not args.pypi_version and not args.github_json:
        parser.error('Select at least one publication channel to verify')
    if args.pypi_version:
        verify_pypi(args.release_dir, args.pypi_version, allow_missing=args.allow_missing, wait_seconds=args.wait_seconds)
    if args.github_json:
        verify_github(args.release_dir, json.loads(args.github_json.read_text(encoding='utf-8')),
                      platform_dir=args.platform_dir)
