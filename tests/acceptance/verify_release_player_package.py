"""Check the sealed Release contract inside a Web fixture or Android APK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import zipfile


def verify(artifact: Path, target: str) -> dict:
    from infernux.engine.player_package_native import read_entry, read_manifest
    from tests.acceptance.release_player_smoke import require_release_manifest

    catalog_path = 'InfernuxPlatformFixture_Data/AssetCatalog.inxcat'
    if target == 'android':
        entry = 'assets/player/' + catalog_path
        with zipfile.ZipFile(artifact) as apk:
            if apk.namelist().count(entry) != 1:
                raise RuntimeError('APK must contain exactly one fixture asset catalog')
            catalog_bytes = apk.read(entry)
    elif target == 'web':
        entries = [item['path'] for item in read_manifest(artifact)['files']]
        if entries.count(catalog_path) != 1:
            raise RuntimeError('Web package must contain exactly one fixture asset catalog')
        catalog_bytes = read_entry(artifact, catalog_path)
    else:
        raise ValueError(f'Unsupported acceptance target: {target}')
    with tempfile.TemporaryDirectory(prefix='infernux-release-catalog-') as temporary:
        catalog = Path(temporary) / 'AssetCatalog.inxcat'
        catalog.write_bytes(catalog_bytes)
        manifest = json.loads(read_entry(catalog, 'BuildManifest.json'))
    require_release_manifest(manifest)
    return dict(status='passed', target=target, artifact=str(artifact), flavor='PlayerRelease', player_control='disabled')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--target', choices=('web', 'android'), required=True)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    # Cross-target fixture builds currently use the explicitly selected source host.
    checkout = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(checkout), str(checkout / 'python')]
    result = verify(args.artifact.resolve(), args.target)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
