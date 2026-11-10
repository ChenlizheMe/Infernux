"""Refresh selected submodule Player payloads and engine-versioned packages locally."""
import argparse
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile
from types import SimpleNamespace

from build_platform_plugins import ROOT, RELEASE, PLATFORM_PLAYERS, PLATFORMS, PYTHON_ABI, build, read, write


def select_targets(targets):
    if targets == ['all']:
        return PLATFORMS
    if targets == ['host']:
        return ('windows',) if sys.platform == 'win32' else ('linux',) if sys.platform.startswith('linux') else ()
    unknown = set(targets) - PLATFORM_PLAYERS.keys()
    if unknown:
        raise ValueError(f'Unknown platform targets: {sorted(unknown)}')
    return tuple(dict.fromkeys(targets))


def replace_payload(source, destination):
    """Publish a complete directory, keeping GUIDs for surviving asset paths."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.player-refresh-', dir=destination.parent))
    candidate, previous = staging / 'candidate', staging / 'previous'
    published = False
    try:
        shutil.copytree(source, candidate, ignore=shutil.ignore_patterns('__pycache__'))
        if destination.exists():
            for meta in destination.rglob('*.meta'):
                relative = meta.relative_to(destination)
                asset = candidate / str(relative)[:-5]
                if asset.exists():
                    shutil.copy2(meta, candidate / relative)
            destination.rename(previous)
        try:
            candidate.rename(destination)
            published = True
        except BaseException:
            if previous.exists():
                previous.rename(destination)
            raise
    finally:
        # Retain the old payload for recovery if even the rollback rename fails.
        if published or not previous.exists():
            shutil.rmtree(staging)


def refresh(targets, *, tools=None, build_root=None, output=None):
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    if abi != PYTHON_ABI:
        raise ValueError(f'Platform release requires {PYTHON_ABI}, running {abi}')
    # Python ABI and release changes select independent compile caches. Native
    # builders reconfigure every invocation and track their source dependencies.
    build_root = build_root or ROOT / 'out/build/platform-plugins' / RELEASE / abi
    output = output or ROOT / 'dist/platform-plugins' / RELEASE
    for platform in targets:
        descriptor = PLATFORM_PLAYERS[platform]
        payload = build_root / platform / 'publish/player'
        args = SimpleNamespace(build_root=build_root, output=output, tools=tools)
        build(platform, args)
        package_root = ROOT / descriptor['repository_path'] / 'package'
        # Find the module from the existing package layout rather than imposing
        # the declaration ID as a Python import name.
        modules = [path for path in (package_root / 'editor').iterdir() if path.is_dir() and not path.name.startswith('.')]
        module = next(path for path in modules if (path / '__init__.py').is_file())
        replace_payload(payload, module / 'player')
        if tools is not None and descriptor['payload_format'] == 'web':
            replace_payload(tools, module / 'tools')
        manifest = read(output / (descriptor['reference'].replace('/', '.') + '.release.json'))
        write(output / (platform + '.refresh.json'), {
            'engine_release': RELEASE, 'python_abi': abi,
            'python_soabi': sysconfig.get_config_var('SOABI'),
            'source_revision': manifest['source_revision'], 'destination': str(module / 'player'),
        })
        print(f'Refreshed {descriptor["reference"]} for Infernux {RELEASE} ({abi})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', default=['host'])
    parser.add_argument('--tools', default='')
    parser.add_argument('--build-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    targets = select_targets(args.targets)
    if not targets:
        parser.error('This host has no native Player builder; select an explicit cross-compilation target')
    if ('windows' in targets and sys.platform != 'win32') or ('linux' in targets and not sys.platform.startswith('linux')):
        parser.error('Windows/Linux require their matching build hosts. Use CI to collect the complete platform set.')
    if 'web' in targets and not args.tools:
        parser.error('Web refresh requires --tools with Windows/Linux shader tools')
    refresh(targets, tools=Path(args.tools).resolve() if args.tools else None,
            build_root=args.build_root, output=args.output)


if __name__ == '__main__':
    main()
