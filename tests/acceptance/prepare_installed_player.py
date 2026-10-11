"""Create a fresh Player fixture using an installed engine and a complete platform package."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def prepare(fixture: Path, project: Path, platform_package: Path) -> dict:
    import infernux
    from infernux.lib import _Infernux
    from infernux.version import ENGINE_RELEASE
    from infernux.engine.project_context import set_project_root
    from infernux.plugins import InxPackage, PluginManager

    checkout = Path(__file__).resolve().parents[2]
    origins = dict(python=str(Path(infernux.__file__).resolve()), native=str(Path(_Infernux.__file__).resolve()))
    if any(Path(path).is_relative_to(checkout) for path in origins.values()):
        raise RuntimeError('Fixture preparation requires the installed candidate wheel')
    if project.exists():
        raise RuntimeError(f'Acceptance requires a fresh project: {project}')
    if not (fixture / 'Assets').is_dir() or not (fixture / 'ProjectSettings').is_dir():
        raise ValueError(f'Invalid acceptance fixture: {fixture}')
    shutil.copytree(fixture, project)
    # This creates a new candidate fixture; it never upgrades an existing project.
    (project / '.infernux-version').write_text(ENGINE_RELEASE + '\n', encoding='utf-8')
    # A candidate build must not overwrite an immutable published plugin
    # version in the developer's shared Hub cache.
    cache = project / '.runtime/package-cache'
    os.environ['INFERNUX_PACKAGE_CACHE_ROOT'] = str(cache)
    archive = project / '.runtime/platform.inxpkg'
    archive.parent.mkdir(parents=True, exist_ok=True)
    set_project_root(str(project))
    InxPackage.export_source(str(platform_package), str(archive))
    manager = PluginManager(str(project), runtime=True)
    try:
        installed = manager.install_package(str(archive), install_dependencies=False)
        if not installed.loaded:
            raise RuntimeError(f'Platform package installation failed: {installed.error}')
    finally:
        manager.shutdown()
    report = dict(project=str(project), engine_release=ENGINE_RELEASE,
                  installed_origins=origins, platform_package=str(platform_package))
    (project / '.runtime/preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('project', type=Path)
    parser.add_argument('platform_package', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.fixture.resolve(), args.project.resolve(), args.platform_package.resolve())))


if __name__ == '__main__':
    main()
