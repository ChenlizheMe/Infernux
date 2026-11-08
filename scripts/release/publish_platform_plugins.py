"""Publish one complete engine-specific platform set; advertise its catalog last."""
import argparse
import os
from pathlib import Path

from build_platform_plugins import RELEASE, PLATFORMS, PLATFORM_PLAYERS, assemble
from publish_hub_objects import Publisher
from plugin_catalog_history import merge_history, published_catalog
from build_platform_plugins import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assemble(args.directory)
    if args.check:
        return
    publisher = Publisher(os.environ['R2_UPLOAD_TOKEN'])
    for platform in PLATFORMS:
        for extension in ('.inxpkg', '.release.json'):
            stem = PLATFORM_PLAYERS[platform]['reference'].replace('/', '.')
            path = args.directory / (stem + extension)
            publisher.upload(path, f'platform-plugins/{RELEASE}/{path.name}')
    publisher.upload(args.directory / 'registry.json', f'platform-plugins/{RELEASE}/registry.json')
    history = merge_history(published_catalog(), read(args.directory / 'registry.json'))
    write(args.directory / 'published-registry.json', history)
    publisher.upload(args.directory / 'published-registry.json', 'plugins/official-registry.json')


if __name__ == '__main__':
    main()
