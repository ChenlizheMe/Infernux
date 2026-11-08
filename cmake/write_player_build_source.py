"""Record the immutable source revision needed to rebuild this SDK's Players."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    release = runpy.run_path(str(args.source / 'python/infernux/version.py'))['ENGINE_RELEASE']
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip()
    dirty = bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'engine_release': release, 'revision': revision, 'dirty': dirty,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
