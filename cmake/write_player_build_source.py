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
    parser.add_argument('--require-clean', action='store_true')
    args = parser.parse_args()
    release = runpy.run_path(str(args.source / 'python/infernux/version.py'))['ENGINE_RELEASE']
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.source, text=True).strip()
    status = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=args.source, text=True)
    if args.require_clean and status:
        parser.exit(1, f'Player SDK release source checkout has changes:\n{status}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'engine_release': release, 'revision': revision, 'dirty': bool(status),
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
