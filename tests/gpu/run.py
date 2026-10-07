"""Run a GPU acceptance script against this checkout, independent of PYTHONPATH."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    directory = Path(__file__).resolve().parent
    scripts = sorted(path.stem for path in directory.glob("*_test.py"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available GPU tests without loading the engine")
    parser.add_argument("test", nargs="?", choices=scripts)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected test")
    options = parser.parse_args()
    if options.list:
        print("\n".join(scripts))
        return
    if options.test is None:
        parser.error("select a test or use --list")
    sys.path[:0] = [str(directory.parents[1] / "python"), str(directory)]
    script = directory / f"{options.test}.py"
    sys.argv = [str(script), *options.arguments]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
