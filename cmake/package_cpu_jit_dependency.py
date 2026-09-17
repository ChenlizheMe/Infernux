"""Build and repair the pinned CPU dependency; never install into the host."""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def build_wheel(source: Path, output: Path) -> None:
    tool = {"win32": "delvewheel", "linux": "auditwheel"}.get(sys.platform)
    if tool is None:
        raise RuntimeError(f"CPU dependency wheel target does not support {sys.platform}")
    if importlib.util.find_spec(tool) is None:
        raise RuntimeError(f"Install the developer wheel repair tool '{tool}' before building")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Each invocation repairs only its own output, never an older wheel left
    # beside it. Native compilation still uses llvmlite's incremental build.
    with tempfile.TemporaryDirectory(prefix="cpu-wheel-", dir=output) as staging:
        subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", staging],
            cwd=source.resolve(), check=True,
            env={**os.environ, "LLVMLITE_PACKAGE_FORMAT": "wheel"},
        )
        wheels = list(Path(staging).glob("llvmlite-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one freshly built CPU dependency wheel, got {len(wheels)}")
        subprocess.run(
            [sys.executable, "-m", tool, "repair", "-w", str(output),
             *(["--analyze-existing"] if tool == "delvewheel" else []), str(wheels[0])],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args()
    build_wheel(args.source_dir, args.wheel_dir)
