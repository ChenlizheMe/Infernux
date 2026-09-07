"""Give the staged Linux wheel its audited PEP 600 platform tag."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PLATFORM_TAG = "manylinux_2_35_x86_64"


def repair(wheel_dir: Path) -> Path:
    wheels = tuple(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected one unaudited Linux wheel in {wheel_dir}, found {len(wheels)}"
        )
    staging = wheel_dir / "auditwheel"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "auditwheel",
            "repair",
            "--plat",
            PLATFORM_TAG,
            "--wheel-dir",
            str(staging),
            str(wheels[0]),
        ],
        check=True,
    )
    repaired = tuple(staging.glob("*.whl"))
    if len(repaired) != 1:
        raise RuntimeError(
            f"Expected one audited Linux wheel in {staging}, found {len(repaired)}"
        )
    destination = wheel_dir / repaired[0].name
    wheels[0].unlink()
    shutil.move(str(repaired[0]), destination)
    staging.rmdir()
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", required=True, type=Path)
    output = repair(parser.parse_args().wheel_dir.resolve())
    print(f"Audited Linux wheel: {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
