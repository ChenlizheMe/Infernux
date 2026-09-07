"""Run the Python regression suite."""

import sys
import subprocess
from pathlib import Path


def main() -> int:
    test_root = Path(__file__).resolve().with_name("test")
    return subprocess.call([sys.executable, "-m", "pytest", str(test_root)])


if __name__ == "__main__":
    raise SystemExit(main())
