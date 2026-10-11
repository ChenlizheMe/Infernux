"""Expose the authoritative release identity to the local PowerShell entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
from hub_release import hub_version_for, project_build_number, project_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    version = project_version(ROOT)
    if args.version != version:
        raise ValueError(f"Requested version {args.version} does not match project.version {version}")
    build = project_build_number(ROOT)
    print(json.dumps({"version": version, "build_number": build, "hub_version": hub_version_for(version, build)}))


if __name__ == "__main__":
    main()
