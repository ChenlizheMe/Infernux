"""Distribute the already published stable Hub without rebuilding the engine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from publish_hub_objects import publish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads((root / "docs/hub-catalog.json").read_text(encoding="utf-8"))
    stable = next(item for item in catalog["releases"] if item["version"] == catalog["stable"])
    version, separator, revision = stable["version"].partition("-")
    build = int(revision) if separator else 1
    hub_tag = stable["release_url"].rsplit("/", 1)[1]
    engine_tag = f"v{version}"
    args.download_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "gh", "release", "download", hub_tag, "--repo", args.repository,
        "--pattern", "InfernuxHub*", "--dir", str(args.download_dir),
    ], check=True)
    publish(args.download_dir, version, build, os.environ["R2_UPLOAD_TOKEN"])

    release = json.loads(subprocess.check_output([
        "gh", "release", "view", engine_tag, "--repo", args.repository, "--json", "assets",
    ], text=True, encoding="utf-8"))
    existing = {asset["name"] for asset in release["assets"]}
    installers = [args.download_dir / platform["installer"]["name"]
                  for platform in stable["platforms"].values()
                  if platform["installer"]["name"] not in existing]
    if installers:
        subprocess.run([
            "gh", "release", "upload", engine_tag, "--repo", args.repository,
            *(str(path) for path in installers),
        ], check=True)
    print(f"Hub {stable['version']} is available in object storage and engine release {engine_tag}.")


if __name__ == "__main__":
    main()
