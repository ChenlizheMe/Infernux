"""Copy the registry's exact GitHub release assets to the official R2 channel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

from publish_hub_objects import Publisher

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "python/infernux/resources/official_packages/official-registry.json"


def release_identity(package: dict) -> tuple[str, str, str]:
    repository = urllib.parse.urlparse(package["repository"])
    source = urllib.parse.urlparse(package["source"]["location"])
    name = package["artifact"]
    expected = f"plugins/{package['reference'].replace('/', '.')}/{package['version']}/{name}"
    if repository.scheme != "https" or repository.netloc != "github.com":
        raise ValueError("Official plugin releases must come from GitHub")
    if source.scheme != "https" or source.netloc != "downloads.infernux-engine.com" or source.path != "/" + expected:
        raise ValueError("Official plugin destination disagrees with its registry identity")
    if Path(name).name != name or not name.endswith(".inxpkg"):
        raise ValueError("Invalid plugin artifact name")
    return repository.path.strip("/"), "v" + package["version"], expected


def stage(package: dict, destination: Path) -> tuple[Path, str]:
    repository, tag, key = release_identity(package)
    release = json.loads(subprocess.check_output([
        "gh", "release", "view", tag, "--repo", repository,
        "--json", "tagName,isDraft,assets",
    ], text=True))
    if release["isDraft"] or release["tagName"] != tag:
        raise ValueError(f"Plugin release is not public: {repository}/{tag}")
    name = package["artifact"]
    assets = [asset for asset in release["assets"] if asset["name"] == name]
    if len(assets) != 1 or assets[0]["size"] <= 0:
        raise ValueError(f"Missing complete plugin payload: {repository}/{tag}")
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "gh", "release", "download", tag, "--repo", repository,
        "--pattern", name, "--dir", str(destination), "--clobber",
    ], check=True)
    path = destination / name
    if path.stat().st_size != assets[0]["size"]:
        raise ValueError(f"Incomplete download: {name}")
    with path.open("rb") as stream:
        if stream.read(8) != b"INXPKG\0\0":
            raise ValueError(f"Invalid InxPackage: {name}")
    return path, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="", help="Publish one reference; empty publishes all platform packages and the registry")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    packages = [item for item in registry["packages"] if (
        item["reference"] == args.reference if args.reference else item["category"] == "platform_build"
    )]
    if not packages:
        raise ValueError(f"No matching official package: {args.reference}")
    publisher = Publisher(os.environ["R2_UPLOAD_TOKEN"])
    staged = [stage(item, ROOT / "out/plugin-publication" / item["reference"] / item["version"]) for item in packages]
    for path, key in staged:
        publisher.upload(path, key)
    # Never advertise missing packages. This also checks unchanged editor tools.
    if not args.reference:
        for item in registry["packages"]:
            request = urllib.request.Request(
                item["source"]["location"], method="HEAD",
                headers={"User-Agent": "Infernux-Release-Publisher"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                if int(response.headers["Content-Length"]) <= 0:
                    raise ValueError(f"Empty public plugin: {item['reference']}")
        publisher.upload(REGISTRY, "plugins/official-registry.json")


if __name__ == "__main__":
    main()
