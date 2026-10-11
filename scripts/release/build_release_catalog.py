"""Generate website release metadata from the actual versioned distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
import urllib.request
import sys

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
from hub_release import hub_version_for, project_build_number, release_tag_for
MINIMUM_UPDATABLE_VERSION = "0.4.0"


def require_new_hub_version() -> str:
    """Require an unpublished application/build identity without replacing old assets."""
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    hub_version = hub_version_for(version, project_build_number(ROOT))
    catalog = json.loads((ROOT / "docs/hub-catalog.json").read_text(encoding="utf-8"))
    published = [
        item["version"] for item in catalog["releases"] if item["published_at"] is not None
    ]
    if published:
        latest = max(published, key=lambda value: Version(value.split("+", 1)[0]))
        if Version(hub_version.split("+", 1)[0]) <= Version(latest.split("+", 1)[0]):
            raise ValueError(
                f"Hub {latest} is already published; increment the build number or project.version "
                "before publishing changed Hub artifacts."
            )
    return version


def wheel_build_number() -> str:
    return str(project_build_number(ROOT))


def pypi_wheel_urls(version: str) -> dict[str, str]:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/infernux/{version}/json",
        headers={"Accept": "application/json", "User-Agent": "Infernux-Release-Publisher"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        document = json.load(response)
    urls = document.get("urls") if isinstance(document, dict) else None
    if not isinstance(urls, list):
        raise ValueError(f"PyPI returned no file catalog for Infernux {version}")
    return {
        str(item["filename"]): str(item["url"])
        for item in urls
        if isinstance(item, dict)
        and isinstance(item.get("filename"), str)
        and isinstance(item.get("url"), str)
    }


def release_wheel_names(
    release_dir: Path,
    version: str,
    wheel_build: str,
    linux_inventory: dict[str, object] | None = None,
) -> dict[str, str]:
    prefix = f"infernux-{version}-{wheel_build}-cp313-cp313-"
    windows = f"{prefix}win_amd64.whl"
    if linux_inventory is None:
        candidates = sorted(
            path.name
            for path in release_dir.glob(f"{prefix}*.whl")
            if path.name != windows
        )
    else:
        files = linux_inventory.get("files")
        if not isinstance(files, dict):
            raise ValueError("Linux release inventory has no file map")
        candidates = sorted(
            Path(str(relative)).name
            for relative in files
            if Path(str(relative)).name.startswith(prefix)
            and Path(str(relative)).name.endswith(".whl")
            and Path(str(relative)).name != windows
        )
    if len(candidates) != 1:
        raise ValueError(
            "Desktop release must contain exactly one audited Linux wheel; "
            f"found {candidates}"
        )
    return {"windows-x64": windows, "linux-x64": candidates[0]}


def build_catalog(
    release_dir: Path,
    published_at: str | None,
    linux_inventory: Path | None = None,
    *,
    resolve_pypi: bool = False,
) -> None:
    version = require_new_hub_version()
    wheel_build = wheel_build_number()
    hub_version = hub_version_for(version, int(wheel_build))
    release_tag = release_tag_for(version, int(wheel_build))
    github_base = f"https://github.com/ChenlizheMe/Infernux/releases/download/{release_tag}"
    object_base = f"https://downloads.infernux-engine.com/hub/{version}/build-{wheel_build}"
    release_url = f"https://github.com/ChenlizheMe/Infernux/releases/tag/{release_tag}"
    wheel_urls = pypi_wheel_urls(version) if resolve_pypi else {}
    platforms = {}
    assets = []
    ci = json.loads(linux_inventory.read_text(encoding="utf-8")) if linux_inventory else None
    wheel_names = release_wheel_names(release_dir, version, wheel_build, ci)
    if resolve_pypi:
        missing = sorted(set(wheel_names.values()) - wheel_urls.keys())
        if missing:
            raise ValueError(f"PyPI has not published the exact release wheels: {missing}")
    for platform, suffix in (
        ("windows-x64", ".exe"),
        ("linux-x64", ""),
    ):
        manifest_name = f"InfernuxHub-{platform}-manifest.json"
        from_ci = platform == "linux-x64" and ci is not None
        manifest = ci["manifest"] if from_ci else json.loads((release_dir / manifest_name).read_text(encoding="utf-8"))
        if manifest["version"] != hub_version or manifest["platform"] != platform:
            raise ValueError(f"{manifest_name} does not describe {hub_version}/{platform}")
        def asset_size(name):
            return ci["files"][f"{version}/{name}"] if from_ci else (release_dir / name).stat().st_size
        installer_name = f"InfernuxHubInstaller-{hub_version}-{platform}{suffix}"
        update_name = f"InfernuxHub-{hub_version}-{platform}-full.zip"
        wheel_name = wheel_names[platform]
        platforms[platform] = {
            "installer": {
                "name": installer_name,
                "url": f"{object_base}/{installer_name}",
                "fallback_url": f"{github_base}/{installer_name}",
            },
            "update": {
                "name": update_name,
                "url": f"{object_base}/{update_name}",
                "fallback_url": f"{github_base}/{update_name}",
                "size": asset_size(update_name),
            },
            "manifest": {
                "name": manifest_name,
                "url": f"{object_base}/{manifest_name}",
                "fallback_url": f"{github_base}/{manifest_name}",
            },
        }
        for kind, name in (("hub-installer", installer_name), ("python-wheel", wheel_name)):
            primary = (
                f"{object_base}/{name}"
                if kind == "hub-installer"
                else (wheel_urls[name] if resolve_pypi else f"https://pypi.org/project/infernux/{version}/")
            )
            assets.append({
                "kind": kind,
                "name": name,
                "size_bytes": asset_size(name),
                "url": primary,
                "fallback_url": f"{github_base}/{name}",
            })
    release = {
        "schema_version": 2, "version": version, "tag": release_tag,
        "name": f"Infernux {version} v{wheel_build}", "channel": "stable", "published_at": published_at,
        "platforms": ["Windows 10/11 x64", "Linux x86_64"], "python_abi": "CPython 3.13 x64",
        "release_url": release_url, "assets": assets,
    }
    catalog_path = ROOT / "docs/hub-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["stable"] = hub_version
    catalog["releases"] = [{
        "version": hub_version, "channel": "stable", "published_at": published_at,
        "release_url": release_url,
        "minimum_updatable_version": MINIMUM_UPDATABLE_VERSION,
        "platforms": platforms,
    }] + [item for item in catalog["releases"] if Version(item["version"]).release != Version(version).release]
    for path, document in ((ROOT / "docs/release.json", release), (catalog_path, catalog)):
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {version} release catalogs from {release_dir}; published_at={published_at!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--check-version", action="store_true", help="Check the Hub publication identity without writing catalogs")
    parser.add_argument("--published-at", help="Actual GitHub publication timestamp; omit while preparing the release")
    parser.add_argument("--linux-inventory", type=Path, help="Verified Linux CI archive inventory instead of local Linux files")
    parser.add_argument("--resolve-pypi", action="store_true", help="Use the published files.pythonhosted.org wheel URLs")
    args = parser.parse_args()
    if args.check_version:
        require_new_hub_version()
    else:
        if args.release_dir is None:
            parser.error("--release-dir is required unless --check-version is used")
        build_catalog(
            args.release_dir,
            args.published_at,
            args.linux_inventory,
            resolve_pypi=args.resolve_pypi,
        )
