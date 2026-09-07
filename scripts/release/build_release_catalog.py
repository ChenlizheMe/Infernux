"""Generate website release metadata from the actual versioned distributions."""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path
import tomllib
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
MINIMUM_UPDATABLE_VERSION = "0.4.0"


def wheel_build_number() -> str:
    configuration = configparser.ConfigParser()
    configuration.read(ROOT / "setup.cfg", encoding="utf-8")
    value = configuration.get("bdist_wheel", "build_number", fallback="").strip()
    if not value.isdigit() or int(value) < 1:
        raise ValueError("setup.cfg must declare a positive bdist_wheel build_number")
    return value


def pypi_wheel_urls(version: str) -> dict[str, str]:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/Infernux/{version}/json",
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


def build_catalog(
    release_dir: Path,
    published_at: str | None,
    linux_inventory: Path | None = None,
    *,
    resolve_pypi: bool = False,
) -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    wheel_build = wheel_build_number()
    github_base = f"https://github.com/ChenlizheMe/Infernux/releases/download/v{version}"
    object_base = f"https://downloads.infernux-engine.com/hub/{version}/build-{wheel_build}"
    release_url = f"https://github.com/ChenlizheMe/Infernux/releases/tag/v{version}"
    wheel_urls = pypi_wheel_urls(version) if resolve_pypi else {}
    platforms = {}
    assets = []
    ci = json.loads(linux_inventory.read_text(encoding="utf-8")) if linux_inventory else None
    for platform, suffix, wheel_suffix in (
        ("windows-x64", ".exe", "win_amd64.whl"),
        ("linux-x64", "", "manylinux_2_35_x86_64.whl"),
    ):
        manifest_name = f"InfernuxHub-{platform}-manifest.json"
        from_ci = platform == "linux-x64" and ci is not None
        manifest = ci["manifest"] if from_ci else json.loads((release_dir / manifest_name).read_text(encoding="utf-8"))
        if manifest["version"] != version or manifest["platform"] != platform:
            raise ValueError(f"{manifest_name} does not describe {version}/{platform}")
        def asset_size(name):
            return ci["files"][f"{version}/{name}"] if from_ci else (release_dir / name).stat().st_size
        installer_name = f"InfernuxHubInstaller-{version}-{platform}{suffix}"
        update_name = f"InfernuxHub-{version}-{platform}-full.zip"
        wheel_name = f"infernux-{version}-{wheel_build}-cp313-cp313-{wheel_suffix}"
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
                else wheel_urls.get(name, f"https://pypi.org/project/Infernux/{version}/")
            )
            assets.append({
                "kind": kind,
                "name": name,
                "size_bytes": asset_size(name),
                "url": primary,
                "fallback_url": f"{github_base}/{name}",
            })
    release = {
        "schema_version": 2, "version": version, "tag": f"v{version}",
        "name": f"Infernux v{version}", "channel": "stable", "published_at": published_at,
        "platforms": ["Windows 10/11 x64", "Linux x86_64"], "python_abi": "CPython 3.13 x64",
        "release_url": release_url, "assets": assets,
    }
    catalog_path = ROOT / "docs/hub-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["stable"] = version
    catalog["releases"] = [{
        "version": version, "channel": "stable", "published_at": published_at,
        "release_url": release_url,
        "minimum_updatable_version": MINIMUM_UPDATABLE_VERSION,
        "platforms": platforms,
    }] + [item for item in catalog["releases"] if item["version"] != version]
    for path, document in ((ROOT / "docs/release.json", release), (catalog_path, catalog)):
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {version} release catalogs from {release_dir}; published_at={published_at!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--published-at", help="Actual GitHub publication timestamp; omit while preparing the release")
    parser.add_argument("--linux-inventory", type=Path, help="Verified Linux CI archive inventory instead of local Linux files")
    parser.add_argument("--resolve-pypi", action="store_true", help="Use the published files.pythonhosted.org wheel URLs")
    args = parser.parse_args()
    build_catalog(
        args.release_dir,
        args.published_at,
        args.linux_inventory,
        resolve_pypi=args.resolve_pypi,
    )
