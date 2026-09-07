from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hub_object_publication_uses_versioned_immutable_keys(tmp_path, monkeypatch):
    module = _load(
        "infernux_publish_hub_objects",
        "scripts/release/publish_hub_objects.py",
    )
    version = "1.2.3"
    for name in module.release_assets(version):
        (tmp_path / name).write_bytes(b"release")
    published = []

    class FakePublisher:
        def __init__(self, token):
            assert token == "secret"

        def upload(self, source, key):
            published.append((source.name, key))

    monkeypatch.setattr(module, "Publisher", FakePublisher)
    module.publish(tmp_path, version, 4, "secret")

    assert [name for name, _key in published] == list(module.release_assets(version))
    assert all(key.startswith("hub/1.2.3/build-4/") for _name, key in published)


def test_release_catalog_reads_the_wheel_build_number(tmp_path):
    module = _load(
        "infernux_build_release_catalog",
        "scripts/release/build_release_catalog.py",
    )
    module.ROOT = tmp_path
    (tmp_path / "docs").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    (tmp_path / "setup.cfg").write_text(
        "[bdist_wheel]\nbuild_number = 4\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/hub-catalog.json").write_text(
        json.dumps({"$schema": "infernux.hub_catalog", "stable": "", "releases": []}),
        encoding="utf-8",
    )
    for platform, suffix, wheel_suffix in (
        ("windows-x64", ".exe", "win_amd64.whl"),
        ("linux-x64", "", "manylinux_2_35_x86_64.whl"),
    ):
        (tmp_path / f"InfernuxHub-{platform}-manifest.json").write_text(
            json.dumps({"version": "1.2.3", "platform": platform}),
            encoding="utf-8",
        )
        for name in (
            f"InfernuxHubInstaller-1.2.3-{platform}{suffix}",
            f"InfernuxHub-1.2.3-{platform}-full.zip",
            f"infernux-1.2.3-4-cp313-cp313-{wheel_suffix}",
        ):
            (tmp_path / name).write_bytes(b"release")

    module.build_catalog(tmp_path, "2026-09-07T00:00:00Z")

    hub = json.loads((tmp_path / "docs/hub-catalog.json").read_text(encoding="utf-8"))
    release = json.loads((tmp_path / "docs/release.json").read_text(encoding="utf-8"))
    assert hub["stable"] == "1.2.3"
    assert hub["releases"][0]["minimum_updatable_version"] == "0.4.0"
    assert all(
        "/hub/1.2.3/build-4/" in asset["url"]
        for asset in hub["releases"][0]["platforms"].values()
        for asset in asset.values()
    )
    assert {
        item["name"]
        for item in release["assets"]
        if item["kind"] == "python-wheel"
    } == {
        "infernux-1.2.3-4-cp313-cp313-win_amd64.whl",
        "infernux-1.2.3-4-cp313-cp313-manylinux_2_35_x86_64.whl",
    }


def test_desktop_ci_exposes_one_click_publication():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "publish_release:" in workflow
    assert "uses: ./.github/workflows/publish-desktop-release.yml" in workflow
    assert "needs: [portable-hub, windows-desktop, linux-desktop]" in workflow
