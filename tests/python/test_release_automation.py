from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


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
    for name in module.release_assets("1.2.3-4"):
        (tmp_path / name).write_bytes(b"release")
    published = []

    class FakePublisher:
        def __init__(self, token):
            assert token == "secret"

        def upload(self, source, key):
            published.append((source.name, key))

    monkeypatch.setattr(module, "Publisher", FakePublisher)
    module.publish(tmp_path, version, 4, "secret")

    assert [name for name, _key in published] == list(module.release_assets("1.2.3-4"))
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
    (tmp_path / "python/infernux").mkdir(parents=True)
    (tmp_path / "python/infernux/version.py").write_text(
        "ENGINE_BUILD_NUMBER = 4\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/hub-catalog.json").write_text(
        json.dumps({"$schema": "infernux.hub_catalog", "stable": "", "releases": []}),
        encoding="utf-8",
    )
    wheel_names = {
        "windows-x64": "infernux-1.2.3-4-cp313-cp313-win_amd64.whl",
        "linux-x64": "infernux-1.2.3-4-cp313-cp313-linux_x86_64.whl",
    }
    for platform, suffix in (
        ("windows-x64", ".exe"),
        ("linux-x64", ""),
    ):
        (tmp_path / f"InfernuxHub-{platform}-manifest.json").write_text(
            json.dumps({"version": "1.2.3-4", "platform": platform}),
            encoding="utf-8",
        )
        for name in (
            f"InfernuxHubInstaller-1.2.3-4-{platform}{suffix}",
            f"InfernuxHub-1.2.3-4-{platform}-full.zip",
            wheel_names[platform],
        ):
            (tmp_path / name).write_bytes(b"release")

    module.build_catalog(tmp_path, "2026-09-07T00:00:00Z")

    hub = json.loads((tmp_path / "docs/hub-catalog.json").read_text(encoding="utf-8"))
    release = json.loads((tmp_path / "docs/release.json").read_text(encoding="utf-8"))
    assert hub["stable"] == "1.2.3-4"
    assert release["tag"] == "v1.2.3-v4"
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
        "infernux-1.2.3-4-cp313-cp313-linux_x86_64.whl",
    }


def test_release_catalog_rejects_republishing_an_existing_hub_version(tmp_path):
    module = _load(
        "infernux_build_release_catalog_republish_guard",
        "scripts/release/build_release_catalog.py",
    )
    module.ROOT = tmp_path
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "1.2.4"\n',
        encoding="utf-8",
    )
    (tmp_path / "python/infernux").mkdir(parents=True)
    (tmp_path / "python/infernux/version.py").write_text("ENGINE_BUILD_NUMBER = 1\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/hub-catalog.json").write_text(
        json.dumps(
            {
                "$schema": "infernux.hub_catalog",
                "stable": "1.2.3",
                "releases": [
                    {
                        "version": "1.2.4",
                        "published_at": "2026-09-20T00:00:00Z",
                    },
                    {
                        "version": "1.2.3",
                        "published_at": "2026-09-01T00:00:00Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already published.*increment the build number"):
        module.require_new_hub_version()
    (tmp_path / "python/infernux/version.py").write_text("ENGINE_BUILD_NUMBER = 2\n")
    assert module.require_new_hub_version() == "1.2.4"


@pytest.mark.parametrize("missing_platform", ["windows-x64", "linux-x64"])
def test_release_catalog_does_not_publish_before_both_pypi_wheels_exist(tmp_path, monkeypatch, missing_platform):
    module = _load("infernux_incomplete_pypi_release", "scripts/release/build_release_catalog.py")
    names = {"windows-x64": "windows.whl", "linux-x64": "linux.whl"}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "require_new_hub_version", lambda: "1.2.3")
    monkeypatch.setattr(module, "wheel_build_number", lambda: "1")
    monkeypatch.setattr(module, "release_wheel_names", lambda *args: names)
    monkeypatch.setattr(module, "pypi_wheel_urls", lambda version: {
        name: f"https://files.pythonhosted.org/{name}"
        for platform, name in names.items() if platform != missing_platform
    })
    (tmp_path / "docs").mkdir()
    catalog = tmp_path / "docs/hub-catalog.json"
    catalog.write_text('{"stable":"1.2.2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="PyPI has not published the exact release wheels"):
        module.build_catalog(tmp_path, "2026-09-30T00:00:00Z", resolve_pypi=True)
    assert json.loads(catalog.read_text(encoding="utf-8"))["stable"] == "1.2.2"
    assert not (tmp_path / "docs/release.json").exists()


def test_desktop_ci_exposes_one_click_publication():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "publish_release:" in workflow
    assert "uses: ./.github/workflows/publish-desktop-release.yml" in workflow
    assert "needs: [portable-hub, windows-desktop, linux-desktop]" in workflow


def test_release_body_is_english_and_reports_actual_signing_state(tmp_path):
    module = _load("infernux_release_notes", "scripts/release/build_release_notes.py")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    for name, text in (("UpdateLog.md", "New worlds"), ("UpdateLog-zh.md", "新的世界")):
        (tmp_path / name).write_text(
            f"# Infernux v1.2.3 · Worlds\n\n{text}\n\n---\n\n# Infernux v1.2.2 · Old\n",
            encoding="utf-8",
        )
    unsigned = module.build_notes(tmp_path, signed=False)
    assert "New worlds" in unsigned and "新的世界" not in unsigned
    assert "v1.2.2" not in unsigned
    assert "this release is unsigned" in unsigned
    assert module.SIGNING_CREDIT not in unsigned
    signed = module.build_notes(tmp_path, signed=True)
    assert module.SIGNING_CREDIT in signed
    assert "this release is unsigned" not in signed
    (tmp_path / "UpdateLog.md").write_text("# Infernux v1.2.2 · Old\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must begin with the release"):
        module.build_notes(tmp_path, signed=False)
