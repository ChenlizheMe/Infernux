"""Hub publication identity checks; these tests never load the engine runtime."""

import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hub_release_catalog_policy", ROOT / "scripts/release/build_release_catalog.py"
)
catalog_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog_builder)


@pytest.mark.parametrize("version", ["0.4.0", "0.3.9", "0.4.0-rc.1", "0.4.0+build.3"])
def test_published_hub_versions_cannot_be_reused_or_downgraded(tmp_path, monkeypatch, version):
    _project(tmp_path, monkeypatch, version)

    with pytest.raises(ValueError, match="increment project.version"):
        catalog_builder.require_new_hub_version()


def test_a_new_hub_patch_version_can_be_published(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, "0.4.1")

    assert catalog_builder.require_new_hub_version() == "0.4.1"


def test_preparing_a_release_does_not_make_it_published(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch, "0.4.1")
    path = tmp_path / "docs/hub-catalog.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["releases"].insert(0, {"version": "0.4.1", "published_at": None})
    path.write_text(json.dumps(document), encoding="utf-8")

    assert catalog_builder.require_new_hub_version() == "0.4.1"


def _project(tmp_path, monkeypatch, version):
    monkeypatch.setattr(catalog_builder, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nversion = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/hub-catalog.json").write_text(
        json.dumps({"releases": [{"version": "0.4.0", "published_at": "2026-09-06T17:56:54Z"}]}),
        encoding="utf-8",
    )


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "hub_catalog_deployment", ROOT / "scripts/release/deploy_hub_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {"stable": "0.4.1", "releases": []}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(module, "MAX_DEPLOYMENT_ATTEMPTS", 3)
    monkeypatch.setattr(module.time, "sleep", lambda *_args: None)
    return module, catalog, expected


def test_catalog_deployment_waits_for_the_requested_commit(deployment, monkeypatch):
    module, catalog, expected = deployment
    calls = []
    builds = iter([
        {"commit": "old", "status": "built"},
        {"commit": "current", "status": "building"},
        {"commit": "current", "status": "built"},
    ])

    def request(repository, path, *, method="GET"):
        calls.append((repository, path, method))
        return {} if method == "POST" else next(builds)

    def public_catalog(request, *, timeout):
        assert timeout == 15
        assert request.full_url == "https://infernux-engine.com/hub-catalog.json"
        assert dict(request.header_items()) == {
            "Accept": "application/json", "User-agent": "InfernuxHub-Updater", "Cache-control": "no-cache",
        }
        return io.BytesIO(json.dumps(expected).encode())

    monkeypatch.setattr(module, "_pages_request", request)
    monkeypatch.setattr(module.urllib.request, "urlopen", public_catalog)

    module.deploy_catalog("owner/repo", "current", catalog)

    assert calls == [("owner/repo", "builds", "POST")] + [
        ("owner/repo", "builds/latest", "GET")
    ] * 3


@pytest.mark.parametrize("status", ["building", "errored"])
def test_catalog_deployment_cannot_claim_success_before_publication(
    deployment, monkeypatch, status,
):
    module, catalog, _expected = deployment
    calls = []

    def request(repository, path, *, method="GET"):
        calls.append(method)
        return {"commit": "current", "status": status, "error": "build failure"}

    monkeypatch.setattr(module, "_pages_request", request)
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("unpublished build"),
    )

    with pytest.raises((RuntimeError, TimeoutError)):
        module.deploy_catalog("owner/repo", "current", catalog)

    assert calls == ["POST"] + ["GET"] * (3 if status == "building" else 1)


def test_catalog_deployment_rejects_a_stale_public_catalog(deployment, monkeypatch):
    module, catalog, _expected = deployment
    monkeypatch.setattr(module, "_pages_request", lambda *_args, **_kwargs: {"commit": "current", "status": "built"})
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b'{"stable":"0.4.0"}'))

    with pytest.raises(TimeoutError, match="public Hub catalog did not reach"):
        module.deploy_catalog("owner/repo", "current", catalog)


def test_catalog_deployment_waits_for_the_actual_client_url_to_refresh(deployment, monkeypatch):
    module, catalog, expected = deployment
    responses = iter([{"stable": "0.4.0"}, expected])
    urls = []
    monkeypatch.setattr(module, "_pages_request", lambda *_args, **_kwargs: {"commit": "current", "status": "built"})

    def public_catalog(request, *, timeout):
        urls.append(request.full_url)
        return io.BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", public_catalog)

    module.deploy_catalog("owner/repo", "current", catalog)

    assert urls == ["https://infernux-engine.com/hub-catalog.json"] * 2
