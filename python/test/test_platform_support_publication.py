"""Keep host metadata and translated public support claims on one matrix."""
from __future__ import annotations

import json
import ast
from pathlib import Path
import tomllib
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]


def _matrix():
    return json.loads((ROOT / "docs/platform-support.json").read_text(encoding="utf-8"))


def test_engine_and_current_release_metadata_use_one_version():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    runtime = ast.parse((ROOT / "python/infernux/version.py").read_text(encoding="utf-8"))
    assignment = next(node for node in runtime.body if isinstance(node, ast.Assign) and node.targets[0].id == "ENGINE_VERSION")
    assert ast.literal_eval(assignment.value) == version
    # Published downloads stay on the last public version while the next
    # release is being built. All published metadata must still agree.
    published = json.loads((ROOT / "docs/release.json").read_text(encoding="utf-8"))["version"]
    assert tuple(map(int, published.split("."))) <= tuple(map(int, version.split(".")))
    if published != version:
        for filename in ("UpdateLog.md", "UpdateLog-zh.md"):
            assert (ROOT / filename).read_text(encoding="utf-8").startswith(f"# Infernux v{version} ")
    for filename, key in (
        ("release.json", "version"),
        ("docs-manifest.json", "documented_release"),
        ("release-notes.json", "version"),
        ("platform-support.json", "released_version"),
    ):
        assert json.loads((ROOT / "docs" / filename).read_text(encoding="utf-8"))[key] == published
    # A rebuilt Hub has a post-release identity while its engine stays on the
    # same base version (for example Hub 0.4.1-2 serves engine 0.4.1 build 2).
    catalog = json.loads((ROOT / "docs/hub-catalog.json").read_text(encoding="utf-8"))
    assert Version(catalog["stable"]).base_version == published


def test_wheel_classifiers_match_supported_host_targets():
    matrix = _matrix()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert metadata["version"] == matrix["development_version"]
    classifiers = {value for value in metadata["classifiers"] if value.startswith("Operating System ::")}
    assert classifiers == {item["classifier"] for item in matrix["platforms"] if item["editor"]}
    assert all("classifier" not in item for item in matrix["platforms"] if not item["editor"])


def test_both_readme_tables_match_the_support_matrix():
    matrix = _matrix()
    for filename, language in (("README.md", "en"), ("README-zh.md", "zh")):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "docs/platform-support.json" in text
        assert "SUPPORT.md#platform-support" in text
        for item in matrix["platforms"]:
            yes, no = ("Yes", "No") if language == "en" else ("有", "无")
            editor = yes if item["editor"] else no
            player = yes if item["player"] == "Yes" else item["player"]
            # README presents capabilities; CI evidence belongs in SUPPORT.md.
            expected = f'| {item["label"]} | {editor} | {player} | {item["graphics"]} |'
            assert expected in text, (filename, item["id"])


def test_released_platform_claims_match_the_public_hub_catalog():
    matrix = _matrix()
    catalog = json.loads((ROOT / "docs/hub-catalog.json").read_text(encoding="utf-8"))
    assert matrix["released_version"] == Version(catalog["stable"]).base_version
    release = next(item for item in catalog["releases"] if item["version"] == catalog["stable"])
    assert {item["id"] for item in matrix["platforms"] if item["released"]} == set(release["platforms"])
    assert set(matrix["unsupported"]) == {"macos", "ios-native", "headless-player"}
    support = (ROOT / "SUPPORT.md").read_text(encoding="utf-8")
    for field in ("commit", "desktop_ci", "player_ci"):
        assert matrix["evidence"][field] in support


def test_download_page_exposes_the_same_matrix_in_both_languages():
    page = (ROOT / "docs/download.html").read_text(encoding="utf-8")
    assert page.count('href="platform-support.json"') == 2
    assert page.count('SUPPORT.md#platform-support"') == 2


def test_readme_plugin_examples_use_repository_and_local_author_contracts():
    for filename in ("README.md", "README-zh.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        for token in ("package.py", "package/", "inx_package.json", "runtime/", "editor/", "plugin_pages/"):
            assert token in text, (filename, token)
        assert "InxPackage.json" not in text
        assert "InxPluginPages/" not in text
