"""Keep host metadata and translated public support claims on one matrix."""
from __future__ import annotations

import json
import ast
from pathlib import Path
import re
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
    availability = {"yes": True, "有": True, "✅": True,
                    "no": False, "无": False, "—": False, "-": False}
    for filename in ("README.md", "README-zh.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "SUPPORT.md#platform-support" in text
        rows = [tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
                for line in text.splitlines() if line.lstrip().startswith("|")]
        for item in matrix["platforms"]:
            label = "".join(item["label"].split()).casefold()
            matches = [row for row in rows if "".join(row[0].split()).casefold() == label]
            assert len(matches) == 1, (filename, item["id"])
            _, editor, player, graphics = matches[0]
            assert availability[editor.casefold()] == item["editor"]
            assert graphics == item["graphics"]
            if item["player"] == "Yes":
                assert availability[player.casefold()]
            else:
                # A short introduction may name only the main delivery formats.
                # Reject unsupported claims without requiring identical prose.
                advertised = {value.strip() for value in re.split(r"[/+]", player)}
                assert advertised and advertised <= set(item["player"].split("/")), (
                    filename, item["id"], player
                )


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


def test_download_page_links_to_platform_support_sources():
    page = (ROOT / "docs/download.html").read_text(encoding="utf-8")
    assert 'href="platform-support.json"' in page
    assert 'SUPPORT.md#platform-support"' in page


def test_readmes_link_to_current_plugin_authoring_contracts():
    for filename, language in (("README.md", "en"), ("README-zh.md", "zh")):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert f"https://infernux-engine.com/wiki/site/{language}/plugin-package-content.html" in text
        assert "https://github.com/InfernuxEngine/infernux_plugin_template" in text
        guide = (ROOT / "docs/wiki/docs" / language / "plugin-package-content.md").read_text(
            encoding="utf-8"
        )
        for token in ("package.py", "package/", "inx_package.json", "runtime/", "editor/", "plugin_pages/"):
            assert token in guide, (language, token)
        assert "InxPackage.json" not in guide
        assert "InxPluginPages/" not in guide
