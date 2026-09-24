from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "setup" / "build_web_toolchain.sh"
DAWN_LOCK = ROOT / "scripts" / "setup" / "dawn_dependencies.lock.json"


def test_web_toolchain_setup_pins_every_downloaded_source():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'EMSCRIPTEN_VERSION="4.0.10"' in source
    assert 'EMSDK_REVISION="e5bd3d0874e302a18f13c5b41f5bacf9a40c8e59"' in source
    assert 'CPYTHON_VERSION="3.13.15"' in source
    assert 'DAWN_REVISION="31e25af254ab572c77054edec4946d2244e184dd"' in source
    assert len(re.findall(r'^[A-Z_]+_SHA256="[0-9a-f]{64}"$', source, re.MULTILINE)) == 2
    assert "sha256sum --check --status" in source
    assert "--retry 5 --retry-all-errors" in source
    assert "max-filesize" not in source.casefold()
    assert "timeout" not in source.casefold()


def test_web_toolchain_setup_builds_webgpu_tools_without_gl_fallbacks():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "Tools/wasm/wasm_build.py emscripten-browser" in source
    assert 'cpython_em_config="$builds/cpython-emscripten-config.py"' in source
    assert 'f"EMSCRIPTEN_ROOT = {str(emscripten_root)!r}\\n"' in source
    assert 'f"BINARYEN_ROOT = {str(binaryen_root)!r}\\n"' in source
    assert 'f"NODE_JS = {str(node_path)!r}\\n"' in source
    assert 'export EM_CONFIG="$cpython_em_config"' in source
    assert "ac_cv_func_pthread_kill=no" in source
    assert 'cpython_wasm_config_changed=1' in source
    assert 'wasm_build.py emscripten-browser cleanall' in source
    assert "--target tint" in source
    assert "-DDAWN_ENABLE_DESKTOP_GL=OFF" in source
    assert "-DDAWN_ENABLE_OPENGLES=OFF" in source
    assert "infernux-web-toolchain.json" in source


def test_web_toolchain_rejects_incomplete_cached_dawn_dependencies():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'dawn_dependencies_complete()' in source
    assert 'dawn_dependencies.lock.json' in source
    assert 'hydrate_dawn_dependencies.py' in source
    assert '--verify-only' in source
    assert 'if ! dawn_dependencies_complete "$dawn_source"; then' in source
    assert "fetch_dawn_dependencies.py" not in source
    assert 'if ! dawn_dependencies_complete "$staging"; then' in source
    assert 'INFERNUX_DAWN_OFFICIAL_SNAPSHOT_DIR' in source
    assert 'mv -- "$dawn_source" "$quarantine_path"' in source
    assert 'rm -rf -- "$dawn_source"' not in source
    assert 'trap - EXIT' in source


def test_dawn_dependency_lock_pins_exact_sources_commits_and_trees():
    payload = json.loads(DAWN_LOCK.read_text(encoding="utf-8"))
    dependencies = payload["dependencies"]

    assert payload["schema"] == "infernux.dawn_dependencies"
    assert payload["dawn_revision"] == "31e25af254ab572c77054edec4946d2244e184dd"
    assert payload["dependency_count"] == len(dependencies) == 19
    assert payload["snapshot_dependency_count"] == 4
    assert len({dependency["path"] for dependency in dependencies}) == 19
    assert sum(
        dependency["transport"] == "official_gitiles_snapshot"
        for dependency in dependencies
    ) == 4
    for dependency in dependencies:
        assert re.fullmatch(r"[0-9a-f]{40}", dependency["commit"])
        assert re.fullmatch(r"[0-9a-f]{40}", dependency["tree"])
        if dependency["transport"] == "github":
            assert dependency["source_url"].startswith("https://github.com/")
            assert dependency["deps_url"].startswith(
                "https://chromium.googlesource.com/external/github.com/"
            )
        else:
            assert dependency["source_url"] == dependency["deps_url"]
            assert dependency["source_url"].startswith(
                "https://chromium.googlesource.com/chromium/src/third_party/"
            )
