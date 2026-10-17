from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from infernux.engine.build import BuildProfile, BuildRequest
from infernux.version import ENGINE_VERSION


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_EDITOR = ROOT / "external" / "plugins" / "infernux_web" / "package" / "editor"


def test_web_bootstrap_publishes_the_normal_runtime_components():
    subprocess.run(
        [sys.executable, str(ROOT / "tests/python/fixtures/web_runtime_namespace.py"), "components", "surface"],
        check=True, capture_output=True, text=True, timeout=30,
    )


def _web_python_runtime_manifest():
    return {
        "implementation": "cpython",
        "version": "3.13.15",
        "abi": "cp313",
        "platform": "emscripten",
        "platform_version": "4.0.10",
        "architecture": "wasm32",
        "extension_linkage": "static-built-in",
        "required_imports": ["numpy"],
        "packages": [
            {
                "name": "numpy",
                "version": "2.2.5",
                "imports": ["numpy"],
                "python_abi": "cp313",
                "platform": "emscripten",
                "platform_version": "4.0.10",
                "architecture": "wasm32",
                "linkage": "static-built-in",
                "source_url": "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.5.tar.gz",
                "source_sha256": "a9c0d994680cd991b1cb772e8b297340085466a6fe964bc9d4e80f5e2f43c291",
                "payload_hash_algorithm": "infernux-tree-sha256-v1",
                "payload_sha256": "1" * 64,
                "payload_bytes": 1,
                "payload_files": 1,
                "license": "BSD-3-Clause",
            }
        ],
    }


def _web_module(monkeypatch):
    monkeypatch.syspath_prepend(str(PLUGIN_EDITOR))
    for name in tuple(sys.modules):
        if name == "infernux_web" or name.startswith("infernux_web."):
            sys.modules.pop(name)
    return importlib.import_module("infernux_web")


def _request(tmp_path: Path) -> BuildRequest:
    return BuildRequest(
        str(tmp_path / "project"),
        "web-wasm32",
        str(tmp_path / "output"),
        BuildProfile(
            options={
                "display_mode": "fullscreen_borderless",
                "window_width": 1280,
                "window_height": 720,
                "window_resizable": False,
                "build_settings": {"scene_guids": []},
            }
        ),
    )


def test_web_payload_preparation_plan_and_successive_publication(monkeypatch, tmp_path):
    from dataclasses import replace

    module = _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    pipeline = importlib.import_module("infernux_web.shader_pipeline")
    from infernux.engine.player_package_native import read_entry

    targets = module.WebPlatformExporter().targets()

    assert [target.id for target in targets] == ["web-wasm32"]
    capabilities = targets[0].capabilities
    assert capabilities.graphics_api == "webgpu"
    assert not capabilities.threads
    assert not capabilities.dynamic_loading
    assert not capabilities.python_native_modules
    assert not capabilities.cpu_jit
    assert "gpu-jit" not in capabilities.features
    assert not capabilities.network
    assert capabilities.audio
    assert capabilities.text_input
    assert capabilities.gamepad_input
    assert not capabilities.persistent_storage
    assert {
        "dom-text-input",
        "gesture-gated-webaudio",
        "multi-pointer",
        "safe-area",
        "webgpu-capability-inventory",
    } <= capabilities.features

    root = _installed_web_payload(tmp_path)
    doctor = importlib.import_module("infernux_web.doctor")
    monkeypatch.setattr(doctor, "__file__", str(root / "doctor.py"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("INFERNUX_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("INFERNUX_BUILD_CACHE_ROOT", raising=False)
    report = module.inspect_web_toolchain("web-wasm32")
    assert report.available
    assert report.diagnostics == ()
    assert report.details["player"]["python_abi"] == "cp313"
    assert Path(report.details["tint"]).is_relative_to(root)
    request = _request(tmp_path)
    assert exporter._web_staging_directory(request) == (
        (tmp_path / "project").resolve() / "Cache/Build/WebHost/web-wasm32"
    )
    shared = tmp_path / "fast-build-disk"
    request = replace(
        request,
        profile=BuildProfile(
            options={
                **request.profile.options,
                "build_cache_root": str(shared),
            }
        ),
    )
    staging = exporter._web_staging_directory(request)
    assert staging == shared.resolve() / "WebHost/web-wasm32"
    plan = module.WebPlatformExporter().create_plan(request)
    assert [step.id for step in plan.steps] == ["cook", "shaders", "imports", "runtime", "package", "audit"]
    assert plan.metadata == {"architecture": "wasm32", "graphics_api": "webgpu", "python": "3.13"}
    obsolete = staging / "host-build/CMakeCache.txt"
    obsolete.parent.mkdir(parents=True)
    obsolete.write_text("cache", encoding="utf-8")
    for name in (".infernux-player-cook", "player-assets", "shader-cook"):
        path = staging / name
        path.mkdir()
        (path / "stale.txt").write_text("stale", encoding="utf-8")
    exporter._prepare_web_staging(staging)
    assert not obsolete.exists()
    assert not (staging / ".infernux-player-cook").exists()
    assert not (staging / "player-assets").exists()
    assert not (staging / "shader-cook").exists()
    assets = staging / "player-assets"
    assets.mkdir(parents=True)
    (assets / "包内文本.txt").write_text("button reads this content", encoding="utf-8")
    branding = assets / "web-branding"
    branding.mkdir()
    for name in (
        "infernux-favicon.png",
        "infernux-icon-192.png",
        "infernux-icon-512.png",
        "infernux.webmanifest",
        "infernux-branding.js",
    ):
        (branding / name).write_bytes(b"branding")
    exporter._stage_webgpu_capability_inventory(assets)
    (staging / "shader-cook").mkdir()
    compiler_calls = []

    def compile_shaders(manifest, output, **tools):
        compiler_calls.append(tools)
        output.mkdir()
        (output / "catalog.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline, "compile_shader_manifest", compile_shaders)
    details = report.details
    mode = "windowed"
    presentation = {"display_mode": mode, "window_width": 960, "window_height": 540}
    exporter._assemble_web_host(
        request,
        staging,
        assets,
        details,
        ROOT / "python/infernux",
        presentation,
        PLUGIN_EDITOR / "infernux_web/templates/host/shell.html",
        None,
    )
    artifacts, revision = exporter._publish_web_player(request, staging / "host-build", None)
    output = Path(request.output_dir)
    package = output / f"infernux-player.{revision}.inxpkg"
    assert read_entry(package, "包内文本.txt").decode() == "button reads this content"
    assert read_entry(package, "web-shaders/catalog.json") == b"{}"
    assert not (output / "player-assets").exists()
    assert not list(output.rglob("CMakeCache.txt"))
    for suffix in ("js", "wasm", "data"):
        assert (output / f"infernux-player.{revision}.{suffix}").read_bytes() == (
            root / "player" / f"infernux-runtime.{suffix}"
        ).read_bytes()
    html = (output / "infernux-player.html").read_text(encoding="utf-8")
    assert "Module.infernuxPresentation = " in html
    assert f'"mode": "{mode}"' in html
    assert f"fetch('infernux-player.{revision}.inxpkg')" in html
    assert "addRunDependency('infernux-project-content')" in html
    assert "Module.FS.writeFile" in html
    assert "Module.abort(error.message)" in html
    assert "@INFERNUX_WEB_" not in html
    assert "{{{ SCRIPT }}}" not in html
    assert len(compiler_calls) == 1
    assert all(Path(item.path).is_relative_to(output) for item in artifacts)

    first_revision = revision
    shutil.rmtree(staging / "host-build")
    shutil.rmtree(assets / "web-shaders")
    presentation["display_mode"] = "fullscreen_borderless"
    exporter._assemble_web_host(
        request,
        staging,
        assets,
        details,
        ROOT / "python/infernux",
        presentation,
        PLUGIN_EDITOR / "infernux_web/templates/host/shell.html",
        None,
    )
    artifacts, revision = exporter._publish_web_player(request, staging / "host-build", None)
    assert revision != first_revision
    html = (output / "infernux-player.html").read_text(encoding="utf-8")
    assert '"mode": "fullscreen_borderless"' in html
    assert f"fetch('infernux-player.{revision}.inxpkg')" in html
    assert (
        read_entry(output / f"infernux-player.{revision}.inxpkg", "包内文本.txt").decode()
        == "button reads this content"
    )
    assert (output / f"infernux-player.{revision}.wasm").read_bytes() == (
        root / "player/infernux-runtime.wasm"
    ).read_bytes()
    assert len(compiler_calls) == 2
    assert all(Path(item.path).is_relative_to(output) for item in artifacts)


def test_web_dependency_gate_accepts_public_cpu_jit_fixture(monkeypatch):
    _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    script = ROOT / "tests/fixtures/multiplatform_player/Assets/Scripts/Bootstrap.py"
    source = script.read_text(encoding="utf-8")
    assert "@inx.jit.compile" in source
    assert "inx.jit.warmup(" in source
    exporter._reject_unshipped_web_dependencies((script,))


def test_web_dependency_gate_accepts_compute_for_build_time_cpu_lowering(monkeypatch, tmp_path):
    _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    script = tmp_path / "compute.py"
    script.write_text(
        "import infernux as inx\n"
        "@inx.compute.function\n"
        "def helper(value):\n"
        "    return value\n"
        "@inx.compute.kernel\n"
        "def sample(values):\n"
        "    i = inx.compute.index(values)\n"
        "    values[i] = helper(values[i])\n",
        encoding="utf-8",
    )

    exporter._reject_unshipped_web_dependencies((script,))


def _installed_web_payload(tmp_path):
    root = tmp_path / "installed-plugin"
    player = root / "player"
    player.mkdir(parents=True)
    manifest = {
        "$schema": "infernux.web_player",
        "engine": ENGINE_VERSION,
        "platform": "web",
        "architecture": "wasm32",
        "python_abi": "cp313",
        "configuration": "Release",
        "python_runtime": _web_python_runtime_manifest(),
    }
    (player / "Player.inxmanifest").write_text(json.dumps(manifest), encoding="utf-8")
    for suffix in ("js", "wasm", "data"):
        (player / f"infernux-runtime.{suffix}").write_bytes(suffix.encode())
    for host, suffix in (("windows-x64", ".exe"), ("linux-x64", "")):
        tools = root / "tools" / host
        tools.mkdir(parents=True)
        for name in ("glslangValidator", "tint"):
            (tools / f"{name}{suffix}").write_bytes(b"native tool")
    return root


def test_web_publish_html_switch_failure_keeps_previous_generation_complete(monkeypatch, tmp_path):
    _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    capabilities = importlib.import_module("infernux_web.capabilities")
    request = _request(tmp_path)
    output = Path(request.output_dir)
    output.mkdir(parents=True)
    host = tmp_path / "host-build"
    host.mkdir()
    old_revision = "a" * 24
    new_revision = "b" * 24
    old_names = [
        "infernux-player.html",
        *(f"infernux-player.{old_revision}.{suffix}" for suffix in ("js", "wasm", "data", "inxpkg")),
    ]
    for name in old_names:
        (output / name).write_bytes(f"old:{name}".encode())
    html = (
        f"<script>const assetRevision = '{new_revision}';"
        "const paths = {'infernux-runtime.wasm': "
        "`infernux-player.${assetRevision}.wasm`, 'infernux-runtime.data': "
        "`infernux-player.${assetRevision}.data`};</script>"
        f'<script src="infernux-player.{new_revision}.js"></script>'
    )
    (host / "infernux-player.html").write_text(html, encoding="utf-8")
    for suffix in ("js", "wasm", "data", "inxpkg"):
        (host / f"infernux-player.{new_revision}.{suffix}").write_bytes(f"new:{suffix}".encode())
    for name in (
        "infernux-logo.png",
        "infernux-favicon.png",
        "infernux-icon-192.png",
        "infernux-icon-512.png",
        "infernux.webmanifest",
        "infernux-branding.js",
    ):
        (host / name).write_bytes(f"new:{name}".encode())
    (host / exporter.WEBGPU_CAPABILITY_FILENAME).write_text(
        json.dumps(capabilities.webgpu_capability_inventory()), encoding="utf-8"
    )
    original_replace = os.replace
    replaced = []

    def fail_html_switch(source, destination, *args, **kwargs):
        replaced.append(Path(destination).name)
        if Path(destination).name == "infernux-player.html":
            raise OSError("injected HTML switch failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(exporter.os, "replace", fail_html_switch)

    with pytest.raises(OSError, match="injected HTML switch failure"):
        exporter._publish_web_player(request, host, None)

    for name in old_names:
        assert (output / name).read_bytes() == f"old:{name}".encode()
    for suffix in ("js", "wasm", "data", "inxpkg"):
        assert (output / f"infernux-player.{new_revision}.{suffix}").is_file()
    assert replaced[-1] == "infernux-player.html"
    assert not list(output.glob(".*.tmp"))


@pytest.mark.parametrize(
    "had_template,failure",
    [
        (True, failure)
        for failure in (
            "template_copy",
            "template_scan",
            "template_publish",
            "html_switch",
            "interrupt",
            "none",
            "remove_template",
            "remove_template_failure",
            "rollback",
        )
    ]
    + [(False, "none"), (False, "html_switch")],
)
def test_web_custom_template_failure_preserves_previous_publication(monkeypatch, tmp_path, had_template, failure):
    _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    capabilities = importlib.import_module("infernux_web.capabilities")
    request = _request(tmp_path)
    output = Path(request.output_dir)
    output.mkdir(parents=True)
    host = tmp_path / "host"
    host.mkdir()
    old_revision, revision = "a" * 24, "b" * 24
    old_html = b"old publication"
    (output / "infernux-player.html").write_bytes(old_html)
    for suffix in ("js", "wasm", "data", "inxpkg"):
        (output / f"infernux-player.{old_revision}.{suffix}").write_bytes(b"old resource")
        (host / f"infernux-player.{revision}.{suffix}").write_bytes(b"new resource")
    html = (
        f"const assetRevision = '{revision}'; infernux-player.{revision}.js;"
        "'infernux-runtime.wasm': `infernux-player.${assetRevision}.wasm`,"
        "'infernux-runtime.data': `infernux-player.${assetRevision}.data`;"
        '<link href="web-template/theme/site.css">'
    )
    if failure.startswith("remove_template"):
        html = html.replace('<link href="web-template/theme/site.css">', "")
    (host / "infernux-player.html").write_text(html, encoding="utf-8")
    for name in (
        "infernux-logo.png",
        "infernux-favicon.png",
        "infernux-icon-192.png",
        "infernux-icon-512.png",
        "infernux.webmanifest",
        "infernux-branding.js",
    ):
        (host / name).write_bytes(b"branding")
    (host / exporter.WEBGPU_CAPABILITY_FILENAME).write_text(
        json.dumps(capabilities.webgpu_capability_inventory()), encoding="utf-8"
    )
    template = tmp_path / "custom-template"
    (template / "theme").mkdir(parents=True)
    (template / "shell.html").write_text("input shell", encoding="utf-8")
    (template / "theme/site.css").write_bytes(b"new css")
    published_template = output / "web-template"
    if had_template:
        (published_template / "theme").mkdir(parents=True)
        (published_template / "theme/site.css").write_bytes(b"old css")
    # This directory belongs to somebody else; publication must never sweep
    # generic staging names it did not allocate.
    foreign = output / ".web-template.tmp"
    foreign.mkdir()
    (foreign / "sentinel").write_bytes(b"unrelated")
    copy = exporter.shutil.copy2
    replace = exporter.os.replace
    scandir = exporter.os.scandir

    def fail_copy(source, destination, *args, **kwargs):
        if failure == "template_copy" and Path(source).name == "site.css":
            Path(destination).write_bytes(b"partial")
            raise OSError("owned template copy failure")
        return copy(source, destination, *args, **kwargs)

    def fail_replace(source, destination, *args, **kwargs):
        target, origin = Path(destination), Path(source)
        if failure == "template_publish" and target == published_template and origin.name != "previous":
            raise OSError("owned template publish failure")
        if (
            failure in {"html_switch", "rollback", "remove_template_failure"}
            and target == output / "infernux-player.html"
        ):
            raise OSError("owned html switch failure")
        if failure == "interrupt" and target == output / "infernux-player.html":
            raise KeyboardInterrupt("owned build interruption")
        if failure == "rollback" and origin.name == "previous":
            raise OSError("owned rollback failure")
        return replace(source, destination, *args, **kwargs)

    def fail_scan(path):
        # POSIX rmtree passes directory descriptors to the same os.scandir.
        # Inject only the authored-path failure, preserving real cleanup.
        if failure == "template_scan" and not isinstance(path, int) and Path(path) == template / "theme":
            raise PermissionError("owned template scan failure")
        return scandir(path)

    monkeypatch.setattr(exporter.shutil, "copy2", fail_copy)
    monkeypatch.setattr(exporter.os, "replace", fail_replace)
    if failure == "template_scan":
        monkeypatch.setattr(exporter.os, "scandir", fail_scan)
    selected_template = None if failure.startswith("remove_template") else template
    if failure in {"none", "remove_template"}:
        artifacts, actual_revision = exporter._publish_web_player(request, host, selected_template)
        assert actual_revision == revision
        assert all(Path(artifact.path).is_file() for artifact in artifacts)
        assert (output / "infernux-player.html").read_text(encoding="utf-8") == html
        if failure == "none":
            assert (published_template / "theme/site.css").read_bytes() == b"new css"
            assert not (published_template / "shell.html").exists()
        else:
            assert not published_template.exists()
        assert not list(output.glob(f"infernux-player.{old_revision}.*"))
        assert (foreign / "sentinel").read_bytes() == b"unrelated"
        assert not [path for path in output.glob(".web-template.*") if path != foreign]
        return
    error_type = {"rollback": RuntimeError, "interrupt": KeyboardInterrupt}.get(failure, OSError)
    error_message = "retained template workspace" if failure == "rollback" else "owned"
    with pytest.raises(error_type, match=error_message):
        exporter._publish_web_player(request, host, selected_template)
    assert (output / "infernux-player.html").read_bytes() == old_html
    for suffix in ("js", "wasm", "data", "inxpkg"):
        assert (output / f"infernux-player.{old_revision}.{suffix}").read_bytes() == b"old resource"
    if failure == "rollback":
        workspaces = [path for path in output.glob(".web-template.*") if path != foreign]
        assert len(workspaces) == 1
        assert (workspaces[0] / "previous/theme/site.css").read_bytes() == b"old css"
        assert (foreign / "sentinel").read_bytes() == b"unrelated"
        return
    if had_template:
        assert (published_template / "theme/site.css").read_bytes() == b"old css"
    else:
        assert not published_template.exists()
    assert (foreign / "sentinel").read_bytes() == b"unrelated"
    assert not [path for path in output.glob(".web-template.*") if path != foreign]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows template junction boundaries")
@pytest.mark.parametrize("placement", ["root", "child"])
def test_web_template_staging_rejects_junctions_without_copying_external_files(monkeypatch, tmp_path, placement):
    import _winapi

    _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    source = tmp_path / "source"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "private.txt"
    sentinel.write_bytes(b"external content")
    if placement == "child":
        source.mkdir()
    junction = source if placement == "root" else source / "linked"
    _winapi.CreateJunction(str(outside), str(junction))
    try:
        with pytest.raises(RuntimeError, match="link.*junction"):
            exporter._stage_project_web_template(source, tmp_path / "staged")
        assert sentinel.read_bytes() == b"external content"
        assert not list((tmp_path / "staged").rglob("private.txt"))
    finally:
        if os.path.isjunction(junction):
            os.rmdir(junction)


def test_web_publish_copy_failure_keeps_previous_generation_complete(monkeypatch, tmp_path):
    _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    capabilities = importlib.import_module("infernux_web.capabilities")
    request = _request(tmp_path)
    output = Path(request.output_dir)
    output.mkdir(parents=True)
    host = tmp_path / "host-build"
    host.mkdir()
    old_revision = "c" * 24
    new_revision = "d" * 24
    old_names = [
        "infernux-player.html",
        *(f"infernux-player.{old_revision}.{suffix}" for suffix in ("js", "wasm", "data", "inxpkg")),
    ]
    old_bytes = {}
    for name in old_names:
        old_bytes[name] = f"old:{name}".encode()
        (output / name).write_bytes(old_bytes[name])
    html = (
        f"<script>const assetRevision = '{new_revision}';"
        "const paths = {'infernux-runtime.wasm': "
        "`infernux-player.${assetRevision}.wasm`, 'infernux-runtime.data': "
        "`infernux-player.${assetRevision}.data`};</script>"
        f'<script src="infernux-player.{new_revision}.js"></script>'
    )
    (host / "infernux-player.html").write_text(html, encoding="utf-8")
    for suffix in ("js", "wasm", "data", "inxpkg"):
        (host / f"infernux-player.{new_revision}.{suffix}").write_bytes(f"new:{suffix}".encode())
    for name in (
        "infernux-logo.png",
        "infernux-favicon.png",
        "infernux-icon-192.png",
        "infernux-icon-512.png",
        "infernux.webmanifest",
        "infernux-branding.js",
    ):
        (host / name).write_bytes(f"new:{name}".encode())
    (host / exporter.WEBGPU_CAPABILITY_FILENAME).write_text(
        json.dumps(capabilities.webgpu_capability_inventory()), encoding="utf-8"
    )
    original_copy2 = exporter.shutil.copy2

    def fail_resource_copy(source, destination, *args, **kwargs):
        if Path(source).name == f"infernux-player.{new_revision}.wasm":
            Path(destination).write_bytes(b"partial")
            raise OSError("injected resource copy failure")
        return original_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(exporter.shutil, "copy2", fail_resource_copy)

    with pytest.raises(OSError, match="injected resource copy failure"):
        exporter._publish_web_player(request, host, None)

    for name, expected in old_bytes.items():
        assert (output / name).read_bytes() == expected
    assert not list(output.glob(f"infernux-player.{new_revision}.*"))
    assert not list(output.glob(".*.tmp"))


def test_web_export_publishes_versioned_cooked_player(monkeypatch, tmp_path):
    module = _web_module(monkeypatch)
    exporter_module = importlib.import_module("infernux_web.exporter")
    exporter = module.WebPlatformExporter()
    monkeypatch.setattr(
        exporter,
        "doctor",
        lambda _request: importlib.import_module("infernux.engine.build").CapabilityReport(
            True,
            details={
                "distribution": "fixture",
                "emsdk_root": "/fixture/emsdk",
                "cpython_root": "/fixture/cpython",
                "source_root": str(ROOT),
                "player": {"python_runtime": _web_python_runtime_manifest()},
            },
        ),
    )
    request = _request(tmp_path)
    project_template = Path(request.project_root) / "ProjectSettings" / "WebTemplate"
    project_template.mkdir(parents=True)
    (project_template / "shell.html").write_text(
        "\n".join(
            (
                '<canvas id="canvas"></canvas>',
                "@INFERNUX_WEB_ASSET_REVISION@",
                "@INFERNUX_WEB_DISPLAY_MODE@",
                "@INFERNUX_WEB_CANVAS_WIDTH@",
                "@INFERNUX_WEB_CANVAS_HEIGHT@",
                "{{{ SCRIPT }}}",
            )
        ),
        encoding="utf-8",
    )
    (project_template / "theme").mkdir()
    (project_template / "theme" / "site.css").write_text(
        "body { background: black; }\n",
        encoding="utf-8",
    )
    player_assets = tmp_path / "player-assets"
    player_assets.mkdir()
    monkeypatch.setattr(
        exporter_module,
        "_cook_web_player_assets",
        lambda _request, _staging: ("Balance", player_assets, ()),
    )
    monkeypatch.setattr(
        exporter_module,
        "_stage_engine_python_package",
        lambda _request, _assets, _source_root: None,
    )
    monkeypatch.setattr(
        exporter_module,
        "_stage_web_branding",
        lambda _assets, _source_root, _game_name: None,
    )
    monkeypatch.setattr(
        exporter_module,
        "_stage_web_shader_sources",
        lambda _request, _staging, _assets, _source_root: None,
    )
    revision = "1234567890abcdef12345678"

    captured_presentation = {}
    captured_template = {}

    def _build_host(
        _request,
        staging,
        _assets,
        _details,
        _source_root,
        presentation,
        web_shell,
        web_template,
    ):
        captured_presentation.update(presentation)
        captured_template.update(shell=web_shell, root=web_template)
        host_build = staging / "host-build"
        host_build.mkdir(parents=True)
        (host_build / "infernux-player.html").write_text(
            "\n".join(
                (
                    f"const assetRevision = '{revision}';",
                    f"infernux-player.{revision}.js",
                    "'infernux-runtime.wasm': `infernux-player.${assetRevision}.wasm`,",
                    "'infernux-runtime.data': `infernux-player.${assetRevision}.data`,",
                )
            ),
            encoding="utf-8",
        )
        for suffix in ("js", "wasm", "data", "inxpkg"):
            (host_build / f"infernux-player.{revision}.{suffix}").write_bytes(suffix.encode("ascii"))
        (host_build / "infernux-logo.png").write_bytes(b"png")
        for name in (
            "infernux-favicon.png",
            "infernux-icon-192.png",
            "infernux-icon-512.png",
            "infernux.webmanifest",
            "infernux-branding.js",
        ):
            (host_build / name).write_bytes(name.encode("ascii"))
        shutil.copy2(
            player_assets / "infernux-webgpu-capabilities.json",
            host_build / "infernux-webgpu-capabilities.json",
        )
        return ("host built",)

    monkeypatch.setattr(exporter_module, "_assemble_web_host", _build_host)

    result = exporter.execute(request, exporter.create_plan(request))

    assert result.success
    assert result.diagnostics == ()
    assert [item.kind for item in result.artifacts] == [
        "html",
        "js",
        "wasm",
        "data",
        "inxpkg",
        "png",
        "png",
        "png",
        "png",
        "webmanifest",
        "js",
        "json",
        "css",
    ]
    assert all(Path(item.path).is_file() for item in result.artifacts)
    assert result.manifest["scope"] == "cooked-player"
    assert result.manifest["game"] == "Balance"
    assert result.manifest["asset_revision"] == revision
    assert result.manifest["entry_point"] == "infernux-player.html"
    assert result.manifest["python_runtime"] == _web_python_runtime_manifest()
    assert result.manifest["web_template"] == "project"
    assert result.manifest["presentation"] == {
        "display_mode": "fullscreen_borderless",
        "window_width": 1280,
        "window_height": 720,
    }
    capability_manifest = result.manifest["webgpu_capabilities"]
    assert capability_manifest["path"] == "infernux-webgpu-capabilities.json"
    assert capability_manifest["open_parity_gates"] == []
    published_capabilities = json.loads(
        (Path(request.output_dir) / "infernux-webgpu-capabilities.json").read_text(encoding="utf-8")
    )
    assert published_capabilities["backend"] == "webgpu"
    assert captured_presentation == result.manifest["presentation"]
    assert captured_template == {
        "shell": (project_template / "shell.html").resolve(),
        "root": project_template.resolve(),
    }
    assert (Path(request.output_dir) / "web-template" / "theme" / "site.css").read_text(
        encoding="utf-8"
    ) == "body { background: black; }\n"
    assert result.logs == ("host built",)


def test_web_scene_shader_passes_pinned_tint_validation():
    renderer = (ROOT / "external" / "plugins" / "infernux_web" / "native" / "WebSceneRenderer.cpp").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r'constexpr char kSceneShader\[\] = R"wgsl\((.*?)\)wgsl";',
        renderer,
        re.DOTALL,
    )
    assert match is not None
    platform_name = "windows-x64" if sys.platform == "win32" else "linux-x64"
    executable = "tint.exe" if sys.platform == "win32" else "tint"
    configured = os.environ.get("INFERNUX_TEST_TINT")
    tint = Path(configured) if configured else PLUGIN_EDITOR / "infernux_web" / "tools" / platform_name / executable
    if configured:
        assert tint.is_file(), f"Configured pinned Tint does not exist: {tint}"
    if not tint.is_file():
        pytest.skip(f"pinned Tint is unavailable for {platform_name}; set INFERNUX_TEST_TINT to the built tool")
    with tempfile.TemporaryDirectory(prefix="web-wgsl-", dir=ROOT / "out") as temp_dir:
        shader = Path(temp_dir) / "WebSceneRenderer.wgsl"
        shader.write_text(match.group(1), encoding="utf-8")
        completed = subprocess.run(
            [str(tint), "--input-format", "wgsl", "--format", "wgsl", str(shader)],
            capture_output=True,
            text=True,
            check=False,
        )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_web_shader_stage_deduplicates_shared_particle_kernel(monkeypatch, tmp_path):
    _web_module(monkeypatch)
    exporter = importlib.import_module("infernux_web.exporter")
    native = importlib.import_module("infernux.lib")._Infernux
    monkeypatch.setattr(
        native,
        "_prepare_authored_shader_glsl",
        lambda source, _path: source,
    )
    request = _request(tmp_path)
    source_root = tmp_path / "source"
    shader_dir = source_root / "resources" / "shaders"
    shader_dir.mkdir(parents=True)
    (shader_dir / "fullscreen_triangle.vert").write_text("#version 450\nvoid main() {}\n", encoding="utf-8")
    player_assets = tmp_path / "player-assets"
    (player_assets / "Balance_Data").mkdir(parents=True)
    # Shader staging consumes the actual Player catalog even when this
    # fixture has no UI materials. Keep its package boundary valid.
    from infernux.engine.player_package_native import write_pack

    runtime_catalog = tmp_path / "RuntimeAssetCatalog.json"
    asset_records = tmp_path / "RuntimeAssetRecords.json"
    asset_records.write_text(json.dumps({"entries": []}), encoding="utf-8")
    record_path = "Library/RuntimeAssetRecords.json"
    runtime_catalog.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "runtime_artifact_id": "records",
                        "runtime_path": record_path,
                        "package": "Balance_Data/Content.inxpkg",
                        "dependencies": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    write_pack(((record_path, asset_records),), player_assets / "Balance_Data" / "Content.inxpkg")
    write_pack((("RuntimeAssetCatalog.json", runtime_catalog),), player_assets / "Balance_Data" / "AssetCatalog.inxcat")
    particle_dir = Path(request.project_root) / "Library" / "Artifacts" / "Particle"
    particle_dir.mkdir(parents=True)
    kernel_hash = "a" * 64
    stage_names = {
        "bootstrap",
        "init",
        "update",
        "update_rendering_fused",
        "contact_prepare",
        "contact_solve",
        "contact_dispatch",
        "render_reset",
        "rendering",
    }
    emitters = []
    for stable_id in ("emitter-b", "emitter-a"):
        emitters.append(
            {
                "stable_id": stable_id,
                "kernel_hash": kernel_hash,
                "state_stride": 80,
                "event_type_count": 0,
                "collision_enabled": False,
                "continuation": None,
                "update_render_fusion": {"eligible": True},
                "data_interface_layout": {
                    "mesh_interfaces": [],
                    "texture2d_parameters": [],
                    "volume_interfaces": [],
                },
                "stages": {name: "#version 450\nlayout(local_size_x=1) in; void main() {}\n" for name in stage_names},
            }
        )
    (particle_dir / "shared.inxparticle").write_text(json.dumps({"gpu_glsl": {"emitters": emitters}}), encoding="utf-8")

    exporter._stage_web_shader_sources(request, tmp_path / "staging", player_assets, source_root)

    catalog = json.loads((player_assets / "web-particles" / "catalog.json").read_text(encoding="utf-8"))
    assert len(catalog["kernels"]) == 1
    assert catalog["kernels"][0]["stable_ids"] == ["emitter-a", "emitter-b"]
    assert len(catalog["kernels"][0]["stages"]) == len(stage_names)


def test_web_asset_revision_covers_content_runtime_and_shader_inputs(monkeypatch, tmp_path):
    _web_module(monkeypatch)
    exporter_module = importlib.import_module("infernux_web.exporter")
    staging = tmp_path / "staging"
    player = staging / "player-assets"
    runtime = tmp_path / "runtime"
    shaders = staging / "shader-cook"
    for root in (player, runtime, shaders):
        root.mkdir(parents=True)
    (player / "Content.inxpkg").write_bytes(b"content")
    (runtime / "main.cpp").write_bytes(b"runtime")
    (shaders / "manifest.json").write_bytes(b"shader")
    project_template = tmp_path / "project-template"
    project_template.mkdir()
    (project_template / "shell.html").write_bytes(b"custom shell")

    fullscreen = {
        "display_mode": "fullscreen_borderless",
        "window_width": 1280,
        "window_height": 720,
    }
    first = exporter_module._web_asset_revision(staging, player, runtime, presentation=fullscreen)
    (runtime / "main.cpp").write_bytes(b"runtime changed")
    second = exporter_module._web_asset_revision(staging, player, runtime, presentation=fullscreen)
    bytecode = player / "python" / "site-packages" / "infernux" / "__pycache__"
    bytecode.mkdir(parents=True)
    (bytecode / "runtime.cpython-313.opt-1.pyc").write_bytes(b"compiled")
    third = exporter_module._web_asset_revision(staging, player, runtime, presentation=fullscreen)
    windowed = exporter_module._web_asset_revision(
        staging,
        player,
        runtime,
        presentation={**fullscreen, "display_mode": "windowed"},
    )
    custom = exporter_module._web_asset_revision(
        staging,
        player,
        runtime,
        presentation=fullscreen,
        project_template_root=project_template,
    )
    (project_template / "shell.html").write_bytes(b"custom shell changed")
    custom_changed = exporter_module._web_asset_revision(
        staging,
        player,
        runtime,
        presentation=fullscreen,
        project_template_root=project_template,
    )

    assert re.fullmatch(r"[0-9a-f]{24}", first)
    assert first != second
    assert second != third
    assert third != windowed
    assert custom != custom_changed
