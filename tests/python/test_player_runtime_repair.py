"""Exercise real package ownership using a small, controlled output recipe.

The recipe writes representative payload files; it does not claim to compile
desktop, Android or Web toolchains. Packages, disk IO and registry publication
use their production implementations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infernux.engine.build.contracts import BuildRequest
from infernux.engine.build_cancellation import BuildCancelled
from infernux.engine.player_runtime_repair import prepare_player_runtime
from infernux.plugins import InxPackage, PluginManager, PluginRegistry
from infernux.plugins.package import current_meta_bytes
from infernux.version import ENGINE_COMPATIBILITY_VERSION


REFERENCE = "test/generated-player"
PAYLOAD = "editor/player"


def _inspect(root):
    root = Path(root)
    if (root / "Player.inxmanifest").read_text(encoding="utf-8") != "compatible":
        raise RuntimeError("Player engine/target/Python ABI does not match")
    for name in ("Runtime.bin", "Generated.bin", "Transient.bin"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Player output is missing {name}")


def _authored_files(project):
    return {path.relative_to(project).as_posix(): path.read_bytes()
            for folder in ("Packages", "ProjectSettings")
            for path in (project / folder).rglob("*") if path.is_file()}


@pytest.fixture
def player_package(tmp_path, monkeypatch):
    from infernux.plugins.platform_releases import PLATFORM_PLAYERS

    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "package-cache"))
    monkeypatch.setenv("INFERNUX_ENGINE_SOURCE_ROOT", str(Path(__file__).resolve().parents[2]))
    project = tmp_path / "project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    source = tmp_path / "source"
    payload = source / PAYLOAD
    payload.mkdir(parents=True)
    (source / "inx_package.json").write_text(json.dumps({
        "reference": REFERENCE, "version": "1.0.0", "engine": "==" + ENGINE_COMPATIBILITY_VERSION,
    }), encoding="utf-8")
    for name, value in {"Player.inxmanifest": "stale", "Runtime.bin": "old-runtime", "Legacy.bin": "old-only"}.items():
        (payload / name).write_text(value, encoding="utf-8")
    (payload / "retired").mkdir()
    (payload / "retired/LegacyNested.bin").write_bytes(b"nested old member")
    outside = source / "runtime/outside.txt"
    outside.parent.mkdir()
    outside.write_text("outside payload", encoding="utf-8")
    archive = tmp_path / "original.inxpkg"
    InxPackage.export_source(str(source), str(archive))
    manager = PluginManager(str(project), runtime=True, installation_only=True)
    manager.install_package(str(archive), install_dependencies=False)
    installed = project / "Packages" / REFERENCE / PAYLOAD
    (installed / "user-note.txt").write_text("user-owned", encoding="utf-8")
    (installed / "user-note.txt.meta").write_bytes(current_meta_bytes("0123456789abcdef0123456789abcdef"))
    recipe = tmp_path / "recipe.py"
    recipe.write_text(
        "from pathlib import Path\nimport sys\n"
        "destination, counter, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]\n"
        "counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')\n"
        "destination.mkdir(parents=True)\n"
        "(destination / 'Player.inxmanifest').write_text('incompatible' if mode == 'invalid' else 'compatible')\n"
        "(destination / 'Runtime.bin').write_bytes(b'new-runtime')\n"
        "(destination / 'Generated.bin').write_bytes(b'generated')\n"
        "(destination / 'Transient.bin').write_bytes(b'transient')\n"
        "(destination / 'Runtime.bin').chmod(0o755)\n"
        "sys.exit(3 if mode == 'failed' else 0)\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(PLATFORM_PLAYERS, "test-generated", {
        "targets": ["test-generated-player"],
        "build_commands": [["{python}", "{recipe}", "{payload}", "{counter}", "{mode}"]],
    })
    request = BuildRequest(str(project), "test-generated-player", str(tmp_path / "output"))
    options = {"recipe": str(recipe), "counter": str(tmp_path / "recipe-count.txt"), "mode": "good"}
    try:
        yield manager, source, installed, request, options
    finally:
        manager.shutdown()


def test_rebuild_tracks_actual_outputs_reuses_them_then_updates_without_old_archive(player_package):
    manager, source, installed, request, options = player_package
    before = manager.registry.installed_record(REFERENCE)
    prior_guids = {item["logical_path"]: item["guid"] for item in before["files"]}
    assert prepare_player_runtime(request, installed, _inspect, options=options) == installed
    after = manager.registry.installed_record(REFERENCE)
    members = {item["logical_path"]: item for item in after["files"]}
    assert after["version"] == before["version"]
    assert after["control"] == before["control"]
    assert after["source"] == before["source"]
    assert after["transaction_id"] != before["transaction_id"]
    for logical in (PAYLOAD + "/Player.inxmanifest", PAYLOAD + "/Runtime.bin", "runtime/outside.txt"):
        assert members[logical]["guid"] == prior_guids[logical]
    for name in ("Generated.bin", "Transient.bin"):
        item = members[PAYLOAD + "/" + name]
        assert item["owned"] and item["role"] == "editor"
        assert item["guid"] in after["available_files"]
        meta = json.loads(Path(str(installed / name) + ".meta").read_bytes())
        assert meta["metadata"]["guid"]["value"] == item["guid"]
    assert PAYLOAD + "/Legacy.bin" not in members
    assert prior_guids[PAYLOAD + "/Legacy.bin"] not in after["available_files"]
    assert not (installed / "Legacy.bin").exists()
    assert not (installed / "Legacy.bin.meta").exists()
    assert not (installed / "retired").exists()
    assert (installed / "user-note.txt").read_text() == "user-owned"
    assert not any(item["logical_path"].endswith("user-note.txt") for item in after["files"])
    assert prepare_player_runtime(request, installed, _inspect, options=options) == installed
    assert Path(options["counter"]).read_text() == "1"
    assert manager.registry.installed_record(REFERENCE) == after

    # An update uses only the current ownership record and the new package.
    Path(manager._installed_archive_path(after)).unlink()
    manifest = source / "inx_package.json"
    metadata = json.loads(manifest.read_bytes())
    metadata["version"] = "2.0.0"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    (source / PAYLOAD / "Generated.bin").write_text("publisher replacement", encoding="utf-8")
    (source / PAYLOAD / "Legacy.bin").unlink()
    archive = source.parent / "next.inxpkg"
    InxPackage.export_source(str(source), str(archive))
    manager.install_package(str(archive), update=True, install_dependencies=False)
    updated = manager.registry.installed_record(REFERENCE)
    generated = next(item for item in updated["files"] if item["logical_path"] == PAYLOAD + "/Generated.bin")
    assert generated["guid"] == members[PAYLOAD + "/Generated.bin"]["guid"]
    assert (installed / "Generated.bin").read_text() == "publisher replacement"
    assert not (installed / "Transient.bin").exists()
    assert not (installed / "Transient.bin.meta").exists()
    assert (installed / "user-note.txt").read_text() == "user-owned"


def test_uninstall_removes_rebuilt_members_and_keeps_user_files(player_package):
    manager, _source, installed, request, options = player_package
    prepare_player_runtime(request, installed, _inspect, options=options)
    manager.uninstall(REFERENCE)
    assert manager.registry.installed_record(REFERENCE) is None
    assert sorted(path.name for path in installed.iterdir()) == ["user-note.txt", "user-note.txt.meta"]


@pytest.mark.parametrize("missing", ["sidecar", "guid"])
def test_rebuild_recovers_known_identity_from_installation_ledger(player_package, missing):
    manager, _source, installed, request, options = player_package
    before = manager.registry.installed_record(REFERENCE)
    member = next(item for item in before["files"] if item["logical_path"] == PAYLOAD + "/Runtime.bin")
    sidecar = installed / "Runtime.bin.meta"
    if missing == "sidecar":
        sidecar.unlink()
    else:
        sidecar.write_text('{"metadata": {}, "importer_settings": {"custom": true}}', encoding="utf-8")
    prepare_player_runtime(request, installed, _inspect, options=options)
    meta = json.loads(sidecar.read_bytes())
    assert meta["metadata"]["guid"]["value"] == member["guid"]
    if missing == "guid":
        assert meta["importer_settings"] == {"custom": True}


@pytest.mark.parametrize("collision", ["new-file", "changed-identity"])
def test_rebuild_rejects_user_occupancy_without_changing_files_or_ledger(player_package, collision):
    manager, _source, installed, request, options = player_package
    if collision == "new-file":
        (installed / "Generated.bin").write_bytes(b"user output")
    else:
        (installed / "Runtime.bin.meta").write_bytes(current_meta_bytes("abcdef0123456789abcdef0123456789"))
    before = _authored_files(Path(request.project_root))
    with pytest.raises(ValueError, match="occupied by another asset|identity changed"):
        prepare_player_runtime(request, installed, _inspect, options=options)
    assert _authored_files(Path(request.project_root)) == before


def test_rebuild_registry_publication_failure_restores_files_and_ownership(player_package, monkeypatch):
    _manager, _source, installed, request, options = player_package
    before = _authored_files(Path(request.project_root))
    original = PluginRegistry.save
    failed = False

    def save(registry, document):
        nonlocal failed
        original(registry, document)
        if not failed:
            failed = True
            raise OSError("generated member publication interrupted")

    monkeypatch.setattr(PluginRegistry, "save", save)
    with pytest.raises(OSError, match="publication interrupted"):
        prepare_player_runtime(request, installed, _inspect, options=options)
    assert _authored_files(Path(request.project_root)) == before
    assert not (installed / "Generated.bin").exists()


@pytest.mark.parametrize("member", ["Runtime.bin", "Legacy.bin"])
def test_rebuild_respects_shared_members_and_transfers_dropped_ownership(player_package, member):
    manager, source, installed, request, options = player_package
    before = manager.registry.installed_record(REFERENCE)
    original = next(item for item in before["files"] if item["logical_path"] == PAYLOAD + "/" + member)
    other = source.parent / "other-source"
    (other / "editor").mkdir(parents=True)
    (other / "inx_package.json").write_text(json.dumps({
        "reference": "test/shared-player-member", "version": "1.0.0",
    }), encoding="utf-8")
    (other / "editor/shared.bin").write_bytes((installed / member).read_bytes())
    (other / "editor/shared.bin.meta").write_bytes((installed / (member + ".meta")).read_bytes())
    archive = source.parent / "other.inxpkg"
    InxPackage.export_source(str(other), str(archive))
    manager.install_package(str(archive), install_dependencies=False)
    snapshot = _authored_files(Path(request.project_root))
    if member == "Runtime.bin":
        with pytest.raises(ValueError, match="Cannot replace a shared generated member"):
            prepare_player_runtime(request, installed, _inspect, options=options)
        assert _authored_files(Path(request.project_root)) == snapshot
    else:
        prepare_player_runtime(request, installed, _inspect, options=options)
        assert (installed / member).read_text() == "old-only"
        assert not any(item["guid"] == original["guid"] for item in manager.registry.installed_record(REFERENCE)["files"])
        other_record = manager.registry.installed_record("test/shared-player-member")
        assert next(item for item in other_record["files"] if item["guid"] == original["guid"])["owned"] is True
        manager.uninstall(REFERENCE)
        assert (installed / member).is_file()
        manager.uninstall("test/shared-player-member")
        assert not (installed / member).exists()


@pytest.mark.parametrize("failure", ["invalid", "failed", "cancelled"])
def test_failed_or_cancelled_recipe_never_publishes_partial_output(player_package, failure):
    _manager, _source, installed, request, options = player_package
    options["mode"] = failure
    before = _authored_files(Path(request.project_root))

    def inspect(root):
        _inspect(root)
        if failure == "cancelled" and Path(root) != installed:
            request.cancellation.cancel()

    with pytest.raises((RuntimeError, BuildCancelled)):
        prepare_player_runtime(request, installed, inspect, options=options)
    assert _authored_files(Path(request.project_root)) == before
    logs = list(Path(request.project_root, "Library/PlayerRuntime").rglob("rebuild.log"))
    assert len(logs) == 1 and logs[0].is_file()
    assert not list(logs[0].parent.glob("candidate-*"))


def test_a_manifest_without_an_installed_package_cannot_claim_generated_files(player_package):
    manager, _source, installed, request, options = player_package
    manager.registry.remove_install(REFERENCE)
    with pytest.raises(RuntimeError, match="Install the platform plugin"):
        prepare_player_runtime(request, installed, lambda _root: None, options=options)
    assert not Path(options["counter"]).exists()


def test_package_updated_during_compilation_cannot_receive_stale_generated_members(player_package):
    manager, _source, installed, request, options = player_package
    def inspect(root):
        _inspect(root)
        if Path(root) != installed:
            document = manager.registry.load()
            document["installed"][0]["transaction_id"] = "another-installation"
            manager.registry.save(document)

    with pytest.raises(RuntimeError, match="installation changed while rebuilding"):
        prepare_player_runtime(request, installed, inspect, options=options)
    assert (installed / "Runtime.bin").read_text() == "old-runtime"
    assert not (installed / "Generated.bin").exists()


@pytest.mark.parametrize("case", ["missing", "disabled", "editor-only", "engine", "invalid-range", "cycle", "compatible"])
def test_player_cook_checks_engine_and_required_runtime_package_closure(player_package, case):
    from infernux.engine.game_builder import GameBuilder
    from test_game_builder import _asset_index_entry

    manager, source, _installed, request, _options = player_package
    dependency = "test/player-dependency"
    project = Path(request.project_root)
    if case not in {"missing", "engine", "invalid-range"}:
        other = source.parent / "dependency-source"
        (other / "editor").mkdir(parents=True)
        (other / "inx_package.json").write_text(json.dumps({
            "reference": dependency, "version": "1.0.0", "engine": "==" + ENGINE_COMPATIBILITY_VERSION,
        }), encoding="utf-8")
        (other / "editor/forbidden.py").write_text("raise RuntimeError('Editor code cannot enter Player')\n", encoding="utf-8")
        if case != "editor-only":
            (other / "runtime").mkdir()
            (other / "runtime/data.txt").write_text("required Runtime content", encoding="utf-8")
        archive = source.parent / "dependency.inxpkg"
        InxPackage.export_source(str(other), str(archive))
        manager.install_package(str(archive), install_dependencies=False)
    document = manager.registry.load()
    records = {item["reference"]: item for item in document["installed"]}
    if case in {"engine", "invalid-range"}:
        records[REFERENCE]["engine"] = ">999" if case == "engine" else "bad range"
    else:
        records[REFERENCE]["dependencies"] = [dependency]
    if case == "disabled":
        records[dependency]["enabled"] = False
    if case == "cycle":
        records[dependency]["dependencies"] = [REFERENCE]
    manager.registry.save(document)
    before = manager.registry.load()
    entries = [_asset_index_entry(project, project / item["path_hint"], item["guid"], "", "Binary")
               for record in document["installed"] for item in record["files"]]
    output = source.parent / "cook"
    data = output / "Data"
    builder = GameBuilder(str(project), str(output), game_name="ContractPlayer")
    builder.freeze_asset_index_entries(entries, runtime_type_records=[])
    errors = {
        "missing": f"{REFERENCE} -> {dependency}: is not installed",
        "disabled": f"{REFERENCE} -> {dependency}: is disabled",
        "editor-only": f"{REFERENCE} -> {dependency}: has no exported Runtime content",
        "engine": f"{REFERENCE} requires Infernux >999",
        "invalid-range": f"{REFERENCE} has an invalid engine range: bad range",
        "cycle": "Circular Player plugin dependency",
    }
    if case == "compatible":
        builder._stage_player_plugins(str(data))
        shipped = json.loads((data / "ProjectSettings/InxPlugins.json").read_bytes())
        assert {item["reference"] for item in shipped["installed"]} == {REFERENCE, dependency}
        assert not list(data.rglob("forbidden.py"))
    else:
        with pytest.raises(RuntimeError, match=errors[case]):
            builder._stage_player_plugins(str(data))
        assert not (data / "ProjectSettings/InxPlugins.json").exists()
    assert manager.registry.load() == before


@pytest.mark.parametrize("field", ["engine_version", "engine_release", "python_abi", "platform", "machine"])
def test_desktop_player_validator_reports_the_actual_incompatible_identity(tmp_path, field):
    import platform
    import sys
    from infernux.engine.precompiled_player import inspect_desktop_runtime
    from infernux.plugins.platform_releases import PLAYER_SOURCE_REVISION
    from infernux.version import ENGINE_RELEASE, ENGINE_VERSION

    manifest = {"engine_version": ENGINE_VERSION, "engine_release": ENGINE_RELEASE,
                "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                "platform": sys.platform, "machine": platform.machine()}
    if PLAYER_SOURCE_REVISION is not None:
        manifest["engine_source_revision"] = PLAYER_SOURCE_REVISION
    for name in ("Runtime.inxrt", "InfernuxPlayerHost.exe" if sys.platform == "win32" else "InfernuxPlayerHost"):
        (tmp_path / name).write_bytes(b"validator-only input")
    path = tmp_path / "Player.inxmanifest"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert inspect_desktop_runtime(str(tmp_path)) == manifest
    manifest[field] = "wrong-identity"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match=field + ": 'wrong-identity'"):
        inspect_desktop_runtime(str(tmp_path))


@pytest.mark.parametrize("missing", [False, True])
def test_local_author_runtime_uses_the_install_requirement_parser(player_package, missing):
    from infernux.engine.game_builder import GameBuilder
    from test_game_builder import _asset_index_entry

    manager, source, _installed, request, _options = player_package
    project = Path(request.project_root)
    root = project / "Packages/local-author"
    (root / "runtime").mkdir(parents=True)
    (root / "inx_package.json").write_text(json.dumps({
        "reference": "future/local-author", "version": "1.0.0", "engine": "==" + ENGINE_COMPATIBILITY_VERSION,
    }), encoding="utf-8")
    runtime = root / "runtime/data.txt"
    runtime.write_text("local Runtime content", encoding="utf-8")
    guid = "6789abcdef0123456789abcdef012345"
    Path(str(runtime) + ".meta").write_bytes(current_meta_bytes(guid))
    dependency = "test/uninstalled-dependency" if missing else REFERENCE
    (root / "requirements.txt").write_text(f"inx:{dependency}\nrequests>=2\n", encoding="utf-8")
    before = manager.registry.load()
    entries = [_asset_index_entry(project, project / item["path_hint"], item["guid"], "", "Binary")
               for record in before["installed"] for item in record["files"]]
    entries.append(_asset_index_entry(project, runtime, guid, "", "Binary"))
    output = source.parent / "local-cook"
    data = output / "Data"
    builder = GameBuilder(str(project), str(output), game_name="LocalContractPlayer")
    builder.freeze_asset_index_entries(entries, runtime_type_records=[])
    if missing:
        with pytest.raises(RuntimeError, match=f"local-author -> {dependency}: is not installed"):
            builder._stage_player_plugins(str(data))
    else:
        builder._stage_player_plugins(str(data))
        document = json.loads((data / "ProjectSettings/InxPlugins.json").read_bytes())
        local = next(item for item in document["installed"] if item["reference"] == "local-author")
        assert local["dependencies"] == [REFERENCE]
        assert builder._player_dependency_constraints == ("requests>=2",)
    assert manager.registry.load() == before
    assert manager.registry.installed_record("local-author") is None
