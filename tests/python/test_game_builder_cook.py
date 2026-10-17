from __future__ import annotations

from test_game_builder import (
    NuitkaBuilder,
    ParticleGraphAsset,
    Path,
    SimpleNamespace,
    _asset_index_entry,
    _make_builder,
    _make_license_distribution,
    _reference_particle_graph,
    _stage_document_inputs,
    _write_asset_index,
    _write_particle_artifact,
    _write_player_executable,
    _write_player_native_contract,
    game_builder_module,
    hashlib,
    json,
    nuitka_builder_module,
    pytest,
    read_entry,
    read_manifest,
    sys,
    _native_package_backend as _native_package_backend,
)


def test_raw_runtime_package_sources_are_replaced_with_adjacent_bytecode(tmp_path):
    package_root = tmp_path / "numba"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    nested = package_root / "nested.py"
    nested.write_text("def value():\n    return 2\n", encoding="utf-8")

    NuitkaBuilder._compile_raw_python_sources(package_root)

    assert not (package_root / "__init__.py").exists()
    assert not nested.exists()
    assert (package_root / "__init__.pyc").is_file()
    assert (package_root / "nested.pyc").is_file()


def test_raw_package_injection_retains_external_wheel_licenses(tmp_path, monkeypatch):
    site = tmp_path / "site"
    notice, relative = _make_license_distribution(site)
    builder = object.__new__(NuitkaBuilder)
    builder._builder_python = sys.executable
    builder.raw_copy_packages = ["llvmlite"]
    monkeypatch.setattr(
        nuitka_builder_module, "_run_python", lambda *a, **kw: SimpleNamespace(stdout=json.dumps([str(site)]))
    )
    dist = tmp_path / "dist"

    builder._inject_jit_packages(str(dist))

    assert (dist / "llvmlite/__init__.pyc").is_file()
    assert not (dist / "llvmlite/__init__.py").exists()
    assert (dist / relative).read_bytes() == notice.read_bytes()


def test_missing_recorded_license_fails_publication(tmp_path):
    site = tmp_path / "site"
    notice, _ = _make_license_distribution(site)
    notice.unlink()
    with pytest.raises(FileNotFoundError):
        NuitkaBuilder._copy_raw_dependency_licenses(site, tmp_path / "dist", ["llvmlite"])


def test_packaged_parallel_runtime_module_round_trip(tmp_path, monkeypatch):
    module_root = tmp_path / "wheel" / "_runtime_modules"
    builder = object.__new__(NuitkaBuilder)
    builder.last_runtime_compatibility_key = "b" * 64
    builder._engine_fingerprint_cache = "engine-content"

    def fake_inject(dist_dir, packages=None):
        for package in packages or []:
            package_dir = Path(dist_dir) / package
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "runtime.pyc").write_bytes(package.encode("utf-8"))
            (package_dir / "runtime.pyi").write_text(package, encoding="utf-8")
            (package_dir / "runtime.c").write_text(package, encoding="utf-8")
            (package_dir / "runtime.h").write_text(package, encoding="utf-8")
            license_dir = package_dir / "_licenses" / package
            license_dir.mkdir(parents=True)
            (license_dir / "LICENSE").write_text(f"Copyright {package}", encoding="utf-8")

    monkeypatch.setattr(builder, "_inject_jit_packages", fake_inject)
    exported = builder.export_runtime_module(str(module_root))
    monkeypatch.setenv("INFERNUX_PREBUILT_RUNTIME_MODULE_DIR", str(module_root))

    dist = tmp_path / "dist"
    dist.mkdir()
    assert builder.install_runtime_module(str(dist)) is True
    assert (dist / "numba" / "runtime.pyc").read_bytes() == b"numba"
    assert (dist / "llvmlite" / "runtime.pyc").read_bytes() == b"llvmlite"
    assert (dist / "llvmlite/_licenses/llvmlite/LICENSE").read_text() == "Copyright llvmlite"
    manifest = json.loads((Path(exported) / "Player.inxmanifest").read_text(encoding="utf-8"))
    assert manifest["archive"] == "Parallel.inxmod"
    assert manifest["compression"] == "store"
    assert manifest["compression_profile"] == "development"
    assert manifest["packages"] == ["llvmlite", "numba"]
    module_manifest = read_manifest(Path(exported) / "Parallel.inxmod")
    assert all(not entry["path"].endswith((".pyi", ".c", ".h")) for entry in module_manifest["files"])

    # Re-exporting the same compatibility payload is idempotent. This is the
    # normal CMake retry/concurrent-build case on Windows and must not require
    # replacing an already published directory.
    assert builder.export_runtime_module(str(module_root)) == exported

    compressed_dist = tmp_path / "compressed-dist"
    compressed_dist.mkdir()
    assert builder.install_runtime_module(str(compressed_dist), archive_only=True) is True
    staged = compressed_dist / "Parallel.inxmod"
    assert staged.read_bytes() == (Path(exported) / "Parallel.inxmod").read_bytes()
    assert not (compressed_dist / "numba").exists()
    assert not (compressed_dist / "llvmlite").exists()

    release_dist = tmp_path / "release-dist"
    release_dist.mkdir()
    assert (
        builder.install_runtime_module(
            str(release_dist),
            archive_only=True,
            profile="release",
        )
        is True
    )
    assert read_manifest(release_dist / "Parallel.inxmod")["compression_profile"] == "release"
    assert (
        read_entry(
            release_dist / "Parallel.inxmod",
            "llvmlite/_licenses/llvmlite/LICENSE",
        )
        == b"Copyright llvmlite"
    )


def test_native_payload_injection_uses_one_override_and_overwrites_stale_files(tmp_path, monkeypatch):
    native_root = tmp_path / "native"
    native_root.mkdir()
    native_module = "_Infernux.cp313-win_amd64.pyd" if sys.platform == "win32" else "_Infernux.so"
    companion = "InfernuxRendererRuntime.dll" if sys.platform == "win32" else "libInfernuxRendererRuntime.so"
    (native_root / native_module).write_bytes(b"current-module")
    bootstrap_module = "_InfernuxBootstrap.cp313-win_amd64.pyd" if sys.platform == "win32" else "_InfernuxBootstrap.so"
    (native_root / bootstrap_module).write_bytes(b"bootstrap-module")
    _write_player_native_contract(native_root)
    if sys.platform == "win32":
        # A stale short-name module can be left by Nuitka discovery. The
        # ABI-tagged build output must win and replace it atomically.
        (native_root / "_Infernux.pyd").write_bytes(b"stale-source-module")
    (native_root / companion).write_bytes(b"current-runtime")
    foundation = (
        native_root / "InfernuxFoundation.dll" if sys.platform == "win32" else native_root / "libInfernuxFoundation.so"
    )
    foundation.write_bytes(b"foundation")
    python_runtime = native_root / "python313.dll" if sys.platform == "win32" else None
    if python_runtime is not None:
        python_runtime.write_bytes(b"python-runtime")
        (native_root / "zlib.dll").write_bytes(b"runtime-zlib")
    monkeypatch.setenv("INFERNUX_NATIVE_MODULE_DIR", str(native_root))

    dist = tmp_path / "boot.dist"
    package_lib = dist / "infernux" / "lib"
    package_lib.mkdir(parents=True)
    (package_lib / native_module).write_bytes(b"stale-module")
    canonical_module = "_Infernux.pyd" if sys.platform == "win32" else "_Infernux.so"
    (package_lib / canonical_module).write_bytes(b"stale-canonical-module")
    (package_lib / companion).write_bytes(b"stale-runtime")
    if sys.platform == "win32":
        (dist / companion).write_bytes(b"stale-root-runtime")

    builder = object.__new__(NuitkaBuilder)
    builder._inject_native_libs(str(dist))

    assert (package_lib / native_module).read_bytes() == b"current-module"
    assert (package_lib / canonical_module).read_bytes() == b"current-module"
    assert (package_lib / companion).read_bytes() == b"current-runtime"
    if sys.platform == "win32":
        assert not (dist / companion).exists()
        assert (dist / "_InfernuxBootstrap.pyd").read_bytes() == b"bootstrap-module"
        assert (dist / "InfernuxFoundation.dll").read_bytes() == b"foundation"
        assert (dist / "python313.dll").read_bytes() == b"python-runtime"
        assert not (package_lib / "python313.dll").exists()
        assert (package_lib / "zlib.dll").read_bytes() == b"runtime-zlib"
        assert not (dist / "zlib.dll").exists()


def test_player_native_payload_requires_complete_current_closure(tmp_path, monkeypatch):
    native_root = tmp_path / "native"
    native_root.mkdir()
    native_module = "_Infernux.cp313-win_amd64.pyd" if sys.platform == "win32" else "_Infernux.so"
    bootstrap_module = "_InfernuxBootstrap.cp313-win_amd64.pyd" if sys.platform == "win32" else "_InfernuxBootstrap.so"
    (native_root / native_module).write_bytes(b"player-module")
    (native_root / bootstrap_module).write_bytes(b"bootstrap-module")
    _write_player_native_contract(native_root)
    missing = sorted(nuitka_builder_module.player_native_library_filenames())[0]
    (native_root / missing).unlink()
    monkeypatch.setenv("INFERNUX_NATIVE_MODULE_DIR", str(native_root))

    with pytest.raises(RuntimeError, match="static Release Player runtime"):
        NuitkaBuilder._native_payload_dir()


def test_player_output_transaction_can_abort_and_retry_without_replacing_the_previous_product(tmp_path):
    output = tmp_path / "已发布游戏"
    output.mkdir()
    previous = output / "TestGame.exe"
    previous.write_bytes(b"previous")
    builder = _make_builder(tmp_path, output)

    builder._begin_output_transaction()
    staging = Path(builder.output_dir)
    (staging / "partial.bin").write_bytes(b"partial")
    builder._abort_output_transaction()
    assert previous.read_bytes() == b"previous"
    assert not staging.exists()
    assert builder.output_dir == str(output.resolve())
    assert not Path(str(output) + ".infernux-build.lock").exists()

    builder._begin_output_transaction()
    staging = Path(builder.output_dir)
    assert previous.read_bytes() == b"previous"
    (staging / "TestGame.exe").write_bytes(b"current")
    (staging / "TestGame_Data").mkdir()

    published = Path(builder._commit_output_transaction(str(staging)))

    assert published == output.resolve()
    assert (published / "TestGame.exe").read_bytes() == b"current"
    assert not Path(str(output) + ".infernux-build.lock").exists()


def test_player_output_transaction_restores_previous_product_on_publish_failure(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "Player"
    output.mkdir()
    (output / "TestGame.exe").write_bytes(b"previous")
    builder = _make_builder(tmp_path, output)
    builder._begin_output_transaction()
    staging = Path(builder.output_dir)
    (staging / "TestGame.exe").write_bytes(b"current")
    native_replace = game_builder_module.os.replace

    def fail_staging_publish(source, destination):
        if Path(source) == staging and Path(destination) == output:
            raise PermissionError("read-only publication target")
        return native_replace(source, destination)

    monkeypatch.setattr(game_builder_module.os, "replace", fail_staging_publish)

    with pytest.raises(PermissionError, match="read-only"):
        builder._commit_output_transaction(str(staging))

    assert (output / "TestGame.exe").read_bytes() == b"previous"
    builder._abort_output_transaction()


def test_player_output_transaction_rejects_live_concurrent_owner(tmp_path, monkeypatch):
    output = tmp_path / "Player"
    lock = Path(str(output) + ".infernux-build.lock")
    lock.write_text(json.dumps({"pid": 123, "staging": ""}), encoding="utf-8")
    builder = _make_builder(tmp_path, output)
    monkeypatch.setattr(
        game_builder_module,
        "_player_builder_process_is_alive",
        lambda pid: pid == 123,
    )

    with pytest.raises(RuntimeError, match="Another Player build owns"):
        builder._begin_output_transaction()

    assert lock.is_file()


def test_player_output_transaction_recovers_stale_interrupted_staging(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "Player"
    stale = tmp_path / ".Player.infernux-staging-44-55"
    stale.mkdir()
    (stale / "partial.bin").write_bytes(b"partial")
    lock = Path(str(output) + ".infernux-build.lock")
    lock.write_text(
        json.dumps({"pid": 44, "staging": str(stale)}),
        encoding="utf-8",
    )
    builder = _make_builder(tmp_path, output)
    monkeypatch.setattr(
        game_builder_module,
        "_player_builder_process_is_alive",
        lambda _pid: False,
    )

    builder._begin_output_transaction()

    assert not stale.exists()
    builder._abort_output_transaction()


def test_player_output_transaction_restores_stale_previous_product(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "Player"
    stale = tmp_path / ".Player.infernux-staging-44-55"
    backup = tmp_path / ".Player.infernux-previous-44-55"
    stale.mkdir()
    (stale / "partial.bin").write_bytes(b"partial")
    backup.mkdir()
    (backup / "TestGame.exe").write_bytes(b"previous")
    lock = Path(str(output) + ".infernux-build.lock")
    lock.write_text(
        json.dumps(
            {
                "pid": 44,
                "staging": str(stale),
                "backup": str(backup),
            }
        ),
        encoding="utf-8",
    )
    builder = _make_builder(tmp_path, output)
    monkeypatch.setattr(
        game_builder_module,
        "_player_builder_process_is_alive",
        lambda _pid: False,
    )

    builder._begin_output_transaction()

    assert (output / "TestGame.exe").read_bytes() == b"previous"
    assert not stale.exists()
    assert not backup.exists()
    builder._abort_output_transaction()


def test_player_cleanup_preserves_runtime_closure_and_removes_authoring_state(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    final_dir = tmp_path / "dist"
    retained = {
        "infernux/lib/keep.dll": b"runtime",
        "infernux/lib/_Infernux.pyd": b"native module",
        "_InfernuxBootstrap.pyd": b"bootstrap module",
        "infernux/resources/icons/gizmo_camera.png": b"camera",
        "infernux/resources/icons/gizmo_light.png": b"light",
        "infernux/resources/fonts/engine.otf": b"package-font",
        "Data/Assets/Scripts/player.py.meta": b"project",
        "numpy/__init__.pyc": b"runtime package",
        "numpy/_core/__init__.pyc": b"runtime core",
    }
    removed = {
        "infernux/lib/_Infernux.pyi": b"build-time stubs",
        "_Infernux.pyd": b"native module",
        "infernux/resources/shaders/lit.frag.meta": b"engine",
        "numpy/__pycache__/stale.pyc": b"cache",
        "Data/Library/Resources/fonts/engine.otf": b"duplicate-font",
    }
    for relative, content in (retained | removed).items():
        path = final_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    # One actual cleanup of one prepared product, rather than six independent
    # products. On Windows the mixed-case _Infernux.pyi still reaches both
    # cleanup discovery paths and must be removed only once.
    builder._cleanup_dist(str(final_dir))

    assert {relative: (final_dir / relative).read_bytes() for relative in retained} == retained
    assert all(not (final_dir / relative).exists() for relative in removed)
    assert not (final_dir / "numpy/__pycache__").exists()
    assert not (final_dir / "Data/Library/Resources").exists()


def test_player_cook_uses_asset_index_snapshot_after_live_index_invalidation(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    builder._asset_index_entries()
    (project / "Library" / "AssetIndex.json").unlink()
    data_dir = tmp_path / "dist" / "Data"

    builder._copy_cooked_assets(str(data_dir))

    assert (data_dir / "Assets" / "Main.scene").is_file()


def test_player_stages_project_shader_as_packed_runtime_glsl(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    shader = project / "Assets" / "Shaders" / "Surface.frag"
    shader.parent.mkdir(parents=True, exist_ok=True)
    shader.write_text('ShaderInfo { Name "Test/Surface" Type Fragment }\n', encoding="utf-8")
    entry = _asset_index_entry(project, shader, "shader-guid", "", "Shader")
    entry["metadata"]["metadata"]["type"] = {"value": "fragment"}
    builder._cooked_asset_entries = {"shader-guid": entry}
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    data_dir = tmp_path / "dist" / "Data"

    _stage_document_inputs(builder, data_dir)
    builder._stage_library_runtime_documents(str(data_dir))

    artifact = data_dir / "Library" / "Artifacts" / "Blob" / "shader-guid.frag"
    assert artifact.read_bytes() == shader.read_bytes()
    assert (
        builder._runtime_artifact_bindings["Library/Artifacts/Blob/shader-guid.frag"]["source_path"]
        == "Assets/Shaders/Surface.frag"
    )


def test_player_stages_font_as_guid_owned_runtime_blob(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    font = project / "Assets" / "Fonts" / "ProjectFont.ttf"
    font.parent.mkdir(parents=True, exist_ok=True)
    font.write_bytes(b"project font payload")
    entry = _asset_index_entry(project, font, "font-guid", "", "Font")
    builder._cooked_asset_entries = {"font-guid": entry}
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    data_dir = tmp_path / "dist" / "Data"

    _stage_document_inputs(builder, data_dir)
    builder._stage_library_runtime_documents(str(data_dir))

    runtime_path = "Library/Artifacts/Blob/font-guid.ttf"
    artifact = data_dir / Path(runtime_path)
    assert artifact.read_bytes() == b"project font payload"
    assert builder._runtime_artifact_bindings[runtime_path]["source_path"] == ("Assets/Fonts/ProjectFont.ttf")
    assert "assets/fonts/projectfont.ttf" in builder._runtime_artifact_source_paths


def test_player_cooks_data_asset_to_binary_infernux_artifact(tmp_path):
    from infernux.core.data_asset import decode_data_asset_artifact

    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    source = project / "Assets" / "Data" / "Settings.inxdata"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            {
                "$type": "data_asset",
                "type_id": "infernux.data_asset",
                "fields": {},
            }
        ),
        encoding="utf-8",
    )
    entry = _asset_index_entry(project, source, "data-guid", "", "DataAsset")
    builder._cooked_asset_entries = {"data-guid": entry}
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    data_dir = tmp_path / "dist" / "Data"

    _stage_document_inputs(builder, data_dir)
    builder._stage_library_runtime_documents(str(data_dir))

    runtime_path = "Library/Artifacts/Data/data-guid.inxasset"
    artifact = data_dir / Path(runtime_path)
    assert artifact.read_bytes().startswith(b"INXDATA\0")
    cooked = decode_data_asset_artifact(artifact.read_bytes())
    assert cooked == {
        "$type": "data_asset",
        "type_id": "infernux.data_asset",
        "fields": {},
    }
    # Cook normalizes runtime values without rewriting the authored source.
    assert "schema_version" not in json.loads(source.read_text(encoding="utf-8"))
    assert builder._runtime_artifact_bindings[runtime_path]["source_path"] == ("Assets/Data/Settings.inxdata")
    assert "assets/data/settings.inxdata" in builder._runtime_artifact_source_paths


@pytest.mark.parametrize("imported", [False, True])
def test_scene_cook_preserves_native_joint_references_and_loads(scene, tmp_path, imported):
    from infernux.lib import SceneManager
    from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction

    target = scene.create_game_object("JointTarget")
    body = target.add_component("Rigidbody")
    owner = scene.create_game_object("JointOwner")
    owner.add_component("Rigidbody")
    joint = owner.add_component("HingeJoint")
    joint.connected_body = body
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    source = project / "Assets/Main.scene"
    assert scene.save_to_file(str(source))
    authored = source.read_bytes()
    if imported:
        source = project / "Assets/Generated.custom"
        source.write_bytes(b"custom importer source")
    entry = _asset_index_entry(project, source, "scene-guid", "", "Scene")
    if imported:
        entry["metadata"]["metadata"].update(
            {
                "import_document": {"value": authored.decode("utf-8")},
                "file_extension": {"value": ".scene"},
            }
        )
    source_before = source.read_bytes()
    builder._cooked_asset_entries = {
        "scene-guid": entry,
    }
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    data_dir = tmp_path / "dist/Data"
    artifact = data_dir / "Library/Artifacts/Document/scene-guid.scene"
    _stage_document_inputs(builder, data_dir)
    builder._stage_library_runtime_documents(str(data_dir))
    first = artifact.read_bytes()
    builder._stage_library_runtime_documents(str(data_dir))
    assert artifact.read_bytes() == first
    assert source.read_bytes() == source_before
    cooked = json.loads(first)
    assert cooked["identity_format"] == "runtime-v1"
    assert not {"authoring_identity", "nextObjectId", "nextComponentId"}.intersection(cooked)
    loaded = SceneManager.instance().create_scene("CookedWorld")
    transaction = SceneDocumentTransaction(loaded, path=str(artifact), clear_registries=False)
    assert transaction.run_to_completion(), transaction.error
    assert loaded.find("JointTarget").id != target.id
    connected = loaded.find("JointOwner").get_component("HingeJoint").connected_body
    assert connected is loaded.find("JointTarget").get_component("Rigidbody")


def test_cooked_python_sources_follow_the_frozen_player_closure(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    runtime = project / "Assets" / "Scripts" / "Gameplay.py"
    editor = project / "Assets" / "Editor" / "Bake.py"
    package = project / "Packages" / "effects" / "Runtime" / "Bloom.py"
    for source in (runtime, editor, package):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("value = 1\n", encoding="utf-8")

    builder._player_python_source_paths = {
        str(runtime.resolve()),
        str(package.resolve()),
    }

    assert builder.cooked_python_source_paths() == tuple(
        sorted((str(runtime.resolve()), str(package.resolve())), key=str.casefold)
    )
    assert str(editor.resolve()) not in builder.cooked_python_source_paths()


def test_game_data_includes_particle_artifacts(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    _reference_particle_graph(Path(builder.project_path), "smoke")
    guid = hashlib.md5(b"smoke").hexdigest()
    artifact = Path(builder.project_path) / "Library" / "Artifacts" / "Particle" / f"{guid}.inxparticle"
    final_dir = tmp_path / "dist"

    builder._copy_game_data(str(final_dir))
    shipped = final_dir / "Data" / "Library" / "Artifacts" / "Particle" / artifact.name
    assert shipped.read_text(encoding="utf-8") == artifact.read_text(encoding="utf-8")
    assert json.loads(shipped.read_text(encoding="utf-8"))["source_hash"]
    runtime_index = json.loads((shipped.parent / builder._PARTICLE_RUNTIME_INDEX_FILENAME).read_text(encoding="utf-8"))
    assert runtime_index == {
        "$schema": "infernux.particle_runtime_index",
        "entries": [
            {
                "guid": hashlib.md5(b"smoke").hexdigest(),
                "path_hint": "Assets/VFX/smoke.particlegraph",
                "stable_id": "smoke",
            }
        ],
    }

    _write_player_executable(final_dir)
    builder._organize_player_layout(str(final_dir))
    builder._pack_content_archive(str(final_dir))
    header = read_manifest(final_dir / "TestGame_Data" / builder._CONTENT_ARCHIVE_FILENAME)
    names = {entry["path"] for entry in header["files"]}
    assert f"Library/Artifacts/Particle/{guid}.inxparticle" in names
    assert "Library/Artifacts/Particle/RuntimeIndex.json" in names


def test_game_data_compiles_missing_particle_artifact(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    _reference_particle_graph(project, "missing")
    (project / "Library" / "Artifacts" / "Particle").mkdir(parents=True, exist_ok=True)
    guid = hashlib.md5(b"missing").hexdigest()

    builder._copy_game_data(str(tmp_path / "dist"))

    shipped = tmp_path / "dist" / "Data" / "Library" / "Artifacts" / "Particle" / f"{guid}.inxparticle"
    assert shipped.is_file()
    payload = json.loads(shipped.read_text(encoding="utf-8"))
    assert payload["$schema"] == "infernux.particle_artifact"
    assert payload["source_hash"]


def test_game_data_fails_when_particle_source_cannot_compile(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    graph_path = _reference_particle_graph(project, "broken")
    graph_path.write_text("{not-json", encoding="utf-8")
    guid = hashlib.md5(b"broken").hexdigest()
    scene_path = project / "Assets" / "Main.scene"
    _write_asset_index(
        project,
        [
            _asset_index_entry(project, scene_path, "scene-guid", "", "Scene"),
            _asset_index_entry(project, graph_path, guid, "", "ParticleGraph"),
        ],
    )
    (project / "Library" / "Artifacts" / "Particle").mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="Library particle artifact compile failed"):
        builder._copy_game_data(str(tmp_path / "dist"))


def test_game_data_keeps_particle_artifacts_that_share_graph_stable_id(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    first = _reference_particle_graph(project, "shared")
    second = project / "Assets" / "VFX" / "copy.particlegraph"
    second.write_text(
        ParticleGraphAsset(stable_id="shared", name="copy").canonical_json(),
        encoding="utf-8",
    )
    guid1 = hashlib.md5(b"shared").hexdigest()
    guid2 = hashlib.md5(b"copy").hexdigest()
    scene_path = project / "Assets" / "Main.scene"
    _write_asset_index(
        project,
        [
            _asset_index_entry(project, scene_path, "scene-guid", "", "Scene"),
            _asset_index_entry(project, first, guid1, "", "ParticleGraph"),
            _asset_index_entry(project, second, guid2, "", "ParticleGraph"),
        ],
    )
    artifact_root = project / "Library" / "Artifacts" / "Particle"
    _write_particle_artifact(artifact_root / f"{guid1}.inxparticle", first)
    _write_particle_artifact(artifact_root / f"{guid2}.inxparticle", second)
    (artifact_root / "RuntimeIndex.json").write_text(
        json.dumps(
            {
                "$schema": "infernux.particle_runtime_index",
                "entries": [
                    {
                        "guid": guid1,
                        "path_hint": "Assets/VFX/shared.particlegraph",
                        "stable_id": "shared",
                    },
                    {
                        "guid": guid2,
                        "path_hint": "Assets/VFX/copy.particlegraph",
                        "stable_id": "shared",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    builder._copy_game_data(str(tmp_path / "dist"))
    shipped = tmp_path / "dist" / "Data" / "Library" / "Artifacts" / "Particle"
    assert (shipped / f"{guid1}.inxparticle").is_file()
    assert (shipped / f"{guid2}.inxparticle").is_file()


def test_content_archive_keeps_compiled_scripts_and_cooked_runtime_documents(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    data = tmp_path / "dist" / "TestGame_Data"
    assets = data / "Assets" / "Scripts"
    assets.mkdir(parents=True)
    (assets / "Player.pyc").write_bytes(b"compiled-player")
    (assets / "Player.py.meta").write_text("editor metadata", encoding="utf-8")
    document = data / "Library" / "Artifacts" / "Document" / "scene-guid.scene"
    document.parent.mkdir(parents=True)
    document.write_text('{"name":"Main","objects":[]}', encoding="utf-8")
    (data / "BuildManifest.json").write_text('{"game_name":"TestGame"}', encoding="utf-8")

    builder._pack_content_archive(str(tmp_path / "dist"))

    header = read_manifest(data / builder._CONTENT_ARCHIVE_FILENAME)
    names = {entry["path"] for entry in header["files"]}
    assert "Assets/Scripts/Player.pyc" in names
    assert "Library/Artifacts/Document/scene-guid.scene" in names
    assert not any(name.endswith(".meta") for name in names)
    assert not (assets / "Player.pyc").exists()
    assert not document.exists()


def test_core_runtime_archive_replaces_loose_numpy_and_resources(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    final_dir = tmp_path / "dist"
    numpy_file = final_dir / "numpy" / "core.py"
    numpy_init = final_dir / "numpy" / "__init__.pyc"
    numpy_core_init = final_dir / "numpy" / "_core" / "__init__.pyc"
    numpy_dll = final_dir / "numpy.libs" / "openblas.dll"
    numpy_header = final_dir / "numpy" / "_core" / "include" / "numpy" / "arrayobject.h"
    numpy_example = final_dir / "numpy" / "random" / "_examples" / "extending.pyx"
    numpy_stub = final_dir / "numpy" / "typing" / "_array_like.pyi"
    numpy_tests_extension = final_dir / "numpy" / "_core" / "_multiarray_tests.cp313-win_amd64.pyd"
    numpy_api_changes = final_dir / "numpy" / "ma" / "API_CHANGES.txt"
    numpy_license = final_dir / "numpy" / "LICENSE.txt"
    font = final_dir / "infernux" / "resources" / "fonts" / "engine.otf"
    gizmo_icon = final_dir / "infernux" / "resources" / "icons" / "gizmo_camera.png"
    light_icon = final_dir / "infernux" / "resources" / "icons" / "gizmo_light.png"
    editor_icon = final_dir / "infernux" / "resources" / "icons" / "file.png"
    numpy_file.parent.mkdir(parents=True)
    numpy_core_init.parent.mkdir(parents=True, exist_ok=True)
    numpy_dll.parent.mkdir(parents=True)
    numpy_header.parent.mkdir(parents=True)
    numpy_example.parent.mkdir(parents=True)
    numpy_stub.parent.mkdir(parents=True)
    numpy_tests_extension.parent.mkdir(parents=True, exist_ok=True)
    numpy_api_changes.parent.mkdir(parents=True, exist_ok=True)
    numpy_license.parent.mkdir(parents=True, exist_ok=True)
    font.parent.mkdir(parents=True)
    gizmo_icon.parent.mkdir(parents=True)
    light_icon.parent.mkdir(parents=True, exist_ok=True)
    numpy_file.write_text("VALUE = 1", encoding="utf-8")
    numpy_init.write_bytes(b"numpy package")
    numpy_core_init.write_bytes(b"numpy core package")
    numpy_dll.write_bytes(b"dll")
    numpy_header.write_text("header", encoding="utf-8")
    numpy_example.write_text("source", encoding="utf-8")
    numpy_stub.write_text("stub", encoding="utf-8")
    numpy_tests_extension.write_bytes(b"test extension")
    numpy_api_changes.write_text("history", encoding="utf-8")
    numpy_license.write_text("license", encoding="utf-8")
    font.write_bytes(b"font")
    gizmo_icon.write_bytes(b"gizmo")
    light_icon.write_bytes(b"light")
    editor_icon.write_bytes(b"editor")
    (final_dir / "TestGame_Data").mkdir(parents=True)

    builder._pack_core_runtime_archive(str(final_dir))

    archive = final_dir / "TestGame_Data" / "Runtime.inxrt"
    assert archive.is_file()
    assert not (final_dir / "numpy").exists()
    assert not (final_dir / "numpy.libs").exists()
    assert not (final_dir / "infernux" / "resources").exists()
    header = read_manifest(archive)
    assert header["compression_profile"] == "release"
    assert {entry["path"] for entry in header["files"]} == {
        "infernux/resources/fonts/engine.otf",
        "infernux/resources/icons/gizmo_camera.png",
        "infernux/resources/icons/gizmo_light.png",
        "numpy.libs/openblas.dll",
        "numpy/__init__.pyc",
        "numpy/_core/__init__.pyc",
        "numpy/core.py",
        "numpy/LICENSE.txt",
    }


def test_platform_cook_packages_reachable_builtin_resources_in_content(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    scene = project / "Assets" / "Main.scene"
    shader = project / "Library" / "Resources" / "shaders" / "standard.vert"
    shader.parent.mkdir(parents=True, exist_ok=True)
    shader.write_text("void main() {}\n", encoding="utf-8")
    scene.write_text(json.dumps({"shader": "builtin-shader-guid"}), encoding="utf-8")
    scene_entry = _asset_index_entry(project, scene, "scene-guid", "", "Scene")
    shader_entry = _asset_index_entry(
        project,
        shader,
        "builtin-shader-guid",
        "",
        "Shader",
    )
    shader_entry["read_only"] = True
    _write_asset_index(project, [scene_entry, shader_entry])
    final_dir = tmp_path / "dist"
    data_dir = final_dir / "Data"

    builder._copy_cooked_assets(
        str(data_dir),
        package_builtin_resources=True,
    )

    packaged_shader = data_dir / "infernux/resources/shaders/standard.vert"
    assert packaged_shader.read_text(encoding="utf-8") == "void main() {}\n"

    builder._cooked_asset_entries = {"builtin-shader-guid": shader_entry}
    builder._runtime_artifact_bindings = {}
    builder._write_runtime_asset_records(
        str(final_dir),
        package_builtin_resources=True,
    )
    records = json.loads((data_dir / "Library" / "RuntimeAssetRecords.json").read_text(encoding="utf-8"))
    shader_record = next(item for item in records["entries"] if item["guid"] == "builtin-shader-guid")
    assert shader_record["runtime_artifacts"] == [
        {
            "package": "Content.inxpkg",
            "runtime_path": "infernux/resources/shaders/standard.vert",
            "runtime_artifact_id": game_builder_module.runtime_artifact_id(
                "Content.inxpkg", "infernux/resources/shaders/standard.vert"
            ),
        }
    ]

    data_root = final_dir / "TestGame_Data"
    data_root.parent.mkdir(parents=True, exist_ok=True)
    data_dir.rename(data_root)
    (data_root / "Assets/Main.scene").unlink()
    (data_root / "Assets").rmdir()
    builder._pack_content_archive(
        str(final_dir),
        package_builtin_resources=True,
    )
    manifest = read_manifest(data_root / "Content.inxpkg")
    assert "infernux/resources/shaders/standard.vert" in {entry["path"] for entry in manifest["files"]}


def test_package_resource_shader_keeps_guid_identity_in_headless_build(tmp_path, monkeypatch):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    scene = project / "Assets" / "Main.scene"
    package_resources = tmp_path / "installed-engine" / "infernux" / "resources"
    shader = package_resources / "shaders" / "particle_sprite.vert"
    shader.parent.mkdir(parents=True, exist_ok=True)
    shader.write_text("void main() {}\n", encoding="utf-8")
    scene.write_text(json.dumps({"shader": "particle-shader-guid"}), encoding="utf-8")
    scene_entry = _asset_index_entry(project, scene, "scene-guid", "", "Scene")
    shader_entry = _asset_index_entry(
        project,
        shader,
        "particle-shader-guid",
        "",
        "Shader",
    )
    shader_entry["read_only"] = True
    shader_entry["metadata"]["metadata"]["shader_id"] = {
        "type": "string",
        "value": "Particle Sprite",
    }
    monkeypatch.setattr(
        game_builder_module._resources,
        "get_package_resources_path",
        lambda: str(package_resources),
    )
    monkeypatch.setattr(
        game_builder_module._resources,
        "resources_path",
        str(package_resources),
    )
    entries = [scene_entry, shader_entry]
    _write_asset_index(project, entries)
    (project / "ProjectSettings" / "BuildSettings.json").write_text(
        json.dumps({"scene_guids": ["scene-guid"]}),
        encoding="utf-8",
    )

    selected = builder._collect_library_asset_entries(entries)

    assert set(selected) == {"scene-guid", "particle-shader-guid"}
    builder._copy_cooked_assets(str(tmp_path / "copy" / "Data"))
    assert (tmp_path / "copy" / "Data" / "Assets" / "Main.scene").is_file()
    assert not (tmp_path / "copy" / "Data" / "Library" / "Resources" / "shaders" / "particle_sprite.vert").exists()
    builder._cooked_asset_entries = {"particle-shader-guid": selected["particle-shader-guid"]}
    builder._runtime_artifact_bindings = {}
    final_dir = tmp_path / "dist"
    builder._write_runtime_asset_records(str(final_dir))
    records = json.loads((final_dir / "Data" / "Library" / "RuntimeAssetRecords.json").read_text(encoding="utf-8"))
    shader_record = next(item for item in records["entries"] if item["guid"] == "particle-shader-guid")
    assert shader_record["runtime_path"] == ("Library/Resources/shaders/particle_sprite.vert")
    assert shader_record["runtime_artifacts"][0]["runtime_path"] == ("infernux/resources/shaders/particle_sprite.vert")
    assert shader_record["metadata"]["metadata"]["shader_id"]["value"] == ("Particle Sprite")
