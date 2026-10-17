from __future__ import annotations

import importlib.machinery
import importlib.util

from test_game_builder import (
    BuildOutputDirectoryError,
    GameBuilder,
    Path,
    _asset_index_entry,
    _bind_staged_script_to_asset_index,
    _collect_fixture_dependencies,
    _fnv1a64,
    _make_builder,
    _make_project,
    _prepare_runtime_catalog_inputs,
    _read_runtime_catalog,
    _scene_with_component,
    _write_asset_index,
    _write_player_executable,
    artifact_source_hash,
    build_catalog,
    game_builder_module,
    json,
    logical_type_for_path,
    payload_kind_for,
    py_compile,
    pytest,
    read_entry,
    read_manifest,
    sys,
    write_pack,
    _native_package_backend as _native_package_backend,
)


def test_player_path_audit_distinguishes_model_names_from_absolute_paths():
    from infernux.engine.player_package_audit import _contains_absolute_author_path

    names = json.dumps(["01_backpack_01_..\\packs\\bomb", "Bip01 Pelvis"])
    metadata = {"model_meshes": {"value": names}, "skeleton_node_names": {"value": names}}
    assert not _contains_absolute_author_path(json.dumps(metadata))
    for path in ("C:/Users/Author/Model.glb", r"\\server\share\Model.glb", "/home/author/Model.glb"):
        assert _contains_absolute_author_path(json.dumps({"source": path}))
        reference = {"$type": "asset_ref", "path_hint": path}
        assert _contains_absolute_author_path(json.dumps(reference), game_content=True)
    assert not _contains_absolute_author_path(json.dumps({"source": "Assets/Models/Model.glb"}))


def test_cooked_python_component_keeps_script_and_runtime_guid_identity(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    scene = project / "Assets" / "Main.scene"
    script = project / "Assets" / "Scripts" / "Mover.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("class Mover:\n    pass\n", encoding="utf-8")
    script_guid = "1234567890abcdef1234567890abcdef"
    scene.write_text(
        json.dumps(_scene_with_component({}, type_id=f"python:{script_guid}:type-guid:Scripts.Mover:Mover")),
        encoding="utf-8",
    )
    _write_asset_index(
        project,
        [
            _asset_index_entry(project, scene, "scene-guid", "", "Scene"),
            _asset_index_entry(project, script, script_guid, "", "Script"),
        ],
    )
    (project / "ProjectSettings" / "BuildSettings.json").write_text(
        json.dumps({"scene_guids": ["scene-guid"]}),
        encoding="utf-8",
    )
    final_dir = tmp_path / "dist"
    data_dir = final_dir / "Data"

    builder._copy_cooked_assets(str(data_dir))
    assert (data_dir / "Assets" / "Scripts" / "Mover.py").is_file()

    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    builder._stage_library_runtime_documents(str(data_dir))
    builder._compile_user_scripts(str(final_dir))
    builder._write_runtime_asset_records(str(final_dir))

    assert not (data_dir / "Assets" / "Scripts" / "Mover.py").exists()
    assert (data_dir / "Assets" / "Scripts" / "Mover.pyc").is_file()
    guid_map = json.loads((data_dir / "_script_guid_map.json").read_text(encoding="utf-8"))
    assert guid_map[script_guid].replace("\\", "/") == "Assets/Scripts/Mover.pyc"
    records = json.loads((data_dir / "Library" / "RuntimeAssetRecords.json").read_text(encoding="utf-8"))
    script_record = next(item for item in records["entries"] if item["guid"] == script_guid)
    assert script_record["runtime_path"] == "Assets/Scripts/Mover.pyc"
    expected_artifact_id = game_builder_module.runtime_artifact_id("Content.inxpkg", "Assets/Scripts/Mover.pyc")
    assert set(records) == {"$schema", "entries"}
    assert script_record["primary_runtime_artifact_id"] == expected_artifact_id
    assert script_record["runtime_artifact_ids"] == [expected_artifact_id]
    assert builder._runtime_asset_identity_bindings["Assets/Scripts/Mover.pyc"]["source_guid"] == script_guid
    assert str(project).replace("\\", "/") not in json.dumps(records, ensure_ascii=False)


def test_cooked_python_component_includes_imported_project_helper(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    scene = project / "Assets" / "Main.scene"
    scripts = project / "Assets" / "Scripts" / "Voxel"
    component = scripts / "VoxelContinentGenerator.py"
    helper = scripts / "VoxelPipeline.py"
    unused = scripts / "Unused.py"
    scripts.mkdir(parents=True, exist_ok=True)
    component.write_text(
        "from Scripts.Voxel.VoxelPipeline import register_voxel_effects\nclass VoxelContinentGenerator:\n    pass\n",
        encoding="utf-8",
    )
    helper.write_text("def register_voxel_effects():\n    return True\n", encoding="utf-8")
    unused.write_text("UNUSED = True\n", encoding="utf-8")
    component_guid = "component-script-guid"
    helper_guid = "helper-script-guid"
    scene.write_text(
        json.dumps(
            {
                "component": (
                    f"python:{component_guid}:type-guid:Scripts.Voxel.VoxelContinentGenerator:VoxelContinentGenerator"
                )
            }
        ),
        encoding="utf-8",
    )
    _write_asset_index(
        project,
        [
            _asset_index_entry(project, scene, "scene-guid", "", "Scene"),
            _asset_index_entry(project, component, component_guid, "", "Script"),
            _asset_index_entry(project, helper, helper_guid, "", "Script"),
            _asset_index_entry(project, unused, "unused-script-guid", "", "Script"),
        ],
    )
    (project / "ProjectSettings" / "BuildSettings.json").write_text(
        json.dumps({"scene_guids": ["scene-guid"]}),
        encoding="utf-8",
    )

    selected = builder._collect_library_asset_entries(builder._asset_index_entries())

    assert set(selected) == {
        "scene-guid",
        component_guid,
        helper_guid,
        "unused-script-guid",
    }


def test_player_build_seals_startup_warmups_for_target_platform(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    builder._runtime_platform = "android"
    data_dir = tmp_path / "build_output" / "Data"
    registry = data_dir / "Library" / "StartupWarmup.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "schema": "infernux.startup-warmup",
                "platform": "windows",
                "entries": {
                    "Scripts.Compute:Compute._infernux_startup_warmup": {
                        "platform": "windows",
                        "status": "ready",
                        "source": "C:/author/project/Assets/Scripts/Compute.py",
                        "source_sha256": "obsolete",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    builder._prepare_startup_warmup_registry(str(data_dir))

    document = json.loads(registry.read_text(encoding="utf-8"))
    entry = next(iter(document["entries"].values()))
    assert document["platform"] == "android"
    assert entry["platform"] == "android"
    assert "source" not in entry
    assert "source_sha256" not in entry


@pytest.mark.parametrize("base_expression", ["Parent", "base_module.Parent"])
def test_player_component_inheritance_keeps_cross_script_identity(tmp_path, monkeypatch, base_expression):
    from types import ModuleType

    from infernux.components.component_identity import bind_asset_script_guid
    from infernux.engine.runtime_type_registry import (
        clear_runtime_type_registry,
        install_runtime_type_registry,
    )
    from infernux.lib import _Infernux as native

    package = ModuleType("player_inheritance")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    sources = {
        "player_inheritance.base": (
            "from infernux import InxComponent, serialized_field\n"
            "class Parent(InxComponent):\n"
            "    speed: float = serialized_field(default=3.0, field_id='motion.speed')\n"
            "    def start(self):\n"
            "        raise AssertionError('cook must not invoke lifecycle methods')\n"
        ),
        "player_inheritance.child": (
            "from player_inheritance.base import Parent\n"
            "import player_inheritance.base as base_module\n"
            f"class Child({base_expression}):\n"
            "    strength: float = 5.0\n"
            "    def update(self, dt):\n"
            "        raise AssertionError('cook must not invoke lifecycle methods')\n"
        ),
    }
    output = tmp_path / "build_output"
    builder = _make_builder(tmp_path, output)
    for index, (name, source) in enumerate(sources.items()):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(package, name.rsplit(".", 1)[1], module, raising=False)
        exec(compile(source, name, "exec"), module.__dict__)
        value_type = module.Parent if index == 0 else module.Child
        guid = f"component-script-{index}"
        bind_asset_script_guid(value_type, guid)
        script = output / "Data" / "Assets" / (name.replace(".", "/") + ".py")
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(source, encoding="utf-8")
        _bind_staged_script_to_asset_index(builder, output, script, guid=guid)

    builder._compile_user_scripts(str(output))
    registry_path = output / "Data" / "Library" / "RuntimeTypeRegistry.json"
    records = {record["qualname"]: record for record in json.loads(registry_path.read_text())["types"]}
    assert set(records) == {"Parent", "Child"}
    parent, child = records["Parent"], records["Child"]
    assert child["script_guid"] == "component-script-1"
    assert child["lifecycle"] == ["update"]
    assert parent["lifecycle"] == ["start"]
    assert child["semantic"]["base_type_guid"] == parent["type_guid"]
    assert [field["attributes"]["field_id"] for field in child["semantic"]["fields"]] == [
        "motion.speed",
        "strength",
    ]
    try:
        assert install_runtime_type_registry(str(registry_path)) == 2
        descriptor = native._semantic_catalog_snapshot().type_document(child["type_guid"])
        assert descriptor["base_type_guid"] == parent["type_guid"]
    finally:
        clear_runtime_type_registry()


@pytest.mark.parametrize("split_modules", [False, True])
@pytest.mark.parametrize(
    "data_base,explicit_ids",
    [
        ("SerializableObject", False),
        ("SerializableObject", True),
        ("DataAsset", True),
    ],
)
def test_player_data_inheritance_survives_cook_and_catalog_publication(
    tmp_path,
    monkeypatch,
    split_modules,
    data_base,
    explicit_ids,
):
    from types import ModuleType

    from infernux.components import serializable_object
    from infernux.engine.runtime_type_registry import (
        clear_runtime_type_registry,
        install_runtime_type_registry,
    )
    from infernux.lib import _Infernux as native

    monkeypatch.setattr(
        serializable_object,
        "_SERIALIZABLE_REGISTRY",
        dict(serializable_object._SERIALIZABLE_REGISTRY),
    )
    base_module = "Scripts.player_base"
    child_module = "Scripts.player_derived" if split_modules else base_module
    base_source = (
        f"from infernux import {data_base}, serialized_field\n"
        f"class BaseConfig({data_base}):\n"
        + ("    __serialized_type_id__ = 'tests.player.base'\n" if explicit_ids or data_base == "DataAsset" else "")
        + "    speed: float = serialized_field(default=3.0, field_id='motion.speed')\n"
        "class IntermediateConfig(BaseConfig):\n"
        + ("    __serialized_type_id__ = 'tests.player.middle'\n" if data_base == "DataAsset" else "")
        + "    weight: float = 2.0\n"
    )
    child_source = (
        "class DerivedConfig(IntermediateConfig):\n"
        + ("    __serialized_type_id__ = 'tests.player.derived'\n" if explicit_ids else "")
        + "    strength: float = 5.0\n"
    )
    sources = {base_module: base_source}
    if split_modules:
        sources[child_module] = f"from {base_module} import IntermediateConfig\n" + child_source
    else:
        sources[base_module] += child_source

    output = tmp_path / "build_output"
    builder = _make_builder(tmp_path, output)
    for index, (name, source) in enumerate(sources.items()):
        if name.startswith("Scripts.") and "Scripts" not in sys.modules:
            monkeypatch.setitem(sys.modules, "Scripts", ModuleType("Scripts"))
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(source, name, "exec"), module.__dict__)
        script = output / "Data" / "Assets" / (name.replace(".", "/") + ".py")
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(source, encoding="utf-8")
        _bind_staged_script_to_asset_index(builder, output, script, guid=f"data-script-{index}")

    builder._compile_user_scripts(str(output))
    registry_path = output / "Data" / "Library" / "RuntimeTypeRegistry.json"
    records = {record["qualname"]: record for record in json.loads(registry_path.read_text())["types"]}
    assert set(records) == {"BaseConfig", "IntermediateConfig", "DerivedConfig"}
    base, intermediate, derived = (records[name] for name in ("BaseConfig", "IntermediateConfig", "DerivedConfig"))
    assert intermediate["semantic"]["base_type_guid"] == base["type_guid"]
    assert derived["semantic"]["base_type_guid"] == intermediate["type_guid"]
    assert "schema_version" not in derived["semantic"]
    assert derived["semantic"]["owner"] == f"script:data-script-{int(split_modules)}"
    assert [field["attributes"]["field_id"] for field in derived["semantic"]["fields"]] == [
        "motion.speed",
        "weight",
        "strength",
    ]
    try:
        assert install_runtime_type_registry(str(registry_path)) == 3
        descriptor = native._semantic_catalog_snapshot().type_document(derived["type_guid"])
        assert descriptor["base_type_guid"] == intermediate["type_guid"]
        assert descriptor["fields"] == derived["semantic"]["fields"]
    finally:
        clear_runtime_type_registry()


@pytest.mark.parametrize("owner", ["Assets/Rendering", "Packages/author/monitor/Runtime"])
def test_render_texture_cook_ships_guid_binary_description_not_authoring_source(tmp_path, owner):
    from infernux.lib import _Infernux as native

    builder = _make_builder(tmp_path, tmp_path / "build_output")
    project = Path(builder.project_path)
    source = project / owner / "CameraTarget.rendertexture"
    source.parent.mkdir(parents=True, exist_ok=True)
    desc = native._RenderTextureDesc()
    desc.relative_size = True
    desc.width_scale = desc.height_scale = 0.5
    source.write_text(native._render_texture_description_to_json(desc), encoding="utf-8")
    if owner.startswith("Packages/"):
        (source.parent.parent / "inx_package.json").write_text(
            json.dumps({"reference": "author/monitor", "name": "Monitor", "version": "1.0.0"}),
            encoding="utf-8",
        )
        Path(str(source) + ".meta").write_text(
            json.dumps({"metadata": {"guid": {"type": "string", "value": "e" * 32}}}), encoding="utf-8"
        )
    scene = project / "Assets" / "Main.scene"
    guid = "e" * 32
    artifact_path = f"Library/Artifacts/RenderTexture/{guid}.inxrtex"
    artifact = project / artifact_path
    artifact.parent.mkdir(parents=True)
    source_hash = _fnv1a64(source.read_bytes())
    payload = native._encode_render_texture_artifact(desc, source_hash)
    artifact.write_bytes(payload)
    entry = _asset_index_entry(project, source, guid, artifact_path, "RenderTexture")
    _write_asset_index(project, [_asset_index_entry(project, scene, "scene-guid", "", "Scene"), entry])
    final_dir = tmp_path / "dist"
    builder._copy_game_data(str(final_dir))
    builder._write_runtime_asset_records(str(final_dir))
    assert (final_dir / "Data" / artifact_path).read_bytes() == payload
    _write_player_executable(final_dir)
    builder._organize_player_layout(str(final_dir))
    builder._pack_content_archive(str(final_dir))
    data_root = final_dir / "TestGame_Data"
    package = data_root / builder._CONTENT_ARCHIVE_FILENAME
    names = {item["path"] for item in read_manifest(package)["files"]}
    assert artifact_path in names
    assert not any(name.endswith(".rendertexture") for name in names)
    assert not any(name.startswith(("Assets/", "Packages/")) for name in names)
    cooked = native._decode_render_texture_artifact(read_entry(package, artifact_path))
    assert cooked.relative_size and cooked.width_scale == 0.5
    assert artifact_source_hash(artifact) == source_hash
    assert logical_type_for_path(artifact_path) == "render_texture_artifact"
    assert payload_kind_for(logical_type_for_path(artifact_path)) == "compiled_artifact"


@pytest.mark.parametrize("component_type", ["UIText", "UIButton"])
def test_ui_font_chain_is_a_guid_backed_player_cook_dependency(component_type):
    from infernux.core.asset_ref import create_asset_ref
    from infernux.ui import UIButton, UIText

    primary_guid = "ui-primary-font-guid"
    fallback_guid = "ui-fallback-font-guid"
    component = {"UIText": UIText, "UIButton": UIButton}[component_type]()
    component.font = create_asset_ref("Font", guid=primary_guid, path_hint="Assets/Fonts/Primary.ttf")
    component.fallback_fonts = [create_asset_ref("Font", guid=fallback_guid, path_hint="Assets/Fonts/Fallback.otf")]

    document = component._serialize_fields_document()
    references = list(GameBuilder._asset_reference_values(document, frozenset({primary_guid, fallback_guid})))

    assert references == [
        (primary_guid, ""),
        (fallback_guid, ""),
    ]
    assert "font_path" not in document
    assert "fallback_font_paths" not in document


def test_cooked_catalog_keeps_ui_material_shader_and_texture_guids():
    material_guid = "11111111111111111111111111111111"
    vertex_guid = "22222222222222222222222222222222"
    fragment_guid = "33333333333333333333333333333333"
    texture_guid = "44444444444444444444444444444444"
    payloads = {
        f"Library/Artifacts/Document/{material_guid}.mat": json.dumps(
            {
                "shaders": {
                    "vertex": {"guid": vertex_guid, "shader_id": "UI/Vertex"},
                    "fragment": {"guid": fragment_guid, "shader_id": "UI/Fragment"},
                },
                "properties": {
                    "gain": {"type": "Float", "value": 0.5},
                    "detailTex": {"type": "Texture2D", "guid": texture_guid},
                },
            }
        ).encode("utf-8"),
        f"Library/Artifacts/Document/{vertex_guid}.vert": b"#version 450\n",
        f"Library/Artifacts/Document/{fragment_guid}.frag": b"#version 450\n",
        f"Library/Artifacts/Textures/{texture_guid}.inxtex": b"texture artifact",
    }
    catalog = build_catalog(
        [
            {
                "package": "Content.inxpkg",
                "runtime_path": path,
                "bytes": len(payload),
                "payload": payload,
                "asset_binding": {
                    "source_guid": Path(path).stem,
                    "source_path": f"Assets/{Path(path).name}",
                    "dependencies": [],
                },
            }
            for path, payload in payloads.items()
        ],
        player_host={"executable": "Game.exe"},
        package_records=[],
    )
    by_path = {item["runtime_path"]: item for item in catalog["artifacts"]}
    material = by_path[f"Library/Artifacts/Document/{material_guid}.mat"]
    expected = {by_path[path]["runtime_artifact_id"] for path in payloads if not path.endswith(".mat")}
    assert set(material["dependencies"]) == expected
    assert material["unresolved_dependencies"] == []


def test_generated_player_boot_requires_build_manifest_in_the_data_tree(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    source = Path(builder._generate_boot_script()).read_text(encoding="utf-8")

    required = "if not os.path.isfile(_BUILD_MANIFEST_PATH):"
    assert required in source
    assert '_BUILD_MANIFEST_PATH = os.path.join(_DATA_DIR, "BuildManifest.json")' in source
    assert "_copy_player_file_atomic" not in source
    assert "if os.path.isfile(_BUILD_MANIFEST_PATH):" not in source


def test_desktop_player_keeps_project_content_in_native_package(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    final_dir = tmp_path / "dist"
    data_root = _prepare_runtime_catalog_inputs(builder, final_dir)
    bootstrap_source = tmp_path / "bootstrap.pyd"
    python_source = tmp_path / "python313.dll"
    bootstrap_source.write_bytes(b"bootstrap")
    python_source.write_bytes(b"python")
    write_pack(
        (
            ("_InfernuxBootstrap.pyd", bootstrap_source),
            ("python313.dll", python_source),
        ),
        data_root / "Bootstrap.inxrt",
    )
    (data_root / "BuildManifest.json").write_text(json.dumps({"scene_guids": ["scene-guid"]}), encoding="utf-8")
    builder._write_payload_manifest(str(final_dir))

    builder._materialize_desktop_player_layout(str(final_dir))
    builder._audit_direct_player_layout(str(final_dir))

    assert (data_root / "Runtime" / "_InfernuxBootstrap.pyd").read_bytes() == b"bootstrap"
    assert (data_root / "Runtime" / "python313.dll").read_bytes() == b"python"
    assert (data_root / "Runtime" / "infernux" / "resources" / "runtime.bin").read_bytes() == b"runtime"
    assert not (data_root / "Bootstrap.inxrt").exists()
    assert not (data_root / builder._RUNTIME_ARCHIVE_FILENAME).exists()
    assert (data_root / builder._CONTENT_ARCHIVE_FILENAME).is_file()
    assert (data_root / builder._ASSET_CATALOG_ARCHIVE_FILENAME).is_file()
    assert (data_root / builder._PLAYER_PACKAGE_INDEX_FILENAME).is_file()
    assert not (data_root / "Library").exists()
    assert not (data_root / "BuildManifest.json").exists()
    assert not (data_root / "RuntimeAssets").exists()
    assert not (data_root / "Cache").exists()
    catalog = _read_runtime_catalog(data_root, builder)
    assert json.loads(
        read_entry(
            data_root / builder._ASSET_CATALOG_ARCHIVE_FILENAME,
            "BuildManifest.json",
        ).decode("utf-8")
    )["scene_guids"] == ["scene-guid"]
    assert len(catalog["packages"]) == 1
    assert Path(catalog["packages"][0]["path"]).name == "Content.inxpkg"
    manifest = json.loads((data_root / builder._PLAYER_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["product"]["layout"] == "single_executable_native_packages"
    package_index = (data_root / builder._PLAYER_PACKAGE_INDEX_FILENAME).read_text(encoding="ascii")
    assert "content\t" in package_index
    assert "catalog\t" in package_index
    assert "runtime\t" not in package_index


def test_desktop_player_indexes_the_sealed_parallel_module(tmp_path):
    project_root = _make_project(tmp_path)
    builder = GameBuilder(
        str(project_root),
        str(tmp_path / "build_output"),
        game_name="TestGame",
        include_jit_runtime=True,
    )
    final_dir = tmp_path / "dist"
    data_root = _prepare_runtime_catalog_inputs(builder, final_dir)
    bootstrap_source = tmp_path / "bootstrap.pyd"
    parallel_source = tmp_path / "parallel.pyd"
    bootstrap_source.write_bytes(b"bootstrap")
    parallel_source.write_bytes(b"parallel")
    write_pack(
        (("_InfernuxBootstrap.pyd", bootstrap_source),),
        data_root / "Bootstrap.inxrt",
    )
    parallel_archive = data_root / "Modules" / builder._PARALLEL_ARCHIVE_FILENAME
    parallel_archive.parent.mkdir(parents=True)
    write_pack(
        (("llvmlite/binding/ffi.pyd", parallel_source),),
        parallel_archive,
    )
    builder._write_payload_manifest(str(final_dir))

    builder._materialize_desktop_player_layout(str(final_dir))
    builder._audit_direct_player_layout(str(final_dir))

    package_index = (data_root / builder._PLAYER_PACKAGE_INDEX_FILENAME).read_text(encoding="ascii").splitlines()
    records = {parts[0]: (parts[1], int(parts[2])) for parts in (line.split("\t") for line in package_index[1:])}
    parallel_manifest = read_manifest(parallel_archive)
    assert records["parallel"] == (
        parallel_manifest["archive_sha256"],
        parallel_manifest["archive_bytes"],
    )
    assert "runtime" not in records


class TestGameBuilderOutputSafety:
    @pytest.mark.parametrize("same_project,same_name", [(True, True), (False, True), (True, False)])
    def test_sealed_output_manifest_controls_rebuild_ownership(self, tmp_path, same_project, same_name):
        output_dir = tmp_path / "build_output"
        builder = _make_builder(tmp_path, output_dir)
        data_root = output_dir / "TestGame_Data"
        data_root.mkdir(parents=True)
        source = tmp_path / "BuildManifest.json"
        source.write_text(
            json.dumps(
                {
                    "build_output": {
                        "tool": "Infernux",
                        "project_name": builder.project_name if same_name else "OtherGame",
                        "project_identity": game_builder_module.path_fingerprint(builder.project_path)
                        if same_project
                        else "0" * 64,
                    }
                }
            ),
            encoding="utf-8",
        )
        metadata = write_pack((("BuildManifest.json", source),), data_root / "AssetCatalog.inxcat")
        (data_root / "PackageIndex.inxmanifest").write_text(
            f"INFERNUX_PLAYER_PACKAGE_INDEX\ncatalog\t{metadata['archive_sha256']}\t{metadata['archive_bytes']}\n",
            encoding="ascii",
        )
        sentinel = output_dir / "preserve.bin"
        sentinel.write_bytes(b"previous product")

        if same_project and same_name:
            builder._validate_output_directory()
        else:
            with pytest.raises(BuildOutputDirectoryError, match="must be empty"):
                builder._validate_output_directory()
        assert sentinel.read_bytes() == b"previous product"
        assert not (data_root / "BuildManifest.json").exists()


class TestGameBuilderDependencyCollection:
    def test_project_build_rejects_an_importable_dependency_with_wrong_version(self, tmp_path, monkeypatch):
        project_root = _make_project(tmp_path)
        (project_root / "ProjectSettings" / "requirements.txt").write_text("demo==2.0\n", encoding="utf-8")
        builder = GameBuilder(str(project_root), str(tmp_path / "build_output"), game_name="TestGame")
        monkeypatch.setattr("infernux.engine._build_dependencies._has_requirement", lambda *_args: False)
        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
        with pytest.raises(RuntimeError, match="do not satisfy project requirements: demo==2.0"):
            _collect_fixture_dependencies(builder)


class TestGameBuilderAutoParallelExport:
    def test_compile_user_scripts_propagates_bytecode_failure(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "build_output"
        assets_dir = output_dir / "Data" / "Assets"
        assets_dir.mkdir(parents=True)
        script_path = assets_dir / "gameplay.py"
        script_path.write_text("score = 1\n", encoding="utf-8")

        builder = _make_builder(tmp_path, output_dir)
        builder.include_jit_runtime = True
        _bind_staged_script_to_asset_index(
            builder,
            output_dir,
            script_path,
            guid="gameplay-script-guid",
        )
        failure = py_compile.PyCompileError(
            SyntaxError,
            SyntaxError("compiler rejected source"),
            str(script_path),
            "compiler rejected source",
        )

        def reject_compile(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(game_builder_module.py_compile, "compile", reject_compile)

        with pytest.raises(py_compile.PyCompileError):
            builder._compile_user_scripts(str(output_dir))

        assert script_path.is_file()
        assert not (assets_dir / "gameplay.pyc").exists()

    @pytest.mark.parametrize(
        "declaration",
        [
            "from infernux import jit\n@jit.compile(cache=True)\n",
        ],
    )
    def test_compile_user_scripts_embeds_auto_parallel_without_sidecar(self, tmp_path, declaration, monkeypatch):
        output_dir = tmp_path / "build_output"
        assets_dir = output_dir / "Data" / "Assets"
        assets_dir.mkdir(parents=True)

        script_path = assets_dir / "stress.py"
        stale_sidecar = assets_dir / "stress.autop.pyc"
        stale_sidecar.write_bytes(b"obsolete")
        script_path.write_text(
            declaration + "def burn(n):\n    acc = 0\n    for i in range(n):\n        acc += i\n    return acc\n",
            encoding="utf-8",
        )

        builder = _make_builder(tmp_path, output_dir)
        builder.include_jit_runtime = True
        _bind_staged_script_to_asset_index(
            builder,
            output_dir,
            script_path,
            guid="auto-parallel-script-guid",
        )
        builder._compile_user_scripts(str(output_dir))

        assert not script_path.exists()
        bytecode_path = assets_dir / "stress.pyc"
        assert bytecode_path.is_file()
        assert not stale_sidecar.exists()

        loader = importlib.machinery.SourcelessFileLoader(
            "infernux_test_embedded_auto_parallel",
            str(bytecode_path),
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        import infernux.application as application

        player_data = tmp_path / "player-data"
        cache = player_data / "Cache" / "Compute" / "CPU"
        cache.mkdir(parents=True)
        # This test exercises the embedded bytecode contract rather than the
        # Player package reader.  A real Player materializes this directory
        # from Content.inxpkg before importing cached JIT declarations.
        (cache / "inx-test-ready.nbi").write_bytes(b"ready")
        monkeypatch.setattr(application, "_runtime_kind", "player")
        monkeypatch.setenv("_INFERNUX_PLAYER_PERSISTENT_DATA_ROOT", str(player_data))
        monkeypatch.setitem(sys.modules, loader.name, module)
        loader.exec_module(module)
        assert module.burn(10) == 45
        assert module.burn.parallel is not module.burn.serial
        manifest = module.__infernux_jit_manifest__["burn"]
        assert len(manifest["hir_fingerprint"]) == 64
        assert module.burn.compiler_fingerprint == manifest["hir_fingerprint"]
        assert list((player_data / "Cache/Compute/CPU").glob("inx-*.nbc"))

    def test_compile_user_scripts_rejects_required_unsafe_kernel(self, tmp_path):
        output_dir = tmp_path / "build_output"
        assets_dir = output_dir / "Data" / "Assets"
        assets_dir.mkdir(parents=True)
        script_path = assets_dir / "unsafe.py"
        script_path.write_text(
            "from infernux import jit\n"
            "@jit.compile(parallel_policy='required')\n"
            "def prefix(values):\n"
            "    for i in range(1, len(values)):\n"
            "        values[i] = values[i - 1] + 1\n",
            encoding="utf-8",
        )

        builder = _make_builder(tmp_path, output_dir)
        builder.include_jit_runtime = True
        _bind_staged_script_to_asset_index(
            builder,
            output_dir,
            script_path,
            guid="unsafe-parallel-script-guid",
        )
        with pytest.raises(RuntimeError, match="compute compilation rejected"):
            builder._compile_user_scripts(str(output_dir))

    def test_compile_user_scripts_without_jit_keeps_cpu_function_as_plain_python(self, tmp_path):
        output_dir = tmp_path / "build_output"
        assets_dir = output_dir / "Data" / "Assets"
        assets_dir.mkdir(parents=True)
        script_path = assets_dir / "compute_user.py"
        script_path.write_text(
            "import infernux as inx\n@inx.jit.compile\ndef square(value):\n    return value * value\n",
            encoding="utf-8",
        )
        builder = _make_builder(tmp_path, output_dir)
        builder.include_jit_runtime = False
        _bind_staged_script_to_asset_index(
            builder,
            output_dir,
            script_path,
            guid="cpu-compute-script-guid",
        )
        builder._compile_user_scripts(str(output_dir))

        assert not script_path.exists()
        bytecode_path = script_path.with_suffix(".pyc")
        assert bytecode_path.is_file()
        loader = importlib.machinery.SourcelessFileLoader("infernux_test_plain_web_jit", str(bytecode_path))
        namespace = {}
        exec(loader.get_code(loader.name), namespace)
        assert namespace["square"](7) == 49
        assert not hasattr(namespace["square"], "serial")
