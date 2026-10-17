from __future__ import annotations
import hashlib as hashlib
import json as json
import os as os
from pathlib import Path as Path
import py_compile as py_compile
import shutil
import subprocess
import sys as sys
import threading as threading
import time
from types import SimpleNamespace as SimpleNamespace
import pytest as pytest
from infernux.engine.build_cancellation import BuildCancelled as BuildCancelled
from infernux.engine import game_builder as game_builder_module
from infernux.engine.game_builder import (
    BuildOutputDirectoryError as BuildOutputDirectoryError,
    GameBuilder as GameBuilder,
)
from infernux.engine.runtime_artifact_catalog import (
    RuntimeArtifactError as RuntimeArtifactError,
    artifact_source_hash as artifact_source_hash,
    build_catalog as build_catalog,
    logical_type_for_path as logical_type_for_path,
    load_asset_index as load_asset_index,
    payload_kind_for as payload_kind_for,
    unix_ns_to_filetime_ticks,
    validate_artifact as validate_artifact,
)
from infernux.engine import nuitka_builder as nuitka_builder_module
from infernux.engine.nuitka_builder import NuitkaBuilder as NuitkaBuilder
from infernux.engine.player_package_native import (
    extract_pack,
    read_entry as read_entry,
    read_manifest as read_manifest,
    set_test_backend,
    write_pack as write_pack,
)
from infernux.engine.player_service_graph import forbidden_player_service_modules as forbidden_player_service_modules
from infernux.lifecycle import PreloadContext
from infernux.particle.asset import ParticleGraphAsset as ParticleGraphAsset
from infernux.plugins.preload import PreloadManager
from infernux.plugins.registry import PluginRegistry


def _stage_document_inputs(builder, data_dir):
    """Prepare the captured source input required by the document cook phase."""
    for entry in builder._cooked_asset_entries.values():
        if "import_document" in entry.get("metadata", {}).get("metadata", {}):
            continue
        source = Path(entry["normalized_path"])
        destination = data_dir / source.relative_to(builder.project_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_player_native_contract(directory: Path) -> None:
    for filename in nuitka_builder_module.player_native_library_filenames():
        path = directory / filename
        if not path.exists():
            path.write_bytes(filename.encode("utf-8"))
    (directory / NuitkaBuilder._PLAYER_NATIVE_CONTRACT_FILENAME).write_text(
        json.dumps(NuitkaBuilder._PLAYER_NATIVE_CONTRACT),
        encoding="utf-8",
    )


class _FakeNativeInxPack:
    """Contract-only backend; production packing remains native C++."""

    manifests: dict[str, dict[str, object]] = {}
    entries: dict[tuple[str, str], bytes] = {}

    @classmethod
    def _key(cls, path) -> str:
        return str(Path(path).resolve())

    @classmethod
    def _write(
        cls,
        sources,
        destination,
        compression_level=None,
        profile="development",
    ):
        archive_key = cls._key(destination)
        cls.entries = {key: value for key, value in cls.entries.items() if key[0] != archive_key}
        records = []
        raw_total = 0
        for offset, (logical, source) in enumerate(sorted(sources)):
            logical = str(logical).replace("\\", "/")
            payload = Path(source).read_bytes()
            records.append(
                {
                    "path": logical,
                    "offset": offset * 64,
                    "stored_bytes": len(payload),
                    "raw_bytes": len(payload),
                    "codec": "store",
                }
            )
            cls.entries[(archive_key, logical)] = payload
            raw_total += len(payload)
        manifest = {
            "format": "infernux-native-inxpack",
            "codec": "store",
            "compression_profile": profile,
            "file_count": len(records),
            "raw_bytes": raw_total,
            "stored_bytes": raw_total,
            "payload_bytes": raw_total,
            "archive_bytes": 128 + len(records) * 64 + raw_total,
            "files": records,
        }
        encoded = json.dumps(manifest, sort_keys=True).encode("utf-8")
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"FAKE-NATIVE-INXPKG\0" + encoded)
        manifest["archive_bytes"] = destination_path.stat().st_size
        manifest["archive_sha256"] = hashlib.sha256(destination_path.read_bytes()).hexdigest()
        cls.manifests[archive_key] = manifest
        return manifest

    @staticmethod
    def _validate_logical_path(logical: str) -> str:
        normalized = str(logical).replace("\\", "/")
        parts = normalized.split("/")
        if (
            not normalized
            or normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or ":" in parts[0]
        ):
            raise RuntimeError(f"unsafe native package path: {logical}")
        return normalized

    @classmethod
    def _read_manifest(cls, path):
        archive_key = cls._key(path)
        try:
            return cls.manifests[archive_key]
        except KeyError:
            try:
                archive_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError as exc:
                raise RuntimeError(f"fake native package is unavailable: {path}") from exc
            source_key = next(
                (key for key, manifest in cls.manifests.items() if manifest.get("archive_sha256") == archive_hash),
                None,
            )
            if source_key is None:
                raise RuntimeError(f"fake native package is unavailable: {path}")
            cls.manifests[archive_key] = cls.manifests[source_key]
            for (entry_archive, logical), payload in list(cls.entries.items()):
                if entry_archive == source_key:
                    cls.entries[(archive_key, logical)] = payload
            return cls.manifests[archive_key]

    @classmethod
    def _read_entry(cls, path, entry_path):
        cls._read_manifest(path)
        return cls.entries[(cls._key(path), str(entry_path).replace("\\", "/"))]

    @classmethod
    def _extract(cls, path, destination, allowed_roots=None):
        manifest = cls._read_manifest(path)
        roots = set(allowed_roots or [])
        for item in manifest["files"]:
            logical = cls._validate_logical_path(str(item["path"]))
            if roots and logical.split("/", 1)[0] not in roots:
                raise RuntimeError(f"unexpected native package root: {logical}")
            target = Path(destination) / Path(logical)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(cls._read_entry(path, logical))
        return manifest

    _inxpack_write = _write
    _inxpack_read_manifest = _read_manifest
    _inxpack_read_entry = _read_entry
    _inxpack_extract = _extract


@pytest.fixture(autouse=True)
def _native_package_backend():
    _FakeNativeInxPack.manifests.clear()
    _FakeNativeInxPack.entries.clear()
    set_test_backend(_FakeNativeInxPack)
    yield
    set_test_backend(None)


def _make_project(tmp_path):
    project_root = tmp_path / "project"
    settings_dir = project_root / "ProjectSettings"
    settings_dir.mkdir(parents=True)
    scene_path = project_root / "Assets" / "Main.scene"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text(
        json.dumps(
            {"identity_format": "guid-v1", "name": "Main", "isPlaying": False, "objects": []}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    (settings_dir / "BuildSettings.json").write_text(
        json.dumps({"scene_guids": ["scene-guid"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_asset_index(
        project_root,
        [_asset_index_entry(project_root, scene_path, "scene-guid", "", "Scene")],
    )
    return project_root


def _collect_fixture_dependencies(builder):
    # Scanner unit fixtures explicitly provide compiled-source inputs. Full
    # staging/compilation coverage lives in test_player_dependency_closure.py.
    sources = sorted((Path(builder.project_path) / "Assets").rglob("*.py"))
    builder._player_python_source_paths = {str(path.resolve()) for path in sources}
    builder._player_python_source_texts = {str(path.resolve()): path.read_text(encoding="utf-8") for path in sources}
    return builder._collect_user_dependencies()


def _make_builder(tmp_path, output_dir):
    project_root = _make_project(tmp_path)
    return GameBuilder(str(project_root), str(output_dir), game_name="TestGame")


def _scene_with_component(fields, *, type_id="python:cook-script:cook-fixture:CookFixture:CookFixture"):
    """Identity-complete authoring input for dependency/cook unit tests."""
    return {
        "identity_format": "guid-v1",
        "name": "Main",
        "isPlaying": False,
        "objects": [
            {
                "id": "1" * 32,
                "name": "CookFixture",
                "active": True,
                "is_static": False,
                "tag": "Untagged",
                "layer": 0,
                "children": [],
                "transform": {
                    "component_id": "2" * 32,
                    "type": "Transform",
                    "enabled": True,
                    "execution_order": 0,
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "scale": [1, 1, 1],
                },
                "components": [
                    {
                        "component_id": "3" * 32,
                        "type_id": type_id,
                        "enabled": True,
                        "execution_order": 0,
                        "data": fields,
                    }
                ],
            }
        ],
    }


def _read_runtime_catalog(data_root: Path, builder: GameBuilder) -> dict:
    return json.loads(
        read_entry(
            data_root / builder._ASSET_CATALOG_ARCHIVE_FILENAME,
            "RuntimeAssetCatalog.json",
        ).decode("utf-8")
    )


def _player_executable_name(game_name: str = "TestGame") -> str:
    return f"{game_name}.exe" if sys.platform == "win32" else game_name


def _write_player_executable(root: Path, payload: bytes = b"Infernux Player") -> Path:
    executable = root / _player_executable_name()
    executable.write_bytes(payload)
    if sys.platform != "win32":
        executable.chmod(executable.stat().st_mode | 0o111)
    return executable


def _bind_staged_script_to_asset_index(
    builder: GameBuilder,
    output_dir: Path,
    staged_script: Path,
    *,
    guid: str,
) -> None:
    runtime_relative = staged_script.relative_to(output_dir / "Data")
    project_source = Path(builder.project_path) / runtime_relative
    project_source.parent.mkdir(parents=True, exist_ok=True)
    project_source.write_bytes(staged_script.read_bytes())
    entries = dict(getattr(builder, "_cooked_asset_entries", {}))
    entries[guid] = {
        "guid": guid,
        "normalized_path": project_source.resolve().as_posix(),
    }
    builder._cooked_asset_entries = entries


def _prepare_runtime_catalog_inputs(
    builder: GameBuilder,
    final_dir: Path,
    *,
    include_runtime: bool = True,
    include_content: bool = True,
    include_executable: bool = True,
) -> Path:
    data_root = final_dir / f"{builder.project_name}_Data"
    data_root.mkdir(parents=True, exist_ok=True)
    source_root = final_dir.parent / f"{final_dir.name}-catalog-sources"
    source_root.mkdir(parents=True, exist_ok=True)
    if include_runtime:
        runtime_source = source_root / "runtime.bin"
        runtime_source.write_bytes(b"runtime")
        write_pack(
            (("infernux/resources/runtime.bin", runtime_source),),
            data_root / builder._RUNTIME_ARCHIVE_FILENAME,
        )
    if include_content:
        content_source = source_root / "content.bin"
        content_source.write_bytes(b"content")
        write_pack(
            (("RuntimeAssets/content.bin", content_source),),
            data_root / builder._CONTENT_ARCHIVE_FILENAME,
        )
    if include_executable:
        executable = final_dir / _player_executable_name(builder.project_name)
        executable.write_bytes(b"Infernux Player")
        if sys.platform != "win32":
            executable.chmod(executable.stat().st_mode | 0o111)
    (data_root / "BuildManifest.json").write_text(
        json.dumps({"game_name": builder.project_name, "scene_guids": ["scene-guid"]}),
        encoding="utf-8",
    )
    return data_root


def _install_runtime_identity_bindings(
    builder: GameBuilder,
    records: dict[str, tuple[str, str]],
) -> None:
    builder._runtime_asset_identity_bindings = {
        runtime_path: {
            "source_guid": guid,
            "source_path": runtime_path,
            "source_fingerprint": {
                "size": 1,
                "modified_ns": 1,
                "content_hash": "a" * 16,
            },
            "dependencies": [],
            "runtime_artifact_reason": reason,
        }
        for runtime_path, (guid, reason) in records.items()
    }


def _write_asset_script(project_root, relative_path: str, source: str) -> None:
    script_path = project_root / "Assets" / relative_path
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(source, encoding="utf-8")


RUNTIME_DOCUMENT_AND_AUDIO_SUFFIXES = (
    ".scene",
    ".prefab",
    ".mat",
    ".effect",
    ".effectgroup",
    ".timeline",
    ".timelinefsm",
    ".animclip",
    ".animclip2d",
    ".animclip3d",
    ".animfsm",
    ".animtimeline",
    ".inxdata",
    ".graph",
    ".particlegraph",
    ".json",
    ".yaml",
    ".yml",
    ".bin",
    ".wav",
    ".ogg",
    ".mp3",
    ".flac",
    ".aiff",
    ".aif",
)


def _reference_particle_graph(project_root: Path, stable_id: str) -> Path:
    graph_path = project_root / "Assets" / "VFX" / f"{stable_id}.particlegraph"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph = ParticleGraphAsset(stable_id=stable_id, name=stable_id)
    graph_path.write_text(graph.canonical_json(), encoding="utf-8")
    guid = hashlib.md5(stable_id.encode("utf-8")).hexdigest()
    scene_path = project_root / "Assets" / "Main.scene"
    scene_path.write_text(
        json.dumps(
            _scene_with_component(
                {
                    "graph": {
                        "$type": "asset_ref",
                        "asset_type": "ParticleGraph",
                        "guid": guid,
                        "path_hint": f"Assets/VFX/{stable_id}.particlegraph",
                    }
                }
            )
        ),
        encoding="utf-8",
    )
    _write_asset_index(
        project_root,
        [
            _asset_index_entry(project_root, scene_path, "scene-guid", "", "Scene"),
            _asset_index_entry(project_root, graph_path, guid, "", "ParticleGraph"),
        ],
    )
    runtime_index = project_root / "Library" / "Artifacts" / "Particle" / "RuntimeIndex.json"
    runtime_index.parent.mkdir(parents=True, exist_ok=True)
    runtime_index.write_text(
        json.dumps(
            {
                "$schema": "infernux.particle_runtime_index",
                "entries": [
                    {
                        "guid": guid,
                        "path_hint": f"Assets/VFX/{stable_id}.particlegraph",
                        "stable_id": stable_id,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return graph_path


def _asset_index_entry(
    project_root: Path,
    source: Path,
    guid: str,
    artifact_path: str,
    resource_type: str,
    content_hash: str | None = None,
) -> dict:
    stat = source.stat()
    normalized_path = str(source.resolve()).replace("\\", "/")
    if sys.platform == "win32":
        normalized_path = normalized_path.casefold()
    return {
        "normalized_path": normalized_path,
        "guid": guid,
        "resource_type": 3,
        "source": {
            "size": stat.st_size,
            "modified_ns": unix_ns_to_filetime_ticks(stat.st_mtime_ns),
        },
        "meta": {"size": 0, "modified_ns": 0},
        "content_hash": content_hash or _fnv1a64(source.read_bytes()),
        "dependencies": [],
        "read_only": False,
        "import_succeeded": True,
        "import_error": "",
        "artifact_path": artifact_path,
        "metadata": {
            "metadata": {
                "guid": {"value": guid},
                "resource_type": {"value": resource_type},
            }
        },
    }


def _write_asset_index(project_root: Path, entries: list[dict]) -> None:
    index = {
        "project_root": str(project_root.resolve()).replace("\\", "/").casefold(),
        "entries": entries,
    }
    index_path = project_root / "Library" / "AssetIndex.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")


def _fnv1a64(payload: bytes) -> str:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _write_texture_asset_index(project_root: Path, source: Path, guid: str, artifact_path: str):
    relative_source = source.resolve().relative_to(project_root.resolve()).as_posix()
    scene_path = project_root / "Assets" / "Main.scene"
    scene_path.write_text(
        json.dumps(
            _scene_with_component(
                {
                    "texture": {
                        "$type": "asset_ref",
                        "asset_type": "Texture",
                        "guid": guid,
                        "path_hint": relative_source,
                    }
                }
            )
        ),
        encoding="utf-8",
    )
    _write_asset_index(
        project_root,
        [
            _asset_index_entry(project_root, scene_path, "scene-guid", "", "Scene"),
            _asset_index_entry(project_root, source, guid, artifact_path, "Texture", "a" * 16),
        ],
    )


def _particle_source_hash(source: Path) -> str:
    graph = ParticleGraphAsset.from_json(source.read_text(encoding="utf-8"))
    return hashlib.sha256(graph.canonical_json().encode("utf-8")).hexdigest()


def _write_particle_artifact(path: Path, source: Path, *, emitters=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "$schema": "infernux.particle_artifact",
                "source_hash": _particle_source_hash(source),
                "kernel_ir": {"emitters": list(emitters or [])},
            }
        ),
        encoding="utf-8",
    )


def _write_texture_artifact(path: Path, payload: bytes, content_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"INXTEXTURE"
        + b"\x04\x03\x02\x01"
        + len(content_hash).to_bytes(4, "little")
        + content_hash.encode("ascii")
        + payload
    )


def _write_animation_texture_asset(
    texture_path: Path,
    *,
    guid: str,
    sprite_frame_ids=(),
    sprite: bool = True,
) -> None:
    from infernux.core.asset_types import SpriteFrame, TextureImportSettings, TextureType

    texture_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path.write_bytes(b"test texture")
    settings = TextureImportSettings(
        texture_type=TextureType.SPRITE if sprite else TextureType.DEFAULT,
        sprite_frames=[
            SpriteFrame(stable_id=stable_id, name=f"frame_{index}", w=16, h=16)
            for index, stable_id in enumerate(sprite_frame_ids)
        ],
    )

    def tagged(value):
        if type(value) is bool:
            tag = "bool"
        elif type(value) is int:
            tag = "int"
        elif type(value) is list:
            tag = "json_array"
        else:
            tag = "string"
        return {"type": tag, "value": value}

    metadata = {"guid": tagged(guid)}
    metadata.update({key: tagged(value) for key, value in settings.to_dict().items()})
    texture_path.with_name(texture_path.name + ".meta").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )


def _write_animation_clip(
    clip_path: Path,
    *,
    texture_guid: str,
    sprite_frame_id: str,
) -> None:
    from infernux.core.animation_clip import AnimationClip, AnimationFrame

    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip = AnimationClip(
        name=clip_path.stem,
        authoring_texture_guid=texture_guid,
        frames=[AnimationFrame(sprite_frame_id=sprite_frame_id)],
    )
    clip_path.write_text(json.dumps(clip.to_dict(), indent=2), encoding="utf-8")


def _make_license_distribution(site, package="llvmlite", *, modern=True):
    package_root = site / package
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    # Import name and distribution name need not be the same.
    metadata = site / f"vendor_{package}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: vendor-{package}\nVersion: 1.0\n",
        encoding="utf-8",
    )
    relative = "licenses/vendor/copyright.txt" if modern else "LICENSE.thirdparty"
    notice = metadata / relative
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text("Original copyright — 保留许可\n", encoding="utf-8")
    (metadata / "RECORD").write_text(
        f"{package}/__init__.py,,\n{metadata.name}/METADATA,,\n{metadata.name}/{relative},,\n",
        encoding="utf-8",
    )
    return notice, Path(package) / "_licenses" / f"vendor_{package}-1.0" / relative


def _write_scene_material_audio_reachability_fixture(
    builder: GameBuilder,
    *,
    include_unreachable: bool,
) -> dict[str, Path]:
    project = Path(builder.project_path)
    assets = project / "Assets"
    scene = assets / "Main.scene"
    material = assets / "Materials" / "Bird.mat"
    audio = assets / "Audio" / "Wing.wav"
    unreachable = assets / "Unused.mat"
    material.parent.mkdir(parents=True, exist_ok=True)
    audio.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(
        json.dumps(
            _scene_with_component(
                {
                    "material": {
                        "$type": "asset_ref",
                        "guid": "material-guid",
                        "path_hint": "Assets/Materials/Bird.mat",
                    }
                }
            )
        ),
        encoding="utf-8",
    )
    material.write_text(
        json.dumps(
            {
                "audio": {
                    "$type": "asset_ref",
                    "guid": "audio-guid",
                    "path_hint": "Assets/Audio/Wing.wav",
                }
            }
        ),
        encoding="utf-8",
    )
    import wave

    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(b"\x00\x00" * 32)
    entries = [
        _asset_index_entry(project, scene, "scene-guid", "", "Scene"),
        _asset_index_entry(project, material, "material-guid", "", "Material"),
        _asset_index_entry(project, audio, "audio-guid", "", "Audio"),
    ]
    if include_unreachable:
        unreachable.write_text("{}", encoding="utf-8")
        entries.append(
            _asset_index_entry(
                project,
                unreachable,
                "unused-guid",
                "",
                "Material",
            )
        )
    _write_asset_index(project, entries)
    (project / "ProjectSettings" / "BuildSettings.json").write_text(
        json.dumps({"scene_guids": ["scene-guid"]}),
        encoding="utf-8",
    )
    return {
        "scene": scene,
        "material": material,
        "audio": audio,
        "unreachable": unreachable,
    }


def test_builder_resolves_relative_output_against_project_after_cwd_changes(monkeypatch, tmp_path):
    project = _make_project(tmp_path)
    elsewhere = tmp_path / "Elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    builder = GameBuilder(
        str(project), r"Builds\Windows", player_runtime_root="Packages/infernux/platform-windows/player"
    )
    assert builder.output_dir == str((project / "Builds" / "Windows").resolve())
    assert builder.player_runtime_root == str((project / "Packages/infernux/platform-windows/player").resolve())


def test_relative_file_import_keeps_guid_after_cook_pack_and_project_move(monkeypatch, tmp_path):
    from infernux.application import Application
    from infernux.core.assets import AssetManager
    from infernux.engine import project_context
    from infernux.engine.path_utils import same_path
    from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog

    builder = _make_builder(tmp_path, tmp_path / "Output")
    project = Path(builder.project_path)
    source = project / "Assets" / "Data" / "配置.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"weapon":"bow","damage":42}', encoding="utf-8")
    guid = "0123456789abcdef0123456789abcdef"
    authored = "Assets/Data/配置.json"
    database = SimpleNamespace(
        get_all_guids=lambda: [guid],
        get_guid_from_path=lambda path: guid if same_path(path, source) else "",
        get_guid_from_catalog_path=lambda path: guid if same_path(path, source) else "",
        get_path_from_guid=lambda identity: str(source) if identity == guid else "",
    )
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: True))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "data_path", staticmethod(lambda: str(project)))
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    before = AssetManager.find_assets(authored)
    original = before[0].read_text()
    assert before[0]._guid == guid
    assert AssetManager.load("Assets/Data") is None

    builder._cooked_asset_entries = {
        guid: _asset_index_entry(project, source, guid, "", "DefaultText"),
    }
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    final = tmp_path / "CookedBuild"
    data_root = final / "Data"
    _stage_document_inputs(builder, data_root)
    builder._stage_library_runtime_documents(str(data_root))
    builder._write_runtime_asset_records(str(final))
    records = json.loads((data_root / "Library/RuntimeAssetRecords.json").read_text(encoding="utf-8"))
    entries = []
    packed = []
    for runtime_path, binding in builder._runtime_asset_identity_bindings.items():
        payload = data_root / runtime_path
        entries.append(
            {
                "package": "Content.inxpkg",
                "runtime_path": runtime_path,
                "bytes": payload.stat().st_size,
                "payload": payload.read_bytes(),
                "asset_binding": binding,
            }
        )
        packed.append((runtime_path, payload))
    catalog_document = build_catalog(entries, player_host={"executable": "Game"}, package_records=[])
    assert records["entries"][0]["runtime_path"] == authored
    assert str(project) not in json.dumps(records, ensure_ascii=False)
    archive = tmp_path / "Content.inxpkg"
    write_pack(packed, archive)

    relocated = tmp_path / "NewDevice" / "PlayerData"
    extract_pack(archive, relocated)
    project.rename(tmp_path / "MovedAuthorProject")
    assert not source.exists()
    runtime = PlayerRuntimeAssetCatalog.from_documents(str(relocated), catalog_document, records)
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(project_context._context(), "asset_query", runtime.query_asset_guids)
    monkeypatch.setattr(project_context._context(), "asset_resolver", runtime.resolve_guid)
    monkeypatch.setattr(AssetManager, "_asset_database", None)
    after = AssetManager.find_assets(authored)
    assert after[0]._guid == guid
    assert after[0].read_text() == original
    assert project_context.resolve_asset_path(authored) == runtime.resolve_guid(guid)
    assert AssetManager.find_assets(r"Assets\Data\配置.json")[0].read_text() == original
    assert AssetManager.load("Assets/Data") is None
    assert project_context.resolve_asset_path("Assets/Data") is None
    assert AssetManager.load("Assets/Data/*.json") is None


@pytest.mark.parametrize("new_runtime_files", [False, True])
@pytest.mark.parametrize("import_style", ["from", "dotted", "alias"])
def test_player_stages_enabled_package_runtime_by_guid_and_excludes_editor(tmp_path, new_runtime_files, import_style):
    project = _make_project(tmp_path)
    runtime = project / "Packages/vendor/gameplay/runtime/lifecycle.py"
    resource = project / "Packages/vendor/gameplay/runtime/message.txt"
    editor = project / "Packages/vendor/gameplay/editor/panel.py"
    content = project / "Assets/Plugins/Scenes/Demo.scene"
    control = project / "Packages/vendor/gameplay/inx_package.json"
    statement, base = {
        "from": ("from infernux.lifecycle import InxPreload", "InxPreload"),
        "dotted": ("import infernux.lifecycle", "infernux.lifecycle.InxPreload"),
        "alias": ("import infernux.lifecycle as lifecycle", "lifecycle.InxPreload"),
    }[import_style]
    lifecycle_source = (
        f"{statement}\nclass RuntimeLifecycle({base}):\n"
        "    def preload(self, context):\n"
        "        self.message = context.package_path('runtime/message.txt')\n"
    ).encode("utf-8")
    files = (
        (runtime, "runtime-guid", "runtime/lifecycle.py", "runtime", lifecycle_source),
        (
            resource,
            "resource-guid",
            "runtime/message.txt",
            "runtime",
            "Player package resource ready\n".encode("utf-8"),
        ),
        (editor, "editor-guid", "editor/panel.py", "editor", b"PANEL = True\n"),
        (content, "content-guid", "Scenes/Demo.scene", "content", b"{}\n"),
    )
    records = []
    for path, guid, logical, role, payload in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        Path(str(path) + ".meta").write_text(
            json.dumps({"metadata": {"guid": {"type": "string", "value": guid}}}),
            encoding="utf-8",
        )
        records.append(
            {
                "logical_path": logical,
                "path_hint": path.relative_to(project).as_posix(),
                "guid": guid,
                "role": role,
                "owned": True,
            }
        )
    control.parent.mkdir(parents=True, exist_ok=True)
    control_payload = b"{}\n"
    control.write_bytes(control_payload)
    Path(str(control) + ".meta").write_text(
        json.dumps({"metadata": {"guid": {"type": "string", "value": "control-guid"}}}),
        encoding="utf-8",
    )
    registry = PluginRegistry(str(project))
    registry.record_install(
        {"reference": "vendor/gameplay", "name": "Gameplay", "version": "1.0"},
        files=records,
        control={
            "logical_path": "inx_package.json",
            "path_hint": control.relative_to(project).as_posix(),
            "guid": "control-guid",
            "role": "control",
            "owned": True,
        },
        package_path="C:/Users/Author/.cache/private.inxpkg",
        source={
            "type": "local",
            "location": "C:/Users/Author/source/plugin.inxpkg",
        },
    )
    original_registry = registry.load()
    if new_runtime_files:
        # Files authored after installation are project assets, not new
        # uninstall ownership. They must nevertheless reach the Player.
        additions = (
            (
                runtime.parent / "new_component.py",
                "new-script-guid",
                "runtime/new_component.py",
                "runtime",
                b"VALUE = 42\n",
            ),
            (
                resource.parent / "new_data.json",
                "new-data-guid",
                "runtime/new_data.json",
                "runtime",
                b'{"value": 42}\n',
            ),
            (
                editor.parent / "new_panel.py",
                "new-editor-guid",
                "editor/new_panel.py",
                "editor",
                b"EDITOR = True\n",
            ),
        )
        for path, guid, _logical, _role, payload in additions:
            path.write_bytes(payload)
            Path(str(path) + ".meta").write_text(
                json.dumps({"metadata": {"guid": {"type": "string", "value": guid}}}),
                encoding="utf-8",
            )
        files += additions
    scene = project / "Assets/Main.scene"
    _write_asset_index(
        project,
        [
            _asset_index_entry(project, scene, "scene-guid", "", "Scene"),
            *[
                _asset_index_entry(project, path, guid, "", "Script" if role != "content" else "Scene")
                for path, guid, _logical, role, _payload in files
            ],
        ],
    )
    output = tmp_path / "build"
    data = output / "Data"
    data.mkdir(parents=True)
    builder = GameBuilder(str(project), str(output), game_name="PackageGame")

    builder._stage_player_plugins(str(data))

    staged_runtime = data / "Packages/vendor/gameplay/runtime/lifecycle.py"
    assert staged_runtime.read_bytes() == lifecycle_source
    staged_resource = data / "Packages/vendor/gameplay/runtime/message.txt"
    assert staged_resource.read_text(encoding="utf-8") == "Player package resource ready\n"
    assert not (data / "Packages/vendor/gameplay/editor/panel.py").exists()
    if new_runtime_files:
        assert (data / "Packages/vendor/gameplay/runtime/new_component.py").read_bytes() == b"VALUE = 42\n"
        assert (data / "Packages/vendor/gameplay/runtime/new_data.json").is_file()
        assert not (data / "Packages/vendor/gameplay/editor/new_panel.py").exists()
    assert registry.load() == original_registry
    shipped = json.loads((data / "ProjectSettings/InxPlugins.json").read_text(encoding="utf-8"))
    assert [item["logical_path"] for item in shipped["installed"][0]["files"]] == [
        "runtime/lifecycle.py",
        "runtime/message.txt",
        "Scenes/Demo.scene",
    ] + (["runtime/new_component.py", "runtime/new_data.json"] if new_runtime_files else [])
    if new_runtime_files:
        assert all(not item["owned"] for item in shipped["installed"][0]["files"][-2:])
    assert "package_path" not in shipped["installed"][0]
    assert "source" not in shipped["installed"][0]
    assert "installed_at" not in shipped["installed"][0]
    assert "C:/Users/Author" not in json.dumps(shipped)

    context = PreloadContext(
        project_root=str(data),
        source_path=str(staged_runtime),
        script_guid="runtime-guid",
        type_id="fixture-preload",
        package_reference="vendor/gameplay",
        runtime=True,
    )
    assert context.package_path("runtime/message.txt") == str(staged_resource.resolve())
    assert (
        Path(context.package_path("runtime/message.txt")).read_text(encoding="utf-8")
        == "Player package resource ready\n"
    )
    assert not builder._runtime_catalog_payload_required("ProjectSettings/InxPlugins.json")

    builder._compile_player_plugin_scripts(str(output))
    assert not staged_runtime.exists()
    assert staged_runtime.with_suffix(".pyc").is_file()
    assert staged_resource.is_file()
    if new_runtime_files:
        new_script = data / "Packages/vendor/gameplay/runtime/new_component.py"
        assert not new_script.exists()
        assert new_script.with_suffix(".pyc").is_file()
        assert {"new-script-guid", "new-data-guid"} <= builder._staged_player_plugin_guids
        assert {"new-script-guid", "new-data-guid"} <= builder._cooked_asset_entries.keys()
    runtime_registry = json.loads((data / "ProjectSettings/InxPlugins.json").read_text(encoding="utf-8"))
    lifecycle_record = runtime_registry["installed"][0]["files"][0]
    assert lifecycle_record["compiled_path_hint"].endswith("runtime/lifecycle.pyc")
    assert lifecycle_record["preload_declarations"] == [
        {
            "name": "RuntimeLifecycle",
            "bases": ["infernux.lifecycle.InxPreload"],
        }
    ]

    manager = PreloadManager(str(data), runtime=True)
    states = manager.reload_all()
    assert len(states) == 1
    assert states[0].loaded is True
    assert states[0].error == ""
    assert states[0].instance is not None
    assert states[0].instance.message == str(staged_resource.resolve())
    assert manager.unload_all() == ()

    registry.set_enabled("vendor/gameplay", False)
    disabled_data = tmp_path / "disabled-build" / "Data"
    builder._stage_player_plugins(str(disabled_data))
    assert not (disabled_data / "Packages").exists()
    assert builder._staged_player_plugin_guids == set()
    assert (
        json.loads((disabled_data / "ProjectSettings/InxPlugins.json").read_text(encoding="utf-8"))["installed"] == []
    )


@pytest.mark.parametrize(
    "reference,authored_manifest",
    [
        ("local_probe", False),
        ("local_probe", True),
        ("vendor/local_probe", True),
    ],
)
def test_player_exports_local_author_package_without_installing(tmp_path, reference, authored_manifest):
    project = _make_project(tmp_path)
    root = project / "Packages" / reference
    runtime = root / "runtime/lifecycle.py"
    message = root / "runtime/message.txt"
    editor = root / "editor/panel.py"
    sources = (
        (
            runtime,
            "local-script-guid",
            (
                "from infernux.lifecycle import InxPreload\n"
                "from pathlib import Path\n"
                "class LocalLifecycle(InxPreload):\n"
                "    def preload(self, context):\n"
                "        self.message = Path(context.package_path('runtime/message.txt')).read_text()\n"
            ),
        ),
        (message, "local-text-guid", "Local author resource ready"),
        (editor, "local-editor-guid", "raise RuntimeError('Editor must not enter Player')\n"),
    )
    for path, guid, payload in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        Path(str(path) + ".meta").write_text(
            json.dumps({"metadata": {"guid": {"type": "string", "value": guid}}}),
            encoding="utf-8",
        )
    if authored_manifest:
        (root / "inx_package.json").write_text(
            json.dumps(
                {
                    "reference": "future/distribution-name",
                    "name": "Local Author",
                    "version": "1.2.3",
                }
            ),
            encoding="utf-8",
        )
    registry = PluginRegistry(str(project))
    original_registry = registry.load()
    _write_asset_index(
        project,
        [
            _asset_index_entry(project, path, guid, "", "Script" if path.suffix == ".py" else "Binary")
            for path, guid, _payload in sources
        ],
    )
    output = tmp_path / "build"
    data = output / "Data"
    builder = GameBuilder(str(project), str(output), game_name="LocalAuthorGame")

    builder._stage_player_plugins(str(data))

    staged = data / "Packages" / reference
    assert (staged / "runtime/lifecycle.py").is_file()
    assert (staged / "runtime/message.txt").read_text() == "Local author resource ready"
    assert not (staged / "editor").exists()
    assert not (staged / "inx_package.json").exists()
    assert registry.load() == original_registry
    assert builder._staged_player_plugin_guids == {"local-script-guid", "local-text-guid"}
    assert builder._cooked_asset_entries.keys() == builder._staged_player_plugin_guids
    shipped = PluginRegistry(str(data)).load()["installed"]
    assert len(shipped) == 1
    assert shipped[0]["reference"] == reference
    assert shipped[0]["name"] == ("Local Author" if authored_manifest else "local_probe")
    assert shipped[0]["version"] == ("1.2.3" if authored_manifest else "0.0.0")
    assert not shipped[0]["control"]["owned"]
    assert all(not item["owned"] for item in shipped[0]["files"])
    builder._compile_player_plugin_scripts(str(output))
    assert not (staged / "runtime/lifecycle.py").exists()
    assert (staged / "runtime/lifecycle.pyc").is_file()
    manager = PreloadManager(str(data), runtime=True)
    try:
        states = manager.reload_all()
        assert len(states) == 1
        assert states[0].loaded, states[0].error
        assert states[0].instance.message == "Local author resource ready"
    finally:
        manager.unload_all()


def test_player_plugin_component_joins_runtime_type_and_guid_catalogs(tmp_path):
    from infernux.components import InxComponent, serialized_field
    from infernux.components.component_identity import bind_asset_script_guid
    from infernux.engine.project_context import get_script_module_name

    project = _make_project(tmp_path)
    authored = project / "Packages/vendor/component-kit/runtime/plugin_component.py"
    authored.parent.mkdir(parents=True)
    authored_source = (
        "import infernux as inx\n"
        "InxComponent = inx.InxComponent\n"
        "class PluginMover(InxComponent):\n"
        "    speed: float = inx.serialized_field(default=4.0)\n"
        "    def start(self):\n"
        "        pass\n"
    )
    authored.write_text(authored_source, encoding="utf-8")
    (authored.parents[1] / "inx_package.json").write_text(
        json.dumps({"reference": "vendor/component-kit"}), encoding="utf-8"
    )
    script_guid = "4d9c67b9b82d4a3292819c56980bc101"
    module_name = get_script_module_name(str(authored), str(project))
    assert module_name == "_infernux_packages.vendor.component_2dkit.runtime.plugin_component"

    class PluginMover(InxComponent):
        speed: float = serialized_field(default=4.0)

        def start(self):
            pass

    PluginMover.__module__ = module_name
    PluginMover.__qualname__ = "PluginMover"
    type_guid = bind_asset_script_guid(PluginMover, script_guid)

    output = tmp_path / "build"
    data = output / "Data"
    staged = data / "Packages/vendor/component-kit/runtime/plugin_component.py"
    staged.parent.mkdir(parents=True)
    staged.write_text(authored_source, encoding="utf-8")
    registry_path = data / "ProjectSettings/InxPlugins.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "installed": [
                    {
                        "reference": "vendor/component-kit",
                        "enabled": True,
                        "files": [
                            {
                                "logical_path": "runtime/plugin_component.py",
                                "path_hint": "Packages/vendor/component-kit/runtime/plugin_component.py",
                                "guid": script_guid,
                                "role": "runtime",
                                "owned": True,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    builder = GameBuilder(str(project), str(output), game_name="PluginComponentGame")
    builder._cooked_asset_entries = {script_guid: _asset_index_entry(project, authored, script_guid, "", "Script")}

    builder._compile_player_plugin_scripts(str(output))

    guid_map = json.loads((data / "_script_guid_map.json").read_text(encoding="utf-8"))
    assert guid_map[script_guid] == ("Packages/vendor/component-kit/runtime/plugin_component.pyc")
    catalog = json.loads((data / "Library/RuntimeTypeRegistry.json").read_text(encoding="utf-8"))
    record = next(item for item in catalog["types"] if item["type_guid"] == type_guid)
    assert record["module"] == module_name
    assert record["qualname"] == "PluginMover"
    assert record["lifecycle"] == ["start"]
    assert record["semantic"]["fields"][0]["attributes"]["default"] == 4.0
    assert staged.with_suffix(".pyc").is_file()
    assert not staged.exists()


class TestGameBuilderAnimationClipPreflight:
    TEXTURE_GUID = "b" * 32
    FRAME_ID = "3" * 32

    def test_validate_runs_animation_preflight_and_resolves_texture_by_guid(self, tmp_path):
        project = _make_project(tmp_path)
        texture = project / "Assets" / "Sprites" / "sheet.png"
        _write_animation_texture_asset(
            texture,
            guid=self.TEXTURE_GUID,
            sprite_frame_ids=(self.FRAME_ID,),
        )
        _write_animation_clip(
            project / "Assets" / "Animations" / "walk.animclip2d",
            texture_guid=self.TEXTURE_GUID,
            sprite_frame_id=self.FRAME_ID,
        )
        scene = project / "Assets" / "Main.scene"
        _write_asset_index(
            project,
            [
                _asset_index_entry(project, scene, "scene-guid", "", "Scene"),
                _asset_index_entry(
                    project,
                    texture,
                    self.TEXTURE_GUID,
                    "",
                    "Texture",
                ),
            ],
        )
        builder = GameBuilder(str(project), str(tmp_path / "build_output"), game_name="TestGame")

        builder._validate()


def test_requested_build_scene_guids_override_disk_and_remain_a_snapshot(tmp_path):
    project = _make_project(tmp_path)
    selected = ["requested-guid", "scene-guid"]
    builder = GameBuilder(str(project), str(tmp_path / "output"), build_scene_guids=selected)
    selected.clear()
    final = tmp_path / "dist"
    settings = final / "Data/ProjectSettings/BuildSettings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text('{"scene_guids":["scene-guid"]}', encoding="utf-8")
    builder._relativize_scenes(str(final))
    assert json.loads(settings.read_text(encoding="utf-8")) == {
        "scene_guids": ["requested-guid", "scene-guid"],
    }
    assert json.loads((project / "ProjectSettings/BuildSettings.json").read_text(encoding="utf-8")) == {
        "scene_guids": ["scene-guid"],
    }


def test_build_cancellation_is_not_reported_as_a_build_failure(tmp_path, monkeypatch):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    error_messages: list[str] = []
    build_log_dir = tmp_path / "build-log"
    monkeypatch.setattr(
        game_builder_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: build_log_dir.mkdir() or str(build_log_dir),
    )
    monkeypatch.setattr(builder, "_build_inner", lambda *_args, **_kwargs: (_ for _ in ()).throw(BuildCancelled()))
    monkeypatch.setattr(game_builder_module.Debug, "log_error", error_messages.append)

    with pytest.raises(BuildCancelled):
        builder.build()

    build_log = (build_log_dir / "build.log").read_text(encoding="utf-8")
    assert "Build cancelled by user." in build_log
    assert "BUILD FAILED" not in build_log
    assert error_messages == []
    assert not (tmp_path / "project" / "Logs").exists()


def test_nuitka_cancellation_does_not_wait_for_the_next_stdout_line(tmp_path, monkeypatch):
    monkeypatch.setattr(nuitka_builder_module, "_ensure_windows_msvc_environment", lambda env: env)
    builder = object.__new__(NuitkaBuilder)
    builder._build_cache_root = str(tmp_path)
    builder._staging_dir = str(tmp_path)
    cancelled = threading.Event()
    cancelled.set()

    started = time.perf_counter()
    with pytest.raises(BuildCancelled):
        builder._run_nuitka(
            [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
            on_progress=None,
            cancel_event=cancelled,
        )

    assert time.perf_counter() - started < 2.5


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction contract")
def test_nuitka_unicode_build_cache_uses_temporary_ascii_junction(tmp_path):
    cache_root = tmp_path / "项目" / "Cache" / "Build" / "Desktop"
    alias = nuitka_builder_module._windows_ascii_build_alias(str(cache_root))
    try:
        assert alias
        assert alias.isascii()
        Path(alias, "probe.txt").write_text("project-owned", encoding="utf-8")
        assert (cache_root / "probe.txt").read_text(encoding="utf-8") == "project-owned"
    finally:
        nuitka_builder_module._remove_windows_build_alias(alias)

    assert not Path(alias).exists()
    assert not Path(alias).parent.exists()
    assert cache_root.is_dir()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction contract")
def test_nuitka_failed_build_link_removes_its_temporary_parent(tmp_path, monkeypatch):
    aliases = []

    def fail(command, **kwargs):
        aliases.append(command[-2])
        return subprocess.CompletedProcess(command, 1, stdout="junction rejected")

    monkeypatch.setattr(nuitka_builder_module.subprocess, "run", fail)
    cache = tmp_path / "项目" / "Library" / "Build"
    with pytest.raises(RuntimeError, match="junction rejected"):
        nuitka_builder_module._windows_ascii_build_alias(str(cache))
    assert len(aliases) == 1
    assert not Path(aliases[0]).parent.exists()
    assert cache.is_dir()


def test_player_module_stages_source_less_engine_runtime(tmp_path, monkeypatch):
    import infernux

    source = tmp_path / "source" / "infernux"
    (source / "engine").mkdir(parents=True)
    (source / "lib").mkdir()
    (source / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "engine" / "__init__.py").write_text("ENGINE = 1\n", encoding="utf-8")
    (source / "engine" / "runtime.py").write_text(
        "from __future__ import annotations\nVALUE = 2\n",
        encoding="utf-8",
    )
    (source / "engine" / "path_utils.py").write_text(
        "def resolved_path(value): return str(value)\n",
        encoding="utf-8",
    )
    (source / "engine" / "runtime_artifact_catalog.py").write_text(
        "from .path_utils import resolved_path\n",
        encoding="utf-8",
    )
    (source / "engine" / "build_settings.py").write_text(
        "def load_build_settings(): return {'scenes': []}\n",
        encoding="utf-8",
    )
    (source / "engine" / "data.json").write_text("{}\n", encoding="utf-8")
    (source / "engine" / "game_builder.py").write_text("BUILD = 1\n", encoding="utf-8")
    (source / "engine" / "nuitka_builder.py").write_text("COMPILE = 1\n", encoding="utf-8")
    (source / "engine" / "interaction").mkdir()
    (source / "engine" / "interaction" / "__init__.py").write_text("EDITOR = 1\n", encoding="utf-8")
    (source / "engine" / "undo").mkdir()
    (source / "engine" / "undo" / "__init__.py").write_text("EDITOR = 1\n", encoding="utf-8")
    (source / "engine" / "ui").mkdir()
    (source / "engine" / "ui" / "__init__.py").write_text("UI = 1\n", encoding="utf-8")
    (source / "engine" / "ui" / "theme.py").write_text("THEME = 1\n", encoding="utf-8")
    (source / "engine" / "ui" / "viewport_utils.py").write_text("VIEWPORT = 1\n", encoding="utf-8")
    (source / "engine" / "ui" / "engine_status.py").write_text("STATUS = 1\n", encoding="utf-8")
    (source / "engine" / "ui" / "runtime_canvas_snapshot.py").write_text("SNAPSHOT = 1\n", encoding="utf-8")
    (source / "engine" / "ui" / "editor_panel.py").write_text("EDITOR = 1\n", encoding="utf-8")
    (source / "lib" / "__init__.py").write_text("NATIVE = True\n", encoding="utf-8")
    (source / "lib" / "stale.dll").write_bytes(b"stale")

    monkeypatch.setattr(infernux, "__file__", str(source / "__init__.py"))
    builder = object.__new__(NuitkaBuilder)
    dist = tmp_path / "dist"
    (dist / "infernux" / "lib").mkdir(parents=True)
    (dist / "infernux" / "lib" / "native.dll").write_bytes(b"native")

    builder._inject_engine_python_runtime(str(dist))

    runtime = dist / "infernux"
    assert (runtime / "__init__.pyc").is_file()
    assert (runtime / "engine" / "runtime.pyc").is_file()
    assert (runtime / "engine" / "path_utils.pyc").is_file()
    assert (runtime / "engine" / "runtime_artifact_catalog.pyc").is_file()
    assert (runtime / "engine" / "build_settings.pyc").is_file()
    assert (runtime / "engine" / "data.json").is_file()
    assert (runtime / "lib" / "__init__.pyc").is_file()
    assert (runtime / "lib" / "native.dll").read_bytes() == b"native"
    assert not (runtime / "lib" / "stale.dll").exists()
    assert not (runtime / "engine" / "game_builder.pyc").exists()
    assert not (runtime / "engine" / "nuitka_builder.pyc").exists()
    assert not (runtime / "engine" / "interaction").exists()
    assert not (runtime / "engine" / "undo").exists()
    assert (runtime / "engine" / "ui" / "__init__.pyc").is_file()
    assert (runtime / "engine" / "ui" / "theme.pyc").is_file()
    assert (runtime / "engine" / "ui" / "viewport_utils.pyc").is_file()
    assert (runtime / "engine" / "ui" / "engine_status.pyc").is_file()
    assert (runtime / "engine" / "ui" / "runtime_canvas_snapshot.pyc").is_file()
    assert not (runtime / "engine" / "ui" / "editor_panel.pyc").exists()
    assert not list(runtime.rglob("*.py"))

    fake_stdlib = tmp_path / "stdlib-source"
    fake_stdlib.mkdir()
    shutil.copy2(Path(os.__file__).parent / "__future__.py", fake_stdlib / "__future__.py")
    shutil.copy2(Path(os.__file__).parent / "inspect.py", fake_stdlib / "inspect.py")
    (fake_stdlib / "README.txt").write_text("development documentation", encoding="utf-8")
    (fake_stdlib / "pydoc_data").mkdir()
    (fake_stdlib / "pydoc_data" / "_pydoc.css").write_text("body {}", encoding="utf-8")
    (fake_stdlib / "lib2to3").mkdir()
    (fake_stdlib / "lib2to3" / "Grammar.txt").write_text("grammar", encoding="utf-8")
    (fake_stdlib / "lib2to3" / "Grammar.pickle").write_bytes(b"pickle")
    builder._inject_python_runtime_stdlib(str(dist), source_root=fake_stdlib)
    assert (dist / "stdlib" / "__future__.pyc").is_file()
    assert (dist / "stdlib" / "inspect.pyc").is_file()
    staged_stdlib_files = [path for path in (dist / "stdlib").rglob("*") if path.is_file()]
    assert staged_stdlib_files
    assert all(path.suffix.casefold() == ".pyc" for path in staged_stdlib_files)
    assert not (dist / "stdlib" / "lib2to3" / "Grammar.txt").exists()

    probe = nuitka_builder_module._run_python(
        sys.executable,
        [
            "-I",
            "-S",
            "-c",
            (
                "import sys; "
                f"sys.path[:] = [{str(dist)!r}, {str(dist / 'stdlib')!r}]; "
                "import infernux.engine.runtime as runtime; "
                "assert runtime.VALUE == 2"
            ),
        ],
    )
    assert probe.returncode == 0, probe.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="validates the Windows Player DLL layout")
def test_pack_core_runtime_moves_full_native_closure_off_root(tmp_path):
    builder = _make_builder(tmp_path, tmp_path / "build_output")
    final_dir = tmp_path / "dist"
    data_root = final_dir / "TestGame_Data"
    package_lib = final_dir / "infernux" / "lib"
    package_lib.mkdir(parents=True)
    data_root.mkdir(parents=True)
    (package_lib / "_Infernux.pyd").write_bytes(b"full bridge")
    (package_lib / "InfernuxFoundation.dll").write_bytes(b"foundation")
    (package_lib / "InfernuxRendererRuntime.dll").write_bytes(b"runtime")
    (final_dir / "_InfernuxBootstrap.pyd").write_bytes(b"bootstrap")
    (final_dir / "InfernuxFoundation.dll").write_bytes(b"foundation")
    (final_dir / "InfernuxRendererRuntime.dll").write_bytes(b"runtime")
    (final_dir / "python313.dll").write_bytes(b"python")
    (final_dir / "_ctypes.pyd").write_bytes(b"ctypes ABI")
    (final_dir / "libffi-8.dll").write_bytes(b"libffi ABI")
    (final_dir / "_socket.pyd").write_bytes(b"socket")

    builder._pack_core_runtime_archive(str(final_dir))

    paths = {entry["path"] for entry in read_manifest(data_root / builder._RUNTIME_ARCHIVE_FILENAME)["files"]}
    assert "infernux/lib/_Infernux.pyd" in paths
    assert "infernux/lib/InfernuxFoundation.dll" in paths
    assert "infernux/lib/InfernuxRendererRuntime.dll" in paths
    assert "stdlib/_socket.pyd" in paths
    assert (final_dir / "_InfernuxBootstrap.pyd").is_file()
    assert (final_dir / "InfernuxFoundation.dll").is_file()
    assert (final_dir / "python313.dll").is_file()
    assert (final_dir / "_ctypes.pyd").is_file()
    assert (final_dir / "libffi-8.dll").is_file()
    assert "stdlib/_ctypes.pyd" not in paths
    assert "stdlib/libffi-8.dll" not in paths
    assert not (final_dir / "InfernuxRendererRuntime.dll").exists()
    assert not (final_dir / "infernux").exists()


def test_runtime_pack_cache_round_trip(tmp_path, monkeypatch):
    cache_root = tmp_path / "runtime-packs"
    builder = object.__new__(NuitkaBuilder)
    builder._runtime_pack_dir = str(cache_root)
    builder._staging_dir = str(tmp_path / "staging")
    builder.console_mode = "disable"
    builder.lto = True
    builder._player_compile_input_fingerprint = lambda: "current-player-runtime"
    os.makedirs(builder._staging_dir)
    dist = tmp_path / "original.dist"
    dist.mkdir(parents=True)
    (dist / "InfernuxPlayer.exe").write_bytes(b"runtime")
    (dist / "bindings.pyi.bak").write_bytes(b"editor backup")
    (dist / "InfernuxPlayer.pdb").write_bytes(b"debug symbols")

    builder._store_runtime_pack("a" * 64, str(dist))
    restored = builder._restore_runtime_pack("a" * 64)

    assert restored == os.path.join(builder._staging_dir, "boot.dist")
    assert (tmp_path / "staging" / "boot.dist" / "InfernuxPlayer.exe").read_bytes() == b"runtime"
    cache_manifest = json.loads((cache_root / ("a" * 64) / "Player.inxmanifest").read_text(encoding="utf-8"))
    assert cache_manifest["archive"] == "Runtime.inxrt"
    assert cache_manifest["compression"] == "store"
    assert cache_manifest["lto"] is True
    assert cache_manifest["file_count"] == 1
    assert cache_manifest["archive_bytes"] > 0
    native_manifest = read_manifest(cache_root / ("a" * 64) / "Runtime.inxrt")
    assert native_manifest["compression_profile"] == "development"
    assert {entry["path"] for entry in native_manifest["files"]} == {"InfernuxPlayer.exe"}
    assert not (tmp_path / "staging" / "boot.dist" / "_infernux_runtime_pack.json").exists()


def test_runtime_pack_cache_publish_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    cache_root = tmp_path / "runtime-packs"
    builder = object.__new__(NuitkaBuilder)
    builder._runtime_pack_dir = str(cache_root)
    builder.console_mode = "disable"
    builder.lto = True
    builder._player_compile_input_fingerprint = lambda: "current-player-runtime"
    dist = tmp_path / "original.dist"
    dist.mkdir()
    (dist / "InfernuxPlayer.exe").write_bytes(b"runtime")
    fingerprint = "a" * 64
    destination = cache_root / fingerprint
    native_replace = nuitka_builder_module.os.replace
    attempts = 0

    def locked_then_publish(source, target):
        nonlocal attempts
        if Path(target) == destination and attempts < 2:
            attempts += 1
            raise PermissionError("scanner retained the generated directory")
        if Path(target) == destination:
            attempts += 1
        return native_replace(source, target)

    monkeypatch.setattr(nuitka_builder_module.os, "replace", locked_then_publish)

    builder._store_runtime_pack(fingerprint, str(dist))

    assert attempts == 3
    assert (destination / "Runtime.inxrt").is_file()
    assert (destination / "Player.inxmanifest").is_file()


def test_packaged_runtime_pack_restores_without_local_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "runtime-packs"
    packaged_root = tmp_path / "wheel" / "_runtime_packs"
    monkeypatch.setenv("INFERNUX_PREBUILT_RUNTIME_PACK_DIR", str(packaged_root))
    builder = object.__new__(NuitkaBuilder)
    builder._runtime_pack_dir = str(cache_root)
    builder._staging_dir = str(tmp_path / "staging")
    builder._engine_fingerprint_cache = "engine-content"
    builder.console_mode = "disable"
    builder.lto = True
    os.makedirs(builder._staging_dir)
    dist = tmp_path / "original.dist"
    dist.mkdir()
    (dist / "InfernuxPlayer.exe").write_bytes(b"prebuilt-runtime")
    fingerprint = "a" * 64
    compatibility_key = "b" * 64

    builder._store_runtime_pack(
        fingerprint,
        str(dist),
        compatibility_key=compatibility_key,
    )
    packaged_pack = packaged_root / compatibility_key
    packaged_pack.parent.mkdir(parents=True)
    shutil.copytree(cache_root / fingerprint, packaged_pack)
    shutil.rmtree(cache_root / fingerprint)

    restored = builder._restore_runtime_pack(
        fingerprint,
        compatibility_key=compatibility_key,
    )

    assert restored == os.path.join(builder._staging_dir, "boot.dist")
    assert (Path(restored) / "InfernuxPlayer.exe").read_bytes() == b"prebuilt-runtime"


def test_runtime_pack_hit_does_not_prepare_compiler_or_install_requirements(tmp_path, monkeypatch):
    entry = tmp_path / "boot.py"
    entry.write_text("pass\n", encoding="utf-8")
    builder = NuitkaBuilder(
        str(entry),
        str(tmp_path / "output"),
        build_cache_root=str(tmp_path / "cache"),
        runtime_pack_cache=True,
        player_module=True,
    )
    monkeypatch.setattr(builder, "_runtime_pack_fingerprint", lambda command: "cached")
    monkeypatch.setattr(builder, "_runtime_pack_compatibility_key", lambda: "compatible")
    monkeypatch.setattr(builder, "_restore_runtime_pack", lambda *args, **kwargs: "restored")

    def unexpected(*args, **kwargs):
        raise AssertionError("A precompiled Player does not need compilation tools")

    monkeypatch.setattr(builder, "_check_nuitka", unexpected)
    monkeypatch.setattr(builder, "_run_nuitka", unexpected)
    monkeypatch.setattr(nuitka_builder_module, "_has_msvc_toolchain", unexpected)
    monkeypatch.setattr(nuitka_builder_module.shutil, "which", unexpected)

    assert builder.build() == "restored"


@pytest.mark.parametrize("force_rebuild", [False, True])
def test_consumer_player_never_compiles_on_a_runtime_pack_miss(tmp_path, monkeypatch, force_rebuild):
    entry = tmp_path / "boot.py"
    entry.write_text("pass\n", encoding="utf-8")
    builder = NuitkaBuilder(
        str(entry),
        str(tmp_path / "output"),
        build_cache_root=str(tmp_path / "cache"),
        runtime_pack_cache=True,
        player_module=True,
    )
    monkeypatch.setattr(builder, "_runtime_pack_fingerprint", lambda command: "missing")
    monkeypatch.setattr(builder, "_runtime_pack_compatibility_key", lambda: "compatible")
    monkeypatch.setattr(builder, "_restore_runtime_pack", lambda *args, **kwargs: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Consumer export must not prepare or invoke a compiler")

    for method in ("_check_nuitka", "_run_nuitka"):
        monkeypatch.setattr(builder, method, unexpected)
    monkeypatch.setattr(nuitka_builder_module, "_has_msvc_toolchain", unexpected)
    monkeypatch.setattr(nuitka_builder_module.shutil, "which", unexpected)
    with pytest.raises(RuntimeError, match="game export does not compile the engine"):
        builder.build(force_runtime_rebuild=force_rebuild)


def test_release_engineering_explicitly_compiles_player_payload(tmp_path, monkeypatch):
    entry = tmp_path / "boot.py"
    entry.write_text("pass\n", encoding="utf-8")
    builder = NuitkaBuilder(
        str(entry),
        str(tmp_path / "output"),
        build_cache_root=str(tmp_path / "cache"),
        runtime_pack_cache=True,
        player_module=True,
        packaged_runtime_lookup=False,
    )
    compiled = []
    monkeypatch.setattr(builder, "_runtime_pack_fingerprint", lambda command: "build-input")
    monkeypatch.setattr(builder, "_runtime_pack_compatibility_key", lambda: "compatible")
    monkeypatch.setattr(builder, "_restore_runtime_pack", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "_check_nuitka", lambda: None)

    def compile_payload(*args):
        compiled.append(True)
        return str(tmp_path / "player.dist")

    monkeypatch.setattr(builder, "_run_nuitka", compile_payload)
    for method in (
        "_inject_native_libs",
        "_inject_engine_python_runtime",
        "_inject_python_runtime_stdlib",
        "_inject_python_bootstrap_runtime",
        "_store_runtime_pack",
    ):
        monkeypatch.setattr(builder, method, lambda *args, **kwargs: None)
    assert builder.build() == str(tmp_path / "player.dist")
    assert compiled == [True]


def test_runtime_pack_stores_the_environment_used_for_compilation(tmp_path, monkeypatch):
    entry = tmp_path / "boot.py"
    entry.write_text("pass\n", encoding="utf-8")
    builder = NuitkaBuilder(
        str(entry), str(tmp_path / "output"), build_cache_root=str(tmp_path / "cache"), runtime_pack_cache=True
    )
    environment = ["before-install"]
    stored = []
    monkeypatch.setattr(builder, "_runtime_pack_fingerprint", lambda command: environment[0])
    monkeypatch.setattr(builder, "_runtime_pack_compatibility_key", lambda: "compatible")
    monkeypatch.setattr(builder, "_restore_runtime_pack", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "_check_nuitka", lambda: environment.__setitem__(0, "compiled-environment"))
    monkeypatch.setattr(builder, "_run_nuitka", lambda *args: str(tmp_path / "compiled.dist"))
    for method in ("_inject_native_libs", "_embed_utf8_manifest"):
        monkeypatch.setattr(builder, method, lambda *args: None)
    monkeypatch.setattr(builder, "_store_runtime_pack", lambda key, *args, **kwargs: stored.append(key))

    builder.build()

    assert stored == ["compiled-environment"]
    assert builder.last_runtime_pack_key == "compiled-environment"
