"""Authoring and cooked Player assets retain query and user-payload semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("hint", [r"rooms\A", r"C:\save\slot", r"D:\Project\Assets\Map.json"])
@pytest.mark.parametrize("shape", ["ordinary", "mixed-reference"])
def test_cook_only_rewrites_typed_reference_hints(tmp_path, hint, shape):
    from infernux.engine.game_builder import GameBuilder

    document = {"path_hint": hint, "nested": [{"Path_Hint": hint}],
                "list": {"path_hint": [hint]}, "message": hint,
                "other_type": {"$type": "puzzle_route", "path_hint": hint}}
    expected = json.loads(json.dumps(document))
    if shape == "mixed-reference":
        document["reference"] = {"$type": "asset_ref", "guid": "1" * 32,
                                 "path_hint": r"D:\Project\Assets\Map.json",
                                 "label": {"path_hint": hint}}
        expected["reference"] = {**document["reference"], "path_hint": "Assets/Map.json"}
    path = tmp_path / "UserConfig.json"
    original = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(original)
    GameBuilder.__new__(GameBuilder)._rewrite_player_document_paths(str(path), ".json")
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    if shape == "ordinary":
        assert path.read_bytes() == original


@pytest.mark.parametrize("contract", ["queries", "user-payload"])
def test_native_cook_pack_move_preserves_asset_contract(tmp_path, contract):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), contract],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("suffix,document,expected_rejected", [
    (".json", {"help_url": "https://example.invalid/home/manual?next=/tmp/page"}, False),
    (".json", {"message": "Open C:/Station/Bridge/access.txt"}, False),
    (".json", {"message": "Read /home/captain/access.log"}, False),
    (".txt", "Read /home/captain/access.log", False),
    (".json", {"metadata": {"file_path": "C:/fictional/game/terminal"}}, False),
    (".json", {"ref": {"$type": "asset_ref", "guid": "1" * 32,
                        "path_hint": "C:/Author/Assets/Main.scene"}}, True),
    (".scene", {"objects": [{"data": {"message": "Read /home/captain/access.log"}}]}, False),
    (".scene", {"objects": [{"data": {"ref": {"$type": "asset_ref", "guid": "1" * 32,
                                              "path_hint": "/opt/author/Assets/Main.scene"}}}]}, True),
])
def test_content_archive_audits_references_not_user_text(tmp_path, suffix, document, expected_rejected):
    from collections import defaultdict
    from infernux.engine.player_package_audit import _archive_entry_records
    from infernux.engine.player_package_native import write_pack, read_entry, using_test_backend

    original = (document if isinstance(document, str) else json.dumps(document)).encode("utf-8")
    source = tmp_path / f"content{suffix}"
    source.write_bytes(original)
    directory = "Document" if suffix == ".scene" else "Blob"
    runtime_path = f"Library/Artifacts/{directory}/{'1' * 32}{suffix}"
    archive = tmp_path / "Content.inxpkg"
    assert not using_test_backend()
    write_pack([(runtime_path, source)], archive)
    buckets = {key: [] for key in ("archive_entries", "forbidden", "author_sources", "meta_files",
               "absolute_paths", "native_files", "hidden_executables", "authoring_tree_files",
               "unknown_author_documents", "unsafe_entry_paths")}
    _archive_entry_records(archive, "Game_Data/Content.inxpkg", payload_candidates=defaultdict(list), **buckets)
    assert not buckets["forbidden"] and not buckets["author_sources"]
    assert bool(buckets["absolute_paths"]) is expected_rejected
    assert read_entry(archive, runtime_path) == original


def exercise(root, contract):
    from infernux.application import Application
    from infernux.core.assets import AssetManager
    from infernux.engine import project_context
    from infernux.engine.engine import Engine
    from infernux.engine.game_builder import GameBuilder
    from infernux.engine.player_package_native import extract_pack, write_pack
    from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.runtime_artifact_catalog import build_catalog, load_asset_index
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    project = root / "作者 项目"
    assets = project / "Assets"
    assets.mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    running = True
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        scene_path = assets / "Main.scene"
        scene_path.write_text(json.dumps(SceneManager.instance().get_active_scene().serialize_asset_document()),
                              encoding="utf-8")
        imported_scene = database.import_asset(str(scene_path))
        assert imported_scene, imported_scene.error
        (project / "ProjectSettings/BuildSettings.json").write_text(
            json.dumps({"scene_guids": [imported_scene.guid]}), encoding="utf-8")
        payload = {"path_hint": r"C:\save\slot", "nested": [{"Path_Hint": r"rooms\A"}],
                   "other": {"$type": "puzzle_route", "path_hint": r"D:\Levels\Room"}}
        original = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        paths = ["Assets/Data/Mixed.JSON", "Assets/Data/Äther.json", "Assets/Data/谜题 数据.json",
                 "Assets/Data/Room/Deep.json", "Assets/Data/Other.txt"]
        glob_paths = ["Assets/Icons/direct.txt", "Assets/A/Icons/middle.txt",
                      "Assets/A/Icons/Deep/leaf.txt", "Assets/A/Nested/Icons/Deep/distant.txt"]
        paths += glob_paths
        guids = {}
        for path in paths:
            file = project / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(original)
            imported = database.import_asset(str(file))
            assert imported, imported.error
            guids[path] = imported.guid
        queries = [
            ("Assets/Data/Mixed.JSON", paths[:1]),
            (r"Assets\Data\Mixed.JSON", paths[:1]),
            ("assets/data/mixed.json", paths[:1]),
            ("Assets/Data/*.json", paths[:3]),
            ("Assets/Data/*.JSON", paths[:3]),
            ("Assets/Data/**/*.JSON", paths[:4]),
            ("assets/**/mixed.*", paths[:1]),
            ("Assets/Data", paths[:3] + paths[4:5]),
            ("assets/data/äTHER.JSON", paths[1:2]),
            ("ASSETS/DATA/谜题 数据.JSON", paths[2:3]),
            ("Assets/Data/[mM]*.?SON", paths[:1]),
            ("Assets/Data/*eep.json", []),
            ("Assets/Missing/*.json", []),
            ("Assets/**/Icons/**/*.txt", glob_paths),
            ("Assets/*/Icons/**/*.txt", glob_paths[1:3]),
            ("Assets/**/Icons/*.txt", glob_paths[:2]),
            ("Assets/*/*/Icons/*/*.txt", glob_paths[3:]),
        ]
        failures = []

        def observe(label):
            for query, wanted in queries:
                handles = AssetManager.find_assets(query)
                actual = [item._guid for item in handles]
                expected = sorted(guids[path] for path in wanted)
                if actual != expected:
                    failures.append((label, query, actual, expected))
                if contract == "user-payload":
                    for handle in handles:
                        assert handle.read_bytes() == original, (label, query)
            for path, guid in guids.items():
                handles = AssetManager.find_assets(guid)
                assert [item._guid for item in handles] == [guid]
                if contract == "user-payload":
                    assert handles[0].read_bytes() == original, (label, path)

        observe("authoring")
        database.flush_derived_index()
        builder = GameBuilder(str(project), str(root / "Cook"), game_name="Portability")
        builder.freeze_asset_index_entries(load_asset_index(project))
        final = root / "Cook"
        builder._copy_game_data(str(final))
        builder._write_runtime_asset_records(str(final))
        data = final / "Data"
        records = json.loads((data / "Library/RuntimeAssetRecords.json").read_text(encoding="utf-8"))
        # This archive exercises project content. Platform-owned builtin
        # records belong to the separately published runtime package.
        managed_guids = {*guids.values(), imported_scene.guid}
        records["entries"] = [entry for entry in records["entries"] if entry["guid"] in managed_guids]
        entries, packed = [], []
        for runtime_path, binding in builder._runtime_asset_identity_bindings.items():
            if binding["source_guid"] not in managed_guids:
                continue
            file = data / runtime_path
            entries.append({"package": "Content.inxpkg", "runtime_path": runtime_path,
                            "bytes": file.stat().st_size, "payload": file.read_bytes(),
                            "asset_binding": binding})
            packed.append((runtime_path, file))
        catalog = build_catalog(entries, player_host={"executable": "Portability"}, package_records=[])
        archive = root / "Content.inxpkg"
        write_pack(packed, archive)
        relocated = root / "新设备 & Player" / "Data"
        extract_pack(archive, relocated)
        for path in paths:
            assert (project / path).read_bytes() == original
        engine.exit()
        running = False
        project.rename(root / "RelocatedAuthorProject")
        assert not project.exists()
        runtime = PlayerRuntimeAssetCatalog.from_documents(str(relocated), catalog, records)
        project_context.set_project_root(str(relocated))
        project_context.set_runtime_asset_query(runtime.query_asset_guids)
        project_context.set_runtime_asset_resolver(runtime.resolve_guid)
        AssetManager._asset_database = None
        Application._bind_engine(engine, "player")
        observe("relocated-player-catalog")
        if contract == "queries":
            assert not failures, failures
    finally:
        if running:
            engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
