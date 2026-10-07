"""Real filesystem, catalog and editor-history coverage for case-only renames."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", [
    "native-file", "native-file-with-meta", "native-batch", "native-directory",
    "editor-file", "editor-directory", "editor-history-file", "editor-history-directory",
    "editor-history-texture", "editor-rollback-directory", "ordinary-file",
    "editor-move-file", "editor-move-directory",
])
def test_case_only_asset_rename(tmp_path, case):
    import infernux
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), case],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(project, case):
    import json
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.interaction import (
        AssetMutationService, DocumentRegistry, DocumentKind, DocumentKey,
        ProjectAssetCommandService, SelectionService,
    )
    from infernux.engine.ui import project_file_ops
    from infernux.engine.undo import UndoManager
    from infernux.lib import AssetRegistry, LogLevel, RuntimeMode

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    mutations = commands = None
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        directory = case.endswith("directory")
        texture_case = case.endswith("texture")
        source_root = assets / ("CASEFolder" if directory else "CASEFile.png" if texture_case else "CASEFile.txt")
        target_root = assets / ("casefolder" if directory else "casefile.png" if texture_case else "casefile.txt")
        if case == "ordinary-file":
            target_root = assets / "Other.txt"
        if directory:
            source_root.mkdir()
        names = ("Child.txt", "Second.txt") if directory else (None,)
        sources = [source_root / name if name else source_root for name in names]
        targets = [target_root / name if name else target_root for name in names]
        guids = []
        for source in sources:
            if texture_case:
                import struct
                import zlib
                def chunk(kind, content):
                    return (struct.pack(">I", len(content)) + kind + content
                            + struct.pack(">I", zlib.crc32(kind + content)))
                source.write_bytes(b"\x89PNG\r\n\x1a\n"
                                   + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
                                   + chunk(b"IDAT", zlib.compress((b"\0" + bytes([255, 0, 0, 255]) * 2) * 2))
                                   + chunk(b"IEND", b""))
            else:
                source.write_text("stable authored asset", encoding="utf-8")
            imported = database.import_asset(str(source))
            assert imported, imported.error
            guids.append(imported.guid)
        texture = AssetRegistry.instance().load_texture_by_guid(guids[0]) if texture_case else None
        if texture_case:
            assert texture is not None and texture.pixel_width == 2
        selection = SelectionService()
        documents = DocumentRegistry()
        mutations = AssetMutationService(documents, selection)
        notifications = []
        mutations.add_observer(notifications.append)
        document = documents.create(
            DocumentKind.PARTICLE_GRAPH, sources[0].stem,
            key=DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, guids[0]),
            resource_path=str(sources[0]), revision=3, saved_revision=2,
        )
        history = UndoManager()
        commands = ProjectAssetCommandService(selection)
        commands.configure(str(project), database)

        def verify(root, paths, editor):
            assert root.name in {entry.name for entry in assets.iterdir()}, "workspace spelling was not changed"
            for path, guid in zip(paths, guids):
                assert database.get_guid_from_path(str(path)) == guid
                assert database.get_path_from_guid(guid).replace("\\", "/") == path.as_posix()
                sidecar = Path(str(path) + ".meta")
                assert sidecar.name in {entry.name for entry in path.parent.iterdir()}
                metadata = json.loads(sidecar.read_bytes())["metadata"]
                assert metadata["guid"]["value"] == guid
                assert metadata["file_path"]["value"] == path.relative_to(project).as_posix()
                entries = database.get_directory_catalog(str(path.parent))
                assert any(entry["guid"] == guid and entry["path"].replace("\\", "/") == path.as_posix()
                           for entry in entries)
            if texture is not None:
                assert texture.file_path.replace("\\", "/") == paths[0].as_posix()
                assert AssetRegistry.instance().load_texture_by_guid(guids[0]).file_path == texture.file_path
            if editor:
                assert document.resource_path.replace("\\", "/") == paths[0].as_posix()
                assert document.revision == 3 and document.saved_revision == 2 and document.is_dirty

        if case.startswith("native-"):
            source_root.rename(target_root)
            if case == "native-file-with-meta":
                Path(str(sources[0]) + ".meta").rename(str(targets[0]) + ".meta")
            if case in {"native-batch", "native-directory"}:
                results = database.move_assets_batch([(str(a), str(b)) for a, b in zip(sources, targets)])
                assert len(results) == len(guids) and all(results), [result.error for result in results]
            else:
                result = database.move_asset(str(sources[0]), str(targets[0]))
                assert result, result.error
        elif case.startswith("editor-history-"):
            assert commands.can_rename(str(source_root), target_root.name)
            assert commands.rename(str(source_root), target_root.name)
        elif case.startswith("editor-move-"):
            previous_selection = selection.snapshot.primary
            assert commands.move(str(source_root), str(target_root))
            assert selection.snapshot.primary == previous_selection
        else:
            if case == "editor-rollback-directory":
                commit = mutations.commit_relocation
                def reject(_plan):
                    raise RuntimeError("injected editor relocation publication failure")
                mutations.commit_relocation = reject
                try:
                    try:
                        project_file_ops.move_path(str(source_root), str(target_root), database, origin="user")
                    except RuntimeError as exc:
                        assert "injected editor relocation" in str(exc)
                    else:
                        raise AssertionError("injected publication failure did not reject")
                finally:
                    mutations.commit_relocation = commit
                verify(source_root, sources, True)
                assert not notifications
            assert project_file_ops.move_path(str(source_root), str(target_root), database, origin="user")
        verify(target_root, targets, case.startswith("editor-"))
        if case.startswith(("editor-history-", "editor-move-")):
            history.undo()
            verify(source_root, sources, True)
            history.redo()
            verify(target_root, targets, True)
            assert len(notifications) == 3
        elif case.startswith("editor-"):
            assert len(notifications) == 1
        database.refresh()
        verify(target_root, targets, case.startswith("editor-"))
    finally:
        if commands:
            commands.shutdown()
        if mutations:
            mutations.shutdown()
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
