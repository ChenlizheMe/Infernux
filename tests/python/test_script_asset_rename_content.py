"""Project file renames preserve authored Python bytes and component identity."""
from pathlib import Path
import sys

import pytest

from infernux.core.assets import AssetManager
from infernux.engine.interaction import (
    EditorActionJournal, ProjectAssetCommandService, SelectionService,
)
from infernux.engine.undo import UndoManager
from model_test_support import remove_model_test_folder


SOURCES = (
    ("Old", b"class Old:\n    value = 13\n    @staticmethod\n    def make():\n        return Old().value\n"),
    ("Old", b'TEMPLATE = """\nclass Old:\n    pass\n"""\nclass Actual:\n    value = 13\n'),
    ("Old", b'from __future__ import annotations\nclass Old:\n    def clone(self) -> Old:\n        return Old()\nclass Derived(Old):\n    pass\nREGISTRY = {"Old": Old}\n'),
    ("old_name", b'class OldName:\n    value = "OldName"\n'),
    ("Old", b'class Old:\n    pass\nclass Secondary:\n    target = Old\n'),
    ("Old", b'# class Old: comment\r\nclass Old:\r\n    pass\r\n'),
    ("Old", b'\xef\xbb\xbf# UTF-8 BOM\nclass Old:\n    pass\n'),
    ("Old", b'class Old:\n    value = 13'),
    ("Old", b'REGISTRY = []\ndef register(cls):\n    REGISTRY.append(cls)\n    return cls\n@register\nclass Old:\n    pass\n'),
    ("Old", b'class Old:\n    def unfinished('),
    ("旧类", 'class 旧类:\n    value = "原始文本"\n'.encode()),
)


@pytest.fixture
def rename_workspace(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    root = Path(database.assets_root) / tmp_path.name
    root.mkdir()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    journal = EditorActionJournal()
    history = UndoManager(journal)
    commands = ProjectAssetCommandService(SelectionService())
    commands.configure(str(Path(database.assets_root).parent), database)
    try:
        yield root, database, commands, history
    finally:
        commands.shutdown()
        history.shutdown()
        remove_model_test_folder(database, root)


@pytest.mark.parametrize("stem,content", SOURCES, ids=(
    "self-reference", "literal", "annotations-inheritance-registry", "snake-case",
    "multiple-classes", "comments-crlf", "bom", "no-final-newline", "decorator",
    "incomplete-edit", "unicode-class",
))
def test_project_rename_and_history_preserve_source_bytes(rename_workspace, stem, content):
    root, database, commands, history = rename_workspace
    source = root / (stem + ".py")
    destination = root / "Renamed.py"
    source.write_bytes(content)
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    guid = str(imported.guid)
    mtime = source.stat().st_mtime_ns
    try:
        original_code = compile(content, str(source), "exec")
    except SyntaxError:
        original_code = None  # File movement must also preserve unfinished edits.

    assert Path(commands.rename(str(source), destination.name)) == destination
    for path, absent, action in (
        (destination, source, history.undo),
        (source, destination, history.redo),
        (destination, source, None),
    ):
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime
        assert not absent.exists()
        assert str(database.get_guid_from_path(str(path))) == guid
        assert Path(database.get_path_from_guid(guid)) == path
        if original_code is not None:
            namespace = {}
            exec(compile(path.read_bytes(), str(path), "exec"), namespace)
            if "Old" in namespace and hasattr(namespace["Old"], "make"):
                assert namespace["Old"].make() == 13
        if action is not None:
            action()


def test_component_restores_by_guid_after_file_rename_and_undo(rename_workspace):
    from infernux.components.component_identity import component_type_guid
    from infernux.components.registry import component_types_for_script_path, unregister_component_script
    from infernux.components.script_loader import load_all_components_from_file
    from infernux.engine.component_restore import create_component_instance

    root, database, commands, history = rename_workspace
    source = root / "Old.py"
    destination = root / "new_filename.py"
    source.write_text(
        'import infernux as inx\nclass Old(inx.InxComponent):\n'
        '    @staticmethod\n    def make():\n        return Old.__name__\n'
        'class Second(inx.InxComponent):\n    pass\n', encoding="utf-8")
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    guid = str(imported.guid)
    identities = {name: component_type_guid(guid, name) for name in ("Old", "Second")}
    modules = set()
    try:
        assert Path(commands.rename(str(source), destination.name)) == destination
        for path, action in ((destination, history.undo), (source, history.redo), (destination, None)):
            classes = load_all_components_from_file(str(path))
            modules.update(cls.__module__ for cls in classes)
            assert [cls.__name__ for cls in classes] == ["Old", "Second"]
            assert classes[0].make() == "Old"
            for name, identity in identities.items():
                instance, resolved = create_component_instance(guid, identity, name, asset_database=database)
                assert Path(resolved) == path and instance is not None
                try:
                    assert type(instance).__name__ == name
                    assert instance._get_type_guid() == identity
                finally:
                    instance._call_on_destroy()
            if action is not None:
                action()
    finally:
        for path in (source, destination):
            modules.update(cls.__module__ for cls in component_types_for_script_path(str(path)))
            unregister_component_script(str(path))
        for name in modules:
            sys.modules.pop(name, None)
