"""Project settings collaboration through the real native document writer."""
import copy
import json
import os
import subprocess
import sys

import pytest

from infernux.core.document_store import DocumentStore
from infernux.engine.build_settings import BUILD_SETTINGS_DEFAULTS
from infernux.engine.interaction import DocumentCapability, DocumentKind, DocumentRegistry, ProjectSettingsDocumentController
from infernux.engine.undo import UndoManager
from infernux.lib import TagLayerManager
from infernux.physics import settings as physics_settings


def _tags():
    layers = [""] * 32
    for index, name in ((0, "Default"), (1, "TransparentFX"), (2, "IgnoreRaycast"), (4, "Water"), (5, "UI")):
        layers[index] = name
    return {"custom_tags": [], "layers": layers, "layer_collision_masks": [0xFFFFFFFF] * 32}


@pytest.fixture
def settings_editor(tmp_path):
    tags = TagLayerManager.instance()
    old_tags = tags.serialize()
    previous_undo = UndoManager._instance
    tags.deserialize(json.dumps(_tags()))
    manager = UndoManager()
    root = tmp_path / "ProjectSettings"
    root.mkdir()
    for filename, value in (
        ("BuildSettings.json", BUILD_SETTINGS_DEFAULTS),
        ("TagLayerSettings.json", _tags()),
        ("PhysicsSettings.json", physics_settings.DEFAULT_PHYSICS_SETTINGS),
    ):
        (root / filename).write_text(json.dumps(value), encoding="utf-8")
    controller = ProjectSettingsDocumentController(str(tmp_path))
    registry = DocumentRegistry.instance()
    document = registry.create(DocumentKind.PROJECT_SETTINGS, "Project Settings",
                               resource_path=controller.settings_path, controller=controller,
                               capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD)
    controller.document_id = document.document_id
    try:
        yield controller, document, root, manager
    finally:
        DocumentStore.flush()
        tags.deserialize(old_tags)
        physics_settings.apply(copy.deepcopy(physics_settings.DEFAULT_PHYSICS_SETTINGS))
        UndoManager._instance = previous_undo


def _edit(controller, section, field, value):
    following = controller.section(section)
    following[field] = value
    assert controller.apply_section(section, following, edit_key=f"{section}.{field}", description="Edit settings")


def _settle(controller):
    DocumentStore.flush()
    controller.poll_pending_writes()


@pytest.mark.parametrize("external", ['{"game_name":"Teammate"}\n', '<<<<<<< HEAD\nconflict\n=======\nother\n>>>>>>> teammate\n'])
def test_physics_edit_preserves_external_build_bytes(settings_editor, external):
    controller, document, root, _ = settings_editor
    build = root / "BuildSettings.json"
    tags_before = (root / "TagLayerSettings.json").read_bytes()
    build.write_text(external, encoding="utf-8")
    external_bytes = build.read_bytes()
    _edit(controller, "physics", "gravity", [0.0, -4.0, 0.0])
    _settle(controller)
    assert build.read_bytes() == external_bytes
    assert (root / "TagLayerSettings.json").read_bytes() == tags_before
    assert json.loads((root / "PhysicsSettings.json").read_text())["gravity"] == [0.0, -4.0, 0.0]
    assert not document.is_dirty


def test_project_loader_resets_missing_tag_file(tmp_path):
    from infernux.engine.engine import Engine

    manager = TagLayerManager.instance()
    original = manager.serialize()
    authored = _tags()
    authored["custom_tags"] = ["OnlyInProjectA"]
    authored["layers"][9] = "OnlyInProjectA"
    authored["layer_collision_masks"][9] &= ~(1 << 10)
    authored["layer_collision_masks"][10] &= ~(1 << 9)
    try:
        assert manager.deserialize(json.dumps(authored))
        Engine._apply_project_settings(str(tmp_path))
        assert json.loads(manager.serialize()) == _tags()
    finally:
        manager.deserialize(original)


@pytest.mark.parametrize("section, filename, field, local", [
    ("build", "BuildSettings.json", "game_name", "Local"),
    ("physics", "PhysicsSettings.json", "gravity", [0.0, -3.0, 0.0]),
    ("tag_layers", "TagLayerSettings.json", "custom_tags", ["Local"]),
])
@pytest.mark.parametrize("change", ["valid", "conflict", "deleted"])
def test_edited_file_rejects_external_change(settings_editor, section, filename, field, local, change):
    from infernux.engine.interaction import DocumentState

    controller, document, root, _ = settings_editor
    path = root / filename
    if change == "deleted":
        path.unlink()
        expected = None
    else:
        expected = (b"<<<<<<< HEAD\ninvalid JSON\n=======\nother\n>>>>>>> teammate\n"
                    if change == "conflict" else path.read_bytes() + b"\n  ")
        path.write_bytes(expected)
    _edit(controller, section, field, local)
    _settle(controller)
    assert (path.read_bytes() if path.exists() else None) == expected
    assert controller.section(section)[field] == local
    assert document.is_dirty
    assert document.state is DocumentState.CONFLICT


def test_rapid_edits_undo_redo_use_one_owned_write_chain(settings_editor):
    controller, document, root, manager = settings_editor
    for number in range(12):
        _edit(controller, "build", "game_name", f"Edit {number}")
    manager.undo()
    manager.redo()
    _settle(controller)
    assert json.loads((root / "BuildSettings.json").read_text())["game_name"] == "Edit 11"
    assert not document.is_dirty
    manager.undo()
    _settle(controller)
    assert json.loads((root / "BuildSettings.json").read_text()) == controller.section("build")
    assert not document.is_dirty


def test_partial_submission_tracks_the_successful_section(settings_editor):
    from infernux.core.document_store import submit_document_text

    controller, document, root, _ = settings_editor
    initial = controller.capture_document()

    def submit(path, content, **options):
        if path.endswith("PhysicsSettings.json"):
            raise OSError("controlled submission failure")
        return submit_document_text(path, content, **options)

    controller._submitter = submit
    following = controller.capture_document()
    following["build"]["game_name"] = "Durable"
    following["physics"]["gravity"] = [0.0, -2.0, 0.0]
    assert controller.apply_document(following, edit_key="combined", description="Two sections")
    _settle(controller)
    assert json.loads((root / "BuildSettings.json").read_text()) == following["build"]
    assert json.loads((root / "PhysicsSettings.json").read_text()) == initial["physics"]
    assert document.is_dirty
    controller._submitter = submit_document_text
    assert controller.schedule_autosave()
    _settle(controller)
    assert json.loads((root / "PhysicsSettings.json").read_text()) == following["physics"]
    assert not document.is_dirty


def test_reload_keeps_conflict_bytes_until_author_resolves_them(settings_editor):
    from infernux.engine.interaction import DocumentActionStatus

    controller, document, root, manager = settings_editor
    path = root / "BuildSettings.json"
    baseline = controller.capture_document()
    baseline_revision = document.revision
    path.write_text("<<<<<<< HEAD\n", encoding="utf-8")
    _edit(controller, "build", "game_name", "Local")
    _settle(controller)
    registry = DocumentRegistry.instance()
    result = registry.request_reload_external(document.document_id)
    assert result.status is DocumentActionStatus.FAILED
    assert path.read_text() == "<<<<<<< HEAD\n"
    assert controller.section("build")["game_name"] == "Local"

    resolved = copy.deepcopy(baseline["build"])
    resolved["game_name"] = "Resolved by Git"
    path.write_text(json.dumps(resolved), encoding="utf-8")
    resolved_bytes = path.read_bytes()
    assert registry.request_reload_external(document.document_id).status is DocumentActionStatus.APPLIED
    assert path.read_bytes() == resolved_bytes
    assert controller.section("build") == resolved
    assert not document.is_dirty
    with pytest.raises(RuntimeError, match="predates"):
        controller.restore_document(baseline, baseline_revision)
    assert path.read_bytes() == resolved_bytes
    _edit(controller, "build", "game_name", "After reload")
    manager.undo()
    _settle(controller)
    assert controller.section("build") == resolved
    assert not document.is_dirty


def test_discard_reads_current_disk_and_never_rewrites_it(settings_editor):
    from infernux.engine.interaction import DocumentActionStatus

    controller, document, root, _ = settings_editor
    path = root / "BuildSettings.json"
    external = controller.section("build")
    external["game_name"] = "External current"
    path.write_text(json.dumps(external), encoding="utf-8")
    expected = {p.name: p.read_bytes() for p in root.iterdir()}
    result = DocumentRegistry.instance().request_discard(document.document_id)
    assert result.status is DocumentActionStatus.APPLIED
    assert controller.section("build") == external
    assert {p.name: p.read_bytes() for p in root.iterdir()} == expected


def test_explicit_save_waits_for_only_edited_settings(settings_editor):
    from infernux.engine.interaction import DocumentActionStatus

    controller, document, root, _ = settings_editor
    tags_bytes = (root / "TagLayerSettings.json").read_bytes()
    _edit(controller, "build", "game_name", "Explicit save")
    result = DocumentRegistry.instance().request_save(document.document_id)
    assert result.status in {DocumentActionStatus.PENDING, DocumentActionStatus.APPLIED}
    _settle(controller)
    DocumentRegistry.instance().process_pending_saves()
    assert not document.is_dirty
    assert (root / "TagLayerSettings.json").read_bytes() == tags_bytes
    assert json.loads((root / "BuildSettings.json").read_text())["game_name"] == "Explicit save"


def test_idle_settings_poll_does_not_publish_registry_changes(settings_editor):
    controller, _, _, _ = settings_editor
    registry = DocumentRegistry.instance()
    revision = registry.revision
    assert controller.poll_pending_writes() == 0
    assert controller.poll_pending_writes() == 0
    assert registry.revision == revision


def test_settings_directory_failure_preserves_authoring_history(settings_editor, monkeypatch):
    import infernux.engine.interaction.project_settings as settings_module

    controller, document, root, manager = settings_editor
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    real_makedirs = os.makedirs

    def deny_settings(path, **kwargs):
        if os.path.normpath(path) == os.path.normpath(str(root)):
            raise PermissionError("controlled settings directory failure")
        return real_makedirs(path, **kwargs)

    monkeypatch.setattr(settings_module.os, "makedirs", deny_settings)
    _edit(controller, "build", "game_name", "Retained draft")
    _settle(controller)
    assert controller.section("build")["game_name"] == "Retained draft"
    assert document.is_dirty
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    monkeypatch.setattr(settings_module.os, "makedirs", real_makedirs)
    manager.undo()
    _settle(controller)
    assert not document.is_dirty
    assert controller.section("build")["game_name"] == ""


def test_missing_settings_do_not_inherit_active_project_or_create_unedited_files(tmp_path):
    tags = TagLayerManager.instance()
    original = tags.serialize()
    previous_undo = UndoManager._instance
    try:
        tags.add_tag("PreviousProject")
        controller = ProjectSettingsDocumentController(str(tmp_path))
        assert controller.section("tag_layers") == _tags()
        assert not (tmp_path / "ProjectSettings").exists()
        document = DocumentRegistry.instance().create(DocumentKind.PROJECT_SETTINGS, "Settings",
                    resource_path=controller.settings_path, controller=controller)
        controller.document_id = document.document_id
        UndoManager()
        _edit(controller, "build", "game_name", "New project")
        _settle(controller)
        assert {p.name for p in (tmp_path / "ProjectSettings").iterdir()} == {"BuildSettings.json"}
        assert not document.is_dirty
    finally:
        DocumentStore.flush()
        tags.deserialize(original)
        UndoManager._instance = previous_undo


@pytest.mark.parametrize("second", ["absent", "default", "custom"])
def test_sequential_real_engine_projects_isolate_settings(tmp_path, second):
    # Separate native process: this test owns both Engine lifetimes, while
    # pytest's session engine and the user's desktop editor remain untouched.
    script = r'''
import json, sys
from pathlib import Path
from infernux.engine.engine import Engine
from infernux.engine.preferences_store import PreferencesStore
from infernux.lib import LogLevel, RuntimeMode, TagLayerManager
root, mode = Path(sys.argv[1]), sys.argv[2]
PreferencesStore()._path = str(root / 'preferences.json')
manager = TagLayerManager.instance()
defaults = json.loads(manager.serialize_defaults())
authored = json.loads(manager.serialize_defaults())
authored['custom_tags'] = ['OnlyInProjectA']
authored['layers'][9] = 'OnlyInProjectA'
authored['layer_collision_masks'][9] &= ~(1 << 10)
authored['layer_collision_masks'][10] &= ~(1 << 9)
expected_b = json.loads(manager.serialize_defaults())
if mode == 'custom':
    expected_b['custom_tags'] = ['OnlyInProjectB']
    expected_b['layers'][8] = 'OnlyInProjectB'
for name, settings in [('A', authored), ('B', expected_b)]:
    project = root / name
    (project / 'Assets').mkdir(parents=True)
    (project / 'ProjectSettings').mkdir()
    if name == 'A' or mode != 'absent':
        (project / 'ProjectSettings/TagLayerSettings.json').write_text(json.dumps(settings), encoding='utf-8')
for name, expected in [('A', authored), ('B', expected_b), ('A', authored), ('B', expected_b)]:
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(root / name))
        assert json.loads(manager.serialize()) == expected, (name, manager.serialize(), expected)
        assert json.loads(manager.serialize_defaults()) == defaults
    finally:
        engine.exit()
print('SETTINGS_PROJECT_ISOLATION_PASSED')
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script, str(tmp_path), second],
                            capture_output=True, text=True, encoding="utf-8", timeout=75, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SETTINGS_PROJECT_ISOLATION_PASSED" in result.stdout
