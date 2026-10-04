from __future__ import annotations

import importlib.util
import json
import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "maintenance" / "audit_project_sync.py"


def _module():
    spec = importlib.util.spec_from_file_location("infernux_project_sync_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "Project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    (project / ".infernux-version").write_text("0.4.1\n", encoding="utf-8")
    (project / "ProjectSettings" / "PythonRuntime.json").write_text('{"pythonVersion": "3.13"}\n', encoding="utf-8")
    (project / "ProjectSettings" / "requirements.txt").write_text("# Project dependencies\n", encoding="utf-8")
    (project / ".gitignore").write_text(
        (ROOT / "python" / "infernux" / "resources" / "project_templates" / "project.gitignore.txt")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / ".gitattributes").write_text(
        (ROOT / "python" / "infernux" / "resources" / "project_templates" / "project.gitattributes.txt")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return project


def _write_asset(project: Path, relative: str, guid: str, *, absolute_hint: str = "") -> None:
    source = project / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{}\n", encoding="utf-8")
    payload = {
        "metadata": {
            "guid": {"type": "string", "value": guid},
            "resource_type": {"type": "enum infernux::ResourceType", "value": "DefaultText"},
        }
    }
    if absolute_hint:
        payload["metadata"]["file_path"] = {"type": "string", "value": absolute_hint}
    (Path(str(source) + ".meta")).write_text(
        json.dumps(payload, indent=4) + "\n", encoding="utf-8"
    )


def test_fixture_project_has_a_portable_sync_contract():
    report = _module().audit_project(ROOT / "tests" / "fixtures" / "multiplatform_player")
    assert report.ok, report.errors
    assert report.package_count == 6


def test_audit_does_not_require_asset_sidecars_for_vcs_control_files(tmp_path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/Vendor/Notes.txt", "1" * 32)
    for relative in (
        "Assets/Vendor/.git/HEAD", "Assets/Vendor/.git/objects/source.py",
        "Assets/Vendor/.hg/state", "Assets/Vendor/.svn/state",
        "Assets/Vendor/.gitignore", "Assets/Vendor/.gitattributes",
        "Assets/Vendor/.gitmodules", "Assets/Vendor/.gitkeep", "Assets/Worktree/.git",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version control state\n", encoding="utf-8")
    report = _module().audit_project(project)
    assert report.ok, report.errors
    assert report.asset_count == report.guid_count == 1


def test_absolute_metadata_path_is_rejected(tmp_path: Path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/Scenes/Main.scene", "0123456789abcdef0123456789abcdef", absolute_hint="C:/work/Main.scene")
    report = _module().audit_project(project)
    assert not report.ok
    assert any("metadata.file_path must be project-relative" in error for error in report.errors)


def test_duplicate_guid_and_missing_sidecar_are_rejected(tmp_path: Path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/A.txt", "0123456789abcdef0123456789abcdef")
    _write_asset(project, "Assets/B.txt", "0123456789abcdef0123456789abcdef")
    (project / "Assets" / "WithoutMeta.txt").write_text("oops", encoding="utf-8")
    report = _module().audit_project(project)
    assert not report.ok
    assert any("duplicate GUID" in error for error in report.errors)
    assert any("missing tracked .meta sidecar" in error for error in report.errors)


def test_orphan_sidecar_and_unresolved_lfs_payload_are_rejected(tmp_path: Path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/Ship.fbx", "0123456789abcdef0123456789abcdef")
    (project / "Assets/Ship.fbx").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:placeholder\nsize 123\n", encoding="utf-8")
    (project / "Assets/Deleted.txt.meta").write_text("{}", encoding="utf-8")
    report = _module().audit_project(project)
    assert any("orphan .meta" in error for error in report.errors)
    assert any("Git LFS payload" in error for error in report.errors)


def _tracked_project(tmp_path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/Main.scene", "1" * 32)
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(project), "add", "."], check=True, capture_output=True)
    assert _module().audit_project(project, require_tracked=True).ok
    return project


def test_effective_ignore_override_cannot_hide_source_sidecars(tmp_path):
    project = _tracked_project(tmp_path)
    (project / "Assets" / ".gitignore").write_text("*.meta\n", encoding="utf-8")
    report = _module().audit_project(project, require_tracked=True)
    assert any("authored project input is ignored" in error for error in report.errors)


def test_effective_attributes_reject_nested_binary_scene_rule(tmp_path):
    project = _tracked_project(tmp_path)
    (project / "Assets" / ".gitattributes").write_text("*.scene binary\n", encoding="utf-8")
    report = _module().audit_project(project, require_tracked=True)
    assert any("Main.scene: effective Git attributes" in error for error in report.errors)


def test_force_added_generated_state_is_rejected(tmp_path):
    project = _tracked_project(tmp_path)
    backup = project / ".infernux-backups" / "old-project.zip"
    backup.parent.mkdir()
    backup.write_bytes(b"private")
    subprocess.run(["git", "-C", str(project), "add", "-f", ".infernux-backups"], check=True, capture_output=True)
    report = _module().audit_project(project, require_tracked=True)
    assert any("old-project.zip: ignored/generated state" in error for error in report.errors)


def test_missing_tracked_asset_and_sidecar_are_not_mistaken_for_a_complete_clone(tmp_path):
    project = _tracked_project(tmp_path)
    (project / "Assets/Main.scene").unlink()
    (project / "Assets/Main.scene.meta").unlink()
    report = _module().audit_project(project, require_tracked=True)
    assert any("Main.scene: tracked project input is missing" in error for error in report.errors)
    assert any("Main.scene.meta: tracked project input is missing" in error for error in report.errors)


def test_case_only_index_collision_is_detected_even_on_windows(tmp_path):
    project = _tracked_project(tmp_path)
    blob = subprocess.run(["git", "-C", str(project), "hash-object", "Assets/Main.scene"],
                          check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "-C", str(project), "update-index", "--add", "--cacheinfo",
                    f"100644,{blob},Assets/main.scene"], check=True, capture_output=True)
    report = _module().audit_project(project, require_tracked=True)
    assert any("case-only Git index collision" in error for error in report.errors)


def test_legacy_view_preferences_are_private_and_audit_is_read_only(tmp_path):
    project = _tracked_project(tmp_path)
    legacy = project / "ProjectSettings/EditorSettings.json"
    legacy.write_text('{"last_scene_path": "C:/old/workstation/Main.scene"}\n', encoding="utf-8")
    before = legacy.read_bytes()
    report = _module().audit_project(project, require_tracked=True)
    assert report.ok, report.errors
    assert legacy.read_bytes() == before
    subprocess.run(["git", "-C", str(project), "add", "-f", str(legacy)], check=True, capture_output=True)
    assert any("EditorSettings.json: ignored/generated state" in error
               for error in _module().audit_project(project, require_tracked=True).errors)


@pytest.mark.parametrize("suffix", [".animclip3d", ".animclip2d", ".timelinefsm", ".inxdata", ".rendertexture", ".physicMaterial"])
def test_audit_rejects_stale_structured_documents_and_absolute_paths(tmp_path, suffix):
    project = _project(tmp_path)
    _write_asset(project, "Assets/Invalid" + suffix, "1" * 32)
    path = project / ("Assets/Invalid" + suffix)
    path.write_text('{"path_hint": "C:/author/machine"}\n', encoding="utf-8")
    assert any("machine-local absolute path" in error for error in _module().audit_project(project).errors)
    path.write_text('{"name": "first", "name": "second"}\n', encoding="utf-8")
    assert any("duplicate field" in error for error in _module().audit_project(project).errors)


@pytest.mark.parametrize("suffix", [".scene", ".prefab", ".mat", ".meta", ".effect", ".json"])
def test_git_merges_independent_asset_fields_without_engine_driver(tmp_path: Path, suffix):
    project = _project(tmp_path)

    def git(*arguments, expected=0):
        result = subprocess.run(
            ["git", "-c", "user.name=Sync Test", "-c", "user.email=sync-test@example.invalid",
             "-c", "commit.gpgsign=false", "-C", str(project), *arguments],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout

    git("init", "-b", "main")
    scene = project / ("ProjectSettings" if suffix == ".json" else "Assets") / ("Main" + suffix)
    document = {"left": 1, "middle_a": 1, "middle_b": 1, "middle_c": 1, "right": 1}
    def write_document(**changes):
        scene.write_text(json.dumps({**document, **changes}, indent=2) + "\n", encoding="utf-8")
    write_document()
    git("add", ".")
    git("commit", "-m", "initial")
    git("checkout", "-b", "other")
    write_document(left=2)
    git("commit", "-am", "other change")
    git("checkout", "main")
    write_document(right=2)
    git("commit", "-am", "main change")
    git("merge", "other", "--no-edit")
    assert not git("ls-files", "--unmerged")
    assert json.loads(scene.read_text(encoding="utf-8")) == {**document, "left": 2, "right": 2}
    assert not git("config", "--get", "merge.infernux.driver", expected=1)
    git("checkout", "-b", "conflicting")
    write_document(left=3, right=2)
    git("commit", "-am", "same property other")
    git("checkout", "main")
    write_document(left=4, right=2)
    git("commit", "-am", "same property main")
    git("merge", "conflicting", expected=1)
    assert scene.relative_to(project).as_posix() in git("ls-files", "--unmerged")
    assert "<<<<<<<" in scene.read_text(encoding="utf-8")
