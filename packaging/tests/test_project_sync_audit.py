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
    (project / ".gitignore").write_text(
        (ROOT / "python" / "Infernux" / "resources" / "project_templates" / "project.gitignore.txt")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / ".gitattributes").write_text(
        (ROOT / "python" / "Infernux" / "resources" / "project_templates" / "project.gitattributes.txt")
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
            "content_hash": {"type": "string", "value": "0000000000000000"},
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
