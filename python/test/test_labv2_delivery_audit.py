from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.acceptance.labv2_delivery_audit import SCENE_SPECS, audit_labv2


def _write_meta(path: Path, guid: str) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "guid": {"type": "string", "value": guid},
                }
            }
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, include_all_build_scenes: bool = True) -> Path:
    project = tmp_path / "Infernux041Labv2"
    (project / "Assets" / "Scenes").mkdir(parents=True)
    (project / "Assets" / "Acceptance" / "A06UIShader").mkdir(parents=True)
    (project / "Assets" / "Scripts").mkdir(parents=True)
    (project / "Assets" / "Materials").mkdir(parents=True)
    (project / "Assets" / "Models").mkdir(parents=True)
    for index, relative in enumerate(
        (
            *[item["path"] for item in SCENE_SPECS],
            "Assets/Scripts/DemoHubController.py",
            "Assets/Scripts/GPUJelly.py",
            "Assets/Scripts/CpuJitDemo.py",
            "Assets/Materials/XPBDJelly.mat",
            "Assets/Models/A16EditorAcceptance.blend",
            "Assets/Models/AAA_Embedded_PBR.gltf",
        ),
        1,
    ):
        candidate = project / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("fixture\n", encoding="utf-8")
        if candidate.suffix == ".scene":
            _write_meta(candidate.with_name(candidate.name + ".meta"), f"{index:032x}")

    all_guids = [
        _write_and_read_guid(project / item["path"])
        for item in SCENE_SPECS
    ]
    selected = all_guids if include_all_build_scenes else all_guids[:2]
    settings = {
        "game_name": "Infernux041Labv2",
        "scene_guids": selected,
    }
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings" / "BuildSettings.json").write_text(
        json.dumps(settings), encoding="utf-8"
    )
    return project


def _write_and_read_guid(scene: Path) -> str:
    payload = json.loads(scene.with_name(scene.name + ".meta").read_text(encoding="utf-8"))
    return str(payload["metadata"]["guid"]["value"])


def test_source_audit_reports_all_documented_scenes_and_never_claims_runtime(tmp_path):
    project = _fixture(tmp_path)

    report = audit_labv2(project)

    assert report["status"] == "passed"
    assert len(report["scenes"]) == len(SCENE_SPECS)
    assert all(item["source_present"] and item["meta_present"] for item in report["scenes"])
    assert report["runtime_acceptance"] == "not_claimed"
    assert report["platform_acceptance"] == "not_claimed"


def test_source_audit_warns_when_authoring_scene_is_not_in_player_settings(tmp_path):
    project = _fixture(tmp_path, include_all_build_scenes=False)

    report = audit_labv2(project)

    assert report["status"] == "passed"
    assert len(report["warnings"]) == len(SCENE_SPECS) - 2
    with pytest.raises(AssertionError):
        assert audit_labv2(project, require_build_scenes=True)["status"] == "passed"


def test_source_audit_rejects_missing_documented_scene(tmp_path):
    project = _fixture(tmp_path)
    (project / SCENE_SPECS[0]["path"]).unlink()

    report = audit_labv2(project)

    assert report["status"] == "incomplete"
    assert any("missing documented scene" in item for item in report["errors"])
