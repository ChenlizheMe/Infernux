from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "acceptance" / "editor_project_smoke.py"
)
_SPEC = importlib.util.spec_from_file_location("editor_project_smoke", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_native_dialog_result_requires_the_exact_selected_path(tmp_path: Path) -> None:
    selected = tmp_path / "selected.txt"

    assert _MODULE._require_dialog_path(
        {"accepted": True, "cancelled": False, "path": str(selected), "error": ""},
        str(selected),
    ) == str(selected.resolve())


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"accepted": False, "cancelled": True, "path": "", "error": ""}, "did not accept"),
        ({"accepted": False, "cancelled": False, "path": "", "error": "portal failed"}, "portal failed"),
        ({"accepted": True, "cancelled": False, "path": "wrong.txt", "error": ""}, "expected"),
    ],
)
def test_native_dialog_result_rejects_cancel_error_and_wrong_path(
    tmp_path: Path, result: dict[str, object], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _MODULE._require_dialog_path(result, str(tmp_path / "selected.txt"))


@pytest.mark.parametrize(
    "diagnostic",
    [
        "[ERROR] asset import failed",
        "X Error of failed request: BadWindow",
        "VK_ERROR_DEVICE_LOST",
        "Aborted (core dumped)",
        "Segmentation fault",
    ],
)
def test_editor_smoke_fatal_scan_covers_native_process_failures(diagnostic: str) -> None:
    assert _MODULE._fatal_lines(f"normal\n{diagnostic}\n") == [diagnostic]


def test_editor_smoke_reads_only_the_new_log_suffix(tmp_path: Path) -> None:
    log = tmp_path / "editor.log"
    log.write_text("[ERROR] stale run\n", encoding="utf-8")
    start = _MODULE._log_start(log)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("current run\nX Error of failed request: BadWindow\n")

    text = _MODULE._new_log_text(log, start)

    assert "stale run" not in text
    assert _MODULE._fatal_lines(text) == ["X Error of failed request: BadWindow"]


def test_editor_smoke_consumes_engine_and_explicit_process_logs(tmp_path: Path) -> None:
    project = tmp_path / "Project"
    process_log = tmp_path / "editor.log"

    assert _MODULE._acceptance_logs(str(project), str(process_log)) == [
        project / "Logs" / "engine.log",
        process_log,
    ]


@pytest.mark.parametrize(
    ("is_loading", "current_scene_path", "expected"),
    [
        (True, "/project/Assets/Scenes/Main.scene", False),
        (False, "", True),
        (False, "/project/Assets/Scenes/Main.scene", True),
    ],
)
def test_editor_smoke_waits_for_scene_manager_deferred_load(
    is_loading: bool,
    current_scene_path: str,
    expected: bool,
) -> None:
    manager = SimpleNamespace(
        is_loading=is_loading,
        current_scene_path=current_scene_path,
    )

    assert _MODULE._scene_manager_ready(manager) is expected


@pytest.mark.parametrize(
    ("is_loading", "current_scene_path", "expected"),
    [
        (True, "", False),
        (False, "", True),
        (False, "/project/Assets/Scenes/Main.scene", False),
    ],
)
def test_editor_smoke_identifies_only_settled_pathless_bootstrap_scene(
    is_loading: bool,
    current_scene_path: str,
    expected: bool,
) -> None:
    manager = SimpleNamespace(
        is_loading=is_loading,
        current_scene_path=current_scene_path,
    )

    assert _MODULE._is_pathless_initial_scene(manager) is expected


@pytest.mark.parametrize(
    ("project_info", "expected"),
    [
        ({}, False),
        ({"active_scene": {"name": "", "path": ""}}, False),
        ({"active_scene": {"name": "Untitled Scene", "path": ""}}, True),
        ({"active_scene": {"name": "Main", "path": "/Assets/Main.scene"}}, True),
    ],
)
def test_editor_smoke_waits_for_active_scene_not_a_durable_path(
    project_info: dict[str, object], expected: bool
) -> None:
    assert _MODULE._project_has_active_scene(project_info) is expected


@pytest.mark.parametrize("log_kind", ["engine", "process"])
def test_editor_smoke_main_rejects_new_fatal_log_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    log_kind: str,
) -> None:
    project = tmp_path / "Project"
    scene = project / "Assets" / "Scenes" / "Main.scene"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}", encoding="utf-8")
    engine_log = project / "Logs" / "engine.log"
    engine_log.parent.mkdir(parents=True)
    engine_log.write_text("[ERROR] stale run\n", encoding="utf-8")
    process_log = tmp_path / "editor-process.log"
    process_log.write_text("[ERROR] stale run\n", encoding="utf-8")

    def fake_smoke(*_args: object, **kwargs: object) -> None:
        target = engine_log if log_kind == "engine" else process_log
        with target.open("a", encoding="utf-8") as stream:
            stream.write("Aborted\n")
        kwargs["outcome"]["passed"] = {"project": str(project)}

    monkeypatch.setattr(_MODULE, "_run_smoke", fake_smoke)
    monkeypatch.setattr(_MODULE, "release_engine", lambda _project: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT_PATH),
            str(project),
            "--scene",
            str(scene),
            "--process-log",
            str(process_log),
        ],
    )

    assert _MODULE.main() == 1
