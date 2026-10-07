"""IDE synchronization preserves team-owned JSONC while changing runtime keys."""
import json
from pathlib import Path

import pytest

from infernux.engine.ide_workspace import synchronize_vscode_workspace
from infernux.engine.ui import project_utils


DOCUMENTS = [
    ('{"editor.fontSize": 18}', '"editor.fontSize": 18'),
    ('{"editor.fontSize": 12, "editor.fontSize": 18}', '"editor.fontSize": 12, "editor.fontSize": 18'),
    ('{\n// Team preference\n"editor.fontSize": 18\n}', '// Team preference'),
    ('{/* Team preference */ "editor.fontSize": 18}', '/* Team preference */'),
    ('{"editor.fontSize": 18, /* keep comma */}', '/* keep comma */'),
    ('\ufeff{\r\n\t"editor.fontSize": 18, // 中文说明\r\n}\r\n', '// 中文说明\r\n'),
    ('{"[python]": {"editor.rulers": [80, 120,],}, "url": "https://host/a//b",}',
     '"[python]": {"editor.rulers": [80, 120,],}'),
    (r'{"quote": "escaped \" /* not a comment */", "path": "C:\\tools\\", "editor.fontSize": 18}',
     r'"quote": "escaped \" /* not a comment */"'),
]


@pytest.mark.parametrize("entrypoint", ["direct", "project"])
@pytest.mark.parametrize("source,untouched", DOCUMENTS)
def test_supported_jsonc_reaches_actual_ide_launch_boundary(tmp_path, monkeypatch, entrypoint, source, untouched):
    script = tmp_path / "Assets/Hello.py"
    script.parent.mkdir()
    script.write_text("import infernux as inx\n", encoding="utf-8")
    settings = tmp_path / ".vscode/settings.json"
    settings.parent.mkdir()
    settings.write_bytes(source.encode("utf-8"))
    monkeypatch.setattr(project_utils, "_find_vscode_executable", lambda: "code.exe")
    monkeypatch.setattr(project_utils, "get_ide", lambda: "vscode")
    monkeypatch.setattr(project_utils, "detect_available_ides", lambda **kwargs: ["vscode"])
    launches = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **kwargs: launches.append(args))
    opener = project_utils.open_in_vscode if entrypoint == "direct" else project_utils.open_file_with_system
    assert opener(str(script), line=17, project_root=str(tmp_path))
    assert len(launches) == 1 and launches[0][-2:] == ["--goto", f"{script}:17"]
    written = settings.read_bytes().decode("utf-8")
    assert untouched in written
    assert '"python.defaultInterpreterPath"' in written
    assert written.startswith("\ufeff") == source.startswith("\ufeff")
    if "\r\n" in source:
        assert "\n" not in written.replace("\r\n", "")
    before = [(path, path.read_bytes(), path.stat().st_mtime_ns)
              for path in (settings, tmp_path / "pyrightconfig.json")]
    synchronize_vscode_workspace(str(tmp_path))
    assert all(path.read_bytes() == payload and path.stat().st_mtime_ns == stamp
               for path, payload, stamp in before)


@pytest.mark.parametrize("settings_bad", [True, False])
@pytest.mark.parametrize("invalid", ['{"x":', '{/* incomplete', '{"x": "unterminated}', '[1,2]', '{"x":,}'])
def test_invalid_configuration_never_changes_either_file(tmp_path, settings_bad, invalid):
    settings = tmp_path / ".vscode/settings.json"
    pyright = tmp_path / "pyrightconfig.json"
    settings.parent.mkdir()
    settings.write_text(invalid if settings_bad else '{"editor.fontSize": 18}', encoding="utf-8")
    pyright.write_text('{"exclude": ["Generated"]}' if settings_bad else invalid, encoding="utf-8")
    before = [path.read_bytes() for path in (settings, pyright)]
    with pytest.raises(ValueError):
        synchronize_vscode_workspace(str(tmp_path))
    assert [path.read_bytes() for path in (settings, pyright)] == before


@pytest.mark.parametrize("obsolete", [
    '"venv": ".venv", /* first */ "exclude": ["Assets/Generated"]',
    '"exclude": ["Assets/Generated"], /* last */ "venv": ".venv"',
    '"venvPath": ".", /* middle */ "venv": ".venv", "exclude": ["Assets/Generated"],',
    '"venv": ".venv", "pythonPath": "old/python", /* only old keys */',
])
def test_pyright_jsonc_removes_owned_obsolete_keys_and_retains_comments(tmp_path, obsolete):
    pyright = tmp_path / "pyrightconfig.json"
    source = "{\n  // Team Pyright settings\n  " + obsolete + "\n}\n"
    pyright.write_bytes(source.encode("utf-8"))
    synchronize_vscode_workspace(str(tmp_path))
    written = pyright.read_text(encoding="utf-8")
    assert "// Team Pyright settings" in written
    assert obsolete[obsolete.index("/*"):obsolete.index("*/") + 2] in written
    assert '"venv"' not in written and '"venvPath"' not in written and '"pythonPath"' not in written
    if '"exclude"' in source:
        assert '"exclude": ["Assets/Generated"]' in written
    assert '"pythonVersion"' in written
    before = pyright.read_bytes()
    synchronize_vscode_workspace(str(tmp_path))
    assert pyright.read_bytes() == before


def test_semantically_current_document_keeps_nonstandard_whitespace(tmp_path):
    synchronize_vscode_workspace(str(tmp_path))
    paths = [tmp_path / ".vscode/settings.json", tmp_path / "pyrightconfig.json"]
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        path.write_bytes((json.dumps(document, separators=(",  ", " : ")) + "\r\n").encode("utf-8"))
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    synchronize_vscode_workspace(str(tmp_path))
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


@pytest.mark.parametrize("target_name", [".vscode/settings.json", "pyrightconfig.json"])
def test_external_edit_during_sync_is_preserved_by_native_atomic_write(tmp_path, monkeypatch, target_name):
    from infernux.engine import ide_workspace

    settings = tmp_path / ".vscode/settings.json"
    settings.parent.mkdir()
    settings.write_text('{"editor.fontSize": 18}', encoding="utf-8")
    pyright = tmp_path / "pyrightconfig.json"
    pyright.write_text('{"exclude": ["Generated"]}', encoding="utf-8")
    target = tmp_path / target_name
    external = '{\n// User saved during synchronization\n"foreign": "latest edit"\n}'
    writer = ide_workspace.write_document_text
    changed = False

    def external_save_before_commit(path, content, **kwargs):
        nonlocal changed
        if Path(path) == target:
            assert not changed
            changed = True
            target.write_text(external, encoding="utf-8")
        return writer(path, content, **kwargs)

    monkeypatch.setattr(ide_workspace, "write_document_text", external_save_before_commit)
    with pytest.raises(RuntimeError, match="changed outside the editor"):
        synchronize_vscode_workspace(str(tmp_path))
    assert changed and target.read_text(encoding="utf-8") == external


@pytest.mark.parametrize("bad_paths", [None, "Libraries", [1, "Libraries"]])
@pytest.mark.parametrize("settings_bad", [False, True])
def test_invalid_extra_paths_leave_configuration_unchanged(tmp_path, bad_paths, settings_bad):
    settings = tmp_path / ".vscode/settings.json"
    settings.parent.mkdir()
    pyright = tmp_path / "pyrightconfig.json"
    settings.write_text(json.dumps({"python.analysis.extraPaths": bad_paths} if settings_bad else {}),
                        encoding="utf-8")
    pyright.write_text(json.dumps({} if settings_bad else {"extraPaths": bad_paths}), encoding="utf-8")
    before = [path.read_bytes() for path in (settings, pyright)]
    with pytest.raises(ValueError, match="extraPaths"):
        synchronize_vscode_workspace(str(tmp_path))
    assert [path.read_bytes() for path in (settings, pyright)] == before
