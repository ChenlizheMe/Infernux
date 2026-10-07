from __future__ import annotations

import importlib
import json
import tempfile
import zipfile
from pathlib import Path


def _load_project_model(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "packaging"))
    monkeypatch.syspath_prepend(str(repo_root / "packaging" / "model"))
    return importlib.import_module("project_model")


def _load_embed_runtime_manager(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "packaging"))
    return importlib.import_module("embed_runtime_manager")


def _bind_python(project: Path, version: str) -> None:
    from project_python_runtime import write_project_python_version

    write_project_python_version(project, version)


class _FakeVersionManager:
    def __init__(self, wheel_path: str) -> None:
        self._wheel_path = wheel_path

    def get_wheel_path(
        self, _engine_version: str, _python_version: str | None = None
    ) -> str:
        return self._wheel_path


def test_hub_generated_assets_are_readable_by_native_asset_database(engine, scene, monkeypatch):
    from infernux.lib import ResourceMeta, ResourceType
    from infernux.engine.build_settings import load_build_settings_for_build
    from infernux.engine.component_restore import deserialize_scene_document_transactionally
    from infernux.engine.scene_authoring import decode_scene_document
    from infernux.renderstack.render_stack import RenderStack
    from infernux.renderstack.render_effect_asset import parse_render_effect_document

    project_model = _load_project_model(monkeypatch)
    database = engine.get_asset_database()
    with tempfile.TemporaryDirectory(prefix="hub-template-", dir=database.assets_root) as directory:
        project = Path(directory)
        project_model._create_default_project_content(str(project), project.name)
        identities = {}
        for path in (project / "Assets").rglob("*.meta"):
            document = json.loads(path.read_text(encoding="utf-8"))
            meta = ResourceMeta()
            meta.deserialize_document(document)
            asset = Path(str(path)[:-5])
            identities[asset] = meta.get_guid()
            assert meta.get_resource_type() == (
                ResourceType.DefaultText if asset.suffix == ".scene" else ResourceType.RenderEffect
            )
            if asset.suffix != ".scene":
                parse_render_effect_document(asset.read_text(encoding="utf-8"))
        assert len(identities) == 4
        database.refresh()
        for asset, guid in identities.items():
            assert database.get_guid_from_path(str(asset)) == guid
            assert database.get_meta_by_guid(guid).get_resource_type() == database.get_resource_type(str(asset))
        settings = load_build_settings_for_build(str(project))
        scene_path = project / "Assets" / "Scenes" / "SampleScene.scene"
        assert settings["scene_guids"] == [identities[scene_path]]
        assert not (project / "ProjectSettings" / "EditorSettings.json").exists()
        document = json.loads(scene_path.read_text(encoding="utf-8"))
        assert document["identity_format"] == "guid-v1"
        assert deserialize_scene_document_transactionally(scene, decode_scene_document(document), database)
        assert scene.find("Main Camera").get_component("Camera") is not None
        assert scene.find("Directional Light").get_component("Light") is not None
        stack = scene.find("RenderStack").get_py_component(RenderStack)
        assert stack is not None
        assert len(stack.effect_slots) == 1
        effect_group = project / "Assets" / "Settings" / "Default Post Processing.effectgroup"
        assert stack.effect_slots[0].effect_ref.guid == identities[effect_group]
        # A second scan must retain seeded GUIDs and accept native-written metadata.
        database.refresh()
        assert all(database.get_guid_from_path(str(asset)) == guid for asset, guid in identities.items())
    database.refresh()


def test_vscode_workspace_uses_current_pyright_interpreter_settings(
    tmp_path, monkeypatch
):
    project_model = _load_project_model(monkeypatch)
    monkeypatch.setattr(project_model, "is_frozen", lambda: False)
    project_dir = tmp_path / "project"

    _bind_python(project_dir, f"{project_model.sys.version_info.major}.{project_model.sys.version_info.minor}")
    project_model.ProjectModel._create_vscode_workspace(str(project_dir))

    settings = json.loads(
        (project_dir / ".vscode" / "settings.json").read_text(encoding="utf-8")
    )
    pyright = json.loads(
        (project_dir / "pyrightconfig.json").read_text(encoding="utf-8")
    )
    assert settings["python.defaultInterpreterPath"] == (
        project_model.ProjectModel._vscode_python_path(str(project_dir))
    )
    assert "python.pythonPath" not in settings
    assert settings["python.defaultInterpreterPath"] == project_model.sys.executable.replace("\\", "/")
    assert settings["python.analysis.extraPaths"] == [
        project_model.sysconfig.get_path("purelib").replace("\\", "/")
    ]
    assert "venvPath" not in pyright
    assert "venv" not in pyright
    assert "pythonPath" not in pyright


def test_frozen_vscode_workspace_keeps_private_runtime_paths_portable(tmp_path, monkeypatch):
    project_model = _load_project_model(monkeypatch)
    monkeypatch.setattr(project_model, "is_frozen", lambda: True)
    project_dir = tmp_path / "project"
    _bind_python(project_dir, "3.13")
    project_model.ProjectModel._create_vscode_workspace(str(project_dir))
    settings = json.loads((project_dir / ".vscode/settings.json").read_text())
    pyright = json.loads((project_dir / "pyrightconfig.json").read_text())
    assert settings["python.defaultInterpreterPath"].startswith("${workspaceFolder}/.runtime/")
    assert settings["python.analysis.extraPaths"][0].startswith(".runtime/")
    assert pyright["extraPaths"] == settings["python.analysis.extraPaths"]
    assert "venv" not in pyright


def _write_infernux_wheel(path: Path, version: str = "0.1.6") -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("infernux/__init__.py", "__version__ = '0.1.6'\n")
        wheel.writestr("infernux/lib/__init__.py", "")
        wheel.writestr("infernux/lib/_Infernux.cp312-win_amd64.pyd", b"native")
        wheel.writestr(
            f"infernux-{version}.dist-info/METADATA",
            f"Name: Infernux\nVersion: {version}\nRequires-Dist: numpy>=1.21.0\n",
        )
        wheel.writestr(f"infernux-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        wheel.writestr(f"infernux-{version}.dist-info/RECORD", "")


def test_frozen_project_runtime_installs_infernux_by_extracting_wheel(tmp_path, monkeypatch):
    project_model = _load_project_model(monkeypatch)
    monkeypatch.setattr(project_model, "is_frozen", lambda: True)

    monkeypatch.setattr(project_model.ProjectModel, "validate_python_runtime", staticmethod(lambda _python: None))

    wheel_path = tmp_path / "infernux-0.1.6-cp312-cp312-win_amd64.whl"
    _write_infernux_wheel(wheel_path)
    project_dir = tmp_path / "project"
    _bind_python(project_dir, "3.12")
    runtime_dir = project_dir / ".runtime" / "python312"
    runtime_dir.mkdir(parents=True)
    project_python = Path(
        project_model.ProjectModel._get_project_python(str(project_dir))
    )
    site_packages = Path(
        project_model.ProjectModel._get_site_packages(str(project_dir))
    )
    (site_packages / "numpy").mkdir(parents=True)
    project_python.parent.mkdir(parents=True, exist_ok=True)
    project_python.write_text("", encoding="utf-8")

    captured_args: list[list[str]] = []

    def fake_run_hidden(args: list[str], *, timeout: int):
        captured_args.append(args)

    monkeypatch.setattr(project_model, "_run_hidden", fake_run_hidden)

    model = project_model.ProjectModel(None, version_manager=_FakeVersionManager(str(wheel_path)))
    model._install_infernux_in_runtime(str(project_dir), "0.1.6")

    assert captured_args == []
    assert (site_packages / "infernux" / "__init__.py").is_file()
    assert (site_packages / "infernux" / "lib" / "_Infernux.cp312-win_amd64.pyd").is_file()
    assert (site_packages / "infernux-0.1.6.dist-info" / "METADATA").is_file()
    assert (site_packages / "numpy").is_dir()


def test_matching_frozen_project_runtime_skips_reinstall_when_native_import_valid(tmp_path, monkeypatch):
    project_model = _load_project_model(monkeypatch)
    monkeypatch.setattr(project_model, "is_frozen", lambda: True)
    monkeypatch.setattr(project_model.ProjectModel, "validate_python_runtime", staticmethod(lambda _python: None))

    wheel_path = tmp_path / "infernux-0.1.6-cp312-cp312-win_amd64.whl"
    wheel_path.write_bytes(b"wheel")
    project_dir = tmp_path / "project"
    _bind_python(project_dir, "3.12")
    runtime_dir = project_dir / ".runtime" / "python312"
    runtime_dir.mkdir(parents=True)
    project_python = Path(
        project_model.ProjectModel._get_project_python(str(project_dir))
    )
    site_packages = Path(
        project_model.ProjectModel._get_site_packages(str(project_dir))
    )
    (site_packages / "infernux-0.1.6.dist-info").mkdir(parents=True)
    project_python.parent.mkdir(parents=True, exist_ok=True)
    project_python.write_text("", encoding="utf-8")

    marker = Path(project_model._project_wheel_marker(str(project_dir)))
    marker.write_text(
        project_model._wheel_install_fingerprint(str(wheel_path)), encoding="utf-8"
    )

    def fail_run_hidden(_args: list[str], *, timeout: int):
        raise AssertionError("pip should not run for a valid matching runtime")

    monkeypatch.setattr(project_model, "_run_hidden", fail_run_hidden)

    model = project_model.ProjectModel(None, version_manager=_FakeVersionManager(str(wheel_path)))
    model._install_infernux_in_runtime(str(project_dir), "0.1.6")


def test_frozen_project_runtime_direct_install_replaces_old_infernux_only(tmp_path, monkeypatch):
    project_model = _load_project_model(monkeypatch)
    monkeypatch.setattr(project_model, "is_frozen", lambda: True)
    monkeypatch.setattr(project_model.ProjectModel, "validate_python_runtime", staticmethod(lambda _python: None))

    wheel_path = tmp_path / "infernux-0.1.6-cp312-cp312-win_amd64.whl"
    _write_infernux_wheel(wheel_path)
    project_dir = tmp_path / "project"
    _bind_python(project_dir, "3.12")
    runtime_dir = project_dir / ".runtime" / "python312"
    runtime_dir.mkdir(parents=True)
    project_python = Path(
        project_model.ProjectModel._get_project_python(str(project_dir))
    )
    site_packages = Path(
        project_model.ProjectModel._get_site_packages(str(project_dir))
    )
    old_package = site_packages / "infernux"
    old_dist_info = site_packages / "infernux-0.1.5.dist-info"
    dependency_dir = site_packages / "numba"
    old_package.mkdir(parents=True)
    old_dist_info.mkdir()
    dependency_dir.mkdir()
    (old_package / "old.py").write_text("old = True\n", encoding="utf-8")
    project_python.parent.mkdir(parents=True, exist_ok=True)
    project_python.write_text("", encoding="utf-8")

    model = project_model.ProjectModel(None, version_manager=_FakeVersionManager(str(wheel_path)))
    model._install_infernux_in_runtime(str(project_dir), "0.1.6")

    assert not (site_packages / "infernux" / "old.py").exists()
    assert not old_dist_info.exists()
    assert (site_packages / "infernux" / "__init__.py").is_file()
    assert (site_packages / "infernux-0.1.6.dist-info" / "METADATA").is_file()
    assert dependency_dir.is_dir()


def test_dev_project_runtime_uses_active_python_and_system_site_packages(tmp_path, monkeypatch):
    project_model = _load_project_model(monkeypatch)
    monkeypatch.setattr(project_model, "is_frozen", lambda: False)

    captured_args: list[list[str]] = []

    def fake_run_hidden(args: list[str], *, timeout: int):
        captured_args.append(args)

    monkeypatch.setattr(project_model, "_run_hidden", fake_run_hidden)

    model = project_model.ProjectModel(None)
    model._create_project_runtime(str(tmp_path / "project"))

    assert captured_args == [
        [project_model.sys.executable, "-m", "venv", "--copies", "--system-site-packages", str(tmp_path / "project" / ".venv")]
    ]


def test_dev_project_install_validates_inherited_current_environment(tmp_path, monkeypatch):
    project_model = _load_project_model(monkeypatch)
    monkeypatch.setattr(project_model, "is_frozen", lambda: False)

    project_dir = tmp_path / "project"
    python_exe = Path(
        project_model.ProjectModel._get_project_python(str(project_dir))
    )
    python_exe.parent.mkdir(parents=True)
    python_exe.write_text("", encoding="utf-8")

    validated: list[str] = []
    monkeypatch.setattr(
        project_model.ProjectModel,
        "validate_python_runtime",
        staticmethod(validated.append),
    )

    model = project_model.ProjectModel(None)
    model._install_infernux_in_runtime(str(project_dir))

    assert validated == [str(python_exe)]


def test_project_runtime_copy_skips_cache_and_test_artifacts(tmp_path, monkeypatch):
    runtime_manager = _load_embed_runtime_manager(monkeypatch)
    source = tmp_path / "source"
    package_dir = source / "Lib" / "site-packages" / "pkg"
    cache_dir = package_dir / "__pycache__"
    tests_dir = package_dir / "tests"
    cache_dir.mkdir(parents=True)
    tests_dir.mkdir()
    (package_dir / "module.py").write_text("x = 1\n", encoding="utf-8")
    (cache_dir / "module.cpython-312.pyc").write_bytes(b"cache")
    (tests_dir / "test_module.py").write_text("def test_x(): pass\n", encoding="utf-8")

    dest = tmp_path / "dest"
    runtime_manager._copy_project_runtime_tree(str(source), str(dest))

    assert (dest / "Lib" / "site-packages" / "pkg" / "module.py").is_file()
    assert not (dest / "Lib" / "site-packages" / "pkg" / "__pycache__").exists()
    assert not (dest / "Lib" / "site-packages" / "pkg" / "tests").exists()
