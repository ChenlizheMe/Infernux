from __future__ import annotations

import json
from pathlib import Path

import pytest


from database import ProjectDatabase
import model.project_model as project_model_module
from model.project_model import ProjectModel
from hub_utils import HubLaunchContext
from viewmodel.control_pane_viewmodel import ControlPaneViewModel
from project_paths import inspect_existing_project


def _make_project(path: Path, *, name: str | None = None, version: str = "") -> Path:
    (path / "Assets").mkdir(parents=True)
    (path / "ProjectSettings").mkdir()
    if name:
        (path / f"{name}.ini").write_text(
            f"[Project]\nname = {name}\npath = {path}\n",
            encoding="utf-8",
        )
    if version:
        (path / ".infernux-version").write_text(f"# pin\n{version}\n", encoding="utf-8")
    return path


def test_source_project_creation_accepts_current_environment_without_version(
    monkeypatch,
):
    class NewProjectDialog:
        def __init__(self, *_args):
            pass

        @staticmethod
        def exec():
            return 1

        @staticmethod
        def get_data():
            return "SourceProject", "C:/Projects", ""

    class InitializationReached(RuntimeError):
        pass

    monkeypatch.setattr("view.new_project_view.NewProjectView", NewProjectDialog)
    monkeypatch.setattr(
        "viewmodel.control_pane_viewmodel.CustomProgressDialog",
        lambda _parent: (_ for _ in ()).throw(InitializationReached()),
    )

    viewmodel = ControlPaneViewModel(
        model=object(),
        project_list=object(),
        launch_context=HubLaunchContext.SOURCE,
    )

    with pytest.raises(InitializationReached):
        viewmodel.create_project(None)


def test_project_creation_requires_the_current_bundled_support_template(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        project_model_module,
        "__file__",
        str(tmp_path / "packaging" / "model" / "project_model.py"),
    )
    destination = tmp_path / "project.gitignore"

    with pytest.raises(RuntimeError, match="Required Infernux project template"):
        ProjectModel(None)._copy_bundled_support_file("project.gitignore.txt", str(destination), "")

    assert not destination.exists()


@pytest.mark.parametrize("name", ["Display Name", "100% Demo"])
def test_inspect_existing_project_reads_name_and_version_without_modifying(tmp_path: Path, name):
    project = _make_project(tmp_path / "folder-name", name=name, version="0.2.1")
    before = sorted(item.relative_to(project) for item in project.rglob("*"))

    info = inspect_existing_project(str(project))

    assert info.name == name
    assert info.engine_version == "0.2.1"
    assert Path(info.path) == project.resolve()
    assert sorted(item.relative_to(project) for item in project.rglob("*")) == before


def test_source_project_creation_rejects_another_engine_before_creating_files(tmp_path, monkeypatch):
    monkeypatch.setattr(project_model_module, "is_frozen", lambda: False)
    model = ProjectModel(None, runtime_manager=object())
    with pytest.raises(RuntimeError, match="exact same engine version"):
        model.init_project_folder("WrongVersion", str(tmp_path), "9.9.9")
    assert not list(tmp_path.iterdir())


def test_project_registry_preserves_identity_across_registration_duplicates_and_relocation(tmp_path: Path):
    first = _make_project(tmp_path / "one", name="Same Name", version="0.2.1")
    second = _make_project(tmp_path / "two", name="Same Name")
    relocated = _make_project(tmp_path / "Relocated", version="0.3.0")
    with ProjectDatabase(tmp_path / "projects.db") as db:
        model = ProjectModel(db)
        first_record, info = model.register_existing_project(str(first))
        second_record, _ = model.register_existing_project(str(second))
        assert first_record.name == second_record.name == "Same Name"
        assert info.engine_version == "0.2.1"
        assert first_record.project_id != second_record.project_id
        assert db.add_project("Another Name", str(first / ".")) is None
        assert len(db.all_projects()) == 2
        with pytest.raises(RuntimeError, match="already"):
            model.register_existing_project(str(first))

        updated, info = model.relocate_project(first_record.project_id, str(relocated))
        assert updated.project_id == first_record.project_id
        assert updated.name == "Relocated"
        assert Path(updated.path) == relocated.resolve()
        assert info.engine_version == "0.3.0"
        assert db.find_project_by_path(str(first)) is None
        assert db.get_project(second_record.project_id).path == str(second.resolve())


def test_new_project_uses_structural_staging_but_creates_runtime_at_final_path(tmp_path: Path, monkeypatch):
    model = ProjectModel(None)
    runtime_locations = []

    monkeypatch.setattr(
        model, "_copy_bundled_requirements", lambda dest, _version: Path(dest).write_text("", encoding="utf-8")
    )

    def create_runtime(project, python_version="", on_status=None):
        runtime_locations.append(Path(project))
        (Path(project) / ".venv").mkdir()

    monkeypatch.setattr(model, "_create_project_runtime", create_runtime)
    monkeypatch.setattr(model, "_install_infernux_in_runtime", lambda *_args, **_kwargs: None)
    default_library_locations = []
    monkeypatch.setattr(
        model,
        "_install_default_libraries",
        lambda project, on_status=None: default_library_locations.append(Path(project)),
    )
    monkeypatch.setattr(model, "_create_vscode_workspace", lambda project: (Path(project) / ".vscode").mkdir())

    result = model.init_project_folder("SafeProject", str(tmp_path), project_model_module.source_engine_version())

    assert Path(result) == (tmp_path / "SafeProject").resolve()
    assert runtime_locations == [Path(result)]
    assert default_library_locations == [Path(result)]
    assert (Path(result) / "Assets").is_dir()
    gitignore = (Path(result) / ".gitignore").read_text(encoding="utf-8")
    assert "/Library/" in gitignore
    assert "/.venv/" in gitignore
    assert "/.runtime/" in gitignore
    assert "*.meta" not in gitignore.splitlines()
    assert "!*.meta" in gitignore.splitlines()
    gitattributes = (Path(result) / ".gitattributes").read_text(encoding="utf-8")
    assert "*.scene text eol=lf" in gitattributes
    assert "*.meta text eol=lf" in gitattributes
    assert "*.fbx binary" in gitattributes
    assert not (Path(result) / "Assets" / "README.md").exists()
    assert (Path(result) / "Assets" / "Scenes" / "SampleScene.scene").is_file()
    assert (Path(result) / "Assets" / "Settings" / "Bloom.effect").is_file()
    assert (Path(result) / "Assets" / "Settings" / "ACES Tone Mapping.effect").is_file()
    assert (Path(result) / "Assets" / "Settings" / "Default Post Processing.effectgroup").is_file()
    bloom = json.loads((Path(result) / "Assets" / "Settings" / "Bloom.effect.meta").read_text(encoding="utf-8"))
    tone_mapping = json.loads(
        (Path(result) / "Assets" / "Settings" / "ACES Tone Mapping.effect.meta").read_text(encoding="utf-8")
    )
    effect_group_meta = json.loads(
        (Path(result) / "Assets" / "Settings" / "Default Post Processing.effectgroup.meta").read_text(encoding="utf-8")
    )
    bloom_guid = bloom["metadata"]["guid"]["value"]
    tone_mapping_guid = tone_mapping["metadata"]["guid"]["value"]
    effect_group_guid = effect_group_meta["metadata"]["guid"]["value"]
    assert all(
        len(guid) == 32 and set(guid) <= set("0123456789abcdef")
        for guid in (bloom_guid, tone_mapping_guid, effect_group_guid)
    )
    effect_group = json.loads(
        (Path(result) / "Assets" / "Settings" / "Default Post Processing.effectgroup").read_text(encoding="utf-8")
    )
    assert [entry["asset"]["guid"] for entry in effect_group["entries"]] == [
        bloom_guid,
        tone_mapping_guid,
    ]
    build_settings = json.loads((Path(result) / "ProjectSettings" / "BuildSettings.json").read_text(encoding="utf-8"))
    scene_path = Path(result) / "Assets" / "Scenes" / "SampleScene.scene"
    scene_meta = json.loads(scene_path.with_suffix(".scene.meta").read_text(encoding="utf-8"))
    scene_guid = scene_meta["metadata"]["guid"]["value"]
    assert scene_meta["metadata"]["resource_type"]["value"] == "DefaultText"
    assert build_settings["scene_guids"] == [scene_guid]
    assert not (Path(result) / "ProjectSettings" / "EditorSettings.json").exists()
    scene = json.loads((Path(result) / "Assets" / "Scenes" / "SampleScene.scene").read_text(encoding="utf-8"))
    assert scene["identity_format"] == "guid-v1"
    camera_id = scene["objects"][0]["components"][0]["component_id"]
    assert len(camera_id) == 32 and set(camera_id) <= set("0123456789abcdef")
    assert scene["mainCameraComponentId"] == camera_id
    assert [item["name"] for item in scene["objects"]] == [
        "Main Camera",
        "Directional Light",
        "RenderStack",
    ]
    assert (
        scene["objects"][2]["components"][0]["data"]["effect_slots"][0]["fields"]["effect"]["guid"] == effect_group_guid
    )
    camera_data = scene["objects"][0]["components"][0]["data"]
    assert camera_data["projectionMode"] == 0
    assert set(camera_data) == {
        "aspectRatio",
        "backgroundColor",
        "clearFlags",
        "cullingMask",
        "depth",
        "dithering",
        "farClip",
        "fov",
        "usePhysicalProperties",
        "iso",
        "shutterSpeed",
        "aperture",
        "focusDistance",
        "focalLength",
        "bladeCount",
        "curvature",
        "barrelClipping",
        "anamorphism",
        "sensorSize",
        "lensShift",
        "gateFit",
        "nearClip",
        "orthoSize",
        "projectionMode",
        "stopNaNs",
        "targetTextureGuid",
    }
    light_data = scene["objects"][1]["components"][0]["data"]
    assert set(light_data) == {
        "areaSize",
        "areaTwoSided",
        "baked",
        "color",
        "useColorTemperature",
        "colorTemperature",
        "cullingMask",
        "influenceDomains",
        "intensity",
        "lightType",
        "outerSpotAngle",
        "range",
        "renderMode",
        "shadowSoftness",
        "shadowStrength",
        "shadows",
        "spotAngle",
    }
    assert light_data["shadowSoftness"] == 1.5
    assert (Path(result) / ".vscode").is_dir()
    assert not list(tmp_path.glob(".infernux-create-*"))


def test_same_named_projects_have_independent_asset_identities(tmp_path: Path):
    identities = []
    for parent in ("first", "second"):
        project = tmp_path / parent / "New Project"
        project_model_module._create_default_project_content(str(project), project.name)
        identities.append(
            {
                json.loads(path.read_text(encoding="utf-8"))["metadata"]["guid"]["value"]
                for path in (project / "Assets").rglob("*.meta")
            }
        )
    assert len(identities[0]) == len(identities[1]) == 4
    assert identities[0].isdisjoint(identities[1])


def test_new_project_failure_removes_only_staging(tmp_path: Path, monkeypatch):
    unrelated = tmp_path / "keep-me"
    unrelated.mkdir()
    model = ProjectModel(None)

    monkeypatch.setattr(
        model, "_copy_bundled_requirements", lambda dest, _version: Path(dest).write_text("", encoding="utf-8")
    )
    monkeypatch.setattr(model, "_create_project_runtime", lambda *_args, **_kwargs: None)

    def fail_install(*_args, **_kwargs):
        raise RuntimeError("expected install failure")

    monkeypatch.setattr(model, "_install_infernux_in_runtime", fail_install)

    with pytest.raises(RuntimeError, match="expected install failure"):
        model.init_project_folder("FailedProject", str(tmp_path), project_model_module.source_engine_version())

    assert unrelated.is_dir()
    assert not (tmp_path / "FailedProject").exists()
    assert not list(tmp_path.glob(".infernux-create-*"))
