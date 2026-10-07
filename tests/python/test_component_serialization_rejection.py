"""Invalid mutable fields must not silently replace authored data on save."""
import math
from pathlib import Path

import pytest

from infernux.components import InxComponent, serialized_field
from infernux.components.fields import get_raw_field_value
from infernux.core.assets import AssetManager
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine import project_context
from infernux.engine.prefab_manager import save_prefab
from infernux.engine.scene_manager import SceneFileManager


class PuzzleSaveData(InxComponent):
    values: list[float] = serialized_field(default=[7.0, 8.0])
    title: str = serialized_field(default="default")


@pytest.mark.parametrize("boundary", ["_serialize_fields_document", "_serialize_fields"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative-inf"])
def test_component_rejects_nonfinite_list_without_default_substitution(boundary, invalid):
    component = PuzzleSaveData()
    component.values = [42.0, 43.0]
    component.title = "authored"
    component.values.append(invalid)
    original = get_raw_field_value(component, "values")
    with pytest.raises(ValueError, match=r"PuzzleSaveData\.values\[2\].*finite"):
        getattr(component, boundary)()
    assert get_raw_field_value(component, "values") is original
    assert original[:2] == [42.0, 43.0] and not math.isfinite(original[2])
    assert component.title == "authored"
    original.pop()
    assert component._serialize_fields_document()["values"] == [42.0, 43.0]


@pytest.mark.parametrize("boundary", ["_serialize_fields_document", "_serialize_fields"])
@pytest.mark.parametrize("invalid", ["wrong element", True])
def test_component_save_rejects_mutated_list_element_type(boundary, invalid):
    component = PuzzleSaveData()
    component.values = [42.0, 43.0]
    component.values.append(invalid)
    with pytest.raises(TypeError, match=r"PuzzleSaveData\.values\[2\]"):
        getattr(component, boundary)()
    assert component.values == [42.0, 43.0, invalid]


@pytest.mark.parametrize("boundary", ["native-scene", "editor-scene", "prefab"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "negative-inf"])
def test_invalid_component_save_preserves_file_and_can_be_explicitly_corrected(
    engine, scene, tmp_path, monkeypatch, boundary, existing, invalid,
):
    database = engine.get_asset_database()
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    path = folder / ("Puzzle.prefab" if boundary == "prefab" else "Puzzle.scene")
    owner = scene.create_game_object("PuzzleRoot")
    component = owner.add_py_component(PuzzleSaveData())
    component.values = [42.0, 43.0]
    component.title = "authored"
    core = None
    previous_root = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
    if boundary == "native-scene":
        save = lambda: scene.save_to_file(str(path))
    elif boundary == "prefab":
        save = lambda: save_prefab(owner, str(path), database)
    else:
        monkeypatch.setattr(AssetManager, "_asset_database", database)
        monkeypatch.setattr(SceneFileManager, "_instance", None)
        core = EditorInteractionCore()
        files = SceneFileManager()
        files.set_asset_database(database)
        files.register_loaded_scene(scene, str(path))
        assert files.activate_loaded_scene(scene)
        save = lambda: files._do_save(str(path))
    try:
        if existing:
            assert save()
        baseline = path.read_bytes() if path.exists() else None
        sidecar = path.with_name(path.name + ".meta")
        meta = sidecar.read_bytes() if sidecar.exists() else None
        component.values.append(invalid)
        assert not save()
        assert (path.read_bytes() if path.exists() else None) == baseline
        assert (sidecar.read_bytes() if sidecar.exists() else None) == meta
        assert component.values[:2] == [42.0, 43.0]
        assert not math.isfinite(component.values[2])
        assert component.title == "authored"
        component.values.pop()
        assert save()
        assert path.is_file()
        assert component._serialize_fields_document()["values"] == [42.0, 43.0]
    finally:
        if core is not None:
            core.shutdown()
        project_context.set_project_root(previous_root)
