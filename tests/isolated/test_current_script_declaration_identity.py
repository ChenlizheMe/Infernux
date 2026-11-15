"""Current-source publication preserves identity without keeping old bodies alive."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

if __name__ != "__main__":
    from tests.isolated import engine_python_path


CASES = (
    "rename-two", "rename-two-reverse", "rename-three-scenes",
    "single", "single-among-stable", "single-unmounted-stable",
    "single-reorder", "reorder", "add", "remove-live", "rename-and-add",
    "remove-unmounted-and-rename", "no-live", "move-and-rename",
    "independent-publications", "move-unmounted-stable", "sequential-renames",
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cds", (False, True), ids=("python-fields", "native-fields"))
def test_current_declaration_identity(tmp_path, case, cds):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(tmp_path), case, str(int(cds))],
        env={**os.environ, "PYTHONPATH": engine_python_path()},
        capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(project, case, cds):
    from infernux.components.missing_script import MissingScript
    from infernux.components.registry import component_types_for_script_path
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.reload.components import ComponentAdapter
    from infernux.engine.reload.coordinator import ReloadCoordinator
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    coordinator = None

    def source(declarations, revision):
        text = "from infernux.components import InxComponent\n"
        for name, role in declarations:
            text += (f"class {name}(InxComponent):\n"
                     f"    _uses_component_data_store = {cds}\n"
                     "    value: float = 1.0\n"
                     f"    def marker(self): return '{role}-{revision}'\n"
                     "    def update(self, delta_time): self.value += delta_time\n")
        return text

    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        scenes = [SceneManager.instance().create_scene(f"Identity-{i}") for i in range(3)]
        SceneManager.instance().set_active_scene(scenes[0])
        coordinator = ReloadCoordinator(str(project), adapter=ComponentAdapter(lambda: scenes))
        declarations = [("Alpha", "alpha"), ("Beta", "beta")]
        if case == "single":
            declarations = declarations[:1]
        elif case == "rename-three-scenes":
            declarations.append(("Gamma", "gamma"))
        path = assets / "Declarations.py"
        path.write_text(source(declarations, "old"), encoding="utf-8")
        imported = database.import_asset(str(path))
        assert imported, imported.error
        guid = imported.guid
        coordinator.ensure_ready((str(path),))
        classes = component_types_for_script_path(str(path))
        assert len(classes) == len(declarations)
        live_classes = classes
        if case in ("single-unmounted-stable", "remove-unmounted-and-rename", "move-unmounted-stable"):
            live_classes = classes[:1]
        elif case == "no-live":
            live_classes = ()
        if case in ("rename-two-reverse", "rename-three-scenes", "single-reorder"):
            live_classes = tuple(reversed(live_classes))
        instances = []
        for i, cls in enumerate(live_classes):
            scene = scenes[i % 3] if case == "rename-three-scenes" else scenes[0]
            obj = scene.create_game_object(cls.__name__)
            instance = obj.add_py_component(cls())
            instance._script_guid = guid
            instance._script_path = str(path)
            instance.value = 17.0 + i
            instances.append((obj, instance, instance.component_id, instance.value, instance.marker()))

        if case.startswith("rename-two") or case in ("rename-three-scenes", "no-live", "independent-publications"):
            updated = [("New" + name, role) for name, role in declarations]
        elif case.startswith(("single", "move-")) or case == "sequential-renames":
            updated = [("NewAlpha" if name == "Alpha" else name, role) for name, role in declarations]
        elif case == "remove-unmounted-and-rename":
            updated = [("NewAlpha", "alpha")]
        elif case == "remove-live":
            updated = declarations[:1]
        elif case == "rename-and-add":
            updated = [("NewAlpha", "alpha"), ("Beta", "beta"), ("Gamma", "gamma")]
        elif case == "add":
            updated = declarations + [("Gamma", "gamma")]
        else:
            updated = declarations
        if case in ("reorder", "single-reorder"):
            updated.reverse()

        paths = []
        other_object = None
        if case == "independent-publications":
            other = assets / "Other.py"
            other.write_text(source([("Other", "other")], "old"), encoding="utf-8")
            assert database.import_asset(str(other))
            coordinator.ensure_ready((str(other),))
            other_class, = component_types_for_script_path(str(other))
            other_object = scenes[2].create_game_object("Other")
            other_instance = other_object.add_py_component(other_class())
            other_instance._script_path = str(other)
            other_instance.value = 29.0
            other.write_text(source([("Other", "other")], "new"), encoding="utf-8")
            paths.append(str(other))

        removed_paths = ()
        if case.startswith("move-"):
            old_path = path
            path = assets / "MovedDeclarations.py"
            old_path.rename(path)
            moved = database.move_asset(str(old_path), str(path))
            assert moved, moved.error
            assert path.exists() and not old_path.exists()
            removed_paths = (str(old_path),)
        path.write_text(source(updated, "new"), encoding="utf-8")
        paths.append(str(path))
        coordinator.request(paths, removed=removed_paths)
        coordinator.compile_pending()
        coordinator.ensure_ready(paths)
        assert {cls.__name__ for cls in component_types_for_script_path(str(path))} == {name for name, _ in updated}
        if removed_paths:
            assert not component_types_for_script_path(removed_paths[0])

        old_names = {name for name, _ in declarations}
        new_names = {name for name, _ in updated}
        removed, added = old_names - new_names, new_names - old_names
        for obj, original, identity, value, marker in instances:
            current, = obj.get_py_components()
            assert current is not original
            assert current.component_id == identity
            assert original._reload_owner.closed
            name = type(original).__name__
            matched = name in new_names or (removed == {name} and len(added) == 1)
            if matched:
                assert not isinstance(current, MissingScript)
                assert current.value == value
                assert current.marker() == marker.replace("-old", "-new")
                assert Path(current._script_path) == path
            else:
                assert isinstance(current, MissingScript)
                assert not current.enabled
                assert current._serialize_fields_document()["value"] == value
        if other_object is not None:
            other_current, = other_object.get_py_components()
            assert other_current.marker() == "other-new" and other_current.value == 29.0

        if case == "sequential-renames":
            path.write_text(source([("NewAlpha", "alpha"), ("NewBeta", "beta")], "second"), encoding="utf-8")
            coordinator.ensure_ready((str(path),))
            for obj, _original, identity, value, marker in instances:
                current, = obj.get_py_components()
                assert current.component_id == identity and current.value == value
                assert current.marker() == marker.replace("-old", "-second")
    finally:
        if coordinator is not None:
            coordinator.close()
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
