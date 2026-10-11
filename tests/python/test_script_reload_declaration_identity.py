"""Native reload publication must not derive class identity from Scene order."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


CASES = (
    "rename-two", "rename-two-reverse", "rename-three-scenes",
    "single", "single-among-stable", "single-unmounted-stable",
    "single-reorder", "reorder", "add", "remove-live", "rename-and-add",
    "remove-unmounted-and-rename", "no-live", "move-and-rename",
    "batch-rejection", "move-unmounted-stable", "sequential-renames",
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cds", (False, True), ids=("python-fields", "native-fields"))
def test_declaration_identity(tmp_path, case, cds):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(tmp_path), case, str(int(cds))],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(project, case, cds):
    from infernux.components.component_identity import bind_asset_script_guid
    from infernux.components.registry import (
        component_types_for_script_path, publish_component_script_types,
        snapshot_component_registry_state,
    )
    from infernux.components.script_loader import (
        ScriptReloadRejected, _snapshot_script_diagnostics,
        load_all_components_from_file, set_script_error,
    )
    from infernux.engine.engine import Engine
    from infernux.engine.play_mode import PlayModeManager, ScriptReloadBatchInput
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.project_context import get_script_module_name
    from infernux.engine.runtime_dispatch import current_runtime_epoch
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    assets = project / "Assets"
    assets.mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)

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
        classes = load_all_components_from_file(str(path), register=False)
        assert len(classes) == len(declarations)
        for cls in classes:
            bind_asset_script_guid(cls, guid)
        publish_component_script_types(str(path), tuple(classes))
        instances = []
        live_classes = classes
        if case in ("single-unmounted-stable", "remove-unmounted-and-rename", "move-unmounted-stable"):
            live_classes = classes[:1]
        elif case == "no-live":
            live_classes = []
        if case in ("rename-two-reverse", "rename-three-scenes", "single-reorder"):
            live_classes = list(reversed(live_classes))
        for i, cls in enumerate(live_classes):
            scene = scenes[i % 3] if case == "rename-three-scenes" else scenes[0]
            instance = scene.create_game_object(cls.__name__).add_py_component(cls())
            instance._script_guid = guid
            instance._script_path = str(path)
            instance.value = 17.0 + i
            instances.append(instance)

        ambiguous = case.startswith("rename-two") or case in (
            "rename-three-scenes", "remove-live", "rename-and-add",
            "remove-unmounted-and-rename", "batch-rejection",
        )
        if case.startswith("rename-two") or case in ("rename-three-scenes", "no-live", "batch-rejection"):
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
            updated = list(reversed(updated))

        revisions = []
        observed_paths = [path]
        if case == "batch-rejection":
            other = assets / "Other.py"
            other.write_text(source([("Other", "other")], "old"), encoding="utf-8")
            other_import = database.import_asset(str(other))
            assert other_import, other_import.error
            other_classes = load_all_components_from_file(str(other), register=False)
            bind_asset_script_guid(other_classes[0], other_import.guid)
            publish_component_script_types(str(other), tuple(other_classes))
            instance = scenes[2].create_game_object("Other").add_py_component(other_classes[0]())
            instance._script_guid = other_import.guid
            instance.value = 29.0
            instances.append(instance)
            other_source = source([("Other", "other")], "new")
            other.write_text(other_source, encoding="utf-8")
            revisions.append(ScriptReloadBatchInput(str(other), other_import.guid, other_source.encode()))
            observed_paths.append(other)

        retire_paths = ()
        if case.startswith("move-"):
            old_path = path
            path = assets / "MovedDeclarations.py"
            old_path.rename(path)
            moved = database.move_asset(str(old_path), str(path))
            assert moved, moved.error
            retire_paths = (str(old_path),)
        candidate_source = source(updated, "new")
        path.write_text(candidate_source, encoding="utf-8")
        revisions.append(ScriptReloadBatchInput(str(path), guid, candidate_source.encode(),
                                                retire_script_paths=retire_paths))
        for observed in observed_paths:
            set_script_error(str(observed), "previous diagnostic witness")
        before_registry = snapshot_component_registry_state()
        before_diagnostics = _snapshot_script_diagnostics()
        before_epoch = current_runtime_epoch()
        before_modules = {get_script_module_name(str(p)): sys.modules[get_script_module_name(str(p))]
                          for p in observed_paths}
        before_instances = [(instance, type(instance), instance.component_id, instance.value, instance.marker())
                            for instance in instances]
        manager = PlayModeManager()
        if ambiguous:
            with pytest.raises(ScriptReloadRejected):
                manager.prepare_script_reload_batch(revisions)
            assert snapshot_component_registry_state() == before_registry
            assert _snapshot_script_diagnostics() == before_diagnostics
            assert current_runtime_epoch() is before_epoch
            for name, module in before_modules.items():
                assert sys.modules[name] is module
        else:
            batch = manager.prepare_script_reload_batch(revisions)
            assert snapshot_component_registry_state() == before_registry
            assert _snapshot_script_diagnostics() == before_diagnostics
            assert current_runtime_epoch() is before_epoch
            outcome = manager.commit_script_reload_batch(batch)
            assert outcome.success, outcome
            manager.finalize_script_reload_batch(batch)
            registered = component_types_for_script_path(str(path))
            assert {cls.__name__ for cls in registered} == {name for name, _ in updated}
            if retire_paths:
                assert not component_types_for_script_path(retire_paths[0])
        for instance, cls, identity, value, marker in before_instances:
            assert type(instance) is cls and instance.component_id == identity
            assert instance.value == value
            assert instance.marker() == (marker if ambiguous else marker.replace("-old", "-new"))
        if case == "sequential-renames":
            second_source = source([("NewAlpha", "alpha"), ("NewBeta", "beta")], "second")
            path.write_text(second_source, encoding="utf-8")
            second_batch = manager.prepare_script_reload_batch((
                ScriptReloadBatchInput(str(path), guid, second_source.encode()),
            ))
            outcome = manager.commit_script_reload_batch(second_batch)
            assert outcome.success, outcome
            manager.finalize_script_reload_batch(second_batch)
            for instance, cls, identity, value, marker in before_instances:
                assert type(instance) is cls and instance.component_id == identity
                assert instance.value == value
                assert instance.marker() == marker.replace("-old", "-second")
            assert {cls.__name__ for cls in component_types_for_script_path(str(path))} == {"NewAlpha", "NewBeta"}
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
