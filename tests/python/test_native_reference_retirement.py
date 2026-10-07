"""Real native retirement across property, method, and argument binding boundaries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run(project, kind, retirement, access):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(project), kind, retirement, access],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((project / "retirement-evidence.json").read_text(encoding="utf-8"))["status"] == "passed"


@pytest.mark.parametrize("kind", ["object", "transform", "component"])
@pytest.mark.parametrize("retirement", ["destroy", "unload", "replace"])
@pytest.mark.parametrize("access", ["truth", "property", "saved_method"])
def test_retired_native_reference_rejects_access(tmp_path, kind, retirement, access):
    _run(tmp_path, kind, retirement, access)


@pytest.mark.parametrize("access", ["truth", "property", "saved_method"])
def test_unloaded_scene_reference_rejects_access(tmp_path, access):
    _run(tmp_path, "scene", "unload", access)


@pytest.mark.parametrize("kind", ["object", "component"])
@pytest.mark.parametrize("retirement", ["destroy", "unload", "replace"])
def test_retired_native_reference_cannot_enter_another_native_call(tmp_path, kind, retirement):
    _run(tmp_path, kind, retirement, "argument")


@pytest.mark.parametrize("retirement", ["destroy", "unload", "replace"])
def test_retained_raycast_record_keeps_values_without_reviving_targets(tmp_path, retirement):
    _run(tmp_path, "hit", retirement, "property")


@pytest.mark.parametrize("retirement", ["destroy", "unload", "replace"])
@pytest.mark.parametrize("conversion", ["scalar", "component", "vector_argument", "vector_iteration", "builtin_argument"])
def test_native_call_revalidates_after_python_argument_conversion(tmp_path, retirement, conversion):
    _run(tmp_path, "object", retirement, f"reentrant_{conversion}")


@pytest.mark.parametrize("retirement", ["destroy", "unload", "replace"])
@pytest.mark.parametrize("operation", ["component_validate", "component_document", "prefab_document",
                                       "object_document", "renderer_parameter"])
def test_native_call_converts_user_data_before_resolving_owner(tmp_path, retirement, operation):
    _run(tmp_path, "object", retirement, f"conversion_{operation}")


@pytest.mark.parametrize("operation", ["scene_document", "scene_retained", "scene_instantiate"])
def test_scene_document_conversion_cannot_use_unloaded_scene(tmp_path, operation):
    _run(tmp_path, "object", "unload", f"conversion_{operation}")


@pytest.mark.parametrize("retirement", ["destroy", "unload", "replace"])
@pytest.mark.parametrize("operation", ["mesh_normals", "mesh_tangents", "clone_rotations", "clone_scales"])
def test_optional_array_conversion_cannot_retire_native_argument(tmp_path, retirement, operation):
    _run(tmp_path, "object", retirement, f"array_{operation}")


def exercise(project, kind, retirement, access):
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import InvalidNativeObjectError, LogLevel, Physics, RuntimeMode, SceneManager, Vector3

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    evidence = dict(kind=kind, retirement=retirement, access=access)

    def checkpoint(stage):
        print(json.dumps(dict(evidence, checkpoint=stage)), flush=True)

    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        source = manager.create_scene("RetiredWorld")
        retained = manager.create_scene("RetainedWorld")
        survivor = retained.create_game_object("Survivor")
        other_collider = survivor.add_component("BoxCollider")._cpp_component
        survivor.transform.position = Vector3(20, 0, 0)
        owner = source.create_game_object("RetiredObject")
        collider = owner.add_component("BoxCollider")._cpp_component
        if kind == "object":
            target, attribute, method = owner, "name", owner.get_transform
        elif kind == "transform":
            target, attribute, method = owner.transform, "position", owner.transform.local_to_world_matrix
        elif kind == "component":
            target, attribute, method = collider, "enabled", collider.serialize
        elif kind == "scene":
            target, attribute, method = source, "name", source.get_root_objects
        else:
            hit = Physics.raycast(Vector3(0, 3, 0), Vector3(0, -1, 0), 10)
            assert hit is not None and hit.game_object.id == owner.id
            distance, point = hit.distance, tuple((hit.point.x, hit.point.y, hit.point.z))
            target = owner
        handle = target.handle if kind != "scene" else None
        world = source.world_id
        checkpoint("before-retirement")
        def retire():
            if retirement == "unload":
                manager.unload_scene(source)
                assert all(manager.get_scene_at(index).world_id != world for index in range(manager.scene_count))
            elif retirement == "replace":
                assert source._commit_document(source.serialize_document())
            elif kind == "component":
                assert owner.remove_component(collider)
            else:
                source.destroy_game_object(owner)
                source.process_pending_destroys()
            checkpoint("retirement-triggered")

        if access.startswith("array_"):
            import numpy as np

            class RetiringArray:
                def __init__(self, shape):
                    self.shape = shape

                def __array__(self, dtype=None, copy=None):
                    retire()
                    return np.zeros(self.shape, dtype=dtype)

            with pytest.raises(InvalidNativeObjectError):
                if access.startswith("array_mesh_"):
                    renderer = owner.add_component("MeshRenderer")._cpp_component
                    renderer.set_inline_mesh_data(
                        np.zeros((3, 3)), RetiringArray((3, 3)) if access.endswith("normals") else None,
                        np.zeros((3, 2)), np.array([0, 1, 2], dtype=np.uint32), "RetirementProbe",
                        RetiringArray((3, 4)) if access.endswith("tangents") else None,
                    )
                else:
                    source._clone_game_objects(
                        owner, np.zeros((1, 3)),
                        RetiringArray((1, 4)) if access.endswith("rotations") else None,
                        RetiringArray((1, 3)) if access.endswith("scales") else None,
                    )
            assert not owner and not collider
            (project / "retirement-evidence.json").write_text(
                json.dumps(dict(evidence, status="passed"), indent=2), encoding="utf-8")
            return

        if access.startswith("conversion_"):
            class RetiringJsonList(list):
                def __iter__(self):
                    retire()
                    yield 0

            class RetiringColor(list):
                def __getitem__(self, index):
                    if index == 0:
                        retire()
                    return super().__getitem__(index)

            if access in ("conversion_component_validate", "conversion_component_document"):
                document = collider.serialize_document()
                invoke = collider.validate_document if access.endswith("validate") else collider.deserialize_document
            elif access in ("conversion_scene_document", "conversion_scene_retained"):
                document = source.serialize_document()
                invoke = source._commit_document if access.endswith("document") else source._commit_document_retaining_world
            else:
                document = owner.serialize_document()
                if access == "conversion_prefab_document":
                    invoke = lambda value: setattr(owner, "_prefab_source_document", value)
                elif access == "conversion_scene_instantiate":
                    invoke = lambda value: source._instantiate_document(value, owner)
                elif access == "conversion_renderer_parameter":
                    from infernux.lib import InxMaterial

                    renderer = owner.add_component("MeshRenderer")._cpp_component
                    # Headless projects have no implicit rendering material.
                    # Establish a valid parameter path before retiring its owner.
                    material = InxMaterial("RetirementProbe", "Unlit")
                    material.set_color("baseColor", (1, 0, 0, 1))
                    renderer.set_material(0, material)
                    renderer.set_parameter("baseColor", (1, 0, 0, 1))
                    assert renderer.get_parameter("baseColor") == (1, 0, 0, 1)
                    invoke = lambda _value: renderer.set_parameter("baseColor", RetiringColor([1, 0, 0, 1]))
                else:
                    invoke = owner._commit_document
            document["native_retirement_probe"] = RetiringJsonList([0])
            with pytest.raises(InvalidNativeObjectError):
                invoke(document)
            assert not owner and not collider
            (project / "retirement-evidence.json").write_text(
                json.dumps(dict(evidence, status="passed"), indent=2), encoding="utf-8")
            return

        if access.startswith("reentrant_"):
            from infernux.lib import _native_module as native

            class RetiringNumber:
                def __index__(self):
                    retire()
                    return 1

                def __bool__(self):
                    retire()
                    return True

            class RetiringList(list):
                def __iter__(self):
                    yield owner
                    retire()

            def retiring_world():
                retire()
                yield survivor

            with pytest.raises(InvalidNativeObjectError):
                if access == "reentrant_scalar":
                    owner.layer = RetiringNumber()
                elif access == "reentrant_component":
                    collider.enabled = RetiringNumber()
                elif access == "reentrant_vector_argument":
                    native._UITransformDependencies([owner], retiring_world())
                elif access == "reentrant_builtin_argument":
                    Physics.ignore_collision(owner.get_component("BoxCollider"), other_collider, RetiringNumber())
                else:
                    native._UITransformDependencies(RetiringList([owner]), [])
            assert not owner and not collider
            (project / "retirement-evidence.json").write_text(
                json.dumps(dict(evidence, status="passed"), indent=2), encoding="utf-8")
            return

        retire()
        if retirement != "unload" and handle is not None:
            resolver = source.resolve_game_object if kind in ("object", "hit") else source.resolve_component
            assert resolver(handle) is None
        checkpoint("native-retired")
        if kind == "hit":
            assert hit.distance == distance
            assert (hit.point.x, hit.point.y, hit.point.z) == point
            for field in ("game_object", "collider"):
                try:
                    value = getattr(hit, field)
                except InvalidNativeObjectError:
                    continue
                assert value is None, f"retained hit revived {field}"
        elif access == "truth":
            assert bool(target) is False, "retired native object still reports alive"
        else:
            try:
                if access == "property":
                    getattr(target, attribute)
                elif access == "saved_method":
                    method()
                elif kind == "object":
                    survivor.set_parent(target)
                else:
                    Physics.ignore_collision(other_collider, target)
            except InvalidNativeObjectError:
                pass
            else:
                raise AssertionError("retired native object was accepted")
        (project / "retirement-evidence.json").write_text(
            json.dumps(dict(evidence, status="passed"), indent=2), encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), *sys.argv[2:5])
