"""Missing object references remain authored identities across a cold scene load."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("deleted", [False, True])
def test_scene_roundtrip_preserves_missing_object_reference(tmp_path, deleted):
    import infernux

    package_root = Path(infernux.__file__).resolve().parent.parent
    for phase in ("author", "reopen"):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(tmp_path), phase, str(int(deleted))],
            env=dict(os.environ, PYTHONPATH=str(package_root)), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "REFERENCE_ROUNDTRIP_OK" in result.stdout


@pytest.mark.parametrize("invalid_id", [-1, True, "12", None])
def test_scene_reference_preflight_still_rejects_malformed_identity(invalid_id):
    from infernux.engine.component_restore import _validate_reference_documents, PythonComponentRestoreError

    with pytest.raises(PythonComponentRestoreError, match="non-negative integer"):
        _validate_reference_documents(
            {"$type": "game_object_ref", "object_id": invalid_id}, "target", None, set(),
        )


def exercise(project, phase, deleted):
    from infernux.components.script_loader import load_all_components_from_file
    from infernux.components.component_identity import bind_asset_script_guid
    from infernux.components.registry import publish_component_script_types
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.runtime_scene_transaction import SceneDocumentTransaction
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    assets = project / "Assets"
    assets.mkdir(exist_ok=True)
    (project / "ProjectSettings").mkdir(exist_ok=True)
    script = assets / "ReferenceProbe.py"
    if phase == "author":
        script.write_text(
            "from infernux.components import InxComponent, serialized_field\n"
            "from infernux.components.ref_wrappers import GameObjectRef\n"
            "class ReferenceProbe(InxComponent):\n"
            "    target: GameObjectRef = serialized_field(default=None)\n",
            encoding="utf-8",
        )
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        classes = load_all_components_from_file(str(script), register=False)
        assert len(classes) == 1
        guid = database.get_guid_from_path(str(script))
        assert guid
        bind_asset_script_guid(classes[0], guid)
        publish_component_script_types(str(script), classes)
        manager = SceneManager.instance()
        scene = manager.create_scene("ReferenceRoundtrip")
        manager.set_active_scene(scene)
        path = assets / "Reference.scene"
        if phase == "author":
            owner = scene.create_game_object("Owner")
            target = scene.create_game_object("Target")
            instance = owner.add_component(classes[0])
            instance.target = target
            target_id = int(target.id)
            if deleted:
                scene.destroy_game_object(target)
                scene.process_pending_destroys()
            path.write_text(json.dumps(scene.serialize_document()), encoding="utf-8")
            (project / "expected.json").write_text(json.dumps({
                "owner": int(owner.id), "component": int(instance.component_id), "target": target_id,
            }), encoding="utf-8")
        else:
            expected = json.loads((project / "expected.json").read_text(encoding="utf-8"))
            transaction = SceneDocumentTransaction(scene, path=str(path), asset_database=database)
            assert transaction.run_to_completion()
            owner = scene.find_by_id(expected["owner"])
            matches = [component for component in owner.get_py_components()
                       if int(component.component_id) == expected["component"]]
            assert len(matches) == 1
            instance = matches[0]
            assert not getattr(instance, "_is_broken", False)
            reference = type(instance).target.get_raw(instance)
            assert reference.persistent_id == expected["target"]
            assert (reference.resolve() is None) is deleted
            if not deleted:
                assert int(reference.resolve().id) == expected["target"]
            document = scene.serialize_document()
            saved_owner = next(item for item in document["objects"] if item["id"] == expected["owner"])
            saved_component = next(item for item in saved_owner["components"]
                                   if item["type_id"].startswith("python:"))
            assert saved_component["data"]["target"]["object_id"] == expected["target"]
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], bool(int(sys.argv[3])))
    print("REFERENCE_ROUNDTRIP_OK")
