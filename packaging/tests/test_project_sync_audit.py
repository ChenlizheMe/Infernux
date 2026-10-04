from __future__ import annotations

import importlib.util
import copy
import json
import sys
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "maintenance" / "audit_project_sync.py"


def _module():
    spec = importlib.util.spec_from_file_location("infernux_project_sync_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "Project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    (project / ".gitignore").write_text(
        (ROOT / "python" / "Infernux" / "resources" / "project_templates" / "project.gitignore.txt")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / ".gitattributes").write_text(
        (ROOT / "python" / "Infernux" / "resources" / "project_templates" / "project.gitattributes.txt")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return project


def _write_asset(project: Path, relative: str, guid: str, *, absolute_hint: str = "") -> None:
    source = project / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("{}\n", encoding="utf-8")
    payload = {
        "metadata": {
            "guid": {"type": "string", "value": guid},
            "content_hash": {"type": "string", "value": "0000000000000000"},
        }
    }
    if absolute_hint:
        payload["metadata"]["file_path"] = {"type": "string", "value": absolute_hint}
    (Path(str(source) + ".meta")).write_text(
        json.dumps(payload, indent=4) + "\n", encoding="utf-8"
    )


def test_fixture_project_has_a_portable_sync_contract():
    report = _module().audit_project(ROOT / "tests" / "fixtures" / "multiplatform_player")
    assert report.ok, report.errors
    assert report.package_count == 6


def test_absolute_metadata_path_is_rejected(tmp_path: Path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/Scenes/Main.scene", "0123456789abcdef0123456789abcdef", absolute_hint="C:/work/Main.scene")
    report = _module().audit_project(project)
    assert not report.ok
    assert any("metadata.file_path must be project-relative" in error for error in report.errors)


def test_duplicate_guid_and_missing_sidecar_are_rejected(tmp_path: Path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/A.txt", "0123456789abcdef0123456789abcdef")
    _write_asset(project, "Assets/B.txt", "0123456789abcdef0123456789abcdef")
    (project / "Assets" / "WithoutMeta.txt").write_text("oops", encoding="utf-8")
    report = _module().audit_project(project)
    assert not report.ok
    assert any("duplicate GUID" in error for error in report.errors)
    assert any("missing tracked .meta sidecar" in error for error in report.errors)


def test_orphan_sidecar_and_unresolved_lfs_payload_are_rejected(tmp_path: Path):
    project = _project(tmp_path)
    _write_asset(project, "Assets/Ship.fbx", "0123456789abcdef0123456789abcdef")
    (project / "Assets/Ship.fbx").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:placeholder\nsize 123\n", encoding="utf-8")
    (project / "Assets/Deleted.txt.meta").write_text("{}", encoding="utf-8")
    report = _module().audit_project(project)
    assert any("orphan .meta" in error for error in report.errors)
    assert any("Git LFS payload" in error for error in report.errors)


def _driver():
    name = "infernux_semantic_merge_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "python/Infernux/collaboration.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _object(identity=1, *, name="Root", prefab=False, components=None, children=None):
    return {
        "local_id" if prefab else "id": identity,
        "name": name, "active": True, "is_static": False, "layer": 0, "tag": "Untagged",
        "transform": {**({} if prefab else {"component_id": identity}),
                      "type": "Transform", "enabled": True, "execution_order": 0,
                      "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]},
        "components": components or [], "children": children or [],
    }


def _component(identity, data=None, kind="python:test:test:Scripts.Test:Test"):
    return {"component_id": identity, "type_id": kind, "data": data or {},
            "enabled": True, "execution_order": 0}


def _scene(*nodes):
    return {"name": "Main", "isPlaying": False, "objects": list(nodes)}


def _branches(document):
    return copy.deepcopy(document), copy.deepcopy(document)


def test_scene_parallel_fields_merge_even_on_the_same_json_line(tmp_path: Path):
    project = _project(tmp_path)

    def git(*arguments, expected=0):
        result = subprocess.run(
            ["git", "-c", "user.name=Sync Test", "-c", "user.email=sync-test@example.invalid",
             "-c", "commit.gpgsign=false", "-C", str(project), *arguments],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout

    git("init", "-b", "main")
    _driver().configure_git_driver(project)
    scene = project / "Assets/Main.scene"
    document = _scene(_object())
    scene.write_text(json.dumps(document), encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "initial")
    git("checkout", "-b", "other")
    document["objects"][0]["name"] = "Renamed"
    scene.write_text(json.dumps(document), encoding="utf-8")
    git("commit", "-am", "other change")
    git("checkout", "main")
    document["objects"][0]["name"] = "Root"
    document["objects"][0]["transform"]["position"][0] = 12
    scene.write_text(json.dumps(document), encoding="utf-8")
    git("commit", "-am", "main change")
    git("merge", "other")
    assert not git("ls-files", "--unmerged")
    merged = json.loads(scene.read_text(encoding="utf-8"))["objects"][0]
    assert merged["name"] == "Renamed"
    assert merged["transform"]["position"] == [12, 0, 0]


def test_scene_combines_vector_coordinates_and_different_components():
    base = _scene(_object(components=[_component(2, {"value": 1}), _component(3, {"value": 1})]))
    ours, theirs = _branches(base)
    ours["objects"][0]["transform"]["position"][0] = 7
    theirs["objects"][0]["transform"]["position"][1] = 8
    ours["objects"][0]["components"][0]["data"]["value"] = 10
    theirs["objects"][0]["components"][1]["data"]["value"] = 20
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert result.clean, result.conflicts
    node = result.document["objects"][0]
    assert node["transform"]["position"] == [7, 8, 0]
    assert [component["data"]["value"] for component in node["components"]] == [10, 20]


@pytest.mark.parametrize("prefab", [False, True])
def test_colliding_additions_preserve_both_objects_and_typed_references(prefab):
    root = _object(prefab=prefab)
    base = {"root_object": root, "next_local_id": 2, "next_component_id": 1} if prefab else _scene(root)
    ours, theirs = _branches(base)
    for branch, name in ((ours, "Sword"), (theirs, "Bow")):
        data = {"self": {"$type": "game_object_ref", "object_id": 2},
                "target": {"$type": "component_ref", "game_object_id": 2, "component_id": 3, "component_type": "Test"},
                "opaque": {"component_id": 3, "connected_body_component_id": 3}}
        added = _object(2, name=name, prefab=prefab, components=[_component(3, data)])
        (branch["root_object"]["children"] if prefab else branch["objects"]).append(added)
        if prefab:
            branch.update(next_local_id=3, next_component_id=4)
    result = _driver().merge_documents(base, ours, theirs, "Main.prefab" if prefab else "Main.scene")
    assert result.clean, result.conflicts
    nodes = result.document["root_object"]["children"] if prefab else result.document["objects"][1:]
    sword, bow = nodes
    identity = "local_id" if prefab else "id"
    assert sword[identity] == 2 and bow[identity] == 3
    new_component_id = 4 if prefab else 5
    assert sword["components"][0]["component_id"] == 3 and bow["components"][0]["component_id"] == new_component_id
    data = bow["components"][0]["data"]
    assert data["self"]["object_id"] == 3
    assert (data["target"]["game_object_id"], data["target"]["component_id"]) == (3, new_component_id)
    assert data["opaque"] == {"component_id": 3, "connected_body_component_id": 3}
    if prefab:
        assert result.document["next_local_id"] == 4
        assert result.document["next_component_id"] == 5
    else:
        assert sword["transform"]["component_id"] != bow["transform"]["component_id"]


def test_identical_concurrent_additions_are_not_duplicated():
    base = _scene(_object())
    ours = copy.deepcopy(base)
    ours["objects"].append(_object(2))
    result = _driver().merge_documents(base, ours, copy.deepcopy(ours), "Main.scene")
    assert result.clean and result.document == ours


def test_new_components_on_existing_objects_get_separate_ids():
    base = _scene(_object())
    ours, theirs = _branches(base)
    ours["objects"][0]["components"].append(_component(2, {"value": "left"}))
    theirs["objects"][0]["components"].append(_component(2, {"value": "right"}))
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert result.clean, result.conflicts
    assert [(component["component_id"], component["data"]["value"])
            for component in result.document["objects"][0]["components"]] == [(2, "left"), (3, "right")]


def test_reparent_and_property_edit_merge_by_identity():
    base = _scene(_object(1, children=[_object(2)]), _object(3))
    ours, theirs = _branches(base)
    ours["objects"][1]["children"].append(ours["objects"][0]["children"].pop())
    theirs["objects"][0]["children"][0]["name"] = "Moved and renamed"
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert result.clean, result.conflicts
    assert not result.document["objects"][0]["children"]
    assert result.document["objects"][1]["children"][0]["name"] == "Moved and renamed"


def test_reparenting_that_jointly_creates_a_cycle_is_rejected():
    base = _scene(_object(1), _object(2))
    ours, theirs = _branches(base)
    first = ours["objects"].pop(0)
    ours["objects"][0]["children"].append(first)
    second = theirs["objects"].pop()
    theirs["objects"][0]["children"].append(second)
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert not result.clean
    assert "cycle" in result.conflicts[0].reason


def test_reorder_and_property_edit_preserve_both_intents():
    base = _scene(_object(1), _object(2), _object(3))
    ours, theirs = _branches(base)
    ours["objects"] = [ours["objects"][2], *ours["objects"][:2]]
    theirs["objects"][1]["name"] = "Changed"
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert result.clean, result.conflicts
    assert [node["id"] for node in result.document["objects"]] == [3, 1, 2]
    assert result.document["objects"][2]["name"] == "Changed"


@pytest.mark.parametrize("action", ["same_field", "delete_edit", "delete_new_reference", "duplicate_id", "invalid_camera"])
def test_real_conflicts_are_explicit_and_keep_valid_json(action):
    base = _scene(_object(1), _object(2))
    ours, theirs = _branches(base)
    if action == "same_field":
        ours["objects"][0]["name"] = "Left"
        theirs["objects"][0]["name"] = "Right"
    elif action == "delete_edit":
        ours["objects"].pop()
        theirs["objects"][1]["name"] = "Right"
    elif action == "delete_new_reference":
        ours["objects"].pop()
        theirs["objects"][0]["components"].append(_component(3, {"ref": {"$type": "game_object_ref", "object_id": 2}}))
    elif action == "duplicate_id":
        ours["objects"][1]["id"] = 1
    else:
        ours["mainCameraComponentId"] = 1
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert not result.clean
    assert all(conflict.path.startswith("/") for conflict in result.conflicts)
    assert json.loads(json.dumps(result.document)) == result.document


def test_physics_joint_links_are_remapped_only_in_native_joint_data():
    base = _scene(_object())
    ours, theirs = _branches(base)
    for branch in (ours, theirs):
        branch["objects"].append(_object(2, components=[
            _component(3, kind="native:infernux.Rigidbody"),
            _component(4, {"connected_body_component_id": 3}, kind="native:infernux.HingeJoint"),
        ]))
    theirs["objects"][1]["name"] = "Other ship"
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert result.clean, result.conflicts
    body, joint = result.document["objects"][2]["components"]
    assert joint["data"]["connected_body_component_id"] == body["component_id"]


def test_new_main_camera_uses_the_remapped_component_id():
    base = _scene(_object())
    ours, theirs = _branches(base)
    ours["objects"].append(_object(2, name="Ship", components=[_component(3)]))
    theirs["objects"].append(_object(2, name="Camera", components=[_component(3, kind="native:infernux.Camera")]))
    theirs["mainCameraComponentId"] = 3
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert result.clean, result.conflicts
    assert result.document["mainCameraComponentId"] == result.document["objects"][2]["components"][0]["component_id"]


def test_material_properties_merge_without_a_line_conflict():
    base = {"shaders": {"vertex": "vertex-guid", "fragment": "fragment-guid"},
            "properties": {"roughness": {"type": "float", "value": 0.5}, "tint": {"type": "vec3", "value": [1, 1, 1]}}}
    ours, theirs = _branches(base)
    ours["properties"]["roughness"]["value"] = 0.8
    theirs["properties"]["tint"]["value"] = [0.2, 1, 1]
    result = _driver().merge_documents(base, ours, theirs, "Ship.mat")
    assert result.clean, result.conflicts
    assert result.document["properties"]["roughness"]["value"] == 0.8
    assert result.document["properties"]["tint"]["value"] == [0.2, 1, 1]
    ours["shaders"]["fragment"] = "new-fragment-guid"
    assert not _driver().merge_documents(base, ours, theirs, "Ship.mat").clean


def test_metadata_import_settings_merge_without_derived_fingerprint_conflicts():
    base = {"metadata": {"guid": {"type": "string", "value": "a" * 32},
                         "content_hash": {"type": "string", "value": "old"},
                         "last_modified": {"type": "string", "value": "old"},
                         "flip": {"type": "bool", "value": False}, "scale": {"type": "float", "value": 1}}}
    ours, theirs = _branches(base)
    ours["metadata"]["flip"]["value"] = True
    theirs["metadata"]["scale"]["value"] = 2
    ours["metadata"]["content_hash"]["value"] = "ours"
    theirs["metadata"]["content_hash"]["value"] = "theirs"
    result = _driver().merge_documents(base, ours, theirs, "Ship.fbx.meta")
    assert result.clean, result.conflicts
    assert "content_hash" not in result.document["metadata"] and "last_modified" not in result.document["metadata"]
    assert result.document["metadata"]["flip"]["value"] is True
    assert result.document["metadata"]["scale"]["value"] == 2
    theirs["metadata"]["guid"]["value"] = "b" * 32
    assert not _driver().merge_documents(base, ours, theirs, "Ship.fbx.meta").clean


def test_graph_nodes_merge_by_uid_and_reject_dangling_links():
    base = {"$schema": "infernux.graph_document", "nodes": [
        {"uid": "a", "type_id": "a", "pos_x": 0, "pos_y": 0, "data": {}},
        {"uid": "b", "type_id": "b", "pos_x": 1, "pos_y": 1, "data": {}},
    ], "links": []}
    ours, theirs = _branches(base)
    ours["nodes"][0]["pos_x"] = 20
    theirs["nodes"][1]["data"]["value"] = 30
    result = _driver().merge_documents(base, ours, theirs, "Bloom.effect")
    assert result.clean, result.conflicts
    assert result.document["nodes"][0]["pos_x"] == 20
    assert result.document["nodes"][1]["data"]["value"] == 30
    ours["nodes"].pop()
    theirs["links"].append({"uid": "link", "source_node": "a", "source_pin": "out", "target_node": "b", "target_pin": "in", "data": {}})
    assert not _driver().merge_documents(base, ours, theirs, "Bloom.effect").clean


def test_package_registry_merges_parallel_additions_by_reference():
    base = {"$schema": "infernux.plugin_registry", "packages": [], "installed": []}
    ours, theirs = _branches(base)
    ours["packages"].append({"reference": "one", "version": "1.0"})
    theirs["packages"].append({"reference": "two", "version": "2.0"})
    result = _driver().merge_documents(base, ours, theirs, "ProjectSettings/InxPlugins.json")
    assert result.clean, result.conflicts
    assert {record["reference"] for record in result.document["packages"]} == {"one", "two"}


@pytest.mark.parametrize("payload", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_invalid_json_is_rejected_before_rewriting_a_document(tmp_path, payload):
    path = tmp_path / "document.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        _driver().read_document(path)


def test_driver_install_migrates_engine_rules_and_preserves_custom_attributes(tmp_path):
    project = _project(tmp_path)
    subprocess.run(["git", "init", str(project)], check=True, capture_output=True)
    path = project / ".gitattributes"
    path.write_text("*.scene text eol=lf merge=binary\n*.png filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    assert _driver().configure_git_driver(project)
    content = path.read_text(encoding="utf-8")
    assert "*.scene text eol=lf merge=infernux" in content
    assert "*.png filter=lfs diff=lfs merge=lfs -text" in content
    before = path.stat().st_mtime_ns
    _driver().configure_git_driver(project)
    assert path.stat().st_mtime_ns == before


def test_merge_cli_preserves_json_and_reports_property_conflicts(tmp_path):
    base = _scene(_object())
    ours, theirs = _branches(base)
    ours["objects"][0]["name"] = "Left"
    theirs["objects"][0]["name"] = "Right"
    theirs["objects"][0]["layer"] = 7
    paths = [tmp_path / name for name in ("base", "ours", "theirs")]
    for path, document in zip(paths, (base, ours, theirs)):
        path.write_text(json.dumps(document), encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "python/Infernux/collaboration.py"),
                             *map(str, paths), "Assets/飞船 demo.scene"], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 1
    assert "/objects/1/name" in result.stderr
    node = json.loads(paths[1].read_text(encoding="utf-8"))["objects"][0]
    assert node["name"] == "Left" and node["layer"] == 7


@pytest.mark.parametrize("suffix", [".mat", ".physicmaterial", ".rendertexture", ".inxdata", ".effect", ".json"])
def test_cloned_projects_merge_parallel_asset_fields_with_spaces_and_unicode(tmp_path, suffix):
    project = _project(tmp_path)
    asset = project / ("ProjectSettings" if suffix == ".json" else "Assets") / ("飞船 material" + suffix)

    def git(root, *arguments, expected=0):
        result = subprocess.run(["git", "-c", "user.name=Sync Test", "-c", "user.email=sync@example.invalid",
                                 "-c", "commit.gpgsign=false", "-C", str(root), *arguments],
                                capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == expected, result.stdout + result.stderr
        return result.stdout

    git(project, "init", "-b", "main")
    original = {"properties": {"roughness": 0.5, "metallic": 0.5}}
    asset.write_text(json.dumps(original), encoding="utf-8")
    git(project, "add", ".")
    git(project, "commit", "-m", "Initial")
    clone = tmp_path / "克隆 with spaces"
    git(project, "clone", str(project), str(clone))
    assert not git(clone, "config", "--get", "merge.infernux.driver", expected=1)
    _driver().configure_git_driver(clone)
    clone_asset = clone / asset.relative_to(project)
    git(clone, "checkout", "-b", "other")
    clone_asset.write_text(json.dumps({"properties": {"roughness": 0.8, "metallic": 0.5}}), encoding="utf-8")
    git(clone, "commit", "-am", "Roughness")
    git(clone, "checkout", "main")
    clone_asset.write_text(json.dumps({"properties": {"roughness": 0.5, "metallic": 0.9}}), encoding="utf-8")
    git(clone, "commit", "-am", "Metallic")
    git(clone, "merge", "other")
    assert not git(clone, "ls-files", "--unmerged")
    assert json.loads(clone_asset.read_text(encoding="utf-8"))["properties"] == {"roughness": 0.8, "metallic": 0.9}


def test_effect_group_parallel_insertions_use_entry_identity():
    base = {"$schema": "infernux.render_effect_group", "entries": [
        {"entry_id": "bloom", "asset": {"guid": "bloom-guid"}, "enabled": True, "overrides": {}},
    ]}
    ours, theirs = _branches(base)
    ours["entries"].append({"entry_id": "tone", "asset": {"guid": "tone-guid"}, "enabled": True, "overrides": {}})
    theirs["entries"].insert(0, {"entry_id": "fog", "asset": {"guid": "fog-guid"}, "enabled": True, "overrides": {}})
    result = _driver().merge_documents(base, ours, theirs, "Post.effectgroup")
    assert result.clean, result.conflicts
    assert [entry["entry_id"] for entry in result.document["entries"]] == ["fog", "bloom", "tone"]


def test_shader_discriminator_change_conflicts_with_other_effect_parameter_edits():
    base = {"$schema": "infernux.render_effect", "feature_type": "Bloom", "parameters": {"strength": 1}, "dependencies": []}
    ours, theirs = _branches(base)
    ours["feature_type"] = "Fog"
    theirs["parameters"]["strength"] = 2
    assert not _driver().merge_documents(base, ours, theirs, "Feature.effect").clean


def test_subprojects_share_the_containing_repository_driver(tmp_path):
    repo = tmp_path / "Repository"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    project = _project(repo)
    assert _driver().configure_git_driver(project)
    result = subprocess.run(["git", "-C", str(repo), "config", "--get", "merge.infernux.driver"], capture_output=True)
    assert result.returncode == 0 and result.stdout


def test_allocation_watermarks_merge_by_maximum_and_reserve_deleted_id_ranges():
    base = {**_scene(_object()), "nextObjectId": 100, "nextComponentId": 200}
    ours, theirs = _branches(base)
    ours["objects"].append(_object(100, name="Sword", components=[_component(200)]))
    theirs["objects"].append(_object(100, name="Bow", components=[_component(200)]))
    # Both editors deleted further additions before saving. Their IDs stay reserved.
    ours.update(nextObjectId=120, nextComponentId=220)
    theirs.update(nextObjectId=130, nextComponentId=230)
    result = _driver().merge_documents(base, ours, theirs, "Main.scene")
    assert result.clean, result.conflicts
    added = result.document["objects"][2]
    assert added["id"] == 130
    assert added["transform"]["component_id"] == 230
    assert added["components"][0]["component_id"] == 231
    assert result.document["nextObjectId"] == 131
    assert result.document["nextComponentId"] == 232


def test_parallel_packages_can_share_python_dependency_ownership():
    base = {"$schema": "infernux.plugin_registry", "packages": [], "installed": [], "python_dependencies": [
        {"name": "numpy", "managed": True, "owners": [{"reference": "base", "requirements": ["numpy>=2"]}]},
    ]}
    ours, theirs = _branches(base)
    ours["python_dependencies"][0]["owners"].append({"reference": "ocean", "requirements": ["numpy>=2"]})
    theirs["python_dependencies"][0]["owners"].append({"reference": "ships", "requirements": ["numpy>=2"]})
    result = _driver().merge_documents(base, ours, theirs, "ProjectSettings/InxPlugins.json")
    assert result.clean, result.conflicts
    assert {owner["reference"] for owner in result.document["python_dependencies"][0]["owners"]} == {"base", "ocean", "ships"}


def test_conflicting_dependency_pins_are_not_silently_combined():
    base = {"$schema": "infernux.plugin_registry", "python_dependencies": [
        {"name": "numpy", "managed": True, "owners": [{"reference": "ocean", "requirements": ["numpy==1"]}]},
    ]}
    ours, theirs = _branches(base)
    ours["python_dependencies"][0]["owners"][0]["requirements"] = ["numpy==2"]
    theirs["python_dependencies"][0]["owners"][0]["requirements"] = ["numpy==3"]
    result = _driver().merge_documents(base, ours, theirs, "ProjectSettings/InxPlugins.json")
    assert not result.clean
    assert result.conflicts[0].path.endswith("/requirements/0")
