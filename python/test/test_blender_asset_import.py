"""The real asset database owns Blender conversion and binary publication."""
import os
import json
from pathlib import Path
import subprocess
import time

import pytest
import numpy as np

from Infernux.core.assets import AssetManager
from Infernux.core.animation_clip3d import AnimationClip3D, embedded_take_descriptors
from Infernux.core.asset_types import read_mesh_import_settings, read_meta_file
from Infernux.engine.model_import.toolchain import export_script
from Infernux.lib import AssetRegistry, Vector3


def _descendants(root):
    result = {}

    def visit(obj):
        for child in obj.get_children():
            result[child.name] = child
            visit(child)

    visit(root)
    return result


def test_blender_tool_requires_explicit_absolute_configuration(engine, tmp_path):
    database = engine.get_asset_database()
    with pytest.raises(ValueError):
        database.configure_blender_import("blender", export_script())
    with pytest.raises(ValueError):
        database.configure_blender_import("", export_script())
    database.configure_blender_import("", "")
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "MissingTool.blend"
    source.write_bytes(b"BLENDER-v410")
    try:
        result = database.import_asset(str(source))
        assert not result and "not configured" in result.error
        assert not (Path(database.project_root) / "Library/ModelImport").exists()
    finally:
        database.delete_asset(str(source))
        source.unlink(missing_ok=True)
        Path(str(source) + ".meta").unlink(missing_ok=True)


@pytest.mark.skipif(not os.environ.get("INFERNUX_TEST_BLENDER"), reason="requires the Blender 5.2 authoring tool")
def test_modern_blend_worker_import_reimport_and_failed_publication(engine, tmp_path, monkeypatch):
    tool = os.environ["INFERNUX_TEST_BLENDER"]
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "嵌套 模型.BLEND"
    # This source is authored by Blender itself, not a renamed glTF fixture.
    script = (
        "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); "
        "root=bpy.data.objects.new('Assembly',None); bpy.context.collection.objects.link(root); "
        "root.location=(2,1,0); "
        "red=bpy.data.materials.new('Red'); red.diffuse_color=(0.8,0.1,0.1,1); "
        "red.use_nodes=True; image=bpy.data.images.new('Embedded Color',width=2,height=2); "
        "image.pixels=[1,0.05,0.05,1, 0.8,0.1,0.1,1, 0.6,0.15,0.15,1, 0.4,0.2,0.2,1]; "
        "nodes=red.node_tree.nodes; links=red.node_tree.links; bsdf=nodes.get('Principled BSDF'); "
        "tex=nodes.new('ShaderNodeTexImage'); tex.image=image; links.new(tex.outputs['Color'],bsdf.inputs['Base Color']); "
        "blue=bpy.data.materials.new('Blue'); blue.diffuse_color=(0.1,0.2,0.8,1); "
        "bpy.ops.mesh.primitive_cube_add(); cube=bpy.context.object; "
        "cube.name='ChildCube'; cube.parent=root; cube.location=(0,0,2); cube.data.materials.append(red); "
        "bpy.ops.mesh.primitive_cube_add(); second=bpy.context.object; "
        "second.name='SecondCube'; second.parent=root; second.location=(2,0,2); second.data.materials.append(blue); "
        "second.hide_render=True; "
        "second.keyframe_insert(data_path='location',frame=1); second.location=(2,1,2); "
        "second.keyframe_insert(data_path='location',frame=24); "
        "bpy.context.scene.frame_end=24; "
        "bpy.ops.wm.save_as_mainfile(filepath=" + repr(str(source)) + ")"
    )
    subprocess.run([tool, "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                    "--python-expr", script], check=True, capture_output=True, timeout=60,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    before = source.read_bytes()
    invalid_source = folder / "ConcurrentInvalid.blend"
    invalid_source.write_bytes(b"not a Blender file")
    database.configure_blender_import(tool, export_script())
    guid = None
    try:
        database.begin_refresh()
        ticks = 0
        deadline = time.monotonic() + 150
        while not database.try_commit_refresh():
            ticks += 1
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert ticks > 1  # Converted on the native worker, not inside BeginRefresh.
        guid = database.get_guid_from_path(str(source))
        assert guid
        mesh = registry.load_mesh(str(source))
        assert mesh and mesh.vertex_count > 0 and mesh.submesh_count >= 2
        assert mesh.name == source.stem
        manifest = json.loads(database.get_meta_by_guid(guid).get_string("model_meshes"))
        assert {entry["name"] for entry in manifest} >= {"ChildCube", "SecondCube"}
        assert all(entry["subresource_id"] for entry in manifest)
        textures = json.loads(database.get_meta_by_guid(guid).get_string("model_textures"))
        assert textures and any(record["name"] == "Embedded Color" for record in textures)
        nodes = mesh.get_model_nodes()
        assert {node["name"] for node in nodes} >= {"Assembly", "ChildCube", "SecondCube"}
        visibility = {node["name"]: node["visible"] for node in nodes}
        assert visibility["ChildCube"] is True
        assert visibility["SecondCube"] is False
        assert database.get_meta_by_guid(guid).get_int("animation_count") > 0
        assert source.read_bytes() == before
        assert database.last_refresh_worker_importer_count > 0
        staging = Path(database.project_root) / "Library/ModelImport"
        assert staging.is_dir() and not list(staging.iterdir())
        settings = read_mesh_import_settings(str(source))
        settings.scale_factor = 2
        started = time.monotonic()
        database.begin_model_reimport(str(source), settings.to_dict())
        assert time.monotonic() - started < 1.0
        assert read_mesh_import_settings(str(source)).scale_factor == 1
        apply_ticks = 0
        deadline = time.monotonic() + 150
        while (result := AssetManager.poll_model_reimport(database)) is None:
            apply_ticks += 1
            assert database.get_guid_from_path(str(source)) == guid
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert apply_ticks > 1  # Blender was running while owner queries remained available.
        assert result, result.error
        assert result.guid == guid
        assert read_mesh_import_settings(str(source)).scale_factor == 2
        sidecar = Path(str(source) + ".meta").read_bytes()
        vertices = mesh.vertex_count
        source.write_bytes(b"invalid Blender source")
        database.begin_model_reimport(str(source), settings.to_dict())
        deadline = time.monotonic() + 150
        while (result := AssetManager.poll_model_reimport(database)) is None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert not result and "Blender import failed" in result.error
        assert Path(str(source) + ".meta").read_bytes() == sidecar
        assert mesh.vertex_count == vertices
        assert not list(staging.iterdir())
        # Binary reload must remain usable without the converter or source.
        database.configure_blender_import("", "")
        registry.invalidate_asset(guid)
        assert registry.load_mesh(str(source)).vertex_count == vertices
    finally:
        if guid:
            registry.invalidate_asset(guid)
        database.configure_blender_import("", "")
        database.delete_asset(str(source))
        database.delete_asset(str(invalid_source))
        for authored in (source, invalid_source):
            authored.unlink(missing_ok=True)
            Path(str(authored) + ".meta").unlink(missing_ok=True)


@pytest.mark.skipif(not os.environ.get("INFERNUX_TEST_BLENDER"), reason="requires the Blender 5.2 authoring tool")
def test_real_blend_reauthor_preserves_mesh_identity_across_rename_and_reparent(
    engine, tmp_path
):
    tool = os.environ["INFERNUX_TEST_BLENDER"]
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "EditorSync.blend"
    fixture_script = (
        Path(__file__).with_name("fixtures")
        / "create_blender_editor_sync_matrix.py"
    )

    def author(revision: str) -> None:
        subprocess.run(
            [
                tool,
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str(fixture_script),
                "--",
                str(source),
                revision,
            ],
            check=True,
            capture_output=True,
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    database.configure_blender_import(tool, export_script())
    guid = ""
    try:
        author("initial")
        imported = database.import_asset(str(source))
        assert imported, imported.error
        guid = imported.guid
        before = json.loads(
            database.get_meta_by_guid(guid).get_string("model_meshes")
        )
        stable_before = next(item for item in before if item["name"] == "RenameMe")
        assert stable_before["path"] == [
            "AuthoringRoot",
            "AuthoringPivot",
            "RenameMe",
        ]

        author("changed")
        result = AssetManager.reimport_asset(str(source), database=database)
        assert result, result.error
        assert result.guid == guid
        after = json.loads(
            database.get_meta_by_guid(guid).get_string("model_meshes")
        )
        stable_after = next(
            item for item in after if item["name"] == "RenamedStable"
        )
        assert stable_after["path"] == ["AuthoringRoot", "RenamedStable"]
        assert stable_after["subresource_id"] == stable_before["subresource_id"]
        assert {item["name"] for item in after} == {
            "RenamedStable",
            "AlwaysHere",
            "AddedFromBlender",
        }
    finally:
        if guid:
            registry.invalidate_asset(guid)
        database.configure_blender_import("", "")
        database.delete_asset(str(source))
        source.unlink(missing_ok=True)
        Path(str(source) + ".meta").unlink(missing_ok=True)


@pytest.mark.skipif(not os.environ.get("INFERNUX_TEST_BLENDER"), reason="requires the Blender 5.2 authoring tool")
def test_real_blend_scene_instance_sync_preserves_authoring_and_roundtrips(
    engine, scene, tmp_path, monkeypatch
):
    """A real .blend revision updates source structure without replacing authored state."""
    tool = os.environ["INFERNUX_TEST_BLENDER"]
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "EditorSceneSync.blend"
    fixture_script = Path(__file__).with_name("fixtures") / "create_blender_editor_sync_matrix.py"

    def author(revision: str) -> None:
        subprocess.run(
            [
                tool,
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str(fixture_script),
                "--",
                str(source),
                revision,
            ],
            check=True,
            capture_output=True,
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    database.configure_blender_import(tool, export_script())
    guid = ""
    try:
        author("initial")
        imported = AssetManager.import_asset(str(source), database=database)
        assert imported, imported.error
        guid = imported.guid
        root = scene.create_from_model(guid, "Authored Blender Instance")
        initial = _descendants(root)
        stable = initial["RenameMe"]
        stable_id = stable.id
        renderer = stable.get_component("MeshRenderer")._require_cpp_component()
        stable_subresource_id = renderer.model_subresource_id
        stable.transform.local_position = Vector3(17.0, 18.0, 19.0)
        stable.add_component("BoxCollider")
        renderer.set_material(0, renderer.get_material(0))
        authored_materials = renderer.serialize_document()["materials"]
        assert authored_materials != [None]

        author("changed")
        result = AssetManager.reimport_asset(str(source), database=database)
        assert result, result.error
        assert result.guid == guid

        changed = _descendants(root)
        assert "RenameMe" not in changed
        assert "DeleteMe" not in changed
        assert "AddedFromBlender" in changed
        assert "FutureCamera" not in changed and "FutureLight" not in changed
        renamed = changed["RenamedStable"]
        assert renamed.id == stable_id
        assert tuple(renamed.transform.local_position) == (17.0, 18.0, 19.0)
        assert renamed.get_parent().name == "AuthoringRoot"
        assert renamed.get_component("BoxCollider") is not None
        renamed_renderer = renamed.get_component("MeshRenderer")._require_cpp_component()
        assert renamed_renderer.model_subresource_id == stable_subresource_id
        assert renamed_renderer.model_node_path == ["AuthoringRoot", "RenamedStable"]
        assert renamed_renderer.serialize_document()["materials"] == authored_materials

        # This is the same serialization/commit boundary used by Save + reopen.
        saved = scene.serialize_document()
        assert scene._commit_document(saved)
        restored_root = next(
            obj for obj in scene.get_root_objects() if obj.name == "Authored Blender Instance"
        )
        restored = _descendants(restored_root)
        restored_renamed = restored["RenamedStable"]
        assert restored_renamed.id == stable_id
        assert tuple(restored_renamed.transform.local_position) == (17.0, 18.0, 19.0)
        assert restored_renamed.get_component("BoxCollider") is not None
        restored_renderer = restored_renamed.get_component("MeshRenderer")._require_cpp_component()
        assert restored_renderer.model_subresource_id == stable_subresource_id
        assert restored_renderer.serialize_document()["materials"] == authored_materials
    finally:
        if guid:
            registry.invalidate_asset(guid)
        database.configure_blender_import("", "")
        database.delete_asset(str(source))
        source.unlink(missing_ok=True)
        Path(str(source) + ".meta").unlink(missing_ok=True)


@pytest.mark.skipif(not os.environ.get("INFERNUX_TEST_BLENDER"), reason="requires the Blender 5.2 authoring tool")
def test_blend_structure_material_rig_and_authoring_boundary_matrix(engine, tmp_path):
    tool = os.environ["INFERNUX_TEST_BLENDER"]
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "AuthoringMatrix.blend"
    external_texture = folder / "ExternalAlbedo.png"
    fixture_script = Path(__file__).with_name("fixtures") / "create_blender_import_matrix.py"
    subprocess.run(
        [tool, "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
         "--python", str(fixture_script), "--", str(source), str(external_texture)],
        check=True, capture_output=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    authored_source = source.read_bytes()
    texture_import = database.import_asset(str(external_texture))
    assert texture_import, texture_import.error
    database.configure_blender_import(tool, export_script())
    imported = database.import_asset(str(source))
    assert imported, imported.error
    try:
        # Coordinate/scale assertions intentionally inspect CPU geometry; make
        # that authoring requirement explicit instead of triggering a hidden
        # GPU readback from the default non-readable import.
        settings = read_mesh_import_settings(str(source))
        settings.is_readable = True
        database.begin_model_reimport(str(source), settings.to_dict())
        deadline = time.monotonic() + 150
        while (readable_result := AssetManager.poll_model_reimport(database)) is None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert readable_result, readable_result.error
        mesh = registry.load_mesh(str(source))
        assert mesh and mesh.has_skinned_data
        assert mesh.skinned_bone_count >= 2
        assert mesh.skinned_animation_names
        animation_descriptors = embedded_take_descriptors(read_meta_file(str(source)))
        assert len(animation_descriptors) == 1
        descriptor = animation_descriptors[0]
        clip = AnimationClip3D.from_embedded_take_virtual_path(
            f"{source}::subanim:{descriptor['id']}"
        )
        assert clip is not None
        assert clip.source_model_guid == imported.guid
        assert clip.take_name == descriptor["id"]
        assert clip.duration_hint > 0.0
        published_clip = json.loads(read_meta_file(str(source))["model_animations"])[0]
        imported_document = json.loads(
            published_clip["metadata"]["metadata"]["import_document"]["value"]
        )
        assert {
            "apply_root_motion", "bone_mask", "curves", "default_loop", "reference_pose"
        } <= imported_document.keys()
        assert imported_document["apply_root_motion"] == clip.apply_root_motion
        assert imported_document["bone_mask"] == clip.bone_mask
        assert imported_document["curves"] == [curve.to_dict() for curve in clip.curves]
        assert imported_document["default_loop"] == clip.default_loop
        assert imported_document["reference_pose"] == clip.reference_pose
        bone_names = {
            name.strip()
            for name in database.get_meta_by_guid(imported.guid).get_string("bone_names_csv").split(",")
            if name.strip()
        }
        assert {"RootBone", "TipBone"} <= bone_names

        # Exercise the actual Blender-authored take through Apply and its
        # published clip document, not only the importer defaults above.
        settings.animation_loop_time = False
        settings.animation_apply_root_motion = True
        settings.animation_reference_pose = "first_frame"
        settings.animation_clip_extras = [{
            "clip_id": descriptor["id"],
            "curves": [{"name": "Grip", "keys": [
                {"time_normalized": 0.0, "value": 0.0},
                {"time_normalized": 1.0, "value": 1.0},
            ]}],
            "events": [{"time_normalized": 0.5, "function": "on_grip",
                        "string_arg": "RootBone", "number_arg": 1.0}],
            "bone_mask": ["RootBone"],
        }]
        applied = AssetManager.reimport_asset(
            str(source), import_settings=settings.to_dict(), database=database,
        )
        assert applied, applied.error
        applied_descriptor = embedded_take_descriptors(read_meta_file(str(source)))[0]
        assert applied_descriptor["id"] == descriptor["id"]
        applied_clip = AnimationClip3D.from_embedded_take_virtual_path(
            f"{source}::subanim:{descriptor['id']}"
        )
        assert applied_clip is not None
        assert applied_clip.default_loop is False
        assert applied_clip.apply_root_motion is True
        assert applied_clip.reference_pose == "first_frame"
        assert applied_clip.bone_mask == ["RootBone"]
        assert applied_clip.sample_curve("Grip", 0.5) == pytest.approx(0.5)
        assert applied_clip.events[0].function == "on_grip"
        applied_document = json.loads(
            applied_descriptor["metadata"]["metadata"]["import_document"]["value"]
        )
        assert applied_document["bone_mask"] == ["RootBone"]
        assert applied_document["curves"][0]["name"] == "Grip"

        node_records = list(mesh.get_model_nodes())
        nodes = {node["name"]: node for node in node_records}
        assert {"Assembly", "AxisContract", "MirroredHardEdge", "MorphWithModifier", "CollectionInstance",
                "ReusableTriangle", "ConstraintTarget", "ConstrainedTriangle", "BakedCurve",
                "UnsupportedCloth", "Rig", "SkinnedQuad"} <= nodes.keys()
        assert "IgnoredCamera" not in nodes and "IgnoredLight" not in nodes
        assert "Inflate" in mesh.morph_target_names
        mirrored = np.asarray(nodes["MirroredHardEdge"]["local_matrix"], dtype=np.float64)
        assert np.linalg.det(mirrored[:3, :3]) < 0
        np.testing.assert_allclose(
            sorted(np.linalg.norm(mirrored[:3, :3], axis=0)),
            sorted((1.0, 1.5, 0.75)),
            atol=1.e-5,
        )
        assert nodes["Assembly"]["node_group"] == -1
        assert node_records[nodes["MirroredHardEdge"]["parent_index"]]["name"] == "Assembly"
        assert nodes["CollectionInstance"]["node_group"] == -1
        assert node_records[nodes["ReusableTriangle"]["parent_index"]]["name"] == "CollectionInstance"
        assert nodes["ReusableTriangle"]["node_group"] >= 0
        assert nodes["Rig"]["node_group"] == -1
        assert node_records[nodes["SkinnedQuad"]["parent_index"]]["name"] == "Rig"
        assert nodes["BakedCurve"]["node_group"] >= 0
        constrained_local = np.asarray(nodes["ConstrainedTriangle"]["local_matrix"], dtype=np.float64)
        np.testing.assert_allclose(constrained_local[:3, 3], (0.75, 0.5, 1.25), atol=1.e-5)

        # Blender's (X, Y, Z) right-handed Z-up coordinates cross the managed
        # GLB boundary exactly once as engine/glTF (X, Z, -Y) Y-up coordinates.
        # Use merged model-space vertices here: applying model-node matrices a
        # second time is expressly outside the Mesh contract.
        axis_group = nodes["AxisContract"]["node_group"]
        axis_submesh = next(
            mesh.get_submesh_info(index)
            for index in range(mesh.submesh_count)
            if mesh.get_submesh_info(index)["node_group"] == axis_group
        )
        axis_positions = mesh.get_vertex_data()["positions"][
            axis_submesh["vertex_start"]:
            axis_submesh["vertex_start"] + axis_submesh["vertex_count"]
        ]
        expected_axis_positions = np.asarray(
            [(3.0, 5.0, -4.0), (4.0, 5.0, -4.0), (3.0, 7.0, -4.0)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            np.asarray(sorted(map(tuple, axis_positions.tolist()))),
            np.asarray(sorted(map(tuple, expected_axis_positions.tolist()))),
            atol=1.e-5,
        )

        hard_group = nodes["MirroredHardEdge"]["node_group"]
        hard_vertices = sum(
            mesh.get_submesh_info(index)["vertex_count"]
            for index in range(mesh.submesh_count)
            if mesh.get_submesh_info(index)["node_group"] == hard_group
        )
        assert hard_vertices > 4  # The bevel modifier was baked by the managed exporter.

        hard_normals = []
        vertex_data = mesh.get_vertex_data()
        for index in range(mesh.submesh_count):
            info = mesh.get_submesh_info(index)
            if info["node_group"] != hard_group:
                continue
            start, count = info["vertex_start"], info["vertex_count"]
            hard_normals.extend(vertex_data["normals"][start:start + count])
        hard_normals = np.asarray(hard_normals, dtype=np.float64)
        assert np.isfinite(hard_normals).all()
        np.testing.assert_allclose(np.linalg.norm(hard_normals, axis=1), 1.0, atol=2.e-5)
        assert len({tuple(np.round(normal, 4)) for normal in hard_normals}) >= 3

        slot = next(
            data for name, data in zip(mesh.material_slot_names, mesh.get_material_slot_data())
            if name == "TransparentExternal"
        )
        assert slot["alpha_mode"] == "blend"
        assert slot["base_color_texture_guid"] == texture_import.guid
        embedded_slot = next(
            data for name, data in zip(mesh.material_slot_names, mesh.get_material_slot_data())
            if name == "OpaqueEmbedded"
        )
        embedded_textures = json.loads(
            database.get_meta_by_guid(imported.guid).get_string("model_textures")
        )
        assert len(embedded_textures) == 1
        assert embedded_textures[0]["name"] == "Embedded Accent"
        assert embedded_slot["base_color_texture_guid"] == embedded_textures[0]["guid"]
        assert embedded_slot["base_color_texture_guid"] != texture_import.guid

        diagnostics = json.loads(
            database.get_meta_by_guid(imported.guid).get_string("blender_import_diagnostics")
        )
        assert diagnostics == [
            {
                "code": "baked_authoring_geometry",
                "owner": "BakedCurve",
                "property": "object/type",
                "detail": "CURVE",
            },
            {
                "code": "baked_geometry_modifier",
                "owner": "MirroredHardEdge",
                "property": "modifier/BakedBevel",
                "detail": "BEVEL",
            },
            {
                "code": "baked_geometry_modifier",
                "owner": "MirroredHardEdge",
                "property": "modifier/BakedGeometryNodes",
                "detail": "NODES",
            },
            {
                "code": "baked_object_constraints",
                "owner": "ConstrainedTriangle",
                "property": "object/constraints",
                "detail": "BakedCopyLocation",
            },
            {
                "code": "ignored_scene_object",
                "owner": "IgnoredCamera",
                "property": "object/type",
                "detail": "CAMERA",
            },
            {
                "code": "ignored_scene_object",
                "owner": "IgnoredLight",
                "property": "object/type",
                "detail": "LIGHT",
            },
            {
                "code": "unsupported_driver",
                "owner": "MirroredHardEdge",
                "property": "object/driver",
                "detail": '[\"driver_probe\"]',
            },
            {
                "code": "unsupported_modifier_with_shape_keys",
                "owner": "MorphWithModifier",
                "property": "modifier/UnbakedMorphBevel",
                "detail": "BEVEL",
            },
            {
                "code": "unsupported_simulation",
                "owner": "UnsupportedCloth",
                "property": "modifier/AuthoringCloth",
                "detail": "CLOTH",
            },
        ]

        # Scale Factor is one model-wide authoring conversion: it scales local
        # geometry and every node translation around the source origin.  It is
        # not applied independently around each submesh pivot.  Baking the
        # source axis basis may redistribute the synthetic root transform but
        # must leave merged geometry unchanged.
        baseline_positions = mesh.get_vertex_data()["positions"].copy()
        baseline_nodes = {node["name"]: np.asarray(node["local_matrix"], dtype=np.float64)
                          for node in mesh.get_model_nodes()}
        settings.scale_factor = 0.25
        database.begin_model_reimport(str(source), settings.to_dict())
        deadline = time.monotonic() + 150
        while (scaled_result := AssetManager.poll_model_reimport(database)) is None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert scaled_result, scaled_result.error
        scaled_positions = mesh.get_vertex_data()["positions"].copy()
        np.testing.assert_allclose(scaled_positions, baseline_positions * 0.25, atol=2.e-5)
        scaled_nodes = {node["name"]: np.asarray(node["local_matrix"], dtype=np.float64)
                        for node in mesh.get_model_nodes()}
        assert scaled_nodes.keys() == baseline_nodes.keys()
        for name, baseline in baseline_nodes.items():
            np.testing.assert_allclose(scaled_nodes[name][:3, :3], baseline[:3, :3], atol=2.e-5)
            np.testing.assert_allclose(scaled_nodes[name][:3, 3], baseline[:3, 3] * 0.25, atol=2.e-5)

        settings.bake_axis_conversion = True
        database.begin_model_reimport(str(source), settings.to_dict())
        deadline = time.monotonic() + 150
        while (baked_result := AssetManager.poll_model_reimport(database)) is None:
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert baked_result, baked_result.error
        baked_positions = mesh.get_vertex_data()["positions"].copy()
        np.testing.assert_allclose(baked_positions, scaled_positions, atol=2.e-5)

        # Cooked reload retains source-space hierarchy, rig and external GUID
        # binding without invoking Blender or reading the .blend source.
        database.configure_blender_import("", "")
        registry.invalidate_asset(imported.guid)
        restored = registry.load_mesh(str(source))
        assert restored.has_skinned_data and restored.skinned_bone_count == mesh.skinned_bone_count
        assert {node["name"] for node in restored.get_model_nodes()} == set(nodes)
        np.testing.assert_allclose(restored.get_vertex_data()["positions"], baked_positions, atol=2.e-5)
        restored_slot = next(
            data for name, data in zip(restored.material_slot_names, restored.get_material_slot_data())
            if name == "TransparentExternal"
        )
        assert restored_slot["base_color_texture_guid"] == texture_import.guid
        assert source.read_bytes() == authored_source
    finally:
        registry.invalidate_asset(imported.guid)
        registry.invalidate_asset(texture_import.guid)
        database.configure_blender_import("", "")
        database.delete_asset(str(source))
        database.delete_asset(str(external_texture))
        for authored in (source, external_texture):
            authored.unlink(missing_ok=True)
            Path(str(authored) + ".meta").unlink(missing_ok=True)
